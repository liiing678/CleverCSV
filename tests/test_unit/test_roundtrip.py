# -*- coding: utf-8 -*-

"""
Unit tests for round-trip verification and score-based dialect selection.

"""

import unittest
import warnings

from clevercsv.consistency import ConsistencyDetector
from clevercsv.consistency import ConsistencyScore
from clevercsv.detect import Detector
from clevercsv.dialect import SimpleDialect
from clevercsv.exceptions import DialectDetectionWarning
from clevercsv.roundtrip import verify_roundtrip


class RoundtripTestCase(unittest.TestCase):
    def test_roundtrip_simple(self) -> None:
        data = "a,b,c\n1,2,3\n4,5,6\n"
        self.assertTrue(
            verify_roundtrip(data, SimpleDialect(",", "", ""))
        )

    def test_roundtrip_quoted(self) -> None:
        data = 'a,"b,c",d\n1,"x,y",2\n'
        self.assertTrue(
            verify_roundtrip(data, SimpleDialect(",", '"', ""))
        )

    def test_roundtrip_unclosed_quote_no_quotechar(self) -> None:
        # A stray quote in an unquoted field is represented faithfully by a
        # dialect without a quotechar.
        data = 'a,b,c\n1,"2,3\n4,5,6\n'
        self.assertTrue(
            verify_roundtrip(data, SimpleDialect(",", "", ""))
        )

    def test_roundtrip_no_delimiter_with_quotechar(self) -> None:
        # A dialect without delimiter but with a quotechar cannot be written
        # by the csv module, so it cannot pass round-trip verification.
        data = 'a,b,c\n1,"2,3\n4,5,6\n'
        self.assertFalse(
            verify_roundtrip(data, SimpleDialect("", '"', ""))
        )

    def test_roundtrip_empty(self) -> None:
        self.assertTrue(verify_roundtrip("", SimpleDialect(",", "", "")))


class SelectionTestCase(unittest.TestCase):
    def test_tied_dialects_warn_and_return_none(self) -> None:
        # Two delimiters give exactly the same consistency score and the tie
        # cannot be broken: no dialect may be picked arbitrarily.
        data = "a;b|c\n1;2|3\n"
        detector = ConsistencyDetector(skip=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = detector.detect(data)
        self.assertIsNone(result)
        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DialectDetectionWarning)
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn("inconclusive", messages[0])
        self.assertIn("SimpleDialect(';', '', '')", messages[0])
        self.assertIn("SimpleDialect('|', '', '')", messages[0])

    def test_scores_are_exposed(self) -> None:
        data = "a;b|c\n1;2|3\n"
        detector = ConsistencyDetector(skip=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector.detect(data)
        self.assertTrue(detector.scores_)
        best = ConsistencyDetector.get_best_dialects(detector.scores_)
        self.assertEqual(
            set(best),
            set([SimpleDialect(";", "", ""), SimpleDialect("|", "", "")]),
        )

    def test_get_best_dialects_tolerance(self) -> None:
        # Scores that differ only by floating point noise must be treated as
        # tied, not as a reason to pick an arbitrary winner.
        scores = {
            SimpleDialect(",", None, None): ConsistencyScore(
                P=1, T=1, Q=1.0
            ),
            SimpleDialect(";", None, None): ConsistencyScore(
                P=1, T=1, Q=1.0 + 1e-12
            ),
        }
        best = ConsistencyDetector.get_best_dialects(scores)
        self.assertEqual(len(best), 2)

    def test_roundtrip_rejection_falls_back_by_score(self) -> None:
        # The highest scoring candidate fails round-trip verification (a
        # dialect without delimiter cannot be written), so the next best
        # candidate that does verify must be selected.
        data = 'a,b,c\n1,"2,3\n4,5,6\n'
        scores = {
            SimpleDialect("", '"', ""): ConsistencyScore(P=10, T=1, Q=10),
            SimpleDialect(",", "", ""): ConsistencyScore(P=1, T=1, Q=1),
        }
        detector = ConsistencyDetector(skip=False)
        result = detector._select_dialect(data, scores)
        self.assertEqual(result, SimpleDialect(",", "", ""))

    def test_no_verified_candidate_warns(self) -> None:
        # If no candidate passes round-trip verification a clear warning is
        # emitted and None is returned.
        data = 'a,b,c\n1,"2,3\n4,5,6\n'
        scores = {
            SimpleDialect("", '"', ""): ConsistencyScore(P=10, T=1, Q=10),
        }
        detector = ConsistencyDetector(skip=False)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = detector._select_dialect(data, scores)
        self.assertIsNone(result)
        messages = [
            str(w.message)
            for w in caught
            if issubclass(w.category, DialectDetectionWarning)
        ]
        self.assertEqual(len(messages), 1)
        self.assertIn("round-trip", messages[0])


class DetectorRoundtripTestCase(unittest.TestCase):
    def test_detected_dialects_verify(self) -> None:
        # Any dialect returned by the Detector must pass round-trip
        # verification on the sample it was detected from.
        samples = [
            "name;age;city\nAlice;30;Paris\nBob;25;Berlin\n",
            'name,note,score\nAlice,"top performer,9\nBob,"ok,7\n',
            "id,name,note\n1,'Alice',ok\n2,'Bob',fine\n",
            'a;b;c\n1;"broken;3\n4;"again;6\n',
            "name\nAlice\nBob\n",
        ]
        detector = Detector()
        for sample in samples:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dialect = detector.detect(sample)
            if dialect is None:
                continue
            self.assertTrue(
                verify_roundtrip(sample, dialect),
                msg="Detected dialect %r does not round-trip" % (dialect,),
            )

    def test_unclosed_quote_not_collapsed(self) -> None:
        # An unclosed quote must not cause the table to collapse into a
        # single row.
        data = 'a,b,c\n1,"2,3\n4,5,6\n7,8,9\n'
        detector = Detector()
        dialect = detector.detect(data)
        self.assertEqual(dialect, SimpleDialect(",", "", ""))

    def test_detector_scores_attribute(self) -> None:
        detector = Detector()
        # normal form detection: no consistency scores
        detector.detect("a,b,c\n1,2,3\n4,5,6\n")
        self.assertIsNone(detector.scores_)
        # consistency detection: scores are available for inspection
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            detector.detect("a;b|c\n1;2|3\n", method="consistency")
        self.assertIsNotNone(detector.scores_)


if __name__ == "__main__":
    unittest.main()
