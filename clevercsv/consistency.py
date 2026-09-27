# -*- coding: utf-8 -*-

"""
Detect the dialect using the data consistency measure.

Author: Gertjan van den Burg

"""

from dataclasses import dataclass
from functools import lru_cache

import math
import warnings

from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional

from . import field_size_limit
from .break_ties import tie_breaker
from .cparser_util import parse_string
from .detect_pattern import pattern_score
from .detect_type import DEFAULT_EPS_TYPE
from .detect_type import TypeDetector
from .dialect import SimpleDialect
from .exceptions import DialectDetectionWarning
from .potential_dialects import get_dialects
from .roundtrip import verify_roundtrip

# Relative tolerance used when comparing consistency scores. Scores that are
# equal up to floating point noise are considered tied, so that an arbitrary
# winner is never picked on the basis of rounding error alone.
SCORE_TOLERANCE: float = 1e-9


@dataclass
class ConsistencyScore:
    """Container to track the consistency score calculation

    Parameters
    ----------
    P : float
        The pattern score

    T : Optional[float]
        The type score. Can be None if not computed for speed.

    Q : Optional[float]
        The consistency score. Can be None if not computed for speed.

    """

    P: float
    T: Optional[float]
    Q: Optional[float]


class ConsistencyDetector:
    """Detect the dialect with the data consistency measure

    This class uses the data consistency measure to detect the dialect. See the
    paper for details.

    Parameters
    ----------
    skip : bool
        Skip computation of the type score for dialects with a low pattern
        score.

    verbose : bool
        Print out the dialects considered and their scores.

    cache_capacity: int
        The size of the cache for type detection. Caching the type detection
        result greatly speeds up the computation of the consistency measure.
        The size of the cache can be changed to trade off memory use and speed.

    Attributes
    ----------
    scores_ : Dict[SimpleDialect, ConsistencyScore]
        The consistency scores computed for each potential dialect during the
        last call to :meth:`detect`. This allows inspecting why a particular
        dialect was selected.

    """

    def __init__(
        self,
        skip: bool = True,
        verbose: bool = False,
        cache_capacity: int = 100_000,
    ) -> None:
        self._skip = skip
        self._verbose = verbose
        self._type_detector = TypeDetector()
        self._cache_capacity = cache_capacity
        self.scores_: Dict[SimpleDialect, ConsistencyScore] = {}

        # NOTE: A bit ugly but allows setting the cache size dynamically
        @lru_cache(cache_capacity)
        def cached_is_known_type(cell: str, is_quoted: bool) -> bool:
            return self._type_detector.is_known_type(cell, is_quoted)

        self._cached_is_known_type = cached_is_known_type

    def detect(
        self, data: str, delimiters: Optional[List[str]] = None
    ) -> Optional[SimpleDialect]:
        """Detect the dialect using the consistency measure

        Parameters
        ----------
        data : str
            The data of the file as a string

        delimiters : iterable
            List of delimiters to consider. If None, the :func:`get_delimiters`
            function is used to automatically detect this (as described in the
            paper).

        Returns
        -------
        dialect : SimpleDialect
            The detected dialect. If no dialect could be detected, returns None.

        Notes
        -----
        The candidate dialects are compared through the consistency score
        (the product of the pattern score and the type score), which is
        stored on the :attr:`scores_` attribute for inspection. If several
        dialects tie for the highest score and the tie cannot be broken, a
        :class:`DialectDetectionWarning` is emitted and None is returned
        instead of arbitrarily picking a winner. The selected dialect is
        only returned if it passes round-trip verification (see
        :func:`clevercsv.roundtrip.verify_roundtrip`); if it does not, the
        remaining candidates are tried in order of decreasing score.

        """
        self._cached_is_known_type.cache_clear()

        # TODO: probably some optimization there too
        dialects = get_dialects(data, delimiters=delimiters)

        # TODO: This is not thread-safe and this object can simply own a Parser
        # for each dialect and set the limit directly there (we can also cache
        # the best parsing result)
        old_limit = field_size_limit(len(data) + 1)

        try:
            scores = self.compute_consistency_scores(data, dialects)
            self.scores_ = scores
            result = self._select_dialect(data, scores)
        finally:
            field_size_limit(old_limit)
        return result

    def _select_dialect(
        self, data: str, scores: Dict[SimpleDialect, ConsistencyScore]
    ) -> Optional[SimpleDialect]:
        """Select a dialect from the computed consistency scores

        The dialect with the highest consistency score wins. Ties are broken
        with :func:`clevercsv.break_ties.tie_breaker`; if the tie cannot be
        broken a :class:`DialectDetectionWarning` is emitted and None is
        returned. The winning dialect is validated with a round-trip check
        and rejected if parsing the data and writing it back does not
        reproduce the same rows. In that case the remaining candidates are
        tried in order of decreasing consistency score.

        """
        best_dialects = ConsistencyDetector.get_best_dialects(scores)
        candidates: List[SimpleDialect] = []
        if not best_dialects:
            warnings.warn(
                "Dialect detection failed: no candidate dialects could be "
                "constructed for the data.",
                DialectDetectionWarning,
                stacklevel=3,
            )
            return None
        if len(best_dialects) == 1:
            candidates = list(best_dialects)
        else:
            winner = tie_breaker(data, best_dialects)
            if winner is None:
                self._warn_inconclusive(best_dialects, scores)
                return None
            candidates = [winner]

        # Fall back to the remaining candidates in order of decreasing
        # consistency score if the winner fails round-trip verification.
        rest = [
            d
            for d, score in scores.items()
            if score.Q is not None and d not in candidates
        ]
        rest.sort(key=lambda d: -(scores[d].Q or 0.0))
        candidates.extend(rest)

        for dialect in candidates:
            if verify_roundtrip(data, dialect):
                return dialect
            if self._verbose:
                print(
                    "%15r:\tfailed round-trip verification, rejecting."
                    % (dialect,)
                )

        warnings.warn(
            "Dialect detection failed: no candidate dialect passed "
            "round-trip verification (parsing the data and writing it back "
            "with the same dialect did not reproduce the same rows).",
            DialectDetectionWarning,
            stacklevel=3,
        )
        return None

    def _warn_inconclusive(
        self,
        best_dialects: List[SimpleDialect],
        scores: Dict[SimpleDialect, ConsistencyScore],
    ) -> None:
        """Emit a warning that detection was inconclusive"""
        lines = [
            "Dialect detection was inconclusive: %i dialects share the "
            "highest consistency score and the tie could not be broken:"
            % len(best_dialects)
        ]
        for dialect in best_dialects:
            score = scores[dialect]
            lines.append(
                "    %r (P=%.6f, T=%s, Q=%.6f)"
                % (
                    dialect,
                    score.P,
                    "%.6f" % score.T if score.T is not None else "n/a",
                    score.Q if score.Q is not None else float("nan"),
                )
            )
        lines.append(
            "No dialect was selected. Inspect the scores on the scores_ "
            "attribute of the detector or specify the dialect manually."
        )
        warnings.warn(
            "\n".join(lines), DialectDetectionWarning, stacklevel=3
        )

    def compute_consistency_scores(
        self, data: str, dialects: List[SimpleDialect]
    ) -> Dict[SimpleDialect, ConsistencyScore]:
        """Compute the consistency score for each dialect

        This function computes the consistency score for each dialect. This is
        done by first computing the pattern score for a dialect. If the class
        is instantiated with ``skip`` set to False, it also computes the type
        score for each dialect. If ``skip`` is True (the default), the type
        score is only computed if the pattern score is larger or equal to the
        current best combined score.

        Parameters
        ----------
        data : str
            The data of the file as a string

        dialects : Iterable[SimpleDialect]
            An iterable of delimiters to consider.

        Returns
        -------
        scores : Dict[SimpleDialect, ConsistencyScore]
            A map with a :class:`ConsistencyScore` object for each dialect
            provided as input.

        """

        scores: Dict[SimpleDialect, ConsistencyScore] = {}
        incumbent_score = -float("inf")
        for dialect in sorted(dialects):
            P = pattern_score(data, dialect)
            if P < incumbent_score and self._skip:
                scores[dialect] = ConsistencyScore(P, None, None)
                if self._verbose:
                    print("%15r:\tP = %15.6f\tskip." % (dialect, P))
                continue

            T = self.compute_type_score(data, dialect)
            Q = P * T
            incumbent_score = max(incumbent_score, Q)
            scores[dialect] = ConsistencyScore(P, T, Q)
            if self._verbose:
                print(
                    "%15r:\tP = %15.6f\tT = %15.6f\tQ = %15.6f"
                    % (dialect, P, T, Q)
                )
        return scores

    @staticmethod
    def get_best_dialects(
        scores: Dict[SimpleDialect, ConsistencyScore]
    ) -> List[SimpleDialect]:
        """Identify the dialects with the highest consistency score

        Scores that are equal up to a small relative tolerance
        (:data:`SCORE_TOLERANCE`) are considered tied, so that a winner is
        never picked arbitrarily on the basis of floating point noise.

        """
        Qscores = [score.Q for score in scores.values()]
        Qmax = -float("inf")
        for q in Qscores:
            if q is None:
                continue
            Qmax = max(Qmax, q)
        return [
            d
            for d, score in scores.items()
            if score.Q is not None
            and math.isclose(score.Q, Qmax, rel_tol=SCORE_TOLERANCE)
        ]

    def compute_type_score(
        self, data: str, dialect: SimpleDialect, eps: float = DEFAULT_EPS_TYPE
    ) -> float:
        """Compute the type score"""
        total = known = 0
        for row in parse_string(data, dialect, return_quoted=True):
            assert all(isinstance(cell, tuple) for cell in row)
            for cell, is_quoted in row:
                total += 1
                known += self._cached_is_known_type(cell, is_quoted=is_quoted)
        if not total:
            return eps
        return max(eps, known / total)


def detect_dialect_consistency(
    data: str,
    delimiters: Optional[Iterable[str]] = None,
    skip: bool = True,
    verbose: bool = False,
) -> Optional[SimpleDialect]:
    """Helper function that wraps ConsistencyDetector"""
    # Mostly kept for backwards compatibility
    consistency_detector = ConsistencyDetector(skip=skip, verbose=verbose)
    if delimiters is not None:
        delimiters = list(delimiters)
    return consistency_detector.detect(data, delimiters=delimiters)
