# -*- coding: utf-8 -*-

"""
Round-trip verification for detected dialects.

A detected dialect is only trustworthy if the parsing result survives a
round-trip: parsing the data with the dialect and writing the resulting rows
back with the same dialect must yield a representation that parses to the
same rows again. Dialects that fail this check can silently mangle the file
(for instance by collapsing the table to a single row through an unclosed
quote, or by introducing spurious columns), so they should not be returned
as a detection result.

Author: Gertjan van den Burg

"""

from io import StringIO

from typing import List

from .cparser_util import parse_string
from .dialect import SimpleDialect
from .write import writer


def verify_roundtrip(data: str, dialect: SimpleDialect) -> bool:
    """Verify that a dialect survives a round-trip on the given data

    The data is parsed with the dialect, the resulting rows are written back
    to a string using the same dialect, and the written string is parsed
    again. The dialect passes verification only if both parses give exactly
    the same rows.

    Parameters
    ----------
    data : str
        The data of the file as a string.

    dialect : SimpleDialect
        The dialect to verify.

    Returns
    -------
    ok : bool
        Whether the dialect passes the round-trip verification. Returns False
        if parsing or writing fails for any reason.

    """
    try:
        rows = _parse(data, dialect)
        return _parse(_write_rows(rows, dialect), dialect) == rows
    except Exception:
        # Any parser or writer error (e.g. a cell that cannot be escaped with
        # the dialect) means the dialect does not round-trip.
        return False


def _write_rows(rows: List[List[str]], dialect: SimpleDialect) -> str:
    """Serialize rows to a string using the given dialect

    For dialects without a quotechar and without an escapechar the cells are
    joined directly with the delimiter. The standard library writer (used by
    :class:`clevercsv.writer`) refuses to write such dialects when a cell
    contains a quote character, even though the dialect represents the data
    faithfully (the cells can never contain the delimiter or a newline, as
    the parser splits on those). The direct join is the exact inverse of
    parsing with such a dialect.

    """
    if dialect.quotechar == "" and dialect.escapechar == "":
        delimiter = dialect.delimiter or ""
        return "".join(delimiter.join(row) + "\n" for row in rows)
    buffer = StringIO(newline="")
    wr = writer(buffer, dialect=dialect)
    wr.writerows(rows)
    return buffer.getvalue()


def _parse(data: str, dialect: SimpleDialect) -> List[List[str]]:
    """Parse a string to rows, normalizing away empty trailing rows"""
    rows = parse_string(data, dialect)
    return [list(row) for row in rows if row != []]
