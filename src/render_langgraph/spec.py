"""Splitting "path:attr" specs.

Windows absolute paths (`C:\\...` or `C:/...`) contain a colon right after
the drive letter, which collides with the `path:attr` separator convention.
`"C:/proj/graph.py:graph".partition(":")` would (wrongly) split into
`"C"` and `"/proj/graph.py:graph"`. Split on the *last* colon instead, but
only after setting aside a leading drive letter so a bare Windows path with
no `:attr` suffix (`C:/proj/graph.py`) isn't misread as `path="", attr="C"`.

NOTE: this logic is intentionally duplicated in `_extractor.py`, which must
stay a standalone script runnable in a venv that doesn't have render-langgraph
installed.
"""
from __future__ import annotations

import re

_DRIVE_RE = re.compile(r"^[A-Za-z]:[\\/]")


def split_spec(spec: str) -> tuple[str, str]:
    """Return (path, attr). attr is "" when the spec has no `:attr` suffix."""
    prefix = ""
    rest = spec
    if _DRIVE_RE.match(spec):
        prefix, rest = spec[:2], spec[2:]
    if ":" not in rest:
        return prefix + rest, ""
    path, _, attr = rest.rpartition(":")
    return prefix + path, attr
