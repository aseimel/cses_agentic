"""Processor-facing text helpers.

Internal workflow services can use precise implementation names, but normal GUI
and chat output must stay in professional CSES processing language.
"""

from __future__ import annotations

import re
from typing import Iterable


BANNED_USER_TERMS = (
    "Evidence packet",
    "Study KB",
    "KB",
    "manifest",
    "chunk",
    "diagnostics",
    "model activity",
    "workflow state",
    "shared context",
    "handoff",
    ".cses",
    "artifact",
    "source hash",
    "tokens",
    "fields_found",
    "standards source",
)

_DROP_LINE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bmodel activity\b",
        r"\bchunk(s)?\b",
        r"\bmanifest hash\b",
        r"\bsource hash\b",
        r"\bfields_found\b",
        r"\binput tokens\b",
        r"\bestimated .*tokens\b",
        r"\bdiagnostics?\b",
        r"\bhandoff\b",
        r"\bshared context\b",
        r"\bworkflow state\b",
        r"\bstandards source\b",
        r"\.cses\b",
    )
]

_REPLACEMENTS = (
    (re.compile(r"\bEvidence packet\b", re.IGNORECASE), "Study materials review"),
    (re.compile(r"\bshared evidence packet\b", re.IGNORECASE), "study materials review"),
    (re.compile(r"\breusable evidence packet\b", re.IGNORECASE), "study materials review"),
    (re.compile(r"\bstudy knowledge base\b", re.IGNORECASE), "study information"),
    (re.compile(r"\bStudy KB\b", re.IGNORECASE), "Study information"),
    (re.compile(r"\bKB\b"), "study information"),
    (re.compile(r"\bstandards\b", re.IGNORECASE), "CSES rules"),
    (re.compile(r"\bartifacts?\b", re.IGNORECASE), "outputs"),
)


def sanitize_processor_text(text: object, *, drop_internal_lines: bool = True) -> str:
    """Return text suitable for the normal processor-facing UI."""
    if text is None:
        return ""
    cleaned = str(text)
    for pattern, replacement in _REPLACEMENTS:
        cleaned = pattern.sub(replacement, cleaned)

    if drop_internal_lines:
        visible_lines = []
        for line in cleaned.splitlines():
            if any(pattern.search(line) for pattern in _DROP_LINE_PATTERNS):
                continue
            visible_lines.append(line)
        cleaned = "\n".join(visible_lines)

    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def assert_no_internal_terms(text: object, *, allowed_terms: Iterable[str] = ()) -> None:
    """Raise AssertionError if normal UI text contains banned implementation terms."""
    value = str(text or "")
    allowed = {term.casefold() for term in allowed_terms}
    violations = []
    for term in BANNED_USER_TERMS:
        if term.casefold() in allowed:
            continue
        if term == "KB":
            if re.search(r"\bKB\b", value):
                violations.append(term)
        elif re.search(re.escape(term), value, flags=re.IGNORECASE):
            violations.append(term)
    if violations:
        joined = ", ".join(sorted(set(violations)))
        raise AssertionError(f"Internal UI terms leaked: {joined}")


def format_study_review_status(status: str) -> str:
    """Map internal review states to processor-facing status text."""
    normalized = (status or "").strip().casefold()
    if normalized == "stale":
        return "Study files changed since last review."
    if normalized == "current":
        return "Study materials reviewed."
    if normalized == "missing":
        return ""
    if normalized:
        return f"Study materials need review: {normalized}."
    return ""


def processor_status_label(kind: str, status: str) -> str:
    """Small shared status labels for GUI panels."""
    kind_key = (kind or "").strip().casefold()
    status_text = format_study_review_status(status)
    if kind_key in {"study_review", "evidence"}:
        return status_text
    return sanitize_processor_text(status_text or status)
