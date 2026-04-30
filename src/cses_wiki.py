"""
Local retrieval over cses_wiki.

The CSES wiki is the compact runtime knowledge base for procedures, standards,
and approved patterns. It is intentionally independent of raw development
sources such as example_studies.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSES_WIKI_ROOT = PROJECT_ROOT / "cses_wiki"


def wiki_root() -> Path:
    """Return the active CSES wiki root."""
    return CSES_WIKI_ROOT


def _wiki_path(*parts: str) -> Path:
    return wiki_root().joinpath(*parts)


TOKEN_RE = re.compile(r"[A-Za-z0-9_]{3,}")


@dataclass(frozen=True)
class WikiChunk:
    chunk_id: str
    title: str
    topic: str
    status: str
    source_path: str
    context: str
    text: str
    year: int | None = None
    doc_type: str = ""


def wiki_available() -> bool:
    return _wiki_path("retrieval", "chunks.jsonl").exists()


def wiki_prompt_hint() -> str:
    """Small always-load hint for the system prompt."""
    if not wiki_available():
        return "CSES standards wiki: not available."
    return (
        "CSES standards wiki is available. Use search_cses_wiki before making uncertain "
        "procedural, eligibility, coding, documentation, data-protection, collaborator-question, "
        "or release-workflow decisions. Prefer a topic filter when obvious: study_eligibility, "
        "documentation_standards, demographic_coding, education_coding, data_processing, "
        "data_protection, district_data, macro_data, party_coding, operations, training, general. "
        "Cite CSES wiki source paths when applying retrieved guidance."
    )


def load_wiki_overview(max_chars: int = 2500) -> str:
    root = wiki_root()
    parts = []
    for path in [_wiki_path("AI_CONTEXT.md"), _wiki_path("ROUTER.md")]:
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                parts.append(f"## {path.relative_to(root)}\n{text}")
    overview = "\n\n".join(parts)
    return overview[:max_chars]


def search_wiki(query: str, topic: str = "", limit: int = 5) -> list[WikiChunk]:
    """Return the most relevant chunks for a query using lexical scoring."""
    query_terms = _tokenize(query)
    if not query_terms:
        return []

    topic_filter = topic.strip().lower().replace("-", "_")
    chunks = _load_chunks()
    if topic_filter:
        chunks = [
            chunk for chunk in chunks
            if chunk.topic.lower() == topic_filter or topic_filter in chunk.topic.lower()
        ]

    doc_freq = _document_frequencies()
    total_docs = max(len(_load_chunks()), 1)
    scored: list[tuple[float, WikiChunk]] = []

    for chunk in chunks:
        haystack = f"{chunk.title} {chunk.topic} {chunk.context} {chunk.text}"
        terms = _tokenize(haystack)
        if not terms:
            continue
        counts = {}
        for term in terms:
            counts[term] = counts.get(term, 0) + 1
        score = 0.0
        for term in query_terms:
            count = counts.get(term, 0)
            if count:
                idf = math.log((total_docs + 1) / (doc_freq.get(term, 0) + 1)) + 1
                score += (1 + math.log(count)) * idf
        if score:
            if chunk.status == "current":
                score *= 1.15
            if chunk.status == "reference":
                score *= 1.05
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[: max(1, min(limit, 10))]]


def format_wiki_results(chunks: list[WikiChunk]) -> str:
    if not chunks:
        return "No relevant CSES standards wiki results found."

    lines = []
    for index, chunk in enumerate(chunks, start=1):
        excerpt = " ".join(chunk.text.split())
        if len(excerpt) > 900:
            excerpt = excerpt[:897] + "..."
        lines.extend(
            [
                f"Result {index}: {chunk.title}",
                f"Topic: {chunk.topic}; status: {chunk.status}; year: {chunk.year or 'unknown'}; type: {chunk.doc_type or 'unknown'}",
                f"Source: {chunk.source_path}",
                f"Context: {chunk.context}",
                f"Excerpt: {excerpt}",
                "",
            ]
        )
    return "\n".join(lines).strip()


@lru_cache(maxsize=1)
def _load_chunks() -> tuple[WikiChunk, ...]:
    chunks_path = _wiki_path("retrieval", "chunks.jsonl")
    if not chunks_path.exists():
        return tuple()

    chunks = []
    with chunks_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunks.append(
                WikiChunk(
                    chunk_id=str(data.get("chunk_id", "")),
                    title=str(data.get("title", "")),
                    topic=str(data.get("topic", "")),
                    status=str(data.get("status", "")),
                    source_path=str(data.get("source_path", "")),
                    context=str(data.get("context", "")),
                    text=str(data.get("text", "")),
                    year=data.get("year") if isinstance(data.get("year"), int) else None,
                    doc_type=str(data.get("doc_type", "")),
                )
            )
    return tuple(chunks)


@lru_cache(maxsize=1)
def _document_frequencies() -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for chunk in _load_chunks():
        terms = set(_tokenize(f"{chunk.title} {chunk.topic} {chunk.context} {chunk.text}"))
        for term in terms:
            frequencies[term] = frequencies.get(term, 0) + 1
    return frequencies


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]
