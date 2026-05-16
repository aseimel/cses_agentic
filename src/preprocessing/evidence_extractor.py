"""
Parallel evidence extraction for CSES deposited files.

This service builds a reusable evidence index from the deposited documents.
It deliberately processes every chunk of each relevant file before reporting a
field as missing.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable

import litellm
import requests
from src.model_runtime import ModelRole, ModelTaskRunner

from src.ingest.doc_parser import DocumentParser
from src.settings import DEFAULT_AGENTIC_MODEL, DEFAULT_LARGE_TEXT_MODEL, apply_settings_to_environment

litellm.suppress_debug_info = True


CHUNK_CHARS = 12000
CHUNK_OVERLAP = 800
LLAMA_MAX_WORKERS = 4
OSS_MAX_WORKERS = 3
GLOBAL_MAX_WORKERS = 5
MAX_RETRIES = 1
LLM_TIMEOUT_SECONDS = 15
MODEL_HEALTH_TIMEOUT_SECONDS = 5


DESIGN_FIELDS = [
    "target_population",
    "eligible_population",
    "sample_design",
    "sampling_frame",
    "sampling_stages",
    "stratification",
    "psu_cluster_details",
    "respondent_selection",
    "sample_size",
    "response_rate",
    "fieldwork_dates",
    "mode",
    "weights",
    "field_lag",
    "consent_data_deposit",
]

MACRO_FIELDS = [
    "election_date",
    "election_type",
    "electoral_system",
    "turnout",
    "election_results",
    "parties",
    "leaders",
    "coalitions",
]

QUESTIONNAIRE_FIELDS = [
    "cses_item_coverage",
    "vote_choice_questions",
    "party_leader_questions",
    "response_categories",
    "missing_value_codes",
]

CODEBOOK_FIELDS = [
    "variable_labels",
    "value_labels",
    "missing_codes",
    "cses_variable_evidence",
]

FIELD_GROUPS = {
    "design": DESIGN_FIELDS,
    "macro": MACRO_FIELDS,
    "questionnaire": QUESTIONNAIRE_FIELDS,
    "codebook": CODEBOOK_FIELDS,
}

ALL_FIELDS = DESIGN_FIELDS + MACRO_FIELDS + QUESTIONNAIRE_FIELDS + CODEBOOK_FIELDS


@dataclass(frozen=True)
class EvidenceChunk:
    source_file: str
    doc_type: str
    chunk_id: int
    start: int
    end: int
    text: str


def _json_from_text(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _safe_text(text: str, limit: int = 900) -> str:
    text = " ".join(str(text or "").split())
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u202f": " ",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    text = text.encode("cp1252", errors="replace").decode("cp1252")
    return text[:limit]


class ParallelEvidenceExtractionService:
    """Builds a full-file, evidence-backed extraction index."""

    def __init__(
        self,
        working_dir: Path,
        large_text_model: str | None = None,
        agentic_model: str | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ):
        self.working_dir = Path(working_dir)
        settings = apply_settings_to_environment()
        self.large_text_model = large_text_model or os.getenv("CSES_LARGE_TEXT_MODEL") or settings.get("CSES_LARGE_TEXT_MODEL") or DEFAULT_LARGE_TEXT_MODEL
        self.agentic_model = (
            agentic_model
            or os.getenv("CSES_VERIFIER_MODEL")
            or settings.get("CSES_VERIFIER_MODEL")
            or os.getenv("CSES_AGENTIC_MODEL")
            or settings.get("CSES_AGENTIC_MODEL")
            or os.getenv("CSES_CHAT_MODEL")
            or DEFAULT_AGENTIC_MODEL
        )
        self.progress_callback = progress_callback
        self.global_semaphore = threading.BoundedSemaphore(GLOBAL_MAX_WORKERS)
        self.runner = ModelTaskRunner(self.working_dir, state=None)

    def build(self, state, force: bool = False) -> dict:
        existing = getattr(state, "evidence_index", {}) or {}
        if existing and existing.get("status") == "ready" and not force:
            return existing

        documents = self._discover_documents(state)
        large_text_available = self._model_health_check(self.large_text_model)
        if not large_text_available:
            self._progress("Using fast document review fallback for this pass.")
        chunks: list[EvidenceChunk] = []
        docs_summary = []
        parser = DocumentParser()
        for doc_type, path in documents:
            parsed = parser.parse(path)
            text = parsed.full_text if parsed else ""
            docs_summary.append(
                {
                    "doc_type": doc_type,
                    "source_file": str(path),
                    "chars": len(text),
                    "questions": len(parsed.questions) if parsed else 0,
                    "parse_errors": parsed.parse_errors if parsed else ["parse failed"],
                }
            )
            chunks.extend(self._chunk_text(path, doc_type, text))

        self._progress(f"Evidence extraction: {len(documents)} files, {len(chunks)} chunks")
        extraction_results = self._extract_chunks_parallel(chunks, use_llm=large_text_available)
        facts = [fact for result in extraction_results for fact in result.get("facts", [])]
        facts_by_field = self._merge_facts(facts)
        missing_fields = [field for field in ALL_FIELDS if not facts_by_field.get(field)]
        verifications = self._verify_parallel(facts_by_field, missing_fields)

        failed_chunks = [
            {
                "source_file": result.get("source_file"),
                "chunk_id": result.get("chunk_id"),
                "error": result.get("error", "invalid or empty extraction"),
            }
            for result in extraction_results
            if not result.get("ok")
        ]
        index = {
            "status": "ready" if not failed_chunks else "needs_review",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "models": {
                "large_text_extraction": self.large_text_model,
                "agentic_verification": self.agentic_model,
            },
            "concurrency": {
                "llama_max_workers": LLAMA_MAX_WORKERS,
                "oss_max_workers": OSS_MAX_WORKERS,
                "global_max_workers": GLOBAL_MAX_WORKERS,
                "max_retries": MAX_RETRIES,
            },
            "documents": docs_summary,
            "diagnostics": {
                "files_scanned": len(documents),
                "chunks_total": len(chunks),
                "chunks_ok": sum(1 for result in extraction_results if result.get("ok")),
                "chunks_failed": len(failed_chunks),
                "facts_total": len(facts),
                "fields_found": len([field for field in ALL_FIELDS if facts_by_field.get(field)]),
                "fields_missing": len(missing_fields),
                "large_text_model_available": large_text_available,
            },
            "failed_chunks": failed_chunks,
            "facts_by_field": facts_by_field,
            "missing_fields": missing_fields,
            "verifications": verifications,
        }
        evidence_path = self.working_dir / ".cses" / "evidence_index.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
        return index

    def _discover_documents(self, state) -> list[tuple[str, Path]]:
        candidates: list[tuple[str, Path]] = []

        def add(doc_type: str, value):
            if not value:
                return
            path = Path(value)
            if path.exists() and path.is_file():
                candidates.append((doc_type, path))

        add("design_report", getattr(state, "design_report_file", None))
        add("codebook", getattr(state, "codebook_file", None))
        for path in getattr(state, "questionnaire_files", []) or []:
            add("questionnaire", path)

        search_roots = [self.working_dir / "micro", self.working_dir / "macro", self.working_dir / "Election Results"]
        suffixes = {".pdf", ".docx", ".txt", ".md", ".rtf"}
        for root in search_roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.suffix.lower() not in suffixes:
                    continue
                name = path.name.lower()
                if "macro" in name:
                    candidates.append(("macro_report", path))
                elif "design" in name:
                    candidates.append(("design_report", path))
                elif "questionnaire" in name or "survey" in name or "translation" in name:
                    candidates.append(("questionnaire", path))
                elif "codebook" in name:
                    candidates.append(("codebook", path))

        deduped = []
        seen = set()
        for doc_type, path in candidates:
            key = str(path.resolve()).lower()
            if key not in seen:
                deduped.append((doc_type, path))
                seen.add(key)
        return deduped

    def _chunk_text(self, path: Path, doc_type: str, text: str) -> list[EvidenceChunk]:
        if not text.strip():
            return []
        chunks = []
        start = 0
        chunk_id = 1
        while start < len(text):
            end = min(len(text), start + CHUNK_CHARS)
            chunks.append(EvidenceChunk(str(path), doc_type, chunk_id, start, end, text[start:end]))
            if end == len(text):
                break
            start = max(0, end - CHUNK_OVERLAP)
            chunk_id += 1
        return chunks

    def _extract_chunks_parallel(self, chunks: list[EvidenceChunk], use_llm: bool = True) -> list[dict]:
        if not use_llm:
            return [self._fallback_chunk_result(chunk, "large text model unavailable") for chunk in chunks]
        results_by_key = {}
        with ThreadPoolExecutor(max_workers=LLAMA_MAX_WORKERS) as executor:
            futures = {executor.submit(self._extract_chunk, chunk): chunk for chunk in chunks}
            for future in as_completed(futures):
                result = future.result()
                key = (result.get("source_file"), result.get("chunk_id"))
                results_by_key[key] = result
                self._progress(
                    f"Extracted {Path(result.get('source_file', '')).name} chunk {result.get('chunk_id')}: "
                    f"{'ok' if result.get('ok') else 'needs review'}"
                )
        return [results_by_key.get((chunk.source_file, chunk.chunk_id), {}) for chunk in chunks]

    def _fallback_chunk_result(self, chunk: EvidenceChunk, reason: str) -> dict:
        return {
            "ok": True,
            "source_file": chunk.source_file,
            "doc_type": chunk.doc_type,
            "chunk_id": chunk.chunk_id,
            "offsets": [chunk.start, chunk.end],
            "attempt": 0,
            "seconds": 0,
            "facts": self._heuristic_facts(chunk),
            "error": reason,
        }

    def _extract_chunk(self, chunk: EvidenceChunk) -> dict:
        fields = FIELD_GROUPS.get(chunk.doc_type, ALL_FIELDS)
        prompt = self._extraction_prompt(chunk, fields)
        started = time.time()
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with self.global_semaphore:
                    content = self._completion_content(
                        role=ModelRole.LARGE_TEXT,
                        model=self.large_text_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=1800,
                        temperature=1,
                    )
                data = _json_from_text(content)
                if isinstance(data.get("facts"), list):
                    facts = self._normalize_facts(data.get("facts", []), chunk)
                    facts = self._combine_facts(facts, self._heuristic_facts(chunk))
                    return {
                        "ok": True,
                        "source_file": chunk.source_file,
                        "doc_type": chunk.doc_type,
                        "chunk_id": chunk.chunk_id,
                        "offsets": [chunk.start, chunk.end],
                        "attempt": attempt,
                        "seconds": round(time.time() - started, 2),
                        "facts": facts,
                        "error": "",
                    }
                last_error = "invalid JSON extraction response"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                time.sleep(2 * attempt)
        fallback_facts = self._heuristic_facts(chunk)
        if fallback_facts:
            return {
                "ok": True,
                "source_file": chunk.source_file,
                "doc_type": chunk.doc_type,
                "chunk_id": chunk.chunk_id,
                "offsets": [chunk.start, chunk.end],
                "attempt": MAX_RETRIES,
                "seconds": round(time.time() - started, 2),
                "facts": fallback_facts,
                "error": f"llm extraction failed; used deterministic fallback ({last_error})",
            }
        return {
            "ok": False,
            "source_file": chunk.source_file,
            "doc_type": chunk.doc_type,
            "chunk_id": chunk.chunk_id,
            "offsets": [chunk.start, chunk.end],
            "attempt": MAX_RETRIES,
            "seconds": round(time.time() - started, 2),
            "facts": [],
            "error": last_error,
        }

    def _heuristic_facts(self, chunk: EvidenceChunk) -> list[dict]:
        """Extract conservative evidence when the model endpoint fails a chunk."""
        text = chunk.text or ""
        if not text.strip():
            return []

        facts: list[dict] = []
        seen = set()

        def add(field: str, value: str, evidence: str, confidence: str = "low") -> None:
            field = self._map_field(field) if field not in set(ALL_FIELDS) else field
            if field not in set(ALL_FIELDS):
                return
            value = _safe_text(value, 700)
            evidence = _safe_text(evidence, 500)
            key = (field, value.casefold(), evidence.casefold())
            if not value or not evidence or key in seen:
                return
            seen.add(key)
            facts.append(
                {
                    "field": field,
                    "value": value,
                    "evidence": evidence,
                    "confidence": confidence,
                    "source_file": chunk.source_file,
                    "doc_type": chunk.doc_type,
                    "chunk_id": chunk.chunk_id,
                    "offsets": [chunk.start, chunk.end],
                    "extraction_method": "deterministic_fallback",
                }
            )

        lines = [line.strip() for line in text.splitlines() if line.strip()]
        lower = text.lower()

        keyword_fields = [
            ("target_population", ["target population", "voting age", "eligible voters", "eligible population", "resident citizens"]),
            ("eligible_population", ["eligible population", "eligibility", "eligible respondent", "eligible voters"]),
            ("sample_design", ["probability sample", "probability sampling", "probability-based", "simple random sample", "stratified", "multi-stage", "multistage", "cluster sampling", "random sampling"]),
            ("sampling_frame", ["sampling frame", "census", "population register", "registry", "enumeration area", "address list"]),
            ("sampling_stages", ["stage sampling", "sampling stage", "primary sampling unit", "psu"]),
            ("stratification", ["stratified", "stratification", "strata"]),
            ("psu_cluster_details", ["primary sampling unit", "psu", "cluster"]),
            ("respondent_selection", ["kish", "birthday", "respondent selection", "randomly selected respondent"]),
            ("response_rate", ["response rate", "rr1", "rr2", "aapor", "completion rate"]),
            ("fieldwork_dates", ["fieldwork", "interviewing began", "interviewing ended", "survey period", "data collection"]),
            ("mode", ["face-to-face", "capi", "cawi", "cati", "telephone", "web survey", "interview mode"]),
            ("weights", ["weight", "weighted", "post-stratification", "poststratification", "design weight"]),
            ("field_lag", ["field lag", "days after election", "after election"]),
            ("consent_data_deposit", ["data deposit", "consent", "archive"]),
            ("election_date", ["election date", "held on", "general election"]),
            ("election_type", ["presidential election", "legislative election", "parliamentary election", "general election"]),
            ("electoral_system", ["electoral system", "proportional representation", "single member", "district magnitude"]),
            ("turnout", ["turnout", "voter turnout"]),
            ("election_results", ["election results", "vote share", "seat share", "valid votes"]),
            ("parties", ["party", "parties", "political party"]),
            ("leaders", ["leader", "candidate", "presidential candidate"]),
            ("coalitions", ["coalition", "alliance"]),
            ("cses_item_coverage", ["cses", "module", "questionnaire item", "item"]),
            ("vote_choice_questions", ["vote choice", "which party", "party-list vote", "constituency vote", "voted for"]),
            ("party_leader_questions", ["party leader", "leader thermometer", "candidate rating"]),
            ("response_categories", ["response categories", "strongly agree", "strongly disagree", "yes", "no"]),
            ("missing_value_codes", ["missing values", "don't know", "refused", "not applicable", "no answer"]),
            ("variable_labels", ["variable label", "variables"]),
            ("value_labels", ["value label", "value labels", "codes"]),
            ("missing_codes", ["missing code", "missing codes", "dk", "refused"]),
            ("cses_variable_evidence", ["cses variable", "variable"]),
        ]

        for line in lines:
            clean = _safe_text(line, 900)
            clean_lower = clean.lower()
            if len(clean) < 8:
                continue
            for field, keywords in keyword_fields:
                if field == "mode":
                    matched = (
                        re.search(r"\b(?:CAPI|CAWI|CATI)\b", clean, flags=re.IGNORECASE)
                        or any(phrase in clean_lower for phrase in ["face-to-face", "telephone", "web survey", "interview mode", "mode of interview"])
                    )
                else:
                    matched = any(keyword in clean_lower for keyword in keywords)
                if matched:
                    add(field, clean, clean, "low")
                    break

        sample_patterns = [
            r"\b(?:sample size|final sample|completed interviews|interviews completed|respondents)\D{0,40}([0-9][0-9,\.]{2,})\b",
            r"\bN\s*=\s*([0-9][0-9,\.]{2,})\b",
        ]
        for pattern in sample_patterns:
            for match in re.finditer(pattern, text, flags=re.IGNORECASE):
                evidence = self._window(text, match.start(), match.end())
                add("sample_size", match.group(0), evidence, "medium")

        response_rate_pattern = r"\b(?:response rate|RR1|RR2|completion rate)\D{0,40}([0-9]{1,3}(?:\.[0-9]+)?\s*%)"
        for match in re.finditer(response_rate_pattern, text, flags=re.IGNORECASE):
            add("response_rate", match.group(0), self._window(text, match.start(), match.end()), "medium")

        date_pattern = r"\b(?:fieldwork|interviewing|data collection|election date|held on)\D{0,80}(?:\d{1,2}[./ -]\d{1,2}[./ -]\d{2,4}|[A-Z][a-z]+ \d{1,2},? \d{4}|\d{1,2} [A-Z][a-z]+ \d{4})"
        for match in re.finditer(date_pattern, text, flags=re.IGNORECASE):
            field = "election_date" if "election" in match.group(0).lower() or "held on" in match.group(0).lower() else "fieldwork_dates"
            add(field, match.group(0), self._window(text, match.start(), match.end()), "medium")

        cses_codes = re.findall(r"\b(?:CSES\s*)?[A-F]\d{3,4}[A-Z]?\b", text, flags=re.IGNORECASE)
        if cses_codes:
            unique_codes = sorted({code.upper().replace(" ", "") for code in cses_codes})[:80]
            add("cses_item_coverage", f"{len(set(cses_codes))} CSES-style item codes detected: {', '.join(unique_codes[:20])}", " ".join(unique_codes[:40]), "medium")

        return facts

    def _combine_facts(self, primary: list[dict], secondary: list[dict]) -> list[dict]:
        combined = []
        seen = set()
        for fact in primary + secondary:
            key = (
                fact.get("field"),
                fact.get("value", "").casefold(),
                fact.get("evidence", "").casefold(),
                fact.get("source_file"),
                fact.get("chunk_id"),
            )
            if key in seen:
                continue
            seen.add(key)
            combined.append(fact)
        return combined

    def _window(self, text: str, start: int, end: int, radius: int = 180) -> str:
        left = max(0, start - radius)
        right = min(len(text), end + radius)
        return text[left:right]

    def _extraction_prompt(self, chunk: EvidenceChunk, fields: list[str]) -> str:
        field_text = ", ".join(fields)
        focused_text = self._focused_text(chunk)
        return f"""Extract CSES processing evidence from this deposited {chunk.doc_type} chunk.

Source file: {Path(chunk.source_file).name}
Chunk: {chunk.chunk_id}
Offsets: {chunk.start}-{chunk.end}

Fields to look for:
{field_text}

Return JSON only:
{{
  "facts": [
    {{
      "field": "one field name from the list",
      "value": "concise extracted fact",
      "evidence": "short exact quote from the chunk",
      "confidence": "high|medium|low"
    }}
  ]
}}

Rules:
- Extract only facts supported by this chunk.
- Do not infer from general knowledge.
- Include exact evidence quotes.
- Use a field from the list above.
- Return an empty facts array if this chunk has no relevant evidence.

TEXT:
{focused_text}
"""

    def _focused_text(self, chunk: EvidenceChunk, limit: int = 5200) -> str:
        """Keep traceable chunk coverage while sending only relevant lines to the model."""
        terms_by_type = {
            "design_report": [
                "target", "eligible", "sample", "sampling", "probability", "random", "strat",
                "census", "frame", "respondent", "response rate", "fieldwork", "interviewing",
                "mode", "capi", "cawi", "cati", "weight", "deposit", "consent",
            ],
            "macro_report": [
                "election", "turnout", "party", "parties", "leader", "candidate", "coalition",
                "result", "vote", "seat", "electoral", "district", "constituency",
            ],
            "questionnaire": [
                "cses", "vote", "party", "leader", "candidate", "district", "constituency",
                "don't know", "refused", "missing", "q", "module",
            ],
            "codebook": [
                "cses", "variable", "label", "value", "missing", "vote", "party", "leader",
                "district", "weight", "q",
            ],
        }
        terms = terms_by_type.get(chunk.doc_type, ["cses", "sample", "election", "vote", "party", "weight"])
        lines = chunk.text.splitlines()
        selected: list[str] = []
        seen = set()
        for index, line in enumerate(lines):
            clean = line.strip()
            lower = clean.lower()
            if not clean:
                continue
            if any(term in lower for term in terms) or re.search(r"\b[A-F]\d{3,4}[A-Z]?\b", clean, flags=re.IGNORECASE):
                for nearby in range(max(0, index - 1), min(len(lines), index + 2)):
                    nearby_line = lines[nearby].strip()
                    if nearby_line and nearby_line not in seen:
                        seen.add(nearby_line)
                        selected.append(nearby_line)
            if sum(len(item) + 1 for item in selected) >= limit:
                break
        text = "\n".join(selected).strip()
        if not text:
            text = chunk.text[:limit]
        return text[:limit]

    def _normalize_facts(self, facts, chunk: EvidenceChunk) -> list[dict]:
        normalized = []
        if not isinstance(facts, list):
            return normalized
        allowed = set(ALL_FIELDS)
        for item in facts:
            if not isinstance(item, dict):
                continue
            field = str(item.get("field", "")).strip()
            if field not in allowed:
                field = self._map_field(field)
            value = _safe_text(item.get("value", ""), 700)
            evidence = _safe_text(item.get("evidence", ""), 500)
            if not field or not value or value.lower() in {"none", "not specified", "n/a", "not provided"}:
                continue
            normalized.append(
                {
                    "field": field,
                    "value": value,
                    "evidence": evidence,
                    "confidence": str(item.get("confidence", "medium")).strip().lower() or "medium",
                    "source_file": chunk.source_file,
                    "doc_type": chunk.doc_type,
                    "chunk_id": chunk.chunk_id,
                    "offsets": [chunk.start, chunk.end],
                }
            )
        return normalized

    def _map_field(self, field: str) -> str:
        compact = field.lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "sample_frame": "sampling_frame",
            "sampling_method": "sample_design",
            "sampling_design": "sample_design",
            "vote_cast": "vote_choice_questions",
            "election_result": "election_results",
            "leader": "leaders",
            "party": "parties",
        }
        return aliases.get(compact, "other")

    def _merge_facts(self, facts: list[dict]) -> dict:
        merged: dict[str, list[dict]] = {}
        seen = set()
        for fact in facts:
            field = fact.get("field", "other")
            key = (
                field,
                fact.get("source_file"),
                fact.get("chunk_id"),
                fact.get("value", "").casefold(),
                fact.get("evidence", "").casefold(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.setdefault(field, []).append(fact)
        for items in merged.values():
            items.sort(key=self._fact_sort_key)
        return merged

    def _fact_sort_key(self, item: dict) -> tuple:
        field = item.get("field", "")
        doc_type = item.get("doc_type", "")
        confidence_rank = {"high": 0, "medium": 1, "low": 2}.get(item.get("confidence", "medium"), 1)
        preferred_docs = {
            **{field_name: ["design_report", "questionnaire", "codebook"] for field_name in DESIGN_FIELDS},
            **{field_name: ["macro_report", "codebook", "questionnaire"] for field_name in MACRO_FIELDS},
            **{field_name: ["questionnaire", "codebook", "design_report"] for field_name in QUESTIONNAIRE_FIELDS},
            **{field_name: ["codebook", "questionnaire", "design_report"] for field_name in CODEBOOK_FIELDS},
        }
        doc_rank = preferred_docs.get(field, []).index(doc_type) if doc_type in preferred_docs.get(field, []) else 9
        fallback_rank = 1 if item.get("extraction_method") == "deterministic_fallback" else 0
        return (
            doc_rank,
            confidence_rank,
            fallback_rank,
            item.get("source_file", ""),
            item.get("chunk_id", 0),
        )

    def _verify_parallel(self, facts_by_field: dict, missing_fields: list[str]) -> dict:
        groups = {
            "sampling_design": DESIGN_FIELDS,
            "macro_election": MACRO_FIELDS,
            "questionnaire_coverage": QUESTIONNAIRE_FIELDS,
            "codebook_labels": CODEBOOK_FIELDS,
            "missing_fields": missing_fields,
        }
        results = {}
        with ThreadPoolExecutor(max_workers=OSS_MAX_WORKERS) as executor:
            futures = {
                executor.submit(self._verify_group, group, fields, facts_by_field): group
                for group, fields in groups.items()
            }
            for future in as_completed(futures):
                group = futures[future]
                results[group] = future.result()
        return results

    def _verify_group(self, group: str, fields: list[str], facts_by_field: dict) -> dict:
        snippets = []
        for field in fields:
            for fact in facts_by_field.get(field, [])[:5]:
                snippets.append(
                    {
                        "field": field,
                        "value": fact.get("value", ""),
                        "evidence": fact.get("evidence", ""),
                        "source": Path(fact.get("source_file", "")).name,
                        "chunk": fact.get("chunk_id"),
                    }
                )
        prompt = f"""Verify CSES evidence group: {group}.

Evidence:
{json.dumps(snippets, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "status": "supported|needs_review",
  "summary": "short processor-facing summary",
  "contradictions": ["..."],
  "insufficient_fields": ["..."]
}}

Rules:
- Only judge evidence shown here.
- Mark needs_review if evidence is weak, contradictory, or missing.
"""
        started = time.time()
        last_error = ""
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with self.global_semaphore:
                    content = self._completion_content(
                        role=ModelRole.VERIFIER,
                        model=self.agentic_model,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=900,
                        temperature=1,
                    )
                data = _json_from_text(content)
                if data:
                    data["attempt"] = attempt
                    data["seconds"] = round(time.time() - started, 2)
                    return data
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:240]}"
                time.sleep(2 * attempt)
        return {
            "status": "needs_review",
            "summary": f"Verification failed for {group}",
            "contradictions": [],
            "insufficient_fields": fields,
            "error": last_error,
            "attempt": MAX_RETRIES,
            "seconds": round(time.time() - started, 2),
        }

    def _completion_content(self, role: ModelRole, model: str, messages: list[dict], max_tokens: int, temperature: float) -> str:
        api_base = os.getenv("OPENAI_API_BASE", "").strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_base and model.startswith("openai/") and api_key:
            endpoint_model = model.split("/", 1)[1]
            response = requests.post(
                api_base.rstrip("/") + "/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": endpoint_model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                timeout=LLM_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "") or ""

        result = self.runner.complete(
            role,
            messages=messages,
            model_override=model,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=LLM_TIMEOUT_SECONDS,
            retries=0,
            purpose=f"Evidence extraction role {role.value}",
        )
        if result.status != "ok":
            raise RuntimeError(result.error or f"Model task failed for {role.value}")
        return result.content

    def _model_health_check(self, model: str) -> bool:
        try:
            api_base = os.getenv("OPENAI_API_BASE", "").strip()
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            if api_base and model.startswith("openai/") and api_key:
                response = requests.post(
                    api_base.rstrip("/") + "/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model.split("/", 1)[1],
                        "messages": [{"role": "user", "content": "Return JSON only: {\"ok\": true}"}],
                        "max_tokens": 32,
                        "temperature": 0,
                    },
                    timeout=MODEL_HEALTH_TIMEOUT_SECONDS,
                )
                if not response.ok:
                    return False
                return bool(_json_from_text(response.text))
            content = self._completion_content(
                ModelRole.LARGE_TEXT,
                model,
                [{"role": "user", "content": "Return JSON only: {\"ok\": true}"}],
                max_tokens=32,
                temperature=0,
            )
            return bool(_json_from_text(content))
        except Exception:
            return False

    def _litellm_connection_kwargs(self) -> dict:
        kwargs = {}
        api_base = os.getenv("OPENAI_API_BASE", "").strip()
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_base:
            kwargs["api_base"] = api_base
        if api_key:
            kwargs["api_key"] = api_key
        return kwargs

    def _progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)


def summarize_evidence_index(index: dict) -> str:
    diagnostics = index.get("diagnostics", {})
    models = index.get("models", {})
    lines = [
        "Evidence extraction diagnostics:",
        f"- Files scanned: {diagnostics.get('files_scanned', 0)}",
        f"- Chunks processed: {diagnostics.get('chunks_ok', 0)}/{diagnostics.get('chunks_total', 0)}",
        f"- Failed chunks: {diagnostics.get('chunks_failed', 0)}",
        f"- Fields found: {diagnostics.get('fields_found', 0)}",
        f"- Fields missing: {diagnostics.get('fields_missing', 0)}",
        f"- Large text model: {models.get('large_text_extraction', '')}",
        f"- Large text model available: {diagnostics.get('large_text_model_available', 'unknown')}",
        f"- Verification model: {models.get('agentic_verification', '')}",
    ]
    missing = index.get("missing_fields", [])
    if missing:
        lines.append("- Missing after full search: " + ", ".join(missing[:12]))
    return "\n".join(lines)
