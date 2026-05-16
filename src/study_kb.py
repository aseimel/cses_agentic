"""Study-specific knowledge base for CSES processing.

The CSES wiki stores general standards. This module builds a compact,
source-backed knowledge base for one deposited election study.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import litellm

from src.ingest.data_loader import DataLoader
from src.ingest.doc_parser import DocumentParser
from src.matching.llm_matcher import CSES_TARGET_VARIABLES
from src.model_runtime import ModelRole, ModelTaskRunner
from src.settings import DEFAULT_STUDY_KB_MODEL, apply_settings_to_environment

litellm.drop_params = True

NARRATIVE_SUFFIXES = {".pdf", ".docx", ".txt", ".md", ".rtf"}
DATA_SUFFIXES = {".dta", ".sav", ".por", ".csv", ".tsv", ".xlsx", ".xls", ".json", ".parquet", ".pq"}
MAX_ONE_SHOT_CHARS = 650_000

REFERENCE_OUTPUT_PARTS = {
    ".cses",
    "benchmark_report",
    "final dataset",
    "data_checks",
    "data checks",
    "checks",
    "labels",
    "frequencies",
    "frequency",
    "documentation",
    "old",
    "_old",
}

REFERENCE_OUTPUT_NAME_PREFIXES = (
    "cses-m6_micro_",
    "cses-m6_log-file_",
    "cses-m6_label",
    "cses-m6_checks",
)

HISTORICAL_REFERENCE_TERMS = (
    "module 5",
    "module_5",
    "module-5",
    "comparison with module 5",
)


@dataclass
class StudyKBStatus:
    status: str
    message: str
    path: str = ""
    stale: bool = False


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


def _safe_ascii(text: Any, limit: int | None = None) -> str:
    value = " ".join(str(text or "").split())
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
        value = value.replace(source, target)
    value = value.encode("cp1252", errors="replace").decode("cp1252")
    return value[:limit] if limit else value


def is_reference_or_historical_source(path: Path, working_dir: Path) -> bool:
    path = Path(path)
    working_dir = Path(working_dir)
    try:
        rel_parts = path.relative_to(working_dir).parts
    except ValueError:
        rel_parts = path.parts
    lowered_parts = [part.casefold() for part in rel_parts]
    lowered_name = path.name.casefold()
    lowered_text = " ".join(lowered_parts)

    if any(part in REFERENCE_OUTPUT_PARTS for part in lowered_parts):
        return True
    if any(term in lowered_text for term in HISTORICAL_REFERENCE_TERMS):
        return True
    if lowered_name.endswith((".log", ".smcl")):
        return True
    if lowered_name.startswith(REFERENCE_OUTPUT_NAME_PREFIXES):
        return True
    return False


class StudyKnowledgeBase:
    """Load and query a study KB from .cses artifacts."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.cses_dir = self.working_dir / ".cses"
        self.json_path = self.cses_dir / "study_kb.json"
        self.toon_path = self.cses_dir / "study_kb.toon"
        self.sources_path = self.cses_dir / "study_kb_sources.json"
        self.diagnostics_path = self.cses_dir / "model_diagnostics.json"
        self.payload = self._load_json(self.json_path)

    def exists(self) -> bool:
        return bool(self.payload)

    def status(self) -> StudyKBStatus:
        if not self.exists():
            return StudyKBStatus("missing", "Study knowledge base has not been built.")
        stale = self.is_stale()
        status = "stale" if stale else self.payload.get("status", "current")
        return StudyKBStatus(
            status=status,
            message=f"Study knowledge base is {status}.",
            path=str(self.json_path),
            stale=stale,
        )

    def is_stale(self) -> bool:
        if not self.exists():
            return True
        manifest = self._load_json(self.sources_path).get("sources", [])
        for source in manifest:
            path = Path(source.get("path", ""))
            if not path.exists():
                return True
            try:
                stat = path.stat()
            except OSError:
                return True
            if stat.st_size != source.get("size") or int(stat.st_mtime) != source.get("mtime"):
                return True
        return False

    def compact_context(self, max_chars: int = 12000) -> str:
        if self.toon_path.exists():
            return self.toon_path.read_text(encoding="utf-8", errors="replace")[:max_chars]
        if not self.exists():
            return ""
        return json.dumps(self.payload, ensure_ascii=False)[:max_chars]

    def get_field(self, field: str) -> list[dict]:
        return (self.payload.get("fields") or {}).get(field, [])

    def missing_fields(self) -> list[str]:
        return self.payload.get("missing_fields") or []

    def contradictions(self) -> list[str]:
        return self.payload.get("contradictions") or []

    def summary_lines(self) -> list[str]:
        if not self.exists():
            return ["Study KB: missing"]
        diagnostics = self.payload.get("diagnostics", {})
        lines = [
            f"Study KB: {self.status().status}",
            f"Model: {self.payload.get('model', '')}",
            f"Narrative files: {diagnostics.get('narrative_files', 0)}",
            f"Data files summarized: {diagnostics.get('data_files', 0)}",
            f"Estimated input tokens: {diagnostics.get('estimated_input_tokens', 0):,}",
            f"Missing fields: {len(self.missing_fields())}",
            f"Contradictions: {len(self.contradictions())}",
        ]
        return lines

    def _load_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


class StudyKnowledgeBaseBuilder:
    """Build a compact, source-backed KB for one study."""

    def __init__(self, working_dir: Path, model: str | None = None, progress_callback=None):
        self.working_dir = Path(working_dir)
        self.cses_dir = self.working_dir / ".cses"
        settings = apply_settings_to_environment()
        self.model = model or os.getenv("CSES_STUDY_KB_MODEL") or settings.get("CSES_STUDY_KB_MODEL") or DEFAULT_STUDY_KB_MODEL
        self.runner = ModelTaskRunner(self.working_dir)
        self.progress_callback = progress_callback

    def build(self, state=None, force: bool = False) -> dict:
        kb = StudyKnowledgeBase(self.working_dir)
        if kb.exists() and not force and not kb.is_stale():
            return kb.payload

        self.cses_dir.mkdir(parents=True, exist_ok=True)
        self._progress("Collecting study documents for knowledge base...")
        narrative_docs = self._collect_narrative_documents(state)
        data_summaries = self._collect_data_summaries(state)
        source_manifest = self._source_manifest([item["path"] for item in narrative_docs] + [item["path"] for item in data_summaries])

        prompt = self._build_prompt(narrative_docs, data_summaries)
        estimated_tokens = max(1, len(prompt) // 4)
        diagnostics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "model": self.model,
            "narrative_files": len(narrative_docs),
            "data_files": len(data_summaries),
            "narrative_chars": sum(len(item.get("text", "")) for item in narrative_docs),
            "prompt_chars": len(prompt),
            "estimated_input_tokens": estimated_tokens,
            "mode": "one_shot" if len(prompt) <= MAX_ONE_SHOT_CHARS else "truncated_one_shot",
            "failed_files": [
                {"path": item["path"], "errors": item.get("errors", [])}
                for item in narrative_docs + data_summaries
                if item.get("errors")
            ],
        }

        self._progress("Reviewing study materials with the selected model...")
        payload = self._call_model(prompt)
        if not payload:
            payload = self._fallback_payload(narrative_docs, data_summaries)
            diagnostics["model_error"] = "Model returned no parseable JSON; deterministic fallback used."

        payload = self._normalize_payload(payload, diagnostics)
        try:
            from src.matching.evidence import MatchingEvidenceBuilder

            matching_evidence = MatchingEvidenceBuilder(self.working_dir).build_from_records(narrative_docs, data_summaries)
            diagnostics["matching_evidence"] = {
                "questionnaire_items": len(matching_evidence.get("questionnaire_items", [])),
                "source_variable_profiles": len(matching_evidence.get("source_variable_profiles", [])),
                "target_variable_profiles": len(matching_evidence.get("target_variable_profiles", [])),
            }
        except Exception as exc:
            diagnostics["matching_evidence_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        self._write_artifacts(payload, source_manifest, diagnostics)
        if state is not None:
            self.update_state(state, payload)
        return payload

    def update_state(self, state, payload: dict) -> None:
        state.study_kb_status = payload.get("status", "current")
        state.study_kb_path = str(self.cses_dir / "study_kb.json")
        state.study_kb_updated_at = payload.get("generated_at", "")
        state.study_kb_source_manifest = str(self.cses_dir / "study_kb_sources.json")
        state.study_kb_model = payload.get("model", self.model)
        state.study_kb_missing_fields = payload.get("missing_fields", [])
        state.study_kb_contradictions = payload.get("contradictions", [])

    def _collect_narrative_documents(self, state=None) -> list[dict]:
        parser = DocumentParser()
        paths = []

        def add(value):
            if not value:
                return
            path = Path(value)
            if path.exists() and path.is_file() and path.suffix.lower() in NARRATIVE_SUFFIXES:
                paths.append(path)

        if state is not None:
            add(getattr(state, "design_report_file", None))
            add(getattr(state, "codebook_file", None))
            for path in getattr(state, "questionnaire_files", []) or []:
                add(path)

        for root_name in ("emails", "E-mails", "micro", "macro", "Election Results"):
            root = self.working_dir / root_name
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in NARRATIVE_SUFFIXES and not self._is_reference_or_historical_source(path):
                    paths.append(path)

        docs = []
        seen = set()
        for path in paths:
            key = str(path.resolve()).lower()
            if key in seen or self._is_reference_or_historical_source(path):
                continue
            seen.add(key)
            parsed = parser.parse(path)
            text = parsed.full_text if parsed else ""
            docs.append(
                {
                    "path": str(path),
                    "relative_path": self._relative(path),
                    "doc_type": self._classify_document(path),
                    "chars": len(text),
                    "questions": len(parsed.questions) if parsed else 0,
                    "parsed_questions": [
                        {
                            "code": question.code,
                            "text": question.text,
                            "response_options": question.response_options,
                        }
                        for question in (parsed.questions if parsed else [])
                    ],
                    "errors": parsed.parse_errors if parsed else ["parse failed"],
                    "text": _safe_ascii(text),
                }
            )
        return docs

    def _collect_data_summaries(self, state=None) -> list[dict]:
        paths = []

        def add(value):
            if not value:
                return
            path = Path(value)
            if path.exists() and path.is_file() and path.suffix.lower() in DATA_SUFFIXES:
                paths.append(path)

        if state is not None:
            add(getattr(state, "data_file", None))
        for root_name in ("emails", "E-mails", "micro", "macro", "Election Results"):
            root = self.working_dir / root_name
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if (
                    path.is_file()
                    and path.suffix.lower() in DATA_SUFFIXES
                    and not self._is_internal_artifact(path)
                    and not self._is_reference_or_historical_source(path)
                ):
                    paths.append(path)

        summaries = []
        seen = set()
        loader = DataLoader()
        targets = {name.upper() for name in CSES_TARGET_VARIABLES}
        for path in paths:
            key = str(path.resolve()).lower()
            if key in seen or self._is_internal_artifact(path) or self._is_reference_or_historical_source(path):
                continue
            seen.add(key)
            errors = []
            dataset = None
            try:
                dataset = loader.load(path)
            except Exception as exc:
                errors.append(str(exc))
            if not dataset:
                summaries.append(
                    {
                        "path": str(path),
                        "relative_path": self._relative(path),
                        "doc_type": "data",
                        "errors": errors or ["could not load data file"],
                    }
                )
                continue
            variables = list(dataset.variables.values())
            direct_cses = [var.name for var in variables if var.name.upper() in targets]
            summaries.append(
                {
                    "path": str(path),
                    "relative_path": self._relative(path),
                    "doc_type": "data",
                    "file_format": dataset.file_format,
                    "rows": dataset.n_rows,
                    "variables": dataset.n_variables,
                    "metadata_quality": dataset.metadata_quality,
                    "direct_cses_variables": direct_cses[:120],
                    "variable_inventory": [
                        {
                            "name": var.name,
                            "label": _safe_ascii(var.description, 180),
                            "dtype": var.dtype,
                            "n_unique": var.n_unique,
                            "sample_values": [_safe_ascii(value, 80) for value in (var.sample_values or [])[:5]],
                            "value_labels": {
                                _safe_ascii(key, 40): _safe_ascii(value, 100)
                                for key, value in list((var.value_labels or {}).items())[:12]
                            },
                        }
                        for var in variables[:500]
                    ],
                    "errors": errors,
                }
            )
        return summaries

    def _build_prompt(self, narrative_docs: list[dict], data_summaries: list[dict]) -> str:
        narrative_payload = [
            {
                "source_id": f"S{i + 1}",
                "path": item["relative_path"],
                "doc_type": item["doc_type"],
                "chars": item["chars"],
                "questions": item["questions"],
                "text": item["text"],
            }
            for i, item in enumerate(narrative_docs)
        ]
        data_payload = [
            {
                "source_id": f"D{i + 1}",
                "path": item["relative_path"],
                "doc_type": item.get("doc_type", "data"),
                "file_format": item.get("file_format"),
                "rows": item.get("rows"),
                "variables": item.get("variables"),
                "metadata_quality": item.get("metadata_quality"),
                "direct_cses_variables": item.get("direct_cses_variables", []),
                "variable_inventory": item.get("variable_inventory", []),
                "errors": item.get("errors", []),
            }
            for i, item in enumerate(data_summaries)
        ]
        expected_items = list(CSES_TARGET_VARIABLES.keys())
        prompt = {
            "task": "Build a compact source-backed CSES study knowledge base.",
            "rules": [
                "Use only the provided sources and deterministic data summaries.",
                "Do not infer unsupported facts.",
                "For every extracted fact include citations with source_id and short quote or variable evidence.",
                "If evidence is missing, list only exact missing study-design facts or exact missing CSES questionnaire items.",
                "Do not treat absence of a local-variable-to-CSES-F-code mapping as missing evidence; CSES variable mapping is done by the processor/workflow.",
                "Do not ask for collaborator questions. Use processor_review_items for items the processor should inspect first.",
                "Core eligibility depends on whether the study is a probability sample; mode is important documentation but not the core eligibility criterion.",
                "Return compact JSON only. Minimal prose.",
                "Do not create collaborator questions. Only identify missing or ambiguous evidence for processor review.",
            ],
            "required_fields": [
                "target_population",
                "eligible_population",
                "sample_design",
                "probability_sample_status",
                "probability_sample_evidence",
                "sample_size",
                "response_rate",
                "fieldwork_dates",
                "mode",
                "weights",
                "field_lag",
                "cses_item_coverage",
                "missing_cses_items",
                "macro_election_context",
                "party_vote_evidence",
                "district_evidence",
                "missing_value_codes",
                "variable_label_evidence",
            ],
            "output_schema": {
                "summary": "one paragraph",
                "fields": {
                    "field_name": [
                        {
                            "value": "compact value",
                            "confidence": "high|medium|low",
                            "citations": [{"source_id": "S1", "evidence": "short quote"}],
                        }
                    ]
                },
                "missing_fields": ["study-design field names or exact CSES questionnaire items missing after full search; never CSES F-code mapping"],
                "contradictions": ["contradictions with source ids"],
                "processor_review_items": ["items the processor should decide before collaborator contact"],
                "source_notes": [{"source_id": "S1", "note": "short note"}],
            },
            "expected_cses_items": expected_items,
            "narrative_sources": narrative_payload,
            "data_summaries": data_payload,
        }
        text = json.dumps(prompt, ensure_ascii=False)
        if len(text) <= MAX_ONE_SHOT_CHARS:
            return text
        # Preserve all data summaries and truncate narrative text proportionally.
        budget = max(50_000, MAX_ONE_SHOT_CHARS - len(json.dumps(data_payload, ensure_ascii=False)) - 50_000)
        per_doc = max(5000, budget // max(1, len(narrative_payload)))
        for item in narrative_payload:
            item["text"] = item["text"][:per_doc]
            item["truncated"] = True
        prompt["narrative_sources"] = narrative_payload
        prompt["truncation_note"] = "Narrative text was truncated to fit the selected model context."
        return json.dumps(prompt, ensure_ascii=False)

    def _call_model(self, prompt: str) -> dict:
        try:
            result = self.runner.complete(
                ModelRole.STUDY_KB,
                messages=[{"role": "user", "content": prompt}],
                purpose="Build compact source-backed study knowledge base",
                include_shared_context=False,
                temperature=1,
                max_tokens=6000,
                timeout=60,
                retries=0,
            )
            self.model = result.model
            if result.status == "ok":
                return _json_from_text(result.content)
            return {}
        except Exception as exc:
            self._progress(f"Study KB model call failed: {type(exc).__name__}")
            return {}

    def _is_reference_or_historical_source(self, path: Path) -> bool:
        return is_reference_or_historical_source(path, self.working_dir)

    def _fallback_payload(self, narrative_docs: list[dict], data_summaries: list[dict]) -> dict:
        fields: dict[str, list[dict]] = {}

        def add(field: str, value: Any, source_id: str, evidence: str, confidence: str = "low") -> None:
            if not value:
                return
            fields.setdefault(field, []).append(
                {
                    "value": _safe_ascii(value, 500),
                    "confidence": confidence,
                    "citations": [{"source_id": source_id, "evidence": _safe_ascii(evidence, 300)}],
                }
            )

        for i, item in enumerate(data_summaries, start=1):
            source_id = f"D{i}"
            if item.get("rows") is not None:
                add("sample_size", f"{item.get('rows')} data rows", source_id, item.get("relative_path", ""), "medium")
            direct = item.get("direct_cses_variables") or []
            if direct:
                add("cses_item_coverage", f"{len(direct)} direct CSES variables detected", source_id, ", ".join(direct[:40]), "medium")

        for i, item in enumerate(narrative_docs, start=1):
            source_id = f"S{i}"
            text_lower = item.get("text", "").lower()
            if "probability sample" in text_lower or "random sampling" in text_lower or "stratified" in text_lower:
                add("probability_sample_evidence", "Sampling document contains probability/random/stratification evidence", source_id, item.get("relative_path", ""), "low")
        return {
            "summary": "Deterministic fallback study KB. Processor review required because the model did not return parseable JSON.",
            "fields": fields,
            "missing_fields": [],
            "contradictions": [],
            "processor_review_items": ["Review fallback KB before collaborator contact."],
            "source_notes": [],
        }

    def _normalize_payload(self, payload: dict, diagnostics: dict) -> dict:
        fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
        normalized_fields = {}
        for field, values in fields.items():
            if isinstance(values, dict):
                values = [values]
            if not isinstance(values, list):
                continue
            clean_values = []
            for value in values[:25]:
                if not isinstance(value, dict):
                    continue
                citations = value.get("citations") if isinstance(value.get("citations"), list) else []
                clean_values.append(
                    {
                        "value": _safe_ascii(value.get("value"), 700),
                        "confidence": str(value.get("confidence", "medium")).lower(),
                        "citations": [
                            {
                                "source_id": _safe_ascii(citation.get("source_id"), 40),
                                "evidence": _safe_ascii(citation.get("evidence"), 500),
                            }
                            for citation in citations[:8]
                            if isinstance(citation, dict)
                        ],
                    }
                )
            if clean_values:
                normalized_fields[_safe_ascii(field, 120)] = clean_values

        generated_at = datetime.now(timezone.utc).isoformat()
        missing = [_safe_ascii(item, 180) for item in payload.get("missing_fields", []) if str(item).strip()]
        contradictions = [_safe_ascii(item, 300) for item in payload.get("contradictions", []) if str(item).strip()]
        review_items = [_safe_ascii(item, 300) for item in payload.get("processor_review_items", []) if str(item).strip()]
        return {
            "status": "current" if not contradictions else "needs_review",
            "generated_at": generated_at,
            "model": self.model,
            "summary": _safe_ascii(payload.get("summary"), 1500),
            "fields": normalized_fields,
            "missing_fields": missing,
            "contradictions": contradictions,
            "processor_review_items": review_items,
            "source_notes": payload.get("source_notes", []) if isinstance(payload.get("source_notes"), list) else [],
            "diagnostics": diagnostics,
        }

    def _write_artifacts(self, payload: dict, source_manifest: dict, diagnostics: dict) -> None:
        (self.cses_dir / "study_kb.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.cses_dir / "study_kb_sources.json").write_text(json.dumps(source_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.cses_dir / "model_diagnostics.json").write_text(json.dumps({"study_kb": diagnostics}, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.cses_dir / "study_kb.toon").write_text(self._to_toon(payload), encoding="utf-8")

    def _to_toon(self, payload: dict) -> str:
        lines = [
            f"study_kb|status={payload.get('status')}|model={payload.get('model')}|updated={payload.get('generated_at')}",
            f"summary|{payload.get('summary', '')}",
        ]
        for field, values in sorted((payload.get("fields") or {}).items()):
            for index, value in enumerate(values, start=1):
                citations = ";".join(
                    f"{citation.get('source_id')}:{citation.get('evidence')}"
                    for citation in value.get("citations", [])[:3]
                )
                lines.append(
                    f"field|{field}|{index}|conf={value.get('confidence')}|value={value.get('value')}|cite={citations}"
                )
        for item in payload.get("missing_fields", []):
            lines.append(f"missing|{item}")
        for item in payload.get("contradictions", []):
            lines.append(f"contradiction|{item}")
        for item in payload.get("processor_review_items", []):
            lines.append(f"review|{item}")
        return "\n".join(lines) + "\n"

    def _source_manifest(self, paths: list[str]) -> dict:
        sources = []
        seen = set()
        for raw_path in paths:
            path = Path(raw_path)
            try:
                resolved = path.resolve()
            except Exception:
                continue
            key = str(resolved).lower()
            if key in seen or not path.exists():
                continue
            seen.add(key)
            stat = path.stat()
            sources.append(
                {
                    "path": str(path),
                    "relative_path": self._relative(path),
                    "size": stat.st_size,
                    "mtime": int(stat.st_mtime),
                }
            )
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "sources": sources}

    def _classify_document(self, path: Path) -> str:
        name = path.name.lower()
        if "design" in name:
            return "design_report"
        if "macro" in name:
            return "macro_report"
        if "codebook" in name:
            return "codebook"
        if "questionnaire" in name or "survey" in name or "translation" in name:
            return "questionnaire"
        if "email" in name:
            return "email"
        if "log" in name:
            return "processing_log"
        return "document"

    def _is_internal_artifact(self, path: Path) -> bool:
        parts = {part.lower() for part in path.parts}
        if ".cses" in parts or "__pycache__" in parts:
            return True
        return path.name.startswith(".") or path.name.lower() in {
            "state.json",
            "evidence_index.json",
            "study_kb.json",
            "study_kb_sources.json",
            "model_diagnostics.json",
        }

    def _relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.working_dir))
        except ValueError:
            return str(path)

    def _progress(self, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(message)
