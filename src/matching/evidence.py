"""Matching evidence and remote-only item similarity for CSES variable matching.

This module deliberately does not import Harmony or any local transformer
embedding runtime. Harmony is used as a design pattern: compare cleaned
questionnaire items and produce reviewable similarity candidates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from src.ingest.data_loader import DatasetInfo
from src.ingest.doc_parser import DocumentInfo
from src.model_runtime import ModelRole, ModelTaskRunner
from src.standards.schema import SchemaRegistry, SchemaVariable
from src.standards.questionnaire_registry import (
    Module6QuestionnaireRegistry,
    canonical_item_id,
    normalize_item_key,
)


@dataclass
class QuestionnaireItem:
    item_id: str
    text: str
    response_options: list[str] = field(default_factory=list)
    source_file: str = ""
    language: str = ""
    citation: str = ""
    grid: str = ""


@dataclass
class SourceVariableProfile:
    name: str
    label: str = ""
    value_labels: dict[str, str] = field(default_factory=dict)
    sample_values: list[str] = field(default_factory=list)
    dtype: str = ""
    n_unique: int | None = None
    missing_count: int | None = None
    missing_percent: float | None = None
    top_values: list[dict[str, Any]] = field(default_factory=list)
    likely_item_id: str = ""
    source_file: str = ""
    citation: str = ""


@dataclass
class TargetVariableProfile:
    name: str
    description: str
    dependency_class: str
    expected_item_ids: list[str] = field(default_factory=list)
    canonical_item_ids: list[str] = field(default_factory=list)
    canonical_item_type: str = ""
    expected_response_structure: str = ""
    missing_value_policy: str = ""
    required_evidence: list[str] = field(default_factory=list)


class QuestionnaireItemExtractor:
    """Extract structured questionnaire items from already parsed documents."""

    def extract_from_document_info(self, document: DocumentInfo) -> list[QuestionnaireItem]:
        items: list[QuestionnaireItem] = []
        for question in document.questions:
            items.append(
                QuestionnaireItem(
                    item_id=question.code,
                    text=_clean_text(question.text, 1000),
                    response_options=[_clean_text(value, 250) for value in question.response_options],
                    source_file=str(document.file_path),
                    language=document.language_detected or "",
                    citation=f"{Path(document.file_path).name}: {question.code}",
                )
            )
        return _dedupe_items(items)

    def extract_from_narrative_record(self, record: dict[str, Any]) -> list[QuestionnaireItem]:
        items: list[QuestionnaireItem] = []
        for question in record.get("parsed_questions", []) or []:
            if not isinstance(question, dict):
                continue
            code = _normalize_item_id(question.get("code", ""))
            text = _clean_text(question.get("text", ""), 1000)
            if not code or not text:
                continue
            items.append(
                QuestionnaireItem(
                    item_id=code,
                    text=text,
                    response_options=[_clean_text(value, 250) for value in question.get("response_options", []) or []],
                    source_file=str(record.get("path", "")),
                    language=str(record.get("language", "") or ""),
                    citation=f"{record.get('relative_path') or Path(str(record.get('path', ''))).name}: {code}",
                )
            )
        return _dedupe_items(items)


class VariableProfileBuilder:
    """Build source variable profiles with deterministic metadata and summaries."""

    def from_dataset_info(self, dataset: DatasetInfo) -> list[SourceVariableProfile]:
        summaries = _column_summaries(dataset.file_path)
        profiles: list[SourceVariableProfile] = []
        for variable in dataset.variables.values():
            summary = summaries.get(variable.name, {})
            value_labels = {
                _clean_text(key, 80): _clean_text(value, 250)
                for key, value in (variable.value_labels or {}).items()
            }
            profile = SourceVariableProfile(
                name=variable.name,
                label=_clean_text(variable.description, 500),
                value_labels=value_labels,
                sample_values=[_clean_text(value, 120) for value in (variable.sample_values or [])[:10]],
                dtype=variable.dtype or "",
                n_unique=variable.n_unique,
                missing_count=summary.get("missing_count"),
                missing_percent=summary.get("missing_percent"),
                top_values=summary.get("top_values", []),
                likely_item_id=_guess_item_id(variable.name, variable.description or ""),
                source_file=str(dataset.file_path),
                citation=f"{Path(dataset.file_path).name}: {variable.name}",
            )
            profiles.append(profile)
        return profiles

    def from_data_summary_record(self, record: dict[str, Any]) -> list[SourceVariableProfile]:
        profiles: list[SourceVariableProfile] = []
        for variable in record.get("variable_inventory", []) or []:
            if not isinstance(variable, dict) or not variable.get("name"):
                continue
            name = str(variable.get("name"))
            label = _clean_text(variable.get("label") or variable.get("description"), 500)
            profiles.append(
                SourceVariableProfile(
                    name=name,
                    label=label,
                    value_labels={
                        _clean_text(key, 80): _clean_text(value, 250)
                        for key, value in (variable.get("value_labels") or {}).items()
                    },
                    sample_values=[_clean_text(value, 120) for value in (variable.get("sample_values") or [])[:10]],
                    dtype=str(variable.get("dtype") or ""),
                    n_unique=variable.get("n_unique"),
                    likely_item_id=_guess_item_id(name, label),
                    source_file=str(record.get("path", "")),
                    citation=f"{record.get('relative_path') or Path(str(record.get('path', ''))).name}: {name}",
                )
            )
        return profiles


class TargetVariableProfileBuilder:
    """Build CSES target profiles from the schema registry."""

    def __init__(self, registry: SchemaRegistry | None = None):
        self.registry = registry or SchemaRegistry()
        self.questionnaire_registry = Module6QuestionnaireRegistry()

    def build(self) -> list[TargetVariableProfile]:
        return [self.from_schema_variable(variable) for variable in self.registry.variables]

    def from_schema_variable(self, variable: SchemaVariable) -> TargetVariableProfile:
        from src.matching.decision_engine import MODULE6_SOURCE_ALIASES

        aliases = MODULE6_SOURCE_ALIASES.get(variable.name, [])
        hinted = _guess_item_id(variable.name, variable.description)
        registry_items = self.questionnaire_registry.find_for_target(variable.name)
        canonical_ids = [item.item_id for item in registry_items]
        canonical_aliases = [
            alias
            for item in registry_items
            for alias in [item.item_id, *item.aliases]
        ]
        expected = list(dict.fromkeys([*aliases, *canonical_aliases, hinted] if hinted else [*aliases, *canonical_aliases]))
        return TargetVariableProfile(
            name=variable.name,
            description=variable.description,
            dependency_class=variable.dependency_class,
            expected_item_ids=expected,
            canonical_item_ids=canonical_ids,
            canonical_item_type=registry_items[0].item_type if registry_items else _fallback_item_type(variable),
            expected_response_structure=_expected_response_structure(variable),
            missing_value_policy=variable.missing_value_policy,
            required_evidence=list(variable.required_evidence),
        )


class CanonicalItemMatcher:
    """Match canonical registry items to collaborator questionnaire items."""

    def __init__(self, registry: Module6QuestionnaireRegistry | None = None):
        self.registry = registry or Module6QuestionnaireRegistry()

    def match_items(self, collaborator_items: list[QuestionnaireItem]) -> list[dict[str, Any]]:
        by_key: dict[str, QuestionnaireItem] = {}
        for item in collaborator_items:
            by_key[normalize_item_key(item.item_id)] = item
        matches = []
        for canonical in self.registry.items:
            matched = None
            for alias in [canonical.item_id, *canonical.aliases]:
                matched = by_key.get(normalize_item_key(alias))
                if matched:
                    break
            status = "matched_by_item_id" if matched else "not_found"
            matches.append(
                {
                    "canonical_item_id": canonical.item_id,
                    "canonical_item_type": canonical.item_type,
                    "section": canonical.section,
                    "title": canonical.title,
                    "collaborator_item_id": matched.item_id if matched else "",
                    "collaborator_text": matched.text if matched else "",
                    "status": status,
                    "evidence": matched.citation if matched else "",
                }
            )
        return matches


class QuestionnaireToDataLinker:
    """Link questionnaire items to source data variables by item id and labels."""

    def link(self, item_matches: list[dict[str, Any]], source_profiles: list[SourceVariableProfile]) -> list[dict[str, Any]]:
        by_item: dict[str, list[SourceVariableProfile]] = {}
        for profile in source_profiles:
            if profile.likely_item_id:
                by_item.setdefault(normalize_item_key(profile.likely_item_id), []).append(profile)
            by_item.setdefault(normalize_item_key(profile.name), []).append(profile)

        links = []
        for match in item_matches:
            keys = [
                normalize_item_key(match.get("canonical_item_id", "")),
                normalize_item_key(match.get("collaborator_item_id", "")),
            ]
            candidates: list[SourceVariableProfile] = []
            for key in keys:
                candidates.extend(by_item.get(key, []))
            seen: set[str] = set()
            clean_candidates = []
            for candidate in candidates:
                if candidate.name in seen:
                    continue
                seen.add(candidate.name)
                clean_candidates.append(candidate)
            links.append(
                {
                    **match,
                    "source_variable_candidates": [
                        {
                            "name": candidate.name,
                            "label": candidate.label,
                            "evidence": candidate.citation,
                        }
                        for candidate in clean_candidates[:5]
                    ],
                    "status": "linked_to_source_variable" if clean_candidates else match.get("status", "not_found"),
                }
            )
        return links


class RegistryDrivenMatchingEngine:
    """Small facade used by tests and callers that need registry-first evidence."""

    def __init__(self, registry: Module6QuestionnaireRegistry | None = None):
        self.registry = registry or Module6QuestionnaireRegistry()

    def build_links(self, collaborator_items: list[QuestionnaireItem], source_profiles: list[SourceVariableProfile]) -> list[dict[str, Any]]:
        item_matches = CanonicalItemMatcher(self.registry).match_items(collaborator_items)
        return QuestionnaireToDataLinker().link(item_matches, source_profiles)


class ResponseOptionMatcher:
    """CSES-specific response/value compatibility checks."""

    def compare(self, target: TargetVariableProfile, source: SourceVariableProfile, item: QuestionnaireItem | None = None) -> dict[str, Any]:
        source_options = [str(value) for value in (source.value_labels or {}).values()]
        if item and item.response_options:
            source_options.extend(item.response_options)
        option_text = " ".join(source_options).lower()
        flags: list[str] = []
        score = 0.5

        if not source_options and target.dependency_class == "direct_survey_item":
            return {"score": 0.35, "flags": ["response_options_missing"], "evidence": "No source value labels or questionnaire options found."}

        expected = target.expected_response_structure
        if expected == "numeric_scale":
            numeric_values = _numeric_values([*source.value_labels.keys(), *source.sample_values])
            if len(numeric_values) >= 2:
                score = 0.78
            else:
                flags.append("numeric_scale_not_confirmed")
        elif expected == "categorical":
            if source.value_labels:
                score = 0.78
            else:
                flags.append("categorical_labels_missing")
        elif expected == "party_or_vote":
            if any(term in option_text for term in ["party", "candidate", "vote", "list"]):
                score = 0.75
            else:
                flags.append("party_vote_options_not_confirmed")
        else:
            score = 0.65 if source_options else 0.45

        if len(source_options) >= 20 and expected not in {"party_or_vote", "open_numeric"}:
            flags.append("many_response_options")
            score = min(score, 0.65)
        return {"score": round(score, 3), "flags": flags, "evidence": f"{len(source_options)} source response/value option(s) inspected."}


class RemoteItemSimilarityService:
    """Remote-only questionnaire item similarity scorer with deterministic fallback."""

    def __init__(self, working_dir: Path | None = None, runner: ModelTaskRunner | None = None, enabled: bool | None = None):
        self.working_dir = Path(working_dir) if working_dir else None
        self.runner = runner or ModelTaskRunner(self.working_dir)
        if enabled is None:
            enabled = os.getenv("CSES_ENABLE_REMOTE_MATCHING", "true").lower() in {"1", "true", "yes"}
        self.enabled = enabled

    def score_pairs(self, pairs: list[dict[str, str]], max_pairs: int = 120) -> dict[str, float]:
        fallback = {self._key(pair): _lexical_similarity(pair.get("target_text", ""), pair.get("source_text", "")) for pair in pairs}
        if not self.enabled or not pairs:
            return fallback

        compact_pairs = [
            {
                "id": self._key(pair),
                "target": _clean_text(pair.get("target_text", ""), 700),
                "source": _clean_text(pair.get("source_text", ""), 700),
            }
            for pair in pairs[:max_pairs]
            if pair.get("target_text") and pair.get("source_text")
        ]
        if not compact_pairs:
            return fallback

        prompt = {
            "task": "Score whether each source questionnaire/data item matches the CSES target item.",
            "rules": [
                "Return JSON only.",
                "Scores are 0.0 to 1.0.",
                "Judge semantic question meaning, not only variable names.",
                "Do not assume a match when the response scale or concept differs.",
            ],
            "output_schema": {"scores": [{"id": "target||source", "score": 0.0, "reason": "short"}]},
            "pairs": compact_pairs,
        }
        result = self.runner.complete(
            ModelRole.MATCH_FAST,
            messages=[{"role": "user", "content": json.dumps(prompt, ensure_ascii=False)}],
            purpose="Remote questionnaire-item similarity scoring",
            include_shared_context=False,
            max_tokens=2500,
            temperature=0.1,
            timeout=45,
            retries=0,
        )
        if result.status != "ok":
            return fallback
        parsed = _json_from_text(result.content)
        for item in parsed.get("scores", []) if isinstance(parsed, dict) else []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("id", ""))
            if key not in fallback:
                continue
            try:
                fallback[key] = max(0.0, min(1.0, float(item.get("score"))))
            except (TypeError, ValueError):
                continue
        return fallback

    def _key(self, pair: dict[str, str]) -> str:
        return f"{pair.get('target_id', '')}||{pair.get('source_id', '')}"


class MatchingEvidenceBuilder:
    """Create and persist study-specific matching evidence artifacts."""

    def __init__(self, working_dir: Path, registry: SchemaRegistry | None = None):
        self.working_dir = Path(working_dir)
        self.cses_dir = self.working_dir / ".cses"
        self.registry = registry or SchemaRegistry()

    def load(self) -> dict[str, Any]:
        path = self.cses_dir / "matching_evidence.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def build_from_loaded(self, dataset: DatasetInfo, documents: list[DocumentInfo] | None = None) -> dict[str, Any]:
        question_extractor = QuestionnaireItemExtractor()
        items: list[QuestionnaireItem] = []
        for document in documents or []:
            items.extend(question_extractor.extract_from_document_info(document))
        source_profiles = VariableProfileBuilder().from_dataset_info(dataset)
        return self._write(items, source_profiles)

    def build_from_records(self, narrative_docs: list[dict[str, Any]], data_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        question_extractor = QuestionnaireItemExtractor()
        items: list[QuestionnaireItem] = []
        for record in narrative_docs:
            items.extend(question_extractor.extract_from_narrative_record(record))
        source_profiles: list[SourceVariableProfile] = []
        profile_builder = VariableProfileBuilder()
        for record in data_summaries:
            source_profiles.extend(profile_builder.from_data_summary_record(record))
        return self._write(items, source_profiles)

    def _write(self, items: list[QuestionnaireItem], source_profiles: list[SourceVariableProfile]) -> dict[str, Any]:
        self.cses_dir.mkdir(parents=True, exist_ok=True)
        items = _dedupe_items(items)
        source_profiles = _dedupe_profiles(source_profiles)
        target_profiles = TargetVariableProfileBuilder(self.registry).build()
        questionnaire_registry = Module6QuestionnaireRegistry()
        item_matches = CanonicalItemMatcher().match_items(items)
        data_links = QuestionnaireToDataLinker().link(item_matches, source_profiles)
        readiness = self._readiness(items, source_profiles, target_profiles)
        payload = {
            "schema_version": 2,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "registry_item_count": len(questionnaire_registry.items),
            "registry_generated_at": questionnaire_registry.payload.get("generated_at", ""),
            "questionnaire_items": [asdict(item) for item in items],
            "source_variable_profiles": [asdict(item) for item in source_profiles],
            "target_variable_profiles": [asdict(item) for item in target_profiles],
            "canonical_item_matches": item_matches,
            "questionnaire_to_data_links": data_links,
            "matching_readiness": readiness,
        }
        (self.cses_dir / "questionnaire_items.json").write_text(json.dumps(payload["questionnaire_items"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (self.cses_dir / "source_variable_profiles.json").write_text(json.dumps(payload["source_variable_profiles"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (self.cses_dir / "target_variable_profiles.json").write_text(json.dumps(payload["target_variable_profiles"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (self.cses_dir / "canonical_item_matches.json").write_text(json.dumps(payload["canonical_item_matches"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (self.cses_dir / "questionnaire_to_data_links.json").write_text(json.dumps(payload["questionnaire_to_data_links"], indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        (self.cses_dir / "matching_evidence.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return payload

    def _readiness(
        self,
        items: list[QuestionnaireItem],
        source_profiles: list[SourceVariableProfile],
        target_profiles: list[TargetVariableProfile],
    ) -> dict[str, Any]:
        item_ids = {normalize_item_key(item.item_id) for item in items}
        source_ids = {normalize_item_key(profile.likely_item_id) for profile in source_profiles if profile.likely_item_id}
        missing_direct_items = []
        for target in target_profiles:
            if target.dependency_class != "direct_survey_item":
                continue
            expected = [normalize_item_key(item) for item in target.expected_item_ids if item]
            if expected and not any(item in item_ids or item in source_ids for item in expected):
                canonical = "/".join(target.canonical_item_ids or target.expected_item_ids[:3])
                missing_direct_items.append(f"{target.name}: expected {canonical}")
        return {
            "questionnaire_items": len(items),
            "source_variables": len(source_profiles),
            "target_variables": len(target_profiles),
            "variables_without_labels": [profile.name for profile in source_profiles if not profile.label][:100],
            "variables_without_response_options": [profile.name for profile in source_profiles if not profile.value_labels][:100],
            "missing_direct_items": missing_direct_items[:120],
        }


def build_similarity_pairs(matching_evidence: dict[str, Any], limit_per_target: int = 8) -> list[dict[str, str]]:
    targets = matching_evidence.get("target_variable_profiles", []) or []
    sources = matching_evidence.get("source_variable_profiles", []) or []
    items = {normalize_item_key(str(item.get("item_id", ""))): item for item in matching_evidence.get("questionnaire_items", []) or []}
    pairs: list[dict[str, str]] = []
    for target in targets:
        if target.get("dependency_class") != "direct_survey_item":
            continue
        target_text = _target_text(target)
        scored_sources = []
        expected = {normalize_item_key(str(item)) for item in target.get("expected_item_ids", []) if item}
        for source in sources:
            likely = normalize_item_key(str(source.get("likely_item_id", "")))
            item = items.get(likely, {})
            source_text = _source_text(source, item)
            lexical = _lexical_similarity(target_text, source_text)
            if likely and likely in expected:
                lexical = max(lexical, 0.9)
            if lexical >= 0.12 or (likely and expected and likely[:3] in {value[:3] for value in expected}):
                scored_sources.append((lexical, source, source_text))
        for _, source, source_text in sorted(scored_sources, key=lambda row: row[0], reverse=True)[:limit_per_target]:
            pairs.append(
                {
                    "target_id": str(target.get("name", "")),
                    "source_id": str(source.get("name", "")),
                    "target_text": target_text,
                    "source_text": source_text,
                }
            )
    return pairs


def _column_summaries(path: Path) -> dict[str, dict[str, Any]]:
    try:
        import pandas as pd
    except Exception:
        return {}
    try:
        suffix = path.suffix.lower()
        if suffix == ".dta":
            df = pd.read_stata(path, convert_categoricals=False)
        elif suffix in {".sav", ".por"}:
            try:
                import pyreadstat
            except Exception:
                return {}
            df, _ = pyreadstat.read_sav(str(path), apply_value_formats=False)
        elif suffix in {".xlsx", ".xls"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, nrows=20000)
    except Exception:
        return {}
    summaries: dict[str, dict[str, Any]] = {}
    total = max(1, len(df))
    for column in df.columns:
        series = df[column]
        missing = int(series.isna().sum())
        counts = series.dropna().value_counts().head(8)
        summaries[str(column)] = {
            "missing_count": missing,
            "missing_percent": round((missing / total) * 100, 2),
            "top_values": [
                {"value": _clean_text(index, 120), "count": int(value)}
                for index, value in counts.items()
            ],
        }
    return summaries


def _expected_response_structure(variable: SchemaVariable) -> str:
    name = variable.name
    desc = variable.description.lower()
    if name.startswith(("F5", "F6")) or any(term in desc for term in ["party", "vote", "candidate"]):
        return "party_or_vote"
    if any(term in desc for term in ["age", "year", "month", "day", "number", "district"]):
        return "open_numeric"
    if variable.dependency_class == "direct_survey_item":
        return "categorical"
    return "metadata"


def _fallback_item_type(variable: SchemaVariable) -> str:
    desc = variable.description.lower()
    if variable.dependency_class == "derived_metadata":
        return "admin_supplied_variable"
    if variable.dependency_class == "district_input":
        return "district_item"
    if variable.dependency_class == "macro_or_party_input":
        return "party_vote_item"
    if any(term in desc for term in ["party", "vote", "candidate", "election", "turnout", "leader"]):
        return "party_vote_item"
    if variable.section == "demographics":
        return "demographic_coding_standard"
    return "core_question"


def _target_text(target: dict[str, Any]) -> str:
    return " ".join(
        str(value)
        for value in [
            target.get("name"),
            target.get("description"),
            " ".join(target.get("expected_item_ids", []) or []),
            target.get("expected_response_structure"),
        ]
        if value
    )


def _source_text(source: dict[str, Any], item: dict[str, Any] | None = None) -> str:
    labels = " ".join(str(value) for value in (source.get("value_labels") or {}).values())
    options = " ".join(str(value) for value in ((item or {}).get("response_options") or []))
    return " ".join(
        str(value)
        for value in [
            source.get("name"),
            source.get("label"),
            source.get("likely_item_id"),
            labels,
            (item or {}).get("text"),
            options,
        ]
        if value
    )


def _guess_item_id(name: str, label: str = "") -> str:
    text = f"{name} {label}"
    for pattern in [
        r"\b([AQD]\d{1,2}[A-Za-z]{0,3}\d?(?:[-_][A-Za-z0-9]+)?)\b",
        r"\b(Q\d{1,2}[A-Za-z0-9_]*)\b",
    ]:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return canonical_item_id(match.group(1).replace("_", "-"))
    return ""


def _normalize_item_id(value: str) -> str:
    return canonical_item_id(value)


def _dedupe_items(items: list[QuestionnaireItem]) -> list[QuestionnaireItem]:
    seen: dict[str, QuestionnaireItem] = {}
    for item in items:
        key = f"{item.item_id.upper()}::{item.text[:80].lower()}"
        if key not in seen:
            seen[key] = item
    return list(seen.values())


def _dedupe_profiles(profiles: list[SourceVariableProfile]) -> list[SourceVariableProfile]:
    seen: dict[str, SourceVariableProfile] = {}
    for profile in profiles:
        if profile.name and profile.name not in seen:
            seen[profile.name] = profile
    return list(seen.values())


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "with", "for", "from", "variable", "component", "cses", "module"}
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in stop
    }


def _lexical_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return round(len(left_tokens & right_tokens) / math.sqrt(len(left_tokens) * len(right_tokens)), 3)


def _numeric_values(values: list[Any]) -> list[float]:
    numeric = []
    for value in values:
        try:
            numeric.append(float(str(value).strip()))
        except (TypeError, ValueError):
            continue
    return numeric


def _clean_text(value: Any, limit: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    text = text.encode("cp1252", errors="replace").decode("cp1252")
    return text[:limit] if limit else text


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
