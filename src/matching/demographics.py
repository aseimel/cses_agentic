"""Demographic recoding assessment for CSES matching.

Demographic variables are not ordinary questionnaire matches. This module
identifies the supplied source concept and classifies the recoding work needed
before CSES syntax can be generated.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.standards.schema import DEFAULT_WIKI_ROOT, SchemaRegistry, SchemaVariable
from src.standards.questionnaire_registry import Module6QuestionnaireRegistry


REGISTRY_PATH = DEFAULT_WIKI_ROOT / "patterns" / "demographic_variable_registry.json"


@dataclass(frozen=True)
class DemographicVariableContract:
    target_variable: str
    canonical_item_ids: list[str]
    concept: str
    source_aliases: list[str]
    format_class: str
    recoding_action: str
    processor_review_required: bool
    notes: str = ""


@dataclass
class DemographicRecodingAssessment:
    target_variable: str
    description: str
    concept: str
    source_variable: str = ""
    source_format: str = ""
    recoding_action: str = ""
    recoding_plan_type: str = "manual_processor_decision"
    status: str = "needs_processor_review"
    confidence: str = "processor_review"
    canonical_item_ids: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    value_map: dict[str, Any] = field(default_factory=dict)
    missing_map: dict[str, Any] = field(default_factory=dict)
    processor_review_required: bool = True
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DemographicRecodingDossier:
    target_variable: str
    description: str
    source_variable: str
    concept: str
    source_format: str
    recoding_action: str
    proposed_plan_type: str
    target_standard: dict[str, Any] = field(default_factory=dict)
    source_evidence: dict[str, Any] = field(default_factory=dict)
    draft_recode_table: list[dict[str, Any]] = field(default_factory=list)
    missing_value_treatment: list[dict[str, Any]] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)
    processor_decision_needed: str = ""
    coding_note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DemographicRecodingDecision:
    target_variable: str
    source_variable: str
    plan_type: str
    approved: bool = False
    value_map: dict[str, Any] = field(default_factory=dict)
    missing_map: dict[str, Any] = field(default_factory=dict)
    processor_note: str = ""
    log_note: str = ""
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class DemographicVariableRegistry:
    """Load the wiki-backed demographic matching contract."""

    def __init__(self, wiki_root: Path = DEFAULT_WIKI_ROOT):
        self.wiki_root = Path(wiki_root)
        self.path = self.wiki_root / "patterns" / "demographic_variable_registry.json"
        self.payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.variables = [DemographicVariableContract(**item) for item in self.payload.get("variables", [])]
        self.by_target = {item.target_variable: item for item in self.variables}

    def validate(self, schema: SchemaRegistry | None = None) -> list[str]:
        schema = schema or SchemaRegistry(self.wiki_root)
        issues: list[str] = []
        schema_demographics = [item.name for item in schema.variables if item.section == "demographics" or item.name.startswith("F2")]
        contract_names = {item.target_variable for item in self.variables}
        for name in schema_demographics:
            if name not in contract_names:
                issues.append(f"Missing demographic contract for {name}")
        for item in self.variables:
            if not item.canonical_item_ids:
                issues.append(f"Missing canonical item id for {item.target_variable}")
            if not item.format_class:
                issues.append(f"Missing source format class for {item.target_variable}")
            if not item.recoding_action:
                issues.append(f"Missing recoding action for {item.target_variable}")
        return issues


class DemographicRecodingAssessmentEngine:
    """Create demographic-specific matching and recoding assessments."""

    def __init__(
        self,
        registry: DemographicVariableRegistry | None = None,
        schema: SchemaRegistry | None = None,
    ):
        self.registry = registry or DemographicVariableRegistry()
        self.schema = schema or SchemaRegistry()

    def assess(
        self,
        matching_evidence: dict[str, Any] | None = None,
        source_contexts: list[dict[str, Any]] | None = None,
    ) -> list[DemographicRecodingAssessment]:
        matching_evidence = matching_evidence or {}
        source_profiles = _source_profiles(matching_evidence, source_contexts or [])
        source_by_name = {profile["name"].upper(): profile for profile in source_profiles if profile.get("name")}
        assessments: list[DemographicRecodingAssessment] = []
        for schema_var in self.schema.variables:
            if schema_var.section != "demographics" and not schema_var.name.startswith("F2"):
                continue
            contract = self.registry.by_target.get(schema_var.name)
            if not contract:
                assessments.append(self._missing_contract(schema_var))
                continue
            source = self._select_source(contract, source_profiles, source_by_name)
            assessments.append(self._assessment_for(schema_var, contract, source))
        return assessments

    def write_artifacts(self, working_dir: Path, assessments: list[DemographicRecodingAssessment]) -> Path:
        path = Path(working_dir) / ".cses" / "demographic_recoding_assessments.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_count": len(assessments),
            "ready_or_reviewable_count": sum(1 for item in assessments if item.status != "missing_source"),
            "needs_processor_review_count": sum(1 for item in assessments if item.processor_review_required),
            "assessments": [item.to_dict() for item in assessments],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def _select_source(
        self,
        contract: DemographicVariableContract,
        source_profiles: list[dict[str, Any]],
        source_by_name: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        for alias in [*contract.canonical_item_ids, *contract.source_aliases]:
            hit = source_by_name.get(alias.upper())
            if hit:
                return hit
        best: tuple[float, dict[str, Any] | None] = (0.0, None)
        target_tokens = _tokens(" ".join([contract.concept, contract.format_class, *contract.source_aliases]))
        for profile in source_profiles:
            text = " ".join(
                str(value)
                for value in [
                    profile.get("name"),
                    profile.get("label"),
                    " ".join(str(v) for v in (profile.get("value_labels") or {}).values()),
                ]
                if value
            )
            score = _token_overlap(target_tokens, _tokens(text))
            likely = str(profile.get("likely_item_id", ""))
            if likely and any(likely.upper() == item.upper() for item in contract.canonical_item_ids):
                score = max(score, 0.95)
            if score > best[0]:
                best = (score, profile)
        return best[1] if best[0] >= 0.28 else None

    def _assessment_for(
        self,
        schema_var: SchemaVariable,
        contract: DemographicVariableContract,
        source: dict[str, Any] | None,
    ) -> DemographicRecodingAssessment:
        if contract.recoding_action == "derive_birth_generation":
            return DemographicRecodingAssessment(
                target_variable=schema_var.name,
                description=schema_var.description,
                concept=contract.concept,
                source_variable="F2001_Y",
                source_format=contract.format_class,
                recoding_action=contract.recoding_action,
                recoding_plan_type="derived_generation",
                status="proposed_derivation",
                confidence="high",
                canonical_item_ids=contract.canonical_item_ids,
                evidence=["Generated from approved birth year using CSES generation thresholds."],
                processor_review_required=False,
                notes=contract.notes,
            )
        if contract.recoding_action == "derive_age_from_birth_year":
            source_name = source.get("name") if source else "F2001_Y"
            return DemographicRecodingAssessment(
                target_variable=schema_var.name,
                description=schema_var.description,
                concept=contract.concept,
                source_variable=str(source_name),
                source_format=contract.format_class,
                recoding_action=contract.recoding_action,
                recoding_plan_type="derived_age",
                status="proposed_derivation",
                confidence="medium",
                canonical_item_ids=contract.canonical_item_ids,
                evidence=["Age can be derived from election year and approved birth year."],
                processor_review_required=True,
                notes=contract.notes,
            )
        if not source:
            return DemographicRecodingAssessment(
                target_variable=schema_var.name,
                description=schema_var.description,
                concept=contract.concept,
                source_format=contract.format_class,
                recoding_action=contract.recoding_action,
                recoding_plan_type="missing_not_collected",
                status="missing_source",
                confidence="low",
                canonical_item_ids=contract.canonical_item_ids,
                warnings=["No source variable candidate found for this demographic concept."],
                processor_review_required=True,
                notes=contract.notes,
            )
        source_name = str(source.get("name", ""))
        observed = _observed_values(source)
        plan_type, value_map, missing_map, warnings = self._classify_transform(contract, observed)
        status = "proposed_demographic_plan" if plan_type in {"direct_copy", "offset_transform", "recode", "derive_from_source"} else "needs_processor_review"
        confidence = "high" if not warnings and not contract.processor_review_required else "medium"
        return DemographicRecodingAssessment(
            target_variable=schema_var.name,
            description=schema_var.description,
            concept=contract.concept,
            source_variable=source_name,
            source_format=contract.format_class,
            recoding_action=contract.recoding_action,
            recoding_plan_type=plan_type,
            status=status,
            confidence=confidence,
            canonical_item_ids=contract.canonical_item_ids,
            evidence=[
                f"Source variable {source_name} selected for canonical item {', '.join(contract.canonical_item_ids)}.",
                f"Observed source values: {', '.join(str(value) for value in observed[:12])}.",
            ],
            warnings=warnings,
            value_map=value_map,
            missing_map=missing_map,
            processor_review_required=contract.processor_review_required or bool(warnings),
            notes=contract.notes,
        )

    def _classify_transform(
        self,
        contract: DemographicVariableContract,
        observed: list[Any],
    ) -> tuple[str, dict[str, Any], dict[str, Any], list[str]]:
        action = contract.recoding_action
        observed_set = {_value_key(value) for value in observed}
        warnings: list[str] = []
        if action == "direct_copy":
            return "direct_copy", {}, {}, warnings
        if action == "gender_recode_review":
            if {"1", "2"}.issubset(observed_set):
                return "offset_transform", {}, {}, ["Confirm source coding before applying 1=male, 2=female to 0/1."]
            return "recode", {}, {}, ["Gender categories need processor review."]
        if action == "yes_no_recode_review":
            if {"1", "2"}.issubset(observed_set):
                return "recode", {"2": 0}, {}, ["Confirm 1=yes and 2=no before applying no=0."]
            if {"1", "5"}.issubset(observed_set):
                return "recode", {"5": 0}, {}, ["Confirm 1=yes and 5=no before applying no=0."]
            return "recode", {}, {}, ["Binary yes/no source format needs review."]
        if action == "direct_copy_or_review":
            return "direct_copy", {}, {}, ["Confirm source categories already match CSES coding."]
        if action == "direct_copy_or_crosswalk":
            return "direct_copy", {}, {"": _missing_value_for(contract.target_variable)}, ["Confirm source is already in CSES/standard coding or supply a crosswalk."]
        if action == "direct_copy_with_large_missing":
            return "recode", {"99": 99999999}, {}, ["Confirm source missing value before applying CSES large missing code."]
        if action == "derive_income_quintiles":
            return "crosswalk_required", {}, {}, ["Income quintiles require a reviewed derivation from source income categories/distribution."]
        if action in {
            "crosswalk_required",
            "frequency_crosswalk_required",
            "country_crosswalk_required",
            "language_crosswalk_required",
            "urban_rural_crosswalk_required",
            "district_review_required",
            "crosswalk_or_missing_review",
            "direct_copy_or_missing_review",
        }:
            return "crosswalk_required", {}, {}, ["A processor-approved demographic crosswalk or missing/not-collected decision is required."]
        return "manual_processor_decision", {}, {}, ["Unsupported demographic recoding action."]

    def _missing_contract(self, schema_var: SchemaVariable) -> DemographicRecodingAssessment:
        return DemographicRecodingAssessment(
            target_variable=schema_var.name,
            description=schema_var.description,
            concept="",
            status="missing_contract",
            confidence="low",
            warnings=["No demographic contract exists in cses_wiki."],
            processor_review_required=True,
        )


class DemographicReferenceDecisionLearner:
    """Infer processor-approved value maps from a processed reference dataset.

    This is for development benchmarks only. Runtime processing still requires
    processor review for crosswalks and judgment-sensitive decisions.
    """

    def infer_maps(
        self,
        source_df: pd.DataFrame,
        final_df: pd.DataFrame,
        assessments: list[DemographicRecodingAssessment],
        source_id: str,
        final_id: str,
    ) -> list[DemographicRecodingAssessment]:
        left = source_df.copy()
        right = final_df.copy()
        left["_cses_key"] = pd.to_numeric(left[source_id], errors="coerce")
        right["_cses_key"] = pd.to_numeric(right[final_id], errors="coerce")
        merged = left.merge(right, on="_cses_key", suffixes=("_source", "_final"))
        updated: list[DemographicRecodingAssessment] = []
        for assessment in assessments:
            item = DemographicRecodingAssessment(**assessment.to_dict())
            target = item.target_variable
            source = item.source_variable
            if target not in merged.columns:
                updated.append(item)
                continue
            if item.recoding_plan_type in {"derived_age", "derived_generation"} or not source or source not in merged.columns:
                item.processor_review_required = False
                item.status = "reference_approved"
                updated.append(item)
                continue
            value_map = _consistent_value_map(merged[source], merged[target])
            if value_map is not None:
                item.value_map = value_map
                item.recoding_plan_type = "direct_copy" if _identity_map(value_map) else "recode"
                item.status = "reference_approved"
                item.confidence = "high"
                item.processor_review_required = False
                item.warnings = []
                item.evidence.append("Processor decision simulated from processed reference benchmark.")
            updated.append(item)
        return updated


class DemographicRecodingDossierBuilder:
    """Build coder-facing demographic dossiers from matching-stage evidence."""

    def __init__(self, questionnaire_registry: Module6QuestionnaireRegistry | None = None):
        self.questionnaire_registry = questionnaire_registry or Module6QuestionnaireRegistry()

    def build(
        self,
        assessments: list[DemographicRecodingAssessment],
        matching_evidence: dict[str, Any] | None = None,
    ) -> list[DemographicRecodingDossier]:
        source_lookup = {
            item.get("name"): item
            for item in (matching_evidence or {}).get("source_variable_profiles", []) or []
            if isinstance(item, dict) and item.get("name")
        }
        dossiers: list[DemographicRecodingDossier] = []
        for assessment in assessments:
            source = source_lookup.get(assessment.source_variable, {})
            target_standard = self._target_standard(assessment)
            draft_table, missing_treatment = self._draft_table(assessment, source, target_standard)
            gaps = self._evidence_gaps(assessment, source, target_standard)
            dossiers.append(
                DemographicRecodingDossier(
                    target_variable=assessment.target_variable,
                    description=assessment.description,
                    source_variable=assessment.source_variable,
                    concept=assessment.concept,
                    source_format=assessment.source_format,
                    recoding_action=assessment.recoding_action,
                    proposed_plan_type=assessment.recoding_plan_type,
                    target_standard=target_standard,
                    source_evidence={
                        "label": source.get("label", ""),
                        "value_labels": source.get("value_labels", {}) or {},
                        "observed_values": _observed_values(source),
                        "top_values": source.get("top_values", []) or [],
                        "missing_percent": source.get("missing_percent"),
                        "citation": source.get("citation", ""),
                    },
                    draft_recode_table=draft_table,
                    missing_value_treatment=missing_treatment,
                    evidence_gaps=gaps,
                    processor_decision_needed=self._decision_prompt(assessment, gaps),
                    coding_note=assessment.notes,
                )
            )
        return dossiers

    def write(self, working_dir: Path, dossiers: list[DemographicRecodingDossier]) -> Path:
        path = Path(working_dir) / ".cses" / "demographic_recoding_dossiers.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dossier_count": len(dossiers),
            "dossiers": [item.to_dict() for item in dossiers],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return path

    def _target_standard(self, assessment: DemographicRecodingAssessment) -> dict[str, Any]:
        items = [
            self.questionnaire_registry.get(item_id)
            for item_id in assessment.canonical_item_ids
            if item_id
        ]
        items = [item for item in items if item]
        response_options = []
        missing_codes = []
        notes = []
        titles = []
        for item in items:
            response_options.extend(item.response_options)
            missing_codes.extend(item.missing_codes)
            if item.notes:
                notes.append(item.notes)
            if item.title:
                titles.append(item.title)
        return {
            "canonical_item_ids": assessment.canonical_item_ids,
            "title": "; ".join(titles),
            "response_options": response_options,
            "missing_codes": list(dict.fromkeys(missing_codes)),
            "notes": " ".join(notes)[:1500],
        }

    def _draft_table(
        self,
        assessment: DemographicRecodingAssessment,
        source: dict[str, Any],
        target_standard: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        values = _observed_values(source)
        for key in assessment.value_map.keys():
            if key and key not in {_value_key(value) for value in values}:
                values.append(key)
        labels = source.get("value_labels", {}) or {}
        draft: list[dict[str, Any]] = []
        for value in values[:200]:
            key = _value_key(value)
            target = ""
            rationale = "Needs processor decision."
            if assessment.recoding_plan_type in {"direct_copy", "direct_copy_or_review"}:
                target = key
                rationale = "Draft direct copy; confirm source categories match CSES."
            elif assessment.recoding_plan_type == "offset_transform" and assessment.target_variable == "F2002":
                if key in {"1", "2"}:
                    target = str(int(key) - 1)
                    rationale = "Draft common CSES gender transform after confirming source labels."
            elif assessment.value_map and key in assessment.value_map:
                target = str(assessment.value_map[key])
                rationale = "Draft mapped value from demographic assessment."
            draft.append(
                {
                    "source_value": key,
                    "source_label": labels.get(key, ""),
                    "proposed_target_value": target,
                    "target_label": _target_label(target_standard, target),
                    "rationale": rationale,
                    "requires_processor_approval": True,
                }
            )
        missing = [
            {
                "source_value": key,
                "target_value": value,
                "rationale": "Draft missing-value treatment from demographic assessment.",
            }
            for key, value in assessment.missing_map.items()
        ]
        return draft, missing

    def _evidence_gaps(
        self,
        assessment: DemographicRecodingAssessment,
        source: dict[str, Any],
        target_standard: dict[str, Any],
    ) -> list[str]:
        gaps = list(assessment.warnings)
        if not assessment.source_variable:
            gaps.append("No source variable has been identified.")
        if source and not source.get("value_labels"):
            gaps.append("Source value labels are not available; inspect questionnaire/codebook evidence before approving.")
        if not target_standard.get("response_options") and assessment.recoding_plan_type in {"crosswalk_required", "recode"}:
            gaps.append("The target standard requires notes or a coding table rather than simple response options.")
        return list(dict.fromkeys(gaps))

    def _decision_prompt(self, assessment: DemographicRecodingAssessment, gaps: list[str]) -> str:
        if assessment.recoding_plan_type == "crosswalk_required":
            return "Review source categories and approve a category-to-CSES crosswalk before syntax generation."
        if assessment.recoding_plan_type in {"derived_age", "derived_generation"}:
            return "Confirm the derivation rule and the source birth-year/election-year values."
        if assessment.status == "missing_source":
            return "Confirm whether the item was not collected or whether another source file contains it."
        if gaps:
            return "Review the listed evidence gaps before approving the recode."
        return "Confirm the proposed recode before syntax generation."


class DemographicRecodingDecisionStore:
    """Persist approved demographic decisions for later Stata generation."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.path = self.working_dir / ".cses" / "demographic_recoding_decisions.json"

    def load(self) -> dict[str, DemographicRecodingDecision]:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        decisions = {}
        for item in payload.get("decisions", []) or []:
            if isinstance(item, dict) and item.get("target_variable"):
                decision = DemographicRecodingDecision(**item)
                decisions[decision.target_variable] = decision
        return decisions

    def write(self, decisions: list[DemographicRecodingDecision]) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "decision_count": len(decisions),
            "approved_count": sum(1 for item in decisions if item.approved),
            "decisions": [item.to_dict() for item in decisions],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return self.path

    def write_from_assessments(
        self,
        assessments: list[DemographicRecodingAssessment],
        approved: bool = False,
    ) -> Path:
        decisions = [
            DemographicRecodingDecision(
                target_variable=item.target_variable,
                source_variable=item.source_variable,
                plan_type=item.recoding_plan_type,
                approved=approved and not item.processor_review_required,
                value_map=item.value_map,
                missing_map=item.missing_map,
                processor_note="",
                log_note=item.notes,
                evidence=item.evidence,
            )
            for item in assessments
        ]
        return self.write(decisions)


class DemographicDataGenerator:
    """Apply demographic assessments to a data frame for benchmark validation."""

    def generate(
        self,
        source_df: pd.DataFrame,
        assessments: list[DemographicRecodingAssessment],
        election_year: int,
    ) -> pd.DataFrame:
        output = pd.DataFrame(index=source_df.index)
        assessment_lookup = {item.target_variable: item for item in assessments}
        for item in assessments:
            target = item.target_variable
            if item.recoding_plan_type == "derived_age":
                birth = output.get("F2001_Y")
                if birth is None and item.source_variable in source_df:
                    birth = pd.to_numeric(source_df[item.source_variable], errors="coerce")
                output[target] = _derive_age(birth, election_year)
            elif item.recoding_plan_type == "derived_generation":
                birth = output.get("F2001_Y")
                if birth is None:
                    birth_item = assessment_lookup.get("F2001_Y")
                    if birth_item and birth_item.source_variable in source_df:
                        birth = pd.to_numeric(source_df[birth_item.source_variable], errors="coerce")
                output[target] = _derive_generation(target, birth)
            elif item.source_variable and item.source_variable in source_df:
                source = source_df[item.source_variable]
                if item.value_map:
                    output[target] = source.map(lambda value: item.value_map.get(_value_key(value), value))
                elif item.recoding_plan_type == "offset_transform":
                    output[target] = pd.to_numeric(source, errors="coerce") - 1
                else:
                    output[target] = source
            else:
                missing = _missing_value_for(target)
                output[target] = missing
        return output


def demographic_assessment_summary(assessments: list[DemographicRecodingAssessment]) -> dict[str, int]:
    return {
        "total": len(assessments),
        "ready": sum(1 for item in assessments if not item.processor_review_required and item.status != "missing_source"),
        "needs_review": sum(1 for item in assessments if item.processor_review_required),
        "missing_source": sum(1 for item in assessments if item.status == "missing_source"),
        "crosswalk_required": sum(1 for item in assessments if item.recoding_plan_type == "crosswalk_required"),
    }


def audit_demographic_registry() -> list[str]:
    return DemographicVariableRegistry().validate()


def _source_profiles(matching_evidence: dict[str, Any], source_contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    profiles = [
        dict(item)
        for item in matching_evidence.get("source_variable_profiles", []) or []
        if isinstance(item, dict) and item.get("name")
    ]
    existing = {item["name"] for item in profiles}
    for source in source_contexts:
        name = source.get("name")
        if name and name not in existing:
            profiles.append(
                {
                    "name": name,
                    "label": source.get("description", ""),
                    "value_labels": source.get("value_labels", {}) or {},
                    "sample_values": source.get("sample_values", []) or [],
                    "top_values": [],
                    "likely_item_id": "",
                }
            )
    return profiles


def _observed_values(source: dict[str, Any]) -> list[Any]:
    values: list[Any] = []
    for item in source.get("sample_values", []) or []:
        values.append(item)
    for item in source.get("top_values", []) or []:
        if isinstance(item, dict) and "value" in item:
            values.append(item["value"])
    for key in (source.get("value_labels") or {}).keys():
        values.append(key)
    clean = []
    seen = set()
    for value in values:
        key = _value_key(value)
        if key and key not in seen:
            seen.add(key)
            clean.append(value)
    return clean


def _consistent_value_map(source: pd.Series, target: pd.Series) -> dict[str, Any] | None:
    mapping: dict[str, Any] = {}
    frame = pd.DataFrame({"source": source, "target": target})
    for value, group in frame.groupby("source", dropna=False):
        targets = group["target"].dropna().unique().tolist()
        if not targets:
            continue
        if len(targets) != 1:
            return None
        mapping[_value_key(value)] = _plain_value(targets[0])
    return mapping


def _identity_map(mapping: dict[str, Any]) -> bool:
    for key, value in mapping.items():
        if _value_key(value) != key:
            return False
    return True


def _derive_age(birth_year: pd.Series | None, election_year: int) -> pd.Series:
    if birth_year is None:
        return pd.Series(dtype="float64")
    birth = pd.to_numeric(birth_year, errors="coerce")
    age = election_year - birth
    age = age.where(birth < 9997)
    age = age.mask(birth == 9997, 9997)
    age = age.mask(birth == 9998, 9998)
    age = age.mask(birth == 9999, 9999)
    return age


def _derive_generation(target: str, birth_year: pd.Series | None) -> pd.Series:
    if birth_year is None:
        return pd.Series(dtype="float64")
    birth = pd.to_numeric(birth_year, errors="coerce")
    result = pd.Series(0, index=birth.index, dtype="float64")
    ranges = {
        "F2001_GG": (None, 1927),
        "F2001_GS": (1928, 1945),
        "F2001_GBB": (1946, 1964),
        "F2001_GX": (1965, 1980),
        "F2001_GY": (1981, 1996),
        "F2001_GZ": (1997, None),
    }
    low, high = ranges.get(target, (None, None))
    valid = birth < 9997
    in_range = valid
    if low is not None:
        in_range &= birth >= low
    if high is not None:
        in_range &= birth <= high
    result.loc[in_range] = 1
    result.loc[birth > 9996] = 9
    return result


def _missing_value_for(target: str) -> int:
    if target in {"F2001_Y", "F2001_A"}:
        return 9999
    if target == "F2021":
        return 99
    if target in {"F2007", "F2013", "F2014", "F2015", "F2017"}:
        return 999
    if target == "F2019":
        return 99999
    if target == "F2010_2":
        return 99999999
    return 9


def _target_label(target_standard: dict[str, Any], value: str) -> str:
    if value in {"", None}:
        return ""
    normalized = _value_key(value)
    for option in target_standard.get("response_options", []) or []:
        if _value_key(option.get("code")) == normalized:
            return str(option.get("label", ""))
    return ""


def _tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if len(token) > 2 and token not in {"the", "and", "with", "for", "source"}
    }


def _token_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / math.sqrt(len(left) * len(right))


def _value_key(value: Any) -> str:
    if pd.isna(value):
        return ""
    try:
        number = float(str(value).strip())
        if number.is_integer():
            return str(int(number))
        return str(number)
    except (TypeError, ValueError):
        return str(value).strip()


def _plain_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    try:
        number = float(value)
        if number.is_integer():
            return int(number)
        return number
    except (TypeError, ValueError):
        return value
