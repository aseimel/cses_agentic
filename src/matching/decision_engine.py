"""Reviewable matching decisions for CSES target variables."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.matching.evidence import (
    QuestionnaireItem,
    ResponseOptionMatcher,
    SourceVariableProfile,
    TargetVariableProfile,
    _lexical_similarity,
)
from src.standards.questionnaire_registry import normalize_item_key
from src.standards.schema import SchemaRegistry, SchemaVariable
from src.standards.administrative import AdministrativeVariablePlan
from src.matching.demographics import DemographicRecodingAssessment
from src.matching.party_order import is_party_order_dependent_variable


MODULE6_SOURCE_ALIASES = {
    "F1016_1": ["mode"],
    "F1019_M": ["A4a"],
    "F1019_D": ["A4b"],
    "F1019_Y": ["A4c"],
    "F1101_2": ["A5"],
    "F2001_Y": ["D01b"],
    "F2002": ["D02"],
    "F2003": ["D03"],
    "F2004": ["D04"],
    "F2005": ["D05"],
    "F2006": ["D06"],
    "F2007": ["D07"],
    "F2008": ["D07a"],
    "F2009": ["D08"],
    "F2010_2": ["D09"],
    "F2011": ["D10"],
    "F2012": ["D11"],
    "F2015": ["D14"],
    "F2016": ["D15"],
    "F2017": ["D16"],
    "F2018": ["D17"],
    "F2019": ["D18"],
    "F2020": ["D19"],
    "F3001": ["Q01"],
    "F3002_1": ["Q02a"],
    "F3002_2": ["Q02b"],
    "F3002_3": ["Q02c"],
    "F3002_4": ["Q02d"],
    "F3002_5": ["Q02e"],
    "F3002_6_1": ["Q02f"],
    "F3002_6_2": ["Q02g"],
    "F3003": ["Q03"],
    "F3004_1": ["Q04a"],
    "F3004_2": ["Q04b"],
    "F3004_3": ["Q04c"],
    "F3004_4": ["Q04d"],
    "F3005_1": ["Q05a"],
    "F3005_2": ["Q05b"],
    "F3005_3": ["Q05c"],
    "F3006": ["Q06"],
    "F3007_1": ["Q07a"],
    "F3007_2": ["Q07b"],
    "F3007_3": ["Q07c"],
    "F3007_4": ["Q07d"],
    "F3007_5": ["Q07e"],
    "F3007_6": ["Q07f"],
    "F3007_7": ["Q07g"],
    "F3008_1": ["Q08a"],
    "F3008_2": ["Q08b"],
    "F3009": ["Q09"],
    "F3010_LH": ["Q10LHa"],
    "F3011_LH_PL": ["Q10LHb"],
    "F3011_LH_PF": ["Q10LHd"],
    "F3012_1": ["Q11a"],
    "F3012_2": ["Q11b"],
    "F3012_3": ["Q11c"],
    "F3013": ["Q12"],
    "F3014": ["Q13"],
    "F3015_LH": ["Q14a"],
    "F3016_LH_PL": ["Q14b"],
    "F3017": ["Q15"],
    "F3018_A": ["Q16a"],
    "F3018_B": ["Q16b"],
    "F3018_C": ["Q16c"],
    "F3018_D": ["Q16d"],
    "F3018_E": ["Q16e"],
    "F3018_F": ["Q16f"],
    "F3018_G": ["Q16g"],
    "F3018_H": ["Q16h"],
    "F3018_I": ["Q16i"],
    "F3019_A": ["Q17a"],
    "F3019_B": ["Q17b"],
    "F3019_C": ["Q17c"],
    "F3019_D": ["Q17d"],
    "F3019_E": ["Q17e"],
    "F3019_F": ["Q17f"],
    "F3019_G": ["Q17g"],
    "F3019_H": ["Q17h"],
    "F3019_I": ["Q17i"],
    "F3020_A": ["Q18a"],
    "F3020_B": ["Q18b"],
    "F3020_C": ["Q18c"],
    "F3020_D": ["Q18d"],
    "F3020_E": ["Q18e"],
    "F3020_F": ["Q18f"],
    "F3020_G": ["Q18g"],
    "F3020_H": ["Q18h"],
    "F3020_I": ["Q18i"],
    "F3020_R": ["Q19"],
    "F3022": ["Q22"],
    "F3023_1": ["Q23a"],
    "F3023_2": ["Q23b"],
    "F3023_3": ["Q23c"],
    "F3023_4": ["Q23d"],
    "F3024": ["Q24"],
    "F3025_1": ["Q25a"],
    "F3025_2": ["Q25b"],
    "F3026": ["Q26a"],
    "F3027": ["Q26b"],
    "F3028_1": ["Q27a"],
    "F3028_2": ["Q27b"],
    "F3028_3": ["Q27c"],
    "F3028_4": ["Q27d"],
}


@dataclass
class SourceCandidate:
    source_variable: str
    score: float
    evidence: list[str] = field(default_factory=list)
    conflict_flags: list[str] = field(default_factory=list)


@dataclass
class MatchingDecision:
    target_variable: str
    description: str
    status: str
    source_variable: str = ""
    confidence: str = "processor_review"
    candidates: list[SourceCandidate] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    conflict_flags: list[str] = field(default_factory=list)
    processor_verification_required: bool = True
    dependency_class: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        data = asdict(self)
        data["candidates"] = [asdict(item) for item in self.candidates]
        return data


class MatchingDecisionEngine:
    """Combine deterministic candidates, LLM proposals, and schema constraints."""

    def __init__(self, registry: SchemaRegistry | None = None):
        self.registry = registry or SchemaRegistry()

    def decide(
        self,
        source_contexts: list[dict[str, Any]],
        llm_proposals: list[Any] | None = None,
        study_kb: dict | None = None,
        matching_evidence: dict | None = None,
        remote_similarity_scores: dict[str, float] | None = None,
        administrative_plans: list[AdministrativeVariablePlan] | None = None,
        demographic_assessments: list[DemographicRecodingAssessment] | None = None,
        party_order_approved: bool = False,
        party_order_summary: dict[str, Any] | None = None,
    ) -> list[MatchingDecision]:
        source_lookup = {str(item.get("name", "")): item for item in source_contexts if item.get("name")}
        proposal_lookup = self._proposal_lookup(llm_proposals or [])
        matching_evidence = matching_evidence or {}
        remote_similarity_scores = remote_similarity_scores or {}
        administrative_lookup = {
            plan.target_variable: plan for plan in (administrative_plans or [])
        }
        demographic_lookup = {
            item.target_variable: item for item in (demographic_assessments or [])
        }
        target_profiles = {
            item.get("name"): item
            for item in matching_evidence.get("target_variable_profiles", []) or []
            if isinstance(item, dict)
        }
        decisions: list[MatchingDecision] = []

        for schema_var in self.registry.variables:
            profile = target_profiles.get(schema_var.name, {})
            if is_party_order_dependent_variable(
                schema_var.name,
                description=schema_var.description,
                dependency_class=schema_var.dependency_class,
                item_type=str(profile.get("canonical_item_type") or ""),
            ):
                if not party_order_approved:
                    decisions.append(self._awaiting_party_order_decision(schema_var, party_order_summary or {}))
                    continue
                if schema_var.dependency_class == "macro_or_party_input":
                    decisions.append(self._party_order_generated_decision(schema_var, party_order_summary or {}))
                    continue
                context_decision = self._party_context_decision(schema_var, party_order_summary or {})
                if context_decision:
                    decisions.append(context_decision)
                    continue
            if schema_var.dependency_class == "derived_metadata":
                decisions.append(self._derived_decision(schema_var, administrative_lookup.get(schema_var.name)))
                continue
            if schema_var.section == "demographics" or schema_var.name.startswith("F2"):
                decisions.append(self._demographic_decision(schema_var, demographic_lookup.get(schema_var.name)))
                continue
            if schema_var.dependency_class in {"macro_or_party_input", "district_input"}:
                decisions.append(self._external_decision(schema_var))
                continue

            candidates = self._deterministic_candidates(
                schema_var,
                source_contexts,
                matching_evidence=matching_evidence,
                remote_similarity_scores=remote_similarity_scores,
            )
            proposal = proposal_lookup.get(schema_var.name)
            if proposal:
                proposal_source = str(proposal.get("source_variable") or proposal.get("source") or "")
                if proposal_source and proposal_source not in {"NOT_FOUND", "ERROR", "NO_CONSENSUS"}:
                    proposal_candidate = SourceCandidate(
                        source_variable=proposal_source,
                        score=max(float(proposal.get("confidence_score", 0.0) or 0.0), 0.55),
                        evidence=[str(proposal.get("reasoning", "LLM proposed this source variable."))[:500]],
                    )
                    candidates = self._merge_candidate(candidates, proposal_candidate)

            candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
            if party_order_approved and is_party_order_dependent_variable(
                schema_var.name,
                description=schema_var.description,
                dependency_class=schema_var.dependency_class,
                item_type=str(profile.get("canonical_item_type") or ""),
            ):
                candidates = self._merge_party_specific_candidates(schema_var, candidates, source_contexts)
            if not candidates:
                party_context_decision = self._party_context_decision(schema_var, party_order_summary or {})
                if party_order_approved and party_context_decision:
                    decisions.append(party_context_decision)
                    continue
                decisions.append(
                    MatchingDecision(
                        target_variable=schema_var.name,
                        description=schema_var.description,
                        status="blocked_for_processor_review",
                        confidence="low",
                        evidence=_kb_evidence(schema_var.name, study_kb),
                        dependency_class=schema_var.dependency_class,
                        notes="No source candidate found after deterministic and model-assisted matching.",
                    )
                )
                continue

            best = candidates[0]
            conflict_flags = list(best.conflict_flags)
            if best.source_variable not in source_lookup:
                conflict_flags.append("source_variable_not_in_selected_data")
            if len(candidates) > 1 and candidates[1].score >= best.score - 0.05:
                conflict_flags.append("near_tie_candidates")
            confidence = "high" if best.score >= 0.82 and not conflict_flags else "medium" if best.score >= 0.55 else "low"
            decisions.append(
                MatchingDecision(
                    target_variable=schema_var.name,
                    description=schema_var.description,
                    status="proposed_match" if confidence != "low" else "blocked_for_processor_review",
                    source_variable=best.source_variable,
                    confidence=confidence,
                    candidates=candidates[:5],
                    evidence=best.evidence + _kb_evidence(schema_var.name, study_kb),
                    conflict_flags=conflict_flags,
                    processor_verification_required=True,
                    dependency_class=schema_var.dependency_class,
                    notes="Candidate requires processor verification before final code generation.",
                )
            )

        self._mark_source_reuse_conflicts(decisions)
        return decisions

    def write_artifacts(self, working_dir: Path, decisions: list[MatchingDecision]) -> tuple[Path, Path]:
        cses_dir = Path(working_dir) / ".cses"
        cses_dir.mkdir(parents=True, exist_ok=True)
        candidates_path = cses_dir / "matching_candidates.json"
        decisions_path = cses_dir / "matching_decisions.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_count": len(decisions),
            "decisions": [item.to_dict() for item in decisions],
        }
        decisions_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        candidates_path.write_text(
            json.dumps(
                {
                    "generated_at": payload["generated_at"],
                    "targets": [
                        {
                            "target_variable": item.target_variable,
                            "candidates": [asdict(candidate) for candidate in item.candidates],
                        }
                        for item in decisions
                    ],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return candidates_path, decisions_path

    def _deterministic_candidates(
        self,
        schema_var: SchemaVariable,
        source_contexts: list[dict[str, Any]],
        matching_evidence: dict | None = None,
        remote_similarity_scores: dict[str, float] | None = None,
    ) -> list[SourceCandidate]:
        target_tokens = _tokens(schema_var.description)
        target_code_hint = _question_hint(schema_var.description)
        source_by_upper = {str(source.get("name", "")).upper(): source for source in source_contexts}
        candidates = []
        for alias in MODULE6_SOURCE_ALIASES.get(schema_var.name, []):
            if alias.upper() in source_by_upper:
                candidates.append(
                    SourceCandidate(
                        source_variable=str(source_by_upper[alias.upper()].get("name")),
                        score=0.94,
                        evidence=[f"Standard CSES Module 6 source item alias {alias}."],
                    )
                )
        for source in source_contexts:
            name = str(source.get("name", ""))
            desc = str(source.get("description", ""))
            labels = source.get("value_labels") or {}
            source_tokens = _tokens(" ".join([name, desc, " ".join(str(v) for v in labels.values())]))
            overlap = len(target_tokens & source_tokens) / max(len(target_tokens), 1)
            score = overlap
            evidence = []
            if target_code_hint and name.upper() == target_code_hint.upper():
                score = max(score, 0.92)
                evidence.append(f"Source variable name matches questionnaire hint {target_code_hint}.")
            if schema_var.name.upper() == name.upper():
                score = max(score, 0.95)
                evidence.append("Source variable is already CSES-named.")
            if overlap:
                evidence.append(f"Token overlap with source label/values: {overlap:.2f}.")
            if score >= 0.18:
                candidates.append(
                    SourceCandidate(
                        source_variable=name,
                        score=round(score, 3),
                        evidence=evidence or ["Deterministic source-label similarity."],
                    )
                )
        evidence_candidates = self._matching_evidence_candidates(
            schema_var,
            matching_evidence or {},
            remote_similarity_scores or {},
        )
        for candidate in evidence_candidates:
            candidates = self._merge_candidate(candidates, candidate)
        return candidates

    def _merge_party_specific_candidates(
        self,
        schema_var: SchemaVariable,
        candidates: list[SourceCandidate],
        source_contexts: list[dict[str, Any]],
    ) -> list[SourceCandidate]:
        """Handle party/leader source-name variants after party order approval."""
        source_by_upper = {str(source.get("name", "")).upper(): source for source in source_contexts}
        for alias in MODULE6_SOURCE_ALIASES.get(schema_var.name, []):
            prefix_hits = [
                str(source.get("name"))
                for key, source in source_by_upper.items()
                if re.match(rf"^{re.escape(alias.upper())}(?:[_-]?\d+)$", key)
            ]
            for source_name in sorted(prefix_hits, key=_natural_sort_key):
                _boost_or_add_candidate(
                    candidates,
                    SourceCandidate(
                        source_variable=source_name,
                        score=1.0,
                        evidence=[f"Source variable is a split variant of expected party/leader item {alias}."],
                        conflict_flags=["split_party_or_leader_item_variant"],
                    )
                )
                break
        if re.match(r"^F3019_[A-I]$", schema_var.name):
            target_letter = schema_var.name.rsplit("_", 1)[-1]
            direct_alias = f"Q17{target_letter.lower()}"
            if direct_alias.upper() not in source_by_upper and not any(name.startswith(direct_alias.upper()) for name in source_by_upper):
                extra = self._additional_split_leader_source(source_contexts)
                if extra:
                    _boost_or_add_candidate(
                        candidates,
                        SourceCandidate(
                            source_variable=extra,
                            score=1.0,
                            evidence=[
                                "Additional leader source item detected from a split party-leader battery.",
                                "Processor must confirm the leader-party association before code generation.",
                            ],
                            conflict_flags=["additional_leader_variant_requires_review"],
                        ),
                    )
        return sorted(candidates, key=lambda item: (item.score, _party_candidate_priority(item)), reverse=True)

    def _additional_split_leader_source(self, source_contexts: list[dict[str, Any]]) -> str:
        split_sources = sorted(
            [
                str(source.get("name"))
                for source in source_contexts
                if re.match(r"^Q17[a-i](?:[_-]?\d+)$", str(source.get("name", "")), re.IGNORECASE)
            ],
            key=_natural_sort_key,
        )
        for source in split_sources:
            if not re.search(r"1$", source, re.IGNORECASE):
                return source
        return ""

    def _matching_evidence_candidates(
        self,
        schema_var: SchemaVariable,
        matching_evidence: dict[str, Any],
        remote_similarity_scores: dict[str, float],
    ) -> list[SourceCandidate]:
        if not matching_evidence or schema_var.dependency_class != "direct_survey_item":
            return []
        targets = {
            item.get("name"): item
            for item in matching_evidence.get("target_variable_profiles", []) or []
            if isinstance(item, dict)
        }
        target = targets.get(schema_var.name)
        if not target:
            return []
        expected_ids = {normalize_item_key(str(item)) for item in target.get("expected_item_ids", []) or [] if item}
        source_profiles = [
            item for item in matching_evidence.get("source_variable_profiles", []) or []
            if isinstance(item, dict) and item.get("name")
        ]
        questionnaire_items = {
            normalize_item_key(str(item.get("item_id", ""))): item
            for item in matching_evidence.get("questionnaire_items", []) or []
            if isinstance(item, dict)
        }
        response_matcher = ResponseOptionMatcher()
        candidates: list[SourceCandidate] = []
        for source in source_profiles:
            source_name = str(source.get("name", ""))
            likely_id = normalize_item_key(str(source.get("likely_item_id", "")))
            item = questionnaire_items.get(likely_id)
            source_text = _source_profile_text(source, item)
            target_text = " ".join(
                str(value)
                for value in [
                    target.get("name"),
                    target.get("description"),
                    " ".join(target.get("expected_item_ids", []) or []),
                ]
                if value
            )
            score_parts: list[float] = []
            evidence: list[str] = []
            flags: list[str] = []
            if likely_id and likely_id in expected_ids:
                score_parts.append(0.96)
                evidence.append(f"Source variable is linked to expected questionnaire item {likely_id}.")
            remote_key = f"{schema_var.name}||{source_name}"
            if remote_key in remote_similarity_scores:
                score_parts.append(float(remote_similarity_scores[remote_key]))
                evidence.append(f"Item text similarity: {remote_similarity_scores[remote_key]:.2f}.")
            lexical = _lexical_similarity(target_text, source_text)
            if lexical:
                score_parts.append(min(0.82, lexical))
                evidence.append(f"Text/label similarity: {lexical:.2f}.")
            response_result = response_matcher.compare(
                _target_profile_to_object(target),
                _source_profile_to_object(source),
                _question_item_to_object(item) if item else None,
            )
            if not score_parts:
                continue
            score_parts.append(float(response_result.get("score", 0.0)) * 0.85)
            flags.extend(response_result.get("flags", []) or [])
            if response_result.get("evidence"):
                evidence.append(str(response_result["evidence"]))
            score = max(score_parts)
            if flags:
                score = min(score, 0.78)
            if score >= 0.18:
                candidates.append(
                    SourceCandidate(
                        source_variable=source_name,
                        score=round(score, 3),
                        evidence=evidence,
                        conflict_flags=flags,
                    )
                )
        return candidates

    def _merge_candidate(self, candidates: list[SourceCandidate], proposal: SourceCandidate) -> list[SourceCandidate]:
        for candidate in candidates:
            if candidate.source_variable == proposal.source_variable:
                candidate.score = max(candidate.score, proposal.score)
                candidate.evidence.extend(item for item in proposal.evidence if item)
                return candidates
        candidates.append(proposal)
        return candidates

    def _derived_decision(self, schema_var: SchemaVariable, administrative_plan: AdministrativeVariablePlan | None = None) -> MatchingDecision:
        if administrative_plan:
            confidence = {
                "confirmed": "high",
                "proposed": "medium",
                "needs_processor_review": "processor_review",
            }.get(administrative_plan.status, "processor_review")
            return MatchingDecision(
                target_variable=schema_var.name,
                description=schema_var.description,
                status="generated_from_administrative_information",
                source_variable=administrative_plan.source_variable or "ADMINISTRATIVE_INFORMATION",
                confidence=confidence,
                evidence=administrative_plan.evidence,
                conflict_flags=[] if administrative_plan.status != "needs_processor_review" else ["processor_review_required"],
                processor_verification_required=administrative_plan.processor_review_required,
                dependency_class=schema_var.dependency_class,
                notes=administrative_plan.notes,
            )
        return MatchingDecision(
            target_variable=schema_var.name,
            description=schema_var.description,
            status="derived_metadata",
            source_variable="DERIVED_METADATA",
            confidence="processor_review",
            dependency_class=schema_var.dependency_class,
            notes="Derived from study metadata/design evidence; processor verification required.",
        )

    def _demographic_decision(
        self,
        schema_var: SchemaVariable,
        assessment: DemographicRecodingAssessment | None = None,
    ) -> MatchingDecision:
        if not assessment:
            return MatchingDecision(
                target_variable=schema_var.name,
                description=schema_var.description,
                status="blocked_for_processor_review",
                confidence="low",
                dependency_class=schema_var.dependency_class,
                notes="Demographic recoding assessment was not available.",
                conflict_flags=["demographic_assessment_missing"],
            )
        source = assessment.source_variable
        has_source = bool(source) and source not in {"NOT_FOUND", "EXTERNAL_INPUT_REQUIRED"}
        status = "demographic_recoding_assessment" if has_source or assessment.status == "proposed_derivation" else "blocked_for_processor_review"
        flags = list(assessment.warnings)
        if assessment.processor_review_required:
            flags.append("processor_review_required")
        return MatchingDecision(
            target_variable=schema_var.name,
            description=schema_var.description,
            status=status,
            source_variable=source or "NOT_FOUND",
            confidence=assessment.confidence,
            candidates=[
                SourceCandidate(
                    source_variable=source,
                    score=0.86 if has_source else 0.0,
                    evidence=assessment.evidence[:3],
                    conflict_flags=assessment.warnings[:3],
                )
            ] if source else [],
            evidence=assessment.evidence,
            conflict_flags=flags,
            processor_verification_required=assessment.processor_review_required,
            dependency_class=schema_var.dependency_class,
            notes=(
                f"Demographic recoding assessment: {assessment.source_format}; "
                f"{assessment.recoding_action}; {assessment.recoding_plan_type}. "
                f"{assessment.notes}"
            ).strip(),
        )

    def _external_decision(self, schema_var: SchemaVariable) -> MatchingDecision:
        status = "external_input_required"
        if schema_var.dependency_class == "district_input":
            status = "external_input_required"
        return MatchingDecision(
            target_variable=schema_var.name,
            description=schema_var.description,
            status=status,
            source_variable="EXTERNAL_INPUT_REQUIRED",
            confidence="processor_review",
            dependency_class=schema_var.dependency_class,
            notes="Requires external election, macro, party, or district material, or a recorded processor decision.",
        )

    def _awaiting_party_order_decision(self, schema_var: SchemaVariable, party_order_summary: dict[str, Any]) -> MatchingDecision:
        return MatchingDecision(
            target_variable=schema_var.name,
            description=schema_var.description,
            status="awaiting_party_order_agreement",
            source_variable="PARTY_ORDER_AGREEMENT",
            confidence="processor_review",
            dependency_class=schema_var.dependency_class,
            evidence=[
                f"Party order proposal covers {party_order_summary.get('party_count', 0)} parties.",
                "Party, vote-choice, leader, and macro-party variables must use the same approved party order.",
            ],
            conflict_flags=["micro_macro_agreement_required"],
            processor_verification_required=True,
            notes="Party Order Agreement must be approved by the micro processor and macro coder before this variable is matched or generated.",
        )

    def _party_order_generated_decision(self, schema_var: SchemaVariable, party_order_summary: dict[str, Any]) -> MatchingDecision:
        return MatchingDecision(
            target_variable=schema_var.name,
            description=schema_var.description,
            status="generated_from_party_order",
            source_variable="APPROVED_PARTY_ORDER",
            confidence="processor_review",
            dependency_class=schema_var.dependency_class,
            evidence=[
                f"Approved party order covers {party_order_summary.get('party_count', 0)} parties.",
                "Macro-party variables must remain consistent with the approved micro/macro party order.",
            ],
            conflict_flags=[],
            processor_verification_required=True,
            notes="Generated from the approved Party Order Agreement and related macro/election inputs.",
        )

    def _party_context_decision(self, schema_var: SchemaVariable, party_order_summary: dict[str, Any]) -> MatchingDecision | None:
        context = str(party_order_summary.get("selected_context") or "")
        name = schema_var.name
        party_count = int(party_order_summary.get("party_count") or 0)
        party_letter_match = re.match(r"^F30(18|20|21)_([A-I])$", name)
        if party_letter_match and party_count:
            letter_index = ord(party_letter_match.group(2)) - ord("A") + 1
            if letter_index > party_count:
                return self._generated_from_party_context(schema_var, "no_approved_party_for_this_slot")
        if name == "F3021" or re.match(r"^F3021(?:_[A-IR])?$", name):
            return self._generated_from_party_context(schema_var, "optional_alternative_scale_not_collected")
        if context == "lower_house" and name in {"F3010", "F3010_ME"}:
            return self._generated_from_party_context(schema_var, "derived_from_lower_house_main_election")
        if context == "lower_house" and re.match(r"^F3010_PR_", name):
            return self._generated_from_party_context(schema_var, "not_applicable_presidential_election")
        if context == "lower_house" and name == "F3010_UH":
            return self._generated_from_party_context(schema_var, "not_applicable_upper_house_election")
        if name in {"F3010_TS", "F3010_FTV"}:
            return self._generated_from_party_context(schema_var, "derived_from_current_and_previous_turnout")
        if context == "lower_house" and re.match(r"^F3015_PR_", name):
            return self._generated_from_party_context(schema_var, "not_applicable_presidential_election")
        if context == "lower_house" and name == "F3015_UH":
            return self._generated_from_party_context(schema_var, "not_applicable_upper_house_election")
        if name.startswith("F3100_"):
            return self._generated_from_party_context(schema_var, "derived_from_approved_party_metadata")
        if context == "lower_house" and re.match(r"^F30(11|16)_PR_", name):
            return self._generated_from_party_context(schema_var, "not_applicable_presidential_election")
        if context == "lower_house" and re.match(r"^F30(11|16)_UH_", name):
            return self._generated_from_party_context(schema_var, "not_applicable_upper_house_election")
        if context == "lower_house" and re.match(r"^F30(11|16)_LH_DC", name):
            return self._generated_from_party_context(schema_var, "not_applicable_district_candidate_vote")
        if name in {"F3011_OUTGOV", "F3011_LR_CSES", "F3011_LR_MARPOR", "F3011_IF_CSES", "F3011_VS_1"}:
            return self._generated_from_party_context(schema_var, "derived_from_approved_vote_choice_and_party_metadata")
        return None

    def _generated_from_party_context(self, schema_var: SchemaVariable, source_variable: str) -> MatchingDecision:
        return MatchingDecision(
            target_variable=schema_var.name,
            description=schema_var.description,
            status="generated_from_party_context",
            source_variable=source_variable.upper(),
            confidence="processor_review",
            dependency_class=schema_var.dependency_class,
            evidence=["Generated from approved election context and party-order decision."],
            conflict_flags=[],
            processor_verification_required=True,
            notes="Not an ordinary source-variable match; generated from election context, party order, and CSES missing/not-applicable rules.",
        )

    def _proposal_lookup(self, proposals: list[Any]) -> dict[str, dict]:
        lookup = {}
        for proposal in proposals:
            data = proposal if isinstance(proposal, dict) else getattr(proposal, "__dict__", {})
            target = data.get("target_variable") or data.get("target") or data.get("cses_target")
            if target:
                lookup[str(target)] = data
        return lookup

    def _mark_source_reuse_conflicts(self, decisions: list[MatchingDecision]) -> None:
        by_source: dict[str, list[MatchingDecision]] = {}
        for decision in decisions:
            source = decision.source_variable
            if not source or source in {"DERIVED_METADATA", "EXTERNAL_INPUT_REQUIRED", "NOT_FOUND"}:
                continue
            if decision.status in {
                "generated_from_administrative_information",
                "demographic_recoding_assessment",
                "generated_from_party_context",
                "generated_from_party_order",
            }:
                continue
            by_source.setdefault(source, []).append(decision)
        for source, items in by_source.items():
            if len(items) <= 1:
                continue
            for decision in items:
                decision.conflict_flags.append(f"source_reused_by_{len(items)}_targets")
                if decision.status == "proposed_match":
                    decision.status = "blocked_for_processor_review"
                decision.confidence = "medium"


def decision_summary(decisions: list[MatchingDecision]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for decision in decisions:
        summary[decision.status] = summary.get(decision.status, 0) + 1
    return summary


def matching_category_summary(decisions: list[MatchingDecision], matching_evidence: dict[str, Any] | None = None) -> dict[str, dict[str, int]]:
    """Summarize matching by processor-relevant CSES categories."""
    target_profiles = {
        item.get("name"): item
        for item in (matching_evidence or {}).get("target_variable_profiles", []) or []
        if isinstance(item, dict)
    }
    categories = {
        "core_questionnaire_items": {"total": 0, "matched": 0, "needs_review": 0},
        "demographic_items": {"total": 0, "matched": 0, "needs_review": 0},
        "administrative_metadata": {"total": 0, "prepared": 0, "ready": 0, "needs_review": 0},
        "party_election_items": {"total": 0, "awaiting_party_ordering": 0, "matched": 0},
        "district_items": {"total": 0, "awaiting_district_input": 0, "matched": 0},
        "unresolved_items": {"total": 0},
    }
    for decision in decisions:
        profile = target_profiles.get(decision.target_variable, {})
        item_type = str(profile.get("canonical_item_type") or "")
        section = str(profile.get("section") or "")
        dep = decision.dependency_class
        matched = decision.status in {"proposed_match", "demographic_recoding_assessment", "generated_from_party_order", "generated_from_party_context"} and decision.source_variable not in {"", "NOT_FOUND", "ERROR"}
        if dep == "derived_metadata":
            bucket = categories["administrative_metadata"]
            bucket["total"] += 1
            if decision.status in {"derived_metadata", "generated_from_administrative_information"}:
                bucket["prepared"] += 1
                if decision.processor_verification_required:
                    bucket["needs_review"] += 1
                else:
                    bucket["ready"] += 1
            else:
                bucket["needs_review"] += 1
        elif dep == "district_input":
            bucket = categories["district_items"]
            bucket["total"] += 1
            if matched:
                bucket["matched"] += 1
            else:
                bucket["awaiting_district_input"] += 1
        elif dep == "macro_or_party_input" or item_type == "party_vote_item":
            bucket = categories["party_election_items"]
            bucket["total"] += 1
            if decision.status == "awaiting_party_order_agreement":
                bucket["awaiting_party_ordering"] += 1
            elif matched:
                bucket["matched"] += 1
            else:
                bucket["awaiting_party_ordering"] += 1
        elif item_type == "demographic_coding_standard" or section == "demographic" or decision.target_variable.startswith("F2"):
            bucket = categories["demographic_items"]
            bucket["total"] += 1
            if matched:
                bucket["matched"] += 1
            else:
                bucket["needs_review"] += 1
        else:
            bucket = categories["core_questionnaire_items"]
            bucket["total"] += 1
            if matched:
                bucket["matched"] += 1
            else:
                bucket["needs_review"] += 1
        if not matched and decision.status not in {
            "derived_metadata",
            "generated_from_administrative_information",
            "external_input_required",
            "awaiting_party_order_agreement",
        }:
            categories["unresolved_items"]["total"] += 1
    return categories


def _tokens(text: str) -> set[str]:
    stop = {"the", "and", "with", "for", "from", "variable", "component", "id", "cses"}
    return {
        token for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2 and token not in stop
    }


def _question_hint(text: str) -> str:
    match = re.search(r"\b([AQD]\d{1,2}[A-Za-z0-9_]*)\b", text)
    return match.group(1) if match else ""


def _natural_sort_key(value: str) -> list[Any]:
    return [
        int(part) if part.isdigit() else part.casefold()
        for part in re.split(r"(\d+)", str(value))
    ]


def _boost_or_add_candidate(candidates: list[SourceCandidate], proposal: SourceCandidate) -> None:
    for candidate in candidates:
        if candidate.source_variable == proposal.source_variable:
            candidate.score = max(candidate.score, proposal.score)
            candidate.evidence.extend(item for item in proposal.evidence if item and item not in candidate.evidence)
            candidate.conflict_flags.extend(
                item for item in proposal.conflict_flags if item and item not in candidate.conflict_flags
            )
            return
    candidates.append(proposal)


def _party_candidate_priority(candidate: SourceCandidate) -> int:
    flags = set(candidate.conflict_flags or [])
    evidence = " ".join(candidate.evidence or []).casefold()
    if "additional_leader_variant_requires_review" in flags:
        return 3
    if "split_party_or_leader_item_variant" in flags:
        return 2
    if "expected party/leader item" in evidence:
        return 1
    return 0


def _kb_evidence(target_variable: str, study_kb: dict | None) -> list[str]:
    if not study_kb:
        return []
    missing = study_kb.get("missing_fields", []) or []
    hits = [item for item in missing if target_variable.lower() in str(item).lower()]
    return [f"Study information missing-field evidence: {item}" for item in hits[:3]]


def _source_profile_text(source: dict[str, Any], item: dict[str, Any] | None = None) -> str:
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


def _target_profile_to_object(target: dict[str, Any]) -> TargetVariableProfile:
    return TargetVariableProfile(
        name=str(target.get("name", "")),
        description=str(target.get("description", "")),
        dependency_class=str(target.get("dependency_class", "")),
        expected_item_ids=list(target.get("expected_item_ids", []) or []),
        canonical_item_ids=list(target.get("canonical_item_ids", []) or []),
        canonical_item_type=str(target.get("canonical_item_type", "")),
        expected_response_structure=str(target.get("expected_response_structure", "")),
        missing_value_policy=str(target.get("missing_value_policy", "")),
        required_evidence=list(target.get("required_evidence", []) or []),
    )


def _source_profile_to_object(source: dict[str, Any]) -> SourceVariableProfile:
    return SourceVariableProfile(
        name=str(source.get("name", "")),
        label=str(source.get("label", "")),
        value_labels=dict(source.get("value_labels", {}) or {}),
        sample_values=list(source.get("sample_values", []) or []),
        dtype=str(source.get("dtype", "")),
        n_unique=source.get("n_unique"),
        missing_count=source.get("missing_count"),
        missing_percent=source.get("missing_percent"),
        top_values=list(source.get("top_values", []) or []),
        likely_item_id=str(source.get("likely_item_id", "")),
        source_file=str(source.get("source_file", "")),
        citation=str(source.get("citation", "")),
    )


def _question_item_to_object(item: dict[str, Any]) -> QuestionnaireItem:
    return QuestionnaireItem(
        item_id=str(item.get("item_id", "")),
        text=str(item.get("text", "")),
        response_options=list(item.get("response_options", []) or []),
        source_file=str(item.get("source_file", "")),
        language=str(item.get("language", "")),
        citation=str(item.get("citation", "")),
        grid=str(item.get("grid", "")),
    )
