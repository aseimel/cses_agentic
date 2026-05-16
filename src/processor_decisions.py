"""Processor corrections, approvals, and targeted rerun bookkeeping."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DECISION_STATUSES = {"pending_confirmation", "approved", "rejected", "needs_review"}


@dataclass
class ProcessorDecision:
    decision_id: str
    area: str
    decision_type: str
    step: int = 0
    target: str = ""
    value: str = ""
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    affected_variables: list[str] = field(default_factory=list)
    affected_outputs: list[str] = field(default_factory=list)
    status: str = "approved"
    source: str = "processor"
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        if not self.updated_at:
            self.updated_at = self.created_at
        if self.status not in DECISION_STATUSES:
            self.status = "needs_review"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProcessorDecision":
        return cls(
            decision_id=str(data.get("decision_id") or data.get("id") or ""),
            area=str(data.get("area") or "general"),
            decision_type=str(data.get("decision_type") or data.get("type") or "correction"),
            step=int(data.get("step") or 0),
            target=str(data.get("target") or ""),
            value=str(data.get("value") or ""),
            reason=str(data.get("reason") or ""),
            evidence=[str(item) for item in data.get("evidence", []) or []],
            affected_variables=[str(item) for item in data.get("affected_variables", []) or []],
            affected_outputs=[str(item) for item in data.get("affected_outputs", []) or []],
            status=str(data.get("status") or "approved"),
            source=str(data.get("source") or "processor"),
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )


@dataclass
class DecisionImpact:
    affected_steps: list[int] = field(default_factory=list)
    affected_outputs: list[str] = field(default_factory=list)
    affected_variables: list[str] = field(default_factory=list)
    rerun_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DependencyInvalidator:
    """Map approved corrections to the smallest downstream work that needs review."""

    AREA_IMPACTS: dict[str, DecisionImpact] = {
        "eligibility_design": DecisionImpact(
            affected_steps=[1, 2, 4, 15, 16],
            affected_outputs=["processing_log", "esn", "final_readiness"],
            rerun_actions=["regenerate documentation", "compare again"],
        ),
        "matching": DecisionImpact(
            affected_steps=[7, 8, 11, 12, 15, 16],
            affected_outputs=["recoding_plan", "stata_syntax", "checks", "documentation", "final_readiness"],
            rerun_actions=["rerun this variable", "rerun checks", "compare again"],
        ),
        "demographic_coding": DecisionImpact(
            affected_steps=[7, 8, 11, 12, 15, 16],
            affected_outputs=["demographic_recode", "stata_syntax", "checks", "documentation", "final_readiness"],
            rerun_actions=["rerun this variable", "rerun checks", "compare again"],
        ),
        "party_order": DecisionImpact(
            affected_steps=[7, 8, 10, 11, 12, 15, 16],
            affected_outputs=["party_recodes", "party_labels", "macro_party_variables", "checks", "documentation", "final_readiness"],
            rerun_actions=["rebuild party recodes", "rerun checks", "regenerate documentation", "compare again"],
        ),
        "macro_context": DecisionImpact(
            affected_steps=[7, 8, 11, 12, 15, 16],
            affected_outputs=["party_context_variables", "stata_syntax", "checks", "documentation", "final_readiness"],
            rerun_actions=["rebuild party recodes", "rerun checks", "regenerate documentation"],
        ),
        "district_data": DecisionImpact(
            affected_steps=[9, 8, 11, 12, 15, 16],
            affected_outputs=["district_merge", "stata_syntax", "district_checks", "documentation", "final_readiness"],
            rerun_actions=["rerun district review", "rerun checks", "compare again"],
        ),
        "recoding_plan": DecisionImpact(
            affected_steps=[7, 8, 11, 12, 15, 16],
            affected_outputs=["recoding_plan", "stata_syntax", "checks", "documentation", "final_readiness"],
            rerun_actions=["rerun this variable", "rerun checks"],
        ),
        "documentation": DecisionImpact(
            affected_steps=[13, 14, 15, 16],
            affected_outputs=["processing_log", "collaborator_questions", "esn", "final_readiness"],
            rerun_actions=["regenerate documentation", "compare again"],
        ),
        "final_readiness": DecisionImpact(
            affected_steps=[16],
            affected_outputs=["final_readiness"],
            rerun_actions=["compare again"],
        ),
    }

    def impact_for(self, decision: ProcessorDecision) -> DecisionImpact:
        base = self.AREA_IMPACTS.get(decision.area, DecisionImpact(affected_steps=[16], affected_outputs=["final_readiness"], rerun_actions=["compare again"]))
        variables = list(dict.fromkeys([*base.affected_variables, *decision.affected_variables]))
        if decision.target and decision.target.upper().startswith("F"):
            variables = list(dict.fromkeys([*variables, decision.target.upper()]))
        return DecisionImpact(
            affected_steps=list(base.affected_steps),
            affected_outputs=list(base.affected_outputs),
            affected_variables=variables,
            rerun_actions=list(base.rerun_actions),
        )


class ProcessorDecisionLedger:
    """Read and write the study-level processor decision ledger."""

    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        self.path = self.working_dir / ".cses" / "processor_decisions.json"

    def load(self) -> list[ProcessorDecision]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return [ProcessorDecision.from_dict(item) for item in payload.get("decisions", []) or []]

    def save(self, decisions: list[ProcessorDecision]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "decision_count": len(decisions),
            "decisions": [decision.to_dict() for decision in decisions],
        }
        self.path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def append(self, decision: ProcessorDecision) -> ProcessorDecision:
        decisions = self.load()
        if not decision.decision_id:
            decision.decision_id = f"PD-{len(decisions) + 1:04d}"
        decisions.append(decision)
        self.save(decisions)
        return decision

    def update_status(self, decision_id: str, status: str, reason: str = "") -> ProcessorDecision | None:
        if status not in DECISION_STATUSES:
            raise ValueError(f"Unsupported decision status: {status}")
        decisions = self.load()
        updated: ProcessorDecision | None = None
        for decision in decisions:
            if decision.decision_id == decision_id:
                decision.status = status
                if reason:
                    decision.reason = reason
                decision.updated_at = datetime.now(timezone.utc).isoformat()
                updated = decision
                break
        self.save(decisions)
        return updated

    def latest_approved(self, area: str | None = None, target: str | None = None) -> list[ProcessorDecision]:
        decisions = [item for item in self.load() if item.status == "approved"]
        if area:
            decisions = [item for item in decisions if item.area == area]
        if target:
            normalized = target.upper()
            decisions = [item for item in decisions if item.target.upper() == normalized]
        return decisions

    def source_match_overrides(self) -> dict[str, ProcessorDecision]:
        overrides: dict[str, ProcessorDecision] = {}
        for decision in self.latest_approved(area="matching"):
            if decision.decision_type == "source_variable_correction" and decision.target and decision.value:
                overrides[decision.target.upper()] = decision
        return overrides


class CorrectionInterpreter:
    """Small deterministic parser for common processor corrections."""

    PARTY_RE = re.compile(r"\bparty\s+([a-i])\b.*?\b(?:should be|is|=)\s+(.+)", re.IGNORECASE)
    DISTRICT_RE = re.compile(r"\bdistrict\s+(?:key|variable)\s+(?:is|=|should be)\s+([A-Za-z_][\w]*)", re.IGNORECASE)
    USE_VAR_RE = re.compile(
        r"\buse\s+(?:variable\s+)?([A-Za-z_][\w]*)\s+(?:for|as)\s+([A-Za-z0-9_ -]+)",
        re.IGNORECASE,
    )
    TARGET_VAR_RE = re.compile(
        r"\b(F\d{4}(?:_[A-Za-z0-9]+)*)\b.*?\b(?:should use|use|is from|source is)\s+(?:variable\s+)?([A-Za-z_][\w]*)",
        re.IGNORECASE,
    )

    def parse(self, text: str, default_step: int = 0) -> ProcessorDecision | None:
        message = " ".join((text or "").strip().split())
        if not message:
            return None
        lowered = message.casefold()

        if "probability sample" in lowered:
            negative = bool(
                re.search(r"\bnot\s+(?:a\s+)?probability sample\b", lowered)
                or re.search(r"\bnon[- ]probability sample\b", lowered)
                or re.search(r"\bprobability sample\s*:\s*no\b", lowered)
            )
            value = "no" if negative else "yes"
            return ProcessorDecision(
                decision_id="",
                step=default_step or 1,
                area="eligibility_design",
                decision_type="study_design_correction",
                target="probability_sample_status",
                value=value,
                reason=message,
                affected_outputs=["processing_log", "esn", "final_readiness"],
                status="approved",
            )

        target_match = self.TARGET_VAR_RE.search(message)
        if target_match:
            target, source = target_match.group(1).upper(), target_match.group(2)
            return ProcessorDecision(
                decision_id="",
                step=default_step or 7,
                area="matching",
                decision_type="source_variable_correction",
                target=target,
                value=source,
                reason=message,
                affected_variables=[target],
                status="pending_confirmation",
            )

        use_match = self.USE_VAR_RE.search(message)
        if use_match:
            source, target_label = use_match.group(1), use_match.group(2).strip(" .")
            return ProcessorDecision(
                decision_id="",
                step=default_step or 7,
                area="matching",
                decision_type="source_variable_correction",
                target=target_label,
                value=source,
                reason=message,
                status="pending_confirmation",
            )

        party_match = self.PARTY_RE.search(message)
        if party_match:
            letter, party = party_match.group(1).upper(), party_match.group(2).strip(" .")
            return ProcessorDecision(
                decision_id="",
                step=default_step or 7,
                area="party_order",
                decision_type="party_order_correction",
                target=f"Party {letter}",
                value=party,
                reason=message,
                status="pending_confirmation",
            )

        district_match = self.DISTRICT_RE.search(message)
        if district_match:
            return ProcessorDecision(
                decision_id="",
                step=default_step or 9,
                area="district_data",
                decision_type="district_key_correction",
                target="district_key",
                value=district_match.group(1),
                reason=message,
                status="pending_confirmation",
            )

        if any(token in lowered for token in ("actually", "correction", "correct ", "should be", "wrong")):
            return ProcessorDecision(
                decision_id="",
                step=default_step,
                area="general",
                decision_type="general_correction",
                reason=message,
                status="needs_review",
            )
        return None


class TargetedRerunInterpreter:
    """Interpret common rerun requests into queued workflow actions."""

    def parse(self, text: str) -> dict[str, Any] | None:
        lowered = " ".join((text or "").strip().lower().split())
        if not lowered:
            return None
        if "rerun this variable" in lowered or "rerun variable" in lowered:
            return {"action": "rerun_variable", "label": "Rerun the affected variable", "steps": [7, 8, 12]}
        if "rebuild party recodes" in lowered:
            return {"action": "rebuild_party_recodes", "label": "Rebuild party recodes", "steps": [7, 8, 10, 12]}
        if "regenerate documentation" in lowered:
            return {"action": "regenerate_documentation", "label": "Regenerate documentation", "steps": [13, 15, 16]}
        if "rerun checks" in lowered:
            return {"action": "rerun_checks", "label": "Rerun checks", "steps": [12, 16]}
        if "compare again" in lowered:
            return {"action": "compare_again", "label": "Compare outputs again", "steps": [16]}
        return None


def diagnose_failure_text(text: str) -> dict[str, Any]:
    """Classify a failure in processor-facing categories."""
    lowered = (text or "").casefold()
    checks = [
        ("district merge problem", ("district", "constituency", "_merge", "f400", "f2019")),
        ("party-order problem", ("party order", "party a", "party b", "leader", "f3018", "f3019", "f3020")),
        ("macro-context problem", ("marpor", "parlgov", "party facts", "ideological", "left-right", "rile")),
        ("source match problem", ("not found", "ambiguous source", "source variable", "no source", "mapping")),
        ("recode logic problem", ("invalid syntax", "type mismatch", "recode", "replace", "expression")),
        ("label/check/documentation issue", ("label", "check", "documentation", "esn", "log")),
    ]
    for category, terms in checks:
        if any(term in lowered for term in terms):
            return {
                "category": category,
                "suggested_fix": _suggested_fix(category),
            }
    return {
        "category": "processor review needed",
        "suggested_fix": "Review the failing variable or output, record the corrected decision, then rerun the affected step.",
    }


def _suggested_fix(category: str) -> str:
    return {
        "district merge problem": "Review the district file, respondent district key, and district-code coverage, then rerun District Data Review.",
        "party-order problem": "Review Party Order Agreement, approve the corrected order, then rebuild party recodes.",
        "macro-context problem": "Review Macro Context Review values with the macro coder, then regenerate party-context variables.",
        "source match problem": "Correct the source-variable match for the affected CSES variable, then rerun that variable.",
        "recode logic problem": "Correct the recoding plan or source value map, then regenerate Stata syntax and rerun checks.",
        "label/check/documentation issue": "Regenerate labels, checks, or documentation from the approved decisions and compare again.",
    }.get(category, "Record the corrected processor decision and rerun the affected output.")
