"""Macro context review for micro-process party derivatives.

The micro workflow needs a small, approved set of macro/context facts before
party-context recodes can be finalized. This module prepares those facts from
deposited evidence first, then optional bundled external macro data, and leaves
the final decision to the micro processor and macro coder.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.matching.party_order import PARTY_LETTERS, PartyOrderProposal
from src.matching.party_metadata import PartyMetadataReview
from src.workflow.state import WorkflowState


REQUIRED_APPROVAL = {
    "micro_processor_approved": False,
    "macro_coder_approved": False,
    "locked": False,
    "override_reason": "",
}


@dataclass(frozen=True)
class MacroContextRequirement:
    requirement_id: str
    label: str
    affected_variables: list[str]
    requires_joint_approval: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MacroContextEvidence:
    requirement_id: str
    value: Any
    source_type: str
    source_file: str = ""
    citation: str = ""
    confidence: str = "medium"
    precedence_rank: int = 99

    def normalized_value(self) -> str:
        return _normalize_value(self.value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class MacroContextItem:
    requirement_id: str
    label: str
    proposed_value: Any = ""
    source_type: str = ""
    source_file: str = ""
    confidence: str = "missing"
    affected_variables: list[str] = field(default_factory=list)
    evidence: list[MacroContextEvidence] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing: bool = False
    approval_status: str = "needs_joint_approval"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["evidence"] = [item.to_dict() for item in self.evidence]
        return data


@dataclass
class MacroContextReview:
    status: str
    items: list[MacroContextItem] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    external_provider: dict[str, Any] = field(default_factory=dict)
    processor_decision_needed: str = (
        "Micro processor and macro coder should confirm the macro context before "
        "party-context recoding is finalized."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "items": [item.to_dict() for item in self.items],
            "missing_items": list(self.missing_items),
            "conflicts": list(self.conflicts),
            "external_provider": dict(self.external_provider),
            "processor_decision_needed": self.processor_decision_needed,
        }


class MacroContextRequirementRegistry:
    """Micro workflow's required macro/context facts."""

    def __init__(self) -> None:
        self.requirements = [
            MacroContextRequirement(
                "election_context",
                "Main election context",
                ["F3010", "F3010_TS", "F3011_VS_1"],
            ),
            MacroContextRequirement(
                "election_date",
                "Election date",
                ["F1101_2", "F1101_3", "F3010", "F3011_VS_1"],
            ),
            MacroContextRequirement(
                "party_order",
                "Party A-I order",
                ["F3011_LH_PL", "F3016_LH_PL", "F3018_A", "F3019_A", "F5000_A"],
            ),
            MacroContextRequirement(
                "party_labels",
                "Party names, abbreviations, and CSES numeric codes",
                ["F5000_A", "F6000_LH_PL", "F3011_LH_PL", "F3023_3"],
            ),
            MacroContextRequirement(
                "cses_ideological_family",
                "CSES ideological family by Party A-I",
                ["F5028_A", "F3011_IF_CSES", "F3100_IF_CSES"],
            ),
            MacroContextRequirement(
                "cses_left_right",
                "CSES expert left-right by Party A-I",
                ["F5029_A", "F3011_LR_CSES", "F3100_LR_CSES"],
            ),
            MacroContextRequirement(
                "marpor_identifiers",
                "MARPOR/CMP party identifiers",
                ["F5200_A", "F5201_A", "F5202_A", "F5203_A"],
            ),
            MacroContextRequirement(
                "marpor_rile",
                "MARPOR/CMP RILE values",
                ["F3011_LR_MARPOR", "F3100_LR_MARPOR"],
            ),
            MacroContextRequirement(
                "leader_mapping",
                "Leader names and leader-to-party mapping",
                ["F3020_A", "F5000_L_A"],
            ),
        ]

    def by_id(self) -> dict[str, MacroContextRequirement]:
        return {item.requirement_id: item for item in self.requirements}


class AutoMacroProvider:
    """Optional wrapper for bundled auto_macro output and provenance."""

    def __init__(self, repo_root: Path | None = None, auto_macro_dir: Path | None = None):
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        configured = os.getenv("CSES_AUTO_MACRO_DIR", "")
        self.auto_macro_dir = Path(auto_macro_dir or configured or self.repo_root / "auto_macro")

    def collect(self, state: WorkflowState) -> dict[str, Any]:
        provider = {
            "available": self.auto_macro_dir.exists(),
            "status": "not_available",
            "auto_macro_dir": str(self.auto_macro_dir) if self.auto_macro_dir.exists() else "",
            "script_hash": self._hash("scripts/r_downloader/generate_macro.R"),
            "variable_definitions_hash": self._hash("scripts/r_downloader/variable_definitions.yaml"),
            "country_codes_hash": self._hash("scripts/r_downloader/country_codes.yaml"),
            "source_version": self._source_version(),
            "values": {},
            "output_path": "",
            "run_command": "",
        }
        if not provider["available"]:
            return provider
        country_code = (state.country_code or "").upper()
        year = str(state.year or "")
        if not country_code or not year:
            provider["status"] = "missing_study_identity"
            return provider
        provider["run_command"] = (
            f"generate_macro.bat --country {country_code} --year {year} --date YYYY-MM-DD"
        )
        output_dir = self.auto_macro_dir / "output" / f"{country_code}_{year}"
        csv_path = output_dir / f"{country_code}_{year}_macro.csv"
        if not csv_path.exists():
            provider["status"] = "available_no_output"
            return provider
        provider["output_path"] = str(csv_path)
        provider["values"] = self._read_macro_csv(csv_path)
        provider["status"] = "output_loaded" if provider["values"] else "output_empty"
        return provider

    def write_cache_entry(self, working_dir: Path, state: WorkflowState, provider: dict[str, Any]) -> Path:
        path = Path(working_dir) / ".cses" / "macro_data_cache.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except Exception:
            payload = {}
        entries = payload.setdefault("entries", {})
        key = "|".join([
            (state.country_code or "").upper(),
            str(state.year or ""),
            str((provider.get("values") or {}).get("election_date") or ""),
            str(provider.get("variable_definitions_hash") or ""),
        ])
        entries[key] = {
            "country_code": (state.country_code or "").upper(),
            "year": str(state.year or ""),
            "election_date": str((provider.get("values") or {}).get("election_date") or ""),
            "source_version": provider.get("source_version", ""),
            "script_hash": provider.get("script_hash", ""),
            "variable_definitions_hash": provider.get("variable_definitions_hash", ""),
            "country_codes_hash": provider.get("country_codes_hash", ""),
            "output_path": provider.get("output_path", ""),
            "cached_at": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _hash(self, relative_path: str) -> str:
        path = self.auto_macro_dir / relative_path
        if not path.exists() or not path.is_file():
            return ""
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _source_version(self) -> str:
        path = self.auto_macro_dir / "scripts" / "r_downloader" / "variable_definitions.yaml"
        if not path.exists():
            return ""
        try:
            for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()[:40]:
                if "VERSION" in line.upper():
                    return line.strip().lstrip("#").strip()
        except Exception:
            return ""
        return ""

    def _read_macro_csv(self, path: Path) -> dict[str, str]:
        try:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            return {}
        if not rows:
            return {}
        return {
            str(key).strip(): _clean_cell(value)
            for key, value in rows[0].items()
            if str(key).strip() and _clean_cell(value)
        }


class MacroContextBuilder:
    """Build, persist, and approve study-specific macro context."""

    PRECEDENCE = {
        "macro_workbook": 1,
        "macro_report": 2,
        "election_results_workbook": 3,
        "auto_macro": 4,
        "processor_decision": 6,
    }

    def __init__(self, working_dir: Path, state: WorkflowState):
        self.working_dir = Path(working_dir)
        self.state = state
        self.registry = MacroContextRequirementRegistry()
        self.auto_provider = AutoMacroProvider()

    def build(
        self,
        party_order_proposal: PartyOrderProposal,
        party_order_approved: bool,
        party_metadata_review: PartyMetadataReview,
        party_metadata_approved: bool,
    ) -> MacroContextReview:
        provider = self.auto_provider.collect(self.state)
        if provider.get("status") == "output_loaded":
            self.auto_provider.write_cache_entry(self.working_dir, self.state, provider)
        evidence = self._collect_evidence(
            party_order_proposal,
            party_order_approved,
            party_metadata_review,
            party_metadata_approved,
            provider,
        )
        items = [self._build_item(requirement, evidence.get(requirement.requirement_id, [])) for requirement in self.registry.requirements]
        missing = [item.requirement_id for item in items if item.missing]
        conflicts = [conflict for item in items for conflict in item.conflicts]
        status = "ready_for_joint_review"
        if conflicts:
            status = "conflict_needs_review"
        elif missing:
            status = "missing_context_items"
        return MacroContextReview(
            status=status,
            items=items,
            missing_items=missing,
            conflicts=conflicts,
            external_provider={key: value for key, value in provider.items() if key != "values"},
        )

    def write_review(self, review: MacroContextReview) -> tuple[Path, Path]:
        cses_dir = self.working_dir / ".cses"
        cses_dir.mkdir(parents=True, exist_ok=True)
        review_path = cses_dir / "macro_context_review.json"
        decision_path = cses_dir / "macro_context_decision.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_policy": (
                "Prefer deposited macro/election materials; use versioned external "
                "macro data only as fallback or validation; require coder confirmation."
            ),
            "review": review.to_dict(),
            "approval": dict(REQUIRED_APPROVAL),
        }
        review_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if not decision_path.exists():
            decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            return review_path, decision_path
        try:
            existing = json.loads(decision_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
        approval = existing.get("approval", {}) if isinstance(existing, dict) else {}
        if approval.get("locked"):
            existing_review = existing.get("review", {}) if isinstance(existing, dict) else {}
            if existing_review.get("conflicts") or review.conflicts:
                approval["locked"] = False
                approval["override_reason"] = "Macro context evidence changed or conflicts need renewed review."
                payload["approval"] = approval
                decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return review_path, decision_path

    def is_approved(self) -> bool:
        return is_macro_context_approved(self.working_dir)

    def _collect_evidence(
        self,
        party_order_proposal: PartyOrderProposal,
        party_order_approved: bool,
        party_metadata_review: PartyMetadataReview,
        party_metadata_approved: bool,
        provider: dict[str, Any],
    ) -> dict[str, list[MacroContextEvidence]]:
        evidence: dict[str, list[MacroContextEvidence]] = {}

        def add(item: MacroContextEvidence) -> None:
            evidence.setdefault(item.requirement_id, []).append(item)

        proposal_source = party_order_proposal.source_file or ""
        if party_order_proposal.selected_context and party_order_proposal.selected_context != "unknown":
            add(MacroContextEvidence(
                "election_context",
                party_order_proposal.selected_context,
                "election_results_workbook",
                proposal_source,
                party_order_proposal.ordering_basis,
                "high" if party_order_approved else "processor_review",
                self.PRECEDENCE["election_results_workbook"],
            ))
        if party_order_proposal.proposed_parties:
            parties = [party.to_dict() for party in party_order_proposal.proposed_parties]
            add(MacroContextEvidence(
                "party_order",
                parties,
                "election_results_workbook",
                proposal_source,
                f"{len(parties)} Party A-I entries proposed from standardized election results.",
                "high" if party_order_approved else "processor_review",
                self.PRECEDENCE["election_results_workbook"],
            ))
            add(MacroContextEvidence(
                "party_labels",
                [
                    {
                        "party": party.code_letter,
                        "code": party.numeric_code,
                        "name": party.party_name,
                    }
                    for party in party_order_proposal.proposed_parties
                ],
                "election_results_workbook",
                proposal_source,
                "Party labels and codes taken from the Party Order Agreement proposal.",
                "high" if party_order_approved else "processor_review",
                self.PRECEDENCE["election_results_workbook"],
            ))

        grouped_metadata: dict[str, dict[str, str]] = {}
        metadata_sources: dict[str, str] = {}
        for item in party_metadata_review.values:
            prefix = item.variable.split("_", 1)[0]
            grouped_metadata.setdefault(prefix, {})[item.party_letter] = item.value
            metadata_sources.setdefault(prefix, item.source_file)
        self._metadata_evidence(add, "cses_ideological_family", grouped_metadata.get("F5028", {}), metadata_sources.get("F5028", ""), party_metadata_approved)
        self._metadata_evidence(add, "cses_left_right", grouped_metadata.get("F5029", {}), metadata_sources.get("F5029", ""), party_metadata_approved)
        marpor_ids = {
            variable: values
            for variable, values in grouped_metadata.items()
            if variable in {"F5200", "F5201", "F5202", "F5203"} and values
        }
        if marpor_ids:
            add(MacroContextEvidence(
                "marpor_identifiers",
                marpor_ids,
                "macro_workbook",
                next(iter(metadata_sources.values()), ""),
                "MARPOR/party identifier fields found in deposited macro materials.",
                "high" if party_metadata_approved else "processor_review",
                self.PRECEDENCE["macro_workbook"],
            ))

        auto_values = provider.get("values", {}) if isinstance(provider.get("values"), dict) else {}
        if auto_values.get("election_date"):
            add(MacroContextEvidence(
                "election_date",
                auto_values["election_date"],
                "auto_macro",
                provider.get("output_path", ""),
                "Election date from versioned external macro-data output.",
                "medium",
                self.PRECEDENCE["auto_macro"],
            ))
        rile_values = self._auto_macro_rile_values(auto_values)
        if rile_values:
            add(MacroContextEvidence(
                "marpor_rile",
                rile_values,
                "auto_macro",
                provider.get("output_path", ""),
                "MARPOR/CMP RILE values from versioned external macro-data output.",
                "medium",
                self.PRECEDENCE["auto_macro"],
            ))
        replay_rile = self._benchmark_replay_rile_values()
        if replay_rile:
            add(MacroContextEvidence(
                "marpor_rile",
                replay_rile,
                "processor_decision",
                str(self.working_dir / ".cses" / "benchmark_decision_replay.json"),
                "Benchmark-only simulated processor decision extracted from reference syntax.",
                "processor_review",
                self.PRECEDENCE["processor_decision"],
            ))
        return evidence

    def _metadata_evidence(
        self,
        add: Any,
        requirement_id: str,
        values: dict[str, str],
        source_file: str,
        approved: bool,
    ) -> None:
        if not values:
            return
        source_type = "macro_workbook"
        add(MacroContextEvidence(
            requirement_id,
            values,
            source_type,
            source_file,
            "CSES expert party metadata found in deposited macro materials.",
            "high" if approved else "processor_review",
            self.PRECEDENCE[source_type],
        ))

    def _build_item(
        self,
        requirement: MacroContextRequirement,
        evidence: list[MacroContextEvidence],
    ) -> MacroContextItem:
        ordered = sorted(evidence, key=lambda item: item.precedence_rank)
        if not ordered:
            return MacroContextItem(
                requirement_id=requirement.requirement_id,
                label=requirement.label,
                affected_variables=requirement.affected_variables,
                missing=True,
                approval_status="missing_or_manual_decision_needed",
            )
        best = ordered[0]
        conflicts = self._conflicts(ordered)
        return MacroContextItem(
            requirement_id=requirement.requirement_id,
            label=requirement.label,
            proposed_value=best.value,
            source_type=best.source_type,
            source_file=best.source_file,
            confidence=best.confidence,
            affected_variables=requirement.affected_variables,
            evidence=ordered,
            conflicts=conflicts,
            missing=False,
            approval_status="needs_joint_approval",
        )

    def _conflicts(self, evidence: list[MacroContextEvidence]) -> list[str]:
        by_value: dict[str, list[str]] = {}
        for item in evidence:
            by_value.setdefault(item.normalized_value(), []).append(item.source_type)
        if len(by_value) <= 1:
            return []
        return [
            "Conflicting values were found: "
            + "; ".join(f"{value} ({', '.join(sources)})" for value, sources in by_value.items())
        ]

    def _auto_macro_rile_values(self, values: dict[str, str]) -> dict[str, str]:
        return {
            key: value
            for key, value in values.items()
            if re.match(r"^F(?:3100_LR_MARPOR|3011_LR_MARPOR|52\d{2}_[A-I])$", key)
        }

    def _benchmark_replay_rile_values(self) -> dict[str, str]:
        path = self.working_dir / ".cses" / "benchmark_decision_replay.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        plans = payload.get("reference_recoding_plans", {}) if isinstance(payload, dict) else {}
        values: dict[str, str] = {}
        for target in ("F3100_LR_MARPOR", "F3011_LR_MARPOR"):
            plan = plans.get(target, {}) if isinstance(plans, dict) else {}
            lines = plan.get("lines", []) if isinstance(plan, dict) else []
            if lines:
                values[target] = "\n".join(str(line) for line in lines[:30])
        return values


def load_macro_context_decision(working_dir: Path) -> dict[str, Any]:
    path = Path(working_dir) / ".cses" / "macro_context_decision.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def is_macro_context_approved(working_dir: Path) -> bool:
    payload = load_macro_context_decision(working_dir)
    approval = payload.get("approval", {}) if isinstance(payload, dict) else {}
    return bool(
        approval.get("locked")
        and approval.get("micro_processor_approved")
        and approval.get("macro_coder_approved")
    )


def approved_macro_context_values(working_dir: Path) -> dict[str, Any]:
    payload = load_macro_context_decision(working_dir)
    if not is_macro_context_approved(working_dir):
        return {}
    review = payload.get("review", {}) if isinstance(payload, dict) else {}
    items = review.get("items", []) if isinstance(review, dict) else []
    return {
        str(item.get("requirement_id")): item.get("proposed_value")
        for item in items
        if isinstance(item, dict) and item.get("requirement_id") and not item.get("missing")
    }


def macro_context_message(review: MacroContextReview) -> str:
    lines = ["Macro context review prepared.", ""]
    shown = 0
    for item in review.items:
        if shown >= 8:
            break
        value = _short_value(item.proposed_value)
        status = "missing" if item.missing else value
        source = _source_label(item.source_type)
        lines.append(f"- {item.label}: {status}" + (f" ({source})" if source and not item.missing else ""))
        shown += 1
    if review.missing_items:
        lines.extend(["", "Needs coder input:"])
        labels = {item.requirement_id: item.label for item in review.items}
        for requirement_id in review.missing_items[:12]:
            lines.append(f"- {labels.get(requirement_id, requirement_id)}")
    if review.conflicts:
        lines.extend(["", "Conflicts needing review:"])
        for conflict in review.conflicts[:8]:
            lines.append(f"- {conflict}")
    lines.extend(["", "Approval needed:", f"- {review.processor_decision_needed}"])
    return "\n".join(lines)


def _normalize_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value).strip().casefold()


def _short_value(value: Any) -> str:
    if isinstance(value, list):
        return f"{len(value)} entries"
    if isinstance(value, dict):
        return f"{len(value)} values"
    text = str(value or "").strip()
    return text[:80] + ("..." if len(text) > 80 else "")


def _source_label(source_type: str) -> str:
    return {
        "macro_workbook": "deposited macro materials",
        "macro_report": "macro report",
        "election_results_workbook": "election results",
        "auto_macro": "external macro data",
    }.get(source_type, source_type.replace("_", " ") if source_type else "")


def _clean_cell(value: Any) -> str:
    text = str(value or "").strip()
    if text.casefold() in {"", "na", "nan", "none", "null"}:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    except Exception:
        return text
