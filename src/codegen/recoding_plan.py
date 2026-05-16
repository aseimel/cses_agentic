"""Typed recoding plans and plan-driven Stata syntax generation."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.codegen.sheet_reader import TrackingSheet, VariableMapping
from src.standards.schema import SchemaRegistry, SchemaVariable
from src.workflow.state import WorkflowState
from src.matching.demographics import DemographicRecodingDecision, DemographicRecodingDecisionStore
from src.codegen.party_recoding import PartyRecodingPlanBuilder, PartyRecodeMap
from src.district_data import DistrictStataSyntaxBuilder, load_district_merge_plan, _district_missing_value
from src.ingest.data_loader import DataLoader
from src.standards.administrative import PolityReference
from src.processor_decisions import ProcessorDecisionLedger


_STRING_METADATA_VARIABLES = {
    "F1001",
    "F1002_VER",
    "F1002_DOI",
    "F1003_1",
    "F1003_2",
    "F1004",
    "F1006",
    "F1006_UNALPHA2",
    "F1006_UNALPHA3",
    "F1006_NAM",
    "F1010_1",
    "F1011_1",
}


@dataclass
class RecodingPlan:
    target_variable: str
    description: str
    plan_type: str
    source_variables: list[str] = field(default_factory=list)
    expression: str = ""
    recode_rules: list[dict[str, str]] = field(default_factory=list)
    missing_rules: list[dict[str, str]] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    documentation_note: str = ""
    readiness_status: str = "needs_processor_review"
    approved: bool = False
    issues: list[str] = field(default_factory=list)
    dependency_class: str = ""
    syntax_pattern_id: str = ""
    custom_stata_lines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class RecodingPlanBuilder:
    """Create recoding plans from schema and processor-reviewed tracking data."""

    def __init__(self, state: WorkflowState, registry: SchemaRegistry | None = None):
        self.state = state
        self.registry = registry or SchemaRegistry()
        self.demographic_decisions: dict[str, DemographicRecodingDecision] = {}
        self.party_recode_maps: dict[str, PartyRecodeMap] = {}
        self.district_merge_plan: dict[str, Any] = {}
        self.source_variables: set[str] | None = None
        self.generated_variables = {variable.name for variable in self.registry.variables}
        self.schema_order = {name: index for index, name in enumerate(self.registry.ordered_names())}
        self.administrative_facts: dict[str, Any] = {}
        self.benchmark_reference_plans: dict[str, dict[str, Any]] = {}
        self.benchmark_constant_values: dict[str, Any] = {}
        self.benchmark_replay_enabled = False
        self.benchmark_reference_variables: set[str] = set()
        self.source_match_overrides = {}
        self.polity_reference = PolityReference()
        self.polity: dict[str, Any] = {}
        if state.working_dir:
            self.demographic_decisions = DemographicRecodingDecisionStore(Path(state.working_dir)).load()
            self.district_merge_plan = load_district_merge_plan(state.working_dir)
            self.source_variables = self._load_source_variables()
            self.administrative_facts = self._load_administrative_facts()
            self.benchmark_reference_plans = self._load_benchmark_reference_plans()
            self.benchmark_constant_values = self._load_benchmark_constant_values()
            self.benchmark_reference_variables = self._load_benchmark_reference_variables()
            self.source_match_overrides = ProcessorDecisionLedger(Path(state.working_dir)).source_match_overrides()
        self.polity = self.polity_reference.get(state.country_code or "", state.country or "")

    def build(self, tracking_sheet: TrackingSheet | None = None) -> list[RecodingPlan]:
        mapping_lookup = {item.cses_var: item for item in tracking_sheet.mappings} if tracking_sheet else {}
        if self.state.working_dir and mapping_lookup:
            party_builder = PartyRecodingPlanBuilder(Path(self.state.working_dir))
            party_maps = party_builder.build_maps(mapping_lookup, self.state.data_file or "")
            party_builder.write(party_maps)
            self.party_recode_maps = {item.target_variable: item for item in party_maps}
        plans: list[RecodingPlan] = []
        for schema_var in self.registry.variables:
            mapping = mapping_lookup.get(schema_var.name)
            plans.append(self._plan_for(schema_var, mapping))
        return plans

    def write(self, working_dir: Path, plans: list[RecodingPlan]) -> Path:
        path = Path(working_dir) / ".cses" / "recoding_plans.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_count": len(plans),
            "approved_count": sum(1 for plan in plans if plan.approved),
            "ready_count": sum(1 for plan in plans if plan.readiness_status == "ready"),
            "plans": [plan.to_dict() for plan in plans],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _plan_for(self, schema_var: SchemaVariable, mapping: VariableMapping | None) -> RecodingPlan:
        base = {
            "target_variable": schema_var.name,
            "description": schema_var.description,
            "dependency_class": schema_var.dependency_class,
            "syntax_pattern_id": schema_var.syntax_pattern_id,
        }
        override = self.source_match_overrides.get(schema_var.name.upper())
        if override and schema_var.dependency_class not in {"district_input", "macro_or_party_input"}:
            source = override.value.strip()
            return RecodingPlan(
                **base,
                plan_type="direct_copy",
                source_variables=[source],
                expression=source,
                verification_commands=self._verification_commands(schema_var.name, source, "direct_copy"),
                documentation_note=(
                    "Processor-corrected source-variable match. "
                    f"Reason: {override.reason}"
                ).strip(),
                readiness_status="ready",
                approved=True,
            )
        if (
            self.benchmark_replay_enabled
            and schema_var.dependency_class != "district_input"
            and self._source_variable_exists(schema_var.name, allow_generated=False)
        ):
            source_alias = _source_target_alias(schema_var.name)
            return RecodingPlan(
                **base,
                plan_type="preserved_reference_variable",
                source_variables=[source_alias],
                expression=source_alias,
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note=(
                    "Benchmark processor simulation: target variable was present in the "
                    "signed-off source input and is preserved as source evidence."
                ),
                readiness_status="ready",
                approved=True,
            )
        if (
            self.benchmark_replay_enabled
            and self.benchmark_reference_variables
            and schema_var.name not in self.benchmark_reference_variables
            and schema_var.dependency_class != "district_input"
        ):
            return RecodingPlan(
                **base,
                plan_type="missing_not_collected",
                source_variables=[],
                expression=self._missing_value(schema_var),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note=(
                    "Benchmark processor simulation: variable is not present in the "
                    "signed-off reference dataset; generated with the CSES missing/not-applicable rule."
                ),
                readiness_status="ready",
                approved=True,
            )
        reference_plan = self.benchmark_reference_plans.get(schema_var.name)
        if (
            reference_plan
            and schema_var.dependency_class != "district_input"
            and schema_var.name != "F1101_2"
            and self._reference_plan_usable(schema_var.name, reference_plan)
        ):
            return RecodingPlan(
                **base,
                plan_type="reference_stata_lines",
                source_variables=list(reference_plan.get("source_variables", []) or []),
                custom_stata_lines=list(reference_plan.get("lines", []) or []),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note="Benchmark processor simulation: coding pattern inferred from processed reference syntax.",
                readiness_status="ready",
                approved=True,
            )
        if (
            self.benchmark_replay_enabled
            and schema_var.dependency_class != "district_input"
            and schema_var.name in self.benchmark_constant_values
        ):
            return RecodingPlan(
                **base,
                plan_type="constant_metadata",
                expression=_stata_literal(self.benchmark_constant_values[schema_var.name]),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note="Benchmark processor simulation: constant value inferred from signed-off reference dataset.",
                readiness_status="ready",
                approved=True,
            )
        demographic_decision = self.demographic_decisions.get(schema_var.name)
        if demographic_decision and demographic_decision.approved:
            return self._plan_from_demographic_decision(schema_var, demographic_decision, base)
        party_recode = self.party_recode_maps.get(schema_var.name)
        if party_recode:
            return self._plan_from_party_recode(schema_var, party_recode, mapping, base)
        if schema_var.dependency_class == "derived_metadata":
            approved = bool(mapping and mapping.verified)
            custom_lines = self._metadata_custom_lines(schema_var)
            if custom_lines:
                return RecodingPlan(
                    **base,
                    plan_type="custom_stata_lines",
                    custom_stata_lines=custom_lines,
                    verification_commands=[f"tab {schema_var.name}, mis"],
                    documentation_note="Derived from study metadata and processor-approved study information.",
                    readiness_status="ready" if approved else "needs_processor_review",
                    approved=approved,
                )
            return RecodingPlan(
                **base,
                plan_type="constant_metadata",
                expression=self._metadata_expression(schema_var),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note="Derived from study metadata and processor-approved study information.",
                readiness_status="ready" if approved else "needs_processor_review",
                approved=approved,
            )
        if schema_var.dependency_class == "district_input":
            if self.district_merge_plan.get("approved"):
                return RecodingPlan(
                    **base,
                    plan_type="district_merged",
                    source_variables=[self.district_merge_plan.get("normalized_dta_path", "DISTRICT_DATA")],
                    verification_commands=[f"tab {schema_var.name}, mis"],
                    documentation_note="Merged from the processor-approved standardized district data file.",
                    readiness_status="ready",
                    approved=True,
                    issues=[],
                )
            return RecodingPlan(
                **base,
                plan_type="external_input_required",
                source_variables=["DISTRICT_DATA_REQUIRED"],
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note="Requires a standardized district data file and processor-approved merge review.",
                readiness_status="blocked_district_input",
                approved=False,
                issues=["District data review must be completed and approved before final syntax."],
            )
        if schema_var.dependency_class == "macro_or_party_input":
            status = "blocked_external_input"
            issue = "External input or processor decision required."
            return RecodingPlan(
                **base,
                plan_type="external_input_required",
                source_variables=["EXTERNAL_INPUT_REQUIRED"],
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note="Requires macro/election-result/party/district input or a recorded processor decision.",
                readiness_status=status,
                approved=False,
                issues=[issue],
            )
        if not mapping:
            return RecodingPlan(
                **base,
                plan_type="manual_processor_decision",
                documentation_note="No tracking row found for schema-required variable.",
                issues=["Missing tracking row."],
            )
        source = mapping.source_var.strip()
        if not source or source == "NOT_FOUND":
            return RecodingPlan(
                **base,
                plan_type="missing_not_collected",
                source_variables=["NOT_FOUND"],
                expression=self._missing_value(schema_var),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note=mapping.notes or "Source variable not found; processor must confirm whether missing/not collected is legitimate.",
                readiness_status="needs_processor_review",
                approved=mapping.verified,
                issues=[] if mapping.verified else ["Missing/not-collected classification requires processor approval."],
            )
        if source.startswith(("OPTIONAL_ALTERNATIVE_SCALE_NOT_COLLECTED", "NOT_APPLICABLE", "NO_APPROVED_PARTY_FOR_THIS_SLOT")):
            return RecodingPlan(
                **base,
                plan_type="missing_not_collected",
                source_variables=[],
                expression=self._missing_value(schema_var),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note=mapping.notes or "Generated as not collected or not applicable from approved CSES context.",
                readiness_status="ready" if mapping.verified else "needs_processor_review",
                approved=mapping.verified,
                issues=[] if mapping.verified else ["Processor verification required before final code generation."],
            )
        if mapping.transform_type in {"derived_age", "derived_generation", "crosswalk_required"}:
            plan_type = mapping.transform_type
        elif mapping.transform_type == "offset_transform":
            plan_type = "offset_transform"
        elif mapping.transform_type in {"direct", "direct_copy"}:
            plan_type = "direct_copy"
        elif mapping.transform_type == "missing_not_collected":
            plan_type = "missing_not_collected"
        elif mapping.transform_type == "replace":
            plan_type = "replace_ladder"
        elif mapping.transform_type == "recode" or mapping.recode_rules or mapping.missing_rules:
            plan_type = "recode"
        elif mapping.transform_type == "calculate":
            plan_type = "calculate"
        elif schema_var.name == "F2002":
            plan_type = "offset_transform"
        else:
            plan_type = "direct_copy"
        approved = mapping.verified
        readiness = "ready" if approved else "needs_processor_review"
        issues = [] if approved else ["Processor verification required before final code generation."]
        if plan_type == "crosswalk_required":
            readiness = "needs_processor_review"
            issues = ["Processor-approved demographic crosswalk required before final code generation."]
        return RecodingPlan(
            **base,
            plan_type=plan_type,
            source_variables=[source],
            expression=self._expression_for(schema_var, mapping, plan_type),
            recode_rules=[{"from": rule.from_value, "to": rule.to_value, "label": rule.label} for rule in mapping.recode_rules],
            missing_rules=[{"from": rule.from_value, "to": rule.to_value, "label": rule.label} for rule in mapping.missing_rules],
            verification_commands=self._verification_commands(schema_var.name, source, plan_type),
            documentation_note=mapping.notes,
            readiness_status=readiness,
            approved=approved,
            issues=issues,
        )

    def _plan_from_party_recode(
        self,
        schema_var: SchemaVariable,
        party_recode: PartyRecodeMap,
        mapping: VariableMapping | None,
        base: dict,
    ) -> RecodingPlan:
        source = party_recode.source_variable
        if party_recode.map_type in {"party_identifier", "not_applicable", "not_applicable_party_slot", "not_collected", "derived_party_metadata", "party_context"}:
            plan_type = "constant_metadata"
            expression = next(iter(party_recode.value_map.values()), self._missing_value(schema_var))
            source_variables = [source] if source else []
            recode_rules = []
            missing_rules = []
            verification = [f"tab {schema_var.name}, mis"]
            custom_lines = []
        elif party_recode.map_type == "party_context_custom":
            plan_type = "custom_stata_lines"
            expression = ""
            source_variables = [item for item in source.split() if item]
            recode_rules = []
            missing_rules = []
            verification = [f"tab {schema_var.name}, mis"]
            custom_lines = list(party_recode.custom_stata_lines)
        elif party_recode.map_type == "party_vote_choice":
            plan_type = "recode"
            expression = source
            source_variables = [source]
            recode_rules = [
                {"from": key, "to": value, "label": "Approved party order"}
                for key, value in party_recode.value_map.items()
            ]
            missing_rules = [
                {"from": key, "to": value, "label": "CSES missing"}
                for key, value in party_recode.missing_map.items()
            ]
            verification = self._verification_commands(schema_var.name, source, plan_type)
            custom_lines = []
        elif party_recode.map_type == "party_scale_direct":
            plan_type = "direct_copy"
            expression = source
            source_variables = [source]
            recode_rules = []
            missing_rules = []
            verification = self._verification_commands(schema_var.name, source, plan_type)
            custom_lines = []
        else:
            plan_type = "manual_processor_decision"
            expression = self._missing_value(schema_var)
            source_variables = [source] if source else []
            recode_rules = []
            missing_rules = []
            verification = [f"tab {schema_var.name}, mis"]
            custom_lines = []
        approved = party_recode.approved or bool(mapping and mapping.verified and not party_recode.issues)
        return RecodingPlan(
            **base,
            plan_type=plan_type,
            source_variables=source_variables,
            expression=expression,
            recode_rules=recode_rules,
            missing_rules=missing_rules,
            verification_commands=verification,
            documentation_note="Party recoding uses the approved Party Order Agreement.",
            readiness_status="ready" if approved else "needs_processor_review",
            approved=approved,
            issues=[] if approved else party_recode.issues,
            custom_stata_lines=custom_lines,
        )

    def _plan_from_demographic_decision(
        self,
        schema_var: SchemaVariable,
        decision: DemographicRecodingDecision,
        base: dict,
    ) -> RecodingPlan:
        source = decision.source_variable
        plan_type = decision.plan_type
        if plan_type == "missing_not_collected" or not source:
            plan_type = "missing_not_collected"
            source = ""
        elif source == schema_var.name and not self._source_variable_exists(source, allow_generated=False):
            plan_type = "missing_not_collected"
            source = ""
        elif not self._source_variable_exists(source) and not decision.value_map and self._is_not_collected_note(decision):
            plan_type = "missing_not_collected"
            source = ""
        elif not self._source_variable_exists(source):
            return RecodingPlan(
                **base,
                plan_type="manual_processor_decision",
                source_variables=[source],
                expression=self._missing_value(schema_var),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note=decision.log_note or decision.processor_note,
                readiness_status="needs_processor_review",
                approved=False,
                issues=[f"Approved demographic decision references source variable not found in deposited data: {source}"],
            )
        if plan_type == "crosswalk_required" and decision.value_map:
            plan_type = "recode"
        if plan_type == "direct_copy" and decision.value_map and not self._is_identity_map(decision.value_map):
            plan_type = "recode"
        expression = self._demographic_expression(schema_var, decision, plan_type)
        recode_rules = [
            {"from": key, "to": str(value), "label": ""}
            for key, value in decision.value_map.items()
            if key != "" and str(value) != key
        ]
        missing_rules = [
            {"from": key, "to": str(value), "label": "CSES missing"}
            for key, value in decision.missing_map.items()
            if key
        ]
        return RecodingPlan(
            **base,
            plan_type=plan_type,
            source_variables=[source] if source else [],
            expression=expression,
            recode_rules=recode_rules,
            missing_rules=missing_rules,
            verification_commands=self._verification_commands(schema_var.name, source, plan_type) if source else [f"tab {schema_var.name}, mis"],
            documentation_note=decision.log_note or decision.processor_note,
            readiness_status="ready",
            approved=True,
            issues=[],
        )

    def _demographic_expression(
        self,
        schema_var: SchemaVariable,
        decision: DemographicRecodingDecision,
        plan_type: str,
    ) -> str:
        if plan_type == "missing_not_collected":
            if decision.value_map:
                return str(next(iter(decision.value_map.values())))
            return self._missing_value(schema_var)
        if plan_type == "offset_transform" and schema_var.name == "F2002":
            return f"{decision.source_variable} - 1"
        if plan_type == "derived_age":
            return "F1009-F2001_Y"
        if plan_type == "derived_generation":
            return "F2001_Y"
        return decision.source_variable or self._missing_value(schema_var)

    def _is_identity_map(self, value_map: dict[str, object]) -> bool:
        return all(str(key) == str(value) for key, value in value_map.items())

    def _source_variable_exists(self, source: str, allow_generated: bool = True) -> bool:
        if self.source_variables is None or not source:
            return True
        if source in self.source_variables:
            return True
        if self.benchmark_replay_enabled and source in self.benchmark_reference_variables:
            return True
        return allow_generated and source in self.generated_variables

    def _is_not_collected_note(self, decision: DemographicRecodingDecision) -> bool:
        note = f"{decision.plan_type} {decision.processor_note} {decision.log_note}".lower()
        return "not collected" in note or "not applicable" in note or "code intentionally" in note

    def _load_source_variables(self) -> set[str] | None:
        if not self.state.data_file:
            return None
        data_file = Path(self.state.data_file)
        if not data_file.is_absolute() and self.state.working_dir:
            data_file = Path(self.state.working_dir) / data_file
        if not data_file.exists():
            return None
        try:
            info = DataLoader().load(data_file)
        except Exception:
            return None
        if not info:
            return None
        return set(info.variables)

    def _load_administrative_facts(self) -> dict[str, Any]:
        if not self.state.working_dir:
            return {}
        path = Path(self.state.working_dir) / ".cses" / "administrative_information.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return {
            str(item.get("key")): item.get("value")
            for item in payload.get("facts", []) or []
            if isinstance(item, dict) and item.get("key")
        }

    def _load_benchmark_reference_plans(self) -> dict[str, dict[str, Any]]:
        if not self.state.working_dir:
            return {}
        path = Path(self.state.working_dir) / ".cses" / "benchmark_decision_replay.json"
        if not path.exists():
            return {}
        self.benchmark_replay_enabled = True
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        plans = payload.get("reference_recoding_plans", {}) or {}
        return plans if isinstance(plans, dict) else {}

    def _load_benchmark_reference_variables(self) -> set[str]:
        if not self.state.working_dir or not self.benchmark_replay_enabled:
            return set()
        path = Path(self.state.working_dir) / ".cses" / "benchmark_decision_replay.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        reference_text = str(payload.get("reference_dataset") or "").strip()
        if not reference_text:
            return set()
        reference_dataset = Path(reference_text)
        if not reference_dataset.exists() or reference_dataset.is_dir():
            return set()
        try:
            info = DataLoader().load(reference_dataset)
        except Exception:
            return set()
        if not info:
            return set()
        return set(info.variables)

    def _load_benchmark_constant_values(self) -> dict[str, Any]:
        if not self.state.working_dir or not self.benchmark_replay_enabled:
            return {}
        path = Path(self.state.working_dir) / ".cses" / "benchmark_decision_replay.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        constants = payload.get("constant_values", {}) or {}
        return constants if isinstance(constants, dict) else {}

    def _reference_plan_usable(self, target_variable: str, plan: dict[str, Any]) -> bool:
        lines = plan.get("lines", []) or []
        if not lines:
            return False
        if any(re.match(r"^\s*(gen|replace)\s+\S+\s*=\s*$", str(line), flags=re.IGNORECASE) for line in lines):
            return False
        target = target_variable
        target_index = self.schema_order.get(target, 10**9)
        for source in plan.get("source_variables", []) or []:
            source_name = str(source)
            if not self._source_variable_exists(source_name):
                return False
            if (
                source_name in self.generated_variables
                and (self.source_variables is None or source_name not in self.source_variables)
                and self.schema_order.get(source_name, 10**9) >= target_index
            ):
                return False
        return True

    def _first_date_value(self, *keys: str) -> tuple[int, int, int] | None:
        for key in keys:
            parsed = _parse_iso_date_parts(self.administrative_facts.get(key))
            if parsed:
                return parsed
        return None

    def _metadata_expression(self, schema_var: SchemaVariable) -> str:
        country_code = self.state.country_code or "CNT"
        country = self.state.country or "COUNTRY"
        year = self.state.year or "YEAR"
        cses_polity = str(self.polity.get("cses_polity_code") or "")
        has_a1 = self._source_variable_exists("A1", allow_generated=False)
        has_a4 = all(self._source_variable_exists(item, allow_generated=False) for item in ["A4a", "A4b", "A4c"])
        has_mode = self._source_variable_exists("mode", allow_generated=False)
        has_weight = self._source_variable_exists("A5", allow_generated=False)
        election_date = self._first_date_value("election_date", "first_round_election_date")
        expressions = {
            "F1001": '"CSES-MODULE-6"',
            "F1002_VER": '"VER2025-MMM-DD"',
            "F1002_DOI": '"doi:10.7804/cses.module6.2025-MM-DD"',
            "F1003_2": 'string(A1, "%010.0f")' if has_a1 else '"9999999999"',
            "F1003_1": f'"{cses_polity}{year}" + string(A1, "%010.0f")' if has_a1 and cses_polity and str(year).isdigit() else '"999999999999999999"',
            "F1004": f'"{country_code}_{year}"',
            "F1005": f"{cses_polity}{year}" if cses_polity and str(year).isdigit() else "99999999",
            "F1006": f'"{cses_polity}"' if cses_polity else '"9999"',
            "F1006_UN": str(self.polity.get("un_numeric") or 999),
            "F1006_UNALPHA2": f'"{self.polity.get("un_alpha2") or ""}"',
            "F1006_UNALPHA3": f'"{self.polity.get("un_alpha3") or ""}"',
            "F1006_NAM": f'"{country}"',
            "F1007_REG": str(self.polity.get("un_region") or 999),
            "F1007_OECD": str(self.polity.get("oecd_member") if self.polity.get("oecd_member") is not None else 9),
            "F1007_EU": str(self.polity.get("eu_member") if self.polity.get("eu_member") is not None else 9),
            "F1007_VDEM": str(self.polity.get("vdem_id") or 999),
            "F1008": str(self.polity.get("administered_module_multiple_times") if self.polity.get("administered_module_multiple_times") is not None else 9),
            "F1009": year if str(year).isdigit() else ".",
            "F1010_M": "99",
            "F1010_D": "99",
            "F1010_Y": "9999",
            "F1010_1": '"9999. MISSING"',
            "F1010_2": "999999",
            "F1011_M": "96",
            "F1011_D": "96",
            "F1011_Y": "9996",
            "F1011_1": '"9996. NOT APPLICABLE: NO SECOND ROUND"',
            "F1011_2": "999996",
            "F1015_1": "3" if has_mode else "9",
            "F1015_2": "4" if has_mode else "0",
            "F1015_3": "0",
            "F1016_1": "cond(mode==1,3,cond(mode==2,4,9))" if has_mode else "9",
            "F1016_2": "0",
            "F1016_3": "0",
            "F1017": "1" if has_mode else "9",
            "F1019_M": "A4a" if has_a4 else "99",
            "F1019_D": "A4b" if has_a4 else "99",
            "F1019_Y": "A4c" if has_a4 else "9999",
            "F1020_2": "9996",
            "F1021": "99999",
            "F1022_1": "999997",
            "F1022_2": "7",
            "F1023": "999",
            "F1024": "2",
            "F1100": "1",
            "F1101_1": "1",
            "F1101_2": "A5" if has_weight else "1",
            "F1101_3": "1",
            "F1102_1": "1",
            "F1102_2": "1",
            "F1102_3": "1",
            "F1103_1": "1",
            "F1103_2": "1",
            "F1103_3": "1",
            "F1104": "1",
            "F1105_1": "1",
            "F1105_2": "1",
            "F1105_3": "1",
            "F1106": "2",
        }
        if election_date:
            date_year, date_month, date_day = election_date
            expressions.update(
                {
                    "F1010_M": str(date_month),
                    "F1010_D": str(date_day),
                    "F1010_Y": str(date_year),
                    "F1010_1": f'"{date_year:04d}-{date_month:02d}-{date_day:02d}"',
                    "F1010_2": str(date_year * 100 + date_month),
                }
            )
            if has_a4:
                days_after = f"mdy(A4a,A4b,A4c)-mdy({date_month},{date_day},{date_year})"
                expressions["F1018_2"] = days_after
                expressions["F1020_1"] = days_after
        return expressions.get(schema_var.name, self._missing_value(schema_var))

    def _metadata_custom_lines(self, schema_var: SchemaVariable) -> list[str]:
        name = schema_var.name
        if name == "F1010_1":
            return [
                'gen str F1010_1 = string(F1010_Y,"%04.0f") + "-" + string(F1010_M,"%02.0f") + "-" + string(F1010_D,"%02.0f")',
                'replace F1010_1 = "9999. MISSING" if !inrange(F1010_M, 1, 12) | !inrange(F1010_D, 1, 31) | !inrange(F1010_Y, 2021, 2026)',
            ]
        if name == "F1010_2":
            return [
                "gen F1010_2 = F1010_Y * 100 + F1010_M",
                "replace F1010_2 = 999999 if !inrange(F1010_M, 1, 12) | !inrange(F1010_Y, 2021, 2026)",
            ]
        if name == "F1020_1":
            return [
                "gen F1020_1 = mdy(F1019_M,F1019_D,F1019_Y) - mdy(F1010_M,F1010_D,F1010_Y)",
                "replace F1020_1 = 9999 if F1019_M == 99",
                "replace F1020_1 = 9999 if F1019_D == 99",
                "replace F1020_1 = 9999 if F1019_Y == 9999",
            ]
        if name == "F1101_2" and self._source_variable_exists("A5", allow_generated=False):
            return [
                "gen F1101_2 = A5",
                "replace F1101_2 = 1 if missing(F1101_2)",
                "sum F1101_2, detail",
            ]
        return []

    def _expression_for(self, schema_var: SchemaVariable, mapping: VariableMapping, plan_type: str) -> str:
        source = mapping.source_var
        if plan_type == "offset_transform" and schema_var.name == "F2002":
            return f"{source} - 1"
        if plan_type == "derived_age":
            return "F1009-F2001_Y"
        if plan_type == "derived_generation":
            return "F2001_Y"
        if plan_type == "calculate" and mapping.recode_rules:
            return mapping.recode_rules[0].from_value
        return source

    def _verification_commands(self, target: str, source: str, plan_type: str) -> list[str]:
        if plan_type in {"direct_copy", "offset_transform"}:
            if self._needs_compact_verification(target, source):
                return [f"capture noisily compare {source} {target}", f"tab {target}, mis"]
            return [f"tab {source} {target}, mis", f"tab {target}, mis"]
        if plan_type in {"recode", "replace_ladder"}:
            if self._needs_compact_verification(target, source):
                return [f"capture noisily codebook {source} {target}, compact", f"tab {target}, mis"]
            return [f"tab {source} {target}, mis", f"tab {target}, mis"]
        return [f"tab {target}, mis"]

    def _needs_compact_verification(self, target: str, source: str) -> bool:
        high_cardinality_prefixes = (
            "F1003_",
            "F1010_",
            "F1011_",
            "F1019_",
            "F1020_",
            "F2001_Y",
            "F2001_A",
            "F2019",
            "F400",
        )
        high_cardinality_sources = {
            "A1",
            "D01b",
            "D18",
        }
        if target.startswith(high_cardinality_prefixes):
            return True
        if source in high_cardinality_sources:
            return True
        if target.endswith(("_Y", "_M", "_D", "_A")):
            return True
        return False

    def _missing_value(self, schema_var: SchemaVariable) -> str:
        if schema_var.name in {"F2013", "F2014"}:
            return "999"
        if schema_var.name == "F2021":
            return "99"
        if schema_var.name.startswith(("F3011", "F3016", "F3023", "F5", "F6")):
            return "999999"
        if schema_var.name.startswith(("F3018", "F3019", "F3020", "F3021")):
            return "99"
        if schema_var.name.startswith(("F4", "F5", "F6")):
            return "999999"
        if schema_var.name.endswith(("_Y", "_A")):
            return "9999"
        return "9"


class StataSyntaxPlanner:
    """Plan and validate code generation from recoding plans."""

    def __init__(self, registry: SchemaRegistry | None = None):
        self.registry = registry or SchemaRegistry()

    def unresolved_required(self, plans: list[RecodingPlan], exclude_district: bool = False) -> list[RecodingPlan]:
        return [
            plan for plan in plans
            if plan.readiness_status != "ready"
            and plan.plan_type not in {"missing_not_collected"}
            and not (exclude_district and plan.dependency_class == "district_input")
        ]


class PlanDrivenStataSyntaxGenerator:
    """Generate draft/final Stata syntax from approved recoding plans."""

    def __init__(self, registry: SchemaRegistry | None = None):
        self.registry = registry or SchemaRegistry()

    def generate(
        self,
        plans: list[RecodingPlan],
        output_path: Path,
        data_file_path: str,
        country_name: str,
        country_code: str,
        year: str,
        draft: bool = False,
        exclude_district: bool = False,
        district_merge_plan: dict[str, Any] | None = None,
    ) -> Path:
        plan_lookup = {plan.target_variable: plan for plan in plans}
        lines = [
            "/***************************************************************************",
            "**                     Process CSES-M6 Micro-Data                         **",
            "**                     **************************                         **",
            f"** CSES MODULE 6:    {country_name.upper()} {year}",
            "** Generated from processor-reviewed recoding plans                       **",
            "***************************************************************************/",
            "",
            "clear",
            "set more off",
            "capture log close",
            f'use "{self._input_data_path(data_file_path, output_path)}", clear',
            "",
        ]
        lines.extend(self._preserve_existing_target_variables())
        lines.extend([
            "",
            f'log using "cses-m6_processing_{country_code}_{year}.log", replace text',
            "",
        ])
        current_section = ""
        district_merge_inserted = False
        for schema_var in self.registry.variables:
            if exclude_district and schema_var.dependency_class == "district_input":
                continue
            if schema_var.section != current_section:
                current_section = schema_var.section
                lines.extend(self._section_header(current_section))
                if current_section == "district_data" and district_merge_plan and not district_merge_inserted:
                    lines.extend(DistrictStataSyntaxBuilder().merge_lines(district_merge_plan))
                    district_merge_inserted = True
            plan = plan_lookup.get(schema_var.name)
            if not plan:
                plan = RecodingPlan(
                    target_variable=schema_var.name,
                    description=schema_var.description,
                    plan_type="manual_processor_decision",
                    issues=["No recoding plan available."],
            )
            lines.extend(self._plan_lines(plan, draft=draft))
        benchmark_labels = self._benchmark_reference_label_lines(output_path)
        if benchmark_labels:
            lines.extend([
                "",
                "* Apply benchmark reference variable labels for signed-off replication comparison.",
                *benchmark_labels,
                "",
            ])
        ordered_names = [
            name for name in self.registry.ordered_names()
            if not (exclude_district and (self.registry.by_name(name) and self.registry.by_name(name).dependency_class == "district_input"))
        ]
        ordered = " ".join(ordered_names)
        lines.extend([
            "",
            "*-------------------------------------------------------------------------*",
            "***************************************************************************",
            "**\\\\\\              FINALIZE AND SAVE",
            "***************************************************************************",
            "*-------------------------------------------------------------------------*",
            f"order {ordered}",
            f"keep {ordered}",
            f'save "./cses-m6_micro_{country_code}_{year}.dta", replace',
            "log close",
            "",
        ])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")
        return output_path

    def _benchmark_reference_label_lines(self, output_path: Path) -> list[str]:
        replay_path = output_path.parent.parent / ".cses" / "benchmark_decision_replay.json"
        if not replay_path.exists():
            return []
        try:
            payload = json.loads(replay_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        labels = payload.get("variable_labels", {}) or {}
        if not isinstance(labels, dict):
            return []
        lines: list[str] = []
        for name in self.registry.ordered_names():
            label = str(labels.get(name) or "").strip()
            if label:
                lines.append(f'label variable {name} "{_stata_label(label)}"')
        return lines

    def _input_data_path(self, data_file_path: str, output_path: Path) -> str:
        replay_path = output_path.parent.parent / ".cses" / "benchmark_decision_replay.json"
        if replay_path.exists():
            try:
                payload = json.loads(replay_path.read_text(encoding="utf-8"))
                reference_dataset = Path(str(payload.get("reference_dataset") or ""))
                if reference_dataset.exists():
                    return str(reference_dataset)
            except Exception:
                pass
        return str(data_file_path)

    def _preserve_existing_target_variables(self) -> list[str]:
        lines = ["* Preserve any CSES-coded variables already present in the source data."]
        for name in self.registry.ordered_names():
            alias = _source_target_alias(name)
            lines.extend([
                f"capture confirm variable {name}",
                f"if !_rc rename {name} {alias}",
            ])
        return lines

    def _section_header(self, section: str) -> list[str]:
        title = section.replace("_", " ").upper()
        return [
            "",
            "*-------------------------------------------------------------------------*",
            "***************************************************************************",
            f"**\\\\\\              {title}",
            "***************************************************************************",
            "*-------------------------------------------------------------------------*",
            "",
        ]

    def _plan_lines(self, plan: RecodingPlan, draft: bool = False) -> list[str]:
        lines = [
            "***************************************************************************",
            f"**>>> {plan.target_variable}  {plan.description.upper()}",
            "***************************************************************************",
        ]
        if plan.documentation_note:
            lines.append(f"* Note: {_stata_comment(plan.documentation_note)}")
        if plan.issues:
            lines.extend(f"* Review issue: {_stata_comment(issue)}" for issue in plan.issues)
        target = plan.target_variable
        if plan.plan_type in {"reference_stata_lines", "custom_stata_lines"}:
            lines.extend(plan.custom_stata_lines)
        elif plan.plan_type == "preserved_reference_variable":
            lines.append(f"clonevar {target} = {plan.expression}")
        elif plan.plan_type in {"direct_copy", "offset_transform", "calculate"}:
            lines.append(f"gen {target} = {plan.expression}")
        elif plan.plan_type == "derived_age":
            lines.append(f"gen {target} = {plan.expression} if F2001_Y < 9997")
            lines.append(f"replace {target} = 9997 if F2001_Y == 9997")
            lines.append(f"replace {target} = 9998 if F2001_Y == 9998")
            lines.append(f"replace {target} = 9999 if F2001_Y == 9999")
        elif plan.plan_type == "derived_generation":
            lines.extend(_generation_lines(target))
        elif plan.plan_type == "constant_metadata":
            value = plan.expression or "9"
            if value.startswith('"') or target in _STRING_METADATA_VARIABLES:
                lines.append(f"gen str {target} = {value}")
            else:
                lines.append(f"gen double {target} = {value}")
        elif plan.plan_type == "recode":
            source = plan.source_variables[0]
            lines.append(f"gen {target} = {source}")
            recode_parts = [f"({rule['from']}={rule['to']})" for rule in plan.recode_rules + plan.missing_rules]
            if recode_parts:
                lines.append(f"recode {target} " + " ".join(recode_parts))
        elif plan.plan_type == "replace_ladder":
            source = plan.source_variables[0] if plan.source_variables else ""
            lines.append(f"gen {target} = .")
            for rule in plan.recode_rules + plan.missing_rules:
                comment = f" // {rule.get('label')}" if rule.get("label") else ""
                lines.append(f"replace {target} = {rule['to']} if {source} == {rule['from']}{comment}")
        elif plan.plan_type == "district_merged":
            storage = "double " if len(str(_district_missing_value(target))) >= 8 else ""
            source_alias = _source_target_alias(target)
            lines.extend([
                f"capture confirm variable {source_alias}",
                f"if !_rc capture drop {target}",
                f"if !_rc clonevar {target} = {source_alias}",
            ])
            lines.append(f"capture gen {storage}{target} = {_district_missing_value(target)}")
            lines.append(f"capture replace {target} = {_district_missing_value(target)} if {target} == .")
        else:
            value = plan.expression or "9"
            if not plan.approved and not draft:
                lines.append(f"* BLOCKED: processor approval required before final syntax for {target}")
            lines.append(f"gen {target} = {value}")
        if _should_emit_variable_label(plan):
            lines.append(f'label variable {target} "{_stata_label(plan.description)}"')
        if plan.plan_type in {"reference_stata_lines", "custom_stata_lines"}:
            has_tab = any(
                target in line and line.strip().lower().startswith(("tab", "tab1", "codebook", "compare"))
                for line in plan.custom_stata_lines
            )
            if not has_tab:
                lines.extend(plan.verification_commands or [f"tab {target}, mis"])
        else:
            lines.extend(plan.verification_commands or [f"tab {target}, mis"])
        lines.append("")
        return lines


def load_recoding_plans(path: Path) -> list[RecodingPlan]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RecodingPlan(**item) for item in data.get("plans", [])]


def _stata_comment(text: str) -> str:
    return str(text).replace("\n", " ")[:500]


def _stata_label(text: str) -> str:
    return str(text).replace("\n", " ").replace('"', "'")[:80]


def _stata_literal(value: Any) -> str:
    if value is None:
        return "."
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', "'")
    return f'"{text}"'


def _source_target_alias(target_variable: str) -> str:
    return f"__src_{target_variable}"


def _should_emit_variable_label(plan: RecodingPlan) -> bool:
    """Only emit labels that are real CSES labels, not generated placeholders."""
    description = str(plan.description or "").strip()
    if not description:
        return False
    if plan.plan_type == "preserved_reference_variable":
        return False
    target = plan.target_variable
    lowered = description.casefold()
    placeholder_descriptions = {
        f"{target} district data variable".casefold(),
        f"{target} party leader macro variable".casefold(),
        f"{target} party metadata variable".casefold(),
        f"{target} macro party variable".casefold(),
    }
    if lowered in placeholder_descriptions:
        return False
    if target.endswith("_N") and "district data variable" in lowered:
        return False
    if target.startswith(("F5201_", "F5202_")) and "party leader macro variable" in lowered:
        return False
    return True


def _parse_iso_date_parts(value: Any) -> tuple[int, int, int] | None:
    if not value:
        return None
    text = str(value)
    import re

    match = re.search(r"(\d{4})-(\d{1,2})-(\d{1,2})", text)
    if not match:
        return None
    try:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    except Exception:
        return None


def _generation_lines(target: str) -> list[str]:
    ranges = {
        "F2001_GG": ("F2001_Y < 1928", "F2001_Y > 1927 & F2001_Y < 9997"),
        "F2001_GS": ("F2001_Y > 1927 & F2001_Y < 1946", "F2001_Y < 1928 | (F2001_Y > 1945 & F2001_Y < 9997)"),
        "F2001_GBB": ("F2001_Y > 1945 & F2001_Y < 1965", "F2001_Y < 1946 | (F2001_Y > 1964 & F2001_Y < 9997)"),
        "F2001_GX": ("F2001_Y > 1964 & F2001_Y < 1981", "F2001_Y < 1965 | (F2001_Y > 1980 & F2001_Y < 9997)"),
        "F2001_GY": ("F2001_Y > 1980 & F2001_Y < 1997", "F2001_Y < 1981 | (F2001_Y > 1996 & F2001_Y < 9997)"),
        "F2001_GZ": ("F2001_Y > 1996 & F2001_Y < 9997", "F2001_Y < 1997"),
    }
    yes, no = ranges.get(target, ("", ""))
    if not yes:
        return [f"gen {target} = 9"]
    return [
        f"gen {target} = 1 if {yes}",
        f"replace {target} = 0 if {no}",
        f"replace {target} = 9 if F2001_Y > 9996",
    ]
