"""Typed recoding plans and plan-driven Stata syntax generation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.codegen.sheet_reader import TrackingSheet, VariableMapping
from src.standards.schema import SchemaRegistry, SchemaVariable
from src.workflow.state import WorkflowState
from src.matching.demographics import DemographicRecodingDecision, DemographicRecodingDecisionStore
from src.codegen.party_recoding import PartyRecodingPlanBuilder, PartyRecodeMap
from src.district_data import DistrictStataSyntaxBuilder, load_district_merge_plan


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
        if state.working_dir:
            self.demographic_decisions = DemographicRecodingDecisionStore(Path(state.working_dir)).load()
            self.district_merge_plan = load_district_merge_plan(state.working_dir)

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
        demographic_decision = self.demographic_decisions.get(schema_var.name)
        if demographic_decision and demographic_decision.approved:
            return self._plan_from_demographic_decision(schema_var, demographic_decision, base)
        party_recode = self.party_recode_maps.get(schema_var.name)
        if party_recode:
            return self._plan_from_party_recode(schema_var, party_recode, mapping, base)
        if schema_var.dependency_class == "derived_metadata":
            return RecodingPlan(
                **base,
                plan_type="constant_metadata",
                expression=self._metadata_expression(schema_var),
                verification_commands=[f"tab {schema_var.name}, mis"],
                documentation_note="Derived from study metadata and processor-approved study information.",
                readiness_status="needs_processor_review",
                approved=bool(mapping and mapping.verified),
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
        elif party_recode.map_type == "party_scale_direct":
            plan_type = "direct_copy"
            expression = source
            source_variables = [source]
            recode_rules = []
            missing_rules = []
            verification = self._verification_commands(schema_var.name, source, plan_type)
        else:
            plan_type = "manual_processor_decision"
            expression = self._missing_value(schema_var)
            source_variables = [source] if source else []
            recode_rules = []
            missing_rules = []
            verification = [f"tab {schema_var.name}, mis"]
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
        )

    def _plan_from_demographic_decision(
        self,
        schema_var: SchemaVariable,
        decision: DemographicRecodingDecision,
        base: dict,
    ) -> RecodingPlan:
        source = decision.source_variable
        plan_type = decision.plan_type
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
        if plan_type == "offset_transform" and schema_var.name == "F2002":
            return f"{decision.source_variable} - 1"
        if plan_type == "derived_age":
            return "F1009-F2001_Y"
        if plan_type == "derived_generation":
            return "F2001_Y"
        return decision.source_variable or self._missing_value(schema_var)

    def _is_identity_map(self, value_map: dict[str, object]) -> bool:
        return all(str(key) == str(value) for key, value in value_map.items())

    def _metadata_expression(self, schema_var: SchemaVariable) -> str:
        country_code = self.state.country_code or "CNT"
        country = self.state.country or "COUNTRY"
        year = self.state.year or "YEAR"
        expressions = {
            "F1001": '"CSES-MODULE-6"',
            "F1004": f'"{country_code}_{year}"',
            "F1006_NAM": f'"{country}"',
            "F1009": year if str(year).isdigit() else ".",
        }
        return expressions.get(schema_var.name, self._missing_value(schema_var))

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
        if plan_type in {"direct_copy", "offset_transform", "recode", "replace_ladder"}:
            return [f"tab {source} {target}, mis", f"tab {target}, mis"]
        return [f"tab {target}, mis"]

    def _missing_value(self, schema_var: SchemaVariable) -> str:
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
            f'use "{data_file_path}", clear',
            "",
            f'log using "cses-m6_micro_{country_code}_{year}.log", replace text',
            "",
        ]
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
        if plan.plan_type in {"direct_copy", "offset_transform", "calculate"}:
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
            if value.startswith('"'):
                lines.append(f"gen str {target} = {value}")
            else:
                lines.append(f"gen {target} = {value}")
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
            lines.append(f"capture confirm variable {target}")
            lines.append(f'if _rc display as error "District variable not found after merge: {target}"')
        else:
            value = plan.expression or "9"
            if not plan.approved and not draft:
                lines.append(f"* BLOCKED: processor approval required before final syntax for {target}")
            lines.append(f"gen {target} = {value}")
        lines.extend(plan.verification_commands or [f"tab {target}, mis"])
        lines.append("")
        return lines


def load_recoding_plans(path: Path) -> list[RecodingPlan]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [RecodingPlan(**item) for item in data.get("plans", [])]


def _stata_comment(text: str) -> str:
    return str(text).replace("\n", " ")[:500]


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
