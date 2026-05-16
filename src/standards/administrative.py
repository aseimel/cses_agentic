"""Administrative information contract and planning for CSES Module 6."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.ingest.data_loader import DatasetInfo
from src.standards.schema import DEFAULT_WIKI_ROOT, SchemaRegistry, SchemaVariable
from src.workflow.state import WorkflowState


REGISTRY_PATH = DEFAULT_WIKI_ROOT / "patterns" / "administrative_variable_registry.json"
POLITY_REFERENCE_PATH = DEFAULT_WIKI_ROOT / "patterns" / "polity_reference.json"


@dataclass(frozen=True)
class AdministrativeVariableContract:
    target_variable: str
    description: str
    generation_category: str
    evidence_rule: str
    stata_rule: str
    requires_processor_confirmation: bool
    validation_checks: list[str] = field(default_factory=list)
    source_aliases: list[str] = field(default_factory=list)
    missing_value_rule: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdministrativeFact:
    key: str
    value: Any
    status: str
    evidence: str
    source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AdministrativeVariablePlan:
    target_variable: str
    description: str
    generation_category: str
    status: str
    source_variable: str = ""
    value: Any = ""
    evidence: list[str] = field(default_factory=list)
    validation_checks: list[str] = field(default_factory=list)
    processor_review_required: bool = True
    missing_value_rule: str = ""
    stata_rule: str = ""
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AdministrativeVariableRegistry:
    """Load the wiki-backed administrative variable contract."""

    def __init__(self, wiki_root: Path = DEFAULT_WIKI_ROOT, schema: SchemaRegistry | None = None):
        self.wiki_root = Path(wiki_root)
        self.registry_path = self.wiki_root / "patterns" / "administrative_variable_registry.json"
        self.schema = schema or SchemaRegistry(self.wiki_root)
        self.payload = self._load()
        self.default_missing_value_rule = str(self.payload.get("default_missing_value_rule", ""))

    def _load(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {"schema_version": 0, "rules": [], "default_missing_value_rule": ""}

    def contract_for(self, item: SchemaVariable | str) -> AdministrativeVariableContract | None:
        if isinstance(item, str):
            schema_item = self.schema.by_name(item)
            if not schema_item:
                return None
        else:
            schema_item = item
        if not _is_administrative(schema_item.name):
            return None
        rule = self._rule_for(schema_item.name)
        if not rule:
            return None
        return AdministrativeVariableContract(
            target_variable=schema_item.name,
            description=schema_item.description,
            generation_category=str(rule.get("generation_category", "")),
            evidence_rule=str(rule.get("evidence_rule", "")),
            stata_rule=str(rule.get("stata_rule", "")),
            requires_processor_confirmation=bool(rule.get("requires_processor_confirmation", True)),
            validation_checks=list(rule.get("validation_checks", []) or []),
            source_aliases=list(rule.get("source_aliases", []) or []),
            missing_value_rule=str(rule.get("missing_value_rule") or self.default_missing_value_rule),
        )

    def contracts(self) -> list[AdministrativeVariableContract]:
        results = []
        for item in self.schema.variables:
            contract = self.contract_for(item)
            if contract:
                results.append(contract)
        return results

    def validate(self) -> list[str]:
        issues: list[str] = []
        if not self.registry_path.exists():
            issues.append(f"Administrative variable registry not found: {self.registry_path}")
            return issues
        for item in self.schema.variables:
            if _is_administrative(item.name) and not self.contract_for(item):
                issues.append(f"Missing administrative contract for {item.name}")
        for contract in self.contracts():
            if not contract.generation_category:
                issues.append(f"Missing generation category for {contract.target_variable}")
            if not contract.evidence_rule:
                issues.append(f"Missing evidence rule for {contract.target_variable}")
            if not contract.validation_checks:
                issues.append(f"Missing validation checks for {contract.target_variable}")
            if not contract.stata_rule:
                issues.append(f"Missing Stata rule for {contract.target_variable}")
        return issues

    def _rule_for(self, variable: str) -> dict[str, Any] | None:
        for rule in self.payload.get("rules", []) or []:
            if variable in set(rule.get("match", []) or []):
                return rule
            prefix = rule.get("prefix")
            if prefix and variable.startswith(str(prefix)):
                return rule
        return None


class PolityReference:
    """Runtime CSES polity/country reference data."""

    def __init__(self, wiki_root: Path = DEFAULT_WIKI_ROOT):
        self.path = Path(wiki_root) / "patterns" / "polity_reference.json"
        self.payload = self._load()

    def _load(self) -> dict[str, Any]:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"polities": []}

    def get(self, country_code: str = "", country_name: str = "") -> dict[str, Any]:
        code = str(country_code or "").upper()
        name = str(country_name or "").casefold()
        for item in self.payload.get("polities", []) or []:
            if code and str(item.get("country_code", "")).upper() == code:
                return item
            if name and str(item.get("country_name", "")).casefold() == name:
                return item
        return {}

    def validate(self) -> list[str]:
        issues = []
        if not self.path.exists():
            issues.append(f"Polity reference not found: {self.path}")
        for item in self.payload.get("polities", []) or []:
            for key in ["country_code", "country_name", "cses_polity_code", "un_numeric", "un_alpha2", "un_alpha3"]:
                if item.get(key) in {"", None}:
                    issues.append(f"Polity reference row missing {key}: {item}")
        return issues


class AdministrativeFactBuilder:
    """Build administrative facts from existing study setup and reviewed materials."""

    def __init__(self, working_dir: Path, registry: AdministrativeVariableRegistry | None = None):
        self.working_dir = Path(working_dir)
        self.registry = registry or AdministrativeVariableRegistry()

    def build(
        self,
        state: WorkflowState,
        dataset_info: DatasetInfo | None = None,
        evidence_packet: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        evidence_packet = evidence_packet or self._load_evidence_packet()
        facts: list[AdministrativeFact] = []
        design_facts = evidence_packet.get("design_facts", {}) if isinstance(evidence_packet, dict) else {}
        facts.extend(self._study_identity_facts(state, dataset_info))
        facts.extend(self._processor_decision_facts(state))
        facts.extend(self._design_facts(design_facts))
        facts.extend(self._source_variable_facts(dataset_info))
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "study": {
                "country": state.country,
                "country_code": state.country_code,
                "year": state.year,
            },
            "facts": [fact.to_dict() for fact in facts],
        }
        self._write_json("administrative_information.json", payload)
        return payload

    def _load_evidence_packet(self) -> dict[str, Any]:
        path = self.working_dir / ".cses" / "evidence_packet.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _study_identity_facts(self, state: WorkflowState, dataset_info: DatasetInfo | None) -> list[AdministrativeFact]:
        facts = [
            AdministrativeFact("country", state.country, _present_status(state.country), "Study setup"),
            AdministrativeFact("country_code", state.country_code, _present_status(state.country_code), "Study setup"),
            AdministrativeFact("election_year", state.year, _present_status(state.year), "Study setup"),
        ]
        if dataset_info:
            facts.extend(
                [
                    AdministrativeFact("row_count", dataset_info.n_rows, "confirmed", "Loaded deposited data file", str(dataset_info.file_path.name)),
                    AdministrativeFact("variable_count", dataset_info.n_variables, "confirmed", "Loaded deposited data file", str(dataset_info.file_path.name)),
                    AdministrativeFact("data_file", str(dataset_info.file_path.name), "confirmed", "Selected deposited data file", str(dataset_info.file_path.name)),
                ]
            )
        return facts

    def _processor_decision_facts(self, state: WorkflowState) -> list[AdministrativeFact]:
        facts = []
        for decision in state.processor_decisions or []:
            key = decision.get("field") or decision.get("key") or decision.get("target_variable")
            if not key:
                continue
            facts.append(
                AdministrativeFact(
                    key=str(key),
                    value=decision.get("value"),
                    status="confirmed",
                    evidence=str(decision.get("evidence") or "Processor decision"),
                    source="processor_decision",
                )
            )
        return facts

    def _design_facts(self, design_facts: dict[str, Any]) -> list[AdministrativeFact]:
        mapping = {
            "mode": "Mode from reviewed study materials",
            "fieldwork_dates": "Fieldwork period from reviewed study materials",
            "weights": "Weighting information from reviewed study materials",
            "response_rate": "Response rate from reviewed study materials",
            "sample_design": "Sample design from reviewed study materials",
            "sample_size": "Sample size from reviewed study materials",
            "election_date": "Election date from reviewed study materials",
            "election_type": "Election type from reviewed study materials",
            "study_timing": "Study timing from reviewed study materials",
        }
        facts: list[AdministrativeFact] = []
        for key, evidence in mapping.items():
            value = design_facts.get(key)
            if value:
                facts.append(AdministrativeFact(key, value, "proposed", evidence))
        return facts

    def _source_variable_facts(self, dataset_info: DatasetInfo | None) -> list[AdministrativeFact]:
        if not dataset_info:
            return []
        facts = []
        for variable in dataset_info.variables.values():
            facts.append(
                AdministrativeFact(
                    key=f"source_variable:{variable.name}",
                    value={
                        "name": variable.name,
                        "description": variable.description or "",
                        "sample_values": variable.sample_values[:5],
                        "value_labels": variable.value_labels or {},
                        "dtype": variable.dtype or "",
                        "n_unique": variable.n_unique,
                    },
                    status="confirmed",
                    evidence="Variable present in deposited data",
                    source=str(dataset_info.file_path.name),
                )
            )
        return facts

    def _write_json(self, filename: str, payload: dict[str, Any]) -> Path:
        output = self.working_dir / ".cses" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        return output


class AdministrativeVariablePlanner:
    """Convert administrative facts into deterministic variable plans."""

    def __init__(self, registry: AdministrativeVariableRegistry | None = None):
        self.registry = registry or AdministrativeVariableRegistry()

    def build_plans(
        self,
        facts_payload: dict[str, Any],
        source_variables: set[str] | None = None,
    ) -> list[AdministrativeVariablePlan]:
        source_variables = {item.upper(): item for item in (source_variables or set())}
        facts = {item.get("key"): item for item in facts_payload.get("facts", []) if isinstance(item, dict)}
        plans = []
        for contract in self.registry.contracts():
            source = self._select_source_variable(contract, source_variables)
            value, value_evidence = self._derive_value(contract, facts)
            status = self._status_for(contract, source, value)
            evidence = [contract.evidence_rule]
            if source:
                evidence.append(f"Source variable available: {source}")
            if value_evidence:
                evidence.append(value_evidence)
            notes = self._notes_for(contract, status)
            plans.append(
                AdministrativeVariablePlan(
                    target_variable=contract.target_variable,
                    description=contract.description,
                    generation_category=contract.generation_category,
                    status=status,
                    source_variable=source,
                    value=value,
                    evidence=evidence,
                    validation_checks=contract.validation_checks,
                    processor_review_required=contract.requires_processor_confirmation or status == "needs_processor_review",
                    missing_value_rule=contract.missing_value_rule,
                    stata_rule=contract.stata_rule,
                    notes=notes,
                )
            )
        return plans

    def write_artifacts(self, working_dir: Path, facts_payload: dict[str, Any], plans: list[AdministrativeVariablePlan]) -> Path:
        output = Path(working_dir) / ".cses" / "administrative_variable_plans.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "study": facts_payload.get("study", {}),
                    "summary": administrative_plan_summary(plans),
                    "plans": [plan.to_dict() for plan in plans],
                },
                indent=2,
                ensure_ascii=False,
                default=str,
            ),
            encoding="utf-8",
        )
        return output

    def _select_source_variable(self, contract: AdministrativeVariableContract, source_variables: dict[str, str]) -> str:
        if contract.target_variable in {"F1019_M"}:
            aliases = ["A4a", "A04a"]
        elif contract.target_variable in {"F1019_D"}:
            aliases = ["A4b", "A04b"]
        elif contract.target_variable in {"F1019_Y"}:
            aliases = ["A4c", "A04c"]
        elif contract.target_variable in {"F1101_1", "F1101_2", "F1101_3"}:
            aliases = ["A5", "A05", "weight", "weights", "wgt"]
        else:
            aliases = contract.source_aliases
        for alias in aliases:
            if alias.upper() in source_variables:
                return source_variables[alias.upper()]
        return ""

    def _derive_value(self, contract: AdministrativeVariableContract, facts: dict[str, dict[str, Any]]) -> tuple[Any, str]:
        study = {
            "F1001": "CSES-MODULE-6",
            "F1004": _study_id(facts),
            "F1006_NAM": _fact_value(facts, "country"),
            "F1009": _fact_value(facts, "election_year"),
        }
        if contract.target_variable in study and study[contract.target_variable]:
            return study[contract.target_variable], "Available from study setup."
        if contract.generation_category == "fieldwork_design_fact":
            for key in ["mode", "fieldwork_dates"]:
                value = _fact_value(facts, key)
                if value:
                    return value, f"Available from reviewed study materials: {key}."
        if contract.generation_category == "election_metadata":
            for key in ["election_date", "election_type", "study_timing"]:
                value = _fact_value(facts, key)
                if value:
                    return value, f"Available from reviewed study materials: {key}."
        if contract.generation_category == "weight_derivation":
            value = _fact_value(facts, "weights")
            if value:
                return value, "Available from reviewed study materials: weights."
        return "", ""

    def _status_for(self, contract: AdministrativeVariableContract, source: str, value: Any) -> str:
        if contract.generation_category == "fixed_release_value" and contract.target_variable == "F1001":
            return "confirmed"
        if source:
            return "proposed"
        if value not in {"", None}:
            return "proposed"
        if contract.stata_rule.endswith("_or_not_applicable"):
            return "needs_processor_review"
        return "needs_processor_review"

    def _notes_for(self, contract: AdministrativeVariableContract, status: str) -> str:
        if status == "confirmed":
            return "Administrative value can be generated deterministically."
        if status == "proposed":
            return "Administrative value/source found; processor should verify before final syntax."
        return "Administrative information not yet sufficient; processor review is required before final readiness."


class AdministrativeDataGenerator:
    """Generate the administrative dataframe block from approved facts."""

    def __init__(self, wiki_root: Path = DEFAULT_WIKI_ROOT):
        self.polities = PolityReference(wiki_root)

    def generate(self, source_df: pd.DataFrame, facts_payload: dict[str, Any]) -> pd.DataFrame:
        facts = {item.get("key"): item for item in facts_payload.get("facts", []) if isinstance(item, dict)}
        study = facts_payload.get("study", {}) or {}
        n = len(source_df)
        out = pd.DataFrame(index=source_df.index)
        country_code = str(study.get("country_code") or _fact_value(facts, "country_code") or "")
        country_name = str(study.get("country") or _fact_value(facts, "country") or "")
        year = int(_fact_value(facts, "election_year") or study.get("year") or 0)
        polity = self.polities.get(country_code, country_name)
        cses_polity = str(polity.get("cses_polity_code") or "")

        out["F1001"] = "CSES-MODULE-6"
        out["F1002_VER"] = str(_fact_value(facts, "F1002_VER") or "VER2025-MMM-DD")
        out["F1002_DOI"] = str(_fact_value(facts, "F1002_DOI") or "doi:10.7804/cses.module6.2025-MM-DD")
        respondent_component = _zero_pad(source_df.get("A1"), 10, n)
        out["F1003_2"] = respondent_component
        out["F1004"] = f"{country_code}_{year}" if country_code and year else ""
        out["F1006"] = cses_polity
        out["F1003_1"] = out["F1006"].astype(str) + str(year) + out["F1003_2"].astype(str)
        out["F1005"] = _numeric_constant(_int_or_missing(f"{cses_polity}{year}"), n, source_df.index)
        out["F1006_UN"] = _numeric_constant(polity.get("un_numeric"), n, source_df.index)
        out["F1006_UNALPHA2"] = str(polity.get("un_alpha2") or "")
        out["F1006_UNALPHA3"] = str(polity.get("un_alpha3") or "")
        out["F1006_NAM"] = str(polity.get("country_name") or country_name)
        out["F1007_REG"] = _numeric_constant(polity.get("un_region"), n, source_df.index)
        out["F1007_OECD"] = _numeric_constant(polity.get("oecd_member"), n, source_df.index)
        out["F1007_EU"] = _numeric_constant(polity.get("eu_member"), n, source_df.index)
        out["F1007_VDEM"] = _numeric_constant(polity.get("vdem_id"), n, source_df.index)
        out["F1008"] = _numeric_constant(polity.get("administered_module_multiple_times", 0), n, source_df.index)
        out["F1009"] = _numeric_constant(year, n, source_df.index)

        election_date = _parse_date(_fact_value(facts, "election_date") or _fact_value(facts, "first_round_election_date"))
        self._fill_election_date(out, "F1010", election_date, year)
        second_date = _parse_date(_fact_value(facts, "second_round_election_date"))
        no_second_round = bool(_fact_value(facts, "no_second_round") or not second_date)
        if second_date and not no_second_round:
            self._fill_election_date(out, "F1011", second_date, year)
        else:
            out["F1011_M"] = 96
            out["F1011_D"] = 96
            out["F1011_Y"] = 9996
            out["F1011_1"] = "9996. NOT APPLICABLE: NO SECOND ROUND"
            out["F1011_2"] = 999996

        out["F1012_1"] = _numeric_constant(_fact_value(facts, "F1012_1") or _fact_value(facts, "study_timing_code") or 9, n, source_df.index)
        out["F1012_2"] = _numeric_constant(_fact_value(facts, "F1012_2") or _fact_value(facts, "covid_timing_code") or 9, n, source_df.index)
        out["F1013"] = _numeric_constant(_fact_value(facts, "F1013") or _fact_value(facts, "study_context_code") or 9, n, source_df.index)
        out["F1014"] = _numeric_constant(_fact_value(facts, "F1014") or _fact_value(facts, "election_type_code") or 99, n, source_df.index)

        mode_map = {1: 3, 2: 4, 3: 1, 4: 2, 5: 5}
        if "mode" in source_df.columns:
            respondent_mode = source_df["mode"].map(mode_map).fillna(9).astype(int)
            modes = [int(item) for item in sorted(respondent_mode.dropna().unique()) if int(item) != 9]
        else:
            respondent_mode = pd.Series([9] * n, index=source_df.index)
            modes = []
        modes = modes[:3] + [0] * max(0, 3 - len(modes))
        out["F1015_1"], out["F1015_2"], out["F1015_3"] = modes[:3]
        out["F1016_1"] = respondent_mode
        out["F1016_2"] = 0
        out["F1016_3"] = 0
        out["F1017"] = 1 if len([m for m in modes if m]) > 1 else 0

        interview_dates = _interview_dates(source_df)
        min_interview = interview_dates.dropna().min() if not interview_dates.dropna().empty else None
        max_interview = interview_dates.dropna().max() if not interview_dates.dropna().empty else None
        out["F1018_1"] = _date_span_days(min_interview, max_interview, inclusive=True) if min_interview is not None and max_interview is not None else 9999
        out["F1018_2"] = _date_diff_days(min_interview, election_date) if min_interview is not None and election_date is not None else 999
        out["F1019_M"] = source_df["A4a"] if "A4a" in source_df.columns else 99
        out["F1019_D"] = source_df["A4b"] if "A4b" in source_df.columns else 99
        out["F1019_Y"] = source_df["A4c"] if "A4c" in source_df.columns else 9999
        if election_date is not None:
            out["F1020_1"] = (interview_dates - pd.Timestamp(election_date)).dt.days
            out.loc[(out["F1019_M"] == 99) | (out["F1019_D"] == 99) | (out["F1019_Y"] == 9999), "F1020_1"] = 9999
        else:
            out["F1020_1"] = 9999
        out["F1020_2"] = 9996 if no_second_round else 9999
        out["F1021"] = int(_fact_value(facts, "F1021") or 99999)
        out["F1022_1"] = int(_fact_value(facts, "F1022_1") or 999997)
        out["F1022_2"] = int(_fact_value(facts, "F1022_2") or 7)
        out["F1023"] = int(_fact_value(facts, "F1023") or _fact_value(facts, "questionnaire_language_code") or 999)
        out["F1024"] = int(_fact_value(facts, "F1024") or 2)
        out["F1100"] = int(_fact_value(facts, "F1100") or 1)

        out["F1101_1"] = 1.0
        if "A5" in source_df.columns:
            weight = source_df["A5"].copy()
            weight = weight.fillna(weight.mean())
            out["F1101_2"] = weight
        else:
            out["F1101_2"] = 1.0
        out["F1101_3"] = 1.0
        for name in ["F1102_1", "F1102_2", "F1102_3", "F1103_1", "F1103_2", "F1103_3", "F1104", "F1105_1", "F1105_2", "F1105_3"]:
            out[name] = 1.0
        out["F1106"] = int(_fact_value(facts, "F1106") or _fact_value(facts, "release_cycle") or 2)
        return out

    def _fill_election_date(self, out: pd.DataFrame, prefix: str, date_value: Any, year: int) -> None:
        if date_value is None:
            out[f"{prefix}_M"] = 99
            out[f"{prefix}_D"] = 99
            out[f"{prefix}_Y"] = 9999
            out[f"{prefix}_1"] = "9999. MISSING"
            out[f"{prefix}_2"] = 999999
            return
        out[f"{prefix}_M"] = date_value.month
        out[f"{prefix}_D"] = date_value.day
        out[f"{prefix}_Y"] = date_value.year
        out[f"{prefix}_1"] = f"{date_value.year:04d}-{date_value.month:02d}-{date_value.day:02d}"
        out[f"{prefix}_2"] = int(f"{date_value.year:04d}{date_value.month:02d}")


def administrative_plan_summary(plans: list[AdministrativeVariablePlan]) -> dict[str, int]:
    summary = {"total": len(plans), "confirmed": 0, "proposed": 0, "needs_processor_review": 0}
    for plan in plans:
        if plan.status in summary:
            summary[plan.status] += 1
    return summary


def audit_administrative_registry(wiki_root: Path = DEFAULT_WIKI_ROOT) -> list[str]:
    issues = AdministrativeVariableRegistry(wiki_root).validate()
    issues.extend(PolityReference(wiki_root).validate())
    return issues


def plans_by_target(plans: list[AdministrativeVariablePlan]) -> dict[str, AdministrativeVariablePlan]:
    return {plan.target_variable: plan for plan in plans}


def _is_administrative(name: str) -> bool:
    return name.startswith("F10") or name.startswith("F11")


def _present_status(value: Any) -> str:
    return "confirmed" if value and str(value).strip() not in {"UNK", "0000", "Unknown"} else "needs_processor_review"


def _fact_value(facts: dict[str, dict[str, Any]], key: str) -> Any:
    fact = facts.get(key) or {}
    return fact.get("value")


def _study_id(facts: dict[str, dict[str, Any]]) -> str:
    country_code = _fact_value(facts, "country_code")
    year = _fact_value(facts, "election_year")
    if country_code and year:
        return f"{country_code}_{year}"
    return ""


def _numeric_constant(value: Any, n: int, index: Any) -> pd.Series:
    return pd.Series([_int_or_missing(value)] * n, index=index)


def _int_or_missing(value: Any, missing: int = 9999) -> int:
    try:
        if value in {"", None}:
            return missing
        return int(float(str(value)))
    except Exception:
        return missing


def _zero_pad(series: Any, width: int, n: int) -> pd.Series:
    if series is None:
        return pd.Series(["9" * width] * n)
    return series.apply(lambda value: str(int(value)).zfill(width) if pd.notna(value) else "9" * width)


def _parse_date(value: Any) -> pd.Timestamp | None:
    if not value:
        return None
    match = re.search(r"\d{4}-\d{1,2}-\d{1,2}", str(value))
    if not match:
        return None
    try:
        return pd.Timestamp(match.group(0))
    except Exception:
        return None


def _interview_dates(source_df: pd.DataFrame) -> pd.Series:
    if not {"A4a", "A4b", "A4c"}.issubset(set(source_df.columns)):
        return pd.Series([pd.NaT] * len(source_df), index=source_df.index)
    dates = pd.to_datetime(
        {
            "year": source_df["A4c"].where(source_df["A4c"] != 9999),
            "month": source_df["A4a"].where(source_df["A4a"] != 99),
            "day": source_df["A4b"].where(source_df["A4b"] != 99),
        },
        errors="coerce",
    )
    return dates


def _date_span_days(start: pd.Timestamp, end: pd.Timestamp, inclusive: bool = False) -> int:
    days = int((end - start).days)
    return days + 1 if inclusive else days


def _date_diff_days(date_value: pd.Timestamp, reference: pd.Timestamp) -> int:
    return int((date_value - reference).days)
