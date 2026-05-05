"""CSES schema registry backed by cses_wiki.

The runtime workflow uses this module as the single source of truth for the
target variable universe. Development references may be used to distill the
JSON file, but normal processing must only read cses_wiki.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIKI_ROOT = PROJECT_ROOT / "cses_wiki"
SCHEMA_PATH = DEFAULT_WIKI_ROOT / "patterns" / "module6_schema.json"


FALLBACK_VARIABLES = [
    "F1001", "F1002_VER", "F1002_DOI", "F1003_1", "F1003_2", "F1004", "F1005",
    "F1006", "F1006_UN", "F1006_UNALPHA2", "F1006_UNALPHA3", "F1006_NAM",
    "F1007_REG", "F1007_OECD", "F1007_EU", "F1007_VDEM", "F1008", "F1009",
    "F1010_M", "F1010_D", "F1010_Y", "F1010_1", "F1010_2", "F1011_M",
    "F1011_D", "F1011_Y", "F1011_1", "F1011_2", "F1012_1", "F1012_2",
    "F1013", "F1014", "F1015_1", "F1015_2", "F1015_3", "F1016_1",
    "F1016_2", "F1016_3", "F1017", "F1018_1", "F1018_2", "F1019_M",
    "F1019_D", "F1019_Y", "F1020_1", "F1020_2", "F1021", "F1022",
    "F1100", "F1101_1", "F1101_2", "F1101_3", "F1102", "F1103_1",
    "F1103_2", "F1103_3", "F1104", "F1105_1", "F1105_2", "F1105_3",
    "F1106", "F2001_Y", "F2001_M", "F2001_A", "F2001_GG", "F2001_GS",
    "F2001_GBB", "F2001_GX", "F2001_GY", "F2001_GZ", "F2002", "F2003",
    "F2004", "F2005", "F2006", "F2007", "F2008", "F2009", "F2010_1",
    "F2010_2", "F2011", "F2012", "F2013", "F2014", "F2015", "F2016",
    "F2017", "F2018", "F2020", "F2021", "F3001", "F3002_1", "F3002_2",
    "F3002_3", "F3002_4", "F3002_5", "F3002_6_1", "F3003", "F3004_1",
    "F3004_2", "F3004_3", "F3004_4", "F3005_1", "F3005_2", "F3005_3",
    "F3006", "F3007_1", "F3007_2", "F3007_3", "F3007_4", "F3007_5",
    "F3007_6", "F3007_7", "F3008_1", "F3008_2", "F3009", "F3010",
    "F3011_PR_1", "F3011_PR_2", "F3011_LH_PL", "F3011_LH_DC", "F3011_LH_PL2",
    "F3011_UH_PL", "F3012_1", "F3012_2", "F3013", "F3014", "F3017",
    "F3018_A", "F3018_B", "F3018_C", "F3018_D", "F3018_E", "F3018_F",
    "F3018_G", "F3018_H", "F3018_I", "F3019_A", "F3019_B", "F3019_C",
    "F3019_D", "F3019_E", "F3019_F", "F3019_G", "F3019_H", "F3019_I",
    "F3020", "F3021", "F3022", "F3023", "F3024", "F4001", "F4002",
    "F4003", "F4004_A", "F4004_B", "F4004_C", "F4004_D", "F4004_E",
    "F4004_F", "F4004_G", "F4004_H", "F4004_I", "F4005_A", "F4005_B",
    "F4005_C", "F4005_D", "F4005_E", "F4005_F", "F4005_G", "F4005_H",
    "F4005_I", "F4006", "F4006_N", "F4007", "F4007_N", "F5000_A",
    "F5000_B", "F5000_C", "F5000_D", "F5000_E", "F5000_F", "F5000_G",
    "F5000_H", "F5000_I", "F5000_L_A", "F5000_L_B", "F5000_L_C",
    "F5000_L_D", "F5000_L_E", "F5000_L_F", "F5000_L_G", "F5000_L_H",
    "F5000_L_I", "F5200_A", "F5200_B", "F5200_C", "F5200_D", "F5200_E",
    "F5200_F", "F5200_G", "F5200_H", "F5200_I", "F5201_A", "F5201_B",
    "F5201_C", "F5201_D", "F5201_E", "F5201_F", "F5201_G", "F5201_H",
    "F5201_I", "F5202_A", "F5202_B", "F5202_C", "F5202_D", "F5202_E",
    "F5202_F", "F5202_G", "F5202_H", "F5202_I", "F5203_A", "F5203_B",
    "F5203_C", "F5203_D", "F5203_E", "F5203_F", "F5203_G", "F5203_H",
    "F5203_I", "F6000_PR_1", "F6000_PR_2", "F6000_LH_PL", "F6000_LH_DC",
]


@dataclass(frozen=True)
class SchemaVariable:
    name: str
    description: str
    order: int
    section: str
    dependency_class: str
    required_evidence: list[str] = field(default_factory=list)
    recode_strategy: str = "direct_or_recode"
    syntax_pattern_id: str = "standard_variable_block"
    validation_checks: list[str] = field(default_factory=list)
    documentation_requirements: list[str] = field(default_factory=list)
    missing_value_policy: str = "cses_standard"
    value_label_required: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class SchemaRegistry:
    """Load and query the CSES Module 6 target schema."""

    def __init__(self, wiki_root: Path = DEFAULT_WIKI_ROOT):
        self.wiki_root = Path(wiki_root)
        self.schema_path = self.wiki_root / "patterns" / "module6_schema.json"
        self._schema = self._load_or_build_fallback()
        self.variables = [
            SchemaVariable(**item) for item in self._schema.get("variables", [])
        ]

    def as_target_dict(self) -> dict[str, str]:
        return {item.name: item.description for item in self.variables}

    def ordered_names(self) -> list[str]:
        return [item.name for item in self.variables]

    def by_name(self, name: str) -> SchemaVariable | None:
        for item in self.variables:
            if item.name == name:
                return item
        return None

    def required_count(self) -> int:
        return len(self.variables)

    def validate(self) -> list[str]:
        issues: list[str] = []
        seen: set[str] = set()
        for item in self.variables:
            if not re.match(r"^F\d{4}(?:_[A-Z0-9]+)*$", item.name):
                issues.append(f"Invalid CSES variable name: {item.name}")
            if item.name in seen:
                issues.append(f"Duplicate CSES variable in schema: {item.name}")
            seen.add(item.name)
            if not item.description:
                issues.append(f"Missing description for {item.name}")
            if not item.syntax_pattern_id:
                issues.append(f"Missing syntax pattern for {item.name}")
            if not item.validation_checks:
                issues.append(f"Missing validation checks for {item.name}")
            if not item.documentation_requirements:
                issues.append(f"Missing documentation requirements for {item.name}")
        return issues

    def _load_or_build_fallback(self) -> dict:
        if self.schema_path.exists():
            return json.loads(self.schema_path.read_text(encoding="utf-8"))
        return build_schema_payload(_fallback_schema_variables())


def build_schema_payload(variables: Iterable[SchemaVariable]) -> dict:
    vars_list = [item.to_dict() for item in variables]
    return {
        "schema_version": 1,
        "module": "CSES Module 6",
        "source_policy": (
            "Runtime schema distilled into cses_wiki. Raw example studies are "
            "development inputs only and are not required by the installed app."
        ),
        "variables": vars_list,
    }


def infer_schema_variable(name: str, description: str = "", order: int = 0) -> SchemaVariable:
    section = infer_section(name)
    dependency_class = infer_dependency_class(name)
    return SchemaVariable(
        name=name,
        description=description or infer_description(name),
        order=order,
        section=section,
        dependency_class=dependency_class,
        required_evidence=infer_required_evidence(name),
        recode_strategy=infer_recode_strategy(name),
        syntax_pattern_id=infer_syntax_pattern(name),
        validation_checks=infer_validation_checks(name),
        documentation_requirements=infer_documentation_requirements(name),
        missing_value_policy=infer_missing_policy(name),
        value_label_required=not name.endswith(("_VER", "_DOI", "_UNALPHA2", "_UNALPHA3", "_NAM")),
    )


def infer_section(name: str) -> str:
    if name.startswith("F1"):
        return "administration"
    if name.startswith("F2"):
        return "demographics"
    if name.startswith("F3"):
        return "survey_questions"
    if name.startswith("F4"):
        return "district_data"
    if name.startswith("F5"):
        return "party_leader_macro"
    if name.startswith("F6"):
        return "integrated_vote_choice"
    return "other"


def infer_dependency_class(name: str) -> str:
    if name.startswith("F4"):
        return "district_input"
    if name.startswith("F5") or name.startswith("F6"):
        return "macro_or_party_input"
    if name in {
        "F3010",
        "F3010_TS",
        "F3011_VS_1",
        "F3011_LR_CSES",
        "F3011_LR_MARPOR",
        "F3011_IF_CSES",
        "F3100_LR_CSES",
        "F3100_LR_MARPOR",
        "F3100_IF_CSES",
    }:
        return "macro_or_party_input"
    if name.startswith("F10") or name.startswith("F11"):
        return "derived_metadata"
    if name.startswith("F2") or name.startswith("F3"):
        return "direct_survey_item"
    return "processor_decision"


def infer_required_evidence(name: str) -> list[str]:
    dep = infer_dependency_class(name)
    return {
        "direct_survey_item": ["questionnaire", "data_variable", "source_frequency"],
        "derived_metadata": ["design_report_or_deposit_metadata", "processor_decision"],
        "district_input": ["district_data_file", "district_definition", "merge_key"],
        "macro_or_party_input": ["election_results_or_macro_file", "party_ordering"],
        "processor_decision": ["processor_decision"],
    }.get(dep, ["source_evidence"])


def infer_recode_strategy(name: str) -> str:
    if name.startswith("F4") or name.startswith("F5") or name.startswith("F6"):
        return "external_input_or_derived"
    if name.startswith("F10") or name.startswith("F11"):
        return "metadata_or_constant"
    return "direct_or_recode"


def infer_syntax_pattern(name: str) -> str:
    if name.startswith("F4"):
        return "district_variable_block"
    if name.startswith("F5") or name.startswith("F6"):
        return "party_macro_variable_block"
    if name.startswith("F10") or name.startswith("F11"):
        return "administrative_variable_block"
    return "standard_variable_block"


def infer_validation_checks(name: str) -> list[str]:
    checks = ["variable_exists", "tab_missing", "missing_codes"]
    if name.startswith("F3") or name.startswith("F2"):
        checks.append("source_target_verification")
    if name.startswith("F5") or name.startswith("F6"):
        checks.append("party_code_check")
    if name.startswith("F4"):
        checks.append("district_consistency_check")
    return checks


def infer_documentation_requirements(name: str) -> list[str]:
    requirements = ["tracking_sheet_row", "processing_log_entry"]
    if name.startswith("F4"):
        requirements.append("district_data_documentation")
    if name.startswith("F5") or name.startswith("F6"):
        requirements.append("party_leader_appendix")
    if name.startswith("F10") or name.startswith("F11"):
        requirements.append("study_design_and_weights")
    return requirements


def infer_missing_policy(name: str) -> str:
    if name.endswith(("_Y", "_A")) or name in {"F2007"}:
        return "extended_cses_missing_codes"
    if name.startswith("F4") or name.startswith("F5") or name.startswith("F6"):
        return "macro_district_missing_codes"
    return "standard_cses_missing_codes"


def infer_description(name: str) -> str:
    section = infer_section(name).replace("_", " ")
    return f"{name} {section} variable"


def _fallback_schema_variables() -> list[SchemaVariable]:
    return [
        infer_schema_variable(name, order=index + 1)
        for index, name in enumerate(FALLBACK_VARIABLES)
    ]


def load_target_variables(wiki_root: Path = DEFAULT_WIKI_ROOT) -> dict[str, str]:
    """Compatibility helper used by matching and tracking code."""
    return SchemaRegistry(wiki_root).as_target_dict()
