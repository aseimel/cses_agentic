"""Standardized district data parsing, validation, and merge planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.codegen.sheet_reader import TrackingSheetReader
from src.matching.party_order import PARTY_LETTERS, PartyOrderingRulesEngine
from src.workflow.state import WorkflowState


DISTRICT_KEY = "F2019"
DISTRICT_REQUIRED_BASE = ("F4001", "F4002", "F4003", "F4006", "F4007")
DISTRICT_PARTY_PREFIXES = ("F4004", "F4005")
DISTRICT_OPTIONAL_TEXT = ("OriginalDistrictName", "EnglishDistrictName", "Notes", "References")
DISTRICT_FILE_TERMS = ("district", "constituency", "riding", "wahlkreis", "circonscription")
DISTRICT_FILE_EXTENSIONS = (".xlsx", ".xls", ".csv", ".dta")


@dataclass
class DistrictDataTable:
    source_file: str
    source_sheet: str
    dataframe: pd.DataFrame
    columns: list[str]
    district_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "source_sheet": self.source_sheet,
            "columns": self.columns,
            "district_count": self.district_count,
        }


@dataclass
class DistrictValidationResult:
    status: str
    approved: bool = False
    source_district_variable: str = ""
    party_order_approved: bool = False
    party_letters: list[str] = field(default_factory=list)
    required_columns_missing: list[str] = field(default_factory=list)
    required_party_columns_missing: list[str] = field(default_factory=list)
    generated_missing_party_slots: list[str] = field(default_factory=list)
    duplicate_district_codes: list[str] = field(default_factory=list)
    missing_observed_district_codes: list[str] = field(default_factory=list)
    extra_district_codes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DistrictMergePlan:
    status: str
    approved: bool
    source_district_variable: str
    source_data_file: str
    district_source_file: str
    district_source_sheet: str
    normalized_dta_path: str
    district_count: int
    variables_merged: list[str] = field(default_factory=list)
    generated_missing_slots: list[str] = field(default_factory=list)
    missing_observed_district_codes: list[str] = field(default_factory=list)
    extra_district_codes: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DistrictDataTemplateParser:
    """Read standardized district files without country-specific assumptions."""

    def discover_files(self, working_dir: Path) -> list[Path]:
        roots = [
            Path(working_dir),
            Path(working_dir) / "micro" / "district data",
            Path(working_dir) / "micro",
            Path(working_dir) / "E-mails",
            Path(working_dir) / "emails",
        ]
        found: dict[str, Path] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*"):
                if not path.is_file() or path.name.startswith("~$"):
                    continue
                if path.suffix.lower() not in DISTRICT_FILE_EXTENSIONS:
                    continue
                lower = str(path).casefold()
                if any(term in lower for term in DISTRICT_FILE_TERMS) or path.parent.name.casefold() == "district data":
                    found[str(path.resolve()).casefold()] = path
        return sorted(found.values(), key=lambda item: str(item).casefold())

    def parse_best(self, working_dir: Path) -> DistrictDataTable | None:
        for path in self.discover_files(working_dir):
            try:
                table = self.parse(path)
            except Exception:
                continue
            if table:
                return table
        return None

    def parse(self, path: Path) -> DistrictDataTable | None:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            return self._parse_excel(path)
        if suffix == ".csv":
            df = pd.read_csv(path)
            return self._table_from_dataframe(path, "", df)
        if suffix == ".dta":
            df = pd.read_stata(path)
            return self._table_from_dataframe(path, "", df)
        return None

    def _parse_excel(self, path: Path) -> DistrictDataTable | None:
        sheets = pd.read_excel(path, sheet_name=None, header=None)
        for sheet_name, raw in sheets.items():
            header_row = self._find_header_row(raw)
            if header_row is None:
                continue
            headers = [raw.iat[header_row, col] for col in range(raw.shape[1])]
            df = raw.iloc[header_row + 1 :].copy()
            df.columns = headers
            table = self._table_from_dataframe(path, sheet_name, df)
            if table:
                return table
        return None

    def _find_header_row(self, raw: pd.DataFrame) -> int | None:
        for row_idx in range(min(len(raw), 50)):
            normalized = {_canonical_column(raw.iat[row_idx, col]) for col in range(raw.shape[1])}
            if DISTRICT_KEY in normalized and any(col.startswith("F400") for col in normalized):
                return row_idx
        return None

    def _table_from_dataframe(self, path: Path, sheet_name: str, df: pd.DataFrame) -> DistrictDataTable | None:
        renamed = {}
        for column in df.columns:
            canonical = _canonical_column(column)
            if canonical:
                renamed[column] = canonical
        if not renamed:
            return None
        df = df.rename(columns=renamed)
        keep_columns = [
            col for col in df.columns
            if col == DISTRICT_KEY or col.startswith("F400") or col in DISTRICT_OPTIONAL_TEXT
        ]
        if DISTRICT_KEY not in keep_columns:
            return None
        district_columns = [col for col in keep_columns if col == DISTRICT_KEY or col.startswith("F400")]
        if len(district_columns) <= 1:
            return None
        df = df[keep_columns].copy()
        df = df[df[DISTRICT_KEY].map(_clean_key) != ""]
        if df.empty:
            return None
        for col in district_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.reset_index(drop=True)
        return DistrictDataTable(
            source_file=str(path),
            source_sheet=sheet_name,
            dataframe=df,
            columns=list(df.columns),
            district_count=len(df),
        )


class DistrictDataValidator:
    """Validate district input against micro data and approved party order."""

    def validate(
        self,
        table: DistrictDataTable,
        state: WorkflowState,
        approve: bool = False,
        source_variable: str = "",
    ) -> DistrictValidationResult:
        source_variable = source_variable or detect_source_district_variable(state)
        party_letters = approved_party_letters(Path(state.working_dir)) if state.working_dir else []
        columns = set(table.columns)
        issues: list[str] = []
        warnings: list[str] = []

        missing_base = [col for col in DISTRICT_REQUIRED_BASE if col not in columns]
        if missing_base:
            issues.append("The district file is missing required district variables.")

        if not source_variable:
            issues.append("The respondent district variable in the survey data needs processor review.")

        duplicate_codes = _duplicate_keys(table.dataframe[DISTRICT_KEY].tolist())
        if duplicate_codes:
            issues.append("The district file contains duplicate district codes.")

        observed_codes = observed_district_codes(state.data_file, source_variable) if source_variable else set()
        district_codes = {_clean_key(value) for value in table.dataframe[DISTRICT_KEY].tolist() if _clean_key(value)}
        missing_observed = sorted(observed_codes - district_codes, key=_sort_key)
        extra_codes = sorted(district_codes - observed_codes, key=_sort_key) if observed_codes else []
        if missing_observed:
            issues.append("Some respondent district codes are absent from the district file.")
        if extra_codes:
            warnings.append("The district file contains district rows not observed in the survey data.")

        required_party_columns: list[str] = []
        generated_slots: list[str] = []
        if party_letters:
            for prefix in DISTRICT_PARTY_PREFIXES:
                for letter in party_letters:
                    required_party_columns.append(f"{prefix}_{letter}")
                for letter in PARTY_LETTERS:
                    if letter not in party_letters:
                        generated_slots.append(f"{prefix}_{letter}")
        else:
            if any(col.startswith(DISTRICT_PARTY_PREFIXES) for col in columns):
                issues.append("Party Order Agreement must be approved before validating party-specific district variables.")
        missing_party = [col for col in required_party_columns if col not in columns]
        if missing_party:
            issues.append("The district file is missing party-specific district columns required by the approved party order.")

        numeric_issues = [
            col for col in columns
            if (col == DISTRICT_KEY or col.startswith("F400")) and table.dataframe[col].isna().all()
        ]
        if numeric_issues:
            warnings.append("Some district variables contain no numeric values after parsing.")

        status = "ready" if not issues else "needs_processor_review"
        approved = bool(approve and status == "ready")
        if status == "ready" and not approved:
            issues.append("Processor approval is needed before district variables are used in final syntax.")
            status = "needs_processor_approval"

        return DistrictValidationResult(
            status=status,
            approved=approved,
            source_district_variable=source_variable,
            party_order_approved=bool(party_letters),
            party_letters=party_letters,
            required_columns_missing=missing_base,
            required_party_columns_missing=missing_party,
            generated_missing_party_slots=generated_slots,
            duplicate_district_codes=duplicate_codes,
            missing_observed_district_codes=missing_observed,
            extra_district_codes=extra_codes[:100],
            issues=issues,
            warnings=warnings,
        )


class DistrictMergePlanner:
    """Create and persist district merge plans."""

    def build(
        self,
        working_dir: Path,
        table: DistrictDataTable,
        validation: DistrictValidationResult,
    ) -> DistrictMergePlan:
        output_dir = Path(working_dir) / "micro" / "district data"
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_path = output_dir / "district_data_normalized.dta"
        self.write_normalized_dta(table, normalized_path)
        variables = [
            col for col in table.columns
            if col.startswith("F400")
        ] + validation.generated_missing_party_slots
        plan = DistrictMergePlan(
            status=validation.status,
            approved=validation.approved,
            source_district_variable=validation.source_district_variable,
            source_data_file="",
            district_source_file=table.source_file,
            district_source_sheet=table.source_sheet,
            normalized_dta_path=str(normalized_path),
            district_count=table.district_count,
            variables_merged=sorted(set(variables), key=_district_variable_sort_key),
            generated_missing_slots=validation.generated_missing_party_slots,
            missing_observed_district_codes=validation.missing_observed_district_codes,
            extra_district_codes=validation.extra_district_codes,
            issues=validation.issues,
            warnings=validation.warnings,
        )
        return plan

    def write(self, working_dir: Path, plan: DistrictMergePlan) -> Path:
        cses_dir = Path(working_dir) / ".cses"
        cses_dir.mkdir(parents=True, exist_ok=True)
        path = cses_dir / "district_merge_plan.json"
        path.write_text(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def write_normalized_dta(self, table: DistrictDataTable, output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = table.dataframe.copy()
        for col in df.columns:
            if col == DISTRICT_KEY or col.startswith("F400"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.to_stata(output_path, write_index=False, version=118)
        return output_path


class DistrictStataSyntaxBuilder:
    """Generate the standard district merge block for plan-driven syntax."""

    def merge_lines(self, plan: dict[str, Any]) -> list[str]:
        if not plan or not plan.get("approved"):
            return []
        source = plan.get("source_district_variable") or DISTRICT_KEY
        using_path = _stata_path(plan.get("normalized_dta_path", ""))
        variables = plan.get("variables_merged", []) or []
        generated_slots = set(plan.get("generated_missing_slots", []) or [])
        lines = [
            "***************************************************************************",
            "**>>> DISTRICT DATA MERGE",
            "***************************************************************************",
            f"* Source district variable: {source}",
        ]
        if source != DISTRICT_KEY:
            lines.extend([
                f"capture drop {DISTRICT_KEY}",
                f"gen {DISTRICT_KEY} = {source}",
                f"recode {DISTRICT_KEY} (. = 99999)",
            ])
        else:
            lines.append(f"recode {DISTRICT_KEY} (. = 99999)")
        lines.extend([
            f"format {DISTRICT_KEY} %005.0f",
            f'merge m:1 {DISTRICT_KEY} using "{using_path}"',
            "tab _merge, mis",
            f"count if _merge == 1 & {DISTRICT_KEY} != 99999",
            "drop _merge",
        ])
        for var in sorted(generated_slots, key=_district_variable_sort_key):
            lines.extend([
                f"capture gen {var} = {_district_missing_value(var)}",
            ])
        for var in sorted(set(variables), key=_district_variable_sort_key):
            lines.append(f"capture replace {var} = {_district_missing_value(var)} if {var} == .")
        lines.append("")
        return lines


def load_district_merge_plan(working_dir: Path | str | None) -> dict[str, Any]:
    if not working_dir:
        return {}
    path = Path(working_dir) / ".cses" / "district_merge_plan.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_source_district_variable(state: WorkflowState) -> str:
    if state.variable_tracking_file:
        try:
            tracking = TrackingSheetReader().read(Path(state.variable_tracking_file))
            mapping = tracking.get_mapping(DISTRICT_KEY)
            if mapping and mapping.source_var and mapping.source_var not in {"NOT_FOUND", "EXTERNAL_INPUT_REQUIRED"}:
                return mapping.source_var
        except Exception:
            pass
    columns = _data_columns(state.data_file)
    preferred = ("F2019", "D18", "DISTRICTID", "DISTRICT_ID", "DISTRICT", "CONSTITUENCY")
    upper_lookup = {col.upper(): col for col in columns}
    for candidate in preferred:
        if candidate in upper_lookup:
            return upper_lookup[candidate]
    for col in columns:
        lowered = col.casefold()
        if "district" in lowered or "constituency" in lowered:
            return col
    return ""


def observed_district_codes(data_file: str | None, source_variable: str) -> set[str]:
    if not data_file or not source_variable:
        return set()
    try:
        df = _read_tabular(Path(data_file), usecols=[source_variable])
    except Exception:
        return set()
    if source_variable not in df.columns:
        return set()
    missing_codes = {"", "99999", "99998", "99997", "."}
    return {
        key for key in (_clean_key(value) for value in df[source_variable].tolist())
        if key and key not in missing_codes
    }


def approved_party_letters(working_dir: Path) -> list[str]:
    if not PartyOrderingRulesEngine().is_approved(working_dir):
        return []
    decision = PartyOrderingRulesEngine().load_decision(working_dir)
    parties = ((decision.get("proposal") or {}).get("proposed_parties") or [])
    letters = [str(party.get("code_letter", "")).strip().upper() for party in parties]
    return [letter for letter in letters if letter in PARTY_LETTERS]


def _read_tabular(path: Path, usecols: list[str] | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".dta":
        return pd.read_stata(path, columns=usecols)
    if suffix == ".csv":
        return pd.read_csv(path, usecols=usecols)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, usecols=usecols)
    if suffix == ".sav":
        import pyreadstat

        df, _ = pyreadstat.read_sav(str(path), usecols=usecols)
        return df
    return pd.read_csv(path, usecols=usecols)


def _data_columns(data_file: str | None) -> list[str]:
    if not data_file:
        return []
    try:
        return list(_read_tabular(Path(data_file), usecols=None).columns)
    except Exception:
        return []


def _canonical_column(value: Any) -> str:
    text = str(value or "").strip()
    compact = "".join(ch for ch in text if ch.isalnum() or ch == "_").upper()
    if compact in {"DISTRICTID", "DISTRICT_ID", "DISTRICTCODE", "DISTRICT_CODE", "F2019"}:
        return DISTRICT_KEY
    if compact in {"ORIGINALDISTRICTNAME", "ORIGINALNAME", "DISTRICTNAMEORIGINAL"}:
        return "OriginalDistrictName"
    if compact in {"ENGLISHDISTRICTNAME", "ENGLISHNAME", "DISTRICTNAMEENGLISH"}:
        return "EnglishDistrictName"
    if compact in {"NOTE", "NOTES"}:
        return "Notes"
    if compact in {"REFERENCE", "REFERENCES", "SOURCE", "SOURCES"}:
        return "References"
    if compact.startswith("F400"):
        if "_" in text:
            return text.strip().upper()
        if len(compact) == 6:
            return compact
        if len(compact) == 7 and compact[-1].isalpha():
            return f"{compact[:5]}_{compact[-1]}"
    return ""


def _clean_key(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
    except Exception:
        pass
    return text


def _duplicate_keys(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        key = _clean_key(value)
        if not key:
            continue
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates, key=_sort_key)


def _sort_key(value: str) -> tuple[int, Any]:
    try:
        return (0, int(value))
    except Exception:
        return (1, value)


def _district_variable_sort_key(name: str) -> tuple[int, str]:
    if name == "F4001":
        return (1, name)
    if name == "F4002":
        return (2, name)
    if name == "F4003":
        return (3, name)
    if name.startswith("F4004_"):
        return (4, name)
    if name.startswith("F4005_"):
        return (5, name)
    if name == "F4006":
        return (6, name)
    if name == "F4007":
        return (7, name)
    return (99, name)


def _district_missing_value(name: str) -> str:
    if name == "F4002":
        return "9999"
    if name == "F4007":
        return "99999999"
    if name.startswith("F4005") or name == "F4003" or name == "F4006":
        return "99"
    return "999"


def _stata_path(path: str) -> str:
    return str(path).replace("\\", "/")
