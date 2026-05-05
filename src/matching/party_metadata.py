"""Party metadata review and party-context derivation support.

This module handles CSES party metadata as macro-process input. It reads
standardized macro materials where possible, prepares a human-reviewable party
metadata table, and provides approved values for deterministic micro
derivatives.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.matching.party_order import PARTY_LETTERS


PARTY_METADATA_PREFIXES = ("F5028", "F5029", "F5200", "F5201", "F5202", "F5203")


@dataclass
class PartyMetadataValue:
    variable: str
    party_letter: str
    value: str
    source_type: str
    source_file: str
    source_sheet: str = ""
    row_number: int = 0
    evidence: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PartyMetadataReview:
    status: str
    values: list[PartyMetadataValue] = field(default_factory=list)
    missing_variables: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    processor_decision_needed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "values": [item.to_dict() for item in self.values],
            "missing_variables": list(self.missing_variables),
            "warnings": list(self.warnings),
            "processor_decision_needed": self.processor_decision_needed,
        }


class PartyMetadataReviewBuilder:
    """Prepare a party metadata review from macro/election materials."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)

    def build(self) -> PartyMetadataReview:
        values = self._workbook_values()
        if not values:
            values = self._macro_report_values()
        decision = self._load_party_order_decision()
        parties = _approved_parties(decision)
        warnings = self._validation_warnings(values, parties)
        expected = self._expected_variables(parties)
        found = {item.variable for item in values}
        missing = [variable for variable in expected if variable not in found]
        status = "ready_for_review" if values else "missing_party_metadata"
        if missing:
            status = "needs_processor_review"
        return PartyMetadataReview(
            status=status,
            values=values,
            missing_variables=missing,
            warnings=warnings,
            processor_decision_needed=(
                "Micro processor and macro coder should approve the party metadata "
                "before derivative party-context variables are finalized."
            ),
        )

    def write_review(self, review: PartyMetadataReview) -> tuple[Path, Path]:
        cses_dir = self.working_dir / ".cses"
        cses_dir.mkdir(parents=True, exist_ok=True)
        review_path = cses_dir / "party_metadata_review.json"
        decision_path = cses_dir / "party_metadata_decision.json"
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_policy": "Prefer deposited macro materials; use public sources only as fallback or validation.",
            "review": review.to_dict(),
            "approval": {
                "micro_processor_approved": False,
                "macro_coder_approved": False,
                "locked": False,
                "override_reason": "",
            },
        }
        review_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        if not decision_path.exists():
            decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        else:
            try:
                existing = json.loads(decision_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}
            approval = existing.get("approval", {}) if isinstance(existing, dict) else {}
            if not approval.get("locked"):
                decision_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return review_path, decision_path

    def is_approved(self) -> bool:
        payload = load_party_metadata_decision(self.working_dir)
        approval = payload.get("approval", {}) if isinstance(payload, dict) else {}
        return bool(
            approval.get("locked")
            and approval.get("micro_processor_approved")
            and approval.get("macro_coder_approved")
        )

    def _workbook_values(self) -> list[PartyMetadataValue]:
        records: dict[str, PartyMetadataValue] = {}
        for workbook in self._discover_workbooks():
            for record in self._parse_workbook(workbook):
                records.setdefault(record.variable, record)
        return [records[key] for key in sorted(records, key=_metadata_sort_key)]

    def _discover_workbooks(self) -> list[Path]:
        roots = [
            self.working_dir,
            self.working_dir / "macro",
            self.working_dir / "E-mails",
            self.working_dir / "emails",
        ]
        found: dict[str, Path] = {}
        for root in roots:
            if not root.exists():
                continue
            for path in root.rglob("*.xlsx"):
                if path.name.startswith("~$"):
                    continue
                text = " ".join(part.casefold() for part in path.parts)
                if "macro" in text or "cses" in text:
                    found[str(path.resolve()).casefold()] = path
        return sorted(found.values(), key=lambda item: _workbook_priority(item))

    def _parse_workbook(self, path: Path) -> list[PartyMetadataValue]:
        try:
            import openpyxl
        except ImportError:
            return []
        records: list[PartyMetadataValue] = []
        try:
            workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
        except Exception:
            return []
        try:
            for sheet in workbook.worksheets:
                for row_number, row in enumerate(sheet.iter_rows(values_only=True), start=1):
                    cells = ["" if cell is None else str(cell).strip() for cell in row]
                    row_text = " | ".join(cells)
                    for variable in _variables_in_text(row_text):
                        value = _value_after_variable(cells, variable)
                        if value == "":
                            continue
                        records.append(
                            PartyMetadataValue(
                                variable=variable,
                                party_letter=variable.rsplit("_", 1)[-1],
                                value=value,
                                source_type="macro_workbook",
                                source_file=str(path),
                                source_sheet=sheet.title,
                                row_number=row_number,
                                evidence=row_text[:700],
                            )
                        )
        finally:
            workbook.close()
        return records

    def _macro_report_values(self) -> list[PartyMetadataValue]:
        reports = [
            path for path in self.working_dir.rglob("*")
            if path.is_file()
            and "macro" in path.name.casefold()
            and path.suffix.casefold() in {".pdf", ".docx", ".txt", ".md"}
        ]
        if not reports:
            return []
        try:
            from src.ingest.doc_parser import DocumentParser
            from src.model_runtime import ModelRole, ModelTaskRunner

            parser = DocumentParser()
            chunks = []
            for path in reports[:6]:
                parsed = parser.parse(path)
                if parsed and parsed.full_text:
                    chunks.append(f"SOURCE: {path}\n{parsed.full_text[:50000]}")
            if not chunks:
                return []
            prompt = (
                "Extract CSES Module 6 party metadata values from the macro report text. "
                "Return only JSON with a list named values. Each item must contain "
                "variable, value, source_file, and evidence. Extract only variables "
                "F5028_A-I, F5029_A-I, F5200_A-I, F5201_A-I, F5202_A-I, F5203_A-I.\n\n"
                + "\n\n".join(chunks)
            )
            result = ModelTaskRunner(self.working_dir).complete(
                ModelRole.LARGE_TEXT,
                [{"role": "user", "content": prompt}],
                purpose="party metadata extraction from macro report",
                include_shared_context=False,
                max_tokens=3000,
                timeout=180,
                retries=0,
            )
            if result.status != "ok":
                return []
            payload = _json_from_text(result.content)
            values = []
            for item in payload.get("values", []) if isinstance(payload, dict) else []:
                variable = str(item.get("variable") or "").strip().upper()
                if not _is_party_metadata_variable(variable):
                    continue
                values.append(
                    PartyMetadataValue(
                        variable=variable,
                        party_letter=variable.rsplit("_", 1)[-1],
                        value=str(item.get("value") or "").strip(),
                        source_type="macro_report",
                        source_file=str(item.get("source_file") or ""),
                        evidence=str(item.get("evidence") or "")[:700],
                    )
                )
            return values
        except Exception:
            return []

    def _expected_variables(self, parties: dict[str, dict[str, Any]]) -> list[str]:
        letters = [letter for letter in PARTY_LETTERS if letter in parties]
        expected: list[str] = []
        for prefix in ("F5028", "F5029"):
            expected.extend(f"{prefix}_{letter}" for letter in letters)
        return expected

    def _validation_warnings(
        self,
        values: list[PartyMetadataValue],
        parties: dict[str, dict[str, Any]],
    ) -> list[str]:
        warnings: list[str] = []
        if not parties:
            warnings.append("Party Order Agreement must be approved before party metadata can be validated.")
        party_letters = set(parties)
        for value in values:
            if party_letters and value.party_letter not in party_letters:
                warnings.append(f"{value.variable} refers to a party slot not present in the approved Party A-I order.")
        if not any(item.variable.startswith("F5028_") for item in values):
            warnings.append("CSES ideological family values were not found in the macro materials.")
        if not any(item.variable.startswith("F5029_") for item in values):
            warnings.append("CSES expert left-right values were not found in the macro materials.")
        return sorted(set(warnings))

    def _load_party_order_decision(self) -> dict[str, Any]:
        path = self.working_dir / ".cses" / "party_order_decision.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


class MarporSourceAdapter:
    """Optional MARPOR/CMP adapter used only when configured."""

    API_ROOT = "https://manifestoproject.wzb.eu/api/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("MARPOR_API_KEY", "")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def request_headers(self) -> dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


def load_party_metadata_decision(working_dir: Path) -> dict[str, Any]:
    path = Path(working_dir) / ".cses" / "party_metadata_decision.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def approved_party_metadata_values(working_dir: Path) -> dict[str, str]:
    payload = load_party_metadata_decision(working_dir)
    approval = payload.get("approval", {}) if isinstance(payload, dict) else {}
    if not (
        approval.get("locked")
        and approval.get("micro_processor_approved")
        and approval.get("macro_coder_approved")
    ):
        return {}
    review = payload.get("review", {}) if isinstance(payload, dict) else {}
    values = review.get("values", []) if isinstance(review, dict) else []
    return {
        str(item.get("variable")): str(item.get("value"))
        for item in values
        if isinstance(item, dict) and item.get("variable") and str(item.get("value", "")).strip()
    }


def party_metadata_message(review: PartyMetadataReview) -> str:
    lines = ["Party metadata review prepared.", ""]
    if review.values:
        lines.append("Found party metadata:")
        grouped: dict[str, dict[str, str]] = {}
        for item in review.values:
            grouped.setdefault(item.party_letter, {})[item.variable.split("_", 1)[0]] = item.value
        for letter in PARTY_LETTERS:
            if letter not in grouped:
                continue
            parts = grouped[letter]
            labels = []
            if "F5028" in parts:
                labels.append(f"ideological family {parts['F5028']}")
            if "F5029" in parts:
                labels.append(f"CSES left-right {parts['F5029']}")
            if "F5200" in parts:
                labels.append(f"MARPOR ID {parts['F5200']}")
            lines.append(f"- Party {letter}: {', '.join(labels)}")
    else:
        lines.append("Found party metadata:")
        lines.append("- No CSES party metadata table was found in the available macro materials.")
    if review.missing_variables:
        lines.extend(["", "Needs processor review:"])
        for variable in review.missing_variables[:30]:
            lines.append(f"- {variable}")
    if review.warnings:
        lines.extend(["", "Checks needing review:"])
        for warning in review.warnings[:10]:
            lines.append(f"- {warning}")
    lines.extend(["", "Approval needed:", f"- {review.processor_decision_needed}"])
    return "\n".join(lines)


def _approved_parties(decision: dict[str, Any]) -> dict[str, dict[str, Any]]:
    proposal = decision.get("proposal", {}) if isinstance(decision, dict) else {}
    parties: dict[str, dict[str, Any]] = {}
    for party in proposal.get("proposed_parties", []) or []:
        letter = str(party.get("code_letter") or "").upper()
        if letter in PARTY_LETTERS:
            parties[letter] = party
    return parties


def _variables_in_text(text: str) -> list[str]:
    return [
        match.group(1).upper()
        for match in re.finditer(r"\b((?:F5028|F5029|F5200|F5201|F5202|F5203)_[A-I])\b", text, re.IGNORECASE)
    ]


def _value_after_variable(cells: list[str], variable: str) -> str:
    variable_index = -1
    for index, cell in enumerate(cells):
        if re.search(rf"\b{re.escape(variable)}\b", cell, re.IGNORECASE):
            variable_index = index
    if variable_index < 0:
        return ""
    equals_index = -1
    for index in range(variable_index + 1, len(cells)):
        if cells[index].strip() == "=":
            equals_index = index
            break
    start = equals_index + 1 if equals_index >= 0 else variable_index + 1
    for cell in cells[start:]:
        value = cell.strip()
        if not value:
            continue
        if value.casefold() in {"gen", "generate", "="}:
            continue
        if re.match(r"^F\d{4}_[A-I]$", value, re.IGNORECASE):
            continue
        return _clean_value(value)
    return ""


def _clean_value(value: Any) -> str:
    text = str(value).strip()
    try:
        number = float(text)
        if number.is_integer():
            return str(int(number))
        return f"{number:g}"
    except Exception:
        return text


def _is_party_metadata_variable(variable: str) -> bool:
    return bool(re.match(r"^(?:F5028|F5029|F5200|F5201|F5202|F5203)_[A-I]$", variable))


def _workbook_priority(path: Path) -> tuple[int, str]:
    text = " ".join(part.casefold() for part in path.parts)
    priority = 0 if "macro" in text else 1
    return priority, str(path).casefold()


def _metadata_sort_key(variable: str) -> tuple[int, int]:
    prefix_order = {prefix: index for index, prefix in enumerate(PARTY_METADATA_PREFIXES)}
    prefix, letter = variable.rsplit("_", 1)
    return prefix_order.get(prefix, 99), PARTY_LETTERS.index(letter) if letter in PARTY_LETTERS else 99


def _json_from_text(text: str) -> dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except Exception:
        return {}
