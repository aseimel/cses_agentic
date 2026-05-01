"""Party-order-backed recoding plans for CSES party variables."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.matching.party_order import PARTY_LETTERS


@dataclass
class PartyRecodeMap:
    target_variable: str
    source_variable: str
    map_type: str
    value_map: dict[str, str] = field(default_factory=dict)
    missing_map: dict[str, str] = field(default_factory=dict)
    approved: bool = False
    issues: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PartyRecodingPlanBuilder:
    """Build recoding maps from a locked Party Order Agreement."""

    VOTE_CODE_TARGETS = {
        "F3011_LH_PL",
        "F3011_LH_PL2",
        "F3011_UH_PL",
        "F3016_LH_PL",
        "F3016_UH_PL",
        "F3023_3",
    }

    SCALE_PREFIXES = ("F3018_", "F3019_", "F3020_", "F3021_")
    PARTY_IDENTIFIER_PREFIXES = ("F5000_", "F5000_L_", "F5200_", "F5201_", "F5202_", "F5203_")

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.decision = self._load_decision()
        self.parties = self._approved_parties()

    @property
    def approved(self) -> bool:
        approval = self.decision.get("approval", {}) if isinstance(self.decision, dict) else {}
        return bool(
            approval.get("locked")
            and approval.get("micro_processor_approved")
            and approval.get("macro_coder_approved")
        )

    def build_maps(
        self,
        mappings: dict[str, Any],
        data_file: str = "",
    ) -> list[PartyRecodeMap]:
        source_values = self._source_values(data_file)
        maps: list[PartyRecodeMap] = []
        for target, mapping in mappings.items():
            source = str(getattr(mapping, "source_var", "") or "")
            if not self._is_party_target(target):
                continue
            maps.append(self._map_for(target, source, source_values.get(source, []), mapping))
        return maps

    def write(self, maps: list[PartyRecodeMap]) -> Path:
        path = self.working_dir / ".cses" / "party_recode_maps.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "approved_party_order": self.approved,
            "map_count": len(maps),
            "approved_count": sum(1 for item in maps if item.approved),
            "maps": [item.to_dict() for item in maps],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def party_code_for_letter(self, letter: str) -> str:
        party = self.parties.get(letter.upper())
        return str(party.get("numeric_code") or "") if party else ""

    def party_name_for_letter(self, letter: str) -> str:
        party = self.parties.get(letter.upper())
        return str(party.get("party_name") or "") if party else ""

    def party_letters(self) -> list[str]:
        return [letter for letter in PARTY_LETTERS if letter in self.parties]

    def _map_for(
        self,
        target: str,
        source: str,
        observed_values: list[str],
        mapping: Any,
    ) -> PartyRecodeMap:
        mapping_approved = bool(getattr(mapping, "verified", False))
        if not self.approved:
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source or "PARTY_ORDER_AGREEMENT",
                map_type="approval_required",
                issues=["Party Order Agreement must be approved before party recoding."],
            )
        letter = _target_party_letter(target)
        if source.startswith("NO_APPROVED_PARTY_FOR_THIS_SLOT"):
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source,
                map_type="not_applicable_party_slot",
                value_map={"*": _party_scale_not_applicable(target)},
                approved=True,
                evidence=["No approved Party A-I entry exists for this slot."],
            )
        if source.startswith("OPTIONAL_ALTERNATIVE_SCALE_NOT_COLLECTED"):
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source,
                map_type="not_collected",
                value_map={"*": _party_scale_missing(target)},
                approved=True,
                evidence=["Optional party scale was not collected or not used."],
            )
        if source.startswith("NOT_APPLICABLE"):
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source,
                map_type="not_applicable",
                value_map={"*": _party_code_not_applicable(target)},
                approved=True,
                evidence=["Election context makes this party variable not applicable."],
            )
        if source.startswith("DERIVED_FROM_APPROVED"):
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source,
                map_type="derived_party_metadata",
                value_map={"*": _party_code_missing(target)},
                approved=mapping_approved,
                issues=[] if mapping_approved else ["Processor must approve derived party metadata before final syntax."],
                evidence=["Derived from approved party order, vote choice, and party metadata."],
            )
        if target.startswith(self.PARTY_IDENTIFIER_PREFIXES):
            code = self.party_code_for_letter(letter or "")
            if letter and not code:
                return PartyRecodeMap(
                    target_variable=target,
                    source_variable="NO_APPROVED_PARTY_FOR_THIS_SLOT",
                    map_type="not_applicable_party_slot",
                    value_map={"*": _party_code_missing(target)},
                    approved=True,
                    evidence=[f"No approved Party {letter} entry exists for this slot."],
                )
            if not letter:
                return PartyRecodeMap(
                    target_variable=target,
                    source_variable=source or "APPROVED_PARTY_ORDER",
                    map_type="party_context",
                    value_map={"*": _party_code_missing(target)},
                    approved=mapping_approved,
                    issues=[] if mapping_approved else ["Processor must approve party or leader metadata generation."],
                    evidence=["Derived from approved party or leader metadata."],
                )
            return PartyRecodeMap(
                target_variable=target,
                source_variable="APPROVED_PARTY_ORDER",
                map_type="party_identifier",
                value_map={"*": code or _party_code_missing(target)},
                approved=bool(code),
                issues=[] if code else ["No approved party code exists for this party slot."],
                evidence=[f"Party {letter}: {self.party_name_for_letter(letter or '')}"],
            )
        if target in self.VOTE_CODE_TARGETS:
            value_map, issues = self._vote_choice_map(target, observed_values, mapping)
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source,
                map_type="party_vote_choice",
                value_map=value_map,
                missing_map=_party_code_missing_map(target),
                approved=mapping_approved and not issues,
                issues=issues + ([] if mapping_approved else ["Processor must approve source category to party-code mapping."]),
                evidence=["Vote-choice source categories mapped to the approved Party A-I order."],
            )
        if target.startswith(self.SCALE_PREFIXES):
            return PartyRecodeMap(
                target_variable=target,
                source_variable=source,
                map_type="party_scale_direct",
                approved=mapping_approved,
                issues=[] if mapping_approved else ["Processor must approve party/leader scale source mapping."],
                evidence=["Party/leader scale copied after Party Order Agreement approval."],
            )
        return PartyRecodeMap(
            target_variable=target,
            source_variable=source or "APPROVED_PARTY_ORDER",
            map_type="party_context",
            value_map={"*": _party_code_missing(target)},
            approved=mapping_approved,
            issues=[] if mapping_approved else ["Processor must approve party-context generation."],
        )

    def _vote_choice_map(self, target: str, observed_values: list[str], mapping: Any) -> tuple[dict[str, str], list[str]]:
        recode_rules = getattr(mapping, "recode_rules", []) or []
        if recode_rules:
            return {
                str(rule.from_value): str(rule.to_value)
                for rule in recode_rules
                if str(rule.from_value).strip() and str(rule.to_value).strip()
            }, []
        values = [value for value in observed_values if _is_simple_numeric(value)]
        party_count = len(self.parties)
        eligible = [value for value in values if 1 <= int(float(value)) <= party_count]
        issues: list[str] = []
        if not eligible:
            issues.append("No numeric source categories could be linked to Party A-I slots.")
        if len(set(eligible)) < party_count:
            issues.append("Source categories do not cover every approved Party A-I slot.")
        value_map = {}
        for value in sorted(set(eligible), key=lambda item: int(float(item))):
            letter = PARTY_LETTERS[int(float(value)) - 1]
            code = self.party_code_for_letter(letter)
            if code:
                value_map[_clean_value(value)] = code
        if target == "F3023_3" and not issues:
            issues.append("Party-identification source order must be confirmed against questionnaire/codebook labels before approval.")
        return value_map, issues

    def _source_values(self, data_file: str) -> dict[str, list[str]]:
        if not data_file:
            return {}
        path = Path(data_file)
        if not path.exists():
            return {}
        try:
            if path.suffix.lower() == ".dta":
                import pyreadstat

                frame, _ = pyreadstat.read_dta(str(path), apply_value_formats=False)
            elif path.suffix.lower() == ".csv":
                frame = pd.read_csv(path)
            elif path.suffix.lower() in {".xlsx", ".xls"}:
                frame = pd.read_excel(path)
            else:
                return {}
        except Exception:
            return {}
        values: dict[str, list[str]] = {}
        for column in frame.columns:
            try:
                observed = frame[column].dropna().unique().tolist()
            except Exception:
                observed = []
            values[str(column)] = [_clean_value(value) for value in observed[:200]]
        return values

    def _load_decision(self) -> dict[str, Any]:
        path = self.working_dir / ".cses" / "party_order_decision.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _approved_parties(self) -> dict[str, dict[str, Any]]:
        proposal = self.decision.get("proposal", {}) if isinstance(self.decision, dict) else {}
        parties = {}
        for party in proposal.get("proposed_parties", []) or []:
            letter = str(party.get("code_letter") or "").upper()
            if letter in PARTY_LETTERS:
                parties[letter] = party
        return parties

    def _is_party_target(self, target: str) -> bool:
        if target.startswith(self.PARTY_IDENTIFIER_PREFIXES):
            return True
        if target in self.VOTE_CODE_TARGETS:
            return True
        if target.startswith(self.SCALE_PREFIXES):
            return True
        if target.startswith(("F3010", "F3011", "F3015", "F3016", "F3023", "F3100", "F6000")):
            return True
        return False


def _target_party_letter(target: str) -> str:
    patterns = [
        r"^F30(?:18|19|20|21)_([A-I])$",
        r"^F5\d{3}_L?_?([A-I])$",
        r"^F52\d{2}_([A-I])$",
    ]
    for pattern in patterns:
        match = re.match(pattern, target)
        if match:
            return match.group(1)
    return ""


def _is_simple_numeric(value: str) -> bool:
    try:
        number = float(str(value))
    except ValueError:
        return False
    return number.is_integer()


def _clean_value(value: Any) -> str:
    try:
        number = float(str(value).strip())
        if number.is_integer():
            return str(int(number))
        return str(number)
    except (TypeError, ValueError):
        return str(value).strip()


def _party_code_missing_map(target: str) -> dict[str, str]:
    if target.startswith(("F3011", "F3016", "F3023", "F5", "F6")):
        return {"97": "999997", "98": "999998", "99": "999999"}
    return {"97": "97", "98": "98", "99": "99"}


def _party_code_not_applicable(target: str) -> str:
    if target.startswith(("F3011", "F3016", "F3023", "F5", "F6")):
        return "999997"
    return "97"


def _party_code_missing(target: str) -> str:
    if target.startswith(("F3011", "F3016", "F3023", "F5", "F6")):
        return "999999"
    return "99"


def _party_scale_not_applicable(target: str) -> str:
    return "97" if target.startswith(("F3018", "F3019", "F3020", "F3021")) else _party_code_not_applicable(target)


def _party_scale_missing(target: str) -> str:
    return "99" if target.startswith(("F3018", "F3019", "F3020", "F3021")) else _party_code_missing(target)
