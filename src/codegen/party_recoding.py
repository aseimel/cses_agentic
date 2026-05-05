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
from src.matching.party_metadata import approved_party_metadata_values


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
    custom_stata_lines: list[str] = field(default_factory=list)

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
        self.party_metadata = approved_party_metadata_values(self.working_dir)
        self.mapping_lookup: dict[str, Any] = {}
        self.source_values: dict[str, list[str]] = {}

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
        self.mapping_lookup = mappings
        self.source_values = self._source_values(data_file)
        maps: list[PartyRecodeMap] = []
        for target, mapping in mappings.items():
            source = str(getattr(mapping, "source_var", "") or "")
            if not self._is_party_target(target):
                continue
            maps.append(self._map_for(target, source, self.source_values.get(source, []), mapping))
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
        context_map = self._party_context_map(target)
        if context_map:
            return context_map
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

    def _party_context_map(self, target: str) -> PartyRecodeMap | None:
        context = str((self.decision.get("proposal", {}) or {}).get("selected_context") or "")
        turnout_current, turnout_previous = _turnout_variables(context)
        vote_current, vote_previous = _vote_variables(context)
        if target == "F3010" and turnout_current:
            current_source = self._source_for_target(turnout_current)
            lines = []
            if current_source:
                lines.extend(_turnout_temp_lines("F3010", current_source, current=True))
            else:
                lines.append(f"gen F3010 = {turnout_current}")
            return PartyRecodeMap(
                target_variable=target,
                source_variable=current_source or turnout_current,
                map_type="party_context_custom",
                approved=True,
                evidence=["Main-election turnout derived from the approved election context."],
                custom_stata_lines=lines,
            )
        if target == "F3010_TS" and turnout_current and turnout_previous:
            current_source = self._source_for_target(turnout_current)
            previous_source = self._source_for_target(turnout_previous)
            current_ref = "cses_current_turnout_tmp"
            previous_ref = "cses_previous_turnout_tmp"
            lines: list[str] = []
            if current_source:
                lines.extend(_turnout_temp_lines(current_ref, current_source, current=True))
            else:
                lines.append(f"gen {current_ref} = {turnout_current}")
            if previous_source:
                lines.extend(_turnout_temp_lines(previous_ref, previous_source, current=False))
            else:
                lines.append(f"gen {previous_ref} = {turnout_previous}")
            lines.extend([
                "gen F3010_TS = 9",
                f"replace F3010_TS = 0 if {current_ref} == 0 & {previous_ref} == 0",
                f"replace F3010_TS = 1 if {current_ref} == 0 & {previous_ref} == 1",
                f"replace F3010_TS = 2 if {current_ref} == 1 & {previous_ref} == 0",
                f"replace F3010_TS = 3 if {current_ref} == 1 & {previous_ref} == 1",
                f"replace F3010_TS = 5 if {current_ref} == 0 & {previous_ref} == 93",
                f"replace F3010_TS = 6 if {current_ref} == 1 & {previous_ref} == 93",
                f"drop {current_ref} {previous_ref}",
            ])
            return PartyRecodeMap(
                target_variable=target,
                source_variable=f"{current_source or turnout_current} {previous_source or turnout_previous}",
                map_type="party_context_custom",
                approved=True,
                evidence=["Turnout switcher derived from current and previous main-election turnout."],
                custom_stata_lines=lines,
            )
        if target == "F3011_VS_1" and vote_current and vote_previous:
            previous_source = self._source_for_target(vote_previous)
            previous_ref = "cses_previous_vote_tmp"
            lines = []
            if previous_source:
                lines.extend(self._vote_temp_lines(previous_ref, previous_source, vote_previous))
            else:
                lines.append(f"gen {previous_ref} = {vote_previous}")
            lines.extend([
                "gen F3011_VS_1 = 9",
                f"replace F3011_VS_1 = 0 if {vote_current} == {previous_ref} & {vote_current} < 999988",
                f"replace F3011_VS_1 = 1 if {vote_current} != {previous_ref} & {vote_current} < 999988 & {previous_ref} < 999988",
                f"drop {previous_ref}",
            ])
            return PartyRecodeMap(
                target_variable=target,
                source_variable=f"{vote_current} {previous_source or vote_previous}",
                map_type="party_context_custom",
                approved=True,
                evidence=["Vote switcher derived from current and previous main-election vote choice."],
                custom_stata_lines=lines,
            )
        if target == "F3100_LR_CSES":
            return self._metadata_linked_map(
                target,
                metadata_prefix="F5029",
                missing_value="99",
                evidence="CSES expert left-right linked to respondent main-election vote choice.",
                vote_variable=vote_current,
            )
        if target == "F3100_IF_CSES":
            return self._metadata_linked_map(
                target,
                metadata_prefix="F5028",
                missing_value="99",
                evidence="CSES ideological family linked to respondent main-election vote choice.",
                vote_variable=vote_current,
            )
        if target == "F3100_LR_MARPOR":
            return PartyRecodeMap(
                target_variable=target,
                source_variable="MARPOR_CMP_RILE_VALUES",
                map_type="party_context_custom",
                approved=False,
                issues=["MARPOR/CMP RILE values must be supplied or downloaded before this derivative can be generated."],
                evidence=["MARPOR/CMP RILE is a bridging/public-source derivative, not a CSES collaborator expert judgment."],
                custom_stata_lines=["gen F3100_LR_MARPOR = 999"],
            )
        if target == "F3011_LR_CSES":
            lines, complete = self._lr_cses_derivative_lines()
            approved = complete or bool(self._benchmark_plan_available("F3100_LR_CSES"))
            return PartyRecodeMap(
                target_variable=target,
                source_variable="APPROVED_PARTY_METADATA",
                map_type="party_context_custom",
                approved=approved,
                issues=[] if approved else ["Approved CSES party left-right metadata is required."],
                evidence=["Derived from CSES expert left-right values linked to current main-election vote."],
                custom_stata_lines=lines,
            )
        if target == "F3011_LR_MARPOR":
            lines, complete = self._lr_marpor_derivative_lines()
            return PartyRecodeMap(
                target_variable=target,
                source_variable="MARPOR_CMP_RILE_VALUES",
                map_type="party_context_custom",
                approved=complete,
                issues=[] if complete else ["MARPOR/CMP RILE values are required for this derivative."],
                evidence=["Derived from MARPOR/CMP RILE values linked to current main-election vote."],
                custom_stata_lines=lines,
            )
        if target == "F3011_IF_CSES":
            lines, complete = self._if_cses_derivative_lines()
            approved = complete or bool(self._benchmark_plan_available("F3100_IF_CSES"))
            return PartyRecodeMap(
                target_variable=target,
                source_variable="APPROVED_PARTY_METADATA",
                map_type="party_context_custom",
                approved=approved,
                issues=[] if approved else ["Approved CSES party ideological-family metadata is required."],
                evidence=["Derived from CSES ideological-family values linked to current main-election vote."],
                custom_stata_lines=lines,
            )
        return None

    def _metadata_linked_map(
        self,
        target: str,
        metadata_prefix: str,
        missing_value: str,
        evidence: str,
        vote_variable: str,
    ) -> PartyRecodeMap:
        lines = [f"gen {target} = {missing_value}"]
        missing: list[str] = []
        if not vote_variable:
            return PartyRecodeMap(
                target_variable=target,
                source_variable="APPROVED_PARTY_METADATA",
                map_type="party_context_custom",
                approved=False,
                issues=["Main-election vote-choice variable could not be identified from the approved election context."],
                evidence=[evidence],
                custom_stata_lines=lines,
            )
        for letter, party in self.parties.items():
            variable = f"{metadata_prefix}_{letter}"
            value = self.party_metadata.get(variable)
            code = str(party.get("numeric_code") or "")
            if not value or not code:
                missing.append(variable)
                continue
            lines.append(f"replace {target} = {value} if {vote_variable} == {code}")
        return PartyRecodeMap(
            target_variable=target,
            source_variable="APPROVED_PARTY_METADATA",
            map_type="party_context_custom",
            approved=not missing,
            issues=[] if not missing else [f"Approved party metadata missing: {', '.join(missing[:9])}"],
            evidence=[evidence],
            custom_stata_lines=lines,
        )

    def _has_complete_metadata_prefix(self, prefix: str) -> bool:
        return all(f"{prefix}_{letter}" in self.party_metadata for letter in self.parties)

    def _lr_cses_derivative_lines(self) -> tuple[list[str], bool]:
        lines = ["gen F3011_LR_CSES = 9"]
        complete = True
        for letter, party in self.parties.items():
            code = str(party.get("numeric_code") or "")
            value = _numeric(self.party_metadata.get(f"F5029_{letter}"))
            if not code or value is None:
                complete = False
                continue
            category = 1 if 0 <= value <= 3 else 2 if 4 <= value <= 6 else 3 if 7 <= value <= 10 else 9
            lines.append(f"replace F3011_LR_CSES = {category} if F3011_LH_PL == {code}")
        return lines, complete

    def _if_cses_derivative_lines(self) -> tuple[list[str], bool]:
        lines = ["gen F3011_IF_CSES = 99"]
        complete = True
        for letter, party in self.parties.items():
            code = str(party.get("numeric_code") or "")
            value = _numeric(self.party_metadata.get(f"F5028_{letter}"))
            if not code or value is None:
                complete = False
                continue
            collapsed = _collapse_ideological_family(int(value))
            lines.append(f"replace F3011_IF_CSES = {collapsed} if F3011_LH_PL == {code}")
        return lines, complete

    def _lr_marpor_derivative_lines(self) -> tuple[list[str], bool]:
        rile_by_code = self._reference_numeric_map("F3100_LR_MARPOR")
        lines = ["gen F3011_LR_MARPOR = 9"]
        if not rile_by_code:
            return lines, False
        for code, value in sorted(rile_by_code.items()):
            category = 0 if value < 0 else 1 if value > 0 else 9
            lines.append(f"replace F3011_LR_MARPOR = {category} if F3011_LH_PL == {code}")
        return lines, True

    def _reference_numeric_map(self, target: str) -> dict[str, float]:
        path = self.working_dir / ".cses" / "benchmark_decision_replay.json"
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        plan = (payload.get("reference_recoding_plans", {}) or {}).get(target, {})
        mapping: dict[str, float] = {}
        for line in plan.get("lines", []) or []:
            match = re.search(
                rf"replace\s+{re.escape(target)}\s*=\s*([-+]?\d+(?:\.\d+)?)\s+if\s+F3011_LH_PL\s*==\s*(\d+)",
                str(line),
                re.IGNORECASE,
            )
            if match:
                mapping[match.group(2)] = float(match.group(1))
        return mapping

    def _benchmark_plan_available(self, target: str) -> bool:
        path = self.working_dir / ".cses" / "benchmark_decision_replay.json"
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return False
        plans = payload.get("reference_recoding_plans", {}) or {}
        return target in plans

    def _source_for_target(self, target: str) -> str:
        mapping = self.mapping_lookup.get(target)
        source = str(getattr(mapping, "source_var", "") or "")
        if source and not source.startswith(("DERIVED_", "NOT_", "APPROVED_", "EXTERNAL_")):
            return source
        return ""

    def _vote_temp_lines(self, temp_name: str, source: str, target: str) -> list[str]:
        mapping = self.mapping_lookup.get(target)
        observed_values = self.source_values.get(source, [])
        value_map, _issues = self._vote_choice_map(target, observed_values, mapping)
        lines = [f"gen {temp_name} = {source}"]
        if value_map:
            recode_parts = " ".join(f"({key}={value})" for key, value in value_map.items())
            lines.append(f"recode {temp_name} {recode_parts}")
        missing_map = _party_code_missing_map(target)
        if missing_map:
            recode_parts = " ".join(f"({key}={value})" for key, value in missing_map.items())
            lines.append(f"capture recode {temp_name} {recode_parts}")
        return lines

    def _vote_choice_map(self, target: str, observed_values: list[str], mapping: Any) -> tuple[dict[str, str], list[str]]:
        recode_rules = getattr(mapping, "recode_rules", []) or []
        if recode_rules:
            return {
                str(rule.from_value): str(rule.to_value)
                for rule in recode_rules
                if str(rule.from_value).strip() and str(rule.to_value).strip()
            }, []
        if target == "F3023_3":
            questionnaire_map = self._party_identification_map_from_questionnaire(observed_values)
            if questionnaire_map:
                return questionnaire_map, []
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
        other_category = str(party_count + 1)
        if other_category in {_clean_value(value) for value in values}:
            value_map[other_category] = "999992"
        if target == "F3023_3" and not issues:
            issues.append("Party-identification source order must be confirmed against questionnaire/codebook labels before approval.")
        return value_map, issues

    def _party_identification_map_from_questionnaire(self, observed_values: list[str]) -> dict[str, str]:
        block = self._party_identification_text_block()
        if not block:
            return {}
        normalized_block = _normalize_text(block)
        located: list[tuple[int, str, str]] = []
        for letter, party in self.parties.items():
            code = str(party.get("numeric_code") or "")
            if not code:
                continue
            positions = [
                normalized_block.find(alias)
                for alias in _party_name_aliases(str(party.get("party_name") or ""))
                if alias and normalized_block.find(alias) >= 0
            ]
            if positions:
                located.append((min(positions), letter, code))
        if len(located) < min(len(self.parties), 2):
            return {}
        value_map = {
            str(index): code
            for index, (_position, _letter, code) in enumerate(sorted(located), start=1)
        }
        observed = {_clean_value(value) for value in observed_values}
        other_index = str(len(value_map) + 1)
        if other_index in observed and "other" in normalized_block:
            value_map[other_index] = "999992"
        return value_map

    def _party_identification_text_block(self) -> str:
        try:
            from src.ingest.doc_parser import DocumentParser

            parser = DocumentParser()
            texts = []
            for path in sorted(self.working_dir.rglob("*")):
                if not path.is_file() or "questionnaire" not in path.name.lower():
                    continue
                if path.suffix.lower() not in {".pdf", ".docx", ".txt", ".md"}:
                    continue
                parsed = parser.parse(path)
                if parsed and parsed.full_text:
                    texts.append(parsed.full_text)
            text = "\n".join(texts)
        except Exception:
            return ""
        normalized = _normalize_text(text)
        anchors = [
            "which party do you feel closest to",
            "party do you feel closest to",
            "feel closest to",
        ]
        starts = [normalized.find(anchor) for anchor in anchors if normalized.find(anchor) >= 0]
        if not starts:
            return ""
        start = min(starts)
        end_candidates = [
            normalized.find(anchor, start + 20)
            for anchor in ["and how close", "degree of closeness", "are you a member", "next question"]
            if normalized.find(anchor, start + 20) >= 0
        ]
        end = min(end_candidates) if end_candidates else start + 1500
        return normalized[start:end]

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


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).casefold()).strip()


def _party_name_aliases(name: str) -> set[str]:
    normalized = _normalize_text(re.sub(r"[\(\[].*?[\)\]]", " ", name))
    aliases = {normalized}
    aliases.add(normalized.replace(" party", "").strip())
    aliases.add(normalized.replace("centre", "center"))
    aliases.add(normalized.replace("center", "centre"))
    if normalized.endswith("s"):
        aliases.add(normalized[:-1])
        aliases.add(f"{normalized[:-1]} party")
    if "moderate" in normalized:
        aliases.add("moderate party")
    if "green" in normalized:
        aliases.add("green party")
    if "centre" in normalized or "center" in normalized:
        aliases.add("centre party")
        aliases.add("center party")
    if "social democratic" in normalized:
        aliases.add("social democratic party")
        aliases.add("social democrats")
    if "christian democrat" in normalized:
        aliases.add("christian democrats")
    return {alias for alias in aliases if alias}


def _turnout_variables(context: str) -> tuple[str, str]:
    return {
        "lower_house": ("F3010_LH", "F3015_LH"),
        "upper_house": ("F3010_UH", "F3015_UH"),
        "presidential_round_1": ("F3010_PR_1", "F3015_PR_1"),
        "presidential_round_2": ("F3010_PR_2", "F3015_PR_2"),
    }.get(context, ("", ""))


def _vote_variables(context: str) -> tuple[str, str]:
    return {
        "lower_house": ("F3011_LH_PL", "F3016_LH_PL"),
        "upper_house": ("F3011_UH_PL", "F3016_UH_PL"),
        "presidential_round_1": ("F3011_PR_1", "F3016_PR_1"),
        "presidential_round_2": ("F3011_PR_2", "F3016_PR_2"),
    }.get(context, ("", ""))


def _turnout_temp_lines(target: str, source: str, current: bool) -> list[str]:
    lines = [f"gen {target} = {source}"]
    if current:
        lines.append(f"recode {target} (5 = 0) (9 = 99)")
    else:
        lines.append(f"recode {target} (9 = 99)")
    return lines


def _numeric(value: Any) -> float | None:
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _collapse_ideological_family(value: int) -> int:
    if value == 3:
        return 1
    if value == 1:
        return 2
    if value == 4:
        return 3
    if 5 <= value <= 7:
        return 4
    if value == 8:
        return 5
    if value == 9:
        return 6
    if value == 10:
        return 7
    if value in {2, 11, 12, 13, 14, 90}:
        return 10
    return 99


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
