"""Canonical CSES Module 6 questionnaire registry.

The raw Module 6 questionnaire text is a development input only. Runtime code
loads the distilled JSON registry from cses_wiki.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from src.standards.schema import DEFAULT_WIKI_ROOT, SchemaRegistry


REGISTRY_PATH = DEFAULT_WIKI_ROOT / "patterns" / "module6_questionnaire_registry.json"


@dataclass
class QuestionnaireRegistryItem:
    item_id: str
    normalized_id: str
    aliases: list[str]
    section: str
    item_type: str
    title: str
    text: str = ""
    notes: str = ""
    help_text: str = ""
    response_options: list[dict[str, str]] = field(default_factory=list)
    missing_codes: list[str] = field(default_factory=list)
    target_variables: list[str] = field(default_factory=list)
    source_line: int = 0


class Module6QuestionnaireRegistry:
    """Runtime loader and lookup service for the distilled registry."""

    def __init__(self, wiki_root: Path = DEFAULT_WIKI_ROOT):
        self.wiki_root = Path(wiki_root)
        self.path = self.wiki_root / "patterns" / "module6_questionnaire_registry.json"
        self.payload = self._load()
        self.items = [
            QuestionnaireRegistryItem(**item)
            for item in self.payload.get("items", [])
        ]
        self.by_normalized = {item.normalized_id: item for item in self.items}
        self.by_alias = {
            normalize_item_key(alias): item
            for item in self.items
            for alias in [item.item_id, *item.aliases]
        }

    def exists(self) -> bool:
        return bool(self.items)

    def get(self, item_id: str) -> QuestionnaireRegistryItem | None:
        return self.by_alias.get(normalize_item_key(item_id))

    def find_for_target(self, target_variable: str) -> list[QuestionnaireRegistryItem]:
        return [item for item in self.items if target_variable in item.target_variables]

    def item_ids_for_target(self, target_variable: str) -> list[str]:
        ids: list[str] = []
        for item in self.find_for_target(target_variable):
            ids.extend([item.item_id, *item.aliases])
        return list(dict.fromkeys(ids))

    def category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.items:
            counts[item.item_type] = counts.get(item.item_type, 0) + 1
        return counts

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"items": []}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {"items": []}


class Module6QuestionnaireDistiller:
    """Distill CSES_Module6_Questionnaire.txt into a runtime JSON registry."""

    HEADER_RE = re.compile(
        r"^\s*(?P<id>[AQD]\d{1,2}(?:[A-Za-z]{0,3}\d?)?(?:-[a-z])?)\.\s*>>>\s*(?P<title>.*)$",
        re.IGNORECASE,
    )

    def __init__(self, schema: SchemaRegistry | None = None):
        self.schema = schema or SchemaRegistry()

    def distill(self, source_path: Path, output_path: Path = REGISTRY_PATH) -> dict[str, Any]:
        text = Path(source_path).read_text(encoding="utf-8", errors="replace")
        items = self.parse_text(text)
        payload = {
            "schema_version": 1,
            "module": "CSES Module 6",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_policy": "Runtime uses this distilled registry. The raw questionnaire text is development-only.",
            "source_file": Path(source_path).name,
            "item_count": len(items),
            "items": [asdict(item) for item in items],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def parse_text(self, text: str) -> list[QuestionnaireRegistryItem]:
        lines = text.splitlines()
        starts: list[tuple[int, re.Match]] = []
        for index, line in enumerate(lines):
            match = self.HEADER_RE.match(line)
            if match:
                starts.append((index, match))

        items: list[QuestionnaireRegistryItem] = []
        for pos, (start, match) in enumerate(starts):
            end = starts[pos + 1][0] if pos + 1 < len(starts) else len(lines)
            block = lines[start:end]
            raw_id = match.group("id").strip()
            title_parts = [match.group("title").strip()]
            for continuation in block[1:8]:
                stripped = continuation.strip()
                if not stripped:
                    continue
                if set(stripped) <= {"-"} or stripped.endswith(".."):
                    break
                if re.match(r"^(NOTE|NOTES|TEXT|HELP):", stripped, flags=re.IGNORECASE):
                    break
                title_parts.append(stripped)
            item_id = canonical_item_id(raw_id)
            section = _section_for_id(item_id)
            notes = _extract_labeled_text(block, {"NOTE", "NOTES"})
            text_value = _extract_labeled_text(block, {"TEXT"})
            help_text = _extract_labeled_text(block, {"HELP"})
            response_options = _extract_response_options(block)
            missing_codes = [
                option["code"]
                for option in response_options
                if "missing" in option["label"].lower()
                or "refused" in option["label"].lower()
                or "don't know" in option["label"].lower()
                or "dont know" in option["label"].lower()
            ]
            item_type = _item_type(item_id, " ".join(title_parts), section)
            items.append(
                QuestionnaireRegistryItem(
                    item_id=item_id,
                    normalized_id=normalize_item_key(item_id),
                    aliases=item_aliases(item_id),
                    section=section,
                    item_type=item_type,
                    title=_clean(" ".join(title_parts)),
                    text=_clean(text_value, 4000),
                    notes=_clean(notes, 3000),
                    help_text=_clean(help_text, 2000),
                    response_options=response_options,
                    missing_codes=missing_codes,
                    target_variables=self._target_variables_for_item(item_id),
                    source_line=start + 1,
                )
            )
        return items

    def _target_variables_for_item(self, item_id: str) -> list[str]:
        aliases = {normalize_item_key(alias) for alias in [item_id, *item_aliases(item_id)]}
        try:
            from src.matching.decision_engine import MODULE6_SOURCE_ALIASES
        except Exception:
            MODULE6_SOURCE_ALIASES = {}
        linked: list[str] = []
        for target in self.schema.variables:
            target_aliases = MODULE6_SOURCE_ALIASES.get(target.name, [])
            if any(normalize_item_key(alias) in aliases for alias in target_aliases):
                linked.append(target.name)
        return linked


def audit_questionnaire_registry(path: Path = REGISTRY_PATH) -> list[str]:
    issues: list[str] = []
    if not path.exists():
        return [f"Missing questionnaire registry: {path}"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"Questionnaire registry is not valid JSON: {exc}"]
    items = payload.get("items", [])
    if len(items) < 50:
        issues.append(f"Questionnaire registry has only {len(items)} items.")
    normalized = [item.get("normalized_id") for item in items]
    duplicates = sorted({item for item in normalized if item and normalized.count(item) > 1})
    for duplicate in duplicates[:10]:
        issues.append(f"Duplicate questionnaire registry item: {duplicate}")
    for required in ["Q01", "Q02a", "Q10LH-b", "Q19", "D02", "A5"]:
        if not any(required in {item.get("item_id"), *(item.get("aliases") or [])} for item in items):
            issues.append(f"Required questionnaire item missing: {required}")
    return issues


def canonical_item_id(value: str) -> str:
    value = str(value or "").strip()
    match = re.match(r"^([AQD])(\d{1,2})(.*)$", value, flags=re.IGNORECASE)
    if not match:
        return value
    prefix, number, suffix = match.groups()
    if prefix.upper() in {"Q", "D"}:
        number_text = f"{int(number):02d}"
    else:
        number_text = str(int(number))
    return f"{prefix.upper()}{number_text}{suffix}"


def normalize_item_key(value: str) -> str:
    value = canonical_item_id(value)
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def item_aliases(item_id: str) -> list[str]:
    canonical = canonical_item_id(item_id)
    aliases = {canonical}
    compact = canonical.replace("-", "")
    aliases.add(compact)
    aliases.add(canonical.replace("-", "_"))
    aliases.add(normalize_item_key(canonical))
    match = re.match(r"^([AQD])0?(\d{1,2})(.*)$", canonical, flags=re.IGNORECASE)
    if match:
        prefix, number, suffix = match.groups()
        aliases.add(f"{prefix.upper()}{int(number)}{suffix}")
        aliases.add(f"{prefix.upper()}{int(number):02d}{suffix}")
    return [alias for alias in aliases if alias != canonical]


def _section_for_id(item_id: str) -> str:
    if item_id.startswith("A"):
        return "administrative"
    if item_id.startswith("D"):
        return "demographic"
    return "cses_module"


def _item_type(item_id: str, title: str, section: str) -> str:
    lowered = title.lower()
    if section == "administrative":
        return "admin_supplied_variable"
    if section == "demographic":
        return "demographic_coding_standard"
    if "optional" in lowered:
        return "optional_question"
    if any(term in lowered for term in ["vote choice", "party", "leader", "election"]):
        return "party_vote_item"
    return "core_question"


def _extract_labeled_text(block: list[str], labels: set[str]) -> str:
    capture = False
    parts: list[str] = []
    label_re = re.compile(r"^(NOTE|NOTES|TEXT|HELP):\s*(.*)$", re.IGNORECASE)
    for line in block:
        stripped = line.strip()
        match = label_re.match(stripped)
        if match:
            label = match.group(1).upper()
            capture = label in labels
            if capture and match.group(2):
                parts.append(match.group(2).strip())
            continue
        if capture:
            if set(stripped) <= {"-"} or stripped.startswith(">>>"):
                continue
            if label_re.match(stripped):
                break
            if re.match(r"^\d{1,3}\.\s+", stripped):
                continue
            parts.append(stripped)
    return _clean(" ".join(parts))


def _extract_response_options(block: list[str]) -> list[dict[str, str]]:
    options: list[dict[str, str]] = []
    for line in block:
        stripped = line.strip()
        match = re.match(r"^(\d{1,3})\.\s+(.+)$", stripped)
        if not match:
            continue
        code, label = match.groups()
        if len(label) > 180:
            continue
        options.append({"code": code, "label": _clean(label, 180)})
    return options


def _clean(text: str, limit: int | None = None) -> str:
    value = " ".join(str(text or "").split())
    value = value.encode("cp1252", errors="replace").decode("cp1252")
    return value[:limit] if limit else value
