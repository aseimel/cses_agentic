"""Validators backed by distilled CSES wiki patterns."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from src.standards.schema import SchemaRegistry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WIKI_ROOT = PROJECT_ROOT / "cses_wiki"


@dataclass
class StandardsValidationResult:
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def validate_stata_syntax_text(text: str, wiki_root: Path = DEFAULT_WIKI_ROOT) -> StandardsValidationResult:
    """Validate Stata syntax against distilled cses_wiki pattern checks."""
    _load_pattern(wiki_root, "validation_checks")
    checks = {
        "header_block": "Process CSES-M6 Micro-Data" in text or "Process CSES" in text,
        "section_markers": "**\\\\\\" in text,
        "variable_headers": bool(re.search(r"\*+\s*>+\s*F\d+", text)),
        "tab_missing": bool(re.search(r"\btab\s+\S+\s*,\s*mis", text)),
        "missing_values": bool(re.search(r"=\s*(97|98|99|997|998|999|9997|9998|9999)\b", text)),
        "labels": "label values" in text.lower() or "label define" in text.lower(),
        "check_files": bool(re.search(r"\bdo\s+.*check", text, re.IGNORECASE)),
        "clean_stata_run": "log close" in text.lower(),
    }
    issues = [check_id for check_id, passed in checks.items() if not passed]
    return StandardsValidationResult(ok=not issues, checks=checks, issues=issues)


def validate_documentation_text(text: str, wiki_root: Path = DEFAULT_WIKI_ROOT) -> StandardsValidationResult:
    """Validate rendered documentation against distilled cses_wiki section rules."""
    patterns = _load_pattern(wiki_root, "documentation_templates")
    required = patterns.get("log_required_sections", [])
    checks = {section: section.lower() in text.lower() for section in required}
    issues = [section for section, passed in checks.items() if not passed]
    return StandardsValidationResult(ok=not issues, checks=checks, issues=issues)


def validate_module6_schema(wiki_root: Path = DEFAULT_WIKI_ROOT) -> StandardsValidationResult:
    """Validate the authoritative Module 6 schema file."""
    registry = SchemaRegistry(wiki_root)
    issues = registry.validate()
    checks = {
        "schema_file_exists": (wiki_root / "patterns" / "module6_schema.json").exists(),
        "full_release_size": registry.required_count() >= 200,
        "no_schema_issues": not issues,
    }
    if not checks["full_release_size"]:
        issues.append(f"Schema has only {registry.required_count()} variables; expected a full release schema.")
    if not checks["schema_file_exists"]:
        issues.append("Missing cses_wiki/patterns/module6_schema.json")
    return StandardsValidationResult(ok=all(checks.values()) and not issues, checks=checks, issues=issues)


def _load_pattern(wiki_root: Path, name: str) -> dict:
    path = wiki_root / "patterns" / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSES wiki pattern file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))
