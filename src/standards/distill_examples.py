"""
Distill durable standards from development-only example studies.

This module reads raw example_studies when available and writes compact JSON
patterns into cses_wiki/patterns. Runtime workflow code should use cses_wiki,
not the raw example folders.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXAMPLES_ROOT = PROJECT_ROOT / "example_studies"
DEFAULT_WIKI_ROOT = PROJECT_ROOT / "cses_wiki"

OLD_PATH_RE = re.compile(r"[/\\](_OLD|old|OLD|E-mails|emails|macro[/\\]OLD)([/\\]|$)")
AUTHOR_RE = re.compile(r"File Author:\s*(.+?)\s*\*{2}", re.IGNORECASE)
DATE_RE = re.compile(r"Date:\s*(.+?)\s*\*{2}", re.IGNORECASE)
MODULE_RE = re.compile(r"CSES MODULE\s+(\d+):\s*(.+?)\s*\*{2}", re.IGNORECASE)
SECTION_RE = re.compile(r"\*+\s*\\\\\\\s*([^\n*]+)")
VARIABLE_HEADING_RE = re.compile(r"\*+\s*>+\s*(F\d+[_A-Z0-9]*)\s*[-\u2013]\s*([^\n*]+)")


@dataclass
class SyntaxExample:
    path: str
    artifact_type: str
    author: str = ""
    date: str = ""
    module_context: str = ""
    priority: str = "secondary"
    size_bytes: int = 0
    header_excerpt: str = ""
    section_headings: list[str] = field(default_factory=list)
    variable_heading_examples: list[dict[str, str]] = field(default_factory=list)
    variable_code_examples: list[dict[str, str]] = field(default_factory=list)
    observed_idioms: dict[str, bool] = field(default_factory=dict)


def distill_example_studies(
    examples_root: Path = DEFAULT_EXAMPLES_ROOT,
    wiki_root: Path = DEFAULT_WIKI_ROOT,
) -> dict[str, Path]:
    """Distill raw example studies into compact CSES wiki pattern files."""
    if not examples_root.exists():
        raise FileNotFoundError(f"Example studies folder not found: {examples_root}")

    patterns_dir = wiki_root / "patterns"
    patterns_dir.mkdir(parents=True, exist_ok=True)

    syntax = _distill_stata_syntax(examples_root)
    documentation = _distill_documentation_templates(examples_root)
    validation = _build_validation_checks(syntax)
    module6_schema = _distill_module6_schema(syntax)
    replication_benchmarks = _build_replication_benchmarks()

    outputs = {
        "stata_syntax": patterns_dir / "stata_syntax.json",
        "documentation_templates": patterns_dir / "documentation_templates.json",
        "validation_checks": patterns_dir / "validation_checks.json",
        "module6_schema": patterns_dir / "module6_schema.json",
        "replication_benchmarks": patterns_dir / "replication_benchmarks.json",
    }

    _write_json(outputs["stata_syntax"], syntax)
    _write_json(outputs["documentation_templates"], documentation)
    _write_json(outputs["validation_checks"], validation)
    _write_json(outputs["module6_schema"], module6_schema)
    _write_json(outputs["replication_benchmarks"], replication_benchmarks)
    return outputs


def audit_distilled_wiki(wiki_root: Path = DEFAULT_WIKI_ROOT) -> list[str]:
    """Return human-readable issues for missing distilled pattern files."""
    issues = []
    required = [
        wiki_root / "patterns" / "stata_syntax.json",
        wiki_root / "patterns" / "documentation_templates.json",
        wiki_root / "patterns" / "validation_checks.json",
        wiki_root / "patterns" / "module6_schema.json",
        wiki_root / "patterns" / "replication_benchmarks.json",
        wiki_root / "retrieval" / "chunks.jsonl",
    ]
    for path in required:
        if not path.exists():
            issues.append(f"Missing required CSES wiki file: {path.relative_to(wiki_root)}")
    return issues


def _distill_stata_syntax(examples_root: Path) -> dict:
    examples = []
    for path in sorted(examples_root.rglob("*.do"), key=lambda p: str(p).lower()):
        if OLD_PATH_RE.search(str(path)):
            continue
        artifact_type = _classify_do_file(path)
        if artifact_type == "other_do":
            continue
        example = _read_syntax_example(path, examples_root, artifact_type)
        examples.append(asdict(example))

    primary = [item for item in examples if item["priority"] == "primary"]
    reference_pool = primary or examples
    required_sections = _most_common_ordered(
        heading
        for item in reference_pool
        for heading in item["section_headings"]
    )

    return {
        "schema_version": 1,
        "generated_at": _timestamp(),
        "source_policy": (
            "Development-only distillation from example_studies. Runtime code must use this "
            "compact cses_wiki file and must not require raw example studies."
        ),
        "selection_policy": (
            "Primary examples are detected by high-quality author metadata such as Katharina "
            "Blinzler or KBK. Other current examples are secondary."
        ),
        "required_micro_syntax_idioms": {
            "header_block": True,
            "open_data_section": True,
            "original_frequencies_log": True,
            "section_markers": "**\\\\\\",
            "variable_markers": "**>>>",
            "tab_missing_after_variable": "tab VARIABLE, mis",
            "explicit_missing_handling": True,
            "label_application": True,
            "check_file_execution": True,
            "save_processed_dataset": True,
        },
        "common_section_order": required_sections[:40],
        "examples": examples[:80],
    }


def _distill_documentation_templates(examples_root: Path) -> dict:
    artifacts = []
    patterns = [
        ("log_file", re.compile(r"log[-_ ]?file|_Log_", re.IGNORECASE)),
        ("election_study_notes", re.compile(r"Election Study Notes|(^|[-_ ])ESN([-_ ]|$)", re.IGNORECASE)),
        ("collaborator_questions", re.compile(r"collaborator.*questions|questions for collaborator", re.IGNORECASE)),
        ("district_data_codebook", re.compile(r"district data.*codebook", re.IGNORECASE)),
        ("macro_report", re.compile(r"macro report|macro data", re.IGNORECASE)),
        ("variable_tracking", re.compile(r"deposited variable|tracking", re.IGNORECASE)),
    ]
    for path in sorted(examples_root.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file() or OLD_PATH_RE.search(str(path)):
            continue
        if path.suffix.lower() not in {".docx", ".txt", ".md", ".qmd", ".xlsx"}:
            continue
        name = path.name
        artifact_type = ""
        for candidate, pattern in patterns:
            if pattern.search(name) or pattern.search(str(path.parent)):
                artifact_type = candidate
                break
        if not artifact_type:
            continue
        artifacts.append(
            {
                "path": _relative(path, examples_root),
                "artifact_type": artifact_type,
                "extension": path.suffix.lower(),
                "size_bytes": path.stat().st_size,
            }
        )

    return {
        "schema_version": 1,
        "generated_at": _timestamp(),
        "required_documentation_artifacts": [
            "processing_log",
            "questions_for_collaborator",
            "things_to_do_before_release",
            "election_study_notes",
            "study_design_and_weights",
            "party_leader_appendix_when_applicable",
        ],
        "log_required_sections": [
            "Log File Instructions",
            "Log File Notes",
            "Questions for Collaborator",
            "Things To Do Before Releasing the Data",
            "Election Study Notes and Appendices",
            "Election Summary",
            "Overview of Study Design and Weights",
            "Parties and Leaders",
        ],
        "collaborator_question_rules": [
            "Group related issues instead of producing long lists of individual failure questions.",
            "Include context and the processor's suggested resolution when possible.",
            "Ask only about genuinely ambiguous or missing information.",
        ],
        "examples": artifacts[:120],
    }


def _build_validation_checks(syntax: dict) -> dict:
    return {
        "schema_version": 1,
        "generated_at": _timestamp(),
        "soft_gate_default": True,
        "runtime_dependency_policy": {
            "required": ["cses_wiki"],
            "not_required": ["example_studies"],
        },
        "stata_syntax_checks": [
            {"id": "header_block", "description": "Micro syntax begins with a CSES processing header."},
            {"id": "section_markers", "description": "Major sections use the CSES section marker style."},
            {"id": "variable_headers", "description": "Processed variables use **>>> F-code headings."},
            {"id": "tab_missing", "description": "Each generated target variable is verified with tab ..., mis."},
            {"id": "missing_values", "description": "Missing values are recoded explicitly to CSES codes."},
            {"id": "labels", "description": "Variable and value labels are applied through label files."},
            {"id": "check_files", "description": "Validation, inconsistency, and theoretical checks are run."},
            {"id": "clean_stata_run", "description": "The final do-file runs cleanly in Stata."},
        ],
        "documentation_checks": [
            {"id": "log_sections", "description": "Processing log contains all required CSES sections."},
            {"id": "study_design_sources", "description": "Study design and weights claims cite deposited sources."},
            {"id": "eligibility_decision", "description": "Eligibility decision records item coverage, sample design, and sample size."},
            {"id": "collaborator_questions", "description": "Collaborator questions are focused, grouped, and contextualized."},
            {"id": "release_todos", "description": "Pre-release TODOs are empty or explicitly resolved before final deposit."},
        ],
        "common_section_order": syntax.get("common_section_order", []),
    }


def _distill_module6_schema(syntax: dict) -> dict:
    from src.standards.schema import build_schema_payload, infer_schema_variable

    ordered: dict[str, str] = {}
    for example in syntax.get("examples", []):
        for item in example.get("variable_heading_examples", []):
            name = str(item.get("variable", "")).strip().upper()
            if name and name not in ordered:
                ordered[name] = str(item.get("description", "")).strip()
        for item in example.get("variable_code_examples", []):
            name = str(item.get("variable", "")).strip().upper()
            if name and name not in ordered:
                ordered[name] = str(item.get("description", "")).strip()
    variables = [
        infer_schema_variable(name, description, order=index + 1)
        for index, (name, description) in enumerate(ordered.items())
        if name.startswith("F")
    ]
    return build_schema_payload(variables)


def _build_replication_benchmarks() -> dict:
    return {
        "schema_version": 1,
        "source_policy": (
            "Development benchmark metadata only. These benchmark profiles must not "
            "introduce runtime country-specific processing logic."
        ),
        "profiles": {
            "email_only": {
                "purpose": "Diagnostic benchmark from e-mail deposit only.",
                "must_not_claim_final_success_when_required_inputs_are_absent": True,
                "required_reports": ["missing_input_report", "replication_scorecard"],
            },
            "full_reference_inputs": {
                "purpose": "Full functional replication benchmark from complete input materials.",
                "acceptance": {
                    "all_workflow_steps_completed": True,
                    "stata_clean": True,
                    "schema_variable_coverage": 1.0,
                    "row_count_match": True,
                    "documentation_valid": True,
                    "no_unresolved_todos": True,
                },
            },
        },
    }


def _read_syntax_example(path: Path, examples_root: Path, artifact_type: str) -> SyntaxExample:
    text = path.read_text(encoding="utf-8", errors="replace")
    header = text[:3000]
    author = _match(AUTHOR_RE, header)
    date = _match(DATE_RE, header)
    module_context = _match(MODULE_RE, header)
    priority = "primary" if _is_primary_author(author) else "secondary"
    headings = _unique_preserving_order(h.strip() for h in SECTION_RE.findall(text))
    variable_examples = [
        {"variable": match.group(1).strip(), "description": match.group(2).strip()}
        for match in VARIABLE_HEADING_RE.finditer(text)
    ][:30]
    variable_code_examples = _variable_code_examples(text)

    return SyntaxExample(
        path=_relative(path, examples_root),
        artifact_type=artifact_type,
        author=author,
        date=date,
        module_context=module_context,
        priority=priority,
        size_bytes=path.stat().st_size,
        header_excerpt=_compact_excerpt(header, 1400),
        section_headings=headings[:60],
        variable_heading_examples=variable_examples,
        variable_code_examples=variable_code_examples,
        observed_idioms={
            "has_original_frequencies": "FREQUENCIES OF ORIGINAL DATA" in text.upper(),
            "has_tab_missing": bool(re.search(r"\btab\s+\S+\s*,\s*mis", text)),
            "has_label_values": "label values" in text.lower(),
            "has_label_define": "label define" in text.lower(),
            "has_check_file_calls": bool(re.search(r"\bdo\s+.*check", text, re.IGNORECASE)),
            "has_save_command": bool(re.search(r"\bsave\s+", text, re.IGNORECASE)),
            "has_log_close": "log close" in text.lower(),
        },
    )


def _variable_code_examples(text: str) -> list[dict[str, str]]:
    matches = list(VARIABLE_HEADING_RE.finditer(text))
    examples = []
    for index, match in enumerate(matches[:12]):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else min(len(text), start + 2000)
        section = text[start:end]
        examples.append(
            {
                "variable": match.group(1).strip(),
                "description": match.group(2).strip(),
                "pattern_type": _classify_variable_pattern(section),
                "code_excerpt": _compact_excerpt(section, 1200),
            }
        )
    return examples


def _classify_variable_pattern(section: str) -> str:
    if re.search(r"\brecode\s+", section):
        return "recode"
    if re.search(r"\breplace\s+", section):
        return "replace"
    if re.search(r"\bgen\s+str", section):
        return "string"
    if re.search(r"\bgen\s+", section):
        return "gen"
    return "other"


def _classify_do_file(path: Path) -> str:
    name = path.name.lower()
    parent = str(path.parent).lower()
    if "labels" in parent or "label" in name:
        return "label_do"
    if "check" in parent or "check" in name or "validation" in name:
        return "check_do"
    if name.startswith("cses-m6_micro") and "label" not in name:
        return "micro_syntax"
    if "generating macro" in name or "construction of micro-macro" in name:
        return "macro_syntax"
    return "other_do"


def _is_primary_author(author: str) -> bool:
    normalized = author.lower()
    return "katharina" in normalized or "kbk" in normalized


def _match(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return " ".join(match.group(1).split()) if match else ""


def _unique_preserving_order(items) -> list[str]:
    seen = set()
    result = []
    for item in items:
        normalized = " ".join(str(item).split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _most_common_ordered(items) -> list[str]:
    counts = {}
    first_seen = {}
    for index, item in enumerate(items):
        value = " ".join(str(item).split())
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
        first_seen.setdefault(value, index)
    return sorted(counts, key=lambda value: (-counts[value], first_seen[value]))


def _compact_excerpt(text: str, max_chars: int) -> str:
    compact = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    compact = _sanitize_excerpt(compact)
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _sanitize_excerpt(text: str) -> str:
    text = re.sub(r"[A-Za-z]:\\[^\n\"]+", "<LOCAL_PATH>", text)
    text = re.sub(r"~/[^\n\"]+", "<LOCAL_PATH>", text)
    text = re.sub(r"/Users/[^\n\"]+", "<LOCAL_PATH>", text)
    return text


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _timestamp() -> str:
    return "deterministic"


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
