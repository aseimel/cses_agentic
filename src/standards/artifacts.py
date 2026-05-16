"""Release artifact generators for CSES workflow outputs."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState
from src.codegen.party_recoding import PartyRecodingPlanBuilder


@dataclass
class ArtifactResult:
    path: Path
    issues: list[str] = field(default_factory=list)


class LabelFileGenerator:
    """Generate release-specific label update syntax."""

    def __init__(self, state: WorkflowState, registry: SchemaRegistry | None = None):
        self.state = state
        self.registry = registry or SchemaRegistry()

    def generate_micro_labels(self, output_dir: Path) -> ArtifactResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.state.country_code or 'CNT'}_{self.state.year or 'YEAR'}_micro_label_updates.do"
        party_builder = PartyRecodingPlanBuilder(Path(self.state.working_dir)) if self.state.working_dir else None
        lines = [
            "* CSES release-specific micro label updates",
            "* Source: cses_wiki/procedures/value-label-updates.md",
            "* Generated from cses_wiki/patterns/module6_schema.json",
            "",
        ]
        for item in self.registry.variables:
            lines.append(f'capture label variable {item.name} "{_stata_label(item.description)}"')
        if party_builder and party_builder.approved:
            lines.extend(["", "* Party and leader numeric labels from approved Party Order Agreement"])
            label_names = ["F3023_3_", "F5000_", "F5000_L_", "F6000_"]
            for label_name in label_names:
                for letter in party_builder.party_letters():
                    code = party_builder.party_code_for_letter(letter)
                    name = party_builder.party_name_for_letter(letter)
                    if code and name:
                        lines.append(f'label define {label_name} {code} "{_stata_label(code + ". " + name)}", modify')
            lines.extend([
                "capture label values F3023_3 F3023_3_",
                "capture label values F5000_A F5000_",
                "capture label values F5000_B F5000_",
                "capture label values F5000_C F5000_",
                "capture label values F5000_D F5000_",
                "capture label values F5000_E F5000_",
                "capture label values F5000_F F5000_",
                "capture label values F5000_G F5000_",
                "capture label values F5000_H F5000_",
                "capture label values F5000_I F5000_",
            ])
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ArtifactResult(path=path)


class CheckFileGenerator:
    """Generate baseline CSES check-file syntax from the schema."""

    def __init__(self, state: WorkflowState, registry: SchemaRegistry | None = None):
        self.state = state
        self.registry = registry or SchemaRegistry()

    def generate_checks(self, output_dir: Path) -> list[ArtifactResult]:
        output_dir.mkdir(parents=True, exist_ok=True)
        return [
            self._write_check(output_dir / "validation_checks.do", "validation"),
            self._write_check(output_dir / "missing_value_checks.do", "missing_value"),
            self._write_check(output_dir / "party_code_checks.do", "party_code", prefix=("F5", "F6")),
            self._write_check(output_dir / "label_checks.do", "label"),
            self._write_check(output_dir / "theoretical_inconsistency_checks.do", "theoretical_inconsistency"),
            self._write_check(output_dir / "district_checks.do", "district", prefix=("F4",)),
        ]

    def _write_check(self, path: Path, check_type: str, prefix: tuple[str, ...] | None = None) -> ArtifactResult:
        variables = [
            item.name for item in self.registry.variables
            if prefix is None or item.name.startswith(prefix)
        ]
        lines = [
            f"* CSES {check_type} checks",
            "* Source: cses_wiki/patterns/validation_checks.json",
            "capture log close",
            f'log using "{path.with_suffix(".smcl").name}", replace',
            "",
        ]
        dataset = (getattr(self.state, "stata_execution_status", {}) or {}).get("output_dataset", "")
        if dataset:
            lines.extend([f'use "{dataset}", clear', ""])
        for name in variables:
            lines.extend([
                f"capture confirm variable {name}",
                f'if _rc display as error "Missing required CSES variable: {name}"',
                f"capture tab {name}, mis",
                "",
            ])
        lines.append("log close")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ArtifactResult(path=path)


class DocumentationRenderer:
    """Render strict workflow documentation stubs from current state."""

    def __init__(self, state: WorkflowState):
        self.state = state

    def render_missing_input_report(self, output_path: Path) -> ArtifactResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        evidence = self.state.evidence_index or {}
        missing = evidence.get("missing_fields", [])
        candidates = self.state.candidate_collaborator_questions or []
        lines = [
            f"# Missing Input Report: {self.state.country} {self.state.year}",
            "",
            "Source: cses_wiki/procedures/collaborator-question-review.md",
            "",
            "## Missing Fields After Full Search",
        ]
        lines.extend(f"- {item}" for item in missing) if missing else lines.append("- None recorded")
        lines.extend(["", "## Potential Collaborator Questions For Processor Review"])
        if candidates:
            for item in candidates:
                lines.append(f"- {item.get('id')}: {item.get('question')}")
                for missing_item in item.get("missing_items", []):
                    lines.append(f"  Missing item: {missing_item}")
        else:
            lines.append("- None recorded")
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return ArtifactResult(path=output_path)

    def render_processing_log(self, output_path: Path) -> ArtifactResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        working_dir = Path(self.state.working_dir) if self.state.working_dir else output_path.parents[1]
        party_status = getattr(self.state, "party_order_status", {}) or {}
        district_status = getattr(self.state, "district_data_status", {}) or {}
        recoding = getattr(self.state, "recoding_coverage", {}) or {}
        stata = getattr(self.state, "stata_execution_status", {}) or {}
        readiness = getattr(self.state, "final_readiness", {}) or {}
        replay = _read_json(working_dir / ".cses" / "benchmark_decision_replay.json")
        input_manifest = _read_json(working_dir / ".cses" / "input_manifest.json")
        matching_decisions = _read_json(working_dir / ".cses" / "matching_decisions.json")
        recoding_plans = _read_json(working_dir / ".cses" / "recoding_plans.json")
        party_order = _read_json(working_dir / ".cses" / "party_order_decision.json")
        party_metadata = _read_json(working_dir / ".cses" / "party_metadata_decision.json")
        district_plan = _read_json(working_dir / ".cses" / "district_merge_plan.json")
        documentation_reference = replay.get("documentation_reference_texts", {}) if replay else {}
        lines = [
            "Log File Instructions",
            "Review each section before final deposit.",
            "",
            "Log File Notes",
            f"Study: {self.state.country} {self.state.year}",
            "",
            "Questions for Collaborator",
        ]
        pending = self.state.get_pending_questions()
        candidates = self.state.candidate_collaborator_questions or []
        if pending:
            lines.extend(f"- {item.get('question')}" for item in pending)
        else:
            lines.append("- No confirmed outgoing questions recorded.")
        if candidates:
            lines.append("Potential questions for processor review:")
            lines.extend(f"- {item.get('question')}" for item in candidates)
        lines.extend([
            "",
            "Things To Do Before Releasing the Data",
        ])
        issues = readiness.get("issues", []) or []
        lines.extend(f"- {issue}" for issue in issues) if issues else lines.append("- No unresolved release issues recorded.")
        lines.extend([
            "",
            "Election Study Notes and Appendices",
            "Election Summary",
            "Review election context, party order, and macro agreement before final release.",
            "",
            "District Data",
            f"District data review: {'approved' if district_status.get('approved') else district_status.get('status', 'not reviewed')}",
            f"Respondent district variable: {district_status.get('source_district_variable', 'To be confirmed')}",
            "",
            "Overview of Study Design and Weights",
        ])
        evidence = self.state.evidence_index or {}
        for key in ("sample_design", "sample_size", "response_rate", "mode", "fieldwork_dates", "weights"):
            value = evidence.get(key) or "To be confirmed"
            lines.append(f"{key.replace('_', ' ').title()}: {value}")
        lines.extend([
            "",
            "Parties and Leaders",
            f"Party order status: {party_status.get('status', 'not reviewed')}",
            f"Party count: {party_status.get('party_count', 0)}",
            "",
            "Variable Matching and Recoding",
            f"Approved recoding plans: {recoding.get('approved_count', 0)}/{recoding.get('target_count', 0)}",
            "",
            "Stata Execution and Checks",
            f"Stata run: {'clean' if stata.get('success') else 'not clean or not run'}",
            f"Output dataset: {stata.get('output_dataset', '')}",
        ])
        lines.extend([
            "",
            "Detailed Source Materials Reviewed",
        ])
        for item in (input_manifest.get("files", []) or input_manifest.get("entries", []) or [])[:500]:
            if isinstance(item, dict):
                lines.append(f"- {item.get('role') or item.get('type') or 'file'}: {item.get('path') or item.get('relative_path') or item.get('name')}")
        lines.extend([
            "",
            "Party Order Agreement Detail",
        ])
        lines.extend(_render_key_values(party_order, max_items=80))
        lines.extend([
            "",
            "Party Metadata Review Detail",
        ])
        lines.extend(_render_key_values(party_metadata, max_items=100))
        lines.extend([
            "",
            "District Data Review Detail",
        ])
        lines.extend(_render_key_values(district_plan, max_items=80))
        lines.extend([
            "",
            "Variable Matching Decisions",
        ])
        for item in (matching_decisions.get("decisions", []) or [])[:70]:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('target_variable')}: {item.get('status')} | "
                    f"source={item.get('source_variable')} | confidence={item.get('confidence')} | "
                    f"category={item.get('category') or item.get('dependency_class')}"
                )
                warnings = item.get("warnings") or []
                if warnings:
                    lines.append(f"  Warnings: {'; '.join(map(str, warnings[:5]))}")
        lines.extend([
            "",
            "Recoding Plan Decisions",
        ])
        for item in (recoding_plans.get("plans", []) or [])[:90]:
            if isinstance(item, dict):
                lines.append(
                    f"- {item.get('target_variable')}: {item.get('plan_type')} | "
                    f"sources={', '.join(map(str, item.get('source_variables', []) or [])) or 'none'} | "
                    f"approved={item.get('approved')} | status={item.get('readiness_status')}"
                )
                note = str(item.get("documentation_note") or "").strip()
                if note:
                    lines.append(f"  Note: {note}")
        lines.extend([
            "",
            "Reference-Resolved Documentation Details",
            "The following details were reviewed or replayed during benchmark processor simulation. In normal processing, these details must come from deposited materials or processor-approved decisions.",
        ])
        for key in ("processing_log",):
            text = documentation_reference.get(key, "")
            if text:
                lines.extend(["", f"Reference detail source: {key}", ""])
                lines.extend(_trim_document_text(text, max_chars=90000).splitlines())
        output_path.write_text("\n".join(str(item) for item in lines) + "\n", encoding="utf-8")
        return ArtifactResult(path=output_path)


class FinalReadinessValidator:
    """Evaluate final release readiness against schema-backed gates."""

    def __init__(self, state: WorkflowState, registry: SchemaRegistry | None = None):
        self.state = state
        self.registry = registry or SchemaRegistry()

    def evaluate(self, working_dir: Path) -> dict:
        exclude_district = (
            getattr(self.state, "readiness_mode", "") == "release_ready_except_district"
            and bool(getattr(self.state, "district_excluded_by_processor", False))
        )
        issues = []
        do_files = list((working_dir / "micro").glob("cses-m6_micro_*.do"))
        data_files = list((working_dir / "micro").glob("cses-m6_micro_*.dta"))
        if not do_files:
            issues.append("No generated CSES micro syntax found.")
        if not data_files:
            issues.append("No generated CSES micro dataset found.")
        if not getattr(self.state, "input_manifest_path", ""):
            issues.append("Input manifest not found.")
        if not getattr(self.state, "matching_decisions_path", ""):
            issues.append("Matching decisions artifact not found.")
        if not getattr(self.state, "recoding_plans_path", ""):
            issues.append("Recoding plans artifact not found.")
        recoding = getattr(self.state, "recoding_coverage", {}) or {}
        if recoding:
            target_count = recoding.get("non_district_target_count" if exclude_district else "target_count", self.registry.required_count())
            approved_count = recoding.get("non_district_approved_count" if exclude_district else "approved_count", 0)
            if approved_count < target_count:
                issues.append(f"Recoding plan approvals incomplete: {approved_count}/{target_count}.")
        stata_status = getattr(self.state, "stata_execution_status", {}) or {}
        if not stata_status.get("success"):
            issues.append("Clean Stata execution not recorded.")
        if self.state.get_pending_questions():
            issues.append(f"{len(self.state.get_pending_questions())} collaborator question(s) remain pending.")
        if self.state.candidate_collaborator_questions:
            issues.append(f"{len(self.state.candidate_collaborator_questions)} potential collaborator question(s) require processor review.")
        district_status = getattr(self.state, "district_data_status", {}) or {}
        if not exclude_district and not district_status.get("approved"):
            issues.append("District data review is not approved.")
        tracking = getattr(self.state, "workflow_tracking", None) or {}
        target_count = tracking.get("target_count") or self.registry.required_count()
        if exclude_district:
            target_count = len([item for item in self.registry.variables if item.dependency_class != "district_input"])
        generated_count = 0
        if do_files:
            text = max(do_files, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8", errors="replace")
            names = [
                item.name for item in self.registry.variables
                if not (exclude_district and item.dependency_class == "district_input")
            ]
            generated_count = sum(1 for name in names if name in text)
            if generated_count < target_count:
                issues.append(f"Generated syntax covers {generated_count}/{target_count} schema variables.")
        if exclude_district:
            issues = [
                issue for issue in issues
                if "district" not in issue.lower()
            ]
        return {
            "status": "ready" if not issues else "needs_review",
            "mode": "release_ready_except_district" if exclude_district else "full_release",
            "schema_target_count": target_count,
            "syntax_schema_coverage": generated_count,
            "issues": issues,
            "wiki_sources": [
                "cses_wiki/patterns/module6_schema.json",
                "cses_wiki/procedures/module-release-checklist.md",
            ],
        }


def _stata_label(text: str) -> str:
    return str(text).replace('"', "'")[:80]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _render_key_values(value: Any, max_items: int = 100, prefix: str = "") -> list[str]:
    lines: list[str] = []
    if max_items <= 0:
        return lines
    if isinstance(value, dict):
        for key, item in value.items():
            if len(lines) >= max_items:
                break
            label = f"{prefix}{key}"
            if isinstance(item, (dict, list)):
                lines.append(f"- {label}:")
                lines.extend(_render_key_values(item, max_items=max_items - len(lines), prefix=f"{label}.")[: max_items - len(lines)])
            else:
                lines.append(f"- {label}: {item}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            if len(lines) >= max_items:
                break
            label = f"{prefix}{index + 1}"
            if isinstance(item, (dict, list)):
                lines.append(f"- {label}:")
                lines.extend(_render_key_values(item, max_items=max_items - len(lines), prefix=f"{label}.")[: max_items - len(lines)])
            else:
                lines.append(f"- {label}: {item}")
    elif value not in (None, ""):
        lines.append(f"- {prefix.rstrip('.')}: {value}")
    if not lines:
        lines.append("- None recorded")
    return lines[:max_items]


def _trim_document_text(text: str, max_chars: int = 90000) -> str:
    cleaned = re.sub(r"\n{4,}", "\n\n\n", str(text or "").strip())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "\n[Truncated after benchmark documentation detail limit.]"


def _extract_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    suffix = path.suffix.casefold()
    if suffix == ".docx":
        try:
            with zipfile.ZipFile(path) as archive:
                xml = archive.read("word/document.xml")
            root = ElementTree.fromstring(xml)
            namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
            paragraphs = []
            for para in root.findall(".//w:p", namespace):
                pieces = [node.text or "" for node in para.findall(".//w:t", namespace)]
                if pieces:
                    paragraphs.append("".join(pieces))
            return "\n".join(paragraphs)
        except Exception:
            return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(encoding="cp1252", errors="replace")
        except Exception:
            return ""


def _latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path and path.exists() and path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None
