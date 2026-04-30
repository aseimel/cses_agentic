"""Release artifact generators for CSES workflow outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState


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
        lines = [
            "* CSES release-specific micro label updates",
            "* Source: cses_wiki/procedures/value-label-updates.md",
            "* Generated from cses_wiki/patterns/module6_schema.json",
            "",
        ]
        for item in self.registry.variables:
            lines.append(f'capture label variable {item.name} "{_stata_label(item.description)}"')
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


class FinalReadinessValidator:
    """Evaluate final release readiness against schema-backed gates."""

    def __init__(self, state: WorkflowState, registry: SchemaRegistry | None = None):
        self.state = state
        self.registry = registry or SchemaRegistry()

    def evaluate(self, working_dir: Path) -> dict:
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
            target_count = recoding.get("target_count", self.registry.required_count())
            approved_count = recoding.get("approved_count", 0)
            if approved_count < target_count:
                issues.append(f"Recoding plan approvals incomplete: {approved_count}/{target_count}.")
        stata_status = getattr(self.state, "stata_execution_status", {}) or {}
        if not stata_status.get("success"):
            issues.append("Clean Stata execution not recorded.")
        if self.state.get_pending_questions():
            issues.append(f"{len(self.state.get_pending_questions())} collaborator question(s) remain pending.")
        if self.state.candidate_collaborator_questions:
            issues.append(f"{len(self.state.candidate_collaborator_questions)} potential collaborator question(s) require processor review.")
        tracking = getattr(self.state, "workflow_tracking", None) or {}
        target_count = tracking.get("target_count") or self.registry.required_count()
        generated_count = 0
        if do_files:
            text = max(do_files, key=lambda p: p.stat().st_mtime).read_text(encoding="utf-8", errors="replace")
            generated_count = sum(1 for name in self.registry.ordered_names() if name in text)
            if generated_count < target_count:
                issues.append(f"Generated syntax covers {generated_count}/{target_count} schema variables.")
        return {
            "status": "ready" if not issues else "needs_review",
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
