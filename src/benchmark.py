"""Replication benchmark harness for CSES workflow development."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class DatasetComparison:
    reference_path: str
    generated_path: str
    row_count_match: bool = False
    reference_rows: int | None = None
    generated_rows: int | None = None
    reference_variables: int = 0
    generated_variables: int = 0
    missing_variables: list[str] = field(default_factory=list)
    extra_variables: list[str] = field(default_factory=list)
    order_mismatches: list[str] = field(default_factory=list)
    label_mismatches: list[str] = field(default_factory=list)
    value_label_mismatches: list[str] = field(default_factory=list)
    frequency_mismatches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.row_count_match
            and not self.missing_variables
            and not self.errors
            and not self.frequency_mismatches
        )


class DatasetComparator:
    """Compare generated and reference Stata datasets functionally."""

    def compare(self, generated_path: Path, reference_path: Path, exclude_prefixes: tuple[str, ...] = ()) -> DatasetComparison:
        result = DatasetComparison(reference_path=str(reference_path), generated_path=str(generated_path))
        try:
            import pyreadstat

            gen_df, gen_meta = pyreadstat.read_dta(str(generated_path), metadataonly=True)
            ref_df, ref_meta = pyreadstat.read_dta(str(reference_path), metadataonly=True)
            gen_vars = [name for name in gen_meta.column_names if not name.startswith(exclude_prefixes)]
            ref_vars = [name for name in ref_meta.column_names if not name.startswith(exclude_prefixes)]
            result.generated_rows = gen_df.shape[0]
            result.reference_rows = ref_df.shape[0]
            result.row_count_match = result.generated_rows == result.reference_rows
            result.generated_variables = len(gen_vars)
            result.reference_variables = len(ref_vars)
            result.missing_variables = [name for name in ref_vars if name not in gen_vars]
            result.extra_variables = [name for name in gen_vars if name not in ref_vars]
            result.order_mismatches = [
                f"{index + 1}: generated {g} vs reference {r}"
                for index, (g, r) in enumerate(zip(gen_vars, ref_vars))
                if g != r
            ][:50]
            result.label_mismatches = self._label_mismatches(gen_meta, ref_meta, set(gen_vars) & set(ref_vars))
            result.value_label_mismatches = self._value_label_mismatches(gen_meta, ref_meta, set(gen_vars) & set(ref_vars))
            result.frequency_mismatches = self._frequency_mismatches(generated_path, reference_path, set(gen_vars) & set(ref_vars))
        except Exception as exc:
            result.errors.append(str(exc))
        return result

    def _label_mismatches(self, gen_meta: Any, ref_meta: Any, variables: set[str]) -> list[str]:
        gen_labels = dict(zip(gen_meta.column_names, gen_meta.column_labels or []))
        ref_labels = dict(zip(ref_meta.column_names, ref_meta.column_labels or []))
        mismatches = []
        for name in sorted(variables):
            if str(gen_labels.get(name, "")).strip() != str(ref_labels.get(name, "")).strip():
                mismatches.append(name)
        return mismatches[:100]

    def _value_label_mismatches(self, gen_meta: Any, ref_meta: Any, variables: set[str]) -> list[str]:
        mismatches = []
        gen_map = getattr(gen_meta, "variable_to_label", {}) or {}
        ref_map = getattr(ref_meta, "variable_to_label", {}) or {}
        for name in sorted(variables):
            if str(gen_map.get(name, "")).strip() != str(ref_map.get(name, "")).strip():
                mismatches.append(name)
        return mismatches[:100]

    def _frequency_mismatches(self, generated_path: Path, reference_path: Path, variables: set[str]) -> list[dict[str, Any]]:
        key_patterns = ("F100", "F200", "F300", "F301", "F302", "F500", "F600")
        key_vars = [name for name in sorted(variables) if name.startswith(key_patterns)][:80]
        if not key_vars:
            return []
        try:
            import pyreadstat
            gen_df, _ = pyreadstat.read_dta(str(generated_path), usecols=key_vars, apply_value_formats=False)
            ref_df, _ = pyreadstat.read_dta(str(reference_path), usecols=key_vars, apply_value_formats=False)
        except Exception:
            return []
        mismatches = []
        for name in key_vars:
            gen_counts = gen_df[name].value_counts(dropna=False).sort_index().to_dict()
            ref_counts = ref_df[name].value_counts(dropna=False).sort_index().to_dict()
            if gen_counts != ref_counts:
                mismatches.append({"variable": name, "generated": _plain_counts(gen_counts), "reference": _plain_counts(ref_counts)})
        return mismatches[:50]


class DocumentationComparator:
    """Minimal documentation artifact comparator."""

    def compare(self, working_dir: Path, reference_dir: Path | None = None) -> dict[str, Any]:
        micro = Path(working_dir) / "micro"
        checks = {
            "processing_log": bool(list(micro.glob("*log*.docx")) or list(micro.glob("*log*.qmd"))),
            "collaborator_questions": bool(list(micro.glob("**/*question*.docx")) or list(micro.glob("**/*question*.txt"))),
            "esn": bool(list(micro.glob("Documentation/*ESN*")) or list(Path(working_dir).glob("macro/*ESN*"))),
            "label_files": bool(list((micro / "labels").glob("*.do"))) if (micro / "labels").exists() else False,
            "check_files": bool(list((micro / "data_checks").glob("*.do"))) if (micro / "data_checks").exists() else False,
            "check_outputs": bool(list((micro / "data_checks").glob("*.log")) or list((micro / "data_checks").glob("*.smcl"))) if (micro / "data_checks").exists() else False,
        }
        return {
            "ok": all(checks.values()),
            "checks": checks,
            "reference_dir": str(reference_dir) if reference_dir else "",
        }


class ReplicationBenchmarkRunner:
    """Run benchmark checks against workflow artifacts without hardcoded runtime logic."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)

    def scorecard(
        self,
        profile: str,
        reference_dataset: Path | None = None,
        reference_dir: Path | None = None,
    ) -> dict[str, Any]:
        micro = self.working_dir / "micro"
        exclude_district = profile in {"release_ready_except_district", "non_district", "full_reference_inputs_except_district"}
        exclude_prefixes = ("F4",) if exclude_district else ()
        generated_datasets = sorted(micro.glob("cses-m6_micro_*.dta"), key=lambda path: path.stat().st_mtime)
        dataset_comparison = None
        if reference_dataset and reference_dataset.exists() and generated_datasets:
            dataset_comparison = DatasetComparator().compare(generated_datasets[-1], reference_dataset, exclude_prefixes=exclude_prefixes)
            dataset_path = self.working_dir / ".cses" / "dataset_comparison.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(json.dumps(asdict(dataset_comparison), indent=2, ensure_ascii=False), encoding="utf-8")

        documentation = DocumentationComparator().compare(self.working_dir, reference_dir)
        syntax = self._syntax_comparison(reference_dir, exclude_district=exclude_district)
        matching_metrics = self._matching_metrics(reference_dataset)
        artifacts = {
            "input_manifest": (self.working_dir / ".cses" / "input_manifest.json").exists(),
            "study_kb": (self.working_dir / ".cses" / "study_kb.json").exists(),
            "matching_decisions": (self.working_dir / ".cses" / "matching_decisions.json").exists(),
            "recoding_plans": (self.working_dir / ".cses" / "recoding_plans.json").exists(),
            "stata_execution": (self.working_dir / ".cses" / "stata_execution.json").exists(),
            "final_readiness": bool(list(micro.glob("*final_readiness_report.md"))),
        }
        issues = []
        for key, ok in artifacts.items():
            if not ok:
                issues.append(f"Missing benchmark artifact: {key}")
        if dataset_comparison and not dataset_comparison.ok:
            issues.append("Generated dataset does not functionally match reference dataset.")
        if not documentation["ok"]:
            issues.append("Documentation/check/label artifact set is incomplete.")
        if not syntax.get("ok", False):
            issues.append("Generated syntax does not yet match the required structure/reference checks.")
        if matching_metrics.get("high_confidence_without_canonical_evidence", 0):
            issues.append("High-confidence matching includes targets without canonical item evidence.")

        scorecard = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "profile": profile,
            "working_dir": str(self.working_dir),
            "status": "pass" if not issues else "needs_review",
            "artifacts": artifacts,
            "dataset_comparison": asdict(dataset_comparison) if dataset_comparison else None,
            "matching_metrics": matching_metrics,
            "documentation": documentation,
            "syntax": syntax,
            "issues": issues,
        }
        path = self.working_dir / ".cses" / "replication_scorecard.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(scorecard, indent=2, ensure_ascii=False), encoding="utf-8")
        return scorecard

    def _matching_metrics(self, reference_dataset: Path | None = None) -> dict[str, Any]:
        decisions_path = self.working_dir / ".cses" / "matching_decisions.json"
        evidence_path = self.working_dir / ".cses" / "matching_evidence.json"
        if not decisions_path.exists():
            return {}
        try:
            decisions_payload = json.loads(decisions_path.read_text(encoding="utf-8"))
            evidence = json.loads(evidence_path.read_text(encoding="utf-8")) if evidence_path.exists() else {}
            from src.matching.decision_engine import MatchingDecision, SourceCandidate, matching_category_summary

            decisions = []
            for item in decisions_payload.get("decisions", []):
                candidates = [SourceCandidate(**candidate) for candidate in item.get("candidates", [])]
                item = {**item, "candidates": candidates}
                decisions.append(MatchingDecision(**item))
            category_summary = matching_category_summary(decisions, evidence)
            reference_variables = self._reference_variables(reference_dataset)
            target_profiles = {
                item.get("name"): item
                for item in evidence.get("target_variable_profiles", []) or []
                if isinstance(item, dict)
            }
            direct = [decision for decision in decisions if decision.dependency_class == "direct_survey_item"]
            matched = [
                decision for decision in direct
                if decision.status == "proposed_match" and decision.source_variable not in {"", "NOT_FOUND", "ERROR"}
            ]
            high_without_canonical = [
                decision.target_variable
                for decision in matched
                if decision.confidence == "high"
                and not (target_profiles.get(decision.target_variable, {}).get("canonical_item_ids"))
                and decision.target_variable.startswith("F3")
            ]
            reference_target_present = [
                decision.target_variable for decision in decisions
                if decision.target_variable in reference_variables
            ] if reference_variables else []
            return {
                "category_summary": category_summary,
                "direct_survey_targets": len(direct),
                "direct_survey_matched": len(matched),
                "reference_targets_present": len(reference_target_present),
                "high_confidence_without_canonical_evidence": len(high_without_canonical),
                "high_confidence_without_canonical_targets": high_without_canonical[:50],
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _reference_variables(self, reference_dataset: Path | None) -> set[str]:
        if not reference_dataset or not reference_dataset.exists():
            return set()
        try:
            import pyreadstat

            _, meta = pyreadstat.read_dta(str(reference_dataset), metadataonly=True)
            return set(meta.column_names)
        except Exception:
            return set()

    def _syntax_comparison(self, reference_dir: Path | None = None, exclude_district: bool = False) -> dict[str, Any]:
        from src.standards.validators import validate_stata_syntax_text

        micro = self.working_dir / "micro"
        generated = sorted(micro.glob("cses-m6_micro_*.do"), key=lambda path: path.stat().st_mtime)
        checks = {"generated_syntax_exists": bool(generated)}
        issues = []
        if generated:
            text = generated[-1].read_text(encoding="utf-8", errors="replace")
            validation = validate_stata_syntax_text(text)
            checks.update(validation.checks)
            issues.extend(validation.issues)
            if exclude_district and "F4001" in text:
                issues.append("district_variables_present_in_non_district_syntax")
        if reference_dir:
            reference_syntax = sorted(Path(reference_dir).glob("**/cses-m6_micro_*.do"))
            checks["reference_syntax_available"] = bool(reference_syntax)
        return {"ok": not issues and bool(generated), "checks": checks, "issues": issues[:50]}


def _plain_counts(counts: dict[Any, Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counts.items()}
