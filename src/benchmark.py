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
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.row_count_match and not self.missing_variables and not self.errors


class DatasetComparator:
    """Compare generated and reference Stata datasets functionally."""

    def compare(self, generated_path: Path, reference_path: Path) -> DatasetComparison:
        result = DatasetComparison(reference_path=str(reference_path), generated_path=str(generated_path))
        try:
            import pyreadstat

            gen_df, gen_meta = pyreadstat.read_dta(str(generated_path), metadataonly=True)
            ref_df, ref_meta = pyreadstat.read_dta(str(reference_path), metadataonly=True)
            gen_vars = list(gen_meta.column_names)
            ref_vars = list(ref_meta.column_names)
            result.generated_rows = gen_df.shape[0]
            result.reference_rows = ref_df.shape[0]
            result.row_count_match = result.generated_rows == result.reference_rows
            result.generated_variables = len(gen_vars)
            result.reference_variables = len(ref_vars)
            result.missing_variables = [name for name in ref_vars if name not in gen_vars]
            result.extra_variables = [name for name in gen_vars if name not in ref_vars]
        except Exception as exc:
            result.errors.append(str(exc))
        return result


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
        generated_datasets = sorted(micro.glob("cses-m6_micro_*.dta"), key=lambda path: path.stat().st_mtime)
        dataset_comparison = None
        if reference_dataset and reference_dataset.exists() and generated_datasets:
            dataset_comparison = DatasetComparator().compare(generated_datasets[-1], reference_dataset)
            dataset_path = self.working_dir / ".cses" / "dataset_comparison.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(json.dumps(asdict(dataset_comparison), indent=2, ensure_ascii=False), encoding="utf-8")

        documentation = DocumentationComparator().compare(self.working_dir, reference_dir)
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
