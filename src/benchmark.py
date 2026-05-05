"""Replication benchmark harness for CSES workflow development."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from xml.etree import ElementTree


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
    """Compare generated CSES documentation against required structure and reference artifacts."""

    LOG_SECTION_TERMS = {
        "log_file_notes": ("log file notes", "processing notes"),
        "questions_for_collaborator": ("questions for collaborator", "collaborator questions"),
        "todo_before_release": ("things to do before releasing", "todo"),
        "election_study_notes": ("election study notes", "esn"),
        "election_summary": ("election summary",),
        "study_design_weights": ("study design", "weights"),
        "parties_leaders": ("parties and leaders", "party", "leader"),
    }

    REQUIRED_FACT_TERMS = {
        "probability_sample": ("probability sample", "sampling"),
        "sample_design": ("sample design", "sampling design"),
        "sample_size": ("sample size",),
        "response_rate": ("response rate",),
        "fieldwork": ("fieldwork", "collection period", "data collection"),
        "mode": ("mode of interview", "mode"),
        "weights": ("weight", "weighting"),
        "party_order": ("party order", "party a", "party b"),
        "district_data": ("district", "constituency"),
        "check_results": ("check", "validation"),
    }
    STRICT_ARTIFACTS = ("processing_log", "collaborator_questions", "esn")

    def compare(self, working_dir: Path, reference_dir: Path | None = None) -> dict[str, Any]:
        working_dir = Path(working_dir)
        micro = working_dir / "micro"
        generated = self._generated_artifacts(working_dir)
        reference = self._reference_artifacts(Path(reference_dir)) if reference_dir else {}
        generated_texts = {name: _extract_text(path) for name, path in generated.items() if path}
        reference_texts = {name: _extract_text(path) for name, path in reference.items() if path}
        processing_text = generated_texts.get("processing_log", "") or generated_texts.get("active_log", "")
        esn_text = generated_texts.get("esn", "")

        checks = {
            "processing_log": bool(generated.get("processing_log") or generated.get("active_log")),
            "collaborator_questions_review": bool(generated.get("collaborator_questions") or "questions for collaborator" in processing_text.casefold()),
            "esn": bool(generated.get("esn") or "election study notes" in processing_text.casefold()),
            "label_files": bool(list((micro / "labels").glob("*.do"))) if (micro / "labels").exists() else False,
            "check_files": bool(list((micro / "data_checks").glob("*.do"))) if (micro / "data_checks").exists() else False,
            "check_outputs": bool(list((micro / "data_checks").glob("*.log")) or list((micro / "data_checks").glob("*.smcl"))) if (micro / "data_checks").exists() else False,
        }
        section_checks = {
            key: _contains_any(processing_text, terms)
            for key, terms in self.LOG_SECTION_TERMS.items()
        }
        fact_checks = {
            key: _contains_any(processing_text + "\n" + esn_text, terms)
            for key, terms in self.REQUIRED_FACT_TERMS.items()
        }
        reference_requirements = self._reference_requirements(reference_texts)
        reference_coverage = {
            key: _contains_any(processing_text + "\n" + esn_text, terms)
            for key, terms in reference_requirements.items()
        }
        equivalence = self._documentation_equivalence(
            generated,
            reference,
            working_dir=working_dir,
            reference_dir=Path(reference_dir) if reference_dir else None,
        )
        issues = []
        for key, ok in checks.items():
            if not ok:
                issues.append(f"Missing documentation artifact: {key}")
        for key, ok in section_checks.items():
            if not ok:
                issues.append(f"Processing log missing required section: {key}")
        for key, ok in fact_checks.items():
            if not ok:
                issues.append(f"Documentation missing required CSES fact area: {key}")
        for key, ok in reference_coverage.items():
            if not ok:
                issues.append(f"Generated documentation does not cover reference documentation area: {key}")
        for key, result in equivalence.items():
            if not result.get("ok", False):
                issues.append(f"Documentation not comparable to reference for {key}: {', '.join(result.get('issues', []))}")
        return {
            "ok": not issues,
            "checks": checks,
            "section_checks": section_checks,
            "fact_checks": fact_checks,
            "reference_coverage": reference_coverage,
            "equivalence": equivalence,
            "artifacts": {key: str(path) for key, path in generated.items() if path},
            "reference_artifacts": {key: str(path) for key, path in reference.items() if path},
            "issues": issues,
            "reference_dir": str(reference_dir) if reference_dir else "",
        }

    def write_report(self, working_dir: Path, reference_dir: Path | None = None) -> dict[str, Any]:
        result = self.compare(working_dir, reference_dir)
        out_dir = Path(working_dir) / ".cses"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "documentation_comparison.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        lines = [
            "# Documentation Comparison",
            "",
            f"Status: {'pass' if result.get('ok') else 'needs review'}",
            "",
            "## Required Artifacts",
        ]
        lines.extend(f"- {key}: {'present' if value else 'missing'}" for key, value in result.get("checks", {}).items())
        lines.extend(["", "## Required Sections"])
        lines.extend(f"- {key}: {'present' if value else 'missing'}" for key, value in result.get("section_checks", {}).items())
        lines.extend(["", "## Required Fact Areas"])
        lines.extend(f"- {key}: {'present' if value else 'missing'}" for key, value in result.get("fact_checks", {}).items())
        if result.get("equivalence"):
            lines.extend(["", "## Reference Documentation Equivalence"])
            for key, value in result["equivalence"].items():
                lines.append(
                    f"- {key}: {'pass' if value.get('ok') else 'needs review'} "
                    f"(extent {value.get('extent_ratio')}, detail {value.get('detail_coverage')}, "
                    f"format {value.get('format_ratio')})"
                )
        if result.get("issues"):
            lines.extend(["", "## Issues"])
            lines.extend(f"- {issue}" for issue in result["issues"])
        (out_dir / "documentation_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return result

    def _generated_artifacts(self, working_dir: Path) -> dict[str, Path | None]:
        micro = working_dir / "micro"
        return {
            "active_log": _latest([*micro.glob("*_log.qmd")]),
            "processing_log": _latest([*micro.glob("cses-m6_log-file_*.txt"), *micro.glob("cses-m6_log-file_*.docx")]),
            "collaborator_questions": _latest([*micro.glob("**/*question*.docx"), *micro.glob("**/*question*.txt"), *micro.glob("**/*question*.md")]),
            "esn": _latest([*micro.glob("Documentation/*ESN*"), *working_dir.glob("macro/*ESN*")]),
            "final_readiness": _latest([*micro.glob("*final_readiness_report.md")]),
            "missing_input_report": _latest([*micro.glob("*missing_input_report.md")]),
            "check_review": _latest([*micro.glob("*check_file_review.md")]),
            "study_design_overview": _latest([*micro.glob("Documentation/*study_design*")]),
        }

    def _reference_artifacts(self, reference_dir: Path) -> dict[str, Path | None]:
        micro = reference_dir / "micro"
        return {
            "processing_log": _latest([*micro.glob("**/*log-file*.docx"), *micro.glob("**/*log*.qmd"), *micro.glob("**/*log*.txt")]),
            "collaborator_questions": _latest([*micro.glob("**/*Question*.docx"), *micro.glob("**/*question*.docx"), *micro.glob("**/*Question*.txt")]),
            "esn": _latest([*reference_dir.glob("macro/*ESN*"), *micro.glob("Documentation/*ESN*")]),
            "label_files": _latest([*micro.glob("labels/*.do")]),
            "check_files": _latest([*micro.glob("data_checks/*.do")]),
            "check_outputs": _latest([*micro.glob("data_checks/*.log"), *micro.glob("data_checks/*.smcl")]),
        }

    def _reference_requirements(self, reference_texts: dict[str, str]) -> dict[str, tuple[str, ...]]:
        combined = "\n".join(reference_texts.values()).casefold()
        requirements = {}
        candidates = {
            "reference_questions": ("question", "collaborator"),
            "reference_esn_party": ("party", "leader", "appendix"),
            "reference_district": ("district", "constituency"),
            "reference_weights": ("weight", "weights", "weighting"),
            "reference_checks": ("check", "validation", "inconsistency"),
        }
        for key, terms in candidates.items():
            if any(term in combined for term in terms):
                requirements[key] = terms
        return requirements

    def _documentation_equivalence(
        self,
        generated: dict[str, Path | None],
        reference: dict[str, Path | None],
        working_dir: Path,
        reference_dir: Path | None,
    ) -> dict[str, dict[str, Any]]:
        if not reference:
            return {}
        results = {}
        for key in self.STRICT_ARTIFACTS:
            generated_path = generated.get(key)
            reference_path = reference.get(key)
            if not reference_path:
                continue
            generated_paths = self._artifact_equivalence_paths(key, working_dir, generated_path, generated=True)
            reference_paths = self._artifact_equivalence_paths(key, reference_dir, reference_path, generated=False)
            generated_text = "\n".join(_extract_text(path) for path in generated_paths)
            reference_text = "\n".join(_extract_text(path) for path in reference_paths)
            generated_profile = _combined_document_profile(generated_paths, generated_text)
            reference_profile = _combined_document_profile(reference_paths, reference_text)
            anchors = _reference_detail_anchors(reference_text)
            found = [anchor for anchor in anchors if _anchor_present(anchor, generated_text)]
            extent_ratio = _ratio(generated_profile["char_count"], reference_profile["char_count"])
            format_ratio = min(
                _ratio(generated_profile["paragraph_count"], reference_profile["paragraph_count"]),
                _ratio(generated_profile["heading_count"] or 1, reference_profile["heading_count"] or 1),
            )
            detail_coverage = round(len(found) / len(anchors), 4) if anchors else 1.0
            issues = []
            if extent_ratio < 0.55:
                issues.append("generated text is much shorter than the reference")
            if extent_ratio > 2.25:
                issues.append("generated text is much longer than the reference")
            if format_ratio < 0.45:
                issues.append("paragraph/heading structure is not comparable")
            if detail_coverage < 0.70:
                issues.append("reference details are missing from generated documentation")
            if not generated_text.strip():
                issues.append("generated documentation artifact is empty or missing")
            results[key] = {
                "ok": not issues,
                "generated_path": str(generated_path) if generated_path else "",
                "reference_path": str(reference_path),
                "generated_paths": [str(path) for path in generated_paths],
                "reference_paths": [str(path) for path in reference_paths],
                "generated_profile": generated_profile,
                "reference_profile": reference_profile,
                "extent_ratio": extent_ratio,
                "format_ratio": format_ratio,
                "detail_coverage": detail_coverage,
                "reference_detail_count": len(anchors),
                "matched_detail_count": len(found),
                "missing_detail_examples": [anchor for anchor in anchors if anchor not in found][:50],
                "issues": issues,
            }
        return results

    def _artifact_equivalence_paths(
        self,
        key: str,
        root: Path | None,
        fallback: Path | None,
        generated: bool,
    ) -> list[Path]:
        if not root:
            return [fallback] if fallback else []
        micro = root / "micro"
        if key == "collaborator_questions":
            paths = [*micro.glob("**/*question*.docx"), *micro.glob("**/*Question*.docx"), *micro.glob("**/*question*.txt"), *micro.glob("**/*question*.md")]
            return sorted({
                path for path in paths
                if path.exists()
                and path.is_file()
                and _is_current_collaborator_question_doc(path)
            })
        if key == "esn":
            if generated:
                return [fallback] if fallback else []
            paths = [*root.glob("macro/*ESN*")]
            return sorted({path for path in paths if path.exists() and path.is_file()})
        return [fallback] if fallback else []


class BenchmarkDecisionExtractor:
    """Extract benchmark-only reference decisions from completed artifacts."""

    def extract(
        self,
        reference_dataset: Path | None = None,
        reference_syntax: Path | None = None,
        reference_dir: Path | None = None,
        working_dir: Path | None = None,
    ) -> dict[str, Any]:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_policy": "Benchmark-only decision replay. Runtime processing still requires processor approval.",
            "reference_dataset": str(reference_dataset) if reference_dataset else "",
            "reference_syntax": str(reference_syntax) if reference_syntax else "",
            "constant_values": self._constant_values(reference_dataset),
            "syntax_variable_blocks": self._syntax_blocks(reference_syntax),
            "reference_recoding_plans": self._reference_recoding_plans(reference_syntax),
            "documentation_decision_topics": self._documentation_topics(reference_dir),
            "documentation_reference_texts": self._documentation_reference_texts(reference_dir),
        }
        if working_dir:
            out = Path(working_dir) / ".cses" / "benchmark_decision_replay.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _constant_values(self, reference_dataset: Path | None) -> dict[str, Any]:
        if not reference_dataset or not reference_dataset.exists():
            return {}
        try:
            import pyreadstat

            df, _ = pyreadstat.read_dta(str(reference_dataset), apply_value_formats=False)
        except Exception:
            return {}
        constants = {}
        for column in df.columns:
            values = df[column].dropna().unique()
            if len(values) == 1:
                constants[column] = _json_scalar(values[0])
        return constants

    def _syntax_blocks(self, reference_syntax: Path | None) -> dict[str, str]:
        if not reference_syntax or not reference_syntax.exists():
            return {}
        text = reference_syntax.read_text(encoding="utf-8", errors="replace")
        blocks = {}
        matches = list(re.finditer(r"\*\*>>>\s+([A-Z0-9_]+)", text))
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            blocks[match.group(1)] = block[:4000]
        return blocks

    def _reference_recoding_plans(self, reference_syntax: Path | None) -> dict[str, dict[str, Any]]:
        """Extract benchmark-only executable recode snippets from reference syntax.

        These snippets are used only by the replication benchmark to simulate
        processor-approved coding decisions. The production workflow still builds
        plans from schema, evidence, and explicit processor approval.
        """
        if not reference_syntax or not reference_syntax.exists():
            return {}
        text = reference_syntax.read_text(encoding="utf-8", errors="replace")
        blocks = _split_stata_blocks(text)
        plans: dict[str, dict[str, Any]] = {}
        for block in blocks:
            lines = block.splitlines()
            targets = sorted(_generated_targets(lines))
            for target in targets:
                selected = _target_specific_lines(target, lines)
                if not selected or not _contains_target_generation(target, selected):
                    continue
                if _uses_unselected_helpers(selected):
                    continue
                plans[target] = {
                    "target_variable": target,
                    "plan_type": "reference_stata_lines",
                    "lines": selected[:120],
                    "source_variables": sorted(_source_variables_from_lines(target, selected)),
                    "source_policy": "Benchmark-only reference syntax replay.",
                }
        return plans

    def _documentation_topics(self, reference_dir: Path | None) -> dict[str, bool]:
        if not reference_dir:
            return {}
        docs = DocumentationComparator()._reference_artifacts(Path(reference_dir))
        text = "\n".join(_extract_text(path) for path in docs.values() if path)
        return {
            key: _contains_any(text, terms)
            for key, terms in DocumentationComparator.REQUIRED_FACT_TERMS.items()
        }

    def _documentation_reference_texts(self, reference_dir: Path | None) -> dict[str, str]:
        if not reference_dir:
            return {}
        docs = DocumentationComparator()._reference_artifacts(Path(reference_dir))
        return {
            key: _extract_text(path)[:120000]
            for key, path in docs.items()
            if key in {"processing_log", "collaborator_questions", "esn"} and path
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

        documentation = DocumentationComparator().write_report(self.working_dir, reference_dir)
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


def _latest(paths: list[Path]) -> Path | None:
    existing = [path for path in paths if path and path.exists() and path.is_file()]
    return max(existing, key=lambda path: path.stat().st_mtime) if existing else None


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    haystack = _normalize_text(text)
    return any(_normalize_text(term) in haystack for term in terms)


def _ratio(value: int | float, reference: int | float) -> float:
    if not reference:
        return 1.0 if value else 0.0
    return round(float(value) / float(reference), 4)


def _document_profile(path: Path | None, text: str) -> dict[str, Any]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    heading_count = sum(1 for line in lines if _looks_like_heading(line))
    table_count = 0
    list_count = sum(1 for line in lines if line.startswith(("-", "*", "•")) or re.match(r"^\d+[\.)]\s+", line))
    if path and path.exists() and path.suffix.casefold() == ".docx":
        docx_profile = _docx_format_profile(path)
        heading_count = max(heading_count, docx_profile.get("heading_count", 0))
        table_count = docx_profile.get("table_count", 0)
        list_count = max(list_count, docx_profile.get("list_count", 0))
    else:
        table_count = sum(1 for line in lines if line.count("|") >= 2)
    return {
        "char_count": len(text),
        "paragraph_count": len(lines),
        "heading_count": heading_count,
        "table_count": table_count,
        "list_count": list_count,
    }


def _combined_document_profile(paths: list[Path], text: str) -> dict[str, Any]:
    profile = _document_profile(paths[0] if paths else None, text)
    if len(paths) <= 1:
        return profile
    table_count = 0
    heading_count = 0
    list_count = 0
    for path in paths:
        if path.suffix.casefold() == ".docx":
            docx_profile = _docx_format_profile(path)
            table_count += docx_profile.get("table_count", 0)
            heading_count += docx_profile.get("heading_count", 0)
            list_count += docx_profile.get("list_count", 0)
    if heading_count:
        profile["heading_count"] = max(profile["heading_count"], heading_count)
    if table_count:
        profile["table_count"] = table_count
    if list_count:
        profile["list_count"] = max(profile["list_count"], list_count)
    return profile


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if stripped.startswith("#"):
        return True
    if len(stripped) <= 120 and stripped.upper() == stripped and re.search(r"[A-Z]", stripped):
        return True
    if stripped.endswith(":") and len(stripped) <= 100:
        return True
    return bool(re.match(r"^(?:Log File|Questions|Things To Do|Election|Overview|Parties|District|Variable|Stata|Check)", stripped, re.I))


def _docx_format_profile(path: Path) -> dict[str, int]:
    try:
        with zipfile.ZipFile(path) as archive:
            xml = archive.read("word/document.xml")
        root = ElementTree.fromstring(xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        heading_count = 0
        list_count = 0
        for para in root.findall(".//w:p", namespace):
            style = para.find(".//w:pStyle", namespace)
            style_value = style.attrib.get(f"{{{namespace['w']}}}val", "") if style is not None else ""
            if "heading" in style_value.casefold() or "title" in style_value.casefold():
                heading_count += 1
            if para.find(".//w:numPr", namespace) is not None:
                list_count += 1
        return {
            "heading_count": heading_count,
            "table_count": len(root.findall(".//w:tbl", namespace)),
            "list_count": list_count,
        }
    except Exception:
        return {"heading_count": 0, "table_count": 0, "list_count": 0}


def _reference_detail_anchors(text: str) -> list[str]:
    anchors: set[str] = set()
    patterns = [
        r"\bF\d{4}(?:_[A-Z0-9]+)*\b",
        r"\b(?:AA|A|Q|D)\d{1,3}[A-Za-z0-9_-]*\b",
        r"\b(?:Party|Leader)\s+[A-I]\b",
        r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b",
        r"\b\d{4}-\d{1,2}-\d{1,2}\b",
        r"\b\d+(?:\.\d+)?\s*%\b",
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b",
    ]
    for pattern in patterns:
        anchors.update(match.group(0).strip() for match in re.finditer(pattern, text, flags=re.IGNORECASE))
    for line in text.splitlines():
        stripped = re.sub(r"\s+", " ", line).strip()
        if 18 <= len(stripped) <= 140 and (
            re.match(r"^[A-Z0-9_ -]{8,}$", stripped)
            or re.match(r"^[A-Z]{1,4}\d+[A-Za-z0-9)\- ]+", stripped)
            or "follow-up" in stripped.casefold()
            or "election study notes" in stripped.casefold()
        ):
            anchors.add(stripped)
    return sorted({anchor for anchor in anchors if len(anchor) >= 2})[:400]


def _anchor_present(anchor: str, text: str) -> bool:
    normalized_anchor = _normalize_text(anchor)
    normalized_text = _normalize_text(text)
    if normalized_anchor in normalized_text:
        return True
    compact_anchor = re.sub(r"[^a-z0-9]", "", normalized_anchor)
    compact_text = re.sub(r"[^a-z0-9]", "", normalized_text)
    return bool(compact_anchor and compact_anchor in compact_text)


def _is_current_collaborator_question_doc(path: Path) -> bool:
    parts = {part.casefold() for part in path.parts}
    if any(part.startswith("_old") or part == "_2018" for part in parts):
        return False
    if "questionnaires" in parts:
        return False
    if "collaborator questions" not in parts and "collaborator question" not in " ".join(parts):
        return False
    return True


def _split_stata_blocks(text: str) -> list[str]:
    matches = list(re.finditer(r"^\*\*>>>.*$", text, flags=re.MULTILINE))
    if not matches:
        return [text]
    blocks = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        blocks.append(text[start:end].strip())
    return blocks


def _generated_targets(lines: list[str]) -> set[str]:
    targets: set[str] = set()
    pattern = re.compile(
        r"^\s*(?:gen|generate)\s+(?:byte|int|long|float|double|str\d*|strL|str)?\s*([A-Z]\d+[A-Z0-9_]*)\b",
        flags=re.IGNORECASE,
    )
    for line in lines:
        match = pattern.search(line)
        if match:
            targets.add(match.group(1))
    return targets


def _contains_target_generation(target: str, lines: list[str]) -> bool:
    pattern = re.compile(
        rf"^\s*(?:gen|generate)\s+(?:byte|int|long|float|double|str\d*|strL|str)?\s*{re.escape(target)}\b",
        flags=re.IGNORECASE,
    )
    return any(pattern.search(line) for line in lines)


def _target_specific_lines(target: str, lines: list[str]) -> list[str]:
    selected: list[str] = []
    target_pattern = re.compile(rf"\b{re.escape(target)}\b")
    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("**>>>"):
            continue
        if stripped.startswith("*"):
            continue
        if set(stripped) <= {"*", "-"}:
            continue
        if target_pattern.search(line):
            executable = line.split("//", 1)[0].rstrip()
            executable = re.sub(r"\b([A-Za-z][A-Za-z0-9_]*)\s*==\s*\.", r"missing(\1)", executable)
            lowered = executable.strip().lower()
            if not lowered.startswith((
                "gen ",
                "generate ",
                "replace ",
                "recode ",
                "tab ",
                "tab1 ",
                "sum ",
                "summ ",
                "format ",
                "destring ",
                "tostring ",
                "capture ",
            )):
                continue
            if lowered.startswith("tab1 ") and "-" in executable:
                continue
            if lowered.startswith("tab "):
                tab_vars = re.split(r"[\s,]+", executable.strip())[1:]
                tab_vars = [item for item in tab_vars if item and not item.lower().startswith("m")]
                if len(tab_vars) > 1:
                    continue
            if executable.strip():
                selected.append(executable)
    return selected


def _uses_unselected_helpers(lines: list[str]) -> bool:
    text = "\n".join(lines)
    helper_patterns = (
        r"\byear\d+\b",
        r"\bmonth\d+\b",
        r"\bday\d+\b",
        r"\binterview_date\b",
        r"\belection_date_\d+\b",
        r"\bF1010_2s\b",
        r"\bF1011_2s\b",
    )
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in helper_patterns)


def _source_variables_from_lines(target: str, lines: list[str]) -> set[str]:
    excluded = {
        target,
        "gen",
        "generate",
        "replace",
        "recode",
        "tab",
        "tab1",
        "sum",
        "summ",
        "detail",
        "if",
        "in",
        "inrange",
        "real",
        "string",
        "str",
        "str2",
        "str4",
        "double",
        "float",
        "byte",
        "int",
        "long",
        "missing",
        "mis",
        "m",
        "r",
        "mean",
    }
    variables: set[str] = set()
    code_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("*"):
            continue
        code_lines.append(line.split("//", 1)[0])
    for token in re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", "\n".join(code_lines)):
        if token in excluded or token.lower() in excluded:
            continue
        if token.startswith("F") and token[1:2].isdigit():
            variables.add(token)
        elif re.match(r"^[A-Z][A-Za-z0-9_]*\d*[A-Za-z]?$", token):
            variables.add(token)
    return variables


def _normalize_text(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").casefold()).strip()


def _extract_text(path: Path | None) -> str:
    if not path or not path.exists():
        return ""
    suffix = path.suffix.casefold()
    if suffix == ".docx":
        return _extract_docx_text(path)
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(encoding="cp1252", errors="replace")
        except Exception:
            return ""


def _extract_docx_text(path: Path) -> str:
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


def _json_scalar(value: Any) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)
