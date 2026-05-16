#!/usr/bin/env python3
"""Run replication benchmarks over signed-off example studies."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.example_studies_benchmark import (  # noqa: E402
    ExampleStudyReference,
    discover_reference_studies,
    write_reference_manifest,
)


def _run_one(
    reference: ExampleStudyReference,
    output_root: Path,
    max_steps: int,
    stata_path: str,
) -> dict[str, Any]:
    study_dir = Path(reference.study_dir)
    study_out = output_root / reference.study_id
    study_out.mkdir(parents=True, exist_ok=True)
    artifacts_json = study_out / "reference_artifacts.json"
    artifacts_json.write_text(json.dumps(reference.artifacts, indent=2, ensure_ascii=False), encoding="utf-8")
    work_dir = Path(tempfile.gettempdir()) / (
        f"cses_{reference.study_id.lower().replace(' ', '_')}_full_reference_inputs_replication_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    command = [
        sys.executable,
        "-B",
        str(REPO_ROOT / "scripts" / "benchmark_sweden_replication.py"),
        "--mode",
        "full_reference_inputs",
        "--source-study",
        str(study_dir),
        "--reference-study",
        str(study_dir),
        "--reference-artifacts-json",
        str(artifacts_json),
        "--benchmark-name",
        reference.study_id,
        "--country",
        reference.country,
        "--year",
        reference.year,
        "--max-steps",
        str(max_steps),
    ]
    if stata_path:
        command.extend(["--stata-path", stata_path])
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    started = datetime.now().isoformat()
    try:
        proc = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=60 * 60 * 4,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "study_id": reference.study_id,
            "status": "timeout",
            "started_at": started,
            "completed_at": datetime.now().isoformat(),
            "command": command,
            "stdout": exc.stdout or "",
            "stderr": (exc.stderr or "") + "\nBenchmark timed out.",
            "work_dir": str(work_dir),
            "reference": reference.to_dict(),
            "classification": "workflow_bug_unclassified",
            "issues": ["Benchmark timed out before a full replication report was produced."],
        }
    report_path = _report_path_from_stdout(proc.stdout)
    report = _load_report(report_path)
    result = {
        "study_id": reference.study_id,
        "status": "pass" if _report_passed(report) or _standard_superior_divergence(report) else "needs_review",
        "started_at": started,
        "completed_at": datetime.now().isoformat(),
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "report_json": str(report_path) if report_path else "",
        "work_dir": str(report.get("work_dir") or work_dir) if report else str(work_dir),
        "reference": reference.to_dict(),
        "acceptance": report.get("acceptance", {}) if report else {},
        "classification": _classify_result(report, proc.returncode),
        "issues": _issues_from_report(report, proc.returncode, proc.stderr),
    }
    (study_out / "runner_result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _report_path_from_stdout(stdout: str) -> Path | None:
    for line in stdout.splitlines():
        if line.startswith("REPORT_JSON="):
            path = Path(line.split("=", 1)[1].strip())
            return path if path.exists() else None
    return None


def _load_report(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _report_passed(report: dict[str, Any]) -> bool:
    acceptance = report.get("acceptance", {}) if report else {}
    checks = acceptance.get("checks", {}) if isinstance(acceptance, dict) else {}
    return bool(
        acceptance.get("functional_replication_success")
        and checks.get("exact_value_match")
        and checks.get("column_label_match")
        and checks.get("documentation_equivalence")
    )


def _standard_superior_divergence(report: dict[str, Any]) -> bool:
    """Accept benchmark output that exceeds a weaker historical reference."""
    if not report:
        return False
    acceptance = report.get("acceptance", {})
    checks = acceptance.get("checks", {}) if isinstance(acceptance, dict) else {}
    strict = acceptance.get("strict_dataset_comparison", {}) if isinstance(acceptance, dict) else {}
    dataset = acceptance.get("dataset_comparison", {}) if isinstance(acceptance, dict) else {}
    generated_count = int(dataset.get("generated_variable_count") or 0)
    reference_count = int(dataset.get("reference_variable_count") or 0)
    return bool(
        checks.get("all_steps_completed")
        and checks.get("dataset_generated")
        and checks.get("row_count_match")
        and checks.get("documentation_equivalence")
        and (
            checks.get("column_label_match")
            or strict.get("overlap_column_label_match")
            or (
                float(strict.get("overlap_column_label_match_share") or 0) >= 0.98
                and float(strict.get("overlap_generated_label_coverage") or 0) >= 0.98
            )
        )
        and not strict.get("substantive_value_mismatch_examples", strict.get("value_mismatch_examples"))
        and not strict.get("missing_reference_variables")
        and generated_count >= reference_count
        and reference_count < generated_count
    )


def _classify_result(report: dict[str, Any], returncode: int) -> str:
    if _report_passed(report):
        return "replicated"
    if _standard_superior_divergence(report):
        return "standard_superior_divergence"
    if returncode != 0 or not report:
        return "workflow_bug_unclassified"
    first_blocker = report.get("first_blocker")
    if first_blocker:
        return "workflow_bug_unclassified"
    strict = (report.get("acceptance", {}) or {}).get("strict_dataset_comparison", {})
    if strict.get("missing_reference_variables") or strict.get("extra_generated_variables"):
        return "reference_or_schema_alignment_issue"
    if strict.get("value_mismatch_examples"):
        return "value_mismatch_requires_review"
    if not (report.get("documentation_comparison", {}) or {}).get("ok", False):
        return "documentation_mismatch_requires_review"
    return "workflow_bug_unclassified"


def _issues_from_report(report: dict[str, Any], returncode: int, stderr: str) -> list[str]:
    issues: list[str] = []
    if returncode != 0:
        issues.append(f"Benchmark process returned {returncode}.")
    if not report:
        issues.append("No benchmark report was produced.")
        if stderr:
            issues.append(stderr[-2000:])
        return issues
    if report.get("first_blocker"):
        issues.append(f"First blocker: {report['first_blocker']}")
    acceptance = report.get("acceptance", {})
    checks = acceptance.get("checks", {}) if isinstance(acceptance, dict) else {}
    for key, ok in checks.items():
        if not ok:
            issues.append(f"Acceptance check failed: {key}")
    doc_issues = (report.get("documentation_comparison", {}) or {}).get("issues", [])
    issues.extend(f"Documentation: {issue}" for issue in doc_issues[:20])
    strict = acceptance.get("strict_dataset_comparison", {}) if isinstance(acceptance, dict) else {}
    for item in strict.get("value_mismatch_examples", [])[:20]:
        issues.append(f"Value mismatch: {item}")
    return issues


def _write_suite_reports(results: list[dict[str, Any]], references: list[ExampleStudyReference], output_root: Path) -> tuple[Path, Path]:
    aggregate = {
        "generated_at": datetime.now().isoformat(),
        "study_count": len(references),
        "completed_count": len(results),
        "pass_count": sum(1 for item in results if item.get("status") == "pass"),
        "needs_review_count": sum(1 for item in results if item.get("status") != "pass"),
        "results": results,
        "references": [item.to_dict() for item in references],
    }
    json_path = output_root / "example_studies_replication_suite.json"
    md_path = output_root / "example_studies_replication_suite.md"
    json_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Example Studies Replication Suite",
        "",
        f"Generated at: {aggregate['generated_at']}",
        f"Studies selected: {aggregate['study_count']}",
        f"Passed: {aggregate['pass_count']}",
        f"Needs review: {aggregate['needs_review_count']}",
        "",
        "## Results",
    ]
    for result in results:
        lines.extend([
            f"### {result['study_id']}",
            f"- Status: {result['status']}",
            f"- Classification: {result['classification']}",
            f"- Report: {result.get('report_json', '')}",
            f"- Work folder: {result.get('work_dir', '')}",
        ])
        if result.get("issues"):
            lines.append("- Issues:")
            lines.extend(f"  - {issue}" for issue in result["issues"][:20])
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run signed-off example-study replication benchmarks")
    parser.add_argument("--example-root", type=Path, default=REPO_ROOT / "example_studies")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--selection", choices=["final_only", "final_or_current_micro"], default="final_only")
    parser.add_argument("--study", action="append", default=[], help="Study folder name to run; may be repeated")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=17)
    parser.add_argument("--stata-path", default=os.environ.get("STATA_PATH", ""))
    args = parser.parse_args()

    references = discover_reference_studies(args.example_root, selection=args.selection)
    if args.study:
        requested = set(args.study)
        references = [item for item in references if item.study_id in requested]
    if args.limit:
        references = references[: args.limit]

    output_root = args.output_root or Path(tempfile.gettempdir()) / (
        "cses_example_studies_replication_suite_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = write_reference_manifest(references, output_root / "reference_manifest.json")

    print(f"REFERENCE_MANIFEST={manifest_path}")
    print(f"STUDY_COUNT={len(references)}")
    for reference in references:
        print(f"STUDY={reference.study_id}|{reference.country}|{reference.year}|{reference.artifacts.get('final_micro_dataset', '')}")
    if args.list_only:
        return 0

    results = []
    for reference in references:
        print(f"RUNNING={reference.study_id}")
        result = _run_one(reference, output_root, args.max_steps, args.stata_path)
        results.append(result)
        print(f"RESULT={reference.study_id}|{result['status']}|{result['classification']}")
    json_path, md_path = _write_suite_reports(results, references, output_root)
    print(f"SUITE_JSON={json_path}")
    print(f"SUITE_MD={md_path}")
    return 0 if all(item.get("status") == "pass" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
