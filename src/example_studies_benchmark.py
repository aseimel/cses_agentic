"""Discovery and reference resolution for example-study replication benchmarks.

This module is development-only support for benchmark runners. It does not feed
country-specific reference information into normal runtime processing.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REFERENCE_LABELS = (
    "final_micro_dataset",
    "reference_micro_syntax",
    "processing_log",
    "election_results",
    "macro_dataset",
    "macro_log",
    "esn",
)


@dataclass
class ExampleStudyReference:
    study_id: str
    country: str
    year: str
    study_dir: str
    artifacts: dict[str, str]
    selection_notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_reference_studies(
    example_root: Path,
    selection: str = "final_only",
) -> list[ExampleStudyReference]:
    """Discover benchmark-eligible studies from folder contents."""
    root = Path(example_root)
    if not root.exists():
        return []
    references: list[ExampleStudyReference] = []
    for study_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        reference = resolve_reference_artifacts(study_dir, selection=selection)
        if reference:
            references.append(reference)
    return references


def resolve_reference_artifacts(
    study_dir: Path,
    selection: str = "final_only",
) -> ExampleStudyReference | None:
    """Resolve current-study Module 6 reference artifacts without hardcoded studies."""
    study_dir = Path(study_dir)
    country, year = _country_year_from_folder(study_dir.name)
    files = [path for path in study_dir.rglob("*") if path.is_file() and not _excluded_reference_path(path)]
    final_dataset = _select_final_dataset(study_dir, files, year, selection)
    if not final_dataset:
        return None
    syntax = _select_latest(
        path for path in files
        if path.suffix.lower() == ".do"
        and path.name.lower().startswith("cses-m6_micro_")
        and _under_named_part(path, "micro")
        and "label" not in _parts_text(path)
        and "check" not in _parts_text(path)
    )
    log = _select_latest(
        path for path in files
        if path.suffix.lower() in {".doc", ".docx", ".txt"}
        and "log-file" in path.name.lower()
        and "macro" not in _parts_text(path.relative_to(study_dir))
        and _under_named_part(path, "micro")
    )
    election = _select_latest(
        path for path in files
        if path.suffix.lower() in {".xlsx", ".xls", ".csv", ".ods"}
        and "election" in _parts_text(path)
        and "result" in _parts_text(path)
    )
    macro_dataset = _select_latest(
        path for path in files
        if path.suffix.lower() in {".xlsx", ".xls"}
        and "macro" in _parts_text(path)
        and "macro data" in path.name.lower()
    )
    macro_log = _select_latest(
        path for path in files
        if path.suffix.lower() in {".doc", ".docx", ".txt"}
        and "macro" in _parts_text(path)
        and "log-file" in path.name.lower()
    )
    esn = _select_latest(
        path for path in files
        if "macro" in _parts_text(path)
        and "esn" in path.name.lower()
        and path.suffix.lower() in {".txt", ".doc", ".docx"}
    )
    artifacts = {
        "final_micro_dataset": _rel(study_dir, final_dataset),
        "reference_micro_syntax": _rel(study_dir, syntax),
        "processing_log": _rel(study_dir, log),
        "election_results": _rel(study_dir, election),
        "macro_dataset": _rel(study_dir, macro_dataset),
        "macro_log": _rel(study_dir, macro_log),
        "esn": _rel(study_dir, esn),
    }
    notes = []
    if "FINAL dataset" not in str(final_dataset):
        notes.append("Reference dataset selected from current micro output because no FINAL dataset folder matched.")
    for label in ("reference_micro_syntax", "processing_log"):
        if not artifacts[label]:
            notes.append(f"Missing required reference artifact: {label}.")
    return ExampleStudyReference(
        study_id=study_dir.name,
        country=country,
        year=year,
        study_dir=str(study_dir),
        artifacts=artifacts,
        selection_notes=notes,
    )


def write_reference_manifest(references: list[ExampleStudyReference], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_count": len(references),
        "studies": [item.to_dict() for item in references],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _country_year_from_folder(name: str) -> tuple[str, str]:
    if "_" in name:
        country, year = name.rsplit("_", 1)
        return country, year
    return name, ""


def _select_final_dataset(
    study_dir: Path,
    files: list[Path],
    year: str,
    selection: str,
) -> Path | None:
    final_candidates = [
        path for path in files
        if path.suffix.lower() == ".dta"
        and "final dataset" in _parts_text(path)
        and _looks_current_module6_dataset(path, year)
        and _under_named_part(path, "micro")
    ]
    selected = _select_latest(final_candidates)
    if selected or selection == "final_only":
        return selected
    fallback = [
        path for path in files
        if path.suffix.lower() == ".dta"
        and path.name.lower().startswith("cses-m6_micro_")
        and _under_named_part(path, "micro")
        and _year_matches(path, year)
    ]
    return _select_latest(fallback)


def _looks_current_module6_dataset(path: Path, year: str) -> bool:
    name = path.name.lower()
    if "cses-m5" in name or "module 5" in _parts_text(path):
        return False
    if "cses-m6_micro" in name:
        return _year_matches(path, year)
    return _year_matches(path, year)


def _excluded_reference_path(path: Path) -> bool:
    text = _parts_text(path)
    parts = {part.lower() for part in path.parts}
    if parts & {"old", "_old", "__pycache__"}:
        return True
    if any(part.lower().startswith("_old") for part in path.parts):
        return True
    if "cses-m5" in path.name.lower() or "module 5" in text:
        return True
    if "old-" in text or "\\old_" in text or "/old_" in text:
        return True
    return False


def _year_matches(path: Path, year: str) -> bool:
    if not year:
        return True
    name = path.name
    return bool(re.search(rf"(^|[_\W]){re.escape(year)}([_\W]|$)", name))


def _select_latest(paths: Any) -> Path | None:
    candidates = [path for path in paths if path and path.exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: (_date_score(path.name), path.stat().st_mtime, str(path).lower()))[-1]


def _date_score(name: str) -> int:
    matches = re.findall(r"(20\d{6})", name)
    if matches:
        return max(int(item) for item in matches)
    year_matches = re.findall(r"(20\d{2})", name)
    return max((int(item) * 10000 for item in year_matches), default=0)


def _under_named_part(path: Path, name: str) -> bool:
    return name.lower() in {part.lower() for part in path.parts}


def _parts_text(path: Path) -> str:
    return " ".join(str(part).lower() for part in path.parts)


def _rel(root: Path, path: Path | None) -> str:
    if not path:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)
