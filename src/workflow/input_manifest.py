"""Input manifest and primary input selection for CSES studies."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.workflow.organizer import DATA_FORMAT_PREFERENCE, DOC_EXTENSIONS, DATA_EXTENSIONS


@dataclass
class ManifestFile:
    path: str
    relative_path: str
    name: str
    suffix: str
    role: str
    size: int
    modified_at: str
    date_score: int
    selected: bool = False
    selection_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InputManifest:
    generated_at: str
    working_dir: str
    files: list[ManifestFile]
    primary_data_file: str = ""
    primary_data_reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["files"] = [asdict(item) for item in self.files]
        return data


class PrimaryInputSelector:
    """Select the most plausible current deposited survey data file."""

    def select_data_file(self, files: list[Path]) -> tuple[Path | None, str, list[str]]:
        warnings: list[str] = []
        candidates = [path for path in files if path.suffix.lower() in DATA_EXTENSIONS]
        candidates = [path for path in candidates if not _looks_like_check_or_output(path)]
        candidates = [path for path in candidates if not _looks_like_district_or_election_input(path)]
        if not candidates:
            return None, "No survey data files detected", warnings

        scored = sorted(candidates, key=self._score_data_file, reverse=True)
        selected = scored[0]
        reason = (
            "Selected by newest date in path/name, newest modification time, "
            "larger file size, and preferred data format."
        )
        if len(scored) > 1:
            groups = _group_by_name(scored)
            if any(len(group) > 1 for group in groups.values()):
                warnings.append(
                    "Multiple deposited data versions detected. Older versions were preserved and not discarded."
                )
        return selected, reason, warnings

    def _score_data_file(self, path: Path) -> tuple:
        try:
            stat = path.stat()
            size = stat.st_size
            mtime = stat.st_mtime
        except OSError:
            size = 0
            mtime = 0
        try:
            format_rank = DATA_FORMAT_PREFERENCE.index(path.suffix.lower())
        except ValueError:
            format_rank = len(DATA_FORMAT_PREFERENCE)
        path_parts = [part.lower() for part in path.parts]
        preferred_location = 1 if path.parent.name.lower() == "micro" else 0
        source_penalty = 1 if any(part in {"emails", "e-mails", "email", "original_deposit"} for part in path_parts) else 0
        return (
            preferred_location,
            -source_penalty,
            _path_date_score(path),
            mtime,
            size,
            -format_rank,
            str(path).lower(),
        )


class InputManifestBuilder:
    """Build and persist a study input manifest."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.selector = PrimaryInputSelector()

    def build(self) -> InputManifest:
        all_files = [
            path for path in self.working_dir.rglob("*")
            if path.is_file()
            and ".cses" not in path.parts
            and "__pycache__" not in path.parts
        ]
        data_candidates = [
            path for path in all_files
            if path.suffix.lower() in DATA_EXTENSIONS
            and not _looks_like_generated_artifact(path)
            and not _looks_like_district_or_election_input(path)
        ]
        selected, reason, warnings = self.selector.select_data_file(data_candidates)
        rows: list[ManifestFile] = []
        for path in sorted(all_files, key=lambda item: str(item).lower()):
            role = classify_input_role(path)
            metadata = _safe_file_metadata(path, role)
            is_selected = selected is not None and path.resolve() == selected.resolve()
            rows.append(
                ManifestFile(
                    path=str(path.resolve()),
                    relative_path=str(path.relative_to(self.working_dir)),
                    name=path.name,
                    suffix=path.suffix.lower(),
                    role=role,
                    size=path.stat().st_size,
                    modified_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    date_score=_path_date_score(path),
                    selected=is_selected,
                    selection_reason=reason if is_selected else "",
                    metadata=metadata,
                )
            )

        if selected:
            selected_meta = _safe_file_metadata(selected, "survey_data")
            for other in data_candidates:
                if other.resolve() == selected.resolve():
                    continue
                other_meta = _safe_file_metadata(other, "survey_data")
                selected_vars = set(selected_meta.get("variables", []))
                other_vars = set(other_meta.get("variables", []))
                if selected_vars and other_vars and selected_vars != other_vars:
                    added = sorted(selected_vars - other_vars)
                    removed = sorted(other_vars - selected_vars)
                    warnings.append(
                        f"Data version differs from {other.name}: "
                        f"{len(added)} variable(s) added, {len(removed)} removed."
                    )
        return InputManifest(
            generated_at=datetime.now(timezone.utc).isoformat(),
            working_dir=str(self.working_dir.resolve()),
            files=rows,
            primary_data_file=str(selected.resolve()) if selected else "",
            primary_data_reason=reason,
            warnings=sorted(set(warnings)),
        )

    def write(self, manifest: InputManifest | None = None) -> Path:
        manifest = manifest or self.build()
        output = self.working_dir / ".cses" / "input_manifest.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return output


def classify_input_role(path: Path) -> str:
    name = path.name.lower()
    path_text = str(path).lower()
    suffix = path.suffix.lower()
    if suffix in DATA_EXTENSIONS:
        if any(term in name for term in ["district", "constituency", "wahlkreis"]):
            return "district_data"
        if any(term in path_text for term in ["final dataset", "data_checks", "labels", "deposited variable list"]):
            return "generated_or_reference_artifact"
        return "survey_data"
    if suffix in DOC_EXTENSIONS:
        if "question" in name or "questionnaire" in name:
            return "questionnaire"
        if "design" in name or "method" in name:
            return "design_report"
        if "macro" in name:
            return "macro_report"
        if "codebook" in name or "variable" in name:
            return "codebook"
        if "collaborator" in name or "email" in name or "e-mail" in name:
            return "correspondence"
        return "narrative_document"
    if suffix in {".do", ".log", ".smcl"}:
        return "generated_or_reference_artifact"
    return "other"


def _safe_file_metadata(path: Path, role: str) -> dict[str, Any]:
    if role not in {"survey_data", "district_data"}:
        return {}
    try:
        from src.ingest.data_loader import DataLoader

        info = DataLoader().load(path)
        if not info:
            return {}
        return {
            "n_rows": info.n_rows,
            "n_variables": len(info.variables),
            "variables": list(info.variables.keys()),
        }
    except Exception as exc:
        return {"metadata_error": str(exc)}


def _looks_like_generated_artifact(path: Path) -> bool:
    text = str(path).lower()
    parts = {part.lower() for part in path.parts}
    return any(
        marker in text
        for marker in [
            "final dataset",
            "data_checks",
            "deposited variable list",
            "deposited variables",
            "variable list",
            "\\labels\\",
            "/labels/",
            "cses-m6_micro_",
            "cses-m6_macro_",
        ]
    ) or "macro" in parts


def _looks_like_check_or_output(path: Path) -> bool:
    return _looks_like_generated_artifact(path) or any(
        term in path.name.lower()
        for term in ["check", "label", "final", "processed", "micro+macro"]
    )


def _looks_like_district_or_election_input(path: Path) -> bool:
    text = str(path).lower()
    return any(
        term in text
        for term in [
            "district",
            "constituency",
            "wahlkreis",
            "election results",
            "election_results",
            "election-result",
        ]
    )


def _path_date_score(path: Path) -> int:
    candidates = []
    for match in re.finditer(r"(?<!\d)(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)(?!\d)", str(path)):
        try:
            candidates.append(int("".join(match.groups())))
        except ValueError:
            pass
    return max(candidates) if candidates else 0


def _group_by_name(paths: list[Path]) -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        groups.setdefault(path.name.lower(), []).append(path)
    return groups
