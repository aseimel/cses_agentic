"""Stata execution verification for generated CSES syntax."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class StataExecutionResult:
    success: bool
    do_file: str
    log_path: str = ""
    output_dataset: str = ""
    error: str = ""
    errors: list[dict[str, Any]] = field(default_factory=list)
    repairs_attempted: int = 0
    repair_log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StataExecutionVerifier:
    """Run generated syntax and verify it produced the expected dataset."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)

    def run(self, do_path: Path, stata_path: str | None = None) -> StataExecutionResult:
        if os.name == "nt" and os.environ.get("CSES_ALLOW_VISIBLE_STATA", "").strip() != "1":
            result = StataExecutionResult(
                success=False,
                do_file=str(do_path),
                error=(
                    "Automatic Stata execution is blocked because the configured Stata "
                    "application opens a visible window. Run the generated .do file in "
                    "Stata manually, then return to the workflow for review."
                ),
            )
            self.write(result)
            return result

        from src.agent.tool_wrappers import run_stata_debug

        result = run_stata_debug(do_path, stata_path=stata_path)
        data = result.data if isinstance(result.data, dict) else {}
        log_path = data.get("log_path", "")
        output_dataset = self._detect_output_dataset(do_path)
        parsed_errors = data.get("errors", []) or []
        if not parsed_errors and log_path:
            parsed_errors = self._parse_log_errors(Path(log_path))
        success = bool(result.success and output_dataset and not parsed_errors)
        error = result.error or ""
        if result.success and not output_dataset:
            error = "Stata completed but no processed CSES dataset was created."
        payload = StataExecutionResult(
            success=success,
            do_file=str(do_path),
            log_path=log_path,
            output_dataset=str(output_dataset) if output_dataset else "",
            error=error,
            errors=parsed_errors[:20],
        )
        self.write(payload)
        return payload

    def write(self, result: StataExecutionResult) -> Path:
        path = self.working_dir / ".cses" / "stata_execution.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _detect_output_dataset(self, do_path: Path) -> Path | None:
        candidates = sorted(do_path.parent.glob("cses-m6_micro_*.dta"), key=lambda path: path.stat().st_mtime)
        return candidates[-1] if candidates else None

    def _parse_log_errors(self, log_path: Path) -> list[dict[str, Any]]:
        if not log_path.exists():
            return []
        text = log_path.read_text(encoding="utf-8", errors="replace")
        errors = []
        for index, line in enumerate(text.splitlines(), start=1):
            if re.search(r"\br\(\d+\)", line) or ("error" in line.lower() and "no error" not in line.lower()):
                errors.append({"line_number": index, "error_line": line.strip()})
        return errors
