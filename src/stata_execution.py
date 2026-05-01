"""Stata execution verification for generated CSES syntax."""

from __future__ import annotations

import json
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
        from src.stata_mcp import MCPStataRunner

        mcp_result = MCPStataRunner(stata_path=stata_path or "").run_do_file(do_path)
        log_path = mcp_result.log_path
        output_dataset = self._detect_output_dataset(do_path)
        parsed_errors = []
        if not parsed_errors and log_path:
            parsed_errors = self._parse_log_errors(Path(log_path))
        if mcp_result.rc not in (None, 0) and not parsed_errors:
            parsed_errors.append({"line_number": 0, "error_line": f"Stata returned r({mcp_result.rc})."})
        success = bool(mcp_result.success and output_dataset and not parsed_errors)
        error = mcp_result.error
        if mcp_result.success and not output_dataset:
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
