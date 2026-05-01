"""
MCP-Stata configuration helpers.

MCP-Stata is distributed as a uvx runnable tool:
uvx --refresh --refresh-package mcp-stata --from mcp-stata@latest mcp-stata
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


MCP_STATA_REPOSITORY = "https://github.com/tmonk/mcp-stata"
MCP_STATA_COMMAND = "uvx"
MCP_STATA_ARGS = [
    "--refresh",
    "--refresh-package",
    "mcp-stata",
    "--from",
    "mcp-stata@latest",
    "mcp-stata",
]


def build_mcp_stata_config(stata_path: str = "") -> dict:
    """Build a generic stdio MCP server config for MCP-capable clients."""
    server = {
        "command": MCP_STATA_COMMAND,
        "args": MCP_STATA_ARGS,
    }
    if stata_path:
        server["env"] = {"STATA_PATH": stata_path}
    return {"mcpServers": {"mcp-stata": server}}


def write_mcp_stata_config(config_dir: Path, stata_path: str = "") -> Path:
    """Write a user-visible MCP-Stata config snippet."""
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "mcp-stata.json"
    config = build_mcp_stata_config(stata_path)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config_path


@dataclass
class MCPStataRunResult:
    success: bool
    log_path: str = ""
    stdout: str = ""
    stderr: str = ""
    rc: int | None = None
    error: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class MCPStataRunner:
    """Run Stata through mcp-stata instead of launching the visible Stata app."""

    def __init__(self, stata_path: str = "", timeout_seconds: int | None = None):
        self.stata_path = stata_path
        self.timeout_seconds = timeout_seconds or int(os.environ.get("CSES_MCP_STATA_TIMEOUT_SECONDS", "120"))

    def available(self) -> bool:
        try:
            import mcp  # noqa: F401
            import mcp_stata.server  # noqa: F401
            return True
        except Exception:
            return False

    def run_do_file(self, do_path: Path) -> MCPStataRunResult:
        if not self.available():
            return MCPStataRunResult(
                success=False,
                error="MCP-Stata is not installed. Install the mcp and mcp-stata Python packages.",
            )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.stata_mcp_runner",
                    str(Path(do_path)),
                    "--stata-path",
                    self.stata_path,
                    "--max-output-lines",
                    os.environ.get("CSES_MCP_STATA_MAX_OUTPUT_LINES", "2000"),
                ],
                cwd=str(Path(__file__).resolve().parent.parent),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return MCPStataRunResult(
                success=False,
                error=f"MCP-Stata did not return within {self.timeout_seconds} seconds while running the .do file.",
            )
        except Exception as exc:
            return MCPStataRunResult(success=False, error=str(exc))
        text = self._extract_runner_payload(completed.stdout)
        if not text:
            return MCPStataRunResult(
                success=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error=completed.stderr.strip() or f"MCP-Stata runner exited with code {completed.returncode}.",
            )
        try:
            runner_payload = json.loads(text)
        except Exception:
            return MCPStataRunResult(
                success=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error="MCP-Stata runner returned non-JSON output.",
                raw={"stdout": completed.stdout, "stderr": completed.stderr},
            )
        if not runner_payload.get("ok"):
            return MCPStataRunResult(
                success=False,
                stdout=completed.stdout,
                stderr=completed.stderr,
                error=runner_payload.get("error") or "MCP-Stata runner failed.",
                raw=runner_payload,
            )
        return self._parse_tool_text(runner_payload.get("text", ""))

    def _extract_runner_payload(self, stdout: str) -> str:
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                return stripped
        return ""

    def _parse_tool_text(self, text: str) -> MCPStataRunResult:
        try:
            payload = json.loads(text)
        except Exception:
            return MCPStataRunResult(
                success=False,
                stdout=text,
                error="MCP-Stata returned non-JSON output.",
                raw={"text": text},
            )
        success = bool(payload.get("success")) and int(payload.get("rc") or 0) == 0
        error_payload = payload.get("error") or {}
        if isinstance(error_payload, dict):
            error_text = error_payload.get("message") or error_payload.get("error") or ""
        else:
            error_text = str(error_payload) if error_payload else ""
        return MCPStataRunResult(
            success=success,
            log_path=payload.get("log_path") or "",
            stdout=payload.get("stdout") or payload.get("smcl_output") or "",
            stderr=payload.get("stderr") or "",
            rc=payload.get("rc"),
            error=error_text,
            raw=payload,
        )
