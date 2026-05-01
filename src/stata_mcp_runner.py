"""Isolated MCP-Stata runner process.

The MCP stdio client can hang during cleanup if the server tool does not return.
Running it in a subprocess lets the workflow enforce a hard timeout without ever
falling back to the visible Stata executable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _split_do_file_commands(do_path: Path) -> list[str]:
    commands: list[str] = []
    pending = ""
    in_block_comment = False
    for raw_line in do_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if in_block_comment:
            if "*/" in line:
                in_block_comment = False
            continue
        if line.startswith("/*"):
            if "*/" not in line:
                in_block_comment = True
            continue
        if line.startswith("*") or line.startswith("//"):
            continue
        if line.endswith("///"):
            pending += line[:-3].rstrip() + " "
            continue
        command = (pending + line).strip()
        pending = ""
        if command:
            commands.append(command)
    if pending.strip():
        commands.append(pending.strip())
    return commands


async def _run(do_path: Path, stata_path: str, max_output_lines: int) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {
        **os.environ,
        "MCP_STATA_LOGLEVEL": os.environ.get("MCP_STATA_LOGLEVEL", "ERROR"),
        "PYTHONWARNINGS": os.environ.get("PYTHONWARNINGS", "ignore::DeprecationWarning"),
    }
    if stata_path:
        env["STATA_PATH"] = stata_path

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "mcp_stata.server"],
        env=env,
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            outputs: list[str] = []
            last_payload: dict = {"success": True, "rc": 0}
            command_failed = False
            for index, command in enumerate(_split_do_file_commands(do_path), start=1):
                result = await session.call_tool(
                    "run_command",
                    {
                        "code": command,
                        "cwd": str(do_path.parent),
                        "echo": True,
                        "as_json": True,
                        "max_output_lines": max_output_lines,
                    },
                )
                text = ""
                if getattr(result, "content", None):
                    text = getattr(result.content[0], "text", "") or ""
                if not text and isinstance(getattr(result, "structuredContent", None), dict):
                    text = result.structuredContent.get("result", "") or ""
                try:
                    payload = json.loads(text)
                except Exception:
                    payload = {"success": False, "rc": 999, "stdout": text, "error": "MCP-Stata returned non-JSON output."}
                last_payload = payload
                outputs.append(f". {command}")
                if payload.get("stdout"):
                    outputs.append(str(payload["stdout"]))
                if payload.get("stderr"):
                    outputs.append(str(payload["stderr"]))
                rc = int(payload.get("rc") or 0)
                if not payload.get("success") or rc != 0:
                    payload["line"] = index
                    command_failed = True
                    break
            log_path = do_path.with_suffix(".log")
            log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
            payload = {
                "command": f"run {do_path.name} via MCP-Stata commands",
                "rc": int(last_payload.get("rc") or 0),
                "stdout": "\n".join(outputs),
                "stderr": last_payload.get("stderr"),
                "log_path": str(log_path),
                "success": (not command_failed) and bool(last_payload.get("success", True)) and int(last_payload.get("rc") or 0) == 0,
                "error": last_payload.get("error"),
            }
            return {"ok": True, "text": json.dumps(payload, ensure_ascii=False)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("do_path")
    parser.add_argument("--stata-path", default="")
    parser.add_argument("--max-output-lines", type=int, default=2000)
    args = parser.parse_args()
    try:
        payload = asyncio.run(_run(Path(args.do_path), args.stata_path, args.max_output_lines))
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
