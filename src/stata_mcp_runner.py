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
from typing import Any


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
            log_path = do_path.with_suffix(".log")
            line_mode = os.environ.get("CSES_MCP_STATA_LINE_MODE") == "1"
            default_timeout = "120" if line_mode else os.environ.get("CSES_MCP_STATA_TIMEOUT_SECONDS", "1800")
            command_timeout = int(os.environ.get("MCP_STATA_COMMAND_TIMEOUT_SECONDS", default_timeout) or default_timeout)
            if not line_mode:
                outputs.append(f". {do_path}")
                log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
                payload = await _run_do_file_background(
                    session=session,
                    do_path=do_path,
                    max_output_lines=max_output_lines,
                    timeout_seconds=command_timeout,
                    outputs=outputs,
                    local_log_path=log_path,
                )
                return {"ok": True, "text": json.dumps(payload, ensure_ascii=False)}
            commands = _split_do_file_commands(do_path) if line_mode else [str(do_path)]
            for index, command in enumerate(commands, start=1):
                outputs.append(f". {command}")
                log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
                try:
                    if line_mode:
                        tool_name = "run_command"
                        tool_args = {
                            "code": command,
                            "cwd": str(do_path.parent),
                            "echo": True,
                            "as_json": True,
                            "max_output_lines": max_output_lines,
                        }
                    else:
                        tool_name = "run_do_file"
                        tool_args = {
                            "path": command,
                            "cwd": str(do_path.parent),
                            "echo": True,
                            "as_json": True,
                            "max_output_lines": max_output_lines,
                        }
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, tool_args),
                        timeout=command_timeout,
                    )
                except asyncio.TimeoutError:
                    payload = {
                        "success": False,
                        "rc": 998,
                        "stdout": "",
                        "stderr": "",
                        "error": f"MCP-Stata command timed out after {command_timeout} seconds.",
                        "line": index,
                        "command": command,
                    }
                    last_payload = payload
                    outputs.append(payload["error"])
                    command_failed = True
                    break
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
                if payload.get("stdout"):
                    outputs.append(str(payload["stdout"]))
                if payload.get("stderr"):
                    outputs.append(str(payload["stderr"]))
                log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
                rc = int(payload.get("rc") or 0)
                if not payload.get("success") or rc != 0:
                    payload["line"] = index
                    command_failed = True
                    break
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


def _tool_text(result: Any) -> str:
    text = ""
    if getattr(result, "content", None):
        text = getattr(result.content[0], "text", "") or ""
    if not text and isinstance(getattr(result, "structuredContent", None), dict):
        text = result.structuredContent.get("result", "") or ""
    return text


async def _run_do_file_background(
    session: Any,
    do_path: Path,
    max_output_lines: int,
    timeout_seconds: int,
    outputs: list[str],
    local_log_path: Path,
) -> dict[str, Any]:
    started = await session.call_tool(
        "run_do_file_background",
        {
            "path": str(do_path),
            "cwd": str(do_path.parent),
            "echo": True,
            "as_json": True,
            "max_output_lines": max_output_lines,
        },
    )
    try:
        start_payload = json.loads(_tool_text(started))
    except Exception:
        start_payload = {"status": "error", "error": _tool_text(started)}
    task_id = start_payload.get("task_id")
    server_log_path = start_payload.get("log_path") or ""
    if not task_id:
        return {
            "command": f"run_do_file_background {do_path.name}",
            "rc": 999,
            "stdout": "\n".join(outputs),
            "stderr": "",
            "log_path": str(local_log_path),
            "success": False,
            "error": start_payload.get("error") or "MCP-Stata did not return a task id.",
        }
    outputs.append(f"Started MCP-Stata task {task_id}.")
    if server_log_path:
        outputs.append(f"MCP-Stata log: {server_log_path}")
    local_log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    status_payload: dict[str, Any] = {}
    while asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(2)
        status_result = await session.call_tool(
            "get_task_status",
            {"task_id": task_id, "allow_polling": True},
        )
        try:
            status_payload = json.loads(_tool_text(status_result))
        except Exception:
            status_payload = {"status": "error", "error": _tool_text(status_result)}
        if status_payload.get("status") == "done":
            break
    else:
        await session.call_tool("cancel_task", {"task_id": task_id})
        outputs.append(f"MCP-Stata task timed out after {timeout_seconds} seconds.")
        local_log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
        return {
            "command": f"run_do_file_background {do_path.name}",
            "rc": 998,
            "stdout": "\n".join(outputs),
            "stderr": "",
            "log_path": server_log_path or str(local_log_path),
            "success": False,
            "error": f"MCP-Stata task timed out after {timeout_seconds} seconds.",
        }
    result = await session.call_tool(
        "get_task_result",
        {"task_id": task_id, "allow_polling": True},
    )
    try:
        result_payload = json.loads(_tool_text(result))
    except Exception:
        result_payload = {"status": "error", "error": _tool_text(result)}
    raw_result = result_payload.get("result") or ""
    try:
        command_payload = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
    except Exception:
        command_payload = {"success": False, "rc": 999, "stdout": raw_result, "error": "MCP-Stata returned non-JSON task result."}
    stdout = command_payload.get("stdout") or ""
    stderr = command_payload.get("stderr") or ""
    if stdout:
        outputs.append(str(stdout))
    if stderr:
        outputs.append(str(stderr))
    log_path = command_payload.get("log_path") or result_payload.get("log_path") or server_log_path or str(local_log_path)
    local_log_path.write_text("\n".join(outputs), encoding="utf-8", errors="replace")
    return {
        "command": f"run_do_file_background {do_path.name}",
        "rc": int(command_payload.get("rc") or 0),
        "stdout": "\n".join(outputs),
        "stderr": stderr,
        "log_path": log_path,
        "success": bool(command_payload.get("success")) and int(command_payload.get("rc") or 0) == 0,
        "error": command_payload.get("error") or result_payload.get("error"),
    }


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
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
