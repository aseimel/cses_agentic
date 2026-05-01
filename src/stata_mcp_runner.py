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


async def _run(do_path: Path, stata_path: str, max_output_lines: int) -> dict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {
        **os.environ,
        "MCP_STATA_LOGLEVEL": os.environ.get("MCP_STATA_LOGLEVEL", "WARNING"),
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
            result = await session.call_tool(
                "run_do_file",
                {
                    "path": str(do_path),
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
            return {"ok": True, "text": text}


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
