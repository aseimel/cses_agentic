"""
MCP-Stata configuration helpers.

MCP-Stata is distributed as a uvx runnable tool:
uvx --refresh --refresh-package mcp-stata --from mcp-stata@latest mcp-stata
"""

from __future__ import annotations

import json
from pathlib import Path


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
