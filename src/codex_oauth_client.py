"""Narrow Codex OAuth client adapter.

The v1 implementation deliberately reuses Codex-owned auth state and does not
rotate refresh tokens. Direct HTTP support can be added once the public Codex
SDK surface exposes a stable local-call contract for third-party desktop apps.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.auth_profiles import CodexOAuthAuthSource


@dataclass
class SimpleMessage:
    content: str = ""
    tool_calls: list | None = None


@dataclass
class SimpleChoice:
    message: SimpleMessage


@dataclass
class SimpleCompletionResponse:
    choices: list[SimpleChoice]


class CodexOAuthClient:
    """Adapter for `openai-codex/*` model refs."""

    def __init__(self, auth_source: CodexOAuthAuthSource | None = None):
        self.auth_source = auth_source or CodexOAuthAuthSource()

    def complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **_: Any,
    ) -> SimpleCompletionResponse:
        status = self.auth_source.check()
        if not status.signed_in:
            raise RuntimeError(f"Codex OAuth is not ready: {status.message}")

        if tools:
            raise RuntimeError(
                "Codex OAuth tool-call transport is not available in this CSES build yet. "
                "Use the Proceed button's deterministic workflow path, or select an API/OpenWebUI model for tool-calling chat."
            )

        text = self._complete_via_codex_subprocess(messages, model, timeout=timeout)
        return SimpleCompletionResponse(choices=[SimpleChoice(message=SimpleMessage(content=text))])

    def _complete_via_codex_subprocess(self, messages: list[dict[str, Any]], model: str, timeout: int | None = None) -> str:
        prompt = _messages_to_prompt(messages)
        model_name = model.split("/", 1)[1] if "/" in model else model
        command = ["codex", "exec", "--model", model_name, "--skip-git-repo-check", prompt]
        try:
            proc = subprocess.run(
                command,
                cwd=str(Path.cwd()),
                text=True,
                capture_output=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout or 180,
            )
        except PermissionError as exc:
            raise RuntimeError(
                "Codex is installed but Windows denied launching it from this app. "
                "Open Codex once manually, or use an API/OpenWebUI model for now."
            ) from exc
        except FileNotFoundError as exc:
            raise RuntimeError("Codex executable was not found. Install or sign in to Codex first.") from exc

        output = (proc.stdout or "").strip()
        if proc.returncode != 0:
            error = (proc.stderr or output or f"exit code {proc.returncode}").strip()
            raise RuntimeError(f"Codex subprocess failed: {error}")
        return output or "Codex returned no text output."


def is_codex_oauth_model(model: str | None) -> bool:
    return bool(model and str(model).strip().startswith("openai-codex/"))


def _messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        if role == "system":
            lines.append(f"System:\n{content}")
        elif role == "assistant":
            lines.append(f"Assistant:\n{content}")
        elif role == "user":
            lines.append(f"User:\n{content}")
    return "\n\n".join(lines).strip()
