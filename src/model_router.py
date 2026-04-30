"""Model routing between LiteLLM providers and Codex OAuth."""

from __future__ import annotations

from typing import Any

from litellm import completion as litellm_completion

from src.codex_oauth_client import CodexOAuthClient, is_codex_oauth_model


def completion(model: str, messages: list[dict[str, Any]], **kwargs: Any):
    """Route completion calls by model prefix."""
    if is_codex_oauth_model(model):
        return CodexOAuthClient().complete(messages=messages, model=model, **kwargs)
    return litellm_completion(model=model, messages=messages, **kwargs)
