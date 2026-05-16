"""
Model discovery and cost display helpers.

This combines LiteLLM's built-in model price metadata with models exposed by an
OpenAI-compatible endpoint. Endpoint models from localhost/127.0.0.1 are marked
as Local so users can distinguish local/private model servers from vendor APIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import json
import urllib.error

import litellm

from src.auth_profiles import CodexOAuthAuthSource
from src.settings import GESIS_OPENWEBUI_BASE, OPENROUTER_API_BASE


KNOWN_PROVIDER_LISTS = {
    "OpenAI": "open_ai_chat_completion_models",
    "Anthropic": "anthropic_models",
    "Google Gemini": "gemini_models",
    "xAI": "xai_models",
}

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


@dataclass
class ModelCatalogRow:
    model: str
    provider: str
    source: str
    input_per_million: str
    output_per_million: str
    context_window: str
    tools: str
    cost_known: bool
    tool_capable: bool


def is_local_api_base(api_base: str) -> bool:
    if not api_base:
        return False
    parsed = urlparse(api_base)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1", "0.0.0.0"} or host.endswith(".local")


def _format_cost(value) -> str:
    if value is None:
        return "Unknown"
    try:
        return f"${float(value) * 1_000_000:.2f}"
    except (TypeError, ValueError):
        return "Unknown"


def _metadata_for(model: str) -> dict:
    return litellm.model_cost.get(model) or litellm.model_cost.get(model.split("/", 1)[-1]) or {}


def model_supports_tools(model: str) -> bool:
    """Return True when LiteLLM metadata says the model supports tool calls."""
    metadata = _metadata_for(model)
    return bool(metadata.get("supports_function_calling") or metadata.get("supports_tool_choice"))


def litellm_model_for_completion(model: str, settings: dict[str, str] | None = None) -> str:
    """Return the LiteLLM model string needed for direct completion calls."""
    model = (model or "").strip()
    if not model:
        return ""
    if model.startswith("openrouter/"):
        return model
    settings = settings or {}
    use_openwebui = (
        settings.get("CSES_USE_OPENWEBUI", "").lower() in {"1", "true", "yes"}
        or settings.get("OPENAI_API_BASE", "").strip()
    )
    if use_openwebui and "/" not in model:
        return f"openai/{model}"
    return model


def _openrouter_row(item: dict) -> ModelCatalogRow | None:
    model_id = item.get("id")
    if not model_id:
        return None
    pricing = item.get("pricing") or {}
    supported = set(item.get("supported_parameters") or [])

    def format_price(value) -> str:
        if value is None:
            return "Unknown"
        try:
            return f"${float(value) * 1_000_000:.3g}"
        except (TypeError, ValueError):
            return "Unknown"

    tool_capable = "tools" in supported or "tool_choice" in supported
    structured = "structured_outputs" in supported or "response_format" in supported
    tools = "Yes" if tool_capable else "No"
    if structured and tool_capable:
        tools = "Yes + JSON"
    return ModelCatalogRow(
        model=f"openrouter/{model_id}",
        provider=(item.get("name") or model_id).split(":", 1)[0],
        source="OpenRouter",
        input_per_million=format_price(pricing.get("prompt")),
        output_per_million=format_price(pricing.get("completion")),
        context_window=str(
            (item.get("top_provider") or {}).get("context_length")
            or item.get("context_length")
            or "Unknown"
        ),
        tools=tools,
        cost_known=pricing.get("prompt") is not None or pricing.get("completion") is not None,
        tool_capable=tool_capable,
    )


def discover_openrouter_models(api_key: str = "", timeout_seconds: int = 20) -> list[ModelCatalogRow]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(OPENROUTER_MODELS_URL, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))
    rows = []
    for item in payload.get("data", []):
        row = _openrouter_row(item)
        if row:
            rows.append(row)
    return rows


def _row(model: str, provider: str, source: str) -> ModelCatalogRow:
    metadata = _metadata_for(model)
    input_cost = metadata.get("input_cost_per_token")
    output_cost = metadata.get("output_cost_per_token")
    tool_capable = model_supports_tools(model)
    return ModelCatalogRow(
        model=model,
        provider=provider or metadata.get("litellm_provider", ""),
        source=source,
        input_per_million=_format_cost(input_cost),
        output_per_million=_format_cost(output_cost),
        context_window=str(metadata.get("max_input_tokens") or metadata.get("max_tokens") or "Unknown"),
        tools="Yes" if tool_capable else "Unknown",
        cost_known=input_cost is not None or output_cost is not None,
        tool_capable=tool_capable,
    )


def codex_oauth_catalog(settings: dict[str, str]) -> list[ModelCatalogRow]:
    if settings.get("CSES_USE_CODEX_OAUTH", "false").lower() not in {"1", "true", "yes"}:
        return []
    status = CodexOAuthAuthSource().check()
    if not status.signed_in:
        return []
    models = [
        ("openai-codex/gpt-5.4", "1050000"),
        ("openai-codex/gpt-5.3-codex", "Unknown"),
        ("openai-codex/gpt-5.4-mini", "Unknown"),
        ("openai-codex/gpt-5.3-codex-spark", "Unknown"),
    ]
    return [
        ModelCatalogRow(
            model=model,
            provider="OpenAI Codex OAuth",
            source="Codex OAuth",
            input_per_million="Included/plan-limited",
            output_per_million="Included/plan-limited",
            context_window=context,
            tools="Codex",
            cost_known=False,
            tool_capable=True,
        )
        for model, context in models
    ]


def litellm_catalog_for_configured_keys(settings: dict[str, str]) -> list[ModelCatalogRow]:
    rows: list[ModelCatalogRow] = []
    provider_keys = {
        "OpenAI": ("OPENAI_API_KEY", "CSES_USE_OPENAI"),
        "Anthropic": ("ANTHROPIC_API_KEY", "CSES_USE_ANTHROPIC"),
        "Google Gemini": ("GEMINI_API_KEY", "CSES_USE_GEMINI"),
        "xAI": ("XAI_API_KEY", "CSES_USE_XAI"),
    }

    for provider, (key_name, toggle_name) in provider_keys.items():
        if settings.get(toggle_name, "false").lower() not in {"1", "true", "yes"}:
            continue
        if not settings.get(key_name):
            continue
        attr = KNOWN_PROVIDER_LISTS[provider]
        for model in sorted(getattr(litellm, attr, []))[:120]:
            rows.append(_row(model, provider, provider))

    return rows


def _probe_openai_compatible_tool_support(
    api_base: str,
    api_key: str,
    model: str,
    timeout_seconds: int = 20,
) -> bool:
    base = api_base.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": "Call the tool get_marker with value 'tool-ok'. Do not answer directly.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_marker",
                    "description": "Return a marker value for a tool capability probe.",
                    "parameters": {
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                },
            }
        ],
        "tool_choice": "auto",
        "max_tokens": 128,
        "temperature": 1,
    }
    request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            result = json.loads(response.read().decode("utf-8", errors="replace"))
    except (OSError, urllib.error.HTTPError, json.JSONDecodeError):
        return False
    if not isinstance(result, dict):
        return False
    choices = result.get("choices") or []
    if not choices:
        return False
    first_choice = choices[0] or {}
    if not isinstance(first_choice, dict):
        return False
    message = first_choice.get("message") or {}
    if not isinstance(message, dict):
        return False
    return bool(message.get("tool_calls"))


def discover_openai_compatible_models(
    api_base: str,
    api_key: str = "",
    timeout_seconds: int = 10,
    probe_tools: bool = False,
    provider_name: str | None = None,
    source_name: str | None = None,
) -> list[ModelCatalogRow]:
    """Fetch /models from an OpenAI-compatible API base if configured."""
    if not api_base:
        return []

    base = api_base.rstrip("/")
    url = base if base.endswith("/models") else f"{base}/models"
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8", errors="replace"))

    source = source_name or ("Local" if is_local_api_base(api_base) else "OpenAI-compatible API")
    provider = provider_name or ("Local" if source == "Local" else "Custom API")

    rows = []
    for item in payload.get("data", []):
        endpoint_model = item.get("id")
        if endpoint_model:
            catalog_model = litellm_model_for_completion(
                endpoint_model,
                {"CSES_USE_OPENWEBUI": "true", "OPENAI_API_BASE": api_base},
            )
            row = _row(catalog_model, provider, source)
            if probe_tools and not row.tool_capable:
                row.tool_capable = _probe_openai_compatible_tool_support(api_base, api_key, endpoint_model)
                row.tools = "Yes" if row.tool_capable else "No"
            rows.append(row)
    return rows


def build_model_catalog(settings: dict[str, str], probe_openai_compatible_tools: bool = False) -> tuple[list[ModelCatalogRow], list[str]]:
    """Return discovered model rows and non-fatal warnings."""
    warnings = []
    rows: list[ModelCatalogRow] = []
    try:
        rows.extend(discover_openrouter_models(settings.get("OPENROUTER_API_KEY", "")))
    except Exception as exc:
        warnings.append(f"Could not refresh OpenRouter models: {exc}")

    deduped = {}
    for row in rows:
        key = (row.model, row.provider, row.source)
        deduped[key] = row

    return sorted(deduped.values(), key=lambda row: (row.source, row.provider, row.model)), warnings
