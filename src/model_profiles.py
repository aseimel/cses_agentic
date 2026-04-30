"""
Curated model profiles for non-technical users.

The GUI exposes profile names, not arbitrary model strings. Advanced model
details stay here so the application remains supportable.
"""

from __future__ import annotations

from dataclasses import dataclass


DEFAULT_PROFILE_ID = "openrouter_cost_efficient"


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    description: str
    required_key: str
    conversation_model: str
    match_model: str
    preprocess_model: str
    recode_model: str
    code_model: str
    quality_model: str
    ensemble_models: tuple[str, str, str]


MODEL_PROFILES: dict[str, ModelProfile] = {
    "openrouter_cost_efficient": ModelProfile(
        id="openrouter_cost_efficient",
        label="OpenRouter cost-efficient",
        description="Recommended OpenRouter setup: cheap large-context extraction, structured verification, and coding-specific escalation.",
        required_key="OPENROUTER_API_KEY",
        conversation_model="openrouter/deepseek/deepseek-v4-flash",
        match_model="openrouter/qwen/qwen-plus",
        preprocess_model="openrouter/deepseek/deepseek-v4-flash",
        recode_model="openrouter/qwen/qwen3-coder-flash",
        code_model="openrouter/qwen/qwen3-coder-flash",
        quality_model="openrouter/deepseek/deepseek-v4-flash",
        ensemble_models=(
            "openrouter/deepseek/deepseek-v4-flash",
            "openrouter/qwen/qwen-plus",
            "openrouter/minimax/minimax-m2.5",
        ),
    ),
    "gesis_recommended": ModelProfile(
        id="gesis_recommended",
        label="CSES recommended",
        description="Recommended tested setup for the CSES workflow through an OpenAI-compatible endpoint.",
        required_key="OPENAI_API_KEY",
        conversation_model="openai/gpt-oss:120b",
        match_model="openai/gpt-oss:120b",
        preprocess_model="openai/llama4:latest",
        recode_model="openai/gpt-oss:120b",
        code_model="openai/gpt-oss:120b",
        quality_model="openai/gpt-oss:120b",
        ensemble_models=("openai/gpt-oss:120b", "openai/gpt-oss:latest", "openai/gpt-4.1-mini"),
    ),
    "codex_oauth_recommended": ModelProfile(
        id="codex_oauth_recommended",
        label="Codex OAuth recommended",
        description="Uses ChatGPT/Codex sign-in for agentic workflow reasoning and GESIS llama for large deposited-file extraction.",
        required_key="Codex sign-in",
        conversation_model="openai-codex/gpt-5.4",
        match_model="openai-codex/gpt-5.4",
        preprocess_model="openai/llama4:latest",
        recode_model="openai-codex/gpt-5.4",
        code_model="openai-codex/gpt-5.4",
        quality_model="openai-codex/gpt-5.4",
        ensemble_models=("openai-codex/gpt-5.4", "openai-codex/gpt-5.3-codex", "openai/gpt-oss:120b"),
    ),
    "openai_balanced": ModelProfile(
        id="openai_balanced",
        label="OpenAI balanced",
        description="Direct OpenAI profile for chat and workflow reasoning.",
        required_key="OPENAI_API_KEY",
        conversation_model="openai/gpt-4.1",
        match_model="openai/gpt-4.1",
        preprocess_model="openai/gpt-4.1-mini",
        recode_model="openai/gpt-4.1",
        code_model="openai/gpt-4.1",
        quality_model="openai/gpt-4.1-mini",
        ensemble_models=("openai/gpt-4.1", "openai/gpt-4.1-mini", "openai/o4-mini"),
    ),
    "anthropic_balanced": ModelProfile(
        id="anthropic_balanced",
        label="Anthropic balanced",
        description="Direct Anthropic profile for teams using Claude API keys.",
        required_key="ANTHROPIC_API_KEY",
        conversation_model="anthropic/claude-sonnet-4-20250514",
        match_model="anthropic/claude-sonnet-4-20250514",
        preprocess_model="anthropic/claude-sonnet-4-20250514",
        recode_model="anthropic/claude-sonnet-4-20250514",
        code_model="anthropic/claude-sonnet-4-20250514",
        quality_model="anthropic/claude-sonnet-4-20250514",
        ensemble_models=(
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-sonnet-4-20250514",
        ),
    ),
    "google_balanced": ModelProfile(
        id="google_balanced",
        label="Google balanced",
        description="Direct Google Gemini profile.",
        required_key="GEMINI_API_KEY",
        conversation_model="gemini/gemini-2.5-pro",
        match_model="gemini/gemini-2.5-pro",
        preprocess_model="gemini/gemini-2.5-flash",
        recode_model="gemini/gemini-2.5-pro",
        code_model="gemini/gemini-2.5-pro",
        quality_model="gemini/gemini-2.5-flash",
        ensemble_models=("gemini/gemini-2.5-pro", "gemini/gemini-2.5-flash", "gemini/gemini-2.5-pro"),
    ),
    "xai_balanced": ModelProfile(
        id="xai_balanced",
        label="xAI balanced",
        description="Direct xAI profile for teams using Grok API keys.",
        required_key="XAI_API_KEY",
        conversation_model="xai/grok-4",
        match_model="xai/grok-4",
        preprocess_model="xai/grok-3-mini",
        recode_model="xai/grok-4",
        code_model="xai/grok-4",
        quality_model="xai/grok-3-mini",
        ensemble_models=("xai/grok-4", "xai/grok-3-mini", "xai/grok-4"),
    ),
    "xai_grok_4_1_fast": ModelProfile(
        id="xai_grok_4_1_fast",
        label="xAI Grok 4.1 Fast",
        description="Fast xAI profile using Grok 4.1 Fast for chat and workflow tasks.",
        required_key="XAI_API_KEY",
        conversation_model="xai/grok-4-1-fast",
        match_model="xai/grok-4-1-fast",
        preprocess_model="xai/grok-4-1-fast",
        recode_model="xai/grok-4-1-fast",
        code_model="xai/grok-4-1-fast",
        quality_model="xai/grok-4-1-fast",
        ensemble_models=("xai/grok-4-1-fast", "xai/grok-4-1-fast-non-reasoning", "xai/grok-4-1-fast"),
    ),
}


def get_profile(profile_id: str | None) -> ModelProfile:
    return MODEL_PROFILES.get(profile_id or "", MODEL_PROFILES[DEFAULT_PROFILE_ID])


def get_profile_choices() -> list[tuple[str, str]]:
    return [(profile.id, profile.label) for profile in MODEL_PROFILES.values()]
