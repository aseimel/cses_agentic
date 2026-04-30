"""
User settings for the CSES Assistant.

The GUI and CLI share the same .env file so non-programmer users can configure
the application once and then use either interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os


SUPPORTED_API_KEYS = {
    "OpenRouter": "OPENROUTER_API_KEY",
    "OpenAI": "OPENAI_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "xAI": "XAI_API_KEY",
    "Google Gemini": "GEMINI_API_KEY",
}

PROVIDER_TOGGLES = {
    "OpenRouter": "CSES_USE_OPENROUTER",
    "OpenAI": "CSES_USE_OPENAI",
    "Anthropic": "CSES_USE_ANTHROPIC",
    "xAI": "CSES_USE_XAI",
    "Google Gemini": "CSES_USE_GEMINI",
}

OPTIONAL_SETTINGS = {
    "Use OpenAI Codex OAuth": "CSES_USE_CODEX_OAUTH",
    "Codex OAuth profile": "CSES_CODEX_OAUTH_PROFILE",
    "Use GESIS OpenWebUI": "CSES_USE_GESIS",
    "GESIS API key": "GESIS_API_KEY",
    "Use OpenWebUI/OpenAI-compatible service": "CSES_USE_OPENWEBUI",
    "OpenAI API base URL": "OPENAI_API_BASE",
    "OpenWebUI/OpenAI-compatible API key": "OPENWEBUI_API_KEY",
    "Stata executable": "STATA_PATH",
    "Model profile": "CSES_MODEL_PROFILE",
    "Selected chat model": "CSES_CHAT_MODEL",
    "Available chat models": "CSES_CHAT_MODELS",
    "Agentic workflow model": "CSES_AGENTIC_MODEL",
    "Large text extraction model": "CSES_LARGE_TEXT_MODEL",
    "Study information review model": "CSES_STUDY_KB_MODEL",
    "Verifier model": "CSES_VERIFIER_MODEL",
    "Matching fast model": "CSES_MATCH_FAST_MODEL",
    "Matching escalation model": "CSES_MATCH_ESCALATION_MODEL",
    "Stata code generation model": "CSES_STATA_CODE_MODEL",
    "Stata repair model": "CSES_STATA_REPAIR_MODEL",
    "Documentation model": "CSES_DOCUMENTATION_MODEL",
    "Final escalation model": "CSES_FINAL_ESCALATION_MODEL",
    "Matching ensemble models": "CSES_MATCH_ENSEMBLE_MODELS",
    "Enable remote matching": "CSES_ENABLE_REMOTE_MATCHING",
    "Enable escalation": "CSES_ENABLE_ESCALATION",
    "Max escalation calls per step": "CSES_MAX_ESCALATION_CALLS_PER_STEP",
    "OpenRouter privacy mode": "CSES_OPENROUTER_PRIVACY_MODE",
}

GESIS_OPENWEBUI_BASE = "https://ai-openwebui.gesis.org/api/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_GESIS_CHAT_MODEL = "openai/gpt-oss:120b"
DEFAULT_AGENTIC_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_LARGE_TEXT_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_CODEX_AGENTIC_MODEL = "openai-codex/gpt-5.4"
DEFAULT_STUDY_KB_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_VERIFIER_MODEL = "openrouter/deepseek/deepseek-v4-flash"
DEFAULT_MATCH_FAST_MODEL = "openrouter/qwen/qwen-plus"
DEFAULT_MATCH_ESCALATION_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_STATA_CODE_MODEL = "openrouter/qwen/qwen3-coder-flash"
DEFAULT_STATA_REPAIR_MODEL = "openrouter/moonshotai/kimi-k2.5"
DEFAULT_DOCUMENTATION_MODEL = "openrouter/minimax/minimax-m2.5"
DEFAULT_FINAL_ESCALATION_MODEL = "openrouter/deepseek/deepseek-v4-pro"
DEFAULT_MATCH_ENSEMBLE_MODELS = ",".join([
    "openrouter/deepseek/deepseek-v4-flash",
    "openrouter/qwen/qwen-plus",
    "openrouter/minimax/minimax-m2.5",
])


@dataclass
class UserSettings:
    """Settings that users may edit from the GUI."""

    values: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: str = "") -> str:
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> None:
        self.values[key] = value.strip()

    def has_any_api_key(self) -> bool:
        return any(self.values.get(env_name, "").strip() for env_name in SUPPORTED_API_KEYS.values())


def get_install_dir(project_root: Path | None = None) -> Path:
    """Return the shared per-user settings directory."""
    override = os.environ.get("CSES_SETTINGS_DIR")
    if override:
        return Path(override).expanduser()
    install_dir = Path.home() / ".cses-agent"
    if install_dir.exists() or project_root is None:
        return install_dir
    return project_root


def get_env_path(project_root: Path | None = None) -> Path:
    return get_install_dir(project_root) / ".env"


def parse_env_file(path: Path) -> UserSettings:
    """Parse a simple KEY=value .env file without requiring python-dotenv."""
    settings = UserSettings()
    if not path.exists():
        return settings

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        settings.set(key.strip(), value)

    return settings


def read_openrouter_key_file(project_root: Path | None = None) -> str:
    """Read an OpenRouter key from local convenience files without logging it."""
    roots = []
    if project_root:
        roots.append(Path(project_root))
    roots.append(Path.cwd())
    roots.append(Path.home() / ".cses-agent")
    seen = set()
    for root in roots:
        try:
            root = root.resolve()
        except Exception:
            continue
        if root in seen:
            continue
        seen.add(root)
        for filename in ("openrouter.txt", "openrouter.txt.txt"):
            path = root / filename
            if not path.exists() or not path.is_file():
                continue
            try:
                value = path.read_text(encoding="utf-8-sig", errors="replace").strip()
            except Exception:
                continue
            if value.startswith("sk-or-v1-"):
                return value
    return ""


def load_settings(project_root: Path | None = None) -> UserSettings:
    effective_project_root = project_root or Path(__file__).resolve().parent.parent
    env_path = get_env_path(project_root)
    settings = parse_env_file(env_path)
    for key in [
        *SUPPORTED_API_KEYS.values(),
        *PROVIDER_TOGGLES.values(),
        *OPTIONAL_SETTINGS.values(),
        "CSES_SETUP_COMPLETE",
    ]:
        if key in os.environ and key not in settings.values:
            settings.set(key, os.environ[key])

    if not settings.get("OPENROUTER_API_KEY"):
        openrouter_key = read_openrouter_key_file(effective_project_root)
        if openrouter_key:
            settings.set("OPENROUTER_API_KEY", openrouter_key)
    if "CSES_USE_OPENROUTER" not in settings.values:
        settings.set("CSES_USE_OPENROUTER", "true" if settings.get("OPENROUTER_API_KEY") else "false")
    if "CSES_USE_GESIS" not in settings.values:
        settings.set("CSES_USE_GESIS", "false" if settings.get("OPENROUTER_API_KEY") else "true")

    if not settings.get("GESIS_API_KEY"):
        base_is_gesis = settings.get("OPENAI_API_BASE").rstrip("/") == GESIS_OPENWEBUI_BASE
        if base_is_gesis and settings.get("OPENWEBUI_API_KEY"):
            settings.set("GESIS_API_KEY", settings.get("OPENWEBUI_API_KEY"))
        elif base_is_gesis and settings.get("OPENAI_API_KEY"):
            settings.set("GESIS_API_KEY", settings.get("OPENAI_API_KEY"))

    for label, key_name in SUPPORTED_API_KEYS.items():
        if label == "OpenRouter":
            continue
        toggle = PROVIDER_TOGGLES[label]
        if toggle not in settings.values:
            settings.set(toggle, "false")

    base_is_gesis = settings.get("OPENAI_API_BASE").rstrip("/") == GESIS_OPENWEBUI_BASE
    use_gesis = settings.get("CSES_USE_GESIS", "true").lower() in {"1", "true", "yes"}
    if (
        settings.get("OPENAI_API_BASE")
        and not (base_is_gesis and use_gesis)
        and settings.get("CSES_USE_OPENWEBUI").lower() not in {"1", "true", "yes"}
    ):
        settings.set("CSES_USE_OPENWEBUI", "true")
    if not settings.get("CSES_CHAT_MODELS"):
        settings.set("CSES_CHAT_MODELS", DEFAULT_AGENTIC_MODEL)
    if not settings.get("CSES_CHAT_MODEL"):
        settings.set("CSES_CHAT_MODEL", DEFAULT_AGENTIC_MODEL)
    if not settings.get("CSES_AGENTIC_MODEL"):
        settings.set("CSES_AGENTIC_MODEL", settings.get("CSES_CHAT_MODEL") or DEFAULT_AGENTIC_MODEL)
    if not settings.get("CSES_LARGE_TEXT_MODEL"):
        settings.set("CSES_LARGE_TEXT_MODEL", DEFAULT_LARGE_TEXT_MODEL)
    defaults = {
        "CSES_STUDY_KB_MODEL": DEFAULT_STUDY_KB_MODEL,
        "CSES_VERIFIER_MODEL": DEFAULT_VERIFIER_MODEL,
        "CSES_MATCH_FAST_MODEL": DEFAULT_MATCH_FAST_MODEL,
        "CSES_MATCH_ESCALATION_MODEL": DEFAULT_MATCH_ESCALATION_MODEL,
        "CSES_STATA_CODE_MODEL": DEFAULT_STATA_CODE_MODEL,
        "CSES_STATA_REPAIR_MODEL": DEFAULT_STATA_REPAIR_MODEL,
        "CSES_DOCUMENTATION_MODEL": DEFAULT_DOCUMENTATION_MODEL,
        "CSES_FINAL_ESCALATION_MODEL": DEFAULT_FINAL_ESCALATION_MODEL,
        "CSES_MATCH_ENSEMBLE_MODELS": DEFAULT_MATCH_ENSEMBLE_MODELS,
        "CSES_ENABLE_REMOTE_MATCHING": "true",
        "CSES_ENABLE_ESCALATION": "true",
        "CSES_MAX_ESCALATION_CALLS_PER_STEP": "3",
        "CSES_OPENROUTER_PRIVACY_MODE": "strict",
    }
    for key, value in defaults.items():
        if not settings.get(key):
            settings.set(key, value)
    use_openrouter = settings.get("CSES_USE_OPENROUTER", "false").lower() in {"1", "true", "yes"}
    if use_openrouter:
        legacy_models = {
            "openai/gpt-oss:120b",
            "openai/gpt-oss:latest",
            "openai/llama4:latest",
            DEFAULT_GESIS_CHAT_MODEL,
        }
        replacements = {
            "CSES_CHAT_MODEL": DEFAULT_AGENTIC_MODEL,
            "CSES_AGENTIC_MODEL": DEFAULT_AGENTIC_MODEL,
            "CSES_LARGE_TEXT_MODEL": DEFAULT_LARGE_TEXT_MODEL,
        }
        for key, replacement in replacements.items():
            if settings.get(key) in legacy_models:
                settings.set(key, replacement)
        chat_models = [
            model.strip()
            for model in (settings.get("CSES_CHAT_MODELS") or "").split(",")
            if model.strip() and model.strip() not in legacy_models
        ]
        for model in [
            DEFAULT_AGENTIC_MODEL,
            DEFAULT_STUDY_KB_MODEL,
            DEFAULT_VERIFIER_MODEL,
            DEFAULT_MATCH_FAST_MODEL,
            DEFAULT_STATA_CODE_MODEL,
            DEFAULT_STATA_REPAIR_MODEL,
            DEFAULT_DOCUMENTATION_MODEL,
            DEFAULT_FINAL_ESCALATION_MODEL,
        ]:
            if model not in chat_models:
                chat_models.append(model)
        settings.set("CSES_CHAT_MODELS", ",".join(chat_models))
    if "CSES_USE_CODEX_OAUTH" not in settings.values:
        settings.set("CSES_USE_CODEX_OAUTH", "false")
    return settings


def apply_settings_to_environment(project_root: Path | None = None) -> UserSettings:
    """Load saved GUI/CLI settings and expose them to LiteLLM."""
    settings = load_settings(project_root)
    for key, value in settings.values.items():
        if value:
            os.environ[key] = value

    use_openrouter = settings.get("CSES_USE_OPENROUTER", "false").lower() in {"1", "true", "yes"}
    if use_openrouter and settings.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = settings.get("OPENROUTER_API_KEY")
        os.environ.pop("OPENAI_API_BASE", None)
        os.environ.pop("OPENAI_BASE_URL", None)

    use_gesis = settings.get("CSES_USE_GESIS", "true").lower() in {"1", "true", "yes"}
    if use_gesis and not use_openrouter:
        os.environ["OPENAI_API_BASE"] = GESIS_OPENWEBUI_BASE
        gesis_key = settings.get("GESIS_API_KEY") or settings.get("OPENWEBUI_API_KEY") or settings.get("OPENAI_API_KEY")
        if gesis_key:
            os.environ["GESIS_API_KEY"] = gesis_key
            os.environ["OPENWEBUI_API_KEY"] = gesis_key
            os.environ["OPENAI_API_KEY"] = gesis_key
    elif settings.get("CSES_USE_OPENWEBUI", "").lower() in {"1", "true", "yes"}:
        if settings.get("OPENAI_API_BASE"):
            os.environ["OPENAI_API_BASE"] = settings.get("OPENAI_API_BASE")
        openwebui_key = settings.get("OPENWEBUI_API_KEY") or settings.get("OPENAI_API_KEY")
        if openwebui_key:
            os.environ["OPENWEBUI_API_KEY"] = openwebui_key
            os.environ["OPENAI_API_KEY"] = openwebui_key

    return settings


def save_settings(settings: UserSettings, project_root: Path | None = None) -> Path:
    """Save user-editable settings to .env."""
    env_path = get_env_path(project_root)
    env_path.parent.mkdir(parents=True, exist_ok=True)

    keys = [
        "CSES_USE_GESIS",
        "CSES_USE_OPENROUTER",
        "OPENROUTER_API_KEY",
        "CSES_USE_CODEX_OAUTH",
        "CSES_CODEX_OAUTH_PROFILE",
        "GESIS_API_KEY",
        "CSES_USE_OPENAI",
        "OPENAI_API_KEY",
        "CSES_USE_ANTHROPIC",
        "ANTHROPIC_API_KEY",
        "CSES_USE_XAI",
        "XAI_API_KEY",
        "CSES_USE_GEMINI",
        "GEMINI_API_KEY",
        "CSES_USE_OPENWEBUI",
        "OPENAI_API_BASE",
        "OPENWEBUI_API_KEY",
        "STATA_PATH",
        "CSES_MODEL_PROFILE",
        "CSES_CHAT_MODEL",
        "CSES_CHAT_MODELS",
        "CSES_AGENTIC_MODEL",
        "CSES_LARGE_TEXT_MODEL",
        "CSES_STUDY_KB_MODEL",
        "CSES_VERIFIER_MODEL",
        "CSES_MATCH_FAST_MODEL",
        "CSES_MATCH_ESCALATION_MODEL",
        "CSES_STATA_CODE_MODEL",
        "CSES_STATA_REPAIR_MODEL",
        "CSES_DOCUMENTATION_MODEL",
        "CSES_FINAL_ESCALATION_MODEL",
        "CSES_MATCH_ENSEMBLE_MODELS",
        "CSES_ENABLE_REMOTE_MATCHING",
        "CSES_ENABLE_ESCALATION",
        "CSES_MAX_ESCALATION_CALLS_PER_STEP",
        "CSES_OPENROUTER_PRIVACY_MODE",
    ]

    lines = [
        "# CSES Assistant Configuration",
        "# Generated by the Windows GUI",
        "",
        "# OpenRouter is the supported AI backend for this build.",
        "CSES_USE_OPENROUTER=true",
    ]
    openrouter_key = settings.get("OPENROUTER_API_KEY")
    if openrouter_key:
        lines.append(f"OPENROUTER_API_KEY={openrouter_key}")
    for key in [
        "CSES_USE_GESIS",
        "CSES_USE_OPENWEBUI",
        "CSES_USE_CODEX_OAUTH",
        "CSES_USE_OPENAI",
        "CSES_USE_ANTHROPIC",
        "CSES_USE_XAI",
        "CSES_USE_GEMINI",
    ]:
        lines.append(f"{key}=false")

    lines.extend(["", "# Model profile"])
    profile = settings.get("CSES_MODEL_PROFILE")
    if profile:
        lines.append(f"CSES_MODEL_PROFILE={profile}")
    chat_model = settings.get("CSES_CHAT_MODEL")
    if chat_model:
        lines.append(f"CSES_CHAT_MODEL={chat_model}")
    chat_models = settings.get("CSES_CHAT_MODELS")
    if chat_models:
        lines.append(f"CSES_CHAT_MODELS={chat_models}")
    agentic_model = settings.get("CSES_AGENTIC_MODEL")
    if agentic_model:
        lines.append(f"CSES_AGENTIC_MODEL={agentic_model}")
    large_text_model = settings.get("CSES_LARGE_TEXT_MODEL")
    if large_text_model:
        lines.append(f"CSES_LARGE_TEXT_MODEL={large_text_model}")
    for key in [
        "CSES_STUDY_KB_MODEL",
        "CSES_VERIFIER_MODEL",
        "CSES_MATCH_FAST_MODEL",
        "CSES_MATCH_ESCALATION_MODEL",
        "CSES_STATA_CODE_MODEL",
        "CSES_STATA_REPAIR_MODEL",
        "CSES_DOCUMENTATION_MODEL",
        "CSES_FINAL_ESCALATION_MODEL",
        "CSES_MATCH_ENSEMBLE_MODELS",
        "CSES_ENABLE_REMOTE_MATCHING",
        "CSES_ENABLE_ESCALATION",
        "CSES_MAX_ESCALATION_CALLS_PER_STEP",
        "CSES_OPENROUTER_PRIVACY_MODE",
    ]:
        value = settings.get(key)
        if value:
            lines.append(f"{key}={value}")

    lines.extend(["", "# Stata integration"])
    stata_path = settings.get("STATA_PATH")
    if stata_path:
        lines.append(f"STATA_PATH={stata_path}")

    lines.extend(["", "CSES_SETUP_COMPLETE=true", ""])
    env_path.write_text("\n".join(lines), encoding="utf-8")
    return env_path
