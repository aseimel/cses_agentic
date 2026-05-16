"""Role-aware model runtime for CSES workflow tasks."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import time
from typing import Any

from src.model_router import completion
from src.settings import (
    DEFAULT_AGENTIC_MODEL,
    DEFAULT_DOCUMENTATION_MODEL,
    DEFAULT_FINAL_ESCALATION_MODEL,
    DEFAULT_LARGE_TEXT_MODEL,
    DEFAULT_MATCH_ESCALATION_MODEL,
    DEFAULT_MATCH_FAST_MODEL,
    DEFAULT_STATA_CODE_MODEL,
    DEFAULT_STATA_REPAIR_MODEL,
    DEFAULT_STUDY_KB_MODEL,
    DEFAULT_VERIFIER_MODEL,
    DEFAULT_MATCH_ENSEMBLE_MODELS,
    apply_settings_to_environment,
)


class ModelRole(str, Enum):
    STUDY_KB = "study_kb"
    LARGE_TEXT = "large_text"
    AGENTIC = "agentic"
    VERIFIER = "verifier"
    MATCH_FAST = "match_fast"
    MATCH_ENSEMBLE = "match_ensemble"
    MATCH_ESCALATION = "match_escalation"
    RECODE_CODEGEN = "recode_codegen"
    STATA_REPAIR = "stata_repair"
    DOCUMENTATION = "documentation"
    FINAL_ESCALATION = "final_escalation"


ROLE_ENV = {
    ModelRole.STUDY_KB: "CSES_STUDY_KB_MODEL",
    ModelRole.LARGE_TEXT: "CSES_LARGE_TEXT_MODEL",
    ModelRole.AGENTIC: "CSES_AGENTIC_MODEL",
    ModelRole.VERIFIER: "CSES_VERIFIER_MODEL",
    ModelRole.MATCH_FAST: "CSES_MATCH_FAST_MODEL",
    ModelRole.MATCH_ESCALATION: "CSES_MATCH_ESCALATION_MODEL",
    ModelRole.RECODE_CODEGEN: "CSES_STATA_CODE_MODEL",
    ModelRole.STATA_REPAIR: "CSES_STATA_REPAIR_MODEL",
    ModelRole.DOCUMENTATION: "CSES_DOCUMENTATION_MODEL",
    ModelRole.FINAL_ESCALATION: "CSES_FINAL_ESCALATION_MODEL",
}

ROLE_DEFAULT = {
    ModelRole.STUDY_KB: DEFAULT_STUDY_KB_MODEL,
    ModelRole.LARGE_TEXT: DEFAULT_LARGE_TEXT_MODEL,
    ModelRole.AGENTIC: DEFAULT_AGENTIC_MODEL,
    ModelRole.VERIFIER: DEFAULT_VERIFIER_MODEL,
    ModelRole.MATCH_FAST: DEFAULT_MATCH_FAST_MODEL,
    ModelRole.MATCH_ESCALATION: DEFAULT_MATCH_ESCALATION_MODEL,
    ModelRole.RECODE_CODEGEN: DEFAULT_STATA_CODE_MODEL,
    ModelRole.STATA_REPAIR: DEFAULT_STATA_REPAIR_MODEL,
    ModelRole.DOCUMENTATION: DEFAULT_DOCUMENTATION_MODEL,
    ModelRole.FINAL_ESCALATION: DEFAULT_FINAL_ESCALATION_MODEL,
}


@dataclass
class ModelTaskResult:
    role: str
    model: str
    provider: str
    status: str
    content: str
    input_tokens_estimate: int
    output_tokens_estimate: int
    elapsed_seconds: float
    retries: int
    escalated: bool = False
    error: str = ""


class EscalationPolicy:
    """Small policy object for controlled escalation."""

    def __init__(self, working_dir: Path | None = None):
        settings = apply_settings_to_environment()
        self.enabled = settings.get("CSES_ENABLE_ESCALATION", "true").lower() in {"1", "true", "yes"}
        try:
            self.max_per_step = int(settings.get("CSES_MAX_ESCALATION_CALLS_PER_STEP", "3"))
        except ValueError:
            self.max_per_step = 3
        self.working_dir = Path(working_dir) if working_dir else None

    def should_escalate(self, reason: str, current_attempts: int = 0) -> bool:
        if not self.enabled or current_attempts >= self.max_per_step:
            return False
        lowered = (reason or "").lower()
        triggers = [
            "contradict",
            "quorum",
            "not_found",
            "required",
            "stata",
            "repair",
            "final readiness",
            "parse error",
            "low confidence",
        ]
        return any(trigger in lowered for trigger in triggers)


class ModelTaskRunner:
    """Single entry point for all role-specific model calls."""

    def __init__(self, working_dir: Path | None = None, state=None):
        self.working_dir = Path(working_dir) if working_dir else None
        self.state = state
        self.settings = apply_settings_to_environment()
        self.escalation_policy = EscalationPolicy(self.working_dir)

    def model_for_role(self, role: ModelRole | str) -> str:
        role = self._role(role)
        if role == ModelRole.MATCH_ENSEMBLE:
            return self.settings.get("CSES_MATCH_ENSEMBLE_MODELS") or os.getenv("CSES_MATCH_ENSEMBLE_MODELS") or DEFAULT_MATCH_ENSEMBLE_MODELS
        env_name = ROLE_ENV.get(role)
        default = ROLE_DEFAULT.get(role, DEFAULT_AGENTIC_MODEL)
        return os.getenv(env_name or "", "") or self.settings.get(env_name or "", "") or default

    def models_for_ensemble(self) -> list[str]:
        raw = self.model_for_role(ModelRole.MATCH_ENSEMBLE)
        return [item.strip() for item in raw.split(",") if item.strip()]

    def complete(
        self,
        role: ModelRole | str,
        messages: list[dict[str, Any]],
        purpose: str = "",
        include_shared_context: bool = True,
        max_tokens: int = 2048,
        temperature: float = 1,
        timeout: int = 120,
        retries: int = 1,
        escalate_on: str = "",
        model_override: str = "",
        **kwargs: Any,
    ) -> ModelTaskResult:
        role = self._role(role)
        model = model_override or self.model_for_role(role)
        kwargs.pop("drop_params", None)
        request_temperature = self._safe_temperature(model, temperature)
        escalated = False
        if escalate_on and self.escalation_policy.should_escalate(escalate_on):
            model = self.model_for_role(self._escalation_role(role))
            request_temperature = self._safe_temperature(model, temperature)
            escalated = True

        final_messages = self._with_shared_context(role, messages, purpose, include_shared_context)
        started = time.time()
        last_error = ""
        attempts = max(1, retries + 1)
        for attempt in range(1, attempts + 1):
            try:
                response = completion(
                    model=model,
                    messages=final_messages,
                    max_tokens=max_tokens,
                    temperature=request_temperature,
                    timeout=timeout,
                    drop_params=True,
                    **kwargs,
                )
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise RuntimeError("Model returned an empty content response")
                result = ModelTaskResult(
                    role=role.value,
                    model=model,
                    provider=self._provider(model),
                    status="ok",
                    content=content,
                    input_tokens_estimate=self._estimate_tokens(final_messages),
                    output_tokens_estimate=max(1, len(content) // 4),
                    elapsed_seconds=round(time.time() - started, 2),
                    retries=attempt - 1,
                    escalated=escalated,
                )
                self._record_result(result, purpose)
                return result
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                time.sleep(min(8, attempt * 2))

        result = ModelTaskResult(
            role=role.value,
            model=model,
            provider=self._provider(model),
            status="failed",
            content="",
            input_tokens_estimate=self._estimate_tokens(final_messages),
            output_tokens_estimate=0,
            elapsed_seconds=round(time.time() - started, 2),
            retries=attempts - 1,
            escalated=escalated,
            error=last_error,
        )
        self._record_result(result, purpose)
        return result

    def content(self, role: ModelRole | str, messages: list[dict[str, Any]], **kwargs: Any) -> str:
        result = self.complete(role, messages, **kwargs)
        if result.status != "ok":
            raise RuntimeError(result.error or f"Model task failed for role {role}")
        return result.content

    def response(
        self,
        role: ModelRole | str,
        messages: list[dict[str, Any]],
        purpose: str = "",
        include_shared_context: bool = True,
        max_tokens: int = 2048,
        temperature: float = 1,
        timeout: int = 120,
        model_override: str = "",
        **kwargs: Any,
    ):
        """Return the raw provider response while still recording diagnostics."""
        role = self._role(role)
        model = model_override or self.model_for_role(role)
        kwargs.pop("drop_params", None)
        request_temperature = self._safe_temperature(model, temperature)
        final_messages = self._with_shared_context(role, messages, purpose, include_shared_context)
        started = time.time()
        response = completion(
            model=model,
            messages=final_messages,
            max_tokens=max_tokens,
            temperature=request_temperature,
            timeout=timeout,
            drop_params=True,
            **kwargs,
        )
        content = getattr(response.choices[0].message, "content", "") or ""
        if not content.strip():
            raise RuntimeError(f"Model task returned an empty content response for role {role.value}")
        self._record_result(
            ModelTaskResult(
                role=role.value,
                model=model,
                provider=self._provider(model),
                status="ok",
                content=content,
                input_tokens_estimate=self._estimate_tokens(final_messages),
                output_tokens_estimate=max(1, len(content) // 4),
                elapsed_seconds=round(time.time() - started, 2),
                retries=0,
            ),
            purpose,
        )
        return response

    def record_handoff(self, role: ModelRole | str, summary: str, artifacts: list[str] | None = None, data: dict | None = None) -> None:
        if not self.working_dir:
            return
        from src.shared_context import SharedWorkflowContext

        SharedWorkflowContext(self.working_dir, self.state).record_handoff(
            self._role(role).value,
            summary,
            artifacts=artifacts,
            data=data,
        )

    def _with_shared_context(
        self,
        role: ModelRole,
        messages: list[dict[str, Any]],
        purpose: str,
        include_shared_context: bool,
    ) -> list[dict[str, Any]]:
        if not include_shared_context or not self.working_dir:
            return messages
        try:
            from src.shared_context import SharedWorkflowContext

            prefix = SharedWorkflowContext(self.working_dir, self.state).prompt_prefix(role.value, purpose=purpose)
        except Exception:
            return messages
        if not prefix:
            return messages
        final = list(messages)
        for index, message in enumerate(final):
            if message.get("role") == "user":
                message = message.copy()
                message["content"] = prefix + str(message.get("content", ""))
                final[index] = message
                return final
        return [{"role": "user", "content": prefix}, *final]

    def _record_result(self, result: ModelTaskResult, purpose: str) -> None:
        if not self.working_dir:
            return
        cses_dir = self.working_dir / ".cses"
        cses_dir.mkdir(parents=True, exist_ok=True)
        diag_path = cses_dir / "model_diagnostics.json"
        try:
            existing = json.loads(diag_path.read_text(encoding="utf-8")) if diag_path.exists() else {}
        except Exception:
            existing = {}
        calls = existing.setdefault("calls", [])
        calls.append(
            {
                **asdict(result),
                "purpose": purpose,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        existing["calls"] = calls[-300:]
        existing["updated_at"] = datetime.now(timezone.utc).isoformat()
        diag_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")

        cost_path = cses_dir / "cost_diagnostics.json"
        try:
            cost = json.loads(cost_path.read_text(encoding="utf-8")) if cost_path.exists() else {}
        except Exception:
            cost = {}
        by_role = cost.setdefault("by_role", {})
        item = by_role.setdefault(result.role, {"input_tokens_estimate": 0, "output_tokens_estimate": 0, "calls": 0})
        item["input_tokens_estimate"] += result.input_tokens_estimate
        item["output_tokens_estimate"] += result.output_tokens_estimate
        item["calls"] += 1
        cost["updated_at"] = datetime.now(timezone.utc).isoformat()
        cost_path.write_text(json.dumps(cost, indent=2, ensure_ascii=False), encoding="utf-8")

        if result.status == "ok" and result.content:
            try:
                self.record_handoff(
                    result.role,
                    result.content[:1500],
                    data={
                        "purpose": purpose,
                        "model": result.model,
                        "provider": result.provider,
                        "input_tokens_estimate": result.input_tokens_estimate,
                        "output_tokens_estimate": result.output_tokens_estimate,
                        "elapsed_seconds": result.elapsed_seconds,
                        "escalated": result.escalated,
                    },
                )
            except Exception:
                pass

    def _estimate_tokens(self, messages: list[dict[str, Any]]) -> int:
        chars = sum(len(str(message.get("content", ""))) for message in messages)
        return max(1, chars // 4)

    def _provider(self, model: str) -> str:
        return model.split("/", 1)[0] if "/" in model else "unknown"

    def _safe_temperature(self, model: str, temperature: float) -> float:
        if model.startswith("openrouter/deepseek/") and temperature <= 0:
            return 0.1
        return temperature

    def _role(self, role: ModelRole | str) -> ModelRole:
        if isinstance(role, ModelRole):
            return role
        return ModelRole(str(role))

    def _escalation_role(self, role: ModelRole) -> ModelRole:
        if role in {ModelRole.MATCH_FAST, ModelRole.MATCH_ENSEMBLE}:
            return ModelRole.MATCH_ESCALATION
        if role in {ModelRole.RECODE_CODEGEN, ModelRole.STATA_REPAIR}:
            return ModelRole.STATA_REPAIR
        return ModelRole.FINAL_ESCALATION
