"""Shared model context for divided CSES workflow tasks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from src.study_kb import StudyKnowledgeBase


@dataclass
class ContextPacket:
    role: str
    json_context: dict[str, Any]
    toon_context: str


def _safe_text(value: Any, limit: int | None = None) -> str:
    text = " ".join(str(value or "").split())
    text = text.encode("cp1252", errors="replace").decode("cp1252")
    return text[:limit] if limit else text


class ModelHandoffStore:
    """Persist compact model-to-model handoffs."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.handoff_dir = self.working_dir / ".cses" / "handoffs"
        self.handoff_dir.mkdir(parents=True, exist_ok=True)

    def write(self, role: str, payload: dict[str, Any]) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = self.handoff_dir / f"{timestamp}_{role}.json"
        payload = {
            "role": role,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def recent(self, limit: int = 8, role: str | None = None) -> list[dict[str, Any]]:
        paths = sorted(self.handoff_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        items = []
        for path in paths:
            if role and not path.name.endswith(f"_{role}.json"):
                continue
            try:
                items.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
            if len(items) >= limit:
                break
        return items


class SharedWorkflowContext:
    """Build task-specific context packets from shared study memory."""

    def __init__(self, working_dir: Path, state=None):
        self.working_dir = Path(working_dir)
        self.state = state
        self.cses_dir = self.working_dir / ".cses"
        self.json_path = self.cses_dir / "shared_context.json"
        self.toon_path = self.cses_dir / "shared_context.toon"
        self.handoffs = ModelHandoffStore(self.working_dir)

    def build_packet(self, role: str, purpose: str = "", max_chars: int = 12000) -> ContextPacket:
        payload = self._base_payload(role, purpose)
        toon = self._to_toon(payload)
        self._write(payload, toon)
        return ContextPacket(role=role, json_context=payload, toon_context=toon[:max_chars])

    def prompt_prefix(self, role: str, purpose: str = "", max_chars: int = 12000) -> str:
        packet = self.build_packet(role=role, purpose=purpose, max_chars=max_chars)
        if not packet.toon_context.strip():
            return ""
        return (
            "SHARED CSES WORKFLOW CONTEXT\n"
            "Use this compact shared context for consistency. Do not invent facts beyond it or the task input.\n"
            f"{packet.toon_context}\n"
            "END SHARED CSES WORKFLOW CONTEXT\n\n"
        )

    def record_handoff(self, role: str, summary: str, artifacts: list[str] | None = None, data: dict | None = None) -> Path:
        return self.handoffs.write(
            role,
            {
                "summary": _safe_text(summary, 2500),
                "artifacts": artifacts or [],
                "data": data or {},
            },
        )

    def _base_payload(self, role: str, purpose: str) -> dict[str, Any]:
        kb = StudyKnowledgeBase(self.working_dir)
        kb_payload = kb.payload if kb.exists() else {}
        state_payload = self._state_payload()
        standards = state_payload.get("standards_checks", {})
        recent_handoffs = self.handoffs.recent(limit=8)
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "purpose": purpose,
            "study": {
                "country": state_payload.get("country"),
                "country_code": state_payload.get("country_code"),
                "year": state_payload.get("year"),
                "working_dir": state_payload.get("working_dir"),
            },
            "study_kb": {
                "status": kb.status().status,
                "summary": kb_payload.get("summary", ""),
                "fields": self._role_fields(role, kb_payload.get("fields", {})),
                "missing_fields": kb_payload.get("missing_fields", [])[:80],
                "contradictions": kb_payload.get("contradictions", [])[:40],
                "processor_review_items": kb_payload.get("processor_review_items", [])[:40],
                "diagnostics": kb_payload.get("diagnostics", {}),
            },
            "workflow": {
                "current_step": state_payload.get("current_step"),
                "standards_checks": standards,
                "processor_decisions": state_payload.get("processor_decisions", [])[-12:],
                "pending_corrections": state_payload.get("pending_corrections", [])[-20:],
                "work_needing_rerun": state_payload.get("invalidated_outputs", [])[-20:],
                "targeted_reruns": state_payload.get("targeted_rerun_queue", [])[-10:],
                "candidate_questions": state_payload.get("candidate_collaborator_questions", [])[-20:],
                "pending_questions": state_payload.get("collaborator_questions", [])[-20:],
                "final_readiness": state_payload.get("final_readiness", {}),
            },
            "tracking": {
                "target_count": (state_payload.get("workflow_tracking") or {}).get("target_count"),
                "tracking_file": state_payload.get("variable_tracking_file"),
                "data_file": state_payload.get("data_file"),
            },
            "recent_handoffs": recent_handoffs,
        }

    def _role_fields(self, role: str, fields: dict[str, Any]) -> dict[str, Any]:
        role_groups = {
            "study_kb": None,
            "large_text": None,
            "verifier": None,
            "agentic": None,
            "match_fast": ["cses_item_coverage", "variable_label_evidence", "missing_value_codes", "party_vote_evidence"],
            "match_ensemble": ["cses_item_coverage", "variable_label_evidence", "missing_value_codes", "party_vote_evidence"],
            "match_escalation": None,
            "recode_codegen": ["missing_value_codes", "variable_label_evidence", "party_vote_evidence", "weights"],
            "stata_repair": ["missing_value_codes", "variable_label_evidence", "party_vote_evidence", "weights"],
            "documentation": None,
            "final_escalation": None,
        }
        wanted = role_groups.get(role)
        if wanted is None:
            return fields
        return {key: fields.get(key, []) for key in wanted if key in fields}

    def _state_payload(self) -> dict[str, Any]:
        if self.state is None:
            state_path = self.working_dir / ".cses" / "state.json"
            if state_path.exists():
                try:
                    return json.loads(state_path.read_text(encoding="utf-8"))
                except Exception:
                    return {}
            return {}
        if hasattr(self.state, "to_dict"):
            return self.state.to_dict()
        return {}

    def _to_toon(self, payload: dict[str, Any]) -> str:
        lines = [
            f"context|role={payload.get('role')}|purpose={_safe_text(payload.get('purpose'), 160)}|updated={payload.get('generated_at')}",
        ]
        study = payload.get("study", {})
        lines.append(f"study|country={study.get('country')}|code={study.get('country_code')}|year={study.get('year')}")
        kb = payload.get("study_kb", {})
        lines.append(f"kb|status={kb.get('status')}|summary={_safe_text(kb.get('summary'), 700)}")
        for field, values in (kb.get("fields") or {}).items():
            for value in values[:5]:
                citations = ";".join(
                    f"{c.get('source_id')}:{_safe_text(c.get('evidence'), 140)}"
                    for c in value.get("citations", [])[:2]
                )
                lines.append(f"kb_field|{field}|conf={value.get('confidence')}|value={_safe_text(value.get('value'), 220)}|cite={citations}")
        for item in kb.get("missing_fields", [])[:30]:
            lines.append(f"missing|{_safe_text(item, 200)}")
        for item in kb.get("contradictions", [])[:20]:
            lines.append(f"contradiction|{_safe_text(item, 240)}")
        for decision in payload.get("workflow", {}).get("processor_decisions", []):
            lines.append(f"decision|step={decision.get('step')}|{_safe_text(decision.get('decision'), 260)}")
        for correction in payload.get("workflow", {}).get("pending_corrections", []):
            lines.append(
                f"pending_correction|id={correction.get('decision_id')}|area={correction.get('area')}|target={correction.get('target')}|value={_safe_text(correction.get('value'), 180)}"
            )
        for item in payload.get("workflow", {}).get("work_needing_rerun", []):
            outputs = ",".join(item.get("affected_outputs", [])[:5])
            lines.append(f"needs_rerun|decision={item.get('decision_id')}|target={item.get('target')}|outputs={_safe_text(outputs, 220)}")
        for handoff in payload.get("recent_handoffs", [])[:6]:
            lines.append(f"handoff|role={handoff.get('role')}|{_safe_text(handoff.get('summary'), 300)}")
        return "\n".join(lines) + "\n"

    def _write(self, payload: dict[str, Any], toon: str) -> None:
        self.cses_dir.mkdir(parents=True, exist_ok=True)
        self.json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.toon_path.write_text(toon, encoding="utf-8")
