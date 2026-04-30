"""Reusable evidence packet for processor-facing workflow steps."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.study_kb import StudyKnowledgeBase
from src.workflow.input_manifest import InputManifestBuilder
from src.workflow.state import WorkflowState


PACKET_JSON = "evidence_packet.json"
PACKET_TOON = "evidence_packet.toon"
PACKET_MANIFEST = "evidence_manifest.json"


@dataclass
class EvidencePacketStatus:
    status: str
    reason: str
    manifest_hash: str = ""


class EvidencePacketBuilder:
    """Build and load compact evidence shared by intake/documentation steps."""

    def __init__(self, working_dir: Path):
        self.working_dir = Path(working_dir)
        self.cses_dir = self.working_dir / ".cses"
        self.packet_path = self.cses_dir / PACKET_JSON
        self.toon_path = self.cses_dir / PACKET_TOON
        self.manifest_path = self.cses_dir / PACKET_MANIFEST

    def status(self) -> EvidencePacketStatus:
        if not self.packet_path.exists() or not self.manifest_path.exists():
            return EvidencePacketStatus("missing", "Evidence packet has not been built.")
        current_manifest = self._source_manifest()
        stored_manifest = self._read_json(self.manifest_path)
        current_hash = current_manifest.get("manifest_hash", "")
        stored_hash = stored_manifest.get("manifest_hash", "")
        if current_hash and stored_hash and current_hash == stored_hash:
            return EvidencePacketStatus("current", "Evidence is current.", current_hash)
        return EvidencePacketStatus("stale", "Deposited files changed; refresh evidence before relying on decisions.", current_hash)

    def load(self) -> dict:
        return self._read_json(self.packet_path)

    def build(self, state: WorkflowState, force: bool = False) -> dict:
        current_status = self.status()
        if current_status.status == "current" and not force:
            payload = self.load()
            self.update_state(state, payload)
            return payload

        manifest = self._source_manifest()
        input_manifest = InputManifestBuilder(self.working_dir).build()
        kb = StudyKnowledgeBase(self.working_dir)
        kb_payload = kb.payload if kb.exists() else {}
        evidence_index = getattr(state, "evidence_index", {}) or {}
        design_facts = self._design_facts(kb, evidence_index)
        data_summaries = self._data_summaries(input_manifest)
        diagnostics = self._diagnostics(kb_payload, evidence_index, input_manifest)

        payload = {
            "status": "current",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "manifest_hash": manifest["manifest_hash"],
            "input_manifest": input_manifest.to_dict(),
            "study_kb_path": str(kb.json_path) if kb.exists() else "",
            "evidence_index_status": evidence_index.get("status", "missing") if evidence_index else "missing",
            "design_facts": design_facts,
            "cses_item_coverage": self._field_value(kb, "cses_item_coverage"),
            "missing_cses_items": self._missing_cses_items(kb),
            "contradictions": kb.contradictions() if kb.exists() else [],
            "data_summaries": data_summaries,
            "source_citations": self._source_citations(kb_payload),
            "diagnostics": diagnostics,
        }
        self.cses_dir.mkdir(parents=True, exist_ok=True)
        self.packet_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self.toon_path.write_text(self._to_toon(payload), encoding="utf-8")
        self.manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        self.update_state(state, payload)
        return payload

    def update_state(self, state: WorkflowState, payload: dict) -> None:
        state.evidence_packet_status = payload.get("status", "missing")
        state.evidence_packet_path = str(self.packet_path)
        state.evidence_manifest_path = str(self.manifest_path)
        state.last_evidence_refresh = payload.get("generated_at", "")
        try:
            from src.workflow.phases import current_phase_id, phase_status_payload

            state.current_phase = current_phase_id(state)
            state.phase_status = phase_status_payload(state)
        except Exception:
            pass

    def _source_manifest(self) -> dict:
        files = []
        for path in sorted(self.working_dir.rglob("*"), key=lambda item: str(item).lower()):
            if not path.is_file() or self._ignore_for_manifest(path):
                continue
            stat = path.stat()
            files.append({
                "relative_path": str(path.relative_to(self.working_dir)),
                "size": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
            })
        encoded = json.dumps(files, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "working_dir": str(self.working_dir.resolve()),
            "manifest_hash": hashlib.sha256(encoded).hexdigest(),
            "files": files,
        }

    def _ignore_for_manifest(self, path: Path) -> bool:
        parts = {part.lower() for part in path.relative_to(self.working_dir).parts}
        generated_parts = {
            ".cses",
            "__pycache__",
            "final dataset",
            "data_checks",
            "frequencies",
            "labels",
            "documentation",
            "deposited variable list",
            "collaborator questions",
        }
        if parts & generated_parts:
            return True
        name = path.name.lower()
        return (
            name.endswith((".log", ".smcl", ".tmp"))
            or name.startswith("~$")
            or name == ".log_data.json"
            or name.endswith("_log.qmd")
            or name.endswith("_final_readiness_report.md")
            or name.endswith("_missing_input_report.md")
        )

    def _design_facts(self, kb: StudyKnowledgeBase, evidence_index: dict) -> dict:
        facts_by_field = evidence_index.get("facts_by_field", {}) if evidence_index else {}

        def value(*fields: str) -> str:
            for field in fields:
                kb_value = self._field_value(kb, field)
                if kb_value:
                    return kb_value
            values = []
            for field in fields:
                for fact in facts_by_field.get(field, [])[:2]:
                    fact_value = str(fact.get("value", "")).strip()
                    if fact_value and "no " not in fact_value.lower()[:40]:
                        values.append(fact_value)
            return " | ".join(dict.fromkeys(values))

        return {
            "target_population": value("target_population", "eligible_population"),
            "sample_design": value("sample_design", "sampling_frame", "sampling_stages", "respondent_selection"),
            "probability_sample_status": value("probability_sample_status"),
            "probability_sample_evidence": value("probability_sample_evidence", "sample_design"),
            "sample_size": value("sample_size"),
            "response_rate": value("response_rate"),
            "fieldwork_dates": value("fieldwork_dates", "collection_period"),
            "mode": value("mode"),
            "weights": value("weights", "weighting"),
            "field_lag": value("field_lag"),
            "consent_data_deposit": value("consent_data_deposit"),
        }

    def _field_value(self, kb: StudyKnowledgeBase, field: str) -> str:
        if not kb.exists():
            return ""
        facts = kb.get_field(field)
        if not facts:
            return ""
        value = facts[0].get("value", "")
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value).strip()

    def _missing_cses_items(self, kb: StudyKnowledgeBase) -> list[str]:
        if not kb.exists():
            return []
        items = []
        for fact in kb.get_field("missing_cses_items"):
            value = fact.get("value", "")
            if isinstance(value, list):
                items.extend(str(item) for item in value)
            else:
                items.extend(part.strip(" -") for part in str(value).replace(";", ",").split(","))
        return [
            item for item in dict.fromkeys(item.strip() for item in items)
            if item and item.lower() not in {"none", "none missing", "not applicable", "n/a"}
        ]

    def _data_summaries(self, input_manifest: Any) -> list[dict]:
        summaries = []
        for item in input_manifest.files:
            if item.role not in {"survey_data", "district_data"}:
                continue
            summaries.append({
                "path": item.relative_path,
                "role": item.role,
                "rows": item.metadata.get("n_rows"),
                "variables": item.metadata.get("n_variables"),
            })
        return summaries

    def _source_citations(self, kb_payload: dict) -> list[dict]:
        citations = []
        fields = kb_payload.get("fields", {}) if isinstance(kb_payload, dict) else {}
        for field, facts in fields.items():
            if not isinstance(facts, list):
                continue
            for fact in facts[:2]:
                for citation in fact.get("citations", [])[:2]:
                    citations.append({
                        "field": field,
                        "source_id": citation.get("source_id", ""),
                        "evidence": citation.get("evidence", ""),
                    })
        return citations[:80]

    def _diagnostics(self, kb_payload: dict, evidence_index: dict, input_manifest: Any) -> dict:
        evidence_diag = evidence_index.get("diagnostics", {}) if evidence_index else {}
        kb_diag = kb_payload.get("diagnostics", {}) if isinstance(kb_payload, dict) else {}
        return {
            "source_files": len(input_manifest.files),
            "narrative_files": kb_diag.get("narrative_files", 0),
            "data_files": kb_diag.get("data_files", 0),
            "files_scanned": evidence_diag.get("files_scanned", 0),
            "chunks_total": evidence_diag.get("chunks_total", 0),
            "chunks_ok": evidence_diag.get("chunks_ok", 0),
            "chunks_failed": evidence_diag.get("chunks_failed", 0),
            "fields_found": evidence_diag.get("fields_found", 0),
        }

    def _to_toon(self, payload: dict) -> str:
        lines = [
            f"status|{payload.get('status')}",
            f"generated_at|{payload.get('generated_at')}",
            f"manifest_hash|{payload.get('manifest_hash')}",
        ]
        for key, value in (payload.get("design_facts") or {}).items():
            if value:
                lines.append(f"design|{key}|{value}")
        coverage = payload.get("cses_item_coverage")
        if coverage:
            lines.append(f"cses_item_coverage|{coverage}")
        for item in payload.get("missing_cses_items", []):
            lines.append(f"missing_cses_item|{item}")
        for item in payload.get("contradictions", []):
            lines.append(f"contradiction|{item}")
        return "\n".join(lines) + "\n"

    def _read_json(self, path: Path) -> dict:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
