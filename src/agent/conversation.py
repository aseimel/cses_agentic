"""
Conversational LLM interface for CSES processing.

This is the core user experience: users talk naturally with a CSES expert
assistant, and the assistant documents findings through controlled tools.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Callable

import litellm
from src.model_runtime import ModelRole, ModelTaskRunner
from src.shared_context import SharedWorkflowContext

from src.config import LLM_TEMPERATURE
from src.cses_wiki import format_wiki_results, search_wiki, wiki_prompt_hint
from src.model_catalog import litellm_model_for_completion
from src.codex_oauth_client import is_codex_oauth_model
from src.settings import apply_settings_to_environment
from src.model_profiles import DEFAULT_PROFILE_ID, get_profile
from src.project_context import load_project_context
from src.processor_decisions import (
    CorrectionInterpreter,
    DependencyInvalidator,
    ProcessorDecision,
    ProcessorDecisionLedger,
    TargetedRerunInterpreter,
    diagnose_failure_text,
)
from src.study_kb import StudyKnowledgeBase, StudyKnowledgeBaseBuilder
from src.ui_text import sanitize_processor_text
from src.workflow.active_logging import ActiveLogger
from src.workflow.state import WORKFLOW_STEPS, StepStatus, WorkflowState
from src.workflow.steps import StepExecutor, StepResult


logger = logging.getLogger(__name__)
litellm.drop_params = True
MAX_TOOL_READ_CHARS = 20000
MAX_LOG_READ_CHARS = 12000
LLM_TIMEOUT_SECONDS = 120


def _windows_safe_text(text: str) -> str:
    replacements = {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2190": "<-",
        "\u2192": "->",
        "\u2713": "OK",
        "\u2714": "OK",
        "\u2705": "OK",
        "\ufe0f": "",
        "\u202f": " ",
        "\u00a0": " ",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text.encode("cp1252", errors="replace").decode("cp1252")


def _is_internal_step_issue(issue: str) -> bool:
    lowered = str(issue or "").casefold()
    internal_fragments = [
        "llm eligibility review",
        "evidence extraction chunk",
        "study kb diagnostics",
        "model endpoint",
        "model non-probability classification downgraded",
        "standards check failed",
        "mapping of local variable names",
        "mapping from local variables",
        "cses standard f-codes",
        "f-code mapping",
        "explicit mapping",
    ]
    return any(fragment in lowered for fragment in internal_fragments)


LOG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "build_or_refresh_study_kb",
            "description": "Build or refresh the study-specific knowledge base from deposited files and deterministic data summaries before making evidence decisions.",
            "parameters": {
                "type": "object",
                "properties": {"force": {"type": "boolean"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_study_kb_context",
            "description": "Read the compact study knowledge base context with citations, missing fields, and contradictions.",
            "parameters": {
                "type": "object",
                "properties": {"max_chars": {"type": "integer"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_shared_workflow_context",
            "description": "Read the compact shared workflow context that combines the CSES wiki, study KB, workflow state, decisions, risks, and recent model handoffs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "purpose": {"type": "string"},
                    "max_chars": {"type": "integer"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_cses_wiki",
            "description": "Search the local CSES standards wiki for procedural, coding, eligibility, documentation, or workflow guidance. Use this when unsure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "topic": {
                        "type": "string",
                        "description": "Optional topic filter. Useful values: study_eligibility, documentation_standards, demographic_coding, education_coding, data_processing, data_protection, district_data, macro_data, party_coding, operations, training, general.",
                    },
                    "limit": {"type": "integer"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_log_entry",
            "description": "Write an observation, issue, or finding to the CSES processing log.",
            "parameters": {
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_study_design",
            "description": "Record a study design field found in collaborator documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": [
                            "sample_design",
                            "sample_size",
                            "probability_sample_status",
                            "probability_sample_assessment",
                            "sampling_evidence",
                            "cses_item_coverage",
                            "eligibility_assessment",
                            "processor_eligibility_decision",
                            "response_rate",
                            "weighting",
                            "collection_period",
                            "mode",
                            "field_lag",
                        ],
                    },
                    "value": {"type": "string"},
                },
                "required": ["field", "value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_election_summary",
            "description": "Record election context such as date, type, outcome, turnout, and significance.",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_parties_leaders",
            "description": "Record parties, candidates, leaders, coalitions, and election actors.",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_collaborator_question",
            "description": "Add a confirmed focused question that should be sent to the collaborator. Only use after the processor explicitly decides collaborator contact is needed.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_candidate_collaborator_question",
            "description": "Record a potential collaborator question for processor review when information is missing after checking available data, documentation, and the CSES wiki.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "context": {"type": "string"},
                    "missing_items": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Clear list of missing items that triggered this potential question.",
                    },
                },
                "required": ["question", "missing_items"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo_item",
            "description": "Add an item to the pre-release checklist.",
            "parameters": {
                "type": "object",
                "properties": {"item": {"type": "string"}},
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "review_deposit_eligibility",
            "description": "Run the Step 1 intake review: sample size, CSES item coverage, and probability-sample eligibility.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_step_standards_guidance",
            "description": "Get CSES wiki-backed standards guidance for a workflow step.",
            "parameters": {
                "type": "object",
                "properties": {"step_num": {"type": "integer"}},
                "required": ["step_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_step_standards_check",
            "description": "Run and record CSES wiki-backed standards checks for a workflow step.",
            "parameters": {
                "type": "object",
                "properties": {"step_num": {"type": "integer"}},
                "required": ["step_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_processor_decision",
            "description": "Record a human processor decision, override, or eligibility judgment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_num": {"type": "integer"},
                    "decision": {"type": "string"},
                    "context": {"type": "string"},
                },
                "required": ["step_num", "decision"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_processor_correction",
            "description": "Record a structured processor correction that may invalidate downstream Stata, checks, labels, or documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "area": {
                        "type": "string",
                        "enum": [
                            "eligibility_design",
                            "matching",
                            "demographic_coding",
                            "party_order",
                            "macro_context",
                            "district_data",
                            "recoding_plan",
                            "documentation",
                            "final_readiness",
                            "general",
                        ],
                    },
                    "decision_type": {"type": "string"},
                    "target": {"type": "string"},
                    "value": {"type": "string"},
                    "reason": {"type": "string"},
                    "step_num": {"type": "integer"},
                    "affected_variables": {"type": "array", "items": {"type": "string"}},
                    "status": {
                        "type": "string",
                        "enum": ["approved", "pending_confirmation", "needs_review"],
                    },
                },
                "required": ["area", "decision_type", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_processor_corrections",
            "description": "List pending and approved processor corrections and the downstream work that needs rerun.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "approve_processor_correction",
            "description": "Approve a pending processor correction by decision ID, then mark affected downstream work for rerun.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision_id": {"type": "string"},
                    "reason": {"type": "string"},
                },
                "required": ["decision_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "queue_targeted_rerun",
            "description": "Queue a focused rerun such as rerun this variable, rebuild party recodes, regenerate documentation, rerun checks, or compare again.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "label": {"type": "string"},
                    "steps": {"type": "array", "items": {"type": "integer"}},
                    "target": {"type": "string"},
                },
                "required": ["action", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "diagnose_workflow_failure",
            "description": "Classify a Stata, check, comparison, label, or documentation failure and suggest the smallest correction path.",
            "parameters": {
                "type": "object",
                "properties": {"failure_text": {"type": "string"}},
                "required": ["failure_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_current_documentation",
            "description": "Validate the current processing log/documentation against CSES wiki documentation standards.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_current_stata_syntax",
            "description": "Validate the latest generated CSES micro Stata syntax against CSES wiki syntax standards.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory relative to the study folder.",
            "parameters": {
                "type": "object",
                "properties": {"directory": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a document or text file relative to the study folder.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "start_step",
            "description": "Start a workflow step. Call before doing step work.",
            "parameters": {
                "type": "object",
                "properties": {"step_num": {"type": "integer"}},
                "required": ["step_num"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "complete_step",
            "description": "Complete the active workflow step after all work is finished.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_num": {"type": "integer"},
                    "summary": {"type": "string"},
                },
                "required": ["step_num", "summary"],
            },
        },
    },
]


CSES_EXPERT_PROMPT = """You are a CSES data processing assistant for survey project harmonization.

You work with non-programmer users. Be clear, calm, and practical. The user is
in charge: do not run ahead through the workflow. When the user says "proceed",
work on exactly one next workflow step, document findings with tools, complete
that step only if the required tool actions succeeded, then stop and ask whether
to proceed.

Study: {country} {year}
Working folder: {working_dir}

Workflow status:
{workflow_status}

Registered files:
{file_info}

Rules:
1. Human in the loop: one step per explicit proceed.
2. Use specific tools to record findings, not just prose.
3. If a required file or fact is missing, add a collaborator question and explain the blocker.
4. Do not skip steps or silently complete steps.
5. Do not invent country-specific logic. Treat the current study as one generic study instance.
6. After finishing a step, stop for coder validation. Do not ask for simple "Proceed" until the coder has explicitly validated the step.
7. Step 1 is not only a file inventory. Before completing Step 1, run the deposit eligibility review and report: CSES item coverage, probability sample evidence, and sample size.
8. When unsure about CSES procedure, coding, documentation standards, study eligibility, collaborator questions, data protection, district data, party coding, or release workflow, call search_cses_wiki before deciding. Cite CSES wiki source paths when applying guidance.
9. For workflow steps, use get_step_standards_guidance or run_step_standards_check so the user sees what was checked, what evidence was found, and what remains unresolved.
10. If a soft gate fails, say clearly: "You can proceed, but this unresolved issue will be recorded."
11. Ask as few collaborator questions as possible. Before suggesting a collaborator question, inspect available data, documentation, current log, and CSES wiki guidance. If information is still missing, use add_candidate_collaborator_question with a clear missing-items list and ask the processor whether it should become a collaborator question. Use add_collaborator_question only when the processor explicitly confirms.
12. Before eligibility, design-report, matching, documentation, or collaborator-question decisions, use the study-specific knowledge base when available. If it is missing or stale, build_or_refresh_study_kb first.
13. If the processor corrects a fact, source-variable match, recode, party order, macro context, district decision, or documentation note, record it as a structured processor correction. Explain what downstream work needs rerun. Do not silently overwrite an approved decision.
14. Treat step validation as a separate human decision. A step prepared by the assistant is not workflow-complete until the processor says it has been validated.

{wiki_hint}
{study_kb_hint}
"""


def _status_value(status) -> str:
    return status.value if hasattr(status, "value") else str(status)


def get_file_info(state: WorkflowState) -> str:
    lines = []
    if state.data_file:
        lines.append(f"- Data file: {Path(state.data_file).name}")
    for path in state.questionnaire_files or []:
        lines.append(f"- Questionnaire: {Path(path).name}")
    if state.codebook_file:
        lines.append(f"- Codebook: {Path(state.codebook_file).name}")
    if state.design_report_file:
        lines.append(f"- Design report: {Path(state.design_report_file).name}")
    if state.variable_tracking_file:
        lines.append(f"- Tracking sheet: {Path(state.variable_tracking_file).name}")
    return "\n".join(lines) if lines else "No files are registered yet."


def get_workflow_status(state: WorkflowState) -> str:
    lines = []
    next_step = state.get_next_step()
    for step_num, step_info in WORKFLOW_STEPS.items():
        step = state.get_step(step_num)
        status = _status_value(step.status)
        if status == StepStatus.COMPLETED.value:
            marker = "[DONE]"
        elif status == StepStatus.IN_PROGRESS.value:
            marker = "[IN PROGRESS]"
        elif step_num == next_step:
            marker = "[NEXT]"
        else:
            marker = "[    ]"
        lines.append(f"{marker} Step {step_num}: {step_info['name']}")
    return "\n".join(lines)


def build_system_prompt(state: WorkflowState) -> str:
    kb = StudyKnowledgeBase(Path(state.working_dir))
    if kb.exists():
        kb_hint = "\nStudy knowledge base:\n" + "\n".join(kb.summary_lines()) + "\nUse get_study_kb_context for compact source-backed details."
    else:
        kb_hint = "\nStudy knowledge base: missing. Build it before evidence decisions."
    prompt = CSES_EXPERT_PROMPT.format(
        country=state.country,
        year=state.year,
        working_dir=state.working_dir,
        workflow_status=get_workflow_status(state),
        file_info=get_file_info(state),
        wiki_hint=wiki_prompt_hint(),
        study_kb_hint=kb_hint,
    )
    project_context = load_project_context(Path(state.working_dir)).to_prompt_section()
    if project_context:
        prompt += "\n\n" + project_context
    return prompt


class ConversationSession:
    """Stateful conversation with the CSES assistant."""

    def __init__(self, state: WorkflowState):
        apply_settings_to_environment()
        self.state = state
        self.history: list[dict] = []
        self.active_logger = ActiveLogger(state)
        self.runner = ModelTaskRunner(Path(state.working_dir), state)
        self.correction_interpreter = CorrectionInterpreter()
        self.rerun_interpreter = TargetedRerunInterpreter()

    def refresh_state(self) -> None:
        loaded = WorkflowState.load(Path(self.state.working_dir))
        if loaded:
            self.state = loaded
            self.active_logger = ActiveLogger(loaded)

    def send(self, message: str, on_tool_output: Callable[[str], None] | None = None) -> str:
        self.refresh_state()
        validation_response = self._handle_step_validation_message(message, on_tool_output)
        if validation_response:
            response = sanitize_processor_text(validation_response)
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response})
            self.state.save()
            return response
        correction_response = self._handle_correction_or_rerun_message(message, on_tool_output)
        if correction_response:
            response = sanitize_processor_text(correction_response)
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response})
            self.state.save()
            return response
        if self._is_proceed_request(message):
            response = self._proceed_one_step(on_tool_output)
            response = sanitize_processor_text(response)
            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": response})
            self.state.save()
            return response

        messages = [{"role": "system", "content": build_system_prompt(self.state)}]
        messages.extend(self.history[-12:])
        messages.append({"role": "user", "content": message})

        self.history.append({"role": "user", "content": message})
        response = self._call_with_tools(messages, on_tool_output)
        response = sanitize_processor_text(response)
        self.history.append({"role": "assistant", "content": response})
        self.state.save()
        return response

    def _is_proceed_request(self, message: str) -> bool:
        normalized = " ".join((message or "").strip().lower().split())
        return normalized in {"proceed", "go ahead", "continue", "ok", "okay"}

    def _handle_step_validation_message(
        self,
        message: str,
        on_tool_output: Callable[[str], None] | None = None,
    ) -> str:
        normalized = " ".join((message or "").strip().split())
        lower = normalized.casefold()
        validation_phrases = [
            "validate step",
            "validated step",
            "approve step",
            "approved step",
            "mark step",
            "i have validated",
            "i validated",
            "this step is correct",
            "step is correct",
        ]
        if not any(phrase in lower for phrase in validation_phrases):
            return ""

        match = re.search(r"\bstep\s+(\d+)\b", lower)
        step_num = int(match.group(1)) if match else self.state.get_next_step()
        if step_num is None or step_num not in WORKFLOW_STEPS:
            return "I could not identify which step to validate."

        step = self.state.get_step(step_num)
        if _status_value(step.status) != StepStatus.NEEDS_VALIDATION.value:
            return f"Step {step_num} is not waiting for coder validation."

        note = normalized
        self.state.validate_step(step_num, note=note, validated_by="processor")
        self.active_logger.record_processor_decision(step_num, f"Validated Step {step_num}", note)
        self.state.save()
        self._notify(on_tool_output, f"Recorded coder validation for Step {step_num}")
        next_step = self.state.get_next_step()
        if next_step is None:
            return f"Step {step_num} validated. All workflow steps are complete."
        return (
            f"Step {step_num} validated and recorded.\n\n"
            f"Next: Step {next_step}: {WORKFLOW_STEPS[next_step]['name']}. "
            "Use Proceed when you want the assistant to work on that step."
        )

    def _handle_correction_or_rerun_message(
        self,
        message: str,
        on_tool_output: Callable[[str], None] | None = None,
    ) -> str:
        normalized = " ".join((message or "").strip().split())
        lower = normalized.casefold()
        if lower.startswith("approve correction"):
            decision_id = normalized.split(maxsplit=2)[-1].strip()
            if decision_id and decision_id.casefold() != "correction":
                decision = self._approve_structured_decision(decision_id)
                if decision:
                    self._notify(on_tool_output, "Recorded processor approval")
                    return self._format_decision_response(decision, approved_now=True)
            return "I could not find that correction ID. Ask to show pending corrections, then approve the exact ID."

        rerun = self.rerun_interpreter.parse(message)
        if rerun:
            self.state.queue_targeted_rerun(rerun)
            self._notify(on_tool_output, "Queued focused rerun")
            steps = ", ".join(str(step) for step in rerun.get("steps", [])) or "the affected step"
            return (
                f"Queued: {rerun.get('label')}.\n\n"
                f"This will revisit Step {steps}. Approved work outside that affected area remains unchanged."
            )

        decision = self.correction_interpreter.parse(message, default_step=self.state.get_next_step() or 0)
        if not decision:
            return ""
        recorded = self._record_structured_decision(decision)
        self._notify(on_tool_output, "Recorded processor correction")
        return self._format_decision_response(recorded)

    def _record_structured_decision(self, decision: ProcessorDecision) -> ProcessorDecision:
        ledger = ProcessorDecisionLedger(self.state.working_dir)
        impact = DependencyInvalidator().impact_for(decision)
        decision.affected_variables = list(dict.fromkeys([*decision.affected_variables, *impact.affected_variables]))
        decision.affected_outputs = list(dict.fromkeys([*decision.affected_outputs, *impact.affected_outputs]))
        recorded = ledger.append(decision)
        self.state.processor_decision_ledger_path = str(ledger.path)
        self.state.record_structured_processor_decision(
            recorded.to_dict(),
            impact.to_dict() if recorded.status == "approved" else None,
        )
        self.active_logger.record_structured_processor_decision(
            recorded.to_dict(),
            impact.to_dict() if recorded.status == "approved" else None,
        )
        return recorded

    def _approve_structured_decision(self, decision_id: str) -> ProcessorDecision | None:
        ledger = ProcessorDecisionLedger(self.state.working_dir)
        decision = ledger.update_status(decision_id, "approved")
        if not decision:
            return None
        impact = DependencyInvalidator().impact_for(decision)
        decision.affected_variables = list(dict.fromkeys([*decision.affected_variables, *impact.affected_variables]))
        decision.affected_outputs = list(dict.fromkeys([*decision.affected_outputs, *impact.affected_outputs]))
        ledger.save([item if item.decision_id != decision.decision_id else decision for item in ledger.load()])
        self.state.processor_decision_ledger_path = str(ledger.path)
        self.state.update_structured_decision_status(decision.to_dict(), impact.to_dict())
        self.active_logger.record_structured_processor_decision(decision.to_dict(), impact.to_dict())
        return decision

    def _format_decision_response(self, decision: ProcessorDecision, approved_now: bool = False) -> str:
        if decision.status == "approved":
            status_line = "Correction approved and recorded." if approved_now else "Correction recorded."
        elif decision.status == "pending_confirmation":
            status_line = (
                "I recorded this as a proposed correction because it affects later coding. "
                f"To approve it, reply: approve correction {decision.decision_id}"
            )
        else:
            status_line = "I recorded this correction for review."
        outputs = ", ".join(decision.affected_outputs[:6]) if decision.affected_outputs else "final readiness"
        target = f"{decision.target}: " if decision.target else ""
        value = decision.value or decision.reason
        return (
            f"{status_line}\n\n"
            f"Decision: {target}{value}\n"
            f"Affected work: {outputs}\n\n"
            "Unaffected approved work remains unchanged."
        )

    def _proceed_one_step(self, on_tool_output: Callable[[str], None] | None = None) -> str:
        next_step = self.state.get_next_step()
        if next_step is None:
            return "All workflow steps are already complete."
        step_name = WORKFLOW_STEPS[next_step]["name"]
        step = self.state.get_step(next_step)
        if _status_value(step.status) == StepStatus.NEEDS_VALIDATION.value:
            return (
                f"Step {next_step} is ready for coder validation: {step_name}.\n\n"
                "Please review the step output. If it is correct, use the Validate Step button "
                f"or say: validate step {next_step}. If something is wrong, tell me the correction."
            )
        if _status_value(step.status) == StepStatus.BLOCKED.value:
            return (
                f"Step {next_step} is waiting for review: {step_name}.\n\n"
                "Please resolve the listed issue or tell me what should be corrected before continuing."
            )
        self._notify(on_tool_output, f"Running Step {next_step}: {step_name}")
        executor = StepExecutor(self.state)
        result = executor.execute_step(next_step, require_processor_validation=True)
        self.refresh_state()
        if result.success:
            self._notify(on_tool_output, f"Step {next_step} is ready for coder validation")
        else:
            self._notify(on_tool_output, f"Step {next_step} needs review: {result.message}")
        return self._summarize_direct_step(next_step, result)

    def _summarize_direct_step(self, step_num: int, result: StepResult) -> str:
        next_step = self.state.get_next_step()
        next_step_text = (
            f"Step {next_step}: {WORKFLOW_STEPS[next_step]['name']}"
            if next_step is not None
            else "all workflow steps complete"
        )
        status = "ready for coder validation" if result.success else "needs review"
        visible_issues = [
            issue for issue in result.issues
            if not _is_internal_step_issue(issue)
        ]
        issue_text = "\n".join(f"- {issue}" for issue in visible_issues[:10]) or "- None recorded"
        review_block = ""
        if visible_issues:
            review_block = f"\n\nProcessor review:\n{issue_text}"
        return sanitize_processor_text(
            f"Step {step_num} {status}: {WORKFLOW_STEPS[step_num]['name']}\n\n"
            f"{result.message}"
            f"{review_block}\n\n"
            + (
                f"Review this step, then use Validate Step or say: validate step {step_num}."
                if result.success
                else f"Next: {next_step_text}."
            )
        )

    def _call_with_tools(self, messages: list[dict], on_tool_output: Callable[[str], None] | None) -> str:
        model = (
            os.getenv("CSES_AGENTIC_MODEL")
            or os.getenv("CSES_CHAT_MODEL")
            or get_profile(os.getenv("CSES_MODEL_PROFILE", DEFAULT_PROFILE_ID)).conversation_model
        )
        model = litellm_model_for_completion(
            model,
            {
                "CSES_USE_OPENWEBUI": os.getenv("CSES_USE_OPENWEBUI", ""),
                "OPENAI_API_BASE": os.getenv("OPENAI_API_BASE", ""),
            },
        )
        if is_codex_oauth_model(model):
            response = ModelTaskRunner(Path(self.state.working_dir), self.state).response(
                ModelRole.AGENTIC,
                messages=messages,
                model_override=model,
                purpose="Codex OAuth conversational response",
                temperature=LLM_TEMPERATURE,
                max_tokens=2048,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            content = response.choices[0].message.content or ""
            return content.strip() or "Codex OAuth returned no text response. Use Proceed to run the next workflow step."
        latest_user = next((m.get("content", "") for m in reversed(messages) if m.get("role") == "user"), "")
        requires_step_tools = "proceed" in latest_user.lower()
        turn_state = {
            "step_started": False,
            "step_completed": False,
            "failed_tools": [],
        }

        runner = ModelTaskRunner(Path(self.state.working_dir), self.state)
        response = runner.response(
            ModelRole.AGENTIC,
            messages=messages,
            model_override=model,
            purpose="Conversation tool-using turn",
            tools=LOG_TOOLS,
            tool_choice="auto",
            parallel_tool_calls=False,
            temperature=LLM_TEMPERATURE,
            max_tokens=2048,
            drop_params=True,
            timeout=LLM_TIMEOUT_SECONDS,
        )
        current_message = response.choices[0].message

        for _ in range(8):
            tool_calls = getattr(current_message, "tool_calls", None)
            if not tool_calls:
                if requires_step_tools and not turn_state.get("step_started"):
                    messages.append({"role": "assistant", "content": current_message.content or ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "The user asked to proceed with the next workflow step. "
                                "You must use the workflow tools: call start_step, inspect or record the required work, "
                                "then call complete_step only if the step is genuinely complete. Do not answer in prose only."
                            ),
                        }
                    )
                    response = runner.response(
                        ModelRole.AGENTIC,
                        messages=messages,
                        model_override=model,
                        purpose="Force workflow tools after proceed request",
                        tools=LOG_TOOLS,
                        tool_choice="auto",
                        parallel_tool_calls=False,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=2048,
                        drop_params=True,
                        timeout=LLM_TIMEOUT_SECONDS,
                    )
                    current_message = response.choices[0].message
                    continue

                if turn_state.get("step_started") and not turn_state.get("step_completed"):
                    if turn_state.get("failed_tools"):
                        failures = "; ".join(turn_state["failed_tools"])
                        content = current_message.content or ""
                        return (
                            content.strip()
                            + f"\n\nStep remains in progress because these tool calls failed: {failures}"
                        ).strip()

                    messages.append({"role": "assistant", "content": current_message.content or ""})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You started a workflow step but did not call complete_step. "
                                "If the step work is done, call complete_step now. If it is blocked, "
                                "explain the blocker and do not claim the step is complete."
                            ),
                        }
                    )
                    response = runner.response(
                        ModelRole.AGENTIC,
                        messages=messages,
                        model_override=model,
                        purpose="Complete started workflow step",
                        tools=LOG_TOOLS,
                        tool_choice="auto",
                        parallel_tool_calls=False,
                        temperature=LLM_TEMPERATURE,
                        max_tokens=2048,
                        drop_params=True,
                        timeout=LLM_TIMEOUT_SECONDS,
                    )
                    current_message = response.choices[0].message
                    continue

                content = current_message.content or ""
                return content.strip() or "I did not receive a text response. Please try again."

            messages.append(
                {
                    "role": "assistant",
                    "content": current_message.content or "",
                    "tool_calls": [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.function.name,
                                "arguments": tool_call.function.arguments,
                            },
                        }
                        for tool_call in tool_calls
                    ],
                }
            )

            for tool_call in tool_calls:
                result = self._execute_tool(tool_call, turn_state, on_tool_output)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    }
                )

            if turn_state.get("step_completed"):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The assistant work for this workflow step is ready for coder validation. "
                            "Provide a concise final message for the processor. "
                            "Include what was checked, key evidence found, CSES wiki or standards guidance used, "
                            "unresolved items, and ask the processor to validate the step or provide corrections. "
                            "Do not call more tools."
                        ),
                    }
                )
                response = runner.response(
                    ModelRole.AGENTIC,
                    messages=messages,
                    model_override=model,
                    purpose="Final processor-facing step message",
                    temperature=LLM_TEMPERATURE,
                    max_tokens=2048,
                    drop_params=True,
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                content = response.choices[0].message.content or ""
                return content.strip() or "This step is ready for coder validation. Please validate it or provide corrections."

            response = runner.response(
                ModelRole.AGENTIC,
                messages=messages,
                model_override=model,
                purpose="Continue conversation tool loop",
                tools=LOG_TOOLS,
                tool_choice="auto",
                parallel_tool_calls=False,
                temperature=LLM_TEMPERATURE,
                max_tokens=2048,
                drop_params=True,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            current_message = response.choices[0].message

        return "I completed the available tool work, but the model did not finish a final message. Check the log and try again."

    def _notify(self, callback: Callable[[str], None] | None, message: str) -> None:
        if callback:
            cleaned = sanitize_processor_text(message)
            if cleaned:
                callback(cleaned)

    def _execute_tool(self, tool_call, turn_state: dict, on_tool_output: Callable[[str], None] | None) -> str:
        name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError:
            return f"FAILED: Invalid arguments for {name}"

        if name == "start_step" and turn_state.get("step_completed"):
            return "BLOCKED: One step per turn. Stop and ask the user whether to proceed."

        if name == "complete_step" and turn_state.get("failed_tools"):
            failures = "; ".join(turn_state["failed_tools"])
            return f"BLOCKED: Cannot complete step because these tools failed: {failures}"

        try:
            result = self._dispatch_tool(name, args, turn_state, on_tool_output)
        except Exception as exc:
            logger.exception("Conversation tool failed")
            turn_state.setdefault("failed_tools", []).append(f"{name}: {exc}")
            result = f"FAILED: {name}: {exc}"
        return result

    def _dispatch_tool(
        self,
        name: str,
        args: dict,
        turn_state: dict,
        on_tool_output: Callable[[str], None] | None,
    ) -> str:
        if name == "write_log_entry":
            message = args.get("message", "")
            success, status = self.active_logger.log_message(message)
            self._notify(on_tool_output, status)
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "build_or_refresh_study_kb":
            force = bool(args.get("force", False))
            builder = StudyKnowledgeBaseBuilder(
                Path(self.state.working_dir),
                progress_callback=lambda message: self._notify(on_tool_output, message),
            )
            payload = builder.build(self.state, force=force)
            self.state.save()
            self._notify(on_tool_output, "Reviewed study materials")
            return json.dumps(
                {
                    "status": payload.get("status"),
                    "model": payload.get("model"),
                    "summary": payload.get("summary"),
                    "missing_fields": payload.get("missing_fields", [])[:40],
                    "contradictions": payload.get("contradictions", [])[:20],
                    "diagnostics": payload.get("diagnostics", {}),
                },
                indent=2,
            )

        if name == "get_study_kb_context":
            max_chars = int(args.get("max_chars", 12000) or 12000)
            kb = StudyKnowledgeBase(Path(self.state.working_dir))
            if not kb.exists():
                return "FAILED: Study materials have not been reviewed yet."
            self._notify(on_tool_output, "Loaded study information")
            return kb.compact_context(max_chars=max_chars)

        if name == "get_shared_workflow_context":
            max_chars = int(args.get("max_chars", 12000) or 12000)
            role = str(args.get("role") or ModelRole.AGENTIC.value)
            purpose = str(args.get("purpose") or "Conversation workflow context")
            context = SharedWorkflowContext(Path(self.state.working_dir), self.state)
            packet = context.build_packet(role=role, purpose=purpose, max_chars=max_chars)
            self._notify(on_tool_output, "Loaded shared workflow context")
            return packet.toon_context

        if name == "search_cses_wiki":
            query = args.get("query", "")
            topic = args.get("topic", "")
            limit = args.get("limit", 5)
            try:
                limit = int(limit)
            except (TypeError, ValueError):
                limit = 5
            results = search_wiki(query=query, topic=topic, limit=limit)
            self._notify(on_tool_output, f"Checked CSES guidance: {query}")
            return format_wiki_results(results)

        if name == "update_study_design":
            field = args.get("field", "")
            value = args.get("value", "")
            success, status = self.active_logger.update_study_design_section({field: value})
            self._notify(on_tool_output, f"Study design: {field} = {value}")
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "update_election_summary":
            success, status = self.active_logger.update_election_summary(args.get("summary", ""))
            self._notify(on_tool_output, status)
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "update_parties_leaders":
            success, status = self.active_logger.update_parties_leaders(args.get("content", ""))
            self._notify(on_tool_output, status)
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "add_collaborator_question":
            question = args.get("question", "")
            success, status = self.active_logger.add_collaborator_question(
                question,
                "From conversation",
                self.state.get_next_step() or 0,
            )
            self._notify(on_tool_output, status)
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "add_candidate_collaborator_question":
            question = args.get("question", "")
            context = args.get("context", "From conversation")
            missing_items = args.get("missing_items", [])
            if not isinstance(missing_items, list):
                missing_items = [str(missing_items)]
            success, status = self.active_logger.add_candidate_collaborator_question(
                question,
                context,
                self.state.get_next_step() or 0,
                missing_items=missing_items,
            )
            self._notify(on_tool_output, status)
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "add_todo_item":
            success, status = self.active_logger.add_todo_item(args.get("item", ""))
            self._notify(on_tool_output, status)
            return f"{'SUCCESS' if success else 'FAILED'}: {status}"

        if name == "review_deposit_eligibility":
            return self._review_deposit_eligibility(on_tool_output)

        if name == "get_step_standards_guidance":
            from src.standards.engine import WorkflowStandardsEngine

            step_num = int(args.get("step_num", self.state.get_next_step() or 0))
            engine = WorkflowStandardsEngine(Path(self.state.working_dir))
            guidance = engine.guidance(step_num)
            self._notify(on_tool_output, f"Loaded CSES guidance for Step {step_num}")
            return guidance

        if name == "run_step_standards_check":
            from src.standards.engine import WorkflowStandardsEngine

            step_num = int(args.get("step_num", self.state.get_next_step() or 0))
            engine = WorkflowStandardsEngine(Path(self.state.working_dir))
            result = engine.evaluate_step(self.state, step_num)
            self.state.record_standards_check(step_num, result.to_dict())
            self.active_logger.record_standards_check(step_num, result.to_dict())
            self._notify(on_tool_output, f"Checked CSES rules for Step {step_num}: {result.status}")
            return json.dumps(result.to_dict(), indent=2)

        if name == "record_processor_decision":
            step_num = int(args.get("step_num", self.state.get_next_step() or 0))
            decision = args.get("decision", "")
            context = args.get("context", "")
            self.state.record_processor_decision(step_num, decision, context)
            self.active_logger.record_processor_decision(step_num, decision, context)
            self._notify(on_tool_output, f"Recorded processor decision for Step {step_num}")
            return "SUCCESS: processor decision recorded"

        if name == "record_processor_correction":
            decision = ProcessorDecision(
                decision_id="",
                step=int(args.get("step_num") or self.state.get_next_step() or 0),
                area=str(args.get("area") or "general"),
                decision_type=str(args.get("decision_type") or "correction"),
                target=str(args.get("target") or ""),
                value=str(args.get("value") or ""),
                reason=str(args.get("reason") or ""),
                affected_variables=[str(item) for item in args.get("affected_variables", []) or []],
                status=str(args.get("status") or "approved"),
            )
            recorded = self._record_structured_decision(decision)
            self._notify(on_tool_output, "Recorded processor correction")
            return json.dumps(recorded.to_dict(), indent=2, ensure_ascii=False)

        if name == "list_processor_corrections":
            ledger = ProcessorDecisionLedger(self.state.working_dir)
            decisions = ledger.load()
            pending = [item.to_dict() for item in decisions if item.status in {"pending_confirmation", "needs_review"}]
            approved = [item.to_dict() for item in decisions if item.status == "approved"]
            self._notify(on_tool_output, "Loaded processor corrections")
            return json.dumps(
                {
                    "pending": pending,
                    "approved_recent": approved[-10:],
                    "work_needing_rerun": self.state.invalidated_outputs[-20:],
                    "queued_reruns": self.state.targeted_rerun_queue[-20:],
                },
                indent=2,
                ensure_ascii=False,
            )

        if name == "approve_processor_correction":
            decision_id = str(args.get("decision_id") or "")
            decision = self._approve_structured_decision(decision_id)
            if not decision:
                return f"FAILED: Correction not found: {decision_id}"
            self._notify(on_tool_output, "Recorded processor approval")
            return json.dumps(decision.to_dict(), indent=2, ensure_ascii=False)

        if name == "queue_targeted_rerun":
            action = {
                "action": args.get("action", "rerun"),
                "label": args.get("label", "Rerun affected work"),
                "steps": args.get("steps", []),
                "target": args.get("target", ""),
            }
            self.state.queue_targeted_rerun(action)
            self._notify(on_tool_output, "Queued focused rerun")
            return json.dumps(action, indent=2, ensure_ascii=False)

        if name == "diagnose_workflow_failure":
            diagnosis = diagnose_failure_text(str(args.get("failure_text") or ""))
            self._notify(on_tool_output, "Reviewed failure and suggested a correction path")
            return json.dumps(diagnosis, indent=2, ensure_ascii=False)

        if name == "validate_current_documentation":
            from src.standards.validators import validate_documentation_text

            if not self.state.log_file or not Path(self.state.log_file).exists():
                return "FAILED: No processing log file is registered."
            text = Path(self.state.log_file).read_text(encoding="utf-8", errors="replace")
            result = validate_documentation_text(text)
            self._notify(on_tool_output, "Checked current documentation against CSES rules")
            return json.dumps({"ok": result.ok, "checks": result.checks, "issues": result.issues}, indent=2)

        if name == "validate_current_stata_syntax":
            from src.standards.validators import validate_stata_syntax_text

            micro_dir = Path(self.state.working_dir) / "micro"
            do_files = list(micro_dir.glob("cses-m6_micro_*.do"))
            if not do_files:
                return "FAILED: No generated CSES micro .do file found."
            do_path = max(do_files, key=lambda path: path.stat().st_mtime)
            result = validate_stata_syntax_text(do_path.read_text(encoding="utf-8", errors="replace"))
            self._notify(on_tool_output, f"Validated Stata syntax: {do_path.name}")
            return json.dumps({"file": str(do_path), "ok": result.ok, "checks": result.checks, "issues": result.issues}, indent=2)

        if name == "list_files":
            directory = args.get("directory", "")
            root = Path(self.state.working_dir).resolve()
            target = (root / directory).resolve()
            if not str(target).startswith(str(root)) or not target.exists():
                return f"FAILED: Directory not found: {directory}"
            files = [str(path.relative_to(root)) for path in target.rglob("*") if path.is_file()]
            self._notify(on_tool_output, f"Listed {len(files)} files in {directory or '.'}")
            return "\n".join(files[:200])

        if name == "read_file":
            return self._read_file(args.get("path", ""), on_tool_output)

        if name == "start_step":
            return self._start_step(args.get("step_num"), turn_state, on_tool_output)

        if name == "complete_step":
            return self._complete_step(args.get("step_num"), args.get("summary", ""), turn_state, on_tool_output)

        return f"FAILED: Unknown tool {name}"

    def _review_deposit_eligibility(self, on_tool_output: Callable[[str], None] | None) -> str:
        from src.workflow.eligibility import review_initial_eligibility

        root = Path(self.state.working_dir)
        review = review_initial_eligibility(
            working_dir=root,
            data_files=[Path(self.state.data_file)] if self.state.data_file else [],
            questionnaire_files=[Path(path) for path in self.state.questionnaire_files or []],
            codebook_files=[Path(self.state.codebook_file)] if self.state.codebook_file else [],
            design_report_files=[Path(self.state.design_report_file)] if self.state.design_report_file else [],
        )
        if review.sample_size_rows is not None:
            self.active_logger.update_study_design_section({"sample_size": str(review.sample_size_rows)})
        self.active_logger.update_study_design_section({
            "probability_sample_status": review.probability_sample_status,
            "probability_sample_assessment": review.probability_sample_assessment,
            "sampling_evidence": "\n".join(review.sampling_evidence[:8]),
            "cses_item_coverage": review.cses_items_evidence,
            "eligibility_assessment": review.eligibility_assessment,
            "processor_eligibility_decision": review.processor_eligibility_decision,
        })
        self.active_logger.log_message(review.to_log_message())
        for question in review.collaborator_questions:
            self.active_logger.add_candidate_collaborator_question(
                question,
                "Initial CSES eligibility review; processor decides whether collaborator contact is needed",
                self.state.get_next_step() or 1,
                missing_items=review.issues,
            )
        summary = review.to_log_message()
        self._notify(on_tool_output, "Completed initial CSES eligibility review")
        if review.issues:
            summary += "\nIssues:\n" + "\n".join(f"- {issue}" for issue in review.issues)
        if review.collaborator_questions:
            summary += "\nCollaborator questions:\n" + "\n".join(f"- {q}" for q in review.collaborator_questions)
        return "SUCCESS: " + summary

    def _read_file(self, relative_path: str, on_tool_output: Callable[[str], None] | None) -> str:
        root = Path(self.state.working_dir).resolve()
        target = (root / relative_path).resolve()
        if not str(target).startswith(str(root)) or not target.exists() or not target.is_file():
            return f"FAILED: File not found: {relative_path}"

        suffix = target.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader

            reader = PdfReader(target)
            content = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif suffix == ".docx":
            from docx import Document

            document = Document(target)
            content = "\n".join(paragraph.text for paragraph in document.paragraphs)
        else:
            content = target.read_text(encoding="utf-8", errors="replace")

        limit = MAX_LOG_READ_CHARS if suffix in {".log", ".smcl"} else MAX_TOOL_READ_CHARS
        if len(content) > limit and suffix in {".log", ".smcl"}:
            content = content[-limit:]
        else:
            content = content[:limit]

        self._notify(on_tool_output, f"Read {relative_path}")
        return content

    def _start_step(self, step_num, turn_state: dict, on_tool_output: Callable[[str], None] | None) -> str:
        if step_num not in WORKFLOW_STEPS:
            return f"FAILED: Invalid step number {step_num}"

        can_proceed, reason = self.state.check_step_prerequisites(step_num)
        if not can_proceed:
            return f"BLOCKED: {reason}"

        step = self.state.get_step(step_num)
        status = _status_value(step.status)
        if status == StepStatus.COMPLETED.value:
            return f"SKIP: Step {step_num} is already completed"

        self.state.set_step_status(step_num, StepStatus.IN_PROGRESS, "Started by assistant")
        self.state.current_step = step_num
        self.state.save()
        turn_state["step_started"] = True
        self._notify(on_tool_output, f"Started Step {step_num}: {WORKFLOW_STEPS[step_num]['name']}")
        return f"SUCCESS: Started Step {step_num}: {WORKFLOW_STEPS[step_num]['name']}"

    def _complete_step(
        self,
        step_num,
        summary: str,
        turn_state: dict,
        on_tool_output: Callable[[str], None] | None,
    ) -> str:
        if step_num not in WORKFLOW_STEPS:
            return f"FAILED: Invalid step number {step_num}"

        step = self.state.get_step(step_num)
        if _status_value(step.status) != StepStatus.IN_PROGRESS.value:
            return f"FAILED: Step {step_num} is not in progress"

        artifact_error = self._validate_step_completion_artifacts(step_num)
        if artifact_error:
            self.state.add_step_issue(step_num, artifact_error)
            self.state.save()
            turn_state.setdefault("failed_tools", []).append(f"complete_step: {artifact_error}")
            return f"BLOCKED: {artifact_error}"

        self.state.mark_step_needs_validation(
            step_num,
            "Assistant work finished. Waiting for coder validation before continuing.",
        )
        self.state.save()
        turn_state["step_completed"] = True
        self._notify(on_tool_output, f"Step {step_num} is ready for coder validation")

        return (
            f"SUCCESS: Step {step_num} is ready for coder validation. "
            f"Stop and ask the user to validate Step {step_num} or provide corrections."
        )

    def _validate_step_completion_artifacts(self, step_num: int) -> str | None:
        """Prevent chat from marking artifact-producing steps complete with notes only."""
        working_dir = Path(self.state.working_dir)
        micro_dir = working_dir / "micro"

        if step_num == 1:
            log_data_path = micro_dir / ".log_data.json"
            if not log_data_path.exists():
                return "Step 1 requires the initial CSES eligibility review to be logged before completion."
            try:
                log_data = json.loads(log_data_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return f"Step 1 requires a readable log data file before completion: {exc}"
            study_design = log_data.get("study_design", {})
            missing = [
                label
                for key, label in [
                    ("sample_size", "sample size"),
                    ("probability_sample_status", "probability-sample status"),
                    ("probability_sample_assessment", "probability-sample assessment"),
                    ("cses_item_coverage", "CSES item coverage"),
                    ("eligibility_assessment", "initial eligibility assessment"),
                ]
                if not str(study_design.get(key, "")).strip()
            ]
            if missing:
                return (
                    "Step 1 requires the initial CSES eligibility review before completion. "
                    f"Missing: {', '.join(missing)}. Run review_deposit_eligibility first."
                )

        if step_num == 7:
            do_files = list(micro_dir.glob("cses-m6_micro_*.do"))
            if not do_files:
                return "Step 7 requires generated Stata syntax in micro/cses-m6_micro_*.do. Run/generate the tracking-sheet based Stata code first."

        if step_num == 8:
            do_files = list(micro_dir.glob("cses-m6_micro_*.do"))
            log_files = list(micro_dir.glob("*.log"))
            if not do_files:
                return "Step 8 requires a generated .do file before Stata debugging can run."
            if not log_files:
                return "Step 8 requires a Stata log artifact showing the .do file was run."

        if step_num == 11:
            final_dir = micro_dir / "FINAL dataset"
            final_files = []
            if final_dir.exists():
                final_files = [
                    path for path in final_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".dta", ".csv", ".xlsx"}
                ]
            processed_files = list(micro_dir.glob("cses-m6_micro_*.dta"))
            if not final_files and not processed_files:
                return "Step 11 requires a processed dataset artifact, such as micro/cses-m6_micro_*.dta."

        if step_num == 12:
            processed_files = list(micro_dir.glob("cses-m6_micro_*.dta"))
            stata_logs = list(micro_dir.glob("cses-m6_micro_*.log"))
            if not processed_files or not stata_logs:
                return "Step 12 requires a processed .dta and Stata execution log before checks can be completed."

        if step_num == 16:
            final_dir = micro_dir / "FINAL dataset"
            final_files = []
            if final_dir.exists():
                final_files = [
                    path for path in final_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in {".dta", ".csv", ".xlsx"}
                ]
            processed_files = list(micro_dir.glob("cses-m6_micro_*.dta"))
            if not final_files and not processed_files:
                return "Step 16 requires a final deposit dataset artifact, such as micro/cses-m6_micro_*.dta."

        return None
