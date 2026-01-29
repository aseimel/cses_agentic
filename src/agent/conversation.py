"""
Conversational Claude Interface for CSES Processing.

After the initial folder setup, this module provides a natural language
interface where users can chat with Claude about the CSES workflow.
Claude has expert knowledge of the CSES process and guides users through
each step.
"""

import subprocess
import shutil
import json
import os
from pathlib import Path
from typing import Optional

from src.workflow.state import WorkflowState, WORKFLOW_STEPS, StepStatus


# CSES Expert System Prompt
CSES_EXPERT_PROMPT = """You are an expert assistant helping process CSES (Comparative Study of Electoral Systems) Module 6 election study data. You're guiding a researcher through the data harmonization workflow.

## Current Study
Country: {country}
Year: {year}
Working Directory: {working_dir}

## Current Workflow Status
{workflow_status}

## Your Role
1. Guide the user through each step of the CSES workflow
2. Explain what needs to be done at each step
3. Answer questions about CSES methodology, coding schemes, and best practices
4. Help troubleshoot issues with data, variables, or documentation
5. Suggest next actions based on the current workflow state

## CSES Workflow Steps (0-16)
0. Set Up Country Folder - Create folder structure (DONE if you're seeing this)
1. Check Completeness of Deposit - Verify all required files are present
2. Read Design Report - Review study methodology and sampling
3. Fill Variable Tracking Sheet - Map source variables to CSES targets
4. Write Study Design & Weights Overview - Document weighting methodology
5. Request Election Results Table - Get party vote/seat data from macro coder
6. Run Frequencies on Original Data - Check value distributions
7. Process Variables in Stata - Recode variables to CSES schema
8. Complete Documentation - Finalize log file and codebook entries
9. Collect and Integrate District Data - Merge constituency-level results
10. Update Stata Label Files - Apply party code labels
11. Finish Data Processing - Clean up and save final dataset
12. Run Check Files - Validate against CSES standards
13. Write Up Collaborator Questions - Document issues needing clarification
14. Follow Up on Collaborator Questions - Track responses
15. Transfer ESNs to Codebook - Update election study numbers
16. Final Deposit - Archive completed study

## CSES Variable Coding Conventions
- Missing values: 7/97/997 = Refused, 8/98/998 = Don't know, 9/99/999 = Missing/NA
- Demographics (F2xxx): Gender, age, education, household composition
- Political attitudes (F3xxx): Interest, ideology, party identification, satisfaction
- Voting behavior: Vote choice, turnout, timing of decision

## Available Files
{file_info}

## Instructions
- Be concise but helpful
- If the user asks to proceed with a step, explain what will happen and what they need to do
- If something is unclear or missing, ask for clarification
- Proactively suggest the next logical action
- You can read files in the study folder if needed to answer questions

Respond naturally as a helpful CSES expert assistant."""


def get_file_info(state: WorkflowState) -> str:
    """Get information about files in the study."""
    lines = []

    if state.data_file:
        lines.append(f"- Data file: {Path(state.data_file).name}")

    if state.questionnaire_files:
        for f in state.questionnaire_files:
            lines.append(f"- Questionnaire: {Path(f).name}")

    if state.codebook_file:
        lines.append(f"- Codebook: {Path(state.codebook_file).name}")

    if state.design_report_file:
        lines.append(f"- Design report: {Path(state.design_report_file).name}")

    return "\n".join(lines) if lines else "No files registered in state."


def get_workflow_status(state: WorkflowState) -> str:
    """Get formatted workflow status (text only, no emojis for Windows compatibility)."""
    lines = []
    next_step = state.get_next_step()

    for step_num, step_info in WORKFLOW_STEPS.items():
        step_key = str(step_num)
        step_state = state.steps.get(step_key)

        if step_state:
            status = step_state.status if hasattr(step_state, 'status') else "not_started"
        else:
            status = "not_started"

        if status == "completed":
            marker = "[DONE]"
        elif status == "in_progress":
            marker = "[IN PROGRESS]"
        elif step_num == next_step:
            marker = "[NEXT]"
        else:
            marker = "[    ]"

        lines.append(f"{marker} Step {step_num}: {step_info['name']}")

    return "\n".join(lines)


def build_system_prompt(state: WorkflowState) -> str:
    """Build the system prompt with current state."""
    return CSES_EXPERT_PROMPT.format(
        country=state.country,
        year=state.year,
        working_dir=state.working_dir,
        workflow_status=get_workflow_status(state),
        file_info=get_file_info(state)
    )


def call_claude_conversation(
    user_message: str,
    state: WorkflowState,
    conversation_history: list = None
) -> str:
    """
    Send a message to Claude and get a response.

    Uses Claude CLI if available, otherwise falls back to LiteLLM API.
    """
    system_prompt = build_system_prompt(state)

    # Check if Claude CLI is available
    claude_path = shutil.which("claude")

    if claude_path:
        return _call_claude_cli(user_message, system_prompt, conversation_history)
    else:
        return _call_litellm(user_message, system_prompt, conversation_history)


def _call_claude_cli(
    user_message: str,
    system_prompt: str,
    conversation_history: list = None
) -> str:
    """Call Claude via CLI."""
    claude_path = shutil.which("claude")

    # Build the full prompt with context
    full_prompt = f"""<system>
{system_prompt}
</system>

"""

    # Add conversation history if any
    if conversation_history:
        for msg in conversation_history[-10:]:  # Last 10 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                full_prompt += f"User: {content}\n\n"
            else:
                full_prompt += f"Assistant: {content}\n\n"

    full_prompt += f"User: {user_message}\n\nAssistant:"

    try:
        # Use UTF-8 encoding explicitly for Windows compatibility
        result = subprocess.run(
            [claude_path, "--print", "--dangerously-skip-permissions"],
            input=full_prompt,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minutes for longer operations
            encoding='utf-8',
            errors='replace'
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"Error: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "Response timed out. Please try again."
    except Exception as e:
        return f"Error calling Claude: {e}"


def _call_litellm(
    user_message: str,
    system_prompt: str,
    conversation_history: list = None
) -> str:
    """Call Claude via LiteLLM API."""
    try:
        from litellm import completion

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": user_message})

        # Get model from environment
        model = os.getenv("LLM_MODEL_VALIDATE") or os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

        response = completion(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.7
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Error calling API: {e}"


class ConversationSession:
    """Manages a conversation session with Claude."""

    def __init__(self, state: WorkflowState):
        self.state = state
        self.history = []

    def send(self, message: str) -> str:
        """Send a message and get a response."""
        # Add user message to history
        self.history.append({"role": "user", "content": message})

        # Get response
        response = call_claude_conversation(message, self.state, self.history)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        return response

    def refresh_state(self):
        """Reload state from disk."""
        new_state = WorkflowState.load(Path(self.state.working_dir))
        if new_state:
            self.state = new_state
