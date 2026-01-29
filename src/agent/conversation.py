"""
Conversational Claude Interface for CSES Processing.

After the initial folder setup, this module provides a natural language
interface where users can chat with Claude about the CSES workflow.
Claude has expert knowledge of the CSES process and guides users through
each step.

Claude has direct tool access to write to the log file in real-time using
tools like write_log_entry, update_study_design, add_collaborator_question,
and update_variable_mapping.
"""

import subprocess
import shutil
import json
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflow.active_logging import ActiveLogger

from src.workflow.state import WorkflowState, WORKFLOW_STEPS, StepStatus
from src.workflow.active_logging import ActiveLogger


# Tool definitions for Claude to update log files directly
LOG_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "write_log_entry",
            "description": "Write an entry to the CSES log file. Use this for any observation, issue, or finding that should be documented.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "The log message to record"
                    }
                },
                "required": ["message"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_study_design",
            "description": "Update study design section with a specific field value. Call this when you discover study design information from documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {
                        "type": "string",
                        "enum": ["sample_design", "sample_size", "response_rate", "weighting", "collection_period", "mode", "field_lag"],
                        "description": "The study design field to update"
                    },
                    "value": {
                        "type": "string",
                        "description": "The value to set for this field"
                    }
                },
                "required": ["field", "value"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_collaborator_question",
            "description": "Add a question that needs to be sent to the collaborator for clarification.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the collaborator"
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_variable_mapping",
            "description": "Record a variable mapping in the tracking sheet. Use when you identify which source variable maps to a CSES target variable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cses_code": {
                        "type": "string",
                        "description": "CSES target variable code (e.g., F2001, F3024)"
                    },
                    "source_variable": {
                        "type": "string",
                        "description": "Source variable name from the deposited data"
                    }
                },
                "required": ["cses_code", "source_variable"]
            }
        }
    }
]


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

## CRITICAL: Live Log File Updates
You MUST write important findings to the log file AS YOU DISCOVER THEM.

If you have tool access, use the tools:
- write_log_entry(message): Log any observation, issue, or finding
- update_study_design(field, value): Record study design details
- add_collaborator_question(question): Add a question for the collaborator
- update_variable_mapping(cses_code, source_variable): Record a variable mapping

If tools are not available, use these markers in your response:
[LOG: your note here] - For any observation, issue, or finding
[STUDY_DESIGN: field=value] - For study design info (sample_design, sample_size, response_rate, weighting, collection_period, mode, field_lag)
[QUESTION: your question] - For collaborator questions
[VARIABLE: cses_code=source_variable] - For variable mappings

IMPORTANT: When you discover information from documents, RECORD IT IMMEDIATELY.
Do not just mention "response rate of 45%" - record it with update_study_design or [STUDY_DESIGN: response_rate=45%].
Do not just note "152 days field lag" - record it with write_log_entry or [LOG: Field lag of 152 days].
When you identify a variable mapping, record it immediately.

Every important finding should be recorded so it persists in the log file.

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
    conversation_history: list = None,
    active_logger: "ActiveLogger" = None
) -> str:
    """
    Send a message to Claude and get a response.

    Uses LiteLLM API with tool support for logging, or CLI if configured.
    """
    system_prompt = build_system_prompt(state)

    # Check model configuration
    model = os.getenv("LLM_MODEL_VALIDATE") or os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

    # If model is "claude-cli", use CLI path (doesn't support tools)
    if model == "claude-cli":
        claude_path = shutil.which("claude")
        if claude_path:
            return _call_claude_cli(user_message, system_prompt, conversation_history)
        else:
            return "Error: claude-cli configured but Claude CLI not found in PATH"

    # Otherwise use LiteLLM with tool support
    return _call_litellm(user_message, system_prompt, conversation_history, active_logger, state)


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
    conversation_history: list = None,
    active_logger: "ActiveLogger" = None,
    state: WorkflowState = None
) -> str:
    """Call Claude via LiteLLM API with tool support for logging."""
    try:
        from litellm import completion

        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": user_message})

        # Get model from environment
        model = os.getenv("LLM_MODEL_VALIDATE") or os.getenv("LLM_MODEL", "anthropic/claude-sonnet-4-20250514")

        # Include tools only if we have an active_logger to execute them
        tools_param = LOG_TOOLS if active_logger else None

        response = completion(
            model=model,
            messages=messages,
            tools=tools_param,
            max_tokens=2048,
            temperature=0.7
        )

        # Check if there are tool calls to execute
        message = response.choices[0].message

        if active_logger and hasattr(message, 'tool_calls') and message.tool_calls:
            # Execute tool calls and continue conversation if needed
            return _execute_tool_loop(messages, message, active_logger, state, model)

        # No tool calls, just return the content
        content = message.content
        return content.strip() if content else ""

    except Exception as e:
        return f"Error calling API: {e}"


def _execute_tool_loop(
    messages: list,
    initial_response_message,
    active_logger: "ActiveLogger",
    state: WorkflowState,
    model: str,
    max_iterations: int = 10
) -> str:
    """
    Execute tool calls in a loop until Claude returns a final text response.

    This handles cases where Claude may need to call multiple tools.
    """
    from litellm import completion

    current_message = initial_response_message

    for _ in range(max_iterations):
        if not hasattr(current_message, 'tool_calls') or not current_message.tool_calls:
            # No more tool calls, return the content
            return current_message.content.strip() if current_message.content else ""

        # Add assistant message with tool calls to history
        messages.append({
            "role": "assistant",
            "content": current_message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in current_message.tool_calls
            ]
        })

        # Execute each tool call
        tool_results = []
        for tool_call in current_message.tool_calls:
            result = _execute_single_tool(tool_call, active_logger, state)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Add tool results to messages
        messages.extend(tool_results)

        # Call Claude again with tool results
        response = completion(
            model=model,
            messages=messages,
            tools=LOG_TOOLS,
            max_tokens=2048,
            temperature=0.7
        )

        current_message = response.choices[0].message

    # Max iterations reached, return whatever content we have
    return current_message.content.strip() if current_message.content else ""


def _execute_single_tool(tool_call, active_logger: "ActiveLogger", state: WorkflowState) -> str:
    """Execute a single tool call and return the result."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return f"Error: Invalid JSON arguments for {name}"

    if name == "write_log_entry":
        message = args.get("message", "")
        active_logger.log_message(message)
        print(f"  [Logged] {message[:60]}...")
        return f"Logged: {message}"

    elif name == "update_study_design":
        field = args.get("field", "")
        value = args.get("value", "")
        active_logger.update_study_design_section({field: value})
        print(f"  [Study Design] {field}: {value[:40]}...")
        return f"Updated study design: {field} = {value}"

    elif name == "add_collaborator_question":
        question = args.get("question", "")
        active_logger.add_collaborator_question(
            question,
            "From conversation",
            state.get_next_step() or 0
        )
        print(f"  [Question added] {question[:60]}...")
        return f"Added question: {question}"

    elif name == "update_variable_mapping":
        cses_code = args.get("cses_code", "")
        source_variable = args.get("source_variable", "")
        active_logger.update_variable_mapping(cses_code, source_variable)
        print(f"  [Variable] {cses_code} <- {source_variable}")
        return f"Mapped variable: {cses_code} = {source_variable}"

    else:
        return f"Unknown tool: {name}"


class ConversationSession:
    """Manages a conversation session with Claude."""

    def __init__(self, state: WorkflowState):
        self.state = state
        self.history = []
        self.active_logger = ActiveLogger(state)

    def send(self, message: str) -> str:
        """Send a message and get a response."""
        import re

        # Add user message to history
        self.history.append({"role": "user", "content": message})

        # Get response - pass active_logger so tools can write to log directly
        response = call_claude_conversation(
            message, self.state, self.history, self.active_logger
        )

        # For CLI path (which doesn't support tools), process markers as fallback
        model = os.getenv("LLM_MODEL_VALIDATE") or os.getenv("LLM_MODEL", "")
        if model == "claude-cli":
            response = self._process_log_markers(response)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        # Save state after any updates
        self.state.save()

        return response

    def _process_log_markers(self, response: str) -> str:
        """
        Parse response for log markers and write to log file.
        Used as fallback for CLI path which doesn't support tools.
        """
        import re

        # Process [LOG: ...] markers
        log_pattern = r'\[LOG:\s*(.+?)\]'
        log_matches = re.findall(log_pattern, response, re.DOTALL)
        for log_entry in log_matches:
            log_entry = log_entry.strip()
            self.active_logger.log_message(log_entry)
            print(f"  [Logged] {log_entry[:60]}...")

        # Process [STUDY_DESIGN: field=value] markers
        design_pattern = r'\[STUDY_DESIGN:\s*(\w+)\s*=\s*(.+?)\]'
        design_matches = re.findall(design_pattern, response)
        if design_matches:
            for field, value in design_matches:
                self.active_logger.update_study_design_section({field.strip(): value.strip()})
                print(f"  [Study Design] {field}: {value[:40]}...")

        # Process [QUESTION: ...] markers
        question_pattern = r'\[QUESTION:\s*(.+?)\]'
        question_matches = re.findall(question_pattern, response, re.DOTALL)
        for question in question_matches:
            question = question.strip()
            self.active_logger.add_collaborator_question(
                question,
                "From conversation",
                self.state.get_next_step() or 0
            )
            print(f"  [Question added] {question[:60]}...")

        # Process [VARIABLE: cses_code=source_variable] markers
        variable_pattern = r'\[VARIABLE:\s*(\w+)\s*=\s*(.+?)\]'
        variable_matches = re.findall(variable_pattern, response)
        for cses_code, source_var in variable_matches:
            self.active_logger.update_variable_mapping(
                cses_code.strip(),
                source_var.strip()
            )

        return response

    def refresh_state(self):
        """Reload state from disk."""
        new_state = WorkflowState.load(Path(self.state.working_dir))
        if new_state:
            self.state = new_state
            self.active_logger = ActiveLogger(new_state)
