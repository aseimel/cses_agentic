"""
Conversational LLM Interface for CSES Processing.

After the initial folder setup, this module provides a natural language
interface where users can chat with the LLM about the CSES workflow.
The LLM has expert knowledge of the CSES process and guides users through
each step.

The LLM has direct tool access to write to the log file in real-time using
tools like write_log_entry, update_study_design, add_collaborator_question,
and update_variable_mapping.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from src.workflow.active_logging import ActiveLogger

from src.workflow.state import WorkflowState, WORKFLOW_STEPS, StepStatus, STEP_PREREQUISITES
from src.workflow.active_logging import ActiveLogger


# Tool definitions for the LLM to update log files directly
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
    },
    {
        "type": "function",
        "function": {
            "name": "update_election_summary",
            "description": "Document election context: date, type, outcome, turnout, significance. Use when reviewing design reports or election documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Election summary text including date, type, outcome, turnout"
                    }
                },
                "required": ["summary"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "update_parties_leaders",
            "description": "Document political parties, candidates, coalitions, and leaders relevant to the election.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "Information about parties, leaders, and candidates"
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_todo_item",
            "description": "Add an item to the pre-release checklist. Use for tasks that must be completed before data release.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "TODO item description"
                    }
                },
                "required": ["item"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file in the study folder. Use for design reports, questionnaires, codebooks, and other documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to study folder (e.g., 'micro/original_deposit/design_report.pdf')"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in a directory of the study folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "Directory path relative to study folder (default: study root)"
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "start_step",
            "description": "Start working on a workflow step. MUST call this BEFORE doing any work on a step. Checks prerequisites and marks step as in progress.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_num": {
                        "type": "integer",
                        "description": "Step number (0-16)"
                    }
                },
                "required": ["step_num"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "complete_step",
            "description": "Complete a workflow step. Call this AFTER finishing all work on a step. Marks step as completed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "step_num": {
                        "type": "integer",
                        "description": "Step number (0-16)"
                    },
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what was accomplished"
                    }
                },
                "required": ["step_num", "summary"]
            }
        }
    }
]


# CSES Expert System Prompt - Active Agent with Workflow Enforcement
CSES_EXPERT_PROMPT = """You are a CSES data processing agent for Module 6. You EXECUTE tasks using tools.

## Current Study
Country: {country}
Year: {year}
Working Dir: {working_dir}

## Workflow Status (work on [NEXT] step ONLY)
{workflow_status}

## Available Files
{file_info}

## MANDATORY WORKFLOW RULES

1. ALWAYS call start_step(N) BEFORE doing any work on step N
   - If it returns BLOCKED, STOP and tell the user which prerequisite step is missing
   - If it returns SKIP (already complete), move to next step
   - If it returns SUCCESS or CONTINUE, proceed with the work

2. When user says "yes/proceed/continue":
   - Call start_step for the [NEXT] step
   - USE TOOLS to do the work (read_file, list_files, write_log_entry)
   - Call complete_step when finished
   - Report: "Done. [summary]. Next: Step N - [name]. Proceed?"

3. ALWAYS call complete_step(N, summary) AFTER finishing step N

4. NEVER skip steps or work on steps out of order

5. NEVER say "I can...", "Would you like...", "Should I..."
   - Just DO IT with tools

## Tools
- start_step(step_num) - MUST call before starting a step
- complete_step(step_num, summary) - MUST call after finishing a step
- list_files(directory) - List files
- read_file(path) - Read any file
- write_log_entry(message) - Log findings
- update_study_design(field, value) - Record study design
- update_election_summary(summary) - Record election info
- add_collaborator_question(question) - Add question for collaborator

## Example Correct Flow
User: "proceed"
1. [call start_step(1)] -> SUCCESS
2. [call list_files] -> find files
3. [call read_file on design report]
4. [call write_log_entry with findings]
5. [call complete_step(1, "Verified deposit: data file, questionnaire, design report")]
Response: "Done. Step 1 complete. Found data file (N=1500), questionnaire, design report. All logged. Next: Step 2 - Read Design Report. Proceed?"

## RESPONSE DETAIL
When reading documents, provide DETAILED findings:
- List specific values (sample size, dates, response rates, modes)
- Quote relevant passages
- Name specific parties, candidates, institutions
- Report what was logged
- Be thorough, not brief"""


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


def _completion_with_retry(
    model: str,
    messages: list,
    tools: list = None,
    tool_choice: str = "auto",
    max_retries: int = 3
):
    """
    Call LiteLLM completion with retry logic for transient failures.

    Uses exponential backoff: 1s, 2s, 4s between retries.
    """
    from litellm import completion

    for attempt in range(max_retries):
        try:
            return completion(
                model=model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                parallel_tool_calls=False,  # Sequential tool execution for reliability
                max_tokens=2048,
                temperature=0.3
            )
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 1s, 2s, 4s
                print(f"[RETRY] Attempt {attempt + 1} failed: {e}. Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise


def call_llm_conversation(
    user_message: str,
    state: WorkflowState,
    conversation_history: list = None,
    active_logger: "ActiveLogger" = None,
    on_tool_output: callable = None
) -> str:
    """
    Send a message to the LLM and get a response.

    Uses LiteLLM API with tool support for logging.

    Args:
        user_message: The user's message
        state: Current workflow state
        conversation_history: Previous conversation messages
        active_logger: Logger for writing to CSES log files
        on_tool_output: Optional callback for tool execution feedback (for TUI)
    """
    system_prompt = build_system_prompt(state)

    # Use LiteLLM with tool support
    return _call_litellm(user_message, system_prompt, conversation_history, active_logger, state, on_tool_output)


def _call_litellm(
    user_message: str,
    system_prompt: str,
    conversation_history: list = None,
    active_logger: "ActiveLogger" = None,
    state: WorkflowState = None,
    on_tool_output: callable = None
) -> str:
    """Call LLM via LiteLLM API with tool support for logging."""
    try:
        messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history
        if conversation_history:
            messages.extend(conversation_history[-10:])

        messages.append({"role": "user", "content": user_message})

        # Get model from environment (default to gpt-4.1 for reliable tool calling)
        model = os.getenv("LLM_MODEL_VALIDATE") or os.getenv("LLM_MODEL", "openai/gpt-4.1")

        # Include tools only if we have an active_logger to execute them
        tools_param = LOG_TOOLS if active_logger else None

        print(f"[DEBUG] Calling {model} with {len(messages)} messages...")

        response = _completion_with_retry(
            model=model,
            messages=messages,
            tools=tools_param,
            tool_choice="auto"
        )

        # Check if there are tool calls to execute
        message = response.choices[0].message

        print(f"[DEBUG] Response: content={bool(message.content)}, tools={bool(getattr(message, 'tool_calls', None))}")

        if active_logger and hasattr(message, 'tool_calls') and message.tool_calls:
            # Execute tool calls and continue conversation if needed
            return _execute_tool_loop(messages, message, active_logger, state, model, on_tool_output=on_tool_output)

        # No tool calls, just return the content (never return empty)
        content = message.content
        if not content or not content.strip():
            print(f"[WARNING] LLM returned empty content")
            return "[No response from model - please try again]"
        return content.strip()

    except Exception as e:
        import traceback
        print(f"[ERROR] API call failed: {e}")
        traceback.print_exc()
        return f"[Error calling API: {e}. Please try again.]"


def _execute_tool_loop(
    messages: list,
    initial_response_message,
    active_logger: "ActiveLogger",
    state: WorkflowState,
    model: str,
    max_iterations: int = 10,
    on_tool_output: callable = None
) -> str:
    """
    Execute tool calls in a loop until the LLM returns a final text response.

    This handles cases where the LLM may need to call multiple tools.
    Uses parallel_tool_calls=False for reliable sequential execution.
    Falls back to forced text synthesis if max iterations reached without text.
    """
    current_message = initial_response_message

    for iteration in range(max_iterations):
        if not hasattr(current_message, 'tool_calls') or not current_message.tool_calls:
            # No more tool calls, return the content (never return empty)
            content = current_message.content
            if not content or not content.strip():
                print(f"[WARNING] Empty response after {iteration} tool iterations")
                return "[Operations completed but no response - check the log file for results]"
            return content.strip()

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
            result = _execute_single_tool(tool_call, active_logger, state, on_tool_output)
            tool_results.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Add tool results to messages
        messages.extend(tool_results)

        # Call LLM again with tool results (using retry helper)
        response = _completion_with_retry(
            model=model,
            messages=messages,
            tools=LOG_TOOLS,
            tool_choice="auto"
        )

        current_message = response.choices[0].message
        print(f"[DEBUG] Iteration {iteration + 1}: content={bool(current_message.content)}, tools={bool(getattr(current_message, 'tool_calls', None))}")

    # Max iterations reached - try fallback synthesis with tool_choice="none"
    content = current_message.content
    if not content or not content.strip():
        print(f"[DEBUG] Max iterations ({max_iterations}) reached. Forcing text synthesis with tool_choice='none'...")

        # Add a prompt to request summary
        messages.append({"role": "user", "content": "[Summarize what was accomplished in the operations above]"})

        try:
            response = _completion_with_retry(
                model=model,
                messages=messages,
                tools=LOG_TOOLS,
                tool_choice="none"  # Force text output, no tool calls
            )
            content = response.choices[0].message.content
        except Exception as e:
            print(f"[WARNING] Fallback synthesis failed: {e}")
            content = None

    if content and content.strip():
        return content.strip()

    # Last resort fallback message
    return "[Operations completed. Check log file for details.]"


def _execute_single_tool(tool_call, active_logger: "ActiveLogger", state: WorkflowState, on_tool_output: callable = None) -> str:
    """Execute a single tool call with verification. Returns result string."""
    name = tool_call.function.name
    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError:
        return f"FAILED: Invalid JSON arguments for {name}"

    def notify(msg: str):
        """Send tool output to callback or print."""
        if on_tool_output:
            on_tool_output(msg)

    if name == "write_log_entry":
        message = args.get("message", "")
        success, status = active_logger.log_message(message)
        if success:
            notify(f"[OK] {status}")
            return f"SUCCESS: {status}"
        else:
            notify(f"[FAILED] {status}")
            return f"FAILED: {status}"

    elif name == "update_study_design":
        field = args.get("field", "")
        value = args.get("value", "")
        success, status = active_logger.update_study_design_section({field: value})
        if success:
            notify(f"[OK] Study design: {field} = {value[:30]}...")
            return f"SUCCESS: Updated {field} = {value}"
        else:
            notify(f"[FAILED] {status}")
            return f"FAILED to update study design: {status}"

    elif name == "add_collaborator_question":
        question = args.get("question", "")
        success, status = active_logger.add_collaborator_question(
            question,
            "From conversation",
            state.get_next_step() or 0
        )
        if success:
            notify(f"[OK] {status}")
            return f"SUCCESS: {status}"
        else:
            notify(f"[FAILED] {status}")
            return f"FAILED: {status}"

    elif name == "update_variable_mapping":
        cses_code = args.get("cses_code", "")
        source_variable = args.get("source_variable", "")
        active_logger.update_variable_mapping(cses_code, source_variable)
        notify(f"[Variable] {cses_code} <- {source_variable}")
        return f"Mapped variable: {cses_code} = {source_variable}"

    elif name == "update_election_summary":
        summary = args.get("summary", "")
        success, status = active_logger.update_election_summary(summary)
        if success:
            notify(f"[OK] {status}")
            return f"SUCCESS: {status}"
        else:
            notify(f"[FAILED] {status}")
            return f"FAILED: {status}"

    elif name == "update_parties_leaders":
        content = args.get("content", "")
        success, status = active_logger.update_parties_leaders(content)
        if success:
            notify(f"[OK] {status}")
            return f"SUCCESS: {status}"
        else:
            notify(f"[FAILED] {status}")
            return f"FAILED: {status}"

    elif name == "add_todo_item":
        item = args.get("item", "")
        success, status = active_logger.add_todo_item(item)
        if success:
            notify(f"[OK] {status}")
            return f"SUCCESS: {status}"
        else:
            notify(f"[FAILED] {status}")
            return f"FAILED: {status}"

    elif name == "read_file":
        path = args.get("path", "")
        full_path = Path(state.working_dir) / path
        if not full_path.exists():
            return f"Error: File not found: {path}"
        try:
            if full_path.suffix.lower() == '.pdf':
                from pypdf import PdfReader
                reader = PdfReader(full_path)
                text = "\n".join(page.extract_text() or "" for page in reader.pages)
                notify(f"[Read] {path} ({len(text)} chars)")
                return text[:50000]  # Limit for context
            else:
                content = full_path.read_text(errors='replace')
                notify(f"[Read] {path} ({len(content)} chars)")
                return content[:50000]
        except Exception as e:
            return f"Error reading file: {e}"

    elif name == "list_files":
        directory = args.get("directory", "")
        dir_path = Path(state.working_dir) / directory
        if not dir_path.exists():
            return f"Error: Directory not found: {directory}"
        files = []
        for f in dir_path.rglob("*"):
            if f.is_file():
                rel_path = f.relative_to(Path(state.working_dir))
                files.append(str(rel_path))
        notify(f"[Listed] {directory or '.'} ({len(files)} files)")
        return "\n".join(files[:100])  # Limit to 100 files

    elif name == "start_step":
        step_num = args.get("step_num")
        if step_num is None or step_num not in WORKFLOW_STEPS:
            return f"FAILED: Invalid step number: {step_num}"

        # Check prerequisites
        can_proceed, reason = state.check_step_prerequisites(step_num)
        if not can_proceed:
            notify(f"[BLOCKED] Cannot start step {step_num}: {reason}")
            return f"BLOCKED: {reason}"

        # Check if step is already in progress or completed
        current_status = state.get_step(step_num).status
        if current_status == StepStatus.COMPLETED.value:
            return f"SKIP: Step {step_num} is already completed"
        if current_status == StepStatus.IN_PROGRESS.value:
            return f"CONTINUE: Step {step_num} is already in progress"

        # Mark as in progress
        state.set_step_status(step_num, StepStatus.IN_PROGRESS, "Started by agent")
        state.current_step = step_num
        state.save()

        step_name = WORKFLOW_STEPS[step_num]["name"]
        notify(f"[STARTED] Step {step_num}: {step_name}")
        return f"SUCCESS: Started Step {step_num} - {step_name}"

    elif name == "complete_step":
        step_num = args.get("step_num")
        summary = args.get("summary", "")

        if step_num is None or step_num not in WORKFLOW_STEPS:
            return f"FAILED: Invalid step number: {step_num}"

        # Verify step is currently in progress
        current_status = state.get_step(step_num).status
        if current_status != StepStatus.IN_PROGRESS.value:
            return f"FAILED: Step {step_num} is not in progress (status: {current_status})"

        # Mark as completed
        state.set_step_status(step_num, StepStatus.COMPLETED, summary)
        state.save()

        step_name = WORKFLOW_STEPS[step_num]["name"]
        notify(f"[DONE] Step {step_num}: {step_name}")

        # Suggest next step
        next_step = state.get_next_step()
        if next_step is not None:
            next_name = WORKFLOW_STEPS[next_step]["name"]
            return f"SUCCESS: Completed Step {step_num}. Next: Step {next_step} - {next_name}"
        return f"SUCCESS: Completed Step {step_num}. All steps complete!"

    else:
        return f"Unknown tool: {name}"


class ConversationSession:
    """Manages a conversation session with the LLM."""

    def __init__(self, state: WorkflowState):
        self.state = state
        self.history = []
        self.active_logger = ActiveLogger(state)

    def send(self, message: str, on_tool_output: callable = None) -> str:
        """Send a message and get a response.

        Args:
            message: The user's message
            on_tool_output: Optional callback for tool execution feedback (for TUI)
        """
        # Add user message to history
        self.history.append({"role": "user", "content": message})

        # Get response - pass active_logger so tools can write to log directly
        response = call_llm_conversation(
            message, self.state, self.history, self.active_logger, on_tool_output
        )

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        # Save state after any updates
        self.state.save()

        return response

    def refresh_state(self):
        """Reload state from disk."""
        new_state = WorkflowState.load(Path(self.state.working_dir))
        if new_state:
            self.state = new_state
            self.active_logger = ActiveLogger(new_state)
