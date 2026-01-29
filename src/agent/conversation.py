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
    }
]


# CSES Expert System Prompt - Active Agent Behavior
CSES_EXPERT_PROMPT = """You are an active CSES data processing agent for Module 6. You execute tasks, not explain them.

## Current Study
Country: {country}
Year: {year}
Working Directory: {working_dir}

## Workflow Status
{workflow_status}

## Available Files
{file_info}

## CRITICAL INSTRUCTIONS

YOU MUST FOLLOW THIS EXACT RESPONSE PATTERN:

1. When user says "yes", "proceed", "continue", "go ahead", or similar:
   - USE TOOLS IMMEDIATELY to execute the task
   - Then report: "Done. I [what you did]. Next: [next step]. Proceed?"

2. When user asks you to do something:
   - USE TOOLS to do it (read_file, list_files, write_log_entry, etc.)
   - Report what you found/did
   - Propose next action: "Next: [action]. Proceed?"

3. NEVER say "I can...", "Would you like me to...", "Should I..."
   - WRONG: "I can read the design report for you"
   - RIGHT: [use read_file tool] "The design report shows sample size 1500. Logged. Next: check questionnaire. Proceed?"

## Example Correct Response
User: "yes"
You: [call list_files tool] [call read_file on design report] [call write_log_entry]
"Done. Found 5 files in deposit. Design report shows:
- Sample: 1500 respondents
- Mode: Face-to-face
- Period: April 2024
Logged to file. Next: Read questionnaire to identify variables. Proceed?"

## Tools (USE THEM)
- list_files(directory) - List files
- read_file(path) - Read any file
- write_log_entry(message) - Log findings
- update_study_design(field, value) - Record study design
- update_election_summary(summary) - Record election info
- add_collaborator_question(question) - Add question for collaborator

## Workflow Steps
0. Set Up Folder - DONE
1. Check Deposit - List and verify files
2. Read Design Report - Extract methodology
3. Fill Tracking Sheet - Map variables
4. Write Study Design - Document weights
5-16. [Later steps]

## RULES
- USE TOOLS for every action
- NEVER ask permission to read files
- ALWAYS end with "Proceed?" after proposing next step

## RESPONSE DETAIL
When reading documents, provide DETAILED findings:
- List specific values found (sample size, dates, response rates, modes)
- Quote relevant passages when appropriate
- Name specific parties, candidates, institutions mentioned
- Report what you logged to the file and confirm success
- Be thorough in your analysis, not brief"""


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
    active_logger: "ActiveLogger" = None,
    on_tool_output: callable = None
) -> str:
    """
    Send a message to Claude and get a response.

    Uses LiteLLM API with tool support for logging, or CLI if configured.

    Args:
        user_message: The user's message
        state: Current workflow state
        conversation_history: Previous conversation messages
        active_logger: Logger for writing to CSES log files
        on_tool_output: Optional callback for tool execution feedback (for TUI)
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
    return _call_litellm(user_message, system_prompt, conversation_history, active_logger, state, on_tool_output)


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
    state: WorkflowState = None,
    on_tool_output: callable = None
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
            return _execute_tool_loop(messages, message, active_logger, state, model, on_tool_output=on_tool_output)

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
    max_iterations: int = 10,
    on_tool_output: callable = None
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
            result = _execute_single_tool(tool_call, active_logger, state, on_tool_output)
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

    else:
        return f"Unknown tool: {name}"


class ConversationSession:
    """Manages a conversation session with Claude."""

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
        response = call_claude_conversation(
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
