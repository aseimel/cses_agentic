"""CSES Chat TUI using Textual with three-panel layout."""
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Header, Footer, Input, Static, Label
from textual.binding import Binding
import asyncio
from pathlib import Path

from src.workflow.state import WorkflowState, WORKFLOW_STEPS


class WorkflowPanel(Static):
    """Left sidebar showing workflow status, questions, and log file info."""

    def __init__(self, state: WorkflowState, **kwargs):
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Label("WORKFLOW STATUS", classes="panel-title")
        yield Static(id="progress-info")
        yield Static(id="steps-list")
        yield Label("PENDING QUESTIONS", classes="panel-title")
        yield Static(id="questions-list")
        yield Label("FILES", classes="panel-title")
        yield Static(id="file-info")

    def on_mount(self):
        """Initial render of workflow status."""
        self.refresh_status()

    def refresh_status(self):
        """Update all workflow status displays."""
        # Progress info
        progress = self.state.get_progress_summary()
        progress_text = (
            f"Progress: {progress['percent_complete']:.0f}%\n"
            f"Done: {progress['completed']}/{progress['total_steps']}\n"
            f"In progress: {progress['in_progress']}\n"
            f"Blocked: {progress['blocked']}"
        )
        self.query_one("#progress-info", Static).update(progress_text)

        # Steps list
        status_markers = {
            "not_started": "[    ]",
            "in_progress": "[....]",
            "blocked": "[WAIT]",
            "completed": "[DONE]",
            "skipped": "[SKIP]"
        }

        steps_lines = []
        for step_num in sorted(WORKFLOW_STEPS.keys()):
            step_info = WORKFLOW_STEPS[step_num]
            step_state = self.state.get_step(step_num)
            marker = status_markers.get(step_state.status, "[????]")
            name = step_info['name']
            # Truncate long names
            if len(name) > 18:
                name = name[:16] + ".."
            steps_lines.append(f"{marker} {step_num}: {name}")

        self.query_one("#steps-list", Static).update("\n".join(steps_lines))

        # Pending questions
        pending = self.state.get_pending_questions()
        if pending:
            q_lines = []
            for q in pending[:5]:  # Show first 5
                text = q.get("question", "")[:25]
                q_lines.append(f"- {text}...")
            if len(pending) > 5:
                q_lines.append(f"  +{len(pending) - 5} more")
            self.query_one("#questions-list", Static).update("\n".join(q_lines))
        else:
            self.query_one("#questions-list", Static).update("(none)")

        # File info
        file_lines = []
        if self.state.log_file:
            log_name = Path(self.state.log_file).name
            if len(log_name) > 25:
                log_name = log_name[:22] + "..."
            file_lines.append(f"Log: {log_name}")
        if self.state.data_file:
            file_lines.append(f"Data: {Path(self.state.data_file).name}")
        self.query_one("#file-info", Static).update("\n".join(file_lines) if file_lines else "(none)")


class ToolLog(Static):
    """Bottom panel showing recent tool execution output."""

    DEFAULT_CSS = """
    ToolLog {
        height: 5;
        border-top: solid $secondary;
        padding: 0 1;
        background: $surface-darken-2;
    }
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._entries = []

    def add_entry(self, msg: str):
        """Add a tool output entry."""
        self._entries.append(msg)
        # Keep only last 10 entries
        if len(self._entries) > 10:
            self._entries = self._entries[-10:]
        # Show last 3 entries in the panel
        display_entries = self._entries[-3:]
        self.update("\n".join(display_entries))


class Message(Static):
    """A chat message."""
    pass


class CSESChat(App):
    """CSES Assistant Terminal Interface with three-panel layout."""

    CSS = """
    /* Main layout */
    #main-container {
        height: 1fr;
    }

    #sidebar {
        width: 28;
        border-right: solid $primary;
        padding: 0 1;
        background: $surface-darken-1;
    }

    #chat-container {
        width: 1fr;
    }

    #chat-view {
        height: 1fr;
        padding: 0 1;
    }

    #tool-log {
        height: 5;
        border-top: solid $secondary;
        padding: 0 1;
        background: $surface-darken-2;
    }

    #input {
        dock: bottom;
        height: auto;
        max-height: 5;
    }

    /* Panel titles */
    .panel-title {
        text-style: bold;
        background: $primary;
        color: $text;
        padding: 0 1;
        margin-top: 1;
    }

    /* Messages */
    Message {
        margin: 1 0;
        padding: 1 2;
    }

    .user {
        background: $primary-darken-3;
        border: tall $primary;
    }

    .assistant {
        background: $surface;
        border: tall $secondary;
    }

    .system {
        background: $warning-darken-3;
        text-style: italic;
    }

    /* Status displays in sidebar */
    #progress-info {
        padding: 0 1;
        margin-bottom: 1;
    }

    #steps-list {
        padding: 0 1;
        height: auto;
    }

    #questions-list {
        padding: 0 1;
    }

    #file-info {
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self, state: WorkflowState, session):
        super().__init__()
        self.state = state
        self.session = session
        self.title = f"CSES Assistant - {state.country} {state.year}"

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            yield WorkflowPanel(self.state, id="sidebar")
            with Vertical(id="chat-container"):
                yield VerticalScroll(id="chat-view")
                yield ToolLog(id="tool-log")
        yield Input(placeholder="Type message... (Ctrl+C to quit)", id="input")
        yield Footer()

    def on_mount(self):
        """Show welcome message and initial status."""
        self._add_message(
            f"Working on {self.state.country} {self.state.year}.\n"
            f"I can read files, document findings, and guide you through CSES workflow.\n"
            f"Type 'status' to refresh, or ask me anything.",
            role="system"
        )
        # Initial tool log message
        self.query_one("#tool-log", ToolLog).update("Tool activity will appear here...")

    def _add_message(self, content: str, role: str = "user"):
        """Add a message to the chat view."""
        chat = self.query_one("#chat-view")
        msg = Message(content, classes=role)
        chat.mount(msg)
        msg.scroll_visible()

    def on_tool_output(self, msg: str):
        """Callback for tool execution feedback - called from background thread."""
        # Use call_from_thread to safely update UI from background thread
        self.call_from_thread(self._update_tool_log, msg)

    def _update_tool_log(self, msg: str):
        """Update tool log from main thread."""
        self.query_one("#tool-log", ToolLog).add_entry(msg)

    async def _refresh_sidebar_async(self):
        """Refresh the workflow status sidebar asynchronously."""
        # Reload state from disk in background thread to avoid blocking UI
        new_state = await asyncio.to_thread(
            WorkflowState.load,
            Path(self.state.working_dir)
        )
        if new_state:
            self.state = new_state
            self.session.state = new_state
        self.query_one("#sidebar", WorkflowPanel).state = self.state
        self.query_one("#sidebar", WorkflowPanel).refresh_status()

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle user input submission."""
        message = event.value.strip()
        if not message:
            return

        # Clear input
        event.input.value = ""

        # Handle special commands
        if message.lower() == "status":
            await self._refresh_sidebar_async()
            self._add_message("Workflow status refreshed.", "system")
            return

        # Show user message
        self._add_message(message, "user")

        # Show thinking indicator
        self._add_message("Thinking...", "system")

        # Get response in background thread, passing our callback
        response = await asyncio.to_thread(
            self.session.send,
            message,
            on_tool_output=self.on_tool_output
        )

        # Remove "Thinking..." and show response
        chat = self.query_one("#chat-view")
        thinking_msgs = chat.query(".system")
        if thinking_msgs:
            thinking_msgs.last().remove()
        self._add_message(response, "assistant")

        # Refresh sidebar after response (state may have changed)
        await self._refresh_sidebar_async()

    def action_clear(self):
        """Clear the chat view."""
        chat = self.query_one("#chat-view")
        chat.remove_children()
        self.on_mount()  # Show welcome message again

    def action_quit(self):
        """Quit the application."""
        self.exit()
