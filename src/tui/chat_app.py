"""CSES Chat TUI using Textual."""
from textual.app import App, ComposeResult
from textual.containers import VerticalScroll
from textual.widgets import Header, Footer, Input, Static
from textual.binding import Binding
import asyncio


class Message(Static):
    """A chat message."""
    pass


class CSESChat(App):
    """CSES Assistant Terminal Interface."""

    CSS = """
    Message {
        margin: 1 2;
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
    #chat-view {
        height: 1fr;
    }
    #input {
        dock: bottom;
        height: auto;
        max-height: 5;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
    ]

    def __init__(self, state, session):
        super().__init__()
        self.state = state
        self.session = session
        self.title = f"CSES Assistant - {state.country} {state.year}"

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat-view")
        yield Input(placeholder="Type message... (Ctrl+C to quit)", id="input")
        yield Footer()

    def on_mount(self):
        """Show welcome message."""
        self._add_message(
            f"Working on {self.state.country} {self.state.year}.\n"
            f"I can read files, document findings, and guide you through CSES workflow.\n"
            f"Try: 'List the files in micro/original_deposit'",
            role="system"
        )

    def _add_message(self, content: str, role: str = "user"):
        """Add a message to the chat view."""
        chat = self.query_one("#chat-view")
        msg = Message(content, classes=role)
        chat.mount(msg)
        msg.scroll_visible()

    async def on_input_submitted(self, event: Input.Submitted):
        """Handle user input submission."""
        message = event.value.strip()
        if not message:
            return

        # Clear input
        event.input.value = ""

        # Show user message
        self._add_message(message, "user")

        # Show thinking indicator
        self._add_message("Thinking...", "system")

        # Get response in background thread
        response = await asyncio.to_thread(self.session.send, message)

        # Remove "Thinking..." and show response
        chat = self.query_one("#chat-view")
        thinking_msgs = chat.query(".system")
        if thinking_msgs:
            thinking_msgs.last().remove()
        self._add_message(response, "assistant")

    def action_clear(self):
        """Clear the chat view."""
        chat = self.query_one("#chat-view")
        chat.remove_children()
        self.on_mount()  # Show welcome message again

    def action_quit(self):
        """Quit the application."""
        self.exit()
