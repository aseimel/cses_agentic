"""
Windows GUI for CSES Assistant.

The primary interface is chat. Workflow buttons are secondary tools for explicit
manual operations, not the main user experience.
"""

from __future__ import annotations

import os
import queue
import json
import subprocess
import sys
import threading
import tempfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.conversation import ConversationSession
from src.auth_profiles import CodexOAuthAuthSource
from src.model_catalog import build_model_catalog, litellm_model_for_completion, model_supports_tools
from src.model_profiles import DEFAULT_PROFILE_ID, MODEL_PROFILES
from src.project_context import create_starter_project_context, describe_project_context
from src.processor_decisions import DependencyInvalidator, ProcessorDecisionLedger
from src.settings import (
    DEFAULT_AGENTIC_MODEL,
    DEFAULT_CODEX_AGENTIC_MODEL,
    DEFAULT_DOCUMENTATION_MODEL,
    DEFAULT_FINAL_ESCALATION_MODEL,
    DEFAULT_GESIS_CHAT_MODEL,
    DEFAULT_LARGE_TEXT_MODEL,
    DEFAULT_MATCH_ESCALATION_MODEL,
    DEFAULT_MATCH_FAST_MODEL,
    DEFAULT_STATA_CODE_MODEL,
    DEFAULT_STATA_REPAIR_MODEL,
    DEFAULT_STUDY_KB_MODEL,
    DEFAULT_VERIFIER_MODEL,
    GESIS_OPENWEBUI_BASE,
    PROVIDER_TOGGLES,
    SUPPORTED_API_KEYS,
    UserSettings,
    load_settings,
    save_settings,
)
from src.stata_mcp import MCP_STATA_REPOSITORY, MCPStataRunner
from src.study_kb import StudyKnowledgeBase, StudyKnowledgeBaseBuilder
from src.ui_text import format_study_review_status, sanitize_processor_text
from src.workflow.evidence_packet import EvidencePacketBuilder
from src.workflow.organizer import FileOrganizer
from src.workflow.phases import WORKFLOW_PHASES, current_phase_id, phase_status, phase_status_payload
from src.workflow.state import WORKFLOW_STEPS, StepStatus, WorkflowState


MODEL_PLACEHOLDER = "Please select a model"


def json_payload_for_models(rows, warnings: list[str]) -> str:
    return json.dumps(
        {
            "rows": [row.__dict__ for row in rows],
            "warnings": warnings,
        }
    )


def parse_models_payload(payload: str) -> dict:
    return json.loads(payload)


class CSESGui(tk.Tk):
    """Main CSES Assistant window."""

    def __init__(self):
        super().__init__()
        self.title("CSES Assistant")
        self.geometry("1040x720")
        self.minsize(900, 620)

        self.settings = load_settings(PROJECT_ROOT)
        self.queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.current_process: subprocess.Popen | None = None
        self.conversation: ConversationSession | None = None
        self.loaded_state: WorkflowState | None = None
        self.setting_vars: dict[str, tk.StringVar] = {}
        self.api_key_entries: dict[str, ttk.Entry] = {}
        self.selected_model_rows: set[str] = set()
        self.model_row_data: dict[str, dict] = {}
        self.role_model_combos: list[ttk.Combobox] = []
        self.profile_label_to_id = {profile.label: profile.id for profile in MODEL_PROFILES.values()}
        self.profile_id_to_label = {profile.id: profile.label for profile in MODEL_PROFILES.values()}
        self.codex_status_var = tk.StringVar(value="Codex OAuth: not checked")

        self._build_style()
        self._build_layout()
        self.after(100, self._drain_queue)

    def _build_style(self) -> None:
        self.style = ttk.Style(self)
        if "vista" in self.style.theme_names():
            self.style.theme_use("vista")
        self.style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        self.style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"))
        self.style.configure("Status.TLabel", foreground="#3b5b7a")
        self.style.configure("Run.TButton", padding=(10, 5))
        self.style.configure("Panel.TFrame", background="#f4f6f8")
        self.style.configure("PanelTitle.TLabel", font=("Segoe UI", 10, "bold"), background="#f4f6f8")
        self.style.configure("PanelText.TLabel", background="#f4f6f8")
        self.style.configure("Composer.TFrame", background="#eef2f5")

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        header = ttk.Frame(container)
        header.pack(fill="x", pady=(0, 12))
        ttk.Label(header, text="CSES Assistant", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            header,
            text="Chat with a workflow assistant that documents decisions and waits for your approval.",
        ).pack(anchor="w", pady=(4, 0))

        notebook = ttk.Notebook(container)
        notebook.pack(fill="both", expand=True)

        self.chat_tab = ttk.Frame(notebook, padding=12)
        self.corrections_tab = ttk.Frame(notebook, padding=12)
        self.settings_tab = ttk.Frame(notebook, padding=12)
        self.about_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.chat_tab, text="Chat")
        notebook.add(self.corrections_tab, text="Corrections")
        notebook.add(self.settings_tab, text="Settings")
        notebook.add(self.about_tab, text="Stata")

        self._build_chat_tab()
        self._build_corrections_tab()
        self._build_settings_tab()
        self._build_about_tab()

    def _build_chat_tab(self) -> None:
        top = ttk.Frame(self.chat_tab)
        top.pack(fill="x")

        ttk.Label(top, text="Study folder", style="Section.TLabel").grid(row=0, column=0, sticky="w")
        self.folder_var = tk.StringVar(value=str(Path.cwd()))
        ttk.Entry(top, textvariable=self.folder_var).grid(row=1, column=0, sticky="ew", pady=(4, 0))
        ttk.Button(top, text="Browse", command=self._choose_folder).grid(row=1, column=1, padx=(8, 0), pady=(4, 0))
        ttk.Button(top, text="Load Study", command=self._load_study).grid(row=1, column=2, padx=(8, 0), pady=(4, 0))
        ttk.Label(top, text="Chat model").grid(row=2, column=0, sticky="w", pady=(8, 0))
        saved_models = [
            model.strip()
            for model in self.settings.get("CSES_CHAT_MODELS", "").split(",")
            if model.strip().startswith("openrouter/")
        ]
        for model in self._default_openrouter_models():
            if model not in saved_models:
                saved_models.append(model)
        saved_chat_model = self.settings.get("CSES_AGENTIC_MODEL") or self.settings.get("CSES_CHAT_MODEL") or DEFAULT_AGENTIC_MODEL
        if not saved_chat_model.startswith("openrouter/") or saved_chat_model not in saved_models:
            saved_chat_model = DEFAULT_AGENTIC_MODEL
        self.chat_model_var = tk.StringVar(value=saved_chat_model)
        self.chat_model_combo = ttk.Combobox(top, textvariable=self.chat_model_var, values=saved_models, state="readonly")
        self.chat_model_combo.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 0))
        self.chat_model_combo.bind("<<ComboboxSelected>>", lambda _event: self._refresh_active_model_display())
        top.columnconfigure(0, weight=1)

        main = ttk.PanedWindow(self.chat_tab, orient="horizontal")
        main.pack(fill="both", expand=True, pady=(12, 8))

        sidebar = ttk.Frame(main, width=310, padding=10, style="Panel.TFrame")
        sidebar.pack_propagate(False)
        main.add(sidebar, weight=0)

        ttk.Label(sidebar, text="Study", style="PanelTitle.TLabel").pack(anchor="w")
        self.study_panel = tk.Text(sidebar, height=6, wrap="word", state="disabled", font=("Segoe UI", 9), bg="#f4f6f8", relief="flat")
        self.study_panel.pack(fill="x", pady=(4, 10))

        ttk.Label(sidebar, text="Workflow", style="PanelTitle.TLabel").pack(anchor="w")
        self.workflow_panel = tk.Text(sidebar, height=21, wrap="word", state="disabled", font=("Segoe UI", 9), bg="#f4f6f8", relief="flat")
        self.workflow_panel.pack(fill="both", expand=True, pady=(4, 10))

        ttk.Label(sidebar, text="Questions and Files", style="PanelTitle.TLabel").pack(anchor="w")
        ttk.Button(sidebar, text="Review Study Materials", command=self._build_refresh_evidence_packet).pack(fill="x", pady=(0, 6))
        self.files_panel = tk.Text(sidebar, height=10, wrap="word", state="disabled", font=("Segoe UI", 9), bg="#f4f6f8", relief="flat")
        self.files_panel.pack(fill="x", pady=(4, 0))

        chat_frame = ttk.Frame(main, padding=(8, 0, 0, 0))
        main.add(chat_frame, weight=1)
        chat_frame.columnconfigure(0, weight=1)
        chat_frame.rowconfigure(0, weight=1)

        transcript_frame = ttk.Frame(chat_frame)
        transcript_frame.grid(row=0, column=0, sticky="nsew")
        transcript_frame.columnconfigure(0, weight=1)
        transcript_frame.rowconfigure(0, weight=1)

        self.chat_text = tk.Text(
            transcript_frame,
            wrap="word",
            state="disabled",
            font=("Segoe UI", 10),
            padx=12,
            pady=10,
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#c8d2dc",
        )
        self.chat_text.grid(row=0, column=0, sticky="nsew")
        chat_scroll = ttk.Scrollbar(transcript_frame, orient="vertical", command=self.chat_text.yview)
        chat_scroll.grid(row=0, column=1, sticky="ns")
        self.chat_text.configure(yscrollcommand=chat_scroll.set)
        self.chat_text.tag_configure("user_label", foreground="#174a7c", font=("Segoe UI", 10, "bold"), spacing1=10, spacing3=2)
        self.chat_text.tag_configure("assistant_label", foreground="#263238", font=("Segoe UI", 10, "bold"), spacing1=10, spacing3=2)
        self.chat_text.tag_configure("tool_label", foreground="#5c6470", font=("Segoe UI", 9, "bold"), spacing1=6, spacing3=2)
        self.chat_text.tag_configure("system_label", foreground="#6b4f00", font=("Segoe UI", 9, "bold"), spacing1=6, spacing3=2)
        self.chat_text.tag_configure("user", foreground="#174a7c", lmargin1=12, lmargin2=12, spacing3=6)
        self.chat_text.tag_configure("assistant", foreground="#222222", lmargin1=12, lmargin2=12, spacing3=8)
        self.chat_text.tag_configure("tool", foreground="#5c6470", lmargin1=12, lmargin2=12, spacing3=4)
        self.chat_text.tag_configure("system", foreground="#6b4f00", lmargin1=12, lmargin2=12, spacing3=6)

        input_row = ttk.Frame(chat_frame, padding=(8, 8, 8, 8), style="Composer.TFrame")
        input_row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        input_row.columnconfigure(0, weight=1)
        ttk.Label(input_row, text="Message", style="PanelTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        self.message_entry = tk.Text(
            input_row,
            height=4,
            wrap="word",
            font=("Segoe UI", 10),
            padx=10,
            pady=8,
            bg="#ffffff",
            relief="solid",
            borderwidth=1,
            highlightthickness=1,
            highlightbackground="#8aa2b5",
            insertbackground="#111111",
        )
        self.message_entry.grid(row=1, column=0, sticky="ew")
        self.message_entry.bind("<Return>", self._send_chat_from_enter)
        self.message_entry.bind("<Shift-Return>", lambda _event: None)
        button_stack = ttk.Frame(input_row, style="Composer.TFrame")
        button_stack.grid(row=1, column=1, sticky="ns", padx=(8, 0))
        self.send_button = ttk.Button(button_stack, text="Send", command=self._send_chat)
        self.send_button.pack(fill="both", expand=True)
        self.proceed_button = ttk.Button(button_stack, text="Proceed", command=self._proceed_chat)
        self.proceed_button.pack(fill="both", expand=True, pady=(6, 0))

        self.status_var = tk.StringVar(value="Choose a study folder and load study.")
        ttk.Label(self.chat_tab, textvariable=self.status_var, style="Status.TLabel").pack(anchor="w", pady=(8, 0))

        self._append_chat(
            "system",
            "Welcome. Load a study folder, then tell the assistant what you want to do. "
            "For workflow progress, say things like: What should we do next? or Proceed.",
        )
        self._refresh_active_model_display()
        self._refresh_sidebar(None)

    def _build_tools_tab(self) -> None:
        ttk.Label(self.tools_tab, text="Advanced actions", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            self.tools_tab,
            text="Use these only when you explicitly want to run a command outside the chat.",
        ).pack(anchor="w", pady=(3, 12))

        actions = ttk.Frame(self.tools_tab)
        actions.pack(fill="x")
        ttk.Button(actions, text="Show Status", style="Run.TButton", command=lambda: self._run_cses(["status"])).pack(side="left")
        ttk.Button(actions, text="Initialize Study", style="Run.TButton", command=lambda: self._run_cses(["init"])).pack(side="left", padx=8)
        ttk.Button(actions, text="Run Matching", style="Run.TButton", command=lambda: self._run_cses(["match"])).pack(side="left")
        ttk.Button(actions, text="Generate Stata Syntax", style="Run.TButton", command=lambda: self._run_cses(["generate"])).pack(side="left", padx=8)
        ttk.Button(actions, text="Review Study Materials", command=self._build_refresh_study_kb).pack(side="left")

        step_row = ttk.Frame(self.tools_tab)
        step_row.pack(fill="x", pady=12)
        ttk.Label(step_row, text="Workflow step").pack(side="left")
        self.step_var = tk.StringVar(value="1")
        ttk.Entry(step_row, textvariable=self.step_var, width=8).pack(side="left", padx=8)
        ttk.Button(step_row, text="Run Step", command=self._run_step).pack(side="left")
        ttk.Button(step_row, text="Stop", command=self._stop_process).pack(side="right")

        self.output_text = tk.Text(self.tools_tab, height=20, wrap="word", state="disabled", font=("Consolas", 10))
        self.output_text.pack(fill="both", expand=True, pady=(4, 0))

    def _build_context_tab(self) -> None:
        ttk.Label(self.context_tab, text="Project context files", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            self.context_tab,
            text="These plain-text files define the assistant role, local workflow notes, and project-specific skills.",
        ).pack(anchor="w", pady=(3, 10))

        button_row = ttk.Frame(self.context_tab)
        button_row.pack(fill="x", pady=(0, 10))
        ttk.Button(button_row, text="Create Starter Files", command=self._create_context_files).pack(side="left")
        ttk.Button(button_row, text="Refresh", command=self._refresh_context_panel).pack(side="left", padx=8)

        self.context_text = tk.Text(self.context_tab, wrap="word", state="disabled", font=("Consolas", 10), height=22)
        self.context_text.pack(fill="both", expand=True)
        self._set_panel_text(
            self.context_text,
            "Load a study in the Chat tab, then use this tab to inspect or create project context files.",
        )

    def _build_corrections_tab(self) -> None:
        ttk.Label(self.corrections_tab, text="Processor corrections", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            self.corrections_tab,
            text="Review corrections and approvals that should drive later coding, checks, and documentation.",
        ).pack(anchor="w", pady=(3, 10))

        self.corrections_tree = ttk.Treeview(
            self.corrections_tab,
            columns=("id", "area", "target", "value", "status", "affected"),
            show="headings",
            height=14,
        )
        for column, label, width in [
            ("id", "ID", 82),
            ("area", "Area", 140),
            ("target", "Item", 150),
            ("value", "Decision", 220),
            ("status", "Status", 130),
            ("affected", "Affected work", 260),
        ]:
            self.corrections_tree.heading(column, text=label)
            self.corrections_tree.column(column, width=width, anchor="w")
        self.corrections_tree.pack(fill="both", expand=True)

        edit = ttk.Frame(self.corrections_tab)
        edit.pack(fill="x", pady=(10, 0))
        edit.columnconfigure(1, weight=1)
        ttk.Label(edit, text="Decision").grid(row=0, column=0, sticky="w")
        self.correction_value_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.correction_value_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        ttk.Label(edit, text="Note").grid(row=1, column=0, sticky="w", pady=(6, 0))
        self.correction_note_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.correction_note_var).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(6, 0))

        buttons = ttk.Frame(self.corrections_tab)
        buttons.pack(fill="x", pady=(10, 0))
        ttk.Button(buttons, text="Refresh", command=self._refresh_corrections_panel).pack(side="left")
        ttk.Button(buttons, text="Approve", command=lambda: self._set_selected_correction_status("approved")).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Needs Review", command=lambda: self._set_selected_correction_status("needs_review")).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Reject", command=lambda: self._set_selected_correction_status("rejected")).pack(side="left", padx=(8, 0))
        ttk.Button(buttons, text="Save Edit", command=self._save_selected_correction_edit).pack(side="left", padx=(8, 0))
        self.corrections_status_var = tk.StringVar(value="Load a study to review corrections.")
        ttk.Label(self.corrections_tab, textvariable=self.corrections_status_var, style="Status.TLabel").pack(anchor="w", pady=(8, 0))

    def _build_settings_tab(self) -> None:
        self.model_profile_var = tk.StringVar(value=self.profile_id_to_label[DEFAULT_PROFILE_ID])
        self.setting_vars["CSES_MODEL_PROFILE"] = self.model_profile_var
        self.setting_vars["CSES_USE_OPENROUTER"] = tk.StringVar(value="true")
        for key in [
            "CSES_USE_GESIS",
            "CSES_USE_OPENWEBUI",
            "CSES_USE_CODEX_OAUTH",
            "CSES_USE_OPENAI",
            "CSES_USE_ANTHROPIC",
            "CSES_USE_XAI",
            "CSES_USE_GEMINI",
        ]:
            self.setting_vars[key] = tk.StringVar(value="false")

        ttk.Label(self.settings_tab, text="OpenRouter", style="Section.TLabel").pack(anchor="w")
        ttk.Label(
            self.settings_tab,
            text="This build uses OpenRouter for all AI roles. Choose cost-efficient models for each task, then chat normally.",
        ).pack(anchor="w", pady=(3, 8))

        openrouter_row = ttk.Frame(self.settings_tab)
        openrouter_row.pack(fill="x", pady=(0, 14))
        ttk.Label(openrouter_row, text="API key").pack(side="left")
        openrouter_var = tk.StringVar(value=self.settings.get("OPENROUTER_API_KEY"))
        self.setting_vars["OPENROUTER_API_KEY"] = openrouter_var
        entry = ttk.Entry(openrouter_row, textvariable=openrouter_var, show="*", width=72)
        entry.pack(side="left", fill="x", expand=True, padx=(12, 0))
        self.api_key_entries["OPENROUTER_API_KEY"] = entry

        ttk.Label(self.settings_tab, text="Model roles", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(
            self.settings_tab,
            text="These roles share the same study context. The defaults are selected for cost and task fit.",
        ).pack(anchor="w", pady=(3, 8))
        role_row = ttk.Frame(self.settings_tab)
        role_row.pack(fill="x", pady=(0, 12))
        role_row.columnconfigure(1, weight=1)
        role_row.columnconfigure(3, weight=1)
        role_values = self._role_model_values()
        role_specs = [
            ("Agentic workflow", "CSES_AGENTIC_MODEL", "agentic_model_var", "agentic_model_combo", DEFAULT_AGENTIC_MODEL),
            ("Study information review", "CSES_STUDY_KB_MODEL", "study_kb_model_var", "study_kb_model_combo", DEFAULT_STUDY_KB_MODEL),
            ("Large text extraction", "CSES_LARGE_TEXT_MODEL", "large_text_model_var", "large_text_model_combo", DEFAULT_LARGE_TEXT_MODEL),
            ("Evidence verifier", "CSES_VERIFIER_MODEL", "verifier_model_var", "verifier_model_combo", DEFAULT_VERIFIER_MODEL),
            ("Matching fast pass", "CSES_MATCH_FAST_MODEL", "match_fast_model_var", "match_fast_model_combo", DEFAULT_MATCH_FAST_MODEL),
            ("Matching escalation", "CSES_MATCH_ESCALATION_MODEL", "match_escalation_model_var", "match_escalation_model_combo", DEFAULT_MATCH_ESCALATION_MODEL),
            ("Stata codegen", "CSES_STATA_CODE_MODEL", "stata_code_model_var", "stata_code_model_combo", DEFAULT_STATA_CODE_MODEL),
            ("Stata repair", "CSES_STATA_REPAIR_MODEL", "stata_repair_model_var", "stata_repair_model_combo", DEFAULT_STATA_REPAIR_MODEL),
            ("Documentation", "CSES_DOCUMENTATION_MODEL", "documentation_model_var", "documentation_model_combo", DEFAULT_DOCUMENTATION_MODEL),
            ("Final escalation", "CSES_FINAL_ESCALATION_MODEL", "final_escalation_model_var", "final_escalation_model_combo", DEFAULT_FINAL_ESCALATION_MODEL),
        ]
        for index, (label, key, var_attr, combo_attr, default) in enumerate(role_specs):
            grid_row = index // 2
            col = 0 if index % 2 == 0 else 2
            ttk.Label(role_row, text=label).grid(row=grid_row, column=col, sticky="w", pady=3)
            var = tk.StringVar(value=self.settings.get(key) or default)
            setattr(self, var_attr, var)
            self.setting_vars[key] = var
            combo = ttk.Combobox(role_row, textvariable=var, values=role_values, state="readonly")
            combo.grid(row=grid_row, column=col + 1, sticky="ew", padx=(8, 18 if col == 0 else 0), pady=3)
            setattr(self, combo_attr, combo)
            self.role_model_combos.append(combo)
            if key == "CSES_AGENTIC_MODEL":
                combo.bind("<<ComboboxSelected>>", lambda _event: self._sync_agentic_role_to_chat())

        ttk.Label(self.settings_tab, text="Available OpenRouter models", style="Section.TLabel").pack(anchor="w", pady=(8, 0))
        ttk.Label(
            self.settings_tab,
            text="Refresh pulls OpenRouter model metadata and per-token pricing when available.",
        ).pack(anchor="w", pady=(3, 8))

        self.tool_capable_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            self.settings_tab,
            text="Only show models with tool-use support",
            variable=self.tool_capable_only_var,
            command=self._refresh_models,
        ).pack(anchor="w", pady=(0, 6))

        model_buttons = ttk.Frame(self.settings_tab)
        model_buttons.pack(fill="x", pady=(0, 6))
        ttk.Button(model_buttons, text="Refresh Models", command=self._refresh_models).pack(side="left")
        self.model_refresh_var = tk.StringVar(value="Model costs are shown when available.")
        ttk.Label(model_buttons, textvariable=self.model_refresh_var, style="Status.TLabel").pack(side="left", padx=(12, 0))

        model_columns = ("use", "model", "provider", "source", "input", "output", "context", "tools")
        self.models_tree = ttk.Treeview(self.settings_tab, columns=model_columns, show="headings", height=5)
        headings = {
            "use": "Use",
            "model": "Model",
            "provider": "Provider",
            "source": "Source",
            "input": "Input / 1M",
            "output": "Output / 1M",
            "context": "Context",
            "tools": "Tools",
        }
        widths = {
            "use": 48,
            "model": 260,
            "provider": 90,
            "source": 140,
            "input": 85,
            "output": 85,
            "context": 90,
            "tools": 70,
        }
        for column in model_columns:
            self.models_tree.heading(column, text=headings[column])
            self.models_tree.column(column, width=widths[column], anchor="w")
        self.models_tree.pack(fill="both", expand=False, pady=(0, 14))
        self.models_tree.bind("<ButtonRelease-1>", self._on_model_tree_click)

        ttk.Separator(self.settings_tab).pack(fill="x", pady=16)
        ttk.Label(self.settings_tab, text="Stata", style="Section.TLabel").pack(anchor="w")
        stata_row = ttk.Frame(self.settings_tab)
        stata_row.pack(fill="x", pady=(8, 0))
        stata_var = tk.StringVar(value=self.settings.get("STATA_PATH"))
        self.setting_vars["STATA_PATH"] = stata_var
        ttk.Entry(stata_row, textvariable=stata_var).pack(side="left", fill="x", expand=True)
        ttk.Button(stata_row, text="Choose Stata EXE", command=self._choose_stata).pack(side="left", padx=(8, 0))

        button_row = ttk.Frame(self.settings_tab)
        button_row.pack(fill="x", pady=18)
        ttk.Button(button_row, text="Save Settings", command=self._save_settings).pack(side="left")
        ttk.Button(button_row, text="Check Stata Connection", command=self._check_stata_connection).pack(side="left", padx=8)

    def _build_about_tab(self) -> None:
        ttk.Label(self.about_tab, text="Stata connection", style="Section.TLabel").pack(anchor="w")
        text = (
            "The app connects to your local licensed Stata installation through its bundled Stata bridge. "
            "If Stata cannot be discovered automatically, choose the Stata executable in Settings.\n\n"
            f"Repository: {MCP_STATA_REPOSITORY}\n\n"
            "The chat assistant remains human-in-the-loop: it should explain what it did, document findings, "
            "and wait before moving to the next workflow step."
        )
        ttk.Label(self.about_tab, text=text, wraplength=850, justify="left").pack(anchor="w")

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose study folder", initialdir=self.folder_var.get() or str(Path.cwd()))
        if folder:
            self.folder_var.set(folder)

    def _refresh_openwebui_state(self) -> None:
        if not hasattr(self, "openwebui_base_entry"):
            return
        enabled = self.setting_vars.get("CSES_USE_OPENWEBUI", tk.StringVar(value="false")).get().lower() in {"1", "true", "yes"}
        self.openwebui_base_entry.configure(state="normal" if enabled else "disabled")
        if hasattr(self, "openwebui_key_entry"):
            self.openwebui_key_entry.configure(state="normal" if enabled else "disabled")

    def _refresh_api_provider_states(self) -> None:
        if hasattr(self, "gesis_key_entry"):
            gesis_enabled = self.setting_vars.get("CSES_USE_GESIS", tk.StringVar(value="true")).get().lower() in {"1", "true", "yes"}
            self.gesis_key_entry.configure(state="normal" if gesis_enabled else "disabled")
        for label, env_name in SUPPORTED_API_KEYS.items():
            entry = self.api_key_entries.get(env_name)
            if not entry:
                continue
            toggle_name = PROVIDER_TOGGLES[label]
            enabled = self.setting_vars.get(toggle_name, tk.StringVar(value="false")).get().lower() in {"1", "true", "yes"}
            entry.configure(state="normal" if enabled else "disabled")
        self._refresh_openwebui_state()

    def _refresh_codex_oauth_status(self, silent: bool = False) -> None:
        status = CodexOAuthAuthSource().check()
        self.codex_status_var.set(f"{status.status}: {status.message}")
        if "CSES_CODEX_OAUTH_PROFILE" in self.setting_vars and (status.account_email or status.account_id):
            self.setting_vars["CSES_CODEX_OAUTH_PROFILE"].set(status.account_email or status.account_id)
        if status.signed_in:
            try:
                CodexOAuthAuthSource().persist_metadata(PROJECT_ROOT)
            except Exception:
                pass
        if not silent and not status.signed_in:
            messagebox.showinfo("Codex sign-in", status.message)

    def _open_codex_sign_in(self) -> None:
        try:
            subprocess.Popen(["codex", "login"], cwd=str(PROJECT_ROOT), creationflags=subprocess.CREATE_NEW_CONSOLE)
            self.codex_status_var.set("Codex sign-in opened. Finish sign-in, then click Check Codex sign-in.")
        except Exception as exc:
            messagebox.showerror(
                "Could not open Codex sign-in",
                f"Codex sign-in could not be launched:\n{exc}\n\nOpen Codex manually, sign in with ChatGPT, then click Check Codex sign-in.",
            )

    def _choose_stata(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose Stata executable",
            filetypes=[("Executable files", "*.exe"), ("All files", "*.*")],
        )
        if path:
            self.setting_vars["STATA_PATH"].set(path)

    def _load_study(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        folder_check = FileOrganizer(folder).validate_study_folder()
        if not folder_check.ok:
            message = self._format_invalid_study_folder_message(folder, folder_check)
            self.conversation = None
            self.loaded_state = None
            self.status_var.set("Select one CSES study folder.")
            self._refresh_sidebar(None)
            self._append_chat("system", message)
            return
        state = self._load_state_from_folder(folder)
        if not state:
            self.conversation = None
            self.status_var.set("Initializing study folder...")
            self._refresh_sidebar(None)
            self._append_chat("system", self._format_uninitialized_folder_summary(folder))
            self._append_chat("system", "Initializing this study folder. The study will load automatically when initialization finishes.")
            self._run_cses(["init"], after="load_study")
            return

        self._open_study_state(state)

    def _open_study_state(self, state: WorkflowState) -> None:
        self.folder_var.set(str(Path(state.working_dir)))
        self.loaded_state = state
        self.conversation = ConversationSession(state)
        self.status_var.set(f"Study loaded: {state.country} {state.year}")
        self._refresh_sidebar(state)
        self._refresh_corrections_panel()
        self._refresh_context_panel()
        self._append_chat("system", self._format_loaded_summary(state))
        self._refresh_active_model_display()

    def _load_conversation(self) -> None:
        self._load_study()

    def _load_state_from_folder(self, folder: Path) -> WorkflowState | None:
        if not folder.exists():
            messagebox.showerror("Folder not found", f"The study folder does not exist:\n{folder}")
            return None

        state = WorkflowState.load(folder)
        if state:
            self._normalize_state_working_dir(state, folder)
            return state
        return None

    def _format_invalid_study_folder_message(self, folder: Path, folder_check) -> str:
        lines = [
            "This folder cannot be loaded as a CSES study.",
            "",
            folder_check.message,
        ]
        if folder_check.details:
            lines.extend(["", "What I found:"])
            lines.extend(f"- {detail}" for detail in folder_check.details)
        lines.extend([
            "",
            "Please choose one study folder directly. It should either already be initialized or contain one email/deposit folder with the deposited study files.",
            f"Selected folder: {folder}",
        ])
        return "\n".join(lines)

    def _normalize_state_working_dir(self, state: WorkflowState, folder: Path) -> None:
        state.rebase_paths(folder)

    def _format_uninitialized_folder_summary(self, folder: Path) -> str:
        if not folder.exists():
            return f"The selected folder does not exist:\n{folder}"

        organizer = FileOrganizer(folder)
        email_folder = organizer.find_email_folder()
        source_dir = email_folder or folder
        detected = organizer.detect_files(source_dir=source_dir, recursive=True)
        counts = [
            ("Survey data", len(detected.data_files)),
            ("Questionnaires", len(detected.questionnaire_files)),
            ("Codebooks", len(detected.codebook_files)),
            ("Design reports", len(detected.design_report_files)),
            ("Macro reports", len(detected.macro_report_files)),
            ("Election results", len(detected.election_results_files)),
            ("District data", len(detected.district_data_files)),
        ]

        lines = [
            "This folder contains deposited materials but has not been initialized as a CSES study yet.",
            "",
            f"Selected folder: {folder}",
            f"Deposit scan location: {source_dir}",
        ]
        if detected.country or detected.year:
            study_parts = [part for part in [detected.country, detected.year] if part]
            lines.append(f"Detected study hint: {' '.join(study_parts)}")

        lines.extend(["", "Detected materials:"])
        for label, count in counts:
            lines.append(f"- {label}: {count}")

        lines.extend(
            [
                "",
                "The study will be initialized automatically when you click Load Study.",
            ]
        )
        return "\n".join(lines)

    def _send_chat(self) -> None:
        message = self.message_entry.get("1.0", "end").strip()
        if not message:
            return
        self._send_chat_message(message, clear_input=True)

    def _proceed_chat(self) -> None:
        self._send_chat_message("Proceed", clear_input=False)

    def _send_chat_message(self, message: str, clear_input: bool) -> None:
        if not self._selected_chat_model():
            messagebox.showwarning("Select a model", "Please select a model from the Chat model dropdown before sending.")
            self.status_var.set("Please select a model before starting chat.")
            return
        if not self.conversation:
            self._load_study()
            if not self.conversation:
                return

        if clear_input:
            self.message_entry.delete("1.0", "end")
        self._append_chat("user", message)
        chat_model = self._selected_chat_model() or MODEL_PLACEHOLDER
        self.status_var.set(f"Assistant is thinking using {chat_model}...")
        self.message_entry.configure(state="disabled")
        self.send_button.configure(state="disabled")
        self.proceed_button.configure(state="disabled")
        threading.Thread(target=self._chat_worker, args=(message,), daemon=True).start()

    def _send_chat_from_enter(self, event) -> str:
        self._send_chat()
        return "break"

    def _chat_worker(self, message: str) -> None:
        try:
            self._apply_settings_to_environment()
            assert self.conversation is not None
            response = self.conversation.send(
                message,
                on_tool_output=lambda text: self.queue.put(("tool", text)),
            )
            self.queue.put(("assistant", response))
            self.queue.put(("refresh_state", ""))
        except Exception as exc:
            self.queue.put(("system", f"Chat error: {exc}"))
        finally:
            self.queue.put(("ready", "Ready"))

    def _apply_settings_to_environment(self) -> None:
        for key, value in load_settings(PROJECT_ROOT).values.items():
            if value:
                os.environ[key] = value
        for key, value in self._settings_from_form().items():
            if value:
                os.environ[key] = value
        for label, env_name in SUPPORTED_API_KEYS.items():
            toggle_name = PROVIDER_TOGGLES[label]
            if os.environ.get(toggle_name, "").lower() not in {"1", "true", "yes"}:
                os.environ.pop(env_name, None)
        if os.environ.get("CSES_USE_OPENROUTER", "").lower() in {"1", "true", "yes"}:
            os.environ.pop("OPENAI_API_BASE", None)
        elif os.environ.get("CSES_USE_GESIS", "").lower() in {"1", "true", "yes"}:
            os.environ["OPENAI_API_BASE"] = GESIS_OPENWEBUI_BASE
            gesis_key = os.environ.get("GESIS_API_KEY", "").strip()
            if gesis_key:
                os.environ["OPENAI_API_KEY"] = gesis_key
                os.environ["OPENWEBUI_API_KEY"] = gesis_key
        elif os.environ.get("CSES_USE_OPENWEBUI", "").lower() in {"1", "true", "yes"}:
            openwebui_key = os.environ.get("OPENWEBUI_API_KEY", "").strip()
            if openwebui_key:
                os.environ["OPENAI_API_KEY"] = openwebui_key
        else:
            os.environ.pop("OPENAI_API_BASE", None)
        chat_model = self._selected_chat_model()
        if chat_model:
            os.environ["CSES_CHAT_MODEL"] = chat_model

    def _update_model_profile_description(self) -> None:
        profile_id = self.profile_label_to_id.get(self.model_profile_var.get(), self.model_profile_var.get())
        profile = MODEL_PROFILES.get(profile_id, MODEL_PROFILES[DEFAULT_PROFILE_ID])
        self.model_profile_label.configure(
            text=f"{profile.label}: {profile.description} Requires {profile.required_key}."
        )
        self._refresh_active_model_display()

    def _current_profile(self):
        return MODEL_PROFILES[DEFAULT_PROFILE_ID]

    def _default_openrouter_models(self) -> list[str]:
        return [
            DEFAULT_AGENTIC_MODEL,
            DEFAULT_STUDY_KB_MODEL,
            DEFAULT_LARGE_TEXT_MODEL,
            DEFAULT_VERIFIER_MODEL,
            DEFAULT_MATCH_FAST_MODEL,
            DEFAULT_MATCH_ESCALATION_MODEL,
            DEFAULT_STATA_CODE_MODEL,
            DEFAULT_STATA_REPAIR_MODEL,
            DEFAULT_DOCUMENTATION_MODEL,
            DEFAULT_FINAL_ESCALATION_MODEL,
        ]

    def _role_model_values(self) -> list[str]:
        values = set()
        for model in (self.settings.get("CSES_CHAT_MODELS", "") or "").split(","):
            if model.strip().startswith("openrouter/"):
                values.add(model.strip())
        values.update(self._default_openrouter_models())
        return sorted(values)

    def _sync_agentic_role_to_chat(self) -> None:
        model = self.agentic_model_var.get().strip() if hasattr(self, "agentic_model_var") else ""
        if model and hasattr(self, "chat_model_var"):
            self.chat_model_var.set(model)
        self._refresh_active_model_display()

    def _selected_chat_model(self) -> str:
        if not hasattr(self, "chat_model_var"):
            return ""
        model = self.chat_model_var.get().strip()
        if not model or model == MODEL_PLACEHOLDER:
            return ""
        return model

    def _format_ai_panel(self) -> str:
        profile = self._current_profile() if hasattr(self, "model_profile_var") else MODEL_PROFILES[DEFAULT_PROFILE_ID]
        agentic_model = self.agentic_model_var.get().strip() if hasattr(self, "agentic_model_var") else self._selected_chat_model()
        large_text_model = self.large_text_model_var.get().strip() if hasattr(self, "large_text_model_var") else DEFAULT_LARGE_TEXT_MODEL
        study_kb_model = self.study_kb_model_var.get().strip() if hasattr(self, "study_kb_model_var") else DEFAULT_STUDY_KB_MODEL
        verifier_model = self.verifier_model_var.get().strip() if hasattr(self, "verifier_model_var") else DEFAULT_VERIFIER_MODEL
        code_model = self.stata_code_model_var.get().strip() if hasattr(self, "stata_code_model_var") else DEFAULT_STATA_CODE_MODEL
        repair_model = self.stata_repair_model_var.get().strip() if hasattr(self, "stata_repair_model_var") else DEFAULT_STATA_REPAIR_MODEL
        chat_model = self._selected_chat_model() or agentic_model or MODEL_PLACEHOLDER
        return (
            f"Profile: {profile.label}\n"
            f"Chat model: {chat_model}\n"
            f"Agentic: {agentic_model or MODEL_PLACEHOLDER}\n"
            f"Study information: {study_kb_model or MODEL_PLACEHOLDER}\n"
            f"Large text: {large_text_model or MODEL_PLACEHOLDER}\n"
            f"Verifier: {verifier_model or MODEL_PLACEHOLDER}\n"
            f"Stata code: {code_model or MODEL_PLACEHOLDER}\n"
            f"Repair: {repair_model or MODEL_PLACEHOLDER}\n"
            f"Provider: OpenRouter"
        )

    def _refresh_active_model_display(self) -> None:
        if not hasattr(self, "active_model_var"):
            return
        profile = self._current_profile() if hasattr(self, "model_profile_var") else MODEL_PROFILES[DEFAULT_PROFILE_ID]
        agentic_model = self.agentic_model_var.get().strip() if hasattr(self, "agentic_model_var") else self._selected_chat_model()
        chat_model = self._selected_chat_model() or agentic_model or MODEL_PLACEHOLDER
        self.active_model_var.set(f"AI: {profile.label} | Agentic: {agentic_model or MODEL_PLACEHOLDER} | Chat: {chat_model}")

    def _settings_from_form(self) -> dict[str, str]:
        values = {}
        for key, var in self.setting_vars.items():
            value = var.get()
            if key == "CSES_MODEL_PROFILE":
                value = self.profile_label_to_id.get(value, value)
            values[key] = value.strip()
        values["CSES_MODEL_PROFILE"] = DEFAULT_PROFILE_ID
        values["CSES_USE_OPENROUTER"] = "true"
        values["CSES_USE_GESIS"] = "false"
        values["CSES_USE_OPENWEBUI"] = "false"
        values["CSES_USE_CODEX_OAUTH"] = "false"
        values["CSES_USE_OPENAI"] = "false"
        values["CSES_USE_ANTHROPIC"] = "false"
        values["CSES_USE_XAI"] = "false"
        values["CSES_USE_GEMINI"] = "false"
        values["OPENAI_API_BASE"] = ""
        values["OPENWEBUI_API_KEY"] = ""
        values["GESIS_API_KEY"] = ""
        if hasattr(self, "chat_model_var"):
            values["CSES_CHAT_MODEL"] = self._selected_chat_model()
        for key in [
            "CSES_AGENTIC_MODEL",
            "CSES_LARGE_TEXT_MODEL",
            "CSES_STUDY_KB_MODEL",
            "CSES_VERIFIER_MODEL",
            "CSES_MATCH_FAST_MODEL",
            "CSES_MATCH_ESCALATION_MODEL",
            "CSES_STATA_CODE_MODEL",
            "CSES_STATA_REPAIR_MODEL",
            "CSES_DOCUMENTATION_MODEL",
            "CSES_FINAL_ESCALATION_MODEL",
        ]:
            if values.get(key) and not values[key].startswith("openrouter/"):
                values[key] = self._default_for_role_key(key)
        if hasattr(self, "chat_model_combo"):
            combo_values = self.chat_model_combo.cget("values")
            if isinstance(combo_values, str):
                model_values = [value for value in combo_values.split() if value]
            else:
                model_values = list(combo_values)
            model_values = [value for value in model_values if str(value).startswith("openrouter/")]
            values["CSES_CHAT_MODELS"] = ",".join(model_values)
        return values

    def _default_for_role_key(self, key: str) -> str:
        return {
            "CSES_AGENTIC_MODEL": DEFAULT_AGENTIC_MODEL,
            "CSES_LARGE_TEXT_MODEL": DEFAULT_LARGE_TEXT_MODEL,
            "CSES_STUDY_KB_MODEL": DEFAULT_STUDY_KB_MODEL,
            "CSES_VERIFIER_MODEL": DEFAULT_VERIFIER_MODEL,
            "CSES_MATCH_FAST_MODEL": DEFAULT_MATCH_FAST_MODEL,
            "CSES_MATCH_ESCALATION_MODEL": DEFAULT_MATCH_ESCALATION_MODEL,
            "CSES_STATA_CODE_MODEL": DEFAULT_STATA_CODE_MODEL,
            "CSES_STATA_REPAIR_MODEL": DEFAULT_STATA_REPAIR_MODEL,
            "CSES_DOCUMENTATION_MODEL": DEFAULT_DOCUMENTATION_MODEL,
            "CSES_FINAL_ESCALATION_MODEL": DEFAULT_FINAL_ESCALATION_MODEL,
        }.get(key, DEFAULT_AGENTIC_MODEL)

    def _refresh_models(self) -> None:
        settings = self._settings_from_form()
        self.model_refresh_var.set("Refreshing models...")
        self.models_tree.delete(*self.models_tree.get_children())
        self.model_row_data = {}

        def worker():
            rows, warnings = build_model_catalog(
                settings,
                probe_openai_compatible_tools=self.tool_capable_only_var.get(),
            )
            if self.tool_capable_only_var.get():
                rows = [row for row in rows if row.tool_capable]
            self.queue.put(("models", json_payload_for_models(rows, warnings)))

        threading.Thread(target=worker, daemon=True).start()

    def _save_settings(self) -> None:
        updated = UserSettings()
        for key, value in self._settings_from_form().items():
            updated.set(key, value)
        env_path = save_settings(updated, PROJECT_ROOT)
        self.settings = updated
        self._apply_settings_to_environment()
        self._refresh_active_model_display()
        messagebox.showinfo("Settings saved", f"Settings were saved to:\n{env_path}")

    def _create_context_files(self) -> None:
        state = self.loaded_state or self._load_state_from_folder(Path(self.folder_var.get()).expanduser())
        if not state:
            messagebox.showwarning("No study loaded", "Load an initialized study before creating project context files.")
            return
        created = create_starter_project_context(Path(state.working_dir))
        self.loaded_state = state
        self._refresh_context_panel()
        if created:
            files = "\n".join(str(path) for path in created)
            messagebox.showinfo("Project context created", f"Created:\n{files}")
        else:
            messagebox.showinfo("Project context exists", "The starter project context files already exist.")

    def _refresh_context_panel(self) -> None:
        if not hasattr(self, "context_text"):
            return
        state = self.loaded_state
        if not state:
            self._set_panel_text(
                self.context_text,
                "No study loaded.\n\nExpected project files after loading a study:\n- agent.md\n- workflow.md\n- .agents/skills/<skill-name>/SKILL.md",
            )
            return
        text = (
            f"Study folder: {state.working_dir}\n\n"
            f"{describe_project_context(Path(state.working_dir))}\n\n"
            "Files the assistant reads automatically:\n"
            "- agent.md or AGENTS.md: role, boundaries, tone, review expectations\n"
            "- workflow.md: project-specific workflow notes supplementing the CSES workflow\n"
            "- .agents/skills/*/SKILL.md: reusable project-specific instructions\n"
        )
        self._set_panel_text(self.context_text, text)

    def _check_stata_connection(self) -> None:
        stata_path = self.setting_vars.get("STATA_PATH", tk.StringVar()).get()
        with tempfile.TemporaryDirectory(prefix="cses_stata_check_") as temp_dir:
            do_path = Path(temp_dir) / "stata_connection_check.do"
            do_path.write_text(
                "\n".join([
                    "clear",
                    "set obs 1",
                    "gen cses_connection_check = 1",
                    'save "cses_connection_check.dta", replace',
                ]),
                encoding="utf-8",
            )
            result = MCPStataRunner(stata_path=stata_path, timeout_seconds=45).run_do_file(do_path)
        if result.success:
            messagebox.showinfo("Stata connection", "Stata connection works.")
        else:
            messagebox.showerror(
                "Stata connection",
                result.error or "Stata could not be reached. Check the selected executable path.",
            )

    def _build_refresh_study_kb(self) -> None:
        state = self.loaded_state or self._load_state_from_folder(Path(self.folder_var.get()).expanduser())
        if not state:
            messagebox.showwarning("No study loaded", "Load a study before reviewing study materials.")
            return
        self.loaded_state = state
        self.status_var.set("Reviewing study materials...")
        self._append_output("\n> Review Study Materials\n")

        def worker():
            try:
                self._apply_settings_to_environment()
                builder = StudyKnowledgeBaseBuilder(
                    Path(state.working_dir),
                    progress_callback=lambda message: self.queue.put(("tool", message)),
                )
                payload = builder.build(state, force=True)
                EvidencePacketBuilder(Path(state.working_dir)).build(state, force=True)
                state.phase_status = phase_status_payload(state)
                state.current_phase = current_phase_id(state)
                state.save()
                self.queue.put(("tool", "Study materials reviewed."))
                self.queue.put(("refresh_state", ""))
            except Exception as exc:
                self.queue.put(("system", f"Study materials review failed: {exc}"))
            finally:
                self.queue.put(("ready", "Ready"))

        threading.Thread(target=worker, daemon=True).start()

    def _build_refresh_evidence_packet(self) -> None:
        state = self.loaded_state or self._load_state_from_folder(Path(self.folder_var.get()).expanduser())
        if not state:
            messagebox.showwarning("No study loaded", "Load a study before reviewing study materials.")
            return
        self.loaded_state = state
        self.status_var.set("Reviewing study materials...")
        self._append_chat("tool", "Reviewing study materials...")

        def worker():
            try:
                self._apply_settings_to_environment()
                from src.preprocessing.evidence_extractor import ParallelEvidenceExtractionService

                StudyKnowledgeBaseBuilder(
                    Path(state.working_dir),
                    progress_callback=lambda message: self.queue.put(("tool", message)),
                ).build(state, force=True)
                state.evidence_index = ParallelEvidenceExtractionService(
                    Path(state.working_dir),
                    progress_callback=lambda message: self.queue.put(("tool", message)),
                ).build(state, force=True)
                EvidencePacketBuilder(Path(state.working_dir)).build(state, force=True)
                state.phase_status = phase_status_payload(state)
                state.current_phase = current_phase_id(state)
                state.save()
                self.queue.put(("tool", "Study materials reviewed."))
                self.queue.put(("refresh_state", ""))
            except Exception as exc:
                self.queue.put(("system", f"Study materials review failed: {exc}"))
            finally:
                self.queue.put(("ready", "Ready"))

        threading.Thread(target=worker, daemon=True).start()

    def _run_step(self) -> None:
        step = self.step_var.get().strip()
        if not step:
            messagebox.showwarning("Missing step", "Enter a workflow step number, for example 1 or 7a.")
            return
        self._run_cses(["step", step])

    def _initialize_selected_folder(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists():
            messagebox.showerror("Folder not found", f"The study folder does not exist:\n{folder}")
            return
        if self._load_state_from_folder(folder):
            self._load_study()
            return
        self._append_chat("system", "Initializing this study folder. Chat will load automatically when initialization finishes.")
        self._run_cses(["init"], after="load_study")

    def _run_cses(self, args: list[str], after: str | None = None) -> None:
        if self.current_process and self.current_process.poll() is None:
            messagebox.showwarning("Workflow already running", "Please wait for the current operation to finish, or stop it first.")
            return

        folder = Path(self.folder_var.get()).expanduser()
        if not folder.exists():
            messagebox.showerror("Folder not found", f"The study folder does not exist:\n{folder}")
            return

        self._append_output(f"\n> cses {' '.join(args)}\n")
        self.status_var.set("Running command...")
        threading.Thread(target=self._run_process, args=(folder, args, after), daemon=True).start()

    def _run_process(self, folder: Path, args: list[str], after: str | None = None) -> None:
        env = os.environ.copy()
        for key, value in load_settings(PROJECT_ROOT).values.items():
            if value:
                env[key] = value
        env["PYTHONIOENCODING"] = "utf-8"

        command = [sys.executable, str(PROJECT_ROOT / "cses_cli.py"), *args]
        return_code = 1
        try:
            self.current_process = subprocess.Popen(
                command,
                cwd=str(folder),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
            )
            assert self.current_process.stdout is not None
            for line in self.current_process.stdout:
                self.queue.put(("output", line))
            return_code = self.current_process.wait()
            if return_code == 0:
                self.queue.put(("output", "\nOperation finished.\n"))
                if after in {"load_chat", "load_study"}:
                    self.queue.put(("load_after_init", ""))
            else:
                self.queue.put(("output", f"\nOperation stopped with exit code {return_code}.\n"))
        except Exception as exc:
            self.queue.put(("output", f"\nCould not run workflow: {exc}\n"))
        finally:
            self.current_process = None
            self.queue.put(("ready", "Ready"))

    def _stop_process(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self.current_process.terminate()
            self.status_var.set("Stopping...")

    def _drain_queue(self) -> None:
        while True:
            try:
                kind, text = self.queue.get_nowait()
            except queue.Empty:
                break

            if kind == "output":
                self._append_output(text)
            elif kind == "ready":
                self.status_var.set(sanitize_processor_text(text))
                self.message_entry.configure(state="normal")
                if hasattr(self, "send_button"):
                    self.send_button.configure(state="normal")
                if hasattr(self, "proceed_button"):
                    self.proceed_button.configure(state="normal")
                self.message_entry.focus_set()
            elif kind == "refresh_state":
                self._reload_loaded_state()
            elif kind == "models":
                self._populate_models(text)
            elif kind == "load_after_init":
                self._load_study()
            else:
                self._append_chat(kind, text)
        self.after(100, self._drain_queue)

    def _populate_models(self, payload: str) -> None:
        data = parse_models_payload(payload)
        self.models_tree.delete(*self.models_tree.get_children())
        if hasattr(self, "chat_model_combo"):
            combo_values = self.chat_model_combo.cget("values")
            saved_models = set(combo_values.split() if isinstance(combo_values, str) else combo_values)
        else:
            saved_models = set()
        new_selected_models = set()
        for row in data["rows"]:
            checked = row["model"] in saved_models or row["model"] in self.selected_model_rows
            tag = "unknown_cost" if not row["cost_known"] else ""
            item_id = self.models_tree.insert(
                "",
                "end",
                values=(
                    "[x]" if checked else "[ ]",
                    row["model"],
                    row["provider"],
                    row["source"],
                    row["input_per_million"],
                    row["output_per_million"],
                    row["context_window"],
                    row["tools"],
                ),
                tags=(tag,),
            )
            self.model_row_data[item_id] = row
            if checked:
                new_selected_models.add(row["model"])
        self.selected_model_rows = new_selected_models
        self.models_tree.tag_configure("unknown_cost", foreground="#6b4f00")
        status = f"{len(data['rows'])} models found."
        if data["warnings"]:
            status += " " + " ".join(data["warnings"])
        self.model_refresh_var.set(status)
        self._sync_checked_models_to_chat_dropdown()

    def _on_model_tree_click(self, event) -> None:
        region = self.models_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        column = self.models_tree.identify_column(event.x)
        if column != "#1":
            return
        item_id = self.models_tree.identify_row(event.y)
        if not item_id:
            return
        values = list(self.models_tree.item(item_id, "values"))
        row = self.model_row_data.get(item_id)
        if not row:
            return
        model = row["model"]
        if values[0] == "[x]":
            values[0] = "[ ]"
            self.selected_model_rows.discard(model)
        else:
            values[0] = "[x]"
            self.selected_model_rows.add(model)
        self.models_tree.item(item_id, values=values)
        self._sync_checked_models_to_chat_dropdown()

    def _sync_checked_models_to_chat_dropdown(self) -> None:
        models = sorted(
            {model for model in self.selected_model_rows if model.startswith("openrouter/")}
            | set(self._default_openrouter_models())
        )
        if not models:
            self.chat_model_combo.configure(values=[])
            self.chat_model_var.set(MODEL_PLACEHOLDER)
            self.model_refresh_var.set("No models checked. Select models in the table to make them available in Chat.")
            self._refresh_active_model_display()
            return
        self.chat_model_combo.configure(values=models)
        for combo in getattr(self, "role_model_combos", []):
            combo.configure(values=models)
        if hasattr(self, "agentic_model_var") and not self.agentic_model_var.get().strip():
            self.agentic_model_var.set(DEFAULT_AGENTIC_MODEL)
        if hasattr(self, "large_text_model_var") and not self.large_text_model_var.get().strip():
            self.large_text_model_var.set(DEFAULT_LARGE_TEXT_MODEL)
        if self._selected_chat_model() not in models:
            self.chat_model_var.set(self.agentic_model_var.get() if hasattr(self, "agentic_model_var") else DEFAULT_AGENTIC_MODEL)
        self._refresh_active_model_display()
        self.model_refresh_var.set(f"{len(models)} selected model(s) are available in the main chat dropdown.")

    def _append_chat(self, role: str, text: str) -> None:
        text = sanitize_processor_text(text)
        if not text:
            return
        labels = {
            "user": "You",
            "assistant": f"Assistant ({self._selected_chat_model() or MODEL_PLACEHOLDER})" if hasattr(self, "chat_model_var") else "Assistant",
            "tool": "Tool",
            "system": "System",
        }
        self.chat_text.configure(state="normal")
        label_tag = f"{role}_label" if role in labels else "system_label"
        self.chat_text.insert("end", f"{labels.get(role, role.title())}\n", label_tag)
        self.chat_text.insert("end", f"{text}\n\n", role)
        self.chat_text.see("end")
        self.chat_text.configure(state="disabled")

    def _set_panel_text(self, widget: tk.Text, text: str) -> None:
        text = sanitize_processor_text(text)
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _reload_loaded_state(self) -> None:
        if not self.loaded_state:
            return
        state = WorkflowState.load(Path(self.loaded_state.working_dir))
        if state:
            self.loaded_state = state
            if self.conversation:
                self.conversation.state = state
            self._refresh_sidebar(state)
            self._refresh_corrections_panel()

    def _refresh_sidebar(self, state: WorkflowState | None) -> None:
        if not state:
            self._set_panel_text(self.study_panel, "No study loaded.\nChoose a folder and click Load Study.")
            self._set_workflow_panel(None)
            self._set_panel_text(self.files_panel, "Pending questions and registered files will appear here.")
            return

        self._set_panel_text(self.study_panel, self._format_study_panel(state))
        self._set_workflow_panel(state)
        self._set_panel_text(self.files_panel, self._format_files_panel(state))

    def _format_study_panel(self, state: WorkflowState) -> str:
        progress = state.get_progress_summary()
        next_step = state.get_next_step()
        current_phase = current_phase_id(state)
        lines = [
            f"{state.country} {state.year}",
            f"Code: {state.country_code}",
            f"Progress: {progress['completed']}/{progress['total_steps']} steps ({progress['percent_complete']:.0f}%)",
            f"Phase: {current_phase.replace('_', ' ').title()}",
        ]
        review_status = format_study_review_status(EvidencePacketBuilder(Path(state.working_dir)).status().status)
        if review_status and "changed" in review_status.lower():
            lines.append(review_status)
        if next_step is not None:
            lines.append(f"Next: Step {next_step} - {WORKFLOW_STEPS[next_step]['name']}")
        else:
            lines.append("Next: All steps complete")
        return "\n".join(lines)

    def _set_workflow_panel(self, state: WorkflowState | None) -> None:
        if not hasattr(self, "workflow_panel"):
            return
        if not state:
            self._set_panel_text(self.workflow_panel, "Workflow status will appear after loading a study.")
            return
        markers = {
            StepStatus.NOT_STARTED.value: "[ ]",
            StepStatus.IN_PROGRESS.value: "[>]",
            StepStatus.BLOCKED.value: "[!]",
            StepStatus.COMPLETED.value: "[x]",
            StepStatus.SKIPPED.value: "[-]",
        }
        current_phase = current_phase_id(state)
        next_step = state.get_next_step()
        lines = []
        for phase in WORKFLOW_PHASES:
            status = phase_status(state, phase)
            phase_marker = {
                "completed": "[x]",
                "in_progress": "[>]",
                "blocked": "[!]",
                "not_started": "[ ]",
            }.get(status, "[ ]")
            phase_title = phase.title.upper() if phase.id == current_phase else phase.title
            lines.append(f"{phase_marker} {phase_title}")
            for step_num in phase.steps:
                step = state.get_step(step_num)
                step_name = WORKFLOW_STEPS[step_num]["name"]
                marker = markers.get(step.status, "[?]")
                next_marker = " <- next" if step_num == next_step else ""
                lines.append(f"  {marker} Step {step_num}: {step_name}{next_marker}")
            lines.append("")
        self._set_panel_text(self.workflow_panel, "\n".join(lines).rstrip())

    def _format_files_panel(self, state: WorkflowState) -> str:
        lines = []
        pending = state.get_pending_questions()
        candidates = getattr(state, "candidate_collaborator_questions", [])
        lines.append(f"Pending questions: {len(pending)}")
        lines.append(f"Potential questions: {len(candidates)}")
        standards = state.standards_checks or {}
        review_count = sum(1 for item in standards.values() if item.get("status") == "needs_review")
        lines.append(f"CSES checks needing review: {review_count}")
        packet_status = EvidencePacketBuilder(Path(state.working_dir)).status()
        review_status = format_study_review_status(packet_status.status)
        if review_status and "changed" in review_status.lower():
            lines.append(review_status)
        if getattr(state, "study_kb_missing_fields", []):
            lines.append(f"Study fields needing review: {len(state.study_kb_missing_fields)}")
        if getattr(state, "study_kb_contradictions", []):
            lines.append(f"Study facts needing review: {len(state.study_kb_contradictions)}")
        corrections = getattr(state, "correction_summary", {}) or {}
        if corrections.get("pending_count"):
            lines.append(f"Corrections needing approval: {corrections.get('pending_count')}")
        if corrections.get("invalidated_count"):
            lines.append(f"Work to revisit after corrections: {corrections.get('invalidated_count')}")
        manifest = getattr(state, "primary_input_selection", {}) or {}
        if manifest.get("primary_data_file"):
            lines.append(f"Primary data: {Path(manifest.get('primary_data_file')).name}")
        matching = getattr(state, "matching_coverage", {}) or {}
        if matching:
            lines.append(
                "Matching: "
                f"{matching.get('proposed_match_count', 0)}/{matching.get('target_count', 0)} proposed, "
                f"{matching.get('blocked_count', 0)} review"
            )
        recoding = getattr(state, "recoding_coverage", {}) or {}
        if recoding:
            lines.append(
                "Recoding: "
                f"{recoding.get('approved_count', 0)}/{recoding.get('target_count', 0)} approved"
            )
        stata = getattr(state, "stata_execution_status", {}) or {}
        if stata:
            lines.append(f"Stata: {'clean' if stata.get('success') else 'needs repair'}")
        if state.final_readiness:
            lines.append(f"Final readiness: {state.final_readiness.get('status', 'unknown')}")
        lines.append("")
        for question in pending[:3]:
            text = question.get("question", "").strip().replace("\n", " ")
            if len(text) > 48:
                text = text[:45] + "..."
            lines.append(f"- {text}")
        if len(pending) > 3:
            lines.append(f"- plus {len(pending) - 3} more")

        lines.append("")
        lines.append("Files")
        if state.log_file:
            lines.append(f"Log: {Path(state.log_file).name}")
        if state.data_file:
            lines.append(f"Data: {Path(state.data_file).name}")
        for path in (state.questionnaire_files or [])[:2]:
            lines.append(f"Questionnaire: {Path(path).name}")
        if state.codebook_file:
            lines.append(f"Codebook: {Path(state.codebook_file).name}")
        if state.design_report_file:
            lines.append(f"Design report: {Path(state.design_report_file).name}")
        if state.variable_tracking_file:
            lines.append(f"Tracking: {Path(state.variable_tracking_file).name}")
        return "\n".join(lines)

    def _refresh_corrections_panel(self) -> None:
        if not hasattr(self, "corrections_tree"):
            return
        self.corrections_tree.delete(*self.corrections_tree.get_children())
        state = self.loaded_state
        if not state:
            self.corrections_status_var.set("Load a study to review corrections.")
            return
        ledger = ProcessorDecisionLedger(state.working_dir)
        decisions = ledger.load()
        for decision in decisions:
            affected = ", ".join(decision.affected_outputs[:4])
            self.corrections_tree.insert(
                "",
                "end",
                iid=decision.decision_id,
                values=(
                    decision.decision_id,
                    decision.area.replace("_", " ").title(),
                    decision.target,
                    decision.value,
                    decision.status.replace("_", " ").title(),
                    affected,
                ),
            )
        pending = len([item for item in decisions if item.status in {"pending_confirmation", "needs_review"}])
        approved = len([item for item in decisions if item.status == "approved"])
        self.corrections_status_var.set(f"{approved} approved correction(s), {pending} needing review.")

    def _selected_correction_id(self) -> str:
        if not hasattr(self, "corrections_tree"):
            return ""
        selection = self.corrections_tree.selection()
        return selection[0] if selection else ""

    def _set_selected_correction_status(self, status: str) -> None:
        state = self.loaded_state
        decision_id = self._selected_correction_id()
        if not state or not decision_id:
            messagebox.showinfo("Processor corrections", "Select a correction first.")
            return
        ledger = ProcessorDecisionLedger(state.working_dir)
        try:
            decision = ledger.update_status(decision_id, status, self.correction_note_var.get().strip())
        except Exception as exc:
            messagebox.showerror("Processor corrections", f"Could not update correction:\n{exc}")
            return
        if decision and status == "approved":
            impact = DependencyInvalidator().impact_for(decision)
            decision.affected_variables = list(dict.fromkeys([*decision.affected_variables, *impact.affected_variables]))
            decision.affected_outputs = list(dict.fromkeys([*decision.affected_outputs, *impact.affected_outputs]))
            decisions = [item if item.decision_id != decision.decision_id else decision for item in ledger.load()]
            ledger.save(decisions)
            state.update_structured_decision_status(decision.to_dict(), impact.to_dict())
        elif decision:
            state.update_structured_decision_status(decision.to_dict())
        state.save()
        self._refresh_sidebar(state)
        self._refresh_corrections_panel()

    def _save_selected_correction_edit(self) -> None:
        state = self.loaded_state
        decision_id = self._selected_correction_id()
        if not state or not decision_id:
            messagebox.showinfo("Processor corrections", "Select a correction first.")
            return
        ledger = ProcessorDecisionLedger(state.working_dir)
        decisions = ledger.load()
        updated = None
        for decision in decisions:
            if decision.decision_id == decision_id:
                value = self.correction_value_var.get().strip()
                note = self.correction_note_var.get().strip()
                if value:
                    decision.value = value
                if note:
                    decision.reason = note
                decision.status = "needs_review" if decision.status == "rejected" else decision.status
                updated = decision
                break
        if not updated:
            messagebox.showerror("Processor corrections", "Selected correction could not be found.")
            return
        ledger.save(decisions)
        if updated.status == "approved":
            impact = DependencyInvalidator().impact_for(updated)
            updated.affected_variables = list(dict.fromkeys([*updated.affected_variables, *impact.affected_variables]))
            updated.affected_outputs = list(dict.fromkeys([*updated.affected_outputs, *impact.affected_outputs]))
            ledger.save([item if item.decision_id != updated.decision_id else updated for item in decisions])
            state.update_structured_decision_status(updated.to_dict(), impact.to_dict())
        else:
            state.update_structured_decision_status(updated.to_dict())
        state.save()
        self._refresh_sidebar(state)
        self._refresh_corrections_panel()

    def _format_loaded_summary(self, state: WorkflowState) -> str:
        progress = state.get_progress_summary()
        next_step = state.get_next_step()
        file_lines = []
        if state.data_file:
            file_lines.append(f"- Data: {Path(state.data_file).name}")
        for path in state.questionnaire_files or []:
            file_lines.append(f"- Questionnaire: {Path(path).name}")
        if state.codebook_file:
            file_lines.append(f"- Codebook: {Path(state.codebook_file).name}")
        if state.design_report_file:
            file_lines.append(f"- Design report: {Path(state.design_report_file).name}")
        if state.log_file:
            file_lines.append(f"- Processing log: {Path(state.log_file).name}")
        if state.collaborator_questions_file:
            file_lines.append(f"- Collaborator questions: {Path(state.collaborator_questions_file).name}")

        completed = []
        in_progress = []
        for step_num in sorted(WORKFLOW_STEPS):
            step = state.get_step(step_num)
            if step.status == StepStatus.COMPLETED.value:
                completed.append(step_num)
            elif step.status == StepStatus.IN_PROGRESS.value:
                in_progress.append(step_num)

        summary = [
            f"Loaded study: {state.country} {state.year}",
            "",
            f"Working folder: {state.working_dir}",
            f"Progress: {progress['completed']}/{progress['total_steps']} steps complete ({progress['percent_complete']:.0f}%).",
            f"Current phase: {current_phase_id(state).replace('_', ' ').title()}.",
        ]
        review_status = format_study_review_status(EvidencePacketBuilder(Path(state.working_dir)).status().status)
        if review_status and "changed" in review_status.lower():
            summary.append(review_status)
        if completed:
            latest = ", ".join(str(step) for step in completed[-5:])
            summary.append(f"Recently completed steps: {latest}.")
        if in_progress:
            current = ", ".join(str(step) for step in in_progress)
            summary.append(f"In progress: {current}.")
        if next_step is not None:
            step_info = WORKFLOW_STEPS[next_step]
            summary.append(f"Next step: Step {next_step} - {step_info['name']}.")
            summary.append(f"Purpose: {step_info['description']}")
        else:
            summary.append("Next step: all workflow steps are complete.")

        summary.extend(["", "Registered files:"])
        summary.extend(file_lines if file_lines else ["- No files registered yet."])

        eligibility_lines = self._format_eligibility_panel(state)
        if eligibility_lines:
            summary.extend(["", "Initial eligibility review:"])
            summary.extend(f"- {line}" for line in eligibility_lines)

        pending = state.get_pending_questions()
        summary.extend(["", f"Pending collaborator questions: {len(pending)}"])
        if pending:
            for question in pending[:5]:
                text = question.get("question", "").strip().replace("\n", " ")
                summary.append(f"- {text}")

        summary.extend(["", "You can ask for a status explanation, inspect files, or say Proceed to work on the next step."])
        return "\n".join(summary)

    def _format_eligibility_panel(self, state: WorkflowState) -> list[str]:
        log_json = Path(state.working_dir) / "micro" / ".log_data.json"
        if not log_json.exists():
            return []
        try:
            data = json.loads(log_json.read_text(encoding="utf-8"))
        except Exception:
            return []
        study_design = data.get("study_design", {})
        fields = [
            ("sample_size", "Sample size"),
            ("probability_sample_status", "Probability sample status"),
            ("probability_sample_assessment", "Probability sample"),
            ("sampling_evidence", "Sampling evidence"),
            ("cses_item_coverage", "CSES item coverage"),
            ("eligibility_assessment", "Eligibility"),
            ("processor_eligibility_decision", "Processor decision"),
        ]
        lines = []
        for key, label in fields:
            value = str(study_design.get(key, "")).strip()
            if value:
                lines.append(f"{label}: {value}")
        return lines

    def _append_output(self, text: str) -> None:
        text = sanitize_processor_text(text)
        if not hasattr(self, "output_text"):
            self._append_chat("tool", text.strip())
            return
        self.output_text.configure(state="normal")
        self.output_text.insert("end", text)
        self.output_text.see("end")
        self.output_text.configure(state="disabled")


def run_gui() -> None:
    app = CSESGui()
    app.mainloop()


if __name__ == "__main__":
    run_gui()
