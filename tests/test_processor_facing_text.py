import tempfile
import unittest
from pathlib import Path

from src.agent.conversation import ConversationSession
from src.gui_app import CSESGui
from src.ui_text import assert_no_internal_terms, sanitize_processor_text
from src.workflow.state import WorkflowState
from src.workflow.steps import StepResult


class ProcessorFacingTextTests(unittest.TestCase):
    def make_state(self, tmp: Path) -> WorkflowState:
        state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(tmp))
        state.data_file = str(tmp / "micro" / "original_deposit" / "study_data.csv")
        state.questionnaire_files = [str(tmp / "questionnaire.pdf")]
        state.design_report_file = str(tmp / "design_report.pdf")
        state.standards_checks = {"1": {"status": "needs_review"}}
        state.study_kb_missing_fields = ["response_rate"]
        state.study_kb_contradictions = ["mode"]
        state.evidence_index = {
            "diagnostics": {
                "chunks_ok": 1,
                "chunks_total": 2,
                "fields_found": 3,
                "chunks_failed": 1,
            }
        }
        return state

    def test_sanitizer_removes_internal_language(self):
        text = (
            "Evidence packet: missing\n"
            "Study KB diagnostics: 1/2 chunks, fields_found=3\n"
            "Model activity\n"
            "Standards source: .cses/model_diagnostics.json\n"
        )
        cleaned = sanitize_processor_text(text)
        assert_no_internal_terms(cleaned)

    def test_common_tool_progress_messages_are_sanitized(self):
        messages = [
            "Building study KB with openrouter/deepseek (~12,000 input tokens)...",
            "Evidence extraction: 14 files, 19 chunks",
            "Evidence packet refreshed: current",
            "Loaded shared workflow context",
            "Ran CSES standards check for Step 1: ready",
            "Saved workflow state to .cses/state.json",
        ]
        for message in messages:
            with self.subTest(message=message):
                assert_no_internal_terms(sanitize_processor_text(message))

    def test_gui_study_and_files_panels_are_processor_facing(self):
        with tempfile.TemporaryDirectory() as folder:
            state = self.make_state(Path(folder))
            app = CSESGui.__new__(CSESGui)
            for text in (
                app._format_study_panel(state),
                app._format_files_panel(state),
                app._format_loaded_summary(state),
            ):
                assert_no_internal_terms(text)

    def test_direct_proceed_summary_is_processor_facing(self):
        with tempfile.TemporaryDirectory() as folder:
            state = self.make_state(Path(folder))
            session = ConversationSession.__new__(ConversationSession)
            session.state = state
            result = StepResult(
                success=True,
                message="Evidence packet: current\nStudy KB: ready\nStandards source: .cses/state.json",
                issues=["Study KB diagnostics need technical review"],
            )
            text = session._summarize_direct_step(1, result)
            assert_no_internal_terms(text)


if __name__ == "__main__":
    unittest.main()
