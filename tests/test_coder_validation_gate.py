import tempfile
import unittest

from src.agent.conversation import ConversationSession
from src.workflow.state import StepStatus, WorkflowState
from src.workflow.steps import StepExecutor, StepResult


class CoderValidationGateTests(unittest.TestCase):
    def test_step_executor_can_pause_successful_step_for_coder_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = WorkflowState(working_dir=tmp, country="Testland", year="2024")
            state.set_step_status(0, StepStatus.COMPLETED, "Initialized")
            executor = StepExecutor(state)
            executor._step_1 = lambda **_kwargs: StepResult(success=True, message="Eligibility reviewed")

            result = executor.execute_step(1, require_processor_validation=True)

            self.assertTrue(result.success)
            self.assertEqual(state.get_step(1).status, StepStatus.NEEDS_VALIDATION.value)
            self.assertFalse(result.ready_for_next_step)
            self.assertEqual(state.get_next_step(), 1)

    def test_next_step_does_not_advance_until_coder_validates(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = WorkflowState(working_dir=tmp, country="Testland", year="2024")
            state.set_step_status(0, StepStatus.COMPLETED, "Initialized")
            state.mark_step_needs_validation(1, "Review needed")

            session = ConversationSession(state)
            proceed_response = session.send("Proceed")
            self.assertIn("ready for coder validation", proceed_response)
            self.assertEqual(session.state.get_next_step(), 1)

            validate_response = session.send("validate step 1")
            self.assertIn("Step 1 validated", validate_response)
            self.assertEqual(session.state.get_step(1).status, StepStatus.COMPLETED.value)
            self.assertEqual(session.state.get_next_step(), 2)
            self.assertIn("1", session.state.step_validations)

    def test_prerequisite_message_names_validation_before_next_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = WorkflowState(working_dir=tmp, country="Testland", year="2024")
            state.set_step_status(0, StepStatus.COMPLETED, "Initialized")
            state.mark_step_needs_validation(1, "Review needed")

            ok, reason = state.check_step_prerequisites(2)

            self.assertFalse(ok)
            self.assertIn("needs coder validation", reason)


if __name__ == "__main__":
    unittest.main()
