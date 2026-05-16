import json
import tempfile
import unittest
from pathlib import Path

from src.agent.conversation import ConversationSession
from src.codegen.recoding_plan import RecodingPlanBuilder
from src.codegen.sheet_reader import TrackingSheet
from src.processor_decisions import (
    CorrectionInterpreter,
    DependencyInvalidator,
    ProcessorDecision,
    ProcessorDecisionLedger,
    TargetedRerunInterpreter,
    diagnose_failure_text,
)
from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState


class ProcessorCorrectionTests(unittest.TestCase):
    def test_processor_decision_ledger_persists_and_filters_approved(self):
        with tempfile.TemporaryDirectory() as folder:
            ledger = ProcessorDecisionLedger(folder)
            first = ledger.append(
                ProcessorDecision(
                    decision_id="",
                    area="matching",
                    decision_type="source_variable_correction",
                    target="F3001",
                    value="q1",
                    status="pending_confirmation",
                )
            )
            ledger.append(
                ProcessorDecision(
                    decision_id="",
                    area="party_order",
                    decision_type="party_order_correction",
                    target="Party A",
                    value="Example Party",
                    status="approved",
                )
            )

            self.assertEqual(first.decision_id, "PD-0001")
            self.assertEqual(len(ledger.load()), 2)
            self.assertEqual(len(ledger.latest_approved()), 1)
            self.assertEqual(ledger.latest_approved()[0].area, "party_order")

    def test_dependency_invalidation_is_area_specific(self):
        decision = ProcessorDecision(
            decision_id="PD-0001",
            area="party_order",
            decision_type="party_order_correction",
            target="Party C",
            value="Centre Party",
            status="approved",
        )

        impact = DependencyInvalidator().impact_for(decision)

        self.assertIn("party_recodes", impact.affected_outputs)
        self.assertIn("party_labels", impact.affected_outputs)
        self.assertIn(10, impact.affected_steps)
        self.assertNotIn("district_merge", impact.affected_outputs)

    def test_correction_interpreter_parses_common_processor_language(self):
        parser = CorrectionInterpreter()

        sample = parser.parse("No, the sample is a probability sample.", default_step=1)
        self.assertIsNotNone(sample)
        self.assertEqual(sample.area, "eligibility_design")
        self.assertEqual(sample.value, "yes")
        self.assertEqual(sample.status, "approved")

        match = parser.parse("F3001 should use variable q1", default_step=7)
        self.assertIsNotNone(match)
        self.assertEqual(match.area, "matching")
        self.assertEqual(match.target, "F3001")
        self.assertEqual(match.value, "q1")
        self.assertEqual(match.status, "pending_confirmation")

        party = parser.parse("Party C should be the Centre Party", default_step=7)
        self.assertIsNotNone(party)
        self.assertEqual(party.area, "party_order")
        self.assertEqual(party.target, "Party C")

    def test_conversation_records_correction_without_model_call(self):
        with tempfile.TemporaryDirectory() as folder:
            state = WorkflowState(working_dir=folder, country="Testland", country_code="TST", year="2024")
            state.save()
            session = ConversationSession(state)

            response = session.send("No, the sample is a probability sample.")
            loaded = WorkflowState.load(Path(folder))
            ledger = ProcessorDecisionLedger(folder).load()

            self.assertIn("Correction recorded", response)
            self.assertEqual(len(ledger), 1)
            self.assertEqual(ledger[0].area, "eligibility_design")
            self.assertGreaterEqual(len(loaded.invalidated_outputs), 1)

    def test_approved_source_match_correction_overrides_recoding_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            state = WorkflowState(working_dir=folder, country="Testland", country_code="TST", year="2024")
            ledger = ProcessorDecisionLedger(study_dir)
            ledger.append(
                ProcessorDecision(
                    decision_id="",
                    area="matching",
                    decision_type="source_variable_correction",
                    target="F3001",
                    value="q1_corrected",
                    reason="Processor selected the correct questionnaire item.",
                    status="approved",
                )
            )
            registry = SchemaRegistry()
            registry.variables = [item for item in registry.variables if item.name == "F3001"]
            plans = RecodingPlanBuilder(state, registry=registry).build(TrackingSheet(mappings=[]))

            self.assertEqual(len(plans), 1)
            self.assertEqual(plans[0].source_variables, ["q1_corrected"])
            self.assertTrue(plans[0].approved)
            self.assertIn("Processor-corrected", plans[0].documentation_note)

    def test_targeted_rerun_and_failure_diagnosis_are_processor_facing(self):
        rerun = TargetedRerunInterpreter().parse("please rebuild party recodes")
        self.assertIsNotNone(rerun)
        self.assertEqual(rerun["action"], "rebuild_party_recodes")

        diagnosis = diagnose_failure_text("F2019 _merge has unmatched district rows")
        self.assertEqual(diagnosis["category"], "district merge problem")
        self.assertIn("District Data Review", diagnosis["suggested_fix"])


if __name__ == "__main__":
    unittest.main()
