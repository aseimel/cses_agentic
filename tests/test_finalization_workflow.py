import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.codegen.party_recoding import PartyRecodingPlanBuilder
from src.codegen.recoding_plan import RecodingPlan, StataSyntaxPlanner
from src.codegen.sheet_reader import TrackingSheet, VariableMapping
from src.standards.artifacts import FinalReadinessValidator, LabelFileGenerator
from src.workflow.state import WorkflowState


class FinalizationWorkflowTests(unittest.TestCase):
    def _write_party_decision(self, study_dir: Path) -> None:
        cses = study_dir / ".cses"
        cses.mkdir()
        payload = {
            "proposal": {
                "proposed_parties": [
                    {"code_letter": "A", "numeric_code": "999001", "party_name": "First Party"},
                    {"code_letter": "B", "numeric_code": "999002", "party_name": "Second Party"},
                    {"code_letter": "C", "numeric_code": "999003", "party_name": "Third Party"},
                ]
            },
            "approval": {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
                "override_reason": "",
            },
        }
        (cses / "party_order_decision.json").write_text(json.dumps(payload), encoding="utf-8")

    def test_party_recode_map_uses_locked_party_order(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_decision(study_dir)
            data_path = study_dir / "source.csv"
            pd.DataFrame({"vote": [1, 2, 3, 97, 98, 99]}).to_csv(data_path, index=False)
            mappings = {
                "F3011_LH_PL": VariableMapping(
                    cses_var="F3011_LH_PL",
                    cses_desc="vote choice",
                    source_var="vote",
                    verified=True,
                )
            }

            maps = PartyRecodingPlanBuilder(study_dir).build_maps(mappings, str(data_path))

            vote_map = {item.target_variable: item for item in maps}["F3011_LH_PL"]
            self.assertTrue(vote_map.approved)
            self.assertEqual(vote_map.value_map["1"], "999001")
            self.assertEqual(vote_map.value_map["2"], "999002")
            self.assertEqual(vote_map.value_map["3"], "999003")
            self.assertEqual(vote_map.missing_map["99"], "999999")

    def test_unapproved_party_category_map_blocks_final_syntax(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_decision(study_dir)
            data_path = study_dir / "source.csv"
            pd.DataFrame({"party_id": [1, 2, 3]}).to_csv(data_path, index=False)
            state = WorkflowState(working_dir=str(study_dir), data_file=str(data_path))
            tracking = TrackingSheet(
                mappings=[
                    VariableMapping(
                        cses_var="F3023_3",
                        cses_desc="party id",
                        source_var="party_id",
                        verified=True,
                    )
                ]
            )

            from src.codegen.recoding_plan import RecodingPlanBuilder

            plans = RecodingPlanBuilder(state).build(tracking)
            party_id = next(plan for plan in plans if plan.target_variable == "F3023_3")

            self.assertEqual(party_id.readiness_status, "needs_processor_review")
            self.assertIn("Party-identification", " ".join(party_id.issues))

    def test_stata_planner_can_exclude_district_for_benchmark_mode(self):
        plans = [
            RecodingPlan(
                target_variable="F3001",
                description="test",
                plan_type="direct_copy",
                readiness_status="ready",
                approved=True,
                dependency_class="direct_survey_item",
            ),
            RecodingPlan(
                target_variable="F4001",
                description="district",
                plan_type="external_input_required",
                readiness_status="blocked_district_input",
                approved=False,
                dependency_class="district_input",
            ),
        ]

        planner = StataSyntaxPlanner()

        self.assertEqual(len(planner.unresolved_required(plans)), 1)
        self.assertEqual(planner.unresolved_required(plans, exclude_district=True), [])

    def test_final_readiness_mode_excludes_district_gate_only_when_recorded(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            micro = study_dir / "micro"
            micro.mkdir()
            (micro / "cses-m6_micro_TST_2024.do").write_text("Process CSES-M6 Micro-Data\n**>>> F3001\nlog close\n", encoding="utf-8")
            (micro / "cses-m6_micro_TST_2024.dta").write_text("placeholder", encoding="utf-8")
            state = WorkflowState(
                working_dir=str(study_dir),
                matching_decisions_path=str(study_dir / ".cses" / "matching_decisions.json"),
                recoding_plans_path=str(study_dir / ".cses" / "recoding_plans.json"),
                input_manifest_path=str(study_dir / ".cses" / "input_manifest.json"),
                readiness_mode="release_ready_except_district",
                district_excluded_by_processor=True,
                recoding_coverage={
                    "target_count": 2,
                    "approved_count": 1,
                    "non_district_target_count": 1,
                    "non_district_approved_count": 1,
                },
                stata_execution_status={"success": True},
            )

            result = FinalReadinessValidator(state).evaluate(study_dir)

            self.assertEqual(result["mode"], "release_ready_except_district")
            self.assertFalse(any("district" in issue.lower() for issue in result["issues"]))

    def test_party_labels_generated_from_approved_order(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_decision(study_dir)
            state = WorkflowState(working_dir=str(study_dir), country_code="TST", year="2024")

            result = LabelFileGenerator(state).generate_micro_labels(study_dir / "micro" / "labels")
            text = result.path.read_text(encoding="utf-8")

            self.assertIn('label define F5000_ 999001 "999001. First Party"', text)
            self.assertIn("capture label values F3023_3 F3023_3_", text)


if __name__ == "__main__":
    unittest.main()
