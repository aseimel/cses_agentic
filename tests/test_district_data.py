import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.codegen.recoding_plan import PlanDrivenStataSyntaxGenerator, RecodingPlanBuilder, StataSyntaxPlanner
from src.codegen.sheet_reader import TrackingSheet, VariableMapping
from src.district_data import (
    DistrictDataTemplateParser,
    DistrictDataValidator,
    DistrictMergePlanner,
    DistrictStataSyntaxBuilder,
)
from src.workflow.state import WorkflowState
from src.workflow.steps import StepExecutor


class DistrictDataWorkflowTests(unittest.TestCase):
    def _write_party_decision(self, study_dir: Path, letters=("A", "B", "C")) -> None:
        cses = study_dir / ".cses"
        cses.mkdir(exist_ok=True)
        parties = [
            {
                "code_letter": letter,
                "numeric_code": f"99900{idx}",
                "party_name": f"Party {letter}",
            }
            for idx, letter in enumerate(letters, 1)
        ]
        payload = {
            "proposal": {"proposed_parties": parties},
            "approval": {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
                "override_reason": "",
            },
        }
        (cses / "party_order_decision.json").write_text(json.dumps(payload), encoding="utf-8")

    def _write_district_xlsx(self, path: Path) -> None:
        df = pd.DataFrame(
            {
                "DistrictID": [1, 2, 3],
                "OriginalDistrictName": ["One", "Two", "Three"],
                "EnglishDistrictName": ["One", "Two", "Three"],
                "F4001": [10, 11, 12],
                "F4002": [1001, 1002, 1003],
                "F4003": [1, 2, 3],
                "F4004_A": [40.0, 30.0, 20.0],
                "F4004_B": [35.0, 45.0, 25.0],
                "F4004_C": [25.0, 25.0, 55.0],
                "F4005_A": [2, 1, 1],
                "F4005_B": [1, 2, 1],
                "F4005_C": [1, 1, 2],
                "F4006": [4, 4, 4],
                "F4007": [100000, 90000, 80000],
            }
        )
        df.to_excel(path, index=False, sheet_name="District Data")

    def test_parser_normalizes_district_id_and_validator_requires_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_decision(study_dir)
            data_path = study_dir / "source.csv"
            pd.DataFrame({"D18": [1, 2, 3, 99999]}).to_csv(data_path, index=False)
            district_path = study_dir / "District Data Template.xlsx"
            self._write_district_xlsx(district_path)
            state = WorkflowState(working_dir=str(study_dir), data_file=str(data_path))

            table = DistrictDataTemplateParser().parse(district_path)
            validation = DistrictDataValidator().validate(table, state, source_variable="D18")

            self.assertIn("F2019", table.columns)
            self.assertEqual(validation.status, "needs_processor_approval")
            self.assertFalse(validation.approved)
            self.assertEqual(validation.missing_observed_district_codes, [])

    def test_validator_rejects_duplicate_keys_and_missing_coverage(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_decision(study_dir)
            data_path = study_dir / "source.csv"
            pd.DataFrame({"D18": [1, 2, 4]}).to_csv(data_path, index=False)
            district_path = study_dir / "District Data Template.xlsx"
            self._write_district_xlsx(district_path)
            df = pd.read_excel(district_path)
            df.loc[2, "DistrictID"] = 2
            df.to_excel(district_path, index=False)
            state = WorkflowState(working_dir=str(study_dir), data_file=str(data_path))

            table = DistrictDataTemplateParser().parse(district_path)
            validation = DistrictDataValidator().validate(table, state, approve=True, source_variable="D18")

            self.assertIn("2", validation.duplicate_district_codes)
            self.assertIn("4", validation.missing_observed_district_codes)
            self.assertFalse(validation.approved)

    def test_approved_district_plan_resolves_f400_recoding_and_generates_merge_syntax(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_decision(study_dir)
            data_path = study_dir / "source.csv"
            pd.DataFrame({"D18": [1, 2, 3]}).to_csv(data_path, index=False)
            district_path = study_dir / "District Data Template.xlsx"
            self._write_district_xlsx(district_path)
            state = WorkflowState(working_dir=str(study_dir), data_file=str(data_path), country_code="TST", year="2024")

            table = DistrictDataTemplateParser().parse(district_path)
            validation = DistrictDataValidator().validate(table, state, approve=True, source_variable="D18")
            plan = DistrictMergePlanner().build(study_dir, table, validation)
            DistrictMergePlanner().write(study_dir, plan)

            tracking = TrackingSheet(
                mappings=[
                    VariableMapping(cses_var="F2019", cses_desc="district", source_var="D18", verified=True)
                ]
            )
            plans = RecodingPlanBuilder(state).build(tracking)
            f4001 = next(item for item in plans if item.target_variable == "F4001")
            self.assertTrue(f4001.approved)
            self.assertEqual(f4001.plan_type, "district_merged")
            self.assertEqual(StataSyntaxPlanner().unresolved_required([f4001]), [])

            syntax_path = study_dir / "micro" / "cses-m6_micro_TST_2024.do"
            PlanDrivenStataSyntaxGenerator().generate(
                plans=plans,
                output_path=syntax_path,
                data_file_path=str(data_path),
                country_name="Test",
                country_code="TST",
                year="2024",
                district_merge_plan=plan.to_dict(),
            )
            text = syntax_path.read_text(encoding="utf-8")
            self.assertIn("merge m:1 F2019 using", text)
            self.assertIn("gen F2019 = D18", text)
        self.assertIn("capture gen F4001 = 999", text)

    def test_district_merge_block_generates_missing_higher_party_slots(self):
        plan = {
            "approved": True,
            "source_district_variable": "D18",
            "normalized_dta_path": "micro/district data/district_data_normalized.dta",
            "variables_merged": ["F4001", "F4004_A", "F4004_I"],
            "generated_missing_slots": ["F4004_I"],
        }

        text = "\n".join(DistrictStataSyntaxBuilder().merge_lines(plan))

        self.assertIn("capture gen F4004_I = 999", text)
        self.assertIn("capture replace F4004_I = 999 if F4004_I == .", text)

    def test_step_9_approves_standardized_district_file_when_processor_confirms(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            (study_dir / "micro").mkdir()
            self._write_party_decision(study_dir)
            data_path = study_dir / "source.csv"
            pd.DataFrame({"D18": [1, 2, 3]}).to_csv(data_path, index=False)
            district_path = study_dir / "District Data Template.xlsx"
            self._write_district_xlsx(district_path)
            state = WorkflowState(
                working_dir=str(study_dir),
                data_file=str(data_path),
                country="Testland",
                country_code="TST",
                year="2024",
            )

            result = StepExecutor(state).execute_step(
                9,
                district_file=str(district_path),
                source_variable="D18",
                approve=True,
            )

            self.assertTrue(result.success)
            self.assertTrue(state.district_data_status["approved"])
            self.assertTrue(Path(state.district_merge_plan_path).exists())
            self.assertIn("District data review approved", result.message)

    def test_sweden_template_matches_processed_district_data_when_reference_available(self):
        root = Path(__file__).resolve().parents[1]
        template = root / "Sweden_2022" / "micro" / "district data" / "District Data Template Sweden 2022.xlsx"
        processed = root / "Sweden_2022" / "micro" / "district data" / "SWE2022_DistrictData.dta"
        if not template.exists() or not processed.exists():
            self.skipTest("Sweden district reference files not available")

        table = DistrictDataTemplateParser().parse(template)
        processed_df = pd.read_stata(processed)
        parsed_df = table.dataframe[[col for col in processed_df.columns if col in table.dataframe.columns]].copy()
        parsed_df = parsed_df[processed_df.columns]
        parsed_df = parsed_df.sort_values("F2019").reset_index(drop=True)
        processed_df = processed_df.sort_values("F2019").reset_index(drop=True)

        self.assertEqual(table.district_count, 29)
        self.assertEqual(list(parsed_df.columns), list(processed_df.columns))
        pd.testing.assert_frame_equal(
            parsed_df,
            processed_df,
            check_dtype=False,
        )


if __name__ == "__main__":
    unittest.main()
