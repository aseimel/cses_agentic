import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import BenchmarkDecisionExtractor
from src.codegen.recoding_plan import RecodingPlanBuilder
from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState


class BenchmarkReferenceReplayTests(unittest.TestCase):
    def test_extracts_target_specific_reference_recoding_lines(self):
        with tempfile.TemporaryDirectory() as folder:
            syntax = Path(folder) / "reference.do"
            syntax.write_text(
                """
**>>> F2003 EDUCATION
gen F2003 = .
replace F2003 = 2 if D03 == 1
replace F2003 = 3 if D03 == 2
replace F2003 = 99 if D03 == .
tab F2003, mis

**>>> F1010_ DATE
gen str2 month1 = string(F1010_M,"%02.0f")
gen str F1010_1 = year1 + "-" + month1
tab F1010_1, mis
""",
                encoding="utf-8",
            )

            payload = BenchmarkDecisionExtractor().extract(reference_syntax=syntax)
            plans = payload["reference_recoding_plans"]

            self.assertIn("F2003", plans)
            self.assertEqual(plans["F2003"]["source_variables"], ["D03"])
            self.assertIn("replace F2003 = 2 if D03 == 1", plans["F2003"]["lines"])
            self.assertNotIn("F1010_1", plans)

    def test_builder_uses_benchmark_reference_plan_when_source_exists(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            (study_dir / ".cses").mkdir()
            (study_dir / ".cses" / "benchmark_decision_replay.json").write_text(
                json.dumps(
                    {
                        "reference_recoding_plans": {
                            "F2003": {
                                "lines": ["gen F2003 = .", "replace F2003 = 2 if D03 == 1", "tab F2003, mis"],
                                "source_variables": ["D03"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(study_dir))
            builder = RecodingPlanBuilder(state, registry=SchemaRegistry())
            builder.source_variables = {"D03"}
            schema_var = SchemaRegistry().by_name("F2003")

            plan = builder._plan_for(schema_var, None)

            self.assertEqual(plan.plan_type, "reference_stata_lines")
            self.assertTrue(plan.approved)
            self.assertIn("replace F2003 = 2 if D03 == 1", plan.custom_stata_lines)


if __name__ == "__main__":
    unittest.main()
