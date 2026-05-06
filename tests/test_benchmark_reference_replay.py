import json
import tempfile
import unittest
from pathlib import Path

from src.benchmark import BenchmarkDecisionExtractor
from src.codegen.recoding_plan import RecodingPlanBuilder
from src.codegen.recoding_plan import PlanDrivenStataSyntaxGenerator
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

    def test_builder_replays_existing_target_when_reference_source_is_unavailable(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            (study_dir / ".cses").mkdir()
            (study_dir / ".cses" / "benchmark_decision_replay.json").write_text(
                json.dumps(
                    {
                        "reference_recoding_plans": {
                            "F2003": {
                                "lines": ["gen F2003 = D03", "tab F2003, mis"],
                                "source_variables": ["D03"],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(study_dir))
            builder = RecodingPlanBuilder(state, registry=SchemaRegistry())
            builder.source_variables = {"F2003"}
            schema_var = SchemaRegistry().by_name("F2003")

            plan = builder._plan_for(schema_var, None)

            self.assertEqual(plan.plan_type, "direct_copy")
            self.assertTrue(plan.approved)
            self.assertEqual(plan.expression, "__src_F2003")

    def test_stata_generator_preserves_existing_target_variables(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated.do"
            registry = SchemaRegistry()
            PlanDrivenStataSyntaxGenerator(registry=registry).generate(
                plans=[],
                data_file_path=Path(folder) / "source.dta",
                output_path=output,
                country_name="Testland",
                country_code="TST",
                year="2024",
                draft=True,
            )
            text = output.read_text(encoding="utf-8")

            self.assertIn("capture confirm variable F2003", text)
            self.assertIn("if !_rc rename F2003 __src_F2003", text)

    def test_reference_plan_with_empty_assignment_is_not_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            (study_dir / ".cses").mkdir()
            (study_dir / ".cses" / "benchmark_decision_replay.json").write_text(
                json.dumps(
                    {
                        "reference_recoding_plans": {
                            "F5201_A": {
                                "lines": ["gen F5201_A ="],
                                "source_variables": [],
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(study_dir))
            builder = RecodingPlanBuilder(state, registry=SchemaRegistry())

            self.assertFalse(builder._reference_plan_usable("F5201_A", builder.benchmark_reference_plans["F5201_A"]))


if __name__ == "__main__":
    unittest.main()
