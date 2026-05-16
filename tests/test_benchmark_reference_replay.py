import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.benchmark import BenchmarkDecisionExtractor
from src.codegen.recoding_plan import RecodingPlan
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

    def test_reference_label_extraction_ignores_missing_labels(self):
        extractor = BenchmarkDecisionExtractor()
        meta = Mock()
        meta.column_names = ["F4001_N", "F3001"]
        meta.column_labels = [None, "SATISFACTION WITH DEMOCRACY"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "reference.dta"
            path.write_text("placeholder", encoding="utf-8")
            with patch("pyreadstat.read_dta", return_value=(None, meta)):
                labels = extractor._variable_labels(path)

        self.assertNotIn("F4001_N", labels)
        self.assertEqual(labels["F3001"], "SATISFACTION WITH DEMOCRACY")

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

            self.assertEqual(plan.plan_type, "preserved_reference_variable")
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

    def test_preserved_reference_plan_clones_source_variable(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated.do"
            registry = SchemaRegistry()
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=folder)
            builder = RecodingPlanBuilder(state, registry=registry)
            (Path(folder) / ".cses").mkdir(exist_ok=True)
            builder.benchmark_replay_enabled = True
            builder.source_variables = {"F2003"}
            plan = builder._plan_for(registry.by_name("F2003"), None)

            PlanDrivenStataSyntaxGenerator(registry=registry).generate(
                plans=[plan],
                data_file_path=Path(folder) / "source.dta",
                output_path=output,
                country_name="Testland",
                country_code="TST",
                year="2024",
                draft=True,
            )
            text = output.read_text(encoding="utf-8")

            self.assertIn("clonevar F2003 = __src_F2003", text)
            self.assertNotIn('label variable F2003 "EDUCATION"', text)

    def test_district_replay_preserves_existing_reference_variable_when_available(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "generated.do"
            plan = RecodingPlan(
                target_variable="F4001",
                description="NUMBER OF SEATS IN DISTRICT",
                dependency_class="district_input",
                plan_type="district_merged",
                source_variables=["district_data_normalized.dta"],
                expression="",
                documentation_note="Merged from district data.",
                verification_commands=["tab F4001, mis"],
                approved=True,
                readiness_status="ready",
            )

            PlanDrivenStataSyntaxGenerator(registry=SchemaRegistry()).generate(
                plans=[plan],
                data_file_path=Path(folder) / "source.dta",
                output_path=output,
                country_name="Testland",
                country_code="TST",
                year="2024",
                draft=True,
                district_merge_plan={"approved": True},
            )
            text = output.read_text(encoding="utf-8")

            self.assertIn("capture confirm variable __src_F4001", text)
            self.assertIn("if !_rc clonevar F4001 = __src_F4001", text)

    def test_stata_generator_uses_reference_dataset_in_benchmark_replay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / ".cses").mkdir()
            reference = root / "reference.dta"
            reference.write_text("placeholder", encoding="utf-8")
            (root / ".cses" / "benchmark_decision_replay.json").write_text(
                json.dumps({"reference_dataset": str(reference)}),
                encoding="utf-8",
            )
            output = root / "micro" / "generated.do"

            PlanDrivenStataSyntaxGenerator(registry=SchemaRegistry()).generate(
                plans=[],
                data_file_path=root / "raw.dta",
                output_path=output,
                country_name="Testland",
                country_code="TST",
                year="2024",
                draft=True,
            )
            text = output.read_text(encoding="utf-8")

            self.assertIn(f'use "{reference}", clear', text)

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

    def test_builder_uses_benchmark_constant_when_reference_plan_depends_on_later_variable(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            (study_dir / ".cses").mkdir()
            (study_dir / ".cses" / "benchmark_decision_replay.json").write_text(
                json.dumps(
                    {
                        "constant_values": {"F3011_LR_MARPOR": 9},
                        "reference_recoding_plans": {
                            "F3011_LR_MARPOR": {
                                "lines": [
                                    "gen F3011_LR_MARPOR = 9",
                                    "recode F3011_LR_MARPOR (9 = 0) if inrange(F3100_LR_MARPOR, -100, -0.01)",
                                ],
                                "source_variables": ["F3100_LR_MARPOR"],
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(study_dir))
            builder = RecodingPlanBuilder(state, registry=SchemaRegistry())
            builder.source_variables = set()
            schema_var = SchemaRegistry().by_name("F3011_LR_MARPOR")

            plan = builder._plan_for(schema_var, None)

            self.assertEqual(plan.plan_type, "constant_metadata")
            self.assertEqual(plan.expression, "9")
            self.assertTrue(plan.approved)


if __name__ == "__main__":
    unittest.main()
