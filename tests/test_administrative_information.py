import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ingest.data_loader import DatasetInfo, VariableInfo
from src.codegen.code_templates import StataTemplates
from src.codegen.sheet_reader import VariableMapping
from src.matching.decision_engine import MatchingDecisionEngine, matching_category_summary
from src.standards.administrative import (
    AdministrativeDataGenerator,
    AdministrativeFactBuilder,
    AdministrativeVariablePlanner,
    AdministrativeVariableRegistry,
)
from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState


class AdministrativeInformationTests(unittest.TestCase):
    def test_every_f10_f11_variable_has_contract(self):
        registry = AdministrativeVariableRegistry()
        issues = registry.validate()

        self.assertEqual(issues, [])
        schema = SchemaRegistry()
        admin_names = [item.name for item in schema.variables if item.name.startswith(("F10", "F11"))]
        contracted = {item.target_variable for item in registry.contracts()}
        self.assertTrue(admin_names)
        self.assertEqual(set(admin_names), contracted)

    def test_builds_administrative_facts_and_plans_without_llm_matching(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            dataset = DatasetInfo(
                file_path=study_dir / "data.csv",
                file_format="csv",
                n_rows=100,
                n_variables=4,
                variables={
                    "A1": VariableInfo("A1", sample_values=[1, 2, 3], n_unique=100),
                    "A4a": VariableInfo("A4a", sample_values=[9], n_unique=1),
                    "A4b": VariableInfo("A4b", sample_values=[11], n_unique=1),
                    "A4c": VariableInfo("A4c", sample_values=[2022], n_unique=1),
                    "mode": VariableInfo("mode", sample_values=[1], n_unique=1),
                },
                metadata_quality="minimal",
            )
            state = WorkflowState(country="Sweden", country_code="SWE", year="2022", working_dir=str(study_dir))
            packet = {"design_facts": {"mode": "web", "fieldwork_dates": "2022-09-27 to 2022-11-21"}}

            facts = AdministrativeFactBuilder(study_dir).build(state, dataset_info=dataset, evidence_packet=packet)
            plans = AdministrativeVariablePlanner().build_plans(facts, source_variables=set(dataset.variables))
            by_target = {plan.target_variable: plan for plan in plans}

            self.assertEqual(by_target["F1001"].status, "confirmed")
            self.assertEqual(by_target["F1019_M"].source_variable, "A4a")
            self.assertEqual(by_target["F1016_1"].source_variable, "mode")
            self.assertEqual(by_target["F1004"].value, "SWE_2022")

    def test_matching_engine_resolves_admin_rows_from_administrative_plans(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            dataset = DatasetInfo(
                file_path=study_dir / "data.csv",
                file_format="csv",
                n_rows=10,
                n_variables=1,
                variables={"A1": VariableInfo("A1", sample_values=[1, 2, 3], n_unique=10)},
                metadata_quality="minimal",
            )
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(study_dir))
            facts = AdministrativeFactBuilder(study_dir).build(state, dataset_info=dataset, evidence_packet={})
            plans = AdministrativeVariablePlanner().build_plans(facts, source_variables=set(dataset.variables))

            decisions = MatchingDecisionEngine().decide(
                source_contexts=[{"name": "A1", "description": "", "value_labels": {}, "sample_values": [1, 2, 3]}],
                administrative_plans=plans,
            )
            admin = [item for item in decisions if item.target_variable.startswith(("F10", "F11"))]

            self.assertTrue(admin)
            self.assertTrue(all(item.status == "generated_from_administrative_information" for item in admin))
            summary = matching_category_summary(decisions)
            self.assertEqual(
                summary["administrative_metadata"]["prepared"],
                summary["administrative_metadata"]["total"],
            )

    def test_administrative_stata_template_generates_constant_or_copy(self):
        template = StataTemplates()
        constant = template.generate(
            VariableMapping(
                cses_var="F1001",
                cses_desc="DATASET",
                source_var="ADMINISTRATIVE_INFORMATION",
                transform_type="administrative_information",
                notes="Administrative value: CSES-MODULE-6 | Stata rule: constant_string",
            )
        )
        copied = template.generate(
            VariableMapping(
                cses_var="F1019_M",
                cses_desc="DATE QUESTIONNAIRE ADMINISTERED - MONTH",
                source_var="A4a",
                transform_type="administrative_information",
            )
        )

        self.assertIn('gen str F1001 = "CSES-MODULE-6"', constant.code)
        self.assertIn("gen F1019_M = A4a", copied.code)
        self.assertIn("tab A4a F1019_M, mis", copied.code)

    def test_sweden_reference_admin_block_matches_after_processor_decisions(self):
        try:
            import pyreadstat
        except ImportError:
            self.skipTest("pyreadstat is not installed")

        root = Path.cwd()
        source = root / "Sweden_2022" / "micro" / "deposited datasets" / "CSES6_SWEDEN.dta"
        final = root / "Sweden_2022" / "micro" / "FINAL dataset" / "cses-m6_micro_SWE_2022_20251014.dta"
        if not source.exists() or not final.exists():
            self.skipTest("Sweden reference files are not available in this checkout")

        source_df, _ = pyreadstat.read_dta(str(source), apply_value_formats=False)
        final_df, _ = pyreadstat.read_dta(str(final), apply_value_formats=False)
        state = WorkflowState(country="Sweden", country_code="SWE", year="2022", working_dir=str(root / "Sweden_2022"))
        state.processor_decisions = [
            {"key": "election_date", "value": "2022-09-11", "evidence": "Sweden reference benchmark"},
            {"key": "no_second_round", "value": True, "evidence": "Sweden reference benchmark"},
            {"key": "F1012_1", "value": 1, "evidence": "Sweden reference benchmark"},
            {"key": "F1012_2", "value": 2, "evidence": "Sweden reference benchmark"},
            {"key": "F1013", "value": 1, "evidence": "Sweden reference benchmark"},
            {"key": "F1014", "value": 10, "evidence": "Sweden reference benchmark"},
            {"key": "F1023", "value": 123, "evidence": "Sweden reference benchmark"},
            {"key": "F1106", "value": 2, "evidence": "Sweden reference benchmark"},
        ]
        dataset = DatasetInfo(
            file_path=source,
            file_format="stata",
            n_rows=len(source_df),
            n_variables=len(source_df.columns),
            variables={
                name: VariableInfo(name=name)
                for name in source_df.columns
            },
            metadata_quality="minimal",
        )
        with tempfile.TemporaryDirectory() as folder:
            facts = AdministrativeFactBuilder(Path(folder)).build(
                state,
                dataset_info=dataset,
                evidence_packet={
                    "design_facts": {
                        "fieldwork_dates": "2022-09-12 to 2023-01-04",
                        "mode": "Self-completion: Paper (by mail) and Self-completion: Internet",
                        "weights": "Demographic weight A5",
                    }
                },
            )
        generated = AdministrativeDataGenerator().generate(source_df, facts)
        admin_vars = [col for col in final_df.columns if col.startswith(("F10", "F11"))]
        left = generated.assign(_key=generated["F1003_2"].astype(str)).sort_values("_key").reset_index(drop=True)
        right = final_df[admin_vars].assign(_key=final_df["F1003_2"].astype(str)).sort_values("_key").reset_index(drop=True)

        self.assertTrue((left["_key"] == right["_key"]).all())
        for col in admin_vars:
            with self.subTest(col=col):
                if pd.api.types.is_numeric_dtype(right[col]):
                    actual = pd.to_numeric(left[col], errors="coerce").fillna(-999999999)
                    expected = pd.to_numeric(right[col], errors="coerce").fillna(-999999999)
                    self.assertTrue(((actual - expected).abs() < 1e-6).all())
                else:
                    self.assertTrue((left[col].astype(str).fillna("") == right[col].astype(str).fillna("")).all())


if __name__ == "__main__":
    unittest.main()
