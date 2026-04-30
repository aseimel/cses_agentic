import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.ingest.data_loader import DataLoader
from src.matching.demographics import (
    DemographicDataGenerator,
    DemographicRecodingDecision,
    DemographicRecodingDecisionStore,
    DemographicRecodingDossierBuilder,
    DemographicRecodingAssessmentEngine,
    DemographicReferenceDecisionLearner,
    DemographicVariableRegistry,
)
from src.matching.decision_engine import MatchingDecisionEngine, matching_category_summary
from src.matching.evidence import MatchingEvidenceBuilder
from src.codegen.recoding_plan import RecodingPlanBuilder
from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState


class DemographicRecodingTests(unittest.TestCase):
    def test_every_f2_variable_has_demographic_contract(self):
        issues = DemographicVariableRegistry().validate()
        self.assertEqual(issues, [])

        schema = SchemaRegistry()
        demographic_names = {item.name for item in schema.variables if item.section == "demographics"}
        contracted = {item.target_variable for item in DemographicVariableRegistry().variables}
        self.assertEqual(demographic_names, contracted)

    def test_demographic_assessment_runs_after_core_matching_and_excludes_external_categories(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            matching_evidence = MatchingEvidenceBuilder(study_dir).build_from_records(
                narrative_docs=[],
                data_summaries=[
                    {
                        "path": str(study_dir / "data.csv"),
                        "relative_path": "data.csv",
                        "variable_inventory": [
                            {"name": "D01b", "label": "Birth year", "sample_values": [1980, 1990]},
                            {"name": "D02", "label": "Gender", "sample_values": [1, 2]},
                            {"name": "D03", "label": "Education", "sample_values": [1, 2, 3]},
                            {"name": "Q01", "label": "Political interest", "sample_values": [1, 2, 3]},
                        ],
                    }
                ],
            )

            assessments = DemographicRecodingAssessmentEngine().assess(matching_evidence)
            by_target = {item.target_variable: item for item in assessments}

            self.assertEqual(by_target["F2001_Y"].source_variable, "D01b")
            self.assertEqual(by_target["F2002"].source_variable, "D02")
            self.assertEqual(by_target["F2003"].recoding_plan_type, "crosswalk_required")
            self.assertFalse(any(item.target_variable.startswith(("F4", "F5", "F6")) for item in assessments))

            decisions = MatchingDecisionEngine().decide(
                source_contexts=[
                    {"name": "D01b", "description": "Birth year"},
                    {"name": "D02", "description": "Gender"},
                    {"name": "D03", "description": "Education"},
                    {"name": "Q01", "description": "Political interest"},
                ],
                matching_evidence=matching_evidence,
                demographic_assessments=assessments,
            )
            summary = matching_category_summary(decisions, matching_evidence)
            self.assertGreater(summary["demographic_items"]["matched"], 0)
            self.assertGreater(summary["party_election_items"]["awaiting_party_ordering"], 0)
            self.assertGreater(summary["district_items"]["awaiting_district_input"], 0)

    def test_demographic_dossier_collects_context_for_later_coding(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            matching_evidence = MatchingEvidenceBuilder(study_dir).build_from_records(
                narrative_docs=[],
                data_summaries=[
                    {
                        "path": str(study_dir / "data.csv"),
                        "relative_path": "data.csv",
                        "variable_inventory": [
                            {
                                "name": "D03",
                                "label": "Highest education",
                                "sample_values": [1, 2, 3],
                                "value_labels": {
                                    "1": "Primary school",
                                    "2": "Lower secondary",
                                    "3": "Upper secondary",
                                },
                            }
                        ],
                    }
                ],
            )
            assessments = DemographicRecodingAssessmentEngine().assess(matching_evidence)
            dossiers = DemographicRecodingDossierBuilder().build(assessments, matching_evidence)
            education = next(item for item in dossiers if item.target_variable == "F2003")

            self.assertEqual(education.source_variable, "D03")
            self.assertEqual(education.proposed_plan_type, "crosswalk_required")
            self.assertTrue(education.target_standard["response_options"])
            self.assertTrue(education.source_evidence["value_labels"])
            self.assertIn("crosswalk", education.processor_decision_needed.lower())

    def test_approved_demographic_decisions_feed_later_recoding_plans(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            DemographicRecodingDecisionStore(study_dir).write(
                [
                    DemographicRecodingDecision(
                        target_variable="F2002",
                        source_variable="D02",
                        plan_type="offset_transform",
                        approved=True,
                        processor_note="Approved gender recode.",
                    )
                ]
            )
            state = WorkflowState(country="Testland", country_code="TST", year="2024", working_dir=str(study_dir))

            plans = RecodingPlanBuilder(state).build()
            gender = next(plan for plan in plans if plan.target_variable == "F2002")

            self.assertTrue(gender.approved)
            self.assertEqual(gender.readiness_status, "ready")
            self.assertEqual(gender.expression, "D02 - 1")

    def test_sweden_demographic_reference_replication(self):
        try:
            import pyreadstat
        except ImportError:
            self.skipTest("pyreadstat is not installed")

        root = Path.cwd()
        source = root / "Sweden_2022" / "micro" / "deposited datasets" / "CSES6_SWEDEN.dta"
        final = root / "Sweden_2022" / "micro" / "FINAL dataset" / "cses-m6_micro_SWE_2022_20251014.dta"
        if not source.exists() or not final.exists():
            self.skipTest("Sweden reference files are not available in this checkout")

        dataset = DataLoader().load(source)
        self.assertIsNotNone(dataset)
        with tempfile.TemporaryDirectory() as folder:
            matching_evidence = MatchingEvidenceBuilder(Path(folder)).build_from_loaded(dataset, [])
            assessments = DemographicRecodingAssessmentEngine().assess(matching_evidence)

        source_df, _ = pyreadstat.read_dta(str(source), apply_value_formats=False)
        final_df, _ = pyreadstat.read_dta(str(final), apply_value_formats=False)
        approved = DemographicReferenceDecisionLearner().infer_maps(
            source_df=source_df,
            final_df=final_df,
            assessments=assessments,
            source_id="A1",
            final_id="F1003_2",
        )
        approved_dossiers = DemographicRecodingDossierBuilder().build(approved, matching_evidence)
        education = next(item for item in approved_dossiers if item.target_variable == "F2003")
        education_map = {
            item["source_value"]: item["proposed_target_value"]
            for item in education.draft_recode_table
            if item.get("proposed_target_value") != ""
        }
        self.assertEqual(
            education_map,
            {
                "1": "2",
                "2": "3",
                "3": "3",
                "4": "4",
                "5": "5",
                "6": "6",
                "7": "6",
                "8": "7",
                "9": "8",
                "10": "9",
            },
        )
        generated = DemographicDataGenerator().generate(source_df, approved, election_year=2022)

        left = generated.assign(_key=pd.to_numeric(source_df["A1"], errors="coerce")).sort_values("_key").reset_index(drop=True)
        demo_vars = [item.target_variable for item in approved]
        right = final_df[["F1003_2", *demo_vars]].assign(
            _key=pd.to_numeric(final_df["F1003_2"], errors="coerce")
        ).sort_values("_key").reset_index(drop=True)

        self.assertTrue((left["_key"] == right["_key"]).all())
        for col in demo_vars:
            with self.subTest(col=col):
                actual = pd.to_numeric(left[col], errors="coerce").fillna(-999999999)
                expected = pd.to_numeric(right[col], errors="coerce").fillna(-999999999)
                self.assertTrue(((actual - expected).abs() < 1e-6).all())


if __name__ == "__main__":
    unittest.main()
