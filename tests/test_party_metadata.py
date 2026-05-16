import json
import tempfile
import unittest
from pathlib import Path

from src.codegen.party_recoding import PartyRecodingPlanBuilder
from src.matching.macro_context import MacroContextBuilder
from src.matching.party_metadata import PartyMetadataReviewBuilder, approved_party_metadata_values
from src.matching.party_order import ElectionResultParty, PartyOrderProposal
from src.standards.schema import SchemaRegistry
from src.workflow.state import WorkflowState


class PartyMetadataTests(unittest.TestCase):
    def _write_party_order_decision(self, study_dir: Path) -> None:
        cses_dir = study_dir / ".cses"
        cses_dir.mkdir(parents=True)
        parties = [
            {"code_letter": "A", "numeric_code": "999001", "party_name": "First Party"},
            {"code_letter": "B", "numeric_code": "999002", "party_name": "Second Party"},
        ]
        payload = {
            "proposal": {
                "selected_context": "lower_house",
                "proposed_parties": parties,
            },
            "approval": {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
            },
        }
        (cses_dir / "party_order_decision.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_macro_workbook(self, study_dir: Path) -> None:
        import openpyxl

        macro_dir = study_dir / "macro"
        macro_dir.mkdir()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "MACRO DATA TEMPLATE"
        rows = [
            ["F5028_A. EXPERT: IDEOLOGICAL FAMILY - PARTY A", "", "gen", "F5028_A", "=", "4"],
            ["F5028_B. EXPERT: IDEOLOGICAL FAMILY - PARTY B", "", "gen", "F5028_B", "=", "10"],
            ["F5029_A. EXPERT: LEFT-RIGHT - PARTY A", "", "gen", "F5029_A", "=", "3"],
            ["F5029_B. EXPERT: LEFT-RIGHT - PARTY B", "", "gen", "F5029_B", "=", "6"],
            ["F5200_A. MARPOR IDENTIFIER - PARTY A", "", "gen", "F5200_A", "=", "12345"],
        ]
        for row in rows:
            sheet.append(row)
        workbook.save(macro_dir / "Macro Data.xlsx")

    def _approve_macro_context(self, study_dir: Path, review) -> None:
        proposal = PartyOrderProposal(
            status="agreement_needed",
            selected_context="lower_house",
            ordering_rule="lower_house_vote_share",
            ordering_basis="Lower-house vote share",
            source_file=str(study_dir / "Election Results.xlsx"),
            proposed_parties=[
                ElectionResultParty(code_letter="A", numeric_code="999001", party_name="First Party", vote_share=40),
                ElectionResultParty(code_letter="B", numeric_code="999002", party_name="Second Party", vote_share=30),
            ],
        )
        state = WorkflowState(country_code="TST", year="2024", working_dir=str(study_dir))
        builder = MacroContextBuilder(study_dir, state)
        context = builder.build(
            party_order_proposal=proposal,
            party_order_approved=True,
            party_metadata_review=review,
            party_metadata_approved=True,
        )
        _review_path, decision_path = builder.write_review(context)
        payload = json.loads(decision_path.read_text(encoding="utf-8"))
        payload["approval"] = {
            "micro_processor_approved": True,
            "macro_coder_approved": True,
            "locked": True,
            "override_reason": "Test approval.",
        }
        decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def test_party_metadata_parser_reads_standardized_macro_workbook(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_order_decision(study_dir)
            self._write_macro_workbook(study_dir)

            builder = PartyMetadataReviewBuilder(study_dir)
            review = builder.build()
            _review_path, decision_path = builder.write_review(review)

            self.assertEqual(review.status, "ready_for_review")
            values = {item.variable: item.value for item in review.values}
            self.assertEqual(values["F5028_A"], "4")
            self.assertEqual(values["F5029_B"], "6")
            self.assertTrue(decision_path.exists())

    def test_approved_party_metadata_values_require_joint_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_order_decision(study_dir)
            self._write_macro_workbook(study_dir)
            builder = PartyMetadataReviewBuilder(study_dir)
            review = builder.build()
            _review_path, decision_path = builder.write_review(review)

            self.assertEqual(approved_party_metadata_values(study_dir), {})
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            payload["approval"] = {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
            }
            decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            values = approved_party_metadata_values(study_dir)
            self.assertEqual(values["F5028_A"], "4")
            self.assertEqual(values["F5029_A"], "3")

    def test_party_context_derivation_uses_approved_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_order_decision(study_dir)
            self._write_macro_workbook(study_dir)
            builder = PartyMetadataReviewBuilder(study_dir)
            review = builder.build()
            _review_path, decision_path = builder.write_review(review)
            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            payload["approval"] = {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
            }
            decision_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._approve_macro_context(study_dir, review)

            recoder = PartyRecodingPlanBuilder(study_dir)
            linked = recoder._party_context_map("F3100_LR_CSES")
            collapsed = recoder._party_context_map("F3011_IF_CSES")
            lr_collapsed = recoder._party_context_map("F3011_LR_CSES")

            self.assertTrue(linked.approved)
            self.assertIn("replace F3100_LR_CSES = 3 if F3011_LH_PL == 999001", linked.custom_stata_lines)
            self.assertTrue(collapsed.approved)
            self.assertIn("replace F3011_IF_CSES = 3 if F3011_LH_PL == 999001", collapsed.custom_stata_lines)
            self.assertIn("replace F3011_LR_CSES = 2 if F3011_LH_PL == 999002", lr_collapsed.custom_stata_lines)

    def test_sweden_macro_workbook_party_metadata_is_detected_when_available(self):
        workbook = Path.cwd() / "Sweden_2022" / "macro" / "SWE_2022_M6 Macro Data_20251229.xlsx"
        if not workbook.exists():
            self.skipTest("Sweden macro workbook is not available in this checkout")
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            self._write_party_order_decision(study_dir)
            macro_dir = study_dir / "macro"
            macro_dir.mkdir(exist_ok=True)
            import shutil

            shutil.copy2(workbook, macro_dir / workbook.name)
            review = PartyMetadataReviewBuilder(study_dir).build()
            values = {item.variable: item.value for item in review.values}

            self.assertEqual(values["F5028_A"], "4")
            self.assertEqual(values["F5028_H"], "6")
            self.assertEqual(values["F5029_A"], "3")
            self.assertEqual(values["F5029_H"], "6")

    def test_party_context_variables_are_schema_derivatives(self):
        registry = SchemaRegistry()
        for name in ["F3010", "F3010_TS", "F3011_VS_1", "F3011_LR_CSES", "F3100_IF_CSES"]:
            with self.subTest(name=name):
                item = registry.by_name(name)
                self.assertEqual(item.dependency_class, "macro_or_party_input")
                self.assertEqual(item.recode_strategy, "party_context_derivative")


if __name__ == "__main__":
    unittest.main()
