import json
import tempfile
import unittest
from pathlib import Path

from src.matching.macro_context import (
    AutoMacroProvider,
    MacroContextBuilder,
    approved_macro_context_values,
    is_macro_context_approved,
)
from src.matching.party_metadata import PartyMetadataReviewBuilder
from src.matching.party_order import ElectionResultParty, PartyOrderProposal
from src.workflow.state import WorkflowState


class MacroContextTests(unittest.TestCase):
    def _write_party_order_decision(self, study_dir: Path) -> PartyOrderProposal:
        cses_dir = study_dir / ".cses"
        cses_dir.mkdir(parents=True, exist_ok=True)
        parties = [
            ElectionResultParty(code_letter="A", numeric_code="999001", party_name="First Party", vote_share=40),
            ElectionResultParty(code_letter="B", numeric_code="999002", party_name="Second Party", vote_share=30),
        ]
        proposal = PartyOrderProposal(
            status="agreement_needed",
            selected_context="lower_house",
            ordering_rule="lower_house_vote_share",
            ordering_basis="lower-house vote share",
            source_file=str(study_dir / "Election Results.xlsx"),
            proposed_parties=parties,
        )
        payload = {
            "proposal": proposal.to_dict(),
            "approval": {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
            },
        }
        (cses_dir / "party_order_decision.json").write_text(json.dumps(payload), encoding="utf-8")
        return proposal

    def _write_macro_workbook(self, study_dir: Path) -> None:
        import openpyxl

        macro_dir = study_dir / "macro"
        macro_dir.mkdir()
        workbook = openpyxl.Workbook()
        sheet = workbook.active
        sheet.title = "MACRO DATA TEMPLATE"
        for row in [
            ["F5028_A. EXPERT: IDEOLOGICAL FAMILY - PARTY A", "", "gen", "F5028_A", "=", "4"],
            ["F5028_B. EXPERT: IDEOLOGICAL FAMILY - PARTY B", "", "gen", "F5028_B", "=", "10"],
            ["F5029_A. EXPERT: LEFT-RIGHT - PARTY A", "", "gen", "F5029_A", "=", "3"],
            ["F5029_B. EXPERT: LEFT-RIGHT - PARTY B", "", "gen", "F5029_B", "=", "6"],
        ]:
            sheet.append(row)
        workbook.save(macro_dir / "Macro Data.xlsx")

    def test_macro_context_prefers_deposited_macro_values(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            proposal = self._write_party_order_decision(study_dir)
            self._write_macro_workbook(study_dir)
            metadata_review = PartyMetadataReviewBuilder(study_dir).build()
            state = WorkflowState(country_code="TST", year="2024", working_dir=str(study_dir))

            review = MacroContextBuilder(study_dir, state).build(
                party_order_proposal=proposal,
                party_order_approved=True,
                party_metadata_review=metadata_review,
                party_metadata_approved=True,
            )

            by_id = {item.requirement_id: item for item in review.items}
            self.assertEqual(by_id["cses_left_right"].source_type, "macro_workbook")
            self.assertEqual(by_id["cses_left_right"].proposed_value["A"], "3")
            self.assertEqual(by_id["party_order"].source_type, "election_results_workbook")

    def test_macro_context_requires_joint_approval(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            proposal = self._write_party_order_decision(study_dir)
            self._write_macro_workbook(study_dir)
            metadata_review = PartyMetadataReviewBuilder(study_dir).build()
            state = WorkflowState(country_code="TST", year="2024", working_dir=str(study_dir))
            builder = MacroContextBuilder(study_dir, state)
            review = builder.build(proposal, True, metadata_review, True)
            _review_path, decision_path = builder.write_review(review)

            self.assertFalse(is_macro_context_approved(study_dir))
            self.assertEqual(approved_macro_context_values(study_dir), {})

            payload = json.loads(decision_path.read_text(encoding="utf-8"))
            payload["approval"] = {
                "micro_processor_approved": True,
                "macro_coder_approved": True,
                "locked": True,
            }
            decision_path.write_text(json.dumps(payload), encoding="utf-8")

            self.assertTrue(is_macro_context_approved(study_dir))
            values = approved_macro_context_values(study_dir)
            self.assertEqual(values["election_context"], "lower_house")
            self.assertIn("cses_ideological_family", values)

    def test_auto_macro_provider_records_versioned_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            auto = root / "auto_macro"
            output = auto / "output" / "TST_2024"
            scripts = auto / "scripts" / "r_downloader"
            output.mkdir(parents=True)
            scripts.mkdir(parents=True)
            (scripts / "generate_macro.R").write_text("print('generate')", encoding="utf-8")
            (scripts / "country_codes.yaml").write_text("TST: Testland", encoding="utf-8")
            (scripts / "variable_definitions.yaml").write_text("# VERSION 2 (2026-04-21)\n", encoding="utf-8")
            (output / "TST_2024_macro.csv").write_text(
                "country_code,election_year,election_date,F5068\nTST,2024,2024-01-02,0.75\n",
                encoding="utf-8",
            )
            state = WorkflowState(country_code="TST", year="2024", working_dir=str(root / "study"))

            provider = AutoMacroProvider(repo_root=root, auto_macro_dir=auto).collect(state)

            self.assertEqual(provider["status"], "output_loaded")
            self.assertEqual(provider["source_version"], "VERSION 2 (2026-04-21)")
            self.assertEqual(provider["values"]["election_date"], "2024-01-02")
            self.assertTrue(provider["variable_definitions_hash"])

    def test_conflicting_context_values_are_surfaced(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            proposal = self._write_party_order_decision(study_dir)
            metadata_review = PartyMetadataReviewBuilder(study_dir).build()
            state = WorkflowState(country_code="TST", year="2024", working_dir=str(study_dir))
            builder = MacroContextBuilder(study_dir, state)
            builder.auto_provider.collect = lambda _state: {
                "status": "output_loaded",
                "values": {"election_date": "2024-01-02"},
                "output_path": "auto.csv",
            }

            review = builder.build(proposal, True, metadata_review, False)

            by_id = {item.requirement_id: item for item in review.items}
            self.assertEqual(by_id["election_date"].proposed_value, "2024-01-02")
            self.assertIn("cses_left_right", review.missing_items)


if __name__ == "__main__":
    unittest.main()
