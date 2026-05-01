import tempfile
import unittest
from pathlib import Path

from src.matching.decision_engine import MatchingDecisionEngine, matching_category_summary
from src.matching.party_order import (
    ElectionResultsWorkbookParser,
    PartyOrderingRulesEngine,
    election_results_intake_summary,
    infer_election_context_from_macro_material,
    party_order_message,
)


class PartyOrderWorkflowTests(unittest.TestCase):
    def _write_workbook(self, path: Path) -> None:
        import openpyxl

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Election Results"
        ws.cell(row=1, column=1, value="TESTLAND: 2024")
        ws.cell(row=3, column=1, value="Lower House")
        headers = ["Party label", "Numeric", "Party Name", "Votes", "% of Vote", "Seats", "% of Seats"]
        for index, header in enumerate(headers, start=1):
            ws.cell(row=5, column=index, value=header)
        rows = [
            ["PARTY A", 999001, "First Party", 1000, 40.0, 40, 40.0],
            ["PARTY B", 999002, "Second Party", 800, 32.0, 32, 32.0],
            ["PARTY C", 999003, "Third Party", 500, 20.0, 20, 20.0],
            ["", "", "Others", 200, 8.0, 8, 8.0],
        ]
        for row_index, row in enumerate(rows, start=6):
            for col_index, value in enumerate(row, start=1):
                ws.cell(row=row_index, column=col_index, value=value)
        wb.save(path)

    def test_parser_reads_standardized_election_results_workbook(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            election_dir = study_dir / "Election Results"
            election_dir.mkdir()
            path = election_dir / "election_results.xlsx"
            self._write_workbook(path)

            tables = ElectionResultsWorkbookParser().parse_directory(study_dir)

            self.assertEqual(len(tables), 1)
            self.assertEqual(tables[0].election_context, "lower_house")
            self.assertEqual([party.code_letter for party in tables[0].parties[:3]], ["A", "B", "C"])
            self.assertEqual(tables[0].parties[0].numeric_code, "999001")

    def test_party_order_proposal_requires_joint_agreement_before_matching(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            election_dir = study_dir / "Election Results"
            election_dir.mkdir()
            self._write_workbook(election_dir / "election_results.xlsx")

            tables = ElectionResultsWorkbookParser().parse_directory(study_dir)
            engine = PartyOrderingRulesEngine()
            proposal = engine.propose(tables, source_variables=["Q16a", "Q17a", "Q10LHb"])
            review_path, decision_path = engine.write_review(study_dir, proposal)

            self.assertTrue(review_path.exists())
            self.assertTrue(decision_path.exists())
            self.assertFalse(engine.is_approved(study_dir))
            self.assertEqual(proposal.status, "agreement_needed")
            self.assertEqual([party.party_name for party in proposal.proposed_parties[:3]], ["First Party", "Second Party", "Third Party"])
            self.assertIn("Micro processor and macro coder", party_order_message(proposal))

    def test_matching_engine_gates_party_variables_without_approved_order(self):
        decisions = MatchingDecisionEngine().decide(
            source_contexts=[{"name": "Q16a", "description": "Like-dislike Party A"}],
            matching_evidence={
                "target_variable_profiles": [
                    {"name": "F3018_A", "canonical_item_type": "party_vote_item", "section": "cses_module"}
                ]
            },
            party_order_approved=False,
            party_order_summary={"party_count": 3},
        )
        by_target = {item.target_variable: item for item in decisions}

        self.assertEqual(by_target["F3018_A"].status, "awaiting_party_order_agreement")
        summary = matching_category_summary(decisions, {"target_variable_profiles": [{"name": "F3018_A", "canonical_item_type": "party_vote_item"}]})
        self.assertGreater(summary["party_election_items"]["awaiting_party_ordering"], 0)

    def test_sweden_reference_workbook_party_order_is_reproduced_when_available(self):
        root = Path.cwd()
        workbook = root / "Sweden_2022" / "Election Results" / "SWE_2022_Election results.xlsx"
        if not workbook.exists():
            self.skipTest("Sweden election-results workbook is not available in this checkout")

        tables = ElectionResultsWorkbookParser().parse_file(workbook)
        proposal = PartyOrderingRulesEngine().propose(
            tables,
            context_hint=infer_election_context_from_macro_material(root / "Sweden_2022"),
        )
        observed = [(party.code_letter, party.numeric_code, party.party_name) for party in proposal.proposed_parties[:8]]

        self.assertEqual([item[0] for item in observed], list("ABCDEFGH"))
        self.assertEqual([item[1] for item in observed], ["752001", "752002", "752003", "752004", "752005", "752006", "752007", "752008"])
        self.assertIn("Social Democratic Party", observed[0][2])
        self.assertEqual(proposal.selected_context, "lower_house")

    def test_intake_summary_detects_standardized_tables(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            election_dir = study_dir / "Election Results"
            election_dir.mkdir()
            self._write_workbook(election_dir / "election_results.xlsx")

            summary = election_results_intake_summary(study_dir)

            self.assertEqual(summary["table_count"], 1)
            self.assertTrue(summary["standardized"])


if __name__ == "__main__":
    unittest.main()
