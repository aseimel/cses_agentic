import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.benchmark import BenchmarkDecisionExtractor, DocumentationComparator


class DocumentationBenchmarkTests(unittest.TestCase):
    def test_documentation_comparator_checks_required_cses_sections_and_facts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            micro = root / "micro"
            doc_dir = micro / "Documentation"
            checks = micro / "data_checks"
            labels = micro / "labels"
            doc_dir.mkdir(parents=True)
            checks.mkdir()
            labels.mkdir()
            (micro / "TST_2024_log.qmd").write_text(
                "\n".join(
                    [
                        "## Log File Notes",
                        "### Processing Notes",
                        "## Questions for Collaborator",
                        "## Things To Do Before Releasing",
                        "## Election Study Notes and Appendices",
                        "### Election Summary",
                        "### Overview of Study Design and Weights",
                        "Probability sample. Sample design. Sample size. Response rate.",
                        "Fieldwork. Mode of interview. Weighting.",
                        "### Parties and Leaders",
                        "Party order: Party A and Party B. District data reviewed. Validation check results.",
                    ]
                ),
                encoding="utf-8",
            )
            (doc_dir / "ESN - Test 2024.txt").write_text("Election Study Notes\nParties and Leaders\n", encoding="utf-8")
            (checks / "validation.do").write_text("* check", encoding="utf-8")
            (checks / "validation.log").write_text("ok", encoding="utf-8")
            (labels / "labels.do").write_text("* labels", encoding="utf-8")

            result = DocumentationComparator().compare(root)

            self.assertTrue(result["ok"], result["issues"])
            self.assertTrue(result["section_checks"]["study_design_weights"])
            self.assertTrue(result["fact_checks"]["party_order"])

    def test_documentation_comparator_rejects_skeletal_reference_replication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "generated"
            reference = Path(folder) / "reference"
            (root / "micro" / "Documentation").mkdir(parents=True)
            (root / "micro" / "data_checks").mkdir()
            (root / "micro" / "labels").mkdir()
            (reference / "micro" / "Collaborator Questions").mkdir(parents=True)
            (reference / "macro").mkdir()
            (root / "micro" / "TST_2024_log.qmd").write_text(
                "Log File Notes\nQuestions for Collaborator\nThings To Do Before Releasing\n"
                "Election Study Notes\nElection Summary\nStudy Design Weights\nParties and Leaders\n"
                "Probability sample. Sample design. Sample size. Response rate. Fieldwork. Mode. Weight. Party order. District. Check.\n",
                encoding="utf-8",
            )
            (root / "micro" / "Documentation" / "ESN - Test 2024.txt").write_text("Election Study Notes\nParty A\n", encoding="utf-8")
            (root / "micro" / "data_checks" / "validation.do").write_text("* check", encoding="utf-8")
            (root / "micro" / "data_checks" / "validation.log").write_text("ok", encoding="utf-8")
            (root / "micro" / "labels" / "labels.do").write_text("* labels", encoding="utf-8")
            long_reference = "\n".join(
                f"F30{i:02d} Party A Q{i} 2024-01-{(i % 28) + 1:02d} detailed processing note for reference documentation."
                for i in range(1, 80)
            )
            (reference / "micro" / "cses-m6_log-file_TST_2024.txt").write_text(long_reference, encoding="utf-8")
            (reference / "micro" / "Collaborator Questions" / "questions.txt").write_text(long_reference, encoding="utf-8")
            (reference / "macro" / "Test_ESN.txt").write_text(long_reference, encoding="utf-8")

            result = DocumentationComparator().compare(root, reference)

            self.assertFalse(result["ok"])
            self.assertFalse(result["equivalence"]["processing_log"]["ok"])
            self.assertIn("generated text is much shorter than the reference", result["equivalence"]["processing_log"]["issues"])

    def test_documentation_comparator_accepts_comparable_reference_detail(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "generated"
            reference = Path(folder) / "reference"
            (root / "micro" / "Documentation").mkdir(parents=True)
            (root / "micro" / "Collaborator Questions").mkdir()
            (root / "micro" / "data_checks").mkdir()
            (root / "micro" / "labels").mkdir()
            (reference / "micro" / "Collaborator Questions").mkdir(parents=True)
            (reference / "macro").mkdir()
            body = "\n".join(
                f"## Section {i}\nF30{i:02d} Party A Q{i} 2024-01-{(i % 28) + 1:02d} detailed processing note."
                for i in range(1, 80)
            )
            required = (
                "Log File Notes\nQuestions for Collaborator\nThings To Do Before Releasing\n"
                "Election Study Notes\nElection Summary\nStudy Design Weights\nParties and Leaders\n"
                "Probability sample. Sample design. Sample size. Response rate. Fieldwork. Mode. Weighting. Party order. District data. Validation check.\n"
            )
            (root / "micro" / "cses-m6_log-file_TST_2024.txt").write_text(required + body, encoding="utf-8")
            (root / "micro" / "Collaborator Questions" / "questions.md").write_text(body, encoding="utf-8")
            (root / "micro" / "Documentation" / "ESN - Test 2024.txt").write_text(required + body, encoding="utf-8")
            (root / "micro" / "data_checks" / "validation.do").write_text("* check", encoding="utf-8")
            (root / "micro" / "data_checks" / "validation.log").write_text("ok", encoding="utf-8")
            (root / "micro" / "labels" / "labels.do").write_text("* labels", encoding="utf-8")
            (reference / "micro" / "cses-m6_log-file_TST_2024.txt").write_text(required + body, encoding="utf-8")
            (reference / "micro" / "Collaborator Questions" / "questions.txt").write_text(body, encoding="utf-8")
            (reference / "macro" / "Test_ESN.txt").write_text(required + body, encoding="utf-8")

            result = DocumentationComparator().compare(root, reference)

            self.assertTrue(result["ok"], result["issues"])
            self.assertTrue(result["equivalence"]["processing_log"]["ok"])

    def test_benchmark_decision_extractor_writes_constant_values_and_syntax_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            reference_dataset = root / "reference.dta"
            reference_syntax = root / "reference.do"
            pd.DataFrame({"F1001": ["CSES-MODULE-6", "CSES-MODULE-6"], "F3001": [1, 2]}).to_stata(reference_dataset, write_index=False)
            reference_syntax.write_text(
                "**>>> F1001 DATASET\n"
                "gen F1001 = \"CSES-MODULE-6\"\n"
                "**>>> F3001 TEST\n"
                "gen F3001 = Q01\n",
                encoding="utf-8",
            )

            payload = BenchmarkDecisionExtractor().extract(
                reference_dataset=reference_dataset,
                reference_syntax=reference_syntax,
                working_dir=root,
            )

            self.assertEqual(payload["constant_values"]["F1001"], "CSES-MODULE-6")
            self.assertIn("F3001", payload["syntax_variable_blocks"])
            self.assertTrue((root / ".cses" / "benchmark_decision_replay.json").exists())


if __name__ == "__main__":
    unittest.main()
