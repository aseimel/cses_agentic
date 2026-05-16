import json
import tempfile
import unittest
from pathlib import Path

from src.ingest.doc_parser import DocumentInfo, QuestionInfo
from src.matching.evidence import (
    MatchingEvidenceBuilder,
    QuestionnaireItemExtractor,
    RemoteItemSimilarityService,
    ResponseOptionMatcher,
    SourceVariableProfile,
    TargetVariableProfile,
    build_similarity_pairs,
)


class MatchingEvidenceTests(unittest.TestCase):
    def test_questionnaire_item_extraction_from_parsed_document(self):
        document = DocumentInfo(
            file_path=Path("questionnaire.pdf"),
            file_format="PDF",
            full_text="",
            questions=[
                QuestionInfo("Q01", "How interested are you in politics?", ["Very interested", "Not at all interested"]),
            ],
            is_questionnaire=True,
        )

        items = QuestionnaireItemExtractor().extract_from_document_info(document)

        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].item_id, "Q01")
        self.assertIn("interested", items[0].text)
        self.assertEqual(len(items[0].response_options), 2)

    def test_response_option_matcher_flags_missing_options(self):
        target = TargetVariableProfile(
            name="F3001",
            description="POLITICAL INTEREST",
            dependency_class="direct_survey_item",
            expected_response_structure="categorical",
        )
        source = SourceVariableProfile(name="Q01", label="Political interest")

        result = ResponseOptionMatcher().compare(target, source)

        self.assertLess(result["score"], 0.5)
        self.assertIn("response_options_missing", result["flags"])

    def test_remote_similarity_service_can_be_mocked_and_batched(self):
        class FakeResult:
            status = "ok"
            content = json.dumps({"scores": [{"id": "F3001||Q01", "score": 0.91, "reason": "same item"}]})

        class FakeRunner:
            def complete(self, *args, **kwargs):
                return FakeResult()

        service = RemoteItemSimilarityService(runner=FakeRunner(), enabled=True)
        scores = service.score_pairs(
            [
                {
                    "target_id": "F3001",
                    "source_id": "Q01",
                    "target_text": "political interest",
                    "source_text": "interest in politics",
                }
            ]
        )

        self.assertEqual(scores["F3001||Q01"], 0.91)

    def test_matching_evidence_builder_writes_profiles_without_local_ml_dependency(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            payload = MatchingEvidenceBuilder(study_dir).build_from_records(
                narrative_docs=[
                    {
                        "path": str(study_dir / "questionnaire.txt"),
                        "relative_path": "questionnaire.txt",
                        "parsed_questions": [
                            {"code": "Q01", "text": "How interested are you in politics?", "response_options": ["Very", "Not at all"]}
                        ],
                    }
                ],
                data_summaries=[
                    {
                        "path": str(study_dir / "data.csv"),
                        "relative_path": "data.csv",
                        "variable_inventory": [
                            {
                                "name": "Q01",
                                "label": "How interested are you in politics?",
                                "value_labels": {"1": "Very", "4": "Not at all"},
                                "sample_values": [1, 2, 3],
                                "dtype": "int",
                                "n_unique": 4,
                            }
                        ],
                    }
                ],
            )

            self.assertTrue((study_dir / ".cses" / "matching_evidence.json").exists())
            self.assertEqual(len(payload["questionnaire_items"]), 1)
            self.assertEqual(len(payload["source_variable_profiles"]), 1)
            self.assertGreaterEqual(len(payload["target_variable_profiles"]), 300)
            pairs = build_similarity_pairs(payload)
            self.assertTrue(any(pair["target_id"] == "F3001" and pair["source_id"] == "Q01" for pair in pairs))

    def test_no_local_ml_imports(self):
        source = Path("src/matching/evidence.py").read_text(encoding="utf-8")
        forbidden = ["import harmony", "from harmony", "sentence_transformers", "sentence-transformers", "transformers import"]
        for term in forbidden:
            with self.subTest(term=term):
                self.assertNotIn(term, source)


if __name__ == "__main__":
    unittest.main()
