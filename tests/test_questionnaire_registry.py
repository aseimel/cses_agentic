import json
import tempfile
import unittest
from pathlib import Path

from src.standards.questionnaire_registry import (
    Module6QuestionnaireDistiller,
    Module6QuestionnaireRegistry,
    audit_questionnaire_registry,
    canonical_item_id,
    normalize_item_key,
)


class QuestionnaireRegistryTests(unittest.TestCase):
    def test_normalizes_module6_item_ids(self):
        self.assertEqual(canonical_item_id("Q1"), "Q01")
        self.assertEqual(canonical_item_id("D2"), "D02")
        self.assertEqual(canonical_item_id("A4a"), "A4a")
        self.assertEqual(normalize_item_key("Q10LH-b"), "Q10LHB")
        self.assertEqual(normalize_item_key("Q10LH_b"), "Q10LHB")

    def test_distills_required_sections_from_questionnaire_text(self):
        source = Path("CSES_Module6_Questionnaire.txt")
        items = Module6QuestionnaireDistiller().parse_text(source.read_text(encoding="utf-8", errors="replace"))
        by_id = {item.item_id: item for item in items}

        self.assertIn("A5", by_id)
        self.assertIn("Q01", by_id)
        self.assertIn("Q10LH-b", by_id)
        self.assertIn("D02", by_id)
        self.assertEqual(by_id["Q01"].section, "cses_module")
        self.assertEqual(by_id["D02"].item_type, "demographic_coding_standard")
        self.assertTrue(by_id["Q01"].response_options)
        self.assertIn("F3001", by_id["Q01"].target_variables)

    def test_runtime_registry_loads_without_raw_questionnaire_file(self):
        with tempfile.TemporaryDirectory() as folder:
            wiki_root = Path(folder) / "cses_wiki"
            output = wiki_root / "patterns" / "module6_questionnaire_registry.json"
            payload = {
                "schema_version": 1,
                "items": [
                    {
                        "item_id": "Q01",
                        "normalized_id": "Q01",
                        "aliases": ["Q1"],
                        "section": "cses_module",
                        "item_type": "core_question",
                        "title": "POLITICAL INTEREST",
                        "text": "How interested are you in politics?",
                        "notes": "",
                        "help_text": "",
                        "response_options": [{"code": "1", "label": "VERY INTERESTED"}],
                        "missing_codes": [],
                        "target_variables": ["F3001"],
                        "source_line": 1,
                    }
                ],
            }
            output.parent.mkdir(parents=True)
            output.write_text(json.dumps(payload), encoding="utf-8")

            registry = Module6QuestionnaireRegistry(wiki_root)

            self.assertTrue(registry.exists())
            self.assertEqual(registry.get("Q1").item_id, "Q01")
            self.assertEqual(registry.item_ids_for_target("F3001")[0], "Q01")

    def test_generated_registry_audit_passes(self):
        self.assertEqual(audit_questionnaire_registry(), [])

    def test_no_local_ml_dependency_in_registry_or_matching(self):
        text = "\n".join(
            Path(path).read_text(encoding="utf-8")
            for path in ["src/standards/questionnaire_registry.py", "src/matching/evidence.py"]
        )
        for forbidden in ["sentence_transformers", "sentence-transformers", "from harmony", "import harmony", "transformers import"]:
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
