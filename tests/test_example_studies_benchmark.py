import tempfile
import unittest
from pathlib import Path

from src.example_studies_benchmark import discover_reference_studies, resolve_reference_artifacts


class ExampleStudiesBenchmarkTests(unittest.TestCase):
    def test_resolver_excludes_old_module5_final_dataset(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder) / "Testland_2024"
            old_final = study / "micro" / "Old 2019" / "FINAL dataset"
            old_final.mkdir(parents=True)
            (old_final / "cses-m5_micro_TST_2019_20240101.dta").write_text("old", encoding="utf-8")

            self.assertIsNone(resolve_reference_artifacts(study))

    def test_resolver_detects_current_module6_final_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder) / "Testland_2024"
            final_dir = study / "micro" / "FINAL dataset"
            final_dir.mkdir(parents=True)
            (final_dir / "cses-m6_micro_TST_2024_20250101.dta").write_text("final", encoding="utf-8")
            (study / "micro").mkdir(exist_ok=True)
            (study / "micro" / "cses-m6_micro_TST_2024_20241201.do").write_text("do", encoding="utf-8")
            (study / "micro" / "cses-m6_log-file_TST_2024_20241201.doc").write_text("log", encoding="utf-8")

            reference = resolve_reference_artifacts(study)

            self.assertIsNotNone(reference)
            self.assertEqual(reference.year, "2024")
            self.assertEqual(reference.artifacts["final_micro_dataset"], "micro/FINAL dataset/cses-m6_micro_TST_2024_20250101.dta")
            self.assertEqual(reference.artifacts["reference_micro_syntax"], "micro/cses-m6_micro_TST_2024_20241201.do")

    def test_discovery_is_generic_and_ignores_incomplete_studies(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            complete = root / "Complete_2024"
            incomplete = root / "Incomplete_2024"
            (complete / "micro" / "FINAL dataset").mkdir(parents=True)
            (complete / "micro" / "FINAL dataset" / "cses-m6_micro_CMP_2024_20250101.dta").write_text("final", encoding="utf-8")
            incomplete.mkdir()

            references = discover_reference_studies(root)

            self.assertEqual([item.study_id for item in references], ["Complete_2024"])

    def test_current_micro_fallback_is_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            study = Path(folder) / "Fallback_2024"
            (study / "micro").mkdir(parents=True)
            (study / "micro" / "cses-m6_micro_FBK_2024_20250101.dta").write_text("current", encoding="utf-8")

            self.assertIsNone(resolve_reference_artifacts(study, selection="final_only"))
            reference = resolve_reference_artifacts(study, selection="final_or_current_micro")
            self.assertIsNotNone(reference)
            self.assertIn("current micro output", " ".join(reference.selection_notes))


if __name__ == "__main__":
    unittest.main()
