import tempfile
import unittest
from pathlib import Path

from src.workflow.input_manifest import PrimaryInputSelector, _looks_like_generated_artifact
from src.study_kb import is_reference_or_historical_source


class InputManifestSelectionTests(unittest.TestCase):
    def test_deposited_variable_workbooks_are_not_primary_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            variable_list = root / "micro" / "micro" / "deposited variables-m6_CNT_YEAR_DATE.xlsx"
            survey = root / "micro" / "MEX_2021_data.sav"
            variable_list.parent.mkdir(parents=True)
            survey.parent.mkdir(parents=True, exist_ok=True)
            variable_list.write_text("variables", encoding="utf-8")
            survey.write_text("survey", encoding="utf-8")

            self.assertTrue(_looks_like_generated_artifact(variable_list))
            selected, _reason, _warnings = PrimaryInputSelector().select_data_file([variable_list, survey])

            self.assertEqual(selected, survey)

    def test_macro_data_folder_is_not_primary_micro_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            macro_data = root / "macro" / "MEX_2021_Micro+Macro.dta"
            survey = root / "micro" / "MEX_2021_data.sav"
            macro_data.parent.mkdir(parents=True)
            survey.parent.mkdir(parents=True, exist_ok=True)
            macro_data.write_text("macro", encoding="utf-8")
            survey.write_text("survey", encoding="utf-8")

            self.assertTrue(_looks_like_generated_artifact(macro_data))
            selected, _reason, _warnings = PrimaryInputSelector().select_data_file([macro_data, survey])

            self.assertEqual(selected, survey)

    def test_study_review_ignores_reference_outputs_and_old_modules(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ignored_paths = [
                root / "micro" / "FINAL dataset" / "cses-m6_micro_CNT_2024.dta",
                root / "micro" / "Documentation" / "cses-m6_log-file_CNT_2024.docx",
                root / "macro" / "Comparison with Module 5" / "module5_notes.pdf",
                root / "macro" / "old" / "old_macro_report.pdf",
                root / "micro" / "Labels" / "cses-m6_label_update.do",
            ]
            kept_path = root / "micro" / "CNT_2024_design_report.pdf"

            for path in [*ignored_paths, kept_path]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("placeholder", encoding="utf-8")

            for path in ignored_paths:
                self.assertTrue(is_reference_or_historical_source(path, root))
            self.assertFalse(is_reference_or_historical_source(kept_path, root))


if __name__ == "__main__":
    unittest.main()
