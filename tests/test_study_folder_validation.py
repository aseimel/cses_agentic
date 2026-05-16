import tempfile
import unittest
from pathlib import Path

from src.workflow.organizer import FileOrganizer
from src.workflow.state import WorkflowState


class StudyFolderValidationTests(unittest.TestCase):
    def test_accepts_initialized_folder_itself(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            state = WorkflowState(working_dir=str(study_dir), country="Testland", country_code="TST", year="2024")
            state.save()

            check = FileOrganizer(study_dir).validate_study_folder()

            self.assertTrue(check.ok)
            self.assertTrue(check.initialized)

    def test_accepts_single_uninitialized_study_with_one_email_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            deposit = study_dir / "E-mails" / "20240101"
            deposit.mkdir(parents=True)
            (deposit / "survey.csv").write_text("id,q1\n1,1\n", encoding="utf-8")

            check = FileOrganizer(study_dir).validate_study_folder()

            self.assertTrue(check.ok)
            self.assertFalse(check.initialized)
            self.assertEqual(check.email_folder, study_dir / "E-mails")

    def test_rejects_parent_with_multiple_initialized_studies(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            for index in range(2):
                study_dir = parent / f"Study_{index}_2024"
                state = WorkflowState(working_dir=str(study_dir), country="Testland", country_code="TST", year="2024")
                state.save()

            check = FileOrganizer(parent).validate_study_folder()

            self.assertFalse(check.ok)
            self.assertIn("parent folder", check.message.lower())
            self.assertTrue(check.details)

    def test_rejects_parent_with_nested_study_email_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            parent = Path(folder)
            for index in range(2):
                deposit = parent / f"Study_{index}_2024" / "emails" / "20240101"
                deposit.mkdir(parents=True)
                (deposit / "survey.csv").write_text("id,q1\n1,1\n", encoding="utf-8")

            check = FileOrganizer(parent).validate_study_folder()

            self.assertFalse(check.ok)
            self.assertIn("parent folder", check.message.lower())

    def test_rejects_folder_with_multiple_email_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            for name in ("email", "E-mails"):
                deposit = study_dir / name / "20240101"
                deposit.mkdir(parents=True)
                (deposit / "survey.csv").write_text("id,q1\n1,1\n", encoding="utf-8")

            check = FileOrganizer(study_dir).validate_study_folder()

            self.assertFalse(check.ok)
            self.assertIn("more than one", check.message.lower())

    def test_initialization_structure_creates_election_and_district_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            FileOrganizer(study_dir).create_study_structure(study_dir)

            self.assertTrue((study_dir / "Election Results").is_dir())
            self.assertTrue((study_dir / "District Data").is_dir())
            self.assertTrue((study_dir / "micro" / "district data").is_dir())

    def test_deposit_election_and_district_files_are_curated_to_dedicated_folders(self):
        with tempfile.TemporaryDirectory() as folder:
            study_dir = Path(folder)
            deposit = study_dir / "E-mails" / "20240101"
            deposit.mkdir(parents=True)
            (deposit / "survey.csv").write_text("id,q1\n1,1\n", encoding="utf-8")
            (deposit / "questionnaire.pdf").write_text("questionnaire", encoding="utf-8")
            election_file = deposit / "Election results.xlsx"
            district_file = deposit / "District Data Template.xlsx"
            election_file.write_text("placeholder", encoding="utf-8")
            district_file.write_text("placeholder", encoding="utf-8")

            organizer = FileOrganizer(study_dir)
            detected = organizer.detect_files(source_dir=study_dir / "E-mails", recursive=True)
            organizer.create_study_structure(study_dir)
            mapping = organizer.copy_files_with_standard_names(detected, study_dir / "E-mails", study_dir, "TST", "2024")

            self.assertIn("election_results", mapping)
            self.assertIn("district_data", mapping)
            self.assertTrue(Path(mapping["election_results"]).is_relative_to(study_dir / "Election Results"))
            self.assertTrue(Path(mapping["district_data"]).is_relative_to(study_dir / "District Data"))


if __name__ == "__main__":
    unittest.main()
