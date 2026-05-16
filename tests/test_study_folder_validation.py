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


if __name__ == "__main__":
    unittest.main()
