"""
Active Logging for CSES Workflow.

Provides real-time logging following CSES naming conventions:
- Log file: micro/cses-m6_log-file_{CODE}_{YEAR}_{DATE}.txt (plain text)
- Questions file: micro/Collaborator Questions/{Country}_{Year}_micro_collaborator_questions_{DATE}.docx (Word)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from docx import Document
from docx.shared import Pt, RGBColor

if TYPE_CHECKING:
    from .state import WorkflowState

logger = logging.getLogger(__name__)


class ActiveLogger:
    """
    Handles real-time logging to CSES-standard Word documents.

    Creates and maintains:
    - Log file with workflow progress, issues, and questions
    - Separate collaborator questions file for easy sharing
    """

    def __init__(self, state: "WorkflowState"):
        """
        Initialize the active logger.

        Args:
            state: Current workflow state with country/year info
        """
        self.state = state
        self.working_dir = Path(state.working_dir) if state.working_dir else Path.cwd()

        # File paths
        self.log_file_path: Optional[Path] = None
        self.questions_file_path: Optional[Path] = None

        # Log entry counter (for descending numbering)
        self._log_entry_count = 0

        # Initialize files if state has country/year
        if state.country and state.year:
            self._initialize_files()

    def _initialize_files(self):
        """Create log and questions files if they don't exist."""
        # Skip if working directory doesn't exist
        if not self.working_dir.exists():
            return

        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        date_str = datetime.now().strftime("%Y%m%d")

        # Ensure micro folder exists
        micro_dir = self.working_dir / "micro"
        micro_dir.mkdir(parents=True, exist_ok=True)

        # Log file path (plain text)
        log_filename = f"cses-m6_log-file_{country_code}_{year}_{date_str}.txt"
        self.log_file_path = micro_dir / log_filename

        # Questions file path
        questions_dir = micro_dir / "Collaborator Questions"
        questions_dir.mkdir(exist_ok=True)
        questions_filename = f"{self.state.country}_{year}_micro_collaborator_questions_{date_str}.docx"
        self.questions_file_path = questions_dir / questions_filename

        # Create log file if it doesn't exist
        if not self.log_file_path.exists():
            self._create_log_file()
            print(f"Created log file: micro/{log_filename}")

        # Create questions file if it doesn't exist
        if not self.questions_file_path.exists():
            self._create_questions_file()
            print(f"Created questions file: micro/Collaborator Questions/{questions_filename}")

        # Update state with file paths
        self.state.log_file = str(self.log_file_path)
        self.state.collaborator_questions_file = str(self.questions_file_path)

    def _create_log_file(self):
        """Create a new log file with CSES standard structure (plain text)."""
        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        country = self.state.country
        date_str = datetime.now().strftime("%Y-%m-%d")

        lines = [
            f"{country_code}_{year}_Mod6",
            "",
            f"Name: [PROCESSOR NAME]",
            f"Date: {date_str}",
            "",
            f"Country: {country}",
            f"Election Year: {year}",
            f"Most recent update: {date_str}",
            "",
            "",
            "=" * 75,
            ">>> Log File Instructions",
            "=" * 75,
            "",
            "This Log File should be used to document any questions, issues, and",
            "challenges that arose while processing the Individual Datasets for",
            f"inclusion in CSES M6.",
            "Navigation: Section headings can be searched using \">>>\".",
            "Subsections can be searched using \"<<>>\".",
            "",
            f">>> Log File Notes: {country_code}_{year}_M6",
            f">>> Questions for Collaborator: {country_code}_{year}_M6",
            f">>> Things To Do Before Releasing the Data: {country_code}_{year}_M6",
            f">>> Election Study Notes and Appendices: {country_code}_{year}_M6",
            f"    <<>> ELECTION SUMMARY - {country_code}_{year}_M6",
            f"    <<>> OVERVIEW OF STUDY DESIGN AND WEIGHTS - {country_code}_{year}_M6",
            f"    <<>> PARTIES AND LEADERS: {country_code}_{year}_M6",
            "",
            "=" * 75,
            f">>> Log File Notes: {country_code}_{year}_M6",
            "=" * 75,
            "",
            "Deposited Files:",
            "",
            "",
            "=" * 75,
            f">>> Questions for Collaborator: {country_code}_{year}_M6",
            "=" * 75,
            "",
            "",
            "=" * 75,
            f">>> Things To Do Before Releasing the Data: {country_code}_{year}_M6",
            "=" * 75,
            "",
            "",
            "=" * 75,
            f">>> Election Study Notes and Appendices: {country_code}_{year}_M6",
            "=" * 75,
            "",
            f"<<>> ELECTION SUMMARY - {country_code}_{year}_M6:",
            "",
            "",
            f"<<>> OVERVIEW OF STUDY DESIGN AND WEIGHTS - {country_code}_{year}_M6:",
            "",
            "",
            f"<<>> PARTIES AND LEADERS: {country_code}_{year}_M6",
            "",
            "",
        ]

        self.log_file_path.write_text("\n".join(lines))

    def _create_questions_file(self):
        """Create a new collaborator questions file."""
        doc = Document()

        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        country = self.state.country
        date_str = datetime.now().strftime("%Y-%m-%d")

        # Title
        title = doc.add_paragraph()
        title_run = title.add_run(f"Collaborator Questions - {country} {year}")
        title_run.bold = True
        title_run.font.size = Pt(16)

        # Metadata
        doc.add_paragraph(f"Study: {country_code}_{year}_M6")
        doc.add_paragraph(f"Generated: {date_str}")
        doc.add_paragraph(f"Processor: [NAME]")
        doc.add_paragraph()

        # Pending Questions section
        doc.add_paragraph("=" * 47)
        pending_header = doc.add_paragraph()
        pending_run = pending_header.add_run("PENDING QUESTIONS")
        pending_run.bold = True
        doc.add_paragraph("=" * 47)
        doc.add_paragraph()

        # Resolved Questions section
        doc.add_paragraph("=" * 47)
        resolved_header = doc.add_paragraph()
        resolved_run = resolved_header.add_run("RESOLVED QUESTIONS")
        resolved_run.bold = True
        doc.add_paragraph("=" * 47)
        doc.add_paragraph("(Questions move here when answered)")

        doc.save(self.questions_file_path)

    def log_step_start(self, step_num: int, step_name: str):
        """
        Log the start of a workflow step.

        Args:
            step_num: Step number (0-16)
            step_name: Name of the step
        """
        message = f"Step {step_num} started: {step_name}"
        self.log_message(message, level="INFO")
        print(f"[Step {step_num}] Starting: {step_name}")

    def log_step_complete(self, step_num: int, message: str, artifacts: list[str] = None):
        """
        Log the completion of a workflow step.

        Args:
            step_num: Step number (0-16)
            message: Completion message
            artifacts: List of artifact paths created
        """
        log_msg = f"Step {step_num} completed: {message}"
        if artifacts:
            log_msg += f" (Artifacts: {', '.join(artifacts)})"

        self.log_message(log_msg, level="INFO")
        print(f"[Step {step_num}] Completed: {message}")

    def log_step_issue(self, step_num: int, issue: str, is_question: bool = False):
        """
        Log an issue encountered during a step.

        Args:
            step_num: Step number where issue occurred
            issue: Description of the issue
            is_question: If True, also add as collaborator question
        """
        log_msg = f"Step {step_num} issue: {issue}"
        self.log_message(log_msg, level="WARNING")
        print(f"[Step {step_num}] Issue: {issue}")

        # Auto-detect if this looks like a collaborator question
        question_keywords = ["request", "ask", "clarify", "confirm", "provide", "missing", "need"]
        if is_question or any(kw in issue.lower() for kw in question_keywords):
            self.add_collaborator_question(issue, f"Step {step_num}", step_num)

    def add_collaborator_question(self, question: str, context: str, step_num: int) -> str:
        """
        Add a question for the collaborator.

        Args:
            question: The question text
            context: Context for the question
            step_num: Step number where question arose

        Returns:
            Question ID (e.g., "CQ AA1")
        """
        from .state import WORKFLOW_STEPS

        # Add to state
        question_id = self.state.add_collaborator_question(question, context, step_num)

        # Get step name
        step_name = WORKFLOW_STEPS.get(step_num, {}).get("name", f"Step {step_num}")

        # Write to questions file
        if self.questions_file_path and self.questions_file_path.exists():
            self._add_question_to_file(question_id, question, context, step_num, step_name)

        # Also add to log file's Questions section
        if self.log_file_path and self.log_file_path.exists():
            self._add_question_to_log(question_id, question)

        print(f"[Question {question_id}] Added: {question[:50]}...")
        return question_id

    def _add_question_to_file(self, question_id: str, question: str, context: str,
                               step_num: int, step_name: str):
        """Add a question to the collaborator questions file."""
        doc = Document(self.questions_file_path)

        # Find the PENDING QUESTIONS section and add before RESOLVED
        # Insert before the "RESOLVED QUESTIONS" line
        insert_index = None
        for i, para in enumerate(doc.paragraphs):
            if "RESOLVED QUESTIONS" in para.text:
                insert_index = i - 1  # Insert before the separator
                break

        if insert_index is None:
            # Fallback: append at end
            insert_index = len(doc.paragraphs) - 1

        # Create new paragraphs for the question
        # We need to insert at a specific position, so we'll rebuild
        doc_new = Document(self.questions_file_path)

        # Find position and insert
        found_pending = False
        for para in doc_new.paragraphs:
            if "PENDING QUESTIONS" in para.text:
                found_pending = True
            elif found_pending and "RESOLVED QUESTIONS" in para.text:
                # Insert question before this
                break

        # Simpler approach: append to the document then save
        # The questions file structure allows appending in the pending section

        # Add the question entry
        q_para = doc.add_paragraph()
        q_run = q_para.add_run(f"{question_id} [Step {step_num}: {step_name}]")
        q_run.bold = True

        doc.add_paragraph(f"Question: {question}")
        doc.add_paragraph(f"Context: {context}")
        doc.add_paragraph("Status: PENDING")
        doc.add_paragraph("-" * 47)

        doc.save(self.questions_file_path)

    def _add_question_to_log(self, question_id: str, question: str):
        """Add a question reference to the log file's Questions section."""
        if not self.log_file_path or not self.log_file_path.exists():
            return

        try:
            content = self.log_file_path.read_text()
            lines = content.split("\n")

            # Find ">>> Questions for Collaborator" section and insert before next section
            found_section = False
            insert_idx = None

            for i, line in enumerate(lines):
                if ">>> Questions for Collaborator" in line:
                    found_section = True
                elif found_section and line.startswith("=" * 10):
                    # Found next section separator, insert before it
                    insert_idx = i
                    break

            if insert_idx:
                lines.insert(insert_idx, f"{question_id}: {question}")
                self.log_file_path.write_text("\n".join(lines))

        except Exception as e:
            logger.error(f"Failed to add question to log: {e}")

    def log_message(self, message: str, level: str = "INFO"):
        """
        Log a message to the log file.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        if not self.log_file_path:
            # Try to initialize files if not already done
            if self.state.country and self.state.year and self.working_dir.exists():
                self._initialize_files()

        if not self.log_file_path or not self.log_file_path.exists():
            return

        try:
            # Read current content
            content = self.log_file_path.read_text()
            lines = content.split("\n")

            # Update "Most recent update" line
            for i, line in enumerate(lines):
                if line.startswith("Most recent update:"):
                    lines[i] = f"Most recent update: {datetime.now().strftime('%Y-%m-%d')}"
                    break

            # Find ">>> Log File Notes" section and add entry after "Deposited Files:"
            self._log_entry_count += 1
            prefix = "" if level == "INFO" else "[!] "
            entry_text = f"{self._log_entry_count:02d}. {prefix}{message}"

            # Find where to insert (after "Deposited Files:" or after the section header)
            found_notes = False
            found_deposited = False
            insert_idx = None
            for i, line in enumerate(lines):
                if ">>> Log File Notes" in line and "===" in lines[i-1] if i > 0 else False:
                    found_notes = True
                elif found_notes and "Deposited Files:" in line:
                    found_deposited = True
                elif found_notes and found_deposited and line.strip() == "":
                    # Insert after the blank line following "Deposited Files:"
                    insert_idx = i + 1
                    break
                elif found_notes and ">>> Questions for Collaborator" in line:
                    # Insert before Questions section
                    insert_idx = i - 2
                    break

            if insert_idx and insert_idx < len(lines):
                lines.insert(insert_idx, entry_text)
            else:
                # Fallback: find Questions section and insert before it
                for i, line in enumerate(lines):
                    if ">>> Questions for Collaborator" in line and "===" in lines[i-1] if i > 0 else False:
                        lines.insert(i - 1, entry_text)
                        break

            self.log_file_path.write_text("\n".join(lines))

        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")

    def update_log_file_path(self):
        """Update state with current log file path."""
        if self.log_file_path:
            self.state.log_file = str(self.log_file_path)

    def update_questions_file_path(self):
        """Update state with current questions file path."""
        if self.questions_file_path:
            self.state.collaborator_questions_file = str(self.questions_file_path)

    def update_study_design_section(self, info: dict):
        """
        Update the OVERVIEW OF STUDY DESIGN AND WEIGHTS section in the log file.

        Args:
            info: Dictionary with keys: sample_design, sample_size, weighting,
                  collection_period, mode, response_rate
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return

        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year

        content_lines = [
            f"Sample Design: {info.get('sample_design', 'TBD')}",
            f"Sample Size: {info.get('sample_size', 'TBD')}",
            f"Response Rate: {info.get('response_rate', 'TBD')}",
            f"Weighting Methodology: {info.get('weighting', 'TBD')}",
            f"Data Collection Period: {info.get('collection_period', 'TBD')}",
            f"Mode of Interview: {info.get('mode', 'TBD')}",
        ]

        self._update_log_section(
            f"<<>> OVERVIEW OF STUDY DESIGN AND WEIGHTS - {country_code}_{year}_M6:",
            "\n".join(content_lines)
        )

    def update_deposit_inventory(self, data_files: list, questionnaires: list,
                                  codebooks: list, design_reports: list, macro_reports: list):
        """
        Update the Deposited Files section in the log file.

        Args:
            data_files: List of data file paths
            questionnaires: List of questionnaire paths
            codebooks: List of codebook paths
            design_reports: List of design report paths
            macro_reports: List of macro report paths
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return

        lines = ["Deposited Files:"]
        lines.append(f"  Data file(s): {len(data_files)}")
        for f in data_files:
            lines.append(f"    - {Path(f).name}")

        lines.append(f"  Questionnaire(s): {len(questionnaires)}")
        for f in questionnaires:
            lines.append(f"    - {Path(f).name}")

        lines.append(f"  Codebook(s): {len(codebooks)}")
        for f in codebooks:
            lines.append(f"    - {Path(f).name}")

        lines.append(f"  Design report(s): {len(design_reports)}")
        for f in design_reports:
            lines.append(f"    - {Path(f).name}")

        if macro_reports:
            lines.append(f"  Macro report(s): {len(macro_reports)}")
            for f in macro_reports:
                lines.append(f"    - {Path(f).name}")

        missing = []
        if not data_files:
            missing.append("DATA FILE")
        if not questionnaires:
            missing.append("QUESTIONNAIRE")
        if not design_reports:
            missing.append("DESIGN REPORT")

        if missing:
            lines.append(f"  MISSING: {', '.join(missing)}")

        self._update_deposited_files_section("\n".join(lines))

    def _update_log_section(self, section_marker: str, content: str):
        """
        Update a specific section in the log file.

        Args:
            section_marker: The section header to find (e.g., "<<>> OVERVIEW OF STUDY DESIGN")
            content: Content to insert after the section header
        """
        try:
            log_content = self.log_file_path.read_text()
            lines = log_content.split("\n")

            # Find the section and replace content until next section
            found_section = False
            start_idx = None
            end_idx = None

            for i, line in enumerate(lines):
                if section_marker in line:
                    found_section = True
                    start_idx = i + 1  # Start after the marker
                elif found_section and (line.startswith("<<>>") or line.startswith(">>>") or line.startswith("=")):
                    end_idx = i
                    break

            if start_idx is not None:
                if end_idx is None:
                    end_idx = len(lines)

                # Replace content between markers
                new_lines = lines[:start_idx] + ["", content, ""] + lines[end_idx:]
                self.log_file_path.write_text("\n".join(new_lines))

        except Exception as e:
            logger.error(f"Failed to update log section: {e}")

    def _update_deposited_files_section(self, content: str):
        """Update the Deposited Files section in the log file."""
        try:
            log_content = self.log_file_path.read_text()
            lines = log_content.split("\n")

            # Find "Deposited Files:" and replace until next blank line or section
            start_idx = None
            end_idx = None

            for i, line in enumerate(lines):
                if "Deposited Files:" in line:
                    start_idx = i
                elif start_idx is not None and (line.startswith(">>>") or line.startswith("=")):
                    end_idx = i
                    break

            if start_idx is not None:
                if end_idx is None:
                    end_idx = start_idx + 1

                new_lines = lines[:start_idx] + [content, ""] + lines[end_idx:]
                self.log_file_path.write_text("\n".join(new_lines))

        except Exception as e:
            logger.error(f"Failed to update deposited files section: {e}")
