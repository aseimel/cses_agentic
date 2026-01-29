"""
Active Logging for CSES Workflow.

Provides real-time logging following CSES naming conventions:
- Log file: micro/cses-m6_log-file_{CODE}_{YEAR}_{DATE}.txt (plain text)
- Questions file: micro/Collaborator Questions/{Country}_{Year}_micro_collaborator_questions_{DATE}.txt (plain text)
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, TYPE_CHECKING

# docx only needed for variable tracking updates
try:
    from docx import Document
    from docx.shared import Pt, RGBColor
except ImportError:
    Document = None
    Pt = None
    RGBColor = None

if TYPE_CHECKING:
    from .state import WorkflowState

logger = logging.getLogger(__name__)


class ActiveLogger:
    """
    Handles real-time logging to CSES-standard text files.

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

        # Questions file path (plain text)
        questions_dir = micro_dir / "Collaborator Questions"
        questions_dir.mkdir(exist_ok=True)
        questions_filename = f"{self.state.country}_{year}_micro_collaborator_questions_{date_str}.txt"
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
        """Create a new collaborator questions file (plain text)."""
        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        country = self.state.country
        date_str = datetime.now().strftime("%Y-%m-%d")

        lines = [
            f"Collaborator Questions - {country} {year}",
            f"Study: {country_code}_{year}_M6",
            f"Generated: {date_str}",
            f"Processor: [NAME]",
            "",
            "=" * 75,
            "PENDING QUESTIONS",
            "=" * 75,
            "",
            "",
            "=" * 75,
            "RESOLVED QUESTIONS",
            "=" * 75,
            "(Questions move here when answered)",
            "",
        ]

        self.questions_file_path.write_text("\n".join(lines))

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

    def add_collaborator_question(self, question: str, context: str, step_num: int) -> tuple:
        """
        Add a question for the collaborator.
        Returns (success, message) tuple. Message includes question ID.
        """
        from .state import WORKFLOW_STEPS

        try:
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

            return True, f"Question {question_id} added: {question[:50]}..."

        except Exception as e:
            return False, f"Failed to add question: {e}"

    def _add_question_to_file(self, question_id: str, question: str, context: str,
                               step_num: int, step_name: str):
        """Add a question to the collaborator questions file (plain text)."""
        if not self.questions_file_path or not self.questions_file_path.exists():
            return

        try:
            content = self.questions_file_path.read_text()
            lines = content.split("\n")

            # Find RESOLVED QUESTIONS section and insert before it
            insert_idx = None
            for i, line in enumerate(lines):
                if "RESOLVED QUESTIONS" in line and i > 0 and "=" in lines[i-1]:
                    insert_idx = i - 1  # Insert before separator
                    break

            if insert_idx:
                question_block = [
                    f"{question_id} [Step {step_num}: {step_name}]",
                    f"Question: {question}",
                    f"Context: {context}",
                    "Status: PENDING",
                    "-" * 75,
                    "",
                ]
                new_lines = lines[:insert_idx] + question_block + lines[insert_idx:]
                self.questions_file_path.write_text("\n".join(new_lines))

        except Exception as e:
            logger.error(f"Failed to add question to file: {e}")

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

    def log_message(self, message: str, level: str = "INFO") -> tuple:
        """
        Log a message to the log file.
        Returns (success, status_message) tuple.
        """
        if not self.log_file_path:
            # Try to initialize files if not already done
            if self.state.country and self.state.year and self.working_dir.exists():
                self._initialize_files()

        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        try:
            # Read current content
            original_content = self.log_file_path.read_text()
            lines = original_content.split("\n")

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

            inserted = False
            if insert_idx and insert_idx < len(lines):
                lines.insert(insert_idx, entry_text)
                inserted = True
            else:
                # Fallback: find Questions section and insert before it
                for i, line in enumerate(lines):
                    if ">>> Questions for Collaborator" in line and "===" in lines[i-1] if i > 0 else False:
                        lines.insert(i - 1, entry_text)
                        inserted = True
                        break

            if not inserted:
                return False, "Could not find insertion point in log file"

            self.log_file_path.write_text("\n".join(lines))

            # Verify the entry was written
            verify_content = self.log_file_path.read_text()
            if entry_text not in verify_content:
                self.log_file_path.write_text(original_content)
                return False, "Verification failed - entry not found after write"

            return True, f"Logged: {message[:50]}..."

        except Exception as e:
            logger.error(f"Failed to write to log file: {e}")
            return False, f"Error writing to log: {e}"

    def update_log_file_path(self):
        """Update state with current log file path."""
        if self.log_file_path:
            self.state.log_file = str(self.log_file_path)

    def update_questions_file_path(self):
        """Update state with current questions file path."""
        if self.questions_file_path:
            self.state.collaborator_questions_file = str(self.questions_file_path)

    def update_study_design_section(self, info: dict) -> tuple:
        """
        Update the OVERVIEW OF STUDY DESIGN AND WEIGHTS section in the log file.
        Returns (success, message) tuple.
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        section_marker = f"<<>> OVERVIEW OF STUDY DESIGN AND WEIGHTS - {country_code}_{year}_M6:"

        # Read current values from the log file
        existing_values = self._read_study_design_values(section_marker)

        # Merge new values with existing (new values take precedence)
        merged = {**existing_values, **info}

        # Format the section content
        content_lines = [
            f"Sample Design: {merged.get('sample_design', 'TBD')}",
            f"Sample Size: {merged.get('sample_size', 'TBD')}",
            f"Response Rate: {merged.get('response_rate', 'TBD')}",
            f"Weighting Methodology: {merged.get('weighting', 'TBD')}",
            f"Data Collection Period: {merged.get('collection_period', 'TBD')}",
            f"Mode of Interview: {merged.get('mode', 'TBD')}",
            f"Field Lag: {merged.get('field_lag', 'TBD')}",
        ]

        success, msg = self._update_log_section(section_marker, "\n".join(content_lines))
        if success:
            fields_updated = ", ".join(info.keys())
            return True, f"Study design updated: {fields_updated}"
        return success, msg

    def _read_study_design_values(self, section_marker: str) -> dict:
        """
        Read existing study design values from the log file.

        Returns a dict with current values for preservation during updates.
        """
        values = {}
        if not self.log_file_path or not self.log_file_path.exists():
            return values

        try:
            content = self.log_file_path.read_text()
            lines = content.split("\n")

            in_section = False
            for line in lines:
                if section_marker in line:
                    in_section = True
                    continue
                elif in_section and (line.startswith("<<>>") or line.startswith(">>>") or line.startswith("=")):
                    break
                elif in_section and ":" in line:
                    # Parse "Field Name: value" format
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        field_name = parts[0].strip()
                        value = parts[1].strip()
                        if value and value != "TBD":
                            # Map display names to internal keys
                            field_map = {
                                "Sample Design": "sample_design",
                                "Sample Size": "sample_size",
                                "Response Rate": "response_rate",
                                "Weighting Methodology": "weighting",
                                "Data Collection Period": "collection_period",
                                "Mode of Interview": "mode",
                                "Field Lag": "field_lag",
                            }
                            if field_name in field_map:
                                values[field_map[field_name]] = value
        except Exception as e:
            logger.error(f"Failed to read study design values: {e}")

        return values

    def update_election_summary(self, summary: str) -> tuple:
        """
        Update the ELECTION SUMMARY section in the log file.
        Returns (success, message) tuple.
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        success, msg = self._update_log_section(
            f"<<>> ELECTION SUMMARY - {country_code}_{year}_M6:",
            summary
        )
        if success:
            return True, "Election summary updated"
        return success, msg

    def update_parties_leaders(self, content: str) -> tuple:
        """
        Update the PARTIES AND LEADERS section in the log file.
        Returns (success, message) tuple.
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        country_code = self.state.country_code or self.state.country[:3].upper()
        year = self.state.year
        success, msg = self._update_log_section(
            f"<<>> PARTIES AND LEADERS: {country_code}_{year}_M6",
            content
        )
        if success:
            return True, "Parties and leaders updated"
        return success, msg

    def add_todo_item(self, item: str) -> tuple:
        """
        Add an item to the Things To Do Before Releasing section.
        Returns (success, message) tuple.
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        try:
            original_content = self.log_file_path.read_text()
            lines = original_content.split("\n")

            # Find ">>> Things To Do Before Releasing" and insert before next section
            found_section = False
            insert_idx = None
            for i, line in enumerate(lines):
                if ">>> Things To Do Before Releasing" in line:
                    found_section = True
                elif found_section and (line.startswith(">>>") or line.startswith("=")):
                    insert_idx = i
                    break

            if insert_idx:
                todo_entry = f"- {item}"
                lines.insert(insert_idx, todo_entry)
                self.log_file_path.write_text("\n".join(lines))

                # Verify
                verify = self.log_file_path.read_text()
                if todo_entry not in verify:
                    self.log_file_path.write_text(original_content)
                    return False, "Verification failed - TODO not found after write"

                return True, f"TODO added: {item[:30]}..."

            return False, "Could not find TODO section in log file"

        except Exception as e:
            logger.error(f"Failed to add TODO item: {e}")
            return False, f"Error adding TODO: {e}"

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

    def _update_log_section(self, section_marker: str, content: str) -> tuple:
        """
        Update a specific section in the log file.
        ESSENTIAL: Content MUST be added - this is not optional.
        Returns (success, message) tuple.
        """
        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        try:
            # Read and preserve original for rollback
            original_content = self.log_file_path.read_text()
            lines = original_content.split("\n")

            # Find section start
            start_idx = None
            for i, line in enumerate(lines):
                if section_marker in line:
                    start_idx = i + 1  # Content starts after marker
                    break

            # If section marker not found, ADD it to the file
            if start_idx is None:
                # Find the Election Study Notes section to add subsections
                insert_point = None
                for i, line in enumerate(lines):
                    if ">>> Election Study Notes" in line:
                        # Find the end of this section (next >>> or end)
                        for j in range(i + 1, len(lines)):
                            if lines[j].startswith(">>>") or (lines[j].startswith("=") and len(lines[j]) > 10):
                                insert_point = j
                                break
                        if insert_point is None:
                            insert_point = len(lines)
                        break

                if insert_point is None:
                    # Last resort: append to end of file
                    insert_point = len(lines)

                # Add the section marker and content
                new_section = ["", section_marker, "", content, ""]
                lines = lines[:insert_point] + new_section + lines[insert_point:]
                self.log_file_path.write_text("\n".join(lines))

                # Verify
                verify = self.log_file_path.read_text()
                if content not in verify:
                    self.log_file_path.write_text(original_content)
                    return False, "Failed to add new section - verification failed"
                return True, f"Section added: {section_marker[:30]}..."

            # Find section end - look for next section marker (<<>> or >>> or ===)
            end_idx = None
            for i in range(start_idx, len(lines)):
                line = lines[i].strip()
                if line.startswith("<<>>") or line.startswith(">>>") or line.startswith("=" * 10):
                    end_idx = i
                    break

            # If no end marker, this is the LAST section - replace to end of file
            # This is correct behavior for sections like PARTIES AND LEADERS
            if end_idx is None:
                end_idx = len(lines)
                # Trim trailing empty lines from end_idx so we don't accumulate blanks
                while end_idx > start_idx and lines[end_idx - 1].strip() == "":
                    end_idx -= 1

            # Build new content - preserve everything before and after
            new_lines = lines[:start_idx] + ["", content, ""] + lines[end_idx:]
            new_content = "\n".join(new_lines)

            # Write new content
            self.log_file_path.write_text(new_content)

            # VERIFY: Read back and check content was written
            verify_content = self.log_file_path.read_text()
            if content not in verify_content:
                # Restore original
                self.log_file_path.write_text(original_content)
                return False, "Verification failed - content not found after write"

            return True, "Section updated successfully"

        except Exception as e:
            logger.error(f"Failed to update log section: {e}")
            return False, f"Error updating section: {e}"

    def _update_deposited_files_section(self, content: str) -> tuple:
        """Update the Deposited Files section - MUST add content."""
        if not self.log_file_path or not self.log_file_path.exists():
            return False, "Log file does not exist"

        try:
            original_content = self.log_file_path.read_text()
            lines = original_content.split("\n")

            # Find "Deposited Files:" line
            start_idx = None
            for i, line in enumerate(lines):
                if "Deposited Files:" in line:
                    start_idx = i
                    break

            # If not found, ADD the section after ">>> Log File Notes"
            if start_idx is None:
                insert_point = None
                for i, line in enumerate(lines):
                    if ">>> Log File Notes" in line:
                        # Insert after the section header line and separator
                        insert_point = i + 2
                        break
                if insert_point is None:
                    insert_point = 10  # Fallback to near start

                lines = lines[:insert_point] + ["", content, ""] + lines[insert_point:]
                self.log_file_path.write_text("\n".join(lines))

                verify = self.log_file_path.read_text()
                if "Deposited Files:" not in verify:
                    self.log_file_path.write_text(original_content)
                    return False, "Failed to add Deposited Files section"
                return True, "Deposited files section added"

            # Find end - next section marker
            end_idx = None
            for i in range(start_idx + 1, len(lines)):
                line = lines[i].strip()
                if line.startswith(">>>") or line.startswith("=" * 10):
                    end_idx = i
                    break

            # If no end marker, replace to end of Log File Notes section
            if end_idx is None:
                # Look for the next major section
                for i in range(start_idx + 1, len(lines)):
                    if ">>> Questions for Collaborator" in lines[i]:
                        end_idx = i - 1  # Before the separator
                        break
                if end_idx is None:
                    end_idx = len(lines)

            new_lines = lines[:start_idx] + [content, ""] + lines[end_idx:]
            new_content = "\n".join(new_lines)

            self.log_file_path.write_text(new_content)

            # Verify
            verify = self.log_file_path.read_text()
            if "Deposited Files:" not in verify:
                self.log_file_path.write_text(original_content)
                return False, "Verification failed"

            return True, "Deposited files updated"

        except Exception as e:
            logger.error(f"Failed to update deposited files section: {e}")
            return False, f"Error: {e}"

    def update_variable_mapping(self, cses_variable: str, source_variable: str, remarks: str = ""):
        """
        Update the variable tracking sheet with a mapping.

        Args:
            cses_variable: The CSES target variable code (e.g., "F1003_2")
            source_variable: The source variable from the deposited data
            remarks: Optional remarks about the mapping
        """
        tracking_file = self.state.variable_tracking_file
        if not tracking_file:
            logger.warning("No variable tracking file set in state")
            return

        tracking_path = Path(tracking_file)
        if not tracking_path.exists():
            logger.warning(f"Variable tracking file not found: {tracking_path}")
            return

        try:
            import openpyxl
            wb = openpyxl.load_workbook(tracking_path)
            ws = wb['deposited variables']

            # Find the row with the CSES variable code
            found = False
            for row_idx, row in enumerate(ws.iter_rows(min_row=1), start=1):
                # Check column B (index 1) for the CSES variable code
                cell_b = row[1] if len(row) > 1 else None
                if cell_b and cell_b.value and str(cell_b.value).strip() == cses_variable:
                    # Found the row - update column C (CNT_YEAR) with source variable
                    ws.cell(row=row_idx, column=3, value=source_variable)
                    # Update column D (REMARKS) if provided
                    if remarks:
                        ws.cell(row=row_idx, column=4, value=remarks)
                    found = True
                    break

            if found:
                wb.save(tracking_path)
                print(f"  [Variable] {cses_variable} <- {source_variable}")
            else:
                logger.warning(f"CSES variable {cses_variable} not found in tracking sheet")

        except Exception as e:
            logger.error(f"Failed to update variable tracking sheet: {e}")
