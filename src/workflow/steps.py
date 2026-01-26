"""
Step Executor for CSES Workflow.

Provides execution logic for each of the 16 workflow steps.
Each step can be executed interactively with user guidance.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any

from .state import WorkflowState, StepStatus, WORKFLOW_STEPS

logger = logging.getLogger(__name__)


@dataclass
class StepResult:
    """Result of executing a workflow step."""
    success: bool
    message: str
    artifacts: list[str] = None
    issues: list[str] = None
    next_action: str = None  # Suggested next action

    def __post_init__(self):
        self.artifacts = self.artifacts or []
        self.issues = self.issues or []


class StepExecutor:
    """
    Executes workflow steps with LLM assistance.

    Each step method:
    1. Checks prerequisites
    2. Executes the step (possibly with LLM)
    3. Updates workflow state
    4. Returns result with artifacts and issues
    """

    def __init__(self, state: WorkflowState, llm_callback: Callable = None):
        """
        Initialize executor.

        Args:
            state: Current workflow state
            llm_callback: Function to call LLM for assistance
        """
        self.state = state
        self.llm_callback = llm_callback
        self.working_dir = Path(state.working_dir) if state.working_dir else Path.cwd()

    def execute_step(self, step_num: int, **kwargs) -> StepResult:
        """
        Execute a specific workflow step.

        Args:
            step_num: Step number (0-16)
            **kwargs: Step-specific arguments

        Returns:
            StepResult with success status and details
        """
        if step_num not in WORKFLOW_STEPS:
            return StepResult(
                success=False,
                message=f"Unknown step number: {step_num}"
            )

        # Get step handler
        handler_name = f"_step_{step_num}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            return StepResult(
                success=False,
                message=f"Step {step_num} handler not implemented yet",
                next_action=f"This step requires manual execution. See workflow.md for details."
            )

        # Mark step as in progress
        self.state.set_step_status(step_num, StepStatus.IN_PROGRESS)
        self.state.save()

        try:
            result = handler(**kwargs)

            # Update state based on result
            if result.success:
                self.state.set_step_status(
                    step_num,
                    StepStatus.COMPLETED,
                    note=result.message
                )
                for artifact in result.artifacts:
                    self.state.add_step_artifact(step_num, artifact)
            else:
                for issue in result.issues:
                    self.state.add_step_issue(step_num, issue)

            self.state.save()
            return result

        except Exception as e:
            logger.error(f"Step {step_num} failed: {e}")
            self.state.add_step_issue(step_num, str(e))
            self.state.save()
            return StepResult(
                success=False,
                message=f"Step failed with error: {e}",
                issues=[str(e)]
            )

    def _step_0(self, **kwargs) -> StepResult:
        """Step 0: Set Up Country Folder"""
        # This is typically done during initialization
        # Check that folder structure exists

        required_folders = [
            self.working_dir / "emails",
            self.working_dir / "micro",
            self.working_dir / "macro",
        ]

        missing = [f for f in required_folders if not f.exists()]

        if missing:
            return StepResult(
                success=False,
                message=f"Missing required folders: {[f.name for f in missing]}",
                issues=[f"Create folder: {f}" for f in missing]
            )

        return StepResult(
            success=True,
            message="Folder structure verified",
            artifacts=[str(self.working_dir)],
            next_action="Proceed to Step 1: Check deposit completeness"
        )

    def _step_1(self, **kwargs) -> StepResult:
        """Step 1: Check Completeness of Deposit"""
        from src.workflow.organizer import FileOrganizer

        organizer = FileOrganizer(self.working_dir)
        detected = organizer.detect_files()

        issues = []
        if not detected.data_files:
            issues.append("No data file found")
        if not detected.questionnaire_files:
            issues.append("No questionnaire found (required per CSES policy)")
        if not detected.design_report_files:
            issues.append("No design report found")
        if not detected.codebook_files:
            issues.append("No codebook found (optional but helpful)")

        # Update state with detected files
        if detected.data_files:
            self.state.data_file = str(detected.data_files[0])
        self.state.questionnaire_files = [str(f) for f in detected.questionnaire_files]
        if detected.codebook_files:
            self.state.codebook_file = str(detected.codebook_files[0])
        if detected.design_report_files:
            self.state.design_report_file = str(detected.design_report_files[0])

        if detected.has_minimum_requirements():
            return StepResult(
                success=True,
                message=f"Deposit complete. Found {len(detected.data_files)} data file(s), "
                        f"{len(detected.questionnaire_files)} questionnaire(s)",
                artifacts=[str(f) for f in detected.data_files + detected.questionnaire_files],
                issues=issues if issues else None,
                next_action="Proceed to Step 2: Read design report"
            )
        else:
            return StepResult(
                success=False,
                message="Deposit incomplete - missing required files",
                issues=issues,
                next_action="Request missing files from collaborator"
            )

    def _step_2(self, **kwargs) -> StepResult:
        """Step 2: Read Design Report"""
        design_report = self.state.design_report_file

        if not design_report or not Path(design_report).exists():
            return StepResult(
                success=False,
                message="Design report not found",
                issues=["Request design report from collaborator"],
                next_action="Cannot proceed without design report"
            )

        # If LLM available, analyze the design report
        if self.llm_callback:
            from src.ingest.doc_parser import DocumentParser
            parser = DocumentParser()
            doc_info = parser.parse(Path(design_report))

            if doc_info and doc_info.full_text:
                # Ask LLM to analyze
                analysis = self.llm_callback(
                    f"Analyze this CSES design report and check if it meets standards:\n\n"
                    f"{doc_info.full_text[:10000]}\n\n"
                    f"Check for: sample size, sampling method, response rate, fieldwork dates, "
                    f"mode of data collection. Flag any concerns."
                )
                return StepResult(
                    success=True,
                    message="Design report analyzed",
                    artifacts=[design_report],
                    next_action=f"Review analysis:\n{analysis}"
                )

        return StepResult(
            success=True,
            message="Design report located - manual review required",
            artifacts=[design_report],
            next_action="Please review the design report manually and note any issues in the logfile"
        )

    def _step_3(self, **kwargs) -> StepResult:
        """Step 3: Fill Variable Tracking Sheet"""
        # This step uses the variable matching functionality
        data_file = self.state.data_file

        if not data_file or not Path(data_file).exists():
            return StepResult(
                success=False,
                message="Data file not found",
                issues=["Data file required for variable tracking"]
            )

        # Load data and check variables
        from src.ingest.data_loader import DataLoader
        loader = DataLoader()
        dataset_info = loader.load(Path(data_file))

        if not dataset_info:
            return StepResult(
                success=False,
                message="Could not load data file",
                issues=["Check data file format"]
            )

        n_vars = len(dataset_info.variables)

        return StepResult(
            success=True,
            message=f"Found {n_vars} variables in deposited data",
            artifacts=[data_file],
            next_action="Run variable matching (Step 7) to fill tracking sheet"
        )

    def _step_6(self, **kwargs) -> StepResult:
        """Step 6: Run Frequencies on Original Data"""
        data_file = self.state.data_file

        if not data_file or not Path(data_file).exists():
            return StepResult(
                success=False,
                message="Data file not found"
            )

        try:
            from src.ingest.data_loader import DataLoader
            import polars as pl

            loader = DataLoader()
            dataset_info = loader.load(Path(data_file))

            if not dataset_info:
                return StepResult(
                    success=False,
                    message="Could not load data file"
                )

            # Generate frequency summary
            freq_lines = ["# Variable Frequencies\n"]
            for var_name, var_info in list(dataset_info.variables.items())[:20]:
                freq_lines.append(f"\n## {var_name}")
                if var_info.description:
                    freq_lines.append(f"Description: {var_info.description}")
                if var_info.value_labels:
                    labels = ", ".join(f"{k}={v}" for k, v in list(var_info.value_labels.items())[:5])
                    freq_lines.append(f"Labels: {labels}")
                if var_info.sample_values:
                    freq_lines.append(f"Sample values: {var_info.sample_values[:5]}")

            # Save frequencies
            freq_file = self.working_dir / "micro" / "frequencies.txt"
            freq_file.parent.mkdir(exist_ok=True)
            freq_file.write_text("\n".join(freq_lines))

            return StepResult(
                success=True,
                message=f"Generated frequencies for {len(dataset_info.variables)} variables",
                artifacts=[str(freq_file)],
                next_action="Review frequencies and proceed to variable processing"
            )

        except Exception as e:
            return StepResult(
                success=False,
                message=f"Failed to run frequencies: {e}",
                issues=[str(e)]
            )

    def _step_7(self, **kwargs) -> StepResult:
        """Step 7: Process Variables - Variable Matching"""
        # This is the main variable matching step
        data_file = self.state.data_file

        if not data_file:
            return StepResult(
                success=False,
                message="Data file not found"
            )

        doc_files = []
        if self.state.questionnaire_files:
            doc_files.extend(self.state.questionnaire_files)
        if self.state.codebook_file:
            doc_files.append(self.state.codebook_file)

        if not doc_files:
            return StepResult(
                success=False,
                message="No documentation files found for matching"
            )

        # Run the dual-model matching
        try:
            from src.agent import run_harmonization

            result = run_harmonization(
                data_file=Path(data_file),
                doc_files=[Path(f) for f in doc_files],
                country=self.state.country,
                year=self.state.year,
                validate=True,
                output_dir=self.working_dir / ".cses"
            )

            # Store mappings in state
            self.state.mappings = [v.to_dict() for v in result.validations]
            self.state.save()

            return StepResult(
                success=True,
                message=f"Matched {result.matched_count}/{result.total_targets} variables. "
                        f"Agreements: {result.agree_count}, Disagreements: {result.disagree_count}",
                artifacts=[str(self.working_dir / ".cses" / "mappings")],
                next_action="Review mappings in the UI or conversation, then approve/reject"
            )

        except Exception as e:
            logger.error(f"Variable matching failed: {e}")
            return StepResult(
                success=False,
                message=f"Variable matching failed: {e}",
                issues=[str(e)]
            )

    def _step_12(self, **kwargs) -> StepResult:
        """Step 12: Run Check Files"""
        # Quality checks on processed data
        data_file = self.state.data_file

        if not data_file:
            return StepResult(
                success=False,
                message="Data file not found"
            )

        if not self.state.mappings:
            return StepResult(
                success=False,
                message="No variable mappings found - complete Step 7 first"
            )

        # Run basic quality checks
        issues = []
        warnings = []

        # Check for mappings with disagreements
        disagreements = [
            m for m in self.state.mappings
            if m.get("validation_verdict") == "DISAGREE"
        ]
        if disagreements:
            warnings.append(f"{len(disagreements)} mappings had model disagreements")

        # Check for NOT_FOUND
        not_found = [
            m for m in self.state.mappings
            if m.get("source_variable") in ["NOT_FOUND", "ERROR"]
        ]
        if not_found:
            warnings.append(f"{len(not_found)} CSES variables not matched")

        return StepResult(
            success=True,
            message="Quality checks completed",
            issues=issues if issues else None,
            next_action="Review any warnings and draft collaborator questions if needed"
        )

    def _step_13(self, **kwargs) -> StepResult:
        """Step 13: Write Up Collaborator Questions"""
        # Compile questions from issues across steps

        questions = []

        # Check for step issues
        for step_num in WORKFLOW_STEPS:
            step = self.state.get_step(step_num)
            if step.issues:
                for issue in step.issues:
                    if "request" in issue.lower() or "ask" in issue.lower() or "clarify" in issue.lower():
                        questions.append({
                            "step": step_num,
                            "issue": issue,
                            "status": "pending"
                        })

        # Check for mapping disagreements
        for m in self.state.mappings:
            if m.get("validation_verdict") == "DISAGREE":
                questions.append({
                    "type": "mapping_clarification",
                    "target": m.get("cses_target"),
                    "source": m.get("source_variable"),
                    "issue": m.get("validation_reasoning", "Models disagreed on mapping"),
                    "status": "pending"
                })

        self.state.pending_questions = questions
        self.state.save()

        if questions:
            # Generate questions document
            questions_file = self.working_dir / "micro" / "collaborator_questions.txt"
            lines = [
                f"# Collaborator Questions: {self.state.country} {self.state.year}",
                "",
                f"Total questions: {len(questions)}",
                ""
            ]
            for i, q in enumerate(questions, 1):
                lines.append(f"## Question {i}")
                lines.append(f"Issue: {q.get('issue', 'N/A')}")
                lines.append(f"Status: {q.get('status', 'pending')}")
                lines.append("")

            questions_file.write_text("\n".join(lines))

            return StepResult(
                success=True,
                message=f"Compiled {len(questions)} questions for collaborator",
                artifacts=[str(questions_file)],
                next_action="Review questions and send to project manager"
            )
        else:
            return StepResult(
                success=True,
                message="No collaborator questions needed",
                next_action="Proceed to Step 14 or 15"
            )

    def get_step_guidance(self, step_num: int) -> str:
        """Get guidance text for a specific step."""
        if step_num not in WORKFLOW_STEPS:
            return "Unknown step"

        step_info = WORKFLOW_STEPS[step_num]

        lines = [
            f"## Step {step_num}: {step_info['name']}",
            "",
            f"**Description:** {step_info['description']}",
            "",
            f"**Automation:** {'High' if step_info['automatable'] else 'Manual'}",
            f"**Requires LLM:** {'Yes' if step_info['requires_llm'] else 'No'}",
        ]

        # Add step-specific guidance
        guidance = {
            0: "Create folder structure from CSES template. Files go in: emails/ (documentation), micro/ (data processing)",
            1: "Check that all required files are present: data file, questionnaire, design report. Mark in tracking sheet.",
            2: "Review the design report for: sample size, sampling method, response rate, fieldwork dates, data collection mode.",
            3: "Check variable list against CSES requirements. Note missing or unusual variables.",
            6: "Run frequency tables on the original data to see available variables and their distributions.",
            7: "For each CSES variable, find the matching source variable and write recode commands.",
            12: "Run inconsistency checks, theoretical checks, and interviewer validation checks.",
            13: "Compile all questions that arose during processing for collaborator clarification."
        }

        if step_num in guidance:
            lines.extend(["", f"**Guidance:** {guidance[step_num]}"])

        return "\n".join(lines)
