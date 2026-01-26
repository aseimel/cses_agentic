#!/usr/bin/env python3
"""
CSES Data Harmonization CLI.

A command-line tool for processing CSES Module 6 election studies.
Run in a folder containing collaborator files to start the workflow.

Usage:
    cses              # Start interactive mode in current folder
    cses init         # Initialize new study from files in folder
    cses status       # Show workflow status
    cses step N       # Work on step N
    cses match        # Run variable matching (Step 7)
    cses export       # Export approved mappings
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow.state import (
    WorkflowState,
    StepStatus,
    WORKFLOW_STEPS,
    format_workflow_status
)
from src.workflow.organizer import FileOrganizer, detect_and_summarize
from src.workflow.steps import StepExecutor
from src.agent.validator import check_claude_cli_available

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           CSES Data Harmonization Agent                      ║
║           Module 6 Processing Workflow                       ║
╚══════════════════════════════════════════════════════════════╝
""")


def check_validation_setup() -> tuple[bool, str]:
    """
    Check validation model configuration.

    Returns:
        Tuple of (is_ready: bool, status_message: str)
    """
    from dotenv import load_dotenv
    load_dotenv()

    validation_model = os.getenv("LLM_MODEL_VALIDATE", "")

    if validation_model == "claude-cli":
        # Using Claude CLI with Max subscription
        available, msg = check_claude_cli_available()
        if available:
            return True, f"✅ Validation: Claude CLI (Max subscription)\n   {msg}"
        else:
            return False, f"⚠️ Validation: Claude CLI not ready\n   {msg}\n   Run 'claude login' to authenticate"

    elif validation_model:
        # Using API model
        return True, f"✅ Validation: {validation_model} (API)"

    else:
        return False, "⚠️ Validation model not configured. Set LLM_MODEL_VALIDATE in .env"


def print_validation_status():
    """Print validation configuration status."""
    ready, message = check_validation_setup()
    print(message)
    if not ready:
        print("\nTo use Claude CLI with your Max subscription:")
        print("  1. Install Claude Code: npm install -g @anthropic-ai/claude-code")
        print("  2. Login: claude login")
        print("  3. Set in .env: LLM_MODEL_VALIDATE=claude-cli")
        print("\nOr use an API model:")
        print("  Set in .env: LLM_MODEL_VALIDATE=anthropic/claude-sonnet-4-20250514")
    print()


def cmd_init(args):
    """Initialize a new study from files in the current folder."""
    working_dir = Path.cwd()

    # Check if already initialized
    existing_state = WorkflowState.load(working_dir)
    if existing_state:
        print(f"Study already initialized: {existing_state.country} {existing_state.year}")
        print(f"Use 'cses status' to see progress or 'cses' for interactive mode.")
        return

    print("Scanning folder for collaborator files...\n")

    # Detect files
    organizer = FileOrganizer(working_dir)
    detected = organizer.detect_files()

    print(detected.summary())
    print()

    # Get country/year (use detected or prompt)
    country = args.country or detected.country
    year = args.year or detected.year

    if not country:
        country = input("Enter country name: ").strip() or "Unknown"
    if not year:
        year = input("Enter election year: ").strip() or "0000"

    # Determine whether to organize files
    if args.no_organize:
        study_dir = working_dir
    else:
        # Check if we should create folder structure
        # Use country code for folder name
        country_code = organizer.COUNTRY_CODES.get(country.lower(), country[:3].upper())
        expected_folder = f"{country_code}_{year}"

        if working_dir.name == expected_folder or (working_dir / ".cses").exists():
            # Already in a study folder
            study_dir = working_dir
            print(f"\nUsing existing folder structure: {working_dir}")
        else:
            # Ask user
            print(f"\nWould you like to organize files into {expected_folder}/?")
            print(f"All files will be renamed with {expected_folder}_ prefix:")
            print(f"  {expected_folder}_original_data.dta")
            print(f"  {expected_folder}_questionnaire.pdf")
            print(f"  {expected_folder}_codebook.docx")
            print(f"  ... plus generated outputs:")
            print(f"  {expected_folder}_processing.do")
            print(f"  {expected_folder}_variable_mappings.xlsx")
            print(f"  {expected_folder}_M6.dta")
            response = input("Organize files? [Y/n]: ").strip().lower()

            if response != 'n':
                study_dir, _ = organizer.initialize_study(
                    country=country,
                    year=year,
                    copy_files=not args.move
                )
                print(f"\nCreated study folder: {study_dir}")
            else:
                study_dir = working_dir

    # Create workflow state
    country_code = organizer.COUNTRY_CODES.get(country.lower(), country[:3].upper())
    state = WorkflowState(
        country=country,
        country_code=country_code,
        year=year,
        working_dir=str(study_dir)
    )

    # Store detected files with new standardized names
    prefix = organizer.get_study_prefix(country, year)

    if detected.data_files:
        if study_dir != working_dir:
            ext = detected.data_files[0].suffix
            state.data_file = str(study_dir / f"{prefix}_original_data{ext}")
        else:
            state.data_file = str(detected.data_files[0])

    if detected.questionnaire_files:
        if study_dir != working_dir:
            state.questionnaire_files = []
            for i, f in enumerate(detected.questionnaire_files):
                ext = f.suffix
                if len(detected.questionnaire_files) == 1:
                    state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire{ext}"))
                else:
                    state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire_{i+1}{ext}"))
        else:
            state.questionnaire_files = [str(f) for f in detected.questionnaire_files]

    if detected.codebook_files:
        if study_dir != working_dir:
            ext = detected.codebook_files[0].suffix
            state.codebook_file = str(study_dir / f"{prefix}_codebook{ext}")
        else:
            state.codebook_file = str(detected.codebook_files[0])

    if detected.design_report_files:
        if study_dir != working_dir:
            ext = detected.design_report_files[0].suffix
            state.design_report_file = str(study_dir / f"{prefix}_design_report{ext}")
        else:
            state.design_report_file = str(detected.design_report_files[0])

    # Mark Step 0 as complete
    state.set_step_status(0, StepStatus.COMPLETED, "Folder initialized")

    # Save state
    state.save(study_dir / ".cses")

    print(f"\n✅ Study initialized: {country} {year}")
    print(f"   Session ID: {state.session_id}")
    print(f"   Working directory: {study_dir}")
    print(f"\nRun 'cses' to enter interactive mode or 'cses status' to see workflow.")


def cmd_status(args):
    """Show workflow status."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized in this folder.")
        print("Run 'cses init' to start a new study.")
        return

    print(format_workflow_status(state))

    # Show next action
    next_step = state.get_next_step()
    if next_step is not None:
        step_info = WORKFLOW_STEPS[next_step]
        print(f"\n**Next:** Step {next_step} - {step_info['name']}")
        print(f"Run 'cses step {next_step}' or 'cses' for interactive mode.")


def cmd_step(args):
    """Execute a specific workflow step."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized. Run 'cses init' first.")
        return

    step_num = args.step_number
    if step_num not in WORKFLOW_STEPS:
        print(f"Invalid step number: {step_num}")
        print(f"Valid steps: 0-16")
        return

    step_info = WORKFLOW_STEPS[step_num]
    print(f"## Step {step_num}: {step_info['name']}")
    print(f"   {step_info['description']}")
    print()

    # Create executor
    executor = StepExecutor(state)

    # Show guidance
    print(executor.get_step_guidance(step_num))
    print()

    # Confirm execution
    response = input("Execute this step? [Y/n]: ").strip().lower()
    if response == 'n':
        print("Step skipped.")
        return

    # Execute
    print("\nExecuting...")
    result = executor.execute_step(step_num)

    if result.success:
        print(f"\n✅ {result.message}")
    else:
        print(f"\n❌ {result.message}")

    if result.artifacts:
        print(f"\nArtifacts created:")
        for a in result.artifacts:
            print(f"  - {a}")

    if result.issues:
        print(f"\nIssues to address:")
        for i in result.issues:
            print(f"  ⚠️ {i}")

    if result.next_action:
        print(f"\nNext: {result.next_action}")


def cmd_match(args):
    """Run variable matching (Step 7)."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized. Run 'cses init' first.")
        return

    print(f"## Variable Matching: {state.country} {state.year}")
    print()

    if not state.data_file:
        print("❌ No data file found. Check deposit completeness.")
        return

    if not state.questionnaire_files and not state.codebook_file:
        print("❌ No documentation found. Need questionnaire or codebook for matching.")
        return

    print(f"Data file: {state.data_file}")
    print(f"Documentation: {len(state.questionnaire_files)} questionnaire(s)")
    if state.codebook_file:
        print(f"Codebook: {state.codebook_file}")
    print()

    # Ask about dual-model validation
    if args.no_validate:
        validate = False
        print("Running without Claude validation (--no-validate)")
    else:
        print("Dual-model validation will run Claude to verify each mapping.")
        response = input("Enable dual-model validation? [Y/n]: ").strip().lower()
        validate = response != 'n'

    print("\nRunning variable matching...")

    executor = StepExecutor(state)
    result = executor.execute_step(7)

    if result.success:
        print(f"\n✅ {result.message}")

        # Show summary of mappings
        if state.mappings:
            agrees = sum(1 for m in state.mappings if m.get("models_agree", False))
            disagrees = sum(1 for m in state.mappings if m.get("validation_verdict") == "DISAGREE")
            print(f"\nDual-model results:")
            print(f"  - Both agree: {agrees}")
            print(f"  - Disagreements: {disagrees}")

            if disagrees > 0:
                print(f"\n⚠️ {disagrees} mappings need review (models disagreed)")
                print("Use 'cses' interactive mode to review.")
    else:
        print(f"\n❌ {result.message}")

    if result.next_action:
        print(f"\nNext: {result.next_action}")


def cmd_export(args):
    """Export approved mappings."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized. Run 'cses init' first.")
        return

    if not state.mappings:
        print("No mappings to export. Run 'cses match' first.")
        return

    print(f"## Export: {state.country} {state.year}")
    print(f"   {len(state.mappings)} mappings available")
    print()

    # Determine format
    fmt = args.format or "both"

    from src.agent.cses_agent import CSESAgent

    agent = CSESAgent(
        country=state.country,
        year=state.year,
        output_dir=Path(state.working_dir) / ".cses"
    )

    # Create a DualModelResult-like object for export
    from src.agent import DualModelResult
    from src.agent.validator import ValidationResult, ValidationVerdict
    from src.matching.llm_matcher import MatchProposal

    # Reconstruct result for export
    validations = []
    for m in state.mappings:
        proposal = MatchProposal(
            source_variable=m.get("source_variable", "NOT_FOUND"),
            target_variable=m.get("cses_target", ""),
            confidence=float(m.get("original_confidence", 0)),
            confidence_level="high" if float(m.get("original_confidence", 0)) >= 0.85 else "medium",
            reasoning=m.get("original_reasoning", ""),
            matched_by="llm_semantic"
        )

        verdict_str = m.get("validation_verdict", "UNCERTAIN")
        try:
            verdict = ValidationVerdict[verdict_str]
        except KeyError:
            verdict = ValidationVerdict.UNCERTAIN

        validations.append(ValidationResult(
            proposal=proposal,
            verdict=verdict,
            reasoning=m.get("validation_reasoning", ""),
            models_agree=m.get("models_agree", False)
        ))

    result = DualModelResult(
        country=state.country,
        year=state.year,
        session_id=state.session_id,
        extraction=None,
        matching=None,
        validations=validations,
        total_targets=len(validations),
        matched_count=sum(1 for v in validations if v.proposal.source_variable not in ["NOT_FOUND", "ERROR"]),
        agree_count=sum(1 for v in validations if v.verdict == ValidationVerdict.AGREE),
        disagree_count=sum(1 for v in validations if v.verdict == ValidationVerdict.DISAGREE),
        uncertain_count=sum(1 for v in validations if v.verdict == ValidationVerdict.UNCERTAIN)
    )

    if fmt in ["json", "both"]:
        path = agent.export(result, "json")
        print(f"✅ Exported JSON: {path}")

    if fmt in ["xlsx", "both"]:
        path = agent.export(result, "xlsx")
        print(f"✅ Exported Excel: {path}")


def cmd_interactive(args):
    """Start interactive conversation mode."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    print_banner()
    print_validation_status()

    if not state:
        print("No study initialized in this folder.")
        print()
        print(detect_and_summarize(working_dir))
        print()

        response = input("Would you like to initialize a new study? [Y/n]: ").strip().lower()
        if response != 'n':
            # Use init command
            init_args = argparse.Namespace(
                country=None,
                year=None,
                no_organize=False,
                move=False
            )
            cmd_init(init_args)
            state = WorkflowState.load(working_dir)

            if not state:
                print("Initialization failed. Please check the files and try again.")
                return

    if state:
        print(format_workflow_status(state))
        print()

    # Interactive loop
    print("Type 'help' for commands, 'quit' to exit.\n")

    executor = StepExecutor(state) if state else None

    while True:
        try:
            user_input = input("cses> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        cmd = user_input.lower().split()[0]
        rest = user_input[len(cmd):].strip()

        if cmd in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        elif cmd == "help":
            print("""
Available commands:
  status       Show workflow progress
  step N       Work on step N (0-16)
  next         Work on the next pending step
  match        Run variable matching (Step 7)
  export       Export approved mappings
  files        List detected files
  questions    Show pending collaborator questions
  help         Show this help
  quit         Exit

You can also type natural language and I'll try to help!
""")

        elif cmd == "status":
            if state:
                print(format_workflow_status(state))
            else:
                print("No study initialized.")

        elif cmd == "step":
            if not state:
                print("Initialize a study first with 'cses init'")
                continue
            try:
                step_num = int(rest) if rest else state.get_next_step()
                if step_num is None:
                    print("All steps complete!")
                    continue

                step_args = argparse.Namespace(step_number=step_num)
                cmd_step(step_args)
                # Reload state
                state = WorkflowState.load(working_dir)
                executor = StepExecutor(state)
            except ValueError:
                print("Usage: step N (where N is 0-16)")

        elif cmd == "next":
            if not state:
                print("Initialize a study first.")
                continue
            next_step = state.get_next_step()
            if next_step is None:
                print("All steps complete!")
            else:
                step_args = argparse.Namespace(step_number=next_step)
                cmd_step(step_args)
                state = WorkflowState.load(working_dir)
                executor = StepExecutor(state)

        elif cmd == "match":
            if not state:
                print("Initialize a study first.")
                continue
            match_args = argparse.Namespace(no_validate=False)
            cmd_match(match_args)
            state = WorkflowState.load(working_dir)

        elif cmd == "export":
            if not state:
                print("Initialize a study first.")
                continue
            export_args = argparse.Namespace(format="both")
            cmd_export(export_args)

        elif cmd == "files":
            print(detect_and_summarize(working_dir))

        elif cmd == "questions":
            if not state or not state.pending_questions:
                print("No pending questions.")
            else:
                print(f"## Pending Questions ({len(state.pending_questions)})")
                for i, q in enumerate(state.pending_questions, 1):
                    print(f"\n{i}. {q.get('issue', 'N/A')}")
                    print(f"   Status: {q.get('status', 'pending')}")

        else:
            # Treat as natural language - provide guidance
            print(f"I don't have a specific command for '{cmd}'.")
            print("Type 'help' to see available commands.")

            # Suggest based on workflow state
            if state:
                next_step = state.get_next_step()
                if next_step is not None:
                    print(f"\nSuggestion: Work on Step {next_step} ({WORKFLOW_STEPS[next_step]['name']})")
                    print(f"Type 'next' or 'step {next_step}' to continue.")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CSES Data Harmonization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cses              Start interactive mode
  cses init         Initialize study from files in folder
  cses status       Show workflow progress
  cses step 7       Work on variable matching
  cses match        Run variable matching with dual-model validation
  cses export       Export approved mappings
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # init command
    init_parser = subparsers.add_parser("init", help="Initialize a new study")
    init_parser.add_argument("--country", "-c", help="Country name")
    init_parser.add_argument("--year", "-y", help="Election year")
    init_parser.add_argument("--no-organize", action="store_true",
                            help="Don't organize files into folder structure")
    init_parser.add_argument("--move", action="store_true",
                            help="Move files instead of copying")

    # status command
    subparsers.add_parser("status", help="Show workflow status")

    # step command
    step_parser = subparsers.add_parser("step", help="Execute a workflow step")
    step_parser.add_argument("step_number", type=int, help="Step number (0-16)")

    # match command
    match_parser = subparsers.add_parser("match", help="Run variable matching")
    match_parser.add_argument("--no-validate", action="store_true",
                             help="Skip Claude validation")

    # export command
    export_parser = subparsers.add_parser("export", help="Export mappings")
    export_parser.add_argument("--format", "-f", choices=["json", "xlsx", "both"],
                              default="both", help="Export format")

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "step":
        cmd_step(args)
    elif args.command == "match":
        cmd_match(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        # No command - start interactive mode
        cmd_interactive(args)


if __name__ == "__main__":
    main()
