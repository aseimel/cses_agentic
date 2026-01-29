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
import shutil
import sys
import os
import atexit
from pathlib import Path

# Platform-specific locking
if os.name != 'nt':  # Unix/Linux/Mac
    import fcntl

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.workflow.state import (
    WorkflowState,
    StepStatus,
    WORKFLOW_STEPS,
    format_workflow_status
)
from src.workflow.organizer import FileOrganizer, detect_and_summarize, detect_questionnaire_language, parse_country_year_from_folder
from src.workflow.steps import StepExecutor
from src.agent.validator import check_claude_cli_available
from src.agent.conversation import ConversationSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


# Global lock file handle (kept open for duration of process)
_lock_file_handle = None


def get_lock_file_path() -> Path:
    """Get the path to the lock file."""
    if os.name == 'nt':  # Windows
        return Path(os.environ.get('TEMP', '.')) / "cses_agent.lock"
    else:
        return Path("/tmp") / "cses_agent.lock"


def acquire_lock() -> bool:
    """
    Acquire an exclusive lock to prevent parallel execution.

    Returns True if lock acquired, False if another instance is running.
    """
    global _lock_file_handle

    lock_path = get_lock_file_path()

    try:
        # Open lock file (create if doesn't exist)
        _lock_file_handle = open(lock_path, 'w')

        if os.name == 'nt':  # Windows
            import msvcrt
            try:
                msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_NBLCK, 1)
                _lock_file_handle.write(str(os.getpid()))
                _lock_file_handle.flush()
                return True
            except IOError:
                _lock_file_handle.close()
                _lock_file_handle = None
                return False
        else:  # Unix/Linux/Mac
            try:
                fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                _lock_file_handle.write(str(os.getpid()))
                _lock_file_handle.flush()
                return True
            except IOError:
                _lock_file_handle.close()
                _lock_file_handle = None
                return False
    except Exception as e:
        logger.warning(f"Could not acquire lock: {e}")
        return True  # Proceed anyway if lock mechanism fails


def release_lock():
    """Release the lock file."""
    global _lock_file_handle

    if _lock_file_handle:
        try:
            if os.name == 'nt':  # Windows
                import msvcrt
                try:
                    msvcrt.locking(_lock_file_handle.fileno(), msvcrt.LK_UNLCK, 1)
                except:
                    pass
            else:  # Unix/Linux/Mac
                try:
                    fcntl.flock(_lock_file_handle.fileno(), fcntl.LOCK_UN)
                except:
                    pass
            _lock_file_handle.close()
        except:
            pass
        _lock_file_handle = None


def get_install_dir() -> Path:
    """Get the installation directory."""
    # Check common locations
    if os.name == 'nt':  # Windows
        install_dir = Path.home() / ".cses-agent"
    else:
        install_dir = Path.home() / ".cses-agent"

    # Fallback to script location
    if not install_dir.exists():
        install_dir = PROJECT_ROOT

    return install_dir


def first_run_setup(force: bool = False) -> bool:
    """
    First-run setup wizard for GESIS infrastructure.

    Prompts user for:
    - GESIS API key
    - Stata executable path

    Args:
        force: If True, run setup even if already configured

    Returns:
        True if setup completed, False if user cancelled
    """
    install_dir = get_install_dir()
    env_file = install_dir / ".env"

    # Check if already configured via setup wizard (unless forced)
    if not force and env_file.exists():
        try:
            from dotenv import dotenv_values
            config = dotenv_values(env_file)
            # Check for setup wizard marker
            if config.get("CSES_SETUP_COMPLETE") == "true":
                return True  # Setup wizard was completed
            if config.get("OPENAI_API_KEY") and config.get("OPENAI_API_KEY") not in ["your-key-here", ""]:
                # Has API key but no setup marker - prompt to complete setup
                print("\nExisting configuration found but setup was not completed.")
                print("Running setup wizard to verify settings...\n")
                # Continue to setup wizard
        except ImportError:
            # dotenv not available - continue to setup
            pass

    print("""
╔══════════════════════════════════════════════════════════════╗
║           First-Time Setup                                   ║
╚══════════════════════════════════════════════════════════════╝
""")
    print("Welcome! Let's configure CSES Assistant for your environment.\n")

    # GESIS API Key
    print("=" * 60)
    print("STEP 1: GESIS API Key")
    print("=" * 60)
    print("Enter your GESIS OpenWebUI API key.")
    print("(Get it from: https://ai-openwebui.gesis.org)")
    print()

    api_key = input("GESIS API Key: ").strip()
    if not api_key:
        print("\n[X] API key is required. Setup cancelled.")
        return False

    # Stata Path
    print()
    print("=" * 60)
    print("STEP 2: Stata Executable (optional)")
    print("=" * 60)
    print("Enter the full path to your Stata executable.")
    print("This allows the agent to run and debug Stata code.")
    print()
    print("Examples:")
    print("  Windows: C:\\Program Files\\Stata18\\StataMP-64.exe")
    print("  Mac:     /Applications/Stata/StataMP.app/Contents/MacOS/stata-mp")
    print()
    print("Press Enter to skip if you don't have Stata installed.")
    print()

    stata_path = input("Stata path: ").strip()

    # Validate Stata path if provided
    if stata_path:
        stata_path_obj = Path(stata_path)
        if not stata_path_obj.exists():
            print(f"\n[!] Warning: File not found: {stata_path}")
            confirm = input("Save anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                stata_path = ""

    # Write .env file
    print()
    print("Saving configuration...")

    env_content = f"""# CSES Assistant Configuration
# Generated by first-run setup

# GESIS OpenWebUI API
OPENAI_API_KEY={api_key}
OPENAI_API_BASE=https://ai-openwebui.gesis.org/api/v1

# LLM Models
LLM_MODEL_MATCH=openai/gpt-oss:120b
LLM_MODEL_PREPROCESS=openai/llama4:latest
LLM_MODEL_VALIDATE=openai/gpt-oss:120b
"""

    if stata_path:
        env_content += f"""
# Stata executable path
STATA_PATH={stata_path}
"""

    # Add setup complete marker
    env_content += """
# Setup wizard completed
CSES_SETUP_COMPLETE=true
"""

    # Save to install directory
    install_dir.mkdir(parents=True, exist_ok=True)
    with open(env_file, 'w') as f:
        f.write(env_content)

    print(f"[OK] Configuration saved to: {env_file}")
    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print("You can edit these settings later by running:")
    print(f"  notepad {env_file}" if os.name == 'nt' else f"  nano {env_file}")
    print()

    input("Press Enter to continue...")
    return True


def print_banner():
    """Print welcome banner."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║           CSES Assistant                                      ║
║           Module 6 Processing Workflow                       ║
╚══════════════════════════════════════════════════════════════╝
""")


def check_validation_setup() -> tuple[bool, str]:
    """
    Check validation model configuration.

    Returns:
        Tuple of (is_ready: bool, status_message: str)
    """
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not available, env vars should be loaded by main()

    validation_model = os.getenv("LLM_MODEL_VALIDATE", "")

    if validation_model == "claude-cli":
        # Using Claude CLI with Max subscription
        available, msg = check_claude_cli_available()
        if available:
            return True, f"[OK] Validation: Claude CLI (Max subscription)\n   {msg}"
        else:
            return False, f"[!] Validation: Claude CLI not ready\n   {msg}\n   Run 'claude login' to authenticate"

    elif validation_model:
        # Using API model
        return True, f"[OK] Validation: {validation_model} (API)"

    else:
        return False, "[!] Validation model not configured. Set LLM_MODEL_VALIDATE in .env"


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


def cmd_init(args) -> Path:
    """
    Initialize a new study from files in the current folder.

    Expects to be run in a country folder (e.g., SouthKorea_2024/) that has
    an email subfolder (emails/, E-mails/, etc.) containing deposited files.

    Returns:
        Path to the study directory (for use by cmd_interactive)
    """
    working_dir = Path.cwd()

    # Check if already initialized
    existing_state = WorkflowState.load(working_dir)
    if existing_state:
        return Path(existing_state.working_dir)

    # Also check subdirectories
    for subdir in working_dir.iterdir():
        if subdir.is_dir() and (subdir / ".cses").exists():
            existing_state = WorkflowState.load(subdir)
            if existing_state:
                return subdir

    print("Scanning for email folder...")

    organizer = FileOrganizer(working_dir)

    # Find email folder with data deposit
    email_folder = organizer.find_email_folder()

    if not email_folder:
        print("\n[X] No email folder found with data files.")
        print("   Expected structure:")
        print("     SouthKorea_2024/")
        print("       emails/           <- or 'E-mails', 'email', etc.")
        print("         20250303/       <- subfolder with deposited files")
        print("           data.dta")
        print("           questionnaire.pdf")
        return None

    print(f"Found email folder: {email_folder.name}/")

    # Detect files recursively in the email folder (scan all subfolders)
    print("Scanning all email subfolders for files...")
    detected = organizer.detect_files(source_dir=email_folder, recursive=True)

    if not detected.data_files:
        print("\n[X] No data files found in deposit folder")
        return None

    # Parse country/year from folder name or use args
    folder_country, folder_year = parse_country_year_from_folder(working_dir.name)
    country = args.country or folder_country
    year = args.year or folder_year or detected.year

    if not country:
        country = input("\nCountry name: ").strip()
        if not country:
            print("Country name is required.")
            return None

    if not year:
        year = input("Election year: ").strip()
        if not year:
            print("Election year is required.")
            return None

    # Get country code
    country_code = organizer.COUNTRY_CODES.get(country.lower(), country[:3].upper())

    print(f"\nSetting up: {country} {year}")
    print(f"  File prefix: {country_code}_{year}")
    print(f"\nDetected files in {email_folder.name}/:")
    print(f"  Data: {len(detected.data_files)} file(s)")
    print(f"  Questionnaires: {len(detected.questionnaire_files)} file(s)")
    print(f"  Codebooks: {len(detected.codebook_files)} file(s)")
    print(f"  Design reports: {len(detected.design_report_files)} file(s)")
    print(f"  Macro reports: {len(detected.macro_report_files)} file(s)")

    # Create CSES structure at root level (alongside email folder)
    print("\nCreating folder structure...")
    organizer.create_study_structure(working_dir)

    # Copy files to micro/ with standardized names
    print("Copying files with standardized names...")
    file_mapping = organizer.copy_files_with_standard_names(
        detected, email_folder, working_dir, country_code, year
    )

    study_dir = working_dir

    # Create workflow state with paths to standardized copies
    state = WorkflowState(
        country=country,
        country_code=country_code,
        year=year,
        working_dir=str(study_dir)
    )

    # Store paths to standardized copies in micro/
    if "data" in file_mapping:
        state.data_file = file_mapping["data"]

    # Collect all questionnaire paths
    questionnaire_paths = [v for k, v in file_mapping.items() if k.startswith("questionnaire")]
    if questionnaire_paths:
        state.questionnaire_files = questionnaire_paths

    if "codebook" in file_mapping:
        state.codebook_file = file_mapping["codebook"]

    if "design_report" in file_mapping:
        state.design_report_file = file_mapping["design_report"]

    print(f"[OK] Files copied to: micro/")
    print(f"   Email folder preserved: {email_folder.name}/ (raw backup)")

    # Mark Step 0 as complete
    state.set_step_status(0, StepStatus.COMPLETED, "Folder initialized (CSES standard)")

    # Save state first
    state.save(study_dir / ".cses")

    # Initialize active logging (creates log and questions files)
    print("Initializing log files...")
    from src.workflow.active_logging import ActiveLogger
    active_logger = ActiveLogger(state)
    active_logger.log_message(f"Study initialized: {country} {year}")
    active_logger.log_message(f"Session ID: {state.session_id}")

    # Save state again with log file paths
    state.save(study_dir / ".cses")

    print(f"\n[OK] Study initialized: {country} {year}")
    print(f"   Session ID: {state.session_id}")
    print(f"   Working directory: {study_dir}")

    return study_dir


def cmd_status(args):
    """Show workflow status."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized in this folder.")
        print("Run 'cses init' to start a new study.")
        return

    print(format_workflow_status(state))

    # Show log file info
    if state.log_file:
        log_path = Path(state.log_file)
        if log_path.exists():
            print(f"\nLog file: {log_path.relative_to(Path(state.working_dir))}")

    # Show collaborator questions info
    pending = state.get_pending_questions()
    if pending:
        print(f"\nPending collaborator questions: {len(pending)}")
        for q in pending[:3]:  # Show first 3
            print(f"  - {q.get('id', '')}: {q.get('question', '')[:50]}...")
        if len(pending) > 3:
            print(f"  ... and {len(pending) - 3} more")

    if state.collaborator_questions_file:
        questions_path = Path(state.collaborator_questions_file)
        if questions_path.exists():
            print(f"Questions file: {questions_path.relative_to(Path(state.working_dir))}")

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
        print(f"\n[OK] {result.message}")
    else:
        print(f"\n[X] {result.message}")

    if result.artifacts:
        print(f"\nArtifacts created:")
        for a in result.artifacts:
            print(f"  - {a}")

    if result.issues:
        print(f"\nIssues to address:")
        for i in result.issues:
            print(f"  [!] {i}")

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
        print("[X] No data file found. Check deposit completeness.")
        return

    if not state.questionnaire_files and not state.codebook_file:
        print("[X] No documentation found. Need questionnaire or codebook for matching.")
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
        print(f"\n[OK] {result.message}")

        # Show summary of mappings
        if state.mappings:
            agrees = sum(1 for m in state.mappings if m.get("models_agree", False))
            disagrees = sum(1 for m in state.mappings if m.get("validation_verdict") == "DISAGREE")
            print(f"\nDual-model results:")
            print(f"  - Both agree: {agrees}")
            print(f"  - Disagreements: {disagrees}")

            if disagrees > 0:
                print(f"\n[!] {disagrees} mappings need review (models disagreed)")
                print("Use 'cses' interactive mode to review.")
    else:
        print(f"\n[X] {result.message}")

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
        print(f"[OK] Exported JSON: {path}")

    if fmt in ["xlsx", "both"]:
        path = agent.export(result, "xlsx")
        print(f"[OK] Exported Excel: {path}")


def cmd_interactive(args):
    """Start interactive conversation mode with Claude."""
    working_dir = Path.cwd()
    study_dir = None
    state = None

    print_banner()

    # Try to find existing state
    state = WorkflowState.load(working_dir)
    if state:
        study_dir = Path(state.working_dir)
    else:
        # Check for study folders in current directory
        for subdir in working_dir.iterdir():
            if subdir.is_dir() and (subdir / ".cses").exists():
                state = WorkflowState.load(subdir)
                if state:
                    study_dir = subdir
                    print(f"Found existing study: {state.country} {state.year}\n")
                    break

    if not state:
        print("No study initialized in this folder.\n")

        # Check for email folder with data
        organizer = FileOrganizer(working_dir)
        email_folder = organizer.find_email_folder()

        if email_folder:
            detected = organizer.detect_files(source_dir=email_folder, recursive=True)
            print(f"Found email folder: {email_folder.name}/")
            print(f"  Data files: {len(detected.data_files)}")
            print(f"  Questionnaires: {len(detected.questionnaire_files)}")
            print(f"  Codebooks: {len(detected.codebook_files)}")
            print(f"  Design reports: {len(detected.design_report_files)}")
            print(f"  Macro reports: {len(detected.macro_report_files)}")
            print()
        else:
            print("[X] No email folder found with data files.")
            print("   Run this in a country folder (e.g., SouthKorea_2024/)")
            print("   that contains an email subfolder with deposited files.")
            return

        response = input("Initialize a new study? [Y/n]: ").strip().lower()
        if response == 'n':
            print("\nExiting.")
            return

        # Initialize
        init_args = argparse.Namespace(
            country=None,
            year=None,
            no_organize=False,
            move=False
        )
        study_dir = cmd_init(init_args)

        if study_dir:
            state = WorkflowState.load(study_dir)

        if not state:
            print("\nInitialization failed.")
            return

    # Show current status
    print()
    print(format_workflow_status(state))
    print()

    # Start conversational mode with Claude
    print("=" * 60)
    print("CSES Expert Assistant")
    print("=" * 60)
    print()
    print("I'm here to guide you through the CSES data processing workflow.")
    print("Ask me anything about the process, or tell me what you'd like to do.")
    print()
    print("Type 'quit' to exit, 'status' to see progress.")
    print()

    # Create conversation session
    conversation = ConversationSession(state)

    # Send initial greeting to get Claude's guidance
    next_step = state.get_next_step()
    if next_step is not None:
        step_info = WORKFLOW_STEPS[next_step]
        initial_prompt = f"The study is initialized. The next step is Step {next_step}: {step_info['name']}. Briefly explain what this step involves and ask if the user is ready to proceed."
        print("Connecting to Claude...")
        try:
            response = conversation.send(initial_prompt)
            print()
            print(f"Assistant: {response}")
            print()
        except Exception as e:
            print(f"Note: Could not connect to Claude ({e})")
            print(f"Next step: Step {next_step} - {step_info['name']}")
            print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # Handle special commands
        cmd_lower = user_input.lower()

        if cmd_lower in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        elif cmd_lower == "status":
            conversation.refresh_state()
            print()
            print(format_workflow_status(conversation.state))
            print()
            continue

        elif cmd_lower == "help":
            print("""
Just chat naturally! You can ask things like:
  - "What should I do next?"
  - "Explain step 3"
  - "I'm having trouble with the questionnaire"
  - "What are the CSES coding conventions?"
  - "Proceed with this step"
  - "Skip this step"

Commands: status, quit, help
""")
            continue

        # Send everything else to Claude
        try:
            print()
            print("Thinking...")
            response = conversation.send(user_input)
            print()
            print(f"Assistant: {response}")
            print()
        except Exception as e:
            print(f"Error: {e}")
            print()


def cmd_setup(args):
    """Re-run the setup wizard."""
    install_dir = get_install_dir()
    env_file = install_dir / ".env"

    if env_file.exists() and not args.force:
        print(f"Configuration already exists at: {env_file}")
        print("Use 'cses setup --force' to reconfigure.")
        return

    # Run setup with force flag
    first_run_setup(force=True)


def cmd_migrate(args):
    """
    Migrate an old folder structure to CSES standard.

    This command:
    1. Moves original_deposit/ -> micro/original_deposit/
    2. Creates missing standard folders
    3. Updates state.json paths if needed
    """
    working_dir = Path.cwd()

    print("CSES Folder Migration")
    print("=" * 40)
    print()

    # Check if this is a study folder
    state = WorkflowState.load(working_dir)
    if not state:
        # Check subdirectories
        for subdir in working_dir.iterdir():
            if subdir.is_dir() and (subdir / ".cses").exists():
                state = WorkflowState.load(subdir)
                if state:
                    working_dir = subdir
                    break

    if not state:
        print("[X] No study found in this folder.")
        print("   Run 'cses init' first to initialize a study.")
        return

    print(f"Study: {state.country} {state.year}")
    print(f"Directory: {working_dir}")
    print()

    # Check for old structure indicators
    old_deposit = working_dir / "original_deposit"
    new_deposit = working_dir / "micro" / "original_deposit"

    needs_migration = False
    if old_deposit.exists() and not new_deposit.exists():
        print("  [!] Found: original_deposit/ at root (should be micro/original_deposit/)")
        needs_migration = True

    if not needs_migration:
        # Check for missing folders
        missing = []
        for folder in ["micro/original_deposit", "micro/FINAL dataset",
                       "micro/deposited variable list", "macro", "Election Results"]:
            if not (working_dir / folder).exists():
                missing.append(folder)
        if missing:
            print(f"  [!] Missing folders: {', '.join(missing)}")
            needs_migration = True

    if not needs_migration:
        print("[OK] Folder structure is already CSES standard.")
        return

    print()
    response = input("Migrate to CSES standard structure? [Y/n]: ").strip().lower()
    if response == 'n':
        print("Migration cancelled.")
        return

    print()
    print("Migrating...")

    # Perform migration
    organizer = FileOrganizer(working_dir)
    results = organizer.migrate_old_structure(working_dir)

    # Update state file paths if files were moved
    paths_updated = False
    if results["moved_files"] or results["renamed_folders"]:
        # Update data_file path if it was in old location
        if state.data_file:
            old_path = Path(state.data_file)
            if "original_deposit" in str(old_path) and "micro" not in str(old_path):
                new_path = working_dir / "micro" / "original_deposit" / old_path.name
                if new_path.exists():
                    state.data_file = str(new_path)
                    paths_updated = True

        # Update questionnaire paths
        if state.questionnaire_files:
            new_questionnaires = []
            for qf in state.questionnaire_files:
                old_path = Path(qf)
                if "original_deposit" in str(old_path) and "micro" not in str(old_path):
                    new_path = working_dir / "micro" / "original_deposit" / old_path.name
                    if new_path.exists():
                        new_questionnaires.append(str(new_path))
                        paths_updated = True
                    else:
                        new_questionnaires.append(qf)
                else:
                    new_questionnaires.append(qf)
            state.questionnaire_files = new_questionnaires

        # Update codebook path
        if state.codebook_file:
            old_path = Path(state.codebook_file)
            if "original_deposit" in str(old_path) and "micro" not in str(old_path):
                new_path = working_dir / "micro" / "original_deposit" / old_path.name
                if new_path.exists():
                    state.codebook_file = str(new_path)
                    paths_updated = True

        # Update design report path
        if state.design_report_file:
            old_path = Path(state.design_report_file)
            if "original_deposit" in str(old_path) and "micro" not in str(old_path):
                new_path = working_dir / "micro" / "original_deposit" / old_path.name
                if new_path.exists():
                    state.design_report_file = str(new_path)
                    paths_updated = True

        if paths_updated:
            state.save()

    # Print results
    print()
    if results["renamed_folders"]:
        print("Moved/renamed folders:")
        for item in results["renamed_folders"]:
            print(f"  - {item}")

    if results["moved_files"]:
        print("Moved files:")
        for item in results["moved_files"]:
            print(f"  - {item}")

    if results["created_folders"]:
        print("Created folders:")
        for item in results["created_folders"]:
            print(f"  - {item}/")

    if paths_updated:
        print("Updated file paths in state.json")

    if results["errors"]:
        print("Errors:")
        for item in results["errors"]:
            print(f"  [!] {item}")

    print()
    print("[OK] Migration complete.")


def cmd_update(args):
    """Update CSES Assistant from GitHub by running the install script."""
    import subprocess
    import tempfile
    import urllib.request

    print("CSES Assistant Update")
    print("=" * 40)
    print()

    # CRITICAL: Release lock before running installer
    # The installer will run as a separate process and needs the lock
    print("Preparing for update...")
    release_lock()

    if os.name == 'nt':
        # Windows - download and run PowerShell install script
        script_url = "https://raw.githubusercontent.com/aseimel/cses_agentic/main/install.ps1"
        print("Downloading update script...")

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ps1', delete=False, encoding='utf-8') as tmp:
                response = urllib.request.urlopen(script_url)
                tmp.write(response.read().decode('utf-8'))
                script_path = tmp.name
        except Exception as e:
            print(f"[X] Download failed: {e}")
            return

        print("Running update...")
        print("(This window will close after update completes)")
        print()

        try:
            # Run PowerShell script - this replaces the current installation
            # so we exit immediately after starting it
            result = subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", script_path],
                check=False
            )
            if result.returncode != 0:
                print(f"[!] Update script returned error code {result.returncode}")
        except Exception as e:
            print(f"[X] Failed to run update script: {e}")
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
        # Exit after update to avoid running old code
        sys.exit(0)
    else:
        # Linux/Mac - download and run bash install script
        script_url = "https://raw.githubusercontent.com/aseimel/cses_agentic/main/install.sh"
        print("Downloading update script...")

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False, encoding='utf-8') as tmp:
                response = urllib.request.urlopen(script_url)
                tmp.write(response.read().decode('utf-8'))
                script_path = tmp.name
        except Exception as e:
            print(f"[X] Download failed: {e}")
            return

        print("Running update...")
        print("(This window will close after update completes)")
        print()

        try:
            os.chmod(script_path, 0o755)
            result = subprocess.run(["bash", script_path], check=False)
            if result.returncode != 0:
                print(f"[!] Update script returned error code {result.returncode}")
        except Exception as e:
            print(f"[X] Failed to run update script: {e}")
        finally:
            try:
                os.unlink(script_path)
            except:
                pass
        # Exit after update to avoid running old code
        sys.exit(0)


def main():
    """Main entry point."""
    # CRITICAL: Acquire lock to prevent parallel execution
    # This prevents the CLI and install script from running simultaneously
    if not acquire_lock():
        print("\n[X] Another CSES instance is already running.")
        print("    Please wait for it to finish or close it first.")
        print()
        print("    If you believe this is an error, delete the lock file:")
        print(f"    {get_lock_file_path()}")
        print()
        input("Press Enter to exit...")
        sys.exit(1)

    # Register cleanup on exit
    atexit.register(release_lock)

    try:
        # Load environment from install directory
        install_dir = get_install_dir()
        env_file = install_dir / ".env"
        if env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(env_file)
            except ImportError:
                print("Warning: python-dotenv not installed, skipping .env loading")
            except Exception as e:
                print(f"Warning: Could not load .env file: {e}")
    except Exception as e:
        print(f"Error during initialization: {e}")
        release_lock()
        input("Press Enter to exit...")
        sys.exit(1)

    parser = argparse.ArgumentParser(
        description="CSES Data Harmonization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cses              Start interactive mode
  cses status       Show workflow progress
  cses step 7       Work on variable matching
  cses match        Run variable matching with dual-model validation
  cses export       Export approved mappings
  cses migrate      Migrate old folder structure to CSES standard
  cses setup        Re-run initial configuration
  cses update       Update to latest version from GitHub
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # setup command
    setup_parser = subparsers.add_parser("setup", help="Configure API keys and Stata path")
    setup_parser.add_argument("--force", "-f", action="store_true",
                             help="Force reconfiguration even if already set up")

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

    # migrate command
    subparsers.add_parser("migrate", help="Migrate old folder structure to CSES standard")

    # update command
    subparsers.add_parser("update", help="Update to latest version from GitHub")

    args = parser.parse_args()

    # Check for first-run setup (except for setup and update commands)
    if args.command not in ["setup", "update"]:
        try:
            if not first_run_setup():
                print("\nSetup cancelled or incomplete.")
                input("Press Enter to exit...")
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\nSetup interrupted.")
            sys.exit(1)
        except Exception as e:
            print(f"\nSetup error: {e}")
            import traceback
            traceback.print_exc()
            input("Press Enter to exit...")
            sys.exit(1)

    try:
        if args.command == "setup":
            cmd_setup(args)
        elif args.command == "update":
            cmd_update(args)
        elif args.command == "init":
            cmd_init(args)
        elif args.command == "status":
            cmd_status(args)
        elif args.command == "step":
            cmd_step(args)
        elif args.command == "match":
            cmd_match(args)
        elif args.command == "export":
            cmd_export(args)
        elif args.command == "migrate":
            cmd_migrate(args)
        else:
            # No command - start interactive mode
            cmd_interactive(args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)
