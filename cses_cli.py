#!/usr/bin/env python3
"""
CSES Data Harmonization CLI.

A command-line tool for processing CSES Module 6 election studies.
Run in a folder containing collaborator files to start the workflow.

Usage:
    cses              # Start conversational assistant
    cses init         # Initialize new study from files in folder
    cses status       # Show workflow status
    cses step N       # Execute step N
    cses step 7a      # AI fills tracking sheet with proposals
    cses step 7c      # Generate Stata code from tracking sheet
    cses match        # Run variable matching (Step 7)
    cses generate     # Generate Stata code (alias for step 7c)
    cses export       # Export approved mappings
"""

import argparse
import logging
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
from src.agent.conversation import ConversationSession
from src.ui_text import sanitize_processor_text

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)

# Suppress LiteLLM and HTTP library noise (prevents API call overlays in TUI)
import os
os.environ["LITELLM_LOG"] = "ERROR"  # Suppress litellm info/debug output
os.environ["HTTPX_LOG_LEVEL"] = "WARNING"  # Suppress httpx output
try:
    import litellm
    litellm.suppress_debug_info = True
    litellm.set_verbose = False
except ImportError:
    pass
logging.getLogger("litellm").setLevel(logging.ERROR)
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)


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


def pause_before_exit(prompt: str = "Press Enter to exit..."):
    """Pause only when running in an interactive console."""
    if sys.stdin and sys.stdin.isatty():
        input(prompt)


def first_run_setup(force: bool = False) -> bool:
    """
    First-run setup wizard.

    Prompts user for:
    - API key
    - optional OpenWebUI/OpenAI-compatible base URL
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
            supported_api_keys = [
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "XAI_API_KEY",
                "GEMINI_API_KEY",
            ]
            has_api_key = any(
                config.get(key) and config.get(key) not in ["your-key-here", ""]
                for key in supported_api_keys
            )
            if has_api_key:
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

    # API Key
    print("=" * 60)
    print("STEP 1: API Key")
    print("=" * 60)
    print("Enter an API key for the provider or service you will use.")
    print("For direct OpenAI, use an OpenAI API key.")
    print("For OpenWebUI, use the key from that OpenWebUI service.")
    print()

    api_key = input("API Key: ").strip()
    if not api_key:
        print("\n[X] API key is required. Setup cancelled.")
        return False

    # Optional OpenAI-compatible service
    print()
    print("=" * 60)
    print("STEP 2: OpenWebUI / OpenAI-compatible service (optional)")
    print("=" * 60)
    print("Use this only if your team provides an OpenWebUI or other OpenAI-compatible endpoint.")
    print("Leave blank for direct vendor APIs.")
    print("Example: https://your-openwebui.example.org/api/v1")
    print()

    api_base = input("OpenAI-compatible base URL: ").strip()

    # Stata Path
    print()
    print("=" * 60)
    print("STEP 3: Stata Executable (optional)")
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

# API key used by LiteLLM
OPENAI_API_KEY={api_key}
"""

    if api_base:
        env_content += f"""
# Optional OpenWebUI / OpenAI-compatible API routing
CSES_USE_OPENWEBUI=true
OPENAI_API_BASE={api_base}
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
    # Model settings are hardcoded in src/config.py - always ready
    from src.config import LLM_MODEL_VALIDATE
    return True, f"[OK] Validation: {LLM_MODEL_VALIDATE}"


def print_validation_status():
    """Print validation configuration status."""
    ready, message = check_validation_setup()
    print(message)
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

    print("Scanning for email folder...")

    organizer = FileOrganizer(working_dir)
    folder_check = organizer.validate_study_folder()
    if not folder_check.ok:
        print("\n[X] This folder cannot be initialized as a CSES study.")
        print(f"   {folder_check.message}")
        for detail in folder_check.details:
            print(f"   {detail}")
        print("   Please run this command from one specific study folder.")
        return None

    # Find email folder with data deposit
    email_folder = folder_check.email_folder or organizer.find_email_folder()

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

    # Build an input manifest immediately so chat/status can explain which
    # deposited data version was selected and which older versions were preserved.
    try:
        from src.workflow.input_manifest import InputManifestBuilder

        manifest_builder = InputManifestBuilder(study_dir)
        manifest = manifest_builder.build()
        manifest_path = manifest_builder.write(manifest)
        state.input_manifest_path = str(manifest_path)
        state.primary_input_selection = {
            "primary_data_file": manifest.primary_data_file,
            "reason": manifest.primary_data_reason,
            "warnings": manifest.warnings,
        }
        if manifest.primary_data_file:
            state.data_file = manifest.primary_data_file
    except Exception as exc:
        print(f"  [!] Input manifest could not be created: {exc}")

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

    # Copy blank variable tracking template (CSES standard format)
    tracking_sheet_path = organizer.copy_variable_tracking_template(study_dir, country_code, year)
    if tracking_sheet_path:
        state.variable_tracking_file = str(tracking_sheet_path)

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

    # CRITICAL: Fill deposit inventory with CURATED filenames (standardized names in micro/)
    # Use the state's file paths which point to the renamed files, not the original deposit
    curated_data = [state.data_file] if state.data_file else []
    curated_questionnaires = state.questionnaire_files or []
    curated_codebooks = [state.codebook_file] if state.codebook_file else []
    curated_design_reports = [state.design_report_file] if state.design_report_file else []
    curated_macro_reports = [file_mapping.get("macro_report")] if file_mapping.get("macro_report") else []

    active_logger.update_deposit_inventory(
        data_files=curated_data,
        questionnaires=curated_questionnaires,
        codebooks=curated_codebooks,
        design_reports=curated_design_reports,
        macro_reports=curated_macro_reports
    )
    print("  Log file updated with curated file inventory")

    # Save state again with log file paths
    state.save(study_dir / ".cses")

    print(f"\n[OK] Study initialized: {country} {year}")
    print(f"   Session ID: {state.session_id}")
    print(f"   Working directory: {study_dir}")

    return study_dir


def cmd_status(args):
    """Show detailed workflow status."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        # Check subdirectories for study
        for subdir in working_dir.iterdir():
            if subdir.is_dir() and (subdir / ".cses").exists():
                state = WorkflowState.load(subdir)
                if state:
                    working_dir = subdir
                    break

    if not state:
        print("No study initialized in this folder.")
        print("Run 'cses init' to start a new study.")
        return

    print()
    print(f"CSES Study Status: {state.country} {state.year}")
    print("=" * 60)
    print()

    # Show all steps with their status
    print("Workflow Steps:")
    for step_num in sorted(WORKFLOW_STEPS.keys()):
        step = state.get_step(step_num)
        step_info = WORKFLOW_STEPS[step_num]
        if step.status == StepStatus.COMPLETED:
            status_mark = "[OK]"
        elif step.status == StepStatus.IN_PROGRESS:
            status_mark = "[..]"
        else:
            status_mark = "[  ]"
        print(f"  {status_mark} Step {step_num}: {step_info['name']}")
        if step.notes:
            print(f"       {step.notes[-1][:60]}..." if step.notes else "")

    print()

    # Show files
    print("Files:")
    if state.data_file:
        print(f"  Data: {state.data_file}")
    if state.questionnaire_files:
        for qf in state.questionnaire_files:
            print(f"  Questionnaire: {qf}")
    if state.codebook_file:
        print(f"  Codebook: {state.codebook_file}")
    if state.design_report_file:
        print(f"  Design Report: {state.design_report_file}")
    if state.variable_tracking_file:
        print(f"  Tracking Sheet: {state.variable_tracking_file}")

    # Show log file info
    if state.log_file:
        print(f"  Log: {state.log_file}")

    print()

    # Show collaborator questions info
    pending = state.get_pending_questions()
    if pending:
        print(f"Pending Collaborator Questions: {len(pending)}")
        for q in pending[:5]:
            print(f"  - {q.get('id', '')}: {q.get('question', '')[:60]}...")
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more")
        print()

    # Show next action
    next_step = state.get_next_step()
    if next_step is not None:
        step_info = WORKFLOW_STEPS[next_step]
        print(f"Next: Step {next_step} - {step_info['name']}")
        print(f"  {step_info['description']}")
        print()
        print(f"Run: cses step {next_step}")
    else:
        print("All steps completed!")


def cmd_step(args):
    """Execute a specific workflow step."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized. Run 'cses init' first.")
        return

    step_arg = args.step_number

    # Handle step variants (7a, 7c)
    step_variant = None
    if isinstance(step_arg, str) and step_arg[-1].isalpha():
        step_variant = step_arg[-1].lower()
        step_num = int(step_arg[:-1])
    else:
        step_num = int(step_arg)

    # Validate step number
    if step_num not in WORKFLOW_STEPS:
        print(f"Invalid step number: {step_num}")
        print(f"Valid steps: 0-16, or 7a/7c for sub-steps")
        return

    # Create executor
    executor = StepExecutor(state)

    # Handle step 7 variants
    if step_num == 7 and step_variant:
        if step_variant == 'a':
            print()
            print("Step 7a: AI Variable Matching")
            print("=" * 40)
            print()
            result = executor._step_7a()
        elif step_variant == 'c':
            print()
            print("Step 7c: Generate Stata Code")
            print("=" * 40)
            print()
            result = executor._step_7c()
        else:
            print(f"Invalid step variant: 7{step_variant}")
            print("Valid variants: 7a (AI fill sheet), 7c (generate code)")
            return
    else:
        step_info = WORKFLOW_STEPS[step_num]
        print()
        print(f"Step {step_num}: {step_info['name']}")
        print("=" * 40)
        print()

        # Execute step
        result = executor.execute_step(step_num)

    # Show results
    print()
    if result.success:
        print(f"[OK] {sanitize_processor_text(result.message)}")
    else:
        print(f"[X] {sanitize_processor_text(result.message)}")

    visible_outputs = [a for a in result.artifacts if ".cses" not in Path(a).parts]
    if visible_outputs:
        print()
        print("Output:")
        for a in visible_outputs:
            # Show relative path if possible
            try:
                rel_path = Path(a).relative_to(working_dir)
                print(f"  {rel_path}")
            except ValueError:
                print(f"  {a}")

    if result.issues:
        print()
        print("Issues:")
        for i in result.issues:
            text = sanitize_processor_text(i)
            if text:
                print(f"  [!] {text}")

    if result.next_action:
        print()
        print(f"Next: {sanitize_processor_text(result.next_action)}")


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
        print("Running without LLM validation (--no-validate)")
    else:
        print("Dual-model validation will run the validation LLM to verify each mapping.")
        response = input("Enable dual-model validation? [Y/n]: ").strip().lower()
        validate = response != 'n'

    print("\nRunning variable matching...")

    executor = StepExecutor(state)
    result = executor.execute_step(7)

    if result.success:
        print(f"\n[OK] {sanitize_processor_text(result.message)}")

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
        print(f"\n[X] {sanitize_processor_text(result.message)}")

    if result.next_action:
        print(f"\nNext: {sanitize_processor_text(result.next_action)}")


def cmd_generate(args):
    """Generate Stata code from tracking sheet (Step 7c)."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        print("No study initialized. Run 'cses init' first.")
        return

    print(f"## Code Generation: {state.country} {state.year}")
    print()

    # Check for tracking sheet
    tracking_sheet = args.sheet if hasattr(args, 'sheet') and args.sheet else None

    # Use the step executor to run step 7c directly
    executor = StepExecutor(state)

    # Call _step_7c directly since it's a sub-step
    result = executor._step_7c(tracking_sheet=tracking_sheet)

    if result.success:
        print(f"\n[OK] {sanitize_processor_text(result.message)}")
        state.save()
    else:
        print(f"\n[X] {sanitize_processor_text(result.message)}")

    visible_outputs = [a for a in result.artifacts if ".cses" not in Path(a).parts]
    if visible_outputs:
        print(f"\nGenerated files:")
        for a in visible_outputs:
            print(f"  - {a}")

    if result.issues:
        print(f"\nWarnings:")
        for i in result.issues[:5]:
            text = sanitize_processor_text(i)
            if text:
                print(f"  [!] {text}")

    if result.next_action:
        print(f"\nNext: {sanitize_processor_text(result.next_action)}")


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


def cmd_default(args):
    """Default command - show status and suggest next step."""
    working_dir = Path.cwd()
    state = None

    # Try to find existing state
    state = WorkflowState.load(working_dir)
    if not state:
        # Check for study folders in current directory
        for subdir in working_dir.iterdir():
            if subdir.is_dir() and (subdir / ".cses").exists():
                state = WorkflowState.load(subdir)
                if state:
                    working_dir = subdir
                    break

    if not state:
        # No study found - check for files to initialize
        print_banner()
        print("No study initialized in this folder.\n")

        organizer = FileOrganizer(working_dir)
        email_folder = organizer.find_email_folder()

        if email_folder:
            detected = organizer.detect_files(source_dir=email_folder, recursive=True)
            print(f"Found email folder: {email_folder.name}/")
            print(f"  Data files: {len(detected.data_files)}")
            print(f"  Questionnaires: {len(detected.questionnaire_files)}")
            print(f"  Codebooks: {len(detected.codebook_files)}")
            print(f"  Design reports: {len(detected.design_report_files)}")
            print()
            print("Run 'cses init' to initialize this study.")
        else:
            print("No email folder found with data files.")
            print("Run this in a country folder that contains an email subfolder with deposited files.")
            print()
            print("Expected structure:")
            print("  CountryName_Year/")
            print("    emails/           <- or 'E-mails', 'email', etc.")
            print("      deposit_date/   <- subfolder with deposited files")
            print("        data.dta")
            print("        questionnaire.pdf")
        return

    # Show status for existing study
    print()
    print(f"CSES Assistant - {state.country} {state.year}")
    print("=" * 50)
    print()

    # Show step status summary
    completed = []
    in_progress = []
    pending = []
    for step_num in sorted(WORKFLOW_STEPS.keys()):
        step = state.get_step(step_num)
        step_info = WORKFLOW_STEPS[step_num]
        if step.status == StepStatus.COMPLETED:
            completed.append(step_num)
        elif step.status == StepStatus.IN_PROGRESS:
            in_progress.append(step_num)
        else:
            pending.append(step_num)

    print("Status:")
    if completed:
        for s in completed[-3:]:  # Show last 3 completed
            print(f"  [OK] Step {s}: {WORKFLOW_STEPS[s]['name']}")
    if in_progress:
        for s in in_progress:
            print(f"  [..] Step {s}: {WORKFLOW_STEPS[s]['name']} (in progress)")
    if pending and not in_progress:
        next_step = pending[0]
        print(f"  [--] Step {next_step}: {WORKFLOW_STEPS[next_step]['name']} (next)")

    print()

    # Show key files
    print("Files:")
    if state.data_file:
        data_path = Path(state.data_file)
        print(f"  Data: {data_path.name}")
    if state.questionnaire_files:
        for qf in state.questionnaire_files[:2]:
            print(f"  Questionnaire: {Path(qf).name}")
    if state.codebook_file:
        print(f"  Codebook: {Path(state.codebook_file).name}")
    if state.variable_tracking_file:
        tracking_path = Path(state.variable_tracking_file)
        if tracking_path.exists():
            print(f"  Tracking sheet: {tracking_path.name}")

    print()

    # Suggest next action
    next_step = state.get_next_step()
    if next_step is not None:
        step_info = WORKFLOW_STEPS[next_step]
        print(f"Next step: Step {next_step} - {step_info['name']}")
        print(f"  {step_info['description']}")
        print()
        print(f"Run: cses step {next_step}")

        # Provide more specific guidance for common steps
        if next_step == 7:
            print()
            print("Or run sub-steps individually:")
            print("  cses step 7a   # AI fills tracking sheet with proposals")
            print("  (review Excel)")
            print("  cses step 7c   # Generate Stata code from sheet")
    else:
        print("All steps completed!")
        print()
        print("Run 'cses status' for full details")

    print()
    print("Commands:")
    print("  cses status      Show detailed workflow status")
    print("  cses step N      Execute step N")
    print("  cses step 7a     AI variable matching")
    print("  cses step 7c     Generate Stata code")
    print("  cses generate    Alias for step 7c")


def cmd_interactive(args):
    """Start conversational assistant mode."""
    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)

    if not state:
        for subdir in working_dir.iterdir():
            if subdir.is_dir() and (subdir / ".cses").exists():
                state = WorkflowState.load(subdir)
                if state:
                    working_dir = subdir
                    break

    if not state:
        print_banner()
        print("No initialized study found in this folder.")
        print()

        organizer = FileOrganizer(working_dir)
        email_folder = organizer.find_email_folder()
        if email_folder:
            detected = organizer.detect_files(source_dir=email_folder, recursive=True)
            print(f"Found email folder: {email_folder.name}/")
            print(f"  Data files: {len(detected.data_files)}")
            print(f"  Questionnaires: {len(detected.questionnaire_files)}")
            print(f"  Codebooks: {len(detected.codebook_files)}")
            print(f"  Design reports: {len(detected.design_report_files)}")
            print()
            response = input("Initialize this study now? [Y/n]: ").strip().lower()
            if response != "n":
                init_args = argparse.Namespace(country=None, year=None, no_organize=False, move=False)
                study_dir = cmd_init(init_args)
                if study_dir:
                    state = WorkflowState.load(study_dir)
        else:
            print("Run this in a study folder with deposited files, or initialize the study first.")

    if not state:
        print("No study loaded.")
        return

    print()
    print(f"CSES Expert Assistant - {state.country} {state.year}")
    print("=" * 60)
    print("Chat naturally. Say 'proceed' to work on the next workflow step.")
    print("Type 'status' for progress, 'quit' to exit.")
    print()

    conversation = ConversationSession(state)
    next_step = state.get_next_step()
    if next_step is not None:
        step_info = WORKFLOW_STEPS[next_step]
        print(f"Assistant: The next step is Step {next_step}: {step_info['name']}. Proceed?")
        print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye.")
            break
        if user_input.lower() == "status":
            conversation.refresh_state()
            print()
            print(format_workflow_status(conversation.state))
            print()
            continue

        print()
        print("Assistant is thinking...")
        response = conversation.send(
            user_input,
            on_tool_output=lambda text: print(f"  {text}")
        )
        print()
        print(f"Assistant: {response}")
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

        print("Launching installer...")
        print()
        print("The installer will run in a NEW window.")
        print("This window will close now.")
        print()

        try:
            # Launch PowerShell in a NEW WINDOW and exit immediately
            # We MUST NOT wait - the installer needs to replace our files
            import subprocess
            subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command",
                 f"Start-Process powershell -ArgumentList '-ExecutionPolicy','Bypass','-File','{script_path}' -Wait; Remove-Item '{script_path}' -ErrorAction SilentlyContinue"],
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )
        except Exception as e:
            print(f"[X] Failed to launch installer: {e}")
            sys.exit(1)

        # Exit IMMEDIATELY - do not wait for installer
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

        print("Launching installer...")
        print()
        print("The installer will run. This process will exit now.")
        print()

        try:
            os.chmod(script_path, 0o755)
            # Launch in background and exit immediately
            # We MUST NOT wait - the installer needs to replace our files
            import subprocess
            subprocess.Popen(
                ["bash", "-c", f"'{script_path}'; rm -f '{script_path}'"],
                start_new_session=True
            )
        except Exception as e:
            print(f"[X] Failed to launch installer: {e}")
            sys.exit(1)

        # Exit IMMEDIATELY - do not wait for installer
        sys.exit(0)


def cmd_wiki(args):
    """Maintain the compact CSES standards wiki."""
    from pathlib import Path
    from src.standards.distill_examples import audit_distilled_wiki, distill_example_studies
    from src.standards.questionnaire_registry import (
        Module6QuestionnaireDistiller,
        audit_questionnaire_registry,
    )
    from src.standards.administrative import audit_administrative_registry
    from src.matching.demographics import audit_demographic_registry

    command = getattr(args, "wiki_command", None) or "audit"

    if command == "audit":
        wiki_root = Path("cses_wiki")
        issues = audit_distilled_wiki(wiki_root)
        issues.extend(audit_questionnaire_registry(wiki_root / "patterns" / "module6_questionnaire_registry.json"))
        issues.extend(audit_administrative_registry(wiki_root))
        issues.extend(audit_demographic_registry())
        if issues:
            print("CSES wiki audit found issues:")
            for issue in issues:
                print(f"  - {issue}")
            print("\nRun 'cses wiki distill' or 'cses wiki distill-questionnaire' in a development checkout to rebuild patterns.")
            return False
        print("CSES wiki audit passed.")
        print("Runtime knowledge base is available without raw example_studies.")
        return True

    if command == "distill":
        examples_root = Path(args.examples)
        wiki_root = Path(args.wiki)
        print("Distilling compact CSES standards from development examples...")
        print(f"  Examples: {examples_root}")
        print(f"  CSES wiki: {wiki_root}")
        outputs = distill_example_studies(examples_root=examples_root, wiki_root=wiki_root)
        print("Distilled pattern files:")
        for name, path in outputs.items():
            print(f"  - {name}: {path}")
        print("Done. Runtime processing now uses cses_wiki; raw example_studies are not required.")
        return True

    if command == "distill-questionnaire":
        source = Path(args.source)
        output = Path(args.output)
        print("Distilling canonical CSES Module 6 questionnaire registry...")
        print(f"  Source: {source}")
        print(f"  Output: {output}")
        payload = Module6QuestionnaireDistiller().distill(source, output)
        issues = audit_questionnaire_registry(output)
        print(f"Distilled {payload.get('item_count', 0)} questionnaire/admin/demographic items.")
        if issues:
            print("Registry audit found issues:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        print("Done. Runtime processing now uses the distilled questionnaire registry; the raw text file is not required.")
        return True

    print(f"Unknown wiki command: {command}")
    return False


def cmd_benchmark(args):
    """Generate a replication benchmark scorecard for the current study."""
    from src.benchmark import ReplicationBenchmarkRunner

    working_dir = Path.cwd()
    state = WorkflowState.load(working_dir)
    profile = args.profile or "workflow"
    reference_dataset = Path(args.reference_dataset).expanduser() if args.reference_dataset else None
    reference_dir = Path(args.reference_dir).expanduser() if args.reference_dir else None
    scorecard = ReplicationBenchmarkRunner(working_dir).scorecard(
        profile=profile,
        reference_dataset=reference_dataset,
        reference_dir=reference_dir,
    )
    state.benchmark_scorecard_path = str(working_dir / ".cses" / "replication_scorecard.json")
    state.save()
    print(f"Benchmark profile: {profile}")
    print(f"Status: {scorecard.get('status')}")
    print(f"Scorecard: {state.benchmark_scorecard_path}")
    if scorecard.get("issues"):
        print("Issues:")
        for issue in scorecard["issues"]:
            print(f"  - {issue}")


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1].lower() in ["gui", "app"]:
        from src.gui_app import run_gui
        run_gui()
        return

    if any(arg in ["--help", "-h"] for arg in sys.argv[1:]):
        build_parser().parse_args()
        return

    # CRITICAL: Acquire lock to prevent parallel execution
    # This prevents the CLI and install script from running simultaneously
    if not acquire_lock():
        print("\n[X] Another CSES instance is already running.")
        print("    Please wait for it to finish or close it first.")
        print()
        print("    If you believe this is an error, delete the lock file:")
        print(f"    {get_lock_file_path()}")
        print()
        pause_before_exit()
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
                use_openwebui = os.environ.get("CSES_USE_OPENWEBUI", "").lower() in {"1", "true", "yes"}
                if not use_openwebui:
                    os.environ.pop("OPENAI_API_BASE", None)
            except ImportError:
                print("Warning: python-dotenv not installed, skipping .env loading")
            except Exception as e:
                print(f"Warning: Could not load .env file: {e}")
    except Exception as e:
        print(f"Error during initialization: {e}")
        release_lock()
        pause_before_exit()
        sys.exit(1)

    parser = build_parser()
    args = parser.parse_args()

    # Check for first-run setup (except for setup and update commands)
    if args.command not in ["setup", "update", "gui", "wiki", "benchmark"]:
        try:
            if not first_run_setup():
                print("\nSetup cancelled or incomplete.")
                pause_before_exit()
                sys.exit(1)
        except (EOFError, KeyboardInterrupt):
            print("\nSetup interrupted.")
            sys.exit(1)
        except Exception as e:
            print(f"\nSetup error: {e}")
            import traceback
            traceback.print_exc()
            pause_before_exit()
            sys.exit(1)

    try:
        if args.command == "setup":
            cmd_setup(args)
        elif args.command == "gui":
            from src.gui_app import run_gui
            run_gui()
        elif args.command == "update":
            cmd_update(args)
        elif args.command == "wiki":
            cmd_wiki(args)
        elif args.command == "benchmark":
            cmd_benchmark(args)
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
        elif args.command == "generate":
            cmd_generate(args)
        else:
            # No command - start conversational mode
            cmd_interactive(args)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        pause_before_exit("\nPress Enter to exit...")
        sys.exit(1)


def build_parser() -> argparse.ArgumentParser:
    """Build the command parser."""
    parser = argparse.ArgumentParser(
        description="CSES Data Harmonization CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  cses              Start conversational assistant
  cses init         Initialize study from files in folder
  cses status       Show detailed workflow progress
  cses step 1       Check deposit completeness
  cses step 2       Extract study design from docs
  cses step 7a      AI fills tracking sheet with proposals
  (edit Excel)      Review and verify mappings manually
  cses step 7c      Generate Stata code from tracking sheet
  cses step 8       Run Stata and debug errors
  cses generate     Alias for step 7c
  cses export       Export approved mappings
  cses migrate      Migrate old folder structure to CSES standard
  cses setup        Re-run initial configuration
  cses gui          Open the Windows GUI
  cses wiki audit   Check the CSES wiki runtime knowledge base
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
    step_parser.add_argument("step_number", help="Step number (0-16) or variant (7a, 7c)")

    # match command
    match_parser = subparsers.add_parser("match", help="Run variable matching")
    match_parser.add_argument("--no-validate", action="store_true",
                             help="Skip LLM validation")

    # export command
    export_parser = subparsers.add_parser("export", help="Export mappings")
    export_parser.add_argument("--format", "-f", choices=["json", "xlsx", "both"],
                              default="both", help="Export format")

    # generate command
    generate_parser = subparsers.add_parser("generate", help="Generate Stata code from tracking sheet")
    generate_parser.add_argument("--sheet", "-s", help="Path to tracking sheet (auto-detect if not provided)")

    # migrate command
    subparsers.add_parser("migrate", help="Migrate old folder structure to CSES standard")

    # update command
    subparsers.add_parser("update", help="Update to latest version from GitHub")

    # gui command
    subparsers.add_parser("gui", help="Open the Windows GUI")

    # benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Generate a replication benchmark scorecard")
    benchmark_parser.add_argument("--profile", choices=["workflow", "email_only", "full_reference_inputs"], default="workflow")
    benchmark_parser.add_argument("--reference-dataset", help="Optional reference .dta for functional comparison")
    benchmark_parser.add_argument("--reference-dir", help="Optional reference study directory for documentation comparison")

    # wiki command
    wiki_parser = subparsers.add_parser("wiki", help="Maintain the CSES standards wiki")
    wiki_subparsers = wiki_parser.add_subparsers(dest="wiki_command")

    wiki_subparsers.add_parser("audit", help="Check required CSES wiki runtime files")

    distill_parser = wiki_subparsers.add_parser(
        "distill",
        help="Development-only: distill compact standards from example_studies"
    )
    distill_parser.add_argument(
        "--examples",
        default="example_studies",
        help="Path to development-only example studies folder"
    )
    distill_parser.add_argument(
        "--wiki",
        default="cses_wiki",
        help="Path to CSES wiki folder"
    )

    questionnaire_parser = wiki_subparsers.add_parser(
        "distill-questionnaire",
        help="Development-only: distill Module 6 questionnaire registry"
    )
    questionnaire_parser.add_argument(
        "--source",
        default="CSES_Module6_Questionnaire.txt",
        help="Path to the authoritative Module 6 questionnaire text"
    )
    questionnaire_parser.add_argument(
        "--output",
        default="cses_wiki/patterns/module6_questionnaire_registry.json",
        help="Output path for the distilled runtime registry"
    )
    return parser


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "__cses_stata_mcp_runner__":
        from src.stata_mcp_runner import main as stata_mcp_runner_main

        sys.argv = [sys.argv[0], *sys.argv[2:]]
        raise SystemExit(stata_mcp_runner_main())
    try:
        main()
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        pause_before_exit("\nPress Enter to exit...")
        sys.exit(1)
