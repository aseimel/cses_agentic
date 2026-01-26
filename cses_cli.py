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
from src.workflow.organizer import FileOrganizer, detect_and_summarize, detect_questionnaire_language
from src.workflow.steps import StepExecutor
from src.agent.validator import check_claude_cli_available
from src.agent.conversation import ConversationSession

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger(__name__)


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

    print("""
╔══════════════════════════════════════════════════════════════╗
║           First-Time Setup                                   ║
╚══════════════════════════════════════════════════════════════╝
""")
    print("Welcome! Let's configure the CSES Agent for your environment.\n")

    # GESIS API Key
    print("=" * 60)
    print("STEP 1: GESIS API Key")
    print("=" * 60)
    print("Enter your GESIS OpenWebUI API key.")
    print("(Get it from: https://ai-openwebui.gesis.org)")
    print()

    api_key = input("GESIS API Key: ").strip()
    if not api_key:
        print("\n❌ API key is required. Setup cancelled.")
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
            print(f"\n⚠️  Warning: File not found: {stata_path}")
            confirm = input("Save anyway? [y/N]: ").strip().lower()
            if confirm != 'y':
                stata_path = ""

    # Claude validation preference
    print()
    print("=" * 60)
    print("STEP 3: Validation Model")
    print("=" * 60)
    print("Choose how to validate variable mappings:")
    print()
    print("  1. Claude CLI (requires Claude Max subscription)")
    print("     - Run 'claude login' first")
    print("     - No additional API key needed")
    print()
    print("  2. GESIS API (uses same API key)")
    print("     - Uses the GESIS OpenWebUI models")
    print()

    choice = input("Choose [1/2] (default: 2): ").strip()
    use_claude_cli = choice == "1"

    # Write .env file
    print()
    print("Saving configuration...")

    env_content = f"""# CSES Agent Configuration
# Generated by first-run setup

# GESIS OpenWebUI API
OPENAI_API_KEY={api_key}
OPENAI_API_BASE=https://ai-openwebui.gesis.org/api/v1

# LLM Models
LLM_MODEL_MATCH=openai/gpt-oss:120b
LLM_MODEL_PREPROCESS=openai/llama4:latest
"""

    if use_claude_cli:
        env_content += """
# Validation via Claude CLI (Max subscription)
LLM_MODEL_VALIDATE=claude-cli
"""
    else:
        env_content += """
# Validation via GESIS API
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

    print(f"✅ Configuration saved to: {env_file}")
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


def cmd_init(args) -> Path:
    """
    Initialize a new study from files in the current folder.

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

    print("Scanning files...")

    # Detect files
    organizer = FileOrganizer(working_dir)
    detected = organizer.detect_files()

    if not detected.data_files:
        print("\n❌ No data files found (.dta, .sav, .csv, .xlsx)")
        print("   Please run this in a folder containing the collaborator's data files.")
        return None

    # Get country/year (use detected or prompt)
    country = args.country or detected.country
    year = args.year or detected.year

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

    # Show what we found
    country_code = organizer.COUNTRY_CODES.get(country.lower(), country[:3].upper())
    print(f"\n📁 Setting up: {country} {year} ({country_code}_{year})")
    print(f"\nDetected files:")
    print(f"  Data: {len(detected.data_files)} file(s)")
    print(f"  Questionnaires: {len(detected.questionnaire_files)} file(s)")
    print(f"  Codebooks: {len(detected.codebook_files)} file(s)")

    # Determine study directory
    expected_folder = f"{country_code}_{year}"

    if working_dir.name == expected_folder or (working_dir / ".cses").exists():
        study_dir = working_dir
    else:
        # Create organized folder
        study_dir, _ = organizer.initialize_study(
            country=country,
            year=year,
            copy_files=True
        )
        print(f"\n✅ Created: {study_dir.name}/")
        print(f"   Originals preserved in: {study_dir.name}/original_deposit/")

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
            state.data_file = str(study_dir / f"{prefix}_data{ext}")
        else:
            state.data_file = str(detected.data_files[0])

    if detected.questionnaire_files:
        if study_dir != working_dir:
            state.questionnaire_files = []
            if len(detected.questionnaire_files) == 1:
                f = detected.questionnaire_files[0]
                state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire{f.suffix}"))
            else:
                # Separate english vs native
                english_files = [f for f in detected.questionnaire_files
                                if detect_questionnaire_language(f.name) == "english"]
                native_files = [f for f in detected.questionnaire_files
                               if detect_questionnaire_language(f.name) == "native"]

                for i, f in enumerate(english_files):
                    if len(english_files) == 1:
                        state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire_english{f.suffix}"))
                    else:
                        state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire_english_{i+1}{f.suffix}"))

                for i, f in enumerate(native_files):
                    if len(native_files) == 1:
                        state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire_native{f.suffix}"))
                    else:
                        state.questionnaire_files.append(str(study_dir / f"{prefix}_questionnaire_native_{i+1}{f.suffix}"))
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
        print(detect_and_summarize(working_dir))
        print()

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
    print("Type 'quit' to exit, 'status' to see progress.\n")

    # Create conversation session
    conversation = ConversationSession(state)

    # Send initial greeting to get Claude's guidance
    next_step = state.get_next_step()
    if next_step is not None:
        step_info = WORKFLOW_STEPS[next_step]
        initial_prompt = f"The study is initialized. The next step is Step {next_step}: {step_info['name']}. Briefly explain what this step involves and ask if the user is ready to proceed."
        try:
            response = conversation.send(initial_prompt)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Note: Could not connect to Claude ({e})")
            print(f"Next step: Step {next_step} - {step_info['name']}\n")

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

Commands: status, quit
""")
            continue

        # Send everything else to Claude
        try:
            print()
            response = conversation.send(user_input)
            print(f"Assistant: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


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


def cmd_update(args):
    """Update CSES agent from GitHub."""
    import tempfile
    import zipfile
    import urllib.request

    install_dir = get_install_dir()
    repo_url = "https://github.com/aseimel/cses_agentic/archive/refs/heads/main.zip"

    print("CSES Agent Update")
    print("=" * 40)
    print()

    # Preserve .env
    env_file = install_dir / ".env"
    env_content = None
    if env_file.exists():
        env_content = env_file.read_text()
        print("Preserving configuration...")

    # Download latest
    print("Downloading latest version...")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            urllib.request.urlretrieve(repo_url, tmp.name)
            zip_path = tmp.name
    except Exception as e:
        print(f"Download failed: {e}")
        return

    # Extract to temp directory
    print("Extracting...")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(tmp_dir)

            extracted_dir = Path(tmp_dir) / "cses_agentic-main"

            # Update Python files (preserve .env and .venv)
            print("Updating files...")
            for item in extracted_dir.iterdir():
                if item.name in [".env", ".venv", ".git"]:
                    continue

                dst = install_dir / item.name
                if item.is_dir():
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(item, dst)
                else:
                    shutil.copy2(item, dst)

            # Restore .env
            if env_content:
                env_file.write_text(env_content)
                print("Configuration restored.")

            # Check if requirements changed and update
            print("Checking dependencies...")
            venv_python = install_dir / ".venv" / ("Scripts" if os.name == "nt" else "bin") / "python"
            if venv_python.exists():
                import subprocess
                result = subprocess.run(
                    [str(venv_python), "-m", "pip", "install", "-q", "-r", str(install_dir / "requirements.txt")],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"Warning: pip install had issues: {result.stderr}")

    except Exception as e:
        print(f"Update failed: {e}")
        import traceback
        traceback.print_exc()
        return
    finally:
        # Clean up zip
        try:
            os.unlink(zip_path)
        except:
            pass

    print()
    print("=" * 40)
    print("Update complete!")
    print()
    print("Changes will take effect on next run.")


def main():
    """Main entry point."""
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
        input("Press Enter to exit...")
        sys.exit(1)

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
