#!/usr/bin/env python3
"""
Build standalone Windows executable using PyInstaller.

This creates a single .exe file that doesn't require Python to be installed.
Perfect for non-technical users.

Usage:
    pip install pyinstaller
    python build_standalone.py

Output:
    dist/cses.exe - Single executable file (~50-100MB)
"""

import subprocess
import sys
from pathlib import Path

def build():
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    project_dir = Path(__file__).parent

    # PyInstaller command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",                          # Single executable
        "--name", "cses",                     # Output name
        "--console",                          # Console application
        "--clean",                            # Clean build
        # Add hidden imports that PyInstaller might miss
        "--hidden-import", "litellm",
        "--hidden-import", "anthropic",
        "--hidden-import", "openai",
        "--hidden-import", "pandas",
        "--hidden-import", "polars",
        "--hidden-import", "pyreadstat",
        "--hidden-import", "openpyxl",
        "--hidden-import", "pypdf",
        "--hidden-import", "docx",
        "--hidden-import", "dotenv",
        # Add data files
        "--add-data", f"{project_dir}/.env.example;.",
        # Entry point
        str(project_dir / "cses_cli.py")
    ]

    print("Building standalone executable...")
    print(f"Command: {' '.join(cmd)}")
    print()

    subprocess.check_call(cmd)

    print()
    print("=" * 60)
    print("BUILD COMPLETE!")
    print("=" * 60)
    print()
    print(f"Executable created: dist/cses.exe")
    print()
    print("To distribute:")
    print("  1. Copy dist/cses.exe to a shared folder")
    print("  2. Users can run it directly - no installation needed")
    print()


if __name__ == "__main__":
    build()
