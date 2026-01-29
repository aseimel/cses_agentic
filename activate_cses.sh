#!/bin/bash
# Activate CSES Assistant environment
# Usage: source activate_cses.sh

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Check if .venv exists
if [ ! -d "$SCRIPT_DIR/.venv" ]; then
    echo "Error: Virtual environment not found at $SCRIPT_DIR/.venv"
    echo "Run: python -m venv .venv && .venv/bin/pip install -r requirements.txt"
    return 1
fi

source "$SCRIPT_DIR/.venv/bin/activate"

# Add project to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           CSES Assistant                                      ║"
echo "║           Environment Activated                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Commands available:"
echo "  cses              Start interactive CLI in current folder"
echo "  cses init         Initialize study from files"
echo "  cses status       Show workflow progress"
echo "  cses match        Run variable matching"
echo "  cses export       Export approved mappings"
echo ""
echo "Quick start:"
echo "  1. cd /path/to/collaborator/files"
echo "  2. cses init"
echo "  3. cses"
echo ""
