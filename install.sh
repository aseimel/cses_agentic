#!/bin/bash
# CSES Agent - One-line installer for Unix/Mac/Linux
# Usage: curl -sSL https://raw.githubusercontent.com/YOUR_ORG/cses_agentic/main/install.sh | bash

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           CSES Data Harmonization Agent                      ║"
echo "║           Installation Script                                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Installation directory (no admin needed)
INSTALL_DIR="$HOME/.cses-agent"
REPO_URL="https://github.com/YOUR_ORG/cses_agentic.git"

# Check for Python 3.10+
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 10 ]; then
            echo "✓ Python $PYTHON_VERSION found"
            return 0
        fi
    fi
    echo "✗ Python 3.10+ required but not found"
    echo "  Please install Python from: https://www.python.org/downloads/"
    exit 1
}

# Check for git
check_git() {
    if command -v git &> /dev/null; then
        echo "✓ Git found"
        return 0
    fi
    echo "✗ Git not found"
    echo "  Please install Git from: https://git-scm.com/downloads"
    exit 1
}

# Check for Claude CLI (optional but recommended)
check_claude() {
    if command -v claude &> /dev/null; then
        echo "✓ Claude CLI found"
    else
        echo "⚠ Claude CLI not found (optional)"
        echo "  Install with: npm install -g @anthropic-ai/claude-code"
        echo "  Then run: claude login"
    fi
}

echo "Checking requirements..."
check_python
check_git
check_claude
echo ""

# Remove old installation if exists
if [ -d "$INSTALL_DIR" ]; then
    echo "Removing old installation..."
    rm -rf "$INSTALL_DIR"
fi

# Clone repository
echo "Downloading CSES Agent..."
git clone --depth 1 "$REPO_URL" "$INSTALL_DIR"

# Create virtual environment
echo "Setting up Python environment..."
cd "$INSTALL_DIR"
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# Create command wrapper
echo "Creating 'cses' command..."
mkdir -p "$HOME/.local/bin"

cat > "$HOME/.local/bin/cses" << 'EOF'
#!/bin/bash
INSTALL_DIR="$HOME/.cses-agent"
source "$INSTALL_DIR/.venv/bin/activate"
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"
python "$INSTALL_DIR/cses_cli.py" "$@"
EOF

chmod +x "$HOME/.local/bin/cses"

# Add to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo ""
    echo "Adding ~/.local/bin to PATH..."

    # Detect shell and update config
    if [ -f "$HOME/.zshrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
        echo "  Added to ~/.zshrc"
    fi
    if [ -f "$HOME/.bashrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        echo "  Added to ~/.bashrc"
    fi
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Installation Complete!                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "To get started:"
echo "  1. Open a NEW terminal (or run: source ~/.bashrc)"
echo "  2. Navigate to a folder with collaborator files"
echo "  3. Run: cses"
echo ""
echo "Configuration:"
echo "  Edit: $INSTALL_DIR/.env"
echo ""
