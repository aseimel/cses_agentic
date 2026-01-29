#!/bin/bash
# CSES Assistant - Installer for Linux/Mac
# Usage: curl -fsSL https://raw.githubusercontent.com/aseimel/cses_agentic/main/install.sh | bash

set -e

echo ""
echo "======================================================================"
echo "           CSES Assistant"
echo "           Linux/Mac Installation"
echo "======================================================================"
echo ""

INSTALL_DIR="$HOME/.cses-agent"
REPO_URL="https://github.com/aseimel/cses_agentic/archive/refs/heads/main.zip"

# Check Python
check_python() {
    for cmd in python3 python; do
        if command -v $cmd &> /dev/null; then
            version=$($cmd --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
            major=$(echo $version | cut -d. -f1)
            minor=$(echo $version | cut -d. -f2)
            if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
                echo "[OK] Python $version"
                PYTHON_CMD=$cmd
                return 0
            fi
        fi
    done
    echo "[X] Python 3.10+ required"
    echo ""
    echo "Install Python from: https://www.python.org/downloads/"
    exit 1
}

echo "Checking requirements..."
check_python
echo ""

# Preserve .env if updating
EXISTING_ENV=""
IS_UPDATE=false
if [ -d "$INSTALL_DIR" ]; then
    IS_UPDATE=true
    echo "Existing installation found - updating..."

    if [ -f "$INSTALL_DIR/.env" ]; then
        echo "Preserving your configuration..."
        EXISTING_ENV=$(cat "$INSTALL_DIR/.env")
    fi

    rm -rf "$INSTALL_DIR"
fi

# Download and extract
echo "Downloading CSES Assistant..."
TMP_ZIP="/tmp/cses_agentic.zip"
curl -fsSL "$REPO_URL" -o "$TMP_ZIP"

echo "Extracting..."
unzip -q "$TMP_ZIP" -d /tmp
mv /tmp/cses_agentic-main "$INSTALL_DIR"
rm "$TMP_ZIP"

# Restore .env
if [ -n "$EXISTING_ENV" ]; then
    echo "Restoring your configuration..."
    echo "$EXISTING_ENV" > "$INSTALL_DIR/.env"
fi

# Create virtual environment
echo "Setting up Python environment..."
cd "$INSTALL_DIR"
$PYTHON_CMD -m venv .venv

# Install dependencies
echo "Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip -q
.venv/bin/python -m pip install -r requirements.txt

# Create launcher script
echo "Creating 'cses' command..."
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/cses" << 'EOF'
#!/bin/bash
INSTALL_DIR="$HOME/.cses-agent"
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"
"$INSTALL_DIR/.venv/bin/python" "$INSTALL_DIR/cses_cli.py" "$@"
EOF

chmod +x "$BIN_DIR/cses"

# Check if bin dir is in PATH
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo "Adding $BIN_DIR to PATH..."

    # Add to appropriate shell config
    if [ -f "$HOME/.zshrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.zshrc"
        echo "Added to ~/.zshrc"
    elif [ -f "$HOME/.bashrc" ]; then
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
        echo "Added to ~/.bashrc"
    fi
fi

echo ""
echo "======================================================================"
if [ "$IS_UPDATE" = true ]; then
    echo "           Update Complete!"
else
    echo "           Installation Complete!"
fi
echo "======================================================================"
echo ""

if [ "$IS_UPDATE" = true ]; then
    echo "Your configuration has been preserved."
    echo ""
    echo "To use: Run 'cses' (may need to restart terminal)"
else
    echo "NEXT STEPS:"
    echo ""
    echo "  1. Restart your terminal (or run: source ~/.bashrc)"
    echo ""
    echo "  2. Run 'cses' to start the setup wizard"
    echo "     (It will ask for your GESIS API key and Stata path)"
    echo ""
    echo "  3. Navigate to a folder with collaborator files and run 'cses'"
fi
echo ""
echo "To reconfigure later: cses setup --force"
echo ""
