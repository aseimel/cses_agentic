# Environment Setup

## Option 1: uv (Recommended - Fast)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
cd /home/armin/cses_agentic
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Option 2: System pip (Arch Linux)

```bash
# Install pip
sudo pacman -S python-pip

# Install dependencies
pip install -r requirements.txt
```

## Option 3: Conda/Mamba

```bash
# Create environment
conda create -n cses python=3.11
conda activate cses
pip install -r requirements.txt
```

## Verify Installation

```bash
python -c "import polars; import pyreadstat; print('OK')"
```

## Run First Test

```bash
python src/test_variable_mapping.py
```
