#!/bin/bash
# PaperSynth — One-step installer
# Usage: bash install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "🔬 PaperSynth Installer"
echo "========================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "✓ Python $PYTHON_VERSION found"

# Create venv if needed
if [ ! -d ".venv" ]; then
    echo "→ Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment exists"
fi

# Install
echo "→ Installing dependencies (this may take a minute)..."
.venv/bin/pip install -e . --quiet 2>&1 | tail -5
echo "✓ Dependencies installed"

# Check .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo "⚠  Created .env from .env.example — edit it with your API keys!"
    fi
fi

# Make wrapper executable
chmod +x papersynth.sh 2>/dev/null || true

echo ""
echo "✅ Installation complete!"
echo ""
echo "Usage:"
echo "  cd $SCRIPT_DIR"
echo "  ./papersynth.sh \"your research query\""
echo ""
echo "Or with venv activated:"
echo "  source .venv/bin/activate"
echo "  papersynth \"your research query\""
echo ""
