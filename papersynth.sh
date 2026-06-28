#!/bin/bash
# PaperSynth launcher — auto-uses venv Python
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
exec "$SCRIPT_DIR/.venv/bin/python" -m papersynth "$@"
