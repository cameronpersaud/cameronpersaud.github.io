#!/usr/bin/env bash
# Start BrainStack 3D (first run installs the libraries it needs).
# Run with:  bash run_mac_linux.sh
set -e
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "Setting up for the first time, this takes a few minutes..."
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
.venv/bin/python main.py
