#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

echo "Starting KAIROS phone microphone server on port 8000."
echo "In another terminal, run: ngrok http 8000"
echo "Open the generated HTTPS URL on your phone."
echo "Press Ctrl+C here to stop the server."

exec .venv/bin/python -m modules.phone_stream
