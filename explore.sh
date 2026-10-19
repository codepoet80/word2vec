#!/bin/bash
# Set up a local Python venv (if needed) and launch the word2vec explorer TUI.
# Usage: ./explore.sh [vectors.bin]
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/.venv"

if [ ! -x "$VENV/bin/python" ]; then
  echo "creating venv in $VENV ..."
  python3 -m venv "$VENV"
fi

if ! "$VENV/bin/python" -c "import numpy" 2>/dev/null; then
  echo "installing numpy ..."
  "$VENV/bin/pip" install -q --disable-pip-version-check numpy
fi

exec "$VENV/bin/python" "$ROOT/scripts/explore.py" "$@"
