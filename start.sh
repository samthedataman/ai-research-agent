#!/bin/bash
# Start the AI Research Agent bot
# Usage: ./start.sh

set -e

cd "$(dirname "$0")"

# Use pyenv Python 3.12 (where deps are installed)
PYTHON="$HOME/.pyenv/versions/3.12.1/bin/python3"
if [ ! -f "$PYTHON" ]; then
    PYTHON=$(which python3.12 2>/dev/null || which python3)
fi
echo "Using: $PYTHON ($($PYTHON --version))"

# Kill any existing bot instances
pkill -f "python3 main.py" 2>/dev/null && echo "Stopped old bot instance" || true
pkill -f "python3.12 main.py" 2>/dev/null || true
sleep 1

# Check Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Starting Ollama..."
    ollama serve &
    sleep 3
fi

echo "Ollama: OK"
echo "Model: $(grep OLLAMA_MODEL .env 2>/dev/null | cut -d= -f2 || echo 'qwen2.5:0.5b')"
echo "Starting @prbuilderbot..."

$PYTHON main.py
