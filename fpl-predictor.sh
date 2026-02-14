#!/usr/bin/env bash
# Launch FPL Predictor Flask app and open in default browser.
set -e

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"

# Kill any existing instance on port 5000
fuser -k 5000/tcp 2>/dev/null || true
sleep 0.3

# Start Flask in background
.venv/bin/python -m src.app &
APP_PID=$!

# Wait for the server to be ready, then open browser
for i in $(seq 1 30); do
    if curl -s -o /dev/null http://127.0.0.1:5000/ 2>/dev/null; then
        xdg-open http://127.0.0.1:5000 2>/dev/null &
        break
    fi
    sleep 0.5
done

# Keep running until the Flask process exits
wait $APP_PID
