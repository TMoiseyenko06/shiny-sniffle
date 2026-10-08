#!/usr/bin/env bash
# Restart the GUI (picks up changes to .env and the code). Optional: ./restart.sh <port>
set -euo pipefail
cd "$(dirname "$0")"
./stop.sh
./start.sh "$@"
