#!/usr/bin/env bash
# Stop the GUI started by start.sh (the server and its restart loop).
set -uo pipefail
cd "$(dirname "$0")"

if [ ! -f server.pid ] || ! kill -0 "$(cat server.pid)" 2>/dev/null; then
  rm -f server.pid
  echo "Not running."
  exit 0
fi
touch .stop
PID=$(cat server.pid)
kill -TERM -- "-$PID" 2>/dev/null || kill -TERM "$PID" 2>/dev/null
for _ in $(seq 1 15); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$PID" 2>/dev/null; then
  echo "Still running after 15 s; forcing it to stop."
  kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID" 2>/dev/null
fi
rm -f server.pid
echo "Stopped. A job that was running will show as interrupted; press Retry after restarting."
