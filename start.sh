#!/usr/bin/env bash
# Start the GUI in the background. It survives SSH logout and restarts itself if it crashes.
#   ./start.sh          port from PORT in .env (default 8000)
#   ./start.sh 7860     one-off port override
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then set -a; . ./.env; set +a; fi
[ -n "${1:-}" ] && PORT="$1"
export PORT="${PORT:-8000}"

if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "Already running (pid $(cat server.pid)). Use ./restart.sh to restart."
  exit 0
fi
[ -x venv/bin/python ] || { echo "No venv found. Run ./setup.sh first." >&2; exit 1; }

rm -f .stop
# The loop below is a tiny supervisor: if the server exits (crash, unrecoverable CUDA error,
# killed for using too much RAM) it is started again. stop.sh creates .stop to end the loop.
setsid nohup bash -c '
  echo $$ > server.pid
  while true; do
    started=$(date +%s)
    echo "[supervisor] $(date "+%F %T") starting server on port $PORT"
    venv/bin/python server.py --port "$PORT" && code=0 || code=$?
    [ -f .stop ] && break
    echo "[supervisor] server exited with code $code; restarting"
    if [ $(( $(date +%s) - started )) -lt 60 ]; then sleep 30; else sleep 3; fi
  done
  rm -f server.pid' >> server.log 2>&1 < /dev/null &

for _ in $(seq 1 30); do
  sleep 1
  curl -s -o /dev/null "http://127.0.0.1:$PORT/login" && break
done
if ! curl -s -o /dev/null "http://127.0.0.1:$PORT/login"; then
  echo "The server did not answer on port $PORT yet. Check: tail -n 50 server.log" >&2
  exit 1
fi

# vast.ai exposes the public IP and the external port mapped to each internal port as env vars
var_from_init() { { tr '\0' '\n' < /proc/1/environ; } 2>/dev/null | sed -n "s/^$1=//p" | head -1 || true; }
PUBLIC_IP="${PUBLIC_IPADDR:-$(var_from_init PUBLIC_IPADDR)}"
PORT_VAR="VAST_TCP_PORT_$PORT"
PUBLIC_PORT="${!PORT_VAR:-$(var_from_init "$PORT_VAR")}"

echo "LongCat GUI is running (log: server.log, pid: $(cat server.pid 2>/dev/null || echo '?'))."
echo "  On this machine:  http://127.0.0.1:$PORT"
if [ -n "$PUBLIC_IP" ] && [ -n "$PUBLIC_PORT" ]; then
  echo "  From your phone:  http://$PUBLIC_IP:$PUBLIC_PORT"
else
  echo "  From your phone:  http://<instance-ip>:<external port that vast.ai maps to $PORT>"
fi
if [ -n "${GUI_PASSWORD:-}" ]; then
  echo "  Password:         $GUI_PASSWORD"
else
  echo "  Password:         random, printed in server.log (set GUI_PASSWORD in .env to keep one)"
fi
echo "The model loads in the background; the status bar shows when it is ready."
