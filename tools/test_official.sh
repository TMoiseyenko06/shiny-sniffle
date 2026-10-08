#!/usr/bin/env bash
# Prove the model works with LongCat-Video's own image-to-video demo, before involving the GUI.
# It renders three clips from assets/girl.png: 480p (50 steps), 480p distilled (16 steps), 720p refined.
# Stop the GUI first (./stop.sh) so the GPU is free. Outputs: LongCat-Video/output_i2v*.mp4
set -euo pipefail
cd "$(dirname "$0")/.."
APP_DIR="$(pwd)"

if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "The GUI server is running and holds the model in VRAM. Run ./stop.sh first." >&2
  exit 1
fi
[ -x venv/bin/torchrun ] || { echo "venv missing: run ./setup.sh first" >&2; exit 1; }

COMPILE=""
[ "${ENABLE_COMPILE:-0}" = 1 ] && COMPILE="--enable_compile"

cd LongCat-Video
rm -f output_i2v.mp4 output_i2v_distill.mp4 output_i2v_refine.mp4
start=$(date +%s)
"$APP_DIR/venv/bin/torchrun" --nproc_per_node=1 run_demo_image_to_video.py \
  --checkpoint_dir="$APP_DIR/models/LongCat-Video" $COMPILE
echo "Official demo finished in $(( $(date +%s) - start )) s"

for f in output_i2v.mp4 output_i2v_distill.mp4 output_i2v_refine.mp4; do
  "$APP_DIR/venv/bin/python" -c "
import sys; sys.path.insert(0, '$APP_DIR'); import backend
print('$f', backend.probe_video('$f'))"
done
