#!/usr/bin/env bash
# Full install on a fresh vast.ai GPU instance. Safe to re-run: finished steps are skipped.
#   ./setup.sh               install, download weights, run a 5-second smoke test
#   ./setup.sh --skip-test   same without the smoke test
set -euo pipefail
cd "$(dirname "$0")"
APP_DIR="$(pwd)"
PY="$APP_DIR/venv/bin/python"
LONGCAT_REF="${LONGCAT_REF:-main}"
RUN_TEST=1
for arg in "$@"; do
  case "$arg" in
    --skip-test) RUN_TEST=0 ;;
    -h|--help) sed -n '2,4p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
warn() { printf '\033[1;33m!!  %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -1)" = "$2" ]; }

# ---------------------------------------------------------------------------------------------
step "GPU"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found. This needs an NVIDIA GPU instance."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: *\([0-9.]*\).*/\1/p' | head -1)
GPU_COUNT=$(nvidia-smi -L | wc -l)
echo "GPU: $GPU_NAME x$GPU_COUNT, $((VRAM_MB / 1024)) GB VRAM, compute capability ${CC:-?}, driver supports CUDA ${DRIVER_CUDA:-?}"
if [ "$VRAM_MB" -lt 40000 ]; then
  warn "Under 40 GB VRAM: the 13.6B DiT (~27 GB in bf16) does not fit, so the server swaps DiT blocks"
  warn "from CPU RAM (slow), offloads the text encoder, tiles the VAE and only offers 480p."
fi
[ "$GPU_COUNT" -gt 1 ] && warn "Only GPU 0 is used (one job at a time)."

if [ -n "${TORCH_PACKAGES:-}" ]; then
  : "${TORCH_INDEX:?set TORCH_INDEX too}" "${FLASH_ATTN_VERSION:?set FLASH_ATTN_VERSION too}"; NEED_CUDA=12.0
elif [ "${CC%%.*}" -ge 10 ] 2>/dev/null; then
  # Blackwell (B200, RTX 50xx, RTX PRO 6000): torch 2.6 + CUDA 12.4 from the README has no kernels for it.
  TORCH_PACKAGES="torch==2.7.1 torchvision==0.22.1"; TORCH_INDEX="https://download.pytorch.org/whl/cu128"
  FLASH_ATTN_VERSION=2.8.3; NEED_CUDA=12.8
  warn "Blackwell GPU: using torch 2.7.1 + CUDA 12.8 instead of the README's torch 2.6.0 + CUDA 12.4."
else
  TORCH_PACKAGES="torch==2.6.0 torchvision==0.21.0"; TORCH_INDEX="https://download.pytorch.org/whl/cu124"
  FLASH_ATTN_VERSION=2.7.4.post1; NEED_CUDA=12.4
fi
if [ -n "$DRIVER_CUDA" ] && ! version_ge "$DRIVER_CUDA" "$NEED_CUDA"; then
  if version_ge "$DRIVER_CUDA" 12.0; then
    warn "The driver supports CUDA $DRIVER_CUDA (< $NEED_CUDA). This usually still works; if torch cannot"
    warn "see the GPU below, rent an instance whose 'Max CUDA' is >= $NEED_CUDA."
  else
    die "The driver only supports CUDA $DRIVER_CUDA. Rent a vast.ai instance with 'Max CUDA' >= $NEED_CUDA."
  fi
fi

# ---------------------------------------------------------------------------------------------
step "Disk space"
df -h "$APP_DIR" | sed 1d | awk '{print "app dir:   " $4 " free of " $2 " (" $6 ")"}'
[ -d /workspace ] && df -h /workspace | sed 1d | awk '{print "/workspace: " $4 " free of " $2}'
echo "Needs roughly 12 GB for the Python environment plus the model weights (size checked before download)."

# ---------------------------------------------------------------------------------------------
step "System packages"
missing=""
for cmd in git curl; do command -v "$cmd" >/dev/null || missing="$missing $cmd"; done
if [ -n "$missing" ]; then
  SUDO=""; [ "$(id -u)" -ne 0 ] && SUDO="sudo"
  $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $missing
fi
echo "git and curl present"

# ---------------------------------------------------------------------------------------------
step "LongCat-Video code (github.com/meituan-longcat/LongCat-Video, $LONGCAT_REF)"
if [ ! -d LongCat-Video/.git ]; then
  git clone --single-branch --branch "$LONGCAT_REF" https://github.com/meituan-longcat/LongCat-Video LongCat-Video
fi
echo "LongCat-Video at commit $(git -C LongCat-Video rev-parse --short HEAD)"

# ---------------------------------------------------------------------------------------------
step "Python 3.10 virtual environment (uv)"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
[ -x "$PY" ] || uv venv --python 3.10 --seed venv
PIP=(uv pip install --python "$PY")
"$PY" --version

# ---------------------------------------------------------------------------------------------
step "PyTorch ($TORCH_PACKAGES, $TORCH_INDEX)"
WANT_TORCH=$(echo "$TORCH_PACKAGES" | sed -n 's/.*torch==\([0-9.]*\).*/\1/p')
if ! "$PY" -c "import torch, sys; sys.exit(torch.__version__.split('+')[0] != '$WANT_TORCH')" 2>/dev/null; then
  # shellcheck disable=SC2086
  "${PIP[@]}" $TORCH_PACKAGES --index-url "$TORCH_INDEX"
fi
"$PY" -c "
import torch
print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU visible:', torch.cuda.is_available())
assert torch.cuda.is_available(), 'PyTorch cannot see the GPU'
print('device:', torch.cuda.get_device_name(0))"

# ---------------------------------------------------------------------------------------------
step "flash-attn $FLASH_ATTN_VERSION"
flash_attn_works() {
  "$PY" - <<'EOF' >/dev/null 2>&1
import torch
from flash_attn import flash_attn_func
q = torch.randn(1, 64, 4, 64, device="cuda", dtype=torch.bfloat16)
flash_attn_func(q, q, q)
torch.cuda.synchronize()
EOF
}
USE_XFORMERS=0
if ! flash_attn_works; then
  "${PIP[@]}" ninja psutil packaging wheel setuptools
  # flash-attn's setup.py fetches the prebuilt wheel matching torch/CUDA/Python; it compiles only if none exists
  MAX_JOBS="${MAX_JOBS:-4}" "$PY" -m pip install "flash-attn==$FLASH_ATTN_VERSION" --no-build-isolation || true
fi
if flash_attn_works; then
  echo "flash-attn works on this GPU"
else
  warn "flash-attn is not usable here; installing xformers and switching the DiT config to it."
  USE_XFORMERS=1
  case "$WANT_TORCH" in 2.6.0) XF=0.0.29.post3 ;; 2.7.1) XF=0.0.31.post1 ;; *) XF="" ;; esac
  "${PIP[@]}" --no-deps "xformers${XF:+==$XF}"
fi

# ---------------------------------------------------------------------------------------------
step "LongCat-Video requirements"
REQS=$(mktemp)
# torch and flash-attn are already installed above (versions matched to this GPU)
grep -viE '^[[:space:]]*(torch|torchvision|torchaudio|flash[-_]attn)[[:space:]]*([=<>!~]|$)' \
  LongCat-Video/requirements.txt > "$REQS"
"${PIP[@]}" -r "$REQS"
rm -f "$REQS"

step "GUI requirements"
"${PIP[@]}" -r requirements-gui.txt

# ---------------------------------------------------------------------------------------------
step "Model weights (huggingface.co/meituan-longcat/LongCat-Video)"
mkdir -p models uploads outputs
"$PY" tools/download_weights.py --dest models/LongCat-Video

if [ "$USE_XFORMERS" = 1 ]; then
  "$PY" - <<'EOF'
import json, pathlib
path = pathlib.Path("models/LongCat-Video/dit/config.json")
config = json.loads(path.read_text())
config.update(enable_flashattn2=False, enable_flashattn3=False, enable_xformers=True)
path.write_text(json.dumps(config, indent=2))
print("dit/config.json now uses xformers attention")
EOF
fi

# ---------------------------------------------------------------------------------------------
step "Configuration (.env)"
if [ ! -f .env ]; then
  cp .env.example .env
  PASSWORD=$("$PY" -c "import secrets; print(secrets.token_urlsafe(12))")
  sed -i "s|^GUI_PASSWORD=.*|GUI_PASSWORD=$PASSWORD|" .env
  chmod 600 .env
  echo "Created .env with a random GUI_PASSWORD"
else
  echo ".env exists; left unchanged"
fi

# ---------------------------------------------------------------------------------------------
if [ "$RUN_TEST" = 1 ]; then
  step "Smoke test: 5 s at 480p straight through backend.py (loads the model first; a few minutes)"
  if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
    warn "The GUI is running and holds the GPU; skipped. Run ./stop.sh && venv/bin/python tools/bench.py"
  else
    "$PY" tools/bench.py --resolutions 480p --duration 5 --tag smoke
  fi
fi

step "Setup complete"
echo "Start the GUI:  ./start.sh       (port: PORT in .env, default 8000)"
echo "Password:       $(sed -n 's/^GUI_PASSWORD=//p' .env)"
echo "More checks:    tools/test_official.sh   (LongCat's own I2V demo)"
echo "                venv/bin/python tools/bench.py   (speed + peak VRAM at 480p and 720p)"
