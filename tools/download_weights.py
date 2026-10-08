"""Download LongCat-Video weights from the official Hugging Face repo into models/ (skipped when complete).

    venv/bin/python tools/download_weights.py [--dest models/LongCat-Video]
"""
import argparse
import shutil
import sys
from pathlib import Path

REPO, OWNER = "meituan-longcat/LongCat-Video", "meituan-longcat"
REQUIRED = ["dit/config.json", "vae", "text_encoder", "tokenizer", "scheduler",
            "lora/cfg_step_lora.safetensors", "lora/refinement_lora.safetensors"]
APP_DIR = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dest", default=str(APP_DIR / "models" / "LongCat-Video"))
    args = parser.parse_args()
    dest = Path(args.dest)
    marker = dest / ".download_complete"
    if marker.exists() and all((dest / r).exists() for r in REQUIRED):
        print(f"Weights already in {dest} (revision {marker.read_text().strip()[:10]}); skipping download.")
        return

    from huggingface_hub import HfApi, snapshot_download
    info = HfApi().model_info(REPO, files_metadata=True)
    if info.id != REPO or (info.author or "").lower() != OWNER:
        sys.exit(f"Refusing to download: {info.id} is not the official {OWNER} repository.")
    total = sum(s.size or 0 for s in info.siblings)
    present = sum((dest / s.rfilename).stat().st_size for s in info.siblings
                  if (dest / s.rfilename).is_file() and (dest / s.rfilename).stat().st_size == s.size)
    dest.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(dest).free
    needed = total - present
    print(f"Official repo {REPO} @ {info.sha[:10]}: {len(info.siblings)} files, {total / 1e9:.1f} GB "
          f"({needed / 1e9:.1f} GB still to download). Free disk at {dest}: {free / 1e9:.1f} GB.")
    if needed > free - 2e9:
        sys.exit(f"Not enough disk space: need {needed / 1e9:.1f} GB plus ~2 GB headroom. "
                 "Rent more disk on vast.ai or free space, then re-run.")

    snapshot_download(REPO, revision=info.sha, local_dir=str(dest))   # resumable; shows progress bars
    missing = [r for r in REQUIRED if not (dest / r).exists()]
    if missing:
        sys.exit(f"Download finished but these are missing: {missing}. The repo layout may have changed.")
    marker.write_text(info.sha)
    print(f"Weights ready in {dest}.")


if __name__ == "__main__":
    main()
