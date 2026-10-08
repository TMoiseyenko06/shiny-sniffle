"""Generate straight through backend.py (no web server) and report speed and peak VRAM.

    venv/bin/python tools/bench.py                                   # 10 s clips at 480p and 720p, Fast mode
    venv/bin/python tools/bench.py --resolutions 480p --duration 5   # quick smoke test (setup.sh runs this)
    venv/bin/python tools/bench.py --mode quality                    # 50-step CFG mode

Stop the GUI first (./stop.sh): it keeps the model in VRAM. Results are appended to bench_results.json.
"""
import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import backend as backends  # noqa: E402

# Prompt from LongCat-Video's run_demo_image_to_video.py, which uses assets/girl.png.
DEMO_PROMPT = ("A woman sits at a wooden table by the window in a cozy café. She reaches out with her right hand, "
               "picks up the white coffee cup from the saucer, and gently brings it to her lips to take a sip. After "
               "drinking, she places the cup back on the table and looks out the window, enjoying the peaceful atmosphere.")


def load_env(path):
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def port_in_use(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def main():
    load_env(APP_DIR / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--image", default=str(APP_DIR / "LongCat-Video" / "assets" / "girl.png"))
    parser.add_argument("--prompt", default=DEMO_PROMPT)
    parser.add_argument("--resolutions", default="480p,720p")
    parser.add_argument("--duration", type=float, default=10)
    parser.add_argument("--mode", choices=list(backends.MODES), default="fast")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default="bench")
    args = parser.parse_args()

    if port_in_use(int(os.environ.get("PORT", "8000"))):
        print("WARNING: something is listening on the GUI port. If it is the GUI, it holds the model in VRAM "
              "and this run may run out of memory. Stop it with ./stop.sh first.\n")
    settings = {k: v for k, v in os.environ.items()}
    settings.setdefault("MODEL_DIR", str(APP_DIR / "models" / "LongCat-Video"))
    settings.setdefault("LONGCAT_REPO", str(APP_DIR / "LongCat-Video"))
    backend = backends.create_backend(settings)
    gpu = backends.gpu_info() or {}
    print(f"GPU: {gpu.get('name', 'none')} ({gpu.get('total_gb', 0)} GB). Loading {backend.name}...", flush=True)
    backend.load()
    print(f"Loaded in {backend.load_seconds} s. {backend.message}\n", flush=True)

    from PIL import Image, ImageOps
    photo = ImageOps.exif_transpose(Image.open(args.image)).convert("RGB")
    out_dir = APP_DIR / "outputs" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = args.steps or backends.MODES[args.mode]["steps"]
    results = []
    for resolution in [r.strip() for r in args.resolutions.split(",") if r.strip()]:
        prepared, info = backend.prepare_image(photo, resolution)
        image_path = out_dir / f"{args.tag}_{resolution}_input.png"
        prepared.save(image_path)
        out_path = out_dir / f"{args.tag}_{resolution}_{args.mode}.mp4"
        print(f"=== {resolution}, {args.duration:g} s, {args.mode} mode, {steps} steps: {info['note']}", flush=True)
        last = {"segment": None}

        def progress(p):
            key = (p["phase"], p["segment"])
            if key != last["segment"]:
                last["segment"] = key
                print(f"  {p['message']}  ({p['percent']:.0f}%)", flush=True)

        row = {"tag": args.tag, "gpu": gpu.get("name"), "resolution": resolution, "mode": args.mode, "steps": steps,
               "duration_requested": args.duration, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        try:
            backend.generate(str(image_path), args.prompt, backends.DEFAULT_NEGATIVE_PROMPT, args.duration, resolution,
                             args.seed, steps, progress, mode=args.mode, out_path=str(out_path))
            video = backends.probe_video(out_path)
            row.update(ok=True, output=str(out_path), video=video, **backend.last_stats)
            print(f"  OK  {out_path}  {video['seconds']} s, {video['width']}x{video['height']} @ {video['fps']} fps, "
                  f"{video['frames']} frames", flush=True)
            print(f"  {row['seconds']} s total, {row['seconds_per_5s_video']} s per 5 s of video, "
                  f"peak VRAM {row['peak_vram_gb']} GB allocated / {row['peak_reserved_gb']} GB reserved\n", flush=True)
        except Exception as exc:  # noqa: BLE001
            message, fatal = backend.describe_error(exc)
            row.update(ok=False, error=message)
            print(f"  FAILED: {message}\n", flush=True)
            backend.release_memory()
            if fatal:
                results.append(row)
                break
        results.append(row)

    history_path = APP_DIR / "bench_results.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else []
    history_path.write_text(json.dumps(history + results, indent=2))

    print(f"{'resolution':<11}{'mode':<9}{'video':>8}{'wall time':>11}{'s / 5 s video':>15}{'peak VRAM':>12}")
    for r in results:
        if r["ok"]:
            print(f"{r['resolution']:<11}{r['mode']:<9}{r['video']['seconds']:>7}s{r['seconds']:>10}s"
                  f"{r['seconds_per_5s_video']:>14}s{str(r['peak_vram_gb']) + ' GB':>12}")
        else:
            print(f"{r['resolution']:<11}{r['mode']:<9}  FAILED: {r['error']}")
    sys.exit(0 if all(r["ok"] for r in results) else 1)


if __name__ == "__main__":
    main()
