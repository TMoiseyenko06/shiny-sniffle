# LongCat Video GUI

A private, phone-friendly web app for turning a photo plus a prompt into a 5–60 second video with
[LongCat-Video](https://github.com/meituan-longcat/LongCat-Video) (weights:
[meituan-longcat/LongCat-Video](https://huggingface.co/meituan-longcat/LongCat-Video)). It runs on a
rented GPU (vast.ai). You open it in your phone's browser, upload a photo, and watch or download the result.

## Quick start on a fresh vast.ai instance

Pick an instance with **≥ 48 GB VRAM** (A6000 / L40S / A100 / H100), **Max CUDA ≥ 12.4**, about
**150 GB of disk** (the weights alone are 83 GB) and **≥ 96 GB RAM**. In the instance's docker options, open the GUI port (`-p 8000:8000`).

```bash
cd /workspace
git clone <this repo> longcat-gui
cd longcat-gui
./setup.sh        # venv, PyTorch, flash-attn, LongCat code + weights, .env, 5 s smoke test
./start.sh        # prints the phone URL and the password
```

`setup.sh` can be re-run at any time. It skips finished steps, and it skips the weight download when
`models/LongCat-Video/` is complete.

## Start, stop, logs

| What | Command |
| --- | --- |
| Start in the background (survives SSH logout, restarts itself after a crash) | `./start.sh` |
| Stop | `./stop.sh` |
| Restart (after editing `.env` or the code) | `./restart.sh` |
| Watch the log | `tail -f server.log` |

To start it automatically when the instance boots, put `/workspace/longcat-gui/start.sh` in vast.ai's on-start script.

## Change the port

Set `PORT=` in `.env`, then run `./restart.sh`. For a one-off override, use `./start.sh 7860`.
The server listens on `0.0.0.0:PORT`. vast.ai maps that port to a random external port, so open
`http://<instance-ip>:<external-port>` on your phone. `start.sh` prints the exact URL when vast.ai
provides `PUBLIC_IPADDR` and `VAST_TCP_PORT_<PORT>`.

## Password

The password is `GUI_PASSWORD` in `.env`; `setup.sh` generates a random one. If you empty it, the server
makes up a new password on every start and prints it in `server.log`. Logins use a signed cookie that
lasts 30 days. Changing the password logs out every session. After 5 wrong passwords from one IP
(or 30 overall) in 15 minutes, further attempts are refused.

## Where files live

```
/workspace/longcat-gui/
├── server.py, backend.py, static/   the app (FastAPI + plain HTML/CSS/JS)
├── setup.sh start.sh stop.sh restart.sh
├── .env                              settings (port, password, memory mode, …)
├── LongCat-Video/                    official LongCat-Video code (cloned by setup.sh)
├── models/LongCat-Video/             weights from Hugging Face
├── uploads/                          your photos: original, model-sized input, thumbnail
├── outputs/                          finished videos (<job id>.mp4)
├── jobs.db                           job history (SQLite)
├── server.log
└── tools/                            bench.py, test_official.sh, e2e_test.py, download_weights.py
```

## How a video is made

* The photo is rotated upright (EXIF), center-cropped to the nearest aspect ratio the model supports,
  and resized. The app tells you exactly what it did.
* **Segment 1** is image-to-video: 93 frames at 480p, which is 6.2 s at the model's native 15 fps.
  **Every further segment** continues from the last 13 frames and adds 80 frames (5.3 s). The video is
  then trimmed to the length you asked for. A 60 s video takes 12 segments.
* **480p** output is 15 fps. **720p** follows the official coarse-to-fine path: the 480p video is
  refined segment by segment with the refinement LoRA into 720p at 30 fps. It is noticeably slower.
* **Fast** mode (default) uses the distilled LoRA: 16 steps, no CFG, and the negative prompt is
  ignored. **Quality** mode runs 50 steps with CFG 4.0 and the negative prompt.
* One job runs on the GPU at a time and the rest wait in a queue. Jobs survive page refreshes and
  server restarts. A job that was running during a restart is marked failed with a Retry button.
* GPU memory (`OFFLOAD=auto` in `.env`):
  * ≥ 70 GB: everything stays on the GPU.
  * ≥ 40 GB: the text encoder waits in CPU RAM and the continuation KV cache is offloaded.
  * < 40 GB: DiT blocks are also swapped from RAM, the VAE is tiled, and only 480p is offered.
  This mode is slow, and a 24 GB card is not a practical target.
* If the GPU runs out of memory, the job fails with a readable message and the server frees the memory
  and keeps running. On an unrecoverable CUDA error, the server exits and `start.sh`'s supervisor
  restarts it.

## Checks and benchmarks

```bash
./stop.sh                                  # these need the GPU to themselves
tools/test_official.sh                     # LongCat's own image-to-video demo, from the command line
venv/bin/python tools/bench.py             # 10 s at 480p and 720p: time per 5 s of video, peak VRAM
venv/bin/python tools/bench.py --mode quality --resolutions 480p
./start.sh
```

`bench.py` appends its results to `bench_results.json`. The UI estimates job time from finished jobs.
Use the measured peak VRAM to set `DEFAULT_RESOLUTION` / `MAX_RESOLUTION` in `.env` if the automatic
choice does not suit your card.

Browser test at phone width (login, upload, progress, playback, download, error display):

```bash
venv/bin/pip install playwright && venv/bin/playwright install --with-deps chromium
venv/bin/python tools/e2e_test.py --url http://127.0.0.1:8000 --password "$(sed -n 's/^GUI_PASSWORD=//p' .env)"
```

### Trying the interface without a GPU

Set `BACKEND=mock` in `.env` to swap the model for a CPU stand-in. It shows fake progress and makes a
slow zoom over the photo. A prompt containing "simulate oom" fails the way a real out-of-memory error
would. All you need is `pip install -r requirements-gui.txt`.

## Troubleshooting

* **Status bar says "Model failed to load"**: the reason is shown under the status bar and in
  `server.log`. Usually this means the weights are missing (`./setup.sh`) or there is not enough CPU RAM
  to load the 13.6B DiT.
* **flash-attn**: `setup.sh` installs a prebuilt wheel. If it cannot run on the card, setup switches the
  DiT to xformers by editing `models/LongCat-Video/dit/config.json`.
* **Blackwell GPUs (RTX 50xx, B200)**: setup uses torch 2.7.1 + CUDA 12.8 instead of the README's
  torch 2.6.0 + CUDA 12.4, which has no kernels for them.
* **Locked out after typing the password wrong**: wait 15 minutes, or run `./restart.sh`.
