"""LongCat-Video web GUI: FastAPI app, single-GPU job queue and password login.

    python server.py [--port 8000] [--host 0.0.0.0]      (start.sh runs this for you)

Settings come from environment variables, with .env next to this file as the fallback.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import hmac
import io
import json
import logging
import math
import os
import queue
import random
import re
import secrets
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from itsdangerous import BadSignature, TimestampSigner
from PIL import Image, ImageOps, UnidentifiedImageError

import backend as backends

APP_DIR = Path(__file__).resolve().parent


def load_env_file(path):
    """Minimal .env reader: KEY=VALUE lines; real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip().removeprefix("export "), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(APP_DIR / ".env")

DATA_DIR = Path(os.environ.get("DATA_DIR", APP_DIR))
UPLOADS, OUTPUTS, DB_PATH = DATA_DIR / "uploads", DATA_DIR / "outputs", DATA_DIR / "jobs.db"
STATIC = APP_DIR / "static"
MAX_UPLOAD_MB = float(os.environ.get("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD = int(MAX_UPLOAD_MB * 1024 * 1024)
SESSION_DAYS = float(os.environ.get("SESSION_DAYS", "30"))
SETTINGS = {key: os.environ.get(key, default) for key, default in {
    "BACKEND": "longcat",
    "MODEL_DIR": str(DATA_DIR / "models" / "LongCat-Video"),
    "LONGCAT_REPO": str(APP_DIR / "LongCat-Video"),
    "OFFLOAD": "auto", "ENABLE_COMPILE": "0", "ACTIVATION_RESERVE_GB": "10",
    "REFINE_STEPS": "50", "GUIDANCE_SCALE": "4.0",
    "DEFAULT_RESOLUTION": "auto", "MAX_RESOLUTION": "auto",
    "MOCK_STEP_SECONDS": "0.15", "MOCK_LOAD_SECONDS": "2",
}.items()}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("longcat-gui")


class QuietPolling(logging.Filter):
    """Keep server.log readable: drop access-log lines for the UI's frequent polling."""
    def filter(self, record):
        message = record.getMessage()
        return not any(p in message for p in ('"GET /api/jobs HTTP', '"GET /api/status HTTP', "/events HTTP"))



# ----- authentication ----------------------------------------------------------------------------

PASSWORD = os.environ.get("GUI_PASSWORD", "").strip()
PASSWORD_GENERATED = not PASSWORD
if PASSWORD_GENERATED:
    PASSWORD = secrets.token_urlsafe(9)


def session_secret():
    if os.environ.get("SESSION_SECRET"):
        return os.environ["SESSION_SECRET"]
    path = DATA_DIR / ".session_secret"
    if not path.exists():
        path.write_text(secrets.token_hex(32))
        path.chmod(0o600)
    return path.read_text().strip()


COOKIE = "lcg_session"
SIGNER = TimestampSigner(session_secret(), salt="longcat-gui-session")
# Changing GUI_PASSWORD invalidates existing sessions.
PASSWORD_TAG = hashlib.sha256(("longcat-gui:" + PASSWORD).encode()).hexdigest()[:16]


def is_authed(request: Request) -> bool:
    token = request.cookies.get(COOKIE)
    if not token:
        return False
    try:
        value = SIGNER.unsign(token, max_age=SESSION_DAYS * 86400).decode()
    except BadSignature:
        return False
    return hmac.compare_digest(value, PASSWORD_TAG)


def require_auth(request: Request):
    if not is_authed(request):
        raise HTTPException(401, "Not logged in.")


class LoginLimiter:
    """At most PER_IP failed logins per IP and GLOBAL failures overall in a sliding WINDOW."""
    WINDOW, PER_IP, GLOBAL = 15 * 60, 5, 30

    def __init__(self):
        self.failures = {}
        self.lock = threading.Lock()

    def _recent(self, key, now):
        times = [t for t in self.failures.get(key, []) if now - t < self.WINDOW]
        self.failures[key] = times
        return times

    def wait_seconds(self, ip):
        now = time.time()
        with self.lock:
            for key, limit in ((ip, self.PER_IP), ("*", self.GLOBAL)):
                times = self._recent(key, now)
                if len(times) >= limit:
                    return int(self.WINDOW - (now - times[0])) + 1
        return 0

    def failed(self, ip):
        with self.lock:
            for key in (ip, "*"):
                self.failures.setdefault(key, []).append(time.time())

    def succeeded(self, ip):
        with self.lock:
            self.failures.pop(ip, None)


limiter = LoginLimiter()


# ----- job storage ---------------------------------------------------------------------------------

class JobStore:
    """Jobs live in memory and are mirrored to SQLite (one JSON blob per job)."""

    def __init__(self, path):
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, created REAL, data TEXT)")
        self.jobs = {row[0]: json.loads(row[1]) for row in self.db.execute("SELECT id, data FROM jobs")}
        self.last_write = {}

    def _write(self, job):
        self.db.execute("INSERT OR REPLACE INTO jobs VALUES (?, ?, ?)", (job["id"], job["created_at"], json.dumps(job)))
        self.last_write[job["id"]] = time.time()

    def add(self, job):
        with self.lock:
            self.jobs[job["id"]] = job
            self._write(job)

    def get(self, job_id):
        with self.lock:
            job = self.jobs.get(job_id)
            return copy.deepcopy(job) if job else None

    def update(self, job_id, persist=True, **fields):
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:          # deleted meanwhile
                return None
            job.update(fields)
            # progress ticks are frequent; write them to disk at most every 5 seconds
            if persist or time.time() - self.last_write.get(job_id, 0) > 5:
                self._write(job)
            return copy.deepcopy(job)

    def delete(self, job_id):
        with self.lock:
            job = self.jobs.pop(job_id, None)
            self.db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            return job

    def all(self):
        with self.lock:
            return sorted(copy.deepcopy(list(self.jobs.values())), key=lambda j: j["created_at"], reverse=True)


# ----- GPU worker ------------------------------------------------------------------------------------

class Worker:
    """One thread owns the GPU: loads the model once, then runs queued jobs one at a time."""

    def __init__(self, store, backend):
        self.store, self.backend = store, backend
        self.queue = queue.Queue()
        self.current = None
        self.cancelled = set()
        threading.Thread(target=self._run, name="gpu-worker", daemon=True).start()

    def submit(self, job_id):
        self.queue.put(job_id)

    def _run(self):
        try:
            self.backend.load()
            log.info("Model ready in %ss (%s)", self.backend.load_seconds, self.backend.message)
        except Exception as exc:  # noqa: BLE001
            log.exception("Model failed to load")
            self.backend.state = "error"
            self.backend.error = self.backend.describe_error(exc)[0]
            self.backend.message = "Model failed to load"
        while True:
            job_id = self.queue.get()
            try:
                self._run_job(job_id)
            except Exception:  # noqa: BLE001 - never let the worker thread die
                log.exception("Unexpected worker error on job %s", job_id)

    def _run_job(self, job_id):
        job = self.store.get(job_id)
        if not job or job["status"] != "queued":
            return
        if self.backend.state != "ready":
            self.store.update(job_id, status="failed", finished_at=time.time(),
                              error=f"Model is not available: {self.backend.error or self.backend.message}")
            return
        log.info("Job %s started: %ss %s %s, %s steps, seed %s", job_id, job["duration_s"], job["resolution"],
                 job["mode"], job["steps"], job["seed"])
        self.current = job_id
        self.store.update(job_id, status="running", started_at=time.time(), finished_at=None, error=None,
                          progress={"phase": "start", "percent": 0, "segment": 0, "segments": job["plan"]["segments"],
                                    "step": 0, "steps": 0, "message": "Starting"})
        part, final = OUTPUTS / f"{job_id}.part.mp4", OUTPUTS / f"{job_id}.mp4"
        error, fatal = None, False

        def progress_cb(progress):
            if job_id in self.cancelled:
                raise backends.Cancelled()
            self.store.update(job_id, persist=False, progress=progress)

        try:
            self.backend.generate(
                image_path=str(UPLOADS / job["input"]), prompt=job["prompt"], negative_prompt=job["negative_prompt"],
                duration_s=job["duration_s"], resolution=job["resolution"], seed=job["seed"], steps=job["steps"],
                progress_cb=progress_cb, mode=job["mode"], out_path=str(part))
            os.replace(part, final)
            try:
                video = backends.probe_video(final)
            except Exception:  # noqa: BLE001
                log.exception("Could not probe %s", final)
                video = None
            stats = dict(self.backend.last_stats)
            done = self.store.update(job_id, status="done", finished_at=time.time(), output=final.name, video=video,
                                     stats=stats, progress={"phase": "done", "percent": 100, "segment": job["plan"]["segments"],
                                                            "segments": job["plan"]["segments"], "step": 0, "steps": 0,
                                                            "message": "Done"})
            if done is None:   # deleted while it was finishing
                final.unlink(missing_ok=True)
            log.info("Job %s done: %s", job_id, stats)
        except backends.Cancelled:
            log.info("Job %s cancelled", job_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("Job %s failed", job_id)
            error, fatal = self.backend.describe_error(exc)
            where = ((self.store.get(job_id) or {}).get("progress") or {}).get("message")
            if where and where not in ("Starting", "Done"):
                error += f" (Failed during: {where}.)"
        finally:
            part.unlink(missing_ok=True)
            self.current = None
            self.cancelled.discard(job_id)
        # outside the except block, so the traceback (and the GPU tensors it references) is gone
        if error is not None:
            self.store.update(job_id, status="failed", finished_at=time.time(), error=error)
        if fatal:   # decide before touching CUDA again: on a broken context even empty_cache() raises
            log.critical("Unrecoverable CUDA error; exiting so start.sh restarts the server.")
            logging.shutdown()
            os._exit(3)
        try:
            self.backend.release_memory()
        except Exception:  # noqa: BLE001
            log.exception("Freeing GPU memory failed")


# ----- helpers -----------------------------------------------------------------------------------

_gpu_cache = {"at": 0.0, "value": None}


def gpu_status():
    if time.time() - _gpu_cache["at"] > 2:
        _gpu_cache.update(at=time.time(), value=backends.gpu_info())
    return _gpu_cache["value"]


def public_job(job, positions):
    out = dict(job)
    out["queue_position"] = positions.get(job["id"])
    out["thumb_url"] = f"/files/uploads/{job['thumb']}"
    out["input_url"] = f"/files/uploads/{job['input']}"
    out["video_url"] = f"/files/outputs/{job['output']}" if job.get("output") else None
    out["download_url"] = f"{out['video_url']}?download=1" if job.get("output") else None
    return out


def queue_positions(jobs):
    queued = sorted((j for j in jobs if j["status"] == "queued"), key=lambda j: j["queued_at"])
    return {j["id"]: i + 1 for i, j in enumerate(queued)}


def estimates(jobs):
    """Median seconds per segment of recent finished jobs, per resolution/mode, for the UI's ETA."""
    samples = {}
    for job in jobs:
        per_segment = (job.get("stats") or {}).get("seconds_per_segment")
        if job["status"] == "done" and per_segment:
            samples.setdefault(f"{job['resolution']}/{job['mode']}", []).append(per_segment)
    return {key: sorted(values[:5])[len(values[:5]) // 2] for key, values in samples.items()}


def bad_request(message, status=400):
    raise HTTPException(status, message)


def delete_job_files(job):
    for name in (job.get("image_orig"), job.get("input"), job.get("thumb")):
        if name:
            (UPLOADS / name).unlink(missing_ok=True)
    (OUTPUTS / f"{job['id']}.mp4").unlink(missing_ok=True)


def read_image(data: bytes):
    """Validate an upload and return (RGB image, extension, note about EXIF rotation)."""
    Image.MAX_IMAGE_PIXELS = 40_000_000   # Pillow refuses images over 2x this
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        if fmt not in ("JPEG", "MPO", "PNG", "WEBP"):   # MPO = JPEG with extra frames (iPhone portrait/HDR)
            bad_request(f"{fmt or 'This'} images are not supported. Use JPG, PNG or WebP.", 415)
        img.load()
    except HTTPException:
        raise
    except Image.DecompressionBombError:
        bad_request("That image has too many pixels (over 80 megapixels).", 413)
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        bad_request("That file is not a readable image. Use JPG, PNG or WebP.", 415)
    orientation = img.getexif().get(0x0112, 1)
    rotated = ImageOps.exif_transpose(img)
    note = "Rotated upright using the photo's EXIF orientation. " if orientation not in (None, 1) else ""
    if rotated.mode in ("RGBA", "LA", "P"):
        rotated = rotated.convert("RGBA")
        background = Image.new("RGB", rotated.size, (0, 0, 0))
        background.paste(rotated, mask=rotated.getchannel("A"))
        rotated = background
    rotated = rotated.convert("RGB")
    if min(rotated.size) < 128:
        bad_request(f"The image is only {rotated.width}×{rotated.height}. Use at least 128 px on the short side.")
    return rotated, {"JPEG": "jpg", "MPO": "jpg", "PNG": "png", "WEBP": "webp"}[fmt], note


def save_inputs(job_id, data, resolution):
    img, ext, rotation_note = read_image(data)
    prepared, info = backend.prepare_image(img, resolution)
    info["note"] = rotation_note + info["note"]
    orig_name, input_name, thumb_name = f"{job_id}_orig.{ext}", f"{job_id}.png", f"{job_id}_thumb.jpg"
    (UPLOADS / orig_name).write_bytes(data)
    prepared.save(UPLOADS / input_name)
    thumb = prepared.copy()
    thumb.thumbnail((320, 320))
    thumb.save(UPLOADS / thumb_name, quality=85)
    return {"image_orig": orig_name, "input": input_name, "thumb": thumb_name, "image_info": info}


# ----- app ------------------------------------------------------------------------------------------

backend = backends.create_backend(SETTINGS)
store: JobStore = None   # created in lifespan
worker: Worker = None


@asynccontextmanager
async def lifespan(app):
    global store, worker
    logging.getLogger("uvicorn.access").addFilter(QuietPolling())
    UPLOADS.mkdir(parents=True, exist_ok=True)
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    store = JobStore(DB_PATH)
    pending = []
    for job in store.all():
        if job["status"] == "running":
            store.update(job["id"], status="failed", finished_at=time.time(),
                         error="Interrupted because the server restarted. Press Retry to run it again.")
        elif job["status"] == "queued":
            pending.append(job)
    worker = Worker(store, backend)
    for job in sorted(pending, key=lambda j: j["queued_at"]):
        worker.submit(job["id"])
    port = os.environ.get("PORT", "8000")
    banner = [
        "=" * 64,
        f" LongCat GUI on http://0.0.0.0:{port}   backend: {backend.name}",
        f" Open from your phone: http://<instance-ip>:<external port mapped to {port}>",
        (f" Password (random, set GUI_PASSWORD in .env to keep one): {PASSWORD}" if PASSWORD_GENERATED
         else " Password: the GUI_PASSWORD value in .env"),
        "=" * 64,
    ]
    print("\n".join(banner), flush=True)
    yield


app = FastAPI(title="LongCat GUI", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
api = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])


def page(name):
    """Serve an HTML page with cache-busting versions for the CSS/JS it references."""
    version = str(int(max((STATIC / f).stat().st_mtime for f in ("app.js", "style.css"))))
    html = (STATIC / name).read_text().replace("__V__", version)
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/")
def index(request: Request):
    return page("index.html") if is_authed(request) else RedirectResponse("/login", 303)


@app.get("/login")
def login_page(request: Request):
    return RedirectResponse("/", 303) if is_authed(request) else page("login.html")


@app.post("/api/login")
async def login(request: Request):
    ip = request.client.host if request.client else "?"
    wait = limiter.wait_seconds(ip)
    if wait:
        return JSONResponse({"detail": f"Too many wrong passwords. Try again in {math.ceil(wait / 60)} min."},
                            429, headers={"Retry-After": str(wait)})
    try:
        password = str((await request.json()).get("password", ""))
    except (ValueError, AttributeError):
        password = ""
    if not hmac.compare_digest(password.encode(), PASSWORD.encode()):
        limiter.failed(ip)
        log.warning("Failed login from %s", ip)
        await asyncio.sleep(1.0)
        return JSONResponse({"detail": "Wrong password."}, 401)
    limiter.succeeded(ip)
    response = JSONResponse({"ok": True})
    response.set_cookie(COOKIE, SIGNER.sign(PASSWORD_TAG).decode(), max_age=int(SESSION_DAYS * 86400),
                        httponly=True, samesite="lax", secure=request.url.scheme == "https", path="/")
    return response


@app.post("/api/logout")
def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE, path="/")
    return response


@api.get("/status")
def status():
    jobs = store.all()
    return {
        "now": time.time(),
        "gpu": gpu_status(),
        "model": backend.info(),
        "queue": {"running": worker.current, "queued": sum(j["status"] == "queued" for j in jobs)},
        "config": {
            "resolutions": backend.resolutions(), "default_resolution": backend.default_resolution(),
            "modes": backends.MODES, "default_mode": "fast",
            "duration": {"min": backends.DURATION_MIN, "max": backends.DURATION_MAX, "default": backends.DURATION_DEFAULT},
            "segment": {"frames": backend.SEG_FRAMES, "cond": backend.COND_FRAMES, "fps": backend.BASE_FPS},
            "default_negative_prompt": backends.DEFAULT_NEGATIVE_PROMPT,
            "max_upload_mb": MAX_UPLOAD_MB,
            "estimates": estimates(jobs),
        },
    }


@api.get("/jobs")
def list_jobs():
    jobs = store.all()
    positions = queue_positions(jobs)
    return {"now": time.time(), "jobs": [public_job(j, positions) for j in jobs]}


def get_job_or_404(job_id):
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


@api.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = get_job_or_404(job_id)
    return public_job(job, queue_positions(store.all()))


@api.post("/jobs", status_code=201)
async def create_job(request: Request):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_UPLOAD + 1024 * 1024:
        bad_request(f"The upload is {int(length) / 2**20:.1f} MB; the limit is {MAX_UPLOAD_MB:g} MB.", 413)
    if backend.state == "error":
        bad_request(f"The model failed to load, so no jobs can run: {backend.error}", 503)
    form = await request.form()

    prompt = str(form.get("prompt") or "").strip()
    if not prompt:
        bad_request("Please describe what should happen in the video.")
    if len(prompt) > 2000:
        bad_request("The prompt is too long (2000 characters max).")
    negative = str(form.get("negative_prompt") if form.get("negative_prompt") is not None
                   else backends.DEFAULT_NEGATIVE_PROMPT).strip()[:2000]
    try:
        duration = float(form.get("duration") or backends.DURATION_DEFAULT)
    except ValueError:
        bad_request("Duration must be a number of seconds.")
    if not backends.DURATION_MIN <= duration <= backends.DURATION_MAX:
        bad_request(f"Duration must be between {backends.DURATION_MIN} and {backends.DURATION_MAX} seconds.")
    resolution = str(form.get("resolution") or backend.default_resolution())
    if resolution not in backend.resolutions():
        bad_request(f"Resolution '{resolution[:20]}' is not available on this GPU. "
                    f"Choose {' or '.join(backend.resolutions())}.")
    mode = str(form.get("mode") or "fast")
    if mode not in backends.MODES:
        bad_request("Mode must be 'fast' or 'quality'.")
    limits = backends.MODES[mode]
    try:
        steps = int(form.get("steps") or limits["steps"])
    except ValueError:
        bad_request("Steps must be a whole number.")
    if not limits["min_steps"] <= steps <= limits["max_steps"]:
        bad_request(f"Steps for {limits['label']} mode must be between {limits['min_steps']} and {limits['max_steps']}.")
    seed_text = str(form.get("seed") or "").strip()
    if seed_text:
        if not re.fullmatch(r"\d{1,10}", seed_text) or int(seed_text) > 2**32 - 1:
            bad_request("Seed must be a whole number between 0 and 4294967295.")
        seed, seed_random = int(seed_text), False
    else:
        seed, seed_random = random.randint(0, 2**31 - 1), True

    upload, source_id = form.get("image"), str(form.get("source_job") or "")
    if upload is not None and not isinstance(upload, str) and upload.filename:
        data = await upload.read(MAX_UPLOAD + 1)
        if len(data) > MAX_UPLOAD:
            bad_request(f"The image is larger than {MAX_UPLOAD_MB:g} MB.", 413)
        name = upload.filename.lower()
        if "." in name and not name.endswith((".jpg", ".jpeg", ".png", ".webp")):
            bad_request("Only JPG, PNG and WebP images are supported.", 415)
    elif source_id:
        source = get_job_or_404(source_id)
        data = (UPLOADS / source["image_orig"]).read_bytes()
    else:
        bad_request("Choose a photo first.")

    job_id = uuid.uuid4().hex[:12]
    files = await run_in_threadpool(save_inputs, job_id, data, resolution)
    now = time.time()
    plan = backend.plan(duration, resolution)
    job = {
        "id": job_id, "created_at": now, "queued_at": now, "started_at": None, "finished_at": None,
        "status": "queued", "prompt": prompt, "negative_prompt": negative, "duration_s": duration,
        "resolution": resolution, "mode": mode, "steps": steps, "seed": seed, "seed_random": seed_random,
        **files, "plan": plan, "output": None, "video": None, "stats": None, "error": None,
        "progress": {"phase": "queued", "percent": 0, "segment": 0, "segments": plan["segments"],
                     "step": 0, "steps": 0, "message": "Waiting in queue"},
    }
    store.add(job)
    worker.submit(job_id)
    log.info("Job %s queued: %s", job_id, files["image_info"]["note"])
    return {"id": job_id, "image_note": files["image_info"]["note"],
            "job": public_job(job, queue_positions(store.all()))}


@api.post("/jobs/{job_id}/retry")
def retry_job(job_id: str):
    job = get_job_or_404(job_id)
    if job["status"] != "failed":
        bad_request("Only failed jobs can be retried.", 409)
    now = time.time()
    store.update(job_id, status="queued", queued_at=now, started_at=None, finished_at=None, error=None,
                 output=None, video=None, stats=None,
                 progress={"phase": "queued", "percent": 0, "segment": 0, "segments": job["plan"]["segments"],
                           "step": 0, "steps": 0, "message": "Waiting in queue"})
    worker.submit(job_id)
    return public_job(store.get(job_id), queue_positions(store.all()))


@api.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    job = get_job_or_404(job_id)
    if job["status"] == "running":
        worker.cancelled.add(job_id)   # the worker stops at the next denoising step
    store.delete(job_id)
    delete_job_files(job)
    return {"ok": True}


@api.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request):
    """Server-sent events: the job JSON whenever it changes, until it is done, failed or deleted."""
    get_job_or_404(job_id)

    async def stream():
        last, last_sent = None, 0.0
        yield "retry: 3000\n\n"
        while not await request.is_disconnected():
            job = store.get(job_id)
            if job is None:
                yield "event: deleted\ndata: {}\n\n"
                return
            payload = json.dumps({"now": time.time(), "job": public_job(job, queue_positions(store.all()))})
            if job != last:
                yield f"data: {payload}\n\n"
                last, last_sent = job, time.time()
            elif time.time() - last_sent > 15:
                yield ": keep-alive\n\n"
                last_sent = time.time()
            if job["status"] in ("done", "failed"):
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


app.include_router(api)


@app.get("/files/{kind}/{name}", dependencies=[Depends(require_auth)])
def files(kind: str, name: str, download: int = 0):
    """Uploads and outputs. FileResponse answers HTTP Range requests, which mobile video seeking needs."""
    base = {"uploads": UPLOADS, "outputs": OUTPUTS}.get(kind)
    if base is None or not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name.startswith("."):
        raise HTTPException(404, "Not found.")
    path = base / name
    if not path.is_file():
        raise HTTPException(404, "Not found.")
    if download:
        return FileResponse(path, filename=f"longcat_{name}", content_disposition_type="attachment")
    return FileResponse(path, headers={"Cache-Control": "private, max-age=86400"})


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    os.environ["PORT"] = str(args.port)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", timeout_graceful_shutdown=3)


if __name__ == "__main__":
    main()
