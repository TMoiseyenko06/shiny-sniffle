"""Video generation backends for the LongCat GUI.

server.py only talks to a backend through this small interface, so another model
(for example Wan 2.2 I2V) can be added later as one more VideoBackend subclass:

    load()                                  load weights once (runs in the GPU worker thread)
    info()                                  state for the status bar
    resolutions() / default_resolution()    what the UI may offer on this GPU
    plan(duration_s, resolution)            segments, fps and frame count for a request
    prepare_image(img, resolution)          crop/resize a photo to the model's input size
    generate(image_path, prompt, negative_prompt, duration_s, resolution,
             seed, steps, progress_cb, mode, out_path) -> out_path
    describe_error(exc) / release_memory()  readable errors and GPU cleanup after a job

LongCatBackend runs meituan-longcat/LongCat-Video. MockBackend fakes it on the CPU
so the web interface can be tested without a GPU (BACKEND=mock).
"""
from __future__ import annotations

import contextlib
import gc
import importlib.util
import inspect
import itertools
import math
import os
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
from PIL import Image

# Negative prompt used by the official LongCat-Video demos.
DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, "
    "static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, "
    "extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, "
    "fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards"
)

# "fast" uses the distilled cfg_step LoRA (16 steps, no CFG), as in run_demo_image_to_video.py.
MODES = {
    "fast": {"label": "Fast", "hint": "Distilled LoRA, no CFG. Several times faster; ignores the negative prompt.",
             "steps": 16, "min_steps": 4, "max_steps": 50},
    "quality": {"label": "Quality", "hint": "Full model with CFG 4.0 and the negative prompt. Much slower.",
                "steps": 50, "min_steps": 10, "max_steps": 100},
}
RESOLUTIONS = ("480p", "720p")
DURATION_MIN, DURATION_MAX, DURATION_DEFAULT = 5, 60, 10


class Cancelled(Exception):
    """Raise from progress_cb to stop the current generation."""


class GenerationError(Exception):
    """An error whose message is written for the person using the GUI."""


def gpu_info():
    """Name and memory of GPU 0 from nvidia-smi (works before torch is imported)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True).stdout
        name, used, total, util = [s.strip() for s in out.splitlines()[0].split(",")]
        return {"name": name, "used_gb": round(float(used) / 1024, 1), "total_gb": round(float(total) / 1024, 1),
                "util": int(util) if util.isdigit() else None}
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class Mp4Writer:
    """Streams RGB frames into an H.264 MP4 that phones can play and seek (yuv420p + faststart)."""

    def __init__(self, path, size, fps, crf=18, preset="medium"):
        import imageio_ffmpeg
        self.size = tuple(size)
        self.frames = 0
        self._gen = imageio_ffmpeg.write_frames(
            str(path), self.size, fps=fps, codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
            quality=None, macro_block_size=1, ffmpeg_log_level="error",
            output_params=["-crf", str(crf), "-preset", preset, "-movflags", "+faststart"])
        self._gen.send(None)

    def add(self, frame):
        if isinstance(frame, Image.Image):
            if frame.size != self.size:
                frame = frame.resize(self.size, Image.BICUBIC)
            frame = np.asarray(frame.convert("RGB"))
        self._gen.send(np.ascontiguousarray(frame, dtype=np.uint8))
        self.frames += 1

    def close(self):
        self._gen.close()


def probe_video(path):
    """Frame count, duration, fps and size of a finished video (also proves it decodes)."""
    import imageio_ffmpeg
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    reader.close()
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(path))
    width, height = meta["size"]
    return {"frames": frames, "seconds": round(seconds, 2), "fps": meta.get("fps"), "width": width, "height": height}


def _to_frames(array, size=None):
    """Pipeline output (T, H, W, 3 floats in 0..1) -> list of PIL frames."""
    frames = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)) for f in array]
    if size:
        frames = [f if f.size == size else f.resize(size, Image.BICUBIC) for f in frames]
    return frames


def _closest_bucket(bucket_config, ratio):
    """Same rule as LongCatVideoPipeline.get_condition_shape: nearest height/width ratio key."""
    key = sorted(bucket_config.keys(), key=lambda k: abs(float(k) - ratio))[0]
    height, width = bucket_config[key][0]
    return int(width), int(height)


class _Progress:
    """Turns per-step callbacks into one overall percentage for the job."""

    def __init__(self, callback, segments, refine, refine_weight):
        self.callback = callback
        self.segments = segments
        self.total = segments * (1 + (refine_weight if refine else 0))
        self.done = 0.0
        self.weight = 1.0
        self.phase, self.segment, self.message = "start", 0, ""

    def begin(self, phase, segment, weight, message):
        self.phase, self.segment, self.weight, self.message = phase, segment, weight, message
        self.step(0, 0)

    def step(self, n, total):
        fraction = n / total if total else 0.0
        percent = min(99.0, 100.0 * (self.done + self.weight * fraction) / self.total)
        self.callback({"phase": self.phase, "segment": self.segment, "segments": self.segments,
                       "step": n, "steps": total, "percent": round(percent, 1), "message": self.message})

    def finish_segment(self):
        self.done += self.weight

    def note(self, phase, message):
        self.phase, self.message, self.weight = phase, message, 0.0
        self.step(0, 0)


class VideoBackend:
    """Shared behaviour. Subclasses implement load(), generate() and _bucket_config()."""

    name = "base"
    SEG_FRAMES = 93      # frames per generated segment (model native rate: 15 fps)
    COND_FRAMES = 13     # overlap frames each continuation segment is conditioned on
    BASE_FPS = 15
    REFINE_WEIGHT = 2.0  # rough cost of one 720p refine segment vs one 480p segment (progress bar only)

    def __init__(self, settings):
        self.settings = settings
        self.state = "idle"          # idle | loading | ready | error
        self.message = "Waiting to load"
        self.error = None
        self.load_started = None
        self.load_seconds = None
        self.memory_mode = None
        self.last_stats = {}
        gpu = gpu_info()
        self.vram_total_gb = gpu["total_gb"] if gpu else 0.0

    # ----- what the UI may offer -------------------------------------------------------------
    def resolutions(self):
        cap = self.settings.get("MAX_RESOLUTION", "auto")
        if cap not in RESOLUTIONS:
            cap = "480p" if 0 < self.vram_total_gb < 40 else "720p"
        return list(RESOLUTIONS[: RESOLUTIONS.index(cap) + 1])

    def default_resolution(self):
        allowed = self.resolutions()
        wanted = self.settings.get("DEFAULT_RESOLUTION", "auto")
        if wanted in allowed:
            return wanted
        return "720p" if "720p" in allowed and self.vram_total_gb >= 70 else "480p"

    def info(self):
        loading_for = None
        if self.state == "loading" and self.load_started:
            loading_for = round(time.time() - self.load_started)
        return {"name": self.name, "state": self.state, "message": self.message, "error": self.error,
                "loading_for": loading_for, "load_seconds": self.load_seconds, "memory_mode": self.memory_mode}

    # ----- request planning --------------------------------------------------------------------
    def plan(self, duration_s, resolution):
        """First segment comes from the image; each continuation adds SEG_FRAMES - COND_FRAMES frames."""
        new_per_segment = self.SEG_FRAMES - self.COND_FRAMES
        needed = math.ceil(duration_s * self.BASE_FPS)
        segments = 1 + max(0, math.ceil((needed - self.SEG_FRAMES) / new_per_segment))
        stage1_frames = self.SEG_FRAMES + (segments - 1) * new_per_segment
        refine = resolution == "720p"          # 720p = 480p video refined by the refinement LoRA (2x fps)
        fps = self.BASE_FPS * (2 if refine else 1)
        frames = min(round(duration_s * fps), stage1_frames * (2 if refine else 1))
        return {"segments": segments, "stage1_frames": stage1_frames, "refine": refine, "fps": fps,
                "frames": frames, "seconds": round(frames / fps, 2)}

    def _bucket_config(self, resolution, scale):
        raise NotImplementedError

    def sizes(self, width, height, resolution):
        """((gen_w, gen_h), (video_w, video_h)) the model will use for a photo of this shape."""
        gen = _closest_bucket(self._bucket_config("480p", 16), height / width)
        if resolution != "720p":
            return gen, gen
        # generate_refine buckets with scale_factor_spatial 8 * 2 * 4 = 64
        return gen, _closest_bucket(self._bucket_config("720p", 64), gen[1] / gen[0])

    def prepare_image(self, img, resolution):
        """Center-crop a photo to the nearest aspect-ratio bucket and resize. Returns (image, info)."""
        width, height = img.size
        (gen_w, gen_h), (vid_w, vid_h) = self.sizes(width, height, resolution)
        aspect = gen_w / gen_h
        if width / height > aspect:
            crop_w, crop_h = round(height * aspect), height
        else:
            crop_w, crop_h = width, round(width / aspect)
        cropped = abs(crop_w - width) > 1 or abs(crop_h - height) > 1
        if cropped:
            left, top = (width - crop_w) // 2, (height - crop_h) // 2
            img = img.crop((left, top, left + crop_w, top + crop_h))
        else:
            crop_w, crop_h = width, height
        if resolution == "720p":
            # keep the exact 480p-bucket aspect, at 720p detail for the refinement stage
            scale = max(vid_w, vid_h) / max(gen_w, gen_h)
            target = (round(gen_w * scale), round(gen_h * scale))
        else:
            target = (gen_w, gen_h)
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)

        fps = self.plan(DURATION_MIN, resolution)["fps"]
        parts = [f"{width}×{height} photo"]
        if cropped:
            parts.append(f"center-cropped to {crop_w}×{crop_h} to match the model's {gen_w}×{gen_h} aspect ratio")
        if target != (crop_w, crop_h):
            parts.append(f"{'upscaled' if target[0] > crop_w else 'downscaled'} to {target[0]}×{target[1]}")
        note = ", ".join(parts) + f". Video will be {vid_w}×{vid_h} at {fps} fps."
        return img, {"original": [width, height], "crop": [crop_w, crop_h] if cropped else None,
                     "input": list(target), "video": [vid_w, vid_h], "fps": fps, "note": note}

    # ----- errors and cleanup ------------------------------------------------------------------
    def describe_error(self, exc):
        """Return (message for the user, fatal). Fatal = CUDA context is broken; restart the process."""
        if isinstance(exc, GenerationError):
            return str(exc), False
        text = f"{type(exc).__name__}: {exc}".strip()
        low = text.lower()
        if "out of memory" in low or "outofmemoryerror" in low or "cublas_status_alloc_failed" in low:
            return ("The GPU ran out of memory. Try 480p, a shorter duration or Fast mode. "
                    "The server freed the memory and is ready for the next job."), False
        fatal = any(s in low for s in ("cuda error", "illegal memory access", "device-side assert",
                                       "cudnn_status_internal_error", "nccl error"))
        message = text if len(text) <= 400 else text[:400] + "…"
        if fatal:
            message += " The GPU is in a bad state, so the server restarts itself. Retry in a minute or two."
        return message, fatal

    def release_memory(self):
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def load(self):
        raise NotImplementedError

    def generate(self, image_path, prompt, negative_prompt, duration_s, resolution, seed, steps,
                 progress_cb, mode="fast", out_path=None):
        raise NotImplementedError


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _clone(value):
    if hasattr(value, "clone"):
        return value.clone()
    if isinstance(value, (list, tuple)):
        return type(value)(_clone(v) for v in value)
    return value


class _BlockOffloader:
    """Runs a DiT that does not fit in VRAM: the first N transformer blocks stay on the GPU and
    the rest live in pinned CPU memory, copied in just before they run and dropped right after."""

    def __init__(self, dit, device, reserve_bytes):
        import torch
        for name, child in dit.named_children():
            if name != "blocks":
                child.to(device)
        for tensors in (dit._parameters, dit._buffers):
            for t in tensors.values():
                if t is not None:
                    t.data = t.data.to(device)
        blocks = dit.blocks
        per_block = sum(t.numel() * t.element_size() for t in itertools.chain(blocks[0].parameters(), blocks[0].buffers()))
        free, _ = torch.cuda.mem_get_info()
        self.kept = max(0, min(len(blocks), int((free - reserve_bytes) // max(per_block, 1))))
        self.total = len(blocks)
        self.swapped = []
        for i, block in enumerate(blocks):
            if i < self.kept:
                block.to(device)
                continue
            pairs = []
            for t in itertools.chain(block.parameters(), block.buffers()):
                cpu = t.data
                with contextlib.suppress(RuntimeError):   # pinning can fail on low-RAM hosts
                    cpu = cpu.pin_memory()
                t.data = cpu
                pairs.append((t, cpu))
            self.swapped.extend(pairs)
            block.register_forward_pre_hook(self._loader(pairs, device))
            block.register_forward_hook(self._unloader(pairs))

    @staticmethod
    def _loader(pairs, device):
        def hook(module, args):
            for t, cpu in pairs:
                t.data = cpu.to(device, non_blocking=True)
        return hook

    @staticmethod
    def _unloader(pairs):
        def hook(module, args, output):
            for t, cpu in pairs:
                t.data = cpu
        return hook

    def reset(self):
        """Put every swapped block back on the CPU (needed after an OOM interrupted a forward pass)."""
        for t, cpu in self.swapped:
            t.data = cpu


class LongCatBackend(VideoBackend):
    """meituan-longcat/LongCat-Video, following run_demo_image_to_video.py and run_demo_long_video.py.

    Segment 1: generate_i2v from the photo (93 frames at 480p / 15 fps).
    Segments 2..N: generate_vc continues from the last 13 frames (+80 new frames each).
    720p: every 93-frame window is refined with the refinement LoRA (720p, 30 fps), like the
    official long-video demo, with the photo as the condition for the first window.
    """

    name = "LongCat-Video"

    def __init__(self, settings):
        super().__init__(settings)
        self.repo_dir = Path(settings["LONGCAT_REPO"])
        self.model_dir = Path(settings["MODEL_DIR"])
        self.pipe = None
        self.offload_text_encoder = False
        self.offload_kv_cache = False
        self._offloader = None
        self._step_hook = None
        self._prompt_cache = {}
        self._te_depth = 0
        self._bucket_module = None

    def _bucket_config(self, resolution, scale):
        if self._bucket_module is None:   # plain dict module; load by path to avoid importing torch
            path = self.repo_dir / "longcat_video" / "utils" / "bukcet_config.py"
            spec = importlib.util.spec_from_file_location("longcat_bucket_config", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self._bucket_module = module
        return self._bucket_module.get_bucket_config(resolution, scale_factor_spatial=scale)

    def _memory_mode(self, total_gb):
        wanted = self.settings.get("OFFLOAD", "auto")
        if wanted in ("none", "text_encoder", "blocks"):
            return wanted
        if total_gb >= 70:
            return "none"           # everything resident (~40 GB of weights)
        if total_gb >= 40:
            return "text_encoder"   # UMT5 text encoder parked in CPU RAM between prompts
        return "blocks"             # + DiT block swapping, VAE tiling

    def load(self):
        self.state, self.load_started = "loading", time.time()
        self.message = "Importing LongCat-Video"
        for required in (self.repo_dir / "longcat_video", self.model_dir / "dit"):
            if not required.exists():
                raise GenerationError(f"{required} is missing. Run ./setup.sh first.")
        import torch
        import torch.distributed as dist
        if not torch.cuda.is_available():
            raise GenerationError("PyTorch cannot see a CUDA GPU. Check nvidia-smi and the torch install.")
        if str(self.repo_dir) not in sys.path:
            sys.path.insert(0, str(self.repo_dir))

        # The LongCat code expects torchrun's single-process "distributed" setup.
        for key, value in {"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(_free_port()),
                           "RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}.items():
            os.environ.setdefault(key, value)
        torch.cuda.set_device(0)
        if not dist.is_initialized():
            try:
                dist.init_process_group(backend="nccl", rank=0, world_size=1, timeout=timedelta(hours=24))
            except Exception:  # noqa: BLE001 - nccl unavailable in some containers; gloo is fine for 1 process
                dist.init_process_group(backend="gloo", rank=0, world_size=1)

        from transformers import AutoTokenizer, UMT5EncoderModel
        import longcat_video.pipeline_longcat_video as pipeline_module
        from longcat_video.pipeline_longcat_video import LongCatVideoPipeline
        from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
        from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
        from longcat_video.modules.longcat_video_dit import LongCatVideoTransformer3DModel
        from longcat_video.context_parallel import context_parallel_util
        from longcat_video.context_parallel.context_parallel_util import init_context_parallel

        init_context_parallel(context_parallel_size=1, global_rank=0, world_size=1)
        cp_split_hw = context_parallel_util.get_optimal_split(context_parallel_util.get_cp_size())

        ckpt, bf16 = str(self.model_dir), torch.bfloat16
        self.message = "Loading tokenizer and text encoder"
        tokenizer = AutoTokenizer.from_pretrained(ckpt, subfolder="tokenizer", torch_dtype=bf16)
        text_encoder = UMT5EncoderModel.from_pretrained(ckpt, subfolder="text_encoder", torch_dtype=bf16)
        self.message = "Loading VAE and scheduler"
        vae = AutoencoderKLWan.from_pretrained(ckpt, subfolder="vae", torch_dtype=bf16)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ckpt, subfolder="scheduler", torch_dtype=bf16)
        self.message = "Loading DiT (13.6B parameters)"
        dit = LongCatVideoTransformer3DModel.from_pretrained(ckpt, subfolder="dit", cp_split_hw=cp_split_hw,
                                                             torch_dtype=bf16)

        total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
        mode = self._memory_mode(total_gb)
        self.memory_mode = mode
        self.offload_text_encoder = mode != "none"
        self.offload_kv_cache = mode != "none"
        if self.settings.get("ENABLE_COMPILE", "0") == "1" and mode != "blocks":
            dit = torch.compile(dit)

        pipe = LongCatVideoPipeline(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae,
                                    scheduler=scheduler, dit=dit)
        self.message = f"Moving weights to the GPU (offload: {mode})"
        if mode == "none":
            pipe.to("cuda")
        else:
            pipe.vae.to("cuda")
            if mode == "blocks":
                reserve = float(self.settings.get("ACTIVATION_RESERVE_GB", "10")) * 2**30
                self._offloader = _BlockOffloader(pipe.dit, "cuda", reserve)
                pipe.vae.enable_tiling()
            else:
                pipe.dit.to("cuda")
            pipe.device = "cuda"
        self.message = "Loading LoRAs"
        lora_dir = self.model_dir / "lora"
        pipe.dit.load_lora(str(lora_dir / "cfg_step_lora.safetensors"), "cfg_step_lora")
        pipe.dit.load_lora(str(lora_dir / "refinement_lora.safetensors"), "refinement_lora")

        self._install_hooks(pipe, pipeline_module)
        self.pipe = pipe
        self.load_seconds = round(time.time() - self.load_started, 1)
        detail = f", {self._offloader.kept}/{self._offloader.total} DiT blocks in VRAM" if self._offloader else ""
        self.message = f"Ready (offload: {mode}{detail})"
        self.state = "ready"

    def _install_hooks(self, pipe, pipeline_module):
        """Per-step progress via the pipeline's tqdm bars, prompt caching and text-encoder offload."""
        backend = self
        base_tqdm = pipeline_module.tqdm

        class StepTqdm(base_tqdm):
            def __init__(self, *args, **kwargs):
                kwargs.setdefault("mininterval", 10.0)   # keep server.log readable
                super().__init__(*args, **kwargs)
                self.steps_done = 0

            def update(self, n=1):
                result = super().update(n)
                self.steps_done += n
                if backend._step_hook:
                    backend._step_hook(self.steps_done, self.total)   # may raise Cancelled
                return result

        pipeline_module.tqdm = StepTqdm

        original_encode = pipe.encode_prompt
        signature = inspect.signature(original_encode)

        def encode_prompt(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            a = bound.arguments
            key = repr((a.get("prompt"), a.get("negative_prompt"), a.get("do_classifier_free_guidance"),
                        a.get("num_videos_per_prompt"), a.get("max_sequence_length"), str(a.get("dtype"))))
            if key not in backend._prompt_cache:   # continuation segments reuse the same prompt
                with backend._text_encoder_on_gpu():
                    backend._prompt_cache[key] = original_encode(*args, **kwargs)
            return _clone(backend._prompt_cache[key])

        pipe.encode_prompt = encode_prompt
        if hasattr(pipe, "_get_t5_prompt_embeds"):
            original_t5 = pipe._get_t5_prompt_embeds

            def get_t5_prompt_embeds(*args, **kwargs):
                with backend._text_encoder_on_gpu():
                    return original_t5(*args, **kwargs)

            pipe._get_t5_prompt_embeds = get_t5_prompt_embeds

    def _enable_lora(self, name):
        self.pipe.dit.enable_loras([name])
        if self._offloader:   # enable_loras follows the DiT's device, which is mixed when blocks are swapped
            self.pipe.dit.lora_dict[name].to("cuda")

    @contextlib.contextmanager
    def _text_encoder_on_gpu(self):
        import torch
        self._te_depth += 1
        try:
            if self._te_depth == 1 and self.offload_text_encoder:
                self.pipe.text_encoder.to("cuda")
            yield
        finally:
            self._te_depth -= 1
            if self._te_depth == 0 and self.offload_text_encoder:
                self.pipe.text_encoder.to("cpu")
                torch.cuda.empty_cache()

    def generate(self, image_path, prompt, negative_prompt, duration_s, resolution, seed, steps,
                 progress_cb, mode="fast", out_path=None):
        import torch
        if self.state != "ready":
            raise GenerationError("The model is not loaded yet.")
        pipe, plan = self.pipe, self.plan(duration_s, resolution)
        n, S, C = plan["segments"], self.SEG_FRAMES, self.COND_FRAMES
        fast = mode == "fast"
        guidance = 1.0 if fast else float(self.settings.get("GUIDANCE_SCALE", "4.0"))
        negative = None if fast else (negative_prompt or None)   # the distilled demo passes none
        out_path = Path(out_path or Path(image_path).with_suffix(".mp4"))
        image = Image.open(image_path).convert("RGB")
        progress = _Progress(progress_cb, n, plan["refine"], self.REFINE_WEIGHT)
        generator = torch.Generator(device="cuda").manual_seed(int(seed))
        torch.cuda.reset_peak_memory_stats()
        started = time.time()
        writer = None
        self._step_hook = progress.step
        self._prompt_cache.clear()
        try:
            with torch.no_grad():
                if fast:
                    self._enable_lora("cfg_step_lora")
                progress.begin("generate", 1, 1.0, f"Generating segment 1 of {n}")
                output = pipe.generate_i2v(
                    image=image, prompt=prompt, negative_prompt=negative, resolution="480p",
                    num_frames=S, num_inference_steps=steps, use_distill=fast,
                    guidance_scale=guidance, generator=generator)[0]
                current = _to_frames(output)
                del output
                size = current[0].size
                frames = list(current)
                progress.finish_segment()

                for segment in range(2, n + 1):
                    progress.begin("generate", segment, 1.0, f"Generating segment {segment} of {n}")
                    output = pipe.generate_vc(
                        video=current, prompt=prompt, negative_prompt=negative, resolution="480p",
                        num_frames=S, num_cond_frames=C, num_inference_steps=steps, use_distill=fast,
                        guidance_scale=guidance, generator=generator, use_kv_cache=True,
                        offload_kv_cache=self.offload_kv_cache, enhance_hf=not fast)[0]   # upstream: not with use_distill
                    current = _to_frames(output, size)
                    del output
                    frames.extend(current[C:])
                    progress.finish_segment()
                del current
                pipe.kv_cache_dict = None   # generate_vc keeps the conditioning KV cache (GBs of VRAM) on the pipeline
                if fast:
                    pipe.dit.disable_all_loras()
                self.release_memory()

                if not plan["refine"]:
                    progress.note("encode", "Encoding video")
                    writer = Mp4Writer(out_path, size, plan["fps"])
                    for frame in frames[: plan["frames"]]:
                        writer.add(frame)
                else:
                    self._enable_lora("refinement_lora")
                    pipe.dit.enable_bsa()
                    refine_steps = int(self.settings.get("REFINE_STEPS", "50"))
                    condition, num_cond, start = None, 1, 0
                    for segment in range(1, n + 1):
                        progress.begin("refine", segment, self.REFINE_WEIGHT, f"Refining to 720p: segment {segment} of {n}")
                        output = pipe.generate_refine(
                            image=image if segment == 1 else None, video=condition, prompt=prompt,
                            stage1_video=frames[start:start + S], num_cond_frames=num_cond,
                            num_inference_steps=refine_steps, generator=generator)[0]
                        refined = _to_frames(output)
                        del output
                        if writer is None:
                            writer = Mp4Writer(out_path, refined[0].size, plan["fps"])
                        for frame in refined[0 if segment == 1 else num_cond:]:
                            if writer.frames < plan["frames"]:
                                writer.add(frame)
                        condition, num_cond, start = refined, C * 2, start + S - C
                        progress.finish_segment()
                writer.close()
                writer = None
        finally:
            self._step_hook = None
            self._prompt_cache.clear()
            pipe.kv_cache_dict = None
            if writer is not None:
                with contextlib.suppress(Exception):
                    writer.close()
            with contextlib.suppress(Exception):
                pipe.dit.disable_all_loras()
            with contextlib.suppress(Exception):
                pipe.dit.disable_bsa()
            if self._offloader:
                self._offloader.reset()

        seconds = time.time() - started
        self.last_stats = {
            "seconds": round(seconds, 1), "segments": n,
            "seconds_per_segment": round(seconds / n, 1),
            "seconds_per_5s_video": round(seconds / plan["seconds"] * 5, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1),
        }
        return str(out_path)


def _synthetic_buckets(area, multiple=16):
    """Approximate LongCat buckets for the mock backend: {ratio: ([h, w], 1)}."""
    ratios = (0.25, 0.33, 0.4, 0.5, 0.56, 0.6, 0.67, 0.75, 0.8, 0.89, 1.0,
              1.12, 1.25, 1.33, 1.5, 1.67, 1.78, 2.0, 2.5, 3.0, 4.0)
    snap = lambda v: max(multiple, round(v / multiple) * multiple)  # noqa: E731
    return {f"{r:.2f}": ([snap(math.sqrt(area * r)), snap(math.sqrt(area / r))], 1) for r in ratios}


class MockBackend(VideoBackend):
    """CPU stand-in for testing the GUI: a slow fake progress bar, then a Ken Burns zoom on the photo.
    A prompt containing "simulate oom" fails the way a real CUDA out-of-memory error would."""

    name = "Mock backend (no GPU)"

    def _bucket_config(self, resolution, scale):
        return _synthetic_buckets(480 * 832 if resolution == "480p" else 720 * 1280, 16 if scale <= 32 else 64)

    def load(self):
        self.state, self.load_started = "loading", time.time()
        self.message = "Loading mock model"
        time.sleep(float(self.settings.get("MOCK_LOAD_SECONDS", "2")))
        self.memory_mode = "none"
        self.load_seconds = round(time.time() - self.load_started, 1)
        self.state, self.message = "ready", "Ready (mock)"

    def generate(self, image_path, prompt, negative_prompt, duration_s, resolution, seed, steps,
                 progress_cb, mode="fast", out_path=None):
        plan = self.plan(duration_s, resolution)
        n = plan["segments"]
        delay = float(self.settings.get("MOCK_STEP_SECONDS", "0.15"))
        progress = _Progress(progress_cb, n, plan["refine"], self.REFINE_WEIGHT)
        started = time.time()
        phases = [("generate", s, 1.0, f"Generating segment {s} of {n}") for s in range(1, n + 1)]
        if plan["refine"]:
            phases += [("refine", s, self.REFINE_WEIGHT, f"Refining to 720p: segment {s} of {n}") for s in range(1, n + 1)]
        for phase, segment, weight, message in phases:
            progress.begin(phase, segment, weight, message)
            for i in range(1, steps + 1):
                time.sleep(delay)
                progress.step(i, steps)
                if "simulate oom" in prompt.lower() and segment == min(2, n) and i == max(1, steps // 2):
                    raise RuntimeError("CUDA out of memory. Tried to allocate 18.00 GiB (simulated by the mock backend)")
            progress.finish_segment()

        progress.note("encode", "Encoding video")
        image = Image.open(image_path).convert("RGB")
        (gen_w, gen_h), (vid_w, vid_h) = self.sizes(image.width, image.height, resolution)
        size = (vid_w, vid_h) if plan["refine"] else (gen_w, gen_h)
        base = image.resize((round(size[0] * 1.25), round(size[1] * 1.25)), Image.BICUBIC)
        writer = Mp4Writer(out_path, size, plan["fps"], preset="veryfast")
        try:
            for k in range(plan["frames"]):
                t = k / max(1, plan["frames"] - 1)
                zoom = 1.25 - 0.2 * t
                w, h = size[0] * zoom, size[1] * zoom
                x = (base.width - w) * (0.5 + 0.3 * math.sin(t * math.pi))
                y = (base.height - h) / 2
                writer.add(base.resize(size, Image.BILINEAR, box=(x, y, x + w, y + h)))
        finally:
            writer.close()
        seconds = time.time() - started
        self.last_stats = {"seconds": round(seconds, 1), "segments": n, "seconds_per_segment": round(seconds / n, 1),
                           "seconds_per_5s_video": round(seconds / plan["seconds"] * 5, 1),
                           "peak_vram_gb": None, "peak_reserved_gb": None}
        return str(out_path)


BACKENDS = {"longcat": LongCatBackend, "mock": MockBackend}


def create_backend(settings):
    name = settings.get("BACKEND", "longcat").lower()
    if name not in BACKENDS:
        raise ValueError(f"Unknown BACKEND={name!r}; choose one of {', '.join(BACKENDS)}")
    return BACKENDS[name](settings)
