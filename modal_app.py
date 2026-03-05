"""
Dotto – Serverless Video Transcription + Relevance Pipeline

Architecture (async job system):
  1. Browser uploads audio to POST /start  → returns { job_id } immediately
  2. process_job() runs in background, updates job_dict with progress
  3. Browser polls GET /status/{job_id} every 3 seconds
  4. When status == "done", browser receives the full result
"""

import json
import os
import uuid as uuid_mod

import modal

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

dotto_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg")
    .pip_install(
        "faster-whisper==1.0.3",
        "torch==2.2.2",
        "torchaudio==2.2.2",
        "ctranslate2>=4.0.0",
        "requests",
        "numpy",
        "fastapi[standard]",
        "openai>=1.30.0",
        "pinecone>=4.0.0",
        "pydantic>=2.0.0",
    )
    .add_local_file("relevance_engine.py", "/app/relevance_engine.py")
)

dotto_secret = modal.Secret.from_name("dotto-secrets")

# ---------------------------------------------------------------------------
# App + Volumes
# ---------------------------------------------------------------------------

app = modal.App("dotto-transcription", image=dotto_image)

# Persistent model cache — avoids re-downloading on every cold start
model_volume = modal.Volume.from_name("dotto-model-cache", create_if_missing=True)
MODEL_CACHE = "/model-cache"

# Temp audio files shared between serve() and process_job()
audio_volume = modal.Volume.from_name("dotto-audio-temp", create_if_missing=True)
AUDIO_MOUNT = "/audio-temp"

# Job state store — keyed by job_id, holds status/progress/result
job_dict = modal.Dict.from_name("dotto-jobs", create_if_missing=True)

# Legacy video volume (kept for the local entrypoint)
video_volume = modal.Volume.from_name("dotto-videos", create_if_missing=True)
VOLUME_MOUNT = "/videos"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_result(segments_iter, info) -> dict:
    """Convert a faster-whisper segment iterator into a JSON-serialisable dict."""
    segments = []
    full_text_parts = []
    for seg in segments_iter:
        full_text_parts.append(seg.text.strip())
        segments.append({
            "start": round(seg.start, 3),
            "end":   round(seg.end, 3),
            "text":  seg.text.strip(),
        })
    return {
        "language":             info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds":     round(info.duration, 2),
        "full_text":            " ".join(full_text_parts),
        "segments":             segments,
    }


# ---------------------------------------------------------------------------
# Background worker — transcription + relevance scoring
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    memory=8192,
    timeout=1800,  # 30 min ceiling for very long lectures
    secrets=[dotto_secret],
    volumes={MODEL_CACHE: model_volume, AUDIO_MOUNT: audio_volume},
)
def process_job(job_id: str) -> None:
    """
    Runs in a dedicated container:
      1. Loads the audio file written by serve() from the shared volume
      2. Transcribes with Whisper large-v3
      3. Scores every segment against Pinecone (parallel, 20 concurrent)
      4. Writes progress updates to job_dict throughout
    """
    import asyncio
    import sys
    sys.path.insert(0, "/app")
    from faster_whisper import WhisperModel
    from relevance_engine import analyze_transcript_relevance

    audio_path = f"{AUDIO_MOUNT}/{job_id}.mp3"

    try:
        # Ensure we see the file committed by serve()
        audio_volume.reload()

        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # ── Step 1: Load model ──────────────────────────────────────────────
        job_dict[job_id] = {
            "status": "transcribing",
            "message": "Loading Whisper large-v3…",
            "progress": 5,
        }
        print(f"[dotto] {job_id}: loading model…")
        model = WhisperModel(
            "large-v3",
            device="cuda",
            compute_type="float16",
            download_root=MODEL_CACHE,
        )

        # ── Step 2: Transcribe ─────────────────────────────────────────────
        job_dict[job_id] = {
            "status": "transcribing",
            "message": "Transcribing lecture with Whisper large-v3…",
            "progress": 10,
        }
        print(f"[dotto] {job_id}: transcribing…")
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        result = _build_result(segments_iter, info)
        total = len(result["segments"])
        print(f"[dotto] {job_id}: transcription done — {total} segments, {result['language']}")

        # ── Step 3: Parallel relevance scoring ────────────────────────────
        job_dict[job_id] = {
            "status": "scoring",
            "message": f"Analyzing exam relevance… 0 / {total} segments",
            "progress": 50,
        }

        MAX_CONCURRENT = 20

        async def score_all() -> list[dict]:
            sem = asyncio.Semaphore(MAX_CONCURRENT)
            scored: list[dict | None] = [None] * total
            done_count = [0]

            async def score_one(idx: int, seg: dict) -> None:
                text = seg.get("text", "").strip()
                if not text:
                    scored[idx] = {**seg, "relevance": None}
                else:
                    async with sem:
                        try:
                            relevance = await asyncio.to_thread(
                                analyze_transcript_relevance, text
                            )
                        except Exception as exc:
                            relevance = {"error": str(exc)}
                        scored[idx] = {**seg, "relevance": relevance}

                done_count[0] += 1
                # Write progress every 10 segments (use async API — we're inside asyncio.run)
                if done_count[0] % 10 == 0 or done_count[0] == total:
                    pct = 50 + int(done_count[0] / total * 49)
                    await job_dict.put.aio(job_id, {
                        "status": "scoring",
                        "message": f"Analyzing exam relevance… {done_count[0]} / {total} segments",
                        "progress": pct,
                    })

            await asyncio.gather(*(score_one(i, s) for i, s in enumerate(result["segments"])))
            return scored  # type: ignore[return-value]

        result["segments"] = asyncio.run(score_all())

        # ── Done ───────────────────────────────────────────────────────────
        job_dict[job_id] = {
            "status":   "done",
            "message":  "Analysis complete!",
            "progress": 100,
            "result":   result,
        }
        print(f"[dotto] {job_id}: done.")

    except Exception as exc:
        print(f"[dotto] {job_id}: ERROR — {exc}")
        job_dict[job_id] = {
            "status":   "error",
            "message":  str(exc),
            "progress": 0,
        }
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)
            audio_volume.commit()


# ---------------------------------------------------------------------------
# Web endpoint — lightweight dispatcher + status polling
# ---------------------------------------------------------------------------

@app.function(
    memory=1024,
    timeout=60,  # just HTTP I/O, returns in < 1 s
    volumes={AUDIO_MOUNT: audio_volume},
)
@modal.asgi_app()
def serve():
    """
    Two endpoints:
      POST /start          — accept audio, spawn process_job, return {job_id}
      GET  /status/{job_id} — return current job state for polling
    """
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from fastapi.middleware.cors import CORSMiddleware

    web_app = FastAPI()
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @web_app.post("/start")
    async def start_route(video: UploadFile = File(...)):
        job_id = str(uuid_mod.uuid4())
        audio_path = f"{AUDIO_MOUNT}/{job_id}.mp3"

        content = await video.read()
        print(f"[dotto-web] Received {len(content) // 1024} KB — job {job_id}")

        with open(audio_path, "wb") as f:
            f.write(content)
        # Use async Modal APIs — blocking calls in async context cause silent failures
        await audio_volume.commit.aio()

        await job_dict.put.aio(job_id, {
            "status":   "queued",
            "message":  "Job queued, starting soon…",
            "progress": 0,
        })
        await process_job.spawn.aio(job_id)

        return {"job_id": job_id}

    @web_app.get("/status/{job_id}")
    async def status_route(job_id: str):
        state = await job_dict.get.aio(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail="Job not found")
        # Never return the full result in status — only return it once done
        if state.get("status") == "done":
            return state  # includes 'result' key
        # Strip heavy result payload from in-progress states
        return {k: v for k, v in state.items() if k != "result"}

    return web_app


# ---------------------------------------------------------------------------
# Local entrypoint — ad-hoc testing
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    memory=8192,
    timeout=600,
    volumes={VOLUME_MOUNT: video_volume, MODEL_CACHE: model_volume},
)
def transcribe_video(video_filename: str, model_size: str = "large-v3") -> dict:
    """Transcribe a video stored in the dotto-videos volume (no relevance scoring)."""
    import subprocess
    from faster_whisper import WhisperModel

    video_path = os.path.join(VOLUME_MOUNT, video_filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    with __import__("tempfile").NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
             "-ar", "16000", "-ac", "1", audio_path],
            capture_output=True, check=True,
        )
        model = WhisperModel(model_size, device="cuda", compute_type="float16",
                             download_root=MODEL_CACHE)
        segments_iter, info = model.transcribe(audio_path, beam_size=5, vad_filter=True)
        return _build_result(segments_iter, info)
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


@app.local_entrypoint()
def main(video_filename: str = "sample.mp4", model_size: str = "large-v3"):
    result = transcribe_video.remote(video_filename, model_size)
    print(json.dumps(result, indent=2, ensure_ascii=False))
