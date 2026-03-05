"""
Dotto – Serverless Video Transcription + Relevance Pipeline
Modal.com application that:
  1. Extracts audio from a video stored in a Modal Volume via ffmpeg
  2. Transcribes the audio using faster-whisper-large
  3. Scores each segment against the Pinecone exam database via GPT-4o
  4. Returns a structured JSON result with relevance metadata per segment
"""

import json
import os
import tempfile

import modal

# ---------------------------------------------------------------------------
# Image definition
# ---------------------------------------------------------------------------
# We build a single image that has:
#   • ffmpeg (system package for audio extraction)
#   • PyTorch + torchaudio (CPU wheels are enough for Whisper on GPU nodes)
#   • faster-whisper (CTranslate2-based Whisper implementation)
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

# Secret that holds OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_INDEX_NAME.
# Create once with:
#   modal secret create dotto-secrets \
#     OPENAI_API_KEY=sk-... \
#     PINECONE_API_KEY=pcsk_... \
#     PINECONE_INDEX_NAME=dotto-exam-questions
dotto_secret = modal.Secret.from_name("dotto-secrets")

# ---------------------------------------------------------------------------
# Modal app + persistent volume
# ---------------------------------------------------------------------------

app = modal.App("dotto-transcription", image=dotto_image)

# A Modal Volume where raw video files are stored.
# Create it once with:  modal volume create dotto-videos
video_volume = modal.Volume.from_name("dotto-videos", create_if_missing=True)

# Path inside the container where the volume is mounted.
VOLUME_MOUNT = "/videos"

# Model cache volume – avoids re-downloading the large model on every cold start.
model_volume = modal.Volume.from_name("dotto-model-cache", create_if_missing=True)
MODEL_CACHE = "/model-cache"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_audio(video_path: str, audio_path: str) -> None:
    """Run ffmpeg to strip the audio track from *video_path* → *audio_path*."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-y",                  # overwrite output if it exists
        "-i", video_path,
        "-vn",                 # drop video stream
        "-acodec", "pcm_s16le",
        "-ar", "16000",        # Whisper expects 16 kHz
        "-ac", "1",            # mono
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )


def _build_result(segments_iter, info) -> dict:
    """Convert a faster-whisper segment iterator into a clean JSON-serialisable dict."""
    segments = []
    full_text_parts = []

    for seg in segments_iter:
        full_text_parts.append(seg.text.strip())
        segments.append(
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text.strip(),
            }
        )

    return {
        "language": info.language,
        "language_probability": round(info.language_probability, 4),
        "duration_seconds": round(info.duration, 2),
        "full_text": " ".join(full_text_parts),
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Main Modal function
# ---------------------------------------------------------------------------

@app.function(
    # Request a GPU only for the transcription step; A10G is a cost-efficient
    # choice for whisper-large.  Use gpu=modal.gpu.T4() for a cheaper option.
    gpu=modal.gpu.A10G(),
    # Keep memory generous enough to hold the model weights in VRAM.
    memory=8192,
    # Cold-start timeout – model download can take a few minutes the first time.
    timeout=600,
    volumes={
        VOLUME_MOUNT: video_volume,
        MODEL_CACHE: model_volume,
    },
)
def transcribe_video(video_filename: str, model_size: str = "large-v3") -> dict:
    """
    Transcribe a video file stored in the dotto-videos Modal Volume.

    Parameters
    ----------
    video_filename : str
        Relative path to the video inside the mounted volume,
        e.g. ``"uploads/interview.mp4"``.
    model_size : str
        Whisper model variant.  Defaults to ``"large-v3"``.

    Returns
    -------
    dict
        ``{language, language_probability, duration_seconds, full_text, segments}``
    """
    from faster_whisper import WhisperModel

    video_path = os.path.join(VOLUME_MOUNT, video_filename)
    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found at {video_path}. "
            "Make sure the file is uploaded to the 'dotto-videos' volume."
        )

    # ── Step 1: Extract audio ──────────────────────────────────────────────
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        print(f"[dotto] Extracting audio from '{video_filename}' …")
        _extract_audio(video_path, audio_path)

        # ── Step 2: Load model (cached in volume after first run) ──────────
        print(f"[dotto] Loading faster-whisper '{model_size}' …")
        model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="float16",       # best throughput on A10G
            download_root=MODEL_CACHE,    # reuse across warm starts
        )

        # ── Step 3: Transcribe ─────────────────────────────────────────────
        print("[dotto] Transcribing …")
        segments_iter, info = model.transcribe(
            audio_path,
            beam_size=5,
            vad_filter=True,              # skip silent regions → faster
            vad_parameters={"min_silence_duration_ms": 500},
        )

        result = _build_result(segments_iter, info)

    finally:
        # Always clean up the temporary audio file to avoid wasting container disk.
        if os.path.exists(audio_path):
            os.remove(audio_path)

    print(f"[dotto] Done. Language: {result['language']} | Segments: {len(result['segments'])}")
    return result


# ---------------------------------------------------------------------------
# Relevance scoring – CPU-only, runs after transcription
# ---------------------------------------------------------------------------

@app.function(
    image=dotto_image,
    secrets=[dotto_secret],
    timeout=300,
)
def analyze_relevance(segments: list[dict]) -> list[dict]:
    """
    Score each transcript segment against the Pinecone exam database.
    Returns the same list with a ``relevance`` key added to every item.
    """
    import sys
    sys.path.insert(0, "/app")
    from relevance_engine import analyze_transcript_relevance

    enriched: list[dict] = []
    total = len(segments)

    for idx, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        print(f"[dotto-relevance] Scoring segment {idx + 1}/{total} …")
        if not text:
            enriched.append({**seg, "relevance": None})
            continue
        try:
            relevance = analyze_transcript_relevance(text)
        except Exception as exc:
            print(f"[dotto-relevance] Segment {idx + 1} failed: {exc}")
            relevance = {"error": str(exc)}
        enriched.append({**seg, "relevance": relevance})

    return enriched


# ---------------------------------------------------------------------------
# Web endpoint – called by the Next.js /api/transcribe proxy
# ---------------------------------------------------------------------------

@app.function(
    gpu=modal.gpu.A10G(),
    memory=8192,
    timeout=600,
    secrets=[dotto_secret],
    volumes={MODEL_CACHE: model_volume},
)
@modal.asgi_app()
def serve():
    """
    Returns a FastAPI ASGI app.  Transcribes the uploaded video with Whisper,
    then scores every segment for exam relevance via GPT-4o + Pinecone.
    """
    from fastapi import FastAPI, File, HTTPException, UploadFile
    from faster_whisper import WhisperModel

    web_app = FastAPI()

    @web_app.post("/")
    async def transcribe_route(video: UploadFile = File(...)):
        suffix = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_vid:
            video_path = tmp_vid.name
            tmp_vid.write(await video.read())

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_aud:
            audio_path = tmp_aud.name

        try:
            print(f"[dotto-web] Extracting audio from '{video.filename}' …")
            _extract_audio(video_path, audio_path)

            print("[dotto-web] Loading faster-whisper large-v3 …")
            model = WhisperModel(
                "large-v3",
                device="cuda",
                compute_type="float16",
                download_root=MODEL_CACHE,
            )

            print("[dotto-web] Transcribing …")
            segments_iter, info = model.transcribe(
                audio_path,
                beam_size=5,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            result = _build_result(segments_iter, info)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            for p in (video_path, audio_path):
                if os.path.exists(p):
                    os.remove(p)

        print(f"[dotto-web] Transcription done. Scoring {len(result['segments'])} segments …")
        result["segments"] = analyze_relevance.remote(result["segments"])
        print("[dotto-web] Relevance scoring complete.")
        return result

    return web_app


# ---------------------------------------------------------------------------
# Local entrypoint – useful for ad-hoc testing from your machine
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(video_filename: str = "sample.mp4", model_size: str = "large-v3"):
    """
    Run from your terminal with:
        modal run modal_app.py --video-filename uploads/interview.mp4
    """
    result = transcribe_video.remote(video_filename, model_size)
    print(json.dumps(result, indent=2, ensure_ascii=False))
