"""
Dotto – Relevance Engine
========================
Evaluates transcript chunks against the Pinecone exam question database using
semantic search + GPT-4o deep reasoning.

Usage (standalone):
    python relevance_engine.py --input transcript.json --output results.json

Usage (as a module):
    from relevance_engine import analyze_transcript_relevance, process_transcript_batch
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any

from openai import OpenAI
from pinecone import Pinecone
from pydantic import BaseModel, ValidationError, field_validator

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dotto.relevance")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "dotto-exam-questions")

EMBEDDING_MODEL: str = "text-embedding-3-small"
CHAT_MODEL: str = "gpt-4o"
TOP_K: int = 5  # number of Pinecone neighbours to retrieve

# ---------------------------------------------------------------------------
# Pydantic schema for LLM output validation
# ---------------------------------------------------------------------------


class RelevanceResult(BaseModel):
    """Strict schema that the LLM must return."""

    relevance_score: float
    reasoning: str
    is_critical: bool

    @field_validator("relevance_score")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"relevance_score must be between 0.0 and 1.0, got {v}")
        return round(v, 4)


# ---------------------------------------------------------------------------
# Client initialisation (lazy singletons)
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_pinecone_index: Any | None = None  # pinecone.Index


def _get_openai() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("OpenAI client initialised.")
    return _openai_client


def _get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise EnvironmentError("PINECONE_API_KEY environment variable is not set.")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        log.info("Pinecone index '%s' connected.", PINECONE_INDEX_NAME)
    return _pinecone_index


# ---------------------------------------------------------------------------
# Step A – Embedding
# ---------------------------------------------------------------------------


def _embed(text: str) -> list[float]:
    """Generate an embedding vector for *text* using OpenAI."""
    response = _get_openai().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Step B – Pinecone retrieval
# ---------------------------------------------------------------------------


def _retrieve_similar_questions(embedding: list[float]) -> list[dict]:
    """
    Query Pinecone for the top-K most semantically similar exam questions.

    Returns a list of dicts, each with at least:
        { "id": str, "score": float, "text": str }
    The ``text`` field is expected to be stored in the vector's metadata
    under the key ``"text"``.  Adjust ``metadata_key`` if your index uses
    a different field name (e.g. ``"question"`` or ``"content"``).
    """
    index = _get_pinecone_index()
    result = index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True,
    )

    questions: list[dict] = []
    for match in result.get("matches", []):
        metadata = match.get("metadata") or {}
        # Support common metadata key names used in exam pipelines.
        text = (
            metadata.get("text")
            or metadata.get("question")
            or metadata.get("content")
            or "(no text in metadata)"
        )
        questions.append(
            {
                "id": match.get("id", ""),
                "score": round(float(match.get("score", 0.0)), 4),
                "text": text,
            }
        )

    log.debug("Retrieved %d similar questions from Pinecone.", len(questions))
    return questions


# ---------------------------------------------------------------------------
# Step C + D – GPT-4o reasoning
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert medical education analyst for the Dotto platform.
Your task is to evaluate whether a transcript chunk is conceptually relevant \
to a given set of exam questions.

You MUST respond with a single, strictly valid JSON object matching this schema:
{
  "relevance_score": <float between 0.0 and 1.0>,
  "reasoning": "<one or two sentences explaining the conceptual link>",
  "is_critical": <true | false>
}

Scoring guide:
  0.0–0.3  → little or no conceptual overlap
  0.3–0.6  → partial or indirect overlap
  0.6–0.8  → clear conceptual link
  0.8–1.0  → direct, high-value teaching moment

Set is_critical=true when the chunk directly explains a concept that is \
frequently tested or commonly misunderstood.

Output ONLY the JSON object. No prose, no markdown fences, no extra keys."""


def _build_user_prompt(chunk_text: str, questions: list[dict]) -> str:
    questions_block = "\n".join(
        f"[Q{i + 1}] (similarity {q['score']}) {q['text']}"
        for i, q in enumerate(questions)
    )
    return (
        f"TRANSCRIPT CHUNK:\n{chunk_text}\n\n"
        f"TOP RELATED EXAM QUESTIONS:\n{questions_block}"
    )


def _call_llm(chunk_text: str, questions: list[dict]) -> RelevanceResult:
    """Send the chunk + retrieved questions to GPT-4o and parse the result."""
    client = _get_openai()
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,  # deterministic for consistent scoring
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(chunk_text, questions)},
        ],
    )

    raw_json: str = response.choices[0].message.content or "{}"

    try:
        data = json.loads(raw_json)
        result = RelevanceResult(**data)
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        log.error("LLM returned invalid JSON – %s\nRaw output: %s", exc, raw_json)
        raise ValueError(f"LLM output failed validation: {exc}") from exc

    return result


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def analyze_transcript_relevance(transcript_chunk_text: str) -> dict:
    """
    Evaluate a single transcript chunk against the Pinecone exam database.

    Parameters
    ----------
    transcript_chunk_text : str
        The raw text of one transcript segment / chunk.

    Returns
    -------
    dict
        {
            "relevance_score": float,   # 0.0 – 1.0
            "reasoning": str,
            "is_critical": bool,
            "retrieved_questions": list[dict]   # the top-K neighbours
        }

    Raises
    ------
    EnvironmentError
        If required API keys are missing.
    ValueError
        If the LLM returns malformed JSON.
    """
    if not transcript_chunk_text or not transcript_chunk_text.strip():
        raise ValueError("transcript_chunk_text must be a non-empty string.")

    log.info("Embedding chunk (%d chars)…", len(transcript_chunk_text))
    embedding = _embed(transcript_chunk_text)

    log.info("Querying Pinecone for top-%d similar questions…", TOP_K)
    questions = _retrieve_similar_questions(embedding)

    log.info("Running GPT-4o relevance analysis…")
    result = _call_llm(transcript_chunk_text, questions)

    return {
        "relevance_score": result.relevance_score,
        "reasoning": result.reasoning,
        "is_critical": result.is_critical,
        "retrieved_questions": questions,
    }


# ---------------------------------------------------------------------------
# Batch processing helper
# ---------------------------------------------------------------------------


def process_transcript_batch(
    transcript_segments: list[dict],
    output_path: str,
    text_field: str = "text",
) -> list[dict]:
    """
    Iterate over a list of transcript segments, run relevance analysis on each,
    and write the enriched results to *output_path* as JSON.

    Parameters
    ----------
    transcript_segments : list[dict]
        Typically the ``"segments"`` list produced by ``modal_app.py``.
        Each element must contain at least the field named by *text_field*.
    output_path : str
        Path to write the output JSON file.
    text_field : str
        Key in each segment dict that holds the transcript text.
        Defaults to ``"text"`` (Dotto whisper format).

    Returns
    -------
    list[dict]
        The enriched segments (same as what is written to *output_path*).
    """
    if not transcript_segments:
        raise ValueError("transcript_segments list is empty.")

    enriched: list[dict] = []
    total = len(transcript_segments)

    for idx, segment in enumerate(transcript_segments):
        chunk_text = segment.get(text_field, "").strip()
        if not chunk_text:
            log.warning("Segment %d/%d has no text – skipping.", idx + 1, total)
            enriched.append({**segment, "relevance": None})
            continue

        log.info("Analysing segment %d/%d…", idx + 1, total)
        try:
            relevance = analyze_transcript_relevance(chunk_text)
        except Exception as exc:
            log.error("Segment %d/%d failed: %s", idx + 1, total, exc)
            relevance = {"error": str(exc)}

        enriched.append({**segment, "relevance": relevance})

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(enriched, fh, indent=2, ensure_ascii=False)

    log.info("Results saved to '%s' (%d segments).", output_path, len(enriched))
    return enriched


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dotto relevance engine – score transcript segments against exam questions."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to transcript JSON file (list of segments or full Dotto output dict).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to write enriched results JSON.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Key inside each segment that holds the text (default: 'text').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    with open(args.input, encoding="utf-8") as fh:
        data = json.load(fh)

    # Accept either the raw segments list or the full Dotto output dict.
    if isinstance(data, dict) and "segments" in data:
        segments = data["segments"]
    elif isinstance(data, list):
        segments = data
    else:
        log.error("Unexpected input format. Expected a list or a dict with 'segments'.")
        sys.exit(1)

    process_transcript_batch(
        transcript_segments=segments,
        output_path=args.output,
        text_field=args.text_field,
    )


if __name__ == "__main__":
    main()
