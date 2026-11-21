"""Transcribe cached audio with Gemini, saving each completed part locally."""
from __future__ import annotations

import base64
import json
import math
import re
import subprocess
import wave
from dataclasses import asdict
from pathlib import Path
from time import monotonic, sleep

from . import config
from .pipeline_progress import report
from .source import Source, Word


MODEL = "gemini-3.5-transcribe"
PART_SECONDS = 600
OVERLAP_SECONDS = 2
MIN_REQUEST_SPACING = 65


def _offset(value: str) -> float:
    parsed = float(value.removesuffix("s"))
    if not math.isfinite(parsed):
        raise ValueError("Invalid timestamp")
    return parsed


def timed_words(payload: dict, offset: float, start: float, end: float, total: float) -> list[Word]:
    words = []
    for step in payload.get("steps", []):
        if step.get("type") != "model_output":
            continue
        for content in step.get("content", []):
            text = content.get("text") or ""
            for annotation in content.get("annotations", []):
                if annotation.get("type") != "word_info":
                    continue
                try:
                    low, high = annotation["start_index"], annotation["end_index"]
                    if not 0 <= low < high <= len(text):
                        continue
                    left = max(0, offset + _offset(annotation["start_offset"]))
                    right = min(total, offset + _offset(annotation["end_offset"]))
                    tokens = text[low:high].split()
                    if not tokens or left >= right:
                        continue
                except (KeyError, TypeError, ValueError, AttributeError):
                    continue
                width = (right - left) / len(tokens)
                for index, token in enumerate(tokens):
                    token_start = left + width * index
                    if start <= token_start + width / 2 < end:
                        words.append(Word(token, token_start, token_start + width))
    return sorted(words, key=lambda word: word.start)


def _transcribe_part(wav: Path, checkpoint: Path, key: str, start: float, end: float, total: float, wait_for_slot) -> list[Word]:
    try:
        return [Word(**word) for word in json.loads(checkpoint.read_text(encoding="utf-8"))]
    except (OSError, TypeError, ValueError):
        pass

    from google import genai
    from google.genai import types

    left = max(0, start - OVERLAP_SECONDS)
    right = min(total, end + OVERLAP_SECONDS)
    audio = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str(left), "-i", str(wav),
         "-t", str(right - left), "-vn", "-ar", "16000", "-ac", "1", "-b:a", "32k", "-f", "mp3", "pipe:1"],
        check=True, capture_output=True,
    ).stdout
    content = {"type": "audio", "mime_type": "audio/mp3", "data": base64.b64encode(audio).decode("ascii")}
    options = types.HttpOptions(api_version="v1beta", timeout=240000, retry_options=types.HttpRetryOptions(attempts=1))
    with genai.Client(api_key=key, http_options=options) as client:
        for attempt in range(3):
            wait_for_slot()
            try:
                result = client.interactions.create(
                    model=MODEL, input=[content],
                    generation_config={"transcription_config": {
                        "mode": {"type": "verbatim", "timestamp_granularities": ["word"]},
                    }},
                )
                break
            except Exception as exc:
                code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
                if type(exc).__name__ == "RateLimitError":
                    code = 429
                message = str(exc).lower()
                if code == 429 and ("perday" in message or "per day" in message):
                    raise RuntimeError("Gemini's daily transcription quota is exhausted. Completed parts are saved.") from None
                if attempt == 2 or code not in (429, 500, 502, 503, 504):
                    reason = "was rate-limited" if code == 429 else f"returned HTTP {code}" if code else "could not connect"
                    raise RuntimeError(f"Gemini transcription {reason}. Completed parts are saved.") from None
                retry = re.search(r"retry in\s+([\d.]+)s", message)
                delay = min(90, math.ceil(float(retry.group(1)))) + 2 if retry else 20 * (attempt + 1)
                while delay > 0:
                    interval = min(15, delay)
                    sleep(interval)
                    delay -= interval
    if result.status != "completed":
        raise RuntimeError("Gemini did not finish this audio part. Completed parts are saved.")
    payload = result.model_dump(mode="json", exclude_none=True)
    words = timed_words(payload, left, start, end, total)
    transcript = re.sub(r"\[[^\]]*\]|\([^)]*\)", "", result.output_text or "").strip()
    annotated = any(
        annotation.get("type") == "word_info"
        for step in payload.get("steps", []) for content in step.get("content", [])
        for annotation in content.get("annotations", [])
    )
    if not words and transcript and not annotated:
        raise RuntimeError("Gemini returned speech without usable word timestamps. Completed parts are saved.")
    staged = checkpoint.with_suffix(".tmp")
    staged.write_text(json.dumps([asdict(word) for word in words], ensure_ascii=False), encoding="utf-8")
    staged.replace(checkpoint)
    return words


def from_cached(video: Path, wav: Path, workdir: Path, title: str, progress) -> Source:
    key = config.ai_key("gemini")
    if not key:
        raise RuntimeError("Save a Gemini API key to use cloud transcription.")
    with wave.open(str(wav), "rb") as audio:
        duration = audio.getnframes() / audio.getframerate()
    if not 0 < duration <= config.MAX_VIDEO_HOURS * 3600:
        raise RuntimeError("The cached audio duration is outside Arsuclip's limits.")

    bounds = [(float(start), min(start + PART_SECONDS, duration)) for start in range(0, math.ceil(duration), PART_SECONDS)]
    parts = len(bounds)
    cache = workdir / "gemini-transcript"
    cache.mkdir(exist_ok=True)
    start_clock = monotonic()
    report(progress, 26, f"Gemini is transcribing {parts} audio parts", phase="transcribe",
           total_seconds=duration, processed_seconds=0, parts=parts, completed=0, provider="gemini", workers=1)
    words: list[Word] = []
    processed = 0.0
    last_request = None
    for index, (start, end) in enumerate(bounds, 1):
        def wait_for_slot() -> None:
            nonlocal last_request
            wait = 0 if last_request is None else last_request + MIN_REQUEST_SPACING - monotonic()
            while wait > 0:
                report(progress, 26 + 14 * processed / duration,
                       f"Waiting for Gemini free-tier quota before audio part {index} of {parts}",
                       phase="transcribe", part=index, parts=parts, completed=index - 1, provider="gemini",
                       workers=1, processed_seconds=processed, total_seconds=duration,
                       chunk_start_seconds=start, chunk_end_seconds=end, words_transcribed=len(words))
                sleep(min(10, wait))
                wait = last_request + MIN_REQUEST_SPACING - monotonic()
            last_request = monotonic()
            report(progress, 26 + 14 * processed / duration, f"Gemini is transcribing audio part {index} of {parts}",
                   phase="transcribe", part=index, parts=parts, completed=index - 1, provider="gemini",
                   workers=1, processed_seconds=processed, total_seconds=duration,
                   chunk_start_seconds=start, chunk_end_seconds=end, words_transcribed=len(words))

        words.extend(_transcribe_part(wav, cache / f"part-{index:03}.json", key, start, end, duration, wait_for_slot))
        processed += end - start
        eta = (duration - processed) * (monotonic() - start_clock) / processed
        report(progress, 26 + 14 * processed / duration, f"Gemini transcribed {index} of {parts} audio parts",
               phase="transcribe", part=index, parts=parts, completed=index, provider="gemini",
               workers=1, processed_seconds=processed, total_seconds=duration,
               chunk_start_seconds=start, chunk_end_seconds=end, words_transcribed=len(words),
               eta_seconds=round(eta) if index >= 2 and last_request is not None else None)
    if not words:
        raise RuntimeError("Gemini did not find any speech in the audio.")
    words.sort(key=lambda word: word.start)
    return Source(video, title, duration, "", words, "gemini-transcribe")
