"""Download a YouTube video and get a word-level transcript for it."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np
import yt_dlp

from . import config
from .pipeline_progress import Progress, report


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Source:
    video_path: Path
    title: str
    duration: float
    language: str
    words: list[Word]
    transcript_source: str  # "youtube-captions" or "whisper"


def _ydl_base() -> dict:
    opts = {"quiet": True, "no_warnings": True, "noplaylist": True}
    # yt-dlp needs a JavaScript runtime for YouTube; it finds Deno itself,
    # Node has to be enabled explicitly.
    if not shutil.which("deno") and shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    return opts


def download(url: str, workdir: Path, progress: Progress) -> Source:
    report(progress, 2, "Reading video details", phase="inspect")
    with yt_dlp.YoutubeDL(_ydl_base()) as ydl:
        info = ydl.extract_info(url, download=False)

    duration = float(info.get("duration") or 0)
    if duration and duration > config.MAX_VIDEO_HOURS * 3600:
        raise ValueError(f"Video is longer than {config.MAX_VIDEO_HOURS} hours.")
    if info.get("is_live"):
        raise ValueError("Live streams can't be clipped until they finish.")

    lang = (info.get("language") or "en").split("-")[0]
    title = info.get("title") or "Untitled"

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            info = d.get("info_dict") or {}
            audio_only = info.get("vcodec") == "none"
            video_only = info.get("acodec") == "none"
            base, span = (21, 3) if audio_only else (3, 18 if video_only else 21)
            done = d.get("downloaded_bytes") or 0
            fraction = min(1, done / total) if total else 0
            report(
                progress, base + span * fraction, "Downloading audio" if audio_only else "Downloading video",
                phase="download_audio" if audio_only else "download_video", title=title,
                total_seconds=duration or None, bytes_done=done, bytes_total=total,
                speed_bytes_per_second=d.get("speed"), eta_seconds=d.get("eta"),
            )

    def merge_hook(d: dict) -> None:
        if d.get("status") == "started" and d.get("postprocessor") == "Merger":
            report(progress, 24, "Merging video and audio", phase="merge", title=title, total_seconds=duration or None)

    opts = _ydl_base() | {
        "format": "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/bv*[height<=1080]+ba/b[height<=1080]/b",
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "retries": 5,
        "progress_hooks": [hook],
        "postprocessor_hooks": [merge_hook],
    }
    long_video = duration > 4 * 3600
    if long_video:
        opts.update(
            format="bv[height<=720][fps<=30][ext=mp4]+ba[ext=m4a]/bv[height<=720]+ba/b[height<=720]/b",
            http_chunk_size=8 * 1024 * 1024,
            concurrent_fragment_downloads=6,
        )
    report(progress, 3, "Connecting to YouTube download", phase="download_video", title=title, total_seconds=duration or None)
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        if not long_video or "HTTP Error 403" not in str(exc):
            raise
        report(progress, 3, "Retrying download with YouTube's streaming format", phase="download_video",
               title=title, total_seconds=duration or None)
        streaming = "bv[height<=720][protocol^=m3u8]+ba[protocol^=m3u8]/b[height<=720][protocol^=m3u8]"
        with yt_dlp.YoutubeDL(opts | {"format": streaming}) as ydl:
            ydl.download([url])

    video_path = workdir / "source.mp4"
    if not video_path.exists():
        found = [p for p in workdir.glob("source.*") if p.suffix in (".mkv", ".webm", ".mp4")]
        if not found:
            raise RuntimeError("Download finished but no video file was produced.")
        video_path = found[0]

    words: list[Word] = []
    transcript_source = "youtube-captions"
    track = pick_caption_track(info, lang)
    if track:
        report(progress, 25, "Getting YouTube captions", phase="captions",
               title=title, total_seconds=duration or None, caption_track=track)
        words = _fetch_captions(url, workdir, track)
        if words:
            report(progress, 26, "Loaded YouTube captions", phase="captions",
                   title=title, total_seconds=duration or None, caption_track=track, words_transcribed=len(words))
    if not words:
        report(progress, 26, "No YouTube captions available; extracting audio", phase="extract_audio",
               title=title, total_seconds=duration or None)
        words = whisper_transcribe(video_path, workdir, progress)
        transcript_source = "whisper"
    if not words:
        raise RuntimeError("Could not get any speech from this video.")

    return Source(
        video_path=video_path,
        title=title,
        duration=duration or words[-1].end,
        language=lang,
        words=words,
        transcript_source=transcript_source,
    )


def pick_caption_track(info: dict, lang: str) -> str | None:
    """The one caption track worth fetching: manual captions beat automatic ones,
    and the video's own language beats English. Asking YouTube for a single track
    (not every variant) keeps us clear of its rate limit."""
    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}
    for pool, codes in ((manual, [lang, "en"]), (auto, [f"{lang}-orig", lang, "en-orig", "en"])):
        for code in codes:
            if code in pool:
                return code
    return None


def _fetch_captions(url: str, workdir: Path, track: str) -> list[Word]:
    """Captions are a shortcut, not a requirement: if YouTube refuses them
    (e.g. HTTP 429 Too Many Requests), return nothing and let Whisper transcribe."""
    opts = _ydl_base() | {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [track],
        "subtitlesformat": "json3",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "retries": 3,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError:
        return []
    sub = workdir / f"source.{track}.json3"
    return parse_json3(sub) if sub.exists() else []


def from_local_file(path: Path, workdir: Path, progress: Progress, title: str | None = None) -> Source:
    """An uploaded video: no captions to fetch, so transcribe it with Whisper."""
    report(progress, 2, "Reading video details", phase="inspect", title=title or path.stem)
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        duration = float(out.stdout.strip())
    except ValueError:
        raise ValueError("That file doesn't look like a video ffmpeg can read.") from None
    if duration > config.MAX_VIDEO_HOURS * 3600:
        raise ValueError(f"Video is longer than {config.MAX_VIDEO_HOURS} hours.")

    report(progress, 26, "Extracting audio for transcription", phase="extract_audio",
           title=title or path.stem, total_seconds=duration or None)
    words = whisper_transcribe(path, workdir, progress)
    if not words:
        raise RuntimeError("Could not get any speech from this video.")
    return Source(
        video_path=path,
        title=title or path.stem or "Untitled",
        duration=duration or words[-1].end,
        language="",
        words=words,
        transcript_source="whisper",
    )


# A token of only dashes is a speaker-change mark in manual captions ("- Joe's dish").
_NOISE = re.compile(r"^\[.*\]$|^\(.*\)$|^>>+$|^♪+$|^[-–—]+$")


def _clean(token: str) -> str:
    token = token.replace("\n", " ").strip()
    token = token.replace(">>", "").strip()
    return "" if _NOISE.match(token) else token


def parse_json3(path: Path) -> list[Word]:
    """YouTube json3 captions -> words. Auto captions carry per-word offsets;
    manual captions carry whole lines, which are spread evenly across the line."""
    data = json.loads(path.read_text(encoding="utf-8"))
    words: list[Word] = []
    for ev in data.get("events", []):
        segs = ev.get("segs")
        if not segs:
            continue
        ev_start = ev.get("tStartMs", 0) / 1000
        ev_end = ev_start + ev.get("dDurationMs", 0) / 1000
        timed = any("tOffsetMs" in s for s in segs[1:])
        if not timed:
            # Manual captions: the whole line, possibly split over several segs
            # with no offsets. Joined and spread evenly below; giving every word
            # the line's start time made them zero-length, so they were skipped.
            # Each seg is cleaned first so a multi-word tag like "[Music playing]"
            # is still dropped whole rather than split into caption words.
            segs = [{"utf8": " ".join(_clean(s.get("utf8", "")) for s in segs)}]
        if timed:
            for s in segs:
                tok = _clean(s.get("utf8", ""))
                if not tok:
                    continue
                start = ev_start + s.get("tOffsetMs", 0) / 1000
                for part in tok.split():
                    words.append(Word(part, start, ev_end))
        else:
            parts = [p for p in (_clean(t) for t in segs[0].get("utf8", "").split()) if p]
            if not parts:
                continue
            step = max(ev_end - ev_start, 0.2) / len(parts)
            for i, part in enumerate(parts):
                words.append(Word(part, ev_start + i * step, ev_start + (i + 1) * step))

    words.sort(key=lambda w: w.start)
    # A word ends when the next one starts (captions overlap), capped so
    # a word before a pause doesn't linger on screen.
    for i, w in enumerate(words):
        nxt = words[i + 1].start if i + 1 < len(words) else w.start + 0.6
        w.end = max(w.start + 0.08, min(nxt, w.start + 0.9))
    return words


def _pcm_floats(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def read_wav_16k(path: Path) -> np.ndarray:
    """Read a mono 16-bit WAV as samples in [-1, 1)."""
    with wave.open(str(path), "rb") as w:
        return _pcm_floats(w.readframes(w.getnframes()))


_TRANSCRIPTION_CHUNK_SECONDS = 300
_TRANSCRIPTION_CONTEXT_SECONDS = 2


def whisper_transcribe(video_path: Path, workdir: Path, progress: Progress) -> list[Word]:
    from faster_whisper import WhisperModel

    wav = workdir / "audio.wav"
    words: list[Word] = []
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000", str(wav)],
            check=True,
        )
        with wave.open(str(wav), "rb") as audio:
            sample_rate = audio.getframerate()
            total_frames = audio.getnframes()
            if not total_frames:
                return words
            batch_frames = _TRANSCRIPTION_CHUNK_SECONDS * sample_rate
            context_frames = _TRANSCRIPTION_CONTEXT_SECONDS * sample_rate
            parts = max(1, -(-total_frames // batch_frames))
            total = total_frames / sample_rate
            report(progress, 26, f"Loading Whisper {config.WHISPER_MODEL}", phase="load_whisper",
                   total_seconds=total, parts=parts)
            cached_models = config.MODELS_DIR / "whisper"
            model_options = {"download_root": str(cached_models)} if cached_models.is_dir() else {}
            model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8", **model_options)
            started = monotonic()
            for part, first_frame in enumerate(range(0, total_frames, batch_frames), 1):
                last_frame = min(first_frame + batch_frames, total_frames)
                chunk_start, chunk_end = first_frame / sample_rate, last_frame / sample_rate
                message = f"Transcribing audio (part {part} of {parts})" if parts > 1 else "Transcribing audio"

                def update(processed: float) -> None:
                    elapsed = monotonic() - started
                    eta = (total - processed) * elapsed / processed if processed >= 30 and elapsed >= 5 else None
                    report(
                        progress, 26 + 14 * processed / total, message, phase="transcribe",
                        part=part, parts=parts, completed=part if processed >= chunk_end else part - 1,
                        processed_seconds=processed, total_seconds=total,
                        chunk_start_seconds=chunk_start, chunk_end_seconds=chunk_end,
                        words_transcribed=len(words), eta_seconds=round(eta) if eta is not None else None,
                    )

                update(chunk_start)
                # Overlap adjacent reads for context, but each word belongs to the
                # chunk containing its midpoint on the original timeline.
                read_start = max(0, first_frame - context_frames)
                read_end = min(total_frames, last_frame + context_frames)
                audio.setpos(read_start)
                offset = read_start / sample_rate
                samples = _pcm_floats(audio.readframes(read_end - read_start))
                # Passing samples also avoids PyAV's version-dependent file decoder.
                segments, _ = model.transcribe(samples, word_timestamps=True, vad_filter=True)
                processed = chunk_start
                for segment in segments:
                    for word in segment.words or []:
                        text = word.word.strip()
                        start, end = offset + word.start, offset + word.end
                        if text and chunk_start <= (start + end) / 2 < chunk_end:
                            words.append(Word(text, start, min(end, total)))
                    processed = max(processed, min(offset + segment.end, chunk_end))
                    update(processed)
                # A silent part still advances the timeline even when VAD returns no words.
                update(chunk_end)
    finally:
        wav.unlink(missing_ok=True)
    words.sort(key=lambda word: word.start)
    return words
