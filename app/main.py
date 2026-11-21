"""Web server: accepts a YouTube link, runs the clipping pipeline in the
background, and serves progress and finished clips to the page."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
import time
import traceback
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.background import BackgroundTask

from . import captions, cloud_transcribe, config, meta, render, selector, settings, source, trim, youtube
from .pipeline_progress import report

@asynccontextmanager
async def _lifespan(app: FastAPI):
    stop = threading.Event()
    scheduler = threading.Thread(target=_instagram_scheduler, args=(stop,), daemon=True)
    scheduler.start()
    try:
        yield
    finally:
        stop.set()
        scheduler.join()


app = FastAPI(title="Arsuclip", lifespan=_lifespan)


_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"} | {
    h.strip().lower() for h in os.environ.get("CLIPSMITH_ALLOWED_HOSTS", "").split(",") if h.strip()
}


def _hostname(host_header: str) -> str:
    host = host_header.strip().lower()
    if host.startswith("["):  # [::1]:8765
        return host[1:host.find("]")] if "]" in host else host
    return host.rsplit(":", 1)[0] if host.count(":") == 1 else host


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """Only this computer's own web pages may use the app.

    - Host check (DNS rebinding): a site can point its own name at 127.0.0.1,
      making its requests look same-origin. Requests not addressed to
      127.0.0.1 / localhost are refused, reads included.
    - Origin check (CSRF): browsers send Origin on cross-site POST/DELETE; a
      page on another site must not delete videos, post clips or disconnect
      YouTube. Extra hostnames can be allowed with CLIPSMITH_ALLOWED_HOSTS."""
    if _hostname(request.headers.get("host", "")) not in _LOCAL_HOSTS:
        return JSONResponse({"detail": "Open Clipsmith at http://127.0.0.1:8765."}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "Request from another website refused."}, status_code=403)
    return await call_next(request)

_jobs: dict[str, dict] = {}
_lock = threading.Lock()
_cache_retry_lock = threading.Lock()
_pool = ThreadPoolExecutor(max_workers=1)  # one video at a time; encoding uses every core
_upload_pool = ThreadPoolExecutor(max_workers=1)  # YouTube uploads don't wait behind rendering
_meta_upload_pool = ThreadPoolExecutor(max_workers=2)
_POST_ACTIVE = ("queued", "uploading")

_YT = re.compile(r"^https?://([\w-]+\.)?(youtube\.com|youtu\.be)/", re.I)


ClipCount = Annotated[int, Field(ge=config.MIN_CLIP_COUNT, le=config.MAX_CLIP_COUNT)] | Literal["auto"]
CaptionLanguage = Literal["hi", "en", "hinglish"]


class JobRequest(BaseModel):
    url: str
    clip_count: ClipCount = config.MAX_CLIP_COUNT
    min_len: int = Field(default=30, ge=10, le=170)
    max_len: int = Field(default=60, ge=15, le=180)
    auto_length: bool = False
    caption_language: CaptionLanguage = "hinglish"
    style: str = "yellow"
    layout: str = "auto"
    hook: bool = True
    prompt: str = Field(default="", max_length=300)
    start_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_time: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    aspect: str = "9:16"
    trim_pauses: bool = True
    auto_post_channel_id: str | None = Field(default=None, max_length=64)
    auto_post_instagram_id: str | None = Field(default=None, pattern=r"^[0-9]{1,32}$")
    auto_post_made_for_kids: bool | None = None
    auto_post_start_at: datetime | None = None
    auto_post_interval_minutes: int | None = Field(default=None, ge=1, le=525600)


_JOB_ID = re.compile(r"[0-9a-f]{10}")  # uuid4().hex[:10], see _new_job


def _job_dir(job_id: str) -> Path:
    """A job's folder. Job IDs come from URLs, so anything that isn't exactly a
    generated ID is refused before it can become part of a file path."""
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ValueError("Not a valid job ID.")
    # Belt and braces: resolve the path and confirm it is still inside the jobs folder.
    root = os.path.realpath(config.JOBS_DIR)
    path = os.path.realpath(os.path.join(root, job_id))
    if not path.startswith(root + os.sep):
        raise ValueError("Not a valid job ID.")
    return Path(path)


def _save(job: dict) -> None:
    d = _job_dir(job["id"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(d / "job.json")


def _update(job_id: str, **fields) -> None:
    with _lock:
        job = _jobs[job_id]
        job.update(fields)
        _save(job)


def _load_existing() -> None:
    for f in config.JOBS_DIR.glob("*/job.json"):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _JOB_ID.fullmatch(str(job.get("id", ""))):
            continue  # not a job this app created
        if job.get("status") in ("queued", "running"):
            job["status"] = "error"
            job["error"] = "The app was closed while this was processing."
        for clip in job.get("clips") or []:
            yt = clip.get("youtube")
            if yt and yt.get("status") in ("queued", "uploading"):
                clip["youtube"] = {"status": "error", "error": "The app was closed during the upload. Post it again."}
            for post in (clip.get("meta_posts") or {}).values():
                if post.get("status") == "queued" and post.get("scheduled_for"):
                    post.update(status="scheduled", message="Scheduled")
                    continue
                if post.get("status") not in _POST_ACTIVE:
                    continue
                if post.get("message") == "Confirming publication":
                    post.update(status="check", message="Arsuclip closed while confirming. Check Published in Meta Business Suite.")
                else:
                    post.update(status="error", message="Arsuclip closed before posting. You can try this Reel again.")
        _jobs[job["id"]] = job


_load_existing()


def _clock(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def friendly_error(exc: BaseException) -> str:
    """Turn a pipeline exception (often raw yt-dlp output) into a message for the page."""
    text = _ANSI.sub("", str(exc)).strip()
    text = re.sub(r"^ERROR:\s*", "", text)
    low = text.lower()
    if "getaddrinfo" in low or "failed to resolve" in low or "connection refused" in low or "timed out" in low:
        return "Couldn't reach YouTube. Check your internet connection and try again."
    if "429" in low or "too many requests" in low:
        return "YouTube is limiting requests from your connection right now. Wait a few minutes and try again."
    text = re.sub(r"^\[\w+\]\s*[\w-]+:\s*", "", text)  # drop yt-dlp's "[youtube] <id>: " prefix
    return text or type(exc).__name__


def _render_bounds(
    clip_start: float,
    clip_end: float,
    words: list,
    duration: float | None,
    range_start: float | None = None,
    range_end: float | None = None,
) -> tuple[float, float]:
    """Pad a clip slightly so its first word isn't cut and the moment can land:
    run on into the pause after the last word (a laugh, a reaction, engine
    noise), never past the video or outside a requested From/To range."""
    start = max(0.0, clip_start - 0.15)
    next_word = next((w.start for w in words if w.start >= clip_end - 0.05), None)
    tail = 1.5 if next_word is None else max(0.3, min(1.5, next_word - clip_end - 0.1))
    end = clip_end + tail
    if duration:
        end = min(end, duration)
    if range_start is not None:
        start = max(start, range_start)
    if range_end is not None:
        end = min(end, range_end)
    return round(start, 3), round(end, 3)


def _run(job_id: str, req: JobRequest) -> None:
    workdir = _job_dir(job_id)
    model_context = config.pin_ai_settings()

    last = {"pct": -1.0, "msg": "", "at": 0.0, "detail": {}}
    progress_lock = threading.Lock()

    def record_progress(pct: float, message: str, detail: dict | None = None) -> None:
        with progress_lock:
            pct = max(last["pct"], round(min(pct, 99), 1))
            now = time.monotonic()
            if detail is None:
                detail = last["detail"]
            phase_changed = detail.get("phase") != last["detail"].get("phase")
            changed = message != last["msg"] or phase_changed
            if not changed and pct == last["pct"] and detail == last["detail"]:
                return
            if not changed and now - last["at"] < 2 and pct - last["pct"] < 1:
                return
            last.update(pct=pct, msg=message, at=now, detail=detail)
            identified = {"title": detail["title"]} if detail.get("title") else {}
            _update(job_id, progress=pct, message=message, work_detail=detail, work_updated=time.time(), **identified)

    def progress(pct: float, message: str) -> None:
        record_progress(pct, message)

    progress.detail = record_progress

    try:
        _update(job_id, status="running")
        job = _jobs[job_id]
        upload = job.get("upload")
        if job.get("cached_from"):
            src = cloud_transcribe.from_cached(workdir / "source.mp4", workdir / "audio.wav", workdir, job["title"], progress)
        elif upload:
            path = next(workdir.glob("source.*"))
            src = source.from_local_file(path, workdir, progress, title=Path(upload).stem)
        else:
            src = source.download(req.url, workdir, progress)
        _update(job_id, title=src.title, duration=src.duration, transcript_source=src.transcript_source)

        report(progress, 42, "Finding the best moments", phase="rank")
        min_len = config.AUTO_CLIP_MIN_SECONDS if req.auto_length else req.min_len
        max_len = config.AUTO_CLIP_MAX_SECONDS if req.auto_length else req.max_len
        sentences = selector.within(selector.to_sentences(src.words), req.start_time, req.end_time)
        clips, method, warning = selector.select(
            sentences, src.title, req.clip_count, min_len, max_len, progress, prompt=req.prompt,
            **({"auto_length": True} if req.auto_length else {}),
        ) if sentences else ([], "basic", None)
        if not clips:
            ranged = req.start_time is not None or req.end_time is not None
            where = (
                f"between {_clock(req.start_time or 0)} and "
                f"{_clock(req.end_time) if req.end_time is not None else 'the end'} of this video"
                if ranged else "in this video"
            )
            raise RuntimeError(
                f"No stretch of speech between {min_len} and {max_len} seconds was found {where}."
            )
        _update(job_id, selection=method, warning=warning)

        done: list[dict] = []
        for i, clip in enumerate(clips):
            start, end = _render_bounds(
                clip.start, clip.end, src.words, src.duration, req.start_time, req.end_time
            )
            position = {"part": i + 1, "parts": len(clips), "source_start": start, "source_end": end,
                        "clip_title": clip.title, "caption_language": req.caption_language}
            pct = 45 + 54 * i / len(clips)
            report(progress, pct, f"Preparing clip {i + 1} of {len(clips)}", phase="prepare_clip", **position)
            name = f"clip_{i + 1:02d}"
            keep = trim.plan(src.video_path, src.words, start, end) if req.trim_pauses else None
            if keep and sum(right - left for left, right in keep) < min_len:
                keep = None
            clip_words = [word for word in src.words if word.end > start and word.start < end]
            caption_words, hook_text = captions.prepare(clip_words, clip.hook if req.hook else "", req.caption_language)
            report(progress, pct, f"Rendering clip {i + 1} of {len(clips)}", phase="render", **position)
            info = render.render_clip(
                src.video_path, caption_words, start, end, workdir, name,
                style=req.style, layout=req.layout, hook=hook_text or None, aspect=req.aspect,
                keep=keep,
            )
            done.append(
                asdict(clip)
                | {
                    "start": round(start, 2),
                    "end": round(end, 2),
                    "length": round(info.get("length", end - start), 1),
                    "video": f"/media/{job_id}/{name}.mp4",
                    "thumb": f"/media/{job_id}/{name}.jpg",
                    "layout": info["layout"],
                    "aspect": req.aspect,
                    "caption_language": req.caption_language,
                }
            )
            _save_ready_clips(job_id, done, req)

        src.video_path.unlink(missing_ok=True)  # the source is large; clips are what matter
        for leftover in workdir.glob("source.*"):
            leftover.unlink(missing_ok=True)
        if job.get("cached_from"):
            (workdir / "audio.wav").unlink(missing_ok=True)
        _update(job_id, status="done", progress=100, message="Done",
                work_detail={"phase": "done", "completed": len(done)}, work_updated=time.time())
    except Exception as exc:
        traceback.print_exc()
        _update(job_id, status="error", error=friendly_error(exc)[:500])
    finally:
        config.reset_ai_settings(model_context)


@app.get("/api/status")
def status() -> dict:
    provider = config.ai_provider()
    model = config.ai_model(provider)
    return {
        "ai": provider,
        "missing": config.missing_tools(),
        "model": f"{model} (local)" if provider == "ollama" else model,
        "studio_settings": True,
        "styles": {k: v["label"] for k, v in render.STYLES.items()},
        "style_looks": {
            k: {"font": v["font"], "active": v["css"], "upper": v["upper"], "width": v["cw"],
                "reveal": bool(v.get("reveal"))} for k, v in render.STYLES.items()
        },
    }


@app.get("/api/settings")
def integration_settings() -> JSONResponse:
    return JSONResponse(settings.status(), headers={"Cache-Control": "no-store"})


@app.put("/api/settings/{category}")
def update_integration_settings(category: Literal["ai", "youtube", "routing"], data: dict) -> JSONResponse:
    try:
        save = {"ai": settings.save_ai, "youtube": settings.save_youtube, "routing": settings.save_routing}[category]
        return JSONResponse(save(data), headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except OSError:
        raise HTTPException(500, "Could not write to .env. Check the folder's write access.") from None


@app.post("/api/settings/check")
def check_model_connection(data: dict | None = None) -> JSONResponse:
    from . import model_connection

    try:
        return JSONResponse(model_connection.test_connection(data or {}), headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from None


@app.post("/api/settings/models")
def list_provider_models(data: dict) -> JSONResponse:
    from . import model_connection

    try:
        return JSONResponse(model_connection.fetch_models(data), headers={"Cache-Control": "no-store"})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    except RuntimeError as exc:
        raise HTTPException(502, str(exc)) from None


def _check_options(req: JobRequest) -> None:
    if not req.auto_length and req.min_len >= req.max_len:
        raise HTTPException(400, "Minimum length must be shorter than maximum length.")
    if req.style not in render.STYLES:
        raise HTTPException(400, "Unknown caption style.")
    if req.layout not in ("auto", "fill", "fit", "split"):
        raise HTTPException(400, "Unknown layout.")
    if req.start_time is not None and req.end_time is not None and req.end_time <= req.start_time:
        raise HTTPException(400, "The 'To' time must be after the 'From' time.")
    if req.aspect not in render.ASPECTS:
        raise HTTPException(400, "Unknown aspect ratio.")
    if req.auto_post_start_at is not None or req.auto_post_interval_minutes is not None:
        start_at = req.auto_post_start_at
        if not (req.auto_post_channel_id or req.auto_post_instagram_id) or start_at is None or req.auto_post_interval_minutes is None:
            raise HTTPException(400, "Set the first publish time and the gap to schedule auto-posts.")
        if start_at.utcoffset() is None or start_at <= datetime.now(timezone.utc):
            raise HTTPException(400, "Choose a future start date and time with a time zone.")
    if req.auto_post_channel_id:
        connection = youtube.status()
        channel_id = (connection.get("channel") or {}).get("id")
        if not connection["connected"] or channel_id != req.auto_post_channel_id:
            raise HTTPException(409, "Auto-post needs a verified YouTube channel. Reconnect YouTube and try again.")
        if req.auto_post_made_for_kids is None:
            raise HTTPException(400, "Choose whether these clips are made for kids before enabling auto-post.")
    if req.auto_post_instagram_id:
        _check_meta_account("instagram", req.auto_post_instagram_id)


def _new_job(req: JobRequest, job_id: str | None = None, **extra) -> dict:
    job_id = job_id or uuid.uuid4().hex[:10]
    job = {
        "id": job_id, "url": req.url.strip(), "status": "queued", "progress": 0, "message": "Waiting to start",
        "title": None, "clips": [], "error": None, "warning": None, "selection": None,
        "created": time.time(), "options": req.model_dump(mode="json", exclude={"url"}),
    } | extra
    with _lock:
        _jobs[job_id] = job
        _save(job)
    _pool.submit(_run, job_id, req)
    return job


@app.post("/api/jobs")
def create_job(req: JobRequest) -> dict:
    if not _YT.match(req.url.strip()):
        raise HTTPException(400, "That doesn't look like a YouTube link.")
    _check_options(req)
    return _new_job(req)


class CachedAudioRequest(BaseModel):
    title: str = Field(default="", max_length=300)
    allow_cloud: bool = False


@app.post("/api/jobs/{source_id}/cloud-transcribe")
def transcribe_cached_job(source_id: str, request: CachedAudioRequest) -> dict:
    """Reuse downloaded media for an explicitly requested cloud transcription."""
    if not _JOB_ID.fullmatch(source_id):
        raise HTTPException(404, "The source job was not found.")
    if not request.allow_cloud:
        raise HTTPException(400, "Cloud transcription sends the source audio to Google Gemini. Explicit approval is required.")
    if not config.ai_key("gemini"):
        raise HTTPException(409, "Save a Gemini API key before using cloud transcription.")
    with _cache_retry_lock:
        previous = next((job for job in _jobs.copy().values() if job.get("cached_from") == source_id), None)
        if previous:
            if previous["status"] != "error":
                return previous
            directory = _job_dir(previous["id"])
            if (directory / "source.mp4").is_file() and (directory / "audio.wav").is_file():
                req = JobRequest(url=previous["url"], **previous["options"])
                _update(previous["id"], status="queued", error=None, message="Waiting to resume cached transcription")
                _pool.submit(_run, previous["id"], req)
                return previous

        for root in dict.fromkeys((config.JOBS_DIR, config.ROOT / "data" / "jobs")):
            candidate = root.resolve() / source_id
            if candidate.resolve() != candidate:
                continue
            assets = [candidate / name for name in ("job.json", "source.mp4", "audio.wav")]
            if all(path.is_file() and not path.is_symlink() for path in assets):
                break
        else:
            raise HTTPException(404, "The downloaded video and extracted audio are no longer available.")

        try:
            original = json.loads(assets[0].read_text(encoding="utf-8"))
            req = JobRequest(url=original["url"], **original["options"])
            if original.get("id") != source_id or not _YT.match(req.url):
                raise ValueError("Invalid source job")
        except (OSError, KeyError, TypeError, ValueError):
            raise HTTPException(409, "The cached job settings could not be read.") from None
        _check_options(req)
        job_id = uuid.uuid4().hex[:10]
        workdir = _job_dir(job_id)
        workdir.mkdir()
        try:
            for media in assets[1:]:
                (workdir / media.name).hardlink_to(media)
        except OSError:
            for media in assets[1:]:
                (workdir / media.name).unlink(missing_ok=True)
            workdir.rmdir()
            raise HTTPException(409, "The cached files could not be reused.") from None
        title = request.title.strip() or original.get("title") or "Cached YouTube video"
        return _new_job(req, job_id, title=title, cached_from=source_id, transcription_engine="gemini-transcribe")


_VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm"}
_MAX_UPLOAD_BYTES = 8 * 1024**3


@app.post("/api/jobs/upload")
async def upload_job(
    file: UploadFile = File(...),
    clip_count: ClipCount = Form(config.MAX_CLIP_COUNT),
    min_len: int = Form(30),
    max_len: int = Form(60),
    auto_length: bool = Form(False),
    caption_language: CaptionLanguage = Form("hinglish"),
    style: str = Form("yellow"),
    layout: str = Form("auto"),
    hook: bool = Form(True),
    prompt: str = Form(""),
    start_time: float | None = Form(None),
    end_time: float | None = Form(None),
    aspect: str = Form("9:16"),
    trim_pauses: bool = Form(True),
    auto_post_channel_id: str | None = Form(None),
    auto_post_instagram_id: str | None = Form(None),
    auto_post_made_for_kids: bool | None = Form(None),
    auto_post_start_at: datetime | None = Form(None),
    auto_post_interval_minutes: int | None = Form(None),
) -> dict:
    name = Path(file.filename or "").name
    ext = Path(name).suffix.lower()
    if ext not in _VIDEO_EXTS:
        raise HTTPException(400, "Please upload a video file: mp4, mov, mkv or webm.")
    try:
        req = JobRequest(url="", clip_count=clip_count, min_len=min_len, max_len=max_len,
                         auto_length=auto_length, caption_language=caption_language,
                         style=style, layout=layout, hook=hook,
                         prompt=prompt, start_time=start_time, end_time=end_time, aspect=aspect,
                         trim_pauses=trim_pauses, auto_post_channel_id=auto_post_channel_id,
                         auto_post_instagram_id=auto_post_instagram_id,
                         auto_post_made_for_kids=auto_post_made_for_kids,
                         auto_post_start_at=auto_post_start_at, auto_post_interval_minutes=auto_post_interval_minutes)
    except ValidationError as exc:
        raise HTTPException(422, str(exc)) from None
    _check_options(req)
    job_id = uuid.uuid4().hex[:10]
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with open(d / f"source{ext}", "wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"File is too large (limit {_MAX_UPLOAD_BYTES // 1024**3} GB).")
                out.write(chunk)
    except BaseException:  # includes a cancelled/interrupted upload
        shutil.rmtree(d, ignore_errors=True)
        raise
    return _new_job(req, job_id, upload=name)


@app.get("/api/jobs/{job_id}/zip")
def download_zip(job_id: str) -> FileResponse:
    with _lock:
        job = _jobs.get(job_id)
        # Copy the list: a clip can be deleted while the zip is being built.
        clips = list(job.get("clips") or []) if job else []
    if not job:
        raise HTTPException(404, "Job not found.")
    d = _job_dir(job_id)
    files = [d / Path(c["video"]).name for c in clips]
    files = [f for f in files if f.is_file()]
    if not files:
        raise HTTPException(409, "This video has no finished clips yet.")
    # Build in a throwaway file (outside the served /media tree) and delete it once sent,
    # so a download doesn't leave a second copy of every clip on disk.
    fd, tmp = tempfile.mkstemp(suffix=".zip", dir=config.DATA_DIR)
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            for f in files:
                try:
                    z.write(f, f.name)
                except FileNotFoundError:
                    continue  # deleted from another tab while zipping
    except BaseException:
        os.unlink(tmp)
        raise
    title = re.sub(r'[\\/:*?"<>|]+', "", job.get("title") or "clips").strip()[:80] or "clips"
    return FileResponse(tmp, media_type="application/zip", filename=f"{title}.zip",
                        background=BackgroundTask(os.unlink, tmp))


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: -j["created"])


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    return job


def _has_active_posts(job: dict) -> bool:
    return any(
        state.get("status") in _POST_ACTIVE
        for clip in job.get("clips") or []
        for state in [(clip.get("youtube") or {}), *(clip.get("meta_posts") or {}).values()]
    )


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
        if job["status"] in ("queued", "running"):
            raise HTTPException(409, "This video is still processing.")
        if _has_active_posts(job):
            raise HTTPException(409, "A clip from this video is being posted to YouTube or Meta. Wait for it to finish.")
        del _jobs[job_id]
    shutil.rmtree(_job_dir(job_id), ignore_errors=True)
    return {"ok": True}


@app.delete("/api/jobs/{job_id}/clips/{index}")
def delete_clip(job_id: str, index: int) -> dict:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found.")
        clips = job.get("clips") or []
        if not 0 <= index < len(clips):
            raise HTTPException(404, "Clip not found.")
        if job["status"] in ("queued", "running"):
            raise HTTPException(409, "This video is still processing.")
        # Uploads find their clip by position, so removing one would point them at the wrong clip.
        if _has_active_posts(job):
            raise HTTPException(409, "A clip from this video is being posted to YouTube or Meta. Wait for it to finish.")
        clip = clips.pop(index)
        _save(job)
    d = _job_dir(job_id)
    for url in (clip.get("video"), clip.get("thumb")):
        if url:
            (d / Path(url).name).unlink(missing_ok=True)
    return {"ok": True}


# ---------- Auto-post ----------

def _save_ready_clips(job_id: str, clips: list[dict], req: JobRequest) -> None:
    index = len(clips) - 1
    clip = clips[index]
    with _lock:
        if req.auto_post_channel_id:
            clip["youtube"] = {"status": "queued", "progress": 0}
        if req.auto_post_instagram_id:
            instagram = {"status": "queued", "message": "Waiting to upload"}
            if req.auto_post_start_at is not None:
                previous = [(earlier.get("meta_posts") or {}).get("instagram", {}).get("scheduled_for")
                            for earlier in clips[:index]]
                try:
                    slot = _next_post_slot(req, index, previous, datetime.now(timezone.utc))
                except (ValueError, OverflowError):
                    instagram.update(status="error", message="Could not assign an Instagram upload time.")
                else:
                    instagram.update(status="scheduled", message="Scheduled", scheduled_for=slot.isoformat())
            clip.setdefault("meta_posts", {})["instagram"] = instagram
        # Persist the upload status together with the clip before it becomes available to manual posting.
        job = _jobs[job_id]
        job["clips"] = list(clips)
        _save(job)
    if req.auto_post_channel_id:
        _upload_pool.submit(_auto_publish, job_id, index, req)
    if req.auto_post_instagram_id and req.auto_post_start_at is None:
        _submit_instagram(job_id, index, clip, req.auto_post_instagram_id)


def _next_post_slot(req: JobRequest, index: int, prior_slots: list[str | None], earliest: datetime) -> datetime:
    gap = timedelta(minutes=req.auto_post_interval_minutes)
    slot = req.auto_post_start_at + index * gap
    last_slot = max((datetime.fromisoformat(value) for value in prior_slots if value), default=None)
    if last_slot is not None:
        slot = max(slot, last_slot + gap)
    if slot < earliest:
        slot += ((earliest - slot) // gap + 1) * gap
    return slot


def _auto_publish(job_id: str, index: int, req: JobRequest) -> None:
    try:
        with _lock:
            clips = _jobs[job_id]["clips"]
            clip = clips[index]
            prior_slots = [(earlier.get("youtube") or {}).get("scheduled_for") for earlier in clips[:index]]
        publish_at = req.auto_post_start_at
        if publish_at is not None:
            # Choose the time just before uploading. A slow render or upload queue may miss a slot.
            earliest = datetime.now(timezone.utc) + timedelta(minutes=1)
            publish_at = _next_post_slot(req, index, prior_slots, earliest)
        hashtags = list(clip.get("hashtags") or [])
        if req.aspect == "9:16":
            hashtags.append("Shorts")
        metadata = youtube.build_metadata(
            title=clip.get("title") or f"Clip {index + 1}",
            description=clip.get("description") or "",
            hashtags=hashtags,
            publish_at=publish_at,
            made_for_kids=req.auto_post_made_for_kids is True,
            source_url=req.url.strip() if _YT.match(req.url.strip()) else None,
        )
    except (ValueError, OverflowError) as exc:
        _set_youtube(job_id, index, {"status": "error", "error": str(exc)})
        return
    _publish(job_id, index, metadata, False, req.auto_post_channel_id)


class PublishRequest(BaseModel):
    title: str = Field(max_length=500)
    description: str = Field(default="", max_length=10000)
    hashtags: list[str] = Field(default_factory=list, max_length=100)
    privacy: str = "public"
    publish_at: datetime | None = None
    made_for_kids: bool = False
    notify_subscribers: bool = True
    add_source_link: bool = False
    # The clip's video path when the dialog opened. Clips are addressed by position, so if one
    # was deleted meanwhile this catches a post landing on a different clip.
    clip_video: str | None = None


@app.get("/api/youtube")
def youtube_status() -> dict:
    return youtube.status()


class MetaConnectionRequest(BaseModel):
    access_token: str = Field(min_length=1, max_length=4096, repr=False)
    page_id: str = Field(pattern=r"^[0-9]{1,32}$")
    instagram_username: str = Field(pattern=r"^@?[A-Za-z0-9._]{1,30}$")


@app.get("/api/meta")
def meta_status() -> dict:
    return meta.status()


@app.post("/api/meta/connect")
def meta_connect(request: MetaConnectionRequest) -> dict:
    try:
        return meta.connect(request.access_token.strip(), request.page_id, request.instagram_username)
    except meta.ConnectionError as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/api/meta/verify")
def meta_verify() -> dict:
    try:
        return meta.verify()
    except meta.ConnectionError as exc:
        raise HTTPException(400, str(exc)) from None


class MetaReelRequest(BaseModel):
    platform: Literal["facebook", "instagram"]
    account_id: str = Field(pattern=r"^[0-9]{1,32}$")
    clip_video: str


def _check_meta_account(platform: str, account_id: str) -> None:
    connection = meta.status()
    if not connection["connected"] or (connection.get(platform) or {}).get("id") != account_id:
        raise HTTPException(409, "Connect the selected account in Integrations first.")
    if platform == "instagram" and not connection.get("instagram_reels"):
        raise HTTPException(409, "Reconnect Facebook & Instagram in Integrations to enable direct Instagram posting.")


@app.post("/api/jobs/{job_id}/clips/{index}/reel", status_code=202)
def queue_meta_reel(job_id: str, index: int, req: MetaReelRequest) -> dict:
    _check_meta_account(req.platform, req.account_id)
    with _lock:
        job = _jobs.get(job_id)
        clips = (job or {}).get("clips") or []
        if not 0 <= index < len(clips):
            raise HTTPException(404, "Clip not found.")
        clip = clips[index]
        if clip.get("video") != req.clip_video:
            raise HTTPException(409, "This clip changed. Refresh the page before posting.")
        if req.platform == "facebook" and (clip.get("aspect", "9:16") != "9:16" or (clip.get("length") or 0) > 90):
            raise HTTPException(400, "Direct Facebook Reel publishing requires a vertical 9:16 clip up to 90 seconds.")
        path = _job_dir(job_id) / Path(req.clip_video).name
        if not path.is_file():
            raise HTTPException(404, "The rendered clip file is missing.")
        posts = clip.setdefault("meta_posts", {})
        if posts.get(req.platform, {}).get("status") not in (None, "error"):
            raise HTTPException(409, "This clip is already posted, queued, or awaiting confirmation on that platform.")
        state = {"status": "queued", "message": "Waiting to upload"}
        posts[req.platform] = state
        _save(job)
    _meta_upload_pool.submit(_publish_meta_reel, job_id, index, req, path)
    return state


def _submit_instagram(job_id: str, index: int, clip: dict, account_id: str) -> None:
    req = MetaReelRequest(platform="instagram", account_id=account_id, clip_video=clip["video"])
    path = _job_dir(job_id) / Path(req.clip_video).name
    _meta_upload_pool.submit(_publish_meta_reel, job_id, index, req, path)


def _dispatch_scheduled_instagram(now: datetime) -> None:
    with _lock:
        for job_id, job in _jobs.items():
            account_id = (job.get("options") or {}).get("auto_post_instagram_id")
            if not account_id:
                continue
            pending = []
            for index, clip in enumerate(job.get("clips") or []):
                post = (clip.get("meta_posts") or {}).get("instagram") or {}
                if post.get("status") == "scheduled":
                    pending.append((index, clip, post, datetime.fromisoformat(post["scheduled_for"])))
            if not pending or pending[0][3] > now:
                continue
            index, clip, post, _ = pending[0]
            post.update(status="queued", message="Waiting to upload")
            # Keep overdue posts spaced when the app was closed across multiple slots.
            if len(pending) > 1 and pending[1][3] <= now:
                gap = timedelta(minutes=job["options"]["auto_post_interval_minutes"])
                due = now
                for _, _, later, slot in pending[1:]:
                    due = max(slot, due + gap)
                    later["scheduled_for"] = due.isoformat()
            _save(job)
            _submit_instagram(job_id, index, clip, account_id)


def _instagram_scheduler(stop: threading.Event) -> None:
    while not stop.is_set():
        try:
            _dispatch_scheduled_instagram(datetime.now(timezone.utc))
        except Exception:
            traceback.print_exc()
        stop.wait(5)


def _publish_meta_reel(job_id: str, index: int, req: MetaReelRequest, path: Path) -> None:
    with _lock:
        clip = _jobs[job_id]["clips"][index]
        title = (clip.get("title") or f"Clip {index + 1}")[:100]
        tags = " ".join("#" + tag.lstrip("#").replace(" ", "") for tag in (clip.get("hashtags") or []))
        caption = "\n\n".join(part for part in (title, clip.get("description"), tags) if part)[:2200]

    def update(status: str, message: str, **result) -> None:
        with _lock:
            job = _jobs[job_id]
            posts = job["clips"][index]["meta_posts"]
            posts[req.platform] = {**posts[req.platform], "status": status, "message": message, **result}
            _save(job)

    try:
        if not path.is_file():
            raise meta.ConnectionError("The rendered clip file is missing.")
        result = meta.publish_reel(req.platform, req.account_id, path, title, caption,
                                  lambda message: update("uploading", message))
    except meta.PublishUnconfirmed as exc:
        update("check", str(exc))
    except meta.ConnectionError as exc:
        update("error", str(exc))
    except Exception:
        traceback.print_exc()
        update("check", "Arsuclip could not confirm the result. Check Published in Meta Business Suite.")
    else:
        update("done", "Reel published", **result)


@app.get("/api/youtube/connect")
def youtube_connect(request: Request) -> RedirectResponse:
    # Google documents the desktop loopback redirect as http://127.0.0.1:port with
    # no path, so Google sends the user back to this app's root URL (see index()).
    redirect = str(request.base_url)
    try:
        return RedirectResponse(youtube.start_auth(redirect), status_code=307)
    except youtube.NotConfigured as exc:
        raise HTTPException(400, str(exc)) from None


@app.post("/api/youtube/disconnect")
def youtube_disconnect() -> dict:
    youtube.disconnect()
    return {"ok": True}


@app.post("/api/jobs/{job_id}/clips/{index}/youtube")
def publish_clip(job_id: str, index: int, req: PublishRequest) -> dict:
    if not youtube.status()["connected"]:
        raise HTTPException(409, "Connect YouTube first.")
    with _lock:
        job = _jobs.get(job_id)
        clips = (job or {}).get("clips") or []
        if not job or not 0 <= index < len(clips):
            raise HTTPException(404, "Clip not found.")
        clip = clips[index]
        if req.clip_video is not None and req.clip_video != clip.get("video"):
            raise HTTPException(409, "This clip changed since you opened it (another clip was deleted). Reload the page and try again.")
        if (clip.get("youtube") or {}).get("status") in ("queued", "uploading"):
            raise HTTPException(409, "This clip is already being posted.")
        source_url = job.get("url") if req.add_source_link and _YT.match(job.get("url") or "") else None
        try:
            meta = youtube.build_metadata(
                title=req.title, description=req.description, hashtags=req.hashtags, privacy=req.privacy,
                publish_at=req.publish_at, made_for_kids=req.made_for_kids, source_url=source_url,
            )
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from None
        clip["youtube"] = {"status": "queued", "progress": 0}
        _save(job)
    _upload_pool.submit(_publish, job_id, index, meta, req.notify_subscribers)
    return clip["youtube"]


def _set_youtube(job_id: str, index: int, value: dict) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job and index < len(job.get("clips") or []):
            job["clips"][index]["youtube"] = value
            _save(job)


def _publish(job_id: str, index: int, meta: dict, notify: bool, channel_id: str | None = None) -> None:
    with _lock:
        job = _jobs.get(job_id)
        clip = job["clips"][index] if job else None
    if not clip:
        return
    path = _job_dir(job_id) / Path(clip["video"]).name
    last = {"pct": -10}

    def progress(pct: int) -> None:
        if pct - last["pct"] >= 5 or pct == 100:  # don't rewrite job.json for every chunk
            last["pct"] = pct
            _set_youtube(job_id, index, {"status": "uploading", "progress": pct})

    try:
        _set_youtube(job_id, index, {"status": "uploading", "progress": 0})
        upload_options = {"expected_channel_id": channel_id} if channel_id else {}
        video_id = youtube.upload(path, meta, notify_subscribers=notify, progress=progress, **upload_options)
        result = {"status": "done", "video_id": video_id, "url": f"https://youtu.be/{video_id}"}
        if meta.get("status", {}).get("publishAt"):
            result["scheduled_for"] = meta["status"]["publishAt"]
        else:
            result["privacy"] = meta.get("status", {}).get("privacyStatus")
        _set_youtube(job_id, index, result)
    except youtube.UploadError as exc:
        _set_youtube(job_id, index, {"status": "error", "error": str(exc)})
    except Exception as exc:
        traceback.print_exc()
        _set_youtube(job_id, index, {"status": "error", "error": f"Upload failed: {exc}"[:300]})


app.mount("/fonts", StaticFiles(directory=render.FONTS_DIR), name="fonts")
app.mount("/media", StaticFiles(directory=config.JOBS_DIR), name="media")


@app.get("/")
def index(state: str | None = None, code: str | None = None, error: str | None = None):
    # Google sends the browser back here after the YouTube sign-in.
    if error and state:
        reason = "access_denied" if error == "access_denied" else "failed"  # never echo the raw value
        return RedirectResponse(f"/?youtube=error&reason={reason}", status_code=303)
    if state and code:
        try:
            youtube.finish_auth(state, code)
        except youtube.AuthError:
            return RedirectResponse("/?youtube=error&reason=expired", status_code=303)
        return RedirectResponse("/?youtube=connected", status_code=303)
    return FileResponse(config.STATIC_DIR / "index.html")
