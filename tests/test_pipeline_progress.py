import wave
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from app import source
from app.pipeline_progress import report


def test_rich_progress_is_optional():
    legacy = Mock()
    report(legacy, 20, "Working", phase="transcribe", part=2)
    legacy.assert_called_once_with(20, "Working")

    rich = Mock()
    rich.detail = Mock()
    report(rich, 21, "Working", phase="transcribe", part=2, eta_seconds=None)
    rich.detail.assert_called_once_with(21, "Working", {"phase": "transcribe", "part": 2})
    rich.assert_not_called()


def test_transcription_chunks_keep_original_timestamps_and_progress(tmp_path, monkeypatch):
    chunks = []

    def fake_ffmpeg(command, **options):
        with wave.open(command[-1], "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8)
            audio.writeframes(np.zeros(96, dtype="<i2").tobytes())

    def segment(*timed_words):
        words = [SimpleNamespace(word=word, start=start, end=end) for word, start, end in timed_words]
        return SimpleNamespace(words=words, end=words[-1].end)

    class Whisper:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, samples, **options):
            assert isinstance(samples, np.ndarray)
            chunks.append(len(samples) / 8)
            segments = (
                [segment(("first", 2.0, 2.5), ("boundary", 4.75, 5.25))],
                [segment(("boundary", 0.75, 1.25), ("last", 3.0, 3.5))],
                [],
            )[len(chunks) - 1]
            return iter(segments), None

    monkeypatch.setattr("faster_whisper.WhisperModel", Whisper)
    monkeypatch.setattr(source.subprocess, "run", fake_ffmpeg)
    monkeypatch.setattr(source, "_TRANSCRIPTION_CHUNK_SECONDS", 5)
    monkeypatch.setattr(source, "_TRANSCRIPTION_CONTEXT_SECONDS", 1)
    progress = Mock()
    progress.detail = Mock()

    words = source.whisper_transcribe(tmp_path / "video.mp4", tmp_path, progress)
    events = [args[2] for args, _ in progress.detail.call_args_list]
    transcript = [event for event in events if event["phase"] == "transcribe"]

    assert chunks == [6, 7, 3]
    assert words == [
        source.Word("first", 2.0, 2.5),
        source.Word("boundary", 4.75, 5.25),
        source.Word("last", 7.0, 7.5),
    ]
    assert events[0] == {"phase": "load_whisper", "total_seconds": 12, "parts": 3}
    assert [(e["part"], e["chunk_start_seconds"], e["chunk_end_seconds"])
            for e in transcript if e["processed_seconds"] == e["chunk_start_seconds"]] == [
        (1, 0, 5), (2, 5, 10), (3, 10, 12),
    ]
    assert transcript[-1]["processed_seconds"] == 12
    assert transcript[-1]["completed"] == 3
    assert transcript[-1]["words_transcribed"] == 3
    assert not (tmp_path / "audio.wav").exists()


def test_download_reports_audio_video_and_merge(tmp_path, monkeypatch):
    class Downloader:
        def __init__(self, options):
            self.options = options

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def extract_info(self, url, download):
            return {"duration": 120, "title": "A title", "subtitles": {"en": [{}]}}

        def download(self, urls):
            hook = self.options["progress_hooks"][0]
            hook({"status": "downloading", "info_dict": {"acodec": "none"}, "downloaded_bytes": 250,
                  "total_bytes": 500, "speed": 100, "eta": 2.5})
            hook({"status": "downloading", "info_dict": {"vcodec": "none"}, "downloaded_bytes": 25})
            self.options["postprocessor_hooks"][0]({"status": "started", "postprocessor": "Merger"})
            (tmp_path / "source.mp4").touch()

    monkeypatch.setattr(source.yt_dlp, "YoutubeDL", Downloader)
    monkeypatch.setattr(source, "_fetch_captions", lambda *_: [source.Word("hello", 0, 1)])
    progress = Mock()
    progress.detail = Mock()
    source.download("https://youtu.be/test", tmp_path, progress)
    events = [args[2] for args, _ in progress.detail.call_args_list]

    assert [event["phase"] for event in events] == [
        "inspect", "download_video", "download_video", "download_audio", "merge", "captions", "captions",
    ]
    assert events[2]["bytes_total"] == 500
    assert events[2]["speed_bytes_per_second"] == 100
    assert events[2]["eta_seconds"] == 2.5
    assert events[3]["bytes_done"] == 25
    assert "bytes_total" not in events[3]
    assert events[-1]["words_transcribed"] == 1
