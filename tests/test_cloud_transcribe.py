import json
import os

from fastapi.testclient import TestClient

from app import cloud_transcribe, config, main


def test_word_offsets_preserve_original_time_and_only_keep_owned_overlap():
    payload = {"steps": [{"type": "model_output", "content": [{
        "text": "पहले बाद",
        "annotations": [
            {"type": "word_info", "start_index": 0, "end_index": 4, "start_offset": "0.5s", "end_offset": "1.2s"},
            {"type": "word_info", "start_index": 5, "end_index": 8, "start_offset": "2.1s", "end_offset": "2.5s"},
            {"type": "word_info", "start_index": 50, "end_index": 55, "start_offset": "4s", "end_offset": "5s"},
        ],
    }]}]}
    words = cloud_transcribe.timed_words(payload, offset=598, start=600, end=1200, total=1200)
    assert [(word.text, word.start, word.end) for word in words] == [("बाद", 600.1, 600.5)]


def test_cloud_job_requires_approval_and_reuses_media_once(tmp_path, monkeypatch):
    source_id = "aabb112233"
    media = tmp_path / "data" / "jobs" / source_id
    media.mkdir(parents=True)
    saved = {"id": source_id, "url": "https://youtu.be/cached", "title": "Cached video", "options": {"clip_count": 12}}
    (media / "job.json").write_text(json.dumps(saved), encoding="utf-8")
    (media / "source.mp4").write_bytes(b"cached video")
    (media / "audio.wav").write_bytes(b"cached audio")
    new_jobs = tmp_path / "preview-jobs"
    new_jobs.mkdir()
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "JOBS_DIR", new_jobs)
    monkeypatch.setattr(main, "_jobs", {})
    submitted = []
    monkeypatch.setattr(main._pool, "submit", lambda *args: submitted.append(args))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    client = TestClient(main.app)
    route = f"/api/jobs/{source_id}/cloud-transcribe"
    assert client.post(route, json={}).status_code == 400
    response = client.post(route, json={"allow_cloud": True})
    assert response.status_code == 200
    job = response.json()
    assert job["cached_from"] == source_id and job["options"]["clip_count"] == 12
    assert os.path.samefile(new_jobs / job["id"] / "source.mp4", media / "source.mp4")
    assert os.path.samefile(new_jobs / job["id"] / "audio.wav", media / "audio.wav")
    assert client.post(route, json={"allow_cloud": True}).json()["id"] == job["id"]
    assert len(submitted) == 1
