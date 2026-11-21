from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app import main, meta


def test_links_only_the_requested_page_and_instagram(tmp_path, monkeypatch):
    token_file = tmp_path / "meta_token.json"
    monkeypatch.setattr(meta, "TOKEN_PATH", token_file)

    def graph(path, token, **params):
        if path == "me/permissions" and token == "user-token":
            return {"data": [{"permission": permission, "status": "granted"}
                             for permission in meta.PUBLISH_PERMISSIONS]}
        if path == "123" and token == "user-token":
            return {"id": "123", "name": "Example Page", "access_token": "page-token",
                    "instagram_business_account": {"id": "456", "username": "example"}}
        if path == "me" and token == "page-token":
            return {"id": "123"}
        if path == "456/content_publishing_limit" and token in ("page-token", "user-token"):
            return {"data": [{"quota_usage": 0}]}
        raise AssertionError((path, token))

    monkeypatch.setattr(meta, "_get", graph)
    client = TestClient(main.app)
    options = {"page_id": "123", "instagram_username": "another", "access_token": "user-token"}
    assert client.post("/api/meta/connect", json=options).status_code == 400
    assert not token_file.exists()

    options["instagram_username"] = "@example"
    result = client.post("/api/meta/connect", json=options)
    assert result.status_code == 200
    expected = {"connected": True, "instagram_reels": True, "facebook": {"id": "123", "name": "Example Page"},
                "instagram": {"id": "456", "username": "example"}}
    assert result.json() == client.get("/api/meta").json() == client.post("/api/meta/verify").json() == expected
    saved = token_file.read_text(encoding="utf-8")
    assert "page-token" in saved and "user-token" in saved
    assert "page-token" not in result.text and "user-token" not in result.text


def test_reel_queue_prevents_duplicate_posts(tmp_path, monkeypatch):
    job_id = "1234567890"
    url = f"/media/{job_id}/clip_01.mp4"
    clip = {"video": url, "title": "Test title", "aspect": "9:16", "length": 30}
    monkeypatch.setattr(main.config, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "_jobs", {job_id: {"id": job_id, "clips": [clip], "status": "done"}})
    monkeypatch.setattr(meta, "status", lambda: {"connected": True, "facebook": {"id": "123"}})
    queued = []
    monkeypatch.setattr(main._meta_upload_pool, "submit", lambda *args: queued.append(args))
    folder = tmp_path / job_id
    folder.mkdir()
    (folder / "clip_01.mp4").write_bytes(b"video")
    client = TestClient(main.app)
    path = f"/api/jobs/{job_id}/clips/0/reel"
    options = {"platform": "facebook", "account_id": "123", "clip_video": url}
    assert client.post(path, json=options).status_code == 202
    assert client.post(path, json=options).status_code == 409
    assert client.delete(f"/api/jobs/{job_id}/clips/0").status_code == 409
    assert len(queued) == 1
    monkeypatch.setattr(meta, "publish_reel", lambda *args: {"id": "987", "url": "https://www.facebook.com/reel/987/"})
    worker, *args = queued[0]
    worker(*args)
    assert clip["meta_posts"]["facebook"]["status"] == "done"
    assert client.post(path, json=options).status_code == 409


@pytest.mark.parametrize("file_upload", [False, True])
def test_instagram_only_auto_post_accepts_verified_account(tmp_path, monkeypatch, file_upload):
    monkeypatch.setattr(main.config, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "_jobs", {})
    monkeypatch.setattr(main._pool, "submit", lambda *args: None)
    monkeypatch.setattr(meta, "status", lambda: {"connected": True, "instagram_reels": True,
                                               "instagram": {"id": "456"}})
    options = {"auto_post_instagram_id": "456", "auto_post_interval_minutes": 23,
               "auto_post_start_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()}
    client = TestClient(main.app)
    if file_upload:
        response = client.post("/api/jobs/upload", data=options, files={"file": ("test.mp4", b"video")})
    else:
        response = client.post("/api/jobs", json={"url": "https://youtu.be/abc", **options})
    assert response.status_code == 200
    assert response.json()["options"]["auto_post_instagram_id"] == "456"
    rejected = client.post("/api/jobs", json={"url": "https://youtu.be/abc", **options, "auto_post_instagram_id": "789"})
    assert rejected.status_code == 409


@pytest.mark.parametrize("scheduled", [False, True])
def test_both_platforms_auto_post_and_instagram_recovers_schedule(tmp_path, monkeypatch, scheduled):
    job_id = "1234567890"
    video = f"/media/{job_id}/clip_01.mp4"
    clip = {"video": video, "title": "Test title", "description": "Test caption", "hashtags": ["test"]}
    start = datetime.now(timezone.utc) + timedelta(hours=1)
    req = main.JobRequest(url="https://youtu.be/abc", auto_post_instagram_id="456",
                          auto_post_channel_id="channel", auto_post_made_for_kids=False,
                          auto_post_start_at=start if scheduled else None,
                          auto_post_interval_minutes=23 if scheduled else None)
    job = {"id": job_id, "clips": [], "status": "done", "options": req.model_dump(mode="json")}
    monkeypatch.setattr(main.config, "JOBS_DIR", tmp_path)
    monkeypatch.setattr(main, "_jobs", {job_id: job})
    youtube_queue, instagram_queue = [], []
    monkeypatch.setattr(main._upload_pool, "submit", lambda *args: youtube_queue.append(args))
    monkeypatch.setattr(main._meta_upload_pool, "submit", lambda *args: instagram_queue.append(args))
    main._save_ready_clips(job_id, [clip], req)
    assert youtube_queue == [(main._auto_publish, job_id, 0, req)]
    (tmp_path / job_id / "clip_01.mp4").write_bytes(b"video")

    if scheduled:
        assert clip["meta_posts"]["instagram"]["status"] == "scheduled"
        assert not instagram_queue
        main._dispatch_scheduled_instagram(start - timedelta(seconds=1))
        assert not instagram_queue
        main._jobs.clear()
        main._load_existing()
        main._dispatch_scheduled_instagram(start)
        main._dispatch_scheduled_instagram(start)
        assert len(instagram_queue) == 1
        # A restart before the worker starts preserves the scheduled post.
        instagram_queue.clear()
        main._jobs.clear()
        main._load_existing()
        main._dispatch_scheduled_instagram(start)
        main._dispatch_scheduled_instagram(start)

    (worker, *args), = instagram_queue
    published = []
    monkeypatch.setattr(meta, "publish_reel", lambda *args: published.append(args) or {"id": "987"})
    worker(*args)
    assert published[0][:2] == ("instagram", "456")
    assert published[0][4] == "Test title\n\nTest caption\n\n#test"
    assert main._jobs[job_id]["clips"][0]["meta_posts"]["instagram"]["status"] == "done"
