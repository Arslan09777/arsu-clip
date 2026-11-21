import json

import pytest
from dotenv import dotenv_values
from fastapi.testclient import TestClient

from app import ai, captions, config, main, selector, settings, youtube
from app.source import Word


@pytest.fixture
def studio(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("# Keep this\nUNRELATED=example\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", path)
    for name in (*config.AI_KEYS, "YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET"):
        monkeypatch.setenv(name, "")
    monkeypatch.setattr(youtube, "TOKEN_PATH", tmp_path / "youtube_token.json")
    return TestClient(main.app), path


def test_custom_settings_save_without_returning_the_key(studio, monkeypatch):
    client, env_path = studio
    request = {"provider": "custom", "model": "test-model", "endpoint": "http://127.0.0.1:20128/v1",
               "api_key": "secret-for-test-only"}
    result = client.put("/api/settings/ai", json=request)
    assert result.status_code == 200 and result.json()["keys"]["custom"]
    assert result.headers["cache-control"] == "no-store"
    assert request["api_key"] not in result.text
    assert request["api_key"] not in client.get("/api/settings").text
    assert dotenv_values(env_path)["CUSTOM_AI_API_KEY"] == request["api_key"]
    assert "# Keep this" in env_path.read_text()
    assert dotenv_values(env_path)["UNRELATED"] == "example"

    class GatewayResponse:
        ok = True
        is_redirect = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"ready":true}'}}]}

    def fake_post(url, *, headers, json, allow_redirects, timeout):
        assert url == "http://127.0.0.1:20128/v1/chat/completions"
        assert headers["Authorization"] == "Bearer secret-for-test-only"
        assert json["model"] == "test-model" and allow_redirects is False
        return GatewayResponse()

    monkeypatch.setattr("requests.post", fake_post)
    assert client.post("/api/settings/check").json() == {"ready": True, "model": "test-model"}


def test_secret_replacement_and_youtube_settings(studio):
    client, env_path = studio
    assert client.put("/api/settings/ai", json={"provider": "gemini", "api_key": "gemini-test"}).status_code == 200
    assert client.put("/api/settings/ai", json={"provider": "gemini", "api_key": ""}).status_code == 200
    assert client.put("/api/settings/youtube", json={"client_id": "test-id", "client_secret": "test-secret"}).status_code == 200
    result = client.get("/api/settings")
    assert result.json()["youtube"] == {"configured": True, "connected": False, "channel": None}
    assert "gemini-test" not in result.text and "test-secret" not in result.text
    values = dotenv_values(env_path)
    assert values["YOUTUBE_CLIENT_SECRET"] == "test-secret"
    assert values["GEMINI_API_KEY"] == "gemini-test"


def test_custom_endpoint_refuses_cleartext_remote_urls(studio):
    client, env_path = studio
    result = client.put("/api/settings/ai", json={"provider": "custom", "endpoint": "http://example.com/v1", "model": "test"})
    assert result.status_code == 422
    assert "CUSTOM_AI_ENDPOINT" not in dotenv_values(env_path)


def test_changing_custom_endpoint_does_not_reuse_the_previous_key(studio):
    client, env_path = studio
    first = {"provider": "custom", "endpoint": "http://127.0.0.1:20128", "model": "local", "api_key": "test"}
    assert client.put("/api/settings/ai", json=first).status_code == 200
    replacement = {"provider": "custom", "endpoint": "https://example.com/v1", "model": "remote"}
    assert client.put("/api/settings/ai", json=replacement).status_code == 422
    assert client.put("/api/settings/ai", json=replacement | {"no_key": True}).status_code == 200
    assert dotenv_values(env_path)["CUSTOM_AI_API_KEY"] == ""


def test_auto_modes_and_subtitle_language_are_preserved_in_jobs(monkeypatch):
    monkeypatch.setattr(main._pool, "submit", lambda *args: None)
    monkeypatch.setattr(main, "_jobs", {})
    client = TestClient(main.app)
    options = {"clip_count": "auto", "auto_length": True, "caption_language": "hi"}
    job = client.post("/api/jobs", json={"url": "https://youtu.be/test", **options}).json()
    assert {key: job["options"][key] for key in options} == options
    upload = client.post("/api/jobs/upload", data=options,
                         files={"file": ("speech.mp4", b"test", "video/mp4")})
    assert upload.status_code == 200
    assert upload.json()["options"]["clip_count"] == "auto"


def test_auto_count_uses_model_scores(monkeypatch):
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    picks = [selector.Clip(i * 60, i * 60 + 40, "Clip", "", 90 if i < 14 else 68, "", "") for i in range(21)]
    monkeypatch.setattr(selector, "select_with_ai", lambda *args, **kwargs: picks)
    result, provider, warning = selector.select([], "Video", "auto", 15, 90, auto_length=True)
    assert len(result) == 14 and provider == "gemini" and warning is None


def test_translation_preserves_the_span_and_hinglish_keeps_word_timing(monkeypatch):
    source = [Word("मैं", 1.0, 1.2), Word("ठीक", 1.2, 1.6), Word("हूँ", 1.6, 2.0)]
    assert captions.prepare(source, "", "hinglish")[0] == [
        Word("main", 1.0, 1.2), Word("theek", 1.2, 1.6), Word("hoon", 1.6, 2.0),
    ]
    assert captions.prepare(source, "", "hi")[0] == source

    def translate(provider, schema, system, prompt):
        assert provider == "gemini" and "natural English" in system
        assert json.loads(prompt)[0]["text"] == "मैं ठीक हूँ"
        return schema(segments=[{"id": 0, "text": "I am fine"}])

    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setattr(ai, "ask_json", translate)
    translated, _ = captions.prepare(source, "", "en")
    assert [word.text for word in translated] == ["I", "am", "fine"]
    assert translated[0].start == 1.0 and translated[-1].end == 2.0
