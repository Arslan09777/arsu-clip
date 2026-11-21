import json

import pytest
import requests

from app import ai, config, model_connection


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def iter_content(self, size):
        yield json.dumps(self.payload).encode("utf-8")


@pytest.fixture
def saved_custom(monkeypatch, tmp_path):
    env_path = tmp_path / ".env"
    original = "# Keep this file intact\n"
    env_path.write_text(original, encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env_path)
    for key, value in {
        "AI_PROVIDER": "custom",
        "AI_ROUTING_PROVIDERS": "",
        "CUSTOM_AI_ENDPOINT": "https://saved.example/v1",
        "CUSTOM_AI_MODEL": "saved-model",
        "CUSTOM_AI_API_KEY": "saved-credential",
    }.items():
        monkeypatch.setenv(key, value)
    return env_path, original


def test_draft_connection_uses_temporary_model_settings(saved_custom, monkeypatch):
    def ask(provider, schema, system, prompt):
        assert provider == "custom"
        assert config.ai_value("CUSTOM_AI_ENDPOINT") == "https://second.example/v1"
        assert config.ai_value("CUSTOM_AI_API_KEY") == "draft-credential"
        assert config.ai_model(provider) == "draft-model"
        return schema(ready=True)

    monkeypatch.setattr(ai, "ask_json", ask)
    result = model_connection.test_connection({
        "provider": "custom", "endpoint": "https://second.example/v1",
        "api_key": "draft-credential", "model": "draft-model",
    })
    assert result == {"ready": True, "model": "draft-model"}
    assert config.ai_model("custom") == "saved-model"
    assert config.ai_value("CUSTOM_AI_API_KEY") == "saved-credential"
    env_path, original = saved_custom
    assert env_path.read_text(encoding="utf-8") == original


def test_empty_test_request_uses_saved_provider(saved_custom, monkeypatch):
    def ask(provider, schema, *_):
        assert provider == "custom" and config.ai_model(provider) == "saved-model"
        return schema(ready=True)

    monkeypatch.setattr(ai, "ask_json", ask)
    assert model_connection.test_connection({}) == {"ready": True, "model": "saved-model"}


def test_custom_model_list_canonicalizes_url_and_filters_ids(saved_custom, monkeypatch):
    def get(url, *, headers, params, timeout, stream, allow_redirects):
        assert url == "https://saved.example/v1/models"
        assert headers == {"Authorization": "Bearer saved-credential"}
        assert stream is True and allow_redirects is False
        return Response({"data": [{"id": "model-a"}, {"id": "model-a"}, {"id": "model-b"},
                                  {"id": "saved-credential"}, {"id": "x" * 401}, {"id": "bad\nid"},
                                  {"id": 72}, "garbage"]})

    monkeypatch.setattr(requests, "get", get)
    assert model_connection.fetch_models({"endpoint": "https://saved.example/v1/chat/completions/"}) == {
        "models": ["model-a", "model-b"]
    }


def test_changed_custom_endpoint_never_receives_the_saved_key(saved_custom, monkeypatch):
    called = []

    def get(url, **kwargs):
        called.append((url, kwargs["headers"]))
        return Response({"data": [{"id": "public-model"}]})

    monkeypatch.setattr(requests, "get", get)
    options = {"provider": "custom", "endpoint": "http://127.0.0.1:20128"}
    with pytest.raises(ValueError, match="this endpoint's API key"):
        model_connection.fetch_models(options)
    assert called == []
    assert model_connection.fetch_models(options | {"no_key": True}) == {"models": ["public-model"]}
    assert called == [("http://127.0.0.1:20128/v1/models", {})]
    assert config.ai_value("CUSTOM_AI_API_KEY") == "saved-credential"


def test_custom_rejects_non_tls_remote_endpoint_and_follows_no_redirect(saved_custom, monkeypatch):
    def get(url, **kwargs):
        assert kwargs["allow_redirects"] is False
        return Response({}, status=302)

    monkeypatch.setattr(requests, "get", get)
    with pytest.raises(ValueError, match="Use HTTPS"):
        model_connection.fetch_models({"endpoint": "http://internet.example/v1", "api_key": "new-key"})
    with pytest.raises(RuntimeError, match="HTTP 302"):
        model_connection.fetch_models({})


def test_gemini_models_use_the_draft_key_in_a_header(monkeypatch):
    def get(url, **kwargs):
        assert url == "https://generativelanguage.googleapis.com/v1beta/models"
        assert kwargs["headers"] == {"x-goog-api-key": "gemini-credential"}
        return Response({"models": [
            {"name": "models/chat-model", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/embedding-model", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/broken", "supportedGenerationMethods": 1},
        ]})

    monkeypatch.setattr(requests, "get", get)
    assert model_connection.fetch_models({"provider": "gemini", "api_key": "gemini-credential"}) == {
        "models": ["chat-model"]
    }


def test_claude_and_ollama_use_native_model_endpoints(monkeypatch):
    def get(url, **kwargs):
        if url == "https://api.anthropic.com/v1/models":
            assert kwargs["headers"]["Authorization"] == "Bearer claude-credential"
            assert kwargs["headers"]["anthropic-version"]
            return Response({"data": [{"id": "claude-model"}]})
        assert url == "http://127.0.0.1:11434/api/tags"
        assert kwargs["headers"] == {}
        return Response({"models": [{"name": "ollama-model"}]})

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(config, "OLLAMA_URL", "http://127.0.0.1:11434")
    assert model_connection.fetch_models({"provider": "claude", "api_key": "claude-credential"}) == {
        "models": ["claude-model"]
    }
    assert model_connection.fetch_models({"provider": "ollama"}) == {"models": ["ollama-model"]}


def test_network_errors_never_echo_draft_credentials(saved_custom, monkeypatch):
    def unreachable(*args, **kwargs):
        raise requests.Timeout("sensitive-draft-key")

    monkeypatch.setattr(requests, "get", unreachable)
    with pytest.raises(RuntimeError, match="Could not read") as response:
        model_connection.fetch_models({"api_key": "sensitive-draft-key"})
    assert "sensitive-draft-key" not in str(response.value)

    error = RuntimeError("sensitive-draft-key")
    error.status_code = 401

    def bad_model(*args, **kwargs):
        raise error

    monkeypatch.setattr(ai, "ask_json", bad_model)
    with pytest.raises(RuntimeError, match="rejected") as response:
        model_connection.test_connection({"api_key": "sensitive-draft-key"})
    assert "sensitive-draft-key" not in str(response.value)
