import json

import pytest
from dotenv import dotenv_values
from pydantic import BaseModel

from app import ai, config, model_connection, settings


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("AI_PROVIDER=gemini\nGEMINI_API_KEY=gemini-fixture\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", env)
    for key in config.AI_KEYS:
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-fixture")
    return env


def test_connections_save_separately_and_can_join_routing(isolated_env):
    result = settings.save_ai({"provider": "openrouter", "api_key": "router-fixture", "activate": False})
    result = settings.save_ai({"provider": "groq", "api_key": "groq-fixture", "activate": False})
    result = settings.save_ai({"provider": "nvidia", "api_key": "nvidia-fixture", "activate": False})
    assert result["default_provider"] == "gemini"
    assert set(result["routing"]["available"]) >= {"gemini", "openrouter", "groq", "nvidia"}
    assert "fixture" not in json.dumps(result)
    saved = dotenv_values(isolated_env)
    assert saved["OPENROUTER_API_KEY"] == "router-fixture"
    assert saved["GROQ_API_KEY"] == "groq-fixture"
    assert saved["NVIDIA_API_KEY"] == "nvidia-fixture"
    routed = settings.save_routing({"providers": ["openrouter", "nvidia", "groq"], "workers": 3})
    assert routed["routing"]["workers"] == 3
    assert routed["routing"]["providers"] == ["openrouter", "nvidia", "groq"]


@pytest.mark.parametrize(("provider", "model"), [
    ("openrouter", "openai/gpt-4o"), ("nvidia", "unlisted-model"), ("groq", "whisper-large-v3"),
])
def test_presets_reject_models_outside_the_free_starters(isolated_env, provider, model):
    before = isolated_env.read_bytes()
    with pytest.raises(ValueError, match="Choose a free model"):
        settings.save_ai({"provider": provider, "api_key": "test", "model": model})
    with pytest.raises(ValueError, match="Choose a free model"):
        model_connection.test_connection({"provider": provider, "api_key": "test", "model": model})
    assert isolated_env.read_bytes() == before


class Reply(BaseModel):
    ready: bool


def test_preset_chat_uses_only_its_fixed_endpoint_and_key(isolated_env, monkeypatch):
    settings.save_ai({"provider": "nvidia", "api_key": "nvidia-fixture", "activate": False})
    monkeypatch.setenv("CUSTOM_AI_ENDPOINT", "https://unrelated.example/v1")
    monkeypatch.setenv("CUSTOM_AI_API_KEY", "custom-fixture")

    class Response:
        ok = True
        is_redirect = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def json(self):
            return {"choices": [{"message": {"content": '{"ready":true}'}}]}

    def post(url, *, json, headers, allow_redirects, timeout):
        assert url == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert headers == {"Authorization": "Bearer nvidia-fixture"}
        assert json["model"] == "nvidia/nemotron-3.5-lightning-30b-a3b"
        assert allow_redirects is False
        return Response()

    monkeypatch.setattr("requests.post", post)
    assert ai.ask_json("nvidia", Reply, "Answer", "Ready?").ready


def test_openrouter_model_discovery_filters_paid_ids_and_ignores_a_custom_endpoint(isolated_env, monkeypatch):
    def models(url, headers, params=None):
        assert url == "https://openrouter.ai/api/v1/models"
        assert headers == {"Authorization": "Bearer router-fixture"}
        return {"data": [{"id": "example/allowed:free"}, {"id": "example/billed"}]}

    monkeypatch.setattr(model_connection, "_request_models", models)
    result = model_connection.fetch_models({
        "provider": "openrouter", "endpoint": "https://unrelated.example/api/v1", "api_key": "router-fixture",
    })
    assert result == {"models": ["openrouter/free", "example/allowed:free"]}
    assert "OPENROUTER_API_KEY" not in dotenv_values(isolated_env)
