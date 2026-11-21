import json
import re
from threading import Barrier

import pytest
from dotenv import dotenv_values

from app import config, selector, settings, youtube


def _sentences():
    # A ten-minute transcript below even the small provider's word limit.
    return [selector.Sentence(i * 10, i * 10 + 9, "Why this moment helps everyone") for i in range(60)]


def _first_line(prompt):
    return int(re.search(r"^\[(\d+)\]", prompt, re.MULTILINE).group(1))


def _one_pick(schema, start):
    return schema(clips=[{
        "start_line": start,
        "end_line": start + 3,
        "title": "A useful moment",
        "hook": "Watch this",
        "score": 82,
        "reason": "Self-contained",
        "description": "A practical takeaway",
        "hashtags": ["video"],
    }])


@pytest.fixture
def isolated_settings(monkeypatch, tmp_path):
    path = tmp_path / ".env"
    path.write_text("AI_PROVIDER=gemini\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_PATH", path)
    monkeypatch.setattr(youtube, "TOKEN_PATH", tmp_path / "youtube.json")
    for key in config.AI_KEYS:
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("AI_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-private-fixture")
    return path


def test_routing_is_opt_in_and_uses_saved_connections(isolated_settings):
    with pytest.raises(ValueError, match="Save the connection"):
        settings.save_routing({"providers": ["gemini", "claude"], "workers": 2})
    status = settings.save_ai({"provider": "claude", "api_key": "claude-private-fixture", "activate": False})
    assert status["provider"] == "gemini"
    assert status["routing"]["providers"] == []
    assert {"gemini", "claude"} <= set(status["routing"]["available"])
    assert "claude-private-fixture" not in json.dumps(status)
    with pytest.raises(ValueError, match="different providers"):
        settings.save_routing({"providers": ["claude", "claude"], "workers": 2})
    with pytest.raises(ValueError, match="number of selected providers"):
        settings.save_routing({"providers": ["gemini", "claude"], "workers": 3})

    status = settings.save_routing({"providers": ["claude", "gemini"], "workers": 2})
    assert status["provider"] == "claude"
    assert status["routing"]["providers"] == ["claude", "gemini"]
    assert dotenv_values(isolated_settings)["AI_ROUTING_PROVIDERS"] == "claude,gemini"
    assert settings.save_routing({"providers": [], "workers": 2})["provider"] == "gemini"


def test_three_workers_split_a_short_transcript_and_keep_pinned_settings(monkeypatch, isolated_settings):
    monkeypatch.setenv("AI_ROUTING_PROVIDERS", "gemini,claude,custom")
    monkeypatch.setenv("AI_ROUTING_WORKERS", "3")
    ready = Barrier(3)
    calls = []
    snapshots = []
    progress = []

    def ask(provider, schema, system, prompt):
        start = _first_line(prompt)
        calls.append((start, provider))
        snapshots.append(config.ai_value("GEMINI_API_KEY"))
        ready.wait(timeout=5)
        return _one_pick(schema, start)

    monkeypatch.setattr(selector.ai, "ask_json", ask)
    token = config.pin_ai_settings()
    try:
        monkeypatch.setenv("GEMINI_API_KEY", "new-key-for-future-jobs")
        clips, provider, warning = selector.select(
            _sentences(), "Video", 3, 30, 60, lambda pct, msg: progress.append((pct, msg))
        )
    finally:
        config.reset_ai_settings(token)
    assert sorted(calls) == [(0, "gemini"), (20, "claude"), (40, "custom")]
    assert snapshots == ["gemini-private-fixture"] * 3
    assert [clip.start for clip in clips] == [0, 200, 400]
    assert provider == "gemini" and warning is None
    assert all(a[0] <= b[0] for a, b in zip(progress, progress[1:]))


def test_failed_part_switches_to_another_selected_provider(monkeypatch, isolated_settings):
    monkeypatch.setenv("AI_ROUTING_PROVIDERS", "gemini,claude")
    calls = []

    def ask(provider, schema, system, prompt):
        start = _first_line(prompt)
        calls.append((start, provider))
        if provider == "gemini":
            raise RuntimeError("Provider timed out")
        return _one_pick(schema, start)

    monkeypatch.setattr(selector.ai, "ask_json", ask)
    clips, provider, warning = selector.select(_sentences(), "Video", 2, 30, 60)
    assert sorted(calls) == [(0, "claude"), (0, "gemini"), (30, "claude")]
    assert [clip.start for clip in clips] == [0, 300]
    assert provider == "claude" and warning is None
