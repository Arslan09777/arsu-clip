"""Test draft AI settings and discover model IDs without saving credentials."""
from __future__ import annotations

import json
from contextlib import contextmanager

import requests
from pydantic import BaseModel

from . import ai, config, settings
from .provider_presets import PRESETS, allows_model, require_free_model


class _ConnectionTest(BaseModel):
    ready: bool


@contextmanager
def _draft_settings(data: dict, *, needs_model: bool):
    with config.SETTINGS_LOCK:
        provider = settings._clean(data, "provider", 20) or config.ai_provider()
        if provider not in config.AI_PROVIDERS:
            raise ValueError("Choose an AI provider before connecting.")
        key = settings._clean(data, "api_key")
        model = settings._clean(data, "model", 300)
        no_key = data.get("no_key", False)
        if type(no_key) is not bool:
            raise ValueError("Choose a valid API key setting.")
        if key and no_key:
            raise ValueError("Enter a key or select No API key.")

        changes = {}
        if provider == "custom":
            saved_endpoint = config.ai_value("CUSTOM_AI_ENDPOINT")
            endpoint = settings._clean(data, "endpoint") or saved_endpoint
            if not endpoint:
                raise ValueError("Enter the custom API endpoint first.")
            normalized = settings.custom_chat_url(endpoint)
            if not key and not no_key:
                key = config.ai_value("CUSTOM_AI_API_KEY")
                if key:
                    try:
                        endpoint_matches = settings.custom_chat_url(saved_endpoint) == normalized
                    except ValueError:
                        endpoint_matches = False
                    if not endpoint_matches:
                        raise ValueError("Enter this endpoint's API key or select No API key.")
            if needs_model and not (model or config.ai_model(provider)):
                raise ValueError("Select or enter a model ID before testing the connection.")
            changes.update(CUSTOM_AI_ENDPOINT=endpoint, CUSTOM_AI_API_KEY=key)
            if model:
                changes["CUSTOM_AI_MODEL"] = model
        elif provider in PRESETS:
            if no_key or not (key or config.ai_key(provider)):
                raise ValueError(f"Enter your {PRESETS[provider].label} API key first.")
            if needs_model:
                require_free_model(provider, model or config.ai_model(provider))
            prefix = provider.upper()
            if key:
                changes[f"{prefix}_API_KEY"] = key
            if model:
                changes[f"{prefix}_MODEL"] = model
        elif provider in ("gemini", "claude"):
            key_name, fallback_key, model_name = (
                ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL") if provider == "gemini"
                else ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_MODEL")
            )
            if no_key or not (key or config.ai_value(key_name) or config.ai_value(fallback_key)):
                raise ValueError("Enter the provider's API key first.")
            if key:
                changes[key_name] = key
                changes[fallback_key] = ""
            if model:
                changes[model_name] = model
        elif needs_model and model and model != config.ai_model(provider):
            raise ValueError("Set the Ollama model in .env before testing it.")

        token = config.pin_ai_settings(overrides=changes)
    try:
        yield provider
    finally:
        config.reset_ai_settings(token)


def _safe_model_id(value: object, key: str = "") -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    if (not result or len(result) > 300 or not result.isprintable()
            or (key and key in result)):
        return None
    return result


def test_connection(data: dict) -> dict:
    """Run a minimal request on the draft model, then discard the draft."""
    with _draft_settings(data, needs_model=True) as provider:
        try:
            result = ai.ask_json(provider, _ConnectionTest, "Answer the connection test.", "Return ready=true.")
        except Exception as exc:
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if status in (401, 403):
                raise RuntimeError("The provider rejected the API credentials.") from None
            if status == 429:
                raise RuntimeError("The provider rate-limited the connection test.") from None
            raise RuntimeError("The model test failed. Check the API key, endpoint, and model ID.") from None
        if not result.ready:
            raise RuntimeError("The model did not confirm the connection.")
        return {"ready": True, "model": _safe_model_id(config.ai_model(provider), config.ai_key(provider))}


_MAX_MODEL_RESULTS = 300
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _request_models(url: str, headers: dict[str, str], params: dict | None = None) -> dict:
    try:
        with requests.get(url, headers=headers, params=params, timeout=(5, 15), stream=True, allow_redirects=False) as response:
            status = response.status_code
            if status in (401, 403):
                raise RuntimeError("The provider rejected the API credentials.")
            if status == 404:
                raise RuntimeError("This endpoint does not expose a model list. Enter a model ID manually.")
            if status == 429:
                raise RuntimeError("The provider rate-limited the model lookup.")
            if not 200 <= status < 300:
                raise RuntimeError(f"The provider could not list models (HTTP {status}).")
            body = bytearray()
            for chunk in response.iter_content(16384):
                body.extend(chunk)
                if len(body) > _MAX_RESPONSE_BYTES:
                    raise RuntimeError("The provider's model list was too large. Enter a model ID manually.")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError("Invalid models response")
        return payload
    except (requests.RequestException, ValueError):
        raise RuntimeError("Could not read the provider's model list. Check the connection or enter a model ID manually.") from None


def fetch_models(data: dict) -> dict:
    """List models available with the entered or saved connection."""
    with _draft_settings(data, needs_model=False) as provider:
        key = config.ai_key(provider)
        if provider in config.OPENAI_COMPATIBLE:
            url = settings.custom_chat_url(config.ai_endpoint(provider)).removesuffix("/chat/completions") + "/models"
            payload = _request_models(url, {"Authorization": f"Bearer {key}"} if key else {})
            rows, field = payload.get("data"), "id"
        elif provider == "gemini":
            payload = _request_models("https://generativelanguage.googleapis.com/v1beta/models",
                                      {"x-goog-api-key": key}, {"pageSize": _MAX_MODEL_RESULTS})
            rows, field = payload.get("models", []), "name"
        elif provider == "claude":
            payload = _request_models("https://api.anthropic.com/v1/models",
                                      {"Authorization": f"Bearer {key}", "anthropic-version": "2026-10-21"},
                                      {"limit": 100})
            rows, field = payload.get("data"), "id"
        else:
            payload = _request_models(f"{config.OLLAMA_URL}/api/tags", {})
            rows, field = payload.get("models"), "name"
        if not isinstance(rows, list):
            raise RuntimeError("The provider returned an invalid model list. Enter a model ID manually.")

        ids = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get(field)
            if provider == "gemini":
                methods = row.get("supportedGenerationMethods")
                if not isinstance(value, str) or not isinstance(methods, list) or "generateContent" not in methods:
                    continue
                value = value.removeprefix("models/")
            model_id = _safe_model_id(value, key)
            if model_id and (provider not in PRESETS or allows_model(provider, model_id)):
                ids[model_id] = None
            if len(ids) >= _MAX_MODEL_RESULTS:
                break
        models = list(ids)
        if provider == "openrouter":
            models = list(dict.fromkeys(("openrouter/free", *models)))[:_MAX_MODEL_RESULTS]
        return {"models": models}
