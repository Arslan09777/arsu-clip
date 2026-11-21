"""Save local integration settings without returning credentials to the browser."""
from __future__ import annotations

import os
import shutil
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit, urlunsplit

from dotenv import set_key

from . import config, youtube
from .provider_presets import PRESETS, require_free_model


def custom_chat_url(endpoint: str) -> str:
    try:
        url = urlsplit(endpoint)
        hostname = url.hostname
        _ = url.port
    except ValueError:
        raise ValueError("Enter a valid API endpoint.") from None
    if (url.scheme not in ("http", "https") or not hostname or url.username is not None
            or url.query or url.fragment or any(char.isspace() for char in endpoint)):
        raise ValueError("Use an HTTP(S) endpoint without a password, query, or fragment.")
    if url.scheme == "http" and hostname not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Use HTTPS for a remote API endpoint.")
    path = url.path.rstrip("/")
    if not path.endswith("/chat/completions"):
        path = (path or "/v1") + "/chat/completions"
    return urlunsplit((url.scheme, url.netloc, path, "", ""))


def status() -> dict:
    with config.SETTINGS_LOCK:
        providers, workers = config.ai_routing()
        return {
            "provider": config.ai_provider() or "none",
            "default_provider": config.default_ai_provider() or "none",
            "connection_tools": True,
            "models": {name: config.ai_model(name) for name in config.AI_PROVIDERS},
            "custom_endpoint": config.ai_value("CUSTOM_AI_ENDPOINT"),
            "keys": {name: bool(config.ai_key(name)) for name in config.AI_PROVIDERS if name != "ollama"},
            "presets": {
                name: {"label": preset.label, "endpoint": preset.endpoint, "models": list(preset.models),
                       "access": preset.access, "docs": preset.docs}
                for name, preset in PRESETS.items()
            },
            "routing": {"providers": list(providers), "workers": workers,
                        "available": list(config.available_ai_providers())},
            "youtube": youtube.status(),
        }


def _clean(data: dict, name: str, limit: int = 2048) -> str:
    value = data.get(name, "")
    if not isinstance(value, str) or len(value) > limit or any(ord(char) < 32 for char in value):
        raise ValueError("Enter valid, single-line settings.")
    return value.strip()


def _write(changes: dict[str, str]) -> None:
    path = config.ENV_PATH
    with NamedTemporaryFile(dir=path.parent, prefix=".env-", delete=False) as temporary:
        staged = Path(temporary.name)
    try:
        if path.exists():
            shutil.copyfile(path, staged)
            shutil.copymode(path, staged)
        for key, value in changes.items():
            set_key(staged, key, value, quote_mode="always")
        staged.replace(path)
        os.environ.update(changes)
    finally:
        staged.unlink(missing_ok=True)


def save_ai(data: dict) -> dict:
    provider = _clean(data, "provider", 20)
    if provider not in (*config.AI_PROVIDERS, "none"):
        raise ValueError("Select an AI provider.")
    activate = data.get("activate", True)
    if type(activate) is not bool:
        raise ValueError("Enter a valid provider activation setting.")
    with config.SETTINGS_LOCK:
        changes = {"AI_PROVIDER": provider} if activate else {}
        if not activate and not config.ai_value("AI_PROVIDER"):
            # Preserve the current selection when saving a key for future use.
            changes["AI_PROVIDER"] = config.default_ai_provider() or "none"
        if provider in PRESETS:
            api_key, model = _clean(data, "api_key"), _clean(data, "model", 300)
            model = model or config.ai_model(provider)
            require_free_model(provider, model)
            if not (api_key or config.ai_key(provider)):
                raise ValueError(f"Enter your {PRESETS[provider].label} API key.")
            prefix = provider.upper()
            if api_key:
                changes[f"{prefix}_API_KEY"] = api_key
            changes[f"{prefix}_MODEL"] = model
        elif provider in ("gemini", "claude", "custom"):
            key_name, model_name = {
                "gemini": ("GEMINI_API_KEY", "GEMINI_MODEL"),
                "claude": ("ANTHROPIC_API_KEY", "CLAUDE_MODEL"),
                "custom": ("CUSTOM_AI_API_KEY", "CUSTOM_AI_MODEL"),
            }[provider]
            api_key, model = _clean(data, "api_key"), _clean(data, "model", 300)
            if api_key:
                changes[key_name] = api_key
            if model:
                changes[model_name] = model
            if provider == "custom":
                endpoint = _clean(data, "endpoint") or config.ai_value("CUSTOM_AI_ENDPOINT")
                if not endpoint or not (model or config.ai_model("custom")):
                    raise ValueError("Custom AI needs an endpoint and a model ID.")
                normalized = custom_chat_url(endpoint)
                old_endpoint = config.ai_value("CUSTOM_AI_ENDPOINT")
                old_key = config.ai_value("CUSTOM_AI_API_KEY")
                no_key = data.get("no_key") is True
                if api_key and no_key:
                    raise ValueError("Enter a key or select No API key.")
                if old_key and not (api_key or no_key) and (
                    not old_endpoint or custom_chat_url(old_endpoint) != normalized
                ):
                    raise ValueError("Enter the new endpoint's key or select No API key.")
                if no_key:
                    changes[key_name] = ""
                changes["CUSTOM_AI_ENDPOINT"] = endpoint
            elif not api_key and not (
                config.ai_value(key_name)
                or config.ai_value("GOOGLE_API_KEY" if provider == "gemini" else "ANTHROPIC_AUTH_TOKEN")
            ):
                raise ValueError("Enter the API key for this provider.")
        if changes:
            _write(changes)
        return status()


def save_routing(data: dict) -> dict:
    providers = data.get("providers")
    workers = data.get("workers", 2)
    if (not isinstance(providers, list) or len(providers) not in (0, 2, 3)
            or any(not isinstance(provider, str) or provider not in config.AI_PROVIDERS for provider in providers)
            or len(set(providers)) != len(providers)):
        raise ValueError("Choose two or three different providers, or turn parallel ranking off.")
    if type(workers) is not int or workers not in (2, 3) or (providers and workers > len(providers)):
        raise ValueError("Choose two or three workers, up to the number of selected providers.")
    with config.SETTINGS_LOCK:
        if any(provider not in config.available_ai_providers() for provider in providers):
            raise ValueError("Save the connection details for every selected provider first.")
        if "custom" in providers:
            custom_chat_url(config.ai_value("CUSTOM_AI_ENDPOINT"))
        _write({"AI_ROUTING_PROVIDERS": ",".join(providers), "AI_ROUTING_WORKERS": str(workers)})
        return status()


def save_youtube(data: dict) -> dict:
    with config.SETTINGS_LOCK:
        old_id = os.environ.get("YOUTUBE_CLIENT_ID", "")
        old_secret = os.environ.get("YOUTUBE_CLIENT_SECRET", "")
        client_id = _clean(data, "client_id") or old_id
        client_secret = _clean(data, "client_secret") or old_secret
        if not client_id or not client_secret:
            raise ValueError("Enter both the YouTube OAuth client ID and secret.")
        if youtube.status()["connected"] and (client_id, client_secret) != (old_id, old_secret):
            raise ValueError("Disconnect the current YouTube channel before replacing its OAuth credentials.")
        _write({"YOUTUBE_CLIENT_ID": client_id, "YOUTUBE_CLIENT_SECRET": client_secret})
        return status()
