"""Structured responses from Arsuclip's configured language model."""
from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel

from . import config
from .provider_presets import PRESETS, require_free_model
from .settings import custom_chat_url


Response = TypeVar("Response", bound=BaseModel)


def ask_json(provider: str, schema: type[Response], system: str, prompt: str) -> Response:
    if provider == "gemini":
        from google import genai

        key = config.ai_value("GEMINI_API_KEY") or config.ai_value("GOOGLE_API_KEY")
        with genai.Client(api_key=key or None) as client:
            result = client.interactions.create(
                model=config.ai_model(provider),
                input=f"{system}\n\n{prompt}",
                response_format={"type": "text", "mime_type": "application/json", "schema": schema.model_json_schema()},
            )
        return schema.model_validate_json(result.output_text)

    if provider == "claude":
        import anthropic

        key = config.ai_value("ANTHROPIC_API_KEY")
        token = config.ai_value("ANTHROPIC_AUTH_TOKEN")
        with anthropic.Anthropic(api_key=key or None, auth_token=token or None) as client:
            result = client.messages.parse(
                model=config.ai_model(provider), max_tokens=16000, system=system,
                messages=[{"role": "user", "content": prompt}], output_format=schema,
            )
        if result.stop_reason == "refusal" or result.parsed_output is None:
            raise RuntimeError("The model returned no usable result.")
        return result.parsed_output

    if provider == "ollama":
        from . import local_llm

        return schema.model_validate_json(local_llm.chat_json(system, prompt, schema.model_json_schema()))

    if provider in config.OPENAI_COMPATIBLE:
        import requests

        label = PRESETS[provider].label if provider in PRESETS else "Custom AI"
        endpoint = custom_chat_url(config.ai_endpoint(provider))
        key = config.ai_key(provider)
        model = config.ai_model(provider)
        if not model:
            raise RuntimeError(f"{label} needs a model ID.")
        if provider in PRESETS:
            require_free_model(provider, model)
            if not key:
                raise RuntimeError(f"{label} needs an API key.")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        system = f"{system}\nReturn only JSON matching this schema: {json.dumps(schema.model_json_schema())}"
        payload = {"model": model, "stream": False,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        try:
            with requests.post(endpoint, json=payload, headers=headers, allow_redirects=False, timeout=(10, 180)) as result:
                if not result.ok or result.is_redirect:
                    raise RuntimeError(f"{label} returned HTTP {result.status_code}.")
                raw = result.json()["choices"][0]["message"]["content"].strip()
            if raw.startswith("```") and raw.endswith("```"):
                raw = raw.split("\n", 1)[-1].removesuffix("```").strip()
            return schema.model_validate_json(raw)
        except requests.Timeout:
            raise RuntimeError(f"{label} timed out.") from None
        except (requests.RequestException, ValueError, KeyError, IndexError, AttributeError, TypeError):
            raise RuntimeError(f"{label} returned an unreadable response or could not be reached.") from None

    raise RuntimeError("Choose an AI provider in Settings first.")
