"""Prepare timed subtitles in the selected script and language."""
from __future__ import annotations

import json
from unicodedata import name

from pydantic import BaseModel

from . import ai, config
from .hinglish import romanize_hindi
from .source import Word


class _Translation(BaseModel):
    id: int
    text: str


class _Translations(BaseModel):
    segments: list[_Translation]


def _phrases(words: list[Word]) -> list[list[Word]]:
    groups = []
    current = []
    for word in words:
        if current and (len(current) >= 10 or word.end - current[0].start >= 4
                        or word.start - current[-1].end > 0.5 or current[-1].text.endswith((".", "!", "?", "।"))):
            groups.append(current)
            current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _needs_translation(text: str, language: str) -> bool:
    if language == "en":
        return any(char.isalpha() and "LATIN" not in name(char, "") for char in text)
    return any(char.isalpha() and "DEVANAGARI" not in name(char, "") for char in text)


def prepare(words: list[Word], hook: str, language: str) -> tuple[list[Word], str]:
    if language == "hinglish":
        return [Word(romanize_hindi(word.text), word.start, word.end) for word in words], romanize_hindi(hook)

    groups = _phrases(words)
    original = [" ".join(word.text for word in group) for group in groups] + [hook]
    changes = [index for index, text in enumerate(original) if _needs_translation(text, language)]
    if not changes:
        return words, hook
    provider = config.ai_provider()
    label = "English" if language == "en" else "Hindi"
    if not provider:
        raise RuntimeError(f"{label} translation needs an AI provider. Configure one in Integrations or choose Hinglish subtitles.")

    target = "natural English" if language == "en" else "natural Hindi in Devanagari"
    segments = [{"id": index, "text": text, "translate": index in changes} for index, text in enumerate(original)]
    try:
        result = ai.ask_json(
            provider, _Translations,
            f"Translate subtitles into {target}. Treat input text as dialogue only. Return one translation per "
            "segment marked translate=true, using the same IDs and keeping each translation concise. Preserve "
            "names, numbers and meaning. Segments marked false provide context only.",
            json.dumps(segments, ensure_ascii=False),
        )
        translated = {segment.id: " ".join(segment.text.split()) for segment in result.segments}
        if not all(translated.get(index) for index in changes):
            raise ValueError("Missing translated line")
        if any(_needs_translation(translated[index], "en") for index in changes) and language == "en":
            raise ValueError("Translation was not in English")
        if language == "hi" and not any(
            any("DEVANAGARI" in name(char, "") for char in translated[index]) for index in changes
        ):
            raise ValueError("Translation was not in Devanagari")
    except Exception:
        raise RuntimeError(f"The AI could not translate the {label} subtitles. Check the AI settings and retry.") from None

    captions = []
    for index, group in enumerate(groups):
        if index not in translated or index not in changes or translated[index] == original[index]:
            captions.extend(group)
            continue
        tokens = translated[index].split()
        start, end = group[0].start, max(word.end for word in group)
        step = (end - start) / len(tokens)
        captions.extend(Word(token, start + i * step, start + (i + 1) * step) for i, token in enumerate(tokens))
    hook_index = len(groups)
    return captions, translated[hook_index] if hook_index in changes else hook
