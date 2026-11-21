"""Hosted starter connections and the chat models offered by each preset."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderPreset:
    label: str
    endpoint: str
    models: tuple[str, ...]
    access: str
    docs: str


PRESETS = {
    "openrouter": ProviderPreset(
        "OpenRouter", "https://openrouter.ai/api/v1", ("openrouter/free",),
        "Free models only", "https://openrouter.ai/openrouter/free",
    ),
    "nvidia": ProviderPreset(
        "NVIDIA NIM", "https://integrate.api.nvidia.com/v1",
        ("nvidia/nemotron-3.5-lightning-30b-a3b", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"),
        "Free prototype endpoint", "https://build.nvidia.com/nvidia/nemotron-3.5-lightning-30b-a3b/build",
    ),
    "groq": ProviderPreset(
        "Groq", "https://api.groq.com/openai/v1", ("openai/gpt-oss-20b", "openai/gpt-oss-120b"),
        "Free plan limits apply", "https://console.groq.com/docs/rate-limits",
    ),
}


def allows_model(provider: str, model: str) -> bool:
    if provider == "openrouter":
        return model == "openrouter/free" or model.endswith(":free")
    return model in PRESETS[provider].models


def require_free_model(provider: str, model: str) -> None:
    if not allows_model(provider, model):
        label = PRESETS[provider].label
        raise ValueError(f"Choose a free model listed by the {label} preset.")
