"""Paths and settings shared across the app."""
import os
import shutil
from contextvars import ContextVar
from pathlib import Path
from threading import RLock

from dotenv import load_dotenv

from .provider_presets import PRESETS, allows_model

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
# override: the app's .env wins over machine-wide variables (e.g. an older OLLAMA_MODELS)
load_dotenv(ENV_PATH, override=True, interpolate=False)
SETTINGS_LOCK = RLock()
AI_KEYS = (
    "AI_PROVIDER", "AI_ROUTING_PROVIDERS", "AI_ROUTING_WORKERS", "GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_MODEL",
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_MODEL",
    "CUSTOM_AI_API_KEY", "CUSTOM_AI_ENDPOINT", "CUSTOM_AI_MODEL",
    *(f"{provider.upper()}_{field}" for provider in PRESETS for field in ("API_KEY", "MODEL")),
)
_ai_context: ContextVar[dict[str, str] | None] = ContextVar("ai_settings", default=None)
AI_PROVIDERS = ("gemini", *PRESETS, "claude", "custom", "ollama")
OPENAI_COMPATIBLE = ("custom", *PRESETS)

DATA_DIR = Path(os.environ.get("CLIPSMITH_DATA_DIR") or ROOT / "data")
JOBS_DIR = DATA_DIR / "jobs"
MODELS_DIR = DATA_DIR / "models"
STATIC_DIR = ROOT / "static"

for _d in (JOBS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Keep model downloads inside the project instead of the user profile.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf"))

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5-5")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")

OUT_W, OUT_H = 1080, 1920
MAX_VIDEO_HOURS = 12
MIN_CLIP_COUNT = 10
MAX_CLIP_COUNT = 30
AUTO_CLIP_MIN_SECONDS = 15
AUTO_CLIP_MAX_SECONDS = 90


OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
# Explicit path from .env, else whatever "ollama" is on PATH (a normal install).
OLLAMA_EXE = os.environ.get("OLLAMA_EXE", "") or shutil.which("ollama") or ""
OLLAMA_MODELS_DIR = os.environ.get("OLLAMA_MODELS", "")
OLLAMA_CTX = int(os.environ.get("OLLAMA_CTX", "8192"))


def pin_ai_settings(overrides: dict[str, str] | None = None):
    with SETTINGS_LOCK:
        current = _ai_context.get()
        snapshot = (current.copy() if current is not None
                    else {key: os.environ[key] for key in AI_KEYS if key in os.environ})
        if overrides:
            snapshot.update(overrides)
        return _ai_context.set(snapshot)


def reset_ai_settings(token) -> None:
    _ai_context.reset(token)


def ai_value(key: str, default: str = "") -> str:
    snapshot = _ai_context.get()
    return (os.environ if snapshot is None else snapshot).get(key, default)


def ai_model(provider: str | None) -> str | None:
    if provider == "gemini":
        return ai_value("GEMINI_MODEL") or GEMINI_MODEL
    if provider == "claude":
        return ai_value("CLAUDE_MODEL") or CLAUDE_MODEL
    if provider == "custom":
        return ai_value("CUSTOM_AI_MODEL") or None
    if provider == "ollama":
        return OLLAMA_MODEL
    if provider in PRESETS:
        return ai_value(f"{provider.upper()}_MODEL") or PRESETS[provider].models[0]
    return None


def ai_key(provider: str) -> str:
    if provider == "gemini":
        return ai_value("GEMINI_API_KEY") or ai_value("GOOGLE_API_KEY")
    if provider == "claude":
        return ai_value("ANTHROPIC_API_KEY") or ai_value("ANTHROPIC_AUTH_TOKEN")
    if provider == "custom":
        return ai_value("CUSTOM_AI_API_KEY")
    return ai_value(f"{provider.upper()}_API_KEY") if provider in PRESETS else ""


def ai_endpoint(provider: str) -> str:
    if provider == "custom":
        return ai_value("CUSTOM_AI_ENDPOINT")
    return PRESETS[provider].endpoint if provider in PRESETS else ""


def available_ai_providers() -> tuple[str, ...]:
    configured = {
        "gemini": ai_value("GEMINI_API_KEY") or ai_value("GOOGLE_API_KEY"),
        "claude": ai_value("ANTHROPIC_API_KEY") or ai_value("ANTHROPIC_AUTH_TOKEN"),
        "custom": ai_value("CUSTOM_AI_ENDPOINT") and ai_model("custom"),
        "ollama": os.environ.get("OLLAMA_URL") or (OLLAMA_EXE and os.path.exists(OLLAMA_EXE)),
    }
    configured.update({name: bool(ai_key(name) and allows_model(name, ai_model(name))) for name in PRESETS})
    return tuple(name for name in AI_PROVIDERS if configured[name])


def ai_routing() -> tuple[tuple[str, ...], int]:
    """The explicitly enabled provider order and worker count."""
    providers = tuple(name.strip().lower() for name in ai_value("AI_ROUTING_PROVIDERS").split(",") if name.strip())
    if (not 2 <= len(providers) <= 3 or len(set(providers)) != len(providers)
            or any(name not in AI_PROVIDERS for name in providers)):
        return (), 2
    workers = 3 if ai_value("AI_ROUTING_WORKERS") == "3" and len(providers) == 3 else 2
    return providers, workers


def default_ai_provider() -> str | None:
    """Use the single-provider setting, or detect its legacy default."""
    forced = ai_value("AI_PROVIDER").strip().lower()
    if forced in AI_PROVIDERS:
        return forced
    if forced in ("none", "basic"):
        return None
    if ai_value("GEMINI_API_KEY") or ai_value("GOOGLE_API_KEY"):
        return "gemini"
    if ai_value("ANTHROPIC_API_KEY") or ai_value("ANTHROPIC_AUTH_TOKEN"):
        return "claude"
    if OLLAMA_EXE and os.path.exists(OLLAMA_EXE):
        return "ollama"
    return None


def ai_provider() -> str | None:
    providers, _ = ai_routing()
    return providers[0] if providers else default_ai_provider()


def missing_tools() -> list[str]:
    """Programs the pipeline shells out to."""
    return [t for t in ("ffmpeg", "ffprobe") if not shutil.which(t)]
