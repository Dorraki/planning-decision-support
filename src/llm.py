"""Shared OpenRouter client, model settings and image preparation."""

# split setup: luna for facts, sol for assessment.
import base64
import io
import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image

DEFAULT_MODEL = "openai/gpt-6.1-sol"
# Fact extraction defaults to a cheaper model than assessment (the "split" setup).
DEFAULT_FACTS_MODEL = "openai/gpt-6-luna"

# Drawings are sent as JPEGs no larger than this on their longest side, which
# keeps written dimensions legible while cutting multi-megabyte PNGs to ~300 KB.
MAX_IMAGE_SIDE_PX = 2000


def model_name() -> str:
    """Return the model to use, overridable with ``PLANNING_MODEL``."""
    load_dotenv()
    return os.environ.get("PLANNING_MODEL", DEFAULT_MODEL)


def facts_model_name() -> str:
    """Return the fact-extraction model, overridable with ``PLANNING_FACTS_MODEL``."""
    load_dotenv()
    return os.environ.get("PLANNING_FACTS_MODEL") or DEFAULT_FACTS_MODEL


def model_slug(name: str | None = None) -> str:
    """Return a model name (default: the current model) as a directory name.

    Only "/" is replaced; model names with other special characters would
    need more handling.
    """
    return (name or model_name()).replace("/", "-")


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    """Return an OpenAI-compatible client pointed at OpenRouter.

    Raises:
        RuntimeError: If the OpenRouter settings are missing from ``.env``.
    """
    load_dotenv()
    api_key = os.environ.get("OPENROUTER_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError(
            "OPENROUTER_API_KEY / OPENROUTER_BASE_URL missing; "
            "copy .env.example to .env and add your key"
        )
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def image_data_url(image_bytes: bytes) -> str:
    """Return a downscaled JPEG data URL for an image."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.thumbnail((MAX_IMAGE_SIDE_PX, MAX_IMAGE_SIDE_PX))
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
