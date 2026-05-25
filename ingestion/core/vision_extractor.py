import os
import base64
import sys
import time
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import config
except ImportError:
    config = None


TABLE_DESCRIPTION_PROMPT = (
    "You are a financial analyst reviewing a table from an SEC filing. "
    "Describe this table in detail. Preserve all numbers, labels, column headers, "
    "row labels, and structure. Note any trends, totals, year-over-year changes, "
    "or notable patterns visible in the data. Be precise and factual.\n\n"
    "Do not add any information not present in the table."
)

DEFAULT_TIMEOUT = 60

_GROQ_RPM = 30
_MIN_INTERVAL = 60.0 / _GROQ_RPM
_last_call_time = 0.0


def _rate_limit_wait():
    global _last_call_time
    now = time.monotonic()
    elapsed = now - _last_call_time
    if elapsed < _MIN_INTERVAL:
        time.sleep(_MIN_INTERVAL - elapsed)
    _last_call_time = time.monotonic()


def _load_vision_config() -> tuple[str, str, str]:
    base_url = (
        (getattr(config, "VISION_BASE_URL", "") if config else "")
        or os.getenv("VISION_BASE_URL", "")
        or (getattr(config, "GROQ_BASE_URL", "") if config else "")
        or os.getenv("GROQ_BASE_URL", "")
    )
    if not base_url:
        base_url = "http://localhost:8001/v1"

    api_key = (
        os.getenv("GROQ_API_KEY", "")
        or (getattr(config, "GROQ_API_KEY", "") if config else "")
        or os.getenv("LLM_API_KEY", "")
        or (getattr(config, "LLM_API_KEY", "") if config else "")
        or "not-needed"
    )
    model = (
        os.getenv("VISION_MODEL", "")
        or (getattr(config, "VISION_MODEL", "") if config else "")
        or "meta-llama/Llama-3.2-11B-Vision-Instruct"
    )
    return base_url, api_key, model


def describe_table_image(image_path: str | Path) -> str:
    image_path = Path(image_path)
    if not image_path.exists():
        return ""

    base_url, api_key, model = _load_vision_config()

    _rate_limit_wait()

    try:
        with open(image_path, "rb") as f:
            img_bytes = f.read()

        encoded = base64.b64encode(img_bytes).decode("utf-8")

        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.post(
                f"{base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": TABLE_DESCRIPTION_PROMPT,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{encoded}",
                                    },
                                },
                            ],
                        }
                    ],
                    "temperature": 0.0,
                    "max_tokens": 512,
                },
            )

            if resp.status_code == 200:
                result = resp.json()
                return result["choices"][0]["message"]["content"].strip()
            else:
                print(f"  VLM error: {resp.status_code} - {resp.text[:200]}")
                return ""

    except Exception as e:
        print(f"  VLM description failed: {e}")
        return ""


def describe_table_images_batch(
    image_paths: list[str | Path],
    batch_size: int = 5,
) -> list[str]:
    """Describe multiple table images with Groq rate limiting.

    Args:
        image_paths: List of paths to PNG table images
        batch_size: Number of images per batch (default 5, respects 30/min Groq limit)

    Returns:
        List of description strings (empty string for failures)
    """
    descriptions = []
    for i, path in enumerate(image_paths):
        desc = describe_table_image(path)
        descriptions.append(desc)
        if (i + 1) % batch_size == 0 and i + 1 < len(image_paths):
            print(f"    VLM: {i + 1}/{len(image_paths)} images processed")
    return descriptions


def extract_vision_features(image_path: str | Path) -> dict:
    image_path = Path(image_path)
    if not image_path.exists():
        return {"text": "", "description": "", "tables": [], "confidence": 0.0}

    description = describe_table_image(image_path)

    return {
        "text": "",
        "description": description,
        "tables": [],
        "confidence": 0.9 if description else 0.0,
    }
