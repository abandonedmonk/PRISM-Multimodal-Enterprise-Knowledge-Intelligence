from __future__ import annotations

from typing import List, Dict


def analyze_vision(prompt: str, image_paths: List[str] | None = None) -> dict:
	_ = image_paths
	return {
		"context_text": "",
		"sources": [],
		"note": "Vision analysis is not configured for this environment.",
	}
