from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ASPECT_RATIO_DIMENSIONS = {
    "1:1": (1024, 1024),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "4:3": (1024, 768),
    "3:4": (768, 1024),
    "3:2": (1152, 768),
    "2:3": (768, 1152),
}


class LeonardoAdapterError(Exception):
    """Raised when Leonardo.AI API request validation or execution fails."""
    pass


class LeonardoAdapter:
    """Adapter for interacting with Leonardo.AI Image Generation API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("LEONARDO_API_KEY")

    def resolve_dimensions(self, image_spec: dict[str, Any]) -> tuple[int, int]:
        ar = image_spec.get("aspect_ratio") or image_spec.get("ar") or "1:1"
        if ar in ASPECT_RATIO_DIMENSIONS:
            return ASPECT_RATIO_DIMENSIONS[ar]
        width = int(image_spec.get("width", 1024))
        height = int(image_spec.get("height", 1024))
        return width, height

    def generate_image(
        self,
        image_spec: dict[str, Any],
        output_dir: Path | None = None,
        save_as: str = "image",
    ) -> dict[str, Any]:
        """
        Processes standard image JSON specification and invokes Leonardo API (or mock generator).
        """
        prompt = image_spec.get("prompt")
        if not prompt or not str(prompt).strip():
            raise LeonardoAdapterError("Missing required 'prompt' in image generation JSON.")

        negative_prompt = image_spec.get("negative_prompt", "")
        num_images = int(image_spec.get("num_images", 1))
        width, height = self.resolve_dimensions(image_spec)

        leonardo_payload = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_images": num_images,
        }

        # Handle missing API key cleanly in test/dev mode
        if not self.api_key or self.api_key == "mock":
            logger.info("Using mock Leonardo.AI adapter execution.")
            mock_url = f"https://mock.leonardo.ai/generations/mock_{save_as}.png"
            artifacts: list[str] = []
            if output_dir:
                img_dir = output_dir / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                mock_file = img_dir / f"{save_as}.json"
                mock_file.write_text(
                    json.dumps(
                        {
                            "generation_id": f"gen_mock_{save_as}",
                            "prompt": prompt,
                            "aspect_ratio": image_spec.get("aspect_ratio") or image_spec.get("ar") or "1:1",
                            "dimensions": f"{width}x{height}",
                            "url": mock_url,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                artifacts.append(str(mock_file))

            return {
                "generation_id": f"gen_mock_{save_as}",
                "status": "completed",
                "request": leonardo_payload,
                "url": mock_url,
                "artifacts": artifacts,
            }

        # Real API request execution using urllib/requests if key is set
        try:
            import urllib.request
            req = urllib.request.Request(
                "https://cloud.leonardo.ai/api/rest/v1/generations",
                data=json.dumps(leonardo_payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                generation_id = resp_data.get("sdGenerationJob", {}).get("generationId", "unknown")
                return {
                    "generation_id": generation_id,
                    "status": "queued",
                    "request": leonardo_payload,
                    "artifacts": [],
                }
        except Exception as exc:
            raise LeonardoAdapterError(f"Leonardo API execution failed: {exc}") from exc
