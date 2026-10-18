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
        raw_key = api_key or os.getenv("LEONARDO_API_KEY", "")
        self.api_key = raw_key.strip().strip('"').strip("'")

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

        leonardo_payload: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_images": num_images,
        }
        if negative_prompt and str(negative_prompt).strip():
            leonardo_payload["negative_prompt"] = str(negative_prompt).strip()

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
            import urllib.error
            import time

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            }

            req = urllib.request.Request(
                "https://cloud.leonardo.ai/api/rest/v1/generations",
                data=json.dumps(leonardo_payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                generation_id = resp_data.get("sdGenerationJob", {}).get("generationId", "unknown")

            image_url = ""
            status = "queued"
            # Poll for generation completion up to ~20 seconds
            if generation_id and generation_id != "unknown":
                for _ in range(10):
                    time.sleep(2)
                    try:
                        poll_req = urllib.request.Request(
                            f"https://cloud.leonardo.ai/api/rest/v1/generations/{generation_id}",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Accept": "application/json",
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                            },
                        )
                        with urllib.request.urlopen(poll_req) as poll_resp:
                            poll_data = json.loads(poll_resp.read().decode("utf-8"))
                            gen_info = poll_data.get("generations_by_pk", {})
                            status = (gen_info.get("status") or status).lower()
                            images = gen_info.get("generated_images", [])
                            if images and images[0].get("url"):
                                image_url = images[0]["url"]
                                status = "completed"
                                break
                    except Exception as poll_err:
                        logger.warning("Polling Leonardo generation failed: %s", poll_err)

            artifacts: list[str] = []
            if output_dir:
                img_dir = output_dir / "images"
                img_dir.mkdir(parents=True, exist_ok=True)
                art_file = img_dir / f"{save_as}.json"
                art_file.write_text(
                    json.dumps(
                        {
                            "generation_id": generation_id,
                            "prompt": prompt,
                            "aspect_ratio": image_spec.get("aspect_ratio") or image_spec.get("ar") or "16:9",
                            "dimensions": f"{width}x{height}",
                            "url": image_url,
                            "status": status,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                artifacts.append(str(art_file))

                if image_url and image_url.startswith("http") and "mock.leonardo.ai" not in image_url:
                    try:
                        ext = ".jpg" if ".jpg" in image_url.lower() else ".png"
                        local_img_file = img_dir / f"{save_as}{ext}"
                        img_req = urllib.request.Request(
                            image_url,
                            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                        )
                        with urllib.request.urlopen(img_req) as img_resp:
                            local_img_file.write_bytes(img_resp.read())
                        artifacts.append(str(local_img_file))
                        logger.info("Image downloaded and saved locally to: %s", local_img_file)
                    except Exception as dl_err:
                        logger.warning("Failed downloading generated image locally: %s", dl_err)

            return {
                "generation_id": generation_id,
                "status": status,
                "request": leonardo_payload,
                "url": image_url,
                "artifacts": artifacts,
            }
        except urllib.error.HTTPError as exc:
            err_msg = str(exc.reason)
            try:
                body = exc.read().decode("utf-8")
                err_data = json.loads(body)
                err_msg = err_data.get("error") or err_data.get("message") or body
            except Exception:
                pass
            raise LeonardoAdapterError(f"Leonardo API returned HTTP {exc.code}: {err_msg}") from exc
        except Exception as exc:
            raise LeonardoAdapterError(f"Leonardo API execution failed: {exc}") from exc
