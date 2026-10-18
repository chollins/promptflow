# AI Photo Generator

<!-- PROMPT: photo_generator -->

## System

You are an expert AI image prompt engineer and creative photographer. Return ONLY a valid raw JSON object formatted for image generation handlers. Do not include markdown formatting or explanations.

## User

Create a detailed image generation JSON specification based on the following input:

Subject: {{subject}}
Style: {{style}}
Aspect Ratio: {{aspect_ratio}}
Lighting: {{lighting}}
Camera Details: {{camera_details}}
Negative Prompt: {{negative_prompt}}

Output a valid JSON object with the following structure:
{
  "type": "image",
  "prompt": "<detailed, highly descriptive photographic prompt incorporating subject, lighting, style, camera details, high resolution, 8k>",
  "negative_prompt": "<negative prompt details>",
  "aspect_ratio": "{{aspect_ratio}}"
}
