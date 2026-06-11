"""
Wrapper for nvidia/LocateAnything-3B visual grounding model.

Provides lazy model loading and inference for use within GlassCV.
The model is loaded once into VRAM on first call and reused across requests.

Dependencies (see scripts/LOCATE_ANYTHING_SETUP.md):
    transformers>=4.51.0,<5.0.0
    accelerate>=0.34.0
    Pillow>=10.0.0
    torch (CUDA recommended, CPU fallback available — ~90s first inference on CPU)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID    = "nvidia/LocateAnything-3B"
COORD_SPACE = 1000  # Model outputs coordinates normalized to a 0–1000 space
DTYPE       = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Lazy model state (module-level singletons)
# ---------------------------------------------------------------------------

_MODEL:     object | None = None
_PROCESSOR: object | None = None


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """A single grounded detection with pixel-space bounding box."""

    label: str
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_locate_anything_model() -> tuple:
    """
    Load and return (processor, model), caching after the first call.

    The model is ~7 GB on disk and takes ~16s to load into VRAM on first call.
    Subsequent calls return the cached instance immediately.

    Raises:
        torch.cuda.OutOfMemoryError: If VRAM is insufficient to load the model.
        ImportError: If transformers or accelerate are not installed.
    """
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        return _PROCESSOR, _MODEL

    from transformers import AutoModel, AutoProcessor

    _PROCESSOR = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    _MODEL = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    _MODEL.eval()
    return _PROCESSOR, _MODEL


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    processor,
    model,
    frame_bgr: np.ndarray,
    prompt: str,
) -> list[Detection]:
    """
    Run visual grounding inference on a BGR numpy frame.

    Args:
        processor:  HuggingFace processor for LocateAnything-3B.
        model:      Loaded LocateAnything-3B model.
        frame_bgr:  Input frame as a BGR numpy array (GlassCV pipeline format).
        prompt:     Natural-language description of the element to locate.

    Returns:
        List of Detection instances with pixel-space bounding-box coordinates.

    Note:
        Inference takes ~72s on an RTX 4060 8 GB (fp16).
        The model returns all visually matching elements, not just the best match.
    """
    import cv2
    from PIL import Image

    # BGR numpy → RGB PIL (required by the model processor)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image     = Image.fromarray(frame_rgb)
    img_w, img_h = image.size

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor(
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=[image],
        return_tensors="pt",
    ).to(DEVICE, dtype=DTYPE)

    with torch.inference_mode():
        raw_output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            tokenizer=processor.tokenizer,
        )

    return parse_detections(raw_output, img_w, img_h)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_detections(raw: str, img_w: int, img_h: int) -> list[Detection]:
    """
    Parse model output string into Detection instances.

    Model output format:
        <ref>label</ref><box><x1><y1><x2><y2></box>

    Coordinates are in a normalized 0–1000 space and are scaled to pixel space
    using the actual image dimensions.
    """
    pattern = re.compile(
        r"<ref>(.*?)</ref>\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    )
    scale_x = img_w / COORD_SPACE
    scale_y = img_h / COORD_SPACE

    detections: list[Detection] = []
    for match in pattern.finditer(raw):
        label, x1, y1, x2, y2 = match.groups()
        detections.append(Detection(
            label=label,
            x1=round(int(x1) * scale_x),
            y1=round(int(y1) * scale_y),
            x2=round(int(x2) * scale_x),
            y2=round(int(y2) * scale_y),
        ))
    return detections
