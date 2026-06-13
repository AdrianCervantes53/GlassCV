"""
Wrapper for nvidia/LocateAnything-3B visual grounding model.

Provides lazy model loading and inference for use within GlassCV.
The model is loaded once into VRAM on first call and reused across requests.

Dependencies (see scripts/LOCATE_ANYTHING_SETUP.md):
    transformers>=4.51.0,<5.0.0
    accelerate>=0.34.0
    Pillow>=10.0.0
    torch (CUDA recommended, CPU fallback available)

Output format (Parallel Box Decoding):
    <ref>label</ref><box><avg></box><box><inst1></box><box><inst2></box>
    When multiple boxes follow one <ref>, the FIRST is a PBD aggregate box
    (average over all instances) and the rest are individual detections.
    parse_detections() handles this by skipping the aggregate when N > 1.

Key implementation notes:
    - Use bfloat16 (not float16) — more numerically stable for this model.
    - Pass inputs individually to model.generate(), NOT via **inputs.to(dtype).
      Doing inputs.to(dtype=float16) corrupts image_grid_hws (int64 → float16)
      which destroys the image patch geometry and produces <0><0><998><998> boxes.
    - image_grid_hws must be passed as an int tensor; never cast it to float.
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
COORD_SPACE = 1000

# bfloat16 preferred over float16 — better numerical stability for this model
DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    Raises:
        torch.cuda.OutOfMemoryError: If VRAM is insufficient to load the model.
        ImportError: If transformers or accelerate are not installed.
    """
    global _MODEL, _PROCESSOR

    if _MODEL is not None and _PROCESSOR is not None:
        return _PROCESSOR, _MODEL

    from transformers import AutoModel, AutoProcessor, AutoTokenizer

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
    """
    import cv2
    from PIL import Image

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

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        images=[image],
        text=text,
        return_tensors="pt",
    )

    # Pass each tensor individually — NEVER do inputs.to(dtype=float) on the
    # whole dict. image_grid_hws is int64 and must stay int64; casting it to
    # float16 corrupts the image patch geometry and produces degenerate boxes.
    with torch.inference_mode():
        raw_output = model.generate(
            input_ids=inputs["input_ids"].to(DEVICE),
            attention_mask=inputs["attention_mask"].to(DEVICE),
            pixel_values=inputs["pixel_values"].to(DEVICE, dtype=DTYPE),
            image_grid_hws=torch.tensor(inputs["image_grid_hws"]).to(DEVICE),
            tokenizer=processor.tokenizer,
            use_cache=True,
            max_new_tokens=64,
        )

    return parse_detections(raw_output, img_w, img_h)


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_detections(raw: str, img_w: int, img_h: int) -> list[Detection]:
    """
    Parse model output into Detection instances.

    The model uses Parallel Box Decoding (PBD): when multiple instances of an
    object are found, the output is:
        <ref>label</ref><box><avg></box><box><inst1></box><box><inst2></box>

    The first <box> is a PBD aggregate (average of all instances) and is
    skipped when N > 1. When only one box is present it is a genuine single
    detection and is kept as-is.
    """
    ref_pattern = re.compile(
        r"<ref>(.*?)</ref>((?:\s*<box><\d+><\d+><\d+><\d+></box>)+)"
    )
    box_pattern = re.compile(
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    )

    scale_x = img_w / COORD_SPACE
    scale_y = img_h / COORD_SPACE

    detections: list[Detection] = []

    for ref_match in ref_pattern.finditer(raw):
        label     = ref_match.group(1)
        boxes_str = ref_match.group(2)
        all_boxes = box_pattern.findall(boxes_str)

        instance_boxes = all_boxes[1:] if len(all_boxes) > 1 else all_boxes

        for x1, y1, x2, y2 in instance_boxes:
            detections.append(Detection(
                label=label,
                x1=round(int(x1) * scale_x),
                y1=round(int(y1) * scale_y),
                x2=round(int(x2) * scale_x),
                y2=round(int(y2) * scale_y),
            ))

    return detections
