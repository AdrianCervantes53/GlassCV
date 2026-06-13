"""
Test script for nvidia/LocateAnything-3B inference.

Uso:
    python scripts/test_locate_anything.py

Requisitos previos:
    Ver scripts/LOCATE_ANYTHING_SETUP.md

Formato de salida del modelo (Parallel Box Decoding):
    <ref>label</ref><box><avg></box><box><inst1></box><box><inst2></box>
    Cuando hay múltiples instancias, el primer <box> es el agregado PBD
    (promedio de todas las instancias) y se descarta. Si solo hay un <box>,
    se trata como detección individual y se mantiene.
    Coordenadas en espacio normalizado 0–1000 → píxeles reales.
"""

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_ID   = "nvidia/LocateAnything-3B"
IMAGE_PATH = Path("img/ui.png")
PROMPT     = "Save image button"

DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COORD_SPACE = 1000
COLORS = ["#FF4444", "#44AAFF", "#44FF88", "#FFB844", "#CC44FF"]


# ---------------------------------------------------------------------------
# Tipos
# ---------------------------------------------------------------------------

@dataclass
class Detection:
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

    def __str__(self) -> str:
        cx, cy = self.center
        return (
            f"  label  : {self.label}\n"
            f"  box    : ({self.x1}, {self.y1}) → ({self.x2}, {self.y2})\n"
            f"  size   : {self.width}x{self.height} px\n"
            f"  center : ({cx}, {cy})"
        )


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def load_model() -> tuple:
    print(f"[INFO] Dispositivo: {DEVICE} | dtype: {DTYPE}")
    print(f"[INFO] Cargando modelo '{MODEL_ID}'...")

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()
    print(f"[INFO] Modelo cargado en {time.time() - t0:.1f}s")

    return processor, model


def run_inference(processor, model, image_path: Path, prompt: str) -> tuple[Image.Image, list[Detection], float, str]:
    if not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size
    print(f"[INFO] Imagen: {image_path} ({img_w}x{img_h} px)")
    print(f"[INFO] Prompt: '{prompt}'")

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

    t0 = time.time()
    with torch.inference_mode():
        raw_output = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            use_cache=True,
            tokenizer=processor.tokenizer,
        )
    elapsed = time.time() - t0

    detections = parse_detections(raw_output, img_w, img_h)
    return image, detections, elapsed, raw_output


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_detections(raw: str, img_w: int, img_h: int) -> list[Detection]:
    """
    Parsea el output del modelo manejando Parallel Box Decoding (PBD).

    Formato: <ref>label</ref><box><avg></box><box><inst1></box>...
    Cuando N > 1: el primer <box> es el agregado PBD → se descarta.
    Cuando N = 1: único box → detección individual → se mantiene.
    """
    ref_pattern = re.compile(
        r"<ref>(.*?)</ref>((?:\s*<box><\d+><\d+><\d+><\d+></box>)+)"
    )
    box_pattern = re.compile(
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    )

    scale_x = img_w / COORD_SPACE
    scale_y = img_h / COORD_SPACE

    detections = []
    for ref_match in ref_pattern.finditer(raw):
        label     = ref_match.group(1)
        boxes_str = ref_match.group(2)
        all_boxes = box_pattern.findall(boxes_str)

        # Saltar el primer box (agregado PBD) cuando hay múltiples instancias
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


# ---------------------------------------------------------------------------
# Visualización
# ---------------------------------------------------------------------------

def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    result  = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", result.size, (0, 0, 0, 0))
    draw    = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype("arial.ttf", size=14)
    except OSError:
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        color_hex   = COLORS[i % len(COLORS)]
        r, g, b     = tuple(int(color_hex.lstrip("#")[j:j+2], 16) for j in (0, 2, 4))
        color_solid = (r, g, b, 255)
        color_fill  = (r, g, b, 50)

        draw.rectangle([det.x1, det.y1, det.x2, det.y2], fill=color_fill, outline=color_solid, width=2)

        label_text = f"[{i+1}] {det.label}"
        bbox = draw.textbbox((det.x1, det.y1), label_text, font=font)
        pad  = 2
        draw.rectangle([bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad], fill=color_solid)
        draw.text((det.x1, det.y1), label_text, fill=(255, 255, 255, 255), font=font)

        cx, cy = det.center
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=color_solid)

    return Image.alpha_composite(result, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("LocateAnything-3B — Test de inferencia")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[WARN] No se detectó GPU CUDA. La inferencia en CPU será lenta.")
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB")

    try:
        processor, model = load_model()
        image, detections, elapsed, raw = run_inference(processor, model, IMAGE_PATH, PROMPT)

        print()
        print("=" * 60)
        print(f"RESULTADO  ({elapsed:.2f}s | {len(detections)} detección/es)")
        print("=" * 60)
        print(f"Raw output: {raw}")
        print()

        if not detections:
            print("[INFO] No se encontraron detecciones.")
        else:
            for i, det in enumerate(detections, 1):
                print(f"\n[{i}] {det}")

            annotated = draw_detections(image, detections)
            annotated.show(title=f"LocateAnything — '{PROMPT}'")

        print("=" * 60)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Inferencia fallida: {e}")
        raise


if __name__ == "__main__":
    main()
