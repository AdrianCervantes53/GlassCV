"""
Test script for nvidia/LocateAnything-3B inference.

Uso:
    python scripts/test_locate_anything.py

Requisitos previos:
    Ver scripts/LOCATE_ANYTHING_SETUP.md

Formato de salida del modelo:
    <ref>label</ref><box><x1><y1><x2><y2></box>
    Coordenadas en espacio normalizado 0–1000. Se convierten a píxeles reales
    multiplicando por (img_w / 1000, img_h / 1000).
"""

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

MODEL_ID   = "nvidia/LocateAnything-3B"
IMAGE_PATH = Path("img/ui.png")
PROMPT     = "Save image button"

DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Espacio de coordenadas que usa el modelo internamente
COORD_SPACE = 1000


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
    """Carga el procesador y el modelo en el dispositivo disponible."""
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


def run_inference(processor, model, image_path: Path, prompt: str) -> tuple[list[Detection], float]:
    """
    Ejecuta inferencia de grounding y devuelve detecciones en píxeles reales.
    """
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
    return detections, elapsed


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse_detections(raw: str, img_w: int, img_h: int) -> list[Detection]:
    """
    Parsea el output del modelo al formato:
        <ref>label</ref><box><x1><y1><x2><y2></box>

    Convierte coordenadas de espacio normalizado (0–1000) a píxeles reales.
    """
    pattern = re.compile(
        r"<ref>(.*?)</ref>\s*<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"
    )

    scale_x = img_w / COORD_SPACE
    scale_y = img_h / COORD_SPACE

    detections = []
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
        detections, elapsed = run_inference(processor, model, IMAGE_PATH, PROMPT)

        print()
        print("=" * 60)
        print(f"RESULTADO  ({elapsed:.2f}s | {len(detections)} detección/es)")
        print("=" * 60)

        if not detections:
            print("[INFO] No se encontraron detecciones.")
        else:
            for i, det in enumerate(detections, 1):
                print(f"\n[{i}] {det}")

        print("=" * 60)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Inferencia fallida: {e}")
        raise


if __name__ == "__main__":
    main()
