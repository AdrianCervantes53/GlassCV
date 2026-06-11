"""
Test script for nvidia/LocateAnything-3B inference.

Uso:
    python scripts/test_locate_anything.py

Requisitos previos:
    Ver scripts/LOCATE_ANYTHING_SETUP.md
"""

import sys
import time
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

# fp16 en CUDA, float32 como fallback en CPU
DTYPE      = torch.float16 if torch.cuda.is_available() else torch.float32
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


def load_model() -> tuple:
    """Carga el procesador y el modelo en el dispositivo disponible."""
    print(f"[INFO] Dispositivo: {DEVICE} | dtype: {DTYPE}")
    print(f"[INFO] Cargando modelo '{MODEL_ID}' (primera vez descarga ~7 GB)...")

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


def parse_output(raw) -> dict:
    """
    Normaliza el output de model.generate() al formato:
        {"text": str, "type": str}

    El generate() personalizado de LocateAnything no devuelve tensor de tokens
    como el generate() estándar de HuggingFace — devuelve el resultado ya
    procesado (str, list, o dict con coordenadas/texto).
    """
    print(f"[DEBUG] output type : {type(raw)}")
    print(f"[DEBUG] output value: {raw}")

    if isinstance(raw, str):
        return {"text": raw, "type": "str"}

    if isinstance(raw, list):
        return {"text": str(raw[0]) if raw else "", "type": "list"}

    if isinstance(raw, dict):
        return {"text": str(raw), "type": "dict"}

    # Tensor de tokens (fallback por si acaso)
    if isinstance(raw, torch.Tensor):
        return {"text": f"<tensor shape={list(raw.shape)}>", "type": "tensor"}

    return {"text": str(raw), "type": type(raw).__name__}


def run_inference(processor, model, image_path: Path, prompt: str) -> dict:
    """Ejecuta inferencia de grounding sobre una imagen con un prompt de texto."""
    if not image_path.exists():
        raise FileNotFoundError(f"Imagen no encontrada: {image_path}")

    image = Image.open(image_path).convert("RGB")
    print(f"[INFO] Imagen cargada: {image_path} ({image.size[0]}x{image.size[1]} px)")
    print(f"[INFO] Prompt: '{prompt}'")

    # Construir mensaje multimodal
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Preprocesar — images debe ser una lista, no un objeto Image suelto
    inputs = processor(
        text=processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
        images=[image],
        return_tensors="pt",
    ).to(DEVICE, dtype=DTYPE)

    # Inferencia — tokenizer requerido por generate() para calcular model_max_length
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

    result = parse_output(raw_output)
    result["elapsed"] = elapsed
    return result


def main() -> None:
    print("=" * 60)
    print("LocateAnything-3B — Test de inferencia")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("[WARN] No se detectó GPU CUDA. La inferencia en CPU será lenta (~30–60s).")
    else:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} | VRAM: {vram_gb:.1f} GB")

    try:
        processor, model = load_model()
        result = run_inference(processor, model, IMAGE_PATH, PROMPT)

        print()
        print("=" * 60)
        print("RESULTADO")
        print("=" * 60)
        print(f"Tiempo de inferencia : {result['elapsed']:.2f}s")
        print(f"Output type          : {result['type']}")
        print(f"Output               : {result['text']}")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Inferencia fallida: {e}")
        raise


if __name__ == "__main__":
    main()
