# Setup — LocateAnything-3B

Instrucciones para instalar las dependencias necesarias para correr el script de prueba **sin romper** el entorno existente de GlassCV.

---

## Dependencias nuevas

| Paquete | Versión mínima | Notas |
|---|---|---|
| `transformers` | `>=4.51.0` | API multimodal estable para VLMs |
| `accelerate` | `>=0.34.0` | Requerido por `device_map` en `AutoModel` |
| `Pillow` | `>=10.0.0` | Procesamiento de imágenes (probablemente ya instalado) |
| `torch` | `>=2.1.0` + CUDA | Ya en el stack de GlassCV |

> `tokenizers` y `safetensors` se instalan automáticamente como dependencias de `transformers`.

---

## Instalación

### Con uv (recomendado)

```bash
uv add "transformers>=4.51.0" "accelerate>=0.34.0" "Pillow>=10.0.0"
```

### Con pip

```bash
pip install "transformers>=4.51.0" "accelerate>=0.34.0" "Pillow>=10.0.0"
```

---

## Primera ejecución

La primera vez el modelo se descarga (~7 GB) desde HuggingFace y queda cacheado en `~/.cache/huggingface/`. Las ejecuciones siguientes cargan desde cache (~5–10s).

```bash
# Desde la raíz del proyecto
python scripts/test_locate_anything.py
```

---

## Posibles conflictos

### `transformers` vs `easyocr`

EasyOCR fija versiones de `tokenizers` que pueden quedar por debajo de lo requerido por `transformers>=4.51`. Si hay conflicto al instalar, verificar con:

```bash
pip show tokenizers transformers easyocr
```

Si la versión de `tokenizers` instalada es `<0.19`, forzar:

```bash
pip install "tokenizers>=0.19" --upgrade
```

### CUDA out of memory

Si tienes menos de 8 GB libres de VRAM (otras apps abiertas), el script fallará al cargar en fp16.
Solución: cerrar otras aplicaciones con GPU antes de correr el script.

---

## Verificar instalación

```bash
python -c "import transformers; import accelerate; print(transformers.__version__, accelerate.__version__)"
```
