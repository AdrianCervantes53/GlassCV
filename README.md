# GlassCV

GlassCV is a high-performance desktop application focused on real-time screen capture, processing, and AI-powered analysis. It is built using Python, PyQt6, OpenCV, and deep learning frameworks (YOLO, EasyOCR, LocateAnything), with a primary focus on low latency and GPU-accelerated processing.

## Key Features

- **Dual Window Architecture**:
  - **Glass Overlay**: A transparent, draggable, and resizable overlay that allows you to select the exact region of the screen you want to capture.
  - **Control Panel**: A dedicated interface to manage captures, configure filters, and control AI models without interfering with your viewing area.
- **Ultra-fast Screen Capture**: Uses `mss` for the best performance and lowest latency in screen capturing, ensuring a continuous stream.
- **Multithreaded Processing**: Image capture and processing are performed in separate background threads to prevent any UI lag. AI inference (OCR, LocateAnything) runs in dedicated workers that keep the video feed running at full speed.
- **HiDPI Support**: Advanced support for screens with different DPI scales, ensuring that capture coordinates are accurate across any monitor configuration.
- **Real-time Image Processing & Filter Chain**: Apply and chain various built-in OpenCV filters (Grayscale, Canny Edges, RGB Mixer, Colorblind Simulation, Smart Inverter, and more) directly from the Control Panel.
- **Template Matching & Object Counting**: Use a dedicated "Template Glass" window to capture a visual template and perform real-time object detection and counting within the main capture region.
- **YOLO Object Detection**: Real-time object detection powered by Ultralytics YOLO with GPU acceleration (CUDA). Features include:
  - Dynamic model selector with auto-download — choose from YOLO11 (Nano to XLarge) and YOLOv8 (Nano to Medium) directly from a dropdown menu.
  - Models are automatically downloaded the first time they are selected and stored in the `models/` directory.
  - Support for loading custom-trained `.pt` models via a file dialog.
  - Adjustable Confidence and IOU thresholds, with toggles for labels and confidence scores.
- **EasyOCR Text Recognition**: Real-time text detection and recognition using EasyOCR with GPU acceleration. Features include:
  - Built-in throttling mechanism (max 2 FPS) to maintain a smooth video feed.
  - Adjustable confidence threshold and language selection.
  - Configurable overlay with toggles for text, boxes, confidence scores, and text background.
  - Independent styling for text, detection boxes, and background (color, thickness, opacity, padding).
  - Translation workflow with target language selection, overlay text source selection, and a Translate action.
- **Nvidia LocateAnything Visual Grounding**: Natural-language UI element localization powered by [nvidia/LocateAnything-3B](https://huggingface.co/nvidia/LocateAnything-3B) (CVPR 2026). Features include:
  - Locate any UI element by describing it in plain English (e.g. *"Save image button"*, *"Search bar"*).
  - On-demand inference: runs once on the current frame when triggered, with the video feed continuing uninterrupted.
  - Bounding boxes rendered in real-time on every subsequent frame until cleared or replaced.
  - Fully customizable overlay: box color, thickness, font scale, font thickness, label toggle, center dot toggle.
  - Model loads once (~16s, ~7 GB) and is cached for the session. Inference runs in a background `QThread` (~72s on RTX 4060 8 GB, fp16).
  - Outputs pixel-space bounding boxes ready for downstream UI automation use cases.
- **GPU Acceleration**: Full NVIDIA CUDA support via PyTorch. All AI models (YOLO, EasyOCR, LocateAnything) automatically detect and utilize your GPU for maximum performance.

## Technologies Used

- **Python >= 3.11**
- **PyQt6**: GUI development and advanced window management.
- **OpenCV (`opencv-python`)**: Image capture processing and annotation.
- **MSS (`mss`)**: Extremely low-latency, cross-platform screen capture.
- **NumPy**: Efficient manipulation of pixel matrices and image data.
- **Ultralytics (`ultralytics`)**: YOLO object detection models.
- **EasyOCR (`easyocr`)**: Optical character recognition.
- **PyTorch (`torch`) with CUDA**: Deep learning backend with GPU acceleration.
- **HuggingFace Transformers (`transformers>=4.51.0,<5.0.0`)**: Model loading and inference for LocateAnything-3B. Pinned to `<5.0.0` due to breaking API changes in v5.
- **Accelerate (`accelerate`)**: Required by Transformers for `device_map` model loading.
- **Pillow**: Image format handling in the LocateAnything inference pipeline.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianCervantes53/GlassCV.git
   cd GlassCV
   ```

2. Create a virtual environment and install dependencies:

   **Using `uv` (recommended):**
   ```bash
   uv sync
   ```

   **Using `pip`:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   pip install -e .
   ```

3. **(Optional but recommended) Enable GPU Acceleration**: By default, `pip install` installs the CPU-only version of PyTorch. For CUDA support on an NVIDIA GPU:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --upgrade --force-reinstall
   ```

4. **(Optional) Pre-download the LocateAnything model**: The model downloads automatically on first use (~7 GB). To pre-download it:
   ```bash
   python -c "from transformers import AutoModel, AutoProcessor; AutoProcessor.from_pretrained('nvidia/LocateAnything-3B', trust_remote_code=True); AutoModel.from_pretrained('nvidia/LocateAnything-3B', trust_remote_code=True)"
   ```

## Usage

```bash
python src/main.py
```

Upon launching, two windows open:
1. **Control Panel** — configure filter chains, AI models, and options.
2. **Glass Window** — drag and resize this transparent overlay over the area you want to capture.

### Filter Chain

All filters are composable. Add filters from the dropdown, drag to reorder, and select any filter to configure its parameters. Removing a filter from the chain clears its associated state (e.g., removing LocateAnything clears rendered detections).

### YOLO Object Detection

1. Select **YOLO Object Detection** → **＋ Add**.
2. Choose a model from the dropdown (auto-downloaded on first use).
3. Adjust **Confidence** and **IOU** sliders.
4. Use **Custom...** to load a custom-trained `.pt` model.

### EasyOCR Text Recognition

1. Select **EasyOCR Text Recognition** → **＋ Add**.
2. Set language and confidence threshold.
3. Customize overlay appearance in the **Overlay** section.
4. Use the **Translation** section to select a target language and click **Translate**.

### Nvidia LocateAnything Visual Grounding

1. Select **Nvidia LocateAnything** → **＋ Add**.
2. Type a natural-language description of the element to find (e.g. `"Save image button"`).
3. Click **Locate Now!** — the video feed continues while inference runs in the background.
4. Bounding boxes appear on the live feed once inference completes. They persist until you click **Clear** or submit a new prompt.
5. Adjust box color, thickness, and font scale in the **Style** section.

> **Performance note:** First run takes ~90s (model load + inference). Subsequent runs take ~72s on an RTX 4060 8 GB (fp16). Ensure at least 7 GB of free VRAM before running alongside other GPU-intensive filters.

### Control Panel with YOLO Object Detection

![GlassCV Control Panel](img/ui_v2.png)

### Glass Overlay with OCR and Translation

![GlassCV Glass Overlay](img/ocr_translated.png)

### OCR with Translation Panel

![GlassCV OCR/Translation Overlay](img/ocr_and_translation_overlay.png)

## Project Structure

```
GlassCV/
├── src/
│   ├── main.py                         # Entry point, app wiring, background workers
│   ├── core/
│   │   ├── capture.py                  # Screen capture thread (MSS)
│   │   ├── processing.py               # Filter registry, ImageProcessor, filter chain
│   │   ├── locate_anything_wrapper.py  # LocateAnything-3B: lazy loader, inference, parser
│   │   ├── ocr.py                      # EasyOCR utilities and translation
│   │   └── utils.py                    # DPI awareness, image conversion helpers
│   └── ui/
│       ├── glass_window.py             # Transparent overlay window
│       ├── control_window.py           # Control panel (filter chain UI, all params)
│       ├── ocr_text_window.py          # OCR results and translation window
│       └── widgets/
│           └── collapsible_section.py  # Reusable collapsible UI section
├── models/                             # YOLO weights (auto-downloaded on first use)
├── scripts/
│   ├── test_locate_anything.py         # Standalone inference test for LocateAnything-3B
│   └── LOCATE_ANYTHING_SETUP.md        # Dependency setup and troubleshooting guide
├── img/                                # UI screenshots and test images
├── pyproject.toml                      # Project config and dependencies
└── .gitignore
```

## Roadmap

- Multi-monitor management improvements.
- Exporting and importing functionality for custom filter chains.
- Asynchronous model downloading with progress indicator.
- PyAutoGUI integration for click-through automation using LocateAnything detections.
