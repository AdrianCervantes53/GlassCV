# GlassCV

GlassCV is a high-performance desktop application focused on real-time screen capture and processing. It is built using Python, PyQt6, and OpenCV, with a primary focus on low latency and efficient processing.

## 🚀 Key Features

- **Dual Window Architecture**: 
  - **Glass Overlay**: A transparent, draggable, and resizable overlay that allows you to select the exact region of the screen you want to capture.
  - **Control Panel**: A dedicated interface to manage captures and configure the application without interfering with your viewing area.
- **Ultra-fast Screen Capture**: Uses `mss` for the best performance and lowest latency in screen capturing, ensuring a continuous stream.
- **Multithreaded Processing**: Image capture and processing are performed in separate background threads (multithreading) to prevent any UI lag.
- **HiDPI Support**: Advanced support for screens with different DPI scales, ensuring that capture coordinates are accurate across any monitor configuration.
- **AI and Computer Vision Ready**: The codebase is structured to scale easily and integrate computer vision models in the future, such as YOLO, for real-time analysis of the captured region.

## 🛠️ Technologies Used

- **Python >= 3.11**
- **PyQt6**: For GUI development and advanced window management.
- **OpenCV (`opencv-python`)**: For processing captured images.
- **MSS (`mss`)**: For extremely low-latency, cross-platform screen capture.
- **NumPy**: For efficient manipulation of pixel matrices and image data.

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AdrianCervantes53/GlassCV.git
   cd GlassCV
   ```

2. (Recommended) Ensure you have an active virtual environment. The project uses a `pyproject.toml` and `uv.lock` file for dependency management. You can create a virtual environment and install dependencies using `uv` or `pip`:

   **Using `uv` (recommended):**
   ```bash
   uv sync
   ```

   **Using `pip`:**
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # Install dependencies
   pip install -e .
   ```

## 🚀 Usage

To run the application, execute the main script located in the `src` directory:

```bash
python src/main.py
```

Upon launching, two windows will open:
1. The **Control Panel**: From here you can control the workflow and configure options.
2. The **Glass Window**: Drag and resize this transparent box over the specific area of your screen that you wish to capture and analyze.

## 📂 Project Structure

- `src/`
  - `main.py`: Main entry point of the application.
  - `ui/`: Modules containing the GUI logic (ControlWindow and GlassWindow).
  - `core/`: Core logic, screen capture via `mss`, background thread image processing, geometry calculations, etc.
- `pyproject.toml`: Project configuration and dependencies.
- `.gitignore`: Exclusion rules to prevent uploading temporary files, binaries, and virtual environments to the repository.

## 📝 Roadmap
- Real-time object detection model integration in the processing pipeline (e.g., YOLO).
- Add OpenCV processing options and filters that can be configured directly from the Control Panel.
- Multi-monitor management improvements.