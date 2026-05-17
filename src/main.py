import sys
from PyQt6.QtWidgets import QApplication

from core.utils import enable_dpi_awareness, cv2_to_qimage
from core.processing import ImageProcessor
from core.capture import CaptureThread
from ui.glass_window import GlassWindow
from ui.control_window import ControlWindow

class GlassCVApp:
    def __init__(self):
        # 1. Enable DPI Awareness before creating QApplication
        enable_dpi_awareness()
        
        self.app = QApplication(sys.argv)
        self.app.setStyle("Fusion")
        
        # 2. Instantiate Logic
        self.processor = ImageProcessor()
        self.capture_thread = CaptureThread(self.processor)
        
        # 3. Instantiate UI
        self.glass = GlassWindow()
        self.control = ControlWindow()
        
        # 4. Connect Signals
        self._connect_signals()
        
        # 5. Start Threads and Show Windows
        self.glass.show()
        self.control.show()
        self.capture_thread.start()

    def _connect_signals(self):
        # --- Glass -> Capture Thread ---
        self.glass.geometry_changed.connect(self.capture_thread.update_region)
        
        # --- Capture Thread -> UI ---
        # The thread emits a numpy array (processed frame)
        self.capture_thread.frame_ready.connect(self._on_frame_ready)
        
        # --- Control Window -> Capture Thread ---
        self.control.mode_changed.connect(self._set_continuous_mode)
        self.control.snapshot_requested.connect(self.capture_thread.request_snapshot)
        self.control.fps_changed.connect(self.capture_thread.set_fps)
        
        # --- Control Window -> Processor ---
        self.control.filter_changed.connect(self.processor.set_filter)
        
        # --- Control Window -> Glass Window ---
        self.control.glass_pinned.connect(self.glass.set_click_through)
        self.control.mirror_mode_changed.connect(self.glass.set_mirror_mode)
        
        # --- Control Window -> File System ---
        self.control.save_requested.connect(self.control.save_current_pixmap)

    def _set_continuous_mode(self, continuous: bool):
        self.capture_thread.continuous_mode = continuous

    def _on_frame_ready(self, frame_np):
        # Convert from numpy array to QImage
        qimg = cv2_to_qimage(frame_np)
        if not qimg.isNull():
            # Update both windows
            self.control.update_image(qimg)
            self.glass.update_image(qimg)

    def run(self):
        exit_code = self.app.exec()
        # Ensure thread is closed on exit
        self.capture_thread.stop()
        sys.exit(exit_code)

if __name__ == "__main__":
    app = GlassCVApp()
    app.run()
