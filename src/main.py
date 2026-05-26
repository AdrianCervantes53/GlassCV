import sys
from PyQt6.QtWidgets import QApplication

from core.utils import enable_dpi_awareness, cv2_to_qimage
from core.processing import ImageProcessor
from core.capture import CaptureThread
from ui.glass_window import GlassWindow
from ui.control_window import ControlWindow

class GlassCV:
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
        from PyQt6.QtGui import QColor
        self.template_glass = GlassWindow(
            title="GlassCV - Template", 
            geometry=(150, 150, 100, 100), 
            border_color=QColor(255, 165, 0)
        )
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
        self.glass.geometry_changed.connect(self.control.update_glass_size)
        
        # --- Capture Thread -> UI ---
        # The thread emits a numpy array (processed frame)
        self.capture_thread.frame_ready.connect(self._on_frame_ready)
        
        # --- Control Window -> Capture Thread ---
        self.control.mode_changed.connect(self._set_continuous_mode)
        self.control.snapshot_requested.connect(self.capture_thread.request_snapshot)
        self.control.fps_changed.connect(self.capture_thread.set_fps)
        
        # --- Control Window -> Processor (chain-based) ---
        self.control.filter_chain_changed.connect(self.processor.set_filter_chain)
        self.control.filter_params_changed_for.connect(self.processor.update_filter_params)
        
        # --- Control Window -> Glass Window ---
        self.control.glass_pinned.connect(self.glass.set_click_through)
        self.control.mirror_mode_changed.connect(self.glass.set_mirror_mode)
        self.control.border_toggled.connect(self.glass.set_show_border)
        
        # --- Control Window -> File System ---
        self.control.save_requested.connect(self.control.save_current_pixmap)
        
        # --- Control Window -> Template Glass ---
        self.control.toggle_template_glass.connect(self._toggle_template_glass)
        self.control.request_template_capture.connect(self._capture_template)

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

    def _toggle_template_glass(self, state: bool):
        if state:
            self.template_glass.show()
        else:
            self.template_glass.hide()

    def _capture_template(self):
        import mss
        import numpy as np
        g = self.template_glass.geometry()
        bw = self.template_glass.border_width
        
        region = {
            "top": g.y() + bw,
            "left": g.x() + bw,
            "width": max(1, g.width() - 2*bw),
            "height": max(1, g.height() - 2*bw)
        }
        
        with mss.mss() as sct:
            img = np.array(sct.grab(region))
            
        self.processor.update_filter_params("object_counter", {"template_img": img})

if __name__ == "__main__":
    app = GlassCV()
    app.run()
