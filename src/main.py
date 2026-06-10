import sys
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication

from core.utils import enable_dpi_awareness, cv2_to_qimage
from core.processing import ImageProcessor
from core.capture import CaptureThread
from ui.glass_window import GlassWindow
from ui.control_window import ControlWindow
from ui.ocr_text_window import OcrTextWindow


class OcrTranslationWorker(QThread):
    translation_finished = pyqtSignal(str, str, str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def run(self):
        try:
            original, translated, error = self.processor.translate_last_ocr()
        except Exception as exc:
            original, translated, error = "", "", str(exc)
        self.translation_finished.emit(original, translated, error)


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
        self.ocr_text_window = OcrTextWindow()
        self.translation_worker = None
        
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
        self.capture_thread.ocr_text_ready.connect(self._on_ocr_text_ready)
        self.control.translate_requested.connect(self._on_translate_requested)
        self.ocr_text_window.translate_requested.connect(self._on_translate_requested)
        
        # --- Control Window -> Capture Thread ---
        self.control.mode_changed.connect(self._set_continuous_mode)
        self.control.snapshot_requested.connect(self.capture_thread.request_snapshot)
        self.control.fps_changed.connect(self.capture_thread.set_fps)
        
        # --- Control Window -> Processor (chain-based) ---
        self.control.filter_chain_changed.connect(self.processor.set_filter_chain)
        self.control.filter_chain_changed.connect(self._on_filter_chain_changed)
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

    def _on_ocr_text_ready(self, original: str, translated: str):
        if self.ocr_text_window.isVisible():
            self.ocr_text_window.update_texts(original, translated or None)

    def _on_translate_requested(self):
        if self.translation_worker is not None:
            return

        self.ocr_text_window.set_translating(True)
        self.translation_worker = OcrTranslationWorker(self.processor)
        self.translation_worker.translation_finished.connect(self._on_translation_finished)
        self.translation_worker.finished.connect(self.translation_worker.deleteLater)
        self.translation_worker.finished.connect(self._clear_translation_worker)
        self.translation_worker.start()

    def _on_translation_finished(self, original: str, translated: str, error: str):
        self.ocr_text_window.set_translating(False)
        if original or translated:
            self.ocr_text_window.update_texts(original, translated or None)

        if error:
            self.ocr_text_window.set_translation_status(error)
        elif translated:
            self.ocr_text_window.set_translation_status("Translation ready.")
        else:
            self.ocr_text_window.set_translation_status("No OCR text detected.")

    def _clear_translation_worker(self):
        self.translation_worker = None

    def _on_filter_chain_changed(self, chain: list[dict]):
        has_ocr = any(entry.get("name") == "ocr" for entry in chain)
        if has_ocr:
            self.ocr_text_window.show()
        else:
            self.ocr_text_window.update_texts("", None)
            self.ocr_text_window.hide()

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
