import time
import numpy as np
import mss
from PyQt6.QtCore import QThread, pyqtSignal

class CaptureThread(QThread):
    # Signal emitted every time a new frame is captured and processed
    frame_ready = pyqtSignal(np.ndarray)
    ocr_text_ready = pyqtSignal(str, str)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.running = False
        self.target_fps = 30
        
        # Initial capture coordinates (will be updated from the Glass)
        self.capture_region = {"top": 0, "left": 0, "width": 100, "height": 100}
        
        # Mode: True for continuous, False for Snapshot (single shot)
        self.continuous_mode = True
        self._single_shot_requested = False

    def update_region(self, x, y, width, height):
        """Updates the screen region to capture."""
        self.capture_region = {"top": y, "left": x, "width": width, "height": height}

    def set_fps(self, fps: int):
        self.target_fps = fps

    def request_snapshot(self):
        """Requests the capture of a single frame when not in continuous mode."""
        if not self.continuous_mode:
            self._single_shot_requested = True

    def run(self):
        self.running = True
        
        with mss.mss() as sct:
            while self.running:
                start_time = time.perf_counter()

                # If we are in continuous mode, or if a shot was requested in manual mode
                if self.continuous_mode or self._single_shot_requested:
                    try:
                        # Ensure valid dimensions for mss
                        if self.capture_region["width"] > 0 and self.capture_region["height"] > 0:
                            # Capture
                            sct_img = sct.grab(self.capture_region)
                            # Convert to numpy array (BGRA format)
                            frame = np.array(sct_img)
                            
                            # Process with OpenCV
                            processed_frame = self.processor.process_frame(frame)
                            
                            # Emit
                            self.frame_ready.emit(processed_frame)
                            if hasattr(self.processor, "get_ocr_texts"):
                                original, translated = self.processor.get_ocr_texts()
                                self.ocr_text_ready.emit(original, translated)
                    except Exception as e:
                        print(f"Capture error: {e}")
                    
                    self._single_shot_requested = False
                
                # FPS control
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep = (1.0 / self.target_fps) - elapsed_time
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    # Prevent 100% CPU usage if capture/processing is too slow
                    time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()
