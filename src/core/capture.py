import time
import numpy as np
import mss
from PyQt6.QtCore import QThread, pyqtSignal

class CaptureThread(QThread):
    # Señal emitida cada vez que se captura y procesa un nuevo frame
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.running = False
        self.target_fps = 30
        
        # Coordenadas de captura iniciales (se actualizarán desde el Glass)
        self.capture_region = {"top": 0, "left": 0, "width": 100, "height": 100}
        
        # Modo: True para continuo, False para Snapshot (disparo único)
        self.continuous_mode = True
        self._single_shot_requested = False

    def update_region(self, x, y, width, height):
        """Actualiza la región de pantalla a capturar."""
        self.capture_region = {"top": y, "left": x, "width": width, "height": height}

    def set_fps(self, fps: int):
        self.target_fps = fps

    def request_snapshot(self):
        """Solicita la captura de un solo frame cuando no estamos en modo continuo."""
        if not self.continuous_mode:
            self._single_shot_requested = True

    def run(self):
        self.running = True
        
        with mss.mss() as sct:
            while self.running:
                start_time = time.perf_counter()

                # Si estamos en modo continuo, o si se pidió un disparo en modo manual
                if self.continuous_mode or self._single_shot_requested:
                    try:
                        # Asegurar dimensiones válidas para mss
                        if self.capture_region["width"] > 0 and self.capture_region["height"] > 0:
                            # Capturar
                            sct_img = sct.grab(self.capture_region)
                            # Convertir a numpy array (formato BGRA)
                            frame = np.array(sct_img)
                            
                            # Procesar con OpenCV
                            processed_frame = self.processor.process_frame(frame)
                            
                            # Emitir
                            self.frame_ready.emit(processed_frame)
                    except Exception as e:
                        print(f"Error en captura: {e}")
                    
                    self._single_shot_requested = False
                
                # Control de FPS
                elapsed_time = time.perf_counter() - start_time
                time_to_sleep = (1.0 / self.target_fps) - elapsed_time
                if time_to_sleep > 0:
                    time.sleep(time_to_sleep)
                else:
                    # Prevenir que use 100% CPU si la captura/procesado es muy lenta
                    time.sleep(0.001)

    def stop(self):
        self.running = False
        self.wait()
