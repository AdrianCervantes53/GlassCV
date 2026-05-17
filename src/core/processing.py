import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.current_filter = "normal"  # "normal", "grayscale", "canny"
    
    def set_filter(self, filter_name: str):
        """Cambia el filtro activo."""
        self.current_filter = filter_name

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Aplica el filtro seleccionado al frame capturado por mss.
        mss captura frames en formato BGRA.
        """
        if self.current_filter == "normal":
            return frame
            
        # Si no es normal, convertimos a BGR para procesar
        if frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr_frame = frame
            
        if self.current_filter == "grayscale":
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            # Retornamos en BGR (3 canales) para consistencia en la UI
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        elif self.current_filter == "canny":
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        return frame
