import ctypes
import sys
from PyQt6.QtGui import QImage
import cv2
import numpy as np

def enable_dpi_awareness():
    """
    Habilita el modo DPI awareness en Windows para evitar 
    que el escalado del sistema operativo distorsione las coordenadas de la pantalla.
    Esto garantiza que la ventana Glass y la captura mss coincidan.
    """
    if sys.platform == "win32":
        try:
            # PROCESS_PER_MONITOR_DPI_AWARE = 2
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception as e:
            print(f"No se pudo habilitar DPI Awareness vía shcore: {e}")
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception as e2:
                print(f"No se pudo habilitar DPI Awareness vía user32: {e2}")

def cv2_to_qimage(cv_img: np.ndarray) -> QImage:
    """Convierte una imagen de OpenCV (numpy array) a QImage de PyQt6."""
    # Verificamos los canales para manejar correctamente Grayscale, BGR o BGRA
    if len(cv_img.shape) == 2:
        # Grayscale
        height, width = cv_img.shape
        bytes_per_line = width
        qimg = QImage(cv_img.data, width, height, bytes_per_line, QImage.Format.Format_Grayscale8)
        return qimg
    
    height, width, channel = cv_img.shape
    bytes_per_line = channel * width
    
    if channel == 4:
        # BGRA to RGBA
        rgba_img = cv2.cvtColor(cv_img, cv2.COLOR_BGRA2RGBA)
        qimg = QImage(rgba_img.data, width, height, bytes_per_line, QImage.Format.Format_RGBA8888)
        return qimg
    elif channel == 3:
        # BGR to RGB
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        return qimg
    
    return QImage()
