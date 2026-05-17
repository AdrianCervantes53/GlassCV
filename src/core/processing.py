import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.current_filter = "normal"  # "normal", "grayscale", "canny"
    
    def set_filter(self, filter_name: str):
        """Changes the active filter."""
        self.current_filter = filter_name

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Applies the selected filter to the frame captured by mss.
        mss captures frames in BGRA format.
        """
        if self.current_filter == "normal":
            return frame
            
        # If not normal, convert to BGR for processing
        if frame.shape[2] == 4:
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        else:
            bgr_frame = frame
            
        if self.current_filter == "grayscale":
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            # Return in BGR (3 channels) for UI consistency
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
        elif self.current_filter == "canny":
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        return frame
