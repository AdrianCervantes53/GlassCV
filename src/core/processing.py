import cv2
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.current_filter = "normal"  # "normal", "grayscale", "canny", "mirror"
        self.filter_params = {}
    
    def set_filter(self, filter_name: str):
        """Changes the active filter."""
        self.current_filter = filter_name

    def set_filter_params(self, params: dict):
        """Updates the active filter parameters."""
        self.filter_params = params

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
            t1 = self.filter_params.get("canny_t1", 100)
            t2 = self.filter_params.get("canny_t2", 200)
            edges = cv2.Canny(gray, t1, t2)
            return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            
        elif self.current_filter == "mirror":
            return cv2.flip(bgr_frame, 1)
            
        elif self.current_filter == "rgb_mixer":
            r_mult = self.filter_params.get("r_mult", 100) / 100.0
            g_mult = self.filter_params.get("g_mult", 100) / 100.0
            b_mult = self.filter_params.get("b_mult", 100) / 100.0
            
            bgr_float = bgr_frame.astype(np.float32)
            bgr_float[:, :, 0] *= b_mult
            bgr_float[:, :, 1] *= g_mult
            bgr_float[:, :, 2] *= r_mult
            return np.clip(bgr_float, 0, 255).astype(np.uint8)
            
        elif self.current_filter == "binary":
            gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
            threshold = self.filter_params.get("binary_threshold", 127)
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
        elif self.current_filter == "pixelated":
            pixel_size = max(2, self.filter_params.get("pixel_size", 10))
            h, w = bgr_frame.shape[:2]
            small = cv2.resize(bgr_frame, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            
        elif self.current_filter == "colorblind":
            cb_type = self.filter_params.get("cb_type", "protanopia")
            
            if cb_type == "protanopia":
                matrix = np.array([[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0.242, 0.758]])
            elif cb_type == "deuteranopia":
                matrix = np.array([[0.625, 0.375, 0], [0.7, 0.3, 0], [0, 0.3, 0.7]])
            elif cb_type == "tritanopia":
                matrix = np.array([[0.95, 0.05, 0], [0, 0.433, 0.567], [0, 0.475, 0.525]])
            else:
                matrix = np.eye(3)
                
            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            cb_rgb = cv2.transform(rgb_frame, matrix)
            return cv2.cvtColor(cb_rgb, cv2.COLOR_RGB2BGR)
            
        elif self.current_filter == "object_counter":
            template = self.filter_params.get("template_img", None)
            confidence = self.filter_params.get("confidence", 80) / 100.0
            
            result_frame = bgr_frame.copy()
            
            if template is not None and template.size > 0:
                th, tw = template.shape[:2]
                fh, fw = bgr_frame.shape[:2]
                if th <= fh and tw <= fw and th > 0 and tw > 0:
                    gray_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
                    
                    if len(template.shape) == 3:
                        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
                    else:
                        template_gray = template
                        
                    res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
                    loc = np.where(res >= confidence)
                    
                    boxes = []
                    for pt in zip(*loc[::-1]):
                        boxes.append([int(pt[0]), int(pt[1]), int(tw), int(th)])
                        
                    if len(boxes) > 0:
                        boxes = np.array(boxes)
                        # groupRectangles requires a list of rects
                        boxes, weights = cv2.groupRectangles(boxes.tolist(), 1, 0.2)
                        count = len(boxes)
                        for (x, y, w, h) in boxes:
                            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(result_frame, f"Objects: {count}", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(result_frame, "Objects: 0", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(result_frame, "Invalid template size", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(result_frame, "No template captured", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result_frame
            
        elif self.current_filter == "smart_inverter":
            intensity = self.filter_params.get("intensity", 100) / 100.0
            
            hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            inv_v = 255 - v
            blended_v = cv2.addWeighted(v, 1.0 - intensity, inv_v, intensity, 0)
            
            hsv_blended = cv2.merge([h, s, blended_v])
            return cv2.cvtColor(hsv_blended, cv2.COLOR_HSV2BGR)
            
        return frame
