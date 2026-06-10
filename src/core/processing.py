import cv2
import numpy as np
import time

from core.ocr import draw_box_overlay, draw_text_overlay, get_translation_source_lang, translate_text

# ---------------------------------------------------------------------------
# Lazy AI Models
# ---------------------------------------------------------------------------
_YOLO_MODEL = None
_YOLO_MODEL_PATH = None
_OCR_READER = None
_OCR_LANGS = None

def get_yolo_model(model_path):
    global _YOLO_MODEL, _YOLO_MODEL_PATH
    if _YOLO_MODEL is None or _YOLO_MODEL_PATH != model_path:
        from ultralytics import YOLO
        _YOLO_MODEL = YOLO(model_path)
        _YOLO_MODEL_PATH = model_path
    return _YOLO_MODEL

def get_ocr_reader(langs):
    global _OCR_READER, _OCR_LANGS
    if _OCR_READER is None or _OCR_LANGS != langs:
        import easyocr
        _OCR_READER = easyocr.Reader(langs)
        _OCR_LANGS = langs
    return _OCR_READER


# ---------------------------------------------------------------------------
# Individual filter functions
# Each function receives a BGR frame (np.ndarray) and a params dict,
# and returns a BGR frame (np.ndarray).
# ---------------------------------------------------------------------------

def _to_bgr(frame: np.ndarray) -> np.ndarray:
    """Ensure frame is BGR (3-channel)."""
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def apply_normal(frame: np.ndarray, params: dict) -> np.ndarray:
    return _to_bgr(frame)


def apply_grayscale(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_canny(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    t1 = params.get("canny_t1", 100)
    t2 = params.get("canny_t2", 200)
    edges = cv2.Canny(gray, t1, t2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def apply_mirror(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    return cv2.flip(bgr, 1)


def apply_symmetry(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    axis = params.get("symmetry_axis", "vertical")
    h, w = bgr.shape[:2]
    result = bgr.copy()
    if axis == "vertical":
        half_w = w // 2
        if half_w > 0:
            left_half = bgr[:, :half_w]
            mirrored_left = cv2.flip(left_half, 1)
            paste_w = min(mirrored_left.shape[1], w - half_w)
            result[:, half_w:half_w + paste_w] = mirrored_left[:, :paste_w]
    else:
        half_h = h // 2
        if half_h > 0:
            top_half = bgr[:half_h, :]
            mirrored_top = cv2.flip(top_half, 0)
            paste_h = min(mirrored_top.shape[0], h - half_h)
            result[half_h:half_h + paste_h, :] = mirrored_top[:paste_h, :]
    return result


def apply_rgb_mixer(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    r_mult = params.get("r_mult", 100) / 100.0
    g_mult = params.get("g_mult", 100) / 100.0
    b_mult = params.get("b_mult", 100) / 100.0
    bgr_float = bgr.astype(np.float32)
    bgr_float[:, :, 0] *= b_mult
    bgr_float[:, :, 1] *= g_mult
    bgr_float[:, :, 2] *= r_mult
    return np.clip(bgr_float, 0, 255).astype(np.uint8)


def apply_binary(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    threshold = params.get("binary_threshold", 127)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


def apply_pixelated(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    pixel_size = max(2, params.get("pixel_size", 10))
    h, w = bgr.shape[:2]
    small = cv2.resize(bgr, (max(1, w // pixel_size), max(1, h // pixel_size)),
                       interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def apply_colorblind(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    cb_type = params.get("cb_type", "protanopia")
    matrices = {
        "protanopia":   np.array([[0.567, 0.433, 0], [0.558, 0.442, 0], [0, 0.242, 0.758]]),
        "deuteranopia": np.array([[0.625, 0.375, 0], [0.7,   0.3,   0], [0, 0.3,   0.7  ]]),
        "tritanopia":   np.array([[0.95,  0.05,  0], [0, 0.433, 0.567], [0, 0.475, 0.525]]),
    }
    matrix = matrices.get(cb_type, np.eye(3))
    rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cb_rgb = cv2.transform(rgb_frame, matrix)
    return cv2.cvtColor(cb_rgb, cv2.COLOR_RGB2BGR)


def apply_object_counter(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    template = params.get("template_img", None)
    confidence = params.get("confidence", 80) / 100.0
    result_frame = bgr.copy()
    if template is not None and template.size > 0:
        th, tw = template.shape[:2]
        fh, fw = bgr.shape[:2]
        if th <= fh and tw <= fw and th > 0 and tw > 0:
            gray_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
            res = cv2.matchTemplate(gray_frame, template_gray, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= confidence)
            boxes = [[int(pt[0]), int(pt[1]), int(tw), int(th)] for pt in zip(*loc[::-1])]
            if boxes:
                boxes, _ = cv2.groupRectangles(boxes, 1, 0.2)
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


def apply_smart_inverter(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    intensity = params.get("intensity", 100) / 100.0
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    inv_v = 255 - v
    blended_v = cv2.addWeighted(v, 1.0 - intensity, inv_v, intensity, 0)
    hsv_blended = cv2.merge([h, s, blended_v])
    return cv2.cvtColor(hsv_blended, cv2.COLOR_HSV2BGR)


def apply_yolo(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    model_path = params.get("yolo_model", "yolo11n.pt")
    confidence = params.get("yolo_conf", 50) / 100.0
    iou = params.get("yolo_iou", 45) / 100.0
    show_labels = params.get("yolo_labels", True)
    show_conf = params.get("yolo_show_conf", True)
    
    try:
        model = get_yolo_model(model_path)
        results = model(bgr, conf=confidence, iou=iou, verbose=False)
        annotated_frame = results[0].plot(labels=show_labels, conf=show_conf)
        return annotated_frame
    except Exception as e:
        cv2.putText(bgr, f"YOLO Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return bgr


def apply_ocr(frame: np.ndarray, params: dict) -> np.ndarray:
    bgr = _to_bgr(frame)
    langs = params.get("ocr_langs", ["en"])
    confidence_thresh = params.get("ocr_conf", 50) / 100.0
    font_color = params.get("ocr_font_color", (255, 255, 255))
    font_size = params.get("ocr_font_size", 14)
    font_scale = max(0.3, font_size / 24.0)
    font_thickness = params.get("ocr_font_thickness", 1)
    text_position = params.get("ocr_text_position", "above")
    box_thickness = params.get("ocr_box_thickness", 1)
    box_color = params.get("ocr_box_color", font_color)
    box_opacity = params.get("ocr_box_opacity", 100)
    show_text = params.get("ocr_show_text", True)
    show_boxes = params.get("ocr_show_boxes", True)
    show_conf = params.get("ocr_show_conf", True)
    text_background = params.get("ocr_text_background", True)
    overlay_text_source = params.get("ocr_overlay_text_source", "original")
    translate_target = params.get("ocr_translate_target", "es")
    subtitle_bg_color = params.get("ocr_subtitle_bg_color", (0, 0, 0))
    subtitle_bg_opacity = params.get("ocr_subtitle_bg_opacity", 100)
    background_padding = params.get("ocr_background_padding", 4)
    
    # Throttling logic
    current_time = time.time()
    last_time = params.get("_last_ocr_time", 0)
    # OCR is slow, let's limit to 2 FPS max (0.5s per frame)
    if current_time - last_time > 0.5:
        try:
            reader = get_ocr_reader(langs)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(gray)
            params["_last_ocr_results"] = results
            params["_last_ocr_time"] = current_time
        except Exception as e:
            params["_last_ocr_texts"] = ""
            params["_last_ocr_translated"] = ""
            cv2.putText(bgr, f"OCR Error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return bgr
    
    results = params.get("_last_ocr_results", [])
    translation_by_text = params.get("_last_ocr_translations", {})
    annotated_frame = bgr.copy()
    original_texts = []
    translated_texts = []

    for (bbox, text, prob) in results:
        if prob >= confidence_thresh:
            original_texts.append(text)
            translated = translation_by_text.get((text, translate_target), "")
            if translated:
                translated_texts.append(translated)

            if show_boxes:
                draw_box_overlay(annotated_frame, bbox, box_color, box_thickness, box_opacity)

            if show_text:
                base_text = translated if overlay_text_source == "translation" and translated else text
                overlay_text = f"{base_text} ({prob:.2f})" if show_conf else base_text
                bg_color = subtitle_bg_color if text_background else None
                draw_text_overlay(
                    annotated_frame,
                    bbox,
                    overlay_text,
                    font_scale,
                    font_color,
                    font_thickness,
                    text_position,
                    bg_color,
                    subtitle_bg_opacity,
                    background_padding,
                )

    params["_last_ocr_texts"] = "\n".join(original_texts)
    params["_last_ocr_translated"] = "\n".join(translated_texts)
    return annotated_frame



# ---------------------------------------------------------------------------
# Registry: maps filter name -> function
# ---------------------------------------------------------------------------
FILTER_REGISTRY = {
    "normal":         apply_normal,
    "grayscale":      apply_grayscale,
    "canny":          apply_canny,
    "mirror":         apply_mirror,
    "symmetry":       apply_symmetry,
    "rgb_mixer":      apply_rgb_mixer,
    "binary":         apply_binary,
    "pixelated":      apply_pixelated,
    "colorblind":     apply_colorblind,
    "object_counter": apply_object_counter,
    "smart_inverter": apply_smart_inverter,
    "yolo":           apply_yolo,
    "ocr":            apply_ocr,
}

# Human-readable display names
FILTER_DISPLAY_NAMES = {
    "normal":         "Normal",
    "grayscale":      "Grayscale",
    "canny":          "Canny Edges",
    "mirror":         "Mirror",
    "symmetry":       "Symmetry",
    "rgb_mixer":      "RGB Mixer",
    "binary":         "Binary",
    "pixelated":      "Pixelated",
    "colorblind":     "Colorblind Sim.",
    "object_counter": "Object Counter",
    "smart_inverter": "Smart Inverter",
    "yolo":           "YOLO Object Detection",
    "ocr":            "EasyOCR Text Recognition",
}


# ---------------------------------------------------------------------------
# ImageProcessor: chain-based processing
# ---------------------------------------------------------------------------
class ImageProcessor:
    def __init__(self):
        # filter_chain: list of dicts {"name": str, "params": dict}
        # An empty chain is equivalent to "normal" (pass-through).
        self.filter_chain: list[dict] = []
        # Shared params store (updated by the UI, keyed by filter name)
        self.all_params: dict[str, dict] = {}

        # Legacy single-filter support (kept for backward compat with
        # parts of the code that still call set_filter / set_filter_params).
        self._legacy_filter = "normal"
        self.filter_params: dict = {}  # kept for object_counter template_img

    # ------------------------------------------------------------------
    # Chain API (new)
    # ------------------------------------------------------------------

    def set_filter_chain(self, chain: list[dict]):
        """Replace the entire filter chain.

        chain is a list of {"name": str, "params": dict} entries.
        """
        self.filter_chain = chain
        if not any(entry.get("name") == "ocr" for entry in chain):
            ocr_params = self.all_params.get("ocr")
            if ocr_params is not None:
                ocr_params["_last_ocr_texts"] = ""
                ocr_params["_last_ocr_translated"] = ""

    def update_filter_params(self, filter_name: str, params: dict):
        """Update the params for a specific filter in the chain.

        Merges the new params into the stored params for that filter.
        All chain entries with the same name share the same params dict.
        """
        if filter_name not in self.all_params:
            self.all_params[filter_name] = {}
        self.all_params[filter_name].update(params)
        # Also propagate special keys to filter_params for legacy compat
        self.filter_params.update(params)

    # ------------------------------------------------------------------
    # Legacy API (kept for backward compat with main.py connections)
    # ------------------------------------------------------------------

    def set_filter(self, filter_name: str):
        """Legacy: sets a single active filter (replaces the chain)."""
        self._legacy_filter = filter_name
        if filter_name == "normal":
            self.filter_chain = []
        else:
            self.filter_chain = [{"name": filter_name, "params": {}}]

    def set_filter_params(self, params: dict):
        """Legacy: updates params for the current legacy filter."""
        self.filter_params.update(params)
        if self._legacy_filter and self._legacy_filter != "normal":
            self.update_filter_params(self._legacy_filter, params)

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the filter chain sequentially and return the result."""
        if not self.filter_chain:
            return _to_bgr(frame)

        result = frame
        for entry in self.filter_chain:
            name = entry["name"]
            # Keep the same params dict alive so filters can cache heavy state.
            merged_params = self.all_params.setdefault(name, {})
            merged_params.update(entry.get("params", {}))
            # Propagate template_img from legacy store (object_counter needs it)
            if "template_img" in self.filter_params:
                merged_params.setdefault("template_img", self.filter_params["template_img"])
            fn = FILTER_REGISTRY.get(name)
            if fn:
                result = fn(result, merged_params)

        return result

    def get_ocr_texts(self) -> tuple[str, str]:
        """Return the last OCR texts produced by the OCR filter."""
        params = self.all_params.get("ocr", {})
        return params.get("_last_ocr_texts", ""), params.get("_last_ocr_translated", "")

    def translate_last_ocr(self) -> tuple[str, str, str]:
        """Translate the latest OCR results without triggering a new OCR pass."""
        params = self.all_params.setdefault("ocr", {})
        original = params.get("_last_ocr_texts", "")
        if not original.strip():
            return "", "", "No OCR text detected yet."

        langs = params.get("ocr_langs", ["en"])
        confidence_thresh = params.get("ocr_conf", 50) / 100.0
        source_lang = get_translation_source_lang(langs)
        target_lang = params.get("ocr_translate_target", "es")
        translation_cache = params.setdefault("_ocr_translation_cache", {})
        translation_by_text = params.setdefault("_last_ocr_translations", {})
        results = params.get("_last_ocr_results", [])

        translated_texts = []
        if results:
            for _, text, prob in results:
                if prob >= confidence_thresh:
                    translated = translate_text(text, source_lang, target_lang, translation_cache)
                    translation_by_text[(text, target_lang)] = translated
                    translated_texts.append(translated)
        else:
            for text in original.splitlines():
                translated = translate_text(text, source_lang, target_lang, translation_cache)
                translation_by_text[(text, target_lang)] = translated
                translated_texts.append(translated)

        translated = "\n".join(translated_texts)
        params["_last_ocr_translated"] = translated
        if any(text.startswith("[Translation error:") for text in translated_texts):
            return original, translated, "Translation failed. Check connection or selected languages."
        return original, translated, ""
