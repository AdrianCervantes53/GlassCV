import cv2
import numpy as np


OCR_LANGUAGES = [
    ("English", "en", "en"),
    ("Spanish", "es", "es"),
    ("French", "fr", "fr"),
    ("German", "de", "de"),
    ("Italian", "it", "it"),
    ("Portuguese", "pt", "pt"),
    ("Russian", "ru", "ru"),
    ("Japanese", "ja", "ja"),
    ("Korean", "ko", "ko"),
    ("Chinese (Simplified)", "ch_sim", "zh-CN"),
    ("Chinese (Traditional)", "ch_tra", "zh-TW"),
    ("Arabic", "ar", "ar"),
    ("Hindi", "hi", "hi"),
    ("Thai", "th", "th"),
    ("Vietnamese", "vi", "vi"),
    ("Turkish", "tr", "tr"),
    ("Polish", "pl", "pl"),
    ("Dutch", "nl", "nl"),
    ("Swedish", "sv", "sv"),
    ("Czech", "cs", "cs"),
]

TRANSLATION_LANGUAGES = [
    ("English", "en"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("German", "de"),
    ("Italian", "it"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Chinese (Simplified)", "zh-CN"),
    ("Chinese (Traditional)", "zh-TW"),
    ("Arabic", "ar"),
    ("Hindi", "hi"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Turkish", "tr"),
    ("Polish", "pl"),
    ("Dutch", "nl"),
    ("Swedish", "sv"),
    ("Czech", "cs"),
]


def get_translation_source_lang(ocr_langs: list[str]) -> str:
    """Return a Google Translate source language for the selected EasyOCR code."""
    if not ocr_langs:
        return "auto"

    easyocr_code = ocr_langs[0]
    for _, ocr_code, translator_code in OCR_LANGUAGES:
        if ocr_code == easyocr_code:
            return translator_code
    return easyocr_code


def translate_text(text: str, source_lang: str, target_lang: str, cache: dict[tuple[str, str, str], str]) -> str:
    """Translate text using a small in-memory cache to avoid repeated requests."""
    clean_text = text.strip()
    if not clean_text:
        return ""

    cache_key = (clean_text, source_lang, target_lang)
    if cache_key in cache:
        return cache[cache_key]

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source=source_lang, target=target_lang).translate(clean_text)
    except Exception as exc:
        translated = f"[Translation error: {exc}]"

    cache[cache_key] = translated
    return translated


def get_bbox_rect(frame: np.ndarray, bbox) -> tuple[int, int, int, int]:
    """Convert EasyOCR bbox points into a clamped rectangle."""
    points = [(int(point[0]), int(point[1])) for point in bbox]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x1 = max(0, min(x_values))
    x2 = min(frame.shape[1] - 1, max(x_values))
    y1 = max(0, min(y_values))
    y2 = min(frame.shape[0] - 1, max(y_values))
    return x1, y1, x2, y2


def draw_box_overlay(
    frame: np.ndarray,
    bbox,
    color: tuple[int, int, int],
    thickness: int = 1,
    opacity: int = 100,
) -> None:
    """Draw only the OCR bounding box."""
    x1, y1, x2, y2 = get_bbox_rect(frame, bbox)
    alpha = max(0, min(opacity, 100)) / 100.0
    if alpha >= 1.0:
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, thickness))
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, max(1, thickness))
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_text_overlay(
    frame: np.ndarray,
    bbox,
    text: str,
    font_scale: float,
    font_color: tuple[int, int, int],
    font_thickness: int = 2,
    text_position: str = "above",
    bg_color: tuple[int, int, int] | None = None,
    bg_opacity: int = 70,
    bg_padding: int = 4,
) -> None:
    """Draw OCR text, optionally with a tight semi-transparent background."""
    x1, y1, _, y2 = get_bbox_rect(frame, bbox)
    thickness = max(1, font_thickness)
    baseline = 0
    text_size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_width, text_height = text_size

    text_x = x1 + 4
    if text_position == "inside":
        text_y = min(max(y1 + text_height + 4, text_height + 4), frame.shape[0] - 5)
    elif text_position == "below":
        text_y = min(frame.shape[0] - 5, y2 + text_height + 8)
        if text_y + baseline > frame.shape[0] - 1:
            text_y = max(text_height + 4, y1 - 8)
    else:
        text_y = max(text_height + 4, y1 - 8)
        if text_y - text_height < 0:
            text_y = min(frame.shape[0] - 5, y2 + text_height + 8)

    if bg_color is not None:
        padding = max(0, bg_padding)
        bg_x1 = max(0, text_x - padding)
        bg_y1 = max(0, text_y - text_height - padding)
        bg_x2 = min(frame.shape[1] - 1, text_x + text_width + padding)
        bg_y2 = min(frame.shape[0] - 1, text_y + baseline + padding)
        opacity = max(0, min(bg_opacity, 100)) / 100.0
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
        cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_color,
        thickness,
        cv2.LINE_AA,
    )
