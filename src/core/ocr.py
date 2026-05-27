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


def draw_text_overlay(
    frame: np.ndarray,
    bbox,
    text: str,
    font_scale: float,
    font_color: tuple[int, int, int],
    bg_color: tuple[int, int, int] | None = None,
    bg_opacity: int = 70,
) -> None:
    """Draw a readable OCR text overlay inside the original bounding box."""
    points = [(int(point[0]), int(point[1])) for point in bbox]
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x1, x2 = max(0, min(x_values)), min(frame.shape[1] - 1, max(x_values))
    y1, y2 = max(0, min(y_values)), min(frame.shape[0] - 1, max(y_values))

    if bg_color is None:
        cv2.rectangle(frame, (x1, y1), (x2, y2), font_color, 2)
        cv2.putText(
            frame,
            text,
            (x1, max(15, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            font_color,
            2,
            cv2.LINE_AA,
        )
        return

    opacity = max(0, min(bg_opacity, 100)) / 100.0
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)

    text_y = min(max(y1 + int(24 * font_scale), 15), frame.shape[0] - 5)
    cv2.putText(
        frame,
        text,
        (x1 + 4, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_color,
        2,
        cv2.LINE_AA,
    )
