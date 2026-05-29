from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class OcrTextWindow(QWidget):
    translate_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - OCR Text")
        self.resize(500, 300)
        self._is_translating = False
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(10)

        self.original_panel = self._build_text_panel("Original", "Copy Original")
        self.translation_panel = self._build_text_panel("Translation", "Copy Translation")

        main_layout.addWidget(self.original_panel["container"])
        main_layout.addWidget(self.translation_panel["container"])

        self.original_panel["button"].clicked.connect(self._copy_original)
        self.translation_panel["button"].clicked.connect(self._copy_translation)
        self.btn_translate.clicked.connect(self.translate_requested.emit)
        self.set_translation_visible(False)
        self._update_translate_button()

    def _build_text_panel(self, title: str, button_text: str) -> dict:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        text_edit = QTextEdit()
        text_edit.setReadOnly(True)

        button = QPushButton(button_text)

        layout.addWidget(label)
        layout.addWidget(text_edit, stretch=1)

        if title == "Original":
            self.btn_translate = QPushButton("Translate")
            self.lbl_status = QLabel("")
            self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(self.btn_translate)
            layout.addWidget(self.lbl_status)

        layout.addWidget(button)

        return {"container": container, "text": text_edit, "button": button}

    def _apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #222;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                font-weight: bold;
            }
            QTextEdit {
                background-color: #151515;
                border: 1px solid #444;
                border-radius: 4px;
                padding: 8px;
                color: #f0f0f0;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 6px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
        """)

    def set_translation_visible(self, visible: bool):
        self.translation_panel["container"].setVisible(visible)

    def update_texts(self, original: str, translated: str | None):
        self.original_panel["text"].setPlainText(original or "")
        self.translation_panel["text"].setPlainText(translated or "")
        self.set_translation_visible(bool(translated))
        if not self._is_translating and not original:
            self.lbl_status.setText("No OCR text detected.")
        elif not self._is_translating and translated:
            self.lbl_status.setText("Translation ready.")
        self._update_translate_button()

    def set_translating(self, is_translating: bool):
        self._is_translating = is_translating
        self.lbl_status.setText("Translating..." if is_translating else "")
        self._update_translate_button()

    def set_translation_status(self, message: str):
        self.lbl_status.setText(message)
        self._update_translate_button()

    def _copy_original(self):
        QApplication.clipboard().setText(self.original_panel["text"].toPlainText())

    def _copy_translation(self):
        QApplication.clipboard().setText(self.translation_panel["text"].toPlainText())

    def _update_translate_button(self):
        has_text = bool(self.original_panel["text"].toPlainText().strip())
        self.btn_translate.setEnabled(has_text and not self._is_translating)
