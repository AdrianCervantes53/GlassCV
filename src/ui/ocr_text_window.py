from PyQt6.QtCore import Qt
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - OCR Text")
        self.resize(500, 300)
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
        self.set_translation_visible(False)

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

    def _copy_original(self):
        QApplication.clipboard().setText(self.original_panel["text"].toPlainText())

    def _copy_translation(self):
        QApplication.clipboard().setText(self.translation_panel["text"].toPlainText())
