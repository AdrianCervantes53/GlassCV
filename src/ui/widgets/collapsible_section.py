from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QFrame, QPushButton, QVBoxLayout, QWidget


class CollapsibleSection(QWidget):
    """Compact section container with a clickable header."""

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self._title = title

        self.toggle_button = QPushButton()
        self.toggle_button.setObjectName("collapsibleHeader")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.clicked.connect(self._on_toggled)

        self.content = QWidget()

        self.container = QFrame()
        self.container.setObjectName("collapsibleContent")
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(8, 8, 8, 8)
        container_layout.addWidget(self.content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.container)

        self._sync_state(expanded)

    def set_content_layout(self, layout):
        self.content.setLayout(layout)

    def _on_toggled(self, checked: bool):
        self._sync_state(checked)

    def _sync_state(self, expanded: bool):
        arrow = "v" if expanded else ">"
        self.toggle_button.setText(f"{arrow} {self._title}")
        self.container.setVisible(expanded)
