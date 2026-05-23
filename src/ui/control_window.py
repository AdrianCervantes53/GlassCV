from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSlider, QComboBox, QGroupBox, QFileDialog, QCheckBox,
    QStackedWidget
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

class ControlWindow(QWidget):
    # Signals to communicate with main logic
    mode_changed = pyqtSignal(bool) # True = Continuous, False = Snapshot
    snapshot_requested = pyqtSignal()
    fps_changed = pyqtSignal(int)
    filter_changed = pyqtSignal(str)
    glass_pinned = pyqtSignal(bool)
    mirror_mode_changed = pyqtSignal(bool)
    save_requested = pyqtSignal(str)
    border_toggled = pyqtSignal(bool)
    filter_params_changed = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - Control Panel")
        self.setMinimumSize(400, 500)
        self._current_pixmap = None
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Image Viewer
        self.image_label = QLabel("Waiting for capture...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("background-color: #222; color: #aaa; border: 1px solid #444; border-radius: 4px;")
        main_layout.addWidget(self.image_label)

        # Controls Panel
        controls_layout = QVBoxLayout()
        
        # Group: Mode and Capture
        group_capture = QGroupBox("Capture Mode")
        capture_layout = QHBoxLayout()
        
        self.btn_mode = QPushButton("Mode: Continuous")
        self.btn_mode.setCheckable(True)
        self.btn_mode.setChecked(True)
        self.btn_mode.toggled.connect(self._on_mode_toggled)
        capture_layout.addWidget(self.btn_mode)
        
        self.btn_snapshot = QPushButton("Take Snapshot")
        self.btn_snapshot.setEnabled(False) # Only active in manual mode
        self.btn_snapshot.clicked.connect(self.snapshot_requested.emit)
        capture_layout.addWidget(self.btn_snapshot)
        
        group_capture.setLayout(capture_layout)
        controls_layout.addWidget(group_capture)
        
        # Group: Performance (FPS)
        group_fps = QGroupBox("Performance (Target FPS)")
        fps_layout = QHBoxLayout()
        
        self.slider_fps = QSlider(Qt.Orientation.Horizontal)
        self.slider_fps.setMinimum(10)
        self.slider_fps.setMaximum(60)
        self.slider_fps.setValue(30)
        self.slider_fps.valueChanged.connect(self._on_fps_changed)
        
        self.lbl_fps_val = QLabel("30")
        
        fps_layout.addWidget(self.slider_fps)
        fps_layout.addWidget(self.lbl_fps_val)
        group_fps.setLayout(fps_layout)
        controls_layout.addWidget(group_fps)

        # Group: Glass Behavior
        group_glass = QGroupBox("Glass Behavior")
        glass_layout = QHBoxLayout()
        
        self.chk_pin = QCheckBox("Pin Window (Click-Through)")
        self.chk_pin.toggled.connect(self.glass_pinned.emit)
        
        self.chk_mirror = QCheckBox("Mirror on Glass")
        self.chk_mirror.toggled.connect(self.mirror_mode_changed.emit)

        self.chk_border = QCheckBox("Show Border")
        self.chk_border.setChecked(True)
        self.chk_border.toggled.connect(self.border_toggled.emit)
        
        glass_layout.addWidget(self.chk_pin)
        glass_layout.addWidget(self.chk_mirror)
        glass_layout.addWidget(self.chk_border)
        group_glass.setLayout(glass_layout)
        controls_layout.addWidget(group_glass)

        # Group: Filters and Output
        group_filters = QGroupBox("Filters and Output")
        filters_main_layout = QVBoxLayout()
        
        top_filters_layout = QHBoxLayout()
        self.combo_filters = QComboBox()
        self.combo_filters.addItems(["normal", "grayscale", "canny", "mirror"])
        self.combo_filters.currentTextChanged.connect(self._on_filter_changed)
        
        self.btn_save = QPushButton("Save Image")
        self.btn_save.clicked.connect(self._on_save_clicked)
        
        top_filters_layout.addWidget(QLabel("Filter:"))
        top_filters_layout.addWidget(self.combo_filters)
        top_filters_layout.addWidget(self.btn_save)
        filters_main_layout.addLayout(top_filters_layout)
        
        # Stacked Widget for parameters
        self.stacked_params = QStackedWidget()
        
        # Empty page
        self.empty_page = QWidget()
        self.stacked_params.addWidget(self.empty_page)
        
        # Canny page
        self.canny_page = QWidget()
        canny_layout = QVBoxLayout(self.canny_page)
        canny_layout.setContentsMargins(0, 5, 0, 0)
        
        t1_layout = QHBoxLayout()
        self.lbl_canny_t1 = QLabel("Threshold 1 (0-500): 100")
        self.slider_canny_t1 = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_t1.setRange(0, 500)
        self.slider_canny_t1.setValue(100)
        self.slider_canny_t1.valueChanged.connect(self._on_canny_params_changed)
        t1_layout.addWidget(self.lbl_canny_t1)
        t1_layout.addWidget(self.slider_canny_t1)
        
        t2_layout = QHBoxLayout()
        self.lbl_canny_t2 = QLabel("Threshold 2 (0-500): 200")
        self.slider_canny_t2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_t2.setRange(0, 500)
        self.slider_canny_t2.setValue(200)
        self.slider_canny_t2.valueChanged.connect(self._on_canny_params_changed)
        t2_layout.addWidget(self.lbl_canny_t2)
        t2_layout.addWidget(self.slider_canny_t2)
        
        canny_layout.addLayout(t1_layout)
        canny_layout.addLayout(t2_layout)
        self.stacked_params.addWidget(self.canny_page)
        
        filters_main_layout.addWidget(self.stacked_params)
        
        group_filters.setLayout(filters_main_layout)
        controls_layout.addWidget(group_filters)

        main_layout.addLayout(controls_layout)

    def _apply_styles(self):
        # Basic dark styles to give a more premium touch
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 1ex;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 6px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #4d4d4d;
            }
            QPushButton:checked {
                background-color: #2b5797;
                border: 1px solid #366cb5;
            }
            QPushButton:disabled {
                background-color: #222;
                color: #555;
            }
            QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #999999;
                height: 8px;
                background: #444;
                margin: 2px 0;
            }
            QSlider::handle:horizontal {
                background: #e0e0e0;
                border: 1px solid #5c5c5c;
                width: 14px;
                margin: -4px 0; 
                border-radius: 2px;
            }
        """)

    def _on_mode_toggled(self, checked: bool):
        if checked:
            self.btn_mode.setText("Mode: Continuous")
            self.btn_snapshot.setEnabled(False)
        else:
            self.btn_mode.setText("Mode: Snapshot")
            self.btn_snapshot.setEnabled(True)
        self.mode_changed.emit(checked)

    def _on_fps_changed(self, value: int):
        self.lbl_fps_val.setText(str(value))
        self.fps_changed.emit(value)

    def _on_filter_changed(self, filter_name: str):
        self.filter_changed.emit(filter_name)
        if filter_name == "canny":
            self.stacked_params.setCurrentWidget(self.canny_page)
            self._on_canny_params_changed()
        else:
            self.stacked_params.setCurrentWidget(self.empty_page)

    def _on_canny_params_changed(self):
        t1 = self.slider_canny_t1.value()
        t2 = self.slider_canny_t2.value()
        self.lbl_canny_t1.setText(f"Threshold 1 (0-500): {t1}")
        self.lbl_canny_t2.setText(f"Threshold 2 (0-500): {t2}")
        self.filter_params_changed.emit({"canny_t1": t1, "canny_t2": t2})

    def _on_save_clicked(self):
        if self._current_pixmap:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Capture", "", "Images (*.png *.jpg)"
            )
            if file_name:
                self.save_requested.emit(file_name)

    def update_image(self, qimg):
        """Updates the image shown in the control panel."""
        pixmap = QPixmap.fromImage(qimg)
        self._current_pixmap = pixmap
        # Scale the image to the label size preserving the aspect ratio
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def save_current_pixmap(self, file_name: str):
        if self._current_pixmap:
            self._current_pixmap.save(file_name)
