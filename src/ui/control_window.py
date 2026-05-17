from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QPushButton, QSlider, QComboBox, QGroupBox, QFileDialog, QCheckBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap

class ControlWindow(QWidget):
    # Señales para comunicar con la lógica principal
    mode_changed = pyqtSignal(bool) # True = Continuo, False = Snapshot
    snapshot_requested = pyqtSignal()
    fps_changed = pyqtSignal(int)
    filter_changed = pyqtSignal(str)
    glass_pinned = pyqtSignal(bool)
    mirror_mode_changed = pyqtSignal(bool)
    save_requested = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - Control Panel")
        self.setMinimumSize(400, 500)
        self._current_pixmap = None
        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Visor de imagen
        self.image_label = QLabel("Esperando captura...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumHeight(300)
        self.image_label.setStyleSheet("background-color: #222; color: #aaa; border: 1px solid #444; border-radius: 4px;")
        main_layout.addWidget(self.image_label)

        # Panel de Controles
        controls_layout = QVBoxLayout()
        
        # Grupo: Modo y Captura
        group_capture = QGroupBox("Modo de Captura")
        capture_layout = QHBoxLayout()
        
        self.btn_mode = QPushButton("Modo: Continuo")
        self.btn_mode.setCheckable(True)
        self.btn_mode.setChecked(True)
        self.btn_mode.toggled.connect(self._on_mode_toggled)
        capture_layout.addWidget(self.btn_mode)
        
        self.btn_snapshot = QPushButton("Tomar Snapshot")
        self.btn_snapshot.setEnabled(False) # Solo activo en modo manual
        self.btn_snapshot.clicked.connect(self.snapshot_requested.emit)
        capture_layout.addWidget(self.btn_snapshot)
        
        group_capture.setLayout(capture_layout)
        controls_layout.addWidget(group_capture)
        
        # Grupo: Rendimiento (FPS)
        group_fps = QGroupBox("Rendimiento (FPS Objetivo)")
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

        # Grupo: Comportamiento Glass
        group_glass = QGroupBox("Comportamiento del Glass")
        glass_layout = QHBoxLayout()
        
        self.chk_pin = QCheckBox("Fijar Ventana (Click-Through)")
        self.chk_pin.toggled.connect(self.glass_pinned.emit)
        
        self.chk_mirror = QCheckBox("Espejo en Glass")
        self.chk_mirror.toggled.connect(self.mirror_mode_changed.emit)
        
        glass_layout.addWidget(self.chk_pin)
        glass_layout.addWidget(self.chk_mirror)
        group_glass.setLayout(glass_layout)
        controls_layout.addWidget(group_glass)

        # Grupo: Filtros y Guardado
        group_filters = QGroupBox("Filtros y Salida")
        filters_layout = QHBoxLayout()
        
        self.combo_filters = QComboBox()
        self.combo_filters.addItems(["normal", "grayscale", "canny"])
        self.combo_filters.currentTextChanged.connect(self.filter_changed.emit)
        
        self.btn_save = QPushButton("Guardar Imagen")
        self.btn_save.clicked.connect(self._on_save_clicked)
        
        filters_layout.addWidget(QLabel("Filtro:"))
        filters_layout.addWidget(self.combo_filters)
        filters_layout.addWidget(self.btn_save)
        group_filters.setLayout(filters_layout)
        controls_layout.addWidget(group_filters)

        main_layout.addLayout(controls_layout)

    def _apply_styles(self):
        # Estilos básicos oscuros para darle un toque más premium
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
            self.btn_mode.setText("Modo: Continuo")
            self.btn_snapshot.setEnabled(False)
        else:
            self.btn_mode.setText("Modo: Snapshot")
            self.btn_snapshot.setEnabled(True)
        self.mode_changed.emit(checked)

    def _on_fps_changed(self, value: int):
        self.lbl_fps_val.setText(str(value))
        self.fps_changed.emit(value)

    def _on_save_clicked(self):
        if self._current_pixmap:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Guardar Captura", "", "Images (*.png *.jpg)"
            )
            if file_name:
                self.save_requested.emit(file_name)

    def update_image(self, qimg):
        """Actualiza la imagen mostrada en el panel de control."""
        pixmap = QPixmap.fromImage(qimg)
        self._current_pixmap = pixmap
        # Escalar la imagen al tamaño del label preservando la relación de aspecto
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def save_current_pixmap(self, file_name: str):
        if self._current_pixmap:
            self._current_pixmap.save(file_name)
