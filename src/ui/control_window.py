from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSlider, QComboBox, QFileDialog, QCheckBox,
    QStackedWidget, QApplication, QListWidget, QAbstractItemView, QListWidgetItem,
    QColorDialog, QSpinBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPixmap

from core.ocr import OCR_LANGUAGES, TRANSLATION_LANGUAGES
from core.processing import FILTER_DISPLAY_NAMES
from ui.widgets.collapsible_section import CollapsibleSection


# Maps internal filter name -> the params it emits
_FILTER_PARAM_KEYS = {
    "canny":          ["canny_t1", "canny_t2"],
    "rgb_mixer":      ["r_mult", "g_mult", "b_mult"],
    "binary":         ["binary_threshold"],
    "pixelated":      ["pixel_size"],
    "colorblind":     ["cb_type"],
    "object_counter": ["confidence"],
    "smart_inverter": ["intensity"],
    "symmetry":       ["symmetry_axis"],
    "yolo":           ["yolo_model", "yolo_conf", "yolo_iou", "yolo_labels", "yolo_show_conf"],
    "ocr":            [
        "ocr_langs", "ocr_conf", "ocr_font_color", "ocr_font_size",
        "ocr_font_thickness", "ocr_text_position", "ocr_show_text", "ocr_show_boxes",
        "ocr_box_thickness", "ocr_text_background", "ocr_overlay_text_source",
        "ocr_translate_target", "ocr_subtitle_bg_color", "ocr_subtitle_bg_opacity",
    ],
}


class ControlWindow(QWidget):
    # Signals to communicate with main logic
    mode_changed = pyqtSignal(bool)       # True = Continuous, False = Snapshot
    snapshot_requested = pyqtSignal()
    fps_changed = pyqtSignal(int)
    # New chain signal — emits the full chain list
    filter_chain_changed = pyqtSignal(list)
    # Per-filter params signal — emits (filter_name, params_dict)
    filter_params_changed_for = pyqtSignal(str, dict)
    # Legacy signals kept for compatibility
    filter_changed = pyqtSignal(str)
    filter_params_changed = pyqtSignal(dict)
    glass_pinned = pyqtSignal(bool)
    mirror_mode_changed = pyqtSignal(bool)
    save_requested = pyqtSignal(str)
    border_toggled = pyqtSignal(bool)
    toggle_template_glass = pyqtSignal(bool)
    request_template_capture = pyqtSignal()

    ALL_FILTERS = [
        "normal", "grayscale", "canny", "mirror", "symmetry",
        "rgb_mixer", "binary", "pixelated",
        "colorblind", "object_counter", "smart_inverter",
        "yolo", "ocr",
    ]

    AVAILABLE_YOLO_MODELS = [
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt"
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - Control Panel")
        self.setMinimumSize(900, 640)
        self._current_pixmap = None
        self._ocr_font_color = (255, 255, 255)
        self._ocr_subtitle_bg_color = (0, 0, 0)
        self._setup_ui()
        self._apply_styles()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)

        # ── Image Viewer ──────────────────────────────────────────────
        self.image_label = QLabel("Waiting for capture...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(360, 280)
        self.image_label.setStyleSheet(
            "background-color: #222; color: #aaa; border: 1px solid #444; border-radius: 4px;"
        )

        controls_panel = QWidget()
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        # ── Capture Mode ──────────────────────────────────────────────
        section_capture = CollapsibleSection("Capture", expanded=True)
        capture_layout = QHBoxLayout()
        capture_layout.setContentsMargins(0, 0, 0, 0)
        self.btn_mode = QPushButton("Mode: Continuous")
        self.btn_mode.setCheckable(True)
        self.btn_mode.setChecked(True)
        self.btn_mode.toggled.connect(self._on_mode_toggled)
        capture_layout.addWidget(self.btn_mode)
        self.btn_snapshot = QPushButton("Take Snapshot")
        self.btn_snapshot.setEnabled(False)
        self.btn_snapshot.clicked.connect(self.snapshot_requested.emit)
        capture_layout.addWidget(self.btn_snapshot)
        section_capture.set_content_layout(capture_layout)
        controls_layout.addWidget(section_capture)

        # ── Performance (FPS) ─────────────────────────────────────────
        section_fps = CollapsibleSection("Performance", expanded=False)
        fps_layout = QHBoxLayout()
        fps_layout.setContentsMargins(0, 0, 0, 0)
        self.slider_fps = QSlider(Qt.Orientation.Horizontal)
        self.slider_fps.setMinimum(10)
        self.slider_fps.setMaximum(60)
        self.slider_fps.setValue(30)
        self.slider_fps.valueChanged.connect(self._on_fps_changed)
        self.lbl_fps_val = QLabel("30")
        fps_layout.addWidget(self.slider_fps)
        fps_layout.addWidget(self.lbl_fps_val)
        section_fps.set_content_layout(fps_layout)
        controls_layout.addWidget(section_fps)

        # ── Glass Behavior ────────────────────────────────────────────
        section_glass = CollapsibleSection("Glass", expanded=False)
        glass_layout = QVBoxLayout()
        glass_layout.setContentsMargins(0, 0, 0, 0)
        glass_options_row = QHBoxLayout()
        glass_status_row = QHBoxLayout()
        self.chk_pin = QCheckBox("Pin (Click-Through)")
        self.chk_pin.toggled.connect(self.glass_pinned.emit)
        self.chk_mirror = QCheckBox("Mirror on Glass")
        self.chk_mirror.toggled.connect(self.mirror_mode_changed.emit)
        self.chk_border = QCheckBox("Show Border")
        self.chk_border.setChecked(True)
        self.chk_border.toggled.connect(self.border_toggled.emit)
        
        self.lbl_glass_size = QLabel("Size: N/A")
        
        glass_options_row.addWidget(self.chk_pin)
        glass_options_row.addWidget(self.chk_mirror)
        glass_options_row.addWidget(self.chk_border)
        glass_status_row.addWidget(self.lbl_glass_size)
        glass_status_row.addStretch()
        glass_layout.addLayout(glass_options_row)
        glass_layout.addLayout(glass_status_row)
        section_glass.set_content_layout(glass_layout)
        controls_layout.addWidget(section_glass)

        # ── Filters ───────────────────────────────────────────────────
        section_filters = CollapsibleSection("Filters", expanded=True)
        filters_main_layout = QVBoxLayout()
        filters_main_layout.setContentsMargins(0, 0, 0, 0)
        filters_main_layout.setSpacing(6)
        # Row 1: combo + Add button
        add_row = QHBoxLayout()
        self.combo_add_filter = QComboBox()
        for name in self.ALL_FILTERS:
            self.combo_add_filter.addItem(FILTER_DISPLAY_NAMES.get(name, name), userData=name)
        self.btn_add_filter = QPushButton("＋ Add")
        self.btn_add_filter.setFixedWidth(70)
        self.btn_add_filter.clicked.connect(self._on_add_filter)
        add_row.addWidget(QLabel("Filter:"))
        add_row.addWidget(self.combo_add_filter, stretch=1)
        add_row.addWidget(self.btn_add_filter)
        filters_main_layout.addLayout(add_row)
        # Row 2: Chain list (drag & drop) + Remove button
        chain_row = QHBoxLayout()
        self.list_chain = QListWidget()
        self.list_chain.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.list_chain.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_chain.setFixedHeight(100)
        self.list_chain.setToolTip("Drag to reorder. Select a filter to edit its parameters below.")
        self.list_chain.currentRowChanged.connect(self._on_chain_selection_changed)
        # Reorder detection via model signals
        self.list_chain.model().rowsMoved.connect(self._on_chain_reordered)
        chain_row.addWidget(self.list_chain, stretch=1)
        chain_btns = QVBoxLayout()
        self.btn_remove_filter = QPushButton("✕")
        self.btn_remove_filter.setFixedWidth(32)
        self.btn_remove_filter.setToolTip("Remove selected filter")
        self.btn_remove_filter.clicked.connect(self._on_remove_filter)
        self.btn_move_up = QPushButton("▲")
        self.btn_move_up.setFixedWidth(32)
        self.btn_move_up.setToolTip("Move up")
        self.btn_move_up.clicked.connect(self._on_move_up)
        self.btn_move_down = QPushButton("▼")
        self.btn_move_down.setFixedWidth(32)
        self.btn_move_down.setToolTip("Move down")
        self.btn_move_down.clicked.connect(self._on_move_down)
        chain_btns.addWidget(self.btn_move_up)
        chain_btns.addWidget(self.btn_move_down)
        chain_btns.addWidget(self.btn_remove_filter)
        chain_btns.addStretch()
        chain_row.addLayout(chain_btns)
        filters_main_layout.addLayout(chain_row)
        # Row 3: Save / Copy buttons
        output_row = QHBoxLayout()
        self.btn_save = QPushButton("Save Image")
        self.btn_save.clicked.connect(self._on_save_clicked)
        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self._on_copy_clicked)
        output_row.addStretch()
        output_row.addWidget(self.btn_save)
        output_row.addWidget(self.btn_copy)
        filters_main_layout.addLayout(output_row)
        # Row 4: Parameter panel (stacked)
        self.stacked_params = QStackedWidget()
        self._build_param_pages()
        filters_main_layout.addWidget(self.stacked_params)
        section_filters.set_content_layout(filters_main_layout)
        controls_layout.addWidget(section_filters)
        controls_layout.addStretch()

        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        controls_scroll.setWidget(controls_panel)
        controls_scroll.setMinimumWidth(360)

        main_layout.addWidget(controls_scroll, stretch=2)
        main_layout.addWidget(self.image_label, stretch=3)
        
    # ------------------------------------------------------------------
    # Parameter pages (one per filter that has parameters)
    # ------------------------------------------------------------------

    def _build_param_pages(self):
        """Build all parameter sub-pages and add them to stacked_params."""

        # Empty (no params)
        self.empty_page = QWidget()
        self.stacked_params.addWidget(self.empty_page)

        # ── Canny ─────────────────────────────────────────────────────
        self.canny_page = QWidget()
        canny_layout = QVBoxLayout(self.canny_page)
        canny_layout.setContentsMargins(0, 4, 0, 0)
        t1_row = QHBoxLayout()
        self.lbl_canny_t1 = QLabel("Threshold 1 (0-500): 100")
        self.slider_canny_t1 = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_t1.setRange(0, 500)
        self.slider_canny_t1.setValue(100)
        self.slider_canny_t1.valueChanged.connect(self._on_canny_params_changed)
        t1_row.addWidget(self.lbl_canny_t1)
        t1_row.addWidget(self.slider_canny_t1)
        t2_row = QHBoxLayout()
        self.lbl_canny_t2 = QLabel("Threshold 2 (0-500): 200")
        self.slider_canny_t2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_canny_t2.setRange(0, 500)
        self.slider_canny_t2.setValue(200)
        self.slider_canny_t2.valueChanged.connect(self._on_canny_params_changed)
        t2_row.addWidget(self.lbl_canny_t2)
        t2_row.addWidget(self.slider_canny_t2)
        canny_layout.addLayout(t1_row)
        canny_layout.addLayout(t2_row)
        self.stacked_params.addWidget(self.canny_page)

        # ── RGB Mixer ─────────────────────────────────────────────────
        self.rgb_page = QWidget()
        rgb_layout = QVBoxLayout(self.rgb_page)
        for color, attr in [("Red", "r"), ("Green", "g"), ("Blue", "b")]:
            row = QHBoxLayout()
            lbl = QLabel(f"{color} (0-200%): 100")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(0, 200)
            slider.setValue(100)
            setattr(self, f"lbl_{attr}", lbl)
            setattr(self, f"slider_{attr}", slider)
            slider.valueChanged.connect(self._on_rgb_params_changed)
            row.addWidget(lbl)
            row.addWidget(slider)
            rgb_layout.addLayout(row)
        self.stacked_params.addWidget(self.rgb_page)

        # ── Binary ────────────────────────────────────────────────────
        self.binary_page = QWidget()
        bin_layout = QVBoxLayout(self.binary_page)
        bin_row = QHBoxLayout()
        self.lbl_bin = QLabel("Threshold (0-255): 127")
        self.slider_bin = QSlider(Qt.Orientation.Horizontal)
        self.slider_bin.setRange(0, 255)
        self.slider_bin.setValue(127)
        self.slider_bin.valueChanged.connect(self._on_binary_params_changed)
        bin_row.addWidget(self.lbl_bin)
        bin_row.addWidget(self.slider_bin)
        bin_layout.addLayout(bin_row)
        self.stacked_params.addWidget(self.binary_page)

        # ── Pixelated ─────────────────────────────────────────────────
        self.pix_page = QWidget()
        pix_layout = QVBoxLayout(self.pix_page)
        pix_row = QHBoxLayout()
        self.lbl_pix = QLabel("Pixel Size (2-50): 10")
        self.slider_pix = QSlider(Qt.Orientation.Horizontal)
        self.slider_pix.setRange(2, 50)
        self.slider_pix.setValue(10)
        self.slider_pix.valueChanged.connect(self._on_pix_params_changed)
        pix_row.addWidget(self.lbl_pix)
        pix_row.addWidget(self.slider_pix)
        pix_layout.addLayout(pix_row)
        self.stacked_params.addWidget(self.pix_page)

        # ── Colorblind ────────────────────────────────────────────────
        self.cb_page = QWidget()
        cb_layout = QHBoxLayout(self.cb_page)
        self.combo_cb = QComboBox()
        self.combo_cb.addItems(["protanopia", "deuteranopia", "tritanopia"])
        self.combo_cb.currentTextChanged.connect(self._on_cb_params_changed)
        cb_layout.addWidget(QLabel("Type:"))
        cb_layout.addWidget(self.combo_cb)
        self.stacked_params.addWidget(self.cb_page)

        # ── Object Counter ────────────────────────────────────────────
        self.obj_page = QWidget()
        obj_layout = QVBoxLayout(self.obj_page)
        btns_row = QHBoxLayout()
        self.btn_toggle_tg = QPushButton("Show Template Glass")
        self.btn_toggle_tg.setCheckable(True)
        self.btn_toggle_tg.toggled.connect(self._on_toggle_tg)
        self.btn_cap_tg = QPushButton("Capture Template")
        self.btn_cap_tg.clicked.connect(self.request_template_capture.emit)
        btns_row.addWidget(self.btn_toggle_tg)
        btns_row.addWidget(self.btn_cap_tg)
        conf_row = QHBoxLayout()
        self.lbl_conf = QLabel("Confidence (1-100%): 80")
        self.slider_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_conf.setRange(1, 100)
        self.slider_conf.setValue(80)
        self.slider_conf.valueChanged.connect(self._on_obj_params_changed)
        conf_row.addWidget(self.lbl_conf)
        conf_row.addWidget(self.slider_conf)
        obj_layout.addLayout(btns_row)
        obj_layout.addLayout(conf_row)
        self.stacked_params.addWidget(self.obj_page)

        # ── Smart Inverter ────────────────────────────────────────────
        self.inv_page = QWidget()
        inv_layout = QVBoxLayout(self.inv_page)
        inv_row = QHBoxLayout()
        self.lbl_inv = QLabel("Intensity (0-100%): 100")
        self.slider_inv = QSlider(Qt.Orientation.Horizontal)
        self.slider_inv.setRange(0, 100)
        self.slider_inv.setValue(100)
        self.slider_inv.valueChanged.connect(self._on_inv_params_changed)
        inv_row.addWidget(self.lbl_inv)
        inv_row.addWidget(self.slider_inv)
        inv_layout.addLayout(inv_row)
        self.stacked_params.addWidget(self.inv_page)

        # ── Symmetry ──────────────────────────────────────────────────
        self.sym_page = QWidget()
        sym_layout = QHBoxLayout(self.sym_page)
        self.combo_sym = QComboBox()
        self.combo_sym.addItems(["vertical", "horizontal"])
        self.combo_sym.currentTextChanged.connect(self._on_sym_params_changed)
        sym_layout.addWidget(QLabel("Axis:"))
        sym_layout.addWidget(self.combo_sym)
        self.stacked_params.addWidget(self.sym_page)

        # ── YOLO ──────────────────────────────────────────────────────
        self.yolo_page = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_page)
        
        # Model selector
        yolo_model_row = QHBoxLayout()
        self.combo_yolo_model = QComboBox()
        self.combo_yolo_model.activated.connect(self._on_yolo_combo_activated)
        self.btn_yolo_custom = QPushButton("Custom...")
        self.btn_yolo_custom.setFixedWidth(70)
        self.btn_yolo_custom.clicked.connect(self._on_yolo_select_custom)
        yolo_model_row.addWidget(QLabel("Model:"))
        yolo_model_row.addWidget(self.combo_yolo_model, stretch=1)
        yolo_model_row.addWidget(self.btn_yolo_custom)
        
        # Confidence slider
        yolo_conf_row = QHBoxLayout()
        self.lbl_yolo_conf = QLabel("Conf (1-100%): 50")
        self.slider_yolo_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_yolo_conf.setRange(1, 100)
        self.slider_yolo_conf.setValue(50)
        self.slider_yolo_conf.valueChanged.connect(self._on_yolo_params_changed)
        yolo_conf_row.addWidget(self.lbl_yolo_conf)
        yolo_conf_row.addWidget(self.slider_yolo_conf)

        # IOU slider
        yolo_iou_row = QHBoxLayout()
        self.lbl_yolo_iou = QLabel("IOU (1-100%): 45")
        self.slider_yolo_iou = QSlider(Qt.Orientation.Horizontal)
        self.slider_yolo_iou.setRange(1, 100)
        self.slider_yolo_iou.setValue(45)
        self.slider_yolo_iou.valueChanged.connect(self._on_yolo_params_changed)
        yolo_iou_row.addWidget(self.lbl_yolo_iou)
        yolo_iou_row.addWidget(self.slider_yolo_iou)
        
        # Checkboxes for labels
        yolo_chk_row = QHBoxLayout()
        self.chk_yolo_labels = QCheckBox("Show Labels")
        self.chk_yolo_labels.setChecked(True)
        self.chk_yolo_labels.toggled.connect(self._on_yolo_params_changed)
        self.chk_yolo_show_conf = QCheckBox("Show Confidences")
        self.chk_yolo_show_conf.setChecked(True)
        self.chk_yolo_show_conf.toggled.connect(self._on_yolo_params_changed)
        yolo_chk_row.addWidget(self.chk_yolo_labels)
        yolo_chk_row.addWidget(self.chk_yolo_show_conf)
        
        yolo_layout.addLayout(yolo_model_row)
        yolo_layout.addLayout(yolo_conf_row)
        yolo_layout.addLayout(yolo_iou_row)
        yolo_layout.addLayout(yolo_chk_row)
        self.stacked_params.addWidget(self.yolo_page)
        
        self._update_yolo_combo()

        # ── OCR ───────────────────────────────────────────────────────
        self.ocr_page = QWidget()
        ocr_layout = QVBoxLayout(self.ocr_page)
        
        # Language combo
        ocr_lang_row = QHBoxLayout()
        self.combo_ocr_lang = QComboBox()
        for name, easyocr_code, _ in OCR_LANGUAGES:
            self.combo_ocr_lang.addItem(f"{name} ({easyocr_code})", userData=easyocr_code)
        self.combo_ocr_lang.currentTextChanged.connect(self._on_ocr_params_changed)
        ocr_lang_row.addWidget(QLabel("Language:"))
        ocr_lang_row.addWidget(self.combo_ocr_lang)
        
        # Confidence slider
        ocr_conf_row = QHBoxLayout()
        self.lbl_ocr_conf = QLabel("Conf (1-100%): 50")
        self.slider_ocr_conf = QSlider(Qt.Orientation.Horizontal)
        self.slider_ocr_conf.setRange(1, 100)
        self.slider_ocr_conf.setValue(50)
        self.slider_ocr_conf.valueChanged.connect(self._on_ocr_params_changed)
        ocr_conf_row.addWidget(self.lbl_ocr_conf)
        ocr_conf_row.addWidget(self.slider_ocr_conf)

        # Font color
        ocr_font_color_row = QHBoxLayout()
        self.btn_ocr_font_color = QPushButton("Font Color")
        self.btn_ocr_font_color.clicked.connect(self._on_ocr_font_color_clicked)
        self._update_color_button(self.btn_ocr_font_color, self._ocr_font_color)
        ocr_font_color_row.addWidget(QLabel("Overlay Text:"))
        ocr_font_color_row.addWidget(self.btn_ocr_font_color)

        # Font size
        ocr_font_size_row = QHBoxLayout()
        self.spin_ocr_font_size = QSpinBox()
        self.spin_ocr_font_size.setRange(8, 72)
        self.spin_ocr_font_size.setValue(16)
        self.spin_ocr_font_size.valueChanged.connect(self._on_ocr_params_changed)
        ocr_font_size_row.addWidget(QLabel("Font Size:"))
        ocr_font_size_row.addWidget(self.spin_ocr_font_size)

        # Font thickness
        ocr_font_thickness_row = QHBoxLayout()
        self.spin_ocr_font_thickness = QSpinBox()
        self.spin_ocr_font_thickness.setRange(1, 5)
        self.spin_ocr_font_thickness.setValue(1)
        self.spin_ocr_font_thickness.valueChanged.connect(self._on_ocr_params_changed)
        ocr_font_thickness_row.addWidget(QLabel("Font Thickness:"))
        ocr_font_thickness_row.addWidget(self.spin_ocr_font_thickness)

        # Text position
        ocr_text_position_row = QHBoxLayout()
        self.combo_ocr_text_position = QComboBox()
        self.combo_ocr_text_position.addItem("Above", userData="above")
        self.combo_ocr_text_position.addItem("Below", userData="below")
        self.combo_ocr_text_position.addItem("On Text", userData="inside")
        self.combo_ocr_text_position.currentTextChanged.connect(self._on_ocr_params_changed)
        ocr_text_position_row.addWidget(QLabel("Text Position:"))
        ocr_text_position_row.addWidget(self.combo_ocr_text_position)

        # Overlay visibility toggles
        ocr_toggle_row = QHBoxLayout()
        self.chk_ocr_show_text = QCheckBox("Show Text")
        self.chk_ocr_show_text.setChecked(True)
        self.chk_ocr_show_text.toggled.connect(self._on_ocr_params_changed)
        self.chk_ocr_show_boxes = QCheckBox("Show Boxes")
        self.chk_ocr_show_boxes.setChecked(True)
        self.chk_ocr_show_boxes.toggled.connect(self._on_ocr_params_changed)
        self.chk_ocr_text_background = QCheckBox("Text Background")
        self.chk_ocr_text_background.toggled.connect(self._on_ocr_text_background_toggled)
        ocr_toggle_row.addWidget(self.chk_ocr_show_text)
        ocr_toggle_row.addWidget(self.chk_ocr_show_boxes)
        ocr_toggle_row.addWidget(self.chk_ocr_text_background)

        # Box thickness
        ocr_box_thickness_row = QHBoxLayout()
        self.spin_ocr_box_thickness = QSpinBox()
        self.spin_ocr_box_thickness.setRange(1, 5)
        self.spin_ocr_box_thickness.setValue(1)
        self.spin_ocr_box_thickness.valueChanged.connect(self._on_ocr_params_changed)
        ocr_box_thickness_row.addWidget(QLabel("Box Thickness:"))
        ocr_box_thickness_row.addWidget(self.spin_ocr_box_thickness)

        # Overlay text source
        ocr_overlay_source_row = QHBoxLayout()
        self.combo_ocr_overlay_source = QComboBox()
        self.combo_ocr_overlay_source.addItem("Original", userData="original")
        self.combo_ocr_overlay_source.addItem("Translation", userData="translation")
        self.combo_ocr_overlay_source.currentTextChanged.connect(self._on_ocr_params_changed)
        ocr_overlay_source_row.addWidget(QLabel("Overlay Text Source:"))
        ocr_overlay_source_row.addWidget(self.combo_ocr_overlay_source)

        # Target language
        ocr_target_row = QHBoxLayout()
        self.combo_ocr_translate_target = QComboBox()
        for name, code in TRANSLATION_LANGUAGES:
            self.combo_ocr_translate_target.addItem(f"{name} ({code})", userData=code)
        self.combo_ocr_translate_target.setCurrentIndex(1)
        self.combo_ocr_translate_target.currentTextChanged.connect(self._on_ocr_params_changed)
        ocr_target_row.addWidget(QLabel("Target Language:"))
        ocr_target_row.addWidget(self.combo_ocr_translate_target)

        # Subtitle background
        ocr_bg_color_row = QHBoxLayout()
        self.btn_ocr_bg_color = QPushButton("Background Color")
        self.btn_ocr_bg_color.clicked.connect(self._on_ocr_bg_color_clicked)
        self._update_color_button(self.btn_ocr_bg_color, self._ocr_subtitle_bg_color)
        ocr_bg_color_row.addWidget(QLabel("Subtitle Bg:"))
        ocr_bg_color_row.addWidget(self.btn_ocr_bg_color)

        ocr_bg_opacity_row = QHBoxLayout()
        self.lbl_ocr_bg_opacity = QLabel("Bg Opacity (0-100%): 70")
        self.slider_ocr_bg_opacity = QSlider(Qt.Orientation.Horizontal)
        self.slider_ocr_bg_opacity.setRange(0, 100)
        self.slider_ocr_bg_opacity.setValue(70)
        self.slider_ocr_bg_opacity.valueChanged.connect(self._on_ocr_params_changed)
        ocr_bg_opacity_row.addWidget(self.lbl_ocr_bg_opacity)
        ocr_bg_opacity_row.addWidget(self.slider_ocr_bg_opacity)
        
        ocr_basic_section = CollapsibleSection("Basic", expanded=True)
        ocr_basic_layout = QVBoxLayout()
        ocr_basic_layout.setContentsMargins(0, 0, 0, 0)
        ocr_basic_layout.addLayout(ocr_lang_row)
        ocr_basic_layout.addLayout(ocr_conf_row)
        ocr_basic_layout.addLayout(ocr_toggle_row)
        ocr_basic_section.set_content_layout(ocr_basic_layout)

        ocr_overlay_section = CollapsibleSection("Overlay", expanded=False)
        ocr_overlay_layout = QVBoxLayout()
        ocr_overlay_layout.setContentsMargins(0, 0, 0, 0)
        ocr_overlay_layout.addLayout(ocr_font_color_row)
        ocr_overlay_layout.addLayout(ocr_font_size_row)
        ocr_overlay_layout.addLayout(ocr_font_thickness_row)
        ocr_overlay_layout.addLayout(ocr_text_position_row)
        ocr_overlay_layout.addLayout(ocr_overlay_source_row)
        ocr_overlay_section.set_content_layout(ocr_overlay_layout)

        ocr_translation_section = CollapsibleSection("Translation", expanded=False)
        ocr_translation_layout = QVBoxLayout()
        ocr_translation_layout.setContentsMargins(0, 0, 0, 0)
        ocr_translation_layout.addLayout(ocr_target_row)
        ocr_translation_section.set_content_layout(ocr_translation_layout)

        ocr_boxes_section = CollapsibleSection("Boxes", expanded=False)
        ocr_boxes_layout = QVBoxLayout()
        ocr_boxes_layout.setContentsMargins(0, 0, 0, 0)
        ocr_boxes_layout.addLayout(ocr_box_thickness_row)
        ocr_boxes_section.set_content_layout(ocr_boxes_layout)

        ocr_background_section = CollapsibleSection("Background", expanded=False)
        ocr_background_layout = QVBoxLayout()
        ocr_background_layout.setContentsMargins(0, 0, 0, 0)
        ocr_background_layout.addLayout(ocr_bg_color_row)
        ocr_background_layout.addLayout(ocr_bg_opacity_row)
        ocr_background_section.set_content_layout(ocr_background_layout)

        ocr_layout.addWidget(ocr_basic_section)
        ocr_layout.addWidget(ocr_overlay_section)
        ocr_layout.addWidget(ocr_translation_section)
        ocr_layout.addWidget(ocr_boxes_section)
        ocr_layout.addWidget(ocr_background_section)
        self.stacked_params.addWidget(self.ocr_page)
        self._on_ocr_text_background_toggled(False)

        # Map filter name -> param page widget
        self._param_page_map = {
            "canny":          self.canny_page,
            "rgb_mixer":      self.rgb_page,
            "binary":         self.binary_page,
            "pixelated":      self.pix_page,
            "colorblind":     self.cb_page,
            "object_counter": self.obj_page,
            "smart_inverter": self.inv_page,
            "symmetry":       self.sym_page,
            "yolo":           self.yolo_page,
            "ocr":            self.ocr_page,
        }

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------

    def _apply_styles(self):
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
            QScrollArea {
                border: none;
            }
            QPushButton#collapsibleHeader {
                background-color: #343434;
                border: 1px solid #4a4a4a;
                border-radius: 3px;
                padding: 6px 8px;
                text-align: left;
                font-weight: bold;
                min-width: 0;
            }
            QPushButton#collapsibleHeader:hover {
                background-color: #3f3f3f;
            }
            QFrame#collapsibleContent {
                border-left: 1px solid #444;
                border-right: 1px solid #444;
                border-bottom: 1px solid #444;
                border-bottom-left-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QPushButton {
                background-color: #3d3d3d;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 6px;
                min-width: 60px;
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
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #555;
                border-radius: 3px;
                color: #e0e0e0;
            }
            QListWidget::item {
                padding: 4px 8px;
                border-bottom: 1px solid #333;
            }
            QListWidget::item:selected {
                background-color: #2b5797;
                color: #fff;
            }
            QListWidget::item:hover {
                background-color: #3a3a3a;
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
        # Style the small chain buttons separately
        small_btn_style = """
            QPushButton { min-width: 28px; padding: 4px 2px; font-size: 11px; }
        """
        self.btn_remove_filter.setStyleSheet(small_btn_style)
        self.btn_move_up.setStyleSheet(small_btn_style)
        self.btn_move_down.setStyleSheet(small_btn_style)

    # ------------------------------------------------------------------
    # Filter Chain helpers
    # ------------------------------------------------------------------

    def _get_chain(self) -> list[dict]:
        """Read the current filter chain from the list widget."""
        chain = []
        for i in range(self.list_chain.count()):
            item = self.list_chain.item(i)
            name = item.data(Qt.ItemDataRole.UserRole)
            chain.append({"name": name, "params": {}})
        return chain

    def _emit_chain(self):
        """Emit the current chain to the processor."""
        self.filter_chain_changed.emit(self._get_chain())

    def _add_item_to_list(self, filter_name: str):
        """Add a filter to the QListWidget."""
        display = FILTER_DISPLAY_NAMES.get(filter_name, filter_name)
        item = QListWidgetItem(display)
        item.setData(Qt.ItemDataRole.UserRole, filter_name)
        item.setToolTip(f"Filter: {filter_name}")
        self.list_chain.addItem(item)

    # ------------------------------------------------------------------
    # Chain slot handlers
    # ------------------------------------------------------------------

    def _on_add_filter(self):
        filter_name = self.combo_add_filter.currentData()
        if filter_name and filter_name != "normal":
            self._add_item_to_list(filter_name)
            # Select the new item
            self.list_chain.setCurrentRow(self.list_chain.count() - 1)
            self._emit_chain()
        elif filter_name == "normal":
            # Normal = clear chain
            self.list_chain.clear()
            self._emit_chain()

    def _on_remove_filter(self):
        row = self.list_chain.currentRow()
        if row >= 0:
            self.list_chain.takeItem(row)
            self._emit_chain()

    def _on_move_up(self):
        row = self.list_chain.currentRow()
        if row > 0:
            item = self.list_chain.takeItem(row)
            self.list_chain.insertItem(row - 1, item)
            self.list_chain.setCurrentRow(row - 1)
            self._emit_chain()

    def _on_move_down(self):
        row = self.list_chain.currentRow()
        if row >= 0 and row < self.list_chain.count() - 1:
            item = self.list_chain.takeItem(row)
            self.list_chain.insertItem(row + 1, item)
            self.list_chain.setCurrentRow(row + 1)
            self._emit_chain()

    def _on_chain_reordered(self, *args):
        """Called when user drag-drops to reorder."""
        self._emit_chain()

    def _on_chain_selection_changed(self, row: int):
        """Show the param page for the selected filter."""
        if row < 0:
            self.stacked_params.setCurrentWidget(self.empty_page)
            return
        item = self.list_chain.item(row)
        if item is None:
            self.stacked_params.setCurrentWidget(self.empty_page)
            return
        filter_name = item.data(Qt.ItemDataRole.UserRole)
        page = self._param_page_map.get(filter_name, self.empty_page)
        self.stacked_params.setCurrentWidget(page)

    # ------------------------------------------------------------------
    # Per-filter param emitters  (emit to the selected filter in chain)
    # ------------------------------------------------------------------

    def _selected_filter_name(self) -> str | None:
        row = self.list_chain.currentRow()
        if row < 0:
            return None
        item = self.list_chain.item(row)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _emit_filter_params(self, params: dict):
        """Emit params for the currently selected filter in the chain."""
        name = self._selected_filter_name()
        if name:
            self.filter_params_changed_for.emit(name, params)
        # Also emit legacy signal
        self.filter_params_changed.emit(params)

    def _on_canny_params_changed(self):
        t1 = self.slider_canny_t1.value()
        t2 = self.slider_canny_t2.value()
        self.lbl_canny_t1.setText(f"Threshold 1 (0-500): {t1}")
        self.lbl_canny_t2.setText(f"Threshold 2 (0-500): {t2}")
        self._emit_filter_params({"canny_t1": t1, "canny_t2": t2})

    def _on_rgb_params_changed(self):
        r = self.slider_r.value()
        g = self.slider_g.value()
        b = self.slider_b.value()
        self.lbl_r.setText(f"Red (0-200%): {r}")
        self.lbl_g.setText(f"Green (0-200%): {g}")
        self.lbl_b.setText(f"Blue (0-200%): {b}")
        self._emit_filter_params({"r_mult": r, "g_mult": g, "b_mult": b})

    def _on_binary_params_changed(self):
        val = self.slider_bin.value()
        self.lbl_bin.setText(f"Threshold (0-255): {val}")
        self._emit_filter_params({"binary_threshold": val})

    def _on_pix_params_changed(self):
        val = self.slider_pix.value()
        self.lbl_pix.setText(f"Pixel Size (2-50): {val}")
        self._emit_filter_params({"pixel_size": val})

    def _on_cb_params_changed(self):
        self._emit_filter_params({"cb_type": self.combo_cb.currentText()})

    def _on_obj_params_changed(self):
        val = self.slider_conf.value()
        self.lbl_conf.setText(f"Confidence (1-100%): {val}")
        self._emit_filter_params({"confidence": val})

    def _on_toggle_tg(self, checked):
        self.btn_toggle_tg.setText("Hide Template Glass" if checked else "Show Template Glass")
        self.toggle_template_glass.emit(checked)

    def _on_inv_params_changed(self):
        val = self.slider_inv.value()
        self.lbl_inv.setText(f"Intensity (0-100%): {val}")
        self._emit_filter_params({"intensity": val})

    def _on_sym_params_changed(self):
        self._emit_filter_params({"symmetry_axis": self.combo_sym.currentText()})

    def _update_yolo_combo(self, select_model=None):
        import os
        current_selection = select_model or self.combo_yolo_model.currentData()
        self.combo_yolo_model.blockSignals(True)
        self.combo_yolo_model.clear()
        
        index_to_select = 0
        for i, model in enumerate(self.AVAILABLE_YOLO_MODELS):
            model_path = os.path.join("models", model)
            if os.path.exists(model_path):
                display = f"{model} (Descargado)"
            else:
                display = model
            self.combo_yolo_model.addItem(display, userData=model_path)
            if current_selection == model_path:
                index_to_select = i
                
        if current_selection and current_selection not in [os.path.join("models", m) for m in self.AVAILABLE_YOLO_MODELS]:
            self.combo_yolo_model.addItem(f"{os.path.basename(current_selection)} (Custom)", userData=current_selection)
            index_to_select = self.combo_yolo_model.count() - 1
            
        self.combo_yolo_model.setCurrentIndex(index_to_select)
        self.combo_yolo_model.blockSignals(False)

    def _on_yolo_combo_activated(self, index):
        self._on_yolo_params_changed()

    def _on_yolo_select_custom(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select YOLO Model", "", "YOLO Models (*.pt);;All Files (*)"
        )
        if file_name:
            self._update_yolo_combo(select_model=file_name)
            self._on_yolo_params_changed()

    def _on_yolo_params_changed(self):
        path = self.combo_yolo_model.currentData()
        if not path:
            import os
            path = os.path.join("models", "yolo11n.pt")
            
        conf = self.slider_yolo_conf.value()
        iou = self.slider_yolo_iou.value()
        self.lbl_yolo_conf.setText(f"Conf (1-100%): {conf}")
        self.lbl_yolo_iou.setText(f"IOU (1-100%): {iou}")
        
        self._emit_filter_params({
            "yolo_model": path,
            "yolo_conf": conf,
            "yolo_iou": iou,
            "yolo_labels": self.chk_yolo_labels.isChecked(),
            "yolo_show_conf": self.chk_yolo_show_conf.isChecked(),
        })
        self._update_yolo_combo()

    def _bgr_to_qcolor(self, bgr: tuple[int, int, int]) -> QColor:
        return QColor(bgr[2], bgr[1], bgr[0])

    def _qcolor_to_bgr(self, color: QColor) -> tuple[int, int, int]:
        return (color.blue(), color.green(), color.red())

    def _update_color_button(self, button: QPushButton, bgr: tuple[int, int, int]):
        color = self._bgr_to_qcolor(bgr)
        button.setStyleSheet(
            f"background-color: {color.name()}; color: #fff; border: 1px solid #777;"
        )

    def _on_ocr_font_color_clicked(self):
        color = QColorDialog.getColor(self._bgr_to_qcolor(self._ocr_font_color), self, "Select OCR Font Color")
        if color.isValid():
            self._ocr_font_color = self._qcolor_to_bgr(color)
            self._update_color_button(self.btn_ocr_font_color, self._ocr_font_color)
            self._on_ocr_params_changed()

    def _on_ocr_bg_color_clicked(self):
        color = QColorDialog.getColor(self._bgr_to_qcolor(self._ocr_subtitle_bg_color), self, "Select Subtitle Background")
        if color.isValid():
            self._ocr_subtitle_bg_color = self._qcolor_to_bgr(color)
            self._update_color_button(self.btn_ocr_bg_color, self._ocr_subtitle_bg_color)
            self._on_ocr_params_changed()

    def _on_ocr_text_background_toggled(self, checked: bool):
        self.btn_ocr_bg_color.setEnabled(checked)
        self.slider_ocr_bg_opacity.setEnabled(checked)
        self._on_ocr_params_changed()

    def _on_ocr_params_changed(self):
        conf = self.slider_ocr_conf.value()
        bg_opacity = self.slider_ocr_bg_opacity.value()
        self.lbl_ocr_conf.setText(f"Conf (1-100%): {conf}")
        self.lbl_ocr_bg_opacity.setText(f"Bg Opacity (0-100%): {bg_opacity}")
        self._emit_filter_params({
            "ocr_langs": [self.combo_ocr_lang.currentData() or "en"],
            "ocr_conf": conf,
            "ocr_font_color": self._ocr_font_color,
            "ocr_font_size": self.spin_ocr_font_size.value(),
            "ocr_font_thickness": self.spin_ocr_font_thickness.value(),
            "ocr_text_position": self.combo_ocr_text_position.currentData() or "above",
            "ocr_show_text": self.chk_ocr_show_text.isChecked(),
            "ocr_show_boxes": self.chk_ocr_show_boxes.isChecked(),
            "ocr_box_thickness": self.spin_ocr_box_thickness.value(),
            "ocr_text_background": self.chk_ocr_text_background.isChecked(),
            "ocr_overlay_text_source": self.combo_ocr_overlay_source.currentData() or "original",
            "ocr_translate_target": self.combo_ocr_translate_target.currentData() or "es",
            "ocr_subtitle_bg_color": self._ocr_subtitle_bg_color,
            "ocr_subtitle_bg_opacity": bg_opacity,
        })

    # ------------------------------------------------------------------
    # Other slot handlers
    # ------------------------------------------------------------------

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

    def _on_save_clicked(self):
        if self._current_pixmap:
            file_name, _ = QFileDialog.getSaveFileName(
                self, "Save Capture", "", "Images (*.png *.jpg)"
            )
            if file_name:
                self.save_requested.emit(file_name)

    def _on_copy_clicked(self):
        if self._current_pixmap:
            QApplication.clipboard().setPixmap(self._current_pixmap)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def update_image(self, qimg):
        """Updates the image shown in the control panel."""
        pixmap = QPixmap.fromImage(qimg)
        self._current_pixmap = pixmap
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled_pixmap)

    def save_current_pixmap(self, file_name: str):
        if self._current_pixmap:
            self._current_pixmap.save(file_name)

    def update_glass_size(self, x: int, y: int, w: int, h: int):
        self.lbl_glass_size.setText(f"Size: {w}x{h}")
