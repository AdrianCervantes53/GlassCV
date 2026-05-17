import sys
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap

class GlassWindow(QWidget):
    geometry_changed = pyqtSignal(int, int, int, int) # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - Glass")
        
        # Initial window attributes
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Hide from taskbar
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Initial geometry
        self.setGeometry(100, 100, 400, 300)
        
        # Variables for manual movement and resizing
        self._is_moving = False
        self._is_resizing = False
        self._resize_edge = None
        self._drag_start_position = QPoint()
        self._window_start_geometry = QRect()
        
        self.border_width = 5
        self.border_color = QColor(0, 255, 0, 200) # Semi-transparent green
        
        # Label for mirror mode
        self.mirror_mode = False
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.hide()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.border_width, self.border_width, self.border_width, self.border_width)
        layout.addWidget(self.image_label)
        
        # To detect mouse on edges without clicking
        self.setMouseTracking(True)

    def set_click_through(self, state: bool):
        """Enables or disables mouse interaction with this window."""
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, state)
        if state:
            self.border_color = QColor(255, 0, 0, 200) # Red when pinned
        else:
            self.border_color = QColor(0, 255, 0, 200) # Green when movable
            
        # When changing WindowFlags in PyQt6 it is sometimes necessary to hide and show the window
        self.hide()
        self.show()
        self.update()

    def set_mirror_mode(self, state: bool):
        """Enables or disables rendering of the processed image inside the glass."""
        self.mirror_mode = state
        if state:
            self.image_label.show()
        else:
            self.image_label.hide()
            self.image_label.clear()

    def update_image(self, qimg):
        """If mirror mode is active, updates the image to display."""
        if self.mirror_mode:
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.FastTransformation
            ))

    def paintEvent(self, event):
        """Draws the border of the capture window."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fill with transparent
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        
        # Draw border
        pen = QPen(self.border_color, self.border_width)
        painter.setPen(pen)
        
        # Adjust the rect so the border is drawn completely inside the widget
        rect = self.rect()
        offset = self.border_width // 2
        rect.adjust(offset, offset, -offset, -offset)
        painter.drawRect(rect)

    # --- Movement and resizing logic ---
    
    def get_resize_edge(self, pos: QPoint):
        x = pos.x()
        y = pos.y()
        w = self.width()
        h = self.height()
        bw = self.border_width * 2 # Larger sensitive area

        edge = []
        if y <= bw:
            edge.append("top")
        elif y >= h - bw:
            edge.append("bottom")
            
        if x <= bw:
            edge.append("left")
        elif x >= w - bw:
            edge.append("right")
            
        return "-".join(edge) if edge else None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._resize_edge = self.get_resize_edge(event.pos())
            if self._resize_edge:
                self._is_resizing = True
            else:
                self._is_moving = True
                
            self._drag_start_position = event.globalPosition().toPoint()
            self._window_start_geometry = self.geometry()

    def mouseMoveEvent(self, event):
        if self._is_moving:
            delta = event.globalPosition().toPoint() - self._drag_start_position
            self.move(self._window_start_geometry.topLeft() + delta)
            self.emit_geometry()
            
        elif self._is_resizing and self._resize_edge:
            delta = event.globalPosition().toPoint() - self._drag_start_position
            geom = self._window_start_geometry
            
            x, y, w, h = geom.x(), geom.y(), geom.width(), geom.height()
            
            if "left" in self._resize_edge:
                w -= delta.x()
                x += delta.x()
            elif "right" in self._resize_edge:
                w += delta.x()
                
            if "top" in self._resize_edge:
                h -= delta.y()
                y += delta.y()
            elif "bottom" in self._resize_edge:
                h += delta.y()
                
            # Minimum constraints
            if w > 50 and h > 50:
                self.setGeometry(x, y, w, h)
                self.emit_geometry()
                
        else:
            # Change the cursor depending on the edge
            edge = self.get_resize_edge(event.pos())
            if edge in ["left", "right"]:
                self.setCursor(Qt.CursorShape.SizeHorCursor)
            elif edge in ["top", "bottom"]:
                self.setCursor(Qt.CursorShape.SizeVerCursor)
            elif edge in ["top-left", "bottom-right"]:
                self.setCursor(Qt.CursorShape.SizeFDiagCursor)
            elif edge in ["top-right", "bottom-left"]:
                self.setCursor(Qt.CursorShape.SizeBDiagCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_moving = False
            self._is_resizing = False
            self._resize_edge = None
            self.emit_geometry()

    def emit_geometry(self):
        # Adjust the capture area to the inside of the glass to not capture the border
        g = self.geometry()
        bw = self.border_width
        self.geometry_changed.emit(g.x() + bw, g.y() + bw, g.width() - 2*bw, g.height() - 2*bw)

    def showEvent(self, event):
        super().showEvent(event)
        self.emit_geometry()
