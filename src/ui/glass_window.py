import sys
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QRect
from PyQt6.QtGui import QPainter, QColor, QPen, QPixmap

class GlassWindow(QWidget):
    geometry_changed = pyqtSignal(int, int, int, int) # x, y, w, h

    def __init__(self):
        super().__init__()
        self.setWindowTitle("GlassCV - Glass")
        
        # Atributos iniciales de la ventana
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool  # Ocultar de la barra de tareas
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Geometría inicial
        self.setGeometry(100, 100, 400, 300)
        
        # Variables para movimiento y redimensionamiento manual
        self._is_moving = False
        self._is_resizing = False
        self._resize_edge = None
        self._drag_start_position = QPoint()
        self._window_start_geometry = QRect()
        
        self.border_width = 5
        self.border_color = QColor(0, 255, 0, 200) # Verde semi-transparente
        
        # Label para el modo espejo
        self.mirror_mode = False
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.hide()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(self.border_width, self.border_width, self.border_width, self.border_width)
        layout.addWidget(self.image_label)
        
        # Para que detecte el mouse en los bordes sin hacer click
        self.setMouseTracking(True)

    def set_click_through(self, state: bool):
        """Activa o desactiva la interacción del mouse con esta ventana."""
        self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, state)
        if state:
            self.border_color = QColor(255, 0, 0, 200) # Rojo cuando está fijado
        else:
            self.border_color = QColor(0, 255, 0, 200) # Verde cuando se puede mover
            
        # Al cambiar WindowFlags en PyQt6 a veces es necesario ocultar y mostrar la ventana
        self.hide()
        self.show()
        self.update()

    def set_mirror_mode(self, state: bool):
        """Activa o desactiva el renderizado de la imagen procesada dentro del glass."""
        self.mirror_mode = state
        if state:
            self.image_label.show()
        else:
            self.image_label.hide()
            self.image_label.clear()

    def update_image(self, qimg):
        """Si el modo espejo está activo, actualiza la imagen a mostrar."""
        if self.mirror_mode:
            pixmap = QPixmap.fromImage(qimg)
            self.image_label.setPixmap(pixmap.scaled(
                self.image_label.size(), 
                Qt.AspectRatioMode.KeepAspectRatio, 
                Qt.TransformationMode.FastTransformation
            ))

    def paintEvent(self, event):
        """Dibuja el borde de la ventana de captura."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Rellenar con transparente
        painter.fillRect(self.rect(), Qt.GlobalColor.transparent)
        
        # Dibujar borde
        pen = QPen(self.border_color, self.border_width)
        painter.setPen(pen)
        
        # Ajustamos el rect para que el borde se dibuje por completo dentro del widget
        rect = self.rect()
        offset = self.border_width // 2
        rect.adjust(offset, offset, -offset, -offset)
        painter.drawRect(rect)

    # --- Lógica de movimiento y redimensionamiento ---
    
    def get_resize_edge(self, pos: QPoint):
        x = pos.x()
        y = pos.y()
        w = self.width()
        h = self.height()
        bw = self.border_width * 2 # Zona sensible más grande

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
                
            # Restricciones mínimas
            if w > 50 and h > 50:
                self.setGeometry(x, y, w, h)
                self.emit_geometry()
                
        else:
            # Cambiar el cursor dependiendo del borde
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
        # Ajustamos el área de captura al interior del glass para no capturar el borde
        g = self.geometry()
        bw = self.border_width
        self.geometry_changed.emit(g.x() + bw, g.y() + bw, g.width() - 2*bw, g.height() - 2*bw)

    def showEvent(self, event):
        super().showEvent(event)
        self.emit_geometry()
