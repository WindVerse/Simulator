"""
Object Library Panel
Drag-and-drop panel for placing objects into the scene.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QLayout
)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal, QSize, QPointF
from PyQt5.QtGui import (
    QDrag, QPixmap, QPainter, QColor, QPen, QPainterPath, QIcon
)


def make_flag_pixmap(size: int, color: QColor) -> QPixmap:
    """
    Render a small waving-flag icon (pole + rippling cloth) at ``size`` px.

    Drawn as a flat silhouette in ``color`` so it reads as a flag without any
    text label.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)

    margin = size * 0.16
    pole_x = size * 0.30
    pole_top = margin * 0.7
    pole_bottom = size - margin * 0.7
    pole_w = max(2.0, size * 0.045)

    # Waving cloth: a band attached to the top of the pole whose top and
    # bottom edges ripple in parallel, suggesting wind.
    cloth_left = pole_x
    cloth_right = size - margin
    cloth_top = pole_top + size * 0.04
    band_h = size * 0.34
    amp = size * 0.06
    w = cloth_right - cloth_left

    path = QPainterPath()
    path.moveTo(cloth_left, cloth_top)
    path.cubicTo(cloth_left + w * 0.33, cloth_top - amp,
                 cloth_left + w * 0.66, cloth_top + amp,
                 cloth_right, cloth_top)
    path.lineTo(cloth_right, cloth_top + band_h)
    path.cubicTo(cloth_left + w * 0.66, cloth_top + band_h + amp,
                 cloth_left + w * 0.33, cloth_top + band_h - amp,
                 cloth_left, cloth_top + band_h)
    path.closeSubpath()

    painter.setPen(Qt.NoPen)
    painter.setBrush(color)
    painter.drawPath(path)

    # Flag pole + finial knob.
    pole_pen = QPen(color, pole_w)
    pole_pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pole_pen)
    painter.drawLine(QPointF(pole_x, pole_top), QPointF(pole_x, pole_bottom))
    painter.setPen(Qt.NoPen)
    finial_r = size * 0.05
    painter.drawEllipse(QPointF(pole_x, pole_top), finial_r, finial_r)

    painter.end()
    return pixmap


class ObjectButton(QPushButton):
    """
    Draggable button representing an object in the library.
    """
    
    def __init__(self, object_type: str, parent=None):
        """
        Initialize the object button.
        
        Args:
            object_type: Type of object this button represents
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.object_type = object_type
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the button appearance."""
        # Structural styling (rounded, padding) comes from ui.theme via the
        # #objectTile selector; only the per-type color lives here.
        self.setObjectName("objectTile")
        self.setMinimumSize(80, 80)
        self.setMaximumSize(100, 100)
        self.setCursor(Qt.OpenHandCursor)

        # Color based on type
        colors = {
            'flag': '#F44336',
        }
        color = colors.get(self.object_type.lower(), '#607D8B')
        self._color = QColor(color)

        # Show the object as an actual flag silhouette instead of a text label.
        self.setText("")
        self.setIcon(QIcon(make_flag_pixmap(56, QColor('white'))))
        self.setIconSize(QSize(56, 56))
        self.setToolTip("Drag onto the grid to place")

        self.setStyleSheet(f"""
            QPushButton#objectTile {{
                background-color: {color};
            }}
            QPushButton#objectTile:hover {{
                background-color: {self._lighten_color(color)};
            }}
            QPushButton#objectTile:pressed {{
                background-color: {self._darken_color(color)};
            }}
        """)
    
    def _lighten_color(self, hex_color: str) -> str:
        """Lighten a hex color."""
        color = QColor(hex_color)
        return color.lighter(120).name()
    
    def _darken_color(self, hex_color: str) -> str:
        """Darken a hex color."""
        color = QColor(hex_color)
        return color.darker(120).name()
    
    def mousePressEvent(self, event):
        """Handle mouse press for drag start."""
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging."""
        if event.buttons() & Qt.LeftButton:
            drag = QDrag(self)
            mime_data = QMimeData()
            mime_data.setText(self.object_type)
            drag.setMimeData(mime_data)

            # Drag cursor: a rounded tile with the flag silhouette on top.
            size = 64
            pixmap = QPixmap(size, size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(self._color)
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, size, size, 12, 12)
            painter.drawPixmap(0, 0, make_flag_pixmap(size, QColor('white')))
            painter.end()

            drag.setPixmap(pixmap)
            drag.setHotSpot(pixmap.rect().center())
            
            drag.exec_(Qt.CopyAction)


class ObjectLibraryPanel(QWidget):
    """
    Panel displaying available objects for drag-and-drop placement.
    
    Signals:
        object_selected: Emitted when an object button is clicked
    """
    
    object_selected = pyqtSignal(str)
    
    def __init__(self, parent=None):
        """
        Initialize the object library panel.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the compact, floating object card."""
        # Rendered as a card that floats over the viewport (see
        # ViewportContainer.set_overlay); #overlayCard is styled in ui.theme.
        self.setObjectName("overlayCard")
        self.setAttribute(Qt.WA_StyledBackground, True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        # Shrink-wrap the card around the flag tile instead of stretching it
        # to a fixed-width panel.
        layout.setSizeConstraint(QLayout.SetFixedSize)

        # The only placeable object is the flag, shown as a flag (no text/name).
        self._button = ObjectButton('flag')
        self._button.clicked.connect(lambda: self.object_selected.emit('flag'))
        layout.addWidget(self._button, alignment=Qt.AlignHCenter)

        # Live scene object count.
        self.count_label = QLabel("In scene: 0")
        self.count_label.setObjectName("panelHint")
        self.count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_label)

    def update_object_count(self, count: int):
        """
        Update the object count display.

        Args:
            count: Number of objects in scene
        """
        self.count_label.setText(f"In scene: {count}")
