"""
Object Library Panel
Drag-and-drop panel for placing objects into the scene.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFrame, QScrollArea, QGridLayout,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QMimeData, pyqtSignal
from PyQt5.QtGui import QDrag, QPixmap, QPainter, QColor, QFont


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
        # Set text
        self.setText(self.object_type.capitalize())
        
        # Style
        self.setMinimumSize(80, 80)
        self.setMaximumSize(100, 100)
        self.setCursor(Qt.OpenHandCursor)
        
        # Color based on type
        colors = {
            'flag': '#F44336',
        }
        color = colors.get(self.object_type.lower(), '#607D8B')
        
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {color};
                color: white;
                border: none;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {self._lighten_color(color)};
            }}
            QPushButton:pressed {{
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
            
            # Create drag pixmap
            pixmap = QPixmap(60, 60)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setBrush(QColor(self.palette().button().color()))
            painter.setPen(Qt.NoPen)
            painter.drawRoundedRect(0, 0, 60, 60, 8, 8)
            painter.setPen(Qt.white)
            painter.setFont(QFont('Arial', 10, QFont.Bold))
            painter.drawText(pixmap.rect(), Qt.AlignCenter, self.object_type[:4])
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
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Title
        title = QLabel("Object Library")
        title.setStyleSheet("""
            QLabel {
                font-size: 14px;
                font-weight: bold;
                color: #ffffff;
                padding: 4px;
            }
        """)
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #444;")
        layout.addWidget(separator)
        
        # Instructions
        instructions = QLabel("Drag objects to the viewport\nor click to select")
        instructions.setStyleSheet("""
            QLabel {
                color: #888;
                font-size: 11px;
            }
        """)
        instructions.setAlignment(Qt.AlignCenter)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Object grid
        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setSpacing(8)
        
        # Available object types
        object_types = ['flag']
        
        for i, obj_type in enumerate(object_types):
            button = ObjectButton(obj_type)
            button.clicked.connect(
                lambda checked, t=obj_type: self.object_selected.emit(t)
            )
            row = i // 2
            col = i % 2
            grid_layout.addWidget(button, row, col)
        
        layout.addWidget(grid_widget)
        
        # Spacer
        layout.addStretch()
        
        # Object count label
        self.count_label = QLabel("Objects in scene: 0")
        self.count_label.setStyleSheet("color: #888; font-size: 11px;")
        self.count_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.count_label)
        
        # Set panel style
        self.setStyleSheet("""
            QWidget {
                background-color: #2d2d2d;
            }
        """)
        
        self.setMinimumWidth(180)
        self.setMaximumWidth(220)
    
    def update_object_count(self, count: int):
        """
        Update the object count display.
        
        Args:
            count: Number of objects in scene
        """
        self.count_label.setText(f"Objects in scene: {count}")
