"""
Main Window
Primary application window containing all UI components.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QAction, QStatusBar, QLabel, QSlider,
    QDockWidget, QMessageBox, QFileDialog, QSplitter,
    QFrame, QGroupBox, QCheckBox, QPushButton
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon, QKeySequence

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer.scene import Scene
from renderer.opengl_widget import OpenGLWidget
from wind_data.wind_field import WindField
from models.deformation_model import DeformationModel
from ui.object_library import ObjectLibraryPanel
from ui.simulation_controller import SimulationController


class ControlPanel(QWidget):
    """
    Panel with simulation controls and settings.
    """
    
    def __init__(self, controller: SimulationController, parent=None):
        """
        Initialize control panel.
        
        Args:
            controller: Simulation controller instance
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.controller = controller
        self._setup_ui()
    
    def _setup_ui(self):
        """Set up the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)
        
        # Simulation controls group
        sim_group = QGroupBox("Simulation")
        sim_layout = QVBoxLayout(sim_group)
        
        # Play/Pause button
        self.play_btn = QPushButton("Play")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._toggle_simulation)
        self.play_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:checked {
                background-color: #FF9800;
            }
            QPushButton:hover {
                opacity: 0.8;
            }
        """)
        sim_layout.addWidget(self.play_btn)
        
        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.controller.reset)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #607D8B;
                color: white;
                border: none;
                padding: 6px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #78909C;
            }
        """)
        sim_layout.addWidget(self.reset_btn)
        
        # FPS slider
        fps_layout = QHBoxLayout()
        fps_label = QLabel("FPS:")
        fps_label.setStyleSheet("color: #ccc;")
        self.fps_slider = QSlider(Qt.Horizontal)
        self.fps_slider.setMinimum(10)
        self.fps_slider.setMaximum(120)
        self.fps_slider.setValue(60)
        self.fps_slider.valueChanged.connect(self.controller.set_target_fps)
        self.fps_value = QLabel("60")
        self.fps_value.setStyleSheet("color: #ccc; min-width: 30px;")
        self.fps_slider.valueChanged.connect(
            lambda v: self.fps_value.setText(str(v))
        )
        fps_layout.addWidget(fps_label)
        fps_layout.addWidget(self.fps_slider)
        fps_layout.addWidget(self.fps_value)
        sim_layout.addLayout(fps_layout)
        
        layout.addWidget(sim_group)
        
        # Display options group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        self.grid_cb.setStyleSheet("color: #ccc;")
        display_layout.addWidget(self.grid_cb)
        
        self.wind_cb = QCheckBox("Show Wind Vectors")
        self.wind_cb.setChecked(True)
        self.wind_cb.setStyleSheet("color: #ccc;")
        display_layout.addWidget(self.wind_cb)
        
        layout.addWidget(display_group)

        # Stats display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("FPS: --\nFrame: 0\nTime: 0.0s")
        self.stats_label.setStyleSheet("color: #888; font-family: monospace;")
        stats_layout.addWidget(self.stats_label)
        
        layout.addWidget(stats_group)
        
        # Spacer
        layout.addStretch()
        
        # Style
        self.setStyleSheet("""
            QGroupBox {
                color: #ccc;
                border: 1px solid #444;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
            }
        """)
        
        self.setMinimumWidth(180)
        self.setMaximumWidth(220)
    
    def _toggle_simulation(self, checked: bool):
        """Toggle simulation play state."""
        if checked:
            self.play_btn.setText("Pause")
            if self.controller.is_paused:
                self.controller.resume()
            else:
                self.controller.start()
        else:
            self.play_btn.setText("Play")
            self.controller.pause()
    
    def update_stats(self):
        """Update statistics display."""
        stats = self.controller.get_stats()
        text = (
            f"FPS: {stats['fps']:.1f}\n"
            f"Frame: {stats['frame_count']}\n"
            f"Time: {stats['simulation_time']:.1f}s\n"
            f"Objects: {stats['object_count']}"
        )
        self.stats_label.setText(text)


class MainWindow(QMainWindow):
    """
    Main application window.
    
    Contains:
    - Central OpenGL viewport
    - Object library dock (left)
    - Control panel dock (right)
    - Toolbar with actions
    - Status bar
    """
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        self._setup_components()
        self._setup_ui()
        self._setup_connections()
        self._setup_update_timer()
    
    def _setup_components(self):
        """Initialize core components."""
        # Create wind field
        self.wind_field = WindField(
            grid_size=(20, 20, 10),
            time_steps=200
        )
        
        # Create scene
        self.scene = Scene(self.wind_field)
        
        # Create deformation model
        self.deformation_model = DeformationModel()
        
        # Create simulation controller
        self.sim_controller = SimulationController(
            self.scene,
            self.deformation_model
        )
    
    def _setup_ui(self):
        """Set up the user interface."""
        self.setWindowTitle("Wind Visualization System")
        self.setMinimumSize(1200, 800)
        
        # Set dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QMenuBar {
                background-color: #2d2d2d;
                color: #ccc;
            }
            QMenuBar::item:selected {
                background-color: #3d3d3d;
            }
            QMenu {
                background-color: #2d2d2d;
                color: #ccc;
            }
            QMenu::item:selected {
                background-color: #3d3d3d;
            }
            QToolBar {
                background-color: #2d2d2d;
                border: none;
                spacing: 4px;
            }
            QStatusBar {
                background-color: #2d2d2d;
                color: #888;
            }
            QDockWidget {
                color: #ccc;
            }
            QDockWidget::title {
                background-color: #2d2d2d;
                padding: 4px;
            }
        """)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # OpenGL viewport
        self.gl_widget = OpenGLWidget(self.scene)
        layout.addWidget(self.gl_widget, stretch=1)
        
        # Left dock - Object Library
        self.object_library = ObjectLibraryPanel()
        left_dock = QDockWidget("Objects", self)
        left_dock.setWidget(self.object_library)
        left_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.LeftDockWidgetArea, left_dock)
        
        # Right dock - Controls
        self.control_panel = ControlPanel(self.sim_controller)
        right_dock = QDockWidget("Controls", self)
        right_dock.setWidget(self.control_panel)
        right_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, right_dock)
        
        # Create toolbar
        self._create_toolbar()
        
        # Create status bar
        self._create_status_bar()
        
        # Create menus
        self._create_menus()
    
    def _create_toolbar(self):
        """Create the main toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        
        # Play action
        self.play_action = QAction("Play", self)
        self.play_action.setShortcut(QKeySequence("Space"))
        self.play_action.setCheckable(True)
        self.play_action.triggered.connect(self._toggle_play)
        toolbar.addAction(self.play_action)
        
        # Reset action
        reset_action = QAction("Reset", self)
        reset_action.setShortcut(QKeySequence("R"))
        reset_action.triggered.connect(self._reset_simulation)
        toolbar.addAction(reset_action)
        
        toolbar.addSeparator()
        
        # Toggle grid
        grid_action = QAction("Grid", self)
        grid_action.setShortcut(QKeySequence("G"))
        grid_action.setCheckable(True)
        grid_action.setChecked(True)
        grid_action.triggered.connect(self._toggle_grid)
        toolbar.addAction(grid_action)
        
        # Toggle wind vectors
        wind_action = QAction("Wind", self)
        wind_action.setShortcut(QKeySequence("W"))
        wind_action.setCheckable(True)
        wind_action.setChecked(True)
        wind_action.triggered.connect(self._toggle_wind)
        toolbar.addAction(wind_action)
        
        toolbar.addSeparator()
        
        # Clear scene
        clear_action = QAction("Clear", self)
        clear_action.triggered.connect(self._clear_scene)
        toolbar.addAction(clear_action)
        
        # Reset camera
        camera_action = QAction("Reset Camera", self)
        camera_action.setShortcut(QKeySequence("C"))
        camera_action.triggered.connect(self._reset_camera)
        toolbar.addAction(camera_action)
    
    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, stretch=1)
        
        self.coord_label = QLabel("X: -- Y: -- Z: --")
        self.coord_label.setMinimumWidth(220)
        self.coord_label.setStyleSheet("font-family: monospace; color: #aaa;")
        self.status_bar.addPermanentWidget(self.coord_label)
        
        self.fps_label = QLabel("FPS: --")
        self.status_bar.addPermanentWidget(self.fps_label)
    
    def _create_menus(self):
        """Create application menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("&File")
        
        save_action = QAction("&Save Scene", self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_scene)
        file_menu.addAction(save_action)
        
        load_action = QAction("&Load Scene", self)
        load_action.setShortcut(QKeySequence.Open)
        load_action.triggered.connect(self._load_scene)
        file_menu.addAction(load_action)

        load_wind_action = QAction("Load &OpenFOAM Wind...", self)
        load_wind_action.triggered.connect(self._load_openfoam_wind)
        file_menu.addAction(load_wind_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")
        
        reset_camera_action = QAction("Reset &Camera", self)
        reset_camera_action.triggered.connect(self._reset_camera)
        view_menu.addAction(reset_camera_action)
        
        # Help menu
        help_menu = menubar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _setup_connections(self):
        """Set up signal connections."""
        # Object library selection
        self.object_library.object_selected.connect(
            self.gl_widget.set_pending_drop
        )
        
        # Object drop
        self.gl_widget.object_dropped.connect(self._add_object)
        
        # Object selection
        self.gl_widget.object_selected.connect(self._on_object_selected)
        
        # Viewport hover coordinates
        self.gl_widget.cursor_world_moved.connect(self._on_cursor_world_moved)
        
        # Simulation updates
        self.sim_controller.simulation_updated.connect(self._on_simulation_update)
        
        # Control panel display toggles
        self.control_panel.grid_cb.toggled.connect(self._toggle_grid)
        self.control_panel.wind_cb.toggled.connect(self._toggle_wind)
    
    def _setup_update_timer(self):
        """Set up UI update timer."""
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(100)  # Update UI at 10 Hz
    
    def _add_object(self, object_type: str, x: float, y: float, z: float):
        """Add an object to the scene."""
        mesh = self.scene.add_object(object_type, (x, y, z))
        
        # Get structural payload with pole center for ML parsing
        payload = mesh.get_info_payload()
        
        # Log to console clearly
        print(f"{payload['object_type']}, {payload['object_position']}, {payload['pole_center']}")
        
        # Make available to simulation/ML pipeline safely
        if hasattr(self.sim_controller, 'register_object_payload'):
            self.sim_controller.register_object_payload(payload)
            
        self.object_library.update_object_count(len(self.scene.objects))
        self.status_label.setText(f"Added {object_type} at ({x:.1f}, {y:.1f}, {z:.1f})")
        self.gl_widget.update()
    
    def _on_object_selected(self, obj):
        """Handle object selection."""
        if obj:
            self.status_label.setText(f"Selected: {obj.name}")
        else:
            self.status_label.setText("Ready")
    
    def _on_cursor_world_moved(self, x: float, y: float, z: float):
        """Update the status bar with the cursor's world coordinates."""
        if not all(math.isfinite(value) for value in (x, y, z)):
            self.coord_label.setText("X: -- Y: -- Z: --")
            return
        
        self.coord_label.setText(f"X: {x:.1f} Y: {y:.1f} Z: {z:.1f}")
    
    def _on_simulation_update(self):
        """Handle simulation update."""
        self.gl_widget.update()
    
    def _update_ui(self):
        """Update UI elements."""
        self.control_panel.update_stats()
        stats = self.sim_controller.get_stats()
        self.fps_label.setText(f"FPS: {stats['fps']:.1f}")
    
    def _toggle_play(self, checked: bool):
        """Toggle simulation play state."""
        self.control_panel.play_btn.setChecked(checked)
        self.control_panel._toggle_simulation(checked)
    
    def _reset_simulation(self):
        """Reset the simulation."""
        self.sim_controller.reset()
        self.play_action.setChecked(False)
        self.control_panel.play_btn.setChecked(False)
        self.gl_widget.update()
    
    def _toggle_grid(self, visible: bool):
        """Toggle grid visibility."""
        self.scene.grid_visible = visible
        self.gl_widget.update()
    
    def _toggle_wind(self, visible: bool):
        """Toggle wind vector visibility."""
        self.scene.wind_vectors_visible = visible
        self.gl_widget.update()
    
    def _clear_scene(self):
        """Clear all objects from the scene."""
        reply = QMessageBox.question(
            self,
            "Clear Scene",
            "Remove all objects from the scene?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.scene.clear_objects()
            self.object_library.update_object_count(0)
            self.gl_widget.update()
            self.status_label.setText("Scene cleared")
    
    def _reset_camera(self):
        """Reset camera to default position."""
        self.scene.camera.reset()
        self.gl_widget.update()
    
    def _save_scene(self):
        """Save the current scene to file."""
        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Scene",
            "",
            "JSON Files (*.json)"
        )
        
        if filepath:
            data = self.scene.serialize()
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            self.status_label.setText(f"Scene saved to {filepath}")
    
    def _load_scene(self):
        """Load a scene from file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Scene",
            "",
            "JSON Files (*.json)"
        )
        
        if filepath:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.scene.deserialize(data)
            self.object_library.update_object_count(len(self.scene.objects))
            self.gl_widget.update()
            self.status_label.setText(f"Scene loaded from {filepath}")

    def _load_openfoam_wind(self):
        """Load OpenFOAM wind data from a folder."""
        base_dir = QFileDialog.getExistingDirectory(
            self,
            "Select OpenFOAM surfaces folder",
            ""
        )

        if not base_dir:
            return

        if self.sim_controller.is_running:
            self.sim_controller.stop()
            self.play_action.setChecked(False)
            self.control_panel.play_btn.setChecked(False)

        self.status_label.setText("Loading OpenFOAM wind data...")
        self.status_bar.repaint()

        try:
            self.wind_field.load_from_openfoam_folder(base_dir)
        except Exception as exc:
            QMessageBox.critical(self, "OpenFOAM Load Failed", str(exc))
            self.status_label.setText("Failed to load OpenFOAM wind data")
            return

        self._fit_grid_to_wind_field()
        self.status_label.setText(f"OpenFOAM wind loaded from {base_dir}")
        self.gl_widget.update()
    
    def _fit_grid_to_wind_field(self):
        """Resize and center the ground grid around the loaded wind field."""
        min_corner, max_corner = self.wind_field.get_bounds()
        min_x, max_x = float(min_corner[0]), float(max_corner[0])
        min_z, max_z = float(min_corner[2]), float(max_corner[2])
        
        spacing = self._infer_horizontal_grid_spacing()
        center_x = (min_x + max_x) / 2.0
        center_z = (min_z + max_z) / 2.0
        half_extent = max((max_x - min_x) / 2.0, (max_z - min_z) / 2.0, spacing)
        half_steps = max(1, int(math.ceil(half_extent / spacing)))
        
        self.scene.grid_center = (center_x, center_z)
        self.scene.grid_spacing = spacing
        self.scene.grid_size = half_steps * 2
    
    def _infer_horizontal_grid_spacing(self) -> float:
        """Infer a display grid spacing from loaded wind X/Z coordinates."""
        spacing_candidates = []
        
        for coords in (self.wind_field.x_coords, self.wind_field.z_coords):
            unique_coords = sorted({float(coord) for coord in coords})
            diffs = [
                b - a
                for a, b in zip(unique_coords, unique_coords[1:])
                if b - a > 1e-6
            ]
            
            if diffs:
                diffs.sort()
                index = min(len(diffs) - 1, int(round((len(diffs) - 1) * 0.75)))
                spacing_candidates.append(diffs[index])
        
        if not spacing_candidates:
            return max(float(self.scene.grid_spacing), 1.0)
        
        return max(min(spacing_candidates), 1e-3)
    
    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Wind Visualization System",
            "Wind Visualization System\n\n"
            "An interactive 3D wind simulation tool with\n"
            "ML-based mesh deformation.\n\n"
            "Features:\n"
            "- Drag and drop object placement\n"
            "- Real-time wind visualization\n"
            "- PyTorch-based deformation prediction\n"
            "- GPU acceleration support"
        )
    
    def closeEvent(self, event):
        """Handle window close."""
        self.sim_controller.stop()
        event.accept()
