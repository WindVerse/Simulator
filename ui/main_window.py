"""
Main Window
Primary application window containing all UI components.
"""

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QToolBar, QAction, QStatusBar, QLabel,
    QDockWidget, QMessageBox, QFileDialog, QSplitter,
    QFrame, QGroupBox, QCheckBox, QPushButton,
    QRadioButton, QButtonGroup, QSpinBox, QActionGroup
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QKeySequence

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer.scene import Scene
from renderer.opengl_widget import OpenGLWidget
from ui.viewports import ViewportContainer
from renderer.wind_colormap import BEAUFORT_BANDS
from wind_data.wind_field import WindField
from wind_data.openfoam_loader import extract_openfoam_case
from objects.object_mesh import ObjectMesh
from models.deformation_model import DeformationModel
from ui.object_library import ObjectLibraryPanel
from ui.simulation_controller import SimulationController
from ui import theme


class _OpenFOAMCaseLoadWorker(QThread):
    """Background thread that parses an OpenFOAM case folder off the UI thread."""

    finished_ok = pyqtSignal(object)  # result dict from extract_openfoam_case
    failed = pyqtSignal(str)

    def __init__(self, selected_path: str, parent=None):
        super().__init__(parent)
        self._selected_path = selected_path

    def run(self):
        try:
            result = extract_openfoam_case(self._selected_path)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished_ok.emit(result)


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
        self.play_btn.setObjectName("playButton")
        self.play_btn.setCheckable(True)
        self.play_btn.clicked.connect(self._toggle_simulation)
        sim_layout.addWidget(self.play_btn)

        # Reset button
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setObjectName("resetButton")
        self.reset_btn.clicked.connect(self.controller.reset)
        sim_layout.addWidget(self.reset_btn)

        layout.addWidget(sim_group)
        
        # Display options group
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)
        
        self.grid_cb = QCheckBox("Show Grid")
        self.grid_cb.setChecked(True)
        display_layout.addWidget(self.grid_cb)

        self.wind_cb = QCheckBox("Show Wind Vectors")
        self.wind_cb.setChecked(True)
        display_layout.addWidget(self.wind_cb)

        # Wind display mode: resultant vs. per-component arrows
        self.wind_mode_resultant_rb = QRadioButton("Resultant")
        self.wind_mode_resultant_rb.setChecked(True)
        self.wind_mode_components_rb = QRadioButton("Components (X/Y/Z)")
        self.wind_mode_group = QButtonGroup(self)
        self.wind_mode_group.addButton(self.wind_mode_resultant_rb)
        self.wind_mode_group.addButton(self.wind_mode_components_rb)
        display_layout.addWidget(self.wind_mode_resultant_rb)
        display_layout.addWidget(self.wind_mode_components_rb)

        # Downsample stride: 1 = show every vector.
        stride_row = QHBoxLayout()
        stride_label = QLabel("Stride:")
        stride_row.addWidget(stride_label)
        self.wind_stride_spin = QSpinBox()
        self.wind_stride_spin.setRange(1, 50)
        self.wind_stride_spin.setValue(1)
        self.wind_stride_spin.setToolTip(
            "1 = show every wind vector. Higher values skip points for performance."
        )
        stride_row.addWidget(self.wind_stride_spin)
        stride_row.addStretch()
        display_layout.addLayout(stride_row)

        # Apply the Beaufort colormap (resultant mode only).
        self.wind_color_cb = QCheckBox("Color by speed")
        self.wind_color_cb.setChecked(True)
        self.wind_color_cb.setToolTip(
            "Color arrows by wind speed (Beaufort bands). Resultant mode only."
        )
        display_layout.addWidget(self.wind_color_cb)

        self.env_cb = QCheckBox("Show Environment")
        self.env_cb.setChecked(True)
        display_layout.addWidget(self.env_cb)

        layout.addWidget(display_group)

        # Wind speed legend (Beaufort scale).
        legend_group = QGroupBox("Wind speed (Beaufort)")
        legend_layout = QVBoxLayout(legend_group)
        legend_layout.setContentsMargins(8, 8, 8, 8)
        legend_layout.setSpacing(3)
        for _max_speed, rgba, label, range_text in BEAUFORT_BANDS:
            row = QHBoxLayout()
            row.setSpacing(8)
            swatch = QFrame()
            swatch.setFixedSize(20, 12)
            r, g, b, _a = rgba
            swatch.setStyleSheet(
                f"background-color: rgb({int(r*255)}, {int(g*255)}, {int(b*255)});"
                f" border: 1px solid {theme.BORDER}; border-radius: 2px;"
            )
            text = QLabel(f"{range_text}  {label}")
            text.setObjectName("statValue")
            row.addWidget(swatch)
            row.addWidget(text)
            row.addStretch()
            legend_layout.addLayout(row)
        self.legend_group = legend_group
        layout.addWidget(legend_group)

        # Stats display
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_label = QLabel("FPS: --\nFrame: 0\nTime: 0.0s")
        self.stats_label.setObjectName("statValue")
        stats_layout.addWidget(self.stats_label)

        layout.addWidget(stats_group)

        # Spacer
        layout.addStretch()

        # Group-box / widget styling is provided globally by ui.theme.

        self.setMinimumWidth(190)
        self.setMaximumWidth(240)
    
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

        self._sample_load_worker = None

        self._setup_components()
        self._setup_ui()
        self._setup_connections()
        self._setup_update_timer()

        # Load bundled OpenFOAM sample asynchronously so the window appears immediately.
        QTimer.singleShot(0, self._load_default_case_async)
    
    def _setup_components(self):
        """Initialize core components."""
        # Create wind field
        self.wind_field = WindField(
            grid_size=(20, 20, 10),
            time_steps=200
        )
        
        # Create scene
        self.scene = Scene(self.wind_field)
        self.scene.compute_wind_vector_scale()

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

        # The dark theme is applied application-wide in ui.theme (apply_theme).

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout with splitter
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 3D viewports (Single / Dual / Quad), all rendering the shared scene.
        self.viewports = ViewportContainer(self.scene)
        layout.addWidget(self.viewports, stretch=1)
        # Primary viewport kept for back-compat (save/load camera, focus default).
        self.gl_widget = self.viewports.primary
        
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

        toolbar.addSeparator()

        # Viewport layout: Single / Dual / Quad (mutually exclusive)
        self.layout_group = QActionGroup(self)
        self.layout_group.setExclusive(True)

        self.layout_single_action = self._make_layout_action("Single", "single", checked=True)
        self.layout_dual_action = self._make_layout_action("Dual", "dual")
        self.layout_quad_action = self._make_layout_action("Quad", "quad")
        for act in (self.layout_single_action, self.layout_dual_action, self.layout_quad_action):
            toolbar.addAction(act)

    def _make_layout_action(self, label: str, mode: str, checked: bool = False) -> QAction:
        """Create a checkable, exclusive viewport-layout action."""
        action = QAction(label, self)
        action.setCheckable(True)
        action.setChecked(checked)
        action.triggered.connect(lambda _=False, m=mode: self.viewports.set_layout(m))
        self.layout_group.addAction(action)
        return action

    def _create_status_bar(self):
        """Create the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label, stretch=1)
        
        self.coord_label = QLabel("X: -- Y: -- Z: --")
        self.coord_label.setObjectName("statValue")
        self.coord_label.setMinimumWidth(220)
        self.status_bar.addPermanentWidget(self.coord_label)

        self.fps_label = QLabel("FPS: --")
        self.fps_label.setObjectName("statValue")
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

        load_case_action = QAction("Load OpenFOAM &Output...", self)
        load_case_action.triggered.connect(self._load_openfoam_case)
        file_menu.addAction(load_case_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View menu
        view_menu = menubar.addMenu("&View")

        layout_menu = view_menu.addMenu("&Layout")
        layout_menu.addAction(self.layout_single_action)
        layout_menu.addAction(self.layout_dual_action)
        layout_menu.addAction(self.layout_quad_action)

        view_menu.addSeparator()

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
        # Object library selection (arm a pending drop on every visible viewport)
        self.object_library.object_selected.connect(
            self.viewports.set_pending_drop
        )

        # Object drop
        self.viewports.object_dropped.connect(self._add_object)

        # Object selection
        self.viewports.object_selected.connect(self._on_object_selected)

        # Viewport hover coordinates
        self.viewports.cursor_world_moved.connect(self._on_cursor_world_moved)
        
        # Simulation updates
        self.sim_controller.simulation_updated.connect(self._on_simulation_update)
        
        # Control panel display toggles
        self.control_panel.grid_cb.toggled.connect(self._toggle_grid)
        self.control_panel.wind_cb.toggled.connect(self._toggle_wind)
        self.control_panel.env_cb.toggled.connect(self._toggle_environment)
        self.control_panel.wind_mode_resultant_rb.toggled.connect(self._on_wind_mode_changed)
        self.control_panel.wind_stride_spin.valueChanged.connect(self._on_wind_stride_changed)
        self.control_panel.wind_color_cb.toggled.connect(self._on_wind_color_changed)
        self._refresh_legend_state()

        # Redirect the ControlPanel reset button to the full reset handler so the
        # Play button unchecks and the viewport repaints explicitly.
        self.control_panel.reset_btn.clicked.disconnect()
        self.control_panel.reset_btn.clicked.connect(self._reset_simulation)
    
    def _setup_update_timer(self):
        """Set up UI update timer."""
        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self._update_ui)
        self.ui_timer.start(100)  # Update UI at 10 Hz
    
    def _add_object(self, object_type: str, x: float, y: float, z: float):
        """Add an object to the scene."""
        mesh = self.scene.add_object(object_type, (x, y, z))

        if object_type.lower() == 'flag':
            world_verts = mesh.get_world_vertices()
            print(f"\n=== FLAG PLANTED IN SCENE ===")
            print(f"Object position: {mesh.position}")
            print(f"Number of vertices: {len(world_verts)}")
            print(f"Vertex position range: X [{world_verts[:, 0].min():.3f}, {world_verts[:, 0].max():.3f}]")
            print(f"                      Y [{world_verts[:, 1].min():.3f}, {world_verts[:, 1].max():.3f}]")
            print(f"                      Z [{world_verts[:, 2].min():.3f}, {world_verts[:, 2].max():.3f}]")
            print(f"First 5 world vertices:\n{world_verts[:5]}")
            print(f"Pole center (attachment point): {mesh.get_pole_center()}\n")
    
        
        # Get structural payload with pole center for ML parsing
        payload = mesh.get_info_payload()
        
        # Log to console clearly
        print(f"{payload['object_type']}, {payload['object_position']}, {payload['pole_center']}")
        
        # Make available to simulation/ML pipeline safely
        if hasattr(self.sim_controller, 'register_object_payload'):
            self.sim_controller.register_object_payload(payload)
            
        self.object_library.update_object_count(len(self.scene.objects))
        self.status_label.setText(f"Added {object_type} at ({x:.1f}, {y:.1f}, {z:.1f})")
        self.viewports.refresh()
        # Frame the new object in the viewport the user dropped into.
        self.viewports.active_view.start_focus_on_object(mesh)
    
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
        self.viewports.refresh()
    
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
        self.viewports.refresh()
    
    def _toggle_grid(self, visible: bool):
        """Toggle grid visibility."""
        self.scene.grid_visible = visible
        self.viewports.refresh()
    
    def _toggle_wind(self, visible: bool):
        """Toggle wind vector visibility."""
        self.scene.wind_vectors_visible = visible
        self.viewports.refresh()

    def _on_wind_mode_changed(self, _checked: bool):
        """Switch between resultant and component vector rendering."""
        self.scene.wind_display_mode = (
            "resultant" if self.control_panel.wind_mode_resultant_rb.isChecked()
            else "components"
        )
        self._refresh_legend_state()
        self.viewports.refresh()

    def _on_wind_stride_changed(self, value: int):
        """Update wind vector downsampling stride."""
        self.scene.wind_downsample_stride = max(1, int(value))
        self.viewports.refresh()

    def _on_wind_color_changed(self, checked: bool):
        """Toggle the Beaufort speed colormap for resultant arrows."""
        self.scene.wind_color_by_speed = bool(checked)
        self._refresh_legend_state()
        self.viewports.refresh()

    def _refresh_legend_state(self):
        """Gray out the legend when the colormap isn't actually applied."""
        in_use = (
            self.scene.wind_display_mode == "resultant"
            and self.scene.wind_color_by_speed
        )
        self.control_panel.legend_group.setEnabled(in_use)

    def _toggle_environment(self, visible: bool):
        """Toggle environment (static STL) mesh visibility."""
        self.scene.environment_visible = visible
        self.viewports.refresh()
    
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
            self.viewports.refresh()
            self.status_label.setText("Scene cleared")
    
    def _reset_camera(self):
        """Reset the focused viewport to the default free perspective view."""
        self.viewports.reset_active_camera()
    
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
            self.viewports.refresh()
            self.status_label.setText(f"Scene loaded from {filepath}")

    def _load_default_case_async(self):
        """Kick off background load of the bundled OpenFOAM sample case."""
        sample_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "wind_data",
                "sample_openfoam_output",
            )
        )
        if not os.path.isdir(sample_path):
            return  # No bundled sample; keep demo wind silently.

        self._start_case_load(sample_path, source_label="bundled sample")

    def _load_openfoam_case(self):
        """Load an OpenFOAM case (or surfaces folder) chosen by the user."""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select OpenFOAM case folder (or postProcessing/surfaces)",
            ""
        )

        if not selected:
            return

        if self.sim_controller.is_running:
            self.sim_controller.stop()
            self.play_action.setChecked(False)
            self.control_panel.play_btn.setChecked(False)

        self._start_case_load(selected, source_label=selected)

    def _start_case_load(self, selected_path: str, source_label: str):
        """Spawn a background worker that loads an OpenFOAM case."""
        if self._sample_load_worker is not None and self._sample_load_worker.isRunning():
            return

        self.status_label.setText(f"Loading OpenFOAM case ({source_label})...")
        self.control_panel.play_btn.setEnabled(False)
        if hasattr(self, "play_action"):
            self.play_action.setEnabled(False)

        worker = _OpenFOAMCaseLoadWorker(selected_path, self)
        worker.finished_ok.connect(self._on_case_load_finished)
        worker.failed.connect(self._on_case_load_failed)
        worker.finished.connect(worker.deleteLater)
        self._sample_load_worker = worker
        worker.start()

    def _on_case_load_finished(self, result: dict):
        """Apply a parsed OpenFOAM case on the UI thread."""
        try:
            self._apply_case_result(result)
        finally:
            self.control_panel.play_btn.setEnabled(True)
            if hasattr(self, "play_action"):
                self.play_action.setEnabled(True)
            self._sample_load_worker = None

    def _on_case_load_failed(self, message: str):
        """Restore controls and surface the error in the status bar."""
        self.status_label.setText(f"OpenFOAM case load failed: {message}")
        self.control_panel.play_btn.setEnabled(True)
        if hasattr(self, "play_action"):
            self.play_action.setEnabled(True)
        self._sample_load_worker = None

    def _apply_case_result(self, result: dict):
        """Apply wind + patches + triSurface geometry from a loaded case."""
        wind_data, x_coords, y_coords, z_coords, time_coords = result["wind"]
        self.wind_field.set_wind_data(wind_data, x_coords, y_coords, z_coords, time_coords)
        self.scene.compute_wind_vector_scale()
        self.scene.reset_all_objects()

        self.scene.clear_environment_meshes()
        for tri in result.get("tri_surfaces", []):
            mesh = ObjectMesh.from_arrays(
                name=tri["name"],
                vertices=tri["vertices"],
                faces=tri["faces"],
                normals=tri.get("normals"),
            )
            self.scene.add_environment_mesh(mesh)

        self._fit_grid_to_wind_field()
        self.status_label.setText(self._format_case_status(result))
        self.viewports.refresh()

    def _format_case_status(self, result: dict) -> str:
        """Compose the status-bar summary for a loaded case."""
        x_n = len(self.wind_field.x_coords)
        y_n = len(self.wind_field.y_coords)
        z_n = len(self.wind_field.z_coords)
        t_n = self.wind_field.time_steps

        parts = [f"Loaded OpenFOAM case: wind {x_n}×{y_n}×{z_n}, {t_n} steps"]

        patches = result.get("patches") or []
        if patches:
            patch_str = ", ".join(f"{p['name']}({p['type']})" for p in patches)
            parts.append(f"patches: {patch_str}")
        else:
            parts.append("patches: (boundary not found)")

        tri_surfaces = result.get("tri_surfaces") or []
        if tri_surfaces:
            parts.append(f"env: {len(tri_surfaces)} mesh" + ("es" if len(tri_surfaces) != 1 else ""))
        else:
            parts.append("env: none")

        text = "; ".join(parts)

        warnings = result.get("warnings") or []
        if warnings:
            text += " (" + "; ".join(warnings) + ")"
        return text
    
    def _fit_grid_to_wind_field(self):
        """Resize and center the ground grid around the loaded wind field (Z-up world)."""
        min_corner, max_corner = self.wind_field.get_bounds()
        min_x, max_x = float(min_corner[0]), float(max_corner[0])
        min_y, max_y = float(min_corner[1]), float(max_corner[1])

        spacing = self._infer_horizontal_grid_spacing()
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0
        half_extent = max((max_x - min_x) / 2.0, (max_y - min_y) / 2.0, spacing)
        half_steps = max(1, int(math.ceil(half_extent / spacing)))

        self.scene.grid_center = (center_x, center_y)
        self.scene.grid_spacing = spacing
        self.scene.grid_size = half_steps * 2

    def _infer_horizontal_grid_spacing(self) -> float:
        """Infer a display grid spacing from loaded wind X/Y coordinates."""
        spacing_candidates = []

        for coords in (self.wind_field.x_coords, self.wind_field.y_coords):
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
