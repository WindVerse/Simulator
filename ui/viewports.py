"""
Viewport Panes & Container
Multiple synchronized 3D viewports over a single shared Scene.

Each viewport renders the same Scene through its own Camera, so the user can
inspect the scene from several angles at once (Single / Dual / Quad layouts).
Top / Front / Side presets are angle-locked orthographic views; Perspective is
a free-orbit perspective camera.
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFrame, QLabel, QComboBox, QSplitter,
    QSizePolicy,
)
from PyQt5.QtCore import Qt, pyqtSignal

from renderer.scene import Scene, Camera
from renderer.opengl_widget import OpenGLWidget


# Preset combo labels <-> Camera.apply_preset names.
_LABEL_TO_PRESET = {
    "Perspective": "perspective",
    "Top": "top",
    "Front": "front",
    "Right": "right",
}
_PRESET_TO_LABEL = {v: k for k, v in _LABEL_TO_PRESET.items()}


class ViewportPane(QWidget):
    """A single 3D view: a thin header (title + preset selector) over a GL view."""

    def __init__(self, scene: Scene, camera: Camera, preset_name: str, parent=None):
        super().__init__(parent)
        self.setObjectName("viewportPane")
        self.setProperty("active", False)
        # Needed so the QSS border/background on a plain QWidget is painted.
        self.setAttribute(Qt.WA_StyledBackground, True)

        self.camera = camera
        self.gl = OpenGLWidget(scene, camera)
        self.gl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.gl.setMinimumSize(120, 120)

        # Header: title label + preset dropdown.
        header = QFrame()
        header.setObjectName("viewportHeader")
        header.setFixedHeight(26)
        h = QHBoxLayout(header)
        h.setContentsMargins(8, 0, 6, 0)
        h.setSpacing(6)

        self.title_label = QLabel(_PRESET_TO_LABEL.get(preset_name, "Perspective"))
        self.title_label.setObjectName("viewportTitle")
        h.addWidget(self.title_label)
        h.addStretch()

        self.preset_combo = QComboBox()
        self.preset_combo.setObjectName("viewportPreset")
        self.preset_combo.addItems(list(_LABEL_TO_PRESET.keys()))
        self.preset_combo.setCurrentText(_PRESET_TO_LABEL.get(preset_name, "Perspective"))
        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        h.addWidget(self.preset_combo)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(header)
        layout.addWidget(self.gl, stretch=1)

    def _on_preset_changed(self, label: str):
        """Apply the chosen view preset to this pane's camera."""
        preset = _LABEL_TO_PRESET.get(label, "perspective")
        self.camera.apply_preset(preset)
        self.title_label.setText(label)
        self.gl.update()

    def set_active(self, active: bool):
        """Toggle the accent border that marks the focused viewport."""
        if bool(self.property("active")) == bool(active):
            return
        self.setProperty("active", bool(active))
        # Re-evaluate the [active] style selector.
        self.style().unpolish(self)
        self.style().polish(self)


class ViewportContainer(QWidget):
    """Holds up to four ViewportPanes and switches between Single/Dual/Quad layouts.

    Re-emits the GL widgets' signals so the main window can stay agnostic of how
    many viewports are visible.
    """

    object_dropped = pyqtSignal(str, float, float, float)
    object_selected = pyqtSignal(object)
    cursor_world_moved = pyqtSignal(float, float, float)

    def __init__(self, scene: Scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self._mode = "single"
        self._overlay = None  # optional floating widget pinned over the views

        # Pane 0 uses the scene's shared camera (the primary view that save/load
        # round-trips); the others get their own camera with a default preset.
        cam_top = Camera(); cam_top.apply_preset("top")
        cam_front = Camera(); cam_front.apply_preset("front")
        cam_right = Camera(); cam_right.apply_preset("right")

        self.panes = [
            ViewportPane(scene, scene.camera, "perspective"),
            ViewportPane(scene, cam_top, "top"),
            ViewportPane(scene, cam_front, "front"),
            ViewportPane(scene, cam_right, "right"),
        ]

        # Resizable nested splitters: outer (vertical) holds two horizontal rows.
        self._top_row = QSplitter(Qt.Horizontal)
        self._top_row.addWidget(self.panes[0])
        self._top_row.addWidget(self.panes[1])
        self._top_row.setChildrenCollapsible(False)

        self._bottom_row = QSplitter(Qt.Horizontal)
        self._bottom_row.addWidget(self.panes[2])
        self._bottom_row.addWidget(self.panes[3])
        self._bottom_row.setChildrenCollapsible(False)

        self._outer = QSplitter(Qt.Vertical)
        self._outer.addWidget(self._top_row)
        self._outer.addWidget(self._bottom_row)
        self._outer.setChildrenCollapsible(False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._outer)

        # Wire each viewport's signals through to aggregate signals.
        for pane in self.panes:
            gl = pane.gl
            gl.object_dropped.connect(
                lambda t, x, y, z, g=gl: self._on_pane_dropped(g, t, x, y, z)
            )
            gl.object_selected.connect(self.object_selected)
            gl.cursor_world_moved.connect(self.cursor_world_moved)
            gl.viewport_activated.connect(lambda g=gl: self._set_active(g))

        self.primary = self.panes[0].gl
        self.active_view = self.primary
        self.set_layout("single")

    # -- layout ------------------------------------------------------------
    def set_layout(self, mode: str):
        """Switch between 'single', 'dual', and 'quad' viewport layouts."""
        mode = mode if mode in ("single", "dual", "quad") else "single"
        self._mode = mode

        self.panes[1].setVisible(mode in ("dual", "quad"))
        self._bottom_row.setVisible(mode == "quad")
        # Pane 0 is always shown; panes 2/3 follow the bottom row's visibility.

        self._set_active(self.panes[0].gl)
        self.refresh()
        self._reposition_overlay()

    # -- overlay -----------------------------------------------------------
    def set_overlay(self, widget: QWidget):
        """Pin a floating widget over the viewports' top-left corner."""
        self._overlay = widget
        widget.setParent(self)
        widget.show()
        self._reposition_overlay()

    def _reposition_overlay(self):
        """Tuck the overlay just below the top-left pane's header, on top."""
        if self._overlay is None:
            return
        header_h = 26  # matches ViewportPane's header height
        margin = 14
        self._overlay.move(margin, header_h + margin)
        self._overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reposition_overlay()

    def _visible_panes(self):
        if self._mode == "single":
            return [self.panes[0]]
        if self._mode == "dual":
            return [self.panes[0], self.panes[1]]
        return list(self.panes)

    # -- helpers used by the main window -----------------------------------
    def refresh(self):
        """Repaint every visible viewport."""
        for pane in self._visible_panes():
            pane.gl.update()

    def set_pending_drop(self, object_type: str):
        """Arm a pending drop on all visible viewports (drop into any of them)."""
        for pane in self._visible_panes():
            pane.gl.set_pending_drop(object_type)

    def reset_active_camera(self):
        """Reset the focused viewport to the default free perspective view."""
        self.active_view.camera.reset()
        self._sync_pane_preset(self.active_view)
        self.refresh()

    # -- internal ----------------------------------------------------------
    def _on_pane_dropped(self, gl, object_type, x, y, z):
        self._set_active(gl)
        self.object_dropped.emit(object_type, x, y, z)

    def _set_active(self, gl):
        self.active_view = gl
        for pane in self.panes:
            pane.set_active(pane.gl is gl)

    def _sync_pane_preset(self, gl):
        """Make a pane's preset dropdown reflect its camera state."""
        for pane in self.panes:
            if pane.gl is gl:
                label = _PRESET_TO_LABEL.get(
                    "perspective" if not pane.camera.locked
                    else _dir_to_preset(pane.camera),
                    "Perspective",
                )
                pane.preset_combo.blockSignals(True)
                pane.preset_combo.setCurrentText(label)
                pane.title_label.setText(label)
                pane.preset_combo.blockSignals(False)
                break


def _dir_to_preset(camera: Camera) -> str:
    """Best-effort name for a locked camera's current direction."""
    import numpy as np
    d = np.asarray(camera._view_dir, dtype=float)
    for name, vec in (("top", (0, 0, 1)), ("front", (0, -1, 0)), ("right", (1, 0, 0))):
        if np.allclose(d, vec):
            return name
    return "perspective"
