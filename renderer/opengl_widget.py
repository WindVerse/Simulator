"""
OpenGL Widget
PyQt5 widget for rendering the 3D scene using OpenGL.
"""

import numpy as np
from typing import Optional, Tuple, List
import sys
import os

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, QElapsedTimer, pyqtSignal, QPoint
from PyQt5.QtGui import QMouseEvent, QWheelEvent

from OpenGL.GL import *
from OpenGL.GLU import *

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from renderer.scene import Scene
from objects.object_mesh import ObjectMesh


class OpenGLWidget(QOpenGLWidget):
    """
    OpenGL widget for 3D scene rendering.
    
    Handles:
    - Scene rendering (objects, grid, wind vectors)
    - Mouse interaction (orbit, pan, zoom)
    - Object picking and placement
    
    Signals:
        object_dropped: Emitted when an object is dropped at a position
        object_selected: Emitted when an object is clicked
        cursor_world_moved: Emitted with the ground-plane world position under the cursor
    """
    
    object_dropped = pyqtSignal(str, float, float, float)
    object_selected = pyqtSignal(object)
    cursor_world_moved = pyqtSignal(float, float, float)
    viewport_activated = pyqtSignal()  # this viewport received interaction

    def __init__(self, scene: Scene, camera=None, parent=None):
        """
        Initialize the OpenGL widget.

        Args:
            scene: The Scene instance to render
            camera: The Camera this viewport renders through. Defaults to the
                scene's shared camera (single-viewport / back-compat).
            parent: Parent widget
        """
        super().__init__(parent)

        self.scene = scene
        # Each viewport renders the shared scene through its own camera.
        self.camera = camera if camera is not None else scene.camera

        # Mouse interaction state
        self._last_mouse_pos: Optional[QPoint] = None
        self._mouse_button: Optional[int] = None
        self._is_dragging = False
        
        # Object dragging state
        self._dragged_object: Optional[ObjectMesh] = None
        self._drag_start_pos: Optional[np.ndarray] = None
        self._drag_offset: Optional[np.ndarray] = None
        
        # Grid hover state
        self._hovered_grid_cell: Optional[Tuple[float, float]] = None
        
        # Drop state
        self._pending_drop_type: Optional[str] = None

        # Camera focus animation
        self._focus_anim_timer = QTimer(self)
        self._focus_anim_timer.setInterval(16)  # ~60 FPS
        self._focus_anim_timer.timeout.connect(self._on_focus_anim_tick)
        self._focus_anim_elapsed = QElapsedTimer()

        # Rendering settings
        self._bg_color = (0.1, 0.1, 0.15, 1.0)
        self._grid_color = (0.3, 0.3, 0.35, 1.0)
        self._wind_vector_color = (0.2, 0.6, 1.0, 0.7)
        self._selection_color = (1.0, 0.8, 0.0, 1.0)

        # Wind-vector geometry is built and cached on the shared Scene
        # (Scene.get_wind_geometry) so every viewport reuses one build.

        # Enable mouse tracking for hover effects
        self.setMouseTracking(True)
        
        # Accept drops
        self.setAcceptDrops(True)
        
        # Set focus policy
        self.setFocusPolicy(Qt.StrongFocus)
    
    def initializeGL(self):
        """Initialize OpenGL settings."""
        glClearColor(*self._bg_color)
        
        # Enable depth testing
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # Enable blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Enable lighting
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set light position
        glLightfv(GL_LIGHT0, GL_POSITION, [1.0, 1.0, 1.0, 0.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.3, 0.3, 0.3, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.7, 0.7, 0.7, 1.0])
        
        # Enable smooth shading
        glShadeModel(GL_SMOOTH)
        
        # Enable line smoothing
        glEnable(GL_LINE_SMOOTH)
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST)
    
    def resizeGL(self, width: int, height: int):
        """Handle widget resize."""
        glViewport(0, 0, width, height)
        self.camera.aspect = width / max(height, 1)
    
    def paintGL(self):
        """Render the scene."""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        self._apply_camera_matrices()
        
        # Draw scene elements
        if self.scene.ground_visible:
            self._draw_ground()
        
        if self.scene.grid_visible:
            self._draw_grid()
        
        if self.scene.wind_vectors_visible:
            self._draw_wind_vectors()
        
        self._draw_objects()
    
    def _apply_camera_matrices(self):
        """Apply the current camera projection and view matrices."""
        cam = self.camera

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        if cam.projection == "orthographic":
            half_h = cam.ortho_half_height()
            half_w = half_h * cam.aspect
            glOrtho(-half_w, half_w, -half_h, half_h, cam.near, cam.far)
        else:
            gluPerspective(cam.fov, cam.aspect, cam.near, cam.far)

        # Set up view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            *cam.position,
            *cam.target,
            *cam.up
        )
    
    def _draw_ground(self):
        """Draw the ground plane on Z=0 (Z-up world)."""
        glDisable(GL_LIGHTING)

        half_size = self.scene.grid_size // 2 * self.scene.grid_spacing
        center_x, center_y = self.scene.grid_center

        glColor4f(0.15, 0.15, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(center_x - half_size, center_y - half_size, -0.01)
        glVertex3f(center_x + half_size, center_y - half_size, -0.01)
        glVertex3f(center_x + half_size, center_y + half_size, -0.01)
        glVertex3f(center_x - half_size, center_y + half_size, -0.01)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_grid(self):
        """Draw the ground grid on Z=0 (Z-up world)."""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)

        # Grid lines come from the scene's shared, cached vertex array so a fine
        # 1 m grid over a large field is one glDrawArrays per viewport (no Python
        # per-line loop). Rebuilt only when the grid extent/spacing/center changes.
        verts = self.scene.get_grid_geometry()
        glColor4f(*self._grid_color)
        glEnableClientState(GL_VERTEX_ARRAY)
        try:
            glVertexPointer(3, GL_FLOAT, 0, verts)
            glDrawArrays(GL_LINES, 0, len(verts))
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)

        # Draw highlighted grid cell under cursor
        self._draw_hovered_grid_cell()

        # Draw coordinate labels
        self._draw_grid_labels()

        glEnable(GL_LIGHTING)

    def _draw_hovered_grid_cell(self):
        """Draw a highlighted square for the grid cell under the cursor."""
        if self._hovered_grid_cell is None:
            return

        x, y = self._hovered_grid_cell
        half_spacing = self.scene.grid_spacing / 2.0

        # Draw semi-transparent highlight quad
        glDisable(GL_LIGHTING)
        glDisable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        # Light blue highlight with transparency
        glColor4f(0.2, 0.6, 1.0, 0.2)
        glBegin(GL_QUADS)
        glVertex3f(x - half_spacing, y - half_spacing, 0.001)
        glVertex3f(x + half_spacing, y - half_spacing, 0.001)
        glVertex3f(x + half_spacing, y + half_spacing, 0.001)
        glVertex3f(x - half_spacing, y + half_spacing, 0.001)
        glEnd()

        # Draw outline
        glLineWidth(2.0)
        glColor4f(0.2, 0.8, 1.0, 0.8)
        glBegin(GL_LINE_LOOP)
        glVertex3f(x - half_spacing, y - half_spacing, 0.002)
        glVertex3f(x + half_spacing, y - half_spacing, 0.002)
        glVertex3f(x + half_spacing, y + half_spacing, 0.002)
        glVertex3f(x - half_spacing, y + half_spacing, 0.002)
        glEnd()

        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

    def _draw_grid_labels(self):
        """Draw X and Y axis ticks at the edge of the ground grid."""
        half_size = self.scene.grid_size // 2
        spacing = self.scene.grid_spacing
        center_x, center_y = self.scene.grid_center
        extent = half_size * spacing

        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor4f(0.5, 0.5, 0.5, 0.8)

        # Draw axis labels every 10 meters
        label_interval = 10
        for i in range(-half_size, half_size + 1, label_interval // int(spacing)):
            x = center_x + i * spacing
            y = center_y + i * spacing

            # X-axis tick (along the -Y edge)
            if abs(i * spacing) < extent + 1:
                glBegin(GL_LINES)
                glVertex3f(x, center_y - extent - 1, 0)
                glVertex3f(x, center_y - extent - 0.5, 0)
                glEnd()

            # Y-axis tick (along the -X edge)
            if abs(i * spacing) < extent + 1:
                glBegin(GL_LINES)
                glVertex3f(center_x - extent - 1, y, 0)
                glVertex3f(center_x - extent - 0.5, y, 0)
                glEnd()
    
    _COMPONENT_COLORS = (
        (1.0, 0.3, 0.3, 0.9),  # X — red
        (0.3, 1.0, 0.3, 0.9),  # Y — green
        (0.4, 0.5, 1.0, 0.9),  # Z — blue
    )

    def _draw_wind_vectors(self):
        """Draw wind velocity vectors from the scene's shared geometry cache.

        Honors scene.wind_display_mode (resultant / components),
        scene.wind_downsample_stride, and scene.wind_vector_scale. The geometry
        is built once per state change by Scene.get_wind_geometry() and reused by
        every viewport — pan/zoom (and sibling viewports) do not rebuild it.
        """
        mode = getattr(self.scene, "wind_display_mode", "resultant")
        geom = self.scene.get_wind_geometry()
        if geom is None:
            return

        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        glEnableClientState(GL_VERTEX_ARRAY)
        try:
            if mode == "components":
                bands = geom or [None, None, None]
                for axis, color in enumerate(self._COMPONENT_COLORS):
                    verts = bands[axis]
                    if verts is None or verts.size == 0:
                        continue
                    glColor4f(*color)
                    glVertexPointer(3, GL_FLOAT, 0, verts)
                    glDrawArrays(GL_LINES, 0, len(verts))
            else:
                verts, colors = geom or (None, None)
                if verts is not None and verts.size > 0:
                    glVertexPointer(3, GL_FLOAT, 0, verts)
                    if colors is not None:
                        glEnableClientState(GL_COLOR_ARRAY)
                        glColorPointer(4, GL_FLOAT, 0, colors)
                    else:
                        glColor4f(*self._wind_vector_color)
                    glDrawArrays(GL_LINES, 0, len(verts))
                    if colors is not None:
                        glDisableClientState(GL_COLOR_ARRAY)
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
            glEnable(GL_LIGHTING)

    def _draw_objects(self):
        """Draw all objects in the scene."""
        for obj in self.scene.objects:
            is_selected = obj is self.scene.selected_object
            self._draw_mesh(obj, is_selected)

        if getattr(self.scene, "environment_visible", True):
            for env_mesh in getattr(self.scene, "environment_meshes", ()):
                self._draw_mesh(env_mesh, selected=False)
    
    def _draw_mesh(self, mesh: ObjectMesh, selected: bool = False):
        """
        Draw a single mesh object with enhanced selection effects.
        
        Args:
            mesh: The mesh to draw
            selected: Whether the mesh is selected
        """
        glPushMatrix()

        # Apply object transform.
        # flag.obj is authored Z-up (pole along X, height along Z) — no rotation needed.
        glTranslatef(*mesh.position)
        glScalef(mesh.scale, mesh.scale, mesh.scale)

        # Set color with glow effect for selected objects
        if selected:
            # Brighten color for selection glow
            color = (
                min(1.0, self._selection_color[0] + 0.3),
                min(1.0, self._selection_color[1] + 0.3),
                min(1.0, self._selection_color[2] + 0.3),
                self._selection_color[3]
            )
            glColor4f(*color)
        else:
            glColor4f(*mesh.color)

        # Draw mesh triangles via client vertex arrays (no per-vertex Python
        # loop, no per-context GL objects — safe across all viewports).
        verts = np.ascontiguousarray(mesh.current_vertices, dtype=np.float32)
        indices = mesh.get_triangle_indices()
        if verts.size == 0 or indices.size == 0:
            glPopMatrix()
            return

        has_normals = mesh.normals.shape == verts.shape
        normals = (
            np.ascontiguousarray(mesh.normals, dtype=np.float32)
            if has_normals else None
        )

        glEnableClientState(GL_VERTEX_ARRAY)
        try:
            glVertexPointer(3, GL_FLOAT, 0, verts)
            if has_normals:
                glEnableClientState(GL_NORMAL_ARRAY)
                glNormalPointer(GL_FLOAT, 0, normals)

            glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, indices)

            if has_normals:
                glDisableClientState(GL_NORMAL_ARRAY)

            # Draw selection effects
            if selected:
                glDisable(GL_LIGHTING)
                glLineWidth(3.0)
                glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
                glColor4f(1.0, 1.0, 0.0, 1.0)

                # Outline: same geometry, wireframe pass
                glDrawElements(GL_TRIANGLES, indices.size, GL_UNSIGNED_INT, indices)

                glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

                # Draw bounding box
                self._draw_bounding_box(mesh)

                glEnable(GL_LIGHTING)
        finally:
            glDisableClientState(GL_VERTEX_ARRAY)
            glPopMatrix()
    
    def _draw_bounding_box(self, mesh: ObjectMesh):
        """Draw a bounding box around a mesh."""
        vertices = mesh.current_vertices
        
        if len(vertices) == 0:
            return
        
        # Find bounding box
        min_point = np.min(vertices, axis=0)
        max_point = np.max(vertices, axis=0)
        
        glLineWidth(2.0)
        glColor4f(0.2, 1.0, 1.0, 0.8)
        
        # Draw box edges
        glBegin(GL_LINES)
        
        # Bottom face
        glVertex3fv(min_point)
        glVertex3fv([max_point[0], min_point[1], min_point[2]])
        
        glVertex3fv([max_point[0], min_point[1], min_point[2]])
        glVertex3fv([max_point[0], min_point[1], max_point[2]])
        
        glVertex3fv([max_point[0], min_point[1], max_point[2]])
        glVertex3fv([min_point[0], min_point[1], max_point[2]])
        
        glVertex3fv([min_point[0], min_point[1], max_point[2]])
        glVertex3fv(min_point)
        
        # Top face
        glVertex3fv([min_point[0], max_point[1], min_point[2]])
        glVertex3fv([max_point[0], max_point[1], min_point[2]])
        
        glVertex3fv([max_point[0], max_point[1], min_point[2]])
        glVertex3fv(max_point)
        
        glVertex3fv(max_point)
        glVertex3fv([min_point[0], max_point[1], max_point[2]])
        
        glVertex3fv([min_point[0], max_point[1], max_point[2]])
        glVertex3fv([min_point[0], max_point[1], min_point[2]])
        
        # Vertical edges
        glVertex3fv(min_point)
        glVertex3fv([min_point[0], max_point[1], min_point[2]])
        
        glVertex3fv([max_point[0], min_point[1], min_point[2]])
        glVertex3fv([max_point[0], max_point[1], min_point[2]])
        
        glVertex3fv([max_point[0], min_point[1], max_point[2]])
        glVertex3fv(max_point)
        
        glVertex3fv([min_point[0], min_point[1], max_point[2]])
        glVertex3fv([min_point[0], max_point[1], max_point[2]])
        
        glEnd()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        self._last_mouse_pos = event.pos()
        self._mouse_button = event.button()
        self.viewport_activated.emit()

        if event.button() == Qt.LeftButton and not self._pending_drop_type:
            # Try to select an object or start dragging it
            world_pos = self._screen_to_world_current(event.x(), event.y())
            if world_pos is not None:
                obj = self.scene.get_object_at_position(world_pos)
                
                if obj is not None:
                    # Start dragging object
                    self._dragged_object = obj
                    self._drag_start_pos = obj.position.copy()
                    self._drag_offset = obj.position - world_pos
                    self.scene.select_object(obj)
                    self.object_selected.emit(obj)
                    self._is_dragging = True
                else:
                    # No object, just select/deselect
                    self.scene.select_object(None)
                    self.object_selected.emit(None)
                
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if self._pending_drop_type and event.button() == Qt.LeftButton:
            # Place object from library
            world_pos = self._screen_to_world_current(event.x(), event.y())
            if world_pos is not None:
                snapped_pos = self.scene.snap_to_grid(world_pos)
                self.object_dropped.emit(
                    self._pending_drop_type,
                    snapped_pos[0],
                    snapped_pos[1],
                    snapped_pos[2]
                )
            self._pending_drop_type = None
            self.setCursor(Qt.ArrowCursor)
        
        # Handle object drag completion
        if self._dragged_object and event.button() == Qt.LeftButton:
            # Snap dragged object's X/Y to grid; preserve Z if lifted off the ground
            current_z = float(self._dragged_object.position[2])
            snapped_pos = self.scene.snap_to_grid(self._dragged_object.position)
            if current_z > 1e-4:
                snapped_pos[2] = current_z
            self.scene.move_object(self._dragged_object, snapped_pos)
            
            # Emit signal that object was moved
            payload = self._dragged_object.get_info_payload()
            print(f"Moved {payload['object_type']} to ({snapped_pos[0]:.1f}, {snapped_pos[1]:.1f}, {snapped_pos[2]:.1f})")
            
            self._dragged_object = None
            self._drag_start_pos = None
            self._drag_offset = None
            self._is_dragging = False
            self.update()
        
        self._last_mouse_pos = None
        self._mouse_button = None
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        self._emit_cursor_world_position(event)
        
        if self._last_mouse_pos is None:
            return
        
        dx = event.x() - self._last_mouse_pos.x()
        dy = event.y() - self._last_mouse_pos.y()
        
        # Handle object dragging
        if self._dragged_object and self._mouse_button == Qt.LeftButton:
            shift_held = bool(event.modifiers() & Qt.ShiftModifier)
            if shift_held:
                # Lift mode: lock X/Y, drive Z by vertical mouse delta (mouse up = lift)
                lift_speed = 0.05
                current = self._dragged_object.position.copy()
                current[2] = max(0.0, float(current[2]) - dy * lift_speed)
                self.scene.move_object(self._dragged_object, current)
                self.update()
            else:
                world_pos = self._screen_to_world_current(event.x(), event.y())
                if world_pos is not None:
                    # Move object with offset, keep on ground (Z-up world)
                    new_pos = world_pos + self._drag_offset
                    new_pos[2] = 0
                    self.scene.move_object(self._dragged_object, new_pos)
                    self.update()
        elif self._mouse_button == Qt.RightButton:
            # Orbit camera (no-op for angle-locked orthographic presets)
            self._cancel_focus_animation()
            self.camera.orbit(dx * 0.5, -dy * 0.5)
            self.update()
        elif self._mouse_button == Qt.MiddleButton:
            # Pan camera
            self._cancel_focus_animation()
            self.camera.pan(-dx * 0.02, dy * 0.02)
            self.update()
        elif self._mouse_button == Qt.LeftButton and not self._dragged_object:
            # Pan camera on left button click on empty grid
            self._cancel_focus_animation()
            self.camera.pan(-dx * 0.02, dy * 0.02)
            self.update()
        
        self._last_mouse_pos = event.pos()
    
    def _emit_cursor_world_position(self, event: QMouseEvent):
        """Emit the current ground-plane world position under the cursor and track hovered grid cell."""
        world_pos = self._screen_to_world_current(event.x(), event.y())
        
        if world_pos is None:
            self.cursor_world_moved.emit(float('nan'), float('nan'), float('nan'))
            self._hovered_grid_cell = None
            return
        
        # Track hovered grid cell for visualization (X/Y horizontal plane)
        snapped = self.scene.snap_to_grid(world_pos)
        self._hovered_grid_cell = (float(snapped[0]), float(snapped[1]))
        
        self.cursor_world_moved.emit(
            float(world_pos[0]),
            float(world_pos[1]),
            float(world_pos[2])
        )
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom."""
        self._cancel_focus_animation()
        delta = event.angleDelta().y() / 120.0
        self.camera.zoom(delta)
        self.update()
    
    def _screen_to_world(
        self,
        screen_x: int,
        screen_y: int
    ) -> Optional[np.ndarray]:
        """
        Convert screen coordinates to world coordinates on the ground plane.
        
        Args:
            screen_x: Screen X coordinate
            screen_y: Screen Y coordinate
            
        Returns:
            World position or None if no intersection
        """
        # Get viewport and matrices
        viewport = glGetIntegerv(GL_VIEWPORT)
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        projection = glGetDoublev(GL_PROJECTION_MATRIX)
        
        # Flip Y coordinate
        screen_y = viewport[3] - screen_y
        
        # Get near and far points
        near_point = gluUnProject(screen_x, screen_y, 0.0, modelview, projection, viewport)
        far_point = gluUnProject(screen_x, screen_y, 1.0, modelview, projection, viewport)
        
        # Ray direction
        ray_dir = np.array(far_point) - np.array(near_point)
        ray_origin = np.array(near_point)
        
        # Intersect with ground plane (z = 0, Z-up world)
        if abs(ray_dir[2]) < 1e-6:
            return None

        t = -ray_origin[2] / ray_dir[2]
        if t < 0:
            return None

        intersection = ray_origin + t * ray_dir
        return intersection.astype(np.float32)
    
    def _screen_to_world_current(
        self,
        screen_x: int,
        screen_y: int
    ) -> Optional[np.ndarray]:
        """Convert screen coordinates using the current GL context and camera."""
        self.makeCurrent()
        self._apply_camera_matrices()
        return self._screen_to_world(screen_x, screen_y)
    
    def set_pending_drop(self, object_type: str):
        """
        Set the object type for next drop.
        
        Args:
            object_type: Type of object to drop
        """
        self._pending_drop_type = object_type
        self.setCursor(Qt.CrossCursor)
    
    def cancel_pending_drop(self):
        """Cancel any pending drop operation."""
        self._pending_drop_type = None
        self.setCursor(Qt.ArrowCursor)

    def start_focus_on_object(self, mesh: ObjectMesh):
        """Center this viewport on the given mesh.

        Free perspective views animate a smooth dolly-in; angle-locked
        orthographic presets just recenter (no rotation).
        """
        if mesh is None:
            return
        center = mesh.get_center()

        if self.camera.locked:
            # Keep the fixed preset angle; recenter on the object.
            self.camera.target = np.asarray(center, dtype=np.float32).reshape(3).copy()
            self.camera._update_position()
            self.update()
            return

        min_c, max_c = mesh.get_bounding_box()
        radius = float(np.linalg.norm(max_c - min_c) * 0.5)
        if not np.isfinite(radius) or radius <= 0.0:
            radius = 1.0
        self.camera.focus_on(center, radius=radius)
        self._focus_anim_elapsed.restart()
        self._focus_anim_timer.start()

    def _cancel_focus_animation(self):
        """Cancel a running focus animation (no-op if not active)."""
        if self._focus_anim_timer.isActive():
            self._focus_anim_timer.stop()
        self.camera.cancel_focus_animation()

    def _on_focus_anim_tick(self):
        """Per-frame tick that advances the camera focus animation."""
        dt = self._focus_anim_elapsed.restart() / 1000.0
        still_animating = self.camera.tick_focus_animation(dt)
        self.update()
        if not still_animating:
            self._focus_anim_timer.stop()

    # Drag and drop support
    def dragEnterEvent(self, event):
        """Handle drag enter."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dragMoveEvent(self, event):
        """Handle drag move."""
        if event.mimeData().hasText():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """Handle drop."""
        if event.mimeData().hasText():
            object_type = event.mimeData().text()
            pos = event.pos()
            
            # Convert to world coordinates
            world_pos = self._screen_to_world_current(pos.x(), pos.y())
            
            if world_pos is not None:
                snapped_pos = self.scene.snap_to_grid(world_pos)
                self.object_dropped.emit(
                    object_type,
                    snapped_pos[0],
                    snapped_pos[1],
                    snapped_pos[2]
                )
            
            event.acceptProposedAction()
