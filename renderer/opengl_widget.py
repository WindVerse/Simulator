"""
OpenGL Widget
PyQt5 widget for rendering the 3D scene using OpenGL.
"""

import numpy as np
from typing import Optional, Tuple, List
import sys
import os

from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QPoint
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
    
    def __init__(self, scene: Scene, parent=None):
        """
        Initialize the OpenGL widget.
        
        Args:
            scene: The Scene instance to render
            parent: Parent widget
        """
        super().__init__(parent)
        
        self.scene = scene
        
        # Mouse interaction state
        self._last_mouse_pos: Optional[QPoint] = None
        self._mouse_button: Optional[int] = None
        self._is_dragging = False
        
        # Drop state
        self._pending_drop_type: Optional[str] = None
        
        # Rendering settings
        self._bg_color = (0.1, 0.1, 0.15, 1.0)
        self._grid_color = (0.3, 0.3, 0.35, 1.0)
        self._wind_vector_color = (0.2, 0.6, 1.0, 0.7)
        self._selection_color = (1.0, 0.8, 0.0, 1.0)
        
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
        self.scene.camera.aspect = width / max(height, 1)
    
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
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.scene.camera.fov,
            self.scene.camera.aspect,
            self.scene.camera.near,
            self.scene.camera.far
        )
        
        # Set up view matrix
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            *self.scene.camera.position,
            *self.scene.camera.target,
            *self.scene.camera.up
        )
    
    def _draw_ground(self):
        """Draw the ground plane."""
        glDisable(GL_LIGHTING)
        
        half_size = self.scene.grid_size // 2 * self.scene.grid_spacing
        center_x, center_z = self.scene.grid_center
        
        glColor4f(0.15, 0.15, 0.18, 1.0)
        glBegin(GL_QUADS)
        glVertex3f(center_x - half_size, -0.01, center_z - half_size)
        glVertex3f(center_x + half_size, -0.01, center_z - half_size)
        glVertex3f(center_x + half_size, -0.01, center_z + half_size)
        glVertex3f(center_x - half_size, -0.01, center_z + half_size)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _draw_grid(self):
        """Draw the ground grid."""
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        
        half_size = self.scene.grid_size // 2
        spacing = self.scene.grid_spacing
        center_x, center_z = self.scene.grid_center
        extent = half_size * spacing
        
        glColor4f(*self._grid_color)
        glBegin(GL_LINES)
        
        # Draw grid lines
        for i in range(-half_size, half_size + 1):
            x = center_x + i * spacing
            z = center_z + i * spacing
            
            # X-parallel lines
            glVertex3f(x, 0, center_z - extent)
            glVertex3f(x, 0, center_z + extent)
            
            # Z-parallel lines
            glVertex3f(center_x - extent, 0, z)
            glVertex3f(center_x + extent, 0, z)
        
        glEnd()
        glEnable(GL_LIGHTING)
    
    def _draw_wind_vectors(self):
        """Draw wind velocity vectors."""
        glDisable(GL_LIGHTING)
        glLineWidth(2.0)
        
        # Sample wind vectors at grid points
        wind_points = self.scene.wind_field.get_grid_points()
        velocities = self.scene.wind_field.get_current_velocities()
        
        # Subsample for performance
        step = max(1, len(wind_points) // 200)
        
        glColor4f(*self._wind_vector_color)
        glBegin(GL_LINES)
        
        for i in range(0, len(wind_points), step):
            point = wind_points[i]
            velocity = velocities[i]
            
            # Scale vector for visibility
            scale = 0.3
            end_point = point + velocity * scale
            
            glVertex3f(*point)
            glVertex3f(*end_point)
        
        glEnd()
        
        # Draw arrowheads
        glPointSize(3.0)
        glBegin(GL_POINTS)
        for i in range(0, len(wind_points), step):
            point = wind_points[i]
            velocity = velocities[i]
            scale = 0.3
            end_point = point + velocity * scale
            glVertex3f(*end_point)
        glEnd()
        
        glEnable(GL_LIGHTING)
    
    def _draw_objects(self):
        """Draw all objects in the scene."""
        for obj in self.scene.objects:
            is_selected = obj is self.scene.selected_object
            self._draw_mesh(obj, is_selected)
    
    def _draw_mesh(self, mesh: ObjectMesh, selected: bool = False):
        """
        Draw a single mesh object.
        
        Args:
            mesh: The mesh to draw
            selected: Whether the mesh is selected
        """
        glPushMatrix()
        
        # Apply object transform
        glTranslatef(*mesh.position)
        glScalef(mesh.scale, mesh.scale, mesh.scale)
        
        # Set color
        if selected:
            glColor4f(*self._selection_color)
        else:
            glColor4f(*mesh.color)
        
        # Draw mesh triangles
        glBegin(GL_TRIANGLES)
        
        for face in mesh.faces:
            for vertex_idx in face:
                if vertex_idx < len(mesh.normals):
                    glNormal3fv(mesh.normals[vertex_idx])
                glVertex3fv(mesh.current_vertices[vertex_idx])
        
        glEnd()
        
        # Draw selection outline
        if selected:
            glDisable(GL_LIGHTING)
            glLineWidth(2.0)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
            glColor4f(1.0, 1.0, 0.0, 1.0)
            
            glBegin(GL_TRIANGLES)
            for face in mesh.faces:
                for vertex_idx in face:
                    glVertex3fv(mesh.current_vertices[vertex_idx])
            glEnd()
            
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
            glEnable(GL_LIGHTING)
        
        glPopMatrix()
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        self._last_mouse_pos = event.pos()
        self._mouse_button = event.button()
        
        if event.button() == Qt.LeftButton and not self._pending_drop_type:
            # Try to select an object
            world_pos = self._screen_to_world_current(event.x(), event.y())
            if world_pos is not None:
                obj = self.scene.get_object_at_position(world_pos)
                self.scene.select_object(obj)
                self.object_selected.emit(obj)
                self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if self._pending_drop_type and event.button() == Qt.LeftButton:
            # Place object
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
        
        self._last_mouse_pos = None
        self._mouse_button = None
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse movement."""
        self._emit_cursor_world_position(event)
        
        if self._last_mouse_pos is None:
            return
        
        dx = event.x() - self._last_mouse_pos.x()
        dy = event.y() - self._last_mouse_pos.y()
        
        if self._mouse_button == Qt.RightButton:
            # Orbit camera
            self.scene.camera.orbit(dx * 0.5, -dy * 0.5)
            self.update()
        elif self._mouse_button == Qt.MiddleButton:
            # Pan camera
            self.scene.camera.pan(-dx * 0.02, dy * 0.02)
            self.update()
        
        self._last_mouse_pos = event.pos()
    
    def _emit_cursor_world_position(self, event: QMouseEvent):
        """Emit the current ground-plane world position under the cursor."""
        world_pos = self._screen_to_world_current(event.x(), event.y())
        
        if world_pos is None:
            self.cursor_world_moved.emit(float('nan'), float('nan'), float('nan'))
            return
        
        self.cursor_world_moved.emit(
            float(world_pos[0]),
            float(world_pos[1]),
            float(world_pos[2])
        )
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y() / 120.0
        self.scene.camera.zoom(delta)
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
        
        # Intersect with ground plane (y = 0)
        if abs(ray_dir[1]) < 1e-6:
            return None
        
        t = -ray_origin[1] / ray_dir[1]
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
