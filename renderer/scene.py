"""
Scene Class
Manages the 3D scene containing objects, camera, and wind field.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import sys
import os
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import config as cfg


from objects.object_mesh import ObjectMesh
from wind_data.wind_field import WindField


class Camera:
    """
    Camera for 3D scene viewing.
    
    Supports orbit, pan, and zoom controls.
    """
    
    def __init__(self):
        """Initialize camera with default values."""
        self.position = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        self.target = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        self.up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        # Orbit parameters
        self.distance = 15.0
        self.azimuth = 45.0  # Horizontal angle in degrees
        self.elevation = 30.0  # Vertical angle in degrees
        
        # Zoom limits for ±50m grid viewing
        self.zoom_min = 0.2  # Allow very close zoom (0.2m minimum)
        self.zoom_max = 200.0  # Allow very far zoom (200m maximum)
        
        # Projection parameters
        self.fov = 60.0
        self.near = 0.01  # Reduced from 0.1 for closer near plane
        self.far = 1000.0
        self.aspect = 1.0
        
        self._update_position()
    
    def _update_position(self):
        """Update camera position based on orbit parameters."""
        azimuth_rad = np.radians(self.azimuth)
        elevation_rad = np.radians(self.elevation)
        
        x = self.distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        y = self.distance * np.sin(elevation_rad)
        z = self.distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        
        self.position = self.target + np.array([x, y, z])
    
    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """
        Orbit the camera around the target.
        
        Args:
            delta_azimuth: Change in horizontal angle
            delta_elevation: Change in vertical angle
        """
        self.azimuth += delta_azimuth
        self.elevation = np.clip(self.elevation + delta_elevation, -89.0, 89.0)
        self._update_position()
    
    def zoom(self, delta: float):
        """
        Zoom in or out with smooth exponential scaling.
        
        Args:
            delta: Zoom amount (positive = zoom in)
        """
        # Use exponential scaling for smooth zoom over large ranges
        zoom_speed = 0.1  # Sensitivity factor
        new_distance = self.distance * (1.0 - zoom_speed * delta)
        
        # Clamp to zoom limits
        self.distance = np.clip(new_distance, self.zoom_min, self.zoom_max)
        self._update_position()
    
    def pan(self, delta_x: float, delta_y: float):
        """
        Pan the camera.
        
        Args:
            delta_x: Horizontal pan amount
            delta_y: Vertical pan amount
        """
        # Calculate right and up vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Apply pan
        pan_vector = right * delta_x + up * delta_y
        self.target += pan_vector
        self._update_position()
    
    def get_view_matrix(self) -> np.ndarray:
        """
        Get the view matrix.
        
        Returns:
            4x4 view matrix
        """
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.dot(view[:3, :3], self.position)
        
        return view
    
    def get_projection_matrix(self) -> np.ndarray:
        """
        Get the perspective projection matrix.
        
        Returns:
            4x4 projection matrix
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)
        
        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = f / self.aspect
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0
        
        return proj
    
    def reset(self):
        """Reset camera to default position."""
        self.distance = 15.0
        self.azimuth = 45.0
        self.elevation = 30.0
        self.target = np.array([0.0, 2.0, 0.0], dtype=np.float32)
        self._update_position()


class Scene:
    """
    Manages the 3D scene with objects and wind field.
    
    Attributes:
        objects: List of ObjectMesh instances in the scene
        wind_field: The wind field data
        camera: Scene camera
        grid_visible: Whether to show the grid
        wind_vectors_visible: Whether to show wind vectors
    """
    
    def __init__(self, wind_field: Optional[WindField] = None):
        """
        Initialize the scene.
        
        Args:
            wind_field: Wind field instance (creates default if None)
        """
        self.objects: List[ObjectMesh] = []
        self.wind_field = wind_field or WindField()
        self.camera = Camera()
        
        # Visibility settings
        self.grid_visible = True
        self.wind_vectors_visible = True
        self.ground_visible = True
        
        # Grid settings
        self.grid_size = 100  # Covers -50 to +50 meters with 1.0m spacing
        self.grid_spacing = 1.0
        self.grid_center = (0.0, 0.0)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Selection
        self.selected_object: Optional[ObjectMesh] = None
        
        # Object ID counter
        self._next_id = 0
        self._object_ids: Dict[int, ObjectMesh] = {}

    def _get_default_obj_path(self, object_type: str) -> Optional[str]:
        """Return the bundled OBJ path for a known object type, if present."""
        candidate = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "objects",
            f"{object_type.lower()}.obj"
        )
        return candidate if os.path.exists(candidate) else None

    def _edge_index_from_faces(self, faces: np.ndarray, num_vertices: int) -> torch.Tensor:
        """Build a directed edge index from mesh faces."""
        if faces is None or len(faces) == 0:
            return torch.empty((2, 0), dtype=torch.long)

        faces = np.asarray(faces, dtype=np.int64)
        if faces.ndim != 2 or faces.shape[1] < 3:
            return torch.empty((2, 0), dtype=torch.long)

        min_index = int(faces.min())
        max_index = int(faces.max())
        if min_index < 0 or max_index >= num_vertices:
            raise ValueError(
                f"Mesh face index out of bounds: valid range is [0, {num_vertices - 1}], "
                f"but faces contain [{min_index}, {max_index}]"
            )

        edges = set()
        for face in faces:
            face_vertices = [int(vertex) for vertex in face]
            for start, end in zip(face_vertices, face_vertices[1:] + face_vertices[:1]):
                if start == end:
                    continue
                edges.add((start, end))
                edges.add((end, start))

        if not edges:
            return torch.empty((2, 0), dtype=torch.long)

        return torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()

    def _load_edge_index_for_mesh(self, mesh: ObjectMesh) -> torch.Tensor:
        """Load the trained topology when it matches; otherwise derive topology from faces."""
        num_vertices = mesh.get_vertex_count()

        if num_vertices == cfg.NUM_VERTICES and os.path.exists(cfg.TOPOLOGY_PATH):
            topology = np.load(cfg.TOPOLOGY_PATH)
            if topology.ndim == 2 and topology.shape[0] == 2 and topology.size > 0:
                min_index = int(topology.min())
                max_index = int(topology.max())
                if min_index >= 0 and max_index < num_vertices:
                    return torch.from_numpy(topology).long()

        return self._edge_index_from_faces(mesh.faces, num_vertices)

    def calculate_edge_lengths(self, pos, edge_index):
        """Computes the length of every edge in the mesh."""
        if edge_index.numel() == 0:
            return torch.empty((0,), device=pos.device, dtype=pos.dtype)

        if edge_index.dim() != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index must have shape [2, E], got {tuple(edge_index.shape)}")

        num_vertices = pos.shape[0]
        min_index = int(edge_index.min().item())
        max_index = int(edge_index.max().item())
        if min_index < 0 or max_index >= num_vertices:
            raise ValueError(
                f"edge_index out of bounds for mesh with {num_vertices} vertices: "
                f"found indices [{min_index}, {max_index}]"
            )

        row, col = edge_index
        vec = pos[row] - pos[col]
        return torch.norm(vec, dim=1)
    
    def add_object(
        self,
        object_type: str,
        position: Tuple[float, float, float],
        obj_path: Optional[str] = None
    ) -> ObjectMesh:
        """
        Add an object to the scene.
        
        Args:
            object_type: Type of object (tree, flag, cloth, pole)
            position: World position
            obj_path: Optional path to OBJ file
            
        Returns:
            The created ObjectMesh
        """
        # Create mesh first
        if obj_path is None:
            obj_path = self._get_default_obj_path(object_type)

        mesh = ObjectMesh(object_type, obj_path, position)
        
        # Load a topology that is valid for this mesh.
        edge_index = self._load_edge_index_for_mesh(mesh).to(self.device)
        mesh.edge_index = edge_index
        
        # Get mesh vertices as tensor
        vertices_tensor = torch.from_numpy(mesh.vertices.astype(np.float32)).to(self.device)
        
        # Calculate and store rest_lengths in the mesh
        mesh.rest_lengths = self.calculate_edge_lengths(vertices_tensor, edge_index)
        
        self.objects.append(mesh)
        self._object_ids[self._next_id] = mesh
        self._next_id += 1
        
        return mesh
    
    def remove_object(self, mesh: ObjectMesh):
        """
        Remove an object from the scene.
        
        Args:
            mesh: The mesh to remove
        """
        if mesh in self.objects:
            self.objects.remove(mesh)
            
            # Remove from ID mapping
            for obj_id, obj in list(self._object_ids.items()):
                if obj is mesh:
                    del self._object_ids[obj_id]
                    break
        
        if self.selected_object is mesh:
            self.selected_object = None
    
    def clear_objects(self):
        """Remove all objects from the scene."""
        self.objects.clear()
        self._object_ids.clear()
        self.selected_object = None
    
    def get_object_at_position(
        self,
        position: np.ndarray,
        tolerance: float = 1.0
    ) -> Optional[ObjectMesh]:
        """
        Find an object near a given position.
        
        Args:
            position: World position to check
            tolerance: Distance tolerance
            
        Returns:
            ObjectMesh if found, None otherwise
        """
        for obj in self.objects:
            center = obj.get_center()
            distance = np.linalg.norm(center - position)
            if distance < tolerance:
                return obj
        return None
    
    def select_object(self, mesh: Optional[ObjectMesh]):
        """
        Select an object in the scene.
        
        Args:
            mesh: Object to select (None to deselect)
        """
        self.selected_object = mesh
    
    def move_object(self, mesh: ObjectMesh, new_position: np.ndarray) -> None:
        """
        Move an object to a new position.
        
        Args:
            mesh: Object to move
            new_position: New world position
        """
        if mesh in self.objects:
            mesh.position = np.array(new_position, dtype=np.float32)
    
    def get_wind_at_object(self, mesh: ObjectMesh) -> np.ndarray:
        """
        Get wind velocity at an object's position.
        
        Args:
            mesh: The object mesh
            
        Returns:
            Wind velocity vector
        """
        center = mesh.get_center()
        return self.wind_field.get_velocity_at_position(center)
    
    def snap_to_grid(self, position: np.ndarray) -> np.ndarray:
        """
        Snap a position to the nearest grid point.
        
        Args:
            position: World position
            
        Returns:
            Snapped position
        """
        center_x, center_z = self.grid_center
        snapped = position.copy()
        snapped[0] = (
            center_x +
            np.round((position[0] - center_x) / self.grid_spacing) * self.grid_spacing
        )
        snapped[2] = (
            center_z +
            np.round((position[2] - center_z) / self.grid_spacing) * self.grid_spacing
        )
        snapped[1] = 0  # Keep y at ground level
        return snapped
    
    def get_grid_points(self) -> np.ndarray:
        """
        Get all grid point positions.
        
        Returns:
            Array of grid point positions
        """
        center_x, center_z = self.grid_center
        x = center_x + np.arange(-self.grid_size // 2, self.grid_size // 2 + 1) * self.grid_spacing
        z = center_z + np.arange(-self.grid_size // 2, self.grid_size // 2 + 1) * self.grid_spacing
        
        X, Z = np.meshgrid(x, z)
        Y = np.zeros_like(X)
        
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return points.astype(np.float32)
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the scene.
        
        Returns:
            Tuple of (min_corner, max_corner)
        """
        center_x, center_z = self.grid_center
        half_size = self.grid_size // 2 * self.grid_spacing
        min_corner = np.array([center_x - half_size, 0, center_z - half_size])
        max_corner = np.array([center_x + half_size, 10, center_z + half_size])
        
        # Include objects
        for obj in self.objects:
            obj_min, obj_max = obj.get_bounding_box()
            min_corner = np.minimum(min_corner, obj_min)
            max_corner = np.maximum(max_corner, obj_max)
        
        return min_corner, max_corner
    
    def toggle_grid(self):
        """Toggle grid visibility."""
        self.grid_visible = not self.grid_visible
    
    def toggle_wind_vectors(self):
        """Toggle wind vector visibility."""
        self.wind_vectors_visible = not self.wind_vectors_visible
    
    def reset_all_objects(self):
        """Reset all objects to their original state."""
        for obj in self.objects:
            obj.reset_to_original()
    
    def serialize(self) -> dict:
        """
        Serialize scene state to dictionary.
        
        Returns:
            Scene state dictionary
        """
        return {
            'objects': [
                {
                    'type': obj.name,
                    'position': obj.position.tolist(),
                    'scale': obj.scale
                }
                for obj in self.objects
            ],
            'camera': {
                'distance': self.camera.distance,
                'azimuth': self.camera.azimuth,
                'elevation': self.camera.elevation,
                'target': self.camera.target.tolist()
            },
            'settings': {
                'grid_visible': self.grid_visible,
                'wind_vectors_visible': self.wind_vectors_visible
            }
        }
    
    def deserialize(self, data: dict):
        """
        Restore scene state from dictionary.
        
        Args:
            data: Scene state dictionary
        """
        self.clear_objects()
        
        for obj_data in data.get('objects', []):
            self.add_object(
                obj_data['type'],
                tuple(obj_data['position'])
            )
        
        if 'camera' in data:
            cam = data['camera']
            self.camera.distance = cam.get('distance', 15.0)
            self.camera.azimuth = cam.get('azimuth', 45.0)
            self.camera.elevation = cam.get('elevation', 30.0)
            self.camera.target = np.array(cam.get('target', [0, 2, 0]))
            self.camera._update_position()
        
        if 'settings' in data:
            settings = data['settings']
            self.grid_visible = settings.get('grid_visible', True)
            self.wind_vectors_visible = settings.get('wind_vectors_visible', True)
