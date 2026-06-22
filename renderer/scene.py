"""
Scene Class
Manages the 3D scene containing objects, camera, and wind field.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict
import sys
import os
import math
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_paths
from models import config as cfg


from objects.object_mesh import ObjectMesh
from wind_data.wind_field import WindField
from renderer.wind_geometry import build_wind_geometry


_DEFAULT_TARGET = np.array([0.0, 0.0, 2.0], dtype=np.float32)
_DEFAULT_DISTANCE = 15.0
_DEFAULT_AZIMUTH = 45.0
_DEFAULT_ELEVATION = 30.0


class Camera:
    """
    Camera for 3D scene viewing.

    Supports orbit, pan, and zoom controls.
    """

    def __init__(self):
        """Initialize camera with default values."""
        self.position = np.array([10.0, 10.0, 10.0], dtype=np.float32)
        self.target = _DEFAULT_TARGET.copy()
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

        # Orbit parameters
        self.distance = _DEFAULT_DISTANCE
        self.azimuth = _DEFAULT_AZIMUTH  # Horizontal angle in degrees
        self.elevation = _DEFAULT_ELEVATION  # Vertical angle in degrees

        # Zoom limits for ±50m grid viewing
        self.zoom_min = 0.2  # Allow very close zoom (0.2m minimum)
        self.zoom_max = 200.0  # Allow very far zoom (200m maximum)

        # Projection parameters
        self.fov = 60.0
        self.near = 0.01  # Reduced from 0.1 for closer near plane
        self.far = 1000.0
        self.aspect = 1.0

        # View preset state. Free perspective by default; orthographic presets
        # (top/front/right) lock the viewing angle so only pan/zoom apply.
        self.projection = "perspective"  # "perspective" | "orthographic"
        self.locked = False              # True => orbit disabled (fixed angle)
        self._view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # target -> camera
        self._view_up = self.up.copy()

        # Focus-animation state
        self._anim_active = False
        self._anim_t = 0.0
        self._anim_duration = 0.45
        self._anim_start_target = self.target.copy()
        self._anim_end_target = self.target.copy()
        self._anim_start_distance = self.distance
        self._anim_end_distance = self.distance
        self._anim_start_azimuth = self.azimuth
        self._anim_end_azimuth = self.azimuth
        self._anim_start_elevation = self.elevation
        self._anim_end_elevation = self.elevation

        self._update_position()
    
    def _update_position(self):
        """Update camera position based on orbit parameters (Z-up world)."""
        if self.locked:
            # Fixed viewing direction (orthographic presets); zoom/pan only.
            self.position = self.target + self._view_dir * self.distance
            return

        azimuth_rad = np.radians(self.azimuth)
        elevation_rad = np.radians(self.elevation)

        # Horizontal (X/Y) ring at radius cos(elevation), vertical (Z) lift = sin(elevation)
        horizontal = self.distance * np.cos(elevation_rad)
        x = horizontal * np.sin(azimuth_rad)
        y = horizontal * np.cos(azimuth_rad)
        z = self.distance * np.sin(elevation_rad)

        self.position = self.target + np.array([x, y, z])

    def apply_preset(self, name: str):
        """
        Switch this camera to a named view preset.

        Args:
            name: "perspective" | "top" | "front" | "right"
                  "perspective" is a free-orbit perspective camera; the others
                  are angle-locked orthographic views (pan + zoom only).
        """
        name = (name or "perspective").lower()

        # target -> camera direction and up vector for each locked preset (Z-up world).
        ortho_presets = {
            "top":   (np.array([0.0, 0.0, 1.0], dtype=np.float32),
                      np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            "front": (np.array([0.0, -1.0, 0.0], dtype=np.float32),
                      np.array([0.0, 0.0, 1.0], dtype=np.float32)),
            "right": (np.array([1.0, 0.0, 0.0], dtype=np.float32),
                      np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        }

        if name in ortho_presets:
            view_dir, view_up = ortho_presets[name]
            self.projection = "orthographic"
            self.locked = True
            self._view_dir = view_dir
            self._view_up = view_up.copy()
            self.up = view_up.copy()
            self._anim_active = False
            self._update_position()
        else:
            # Free perspective: restore the default inspect angle.
            self.projection = "perspective"
            self.locked = False
            self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
            self.reset()

    def ortho_half_height(self) -> float:
        """Half-height of the orthographic view volume (world units).

        Tied to ``distance`` so the existing zoom control scales the ortho view.
        """
        return max(0.01, float(self.distance) * 0.5)

    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """
        Orbit the camera around the target.

        Args:
            delta_azimuth: Change in horizontal angle
            delta_elevation: Change in vertical angle
        """
        if self.locked:
            return  # Presets keep a fixed viewing angle.
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
        """Reset camera to the default free perspective view."""
        self.distance = _DEFAULT_DISTANCE
        self.azimuth = _DEFAULT_AZIMUTH
        self.elevation = _DEFAULT_ELEVATION
        self.target = _DEFAULT_TARGET.copy()
        self.projection = "perspective"
        self.locked = False
        self.up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._anim_active = False
        self._update_position()

    def focus_on(self, target_world: np.ndarray, radius: float = 1.5, duration: float = 0.45):
        """
        Begin a smooth animated focus on a world-space point.

        Args:
            target_world: World position to center on (3,)
            radius: Approximate object radius used to pick a framing distance
            duration: Animation duration in seconds
        """
        target_world = np.asarray(target_world, dtype=np.float32).reshape(3)

        fov_rad = np.radians(self.fov)
        end_distance = max(2.5, float(radius) * 2.5 / np.tan(fov_rad / 2.0))
        end_distance = float(np.clip(end_distance, self.zoom_min, self.zoom_max))

        # Snapshot current state
        self._anim_start_target = self.target.copy()
        self._anim_start_distance = float(self.distance)
        self._anim_start_azimuth = float(self.azimuth)
        self._anim_start_elevation = float(self.elevation)

        # End state — standard inspect angle (matches reset())
        self._anim_end_target = target_world.copy()
        self._anim_end_distance = end_distance

        # Azimuth shortest-arc: pick end angle within ±180° of start
        end_az = _DEFAULT_AZIMUTH
        delta = ((end_az - self._anim_start_azimuth) + 180.0) % 360.0 - 180.0
        self._anim_end_azimuth = self._anim_start_azimuth + delta
        self._anim_end_elevation = _DEFAULT_ELEVATION

        self._anim_duration = max(0.05, float(duration))
        self._anim_t = 0.0
        self._anim_active = True

    def tick_focus_animation(self, dt_seconds: float) -> bool:
        """
        Advance the focus animation by dt_seconds.

        Returns:
            True while still animating, False once finished (or inactive).
        """
        if not self._anim_active:
            return False

        self._anim_t = min(1.0, self._anim_t + float(dt_seconds) / self._anim_duration)
        # Ease-out cubic
        t = self._anim_t
        e = 1.0 - (1.0 - t) ** 3

        self.target = self._anim_start_target + (self._anim_end_target - self._anim_start_target) * e
        self.distance = self._anim_start_distance + (self._anim_end_distance - self._anim_start_distance) * e
        self.azimuth = self._anim_start_azimuth + (self._anim_end_azimuth - self._anim_start_azimuth) * e
        self.elevation = self._anim_start_elevation + (self._anim_end_elevation - self._anim_start_elevation) * e

        self._update_position()

        if self._anim_t >= 1.0:
            self._anim_active = False
            return False
        return True

    def cancel_focus_animation(self):
        """Stop the focus animation, leaving the camera at its current interpolated state."""
        self._anim_active = False


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
        self.environment_meshes: List[ObjectMesh] = []
        self.wind_field = wind_field or WindField()
        self.camera = Camera()

        # Visibility settings
        self.grid_visible = True
        self.wind_vectors_visible = True
        self.ground_visible = True
        self.environment_visible = True

        # Wind vector display
        self.wind_display_mode = "resultant"   # "resultant" or "components"
        self.wind_downsample_stride = 1        # 1 = show every vector
        self.wind_vector_scale = 0.3           # auto-recomputed after data load
        self.wind_color_by_speed = True        # apply Beaufort colormap in resultant mode

        # Two-tier display for large (streaming) cases: a thinned global field
        # plus a full-resolution patch around each placed object.
        self.wind_coarse_spacing_m = 5.0       # global field spacing (matches cache)
        self.wind_fine_box_m = 12.0            # half-extent of the per-object patch
        self.wind_fine_patch_enabled = True
        # Cap the global arrow count so the per-frame draw stays cheap on the dense
        # grid (the field is thinned in X/Y to fit). ~60k -> 10 m on F:\Output\run.
        self.wind_global_max_arrows = 60000

        # Shared wind-vector geometry cache. Built once per state change and
        # reused by every viewport (see get_wind_geometry).
        self._wind_geom_cache = None
        self._wind_geom_cache_key = None

        # Grid settings
        self.grid_size = 100  # Covers -50 to +50 meters with 1.0m spacing
        self.grid_spacing = 1.0
        self.grid_center = (0.0, 0.0)

        # Fixed mount height (m) for a flag pole's center. Matches the center
        # of the wind field's lowest 1m sampling layer (z=1..2m) so the pole
        # lands exactly where get_velocity_at_8_octants expects to sample.
        self.flag_pole_height = 1.5

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Selection
        self.selected_object: Optional[ObjectMesh] = None
        
        # Object ID counter
        self._next_id = 0
        self._object_ids: Dict[int, ObjectMesh] = {}

    def get_wind_geometry(self):
        """Return the wind-vector geometry for the current state, built once and shared.

        The result is cached against the wind/display state so every viewport
        reuses the same arrays (rebuilt only when that state changes).

        Returns:
            Resultant mode: (vertex_array, color_array_or_None).
            Components mode: list of three vertex arrays.
            None when there is nothing to draw.
        """
        wf = self.wind_field
        if wf is None:
            return None

        flat_stride = max(1, int(getattr(self, "wind_downsample_stride", 1)))
        scale = float(getattr(self, "wind_vector_scale", 0.3))
        mode = getattr(self, "wind_display_mode", "resultant")
        color_by_speed = bool(getattr(self, "wind_color_by_speed", True))

        # The dense (streaming) global field is thinned in X/Y to an arrow budget;
        # a full-resolution patch around each object adds local detail back.
        streaming = bool(getattr(wf, "is_streaming", False))
        global_stride = self._global_display_stride(wf) if streaming else 1
        fine_box = float(getattr(self, "wind_fine_box_m", 12.0))
        fine_enabled = (
            streaming and mode == "resultant"
            and bool(getattr(self, "wind_fine_patch_enabled", True))
        )
        centers = self._fine_patch_centers() if fine_enabled else []
        obj_key = tuple(
            (round(float(c[0]), 1), round(float(c[1]), 1), round(float(c[2]), 1))
            for c in centers
        )

        cache_key = (
            wf.current_time, flat_stride, global_stride, scale, mode, color_by_speed,
            streaming, fine_enabled, fine_box, obj_key,
            id(wf.data) if not streaming else id(getattr(wf, "_source", None)),
        )
        if cache_key != self._wind_geom_cache_key:
            # Fetch (and thus copy) velocities only on an actual rebuild, so plain
            # repaints (pan/zoom, sibling viewports) cost nothing here.
            if streaming:
                points, velocities = wf.coarse_display_grid(global_stride)
            else:
                points = wf.get_grid_points()
                velocities = wf.get_current_velocities()

            n = len(points)
            if n == 0 or len(velocities) != n:
                self._wind_geom_cache = None
                self._wind_geom_cache_key = cache_key
                return None

            geom = build_wind_geometry(
                points, velocities, flat_stride, scale, mode, color_by_speed
            )
            if fine_enabled and centers:
                geom = self._append_fine_patches(geom, centers, fine_box, scale,
                                                 color_by_speed)
            self._wind_geom_cache = geom
            self._wind_geom_cache_key = cache_key

        return self._wind_geom_cache

    def _global_display_stride(self, wf) -> int:
        """X/Y stride over the coarse grid so the global arrow count stays within
        ``wind_global_max_arrows`` (keeps the per-frame draw cheap on dense data)."""
        src = getattr(wf, "_source", None)
        if src is None:
            return 1
        Xc, Yc, Zc = len(src.xc), len(src.yc), len(src.zc)
        budget = max(1000, int(getattr(self, "wind_global_max_arrows", 60000)))
        s = 1
        while s < 64 and math.ceil(Yc / s) * math.ceil(Xc / s) * Zc > budget:
            s += 1
        return s

    def _fine_patch_centers(self) -> List[np.ndarray]:
        """World-space centers used to anchor full-resolution wind patches."""
        centers: List[np.ndarray] = []
        for obj in self.objects:
            try:
                if obj.name.lower() == "flag":
                    centers.append(np.asarray(obj.get_pole_center(), dtype=np.float32))
                else:
                    centers.append(np.asarray(obj.get_center(), dtype=np.float32))
            except Exception:
                continue
        return centers

    def _append_fine_patches(self, geom, centers, fine_box, scale, color_by_speed):
        """Concatenate full-res per-object arrow patches onto the global geometry."""
        wf = self.wind_field
        verts, colors = geom if isinstance(geom, tuple) else (geom, None)
        v_parts = [verts] if verts is not None and verts.size else []
        c_parts = [colors] if colors is not None and getattr(colors, "size", 0) else []

        for center in centers:
            box_points, box_vels = wf.get_box_grid(center, fine_box)
            if len(box_points) == 0:
                continue
            vb, cb = build_wind_geometry(
                box_points, box_vels, 1, scale, "resultant", color_by_speed
            )
            if vb is None or vb.size == 0:
                continue
            v_parts.append(vb)
            if color_by_speed and cb is not None and cb.size:
                c_parts.append(cb)

        if not v_parts:
            return (verts, colors)

        merged_v = np.concatenate(v_parts, axis=0) if len(v_parts) > 1 else v_parts[0]
        merged_c = None
        if color_by_speed and c_parts:
            total_c = sum(len(c) for c in c_parts)
            if total_c == len(merged_v):
                merged_c = (np.concatenate(c_parts, axis=0)
                            if len(c_parts) > 1 else c_parts[0])
        return (np.ascontiguousarray(merged_v, dtype=np.float32), merged_c)

    def _get_default_obj_path(self, object_type: str) -> Optional[str]:
        """Return the bundled OBJ path for a known object type, if present."""
        candidate = app_paths.resource_path("objects", f"{object_type.lower()}.obj")
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
            object_type: Type of object (flag)
            position: World position
            obj_path: Optional path to OBJ file
            
        Returns:
            The created ObjectMesh
        """
        # Create mesh first
        if obj_path is None:
            obj_path = self._get_default_obj_path(object_type)

        mesh = ObjectMesh(object_type, obj_path, position)

        if len(mesh.vertices) > 0:
            mesh_z_min = np.min(mesh.vertices[:, 2])
            mesh_z_max = np.max(mesh.vertices[:, 2])
            extra_offset = 0.0
            if object_type.lower() == 'flag':
                extra_offset = 1.0  # Increase this value to lift flag higher (1.0 = 1 meter)
            
            # Position so bottom sits at ground level + any extra offset
            mesh.position[2] = position[2] + (-mesh_z_min) + extra_offset
        
        
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

    def add_environment_mesh(self, mesh: ObjectMesh):
        """Add a static, non-interactive mesh (e.g. STL building geometry)."""
        self.environment_meshes.append(mesh)

    def clear_environment_meshes(self):
        """Remove all static environment meshes."""
        self.environment_meshes.clear()
    
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
        Get wind velocities around 8 cubes at an object's position.
        
        Args:
            mesh: The object mesh
            
        Returns:
            Wind velocity vector
        """
        if mesh.name.lower() == 'flag':
            center = mesh.get_pole_center()
        else:
            center = mesh.get_center()
        return self.wind_field.get_velocity_at_8_octants(center, cube_size=1.0)
    
    def snap_to_grid(self, position: np.ndarray) -> np.ndarray:
        """
        Snap a position to the center of the nearest 1x1 grid cell on the
        X/Y ground plane (e.g. x=0.5, 1.5, 2.5, ... not the grid line
        intersections at 0, 1, 2, ...), and pin height to the flag pole's
        fixed mount height.

        Args:
            position: World position

        Returns:
            Snapped position
        """
        center_x, center_y = self.grid_center
        snapped = position.copy()
        snapped[0] = (
            center_x +
            (np.floor((position[0] - center_x) / self.grid_spacing) + 0.5) * self.grid_spacing
        )
        snapped[1] = (
            center_y +
            (np.floor((position[1] - center_y) / self.grid_spacing) + 0.5) * self.grid_spacing
        )
        snapped[2] = self.flag_pole_height
        return snapped

    def get_grid_points(self) -> np.ndarray:
        """
        Get all grid point positions on the X/Y ground plane.

        Returns:
            Array of grid point positions
        """
        center_x, center_y = self.grid_center
        x = center_x + np.arange(-self.grid_size // 2, self.grid_size // 2 + 1) * self.grid_spacing
        y = center_y + np.arange(-self.grid_size // 2, self.grid_size // 2 + 1) * self.grid_spacing

        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)

        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return points.astype(np.float32)

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the bounding box of the scene.

        Returns:
            Tuple of (min_corner, max_corner)
        """
        center_x, center_y = self.grid_center
        half_size = self.grid_size // 2 * self.grid_spacing
        min_corner = np.array([center_x - half_size, center_y - half_size, 0])
        max_corner = np.array([center_x + half_size, center_y + half_size, 10])
        
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

    def compute_wind_vector_scale(self) -> float:
        """Auto-scale arrows so the strongest gust fits inside one grid cell.

        Picks a scale s such that max(|v|) * s ≈ 0.8 * min(dx, dy, dz),
        where dx/dy/dz are the median spacings of the wind grid axes.
        Falls back to a default when data is empty or constant.
        """
        wf = self.wind_field

        if getattr(wf, "is_streaming", False):
            # No dense array in RAM; scale to the coarse display cell using the
            # precomputed peak speed from the cache.
            max_mag = float(wf._source.max_speed())
            cell = float(getattr(self, "wind_coarse_spacing_m", 5.0))
            if not np.isfinite(max_mag) or max_mag < 1e-6:
                self.wind_vector_scale = 0.3
            else:
                self.wind_vector_scale = 0.8 * cell / max_mag
            return self.wind_vector_scale

        data = wf.data
        if data is None or data.size == 0:
            self.wind_vector_scale = 0.3
            return self.wind_vector_scale

        magnitudes = np.sqrt(np.sum(data * data, axis=0))
        max_mag = float(magnitudes.max())
        if not np.isfinite(max_mag) or max_mag < 1e-6:
            self.wind_vector_scale = 0.3
            return self.wind_vector_scale

        def _axis_step(coords: np.ndarray) -> float:
            if coords.size < 2:
                return 1.0
            diffs = np.diff(np.sort(coords.astype(np.float64)))
            diffs = diffs[diffs > 1e-9]
            return float(np.median(diffs)) if diffs.size else 1.0

        dx = _axis_step(wf.x_coords)
        dy = _axis_step(wf.y_coords)
        dz = _axis_step(wf.z_coords)
        cell = min(dx, dy, dz)

        self.wind_vector_scale = 0.8 * cell / max_mag
        return self.wind_vector_scale
    
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
