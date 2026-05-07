"""
ObjectMesh Class
Handles mesh representation, loading from OBJ files, and vertex manipulation.
"""

import numpy as np
from typing import Optional, Tuple, List
import os


class ObjectMesh:
    """
    Represents a 3D mesh object that can be deformed by wind.
    
    Meshes are loaded from OBJ files and consist of:
    - Vertices: 3D positions that can be updated
    - Faces: Triangular indices (topology is fixed)
    - Normals: Surface normals for rendering
    
    Attributes:
        name: Name/type of the object (tree, flag, cloth, pole)
        vertices: Original vertex positions (N, 3)
        current_vertices: Current deformed vertex positions
        faces: Triangle indices (M, 3)
        normals: Vertex normals (N, 3)
        position: World position of the object
        previous_vertices: Previous frame vertices for model input
    """
    
    def __init__(
        self,
        name: str,
        obj_path: Optional[str] = None,
        position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        rest_lengths = None
    ):
        """
        Initialize the mesh object.
        
        Args:
            name: Object type name (tree, flag, cloth, pole)
            obj_path: Path to OBJ file (generates default if None)
            position: Initial world position
        """
        self.name = name
        self.position = np.array(position, dtype=np.float32)
        self.scale = 1.0
        
        # Mesh data
        self.vertices: np.ndarray = np.array([], dtype=np.float32)
        self.faces: np.ndarray = np.array([], dtype=np.int32)
        self.normals: np.ndarray = np.array([], dtype=np.float32)
        self.texture_coords: np.ndarray = np.array([], dtype=np.float32)
        
        # Current state
        self.current_vertices: np.ndarray = np.array([], dtype=np.float32)
        self.previous_vertices: np.ndarray = np.array([], dtype=np.float32)
        
        # Color for rendering
        self.color = self._get_default_color()
        
        if obj_path and os.path.exists(obj_path):
            self.load_from_obj(obj_path)
        else:
            self._generate_default_mesh()
    
    def _get_default_color(self) -> np.ndarray:
        """Get default color based on object type."""
        colors = {
            'tree': np.array([0.2, 0.6, 0.2, 1.0]),
            'flag': np.array([0.8, 0.2, 0.2, 1.0]),
            'cloth': np.array([0.9, 0.9, 0.9, 1.0]),
            'pole': np.array([0.5, 0.5, 0.5, 1.0]),
        }
        return colors.get(self.name.lower(), np.array([0.7, 0.7, 0.7, 1.0]))
    
    def _generate_default_mesh(self):
        """Generate a default mesh based on object type."""
        generators = {
            'tree': self._generate_tree,
            'flag': self._generate_flag,
            'cloth': self._generate_cloth,
            'pole': self._generate_pole,
        }
        
        generator = generators.get(self.name.lower(), self._generate_cube)
        generator()
        
        # Initialize current and previous vertices
        self.current_vertices = self.vertices.copy()
        self.previous_vertices = self.vertices.copy()
        
        # Compute initial normals
        self._compute_normals()
    
    def _generate_tree(self):
        """Generate a simple tree mesh (trunk + foliage)."""
        vertices = []
        faces = []
        
        # Trunk (cylinder approximation)
        trunk_height = 2.0
        trunk_radius = 0.2
        segments = 8
        
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = trunk_radius * np.cos(angle)
            z = trunk_radius * np.sin(angle)
            vertices.append([x, 0, z])
            vertices.append([x, trunk_height, z])
        
        # Trunk faces
        for i in range(segments):
            i1 = i * 2
            i2 = i * 2 + 1
            i3 = ((i + 1) % segments) * 2
            i4 = ((i + 1) % segments) * 2 + 1
            faces.append([i1, i3, i2])
            faces.append([i2, i3, i4])
        
        trunk_vertex_count = len(vertices)
        
        # Foliage (cone approximation)
        foliage_height = 3.0
        foliage_radius = 1.5
        foliage_base = trunk_height
        
        # Foliage base vertices
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = foliage_radius * np.cos(angle)
            z = foliage_radius * np.sin(angle)
            vertices.append([x, foliage_base, z])
        
        # Foliage tip
        tip_index = len(vertices)
        vertices.append([0, foliage_base + foliage_height, 0])
        
        # Foliage faces
        for i in range(segments):
            i1 = trunk_vertex_count + i
            i2 = trunk_vertex_count + ((i + 1) % segments)
            faces.append([i1, i2, tip_index])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
    
    def _generate_flag(self):
        """Generate a flag mesh (rectangular cloth)."""
        width = 2.0
        height = 1.5
        subdivisions_x = 10
        subdivisions_y = 8
        
        vertices = []
        faces = []
        
        for j in range(subdivisions_y + 1):
            for i in range(subdivisions_x + 1):
                x = (i / subdivisions_x) * width
                y = (j / subdivisions_y) * height + 1.5  # Offset up for pole
                z = 0
                vertices.append([x, y, z])
        
        # Generate faces
        for j in range(subdivisions_y):
            for i in range(subdivisions_x):
                i1 = j * (subdivisions_x + 1) + i
                i2 = i1 + 1
                i3 = i1 + subdivisions_x + 1
                i4 = i3 + 1
                faces.append([i1, i3, i2])
                faces.append([i2, i3, i4])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
    
    def _generate_cloth(self):
        """Generate a cloth mesh (square grid)."""
        size = 3.0
        subdivisions = 15
        
        vertices = []
        faces = []
        
        for j in range(subdivisions + 1):
            for i in range(subdivisions + 1):
                x = (i / subdivisions - 0.5) * size
                y = 3.0  # Height above ground
                z = (j / subdivisions - 0.5) * size
                vertices.append([x, y, z])
        
        # Generate faces
        for j in range(subdivisions):
            for i in range(subdivisions):
                i1 = j * (subdivisions + 1) + i
                i2 = i1 + 1
                i3 = i1 + subdivisions + 1
                i4 = i3 + 1
                faces.append([i1, i3, i2])
                faces.append([i2, i3, i4])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
    
    def _generate_pole(self):
        """Generate a pole mesh (tall cylinder)."""
        height = 4.0
        radius = 0.1
        segments = 8
        
        vertices = []
        faces = []
        
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            vertices.append([x, 0, z])
            vertices.append([x, height, z])
        
        # Side faces
        for i in range(segments):
            i1 = i * 2
            i2 = i * 2 + 1
            i3 = ((i + 1) % segments) * 2
            i4 = ((i + 1) % segments) * 2 + 1
            faces.append([i1, i3, i2])
            faces.append([i2, i3, i4])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)
    
    def _generate_cube(self):
        """Generate a simple cube mesh as fallback."""
        self.vertices = np.array([
            [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5], [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5], [-0.5, 0.5, 0.5],
        ], dtype=np.float32)
        
        self.faces = np.array([
            [0, 1, 2], [0, 2, 3],  # Back
            [4, 6, 5], [4, 7, 6],  # Front
            [0, 4, 5], [0, 5, 1],  # Bottom
            [2, 6, 7], [2, 7, 3],  # Top
            [0, 3, 7], [0, 7, 4],  # Left
            [1, 5, 6], [1, 6, 2],  # Right
        ], dtype=np.int32)
    
    def load_from_obj(self, filepath: str):
        """
        Load mesh from an OBJ file.
        
        Args:
            filepath: Path to the OBJ file
        """
        vertices = []
        faces = []
        normals = []
        texture_coords = []
        
        with open(filepath, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                
                if parts[0] == 'v':
                    # Vertex position
                    vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vn':
                    # Vertex normal
                    normals.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif parts[0] == 'vt':
                    # Texture coordinate
                    texture_coords.append([float(parts[1]), float(parts[2])])
                elif parts[0] == 'f':
                    # Face (handles v, v/vt, v/vt/vn, v//vn formats)
                    face_vertices = []
                    for part in parts[1:]:
                        indices = part.split('/')
                        vertex_index = int(indices[0]) - 1  # OBJ is 1-indexed
                        face_vertices.append(vertex_index)
                    
                    # Triangulate if needed (simple fan triangulation)
                    for i in range(1, len(face_vertices) - 1):
                        faces.append([
                            face_vertices[0],
                            face_vertices[i],
                            face_vertices[i + 1]
                        ])
        
        self.vertices = np.array(vertices, dtype=np.float32)
        self.faces = np.array(faces, dtype=np.int32)

        # Initialize current and previous vertices before normal computation.
        self.current_vertices = self.vertices.copy()
        self.previous_vertices = self.vertices.copy()
        
        if normals:
            self.normals = np.array(normals, dtype=np.float32)
        else:
            self._compute_normals()
        
        if texture_coords:
            self.texture_coords = np.array(texture_coords, dtype=np.float32)
    
    def save_to_obj(self, filepath: str):
        """
        Save mesh to an OBJ file.
        
        Args:
            filepath: Path to save the OBJ file
        """
        with open(filepath, 'w') as f:
            f.write(f"# {self.name} mesh\n")
            
            # Write vertices
            for v in self.current_vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            # Write normals
            for n in self.normals:
                f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            
            # Write faces (1-indexed)
            for face in self.faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    def _compute_normals(self):
        """Compute vertex normals from face normals."""
        if len(self.vertices) == 0 or len(self.faces) == 0:
            return
        
        # Initialize vertex normals to zero
        self.normals = np.zeros_like(self.current_vertices)
        
        # Compute face normals and accumulate to vertices
        for face in self.faces:
            v0 = self.current_vertices[face[0]]
            v1 = self.current_vertices[face[1]]
            v2 = self.current_vertices[face[2]]
            
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = np.cross(edge1, edge2)
            
            # Add face normal to each vertex
            self.normals[face[0]] += face_normal
            self.normals[face[1]] += face_normal
            self.normals[face[2]] += face_normal
        
        # Normalize
        norms = np.linalg.norm(self.normals, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        self.normals /= norms
    
    def update_vertices(self, new_vertices: np.ndarray):
        """
        Update vertex positions.
        
        Args:
            new_vertices: New vertex positions (N, 3)
        """
        # Store previous state
        self.previous_vertices = self.current_vertices.copy()
        
        # Update current vertices
        self.current_vertices = new_vertices.astype(np.float32)
        
        # Recompute normals
        self._compute_normals()
    
    def apply_displacement(self, displacement: np.ndarray):
        """
        Apply displacement to current vertices.
        
        Args:
            displacement: Displacement vectors (N, 3)
        """
        new_vertices = self.current_vertices + displacement
        self.update_vertices(new_vertices)
    
    def reset_to_original(self):
        """Reset vertices to original positions."""
        self.current_vertices = self.vertices.copy()
        self.previous_vertices = self.vertices.copy()
        self._compute_normals()
    
    def get_world_vertices(self) -> np.ndarray:
        """
        Get vertex positions in world space.
        
        Returns:
            World space vertex positions (N, 3)
        """
        return self.current_vertices * self.scale + self.position
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the axis-aligned bounding box.
        
        Returns:
            Tuple of (min_corner, max_corner)
        """
        world_verts = self.get_world_vertices()
        min_corner = world_verts.min(axis=0)
        max_corner = world_verts.max(axis=0)
        return min_corner, max_corner
    
    def get_center(self) -> np.ndarray:
        """
        Get the center of the mesh in world space.
        
        Returns:
            Center position (3,)
        """
        world_verts = self.get_world_vertices()
        return world_verts.mean(axis=0)

    def get_pole_center(self) -> np.ndarray:
        """
        Compute and expose the pole center coordinates.
        For flags, compute pole center from the flag's pole-attachment side (x-min side).
        For other objects, return the full mesh center.
        
        Returns:
            Pole center position (3,)
        """
        world_verts = self.get_world_vertices()
        if len(world_verts) == 0:
            return self.position.copy()
            
        if self.name.lower() == 'flag':
            # Extract x-min side (pole attachment side)
            min_x = world_verts[:, 0].min()
            # Tolerance for finding vertices on the left edge
            mask = np.abs(world_verts[:, 0] - min_x) < 0.1
            left_edge = world_verts[mask]
            
            if len(left_edge) > 0:
                return left_edge.mean(axis=0)
                
        # Default behavior: use the object's centroid
        return self.get_center()

    def get_info_payload(self) -> dict:
        """
        Return structured payload with object parameters (for ML parsing).
        Includes object_type, object_position, and pole_center.
        """
        pole_center = self.get_pole_center()
        
        # Format as easily serializable tuples
        return {
            "object_type": self.name,
            "object_position": tuple(float(v) for v in self.position),
            "pole_center": tuple(float(v) for v in pole_center)
        }
    
    def get_vertex_count(self) -> int:
        """Return the number of vertices."""
        return len(self.vertices)
    
    def get_face_count(self) -> int:
        """Return the number of faces."""
        return len(self.faces)
    
    def clone(self) -> 'ObjectMesh':
        """
        Create a copy of this mesh.
        
        Returns:
            New ObjectMesh instance with copied data
        """
        new_mesh = ObjectMesh(self.name)
        new_mesh.vertices = self.vertices.copy()
        new_mesh.faces = self.faces.copy()
        new_mesh.normals = self.normals.copy()
        new_mesh.current_vertices = self.current_vertices.copy()
        new_mesh.previous_vertices = self.previous_vertices.copy()
        new_mesh.position = self.position.copy()
        new_mesh.scale = self.scale
        new_mesh.color = self.color.copy()
        return new_mesh
