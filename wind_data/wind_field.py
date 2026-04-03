"""
WindField Class
Handles 4D wind data (time, x, y, z, velocity_components).
Provides wind velocity vectors at any point in the field.
"""

import numpy as np
from typing import Tuple, Optional


class WindField:
    """
    Represents a 4D wind field with time-varying velocity data.
    
    The wind data is stored as a 5D numpy array with shape:
    (time_steps, grid_x, grid_y, grid_z, 3)
    where the last dimension contains (u, v, w) velocity components.
    
    Attributes:
        data: The wind velocity data array
        grid_size: Tuple of (x, y, z) grid dimensions
        time_steps: Number of time steps in the data
        current_time: Current time index in the simulation
        grid_spacing: Physical spacing between grid points
        origin: Physical position of grid origin
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int, int] = (20, 20, 10),
        time_steps: int = 100,
        grid_spacing: float = 1.0,
        origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ):
        """
        Initialize the wind field.
        
        Args:
            grid_size: Number of grid points in (x, y, z) dimensions
            time_steps: Number of time steps
            grid_spacing: Physical distance between grid points
            origin: Physical position of the grid origin (0, 0, 0)
        """
        self.grid_size = grid_size
        self.time_steps = time_steps
        self.grid_spacing = grid_spacing
        self.origin = np.array(origin)
        self.current_time = 0
        
        # Initialize empty wind data array
        # Shape: (time, x, y, z, 3) for velocity components (u, v, w)
        self.data = np.zeros((time_steps, *grid_size, 3), dtype=np.float32)
        
        # Generate default wind pattern
        self._generate_default_wind()
    
    def _generate_default_wind(self):
        """
        Generate a default wind pattern for demonstration.
        Creates a time-varying wind field with turbulence.
        """
        t = np.linspace(0, 2 * np.pi, self.time_steps)
        x = np.arange(self.grid_size[0])
        y = np.arange(self.grid_size[1])
        z = np.arange(self.grid_size[2])
        
        # Create meshgrid for spatial coordinates
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        for ti in range(self.time_steps):
            # Base wind in x direction with time variation
            base_wind_x = 2.0 + 0.5 * np.sin(t[ti])
            
            # Add spatial variation and turbulence
            self.data[ti, :, :, :, 0] = base_wind_x + 0.3 * np.sin(
                X * 0.3 + t[ti]
            ) * np.cos(Y * 0.2)
            
            # Y component (cross-wind)
            self.data[ti, :, :, :, 1] = 0.5 * np.sin(
                Y * 0.4 + t[ti] * 0.7
            ) * np.cos(X * 0.3)
            
            # Z component (vertical, decreasing with height)
            height_factor = 1.0 - Z / self.grid_size[2]
            self.data[ti, :, :, :, 2] = 0.2 * np.sin(
                t[ti] * 1.5
            ) * height_factor
    
    def load_from_file(self, filepath: str):
        """
        Load wind data from a numpy file.
        
        Args:
            filepath: Path to the .npy or .npz file
        """
        loaded_data = np.load(filepath)
        if isinstance(loaded_data, np.lib.npyio.NpzFile):
            self.data = loaded_data['wind_data']
        else:
            self.data = loaded_data
        
        # Update dimensions based on loaded data
        self.time_steps = self.data.shape[0]
        self.grid_size = self.data.shape[1:4]
    
    def save_to_file(self, filepath: str):
        """
        Save wind data to a numpy file.
        
        Args:
            filepath: Path to save the file
        """
        np.savez_compressed(filepath, wind_data=self.data)
    
    def get_velocity_at_grid(
        self,
        x: int,
        y: int,
        z: int,
        time: Optional[int] = None
    ) -> np.ndarray:
        """
        Get wind velocity at a specific grid point.
        
        Args:
            x, y, z: Grid indices
            time: Time step index (uses current_time if None)
            
        Returns:
            Velocity vector (u, v, w)
        """
        if time is None:
            time = self.current_time
        
        # Clamp indices to valid range
        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)
        z = np.clip(z, 0, self.grid_size[2] - 1)
        time = time % self.time_steps
        
        return self.data[time, x, y, z].copy()
    
    def get_velocity_at_position(
        self,
        position: np.ndarray,
        time: Optional[int] = None
    ) -> np.ndarray:
        """
        Get interpolated wind velocity at a physical position.
        Uses trilinear interpolation.
        
        Args:
            position: Physical position (x, y, z)
            time: Time step index (uses current_time if None)
            
        Returns:
            Interpolated velocity vector (u, v, w)
        """
        if time is None:
            time = self.current_time
        
        # Convert physical position to grid coordinates
        grid_pos = (position - self.origin) / self.grid_spacing
        
        # Get integer indices and fractional parts
        ix = int(np.floor(grid_pos[0]))
        iy = int(np.floor(grid_pos[1]))
        iz = int(np.floor(grid_pos[2]))
        
        fx = grid_pos[0] - ix
        fy = grid_pos[1] - iy
        fz = grid_pos[2] - iz
        
        # Trilinear interpolation
        result = np.zeros(3, dtype=np.float32)
        
        for dx in [0, 1]:
            for dy in [0, 1]:
                for dz in [0, 1]:
                    weight = (
                        (1 - fx if dx == 0 else fx) *
                        (1 - fy if dy == 0 else fy) *
                        (1 - fz if dz == 0 else fz)
                    )
                    v = self.get_velocity_at_grid(
                        ix + dx, iy + dy, iz + dz, time
                    )
                    result += weight * v
        
        return result
    
    def advance_time(self, delta: int = 1):
        """
        Advance the current time index.
        
        Args:
            delta: Number of time steps to advance
        """
        self.current_time = (self.current_time + delta) % self.time_steps
    
    def reset_time(self):
        """Reset the current time to 0."""
        self.current_time = 0
    
    def get_grid_points(self) -> np.ndarray:
        """
        Get all grid point positions in physical space.
        
        Returns:
            Array of shape (N, 3) with grid point positions
        """
        x = np.arange(self.grid_size[0]) * self.grid_spacing + self.origin[0]
        y = np.arange(self.grid_size[1]) * self.grid_spacing + self.origin[1]
        z = np.arange(self.grid_size[2]) * self.grid_spacing + self.origin[2]
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        
        return points.astype(np.float32)
    
    def get_current_velocities(self) -> np.ndarray:
        """
        Get all velocity vectors at the current time step.
        
        Returns:
            Array of shape (N, 3) with velocity vectors
        """
        velocities = self.data[self.current_time].reshape(-1, 3)
        return velocities.copy()
    
    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the physical bounds of the wind field.
        
        Returns:
            Tuple of (min_corner, max_corner)
        """
        min_corner = self.origin.copy()
        max_corner = self.origin + np.array(self.grid_size) * self.grid_spacing
        return min_corner, max_corner
    
    def set_uniform_wind(self, velocity: Tuple[float, float, float]):
        """
        Set a uniform wind velocity throughout the field.
        
        Args:
            velocity: Uniform velocity vector (u, v, w)
        """
        self.data[:, :, :, :, :] = np.array(velocity)
    
    def add_turbulence(self, intensity: float = 0.5):
        """
        Add random turbulence to the wind field.
        
        Args:
            intensity: Turbulence intensity multiplier
        """
        noise = np.random.randn(*self.data.shape).astype(np.float32)
        self.data += noise * intensity
