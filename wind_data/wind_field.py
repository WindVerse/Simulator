"""
WindField Class
Handles 5D wind data (component, z, y, x, time).
Provides wind velocity vectors at any point in the field.
"""

import numpy as np
from typing import Tuple, Optional

from .openfoam_loader import extract_openfoam_wind


class WindField:
    """
    Represents a time-varying wind field.

    The wind data is stored as a 5D numpy array with shape:
    (component, grid_z, grid_y, grid_x, time_steps)
    where the component axis is (u, v, w).

    Attributes:
        data: The wind velocity data array
        grid_size: Tuple of (x, y, z) grid dimensions
        time_steps: Number of time steps in the data
        current_time: Current time index in the simulation
        grid_spacing: Physical spacing between grid points (uniform default)
        origin: Physical position of grid origin
        x_coords: Physical x coordinates
        y_coords: Physical y coordinates
        z_coords: Physical z coordinates
        time_coords: Time coordinate values
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
        self.origin = np.array(origin, dtype=np.float32)
        self.current_time = 0

        self._set_default_coords()

        # Shape: (component, z, y, x, time)
        self.data = np.zeros(
            (3, grid_size[2], grid_size[1], grid_size[0], time_steps),
            dtype=np.float32
        )

        # Generate default wind pattern
        self._generate_default_wind()

    def _set_default_coords(self):
        self.x_coords = (
            self.origin[0] + np.arange(self.grid_size[0]) * self.grid_spacing
        ).astype(np.float32)
        self.y_coords = (
            self.origin[1] + np.arange(self.grid_size[1]) * self.grid_spacing
        ).astype(np.float32)
        self.z_coords = (
            self.origin[2] + np.arange(self.grid_size[2]) * self.grid_spacing
        ).astype(np.float32)
        self.time_coords = np.arange(self.time_steps, dtype=np.float32)

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
            base_wind_x = 2.0 + 0.5 * np.sin(t[ti])

            u = base_wind_x + 0.3 * np.sin(X * 0.3 + t[ti]) * np.cos(Y * 0.2)
            v = 0.5 * np.sin(Y * 0.4 + t[ti] * 0.7) * np.cos(X * 0.3)
            height_factor = 1.0 - Z / max(self.grid_size[2], 1)
            w = 0.2 * np.sin(t[ti] * 1.5) * height_factor

            self.data[0, :, :, :, ti] = np.transpose(u, (2, 1, 0))
            self.data[1, :, :, :, ti] = np.transpose(v, (2, 1, 0))
            self.data[2, :, :, :, ti] = np.transpose(w, (2, 1, 0))

    def set_wind_data(
        self,
        wind_data: np.ndarray,
        x_coords: np.ndarray,
        y_coords: np.ndarray,
        z_coords: np.ndarray,
        time_coords: np.ndarray
    ):
        """
        Replace wind data and coordinate metadata.
        """
        self.data = wind_data.astype(np.float32, copy=False)
        self.x_coords = np.array(x_coords, dtype=np.float32)
        self.y_coords = np.array(y_coords, dtype=np.float32)
        self.z_coords = np.array(z_coords, dtype=np.float32)
        self.time_coords = np.array(time_coords, dtype=np.float32)

        self.grid_size = (
            len(self.x_coords),
            len(self.y_coords),
            len(self.z_coords)
        )
        self.time_steps = len(self.time_coords)
        self.current_time = 0

    def load_from_openfoam_folder(self, base_dir: str):
        """
        Load wind data from an OpenFOAM postProcessing/surfaces folder.
        """
        wind_data, x_coords, y_coords, z_coords, time_coords = extract_openfoam_wind(base_dir)
        self.set_wind_data(wind_data, x_coords, y_coords, z_coords, time_coords)

    def load_from_file(self, filepath: str):
        """
        Load wind data from a numpy file.

        Args:
            filepath: Path to the .npy or .npz file
        """
        loaded_data = np.load(filepath, allow_pickle=True)
        if isinstance(loaded_data, np.lib.npyio.NpzFile):
            wind_data = loaded_data['wind_data']
            x_coords = loaded_data['x_coords'] if 'x_coords' in loaded_data else None
            y_coords = loaded_data['y_coords'] if 'y_coords' in loaded_data else None
            z_coords = loaded_data['z_coords'] if 'z_coords' in loaded_data else None
            time_coords = loaded_data['time_coords'] if 'time_coords' in loaded_data else None
        else:
            wind_data = loaded_data
            x_coords = None
            y_coords = None
            z_coords = None
            time_coords = None

        if wind_data.ndim != 5:
            raise ValueError("Wind data must be a 5D array.")

        if wind_data.shape[0] == 3:
            data = wind_data
        elif wind_data.shape[-1] == 3:
            # Legacy layout: (time, x, y, z, 3)
            data = np.transpose(wind_data, (4, 3, 2, 1, 0))
        else:
            raise ValueError("Unrecognized wind data layout.")

        self.data = data.astype(np.float32, copy=False)
        self.time_steps = self.data.shape[4]
        self.grid_size = (self.data.shape[3], self.data.shape[2], self.data.shape[1])

        if x_coords is not None and y_coords is not None and z_coords is not None:
            self.x_coords = np.array(x_coords, dtype=np.float32)
            self.y_coords = np.array(y_coords, dtype=np.float32)
            self.z_coords = np.array(z_coords, dtype=np.float32)
        else:
            self._set_default_coords()

        if time_coords is not None:
            self.time_coords = np.array(time_coords, dtype=np.float32)
        else:
            self.time_coords = np.arange(self.time_steps, dtype=np.float32)

        self.current_time = 0

    def save_to_file(self, filepath: str):
        """
        Save wind data to a numpy file.

        Args:
            filepath: Path to save the file
        """
        np.savez_compressed(
            filepath,
            wind_data=self.data,
            x_coords=self.x_coords,
            y_coords=self.y_coords,
            z_coords=self.z_coords,
            time_coords=self.time_coords
        )

    @staticmethod
    def _axis_index_and_frac(coords: np.ndarray, value: float) -> Tuple[int, float]:
        if len(coords) < 2:
            return 0, 0.0

        if value <= coords[0]:
            return 0, 0.0
        if value >= coords[-1]:
            return len(coords) - 2, 1.0

        idx = int(np.searchsorted(coords, value) - 1)
        span = coords[idx + 1] - coords[idx]
        if span == 0:
            return idx, 0.0
        return idx, float((value - coords[idx]) / span)

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

        x = np.clip(x, 0, self.grid_size[0] - 1)
        y = np.clip(y, 0, self.grid_size[1] - 1)
        z = np.clip(z, 0, self.grid_size[2] - 1)
        time = time % self.time_steps

        return self.data[:, z, y, x, time].copy()

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

        ix, fx = self._axis_index_and_frac(self.x_coords, position[0])
        iy, fy = self._axis_index_and_frac(self.y_coords, position[1])
        iz, fz = self._axis_index_and_frac(self.z_coords, position[2])

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
        X, Y, Z = np.meshgrid(self.x_coords, self.y_coords, self.z_coords, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        return points.astype(np.float32)

    def get_current_velocities(self) -> np.ndarray:
        """
        Get all velocity vectors at the current time step.

        Returns:
            Array of shape (N, 3) with velocity vectors
        """
        current = self.data[:, :, :, :, self.current_time]
        velocities = np.transpose(current, (3, 2, 1, 0)).reshape(-1, 3)
        return velocities.copy()

    def get_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the physical bounds of the wind field.

        Returns:
            Tuple of (min_corner, max_corner)
        """
        min_corner = np.array(
            [self.x_coords[0], self.y_coords[0], self.z_coords[0]],
            dtype=np.float32
        )
        max_corner = np.array(
            [self.x_coords[-1], self.y_coords[-1], self.z_coords[-1]],
            dtype=np.float32
        )
        return min_corner, max_corner

    def set_uniform_wind(self, velocity: Tuple[float, float, float]):
        """
        Set a uniform wind velocity throughout the field.

        Args:
            velocity: Uniform velocity vector (u, v, w)
        """
        self.data[0, :, :, :, :] = velocity[0]
        self.data[1, :, :, :, :] = velocity[1]
        self.data[2, :, :, :, :] = velocity[2]

    def add_turbulence(self, intensity: float = 0.5):
        """
        Add random turbulence to the wind field.

        Args:
            intensity: Turbulence intensity multiplier
        """
        noise = np.random.randn(*self.data.shape).astype(np.float32)
        self.data += noise * intensity
