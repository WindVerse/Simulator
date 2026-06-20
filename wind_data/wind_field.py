"""
WindField Class
Handles 5D wind data (component, z, y, x, time).
Provides wind velocity vectors at any point in the field.

Coordinate convention: world frame is right-handed, +Z up. The horizontal
plane is X/Y, with Z=0 as the ground. Velocity components are (u_x, u_y, u_z)
in the same world frame; u_z is vertical wind.
"""

import numpy as np
from typing import Tuple, Optional

from .openfoam_loader import extract_openfoam_wind


class WindField:
    """
    Represents a time-varying wind field.

    The wind data is stored as a 5D numpy array with shape:
    (component, grid_z, grid_y, grid_x, time_steps)
    where the component axis is (u_x, u_y, u_z) in the world frame
    (Z is vertical / up).

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
        self._grid_points_cache: Optional[np.ndarray] = None

        # Streaming (large-case) backing store; None for eager in-RAM data.
        self._source = None
        self._coarse_points_cache: Optional[np.ndarray] = None
        self._coarse_disp_points: dict = {}  # xy_stride -> cached display points

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
        self._grid_points_cache = None

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
        self._source = None
        self._grid_points_cache = None
        self._coarse_points_cache = None

    @property
    def is_streaming(self) -> bool:
        """True when wind is served from a memmapped cache instead of self.data."""
        return self._source is not None

    def set_streaming_source(self, source):
        """Back this field with an OpenFOAMStreamingSource (large cached case).

        ``self.data`` is left as None; display sampling uses the source's in-RAM
        coarse field, while point/position sampling uses its full-res memmap. The
        coordinate arrays are the full-resolution grid so bounds, ML sampling and
        the ground grid all see the true domain.
        """
        self._source = source
        self.data = None
        self.x_coords = np.asarray(source.x_coords, dtype=np.float32)
        self.y_coords = np.asarray(source.y_coords, dtype=np.float32)
        self.z_coords = np.asarray(source.z_coords, dtype=np.float32)
        self.time_coords = np.asarray(source.time_coords, dtype=np.float32)
        self.grid_size = (len(self.x_coords), len(self.y_coords), len(self.z_coords))
        self.time_steps = int(source.time_steps)
        self.current_time = 0
        self._grid_points_cache = None
        self._coarse_points_cache = None
        self._coarse_disp_points = {}

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
        self._grid_points_cache = None

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

        x = int(np.clip(x, 0, self.grid_size[0] - 1))
        y = int(np.clip(y, 0, self.grid_size[1] - 1))
        z = int(np.clip(z, 0, self.grid_size[2] - 1))
        time = time % self.time_steps

        if self._source is not None:
            # Full-resolution read from the memmapped cache (accurate sampling).
            return self._source.velocity_at_grid(time, z, y, x)

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

    def get_velocity_at_8_octants(
        self,
        position: np.ndarray,
        cube_size: float = 1.0,
        time: Optional[int] = None
    ) -> np.ndarray:
        """
        Get wind velocity at 8 octant positions around a center point.
        Useful for spatial wind variation sampling.

        Args:
            position: Physical position (x, y, z) of the cube center
            cube_size: Size of each cube (default 1.0m). Offset is ±cube_size/2
            time: Time step index (uses current_time if None)

        Returns:
            Array of shape (8, 3) with velocities at each octant:
            [0: (−,−,−), 1: (−,−,+), 2: (−,+,−), 3: (−,+,+),
            4: (+,−,−), 5: (+,−,+), 6: (+,+,−), 7: (+,+,+)]
        """
        if time is None:
            time = self.current_time

        offset = cube_size / 2.0
        
        # Octant offsets matching cube_index = ix*4 + iy*2 + iz
        # where ix,iy,iz ∈ {0,1}: 0→-offset, 1→+offset
        octant_offsets = [
            [-offset, -offset, -offset],  # 0: (0,0,0)
            [-offset, -offset, +offset],  # 1: (0,0,1)
            [-offset, +offset, -offset],  # 2: (0,1,0)
            [-offset, +offset, +offset],  # 3: (0,1,1)
            [+offset, -offset, -offset],  # 4: (1,0,0)
            [+offset, -offset, +offset],  # 5: (1,0,1)
            [+offset, +offset, -offset],  # 6: (1,1,0)
            [+offset, +offset, +offset],  # 7: (1,1,1)
        ]
        
        velocities = np.zeros((8, 3), dtype=np.float32)
        for i, offset_vec in enumerate(octant_offsets):
            sample_pos = position + np.array(offset_vec, dtype=np.float32)
            velocities[i] = self.get_velocity_at_position(sample_pos, time)
        
        return velocities

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

        Cached: rebuilt only when the underlying coordinate arrays change
        (set_wind_data / load_from_file / _set_default_coords reset the cache).
        Callers must treat the returned array as read-only.

        In streaming mode this returns the *coarse* display grid (the global
        vector field is thinned for performance); full-resolution sampling still
        goes through get_velocity_at_grid / get_box_grid.
        """
        if self._source is not None:
            if self._coarse_points_cache is None:
                X, Y, Z = np.meshgrid(
                    self._source.xc, self._source.yc, self._source.zc, indexing='ij'
                )
                points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
                self._coarse_points_cache = np.ascontiguousarray(points, dtype=np.float32)
            return self._coarse_points_cache

        if self._grid_points_cache is None:
            X, Y, Z = np.meshgrid(
                self.x_coords, self.y_coords, self.z_coords, indexing='ij'
            )
            points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
            self._grid_points_cache = np.ascontiguousarray(points, dtype=np.float32)
        return self._grid_points_cache

    def get_current_velocities(self) -> np.ndarray:
        """
        Get all velocity vectors at the current time step (matches get_grid_points).

        Returns:
            Array of shape (N, 3) with velocity vectors. In streaming mode these
            are the coarse-field velocities aligned with the coarse grid points.
        """
        if self._source is not None:
            return self._source.coarse_velocities(self.current_time)

        current = self.data[:, :, :, :, self.current_time]
        velocities = np.transpose(current, (3, 2, 1, 0)).reshape(-1, 3)
        return velocities.copy()

    def coarse_display_grid(self, xy_stride: int) -> Tuple[np.ndarray, np.ndarray]:
        """Global display tier (streaming): coarse field thinned by ``xy_stride``.

        The 5 m coarse cache is far denser than a wide view can resolve, so the
        global vector field is thinned in X and Y (Z kept full) to bound the arrow
        count and keep playback smooth. Returns (points (M,3), velocities (M,3))
        in get_grid_points order. Points are cached per stride; velocities follow
        ``current_time``. Falls back to the eager arrays when not streaming.
        """
        if self._source is None:
            return self.get_grid_points(), self.get_current_velocities()

        s = max(1, int(xy_stride))
        src = self._source
        points = self._coarse_disp_points.get(s)
        if points is None:
            xs, ys, zs = src.xc[::s], src.yc[::s], src.zc
            X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
            points = np.ascontiguousarray(
                np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1), dtype=np.float32
            )
            self._coarse_disp_points[s] = points

        cur = src.coarse[self.current_time % self.time_steps][:, :, ::s, ::s]
        vels = np.ascontiguousarray(
            np.transpose(cur, (3, 2, 1, 0)).reshape(-1, 3), dtype=np.float32
        )
        return points, vels

    def get_box_grid(
        self, center: np.ndarray, half_extent_m: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Full-resolution wind in a box around ``center`` at the current time.

        Used to draw a dense "fine patch" of vectors around placed objects on top
        of the coarse global field. Returns (points (M,3), velocities (M,3)) in the
        same ordering as get_grid_points. Empty arrays when not streaming or when
        the box falls outside the domain.
        """
        empty = (np.empty((0, 3), np.float32), np.empty((0, 3), np.float32))
        if self._source is None:
            return empty

        cx, cy = float(center[0]), float(center[1])
        x0 = int(np.searchsorted(self.x_coords, cx - half_extent_m, side="left"))
        x1 = int(np.searchsorted(self.x_coords, cx + half_extent_m, side="right"))
        y0 = int(np.searchsorted(self.y_coords, cy - half_extent_m, side="left"))
        y1 = int(np.searchsorted(self.y_coords, cy + half_extent_m, side="right"))
        x0 = max(0, min(x0, self.grid_size[0]))
        x1 = max(0, min(x1, self.grid_size[0]))
        y0 = max(0, min(y0, self.grid_size[1]))
        y1 = max(0, min(y1, self.grid_size[1]))
        if x1 <= x0 or y1 <= y0:
            return empty

        box = self._source.box_at(self.current_time, x0, x1, y0, y1)  # (3,Z,yb,xb)
        xs = self.x_coords[x0:x1]
        ys = self.y_coords[y0:y1]
        zs = self.z_coords
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        vels = np.transpose(box, (3, 2, 1, 0)).reshape(-1, 3)
        return (
            np.ascontiguousarray(points, dtype=np.float32),
            np.ascontiguousarray(vels, dtype=np.float32),
        )

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
