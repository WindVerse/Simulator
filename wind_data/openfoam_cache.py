"""
OpenFOAM dense-case binary cache + streaming source.

Large OpenFOAM surface exports (e.g. ``F:\\Output\\run``: a uniform 1 m grid,
1201x901, 5 z-levels, 1001 timesteps) are far too big to hold densely in RAM
(~65 GB float32) and far too slow to re-parse from ASCII every frame (~195 GB of
text). This module converts such a case **once** into a compact binary cache on a
roomy disk, then serves it at runtime with O(1) per-frame access:

* ``field_f16.dat`` - a memmapped ``(T, 3, Z, Y, X)`` float16 array (~32.5 GB for
  the run above). One timestep ``mm[t]`` is a contiguous block, so a full slice or
  a small spatial box is a cheap read. Used for ML/object sampling and the
  full-resolution "fine patch" around placed objects.
* ``coarse_f32.npy`` - a spatially subsampled ``(3, Zc, Yc, Xc, T)`` float32 field
  (~5 m spacing) held fully in RAM for instant per-frame global vector display.
* ``meta.npz`` / ``manifest.json`` - coordinates, shapes, and a source fingerprint.
  ``manifest.json`` is written last and doubles as a DONE marker, so a partial or
  interrupted build reads as invalid.

The grid layout (row order within each ``U_zNormal_*.raw``) is identical across
all timesteps, so the per-z scatter indices and the per-z nearest-neighbour
hole-fill map are computed once from timestep 0 and reused for every step.

Small cases stay on the eager path in :mod:`wind_data.openfoam_loader`; this
module is only engaged above ``STREAMING_THRESHOLD_GB``.
"""

from typing import Callable, Dict, List, Optional, Tuple
import hashlib
import json
import os

import numpy as np
from scipy import ndimage

import app_paths


# Bump when the on-disk format changes so stale caches are rebuilt.
CACHE_VERSION = 1

# Dense float32 size above which a case is cached/streamed instead of loaded eager.
STREAMING_THRESHOLD_GB = 8.0

# Default coarse display spacing (metres). Drives the in-RAM global vector field.
DEFAULT_COARSE_SPACING_M = 5.0


# --------------------------------------------------------------------------- #
# Fast ASCII parsing
# --------------------------------------------------------------------------- #
def fast_parse(file_path: str, full: bool = False) -> np.ndarray:
    """Parse an ``U_zNormal_*.raw`` surface file with numpy's C ``loadtxt`` reader.

    The files are ``# header`` x2 then whitespace-separated ``x y z U_x U_y U_z``
    rows. numpy >= 1.23's ``loadtxt`` is a fast C reader; ``usecols`` lets it skip
    converting the three coordinate columns when only velocity is needed (which is
    every timestep after the grid is indexed) - roughly 0.47 s vs 0.66 s per ~1 M
    rows here, and ~13x faster than ``np.fromstring``.

    Args:
        file_path: path to the .raw file.
        full: when True return all 6 columns ``(N, 6)``; otherwise just the three
              velocity columns ``(N, 3)``.

    Returns:
        float32 array; empty ``(0, 6)`` / ``(0, 3)`` on malformed/missing input.
    """
    ncols = 6 if full else 3
    try:
        data = np.loadtxt(
            file_path,
            skiprows=2,
            dtype=np.float32,
            usecols=None if full else (3, 4, 5),
        )
    except (OSError, ValueError):
        return np.empty((0, ncols), dtype=np.float32)

    if data.size == 0:
        return np.empty((0, ncols), dtype=np.float32)
    if data.ndim == 1:  # single row collapses to 1-D
        data = data.reshape(1, -1)
    if data.shape[1] != ncols:
        return np.empty((0, ncols), dtype=np.float32)
    return data


def _read_point_count(file_path: str) -> int:
    """Read POINT_DATA N from the first header line without parsing the body."""
    try:
        with open(file_path, "rb") as fh:
            header = fh.readline().decode("ascii", "replace")
    except OSError:
        return 0
    for token in header.split():
        if token.isdigit():
            return int(token)
    return 0


# --------------------------------------------------------------------------- #
# Cache location / fingerprint
# --------------------------------------------------------------------------- #
def _cache_root() -> str:
    """Writable, persistent cache root (see app_paths.user_cache_root)."""
    return app_paths.user_cache_root()


def cache_dir_for(surfaces_dir: str) -> str:
    """Per-case cache folder, keyed by a hash of the absolute surfaces path."""
    key = os.path.abspath(surfaces_dir).lower().encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:16]
    return os.path.join(_cache_root(), digest)


def _field_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "field_f16.dat")


def _coarse_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "coarse_f32.npy")


def _meta_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "meta.npz")


def _manifest_path(cache_dir: str) -> str:
    return os.path.join(cache_dir, "manifest.json")


def _list_time_dirs(surfaces_dir: str):
    # Imported lazily to avoid a circular import (loader imports this module).
    from .openfoam_loader import _list_time_dirs as _ltd
    return _ltd(surfaces_dir)


def _list_z_files(time_dir: str):
    from .openfoam_loader import _list_z_files as _lzf
    return _lzf(time_dir)


def _source_fingerprint(surfaces_dir: str) -> Dict:
    """Cheap fingerprint of the source case to detect stale caches.

    Uses the time-dir count plus the size/mtime of the first timestep's z-files -
    enough to catch a re-run/regenerated case without scanning all 195 GB.
    """
    time_dirs, _ = _list_time_dirs(surfaces_dir)
    fp: Dict = {"n_times": len(time_dirs), "first": None, "last": None, "z_files": []}
    if not time_dirs:
        return fp
    fp["first"] = time_dirs[0]
    fp["last"] = time_dirs[-1]
    first_dir = os.path.join(surfaces_dir, time_dirs[0])
    for name in _list_z_files(first_dir):
        path = os.path.join(first_dir, name)
        try:
            st = os.stat(path)
            fp["z_files"].append([name, st.st_size, int(st.st_mtime)])
        except OSError:
            fp["z_files"].append([name, -1, -1])
    return fp


def estimate_dense_gb(surfaces_dir: str) -> float:
    """Estimate the dense float32 size (GB) without parsing the body.

    Approximates the horizontal grid by the first z-file's POINT_DATA count
    (holes are a tiny fraction), times z-levels, times timesteps.
    """
    time_dirs, _ = _list_time_dirs(surfaces_dir)
    if not time_dirs:
        return 0.0
    first_dir = os.path.join(surfaces_dir, time_dirs[0])
    z_files = _list_z_files(first_dir)
    if not z_files:
        return 0.0
    n_points = _read_point_count(os.path.join(first_dir, z_files[0]))
    if n_points <= 0:
        return 0.0
    return 3 * len(z_files) * n_points * len(time_dirs) * 4 / 1e9


def estimated_cache_gb(surfaces_dir: str) -> float:
    """Estimate on-disk cache size (GB): f16 memmap + coarse field + ~10% headroom.

    Used to space-check the chosen cache drive before a build.
    """
    dense_f32 = estimate_dense_gb(surfaces_dir)
    if dense_f32 <= 0.0:
        return 0.0
    f16 = dense_f32 / 2.0
    coarse = f16 / float(DEFAULT_COARSE_SPACING_M ** 2)  # ~stride^2 fewer points
    return (f16 + coarse) * 1.10


# --------------------------------------------------------------------------- #
# Cache validity
# --------------------------------------------------------------------------- #
def _read_manifest(cache_dir: str) -> Optional[Dict]:
    try:
        with open(_manifest_path(cache_dir), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def is_cached(surfaces_dir: str) -> bool:
    """True if a complete, current cache exists for this surfaces folder."""
    cache_dir = cache_dir_for(surfaces_dir)
    manifest = _read_manifest(cache_dir)
    if not manifest or manifest.get("cache_version") != CACHE_VERSION:
        return False
    if not (os.path.isfile(_field_path(cache_dir))
            and os.path.isfile(_coarse_path(cache_dir))
            and os.path.isfile(_meta_path(cache_dir))):
        return False
    return manifest.get("fingerprint") == _source_fingerprint(surfaces_dir)


def is_case_cached(selected_path: str) -> bool:
    """Like :func:`is_cached` but accepts a case root or surfaces folder.

    Resolves the surfaces directory first; returns False if it cannot be resolved
    (e.g. the drive is absent) so startup can fall back silently.
    """
    from .openfoam_loader import _resolve_surfaces_dir
    try:
        surfaces_dir, _ = _resolve_surfaces_dir(selected_path)
    except (FileNotFoundError, OSError):
        return False
    return is_cached(surfaces_dir)


# --------------------------------------------------------------------------- #
# Grid indexing (computed once from timestep 0)
# --------------------------------------------------------------------------- #
class _ZIndex:
    """Per-z-level scatter + hole-fill mapping, derived once from timestep 0."""

    __slots__ = ("n_rows", "y_idx", "x_idx", "gather")

    def __init__(self, n_rows, y_idx, x_idx, gather):
        self.n_rows = n_rows
        self.y_idx = y_idx
        self.x_idx = x_idx
        self.gather = gather  # flat (Y*X,) nearest-valid source index


def _build_grid_index(surfaces_dir: str):
    """Parse timestep 0 to build coords + per-z scatter/fill maps.

    Returns:
        (x_coords, y_coords, z_coords, z_files, [ _ZIndex per z ])
    """
    time_dirs, _ = _list_time_dirs(surfaces_dir)
    if not time_dirs:
        raise ValueError("No OpenFOAM time directories found.")
    first_dir = os.path.join(surfaces_dir, time_dirs[0])
    z_files = _list_z_files(first_dir)
    if not z_files:
        raise ValueError("No U_zNormal_*.raw files in the first time directory.")

    parsed: List[np.ndarray] = []
    x_set, y_set, z_levels = set(), set(), []
    for name in z_files:
        data = fast_parse(os.path.join(first_dir, name), full=True)
        if data.size == 0:
            raise ValueError(f"Empty/unreadable surface file: {name}")
        parsed.append(data)
        x_set.update(np.rint(data[:, 0]).astype(np.int64).tolist())
        y_set.update(np.rint(data[:, 1]).astype(np.int64).tolist())
        z_levels.append(float(np.round(np.median(data[:, 2]), 3)))

    x_coords = np.array(sorted(x_set), dtype=np.float32)
    y_coords = np.array(sorted(y_set), dtype=np.float32)
    z_coords = np.array(z_levels, dtype=np.float32)
    X, Y = len(x_coords), len(y_coords)

    z_indices: List[_ZIndex] = []
    for data in parsed:
        xr = np.rint(data[:, 0]).astype(np.int64)
        yr = np.rint(data[:, 1]).astype(np.int64)
        x_idx = np.searchsorted(x_coords, xr).astype(np.int32)
        y_idx = np.searchsorted(y_coords, yr).astype(np.int32)

        valid = np.zeros((Y, X), dtype=bool)
        valid[y_idx, x_idx] = True
        # Nearest valid cell for every cell (identity at valid cells, nearest
        # neighbour at holes). EDT on the hole mask -> indices of nearest zero.
        inds = ndimage.distance_transform_edt(
            ~valid, return_distances=False, return_indices=True
        )
        gather = (inds[0] * X + inds[1]).astype(np.int64).ravel()
        z_indices.append(_ZIndex(data.shape[0], y_idx, x_idx, gather))

    return x_coords, y_coords, z_coords, z_files, z_indices


def _fill_slice(grid: np.ndarray, zidx: _ZIndex, X: int, Y: int) -> np.ndarray:
    """Apply the precomputed nearest-valid gather to a scattered (3, Y, X) grid."""
    return grid.reshape(3, -1)[:, zidx.gather].reshape(3, Y, X)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_cache(
    surfaces_dir: str,
    progress_cb: Optional[Callable[[float], None]] = None,
    cancel_flag: Optional[Callable[[], bool]] = None,
    coarse_spacing_m: float = DEFAULT_COARSE_SPACING_M,
    max_times: Optional[int] = None,
) -> str:
    """Convert a large OpenFOAM case into the binary cache. Returns the cache dir.

    One-time, single pass over every timestep. ``progress_cb(fraction)`` is called
    per timestep; ``cancel_flag()`` returning True aborts cleanly (leaving no DONE
    marker, so the partial build is treated as invalid). ``max_times`` caps the
    number of timesteps converted (used for quick previews/tests; None = all).
    """
    surfaces_dir = os.path.abspath(surfaces_dir)
    time_dirs, time_coords = _list_time_dirs(surfaces_dir)
    if max_times is not None:
        time_dirs = time_dirs[:max_times]
        time_coords = time_coords[:max_times]
    T = len(time_dirs)
    if T == 0:
        raise ValueError("No OpenFOAM time directories found.")

    x_coords, y_coords, z_coords, z_files, z_indices = _build_grid_index(surfaces_dir)
    X, Y, Z = len(x_coords), len(y_coords), len(z_coords)

    sx = max(1, int(round(coarse_spacing_m)))  # 1 m grid -> stride == spacing
    sy = max(1, int(round(coarse_spacing_m)))
    xc = x_coords[::sx]
    yc = y_coords[::sy]
    zc = z_coords
    Xc, Yc, Zc = len(xc), len(yc), len(zc)

    cache_dir = cache_dir_for(surfaces_dir)
    os.makedirs(cache_dir, exist_ok=True)

    # Remove any stale manifest up front so an interrupted build never validates.
    try:
        os.remove(_manifest_path(cache_dir))
    except OSError:
        pass

    mm = np.memmap(
        _field_path(cache_dir), dtype=np.float16, mode="w+", shape=(T, 3, Z, Y, X)
    )
    coarse = np.zeros((3, Zc, Yc, Xc, T), dtype=np.float32)
    max_speed = 0.0

    try:
        for t, time_dir in enumerate(time_dirs):
            if cancel_flag is not None and cancel_flag():
                raise RuntimeError("Cache build canceled.")

            time_path = os.path.join(surfaces_dir, time_dir)
            slice_full = np.zeros((3, Z, Y, X), dtype=np.float32)

            for z, (name, zidx) in enumerate(zip(z_files, z_indices)):
                file_path = os.path.join(time_path, name)
                u = fast_parse(file_path)  # (N, 3)
                if u.shape[0] == zidx.n_rows:
                    grid = np.zeros((3, Y, X), dtype=np.float32)
                    grid[:, zidx.y_idx, zidx.x_idx] = u.T
                    filled = _fill_slice(grid, zidx, X, Y)
                else:
                    # Layout drift (should not happen): rebuild this file's mapping.
                    filled = _scatter_fill_adhoc(file_path, x_coords, y_coords)
                slice_full[:, z] = filled

            mm[t] = slice_full.astype(np.float16)
            coarse[:, :, :, :, t] = slice_full[:, :, ::sy, ::sx]

            cmag = np.sqrt(np.einsum("czyx,czyx->zyx",
                                     coarse[:, :, :, :, t], coarse[:, :, :, :, t]))
            max_speed = max(max_speed, float(cmag.max()))

            if progress_cb is not None:
                progress_cb((t + 1) / T)

        mm.flush()
        np.save(_coarse_path(cache_dir), coarse)
        np.savez(
            _meta_path(cache_dir),
            x_coords=x_coords, y_coords=y_coords, z_coords=z_coords,
            time_coords=time_coords.astype(np.float32),
            xc=xc, yc=yc, zc=zc,
            sx=np.int32(sx), sy=np.int32(sy),
            field_shape=np.array([T, 3, Z, Y, X], dtype=np.int64),
            coarse_shape=np.array([3, Zc, Yc, Xc, T], dtype=np.int64),
            max_speed=np.float32(max_speed),
            coarse_spacing_m=np.float32(coarse_spacing_m),
        )
        # Manifest written last == DONE marker.
        with open(_manifest_path(cache_dir), "w", encoding="utf-8") as fh:
            json.dump({
                "cache_version": CACHE_VERSION,
                "surfaces_dir": surfaces_dir,
                "fingerprint": _source_fingerprint(surfaces_dir),
                "coarse_spacing_m": coarse_spacing_m,
            }, fh, indent=2)
    finally:
        del mm  # close the memmap handle

    return cache_dir


def _scatter_fill_adhoc(file_path, x_coords, y_coords) -> np.ndarray:
    """Fallback scatter+fill for a timestep whose row layout differs from t=0."""
    X, Y = len(x_coords), len(y_coords)
    data = fast_parse(file_path, full=True)
    xr = np.rint(data[:, 0]).astype(np.int64)
    yr = np.rint(data[:, 1]).astype(np.int64)
    x_idx = np.searchsorted(x_coords, xr)
    y_idx = np.searchsorted(y_coords, yr)
    valid = np.zeros((Y, X), dtype=bool)
    valid[y_idx, x_idx] = True
    grid = np.zeros((3, Y, X), dtype=np.float32)
    grid[:, y_idx, x_idx] = data[:, 3:6].T
    inds = ndimage.distance_transform_edt(~valid, return_distances=False,
                                          return_indices=True)
    gather = (inds[0] * X + inds[1]).ravel()
    return grid.reshape(3, -1)[:, gather].reshape(3, Y, X)


# --------------------------------------------------------------------------- #
# Runtime streaming source
# --------------------------------------------------------------------------- #
class OpenFOAMStreamingSource:
    """Runtime accessor over a built cache.

    Display reads the in-RAM coarse field; ML/object sampling and the fine patch
    read the memmapped full-resolution field. Coordinate/velocity ordering matches
    :class:`wind_data.wind_field.WindField`'s eager convention
    (``meshgrid(..., indexing='ij')`` with z varying fastest).
    """

    def __init__(self, cache_dir: str):
        meta = np.load(_meta_path(cache_dir))
        self.x_coords = meta["x_coords"].astype(np.float32)
        self.y_coords = meta["y_coords"].astype(np.float32)
        self.z_coords = meta["z_coords"].astype(np.float32)
        self.time_coords = meta["time_coords"].astype(np.float32)
        self.xc = meta["xc"].astype(np.float32)
        self.yc = meta["yc"].astype(np.float32)
        self.zc = meta["zc"].astype(np.float32)
        self.sx = int(meta["sx"])
        self.sy = int(meta["sy"])
        self._max_speed = float(meta["max_speed"])
        self.coarse_spacing_m = float(meta["coarse_spacing_m"])

        T, C, Z, Y, X = (int(v) for v in meta["field_shape"])
        self.shape = (T, C, Z, Y, X)
        self.mm = np.memmap(
            _field_path(cache_dir), dtype=np.float16, mode="r", shape=(T, C, Z, Y, X)
        )
        # Coarse field is small enough to hold resident for instant playback.
        self.coarse = np.load(_coarse_path(cache_dir))  # (3, Zc, Yc, Xc, T) f32

    @property
    def time_steps(self) -> int:
        return self.shape[0]

    def max_speed(self) -> float:
        return self._max_speed

    def coarse_velocities(self, t: int) -> np.ndarray:
        """(N, 3) coarse velocities aligned with the coarse meshgrid point order."""
        t = int(t) % self.shape[0]
        cur = self.coarse[:, :, :, :, t]                 # (3, Zc, Yc, Xc)
        return np.transpose(cur, (3, 2, 1, 0)).reshape(-1, 3).copy()

    def velocity_at_grid(self, t: int, z: int, y: int, x: int) -> np.ndarray:
        t = int(t) % self.shape[0]
        return self.mm[t, :, z, y, x].astype(np.float32)

    def box_at(self, t: int, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
        """Full-res velocity box ``(3, Z, y1-y0, x1-x0)`` at timestep ``t``."""
        t = int(t) % self.shape[0]
        return self.mm[t, :, :, y0:y1, x0:x1].astype(np.float32)


def open_source(surfaces_dir: str) -> OpenFOAMStreamingSource:
    """Open a streaming source for an already-cached case."""
    cache_dir = cache_dir_for(surfaces_dir)
    if not is_cached(surfaces_dir):
        raise FileNotFoundError(f"No valid cache for: {surfaces_dir}")
    return OpenFOAMStreamingSource(cache_dir)
