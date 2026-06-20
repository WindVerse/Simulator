"""
Wind Vector Geometry
Builds line-segment vertex arrays (and per-vertex colors) for wind vectors.

Pure functions of the wind state — no GL calls, no caching. The Scene owns the
cache so the geometry is built once and shared by every viewport.
"""

import numpy as np

from renderer.wind_colormap import magnitudes_to_colors


def build_wind_geometry(points, velocities, stride, scale, mode, color_by_speed):
    """Build line-segment vertex array(s) for the current wind state.

    Returns:
        Resultant mode: (vertex_array (M*6,3) float32, color_array (M*6,4) float32 or None).
        Components mode: list of three (M_axis*6, 3) float32 vertex arrays.
    """
    origins = points[::stride]
    vels = velocities[::stride]

    if mode == "components":
        return [
            _segments_for_displacements(
                origins, _axis_displacement(vels, axis, scale)
            )[0]
            for axis in range(3)
        ]

    verts, lengths_kept = _segments_for_displacements(
        origins, vels * scale
    )
    if not color_by_speed or lengths_kept is None or lengths_kept.size == 0:
        return (verts, None)

    # Visual length = scale * physical speed (m/s). Recover physical speed.
    speeds = lengths_kept / max(scale, 1e-12)
    arrow_colors = magnitudes_to_colors(speeds)             # (M_kept, 4)
    vertex_colors = np.repeat(arrow_colors, 6, axis=0)      # (M_kept*6, 4)
    vertex_colors = np.ascontiguousarray(vertex_colors, dtype=np.float32)
    return (verts, vertex_colors)


def _axis_displacement(vels, axis, scale):
    """Build (M,3) displacements that are zero except on `axis`."""
    out = np.zeros_like(vels)
    out[:, axis] = vels[:, axis] * scale
    return out


def _segments_for_displacements(origins, displacements):
    """Vectorized arrow geometry → flat GL_LINES vertex array.

    Filters near-zero displacements (they wouldn't draw and would cause
    NaN units), then emits six vertices per arrow:
        [origin, tip, tip, left, tip, right]
    for the shaft line plus the two arrowhead wings.

    Returns:
        (segments (M*6, 3) float32, kept_lengths (M,) float32 or None).
    """
    sq_len = np.einsum('ij,ij->i', displacements, displacements)
    mask = sq_len > 1e-12
    if not np.any(mask):
        return np.empty((0, 3), dtype=np.float32), None

    origins = origins[mask]
    disp = displacements[mask]
    lengths = np.sqrt(sq_len[mask])
    tips = origins + disp
    units = disp / lengths[:, None]

    # Pick a reference axis least aligned with each unit to avoid
    # degenerate cross products when units ≈ ±X.
    x_dominant = np.abs(units[:, 0]) > 0.9
    ref = np.where(
        x_dominant[:, None],
        np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
    )
    side = np.cross(units, ref)
    side_norm = np.linalg.norm(side, axis=1)
    # Any zero side-norm here is impossible given the ref choice, but
    # guard anyway by leaving those rows' wings collapsed to the tip.
    safe = side_norm > 1e-6
    side[safe] = side[safe] / side_norm[safe, None]
    side[~safe] = 0.0

    head_len = np.clip(0.2 * lengths, 0.05, 0.3)[:, None]
    base = tips - units * head_len
    wing = side * (head_len * 0.5)
    left = base + wing
    right = base - wing

    # Stack as (M, 6, 3) then flatten: each row contributes shaft+wings.
    segments = np.stack([origins, tips, tips, left, tips, right], axis=1)
    return (
        np.ascontiguousarray(segments.reshape(-1, 3), dtype=np.float32),
        lengths.astype(np.float32, copy=False),
    )
