"""
OpenFOAM wind data loader.
Extracts wind data from postProcessing/surfaces output into a 5D array,
plus higher-level helpers for parsing a full OpenFOAM case (boundary
patches, triSurface STL geometry).

Data layout: (component, z, y, x, time). OpenFOAM is Z-up and the simulator
world is also Z-up, so coordinates and velocity components are passed through
without remapping.
"""

from collections import deque
from typing import Dict, List, Optional, Tuple
import os
import re
import struct

import numpy as np


_TIME_DIR_RE = re.compile(r"^\d+(?:\.\d+)?$")
_Z_FILE_RE = re.compile(r"^U_zNormal_(\d+)\.raw$")
_SURFACES_SUBPATH = os.path.join("postProcessing", "surfaces")


def _list_time_dirs(base_dir: str) -> Tuple[List[str], np.ndarray]:
    time_dirs = [
        name for name in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, name)) and _TIME_DIR_RE.match(name)
    ]
    time_dirs = sorted(time_dirs, key=lambda v: float(v))
    time_coords = np.array([float(v) for v in time_dirs], dtype=np.float32)
    return time_dirs, time_coords


def _list_z_files(time_dir: str) -> List[str]:
    candidates = []
    for name in os.listdir(time_dir):
        match = _Z_FILE_RE.match(name)
        if match:
            candidates.append((int(match.group(1)), name))
    candidates.sort(key=lambda item: item[0])
    return [name for _, name in candidates]


def _load_raw_file(file_path: str) -> np.ndarray:
    try:
        data = np.loadtxt(file_path, skiprows=2)
    except Exception:
        return np.empty((0, 6), dtype=np.float32)

    if data.size == 0:
        return np.empty((0, 6), dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 6:
        return np.empty((0, 6), dtype=np.float32)
    return data.astype(np.float32, copy=False)


def _fill_slice_nearest(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    valid = ~np.isnan(filled).any(axis=0)

    if valid.all():
        return filled

    if not valid.any():
        return np.zeros_like(filled)

    height, width = valid.shape
    src_y = np.full((height, width), -1, dtype=np.int32)
    src_x = np.full((height, width), -1, dtype=np.int32)
    dist = np.full((height, width), -1, dtype=np.int32)
    queue = deque()

    for y in range(height):
        for x in range(width):
            if valid[y, x]:
                src_y[y, x] = y
                src_x[y, x] = x
                dist[y, x] = 0
                queue.append((y, x))

    while queue:
        y, x = queue.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny = y + dy
            nx = x + dx
            if 0 <= ny < height and 0 <= nx < width and dist[ny, nx] == -1:
                dist[ny, nx] = dist[y, x] + 1
                src_y[ny, nx] = src_y[y, x]
                src_x[ny, nx] = src_x[y, x]
                queue.append((ny, nx))

    missing = ~valid
    if missing.any():
        my, mx = np.where(missing)
        filled[:, my, mx] = filled[:, src_y[my, mx], src_x[my, mx]]

    return filled


def _fill_missing_nearest(wind_data: np.ndarray) -> np.ndarray:
    filled = wind_data.copy()
    _, z_count, _, _, t_count = filled.shape

    for t in range(t_count):
        for z in range(z_count):
            slice_data = filled[:, z, :, :, t]
            if np.isnan(slice_data).any():
                filled[:, z, :, :, t] = _fill_slice_nearest(slice_data)

    return filled


def extract_openfoam_wind(base_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    time_dirs, time_coords = _list_time_dirs(base_dir)
    if not time_dirs:
        raise ValueError("No OpenFOAM time directories found.")

    first_time_dir = os.path.join(base_dir, time_dirs[0])
    z_files = _list_z_files(first_time_dir)
    if not z_files:
        raise ValueError("No U_zNormal_*.raw files found in the first time directory.")

    x_set = set()
    y_set = set()
    z_set = set()

    for time_dir in time_dirs:
        time_path = os.path.join(base_dir, time_dir)
        for z_file in z_files:
            file_path = os.path.join(time_path, z_file)
            if not os.path.exists(file_path):
                continue
            data = _load_raw_file(file_path)
            if data.size == 0:
                continue

            x_round = np.rint(data[:, 0]).astype(np.int32)
            y_round = np.rint(data[:, 1]).astype(np.int32)
            z_level = float(np.round(np.median(data[:, 2]), 3))

            x_set.update(x_round.tolist())
            y_set.update(y_round.tolist())
            z_set.add(z_level)

    if not x_set or not y_set or not z_set:
        raise ValueError("No usable wind coordinate data found.")

    x_coords = np.array(sorted(x_set), dtype=np.int32)
    y_coords = np.array(sorted(y_set), dtype=np.int32)
    z_coords = np.array(sorted(z_set), dtype=np.float32)

    wind_data = np.full(
        (3, len(z_coords), len(y_coords), len(x_coords), len(time_coords)),
        np.nan,
        dtype=np.float32
    )
    z_index = {float(value): idx for idx, value in enumerate(z_coords)}

    for time_idx, time_dir in enumerate(time_dirs):
        time_path = os.path.join(base_dir, time_dir)
        for z_file in z_files:
            file_path = os.path.join(time_path, z_file)
            if not os.path.exists(file_path):
                continue
            data = _load_raw_file(file_path)
            if data.size == 0:
                continue

            z_level = float(np.round(np.median(data[:, 2]), 3))
            z_idx = z_index.get(z_level)
            if z_idx is None:
                continue

            x_round = np.rint(data[:, 0]).astype(np.int32)
            y_round = np.rint(data[:, 1]).astype(np.int32)

            x_idx = np.searchsorted(x_coords, x_round)
            y_idx = np.searchsorted(y_coords, y_round)

            wind_data[0, z_idx, y_idx, x_idx, time_idx] = data[:, 3]
            wind_data[1, z_idx, y_idx, x_idx, time_idx] = data[:, 4]
            wind_data[2, z_idx, y_idx, x_idx, time_idx] = data[:, 5]

    wind_data = _fill_missing_nearest(wind_data)

    # OpenFOAM is Z-up; simulator world is also Z-up - no remap needed.
    return (
        wind_data.astype(np.float32, copy=False),
        x_coords.astype(np.float32),
        y_coords.astype(np.float32),
        z_coords.astype(np.float32),
        time_coords
    )


def _looks_like_surfaces_dir(path: str) -> bool:
    """Return True if `path` looks like a postProcessing/surfaces folder."""
    if not os.path.isdir(path):
        return False
    try:
        entries = os.listdir(path)
    except OSError:
        return False
    for name in entries:
        if not _TIME_DIR_RE.match(name):
            continue
        time_path = os.path.join(path, name)
        if not os.path.isdir(time_path):
            continue
        try:
            for fname in os.listdir(time_path):
                if _Z_FILE_RE.match(fname):
                    return True
        except OSError:
            continue
    return False


def _resolve_surfaces_dir(selected_path: str) -> Tuple[str, Optional[str]]:
    """Resolve the selected folder to (surfaces_dir, case_root_or_None).

    Accepts either an OpenFOAM case root (containing postProcessing/surfaces)
    or a surfaces folder directly.
    """
    selected_path = os.path.abspath(selected_path)
    if _looks_like_surfaces_dir(selected_path):
        return selected_path, None

    candidate = os.path.join(selected_path, _SURFACES_SUBPATH)
    if _looks_like_surfaces_dir(candidate):
        return candidate, selected_path

    raise FileNotFoundError(
        f"No OpenFOAM surfaces (postProcessing/surfaces with U_zNormal_*.raw) "
        f"found at or below: {selected_path}"
    )


_COMMENT_BLOCK_RE = re.compile(r"/\*.*?\*/", re.DOTALL)
_COMMENT_LINE_RE = re.compile(r"//[^\n]*")
_PATCH_BLOCK_RE = re.compile(
    r"([A-Za-z_][A-Za-z0-9_]*)\s*\{([^{}]*)\}",
    re.DOTALL,
)
_TYPE_FIELD_RE = re.compile(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)\s*;")


def parse_boundary_file(boundary_path: str) -> List[Dict[str, str]]:
    """Parse an OpenFOAM constant/polyMesh/boundary dictionary.

    Returns a list of {"name": str, "type": str} in file order.
    """
    with open(boundary_path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    text = _COMMENT_BLOCK_RE.sub(" ", text)
    text = _COMMENT_LINE_RE.sub(" ", text)

    # Skip the FoamFile header block (the first {...} block).
    foam_file_idx = text.find("FoamFile")
    if foam_file_idx != -1:
        brace_open = text.find("{", foam_file_idx)
        if brace_open != -1:
            depth = 0
            for i in range(brace_open, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        text = text[i + 1:]
                        break

    patches: List[Dict[str, str]] = []
    for match in _PATCH_BLOCK_RE.finditer(text):
        name = match.group(1)
        body = match.group(2)
        type_match = _TYPE_FIELD_RE.search(body)
        if type_match is None:
            continue
        patches.append({"name": name, "type": type_match.group(1)})

    return patches


def _parse_ascii_stl(stl_path: str) -> Dict[str, np.ndarray]:
    """Parse an ASCII STL file. Returns dict with vertices, faces, normals."""
    vertices: List[List[float]] = []
    faces: List[List[int]] = []
    normals_per_tri: List[List[float]] = []

    current_normal: Optional[List[float]] = None
    current_tri: List[int] = []

    with open(stl_path, "r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("facet normal"):
                parts = line.split()
                current_normal = [float(parts[2]), float(parts[3]), float(parts[4])]
                current_tri = []
            elif line.startswith("vertex"):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
                current_tri.append(len(vertices) - 1)
            elif line.startswith("endfacet"):
                if len(current_tri) == 3 and current_normal is not None:
                    faces.append(current_tri)
                    normals_per_tri.append(current_normal)
                current_normal = None
                current_tri = []

    if not vertices or not faces:
        raise ValueError(f"ASCII STL contained no triangles: {stl_path}")

    vertices_arr = np.asarray(vertices, dtype=np.float32)
    faces_arr = np.asarray(faces, dtype=np.int32)
    tri_normals = np.asarray(normals_per_tri, dtype=np.float32)
    normals_arr = np.zeros_like(vertices_arr)
    for tri_idx, face in enumerate(faces):
        for vert_idx in face:
            normals_arr[vert_idx] = tri_normals[tri_idx]
    return {"vertices": vertices_arr, "faces": faces_arr, "normals": normals_arr}


def parse_binary_stl(stl_path: str) -> Dict[str, np.ndarray]:
    """Parse a binary or ASCII STL file. Returns dict with vertices, faces, normals.

    Binary STL layout: 80-byte header, uint32 triangle count, then per-triangle
    [3 floats normal, 9 floats vertices, 2 bytes attribute] = 50 bytes.

    The "solid" prefix is not a reliable ASCII indicator (binary STLs sometimes
    use it too) - file size against the binary layout is authoritative.
    """
    file_size = os.path.getsize(stl_path)
    if file_size < 84:
        raise ValueError(f"STL file too small to be valid: {stl_path}")

    with open(stl_path, "rb") as f:
        header = f.read(80)
        tri_count_bytes = f.read(4)
        tri_count = struct.unpack("<I", tri_count_bytes)[0]
        expected_size = 84 + tri_count * 50

        if expected_size == file_size:
            payload = f.read(tri_count * 50)
        else:
            # Size mismatch -> treat as ASCII STL.
            return _parse_ascii_stl(stl_path)

    if tri_count == 0:
        raise ValueError(f"STL file contained no triangles: {stl_path}")

    dtype = np.dtype([
        ("normal", "<f4", (3,)),
        ("v0", "<f4", (3,)),
        ("v1", "<f4", (3,)),
        ("v2", "<f4", (3,)),
        ("attr", "<u2"),
    ])
    tris = np.frombuffer(payload, dtype=dtype, count=tri_count)

    vertices = np.empty((tri_count * 3, 3), dtype=np.float32)
    vertices[0::3] = tris["v0"]
    vertices[1::3] = tris["v1"]
    vertices[2::3] = tris["v2"]

    faces = np.arange(tri_count * 3, dtype=np.int32).reshape(tri_count, 3)

    normals = np.empty_like(vertices)
    tri_normals = tris["normal"].astype(np.float32)
    normals[0::3] = tri_normals
    normals[1::3] = tri_normals
    normals[2::3] = tri_normals

    return {"vertices": vertices, "faces": faces, "normals": normals}


def _load_tri_surfaces(tri_surface_dir: str, warnings: List[str]) -> List[Dict]:
    """Load every .stl in a constant/triSurface directory. Best-effort."""
    if not os.path.isdir(tri_surface_dir):
        return []

    results: List[Dict] = []
    for name in sorted(os.listdir(tri_surface_dir)):
        if not name.lower().endswith(".stl"):
            continue
        stl_path = os.path.join(tri_surface_dir, name)
        try:
            mesh = parse_binary_stl(stl_path)
        except Exception as exc:
            warnings.append(f"Failed to load triSurface {name}: {exc}")
            continue
        mesh["name"] = os.path.splitext(name)[0]
        results.append(mesh)
    return results


def extract_openfoam_case(selected_path: str) -> Dict:
    """Load wind + patches + triSurface geometry from an OpenFOAM case.

    `selected_path` may be the case root (containing postProcessing/) or the
    postProcessing/surfaces folder directly.

    Returns a dict with keys:
        surfaces_dir: str
        case_root: Optional[str]
        wind: (data, x_coords, y_coords, z_coords, time_coords)   # required
        patches: List[{"name": str, "type": str}]                 # may be []
        tri_surfaces: List[{"name", "vertices", "faces", "normals"}]
        warnings: List[str]
    """
    warnings: List[str] = []
    surfaces_dir, case_root = _resolve_surfaces_dir(selected_path)

    wind = extract_openfoam_wind(surfaces_dir)

    patches: List[Dict[str, str]] = []
    tri_surfaces: List[Dict] = []

    if case_root is not None:
        boundary_path = os.path.join(case_root, "constant", "polyMesh", "boundary")
        if os.path.isfile(boundary_path):
            try:
                patches = parse_boundary_file(boundary_path)
            except Exception as exc:
                warnings.append(f"Failed to parse boundary file: {exc}")
        else:
            warnings.append("No constant/polyMesh/boundary file found.")

        tri_surface_dir = os.path.join(case_root, "constant", "triSurface")
        if os.path.isdir(tri_surface_dir):
            tri_surfaces = _load_tri_surfaces(tri_surface_dir, warnings)
            if not tri_surfaces and not warnings[-1:] == ["Failed to load triSurface"]:
                # Empty triSurface dir is a soft warning only.
                if not any(name.lower().endswith(".stl") for name in os.listdir(tri_surface_dir)):
                    warnings.append("constant/triSurface contains no STL files.")
        else:
            warnings.append("No constant/triSurface directory found.")
    else:
        warnings.append(
            "Selected path is the surfaces folder directly; case root unknown, "
            "skipping boundary patches and triSurface geometry."
        )

    return {
        "surfaces_dir": surfaces_dir,
        "case_root": case_root,
        "wind": wind,
        "patches": patches,
        "tri_surfaces": tri_surfaces,
        "warnings": warnings,
    }
