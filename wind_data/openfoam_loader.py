"""
OpenFOAM wind data loader.
Extracts wind data from postProcessing/surfaces output into a 5D array.

Data layout: (component, z, y, x, time)
"""

from collections import deque
from typing import List, Tuple
import os
import re

import numpy as np


_TIME_DIR_RE = re.compile(r"^\d+(?:\.\d+)?$")
_Z_FILE_RE = re.compile(r"^U_zNormal_(\d+)\.raw$")


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
    return wind_data, x_coords, y_coords, z_coords, time_coords
