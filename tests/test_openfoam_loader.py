"""Unit tests for wind_data.openfoam_loader."""
import struct

import numpy as np
import pytest

from wind_data.openfoam_loader import (
    _fill_missing_nearest,
    _fill_slice_nearest,
    _looks_like_surfaces_dir,
    _resolve_surfaces_dir,
    extract_openfoam_case,
    extract_openfoam_wind,
    parse_binary_stl,
    parse_boundary_file,
)


def _write_raw(path, rows):
    """Write a U_zNormal_*.raw file: 2 header lines + 'x y z ux uy uz' rows."""
    lines = ["# header line 1", "# header line 2"]
    for row in rows:
        lines.append(" ".join(str(v) for v in row))
    path.write_text("\n".join(lines) + "\n")


def _make_case(base_dir, time_to_ux):
    """Build a minimal synthetic surfaces case.

    Two z-levels (2.0, 5.0) and a 2x2 x/y grid. For each requested time
    directory, `time_to_ux[time]` is (ux_at_z2, ux_at_z5) applied uniformly.
    """
    grid_xy = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for time_label, (ux_z2, ux_z5) in time_to_ux.items():
        time_dir = base_dir / time_label
        time_dir.mkdir(parents=True)

        rows_z2 = [(x, y, 2.0, ux_z2, 0.0, 0.0) for x, y in grid_xy]
        rows_z5 = [(x, y, 5.0, ux_z5, 0.0, 0.0) for x, y in grid_xy]
        _write_raw(time_dir / "U_zNormal_2.raw", rows_z2)
        _write_raw(time_dir / "U_zNormal_5.raw", rows_z5)


# ---------------------------------------------------------------------------
# extract_openfoam_wind
# ---------------------------------------------------------------------------

def test_extract_openfoam_wind_builds_correct_shape_and_coords(tmp_path):
    _make_case(tmp_path, {"1": (1.0, 2.0), "2": (10.0, 20.0)})

    wind_data, x_coords, y_coords, z_coords, time_coords = extract_openfoam_wind(str(tmp_path))

    assert wind_data.shape == (3, 2, 2, 2, 2)
    assert np.array_equal(x_coords, [0, 1])
    assert np.array_equal(y_coords, [0, 1])
    assert np.allclose(z_coords, [2.0, 5.0])
    assert np.allclose(sorted(time_coords.tolist()), [1.0, 2.0])


def test_extract_openfoam_wind_places_values_at_correct_time_and_z(tmp_path):
    _make_case(tmp_path, {"1": (1.0, 2.0), "2": (10.0, 20.0)})
    wind_data, x_coords, y_coords, z_coords, time_coords = extract_openfoam_wind(str(tmp_path))

    t1 = int(np.where(time_coords == 1.0)[0][0])
    t2 = int(np.where(time_coords == 2.0)[0][0])
    z2_idx = int(np.where(z_coords == 2.0)[0][0])
    z5_idx = int(np.where(z_coords == 5.0)[0][0])

    assert np.allclose(wind_data[0, z2_idx, :, :, t1], 1.0)
    assert np.allclose(wind_data[0, z5_idx, :, :, t1], 2.0)
    assert np.allclose(wind_data[0, z2_idx, :, :, t2], 10.0)
    assert np.allclose(wind_data[0, z5_idx, :, :, t2], 20.0)
    # uy/uz were written as zero everywhere.
    assert np.allclose(wind_data[1:], 0.0)


def test_extract_openfoam_wind_raises_without_time_dirs(tmp_path):
    with pytest.raises(ValueError):
        extract_openfoam_wind(str(tmp_path))


def test_extract_openfoam_wind_raises_without_z_files(tmp_path):
    (tmp_path / "1").mkdir()
    with pytest.raises(ValueError):
        extract_openfoam_wind(str(tmp_path))


# ---------------------------------------------------------------------------
# Missing-data nearest-neighbor fill
# ---------------------------------------------------------------------------

def test_fill_slice_nearest_fills_gap_from_nearest_valid_point():
    # 3-component slice over a 1x3 grid; middle point is missing.
    values = np.array([
        [[1.0, np.nan, 3.0]],
        [[0.0, np.nan, 0.0]],
        [[0.0, np.nan, 0.0]],
    ])
    filled = _fill_slice_nearest(values)
    assert not np.isnan(filled).any()
    # Nearest neighbor (tie -> BFS picks one of the equidistant sources).
    assert filled[0, 0, 1] in (1.0, 3.0)


def test_fill_slice_nearest_returns_zeros_when_all_missing():
    values = np.full((3, 2, 2), np.nan)
    filled = _fill_slice_nearest(values)
    assert np.array_equal(filled, np.zeros_like(values))


def test_fill_slice_nearest_is_noop_when_fully_valid():
    values = np.ones((3, 2, 2))
    filled = _fill_slice_nearest(values)
    assert np.array_equal(filled, values)


def test_fill_missing_nearest_only_touches_slices_with_nans():
    wind_data = np.zeros((3, 1, 2, 2, 1))
    wind_data[0, 0, 0, 0, 0] = np.nan
    wind_data[0, 0, 1, 1, 0] = 5.0
    filled = _fill_missing_nearest(wind_data)
    assert not np.isnan(filled).any()
    assert filled[0, 0, 1, 1, 0] == 5.0


# ---------------------------------------------------------------------------
# Surfaces-directory resolution
# ---------------------------------------------------------------------------

def test_looks_like_surfaces_dir_true_for_valid_case(tmp_path):
    _make_case(tmp_path, {"1": (1.0, 2.0)})
    assert _looks_like_surfaces_dir(str(tmp_path)) is True


def test_looks_like_surfaces_dir_false_for_empty_dir(tmp_path):
    assert _looks_like_surfaces_dir(str(tmp_path)) is False


def test_resolve_surfaces_dir_accepts_surfaces_dir_directly(tmp_path):
    _make_case(tmp_path, {"1": (1.0, 2.0)})
    surfaces_dir, case_root = _resolve_surfaces_dir(str(tmp_path))
    assert surfaces_dir == str(tmp_path)
    assert case_root is None


def test_resolve_surfaces_dir_accepts_case_root(tmp_path):
    surfaces = tmp_path / "postProcessing" / "surfaces"
    _make_case(surfaces, {"1": (1.0, 2.0)})
    surfaces_dir, case_root = _resolve_surfaces_dir(str(tmp_path))
    assert surfaces_dir == str(surfaces)
    assert case_root == str(tmp_path)


def test_resolve_surfaces_dir_raises_when_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        _resolve_surfaces_dir(str(tmp_path))


def test_extract_openfoam_case_surfaces_dir_only_skips_patches(tmp_path):
    _make_case(tmp_path, {"1": (1.0, 2.0)})
    result = extract_openfoam_case(str(tmp_path))
    assert result["case_root"] is None
    assert result["patches"] == []
    assert result["tri_surfaces"] == []
    assert any("case root unknown" in w for w in result["warnings"])


def test_extract_openfoam_case_with_case_root_parses_boundary_and_warns_on_missing_tri(tmp_path):
    surfaces = tmp_path / "postProcessing" / "surfaces"
    _make_case(surfaces, {"1": (1.0, 2.0)})

    poly_mesh = tmp_path / "constant" / "polyMesh"
    poly_mesh.mkdir(parents=True)
    (poly_mesh / "boundary").write_text(
        "FoamFile\n{\n    version 2.0;\n}\n"
        "2\n(\n"
        "inlet\n{\n    type patch;\n    nFaces 10;\n}\n"
        "ground\n{\n    type wall;\n    nFaces 20;\n}\n"
        ")\n"
    )

    result = extract_openfoam_case(str(tmp_path))
    assert result["case_root"] == str(tmp_path)
    assert {"name": "inlet", "type": "patch"} in result["patches"]
    assert {"name": "ground", "type": "wall"} in result["patches"]
    assert any("triSurface" in w for w in result["warnings"])


# ---------------------------------------------------------------------------
# parse_boundary_file
# ---------------------------------------------------------------------------

def test_parse_boundary_file_skips_foamfile_header(tmp_path):
    boundary = tmp_path / "boundary"
    boundary.write_text(
        "FoamFile\n{\n    version 2.0;\n    type wrongtype;\n}\n"
        "1\n(\n"
        "outlet\n{\n    type patch;\n    nFaces 5;\n}\n"
        ")\n"
    )
    patches = parse_boundary_file(str(boundary))
    assert patches == [{"name": "outlet", "type": "patch"}]


def test_parse_boundary_file_strips_comments(tmp_path):
    boundary = tmp_path / "boundary"
    boundary.write_text(
        "/* block comment\n spanning lines */\n"
        "wall1\n{\n    // a line comment\n    type wall;\n}\n"
    )
    patches = parse_boundary_file(str(boundary))
    assert patches == [{"name": "wall1", "type": "wall"}]


# ---------------------------------------------------------------------------
# parse_binary_stl (binary + ASCII fallback)
# ---------------------------------------------------------------------------

def _write_binary_stl(path, triangles):
    """triangles: list of (normal(3,), v0(3,), v1(3,), v2(3,))."""
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", len(triangles)))
        for normal, v0, v1, v2 in triangles:
            f.write(struct.pack("<3f", *normal))
            f.write(struct.pack("<3f", *v0))
            f.write(struct.pack("<3f", *v1))
            f.write(struct.pack("<3f", *v2))
            f.write(struct.pack("<H", 0))


def test_parse_binary_stl_reads_vertices_faces_normals(tmp_path):
    stl_path = tmp_path / "shape.stl"
    triangles = [
        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)),
    ]
    _write_binary_stl(stl_path, triangles)

    mesh = parse_binary_stl(str(stl_path))
    assert mesh["vertices"].shape == (6, 3)
    assert mesh["faces"].shape == (2, 3)
    assert mesh["normals"].shape == (6, 3)
    assert np.allclose(mesh["vertices"][0], [0.0, 0.0, 0.0])
    assert np.allclose(mesh["normals"][0], [0.0, 0.0, 1.0])


def test_parse_binary_stl_too_small_raises(tmp_path):
    stl_path = tmp_path / "tiny.stl"
    stl_path.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError):
        parse_binary_stl(str(stl_path))


def test_parse_binary_stl_zero_triangles_raises(tmp_path):
    stl_path = tmp_path / "empty.stl"
    with open(stl_path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", 0))
    with pytest.raises(ValueError):
        parse_binary_stl(str(stl_path))


def test_parse_binary_stl_falls_back_to_ascii_on_size_mismatch(tmp_path):
    stl_path = tmp_path / "ascii.stl"
    stl_path.write_text(
        "solid test\n"
        "facet normal 0.0 0.0 1.0\n"
        "outer loop\n"
        "vertex 0.0 0.0 0.0\n"
        "vertex 1.0 0.0 0.0\n"
        "vertex 0.0 1.0 0.0\n"
        "endloop\n"
        "endfacet\n"
        "endsolid test\n"
    )
    mesh = parse_binary_stl(str(stl_path))
    assert mesh["vertices"].shape == (3, 3)
    assert mesh["faces"].shape == (1, 3)
    assert np.allclose(mesh["normals"][0], [0.0, 0.0, 1.0])
