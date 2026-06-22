"""Unit tests for renderer.scene.Scene: grid/flag-pole logic and object management."""
import numpy as np
import pytest

from models import config as cfg
from renderer.scene import Scene


@pytest.fixture
def scene():
    return Scene()


# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------

def test_grid_defaults_cover_plus_minus_50m(scene):
    assert scene.grid_size == 100
    assert scene.grid_spacing == cfg.GRID_SPACING
    assert scene.grid_center == (0.0, 0.0)
    half_extent = scene.grid_size // 2 * scene.grid_spacing
    assert half_extent == 50.0


def test_flag_pole_height_comes_from_config(scene):
    assert scene.flag_pole_height == cfg.FLAG_POLE_HEIGHT


# ---------------------------------------------------------------------------
# snap_to_grid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_pos,expected_xy", [
    ([1.5, 0.5, 2.3], [1.5, 0.5]),
    ([1.4, 1.0, 2.4], [1.5, 1.5]),
    ([0.0, 5.0, 0.0], [0.5, 5.5]),
    ([-1.5, 2.0, -2.3], [-1.5, 2.5]),
    ([25.7, 0.0, -30.6], [25.5, 0.5]),
])
def test_snap_to_grid_snaps_xy_to_cell_centers(scene, input_pos, expected_xy):
    snapped = scene.snap_to_grid(np.array(input_pos))
    assert np.allclose(snapped[:2], expected_xy, atol=1e-6)


def test_snap_to_grid_pins_height_to_flag_pole(scene):
    snapped = scene.snap_to_grid(np.array([3.2, -4.1, 99.0]))
    assert snapped[2] == scene.flag_pole_height


def test_snap_to_grid_does_not_mutate_input(scene):
    original = np.array([1.4, 1.0, 2.4])
    original_copy = original.copy()
    scene.snap_to_grid(original)
    assert np.array_equal(original, original_copy)


# ---------------------------------------------------------------------------
# get_grid_points
# ---------------------------------------------------------------------------

def test_grid_points_shape_and_range(scene):
    points = scene.get_grid_points()
    assert points.shape == (101 * 101, 3)
    assert points[:, 0].min() >= -50.0
    assert points[:, 0].max() <= 50.0
    assert points[:, 1].min() >= -50.0
    assert points[:, 1].max() <= 50.0
    assert np.all(points[:, 2] == 0.0)


# ---------------------------------------------------------------------------
# get_grid_geometry (cached ground-grid lines)
# ---------------------------------------------------------------------------

def test_grid_geometry_shape(scene):
    verts = scene.get_grid_geometry()
    half = scene.grid_size // 2
    n = 2 * half + 1
    assert verts.shape == (4 * n, 3)
    assert verts.dtype == np.float32


def test_grid_geometry_is_cached_between_calls(scene):
    first = scene.get_grid_geometry()
    second = scene.get_grid_geometry()
    assert first is second


def test_grid_geometry_cache_invalidated_on_spacing_change(scene):
    first = scene.get_grid_geometry()
    scene.grid_spacing = 2.0
    second = scene.get_grid_geometry()
    assert second is not first
    # Shape is unchanged (line count depends on grid_size, not spacing) but
    # the line extent should double along with the spacing.
    assert second.shape == first.shape
    assert second[:, :2].max() == pytest.approx(first[:, :2].max() * 2)


def test_grid_geometry_cache_invalidated_on_center_change(scene):
    first = scene.get_grid_geometry()
    scene.grid_center = (5.0, 5.0)
    second = scene.get_grid_geometry()
    assert second is not first
    assert not np.allclose(first, second)


def test_grid_geometry_lines_span_full_extent(scene):
    verts = scene.get_grid_geometry()
    extent = scene.grid_size // 2 * scene.grid_spacing
    assert verts[:, 0].max() == pytest.approx(extent)
    assert verts[:, 0].min() == pytest.approx(-extent)
    assert verts[:, 1].max() == pytest.approx(extent)
    assert verts[:, 1].min() == pytest.approx(-extent)


# ---------------------------------------------------------------------------
# Object management
# ---------------------------------------------------------------------------

def test_add_object_appends_to_scene_and_tracks_id(scene):
    mesh = scene.add_object('flag', (1.5, 1.5, 1.5))
    assert mesh in scene.objects
    assert len(scene.objects) == 1
    assert scene._object_ids[0] is mesh


def test_add_flag_lifts_bottom_to_ground_plus_offset(scene):
    requested_z = 1.5
    mesh = scene.add_object('flag', (0.0, 0.0, requested_z))
    mesh_z_min = float(np.min(mesh.vertices[:, 2]))
    # Flags get an extra 1.0m lift beyond resting their bottom on the ground.
    assert mesh.position[2] == pytest.approx(requested_z - mesh_z_min + 1.0)


def test_add_object_computes_edge_index_and_rest_lengths(scene):
    mesh = scene.add_object('flag', (0.0, 0.0, 1.5))
    assert mesh.edge_index.shape[0] == 2
    assert mesh.edge_index.numel() > 0
    assert mesh.rest_lengths.shape[0] == mesh.edge_index.shape[1]


def test_remove_object_clears_selection_and_id_mapping(scene):
    mesh = scene.add_object('flag', (0.0, 0.0, 1.5))
    scene.select_object(mesh)
    scene.remove_object(mesh)
    assert mesh not in scene.objects
    assert scene.selected_object is None
    assert mesh not in scene._object_ids.values()


def test_clear_objects_empties_scene(scene):
    scene.add_object('flag', (0.0, 0.0, 1.5))
    scene.add_object('flag', (2.0, 2.0, 1.5))
    scene.select_object(scene.objects[0])
    scene.clear_objects()
    assert scene.objects == []
    assert scene._object_ids == {}
    assert scene.selected_object is None


def test_move_object_updates_position_for_tracked_object(scene):
    mesh = scene.add_object('flag', (0.0, 0.0, 1.5))
    scene.move_object(mesh, np.array([3.0, 4.0, 1.5]))
    assert np.allclose(mesh.position, [3.0, 4.0, 1.5])


def test_move_object_ignores_untracked_mesh(scene):
    mesh = scene.add_object('flag', (0.0, 0.0, 1.5))
    scene.remove_object(mesh)
    original_position = mesh.position.copy()
    scene.move_object(mesh, np.array([9.0, 9.0, 9.0]))
    assert np.array_equal(mesh.position, original_position)


def test_get_object_at_position_finds_nearby_object(scene):
    mesh = scene.add_object('flag', (5.0, 5.0, 1.5))
    found = scene.get_object_at_position(mesh.get_center(), tolerance=1.0)
    assert found is mesh


def test_get_object_at_position_returns_none_when_far(scene):
    scene.add_object('flag', (5.0, 5.0, 1.5))
    found = scene.get_object_at_position(np.array([500.0, 500.0, 0.0]), tolerance=1.0)
    assert found is None


# ---------------------------------------------------------------------------
# Bounds / toggles
# ---------------------------------------------------------------------------

def test_get_bounds_matches_grid_when_no_objects(scene):
    min_corner, max_corner = scene.get_bounds()
    half = scene.grid_size // 2 * scene.grid_spacing
    assert min_corner[0] == -half
    assert max_corner[0] == half


def test_get_bounds_expands_to_include_objects(scene):
    min_before, max_before = scene.get_bounds()
    scene.add_object('flag', (60.0, 60.0, 1.5))
    min_after, max_after = scene.get_bounds()
    assert max_after[0] >= max_before[0]
    assert max_after[1] >= max_before[1]


def test_toggle_grid_flips_visibility(scene):
    initial = scene.grid_visible
    scene.toggle_grid()
    assert scene.grid_visible is not initial


def test_toggle_wind_vectors_flips_visibility(scene):
    initial = scene.wind_vectors_visible
    scene.toggle_wind_vectors()
    assert scene.wind_vectors_visible is not initial


# ---------------------------------------------------------------------------
# Serialize / deserialize round trip
# ---------------------------------------------------------------------------

def test_serialize_deserialize_round_trip_restores_camera_and_settings(scene):
    scene.add_object('flag', (2.5, 2.5, 1.5))
    scene.camera.zoom(5.0)
    scene.camera.orbit(10.0, 5.0)
    scene.grid_visible = False

    data = scene.serialize()

    fresh = Scene()
    fresh.deserialize(data)

    assert len(fresh.objects) == 1
    assert fresh.objects[0].name == 'flag'
    assert fresh.camera.distance == pytest.approx(scene.camera.distance)
    assert fresh.camera.azimuth == pytest.approx(scene.camera.azimuth)
    assert fresh.grid_visible == scene.grid_visible


def test_serialize_deserialize_known_bug_flag_position_is_not_round_tripped(scene):
    """Documents a known bug (not fixed here, per explicit instruction):

    serialize() stores a flag's already-lifted position (ground offset +
    1.0m applied by add_object), and deserialize() feeds that lifted value
    back into add_object, which lifts it a second time. So reloading a saved
    scene raises every flag's height, compounding on each save/load cycle.
    """
    original = scene.add_object('flag', (2.5, 2.5, 1.5))

    data = scene.serialize()
    fresh = Scene()
    fresh.deserialize(data)
    reloaded = fresh.objects[0]

    lift = original.position[2] - 1.5  # the lift add_object applied once
    assert reloaded.position[2] == pytest.approx(original.position[2] + lift)
    assert not np.allclose(reloaded.position, original.position)
