"""Unit tests for renderer.scene.Camera (orbit/zoom/pan/presets/focus animation)."""
import numpy as np
import pytest

from renderer.scene import Camera, _DEFAULT_AZIMUTH, _DEFAULT_DISTANCE, _DEFAULT_ELEVATION, _DEFAULT_TARGET


@pytest.fixture
def camera():
    return Camera()


def test_initial_state_matches_defaults(camera):
    assert camera.distance == _DEFAULT_DISTANCE
    assert camera.azimuth == _DEFAULT_AZIMUTH
    assert camera.elevation == _DEFAULT_ELEVATION
    assert np.allclose(camera.target, _DEFAULT_TARGET)
    assert camera.zoom_min == 0.2
    assert camera.zoom_max == 200.0
    assert camera.near == 0.01
    assert camera.projection == "perspective"
    assert camera.locked is False


@pytest.mark.parametrize("delta", [5.0, 10.0, 1.0])
def test_zoom_in_decreases_distance_and_stays_in_bounds(camera, delta):
    initial = camera.distance
    camera.zoom(delta)
    assert camera.distance < initial
    assert camera.zoom_min <= camera.distance <= camera.zoom_max


@pytest.mark.parametrize("delta", [-5.0, -10.0, -1.0])
def test_zoom_out_increases_distance_and_stays_in_bounds(camera, delta):
    initial = camera.distance
    camera.zoom(delta)
    assert camera.distance > initial
    assert camera.zoom_min <= camera.distance <= camera.zoom_max


def test_zoom_formula_is_exponential_scaling(camera):
    camera.distance = 15.0
    camera.zoom(5.0)
    # new_distance = distance * (1.0 - 0.1 * delta)
    assert camera.distance == pytest.approx(15.0 * (1.0 - 0.1 * 5.0))


def test_zoom_in_clamps_to_minimum(camera):
    camera.distance = 15.0
    for _ in range(50):
        camera.zoom(1.0)
    assert camera.distance == camera.zoom_min


def test_zoom_out_clamps_to_maximum(camera):
    camera.distance = 15.0
    for _ in range(50):
        camera.zoom(-1.0)
    assert camera.distance == camera.zoom_max


def test_zoom_updates_camera_position(camera):
    before = camera.position.copy()
    camera.zoom(5.0)
    assert not np.allclose(before, camera.position)


def test_orbit_changes_azimuth_and_elevation(camera):
    camera.orbit(delta_azimuth=10.0, delta_elevation=5.0)
    assert camera.azimuth == pytest.approx(_DEFAULT_AZIMUTH + 10.0)
    assert camera.elevation == pytest.approx(_DEFAULT_ELEVATION + 5.0)


def test_orbit_clamps_elevation_to_plus_minus_89(camera):
    camera.orbit(delta_azimuth=0.0, delta_elevation=1000.0)
    assert camera.elevation == 89.0
    camera.orbit(delta_azimuth=0.0, delta_elevation=-1000.0)
    assert camera.elevation == -89.0


def test_orbit_is_noop_when_locked(camera):
    camera.apply_preset("top")
    azimuth_before = camera.azimuth
    elevation_before = camera.elevation
    camera.orbit(delta_azimuth=30.0, delta_elevation=30.0)
    assert camera.azimuth == azimuth_before
    assert camera.elevation == elevation_before


def test_pan_moves_target(camera):
    target_before = camera.target.copy()
    camera.pan(1.0, 1.0)
    assert not np.allclose(target_before, camera.target)


@pytest.mark.parametrize("preset,expected_dir,expected_up", [
    ("top", [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
    ("front", [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    ("right", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
])
def test_apply_ortho_preset_locks_view_direction(camera, preset, expected_dir, expected_up):
    camera.apply_preset(preset)
    assert camera.projection == "orthographic"
    assert camera.locked is True
    assert np.allclose(camera._view_dir, expected_dir)
    assert np.allclose(camera.up, expected_up)


def test_apply_preset_perspective_resets_to_default(camera):
    camera.apply_preset("top")
    camera.apply_preset("perspective")
    assert camera.projection == "perspective"
    assert camera.locked is False
    assert camera.distance == _DEFAULT_DISTANCE
    assert camera.azimuth == _DEFAULT_AZIMUTH


def test_apply_preset_unknown_name_falls_back_to_perspective(camera):
    camera.apply_preset("bogus")
    assert camera.projection == "perspective"
    assert camera.locked is False


def test_apply_preset_is_case_insensitive(camera):
    camera.apply_preset("TOP")
    assert camera.projection == "orthographic"
    assert camera.locked is True


def test_ortho_half_height_scales_with_distance(camera):
    camera.distance = 10.0
    assert camera.ortho_half_height() == pytest.approx(5.0)
    camera.distance = 0.0
    assert camera.ortho_half_height() == pytest.approx(0.01)


def test_reset_restores_defaults_after_mutation(camera):
    camera.zoom(5.0)
    camera.orbit(20.0, 10.0)
    camera.pan(1.0, 1.0)
    camera.apply_preset("top")
    camera.reset()
    assert camera.distance == _DEFAULT_DISTANCE
    assert camera.azimuth == _DEFAULT_AZIMUTH
    assert camera.elevation == _DEFAULT_ELEVATION
    assert np.allclose(camera.target, _DEFAULT_TARGET)
    assert camera.projection == "perspective"
    assert camera.locked is False


def test_view_matrix_is_4x4_and_orthonormal_rows(camera):
    view = camera.get_view_matrix()
    assert view.shape == (4, 4)
    rot = view[:3, :3]
    # Rows of a valid view matrix's rotation block should be unit length.
    norms = np.linalg.norm(rot, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_projection_matrix_is_4x4(camera):
    proj = camera.get_projection_matrix()
    assert proj.shape == (4, 4)
    assert proj[3, 2] == -1.0


def test_focus_on_starts_animation_towards_target(camera):
    target = np.array([5.0, 5.0, 5.0])
    camera.focus_on(target, radius=2.0, duration=0.5)
    assert camera._anim_active is True
    assert np.allclose(camera._anim_end_target, target)
    assert camera.zoom_min <= camera._anim_end_distance <= camera.zoom_max


def test_tick_focus_animation_progresses_then_completes(camera):
    camera.focus_on(np.array([5.0, 5.0, 5.0]), radius=2.0, duration=0.5)
    still_running = camera.tick_focus_animation(0.1)
    assert still_running is True
    assert camera._anim_active is True

    finished = camera.tick_focus_animation(10.0)  # overshoot duration
    assert finished is False
    assert camera._anim_active is False
    assert np.allclose(camera.target, camera._anim_end_target)


def test_tick_focus_animation_inactive_returns_false(camera):
    assert camera.tick_focus_animation(0.1) is False


def test_cancel_focus_animation_stops_without_snapping(camera):
    camera.focus_on(np.array([5.0, 5.0, 5.0]), radius=2.0, duration=10.0)
    camera.tick_focus_animation(0.1)
    camera.cancel_focus_animation()
    assert camera._anim_active is False
    # Target stays at the partially-interpolated position, not the end target.
    assert not np.allclose(camera.target, camera._anim_end_target)
