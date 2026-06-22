"""Unit tests for wind_data.wind_field.WindField."""
import numpy as np
import pytest

from wind_data.wind_field import WindField


@pytest.fixture
def small_field():
    """A tiny (2,2,2) field, 1 time step, with ux = x-grid-index everywhere."""
    field = WindField(grid_size=(2, 2, 2), time_steps=1, grid_spacing=1.0, origin=(0.0, 0.0, 0.0))
    field.data[:] = 0.0
    # data axes: (component, z, y, x, time); set ux to vary only with x.
    field.data[0, :, :, 0, 0] = 0.0
    field.data[0, :, :, 1, 0] = 1.0
    return field


def test_default_constructor_shape_and_coords():
    field = WindField(grid_size=(4, 3, 2), time_steps=5, grid_spacing=2.0, origin=(1.0, 0.0, 0.0))
    assert field.data.shape == (3, 2, 3, 4, 5)
    assert np.allclose(field.x_coords, [1.0, 3.0, 5.0, 7.0])
    assert np.allclose(field.y_coords, [0.0, 2.0, 4.0])
    assert field.time_steps == 5
    assert field.current_time == 0


def test_default_wind_pattern_is_nonzero():
    field = WindField()
    assert np.any(field.data != 0.0)


# ---------------------------------------------------------------------------
# set_wind_data
# ---------------------------------------------------------------------------

def test_set_wind_data_updates_shape_and_coords():
    field = WindField()
    data = np.random.rand(3, 2, 3, 4, 5).astype(np.float32)
    field.set_wind_data(data, x_coords=[0, 1, 2, 3], y_coords=[0, 1, 2], z_coords=[0, 1], time_coords=[0, 1, 2, 3, 4])
    assert field.data.shape == data.shape
    assert field.grid_size == (4, 3, 2)
    assert field.time_steps == 5
    assert field.current_time == 0


def test_set_wind_data_invalidates_grid_points_cache():
    field = WindField()
    field.get_grid_points()
    assert field._grid_points_cache is not None
    field.set_wind_data(
        np.zeros((3, 1, 1, 1, 1), dtype=np.float32),
        x_coords=[0], y_coords=[0], z_coords=[0], time_coords=[0],
    )
    assert field._grid_points_cache is None


# ---------------------------------------------------------------------------
# get_velocity_at_grid
# ---------------------------------------------------------------------------

def test_get_velocity_at_grid_returns_exact_component(small_field):
    v0 = small_field.get_velocity_at_grid(0, 0, 0, time=0)
    v1 = small_field.get_velocity_at_grid(1, 0, 0, time=0)
    assert v0[0] == 0.0
    assert v1[0] == 1.0


def test_get_velocity_at_grid_clips_out_of_range_indices(small_field):
    v_neg = small_field.get_velocity_at_grid(-5, 0, 0, time=0)
    v_over = small_field.get_velocity_at_grid(99, 0, 0, time=0)
    assert v_neg[0] == 0.0   # clipped to index 0
    assert v_over[0] == 1.0  # clipped to index 1 (grid_size[0]-1)


def test_get_velocity_at_grid_wraps_time(small_field):
    # Only 1 time step; time=5 should wrap to 0 via modulo.
    v = small_field.get_velocity_at_grid(0, 0, 0, time=5)
    assert v[0] == 0.0


def test_get_velocity_at_grid_uses_current_time_when_none(small_field):
    small_field.current_time = 0
    v = small_field.get_velocity_at_grid(1, 0, 0, time=None)
    assert v[0] == 1.0


# ---------------------------------------------------------------------------
# get_velocity_at_position (trilinear interpolation)
# ---------------------------------------------------------------------------

def test_interpolation_at_grid_points_matches_exact_values(small_field):
    v0 = small_field.get_velocity_at_position(np.array([0.0, 0.0, 0.0]))
    v1 = small_field.get_velocity_at_position(np.array([1.0, 0.0, 0.0]))
    assert v0[0] == pytest.approx(0.0)
    assert v1[0] == pytest.approx(1.0)


def test_interpolation_midpoint_is_linear(small_field):
    v_mid = small_field.get_velocity_at_position(np.array([0.5, 0.0, 0.0]))
    assert v_mid[0] == pytest.approx(0.5)


def test_interpolation_clamps_outside_bounds(small_field):
    v_below = small_field.get_velocity_at_position(np.array([-10.0, 0.0, 0.0]))
    v_above = small_field.get_velocity_at_position(np.array([10.0, 0.0, 0.0]))
    assert v_below[0] == pytest.approx(0.0)
    assert v_above[0] == pytest.approx(1.0)


def test_uniform_wind_interpolates_to_same_value_everywhere():
    field = WindField(grid_size=(3, 3, 3), time_steps=1)
    field.set_uniform_wind((2.0, -1.0, 0.5))
    for pos in [[0.0, 0.0, 0.0], [1.4, 0.6, 1.9], [2.0, 2.0, 2.0]]:
        v = field.get_velocity_at_position(np.array(pos))
        assert np.allclose(v, [2.0, -1.0, 0.5])


# ---------------------------------------------------------------------------
# get_velocity_at_8_octants
# ---------------------------------------------------------------------------

def test_octants_shape_is_8_by_3():
    field = WindField(grid_size=(3, 3, 3), time_steps=1)
    field.set_uniform_wind((1.0, 0.0, 0.0))
    velocities = field.get_velocity_at_8_octants(np.array([1.0, 1.0, 1.0]), cube_size=1.0)
    assert velocities.shape == (8, 3)


def test_octants_match_uniform_wind():
    field = WindField(grid_size=(3, 3, 3), time_steps=1)
    field.set_uniform_wind((3.0, 4.0, 5.0))
    velocities = field.get_velocity_at_8_octants(np.array([1.0, 1.0, 1.0]), cube_size=1.0)
    assert np.allclose(velocities, np.tile([3.0, 4.0, 5.0], (8, 1)))


# ---------------------------------------------------------------------------
# Time control
# ---------------------------------------------------------------------------

def test_advance_time_wraps_with_modulo():
    field = WindField(grid_size=(2, 2, 2), time_steps=3)
    field.current_time = 0
    field.advance_time(2)
    assert field.current_time == 2
    field.advance_time(2)
    assert field.current_time == 1  # (2 + 2) % 3


def test_reset_time_sets_to_zero():
    field = WindField(grid_size=(2, 2, 2), time_steps=3)
    field.current_time = 2
    field.reset_time()
    assert field.current_time == 0


# ---------------------------------------------------------------------------
# get_grid_points / get_current_velocities
# ---------------------------------------------------------------------------

def test_get_grid_points_shape_and_caching():
    field = WindField(grid_size=(2, 3, 4), time_steps=1)
    points = field.get_grid_points()
    assert points.shape == (2 * 3 * 4, 3)
    assert field.get_grid_points() is points  # cached


def test_get_current_velocities_shape_matches_grid_points():
    field = WindField(grid_size=(2, 3, 4), time_steps=2)
    velocities = field.get_current_velocities()
    assert velocities.shape == (2 * 3 * 4, 3)


# ---------------------------------------------------------------------------
# get_bounds
# ---------------------------------------------------------------------------

def test_get_bounds_returns_coordinate_extremes():
    field = WindField(grid_size=(4, 3, 2), grid_spacing=2.0, origin=(0.0, 0.0, 0.0))
    min_corner, max_corner = field.get_bounds()
    assert np.allclose(min_corner, [0.0, 0.0, 0.0])
    assert np.allclose(max_corner, [6.0, 4.0, 2.0])


# ---------------------------------------------------------------------------
# set_uniform_wind / add_turbulence
# ---------------------------------------------------------------------------

def test_set_uniform_wind_fills_every_cell():
    field = WindField(grid_size=(2, 2, 2), time_steps=2)
    field.set_uniform_wind((1.0, 2.0, 3.0))
    assert np.all(field.data[0] == 1.0)
    assert np.all(field.data[1] == 2.0)
    assert np.all(field.data[2] == 3.0)


def test_add_turbulence_perturbs_data():
    field = WindField(grid_size=(2, 2, 2), time_steps=1)
    field.set_uniform_wind((1.0, 1.0, 1.0))
    before = field.data.copy()
    field.add_turbulence(intensity=1.0)
    assert not np.array_equal(before, field.data)


# ---------------------------------------------------------------------------
# save_to_file / load_from_file round trip
# ---------------------------------------------------------------------------

def test_save_and_load_file_round_trip(tmp_path):
    field = WindField(grid_size=(2, 3, 4), time_steps=2, grid_spacing=1.5, origin=(1.0, 2.0, 3.0))
    field.set_uniform_wind((1.0, -2.0, 0.5))
    filepath = tmp_path / "wind.npz"
    field.save_to_file(str(filepath))

    loaded = WindField()
    loaded.load_from_file(str(filepath))

    assert loaded.data.shape == field.data.shape
    assert np.allclose(loaded.data, field.data)
    assert np.allclose(loaded.x_coords, field.x_coords)
    assert np.allclose(loaded.y_coords, field.y_coords)
    assert np.allclose(loaded.z_coords, field.z_coords)
    assert loaded.current_time == 0


def test_load_from_file_legacy_layout_is_transposed(tmp_path):
    # Legacy layout: (time, x, y, z, 3)
    legacy = np.zeros((2, 3, 2, 1, 3), dtype=np.float32)
    legacy[0, 1, 0, 0, 0] = 7.0  # time=0, x=1, y=0, z=0, component=u
    filepath = tmp_path / "legacy.npy"
    np.save(filepath, legacy)

    field = WindField()
    field.load_from_file(str(filepath))

    assert field.data.shape == (3, 1, 2, 3, 2)
    assert field.data[0, 0, 0, 1, 0] == 7.0


def test_load_from_file_rejects_non_5d_array(tmp_path):
    filepath = tmp_path / "bad.npy"
    np.save(filepath, np.zeros((3, 3)))
    field = WindField()
    with pytest.raises(ValueError):
        field.load_from_file(str(filepath))
