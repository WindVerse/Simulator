#!/usr/bin/env python3
"""
Test script to verify grid expansion and snapping behavior.
"""
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from renderer.scene import Scene
from wind_data.wind_field import WindField

def test_grid_configuration():
    """Test that grid is properly configured for ±50m coverage."""
    print("=" * 60)
    print("TEST 1: Grid Configuration")
    print("=" * 60)
    
    # Create scene
    wind_field = WindField()
    scene = Scene(wind_field)
    
    # Verify grid size
    print(f"Grid size: {scene.grid_size}")
    print(f"Grid spacing: {scene.grid_spacing}")
    print(f"Grid center: {scene.grid_center}")
    
    half_size = scene.grid_size // 2
    max_coord = half_size * scene.grid_spacing
    
    print(f"\nGrid coverage:")
    print(f"  X range: {-max_coord} to {max_coord} meters")
    print(f"  Z range: {-max_coord} to {max_coord} meters")
    print(f"  Y: 0 (ground level)")
    
    # Expected: ±50 meters
    expected_max = 50.0
    if max_coord == expected_max:
        print(f"\n✓ PASS: Grid covers ±{expected_max}m as expected")
        return True
    else:
        print(f"\n✗ FAIL: Grid covers ±{max_coord}m, expected ±{expected_max}m")
        return False

def test_snapping_logic():
    """Test that snapping works correctly with various inputs."""
    print("\n" + "=" * 60)
    print("TEST 2: Snapping Logic")
    print("=" * 60)
    
    wind_field = WindField()
    scene = Scene(wind_field)
    
    pole_z = scene.flag_pole_height

    test_cases = [
        # (input_pos, expected_snapped, description)
        (np.array([1.5, 0.5, 2.3]), np.array([1.5, 0.5, pole_z]), "Already a cell center (1.5, 0.5)"),
        (np.array([1.4, 1.0, 2.4]), np.array([1.5, 1.5, pole_z]), "Snap 1.4→1.5, 1.0→1.5 (nearest cell center)"),
        (np.array([0.0, 5.0, 0.0]), np.array([0.5, 5.5, pole_z]), "Snap to cell center (0.5, 5.5)"),
        (np.array([-1.5, 2.0, -2.3]), np.array([-1.5, 2.5, pole_z]), "Snap -1.5→-1.5, 2.0→2.5"),
        (np.array([25.7, 0.0, -30.6]), np.array([25.5, 0.5, pole_z]), "Large coords"),
    ]
    
    all_pass = True
    for input_pos, expected, description in test_cases:
        snapped = scene.snap_to_grid(input_pos)
        match = np.allclose(snapped, expected, atol=1e-6)
        
        status = "✓ PASS" if match else "✗ FAIL"
        print(f"\n{status}: {description}")
        print(f"  Input:    {input_pos}")
        print(f"  Snapped:  {snapped}")
        print(f"  Expected: {expected}")
        
        if not match:
            all_pass = False
    
    return all_pass

def test_grid_points():
    """Test that grid points are correctly calculated."""
    print("\n" + "=" * 60)
    print("TEST 3: Grid Points Generation")
    print("=" * 60)
    
    wind_field = WindField()
    scene = Scene(wind_field)
    
    grid_points = scene.get_grid_points()
    
    print(f"Grid points shape: {grid_points.shape}")
    print(f"Expected: (10201, 3) for 101x101=10201 points (flattened)")
    
    # Check dimensions: 101x101 = 10201 points flattened to (10201, 3)
    expected_shape = (10201, 3)
    if grid_points.shape == expected_shape:
        print(f"✓ PASS: Grid points shape is correct")
    else:
        print(f"✗ FAIL: Grid points shape {grid_points.shape} != {expected_shape}")
        return False
    
    # Check range
    x_vals = np.unique(grid_points[:, 0])
    z_vals = np.unique(grid_points[:, 2])
    
    print(f"\nX range: {x_vals.min()} to {x_vals.max()}")
    print(f"Z range: {z_vals.min()} to {z_vals.max()}")
    
    if x_vals.min() >= -50 and x_vals.max() <= 50:
        print("✓ PASS: X range is within ±50m")
    else:
        print("✗ FAIL: X range is not within ±50m")
        return False
    
    if z_vals.min() >= -50 and z_vals.max() <= 50:
        print("✓ PASS: Z range is within ±50m")
    else:
        print("✗ FAIL: Z range is not within ±50m")
        return False
    
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GRID SNAPPING VERIFICATION TEST SUITE")
    print("=" * 60)
    
    results = []
    try:
        results.append(("Grid Configuration", test_grid_configuration()))
        results.append(("Snapping Logic", test_snapping_logic()))
        results.append(("Grid Points Generation", test_grid_points()))
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
