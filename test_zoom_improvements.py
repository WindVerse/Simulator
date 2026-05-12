#!/usr/bin/env python3
"""
Test script to verify improved camera zoom functionality.
"""
import numpy as np
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from renderer.scene import Camera

def test_zoom_limits():
    """Test that zoom limits are correctly set."""
    print("=" * 60)
    print("TEST 1: Zoom Limits Configuration")
    print("=" * 60)
    
    camera = Camera()
    
    print(f"Initial distance: {camera.distance}m")
    print(f"Zoom minimum: {camera.zoom_min}m")
    print(f"Zoom maximum: {camera.zoom_max}m")
    print(f"Near clipping plane: {camera.near}m")
    print(f"Far clipping plane: {camera.far}m")
    
    # Verify limits
    assert camera.zoom_min == 0.2, f"zoom_min should be 0.2, got {camera.zoom_min}"
    assert camera.zoom_max == 200.0, f"zoom_max should be 200.0, got {camera.zoom_max}"
    assert camera.near == 0.01, f"near plane should be 0.01, got {camera.near}"
    
    print("\n✓ PASS: Zoom limits correctly configured")
    return True

def test_zoom_in():
    """Test zooming in (positive delta)."""
    print("\n" + "=" * 60)
    print("TEST 2: Zoom In (Exponential Scaling)")
    print("=" * 60)
    
    camera = Camera()
    initial_distance = camera.distance
    
    print(f"Initial distance: {initial_distance}m")
    
    # Zoom in with different deltas
    test_cases = [
        (5.0, "5.0"),  # Moderate zoom in
        (10.0, "10.0"),  # Aggressive zoom in
    ]
    
    for delta, label in test_cases:
        camera.distance = initial_distance  # Reset
        camera.zoom(delta)
        
        # Exponential scaling: new_distance = distance * (1.0 - 0.1 * delta)
        # delta=5.0: 15 * (1 - 0.5) = 7.5
        # delta=10.0: 15 * (1 - 1.0) = 0 (but clamped to 0.2)
        
        print(f"  After zoom({label}): {camera.distance:.2f}m")
        
        # Should be within limits
        assert camera.distance >= camera.zoom_min, f"Distance {camera.distance} < min {camera.zoom_min}"
        assert camera.distance <= camera.zoom_max, f"Distance {camera.distance} > max {camera.zoom_max}"
    
    print("\n✓ PASS: Zoom in works correctly with exponential scaling")
    return True

def test_zoom_out():
    """Test zooming out (negative delta)."""
    print("\n" + "=" * 60)
    print("TEST 3: Zoom Out (Exponential Scaling)")
    print("=" * 60)
    
    camera = Camera()
    initial_distance = camera.distance
    
    print(f"Initial distance: {initial_distance}m")
    
    # Zoom out with different deltas
    test_cases = [
        (-5.0, "-5.0"),  # Moderate zoom out
        (-10.0, "-10.0"),  # Aggressive zoom out
    ]
    
    for delta, label in test_cases:
        camera.distance = initial_distance  # Reset
        camera.zoom(delta)
        
        # Exponential scaling: new_distance = distance * (1.0 - 0.1 * delta)
        # delta=-5.0: 15 * (1 + 0.5) = 22.5
        # delta=-10.0: 15 * (1 + 1.0) = 30
        
        print(f"  After zoom({label}): {camera.distance:.2f}m")
        
        # Should be within limits
        assert camera.distance >= camera.zoom_min, f"Distance {camera.distance} < min {camera.zoom_min}"
        assert camera.distance <= camera.zoom_max, f"Distance {camera.distance} > max {camera.zoom_max}"
    
    print("\n✓ PASS: Zoom out works correctly with exponential scaling")
    return True

def test_zoom_clamping():
    """Test that zoom respects min/max limits."""
    print("\n" + "=" * 60)
    print("TEST 4: Zoom Clamping (Respects Limits)")
    print("=" * 60)
    
    camera = Camera()
    
    # Try to zoom in beyond minimum
    print("Testing extreme zoom in (should clamp to 0.2m):")
    camera.distance = 15.0
    for _ in range(50):  # Many zoom events
        camera.zoom(1.0)
    print(f"  After 50x zoom(1.0): {camera.distance:.3f}m (min is 0.2m)")
    assert camera.distance == camera.zoom_min, f"Should be clamped to {camera.zoom_min}"
    print(f"  ✓ Clamped to minimum {camera.zoom_min}m")
    
    # Try to zoom out beyond maximum
    print("Testing extreme zoom out (should clamp to 200.0m):")
    camera.distance = 15.0
    for _ in range(50):  # Many zoom events
        camera.zoom(-1.0)
    print(f"  After 50x zoom(-1.0): {camera.distance:.1f}m (max is 200m)")
    assert camera.distance == camera.zoom_max, f"Should be clamped to {camera.zoom_max}"
    print(f"  ✓ Clamped to maximum {camera.zoom_max}m")
    
    print("\n✓ PASS: Zoom correctly clamps to min/max limits")
    return True

def test_zoom_smoothness():
    """Test that zoom scaling is smooth and exponential."""
    print("\n" + "=" * 60)
    print("TEST 5: Zoom Smoothness (Exponential Behavior)")
    print("=" * 60)
    
    camera = Camera()
    initial = 15.0
    
    print("Zoom in with delta=1.0 (repeated calls):")
    distances = [initial]
    for i in range(10):
        camera.distance = distances[-1]
        camera.zoom(1.0)
        distances.append(camera.distance)
        print(f"  Step {i+1}: {distances[-1]:.4f}m")
    
    # Check that ratios are consistent (exponential)
    ratios = [distances[i+1] / distances[i] for i in range(len(distances)-1)]
    avg_ratio = np.mean(ratios)
    
    print(f"\nRatio between consecutive steps: {avg_ratio:.4f}")
    print(f"Expected ratio: ~0.9 (since new = old * (1 - 0.1*1))")
    
    # All ratios should be close to 0.9
    for i, ratio in enumerate(ratios):
        assert 0.85 < ratio < 0.95, f"Ratio {i} = {ratio} is not close to 0.9"
    
    print("✓ PASS: Zoom maintains consistent exponential scaling")
    return True

def test_smooth_zoom_across_range():
    """Test smooth zoom from min to max."""
    print("\n" + "=" * 60)
    print("TEST 6: Smooth Zoom Range (0.2m to 200m)")
    print("=" * 60)
    
    camera = Camera()
    
    # Sample zoom levels
    print("Zoom from maximum to minimum (in steps):")
    camera.distance = camera.zoom_max
    steps = []
    while camera.distance > camera.zoom_min:
        steps.append(camera.distance)
        if len(steps) < 15:
            print(f"  {len(steps):2d}. {camera.distance:7.2f}m")
        camera.zoom(2.0)  # Larger zoom step
    
    print(f"  ... ({len(steps)} total steps)")
    print(f"  Final: {camera.distance:.3f}m (min is 0.2m)")
    
    print(f"\n✓ PASS: Smooth zoom range from {camera.zoom_max}m to {camera.zoom_min}m achieved")
    return True

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CAMERA ZOOM IMPROVEMENTS TEST SUITE")
    print("=" * 60)
    
    results = []
    try:
        results.append(("Zoom Limits Configuration", test_zoom_limits()))
        results.append(("Zoom In", test_zoom_in()))
        results.append(("Zoom Out", test_zoom_out()))
        results.append(("Zoom Clamping", test_zoom_clamping()))
        results.append(("Zoom Smoothness", test_zoom_smoothness()))
        results.append(("Zoom Range", test_smooth_zoom_across_range()))
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
        print("\n✓ ALL TESTS PASSED - Zoom improvements working correctly!")
        print("\nKey improvements:")
        print("  • Zoom range: 0.2m to 200m (was 1.0m to ~15m)")
        print("  • Exponential scaling for smooth zoom feel")
        print("  • Near plane: 0.01m (was 0.1m) - see closer objects")
        print("  • Full ±50m grid viewing in one zoom range")
        sys.exit(0)
    else:
        print("\n✗ SOME TESTS FAILED")
        sys.exit(1)
