import numpy as np
import quaternion
import pytest
from locate_camera import Camera, Curve
import tempfile
import os
import sys
import matplotlib.pyplot as plt

# Check for visualization using environment variable
VISUALIZE = os.environ.get('VISUALIZE', 'false').lower() == 'true'

def plot_curve_and_point(points, curve, parameter, sampled_point, k):
    points = np.array(points)
    t_vals = np.linspace(0, 1, 100)
    curve_points = np.array([curve.get_point_along_curve(t, k=k) for t in t_vals])
    plt.figure()
    if points.shape[1] == 2:
        plt.plot(points[:,0], points[:,1], 'ro-', label='Original Points')
        plt.plot(curve_points[:,0], curve_points[:,1], 'b-', label=f'Interpolated Curve (k={k})')
        plt.plot(sampled_point[0], sampled_point[1], 'gs', label=f'Sampled Point (t={parameter})')
        plt.legend()
        plt.title(f'Curve Interpolation (k={k})')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()
    # For 3D, you could add a 3D plot if needed

def test_raw_photo_to_ideal_pinhole_center():
    cam = Camera(fov=np.pi/2, res=np.array([100, 100]))
    # Center of the image should map to (0, 0)
    pt = np.array([50, 50])
    result = cam.raw_photo_to_ideal_pinhole(pt)
    np.testing.assert_allclose(result, [0, 0], atol=1e-6)

def test_raw_photo_to_ideal_pinhole_corner():
    cam = Camera(fov=np.pi/2, res=np.array([100, 100]))
    # Top-left corner
    pt = np.array([0, 0])
    result = cam.raw_photo_to_ideal_pinhole(pt)
    np.testing.assert_allclose(result, [-0.5, -0.5], atol=1e-6)

def test_poi_to_pinhole_projection_in_view():
    cam = Camera(fov=np.pi/2, res=np.array([100, 100]))
    # Camera at origin, looking along x-axis
    r = np.array([0, 0, 0])
    q = quaternion.from_rotation_vector([0, 0, 0])  # No rotation
    # Point directly in front of camera
    pt = np.array([-1, 0, 0])
    result = cam.poi_to_pinhole_projection(pt, r, q)
    assert result is not None
    np.testing.assert_allclose(result, [0, 0], atol=1e-6)

def test_poi_to_pinhole_projection_out_of_view():
    cam = Camera(fov=np.pi/2, res=np.array([100, 100]))
    r = np.array([0, 0, 0])
    q = quaternion.from_rotation_vector([0, 0, 0])
    # Point behind the camera
    pt = np.array([1, 0, 0])
    result = cam.poi_to_pinhole_projection(pt, r, q)
    assert result is None

def test_poi_to_pinhole_projection_off_axis():
    cam = Camera(fov=np.pi/2, res=np.array([100, 100]))
    poi = np.array([1, 0, 0])
    r_cam = np.array([-1,-1,0])
    theta = 5 * np.pi/4
    result_mag = np.dot(
        (poi-r_cam)/np.linalg.norm(poi-r_cam),
        np.array([1/np.sqrt(2), -1/np.sqrt(2),0])
    )/2

    #orientation 1 - camera pointed towards the origin. Z axes aligned
    rot_mat = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0,0,1]
    ])
    q1 = quaternion.from_rotation_matrix(rot_mat)
    result1 = cam.poi_to_pinhole_projection(poi, r_cam, q1)
    assert result1 is not None
    np.testing.assert_allclose(result1[0], result_mag, atol=1e-8)

    #orientation 2 - camera pointed towards the origin. Z axis aligned with global Y axis (portrait :))
    rot_mat = np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0]
    ]) @ rot_mat
    q2 = quaternion.from_rotation_matrix(rot_mat)
    result2 = cam.poi_to_pinhole_projection(poi, r_cam, q2)
    assert result2 is not None
    np.testing.assert_allclose(result2[1], result_mag, atol=1e-8)

# --- Curve class tests ---
def test_curve_from_points():
    points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 0])]
    curve = Curve.from_points(points)
    assert isinstance(curve, Curve)
    assert len(curve.points) == 3
    np.testing.assert_array_equal(curve.points[1], np.array([1, 1]))

def test_curve_to_file_and_from_file():
    points = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 0, 2])]
    curve = Curve.from_points(points)
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmp:
        curve.to_file(tmp.name)
        tmp.close()
        loaded_curve = Curve.from_file(tmp.name)
        assert isinstance(loaded_curve, Curve)
        assert len(loaded_curve.points) == 3
        np.testing.assert_array_almost_equal(loaded_curve.points[2], np.array([2, 0, 2]))
    os.remove(tmp.name)

def test_curve_get_point_along_curve_linear():
    points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 0])]
    curve = Curve.from_points(points)
    # At parameter=0, should be first point
    pt0 = curve.get_point_along_curve(0, k=1)
    np.testing.assert_allclose(pt0, [0, 0], atol=1e-6)
    # At parameter=1, should be last point
    pt1 = curve.get_point_along_curve(1, k=1)
    np.testing.assert_allclose(pt1, [2, 0], atol=1e-6)
    # At parameter=0.5, should be near the middle (linear interpolation)
    pt_half = curve.get_point_along_curve(0.5, k=1)
    # For a V shape, the middle is at [1, 1]
    np.testing.assert_allclose(pt_half, [1, 1], atol=1e-6)
  
    # At parameter=0.25, should be near the middle of the first leg (linear interpolation)
    pt_half = curve.get_point_along_curve(0.25, k=1)
    # For a V shape, the middle is at [0.5, 0.5]
    np.testing.assert_allclose(pt_half, [0.5, 0.5], atol=1e-6)
    if VISUALIZE:
        plot_curve_and_point(points, curve, 0.25, pt_half, k=1)

def test_curve_get_point_along_curve_quadratic():
    # Parabola: y = x^2, sampled at x = 0, 1, 2
    points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 4])]
    curve = Curve.from_points(points)
    pt0 = curve.get_point_along_curve(0, k=2)
    np.testing.assert_allclose(pt0, [0, 0], atol=1e-6)
    pt1 = curve.get_point_along_curve(1, k=2)
    np.testing.assert_allclose(pt1, [2, 4], atol=1e-6)
    pt_half = curve.get_point_along_curve(0.5, k=2)
    if VISUALIZE:
        plot_curve_and_point(points, curve, 0.5, pt_half, k=2)
    np.testing.assert_allclose(pt_half, [1.447174, 1.723649], atol=1e-3)  # Allow some tolerance for interpolation