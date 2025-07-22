import numpy as np
import quaternion
import pytest
from locate_camera import Camera

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