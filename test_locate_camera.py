import numpy as np
import quaternion
import pytest
import tempfile
import os
import sys
import matplotlib.pyplot as plt
import locate_camera as lc

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
    cam = lc.Camera(fov=np.pi/2, res=np.array([100, 100]))
    # Center of the image should map to (0, 0)
    pt = np.array([50, 50])
    result = cam.raw_photo_to_ideal_pinhole(pt)
    np.testing.assert_allclose(result, [0, 0], atol=1e-6)

def test_raw_photo_to_ideal_pinhole_corner():
    cam = lc.Camera(fov=np.pi/2, res=np.array([100, 100]))
    # Top-left corner
    pt = np.array([0, 0])
    result = cam.raw_photo_to_ideal_pinhole(pt)
    np.testing.assert_allclose(result, [-0.5, -0.5], atol=1e-6)

def test_poi_to_pinhole_projection_in_view():
    cam = lc.Camera(fov=np.pi/2, res=np.array([100, 100]))
    # Camera at origin, looking along x-axis
    r = np.array([0, 0, 0])
    q = quaternion.from_rotation_vector([0, 0, 0])  # No rotation
    # Point directly in front of camera
    pt = np.array([-1, 0, 0])
    result = cam.poi_to_pinhole_projection(pt, r, q)
    assert result is not None
    np.testing.assert_allclose(result, [0, 0], atol=1e-6)

def test_poi_to_pinhole_projection_out_of_view():
    cam = lc.Camera(fov=np.pi/2, res=np.array([100, 100]))
    r = np.array([0, 0, 0])
    q = quaternion.from_rotation_vector([0, 0, 0])
    # Point behind the camera
    pt = np.array([1, 0, 0])
    result = cam.poi_to_pinhole_projection(pt, r, q)
    assert result is None

def test_poi_to_pinhole_projection_off_axis():
    cam = lc.Camera(fov=np.pi/2, res=np.array([100, 100]))
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
    curve = lc.Curve.from_points(points)
    assert isinstance(curve, lc.Curve)
    assert len(curve.points) == 3
    np.testing.assert_array_equal(curve.points[1], np.array([1, 1]))

def test_curve_to_file_and_from_file():
    points = [np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([2, 0, 2])]
    curve = lc.Curve.from_points(points)
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, mode='w+') as tmp:
        curve.to_file(tmp.name)
        tmp.close()
        loaded_curve = lc.Curve.from_file(tmp.name)
        assert isinstance(loaded_curve, lc.Curve)
        assert len(loaded_curve.points) == 3
        np.testing.assert_array_almost_equal(loaded_curve.points[2], np.array([2, 0, 2]))
    os.remove(tmp.name)

def test_curve_get_point_along_curve_linear():
    points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 0])]
    curve = lc.Curve.from_points(points)
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
    curve = lc.Curve.from_points(points)
    pt0 = curve.get_point_along_curve(0, k=2)
    np.testing.assert_allclose(pt0, [0, 0], atol=1e-6)
    pt1 = curve.get_point_along_curve(1, k=2)
    np.testing.assert_allclose(pt1, [2, 4], atol=1e-6)
    pt_half = curve.get_point_along_curve(0.5, k=2)
    if VISUALIZE:
        plot_curve_and_point(points, curve, 0.5, pt_half, k=2)
    np.testing.assert_allclose(pt_half, [1.447174, 1.723649], atol=1e-3)  # Allow some tolerance for interpolation

# ============================================================================
# DIMENSION TESTS - Verify input/output dimensions match function specifications
# ============================================================================

class TestCameraDimensions:
    """Test that Camera class methods have correct input/output dimensions."""
    
    def test_camera_init_dimensions(self):
        """Test Camera.__init__ input dimensions."""
        # Valid inputs
        fov = np.pi/2  # float
        res = np.array([1920, 1080])  # shape (2,)
        camera = lc.Camera(fov, res)
        assert camera.fov == fov
        assert np.array_equal(camera.res, res)
        
        # Test invalid resolution dimension
        with pytest.raises(IndexError):
            lc.Camera(fov, np.array([1920]))  # shape (1,) - should fail
    
    def test_raw_photo_to_ideal_pinhole_dimensions(self):
        """Test Camera.raw_photo_to_ideal_pinhole input/output dimensions."""
        camera = lc.Camera(np.pi/2, np.array([1920, 1080]))
        
        # Valid input: shape (2,)
        point = np.array([960, 540])  # shape (2,)
        result = camera.raw_photo_to_ideal_pinhole(point)
        
        # Check output shape
        assert result.shape == (2,)
        assert isinstance(result, np.ndarray)
        
        # Test invalid input dimensions
        with pytest.raises(IndexError):
            camera.raw_photo_to_ideal_pinhole(np.array([960]))  # shape (1,)
        with pytest.raises(IndexError):
            camera.raw_photo_to_ideal_pinhole(np.array([960, 540, 0]))  # shape (3,)
    
    def test_poi_to_pinhole_projection_dimensions(self):
        """Test Camera.poi_to_pinhole_projection input/output dimensions."""
        camera = lc.Camera(np.pi/2, np.array([1920, 1080]))
        
        # Valid inputs
        point = np.array([1.0, 2.0, 3.0])  # shape (3,)
        r = np.array([0.0, 0.0, 0.0])      # shape (3,)
        q = quaternion.from_rotation_matrix(np.eye(3))  # quaternion object
        
        result = camera.poi_to_pinhole_projection(point, r, q)
        
        # Check output shape
        assert result.shape == (2,)
        assert isinstance(result, np.ndarray)
        
        # Test invalid input dimensions
        with pytest.raises(IndexError):
            camera.poi_to_pinhole_projection(np.array([1.0, 2.0]), r, q)  # point shape (2,)
        with pytest.raises(IndexError):
            camera.poi_to_pinhole_projection(point, np.array([0.0, 0.0]), q)  # r shape (2,)
        with pytest.raises(IndexError):
            camera.poi_to_pinhole_projection(point, r, np.array([1, 0, 0, 0]))  # q not quaternion


class TestCurveDimensions:
    """Test that Curve class methods have correct input/output dimensions."""
    
    def test_curve_init_dimensions(self):
        """Test Curve.__init__ output dimensions."""
        curve = lc.Curve()
        assert isinstance(curve.points, list)
        assert len(curve.points) == 0
    
    def test_from_points_dimensions_2d(self):
        """Test Curve.from_points with 2D points."""
        # Valid 2D points
        points = [
            np.array([1.0, 2.0]),  # shape (2,)
            np.array([3.0, 4.0]),  # shape (2,)
            np.array([5.0, 6.0])   # shape (2,)
        ]
        curve = lc.Curve.from_points(points)
        
        # Check all points have correct shape
        for point in curve.points:
            assert point.shape == (2,)
            assert isinstance(point, np.ndarray)
        
        # Test mixed dimensions (should raise ValueError)
        mixed_points = [
            np.array([1.0, 2.0]),  # shape (2,)
            np.array([3.0, 4.0, 5.0])  # shape (3,)
        ]
        with pytest.raises(ValueError, match="Point 1 has dimension 3, expected 2"):
            lc.Curve.from_points(mixed_points)
    
    def test_from_points_dimensions_3d(self):
        """Test Curve.from_points with 3D points."""
        # Valid 3D points
        points = [
            np.array([1.0, 2.0, 3.0]),  # shape (3,)
            np.array([4.0, 5.0, 6.0]),  # shape (3,)
            np.array([7.0, 8.0, 9.0])   # shape (3,)
        ]
        curve = lc.Curve.from_points(points)
        
        # Check all points have correct shape
        for point in curve.points:
            assert point.shape == (3,)
            assert isinstance(point, np.ndarray)
    
    def test_from_points_empty(self):
        """Test Curve.from_points with empty list."""
        curve = lc.Curve.from_points([])
        assert isinstance(curve.points, list)
        assert len(curve.points) == 0
    
    def test_from_file_dimensions(self):
        """Test Curve.from_file input/output dimensions."""
        # Create temporary file with 2D points
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# Test curve\n")
            f.write("1.0, 2.0\n")
            f.write("3.0, 4.0\n")
            f.write("5.0, 6.0\n")
            temp_file = f.name
        
        try:
            curve = lc.Curve.from_file(temp_file)
            
            # Check all points have correct shape
            for point in curve.points:
                assert point.shape == (2,)
                assert isinstance(point, np.ndarray)
        finally:
            os.unlink(temp_file)
    
    def test_to_file_dimensions(self):
        """Test Curve.to_file input dimensions."""
        curve = lc.Curve.from_points([
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0])
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_file = f.name
        
        try:
            curve.to_file(temp_file)
            # Check file was created and has content
            assert os.path.exists(temp_file)
            with open(temp_file, 'r') as f:
                content = f.read()
                assert "1.0, 2.0" in content
                assert "3.0, 4.0" in content
        finally:
            os.unlink(temp_file)
    
    def test_get_point_along_curve_dimensions_2d(self):
        """Test Curve.get_point_along_curve with 2D points."""
        curve = lc.Curve.from_points([
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 0.0])
        ])
        
        # Test output dimensions
        result = curve.get_point_along_curve(0.5)
        assert result.shape == (2,)
        assert isinstance(result, np.ndarray)
        
        # Test parameter bounds
        result_start = curve.get_point_along_curve(0.0)
        result_end = curve.get_point_along_curve(1.0)
        assert result_start.shape == (2,)
        assert result_end.shape == (2,)
        
        # Test invalid parameter
        with pytest.raises(ValueError):
            curve.get_point_along_curve(1.5)  # > 1.0
        with pytest.raises(ValueError):
            curve.get_point_along_curve(-0.1)  # < 0.0
    
    def test_get_point_along_curve_dimensions_3d(self):
        """Test Curve.get_point_along_curve with 3D points."""
        curve = lc.Curve.from_points([
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 0.0, 2.0])
        ])
        
        # Test output dimensions
        result = curve.get_point_along_curve(0.5)
        assert result.shape == (3,)
        assert isinstance(result, np.ndarray)
    
    def test_get_point_along_curve_insufficient_points(self):
        """Test Curve.get_point_along_curve with insufficient points."""
        # Single point
        curve = lc.Curve.from_points([np.array([1.0, 2.0])])
        with pytest.raises(ValueError):
            curve.get_point_along_curve(0.5)
        
        # No points
        curve = lc.Curve()
        with pytest.raises(ValueError):
            curve.get_point_along_curve(0.5)
    
    def test_center_of_mass_dimensions_2d(self):
        """Test Curve.center_of_mass with 2D points."""
        curve = lc.Curve.from_points([
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([0.0, 1.0])
        ])
        
        result = curve.center_of_mass()
        assert result.shape == (2,)
        assert isinstance(result, np.ndarray)
    
    def test_center_of_mass_dimensions_3d(self):
        """Test Curve.center_of_mass with 3D points."""
        curve = lc.Curve.from_points([
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 1.0])
        ])
        
        result = curve.center_of_mass()
        assert result.shape == (3,)
        assert isinstance(result, np.ndarray)
    
    def test_center_of_mass_single_point(self):
        """Test Curve.center_of_mass with single point."""
        curve = lc.Curve.from_points([np.array([1.0, 2.0, 3.0])])
        result = curve.center_of_mass()
        assert result.shape == (3,)
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))
    
    def test_center_of_mass_empty(self):
        """Test Curve.center_of_mass with empty curve."""
        curve = lc.Curve()
        with pytest.raises(ValueError):
            curve.center_of_mass()
    
    def test_project_to_camera_dimensions(self):
        """Test Curve.project_to_camera input/output dimensions."""
        # Create 3D curve
        curve = lc.Curve.from_points([
            np.array([1.0, 2.0, 3.0]),
            np.array([4.0, 5.0, 6.0]),
            np.array([7.0, 8.0, 9.0])
        ])
        
        # Create camera
        camera = lc.Camera(np.pi/2, np.array([1920, 1080]))
        r = np.array([0.0, 0.0, 0.0])  # shape (3,)
        q = quaternion.from_rotation_matrix(np.eye(3))  # quaternion
        
        # Test projection
        projected_curve = curve.project_to_camera(camera, r, q)
        
        # Check output dimensions
        for point in projected_curve.points:
            assert point.shape == (2,)  # Should be 2D after projection
            assert isinstance(point, np.ndarray)
    
    def test_project_to_camera_invalid_dimensions(self):
        """Test Curve.project_to_camera with invalid input dimensions."""
        # Create 2D curve (should fail)
        curve = lc.Curve.from_points([
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0])
        ])
        
        camera = lc.Camera(np.pi/2, np.array([1920, 1080]))
        r = np.array([0.0, 0.0, 0.0])
        q = quaternion.from_rotation_matrix(np.eye(3))
        
        with pytest.raises(ValueError, match="Points must be 3D"):
            curve.project_to_camera(camera, r, q)
    
    def test_plot_dimensions_2d(self):
        """Test Curve.plot with 2D points."""
        curve = lc.Curve.from_points([
            np.array([0.0, 0.0]),
            np.array([1.0, 1.0]),
            np.array([2.0, 0.0])
        ])
        
        # Test that plot doesn't raise dimension errors
        ax = curve.plot(show=False)
        assert ax is not None
    
    def test_plot_dimensions_3d(self):
        """Test Curve.plot with 3D points."""
        curve = lc.Curve.from_points([
            np.array([0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 0.0, 2.0])
        ])
        
        # Test that plot doesn't raise dimension errors
        ax = curve.plot(show=False)
        assert ax is not None
    
    def test_plot_empty_curve(self):
        """Test Curve.plot with empty curve."""
        curve = lc.Curve()
        with pytest.raises(ValueError):
            curve.plot(show=False)
    
    def test_plot_single_point(self):
        """Test Curve.plot with single point."""
        curve = lc.Curve.from_points([np.array([1.0, 2.0])])
        with pytest.raises(ValueError):
            curve.plot(show=False)


class TestMatchFramesDimensions:
    """Test that MatchFrames class methods have correct input/output dimensions."""
    
    def test_auto_initial_r_q_dimensions(self):
        """Test MatchFrames.auto_initial_r_q input/output dimensions."""
        # Create test curves
        geo_curves = [
            lc.Curve.from_points([
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0])
            ]),
            lc.Curve.from_points([
                np.array([7.0, 8.0, 9.0]),
                np.array([10.0, 11.0, 12.0])
            ])
        ]
        
        cam_curves = [
            lc.Curve.from_points([
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4])
            ]),
            lc.Curve.from_points([
                np.array([0.5, 0.6]),
                np.array([0.7, 0.8])
            ])
        ]
        
        camera = lc.Camera(np.pi/2, np.array([1920, 1080]))
        match_frames = lc.MatchFrames(geo_curves, cam_curves, camera)
        
        # Test output dimensions
        r, q = match_frames.auto_initial_r_q()
        
        assert r.shape == (3,)  # Camera position
        assert isinstance(r, np.ndarray)
        assert isinstance(q, quaternion.quaternion)  # Camera orientation
    
    def test_run_unconstrained_dimensions(self):
        """Test MatchFrames.run_unconstrained input/output dimensions."""
        # Create test curves
        geo_curves = [
            lc.Curve.from_points([
                np.array([1.0, 2.0, 3.0]),
                np.array([4.0, 5.0, 6.0])
            ])
        ]
        
        cam_curves = [
            lc.Curve.from_points([
                np.array([0.1, 0.2]),
                np.array([0.3, 0.4])
            ])
        ]
        
        camera = lc.Camera(np.pi/2, np.array([1920, 1080]))
        match_frames = lc.MatchFrames(geo_curves, cam_curves, camera)
        
        # Test output dimensions
        r, q = match_frames.run_unconstrained()
        
        assert r.shape == (3,)  # Camera position
        assert isinstance(r, np.ndarray)
        assert isinstance(q, quaternion.quaternion)  # Camera orientation