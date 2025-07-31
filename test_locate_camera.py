import numpy as np
import quaternion
import pytest
import tempfile
import os
import sys
import matplotlib.pyplot as plt
import locate_camera as lc
from mpl_toolkits.mplot3d import Axes3D
import json

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

def load_lens_profile_from_frame(frame_dir):
    """
    Load lens profile path from a JSON file in the frame directory
    
    Args:
        frame_dir (str): Path to the frame directory
        
    Returns:
        str: Path to the lens profile file
        
    Raises:
        FileNotFoundError: If lens_profile.json doesn't exist
        KeyError: If lens_profile_path key is missing
    """
    lens_profile_file = os.path.join(frame_dir, "lens_profile.json")
    
    if not os.path.exists(lens_profile_file):
        raise FileNotFoundError(f"Lens profile configuration not found: {lens_profile_file}")
    
    with open(lens_profile_file, 'r') as f:
        config = json.load(f)
    
    if 'lens_profile_path' not in config:
        raise KeyError(f"Missing 'lens_profile_path' key in {lens_profile_file}")
    
    lens_profile_path = config['lens_profile_path']
    
    # Check if the lens profile file exists
    if not os.path.exists(lens_profile_path):
        raise FileNotFoundError(f"Lens profile file not found: {lens_profile_path}")
    
    return lens_profile_path

class TestFisheyeCamera:
    def test_fisheye_camera_initialization(self):
        """Test FisheyeCamera initialization with a Gyroflow lens profile."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            cam = lc.FisheyeCamera(profile_path)
            # Check that camera matrix and distortion coefficients are loaded
            assert cam.camera_matrix is not None
            assert cam.distortion_coeffs is not None
            assert cam.size is not None
            # Check that size is a 2D array
            assert cam.size.shape == (2,)
            print(f"Successfully loaded fisheye camera with size: {cam.size}")
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")
        except Exception as e:
            pytest.fail(f"Failed to initialize FisheyeCamera: {e}")

    def test_project_point_in_view(self):
        """Test projecting a point that should be in view."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            cam = lc.FisheyeCamera(profile_path)
            # Camera at origin, looking along z-axis
            r = np.array([0, 0, 0]).astype(np.float64)
            q = quaternion.from_rotation_vector([0, 0, 0])  # No rotation
            # point 10 degrees off center
            pt = np.array([np.sin(np.pi/18), 0, np.cos(np.pi/18)]).astype(np.float64)
            result = cam.project_point(pt, r, q)
            # Should be near center of image
            expected = np.array([
                cam.size[0] / 2 + cam.camera_matrix[0,0] * np.tan(np.pi/18),
                cam.size[1] / 2
            ])
            print(f"Projected result: {result}")
            print(f"Expected (approx): {expected}")
            # comparison not exact due to distortion
            assert abs(result[0] - expected[0]) < 20 # within 20 pixels of expected x
            assert abs(result[1] - expected[1]) < 20 # within 20 pixels of center
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")

    def test_project_point_out_of_view(self):
        """Test projecting a point that should be out of view."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            cam = lc.FisheyeCamera(profile_path)
            # Camera at origin, looking along z-axis
            r = np.array([0, 0, 0]).astype(np.float64)
            q = quaternion.from_rotation_vector([0, 0, 0])  # No rotation
            # point 10 degrees off center, behind camera
            pt = np.array([np.sin(np.pi/18), 0, -np.cos(np.pi/18)]).astype(np.float64)
            result = cam.project_point(pt, r, q)
            # Should be near center of image
            expected = np.array([
                cam.size[0] / 2 - cam.camera_matrix[0,0] * np.tan(np.pi/18),
                cam.size[1] / 2
            ])
            print(f"Projected result: {result}")
            print(f"Expected (approx): {expected}")
            # comparison not exact due to distortion
            assert abs(result[0] - expected[0]) < 20 # within 20 pixels of expected x
            assert abs(result[1] - expected[1]) < 20 # within 20 pixels of center
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")

    def test_project_point_off_axis(self):
        """Test projecting a point off the camera axis."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            cam = lc.FisheyeCamera(profile_path)
            print(f"Camera matrix: {cam.camera_matrix}")
            poi = np.array([0, 0, 1]).astype(np.float64)
            r_cam = np.array([0, -1, -1]).astype(np.float64)
            theta = np.pi/4
            z_cam_hat = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])
            y_cam_hat = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])
            # Calculate the angle to the poi from the camera axis
            angle_to_poi = np.arccos(np.dot(z_cam_hat, (poi-r_cam) / np.linalg.norm(poi-r_cam)))
            print(f"angle_to_poi (deg): {np.degrees(angle_to_poi)}")

            # Orientation 1 - camera pointed towards the origin. X axes aligned
            rot_mat = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
            q1 = quaternion.from_rotation_matrix(rot_mat)
            result1 = cam.project_point(poi, r_cam, q1)

            if VISUALIZE:
                fig = plt.figure()
                ax=fig.add_subplot(1,1,1, projection='3d')
                lc.visualize_camera_model(cam, r_cam, q1, ax=ax, axis_length=0.5)
                ax.scatter(poi[0], poi[1], poi[2], color='r', s=100, marker='o', label='POI')
                # Set equal axis scales for 3D plot
                lc.ensure_equal_aspect_3d(ax)
                plt.show()

            expected = np.array([
                cam.size[0] /2, 
                cam.size[1] /2 - cam.camera_matrix[1,1] * np.tan(angle_to_poi)
            ])
            print(f"Projected result1: {result1}")
            print(f"Expected (approx): {expected}")
            #comparison not exact due to distortion
            assert abs(result1[0] - expected[0]) < 20 # within 20 pixels of expected x
            assert abs(result1[1] - expected[1]) < 20 # within 20 pixels of center

            # Orientation 2 - camera pointed towards the origin. Z axis aligned with global Y axis
            rot_mat = np.array([
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1]
            ]) @ rot_mat
            q2 = quaternion.from_rotation_matrix(rot_mat)
            result2 = cam.project_point(poi, r_cam, q2)
            # Convert pixel coordinates to normalized coordinates for comparison
            expected = np.array([
                cam.size[0] / 2 + cam.camera_matrix[0,0] * np.tan(angle_to_poi),
                cam.size[1] / 2
            ])
            print(f"Projected result2: {result2}")
            print(f"Expected (approx): {expected}")
            if VISUALIZE:
                fig = plt.figure()
                ax=fig.add_subplot(1,1,1, projection='3d')
                lc.visualize_camera_model(cam, r_cam, q2, ax=ax, axis_length=0.5)
                ax.scatter(poi[0], poi[1], poi[2], color='r', s=100, marker='o', label='POI')
                lc.ensure_equal_aspect_3d(ax)
                plt.show()
            #comparison not exact due to distortion
            assert abs(result2[0] - expected[0]) < 25 # within 25 pixels of expected x
            assert abs(result2[1] - expected[1]) < 25 # within 25 pixels of center

        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")

    def test_fisheye_camera_attributes(self):
        """Test that FisheyeCamera has the expected attributes."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            cam = lc.FisheyeCamera(profile_path)
            # Check required attributes
            assert hasattr(cam, 'camera_matrix')
            assert hasattr(cam, 'distortion_coeffs')
            assert hasattr(cam, 'size')
            assert hasattr(cam, 'project_point')
            # Check types
            assert isinstance(cam.camera_matrix, np.ndarray)
            assert isinstance(cam.distortion_coeffs, np.ndarray)
            assert isinstance(cam.size, np.ndarray)
            # Check shapes
            assert cam.camera_matrix.shape == (3, 3)
            assert cam.size.shape == (2,)
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")

class CurveTest:
    # --- Curve class tests ---
    def test_curve_from_points(self):
        points = [np.array([0, 0]), np.array([1, 1]), np.array([2, 0])]
        curve = lc.Curve.from_points(points)
        assert isinstance(curve, lc.Curve)
        assert len(curve.points) == 3
        np.testing.assert_array_equal(curve.points[1], np.array([1, 1]))

    def test_curve_to_file_and_from_file(self):
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

    def test_curve_get_point_along_curve_linear(self):
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

    def test_curve_get_point_along_curve_quadratic(self):
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

class TestFisheyeCameraDimensions:
    """Test that FisheyeCamera class methods have correct input/output dimensions."""
    
    def test_fisheye_camera_init_dimensions(self):
        """Test FisheyeCamera.__init__ input dimensions."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            camera = lc.FisheyeCamera(profile_path)
            # Check that required attributes are loaded with correct dimensions
            assert camera.camera_matrix is not None
            assert camera.camera_matrix.shape == (3, 3)
            assert camera.distortion_coeffs is not None
            assert camera.distortion_coeffs.shape == (4,)  # OpenCV fisheye has 4 distortion coeffs
            assert camera.size is not None
            assert camera.size.shape == (2,)
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")
    
    def test_project_point_dimensions(self):
        """Test FisheyeCamera.project_point input/output dimensions."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            camera = lc.FisheyeCamera(profile_path)
            
            # Valid inputs
            point = np.array([1.0, 2.0, 3.0])  # shape (3,)
            r = np.array([0.0, 0.0, 0.0])      # shape (3,)
            q = quaternion.from_rotation_matrix(np.eye(3))  # quaternion object
            
            result = camera.project_point(point, r, q)
            
            # Check output shape
            assert result.shape == (2,)
            assert isinstance(result, np.ndarray)
            
            # Test invalid input dimensions
            with pytest.raises(IndexError):
                camera.project_point(np.array([1.0, 2.0]), r, q)  # point shape (2,)
            with pytest.raises(IndexError):
                camera.project_point(point, np.array([0.0, 0.0]), q)  # r shape (2,)
            with pytest.raises(IndexError):
                camera.project_point(point, r, np.array([1, 0, 0, 0]))  # q not quaternion
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")
    
    def test_load_lens_profile_dimensions(self):
        """Test FisheyeCamera.load_lens_profile loads data with correct dimensions."""
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        try:
            camera = lc.FisheyeCamera(profile_path)
            
            # Check camera matrix dimensions
            assert camera.camera_matrix.shape == (3, 3)
            assert camera.camera_matrix.dtype == np.float64
            
            # Check distortion coefficients dimensions
            assert camera.distortion_coeffs.shape == (4,)
            assert camera.distortion_coeffs.dtype == np.float64
            
            # Check size dimensions
            assert camera.size.shape == (2,)
            assert camera.size.dtype == np.int64 or camera.size.dtype == np.float64
            
        except FileNotFoundError:
            pytest.skip(f"Lens profile not found: {profile_path}")


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
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        camera = lc.FisheyeCamera(profile_path)
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
        
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        camera = lc.FisheyeCamera(profile_path)
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
        
        profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
        camera = lc.FisheyeCamera(profile_path)
        match_frames = lc.MatchFrames(cam_curves, geo_curves, camera)
        
        # Test output dimensions
        match_frames.auto_initial_r_q()

        #get the initial r and q
        r = match_frames.initial_r
        q = match_frames.initial_q
        
        assert r.shape == (3,)  # Camera position
        assert isinstance(r, np.ndarray)
        assert isinstance(q, quaternion.quaternion)  # Camera orientation
    

def test_visualize_camera_model():
    """Test the camera model visualization function."""
    
    # Create a camera
    profile_path = "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
    camera = lc.FisheyeCamera(profile_path)
    
    # Create camera position and orientation
    r = np.array([0.0, 0.0, 0.0])  # At origin
    q = quaternion.from_rotation_matrix(np.eye(3))  # Identity rotation
    
    # Test visualization
    ax = lc.visualize_camera_model(camera, r, q)
    
    # Check that the plot was created
    assert ax is not None
    assert isinstance(ax, Axes3D)
    
    # Test without FOV
    ax2 = lc.visualize_camera_model(camera, r, q)
    assert ax2 is not None
    
    # Test with different camera orientation
    q_rotated = quaternion.from_rotation_vector([0, 0, np.pi/4])  # 45 degree rotation around z
    ax3 = lc.visualize_camera_model(camera, r, q_rotated)
    assert ax3 is not None
    
    plt.close('all')  # Clean up plots


def test_r_q_rvec_tvec_inverse():
    """Test that r_q_2_rvec_tvec and rvec_tvec_2_r_q are inverse functions."""
    
    # Test case 1: Identity rotation, zero position
    r1 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    q1 = quaternion.from_rotation_matrix(np.eye(3))
    
    # Apply forward transformation
    rvec1, tvec1 = lc.r_q_2_rvec_tvec(r1, q1)
    print(f"{rvec1=}")
    print(f"{tvec1=}")
    
    # Apply inverse transformation
    r1_recovered, q1_recovered = lc.rvec_tvec_2_r_q(rvec1, tvec1)
    
    # Check that we get back the original values
    np.testing.assert_allclose(r1, r1_recovered, atol=1e-10)
    np.testing.assert_allclose(quaternion.as_rotation_matrix(q1), 
                              quaternion.as_rotation_matrix(q1_recovered), atol=1e-10)
    
    # Test case 2: Non-zero position, identity rotation
    r2 = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    q2 = quaternion.from_rotation_matrix(np.eye(3))
    
    rvec2, tvec2 = lc.r_q_2_rvec_tvec(r2, q2)
    r2_recovered, q2_recovered = lc.rvec_tvec_2_r_q(rvec2, tvec2)
    
    np.testing.assert_allclose(r2, r2_recovered, atol=1e-10)
    np.testing.assert_allclose(quaternion.as_rotation_matrix(q2), 
                              quaternion.as_rotation_matrix(q2_recovered), atol=1e-10)
    
    # Test case 3: Zero position, non-identity rotation (90 degrees around z-axis)
    r3 = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    q3 = quaternion.from_rotation_vector([0, 0, np.pi/2])
    
    rvec3, tvec3 = lc.r_q_2_rvec_tvec(r3, q3)
    r3_recovered, q3_recovered = lc.rvec_tvec_2_r_q(rvec3, tvec3)
    
    np.testing.assert_allclose(r3, r3_recovered, atol=1e-10)
    np.testing.assert_allclose(quaternion.as_rotation_matrix(q3), 
                              quaternion.as_rotation_matrix(q3_recovered), atol=1e-10)
    
    # Test case 4: Non-zero position, non-identity rotation (45 degrees around x-axis)
    r4 = np.array([50.0, -25.0, 75.0], dtype=np.float64)
    q4 = quaternion.from_rotation_vector([np.pi/4, 0, 0])
    
    rvec4, tvec4 = lc.r_q_2_rvec_tvec(r4, q4)
    r4_recovered, q4_recovered = lc.rvec_tvec_2_r_q(rvec4, tvec4)
    
    np.testing.assert_allclose(r4, r4_recovered, atol=1e-10)
    np.testing.assert_allclose(quaternion.as_rotation_matrix(q4), 
                              quaternion.as_rotation_matrix(q4_recovered), atol=1e-10)
    
    # Test case 5: Complex rotation (multiple axes)
    r5 = np.array([123.456, -789.012, 345.678], dtype=np.float64)
    q5 = quaternion.from_rotation_vector([np.pi/6, np.pi/4, np.pi/3])
    
    rvec5, tvec5 = lc.r_q_2_rvec_tvec(r5, q5)
    r5_recovered, q5_recovered = lc.rvec_tvec_2_r_q(rvec5, tvec5)
    
    np.testing.assert_allclose(r5, r5_recovered, atol=1e-10)
    np.testing.assert_allclose(quaternion.as_rotation_matrix(q5), 
                              quaternion.as_rotation_matrix(q5_recovered), atol=1e-10)