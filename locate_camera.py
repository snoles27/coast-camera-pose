
import numpy as np
import quaternion
from scipy.interpolate import splprep, splev
from pyproj import Transformer
import matplotlib.pyplot as plt
import json
import cv2
import geopandas as gpd
from shapely.geometry import Point
from geodatasets import get_path

class FisheyeCamera:
    """
    Class to represent a fisheye camera using Gyroflow lens profiles.
    All it does is reads in one of the gyroflow lens profiles and saves the camera matrix,
    distortion coefficients, and image size as attributes.
    """

    def __init__(self, lens_profile_path):
        """
        Initialize fisheye camera from Gyroflow lens profile.
        Init should check that it is reading from the fisheye_params and not calib_params.
        
        Args:
            lens_profile_path: str, path to Gyroflow lens profile JSON file
        """
        self.load_lens_profile(lens_profile_path)

    def load_lens_profile(self, profile_path):
        """
        Load a Gyroflow lens profile and extract OpenCV-compatible parameters.
        
        Args:
            profile_path: str, path to Gyroflow lens profile JSON file
        """
        with open(profile_path, 'r') as f:
            profile = json.load(f)
        
        # Check if using OpenCV fisheye model
        if profile.get('fisheye_params', None) is not None:
            fisheye_params = profile['fisheye_params']
            self.camera_matrix = np.array(fisheye_params['camera_matrix'], dtype=np.float64)
            self.distortion_coeffs = np.array(fisheye_params['distortion_coeffs'], dtype=np.float64)
        else:
            raise ValueError("Profile must use OpenCV fisheye")

        self.size = np.array([profile['calib_dimension']['w'], profile['calib_dimension']['h']])
        
        # print(f"Loaded fisheye camera: {profile['camera_model']} - {profile['lens_model']}")
        # print(f"Resolution: {self.size[0]} x {self.size[1]}")

    def project_point(self, point, r, q):
        """
        Project a 3D point to the camera given camera position and orientation.
        Note this does not handle the case where the point is behind the camera. It will still project the point. 
        
        Args:
            point: np.ndarray, shape (3,), 3D point in ECEF coordinates
            r: np.ndarray, shape (3,), camera position in ECEF coordinates
            q: quaternion object, camera orientation quaternion (ECEF to camera frame)
            
        Returns:
            np.ndarray, shape (2,), pixel coordinates [x, y] or [OUT_OF_FRAME_VALUE, OUT_OF_FRAME_VALUE] if not visible
        """
        # Input validation
        if not isinstance(point, np.ndarray) or point.shape != (3,):
            raise IndexError(f"Input 'point' must be a numpy array of shape (3,), got {type(point)} with shape {getattr(point, 'shape', None)}")
        if not isinstance(r, np.ndarray) or r.shape != (3,):
            raise IndexError(f"Input 'r' must be a numpy array of shape (3,), got {type(r)} with shape {getattr(r, 'shape', None)}")
        if not isinstance(q, quaternion.quaternion):
            raise IndexError(f"Input 'q' must be a quaternion object, got {type(q)}")

        rotation_vector, tvec = r_q_2_rvec_tvec(r, q)

        # Reshape point for OpenCV fisheye projection
        point_3d = point.reshape(1, 1, 3).astype(np.float64)

        # Project using OpenCV fisheye
        pixel_points, jacobian = cv2.fisheye.projectPoints(
            objectPoints=point_3d, 
            rvec=rotation_vector,
            tvec=tvec, 
            K=self.camera_matrix,
            D=self.distortion_coeffs
        )

        # Extract pixel coordinates
        pixel_coords = np.array(pixel_points[0, 0, :])
        
        return pixel_coords

class Curve: 
    """
    Class to represent a curve. The only attributes are the points that define the curve. 
    The points can be either 2D or 3D.
    This class has a variety of functions to manipulate the curves.
    This class should also include read and write functions to store the point data in simple files. 
    
    Attributes:
        points: list of np.ndarray, each shape (2,) or (3,) for 2D or 3D points respectively
    
    Example file format: 
    #Description
    1.0, 2.0, 3.0
    3.123, 7.234, 8.45
    ...
    """

    def __init__(self):
        """
        Initialize empty curve.
        
        Attributes:
            points: list, initially empty, will contain np.ndarray objects
        """
        self.points = []

    @classmethod
    def from_points(cls, points):
        """
        Generate Curve instance by providing a list of numpy arrays representing the points.
        The points can be 2 or 3 dimensional.
        
        Args:
            points: list of np.ndarray, each shape (2,) or (3,) for 2D or 3D points respectively
            
        Returns:
            Curve: new Curve instance with the provided points
            
        Raises:
            ValueError: if points have inconsistent dimensions
        """
        obj = cls()
        if not points:
            return obj
            
        # Check that all points have the same dimension
        first_dim = len(points[0])
        for i, p in enumerate(points):
            if len(p) != first_dim:
                raise ValueError(f"Point {i} has dimension {len(p)}, expected {first_dim}")
        
        obj.points = [np.array(p) for p in points]
        return obj

    @classmethod
    def from_file(cls, file_path):
        """
        Generate Curve instance by reading a file with a list of points.
        
        Args:
            file_path: str, path to file containing point data
            
        Returns:
            Curve: new Curve instance with points from file
            
        File format:
            Lines starting with # are comments
            Each non-comment line contains comma-separated coordinates
        """
        points = []
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = [float(x) for x in line.split(',')]
                points.append(np.array(parts))
        return cls.from_points(points)

    def to_file(self, file_path, description=None):
        """
        Write the points in the curve to a file.
        
        Args:
            file_path: str, path to output file
            description: str, optional, custom description to write as header comment
        """
        with open(file_path, 'w') as f:
            if description:
                f.write(f"# {description}\n")
            else:
                f.write("#Curve points\n")
            for pt in self.points:
                f.write(', '.join(str(x) for x in pt) + '\n')

    def get_point_along_curve(self, parameter, k=1):
        """
        Generate an interpolated point along the curve.
        
        Args:
            parameter: float, input from [0,1] indicating what point to extract along the length of the curve
            k: int, default=1, order of spline to interpolate the curve with. Default is linear interpolation
            
        Returns:
            np.ndarray, shape (2,) or (3,), interpolated point coordinates
            
        Raises:
            ValueError: if curve has no points, insufficient points, or parameter out of range
        """

        if not self.points:
            raise ValueError("Curve has no points.")
        pts = np.array(self.points)
        if pts.shape[0] < 2:
            raise ValueError("Need at least two points to interpolate.")
        if parameter < 0 or parameter > 1:
            raise ValueError("Parameter must be between 0 and 1")

        # Prepare data for splprep (shape: (dim, N))
        pts_T = pts.T
        # Fit a parametric spline to the points
        tck, u = splprep(pts_T, s=0, k=k)
        pt_interp = splev(parameter, tck)
        return np.array(pt_interp)

    def plot(self, k=1, show=True, ax=None, num_samples=100, label="", color='blue', **plot_kwargs):
        """
        Plot the curve using matplotlib.
        
        Args:
            k: int, default=1, spline order (default=1, linear)
            show: bool, default=True, whether to call plt.show()
            ax: matplotlib.axes.Axes, optional, axis to plot on (2D or 3D)
            num_samples: int, default=100, number of points to sample along the curve
            label: str, default="", label for the plot
            color: str, default='blue', color for both points and curve
            plot_kwargs: dict, additional keyword arguments for ax.plot
            
        Returns:
            matplotlib.axes.Axes: the axis object used for plotting
            
        Raises:
            ValueError: if curve has no points or insufficient points
        """

        if not self.points:
            raise ValueError("Curve has no points to plot.")
        pts = np.array(self.points)
        if pts.shape[0] < 2:
            raise ValueError("Need at least two points to plot a curve.")


        # Determine if curve is 2D or 3D
        is_3d = pts.shape[1] == 3

        # Interpolate along the curve
        pts_T = pts.T
        from scipy.interpolate import splprep, splev
        tck, u = splprep(pts_T, s=0, k=k)
        u_fine = np.linspace(0, 1, num_samples)
        interp_pts = splev(u_fine, tck)

        # Set up plot based on dimension
        if ax is None:
            fig = plt.figure()
            if is_3d:
                ax = fig.add_subplot(111, projection='3d')
            else:
                ax = fig.add_subplot(111)

        if is_3d:
            # 3D plotting
            x_fine, y_fine, z_fine = interp_pts
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 'o', color=color, label=label+' (points)')
            ax.plot(x_fine, y_fine, z_fine, '-', color=color, label=label + f' curve (k={k})', **plot_kwargs)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
        else:
            # 2D plotting (X,Y frame)
            x_fine, y_fine = interp_pts  # 2D points are in X,Y
            ax.plot(pts[:,0], pts[:,1], 'o', color=color, label=label+' (points)')  # pts[:,0]=X, pts[:,1]=Y
            ax.plot(x_fine, y_fine, '-', color=color, label=label + f' curve (k={k})', **plot_kwargs)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

        ax.legend()
        if show:
            plt.show()
        return ax
    
    def __str__(self):
        return str(self.points)

    def center_of_mass(self):
        """
        Compute the center of mass of the curve, assuming linear interpolation between points.
        The center of mass is calculated as the weighted average of the midpoints of each segment,
        weighted by the segment length.
        
        Returns:
            np.ndarray, shape (2,) or (3,), the center of mass coordinates
            
        Raises:
            ValueError: if curve has no points
        """
        
        pts = np.array(self.points)
        if len(pts) == 0:
            raise ValueError("Curve has no points.")
        if len(pts) == 1:
            return pts[0]
        # Compute segment midpoints and lengths
        seg_starts = pts[:-1]  # Shape: (N-1, dim)
        seg_ends = pts[1:]     # Shape: (N-1, dim)
        midpoints = (seg_starts + seg_ends) / 2  # Shape: (N-1, dim)
        lengths = np.linalg.norm(seg_ends - seg_starts, axis=1)  # Shape: (N-1,)
        total_length = np.sum(lengths)
        if total_length == 0:
            # All points are coincident; return the first point
            return pts[0]
        # Weighted average of midpoints
        center = np.average(midpoints, axis=0, weights=lengths)  # Shape: (dim,)
        return center
    
    def project_to_camera(self, camera, r, q):
        """
        Return a new curve object that is the object projected to a 2D curve using the camera object.
        
        Args:
            camera: Camera object representing camera doing the projecting
            r: np.ndarray, shape (3,), location of camera in ECEF coordinates
            q: quaternion object, orientation of camera (rotation from ECEF to camera frame)
            
        Returns:
            Curve: new 2D curve with projected points
            
        Raises:
            ValueError: if the points of self are not 3D
        """
        # enforce that the points of self are 3D, or raise error
        if len(self.points[0]) != 3:
            raise ValueError("Points must be 3D")

        # Use the camera's project_point method for each point
        new_points = []
        for point in self.points:
            projected_point = camera.project_point(point, r, q)
            new_points.append(projected_point)
        return Curve.from_points(new_points)

class MatchFrames:
    """
    Class representing a framed problem ready for optimization
    photo_curves: list of curve objects with 2D elements representing the coastline curves detected in the photos. They are still in the raw photo pixel representation.
    geo_curves: list of curve objects with 3D elements representing the coastline curves in ECEF

    """
    def __init__(self, photo_curves, geo_curves, camera, initial_r = None, initial_q = None):
        self.photo_curves = photo_curves
        self.geo_curves = geo_curves
        self.camera = camera
        self.initial_r = initial_r
        self.initial_q = initial_q
        assert len(photo_curves) == len(geo_curves), "Number of photo and geo curves must match"
    
    def set_initial_r(self, initial_r):
        self.initial_r = initial_r

    def set_initial_q(self, initial_q):
        self.initial_q = initial_q

    def set_initial_r_q(self, initial_r, initial_q):
        self.initial_r = initial_r
        self.initial_q = initial_q
    
    def auto_initial_r_q(self):
        """
        Generate automatic inital r and q values based on geo curves and camera
        """
        # generate average value of the geo curve center of masses 
        geo_coms = [curve.center_of_mass() for curve in self.geo_curves]
        geo_com_avg = np.mean(geo_coms, axis=0)  # Shape: (3,) - 3D vector
        
        # calculate furthest distance from collective center of mass to any geo curve center of mass
        # Concatenate all points from all geo curves and compute the maximum distance from geo_com_avg
        all_points = np.concatenate([curve.points for curve in self.geo_curves], axis=0)
        max_dist = np.max(np.linalg.norm(all_points - geo_com_avg, axis=1))

        z_hat_ecef = np.array([0, 0, 1])  # Shape: (3,) - unit vector in ECEF z direction

        # x_hat_cam: unit vector pointing from origin toward geo_com_avg
        x_hat_cam = geo_com_avg / np.linalg.norm(geo_com_avg)  # Shape: (3,) - unit vector
        
        # z_hat_cam: project ECEF z onto plane perpendicular to x_hat_cam, then normalize
        z_projection = z_hat_ecef - np.dot(z_hat_ecef, x_hat_cam) * x_hat_cam  # Shape: (3,)
        z_hat_cam = z_projection / np.linalg.norm(z_projection)  # Shape: (3,) - unit vector
        
        # y_hat_cam: cross product to complete right-handed coordinate system
        y_hat_cam = np.cross(z_hat_cam, x_hat_cam)  # Shape: (3,) - unit vector
        
        # Create rotation matrix: each row is a unit vector of the camera frame
        rotation_matrix = np.array([x_hat_cam, y_hat_cam, z_hat_cam])  # Shape: (3, 3)
        q_cam = quaternion.from_rotation_matrix(rotation_matrix)
        
        # Position camera at distance that ensures all curves are in view
        # Distance = max_dist * fov_x_to_pixel_ratio to ensure everything fits in FOV
        r_cam = geo_com_avg + (max_dist * self.camera.camera_matrix[0,0]/self.camera.size[0]) * x_hat_cam  # Shape: (3,)
        
        self.set_initial_r_q(r_cam, q_cam)
    
    def run_PnP(self, N=20, k=1, plot=False, flags=cv2.SOLVEPNP_ITERATIVE):
        """
        run the PnP optimization. 
        Args:
            N: int, number of points to sample along the curves
            k: int, spline order for interpolation
            plot: bool, whether to plot the results
        Returns:
            tuple (r,q)
            r: vector that points from the center of the earth (ECEF origin) to the camera position (camera frame origin)
            q: quaternion that when applied to vectors written in the ECEF frame will give the vector as written in the camera frame 
        """

        #select a set of matching points from the photo curves and geo curves
        photo_points = []
        geo_points = []
        parameters = np.linspace(0, 1, N)
        for photo_curve in self.photo_curves:
            photo_points.extend([photo_curve.get_point_along_curve(parameter, k=k) for parameter in parameters])
        for geo_curve in self.geo_curves:
            geo_points.extend([geo_curve.get_point_along_curve(parameter, k=k) for parameter in parameters])

        photo_points = np.array(photo_points).astype(np.float64)
        geo_points = np.array(geo_points).astype(np.float64)

        # Reshape points to match OpenCV's expected format
        # OpenCV expects imagePoints to be (N, 1, 2) for fisheye.solvePnP
        photo_points = photo_points.reshape(-1, 1, 2)
        geo_points = geo_points.reshape(-1, 1, 3)

        if self.initial_r is None or self.initial_q is None:
            return_value,rvec, tvec = cv2.fisheye.solvePnP(
                objectPoints=geo_points, 
                imagePoints=photo_points, 
                cameraMatrix=self.camera.camera_matrix, 
                distCoeffs=self.camera.distortion_coeffs,
                flags=flags
                )
        else: 
            rvec_initial,tvec_initial = r_q_2_rvec_tvec(self.initial_r, self.initial_q)
            return_value,rvec, tvec = cv2.fisheye.solvePnP(
                objectPoints=geo_points, 
                imagePoints=photo_points, 
                cameraMatrix=self.camera.camera_matrix, 
                distCoeffs=self.camera.distortion_coeffs, 
                rvec=rvec_initial, 
                tvec=tvec_initial, 
                useExtrinsicGuess=True,
                flags=flags
                )

        #reshape rvec to (3,)
        rvec = rvec.reshape(3)
        tvec = tvec.reshape(3)

        #convert solutions to r and q in ECEF convention
        r, q = rvec_tvec_2_r_q(rvec, tvec)

        #optionally plot the results
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            self.plot_results(r, q, ax=ax)
            plt.show()

        return r, q

    def plot_results(self, r, q, ax=None):
        """
        Plot the results of the optimization
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)
    
        #plot the projected geo curves
        projected_geo_curves = [curve.project_to_camera(self.camera, r, q) for curve in self.geo_curves]
        for projected_geo_curve in projected_geo_curves:
            projected_geo_curve.plot(ax=ax, show=False, label = "geo", color='red')
        
        #plot the photo curves
        for photo_curve in self.photo_curves:
            photo_curve.plot(ax=ax, show=False, label="photo", color='blue')
        
        return ax


#untested 
def pixel_coords_plot(ax, pixel_points, photo_size, color='blue', label='', linestyle='-'):
    """
    take pixel coordinates (with origin in upper left corner) and plot them so they are in the correct orientation
    for the camera model.

    Args:
        ax: matplotlib.axes.Axes, axis to plot on
        pixel_points: np.ndarray, shape (N, 2), pixel coordinates
        photo_size: tuple, shape (2,), width and height of the photo
        color: str, color of the points
        label: str, label of the points
        linestyle: str, linestyle of the points
    """
    points_x = [x_pixel for x_pixel in pixel_points[:,0]]
    points_y = [photo_size[1] - y_pixel for y_pixel in pixel_points[:,1]]
    ax.plot(points_x, points_y, linestyle=linestyle, color=color, label=label)
    return ax
    
#probably not needed anymore but doesn't hurt to have it for the visualization section
def curve_difference_cost_2d(curve1, curve2, N, k, plot=False):
    """
    Cost function for 2D curves
    
    Args:
        curve1: First 2D curve
        curve2: Second 2D curve  
        N: Number of sample points along the curves
        k: Spline order for interpolation
        plot: bool, whether to plot visualization of the cost calculation
    """
    cost = 0
    if plot:
        sampled_points1 = []
        sampled_points2 = []
        displacements = []
        
    for p in np.linspace(0, 1, N):
        sampled_point_curve1 = curve1.get_point_along_curve(p, k=k)
        sampled_point_curve2 = curve2.get_point_along_curve(p, k=k)

        displacement = sampled_point_curve2 - sampled_point_curve1
        displacement = np.where(np.isnan(displacement), np.inf, displacement)
        print(f"{displacement=}")

        if plot:
            displacements.append(displacement)
            sampled_points1.append(sampled_point_curve1)
            sampled_points2.append(sampled_point_curve2)
        
        cost += np.linalg.norm(displacement)
    
    cost /= N
    
    # Visualization
    if plot:
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot both curves
        curve1.plot(ax=ax, show=False, color='blue', label='Curve 1', linewidth=2, k=k)
        curve2.plot(ax=ax, show=False, color='red', label='Curve 2', linewidth=2, k=k)
        
        # Convert to numpy arrays for easier plotting
        points1 = np.array(sampled_points1)
        points2 = np.array(sampled_points2)
        displacements = np.array(displacements)
        
        # Plot sampled points
        ax.plot(points1[:, 0], points1[:, 1], 'bo', markersize=6, label='Curve 1 samples')
        ax.plot(points2[:, 0], points2[:, 1], 'ro', markersize=6, label='Curve 2 samples')

        # Get the xlimits of the plot to set the arrow head size
        xlim = ax.get_xlim()
        x_range = abs(xlim[1] - xlim[0])
        arrow_head_width = 0.02 * x_range
        
        # Plot displacement vectors
        for i, (p1, p2, disp) in enumerate(zip(points1, points2, displacements)):
            # Draw arrow from curve1 point to curve2 point
            ax.arrow(p1[0], p1[1], disp[0], disp[1], 
                    head_width=arrow_head_width, head_length=arrow_head_width, fc='green', ec='green', alpha=0.7)
            
            # Add magnitude label
            mag = np.linalg.norm(disp)
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            ax.text(mid_x, mid_y, f'{mag:.3f}', fontsize=8, ha='center', va='bottom')
        
        ax.set_title(f'2D Curve Cost Visualization (Total Cost: {cost:.4f})')
        ax.set_xlabel('Y')
        ax.set_ylabel('Z')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Ensure equal aspect ratio
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.show()
    
    return cost
            
def file_latlong_to_ecef(filename): 
    """
    Read a file with a list of points in lat,long,altitude format (degrees, degrees, meters).
    File name should have format ending in "_latlong".
    Write new file with list of points with same file name except "_ecef" replaces "_latlong".
    """

    if not filename.endswith("_latlong"):
        raise ValueError("Input filename must end with '_latlong'")

    output_filename = filename.replace("_latlong", "_ecef")

    # Set up transformer: WGS84 geodetic to ECEF
    transformer = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)

    with open(filename, 'r') as fin, open(output_filename, 'w') as fout:
        fout.write("# ECEF points (meters)\n")
        for line in fin:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            # Try to parse as lat,lon,alt (comma or space separated)
            if ',' in line:
                parts = [float(x.strip()) for x in line.split(',')]
            else:
                parts = [float(x) for x in line.split()]
            if len(parts) < 2:
                continue  # skip malformed lines
            lat = parts[0]
            lon = parts[1]
            alt = parts[2] if len(parts) > 2 else 0.0
            # pyproj expects lon, lat, alt order
            x, y, z = transformer.transform(lon, lat, alt)
            fout.write(f"{x:.6f}, {y:.6f}, {z:.6f}\n")

#convert between ECEF and camera frame conventions
def r_q_2_rvec_tvec(r,q):
    """
    Convert a camera position and orientation in ECEF convention to a rotation vector and translation vector with the cv2 convention
    
    Args:
        r: vector that points from the center of the earth (ECEF origin) to the camera position (camera frame origin)
        q: quaternion that when applied to vectors written in the ECEF frame will give the vector as written in the camera frame 
    Returns:
        rvec: rotation vector representing the oritentation of the object frame in the camera frame (cv2 convention)
        tvec: vector that points from the camera frame origin to the center of the earth (ECEF origin) written in the camera frame
    """

    #convert quaternion to rodrigues rotation vector
    #project points function takes orientation of ECEF in camera frame, so we need to transpose the quaternion
    rotation_vector = quaternion_to_rotation_vector(q)
    
    #tvec is the location of the object frame in the camera frame
    # so we need to invert the translation vector and than rotate to the camera frame
    tvec = -r
    tvec = quaternion.rotate_vectors(q, tvec)

    return rotation_vector, tvec

def rvec_tvec_2_r_q(rvec, tvec):
    """
    Convert a rotation vector and translation vector with the cv2 convention to a quaternion and position in ECEF convention

    Args:
        rvec: rotation vector representing the oritentation of the object frame in the camera frame (cv2 convention)
        tvec: vector that points from the camera frame origin to the center of the earth (ECEF origin) written in the camera frame
    Returns:
        r: vector that points from the center of the earth (ECEF origin) to the camera position (camera frame origin)
        q: quaternion that when applied to vectors written in the ECEF frame will give the vector as written in the camera frame 
    """
    #convert rotation vector to quaternion
    q = quaternion.from_rotation_vector(rvec)
    print(f"{q=}")

    #invert the transation vector so it points from the center of the earth to the camera position
    #then rotate it to the ECEF frame. q applied to a vector rotates it from the ECEF frame to the camera frame, so we need to apply q^-1 to the translation vector
    r = -tvec
    r = quaternion.rotate_vectors(q.inverse(), r)

    return r, q

# Quaternion to rotation vector
def quaternion_to_rotation_vector(q):
    """Convert quaternion to rotation vector using quaternion library."""
    # Convert quaternion to rotation matrix
    rotation_matrix = quaternion.as_rotation_matrix(q)
    
    # Convert to rotation vector using OpenCV
    rotation_vector, _ = cv2.Rodrigues(rotation_matrix)
    return rotation_vector.reshape(3) #shape (3,)

def visualize_camera_model(camera, r, q, ax=None, axis_length=1000):
    """
    Visualization of the camera model including position and orientation.

    Args:
        camera (Camera): Camera object
        r (np.ndarray): Camera position in ECEF (3,)
        q (np.quaternion): Camera orientation as quaternion (rotation from ECEF to camera frame)
        ax (mpl_toolkits.mplot3d.Axes3D, optional): 3D axes to plot on. If None, creates a new one.
        axis_length (float): Length of camera coordinate axes in meters
    Returns:
        ax: The matplotlib 3D axes with the camera model plotted.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import quaternion

    # Create axes if not provided
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    # Camera axes in camera frame (unit vectors)
    axes_cam = np.eye(3)  # x, y, z axes as columns

    # Rotate camera axes to ECEF using quaternion (rotation from ECEF to camera)
    # To get camera axes in ECEF, rotate camera axes by q^-1
    q_inv = np.conjugate(q)
    axes_ecef = quaternion.rotate_vectors(q_inv, axes_cam.T).T  # shape (3,3)

    # Set axis colors and labels
    colors = ['r', 'g', 'b']
    labels = ['x_cam', 'y_cam', 'z_cam']

    # Plot camera coordinate axes
    for i in range(3):
        start = r
        end = r + axes_ecef[:, i] * axis_length
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=colors[i], label=labels[i], linewidth=2)
        # Add a label at the tip
        ax.text(end[0], end[1], end[2], labels[i], color=colors[i], fontsize=10)

    # Mark the camera origin
    ax.scatter([r[0]], [r[1]], [r[2]], color='k', s=100, marker='o', label='Camera Origin')

    # Set axis labels
    ax.set_xlabel('X (ECEF) [m]')
    ax.set_ylabel('Y (ECEF) [m]')
    ax.set_zlabel('Z (ECEF) [m]')

    # Show legend (avoid duplicate labels)
    handles, labels_ = ax.get_legend_handles_labels()
    unique = dict(zip(labels_, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    # Set title
    ax.set_title('Camera Model Visualization', fontsize=14, fontweight='bold')

    return ax

def ensure_equal_aspect_3d(ax):
    """
    Ensure equal aspect ratio for a 3D matplotlib plot.
    
    Args:
        ax (mpl_toolkits.mplot3d.Axes3D): The 3D axes to fix
        
    This function forces all axes to have the same scale, preventing
    circles from appearing as ellipses.
    """
    # Try the modern method first
    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass
    
    # Force equal scaling by setting limits manually
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    
    # Find the maximum range across all axes
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]
    max_range = max(x_range, y_range, z_range)
    
    # Calculate centers of each axis
    x_center = (x_limits[0] + x_limits[1]) / 2
    y_center = (y_limits[0] + y_limits[1]) / 2
    z_center = (z_limits[0] + z_limits[1]) / 2
    
    # Set equal limits for all axes
    half_range = max_range / 2
    ax.set_xlim(x_center - half_range, x_center + half_range)
    ax.set_ylim(y_center - half_range, y_center + half_range)
    ax.set_zlim(z_center - half_range, z_center + half_range)

def ecef_to_latlonalt(ecef_point):
    """
    Convert ECEF coordinates to latitude, longitude.
    
    Args:
        ecef_point (np.ndarray): ECEF coordinates [x, y, z] in meters
        
    Returns:
        tuple: (latitude, longitude) in degrees
    """
    # Set up transformer: ECEF to WGS84 geodetic
    transformer = Transformer.from_crs("epsg:4978", "epsg:4979", always_xy=True)
    
    x, y, z = ecef_point
    lon, lat, alt = transformer.transform(x, y, z)
    
    return lat, lon, alt

def plot_point_on_simple_map(lat, lon, point_label="Camera Position", figsize=(10, 6), show=True):
    """
    Plot a latitude/longitude point on a simple world map using matplotlib.
    Warning this is very slow 
    
    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees  
        point_label (str): Label for the point on the map
        figsize (tuple): Figure size (width, height)
        show (bool): Whether to display the plot
        
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set up the world map boundaries
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    
    # Add labels
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')
    
    # Load a base map and clip to the plot region
    shapefile_path = 'gshhg-shp-2/GSHHS_shp/h/GSHHS_h_L1.shp'
    world = gpd.read_file(shapefile_path)

    world.plot(ax=ax, color='white', edgecolor='black')
    
    # Plot the point
    ax.plot(lon, lat, 'ro', markersize=12, markeredgecolor='black', 
            markeredgewidth=2, label=point_label)
    
    # Add text label
    ax.text(lon + 2, lat + 2, point_label, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Set title
    ax.set_title(f'World Map - {point_label}\nLatitude: {lat:.4f}°, Longitude: {lon:.4f}°', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set aspect ratio to be more map-like
    ax.set_aspect('equal')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, ax

def plot_point_on_zoom_map(lat, lon, point_label="Camera Position", 
                          lat_margin=5, lon_margin=5, figsize=(10, 8), show=True):
    """
    Plot a latitude/longitude point on a zoomed-in map centered around the point.
    
    Args:
        lat (float): Latitude in degrees
        lon (float): Longitude in degrees
        point_label (str): Label for the point on the map
        lat_margin (float): Degrees of latitude margin around the point
        lon_margin (float): Degrees of longitude margin around the point
        figsize (tuple): Figure size (width, height)
        show (bool): Whether to display the plot
        
    Returns:
        fig, ax: matplotlib figure and axes objects
    """
    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)
    
    # Set the map extent around the point
    ax.set_xlim(lon - lon_margin, lon + lon_margin)
    ax.set_ylim(lat - lat_margin, lat + lat_margin)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add labels
    ax.set_xlabel('Longitude (°)')
    ax.set_ylabel('Latitude (°)')

    # Load a base map (e.g., naturalearth_lowres)
    # world = gpd.read_file(get_path('naturalearth.land'))
    shapefile_path = 'gshhg-shp-2/GSHHS_shp/h/GSHHS_h_L1.shp'
    world = gpd.read_file(shapefile_path)
    
    # Create a bounding box for the zoom region
    from shapely.geometry import box
    plot_bounds = box(lon - lon_margin, lat - lat_margin, lon + lon_margin, lat + lat_margin)
    world_clipped = gpd.clip(world, plot_bounds)
    world_clipped.plot(ax=ax, color='white', edgecolor='black')
    
    # Plot the point
    ax.plot(lon, lat, 'ro', markersize=15, markeredgecolor='black', 
            markeredgewidth=2, label=point_label)
    
    # Add text label
    ax.text(lon + 0.1, lat + 0.1, point_label, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    # Set title
    ax.set_title(f'Regional Map - {point_label}\nLatitude: {lat:.4f}°, Longitude: {lon:.4f}°', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    ax.legend(loc='upper right')
    
    # Set aspect ratio to be more map-like
    ax.set_aspect('equal')
    
    if show:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


if __name__ == "__main__":
    
    photo_curves = [
        Curve.from_file("frames/w3_full_low_f47533/photo_curves/curveA_island_1_rescaled"),
        Curve.from_file("frames/w3_full_low_f47533/photo_curves/curveB_main_coast_rescaled"),
        Curve.from_file("frames/w3_full_low_f47533/photo_curves/curveC_st_peter_south_rescaled")
    ]
    geo_curves = [
        Curve.from_file("frames/w3_full_low_f47533/geo_curves/curveA_island_1_ecef"),
        Curve.from_file("frames/w3_full_low_f47533/geo_curves/curveB_main_coast_ecef"),
        Curve.from_file("frames/w3_full_low_f47533/geo_curves/curveC_st_peter_south_ecef")
    ]

    camera = FisheyeCamera("gyroflow_lens_profiles/GoPro/GoPro_HERO8 Black_Narrow_HS Boost_2.7k_16by9.json")
    match_frames = MatchFrames(photo_curves, geo_curves, camera)

    r, q = match_frames.run_PnP(N=300, k=1, plot=True, flags=cv2.SOLVEPNP_SQPNP)

    #print the comparison of the correct r,q and the estimated r,q
    print(f"Estimated r: {r}")
    print(f"Estimated q: {q}")

    #convert r to lat,lon
    lat, lon, alt = ecef_to_latlonalt(r)
    print(f"Camera position: {lat:.4f}°, {lon:.4f}°, {alt:.4f}m")
    
    # Plot camera position on zoomed map
    fig, ax = plot_point_on_zoom_map(lat, lon, "Estimated Camera Position", lat_margin=5, lon_margin=5, show=False)

    geo_curves_lat_long = [
        Curve.from_file("frames/w3_full_low_f47533/geo_curves/curveA_island_1"),
        Curve.from_file("frames/w3_full_low_f47533/geo_curves/curveB_main_coast"),
        Curve.from_file("frames/w3_full_low_f47533/geo_curves/curveC_st_peter_south")
    ]

    # Plot the geo curves on the same map
    for i, curve in enumerate(geo_curves_lat_long):
        # Extract lat/lon coordinates from the curve points
        lats = [point[0] for point in curve.points]  # latitude is first column
        lons = [point[1] for point in curve.points]  # longitude is second column
        
        # Plot the curve on the same axes
        ax.plot(lons, lats, linewidth=2, label=f'Geo Curve {chr(65+i)}', alpha=0.8)
    
    # Update the legend to include the new curves
    ax.legend(loc='upper right')
    
    # Refresh the plot
    plt.tight_layout()
    plt.show()

    