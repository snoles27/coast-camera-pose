from ast import Pass
from math import cos
import numpy as np
import quaternion
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from pyproj import Transformer
import matplotlib.pyplot as plt

OUT_OF_FRAME_VALUE = 5

class Camera:
    """
    Class to represent a camera convert between different 2D representations
    """

    def __init__(self, fov, res):
        """
        Initialize camera with field of view and resolution.
        
        Args:
            fov: float, field of view in radians
            res: np.ndarray, shape (2,), resolution [width, height] in pixels
        """
        self.fov = fov  # initialize camera field of view attribute (in radians)
        self.res = res  # initialize camera resolution attribute (np array 2x1 wxh in pixel count)
        self.aspect_ratio = res[0]/res[1]
        self.f = 0.5/np.tan(fov/2) #set the focual length such that the image is 1 unit wide
        # will eventually initialize camera parameters to convert between image and ideal pinhole camera

    def raw_photo_to_ideal_pinhole(self, point):
        """
        Take point as taken by camera and convert to point location if it was taken with an ideal pinhole camera.
        Pinhole camera coordinates use 0,0 at the center of the frame with the width normalized to 1.
        
        Args:
            point: np.ndarray, shape (2,), pixel coordinates [x, y]
            
        Returns:
            np.ndarray, shape (2,), normalized pinhole coordinates [x_norm, y_norm] in range [-0.5, 0.5]
        """
        return np.array([
            (point[0] - self.res[0]/2)/self.res[0],
            (point[1] - self.res[1]/2)/self.res[0]
        ]) # assuming camera is pinhole for now, really just a placeholder till I can get a better camera model going
    
    def poi_to_pinhole_projection(self, point, r, q):
        """
        Take a point in 3D cartesian space and project it to an ideal pinhole camera with 
        focal point at location r and camera orientation encoded with quaternion q.
        Quaternion q represents the rotation from ECEF to the camera frame.
        
        Args:
            point: np.ndarray, shape (3,), 3D point in ECEF coordinates [x, y, z]
            r: np.ndarray, shape (3,), camera position in ECEF coordinates [x, y, z]
            q: quaternion object, camera orientation quaternion (ECEF to camera frame)
            
        Returns:
            np.ndarray, shape (2,), normalized pinhole coordinates [y_norm, z_norm] in range [-0.5, 0.5]
            OR [OUT_OF_FRAME_VALUE, OUT_OF_FRAME_VALUE] if point is not visible
        """

        #generate the point from the camera focal point to the point of interest
        r_poi_cam_ecef_norm = (point - r) / np.linalg.norm(point - r)

        #transform this vector to camera frame coordinates using quaternion q
        # Ensure input is 2D shape (N, 3) for quaternion.rotate_vectors, even if single vector
        r_poi_cam_cam_norm = quaternion.rotate_vectors(q, np.atleast_2d(r_poi_cam_ecef_norm))[0]
        r_poi_cam_cam_norm = r_poi_cam_cam_norm.flatten()
        r_x, r_y, r_z = r_poi_cam_cam_norm
        # Only return if point is in front of the camera (x < 0)
        if r_x>=0:
            return [np.inf, np.inf]
        
        return np.array([-r_y/r_x * self.f, -r_z/r_x * self.f])

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

    def to_file(self, file_path):
        """
        Write the points in the curve to a file.
        
        Args:
            file_path: str, path to output file
        """
        with open(file_path, 'w') as f:
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

    def plot(self, k=1, show=True, ax=None, num_samples=100, label="", **plot_kwargs):
        """
        Plot the curve using matplotlib.
        k: spline order (default=1, linear)
        show: whether to call plt.show() (default True)
        ax: optional matplotlib axis to plot on (2D or 3D)
        num_samples: number of points to sample along the curve
        plot_kwargs: additional keyword arguments for ax.plot
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
            ax.plot(pts[:,0], pts[:,1], pts[:,2], 'ro', label=label+' (points)')
            ax.plot(x_fine, y_fine, z_fine, 'b-', label=label + f' curve (k={k})', **plot_kwargs)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
        else:
            # 2D plotting (Y,Z frame)
            y_fine, z_fine = interp_pts  # 2D points are in Y,Z
            ax.plot(pts[:,0], pts[:,1], 'ro', label=label+' (points)')  # pts[:,0]=Y, pts[:,1]=Z
            ax.plot(y_fine, z_fine, 'b-', label=label + f' curve (k={k})', **plot_kwargs)
            ax.set_xlabel('Y')
            ax.set_ylabel('Z')

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

        # call camera.poi_to_pinhole_projection for each point in self
        # return a new curve object with the projected points
        new_points = []
        for point in self.points:
            projected_point = camera.poi_to_pinhole_projection(point, r, q)
            new_points.append(projected_point)
        return Curve.from_points(new_points)

class MatchFrames:
    """
    Class representing a framed problem ready for optimization
    photo_curves: list of curve objects with 2D elements representing the coastline curves detected in the photos in the pinhole camera representation
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
        # Distance = max_dist / tan(fov/2) to ensure everything fits in FOV
        r_cam = geo_com_avg + (max_dist / np.tan(self.camera.fov/2)) * x_hat_cam  # Shape: (3,)
        
        self.set_initial_r_q(r_cam, q_cam)
    
    def get_inital_x(self):
        """
        Return the initial x vector for the optimization
        """
        return np.concatenate((self.initial_r, np.array([self.initial_q.w, self.initial_q.x, self.initial_q.y, self.initial_q.z])))

    def com_difference_cost(self, r, q):
        """
        Cost fucntion that generates the difference between the center of masses
        """

        total_cost = 0
        for index, _ in enumerate(self.photo_curves):
            projected_geo_curve = self.geo_curves[index].project_to_camera(self.camera, r, q)

            photo_com = self.photo_curves[index].center_of_mass()
            geo_com = projected_geo_curve.center_of_mass()
            total_cost += np.linalg.norm(photo_com - geo_com)
        return total_cost
    
    def curve_difference_cost(self, r, q, N, k):
        """
        r vector representing camera location in ECEF
        q quaternion representing camera orientation in ECEF
        n: number of points to sample from the curve to compare
        """

        total_cost = 0
        for index, _ in enumerate(self.photo_curves):
            projected_geo_curve = self.geo_curves[index].project_to_camera(self.camera, r, q)

            cost = 0
            for p in np.linspace(0, 1, N):
                sampled_photo_point = self.photo_curves[index].get_point_along_curve(p,k=k)
                sampled_geo_point = projected_geo_curve.get_point_along_curve(p,k=k)
                cost += np.linalg.norm(sampled_photo_point - sampled_geo_point)
            cost /= N
            total_cost += cost
        return total_cost

    def create_com_objective(self):
        """
        Assumes q is constrained to be normalized
        """
        def cost(x):
            return self.com_difference_cost(x[0:3], x[3:7])
        return cost
    
    def create_com_objective_unconstrained(self, penatly_factor):
        """
        Assumes optimization running on cost is not constrained to normalize q
        """
        def cost(x):
            return self.com_difference_cost(x[0:3], x[3:7]/np.linalg.norm(x[3:7])) + penatly_factor * np.linalg.norm(x[3:7])
        return cost
    
    def create_curve_objective(self, N, k):
        """
        Assumes q is constrained to be normalized
        """
        def cost(x):
            return self.curve_difference_cost(x[0:3], x[3:7], N, k)
        return cost

    def create_curve_objective_unconstrained(self, N, k, penatly_factor):
        """
        Assumes optimization running on cost is not constrained to normalize q
        """
        def cost(x):
            return self.curve_difference_cost(x[0:3], x[3:7]/np.linalg.norm(x[3:7]), N, k) + penatly_factor * np.linalg.norm(x[3:7])
        return cost
    
    def run_unconstrained(self, spline_order=1, number_samples=20, penatly_factor=100, max_iterations=1000):
        """
        run the optimization after setting it up

        Parameters:
        spline_order: int, order of the spline
        number_samples: int, number of samples along the curve
        penatly_factor: float, penalty factor for unconstrained optimization
        max_iterations: int, maximum number of iterations for the optimizer
        """ 

        # check if the initial condition is set
        if self.initial_r is None or self.initial_q is None:
            self.auto_initial_r_q()
        
        # create the objective function
        com_objective = self.create_com_objective_unconstrained(penatly_factor)
        curve_objective = self.create_curve_objective_unconstrained(number_samples, spline_order, penatly_factor)
        
        # run the coarse optimization first on the center of masses
        x0 = self.get_inital_x()
        res_initial = minimize(
            com_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iterations}
        )
        print(f"Completed course optimization with result {res_initial.x=}, {res_initial.success=}")
        # run the full optimization on the curves with the solution to the center of mass optimization as the initial condition
        x0 = res_initial.x
        res = minimize(
            curve_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iterations}
        )
        print(f"Completed optimization with result {res_initial.x=}, {res_initial.success=}")

        # return r and q solutions
        return res.x[0:3], quaternion.from_float_array(res.x[3:7])

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
            projected_geo_curve.plot(ax=ax, show=False, label = "geo")
        
        #plot the photo curves
        for photo_curve in self.photo_curves:
            print(photo_curve)
            photo_curve.plot(ax=ax, show=False, label="photo")
        
        return ax
            
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

def plot_camera_location_orientation(r, q, ax=None):
    """
    Plot the camera coordinate axes and origin within the ECEF frame.

    Args:
        r (np.ndarray): Camera position in ECEF (3,)
        q (np.quaternion): Camera orientation as quaternion (rotation from ECEF to camera frame)
        ax (mpl_toolkits.mplot3d.Axes3D, optional): 3D axes to plot on. If None, creates a new one.

    Returns:
        ax: The matplotlib 3D axes with the camera axes plotted.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import quaternion

    # Create axes if not provided
    if ax is None:
        fig = plt.figure()
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

    # Set axis length (in meters)
    axis_length = 1000  # You may want to adjust this for your scene

    for i in range(3):
        start = r
        end = r + axes_ecef[:, i] * axis_length
        ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
                color=colors[i], label=labels[i])
        # Optionally, add a label at the tip
        ax.text(end[0], end[1], end[2], labels[i], color=colors[i])

    # Mark the camera origin
    ax.scatter([r[0]], [r[1]], [r[2]], color='k', s=50, marker='o', label='Camera Origin')

    # Optionally, set equal aspect ratio for better visualization
    try:
        # This works for matplotlib >= 3.3
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass

    # Set axis labels
    ax.set_xlabel('X (ECEF)')
    ax.set_ylabel('Y (ECEF)')
    ax.set_zlabel('Z (ECEF)')

    # Show legend (avoid duplicate labels)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())

    return ax
    







    