from ast import Pass
import numpy as np
import quaternion
import numpy as np
from scipy.interpolate import splprep, splev
from pyproj import Transformer
import matplotlib.pyplot as plt



class Camera:
    """
    Class to represent a camera convert between different 2D representations
    """

    def __init__(self, fov, res):
        self.fov = fov  #intialize camera field of veiw attribute (in radians)
        self.res = res  #intialize camera resolution attribute (np array 2x1 wxh in pixel count)
        self.aspect_ratio = res[0]/res[1]
        #will eventually initialize camera parameters to convert between image and ideal pinhole camera

    def raw_photo_to_ideal_pinhole(self, point):
        """
        take point as taken by camera and convert to point location if it was taken with an ideal pinhole camera
        Pinhole camera coordinates use 0,0 at the center of the frame with the width normalized to 1
        """
        return np.array([
            (point[0] - self.res[0]/2)/self.res[0],
            (point[1] - self.res[1]/2)/self.res[0]
        ]) #assuming camera is pinhole for now, really just a placeholder till I can get a better camera model going
    
    def poi_to_pinhole_projection(self, point, r, q):
        """
        Take a point in 3D cartesian space and project it to an ideal pinhole camera with 
        focal point at location r and camera orentaion encoded with quaternion q
        quaternion q represents the rotation from ECEF to the camera frame
        return location of with width normalized to 1
        return none if the point of interest is not within the camera field of view
        """

        #generate the point from the camera focal point to the point of interest
        r_poi_cam_ecef_norm = (point - r) / np.linalg.norm(point - r)

        #transform this vector to camera frame coordinates using quaternion q
        r_poi_cam_cam_norm = quaternion.rotate_vectors(q, r_poi_cam_ecef_norm)
        x,y,z = r_poi_cam_cam_norm
        # Only return if point is in front of the camera (x < 0)
        if x>=0:
            return None
        #only return if point is in the horozontal (y) FOV
        if abs(y)>np.sin(self.fov/2):
            return None
        #only return if point is in the vertical (z) FOV
        if abs(z)>np.sin(self.fov/2)/self.aspect_ratio:
            return None
        #convert to pinhole camera convention and return
        return np.array([y/2,z/2])

class Curve: 
    """
    Class to represent a curve. The only attributes are the points that define the curve. The points can be either 2D or 3D
    This class has a variety of functions to manipulate the curves
    This class should also include read and write functions to store the point data in simple files. 
    
    Example file format: 

    #Description
    1.0, 2.0, 3.0
    3.123, 7.234, 8.45
    ...
    """

    def __init__(self):
        self.points = []

    @classmethod
    def from_points(cls, points):
        """
        Generate Curve instance by providing a list of numpy arrays representing the points. The points can be 2 or 3 dimensional
        """
        obj = cls()
        obj.points = [np.array(p) for p in points]
        return obj

    @classmethod
    def from_file(cls, file_path):
        """
        Generate Curve instance by reading a file with a list of points 
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
        Write the points in the curve to a file
        """
        with open(file_path, 'w') as f:
            f.write("#Curve points\n")
            for pt in self.points:
                f.write(', '.join(str(x) for x in pt) + '\n')

    def get_point_along_curve(self, parameter, k=1):
        """
        Generate an interpolated point along the curve
        parameter: input from [0,1] indicating what point to extract along the length of the curve
        k: default = 1. Order of spline to interpolate the curve with. Default is linear interpolation
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

    def plot(self, k=1, show=True, ax=None, num_samples=100, **plot_kwargs):
        """
        Plot the curve using matplotlib.
        k: spline order (default=1, linear)
        show: whether to call plt.show() (default True)
        ax: optional matplotlib 3D axis to plot on
        num_samples: number of points to sample along the curve
        plot_kwargs: additional keyword arguments for ax.plot
        """

        if not self.points:
            raise ValueError("Curve has no points to plot.")
        pts = np.array(self.points)
        if pts.shape[0] < 2:
            raise ValueError("Need at least two points to plot a curve.")

        # Interpolate along the curve
        pts_T = pts.T
        from scipy.interpolate import splprep, splev
        tck, u = splprep(pts_T, s=0, k=k)
        u_fine = np.linspace(0, 1, num_samples)
        interp_pts = splev(u_fine, tck)
        x_fine, y_fine, z_fine = interp_pts

        # Set up 3D plot
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        ax.plot(pts[:,0], pts[:,1], pts[:,2], 'ro', label='Original Points')
        ax.plot(x_fine, y_fine, z_fine, 'b-', label=f'Spline Curve (k={k})', **plot_kwargs)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
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
        If there is only one point, return that point as the center of mass.
        Returns:
            np.ndarray: The center of mass as a 1D array of shape (dim,)
        """
        pts = np.array(self.points)
        if len(pts) == 0:
            raise ValueError("Curve has no points.")
        if len(pts) == 1:
            return pts[0]
        # Compute segment midpoints and lengths
        seg_starts = pts[:-1]
        seg_ends = pts[1:]
        midpoints = (seg_starts + seg_ends) / 2
        lengths = np.linalg.norm(seg_ends - seg_starts, axis=1)
        total_length = np.sum(lengths)
        if total_length == 0:
            # All points are coincident; return the first point
            return pts[0]
        # Weighted average of midpoints
        center = np.average(midpoints, axis=0, weights=lengths)
        return center


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








    