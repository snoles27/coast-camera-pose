import numpy as np
import quaternion


class Camera:
    """
    Class to represent a camera convert between different 2D representations
    """

    def __init__(self, fov, res):
        self.fov = fov  #intialize camera field of veiw attribute
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
        return np.array([y,z]/2)












    