import cv2
import numpy as np

# Load the lens profile
lens_profile_path = "gyroflow_lens_profiles/GoPro/GoPro_HERO8 Black_Narrow_HS Boost_2.7k_16by9.json"
# lens_profile_path = "gyroflow_lens_profiles/GoPro/GoPro_HERO10 Black_Linear_4by3.json"

# Load the Gyroflow lens profile JSON
import json

with open(lens_profile_path, 'r') as f:
    profile = json.load(f)

# Extract camera matrix, distortion coefficients, and image size
if profile.get('use_opencv_fisheye', False):
    camera_matrix = np.array(profile['fisheye_params']['camera_matrix'], dtype=np.float64)
    distortion_coeffs = np.array(profile['fisheye_params']['distortion_coeffs'], dtype=np.float64)
elif profile.get('use_opencv_standard', False):
    camera_matrix = np.array(profile['calib_params']['camera_matrix'], dtype=np.float64)
    distortion_coeffs = np.array(profile['calib_params']['distortion_coeffs'], dtype=np.float64)
else:
    raise ValueError("Profile must use either OpenCV fisheye or standard model")

image_width = profile['calib_dimension']['w']
image_height = profile['calib_dimension']['h']
image_size = (image_width, image_height)

print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)
print("Image Size:", image_size)

alpha = 1
new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    camera_matrix,
    distortion_coeffs,
    image_size,
    R=np.eye(3),
    balance=alpha
)
print(f"Optimal New Camera Matrix (alpha={alpha}):\n", new_camera_matrix)








