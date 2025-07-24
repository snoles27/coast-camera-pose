import quaternion
import locate_camera as lc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
import os

MOVE_WINDOW_STR="+1800+0"

def axis_angle_to_quaternion(axis, angle):
    """
    Convert an axis-angle rotation to a quaternion.

    Args:
        axis (array-like): 3-element iterable, axis of rotation (does not need to be normalized)
        angle (float): rotation angle in radians

    Returns:
        np.quaternion: Quaternion representing the rotation
    """
    axis = np.asarray(axis, dtype=float)
    if np.linalg.norm(axis) == 0:
        raise ValueError("Rotation axis cannot be zero vector.")
    axis = axis / np.linalg.norm(axis)
    q_array = [
        np.cos(angle/2),
        np.sin(angle/2)*axis[0],
        np.sin(angle/2)*axis[1],
        np.sin(angle/2)*axis[2]
    ]
    return quaternion.from_float_array(q_array)


# manual file useful for generating views used for testing
if __name__ == "__main__":

    cam = lc.Camera(fov=np.pi/2, res=[100,100])

    #inital (correct) camera location and orientation used to generate curve B photo pinhole
    d = 20e3 #meters
    Re = np.linalg.norm(np.array([5584.698255, 6356749.877461]))
    r_cam_ecef = Re*np.array([0,0,1]) + d * np.array([1,0,1])

    phi = np.pi/4
    u = [0,1,0]
    q_inital = axis_angle_to_quaternion(u, phi)

    #set up tuple for initial r,q
    inital_r_q = (r_cam_ecef, q_inital)

    cam_pos_q_list = [inital_r_q]

    #generate two more camera orientations at 10 degree rotations about the y axis
    q2 = axis_angle_to_quaternion(u, np.pi/18)
    q3 = axis_angle_to_quaternion(u, -np.pi/18)
    cam_pos_q_list.append((r_cam_ecef, q2*q_inital))
    cam_pos_q_list.append((r_cam_ecef, q3*q_inital))

    #generate from a shallower viewing angle 
    phi = np.pi/12
    r4  = Re*np.array([0,0,1]) + d * np.sqrt(2) * np.array([np.cos(phi),0,np.sin(phi)])
    q4 = axis_angle_to_quaternion(u, phi)
    cam_pos_q_list.append((r4, q4))

    #generate from steeper viewing angle 
    phi = np.pi/3
    r5  = Re*np.array([0,0,1]) + d * np.sqrt(2) * np.array([np.cos(phi),0,np.sin(phi)])
    q5 = axis_angle_to_quaternion(u, phi)
    cam_pos_q_list.append((r5, q5))

    #keep correct orientation but move it to the left and right 
    shift = 10e3
    r6 = r_cam_ecef + shift * np.array([0,-1,0])
    r7 = r_cam_ecef + shift * np.array([0,1,0])
    cam_pos_q_list.append((r6, q_inital))
    cam_pos_q_list.append((r7, q_inital))

    #generate closer and further from the curve
    d_list = [1e3, 5e3, 10e3, 20e3, 30e3, 40e3, 50e3]
    for d in d_list:
        r_cam_ecef = Re*np.array([0,0,1]) + d * np.array([1,0,1])
        cam_pos_q_list.append((r_cam_ecef, q_inital))

    #plot the different cameras
    curveB_geo_ecef = lc.Curve.from_file("frames/test_frame_1/curveB_geo_ecef")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    curveB_geo_ecef.plot(ax=ax, show=False)
    for r, q in cam_pos_q_list:
        lc.visualize_camera_model(cam, r, q, ax=ax, show_fov=False, axis_length=5000)
    lc.ensure_equal_aspect_3d(ax)
    plt.get_current_fig_manager().window.wm_geometry(MOVE_WINDOW_STR) # move the window
    plt.show()


    # Project the geo curve to all camera positions/orientations and plot all 2D projections overlayed
    # Prepare a list of colors for different projections
    # Use the recommended way to get a colormap and sample colors
    cmap = plt.get_cmap('tab10')
    color_list = [cmap(i % cmap.N) for i in range(len(cam_pos_q_list))]

    fig2, ax2 = plt.subplots(figsize=(8, 8))
    for idx, (r, q) in enumerate(cam_pos_q_list):
        # Project the geo curve to the camera
        projected_curve = curveB_geo_ecef.project_to_camera(cam, r, q)
        # Plot the projected curve in 2D (Y,Z), with a different color
        color = color_list[idx]
        label = f"Proj {idx}"
        projected_curve.plot(ax=ax2, show=False, color=color, label=label)
    ax2.set_title("2D Projections of curveB_geo_ecef for Different Camera Poses")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Z")
    ax2.legend()
    ax2.grid(True)
    plt.get_current_fig_manager().window.wm_geometry(MOVE_WINDOW_STR) # move the window
    plt.show()

    # Directory to save the projected curves
    output_dir = "frames/test_frame_1/curveB_geo_projections"
    os.makedirs(output_dir, exist_ok=True)

    # For each camera position and orientation, project the curve and save
    for idx, (r, q) in enumerate(cam_pos_q_list):
        # Project the geo curve to the camera
        projected_curve = curveB_geo_ecef.project_to_camera(cam, r, q)
        # Prepare output file path
        out_path = os.path.join(output_dir, f"curveB_geo_projection_{idx}")
        
        # Create description with camera position and quaternion
        # q is assumed to be a quaternion object
        q_arr = q.components
        
        description = f"Projected curve for camera position {idx}\n# Camera position (ECEF): {r.tolist()}\n# Camera quaternion (w, x, y, z): {q_arr}"
        
        # Use the enhanced to_file method
        projected_curve.to_file(out_path, description=description)
