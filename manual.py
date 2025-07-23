import quaternion
import locate_camera as lc
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    theta_deg = 0.05
    d = 100e3 #meters
    Re = np.linalg.norm(np.array([5584.698255, 6356749.877461]))
    r_cam_ecef = Re*np.array([0,0,1]) + d * np.array([1,0,1])
    # z = Re * np.cos(theta_deg * np.pi/180)
    # r = Re * np.sin(theta_deg * np.pi/180)

    # points_ecef = [
    #     np.array([r,0,z]),
    #     np.array([0,r,z]),
    #     np.array([-r,0,z]),
    #     np.array([0,-r,z])
    # ]

    # poi_curve = lc.Curve.from_points(points_ecef)

    #generate the coordinate rotation 
    phi = np.pi/4
    u = [0,1,0]
    q_array = [
        np.cos(phi/2),
        np.sin(phi/2)*u[0],
        np.sin(phi/2)*u[1],
        np.sin(phi/2)*u[2]
    ]
    q = quaternion.from_float_array(q_array)
    cam = lc.Camera(fov=np.pi/2, res=[100,100])
    #test
    geo_points = lc.Curve.from_file("frames/test_frame_1/curveA_geo_ecef")
    cam_points = lc.Curve.from_file("frames/test_frame_1/curveA_photo_pinhole")

    geo_points_transform = geo_points.project_to_camera(cam, r_cam_ecef, q)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cam_points.plot(k=2, show=False, ax=ax)
    geo_points_transform.plot(show=False,ax=ax)
    plt.show()


    ## OLD HOLD

    # cam = lc.Camera(fov=np.pi/2, res=[100,100])

    # poi_curve_from_camera = poi_curve.project_to_camera(cam, r_cam_ecef, q)
    # print(poi_curve_from_camera)
    # poi_curve_from_camera.plot()
    # poi_curve_from_camera.to_file("frames/test_frame_1/curveA_photo_pinhole")

    #  for i, curve in enumerate(match_frames.photo_curves):
    #     dims = [p.shape for p in curve.points]
    #     all_same = all(d == dims[0] for d in dims) if dims else True
    #     print(f"photo_curves[{i}] has {len(curve.points)} points, dimensions: {dims}, all same: {all_same}")
    # for i, curve in enumerate(match_frames.geo_curves):
    #     dims = [p.shape for p in curve.points]
    #     all_same = all(d == dims[0] for d in dims) if dims else True
    #     print(f"geo_curves[{i}] has {len(curve.points)} points, dimensions: {dims}, all same: {all_same}")


    # # Debug: Check the initial values
    # print("Initial r:", match_frames.initial_r)
    # print("Initial q:", match_frames.initial_q)
    # print("Initial q type:", type(match_frames.initial_q))
    
    # # Debug: Check the geo curve points
    # print("Geo curve points shape:", len(geo_curves[0].points))
    # print("First point:", geo_curves[0].points[0])
    # print("First point type:", type(geo_curves[0].points[0]))
    
    # # Debug: Step through projection manually
    # projected_points = []
    # for i, point in enumerate(geo_curves[0].points):
    #     print(f"Projecting point {i}: {point}")
    #     projected_point = camera.poi_to_pinhole_projection(point, match_frames.initial_r, match_frames.initial_q)
    #     print(f"Projected result: {projected_point}")
    #     if projected_point is not None:
    #         projected_points.append(projected_point)
    #     else:
    #         print(f"Point {i} was not visible to camera")
    
    # print("Projected points:", projected_points)
    # print("Number of visible points:", len(projected_points))
    
    # # Try to create the projected curve
    # if projected_points:
    #     projected_curve = lc.Curve.from_points(projected_points)
    #     projected_curve.plot()
    # else:
    #     print("No points were visible to the camera!")

    # lc.file_latlong_to_ecef("frames/test_frame_1/curveA_geo_latlong")
    # curve = lc.Curve.from_file("frames/test_frame_1/curveA_geo_ecef")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x,y,z = curve.center_of_mass()
    # ax.plot(x,y,z, 'ro')
    # curve.plot(ax=ax) 



    