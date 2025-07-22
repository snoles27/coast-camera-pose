import locate_camera as lc
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    
    photo_curves = [lc.Curve.from_file("frames/test_frame_1/curveA_photo_pinhole")]
    geo_curves = [lc.Curve.from_file("frames/test_frame_1/curveA_geo_ecef")]

    camera = lc.Camera(fov=np.pi/2, res=(1024, 1024)) #resolution doesn't really matter for this example (I think)
    match_frames = lc.MatchFrames(photo_curves, geo_curves, camera)
    match_frames.auto_initial_r_q()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lc.plot_camera_location_orientation(match_frames.initial_r, match_frames.initial_q, ax=ax)
    geo_curves[0].plot(ax=ax)
    plt.show()
    
    # Debug: Check the initial values
    print("Initial r:", match_frames.initial_r)
    print("Initial q:", match_frames.initial_q)
    print("Initial q type:", type(match_frames.initial_q))
    
    # Debug: Check the geo curve points
    print("Geo curve points shape:", len(geo_curves[0].points))
    print("First point:", geo_curves[0].points[0])
    print("First point type:", type(geo_curves[0].points[0]))
    
    # Debug: Step through projection manually
    projected_points = []
    for i, point in enumerate(geo_curves[0].points):
        print(f"Projecting point {i}: {point}")
        projected_point = camera.poi_to_pinhole_projection(point, match_frames.initial_r, match_frames.initial_q)
        print(f"Projected result: {projected_point}")
        if projected_point is not None:
            projected_points.append(projected_point)
        else:
            print(f"Point {i} was not visible to camera")
    
    print("Projected points:", projected_points)
    print("Number of visible points:", len(projected_points))
    
    # Try to create the projected curve
    if projected_points:
        projected_curve = lc.Curve.from_points(projected_points)
        projected_curve.plot()
    else:
        print("No points were visible to the camera!")

    # r, q = match_frames.run_unconstrained()
    # fig = plt.figure()
    # ax = plt.add_subplot(111)
    # match_frames.plot_results(r, q, ax=ax)
    # plt.show()  


    # lc.file_latlong_to_ecef("frames/test_frame_1/curveA_geo_latlong")
    # curve = lc.Curve.from_file("frames/test_frame_1/curveA_geo_ecef")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x,y,z = curve.center_of_mass()
    # ax.plot(x,y,z, 'ro')
    # curve.plot(ax=ax) 
    