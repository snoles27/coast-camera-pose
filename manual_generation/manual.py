import quaternion
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import locate_camera as lc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

MOVE_WINDOW_STR="+1800+0"


if __name__ == "__main__":
    
    # Test: Compare all projected curves to the correct curve using curve_difference_cost_2d
    print("=== Testing 2D Curve Comparison ===")
    
    # Read the correct curve
    correct_curve = lc.Curve.from_file("frames/test_frame_1/curveB_photo_pinhole")
    print(f"Loaded correct curve with {len(correct_curve.points)} points")
    
    # Read all projected curves
    projections_dir = "frames/test_frame_1/curveB_geo_projections"
    projected_curves = []
    curve_indices = []
    
    if os.path.exists(projections_dir):
        for filename in sorted(os.listdir(projections_dir)):
            if filename.startswith("curveB_geo_projection_"):
                try:
                    curve_idx = int(filename.split("_")[-1])
                    curve_path = os.path.join(projections_dir, filename)
                    curve = lc.Curve.from_file(curve_path)
                    projected_curves.append(curve)
                    curve_indices.append(curve_idx)
                    print(f"Loaded projection {curve_idx} with {len(curve.points)} points")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    else:
        print(f"Projections directory not found: {projections_dir}")
        exit(1)
    
    print(f"\nLoaded {len(projected_curves)} projected curves")
    
    # Calculate 2D curve difference costs for each projection
    curve_costs = []
    
    print("\n=== 2D Curve Cost Analysis ===")
    print("Index | Curve Cost (2D)")
    print("-" * 25)
    
    for idx, projected_curve in enumerate(projected_curves):
        # Use the curve_difference_cost_2d function directly
        # Parameters: N=20 sample points, k=1 (linear interpolation)
        curve_cost = lc.curve_difference_cost_2d(correct_curve, projected_curve, N=20, k=2, plot=True)
        curve_costs.append(curve_cost)
        
        print(f"{curve_indices[idx]:5d} | {curve_cost:14.4f}")
    
    # Find the best match
    best_idx = np.argmin(curve_costs)
    best_curve_idx = curve_indices[best_idx]
    
    print(f"\n=== Results ===")
    print(f"Best match: Projection {best_curve_idx} (index {best_idx})")
    print(f"Best 2D curve cost: {curve_costs[best_idx]:.4f}")
    print(f"Worst 2D curve cost: {max(curve_costs):.4f}")
    print(f"Cost range: {max(curve_costs) - min(curve_costs):.4f}")
    
    # Visualize the results
    print("\n=== Visualization ===")
    
    # Plot all projected curves vs correct curve
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot correct curve in bold
    correct_curve.plot(ax=ax, show=False, color='black', label='Correct Curve', linewidth=3)
    
    # Plot all projected curves with different colors
    cmap = plt.get_cmap('tab10')
    for idx, (projected_curve, curve_idx) in enumerate(zip(projected_curves, curve_indices)):
        color = cmap(idx % cmap.N)
        alpha = 0.7 if idx != best_idx else 1.0
        linewidth = 1 if idx != best_idx else 2
        label = f"Proj {curve_idx}" + (" (BEST)" if idx == best_idx else "")
        
        projected_curve.plot(ax=ax, show=False, color=color, label=label, alpha=alpha, linewidth=linewidth)
    
    ax.set_title("All Projected Curves vs Correct Curve (2D Comparison)")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)
    
    # Move window to secondary monitor
    try:
        fig.canvas.manager.window.geometry(MOVE_WINDOW_STR)
    except:
        pass
    
    plt.tight_layout()
    plt.show()
    
    # Plot cost comparison
    fig2, ax = plt.subplots(figsize=(12, 6))
    
    # Bar plot of 2D curve costs
    bars = ax.bar(curve_indices, curve_costs, color='lightblue', alpha=0.7)
    
    # Highlight the best match
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(0.8)
    
    ax.set_title("2D Curve Difference Costs")
    ax.set_xlabel("Projection Index")
    ax.set_ylabel("Cost")
    ax.axhline(y=curve_costs[best_idx], color='red', linestyle='--', label=f'Best: {curve_costs[best_idx]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, cost) in enumerate(zip(bars, curve_costs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(curve_costs)*0.01,
                f'{cost:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Move window to secondary monitor
    try:
        fig2.canvas.manager.window.geometry(MOVE_WINDOW_STR)
    except:
        pass
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nTest completed. Best match was projection {best_curve_idx} with 2D curve cost {curve_costs[best_idx]:.4f}.")
    
    
    ## OLD HOLD
    ## from setting up Cruve B and some plotting from 7/23

    # theta_deg = 0.05
    # d = 20e3 #meters
    # Re = np.linalg.norm(np.array([5584.698255, 6356749.877461]))
    # r_cam_ecef = Re*np.array([0,0,1]) + d * np.array([1,0,1])
    # z = Re * np.cos(theta_deg * np.pi/180)
    # r = Re * np.sin(theta_deg * np.pi/180)

    # points_ecef = [np.array([r*np.cos(phi),r*np.sin(phi),z]) for phi in np.linspace(0, np.pi/2, 5)]
    # print(points_ecef)

    # poi_curve = lc.Curve.from_points(points_ecef)

    # #generate the coordinate rotation 
    # phi = np.pi/4
    # u = [0,1,0]
    # q_array = [
    #     np.cos(phi/2),
    #     np.sin(phi/2)*u[0],
    #     np.sin(phi/2)*u[1],
    #     np.sin(phi/2)*u[2]
    # ]
    # q = quaternion.from_float_array(q_array)
    # cam = lc.Camera(fov=np.pi/2, res=[100,100])

    # poi_curve_from_camera = poi_curve.project_to_camera(cam, r_cam_ecef, q)
    # poi_curve_from_camera.to_file("frames/test_frame_1/curveB_photo_pinhole")
    
    # #test
    # geo_points = lc.Curve.from_file("frames/test_frame_1/curveB_geo_ecef")
    # cam_points = lc.Curve.from_file("frames/test_frame_1/curveB_photo_pinhole")

    # geo_points_transform = geo_points.project_to_camera(cam, r_cam_ecef, q)

    # fig =  plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # geo_points.plot(ax=ax, show=False)
    # lc.visualize_camera_model(cam, r_cam_ecef, q, ax=ax, show_fov=False, axis_length=5000)
    # lc.ensure_equal_aspect_3d(ax)
    # plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # cam_points.plot(k=2, show=False, ax=ax)
    # geo_points_transform.plot(show=False,ax=ax)
    # plt.show()

  

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



    