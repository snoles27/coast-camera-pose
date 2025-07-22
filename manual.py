import locate_camera as lc
import matplotlib.pyplot as plt


if __name__ == "__main__":
    lc.file_latlong_to_ecef("frames/test_frame_1/curveA_geo_latlong")
    curve = lc.Curve.from_file("frames/test_frame_1/curveA_geo_ecef")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x,y,z = curve.center_of_mass()
    ax.plot(x,y,z, 'ro')
    curve.plot(ax=ax) 
    