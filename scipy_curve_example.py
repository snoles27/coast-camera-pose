import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example 3D points
points = np.array([
    [0, 0, 0],
    [1, 2, 1],
    [2, 0, 2],
    [3, 3, 3],
    [4, 0, 4]
])

# Prepare data for splprep (needs shape (3, N))
x, y, z = points.T

# Fit a parametric spline to the 3D points
# s=0 means the spline will pass through all points
# k=3 for cubic spline (default)
tck, u = splprep([x, y, z], s=0)

# Interpolate along the curve
num_samples = 100
u_fine = np.linspace(0, 1, num_samples)
x_fine, y_fine, z_fine = splev(u_fine, tck)

# Example: get a point at parameter t (0 <= t <= 1)
t = 0.35
point_at_t = np.array(splev(t, tck))  # shape (3,)
print(f"Point at t={t}: {point_at_t}")

# Plot the original points and the interpolated curve
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z, 'ro', label='Original Points')
ax.plot(x_fine, y_fine, z_fine, 'b-', label='Spline Curve')
ax.scatter(*point_at_t, color='g', s=50, label=f'Point at t={t}')
ax.legend()
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Curve Interpolation with SciPy')
plt.show() 