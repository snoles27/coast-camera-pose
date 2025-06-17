import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Button

# Create output directory if it doesn't exist
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

class PointManager:
    def __init__(self):
        self.selected_points = []
        self.filename = None
    
    def add_point(self, point):
        """Add a point to the selected points list"""
        self.selected_points.append(point)
    
    def remove_point(self, point):
        """Remove a point from the selected points list"""
        if point in self.selected_points:
            self.selected_points.remove(point)
    
    def clear_points(self):
        """Clear all selected points"""
        self.selected_points.clear()
    
    def select_all(self, points):
        """Select all points"""
        self.selected_points = points.copy()
    
    def deselect_all(self):
        """Deselect all points"""
        self.selected_points.clear()
    
    def save_points(self, filename):
        """Save points to a text file"""
        self.filename = filename
        with open(filename, 'w') as f:
            for point in self.selected_points:
                f.write(f"{point[0]},{point[1]}\n")
        print(f"Points saved to {filename}")
    
    def load_points(self, filename):
        """Load points from a text file"""
        self.filename = filename
        self.clear_points()
        with open(filename, 'r') as f:
            for line in f:
                x, y = map(int, line.strip().split(','))
                self.selected_points.append((x, y))
        print(f"Points loaded from {filename}")
        return self.selected_points

def draw_wireframe(ax, points, line_width=0.5, point_size=1, color='b'):
    """Draw a wireframe visualization of the given points
    
    Args:
        ax: matplotlib axis to draw on
        points: list of (x,y) points
        line_width: width of connecting lines
        point_size: size of point markers
        color: color of lines and points
    """
    ax.clear()
    if len(points) > 1:
        points = np.array(points)
        # Draw lines between consecutive points
        for i in range(len(points)-1):
            ax.plot([points[i][0], points[i+1][0]], 
                    [points[i][1], points[i+1][1]], 
                    f'{color}-', linewidth=line_width)
        # Draw line from last point to first point to close the shape
        ax.plot([points[-1][0], points[0][0]], 
                [points[-1][1], points[0][1]], 
                f'{color}-', linewidth=line_width)
        # Draw small points
        ax.plot(points[:, 0], points[:, 1], f'{color}.', markersize=point_size)
    ax.set_title('Wireframe Outline')
    ax.axis('off')
    ax.invert_yaxis()  # Invert y-axis to match image coordinates

def on_click(event, point_manager, img_display, all_points, ax2):
    """Handle mouse click events"""
    if event.inaxes is not None and event.inaxes not in [ax_select, ax_deselect]:  # Ignore clicks on buttons
        x, y = int(event.xdata), int(event.ydata)
        clicked_point = (x, y)
        
        # Find the closest point within a small radius
        min_dist = float('inf')
        closest_point = None
        for point in all_points:
            dist = ((point[0] - x) ** 2 + (point[1] - y) ** 2) ** 0.5
            if dist < min_dist and dist < 5:  # 5 pixel radius for selection
                min_dist = dist
                closest_point = point
        
        if closest_point is not None:
            # Toggle point selection
            if closest_point in point_manager.selected_points:
                point_manager.remove_point(closest_point)
                color = (0, 0, 255)  # Red for unselected
            else:
                point_manager.add_point(closest_point)
                color = (0, 255, 0)  # Green for selected
            
            # Draw/update point on image
            cv.circle(img_display, closest_point, 2, color, -1)  # Reduced point size to 2
            
            # Update both plots
            plt.subplot(121)
            plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
            
            # Update wireframe plot
            draw_wireframe(ax2, point_manager.selected_points)
            
            plt.draw()

def select_all_callback(event, point_manager, all_points, img_display, ax2):
    """Callback for select all button"""
    point_manager.select_all(all_points)
    # Update display
    img_display = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Reset image
    for point in all_points:
        cv.circle(img_display, point, 2, (0, 255, 0), -1)  # Green for selected
    plt.subplot(121)
    plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
    draw_wireframe(ax2, point_manager.selected_points)
    plt.draw()

def deselect_all_callback(event, point_manager, all_points, img_display, ax2):
    """Callback for deselect all button"""
    point_manager.deselect_all()
    # Update display
    img_display = cv.cvtColor(img, cv.COLOR_GRAY2BGR)  # Reset image
    for point in all_points:
        cv.circle(img_display, point, 2, (0, 0, 255), -1)  # Red for unselected
    plt.subplot(121)
    plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
    draw_wireframe(ax2, point_manager.selected_points)
    plt.draw()

# Read the image
img = cv.imread('test_photos/w3_youtube_0838_1032_edward_island.png', cv.IMREAD_GRAYSCALE)
assert img is not None, "file could not be read, check with os.path.exists()"

# Apply edge detection
edges = cv.Canny(img, 50, 150)

# Find contours
contours, hierarchy = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Create a copy of the image for drawing
img_display = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

# Store all points and initialize them as selected
all_points = []
point_manager = PointManager()

# Process contours and initialize all points as selected
for contour in contours:
    epsilon = 0.001 * cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, epsilon, True)
    
    for point in approx:
        x, y = point[0]
        all_points.append((x, y))
        point_manager.add_point((x, y))  # Add all points to selected points initially
        cv.circle(img_display, (x, y), 2, (0, 255, 0), -1)  # Reduced point size to 2

# Display image and set up click handler
plt.figure(figsize=(12, 8))  # Made figure taller to accommodate buttons

# Create subplots with more space at the bottom
plt.subplot(121)
plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
plt.title('Click to deselect points (green=selected, red=deselected)')
plt.axis('off')

# Create second subplot for wireframe
ax2 = plt.subplot(122)
ax2.set_title('Wireframe Outline')
ax2.axis('off')
ax2.invert_yaxis()

# Add buttons below the plots
plt.subplots_adjust(bottom=0.15)  # Increased bottom margin
ax_select = plt.axes((0.3, 0.05, 0.2, 0.05))  # Moved buttons down
ax_deselect = plt.axes((0.6, 0.05, 0.2, 0.05))  # Moved buttons down
btn_select = Button(ax_select, 'Select All')
btn_deselect = Button(ax_deselect, 'Deselect All')

# Connect the click event and button callbacks
plt.connect('button_press_event', lambda event: on_click(event, point_manager, img_display, all_points, ax2))
btn_select.on_clicked(lambda event: select_all_callback(event, point_manager, all_points, img_display, ax2))
btn_deselect.on_clicked(lambda event: deselect_all_callback(event, point_manager, all_points, img_display, ax2))

plt.tight_layout()
plt.show()

# After user closes the plot window, save the points
if point_manager.selected_points:
    save_path = output_dir / 'selected_points.txt'
    point_manager.save_points(save_path)
    print(f"Selected {len(point_manager.selected_points)} points")

#connected component analysis then point extraction (endpoint or midpoint? (or both))
#hough transform 
