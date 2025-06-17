import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from pathlib import Path

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

def on_click(event, point_manager, img_display, all_points):
    """Handle mouse click events"""
    if event.inaxes is not None:
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
            cv.circle(img_display, closest_point, 3, color, -1)
            plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
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
        cv.circle(img_display, (x, y), 3, (0, 255, 0), -1)  # Green for selected

# Display image and set up click handler
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.imshow(cv.cvtColor(img_display, cv.COLOR_BGR2RGB))
plt.title('Click to deselect points (green=selected, red=deselected)')
plt.axis('off')

# Connect the click event
plt.connect('button_press_event', lambda event: on_click(event, point_manager, img_display, all_points))

plt.tight_layout()
plt.show()

# After user closes the plot window, save the points
if point_manager.selected_points:
    save_path = output_dir / 'selected_points.txt'
    point_manager.save_points(save_path)
    print(f"Selected {len(point_manager.selected_points)} points")

#connected component analysis then point extraction (endpoint or midpoint? (or both))
#hough transform 
