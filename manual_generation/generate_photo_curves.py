import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import locate_camera as lc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
from matplotlib.patches import Circle
import cv2
from scipy.ndimage import gaussian_filter

class PointSelector:
    def __init__(self, image_path, save_path, description="", edge_threshold=50, snap_radius=10):
        self.image_path = image_path
        self.save_path = save_path
        self.description = description
        self.edge_threshold = edge_threshold  # Threshold for edge detection
        self.snap_radius = snap_radius  # Radius to search for nearby edges
        self.selected_points = []
        self.fig, self.ax = None, None
        self.original_image = None
        self.edge_image = None
        self.edge_coordinates = None
        
    def detect_edges(self, image):
        """Detect edges in the image using Canny edge detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = gaussian_filter(gray, sigma=1.0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.edge_threshold, self.edge_threshold * 2)
        
        # Find coordinates of edge pixels
        edge_coords = np.column_stack(np.where(edges > 0))
        
        return edges, edge_coords
    
    def find_nearest_edge(self, x, y):
        """Find the nearest edge pixel within snap_radius of the clicked point."""
        if self.edge_coordinates is None or len(self.edge_coordinates) == 0:
            return x, y
        
        # Convert to integer coordinates
        x_int, y_int = int(x), int(y)
        
        # Create a mask for points within snap_radius
        distances = np.sqrt((self.edge_coordinates[:, 1] - x_int)**2 + 
                           (self.edge_coordinates[:, 0] - y_int)**2)
        
        # Find points within snap radius
        within_radius = distances <= self.snap_radius
        
        if not np.any(within_radius):
            return x, y  # No edge found within radius
        
        # Get the closest edge point
        closest_idx = np.argmin(distances[within_radius])
        edge_y, edge_x = self.edge_coordinates[within_radius][closest_idx]
        
        return float(edge_x), float(edge_y)
    
    def load_image(self):
        """Load and display the image with edge detection overlay."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        # Load the original image
        self.original_image = mpimg.imread(self.image_path)
        
        # Detect edges
        self.edge_image, self.edge_coordinates = self.detect_edges(self.original_image)
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        
        # Display original image
        self.ax.imshow(self.original_image)
        
        # Create a proper edge overlay - only color the edge pixels
        edge_overlay = np.zeros_like(self.original_image)
        edge_overlay[self.edge_image > 0] = [0, 255, 0]  # Cyan edges
        self.ax.imshow(edge_overlay, alpha=0.3)
        
        self.ax.set_title(f"Click to select points. Press 'Enter' to finish.\nImage: {os.path.basename(self.image_path)} (Cyan overlay shows detected edges)")
        self.ax.set_xlabel("X coordinate (pixels)")
        self.ax.set_ylabel("Y coordinate (pixels)")
        
        # Connect the click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
    def on_click(self, event):
        """Handle mouse clicks to select points."""
        if event.inaxes != self.ax:
            return
        
        # Get the click coordinates
        x, y = event.xdata, event.ydata
        
        if x is not None and y is not None:
            # Find the nearest edge pixel
            snapped_x, snapped_y = self.find_nearest_edge(x, y)
            
            # Add snapped point to list
            self.selected_points.append((snapped_x, snapped_y))
            
            # Draw a circle at the snapped point
            circle = Circle((snapped_x, snapped_y), radius=1, color='yellow', fill=True, alpha=0.8)
            self.ax.add_patch(circle)
            
            # Add point number
            self.ax.text(snapped_x + 10, snapped_y + 10, str(len(self.selected_points)), 
                        color='yellow', fontsize=12, fontweight='bold')
            
            # Draw a line from original click to snapped point if they're different
            if abs(x - snapped_x) > 0.5 or abs(y - snapped_y) > 0.5:
                self.ax.plot([x, snapped_x], [y, snapped_y], 'g--', linewidth=1, alpha=0.7)
                print(f"Point {len(self.selected_points)}: clicked at ({x:.1f}, {y:.1f}), snapped to ({snapped_x:.1f}, {snapped_y:.1f})")
            else:
                print(f"Point {len(self.selected_points)} selected at ({snapped_x:.1f}, {snapped_y:.1f})")
            
            # Update display
            self.fig.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'enter':
            self.finish_selection()
        elif event.key == 'escape':
            plt.close()
            print("Selection cancelled.")
    
    def finish_selection(self):
        """Finish point selection and save to file."""
        if not self.selected_points:
            print("No points selected.")
            plt.close()
            return
        
        # Save points to file
        self.save_points()
        
        # Close the plot
        plt.close()
        
        print(f"Selected {len(self.selected_points)} points saved to {self.save_path}")
    
    def save_points(self):
        """Save selected points to file in the specified format."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        with open(self.save_path, 'w') as f:
            # Write description if provided
            if self.description:
                f.write(f"# {self.description}\n")
            
            # Write image path as comment
            f.write(f"# Image: {self.image_path}\n")
            
            # Write number of points as comment
            f.write(f"# Number of points: {len(self.selected_points)}\n")
            
            # Write each point coordinate
            for i, (x, y) in enumerate(self.selected_points, 1):
                f.write(f"{x:.6f}, {y:.6f}\n")
    
    def run(self):
        """Run the point selector."""
        try:
            self.load_image()
            plt.show()
        except Exception as e:
            print(f"Error: {e}")
            return False
        return True

def main():
    parser = argparse.ArgumentParser(description="Select points on an image and save coordinates")
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("save_path", help="Path to save the coordinates file")
    parser.add_argument("-d", "--description", default="", 
                       help="Description to add to the output file (optional)")
    parser.add_argument("-t", "--threshold", type=int, default=50,
                       help="Edge detection threshold (default: 50)")
    parser.add_argument("-r", "--radius", type=int, default=10,
                       help="Snap radius for edge detection (default: 10 pixels)")
    
    args = parser.parse_args()
    
    # Create and run the point selector
    selector = PointSelector(args.image_path, args.save_path, args.description, 
                           edge_threshold=args.threshold, snap_radius=args.radius)
    selector.run()

if __name__ == "__main__":
    main()

