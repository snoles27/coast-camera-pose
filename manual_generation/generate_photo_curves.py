import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import locate_camera as lc

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import argparse
from matplotlib.patches import Circle

class PointSelector:
    def __init__(self, image_path, save_path, description=""):
        self.image_path = image_path
        self.save_path = save_path
        self.description = description
        self.selected_points = []
        self.fig, self.ax = None, None
        
    def load_image(self):
        """Load and display the image."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")
        
        # Load the image
        img = mpimg.imread(self.image_path)
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(img)
        self.ax.set_title(f"Click to select points. Press 'Enter' to finish.\nImage: {os.path.basename(self.image_path)}")
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
            # Add point to list
            self.selected_points.append((x, y))
            
            # Draw a circle at the selected point
            circle = Circle((x, y), radius=1, color='red', fill=True, alpha=0.7)
            self.ax.add_patch(circle)
            
            # Add point number
            self.ax.text(x + 10, y + 10, str(len(self.selected_points)), 
                        color='red', fontsize=12, fontweight='bold')
            
            # Update display
            self.fig.canvas.draw()
            
            print(f"Point {len(self.selected_points)} selected at ({x:.2f}, {y:.2f})")
    
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
    
    args = parser.parse_args()
    
    # Create and run the point selector
    selector = PointSelector(args.image_path, args.save_path, args.description)
    selector.run()

if __name__ == "__main__":
    main()

