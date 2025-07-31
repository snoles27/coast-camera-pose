import quaternion
import sys
import os
import glob
import argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import locate_camera as lc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use("TkAgg")

MOVE_WINDOW_STR="+1800+0"

def plot_curve_over_photo(photo_path, curve_input, title="Curve Over Photo", save_path=None):
    """
    Plot a curve over a photo.
    
    Args:
        photo_path (str): Path to the photo file
        curve_input (str or Curve): Either a path to the curve file or a Curve object
        title (str): Title for the plot
        save_path (str, optional): Path to save the plot image
    """
    # Load the photo
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Photo file not found: {photo_path}")
    
    img = mpimg.imread(photo_path)
    
    # Load the curve using the Curve class
    if isinstance(curve_input, str):
        # Input is a file path - use Curve class to read it
        curve = lc.Curve.from_file(curve_input)
    elif isinstance(curve_input, lc.Curve):
        # Input is already a Curve object
        curve = curve_input
    else:
        raise TypeError(f"curve_input must be a string (file path) or Curve object, got {type(curve_input)}")
    
    # Check if curve has points and is 2D
    if not curve.points:
        raise ValueError("Curve has no points")
    
    curve_points = np.array(curve.points)
    if curve_points.shape[1] != 2:
        raise ValueError(f"Curve must be 2D, got shape {curve_points.shape}")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the photo
    ax.imshow(img)
    
    # Plot the curve line
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'r-', linewidth=2, label='Curve')
    
    # Plot all points as small circles
    ax.plot(curve_points[:, 0], curve_points[:, 1], 'ro', markersize=2, alpha=0.5)
    
    # Mark the beginning (first point) with a triangle
    ax.plot(curve_points[0, 0], curve_points[0, 1], 'r^', markersize=8, alpha=0.9, markeredgecolor='white', markeredgewidth=1)
    
    # Mark the end (last point) with a square
    ax.plot(curve_points[-1, 0], curve_points[-1, 1], 'rs', markersize=8, alpha=0.9, markeredgecolor='white', markeredgewidth=1)
    
    # Set labels and title
    ax.set_title(title)
    ax.set_xlabel("X coordinate (pixels)")
    ax.set_ylabel("Y coordinate (pixels)")
    ax.legend()
    
    # Move window to secondary monitor if available
    try:
        fig.canvas.manager.window.geometry(MOVE_WINDOW_STR)
    except:
        pass
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()
    
    return fig, ax

def plot_all_curves_over_photo(parent_folder, save_path=None):
    """
    Plot all curves from photo_curves directory over the photo in the parent folder.
    
    Args:
        parent_folder (str): Path to the parent folder containing the photo and photo_curves directory
        save_path (str, optional): Path to save the plot image
    """
    # Find the photo file (same name as folder with image extension)
    folder_name = os.path.basename(parent_folder)
    photo_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    photo_path = None
    
    for ext in photo_extensions:
        potential_photo = os.path.join(parent_folder, folder_name + ext)
        if os.path.exists(potential_photo):
            photo_path = potential_photo
            break
    
    if photo_path is None:
        # Try to find any image file in the folder
        for ext in photo_extensions:
            pattern = os.path.join(parent_folder, f"*{ext}")
            files = glob.glob(pattern)
            if files:
                photo_path = files[0]
                break
    
    if photo_path is None:
        raise FileNotFoundError(f"No photo found in {parent_folder}")
    
    print(f"Found photo: {photo_path}")
    
    # Find the photo_curves directory
    photo_curves_dir = os.path.join(parent_folder, "photo_curves")
    if not os.path.exists(photo_curves_dir):
        raise FileNotFoundError(f"photo_curves directory not found: {photo_curves_dir}")
    
    # Find all curve files
    curve_files = []
    for ext in ['', '.txt']:  # Files with no extension or .txt extension
        pattern = os.path.join(photo_curves_dir, f"*{ext}")
        curve_files.extend(glob.glob(pattern))
    
    # Filter out directories, non-curve files, and rescaled files
    curve_files = [f for f in curve_files if os.path.isfile(f) and 
                   not f.endswith('.txt') and 
                   not os.path.basename(f).endswith('_rescaled')]
    
    if not curve_files:
        raise FileNotFoundError(f"No curve files found in {photo_curves_dir} (excluding rescaled files)")
    
    print(f"Found {len(curve_files)} curve files (excluding rescaled):")
    for curve_file in curve_files:
        print(f"  - {os.path.basename(curve_file)}")
    
    # Load the photo
    if not os.path.exists(photo_path):
        raise FileNotFoundError(f"Photo file not found: {photo_path}")
    
    img = mpimg.imread(photo_path)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the photo
    ax.imshow(img)
    
    # Plot each curve with a different color - use warm colors that contrast with blue
    colors = plt.cm.hot(np.linspace(0.1, 0.9, len(curve_files)))
    
    for i, curve_file in enumerate(curve_files):
        try:
            # Load the curve
            curve = lc.Curve.from_file(curve_file)
            
            if not curve.points:
                print(f"Warning: Curve file {curve_file} has no points, skipping")
                continue
            
            curve_points = np.array(curve.points)
            if curve_points.shape[1] != 2:
                print(f"Warning: Curve file {curve_file} is not 2D, skipping")
                continue
            
            # Plot the curve
            curve_name = os.path.basename(curve_file)
            color = colors[i]
            
            # Plot the curve line
            ax.plot(curve_points[:, 0], curve_points[:, 1], '-', 
                   color=color, linewidth=2, label=curve_name, alpha=0.8)
            
            # Plot all points as small circles
            ax.plot(curve_points[:, 0], curve_points[:, 1], 'o', 
                   color=color, markersize=2, alpha=0.5)
            
            # Mark the beginning (first point) with a triangle
            ax.plot(curve_points[0, 0], curve_points[0, 1], '^', 
                   color=color, markersize=8, alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            
            # Mark the end (last point) with a square
            ax.plot(curve_points[-1, 0], curve_points[-1, 1], 's', 
                   color=color, markersize=8, alpha=0.9, markeredgecolor='white', markeredgewidth=1)
            
        except Exception as e:
            print(f"Error loading curve {curve_file}: {e}")
    
    # Set labels and title
    ax.set_title(f"All Curves Over Photo - {folder_name}")
    ax.set_xlabel("X coordinate (pixels)")
    ax.set_ylabel("Y coordinate (pixels)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Move window to secondary monitor if available
    try:
        fig.canvas.manager.window.geometry(MOVE_WINDOW_STR)
    except:
        pass
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

def main():
    parser = argparse.ArgumentParser(description="Overlay all curves from photo_curves directory onto a photo")
    parser.add_argument("parent_folder", help="Path to the parent folder containing the photo and photo_curves directory")
    parser.add_argument("-s", "--save", help="Path to save the plot image (optional)")
    
    args = parser.parse_args()
    
    # Check if parent folder exists
    if not os.path.exists(args.parent_folder):
        print(f"Error: Parent folder not found: {args.parent_folder}")
        return 1
    
    try:
        plot_all_curves_over_photo(args.parent_folder, args.save)
        return 0
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())