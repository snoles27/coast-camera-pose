


"""
Program to take a photo curve file and rescale it to the camera's native resolution

## Inputs/Arguments

- photo_curve_file: path to the photo curve file with pixels in the original photo pixel coordinates
- original photo resoltion. eg [1280, 720]
- camera native resolution eg [1920, 1080]

## Functions
- Verify original photo resolution and camera native resolution have the same aspect ratio
- Rescale the photo curve file to the camera's native resolution
- Save the photo curve file to the output file. Output file is the original file name with _rescaled appended to the end

"""

import sys
import os
import numpy as np

def verify_aspect_ratio(original_res, native_res):
    """
    Verify that original photo resolution and camera native resolution have the same aspect ratio
    
    Args:
        original_res (list): [width, height] of original photo
        native_res (list): [width, height] of camera native resolution
        
    Returns:
        bool: True if aspect ratios match (within tolerance), False otherwise
    """
    original_aspect = original_res[0] / original_res[1]
    native_aspect = native_res[0] / native_res[1]
    
    # Allow for small floating point differences
    tolerance = 1e-2
    return abs(original_aspect - native_aspect) < tolerance

def read_photo_curve(file_path):
    """
    Read a photo curve file and return the header and pixel coordinates
    
    Args:
        file_path (str): Path to the photo curve file
        
    Returns:
        tuple: (header_lines, pixel_coords) where header_lines is a list of comment lines
               and pixel_coords is a numpy array of [x, y] coordinates
    """
    header_lines = []
    pixel_coords = []
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                header_lines.append(line)
            elif line:  # Non-empty line with coordinates
                coords = line.split(',')
                if len(coords) == 2:
                    x, y = float(coords[0]), float(coords[1])
                    pixel_coords.append([x, y])
    
    return header_lines, np.array(pixel_coords)

def rescale_coordinates(pixel_coords, original_res, native_res):
    """
    Rescale pixel coordinates from original resolution to native resolution
    
    Args:
        pixel_coords (np.array): Array of [x, y] coordinates in original resolution
        original_res (list): [width, height] of original photo
        native_res (list): [width, height] of camera native resolution
        
    Returns:
        np.array: Rescaled coordinates in native resolution
    """
    # Calculate scaling factors
    scale_x = native_res[0] / original_res[0]
    scale_y = native_res[1] / original_res[1]
    
    # Rescale coordinates
    rescaled_coords = pixel_coords.copy()
    rescaled_coords[:, 0] *= scale_x  # Scale x coordinates
    rescaled_coords[:, 1] *= scale_y  # Scale y coordinates
    
    return rescaled_coords

def write_photo_curve(file_path, header_lines, pixel_coords):
    """
    Write a photo curve file with header and pixel coordinates
    
    Args:
        file_path (str): Path to output file
        header_lines (list): List of header comment lines
        pixel_coords (np.array): Array of [x, y] coordinates
    """
    with open(file_path, 'w') as f:
        # Write header lines
        for line in header_lines:
            f.write(line + '\n')
        
        # Write pixel coordinates
        for coord in pixel_coords:
            f.write(f"{coord[0]:.6f}, {coord[1]:.6f}\n")

def main():
    """
    Main function to rescale photo curve files
    """
    if len(sys.argv) != 4:
        print("Usage: python rescale_photo.py <photo_curve_file> <original_resolution> <native_resolution>")
        print("Example: python rescale_photo.py curveA_island_1 [1280,720] [1920,1080]")
        sys.exit(1)
    
    # Parse command line arguments
    photo_curve_file = sys.argv[1]
    
    # Parse resolution strings (e.g., "[1280,720]" -> [1280, 720])
    try:
        original_res_str = sys.argv[2].strip('[]').split(',')
        original_res = [int(original_res_str[0]), int(original_res_str[1])]
        
        native_res_str = sys.argv[3].strip('[]').split(',')
        native_res = [int(native_res_str[0]), int(native_res_str[1])]
    except (ValueError, IndexError):
        print("Error: Resolution format should be [width,height] (e.g., [1280,720])")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(photo_curve_file):
        print(f"Error: File '{photo_curve_file}' not found")
        sys.exit(1)
    
    # Verify aspect ratios match
    if not verify_aspect_ratio(original_res, native_res):
        print("Error: Original photo resolution and camera native resolution must have the same aspect ratio")
        print(f"Original aspect ratio: {original_res[0]}/{original_res[1]} = {original_res[0]/original_res[1]:.6f}")
        print(f"Native aspect ratio: {native_res[0]}/{native_res[1]} = {native_res[0]/native_res[1]:.6f}")
        sys.exit(1)
    
    print(f"Rescaling photo curve: {photo_curve_file}")
    print(f"From resolution: {original_res[0]}x{original_res[1]}")
    print(f"To resolution: {native_res[0]}x{native_res[1]}")
    
    # Read the photo curve file
    try:
        header_lines, pixel_coords = read_photo_curve(photo_curve_file)
        print(f"Read {len(pixel_coords)} coordinate points")
    except Exception as e:
        print(f"Error reading photo curve file: {e}")
        sys.exit(1)
    
    # Rescale the coordinates
    rescaled_coords = rescale_coordinates(pixel_coords, original_res, native_res)
    
    # Generate output filename
    base_name, ext = os.path.splitext(photo_curve_file)
    output_file = base_name + "_rescaled" + ext
    
    # Write the rescaled photo curve file
    try:
        write_photo_curve(output_file, header_lines, rescaled_coords)
        print(f"Rescaled photo curve saved to: {output_file}")
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    # Print some statistics
    print(f"\nRescaling complete!")
    print(f"Original coordinate range: X[{pixel_coords[:, 0].min():.1f}, {pixel_coords[:, 0].max():.1f}], "
          f"Y[{pixel_coords[:, 1].min():.1f}, {pixel_coords[:, 1].max():.1f}]")
    print(f"Rescaled coordinate range: X[{rescaled_coords[:, 0].min():.1f}, {rescaled_coords[:, 0].max():.1f}], "
          f"Y[{rescaled_coords[:, 1].min():.1f}, {rescaled_coords[:, 1].max():.1f}]")

if __name__ == "__main__":
    main()