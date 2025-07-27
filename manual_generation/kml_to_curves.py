#!/usr/bin/env python3
"""
KML to Curves Converter
Extracts coordinate data from KML placemarks and saves them as text files.
"""

import sys
import os
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from pyproj import Transformer

def parse_kml_coordinates(coordinates_text):
    """
    Parse KML coordinates string into list of [latitude, longitude, altitude] tuples.
    
    Args:
        coordinates_text (str): KML coordinates string (comma-separated lon,lat,alt triplets)
        
    Returns:
        list: List of [latitude, longitude, altitude] tuples
    """
    coordinates = []
    
    # Split by whitespace to get each coordinate triplet
    coord_triplets = coordinates_text.strip().split()
    
    for triplet in coord_triplets:
        try:
            # Split each triplet by comma to get lon, lat, alt
            parts = triplet.split(',')
            if len(parts) >= 3:
                lon = float(parts[0])
                lat = float(parts[1])
                alt = float(parts[2])
                coordinates.append([lat, lon, alt])  # Reorder to lat, lon, alt
        except ValueError as e:
            print(f"Warning: Could not parse coordinate triplet: {triplet}")
            continue
    
    return coordinates

def extract_placemarks_from_kml(kml_file_path):
    """
    Extract placemarks from KML file.
    
    Args:
        kml_file_path (str): Path to the KML file
        
    Returns:
        list: List of dictionaries with placemark data
    """
    try:
        # Parse the KML file
        tree = ET.parse(kml_file_path)
        root = tree.getroot()
        
        # Define KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        placemarks = []
        
        # Find all Placemark elements
        for placemark in root.findall('.//kml:Placemark', ns):
            placemark_data = {}
            
            # Get placemark name
            name_elem = placemark.find('kml:name', ns)
            if name_elem is not None:
                placemark_data['name'] = name_elem.text
            else:
                placemark_data['name'] = f"placemark_{len(placemarks)}"
            
            # Get coordinates from LineString
            linestring = placemark.find('.//kml:LineString', ns)
            if linestring is not None:
                coords_elem = linestring.find('kml:coordinates', ns)
                if coords_elem is not None:
                    placemark_data['coordinates'] = parse_kml_coordinates(coords_elem.text)
                    placemark_data['type'] = 'LineString'
            
            # Get coordinates from Polygon (outer boundary)
            polygon = placemark.find('.//kml:Polygon', ns)
            if polygon is not None:
                outer_boundary = polygon.find('.//kml:outerBoundaryIs//kml:LinearRing//kml:coordinates', ns)
                if outer_boundary is not None:
                    placemark_data['coordinates'] = parse_kml_coordinates(outer_boundary.text)
                    placemark_data['type'] = 'Polygon'
            
            # Only add placemarks that have coordinates
            if 'coordinates' in placemark_data and placemark_data['coordinates']:
                placemarks.append(placemark_data)
                print(f"Found placemark: {placemark_data['name']} ({placemark_data['type']}) with {len(placemark_data['coordinates'])} points")
        
        return placemarks
        
    except ET.ParseError as e:
        print(f"Error parsing KML file: {e}")
        return []
    except Exception as e:
        print(f"Error reading KML file: {e}")
        return []

def convert_latlong_to_ecef(coordinates):
    """
    Convert latitude, longitude, altitude coordinates to ECEF coordinates.
    
    Args:
        coordinates (list): List of [lat, lon, alt] coordinate tuples
        
    Returns:
        list: List of [x, y, z] ECEF coordinate tuples
    """
    # Set up transformer: WGS84 geodetic to ECEF
    transformer = Transformer.from_crs("epsg:4979", "epsg:4978", always_xy=True)
    
    ecef_coordinates = []
    for coord in coordinates:
        lat, lon, alt = coord
        # pyproj expects lon, lat, alt order
        x, y, z = transformer.transform(lon, lat, alt)
        ecef_coordinates.append([x, y, z])
    
    return ecef_coordinates

def save_placemark_to_file(placemark_data, output_dir, save_ecef=False):
    """
    Save placemark coordinates to a text file.
    
    Args:
        placemark_data (dict): Placemark data with name and coordinates
        output_dir (str): Directory to save the file
        save_ecef (bool): Whether to also save ECEF coordinates
        
    Returns:
        str: Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from placemark name (sanitize for filesystem)
    filename = placemark_data['name'].replace(' ', '_').replace('/', '_').replace('\\', '_')
    filepath = os.path.join(output_dir, filename)
    
    # Write coordinates to file
    with open(filepath, 'w') as f:
        f.write(f"# {placemark_data['name']} ({placemark_data['type']})\n")
        f.write(f"# Extracted from KML file\n")
        f.write(f"# Format: latitude, longitude, altitude\n")
        f.write(f"# Number of points: {len(placemark_data['coordinates'])}\n")
        
        for i, coord in enumerate(placemark_data['coordinates']):
            f.write(f"{coord[0]:.8f}, {coord[1]:.8f}, {coord[2]:.2f}\n")
    
    print(f"Saved {len(placemark_data['coordinates'])} points to: {filepath}")
    
    # Save ECEF coordinates if requested
    if save_ecef:
        ecef_coordinates = convert_latlong_to_ecef(placemark_data['coordinates'])
        ecef_filepath = filepath + "_ecef"
        
        with open(ecef_filepath, 'w') as f:
            f.write(f"# {placemark_data['name']} ({placemark_data['type']}) - ECEF coordinates\n")
            f.write(f"# Converted from KML file\n")
            f.write(f"# Format: x, y, z (meters)\n")
            f.write(f"# Number of points: {len(ecef_coordinates)}\n")
            
            for i, coord in enumerate(ecef_coordinates):
                f.write(f"{coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f}\n")
        
        print(f"Saved {len(ecef_coordinates)} ECEF points to: {ecef_filepath}")
    
    return filepath

def process_kml_folder(parent_folder, save_ecef=False):
    """
    Process KML file in the parent folder and extract placemarks.
    
    Args:
        parent_folder (str): Path to the parent folder containing the KML file
        save_ecef (bool): Whether to also save ECEF coordinates
        
    Returns:
        bool: True if successful, False otherwise
    """
    parent_path = Path(parent_folder)
    folder_name = parent_path.name
    
    # Find the KML file
    kml_file = parent_path / f"{folder_name}.kml"
    
    if not kml_file.exists():
        print(f"Error: KML file not found: {kml_file}")
        return False
    
    print(f"Processing KML file: {kml_file}")
    
    # Extract placemarks from KML
    placemarks = extract_placemarks_from_kml(kml_file)
    
    if not placemarks:
        print("No placemarks found in KML file")
        return False
    
    # Create geo_curves directory
    geo_curves_dir = parent_path / "geo_curves"
    
    # Save each placemark to a separate file
    saved_files = []
    for placemark in placemarks:
        try:
            filepath = save_placemark_to_file(placemark, geo_curves_dir, save_ecef)
            saved_files.append(filepath)
        except Exception as e:
            print(f"Error saving placemark {placemark['name']}: {e}")
    
    print(f"\nSuccessfully processed {len(saved_files)} placemarks from KML file")
    print(f"Files saved to: {geo_curves_dir}")
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Extract coordinate data from KML files and save as text files")
    parser.add_argument("parent_folder", help="Path to the parent folder containing the KML file")
    parser.add_argument("-e", "--ecef", action="store_true", 
                       help="Also save ECEF coordinates (x, y, z in meters)")
    
    args = parser.parse_args()
    
    # Check if parent folder exists
    if not os.path.exists(args.parent_folder):
        print(f"Error: Parent folder not found: {args.parent_folder}")
        return 1
    
    try:
        success = process_kml_folder(args.parent_folder, save_ecef=args.ecef)
        return 0 if success else 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main()) 