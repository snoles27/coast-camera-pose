# Camera Location Estimation Tool

A Python tool for estimating camera position and orientation using photo curves and geographic curves. The system uses fisheye lens calibration and PnP (Perspective-n-Point) algorithms to determine where a camera was located when a photo was taken.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run camera location estimation
python locate_camera.py frames/w3_full_low_f47533

# Run tests
pytest test_locate_camera.py
```

## Main Tool: `locate_camera.py`

The primary tool for estimating camera position from photo and geographic curves.

### Usage

```bash
python locate_camera.py <frame_directory> [options]
```

### Arguments

- `frame_directory`: Path to the frame directory containing photo_curves and geo_curves
- `-c, --curves`: Specify which curves to use (e.g., "a" for curveA, "ab" for curveA and curveB). Default: use all available curves
- `-n, --num_points`: Number of points N to use in SolvePnP (default: 300)

### Examples

```bash
# Use all available curves with default settings
python locate_camera.py frames/w3_full_low_f47533

# Use only curveA with N=500
python locate_camera.py frames/w3_full_low_f47533 -c a -n 500

# Use curveA and curveB with N=200
python locate_camera.py frames/w3_full_low_f47533 -c ab -n 200
```

## Manual Generation Tools

### 1. `kml_to_curves.py` - Convert KML files to curve files

Extracts coordinate data from KML placemarks and saves them as text files. The KML files are exports from Google Earth's path drawing tool.

```bash
python manual_generation/kml_to_curves.py <kml_file> [options]
```

**Arguments:**
- `kml_file`: Path to the KML file
- `-o, --output`: Output directory (default: same directory as KML file)
- `-e, --ecef`: Also save ECEF coordinate versions

**Example:**
```bash
python manual_generation/kml_to_curves.py frames/w3_full_low_f47533/w3_full_low_f47533.kml -e
```

### 2. `generate_photo_curves.py` - Create photo curves by clicking on images

Interactive tool for manually selecting points on detected edges within photos. Program outputs the selected points in text files.

```bash
python manual_generation/generate_photo_curves.py <image_path> <output_path> [options]
```

**Arguments:**
- `image_path`: Path to the photo image
- `output_path`: Path to save the curve file
- `-d, --description`: Description for the curve. Inserted in file as a commented header.
- `-t, --threshold`: Edge detection threshold (default: 50)
- `-r, --radius`: Snap radius to edge for point selection (default: 10)

**Example:**
```bash
python manual_generation/generate_photo_curves.py frames/w3_full_low_f47533/w3_full_low_f47533.jpg frames/w3_full_low_f47533/photo_curves/curveA_island_1 -d "Island coastline"
```

**Usage:**
1. Click on the image to select points along the curve
2. The tool will snap to detected edges automatically
3. Press Enter to finish and save the curve

### 3. `rescale_photo.py` - Rescale photo curves to different resolutions

Rescales photo curve coordinates to match different camera resolutions.

```bash
python manual_generation/rescale_photo.py <photo_curve_file> <original_resolution> <native_resolution>
```

**Arguments:**
- `photo_curve_file`: Path to the photo curve file
- `original_resolution`: Original photo resolution as [width,height]
- `native_resolution`: Target camera resolution as [width,height]

**Example:**
```bash
python manual_generation/rescale_photo.py frames/w3_full_low_f47533/photo_curves/curveA_island_1 "[1280,720]" "[1920,1080]"
```

### 4. `curve_photo_overlay.py` - Visualize curves overlaid on photos

Displays all curve files overlaid on the original photo for verification.

```bash
python manual_generation/curve_photo_overlay.py <frame_directory> [options]
```

**Arguments:**
- `frame_directory`: Path to the frame directory containing the photo and photo_curves
- `-s, --save`: Path to save the overlay image

**Example:**
```bash
python manual_generation/curve_photo_overlay.py frames/w3_full_low_f47533
```

**What it does:**
1. Automatically finds the photo file in the frame directory
2. Loads all curve files from the `photo_curves` subdirectory
3. Displays each curve overlaid on the photo with different colors
4. Shows curve names and point counts

## Testing

Run the test suite using pytest:

```bash
# Run all tests
pytest test_locate_camera.py

# Run with verbose output
pytest test_locate_camera.py -v

# Run specific test
pytest test_locate_camera.py::test_specific_function -v
```

## Frame Directory Structure

Each frame directory must follow this structure:

```
frames/
└── your_frame_name/
    ├── lens_profile.json                            # Camera lens profile configuration
    ├── your_frame_name.jpg                          # Original photo
    ├── your_frame_name.kml                          # KML file with geographic data
    ├── photo_curves/                                # Photo curve files
    │   ├── curveA_your_description                  # Photo curve A
    │   ├── curveA_your_description_rescaled         # Used if photo curve isn't in original camera resolution
    │   ├── curveB_your_description                  # Photo curve B
    │   └── curveB_your_description_rescaled         # Used if photo curve isn't in original camera resolution
    └── geo_curves/                                  # Geographic curve files
        ├── curveA_your_description                  # Lat/lon coordinates
        ├── curveA_your_description_ecef             # ECEF coordinates (required for PnP)
        ├── curveB_your_description                  # Lat/lon coordinates
        └── curveB_your_description_ecef             # ECEF coordinates (required for PnP)
```

### Required Files

1. **`lens_profile.json`** - Camera lens profile configuration:
   ```json
   {
       "lens_profile_path": "gyroflow_lens_profiles/Sony/Sony_a7sIII_Sigma 24-70mm 2.8 Art__4k_16by9_3840x2160-29.97fps.json"
   }
   ```

2. **Photo curves** - Text files with pixel coordinates (x, y):
   - **_rescaled**: must match camera resolution in lens_profile.json
   ```
   # Image: frames/your_frame_name/your_frame_name.jpg
   # Number of points: 51
   897.000000, 172.000000
   895.000000, 173.000000
   ...
   ```

3. **Geo curves** - Text files with geographic coordinates:
   - **Lat/lon format** (for display): `latitude, longitude, altitude`
   - **ECEF format** (for PnP): `x, y, z` in ECEF coordinates

### Curve Naming Convention

- Photo curves: `curveA_*`, `curveB_*`, etc.
- Geo curves: Same prefix as photo curves
- ECEF curves: Must end with `_ecef`
- Rescaled photo curves: Preferred if available (ends with `_rescaled`). If not available, assumes normal photo curve file is correct resolution for SolvePnP.

## Dependencies

- Python 3.7+
- numpy
- matplotlib
- opencv-python
- quaternion
- pyproj
- geopandas
- scipy

## Lens Profiles

The system uses Gyroflow lens profiles for camera calibration. Profiles are stored in `gyroflow_lens_profiles/` and referenced in each frame's `lens_profile.json` file.

## Output

The tool outputs:
- Estimated camera position in ECEF coordinates
- Camera position in latitude/longitude/altitude
- Interactive plot showing camera position on a map with geographic curves
- Console output with detailed results

## Troubleshooting

1. **No matching curves found**: Ensure photo and geo curves have matching prefixes (curveA, curveB, etc.)
2. **Lens profile not found**: Check the path in `lens_profile.json` and ensure the file exists
3. **ECEF curves missing**: Geo curves for PnP must end with `_ecef`
