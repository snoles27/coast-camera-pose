import geopandas as gpd
import matplotlib.pyplot as plt

# Path to the high-resolution GSHHG shapefile (change as needed)
shapefile_path = 'gshhg-shp-2/GSHHS_shp/h/GSHHS_h_L1.shp'

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Optional: filter to a region (e.g., Australia)
bbox = (110, -45, 155, -10)
gdf_aus = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

# Plot
ax = gdf_aus.plot(figsize=(10, 8), color='blue')
plt.title('GSHHG High-Resolution Coastline (Australia)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# # Print the first few coordinates of the first geometry
# for geom in gdf_aus.geometry:
#     if geom.geom_type == 'LineString':
#         print(list(geom.coords)[:5])
#     elif geom.geom_type == 'MultiLineString':
#         for line in geom:
#             print(list(line.coords)[:5])

