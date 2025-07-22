import geopandas as gpd
import matplotlib.pyplot as plt

# Path to the high-resolution GSHHG shapefile (change as needed)
shapefile_path = 'gshhg-shp-2/GSHHS_shp/h/GSHHS_h_L1.shp'

# Read the shapefile
gdf = gpd.read_file(shapefile_path)

# Optional: filter to a region (e.g., Australia)
bbox = (110, -45, 155, -10)
#bbox = (133.3, -32.4, 133.8, -32.15)
gdf_aus = gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]]

# Plot
ax = gdf_aus.plot(figsize=(10, 8), color='blue')
plt.title('GSHHG High-Resolution Coastline (Australia)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Print the first few coordinates of the first geometry
print(type(gdf_aus.geometry))
coord_list = []
for geom in gdf_aus.geometry[:5]:
    print(geom)
    # coord_list.append(geom.coords)

print(coord_list)