"""Clip HRM building footprints to the Halifax peninsula study area."""
import geopandas as gpd
from shapely.geometry import Polygon
from pathlib import Path

DATA_ROOT = Path(__file__).parents[2] / "data"
INPUT = DATA_ROOT / "raw_data" / "footprints_source_hrm" / "Building_Polygons.shp"
OUTPUT_DIR = DATA_ROOT / "interim_data" / "footprints_peninsula_hrm"

# Same peninsula polygon used by the antenna filter.
peninsula = Polygon([
    (-63.6194204, 44.6410835),
    (-63.6308541, 44.6643012),
    (-63.6220173, 44.6817374),
    (-63.5992474, 44.6736948),
    (-63.5531831, 44.6413834),
    (-63.5565099, 44.6169733),
    (-63.5641958, 44.6133803),
    (-63.5844221, 44.6253175),
    (-63.6194298, 44.6410626),
    (-63.6194204, 44.6410835),
])

# --- 1. Load and clip the shapefile ---
print("Loading the full HRM shapefile...")
gdf = gpd.read_file(INPUT)
print(f"  Total HRM buildings: {len(gdf):,}")

# Reproject to WGS84 so the hand-drawn lon/lat polygon can be used directly.
gdf_wgs = gdf.to_crs(epsg=4326)

# Keep buildings whose centroid falls inside the peninsula polygon.
print("Filtering to the Halifax peninsula...")
gdf_wgs['in_peninsula'] = gdf_wgs.geometry.centroid.within(peninsula)
gdf_peninsula = gdf_wgs[gdf_wgs['in_peninsula']].drop(columns=['in_peninsula'])

print(f"  Buildings on the peninsula: {len(gdf_peninsula):,}")
print(f"  Reduction: {100 - len(gdf_peninsula)/len(gdf)*100:.1f}% removed")

# Save the clipped shapefile for the rest of the preprocessing pipeline.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_shp = OUTPUT_DIR / "Building_Polygons_Peninsula.shp"
gdf_peninsula.to_file(output_shp)
print(f"\nShapefile saved: {output_shp}")

