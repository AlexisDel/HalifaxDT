# Halifax Data Setup

This folder is intentionally mostly empty in Git. It documents where each local dataset must be placed before running the pipeline.

Large source files, intermediate files, and regenerated visualizations are
ignored by Git. The publication-ready processed datasets and scene assets are
tracked in this repository.

## Folder Layout

```text
data/
|-- raw_data/          # Downloaded source datasets
|-- interim_data/      # Cropped/prepared inputs used by later scripts
|-- processed_data/    # Final processed datasets
|-- scenes/            # Generated Sionna XML scene and PLY meshes
`-- visuals/           # Generated maps and dashboards
```

## Source Data To Download

| Dataset | Source | Local path expected by scripts |
|---|---|---|
| HRM building footprints | [HRM Buildings](https://data-hrm.hub.arcgis.com/datasets/HRM::buildings-1/about?layer=0) | `raw_data/footprints_source_hrm/Building_Polygons.shp` |
| HRM LiDAR DSM 2018 1 m | [HRM LiDAR Resources](https://data-hrm.hub.arcgis.com/documents/c24cdc23a62e403c989cf6bdcd217ac2/explore), download `LiDAR DSM 1M 2018` | `raw_data/HRM_LiDAR_DSM_2018_1m_wgs84/HRM_LiDAR_DSM_2018_1m_wgs84.tif` |
| HRM LiDAR DEM 2018 5 m | [HRM LiDAR Resources](https://data-hrm.hub.arcgis.com/documents/c24cdc23a62e403c989cf6bdcd217ac2/explore), download `LiDAR DEM 5M 2018` | `raw_data/HRM_LiDAR_DEM_2018_5m_wgs84/HRM_LiDAR_DEM_2018_5m_wgs84.tif` |
| NSTDB building reference data | [Nova Scotia Geographic Data Directory](https://nsgi.novascotia.ca/gdd/), search `Nova Scotia Topographic Database - Buildings` | `raw_data/nstdb_shapefile/BL_POINT_10K.shp` and `raw_data/nstdb_shapefile/BL_POLY_10K.shp` |
| Cellular antenna records | [ISED SMS Data Downloads](https://ised-isde.canada.ca/site/spectrum-management-system/en/download-sms-data), download the terrestrial spectrum licence site data extract CSV (`Site_Data_Extract_FX.csv`) | `raw_data/Site_Data_Extract_FX.csv` |
| OSM buildings | [OpenStreetMap](https://www.openstreetmap.org/) via [Overpass Turbo](https://overpass-turbo.eu/) | `interim_data/osm_buildings_height_levels.geojson` |

### NSTDB Download Choice

In the Nova Scotia Geographic Data Directory, search for:

```text
Nova Scotia Topographic Database - Buildings
```

Prefer the newer datum package:

```text
NAD83(CSRS)v6, UTM Zone 20, CGVD2013
```

For that package, choose the `SHP` download option. The site may also offer `DXF` or `GDB`, but the preprocessing scripts expect shapefiles.

If that package is unavailable, the older package can still be used as a fallback:

```text
NAD83(CSRS)v3, UTM Zone 20, CGVD28
```

Again, click the `SHP` option for the download.

After downloading, extract the shapefile content into:

```text
data/raw_data/nstdb_shapefile/
```

The pipeline expects at least:

```text
BL_POINT_10K.shp
BL_POLY_10K.shp
```

## OSM Buildings

The OSM building file can be generated automatically from the repository root:

```bash
python scripts/preprocessing/download_osm_buildings.py
```

This calls the Overpass API and writes:

```text
data/interim_data/osm_buildings_height_levels.geojson
```

The query itself is kept in `interim_data/osm_buildings_halifax_peninsula.overpassql` so it can also be inspected or run manually in Overpass Turbo:

1. Open <https://overpass-turbo.eu/>.
2. Paste the query.
3. Run it.
4. Export the result as GeoJSON.
5. Save it as:

```text
data/interim_data/osm_buildings_height_levels.geojson
```

The export should include building geometry and tags such as `building`, `building:levels`, `height`, `roof:height`, `building:material`, `roof:material`, and `roof:shape` when available.

## Expected Data Layout After Download

```text
data/
|-- raw_data/
|   |-- footprints_source_hrm/
|   |   |-- Building_Polygons.shp
|   |   |-- Building_Polygons.dbf
|   |   |-- Building_Polygons.shx
|   |   `-- Building_Polygons.prj
|   |
|   |-- HRM_LiDAR_DSM_2018_1m_wgs84/
|   |   `-- HRM_LiDAR_DSM_2018_1m_wgs84.tif
|   |
|   |-- HRM_LiDAR_DEM_2018_5m_wgs84/
|   |   `-- HRM_LiDAR_DEM_2018_5m_wgs84.tif
|   |
|   |-- nstdb_shapefile/
|   |   |-- BL_POINT_10K.shp
|   |   |-- BL_POINT_10K.dbf
|   |   |-- BL_POINT_10K.shx
|   |   |-- BL_POINT_10K.prj
|   |   |-- BL_POLY_10K.shp
|   |   |-- BL_POLY_10K.dbf
|   |   |-- BL_POLY_10K.shx
|   |   `-- BL_POLY_10K.prj
|   |
|   `-- Site_Data_Extract_FX.csv
|
`-- interim_data/
    `-- osm_buildings_height_levels.geojson
```

Once all required datasets are downloaded and placed correctly, return to the main `README.md` at the repository root and follow the pipeline steps to continue building the Halifax digital twin.
