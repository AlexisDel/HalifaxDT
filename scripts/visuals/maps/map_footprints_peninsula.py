"""
Build an interactive QA map for the prepared Halifax peninsula footprints.

This script only creates the map. The footprint clipping itself is handled by
scripts/prepare_footprints_peninsula.py.
"""
from pathlib import Path

import folium
import geopandas as gpd


DATA_ROOT = Path(__file__).parents[3] / "data"
FOOTPRINTS = DATA_ROOT / "interim_data" / "footprints_peninsula_hrm" / "Building_Polygons_Peninsula.shp"
OUTPUT_HTML = DATA_ROOT / "visuals" / "maps" / "map_footprints_peninsula.html"
SIMPLIFY_TOLERANCE = 0.000006


def main() -> None:
    print("Loading peninsula footprints...")
    footprints = gpd.read_file(FOOTPRINTS).to_crs(4326)
    print(f"  Buildings loaded: {len(footprints):,}")

    bounds = footprints.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=14,
        tiles="CartoDB Positron",
        control_scale=True,
        prefer_canvas=True,
    )

    print("Adding footprint polygons to the map...")
    display = footprints[["BL_ID", "FCODE", "geometry"]].copy()
    display["geometry"] = display.geometry.simplify(SIMPLIFY_TOLERANCE, preserve_topology=True)
    display = display[~display.geometry.is_empty].copy()

    folium.GeoJson(
        data=display.__geo_interface__,
        name=f"HRM building footprints ({len(display):,})",
        style_function=lambda _feature: {
            "color": "steelblue",
            "weight": 0.45,
            "fillColor": "steelblue",
            "fillOpacity": 0.45,
        },
        popup=folium.GeoJsonPopup(
            fields=["BL_ID", "FCODE"],
            aliases=["BL_ID", "FCODE"],
            localize=True,
            labels=True,
        ),
    ).add_to(m)

    legend_html = f"""
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:14px 18px; border-radius:8px;
                border:1px solid #ccc; box-shadow:0 2px 6px rgba(0,0,0,0.2);
                font-family:Arial; font-size:13px;">
        <h4 style="margin:0 0 8px 0;">Building Footprints</h4>
        <p style="margin:0; font-size:11px; color:#666;">
            {len(footprints):,} buildings
        </p>
        <ul style="list-style:none; padding:0; margin:6px 0 0 0;">
            <li><span style="color:steelblue;">&#9632;</span> Building footprint (HRM)</li>
        </ul>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    folium.LayerControl(collapsed=False).add_to(m)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(OUTPUT_HTML)
    print(f"Map saved: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()


