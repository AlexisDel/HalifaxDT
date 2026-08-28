"""
Build an interactive HTML map of Halifax building material classes.

Input:
  data/processed_data/building_materials_selected.gpkg

Output:
  data/visuals/maps/building_materials_selected_map.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import pandas as pd


MATERIAL_COLORS = {
    "concrete": "#8f9aa6",
    "brick": "#b5523b",
    "wood": "#c7925b",
    "metal": "#55a6b8",
    "glass": "#6ec7dd",
    "missing": "#9ca3af",
}


def data_root() -> Path:
    return Path(__file__).parents[3] / "data"


def parse_args() -> argparse.Namespace:
    root = data_root()
    parser = argparse.ArgumentParser(description="Build an interactive map of selected building materials.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "processed_data" / "building_materials_selected.gpkg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "visuals" / "maps" / "building_materials_selected_map.html",
    )
    parser.add_argument("--layer", default="buildings")
    parser.add_argument("--simplify-m", type=float, default=0.4)
    return parser.parse_args()


def material_color(material: object) -> str:
    key = "missing" if material is None or str(material).strip() == "" else str(material).strip().lower()
    return MATERIAL_COLORS.get(key, MATERIAL_COLORS["missing"])


def prepare_map_data(buildings: gpd.GeoDataFrame, simplify_m: float) -> gpd.GeoDataFrame:
    keep = [
        "BL_ID",
        "wall_material",
        "roof_material",
        "material_source",
        "material_confidence",
        "material_confidence_score",
        "material_reason",
        "material_zone",
        "final_height_m",
        "footprint_area_m2",
        "footprint_compactness",
        "footprint_elongation",
        "osm_material",
        "osm_roof_material",
        "osm_building_type",
        "osm_overlap_ratio",
        "score_concrete",
        "score_brick",
        "score_wood",
        "score_metal",
        "score_glass",
        "geometry",
    ]
    out = buildings[[column for column in keep if column in buildings.columns]].copy()
    if simplify_m > 0:
        metric_crs = out.estimate_utm_crs()
        out = out.to_crs(metric_crs)
        out["geometry"] = out.geometry.simplify(simplify_m, preserve_topology=True)
    out = out.to_crs(4326)
    out["map_color"] = out["wall_material"].apply(material_color)
    for column in out.columns:
        if column != "geometry" and pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].round(3)
    return out


def normalized(value: object) -> str:
    if value is None or str(value).strip() == "":
        return "missing"
    return str(value)


def write_material_map(out_path: Path, buildings: gpd.GeoDataFrame) -> None:
    bounds = buildings.total_bounds
    center_lat = float((bounds[1] + bounds[3]) / 2.0)
    center_lon = float((bounds[0] + bounds[2]) / 2.0)
    building_count = int(len(buildings))

    material_counts = (
        buildings["wall_material"]
        .fillna("missing")
        .astype(str)
        .replace("", "missing")
        .value_counts()
        .sort_index()
    )
    confidence_counts = (
        buildings["material_confidence"]
        .fillna("missing")
        .astype(str)
        .replace("", "missing")
        .value_counts()
        .sort_index()
    )
    material_stats = [{"material": material, "count": int(count)} for material, count in material_counts.items()]
    confidence_stats = [{"confidence": confidence, "count": int(count)} for confidence, count in confidence_counts.items()]

    geojson = buildings.to_json()
    material_stats_json = json.dumps(material_stats)
    confidence_stats_json = json.dumps(confidence_stats)
    color_json = json.dumps(MATERIAL_COLORS)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Halifax Building Materials</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <style>
    html, body, #map {{ height: 100%; margin: 0; }}
    body {{ background: #f8fafc; font-family: Arial, sans-serif; }}
    .panel {{
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      box-shadow: 0 2px 12px rgba(0,0,0,.16);
      color: #111827;
      line-height: 1.35;
      padding: 10px 12px;
    }}
    .filter {{ max-width: 290px; }}
    .filter-title {{ font-weight: 700; margin-bottom: 7px; }}
    .filter-actions {{ display: flex; gap: 6px; margin: 8px 0; }}
    .filter-actions button {{
      background: #f3f4f6;
      border: 1px solid #d1d5db;
      border-radius: 4px;
      color: #111827;
      cursor: pointer;
      font-size: 12px;
      padding: 3px 7px;
    }}
    .filter-row {{
      align-items: center;
      display: grid;
      gap: 7px;
      grid-template-columns: auto auto 1fr auto;
      margin: 4px 0;
      white-space: nowrap;
    }}
    .filter-row input {{ margin: 0; }}
    .source-name {{ overflow: hidden; text-overflow: ellipsis; }}
    .count {{ color: #4b5563; font-variant-numeric: tabular-nums; }}
    .swatch {{ width: 18px; height: 12px; border: 1px solid #6b7280; display: inline-block; }}
    .visible-count {{ border-top: 1px solid #e5e7eb; margin-top: 8px; padding-top: 7px; }}
    .popup-title {{ font-weight: 700; margin-bottom: 6px; }}
    .popup-grid {{ display: grid; grid-template-columns: auto auto; column-gap: 10px; row-gap: 2px; }}
    .popup-grid span:nth-child(odd) {{ color: #4b5563; }}
    .popup-section {{ border-top: 1px solid #e5e7eb; font-weight: 700; margin-top: 8px; padding-top: 6px; }}
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
  </style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const buildings = {geojson};
const materialStats = {material_stats_json};
const confidenceStats = {confidence_stats_json};
const materialColors = {color_json};
const map = L.map('map', {{ preferCanvas: true }}).setView([{center_lat:.7f}, {center_lon:.7f}], 13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 20,
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
}}).addTo(map);

function value(v, suffix = '') {{
  return v === null || v === undefined || Number.isNaN(v) || v === '' ? '' : `${{v}}${{suffix}}`;
}}

function materialOf(feature) {{
  const material = feature.properties.wall_material;
  return material === null || material === undefined || material === '' ? 'missing' : material;
}}

function popup(feature) {{
  const p = feature.properties;
  return `
    <div class="popup-title">Building ${{value(p.BL_ID)}}</div>
    <div class="popup-grid">
      <span>Wall material</span><b>${{value(p.wall_material)}}</b>
      <span>Roof material</span><b>${{value(p.roof_material)}}</b>
      <span>Source</span><b>${{value(p.material_source)}}</b>
      <span>Confidence</span><b>${{value(p.material_confidence)}}</b>
      <span>Confidence score</span><b>${{value(p.material_confidence_score)}}</b>
      <span>Zone</span><b>${{value(p.material_zone)}}</b>
      <span>Height</span><b>${{value(p.final_height_m, ' m')}}</b>
      <span>Area</span><b>${{value(p.footprint_area_m2, ' m2')}}</b>
      <span>Compactness</span><b>${{value(p.footprint_compactness)}}</b>
      <span>Elongation</span><b>${{value(p.footprint_elongation)}}</b>
      <span>OSM material</span><b>${{value(p.osm_material)}}</b>
      <span>OSM type</span><b>${{value(p.osm_building_type)}}</b>
      <span class="popup-section">Scores</span><span></span>
      <span>Concrete</span><b>${{value(p.score_concrete)}}</b>
      <span>Brick</span><b>${{value(p.score_brick)}}</b>
      <span>Wood</span><b>${{value(p.score_wood)}}</b>
      <span>Metal</span><b>${{value(p.score_metal)}}</b>
      <span>Glass</span><b>${{value(p.score_glass)}}</b>
      <span class="popup-section">Reason</span><span></span>
      <span></span><b>${{value(p.material_reason)}}</b>
    </div>`;
}}

let activeMaterials = new Set(materialStats.map(item => item.material));
let layer = null;

function isVisible(feature) {{
  return activeMaterials.has(materialOf(feature));
}}

function visibleFeatureCount() {{
  return buildings.features.filter(isVisible).length;
}}

function layerStyle(feature) {{
  const material = materialOf(feature);
  return {{
    color: '#374151',
    weight: 0.35,
    fillColor: feature.properties.map_color || materialColors[material] || materialColors.missing,
    fillOpacity: 0.76
  }};
}}

function refreshLayer(fit = false) {{
  if (layer !== null) {{
    map.removeLayer(layer);
  }}
  layer = L.geoJSON(buildings, {{
    filter: isVisible,
    style: layerStyle,
    onEachFeature: (feature, lyr) => {{
      lyr.bindPopup(popup(feature), {{ maxWidth: 390 }});
      lyr.on('mouseover', () => lyr.setStyle({{ weight: 2, fillOpacity: 0.93 }}));
      lyr.on('mouseout', () => layer.resetStyle(lyr));
    }}
  }}).addTo(map);
  if (fit && layer.getLayers().length > 0) {{
    map.fitBounds(layer.getBounds(), {{ padding: [20, 20] }});
  }}
  const count = document.getElementById('visible-count');
  if (count) {{
    count.textContent = `${{visibleFeatureCount()}} / {building_count}`;
  }}
}}

refreshLayer(true);

const filter = L.control({{ position: 'topright' }});
filter.onAdd = function() {{
  const div = L.DomUtil.create('div', 'panel filter');
  const rows = materialStats.map(item => `
    <label class="filter-row" title="${{item.material}}">
      <input type="checkbox" data-material="${{item.material}}" checked>
      <span class="swatch" style="background:${{materialColors[item.material] || materialColors.missing}}"></span>
      <span class="source-name">${{item.material}}</span>
      <span class="count">${{item.count}}</span>
    </label>`).join('');
  div.innerHTML = `
    <div class="filter-title">Wall material</div>
    <div class="filter-actions">
      <button type="button" data-action="all">All</button>
      <button type="button" data-action="none">None</button>
    </div>
    ${{rows}}
    <div class="visible-count">Visible: <b id="visible-count">{building_count} / {building_count}</b></div>`;
  L.DomEvent.disableClickPropagation(div);
  L.DomEvent.disableScrollPropagation(div);

  div.querySelectorAll('input[data-material]').forEach(input => {{
    input.addEventListener('change', () => {{
      if (input.checked) {{
        activeMaterials.add(input.dataset.material);
      }} else {{
        activeMaterials.delete(input.dataset.material);
      }}
      refreshLayer(false);
    }});
  }});
  div.querySelectorAll('button[data-action]').forEach(button => {{
    button.addEventListener('click', () => {{
      const checked = button.dataset.action === 'all';
      activeMaterials = checked ? new Set(materialStats.map(item => item.material)) : new Set();
      div.querySelectorAll('input[data-material]').forEach(input => {{
        input.checked = checked;
      }});
      refreshLayer(false);
    }});
  }});
  return div;
}};
filter.addTo(map);

const legend = L.control({{ position: 'bottomright' }});
legend.onAdd = function() {{
  const div = L.DomUtil.create('div', 'panel');
  const materialRows = materialStats.map(item => `
    <div class="legend-row">
      <span class="swatch" style="background:${{materialColors[item.material] || materialColors.missing}}"></span>
      <span>${{item.material}}: <b>${{item.count}}</b></span>
    </div>`).join('');
  const confidenceRows = confidenceStats.map(item => `
    <div>${{item.confidence}}: <b>${{item.count}}</b></div>`).join('');
  div.innerHTML = `
    <b>Wall materials</b>
    ${{materialRows}}
    <div style="border-top:1px solid #e5e7eb; margin-top:7px; padding-top:7px">
      <b>Confidence</b>
      ${{confidenceRows}}
    </div>`;
  return div;
}};
legend.addTo(map);
</script>
</body>
</html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    buildings = gpd.read_file(args.input, layer=args.layer)
    map_data = prepare_map_data(buildings, args.simplify_m)
    write_material_map(args.output, map_data)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
