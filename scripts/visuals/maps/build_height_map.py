"""
Build an interactive HTML map from the selected Halifax building heights.

Input:
  data/processed_data/building_heights_selected.gpkg

Output:
  data/visuals/maps/building_heights_selected_map.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


def data_root() -> Path:
    return Path(__file__).parents[3] / "data"


def parse_args() -> argparse.Namespace:
    root = data_root()
    parser = argparse.ArgumentParser(description="Build an interactive map of selected building heights.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "processed_data" / "building_heights_selected.gpkg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "visuals" / "maps" / "building_heights_selected_map.html",
    )
    parser.add_argument("--layer", default="buildings")
    parser.add_argument("--simplify-m", type=float, default=0.4)
    return parser.parse_args()


def map_color(height: float) -> str:
    if not np.isfinite(height):
        return "#9ca3af"
    if height < 4:
        return "#2c7bb6"
    if height < 7:
        return "#00a6ca"
    if height < 10:
        return "#00ccbc"
    if height < 15:
        return "#90eb9d"
    if height < 25:
        return "#ffff8c"
    if height < 40:
        return "#f9d057"
    return "#d7191c"


def prepare_map_data(buildings: gpd.GeoDataFrame, simplify_m: float) -> gpd.GeoDataFrame:
    keep = [
        "BL_ID",
        "final_height_m",
        "roof_elevation_m",
        "final_height_source",
        "final_height_confidence",
        "height_selection_flag",
        "external_dsm_conflict",
        "dsm_shape_class",
        "dsm_h50_m",
        "dsm_h75_m",
        "dsm_h90_m",
        "dsm_h95_m",
        "dsm_h98_m",
        "candidate_osm_m",
        "candidate_nstdb_m",
        "osm_match_quality",
        "nstdb_match_method",
        "geometry",
    ]
    out = buildings[[column for column in keep if column in buildings.columns]].copy()
    if {"dsm_shape_class", "dsm_h75_m", "dsm_h95_m"}.issubset(out.columns):
        is_multi_volume = out["dsm_shape_class"].isin(["high_tail_or_multi_volume", "variable_roof"])
        out["candidate_dsm_dominant_m"] = np.where(is_multi_volume, out["dsm_h75_m"], out["dsm_h95_m"])
    if "dsm_h95_m" in out.columns:
        out["candidate_dsm_p95_m"] = out["dsm_h95_m"]
    if simplify_m > 0:
        metric_crs = out.estimate_utm_crs()
        out = out.to_crs(metric_crs)
        out["geometry"] = out.geometry.simplify(simplify_m, preserve_topology=True)
    out = out.to_crs(4326)
    out["map_color"] = out["final_height_m"].apply(map_color)
    out["is_low_confidence"] = out["final_height_confidence"].eq("low")
    for column in out.columns:
        if column != "geometry" and pd.api.types.is_numeric_dtype(out[column]):
            out[column] = out[column].round(2)
    return out


def write_height_map(out_path: Path, buildings: gpd.GeoDataFrame) -> None:
    bounds = buildings.total_bounds
    center_lat = float((bounds[1] + bounds[3]) / 2.0)
    center_lon = float((bounds[0] + bounds[2]) / 2.0)
    low_confidence_count = int(buildings["is_low_confidence"].sum()) if "is_low_confidence" in buildings.columns else 0
    building_count = int(len(buildings))
    source_counts = (
        buildings["final_height_source"]
        .fillna("missing")
        .astype(str)
        .replace("", "missing")
        .value_counts()
        .sort_index()
    )
    source_stats = [{"source": source, "count": int(count)} for source, count in source_counts.items()]
    source_stats_json = json.dumps(source_stats)
    geojson = buildings.to_json()
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Halifax Building Heights</title>
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
    .legend-row {{ display: flex; align-items: center; gap: 8px; margin: 3px 0; }}
    .swatch {{ width: 18px; height: 12px; border: 1px solid #6b7280; }}
    .source-filter {{ max-width: 270px; }}
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
    .source-row {{
      align-items: center;
      display: grid;
      gap: 7px;
      grid-template-columns: auto 1fr auto;
      margin: 4px 0;
      white-space: nowrap;
    }}
    .source-row input {{ margin: 0; }}
    .source-name {{ overflow: hidden; text-overflow: ellipsis; }}
    .source-count {{ color: #4b5563; font-variant-numeric: tabular-nums; }}
    .visible-count {{ border-top: 1px solid #e5e7eb; margin-top: 8px; padding-top: 7px; }}
    .building-search {{ min-width: 280px; padding: 9px; }}
    .building-search input {{
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      box-sizing: border-box;
      font-size: 14px;
      padding: 7px 8px;
      width: 100%;
    }}
    .search-results {{
      border-top: 1px solid #e5e7eb;
      margin-top: 7px;
      max-height: 230px;
      overflow-y: auto;
    }}
    .search-result {{
      border-bottom: 1px solid #f1f5f9;
      cursor: pointer;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 6px 2px;
    }}
    .search-result:hover {{ background: #f8fafc; }}
    .search-height {{ color: #4b5563; white-space: nowrap; }}
    .search-empty {{ color: #6b7280; padding-top: 7px; }}
    .popup-title {{ font-weight: 700; margin-bottom: 6px; }}
    .popup-grid {{ display: grid; grid-template-columns: auto auto; column-gap: 10px; row-gap: 2px; }}
    .popup-grid span:nth-child(odd) {{ color: #4b5563; }}
    .popup-section {{ border-top: 1px solid #e5e7eb; font-weight: 700; margin-top: 8px; padding-top: 6px; }}
  </style>
</head>
<body>
<div id="map"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
const buildings = {geojson};
const sourceStats = {source_stats_json};
const map = L.map('map', {{ preferCanvas: true }}).setView([{center_lat:.7f}, {center_lon:.7f}], 13);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  maxZoom: 20,
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO'
}}).addTo(map);

function value(v, suffix = '') {{
  return v === null || v === undefined || Number.isNaN(v) ? '' : `${{v}}${{suffix}}`;
}}

function isLowConfidence(feature) {{
  return feature.properties.is_low_confidence === true;
}}

function featureSource(feature) {{
  const source = feature.properties.final_height_source;
  return source === null || source === undefined || source === '' ? 'missing' : source;
}}

function popup(feature) {{
  const p = feature.properties;
  return `
    <div class="popup-title">Building ${{value(p.BL_ID)}}</div>
    <div class="popup-grid">
      <span>Height</span><b>${{value(p.final_height_m, ' m')}}</b>
      <span>Roof Z</span><b>${{value(p.roof_elevation_m, ' m')}}</b>
      <span>Source</span><b>${{value(p.final_height_source)}}</b>
      <span>Confidence</span><b>${{value(p.final_height_confidence)}}</b>
      <span>Flag</span><b>${{value(p.height_selection_flag)}}</b>
      <span>DSM conflict</span><b>${{value(p.external_dsm_conflict)}}</b>
      <span>DSM shape</span><b>${{value(p.dsm_shape_class)}}</b>
      <span class="popup-section">Source heights</span><span></span>
      <span>DSM p95</span><b>${{value(p.candidate_dsm_p95_m, ' m')}}</b>
      <span>DSM dominant</span><b>${{value(p.candidate_dsm_dominant_m, ' m')}}</b>
      <span>OSM</span><b>${{value(p.candidate_osm_m, ' m')}}</b>
      <span>NSTDB</span><b>${{value(p.candidate_nstdb_m, ' m')}}</b>
    </div>`;
}}

let activeSources = new Set(sourceStats.map(item => item.source));
let layer = null;

function layerStyle(feature) {{
  return {{
    color: isLowConfidence(feature) ? '#111827' : '#4b5563',
    dashArray: isLowConfidence(feature) ? '4 3' : null,
    weight: isLowConfidence(feature) ? 1.8 : 0.35,
    fillColor: feature.properties.map_color,
    fillOpacity: isLowConfidence(feature) ? 0.88 : 0.72
  }};
}}

function sourceIsVisible(feature) {{
  return activeSources.has(featureSource(feature));
}}

function visibleFeatureCount() {{
  return buildings.features.filter(sourceIsVisible).length;
}}

function refreshLayer(fit = false) {{
  if (layer !== null) {{
    map.removeLayer(layer);
  }}
  layer = L.geoJSON(buildings, {{
    filter: sourceIsVisible,
    style: layerStyle,
    onEachFeature: (feature, lyr) => {{
      lyr.bindPopup(popup(feature), {{ maxWidth: 340 }});
      lyr.on('mouseover', () => lyr.setStyle({{ weight: 2, fillOpacity: 0.92 }}));
      lyr.on('mouseout', () => layer.resetStyle(lyr));
    }}
  }}).addTo(map);
  if (fit && layer.getLayers().length > 0) {{
    map.fitBounds(layer.getBounds(), {{ padding: [20, 20] }});
  }}
  const count = document.getElementById('visible-source-count');
  if (count) {{
    count.textContent = `${{visibleFeatureCount()}} / {building_count}`;
  }}
}}

refreshLayer(true);
map.fitBounds(layer.getBounds(), {{ padding: [20, 20] }});

const buildingSearchIndex = new Map();
buildings.features.forEach(feature => {{
  const id = feature.properties.BL_ID;
  if (id === null || id === undefined || id === '') {{
    return;
  }}
  const key = String(id).toUpperCase();
  if (!buildingSearchIndex.has(key)) {{
    buildingSearchIndex.set(key, []);
  }}
  buildingSearchIndex.get(key).push(feature);
}});
const buildingSearchIds = Array.from(buildingSearchIndex.keys()).sort();
let selectedBuildingMarker = null;
let searchTimer = null;

function buildingGroupBounds(features) {{
  let bounds = null;
  features.forEach(feature => {{
    const featureBounds = L.geoJSON(feature).getBounds();
    bounds = bounds === null ? featureBounds : bounds.extend(featureBounds);
  }});
  return bounds;
}}

function selectBuildingById(id) {{
  const key = String(id).toUpperCase();
  const features = buildingSearchIndex.get(key);
  if (!features || features.length === 0) {{
    return;
  }}
  const bounds = buildingGroupBounds(features);
  if (!bounds || !bounds.isValid()) {{
    return;
  }}
  const center = bounds.getCenter();
  map.fitBounds(bounds, {{ padding: [80, 80], maxZoom: 19 }});
  if (selectedBuildingMarker !== null) {{
    map.removeLayer(selectedBuildingMarker);
  }}
  selectedBuildingMarker = L.marker(center, {{ title: key }})
    .addTo(map)
    .bindPopup(popup(features[0]), {{ maxWidth: 340 }})
    .openPopup();
}}

const searchControl = L.control({{ position: 'topleft' }});
searchControl.onAdd = function() {{
  const div = L.DomUtil.create('div', 'panel building-search');
  div.innerHTML = `
    <input type="search" id="building-id-search" placeholder="Search building ID" autocomplete="off">
    <div class="search-results" id="building-id-results"></div>`;
  L.DomEvent.disableClickPropagation(div);
  L.DomEvent.disableScrollPropagation(div);

  const input = div.querySelector('#building-id-search');
  const results = div.querySelector('#building-id-results');

  function renderResults(query) {{
    const normalized = query.trim().toUpperCase();
    results.innerHTML = '';
    if (!normalized) {{
      return;
    }}
    const matches = buildingSearchIds
      .filter(id => id.includes(normalized))
      .slice(0, 10);
    if (matches.length === 0) {{
      results.innerHTML = '<div class="search-empty">No building found</div>';
      return;
    }}
    results.innerHTML = matches.map(id => {{
      const feature = buildingSearchIndex.get(id)[0];
      const height = value(feature.properties.final_height_m, ' m');
      return `
        <div class="search-result" data-id="${{id}}">
          <b>${{id}}</b>
          <span class="search-height">${{height}}</span>
        </div>`;
    }}).join('');
  }}

  input.addEventListener('input', () => {{
    const query = input.value;
    renderResults(query);
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {{
      const key = query.trim().toUpperCase();
      if (buildingSearchIndex.has(key)) {{
        selectBuildingById(key);
      }}
    }}, 450);
  }});

  input.addEventListener('keydown', event => {{
    if (event.key === 'Enter') {{
      const key = input.value.trim().toUpperCase();
      if (buildingSearchIndex.has(key)) {{
        selectBuildingById(key);
        results.innerHTML = '';
      }}
    }}
  }});

  results.addEventListener('click', event => {{
    const row = event.target.closest('.search-result');
    if (!row) {{
      return;
    }}
    input.value = row.dataset.id;
    results.innerHTML = '';
    selectBuildingById(row.dataset.id);
  }});

  return div;
}};
searchControl.addTo(map);

const sourceControl = L.control({{ position: 'topright' }});
sourceControl.onAdd = function() {{
  const div = L.DomUtil.create('div', 'panel source-filter');
  const rows = sourceStats.map(item => `
    <label class="source-row" title="${{item.source}}">
      <input type="checkbox" data-source="${{item.source}}" checked>
      <span class="source-name">${{item.source}}</span>
      <span class="source-count">${{item.count}}</span>
    </label>`).join('');
  div.innerHTML = `
    <div class="filter-title">Filter by source</div>
    <div class="filter-actions">
      <button type="button" data-action="all">All</button>
      <button type="button" data-action="none">None</button>
    </div>
    ${{rows}}
    <div class="visible-count">Visible: <b id="visible-source-count">{building_count} / {building_count}</b></div>`;
  L.DomEvent.disableClickPropagation(div);
  L.DomEvent.disableScrollPropagation(div);

  function setSource(source, checked) {{
    if (checked) {{
      activeSources.add(source);
    }} else {{
      activeSources.delete(source);
    }}
  }}

  div.querySelectorAll('input[data-source]').forEach(input => {{
    input.addEventListener('change', () => {{
      setSource(input.dataset.source, input.checked);
      refreshLayer(false);
    }});
  }});
  div.querySelectorAll('button[data-action]').forEach(button => {{
    button.addEventListener('click', () => {{
      const checked = button.dataset.action === 'all';
      activeSources = checked ? new Set(sourceStats.map(item => item.source)) : new Set();
      div.querySelectorAll('input[data-source]').forEach(input => {{
        input.checked = checked;
      }});
      refreshLayer(false);
    }});
  }});
  return div;
}};
sourceControl.addTo(map);

const legend = L.control({{ position: 'bottomright' }});
legend.onAdd = function() {{
  const div = L.DomUtil.create('div', 'panel');
  div.innerHTML = `
    <b>Final height</b>
    <div class="legend-row"><span class="swatch" style="background:#2c7bb6"></span>&lt; 4 m</div>
    <div class="legend-row"><span class="swatch" style="background:#00a6ca"></span>4-7 m</div>
    <div class="legend-row"><span class="swatch" style="background:#00ccbc"></span>7-10 m</div>
    <div class="legend-row"><span class="swatch" style="background:#90eb9d"></span>10-15 m</div>
    <div class="legend-row"><span class="swatch" style="background:#ffff8c"></span>15-25 m</div>
    <div class="legend-row"><span class="swatch" style="background:#f9d057"></span>25-40 m</div>
    <div class="legend-row"><span class="swatch" style="background:#d7191c"></span>&ge; 40 m</div>
    <div style="border-top:1px solid #e5e7eb; margin-top:7px; padding-top:7px">
      <div class="legend-row"><span class="swatch" style="background:#fff; border:2px dashed #111827"></span>Low confidence: {low_confidence_count} / {building_count}</div>
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
    write_height_map(args.output, map_data)
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
