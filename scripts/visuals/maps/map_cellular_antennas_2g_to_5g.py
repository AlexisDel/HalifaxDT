"""
Build an interactive map of Halifax peninsula 2G-to-5G cellular tower locations.

Input:
  data/interim_data/peninsula_cellular_antennas_2g_to_5g.csv

Output:
  data/visuals/maps/map_cellular_antennas_2g_to_5g.html
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import folium
import pandas as pd
from shapely.geometry import Polygon


DATA_ROOT = Path(__file__).parents[3] / "data"
INPUT_CSV = DATA_ROOT / "interim_data" / "peninsula_cellular_antennas_2g_to_5g.csv"
OUTPUT_HTML = DATA_ROOT / "visuals" / "maps" / "map_cellular_antennas_2g_to_5g.html"

PENINSULA = Polygon(
    [
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
    ]
)

SERVICE_COLORS = {
    "CELL": "green",
    "PCS": "blue",
    "PCSG": "cadetblue",
    "AWS": "orange",
    "AWS-3": "darkred",
    "BRS": "purple",
    "MBS": "darkblue",
    "600B": "darkgreen",
    "3500B": "red",
}


def value(row: pd.Series, column: str, suffix: str = "") -> str:
    item = row.get(column, "")
    if pd.isna(item) or str(item).strip() == "":
        return ""
    return f"{item}{suffix}"


def compact_unique(group: pd.DataFrame, column: str, limit: int = 6) -> str:
    if column not in group.columns:
        return ""

    values = [
        str(item).strip()
        for item in group[column].dropna().unique()
        if str(item).strip()
    ]
    if not values:
        return ""

    visible = values[:limit]
    suffix = f" (+{len(values) - limit})" if len(values) > limit else ""
    return ", ".join(visible) + suffix


def counts_text(group: pd.DataFrame, column: str) -> str:
    counts = group[column].fillna("missing").astype(str).value_counts().sort_index()
    return ", ".join(f"{name}: {count}" for name, count in counts.items())


def tower_popup_html(group: pd.DataFrame, tower_index: int) -> str:
    first = group.iloc[0]
    summary_rows = [
        ("Location", compact_unique(group, "LOCATION", limit=3)),
        ("Licensees", compact_unique(group, "LICENSEE", limit=5)),
        ("Services", counts_text(group, "SERVICE")),
        ("Technologies", counts_text(group, "TECHNOLOGY")),
        ("Antenna rows", f"{len(group):,}"),
        ("Latitude", f"{first['LATITUDE']}"),
        ("Longitude", f"{first['LONGITUDE']}"),
        ("DEM ground elevation", value(first, "DEM_GROUND_ELEV_M", " m")),
        ("Structure height", value(first, "STUCT_HT", " m")),
    ]
    summary_body = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{escape(text)}</td></tr>"
        for label, text in summary_rows
        if text
    )

    detail_rows = []
    sorted_group = group.sort_values(
        ["SERVICE", "TECHNOLOGY", "LICENSEE", "TRANSMIT_FREQ", "TX_ANT_AZIM"],
        na_position="last",
    )
    for index, row in sorted_group.iterrows():
        detail_rows.append(
            f"""
            <tr>
              <td>{index + 1}</td>
              <td>{escape(value(row, "SERVICE"))}</td>
              <td>{escape(value(row, "TECHNOLOGY"))}</td>
              <td>{escape(value(row, "LICENSEE"))}</td>
              <td>{escape(value(row, "TRANSMIT_FREQ", " MHz"))}</td>
              <td>{escape(value(row, "TX_PWR", " W"))}</td>
              <td>{escape(value(row, "TX_ANT_HT", " m"))}</td>
              <td>{escape(value(row, "ANTENNA_Z_M", " m"))}</td>
              <td>{escape(value(row, "TX_ANT_AZIM", " deg"))}</td>
              <td>{escape(value(row, "TX_ANT_ELEV_ANGLE", " deg"))}</td>
              <td>{escape(value(row, "TX_ANT_GAIN", " dBi"))}</td>
            </tr>
            """
        )

    return f"""
    <div style="font-family:Arial,sans-serif; min-width:760px; max-width:960px;">
      <h4 style="margin:0 0 8px 0;">Cellular tower #{tower_index}</h4>
      <table class="tower-summary" style="border-collapse:collapse; font-size:12px; margin-bottom:10px;">
        {summary_body}
      </table>
      <div style="max-height:340px; overflow:auto; border-top:1px solid #e5e7eb; padding-top:8px;">
        <table class="tower-details" style="border-collapse:collapse; font-size:11px; width:100%;">
          <thead>
            <tr>
              <th>#</th>
              <th>Service</th>
              <th>Tech</th>
              <th>Licensee</th>
              <th>TX freq</th>
              <th>TX power</th>
              <th>Ant h</th>
              <th>Ant Z</th>
              <th>Azimuth</th>
              <th>Tilt</th>
              <th>Gain</th>
            </tr>
          </thead>
          <tbody>{"".join(detail_rows)}</tbody>
        </table>
      </div>
    </div>
    """


def add_peninsula_boundary(m: folium.Map) -> None:
    peninsula_latlng = [(lat, lon) for lon, lat in PENINSULA.exterior.coords]
    folium.Polygon(
        locations=peninsula_latlng,
        color="#111827",
        weight=2,
        fill=True,
        fill_color="#111827",
        fill_opacity=0.04,
        name="Halifax peninsula boundary",
    ).add_to(m)


def add_summary_panel(m: folium.Map, antennas: pd.DataFrame) -> None:
    service_counts = antennas["SERVICE"].fillna("missing").astype(str).value_counts().sort_index()
    technology_counts = antennas["TECHNOLOGY"].fillna("missing").astype(str).value_counts().sort_index()
    licensee_counts = antennas["LICENSEE"].fillna("missing").astype(str).value_counts().head(6)
    tower_count = antennas[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]

    service_rows = "".join(
        f'<li class="antenna-summary-row"><span>{escape(service)}</span><b>{count:,}</b></li>'
        for service, count in service_counts.items()
    )
    technology_rows = "".join(
        f'<li class="antenna-summary-row"><span>{escape(technology)}</span><b>{count:,}</b></li>'
        for technology, count in technology_counts.items()
    )
    licensee_rows = "".join(
        f'<li class="antenna-summary-row"><span>{escape(licensee)}</span><b>{count:,}</b></li>'
        for licensee, count in licensee_counts.items()
    )

    panel = f"""
    <div style="
      position: fixed;
      bottom: 24px;
      left: 24px;
      z-index: 9999;
      background: white;
      border: 1px solid #d1d5db;
      border-radius: 6px;
      box-shadow: 0 2px 12px rgba(0,0,0,.18);
      color: #111827;
      font-family: Arial, sans-serif;
      font-size: 12px;
      line-height: 1.35;
      max-width: 320px;
      padding: 10px 12px;
    ">
      <b>2G-to-5G cellular antennas</b><br>
      Antenna rows: {len(antennas):,}<br>
      Tower locations: {tower_count:,}
      <div style="border-top:1px solid #e5e7eb; margin-top:7px; padding-top:7px;">
        <b>Services</b>
        <ul style="list-style:none; padding:0; margin:5px 0 0 0;">
          {service_rows}
        </ul>
      </div>
      <div style="border-top:1px solid #e5e7eb; margin-top:7px; padding-top:7px;">
        <b>Technologies</b>
        <ul style="list-style:none; padding:0; margin:5px 0 0 0;">
          {technology_rows}
        </ul>
      </div>
      <div style="border-top:1px solid #e5e7eb; margin-top:7px; padding-top:7px;">
        <b>Top licensees</b>
        <ul style="list-style:none; padding:0; margin:5px 0 0 0;">
          {licensee_rows}
        </ul>
      </div>
    </div>
    <style>
      .antenna-summary-row {{ display:flex; justify-content:space-between; gap:14px; }}
      .leaflet-popup-content .tower-summary th {{ color:#4b5563; font-weight:normal; padding:2px 8px 2px 0; text-align:left; vertical-align:top; }}
      .leaflet-popup-content .tower-summary td {{ color:#111827; font-weight:bold; padding:2px 0; vertical-align:top; }}
      .leaflet-popup-content .tower-details th {{ background:#f3f4f6; color:#374151; font-weight:bold; padding:4px 6px; position:sticky; top:0; text-align:left; }}
      .leaflet-popup-content .tower-details td {{ border-bottom:1px solid #e5e7eb; padding:3px 6px; vertical-align:top; }}
    </style>
    """
    m.get_root().html.add_child(folium.Element(panel))


def dominant_service(group: pd.DataFrame) -> str:
    counts = group["SERVICE"].fillna("missing").astype(str).value_counts()
    return str(counts.index[0]) if not counts.empty else "missing"


def main() -> None:
    antennas = pd.read_csv(INPUT_CSV)
    antennas["LATITUDE"] = pd.to_numeric(antennas["LATITUDE"], errors="coerce")
    antennas["LONGITUDE"] = pd.to_numeric(antennas["LONGITUDE"], errors="coerce")
    antennas = antennas.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()

    center_lat = float(antennas["LATITUDE"].mean())
    center_lon = float(antennas["LONGITUDE"].mean())
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles="CartoDB Positron",
        control_scale=True,
        prefer_canvas=True,
    )

    add_peninsula_boundary(m)

    tower_groups = list(antennas.groupby(["LATITUDE", "LONGITUDE"], sort=True))
    for tower_index, ((lat, lon), group) in enumerate(tower_groups, start=1):
        service = dominant_service(group)
        tooltip = (
            f"Tower #{tower_index} | {len(group):,} antenna rows | "
            f"{compact_unique(group, 'LICENSEE', limit=2)} | {counts_text(group, 'TECHNOLOGY')}"
        )
        folium.Marker(
            location=[lat, lon],
            tooltip=tooltip,
            popup=folium.Popup(tower_popup_html(group, tower_index), max_width=980),
            icon=folium.Icon(color=SERVICE_COLORS.get(service, "gray"), icon="signal", prefix="fa"),
        ).add_to(m)

    add_summary_panel(m, antennas)

    bounds = [
        [float(antennas["LATITUDE"].min()), float(antennas["LONGITUDE"].min())],
        [float(antennas["LATITUDE"].max()), float(antennas["LONGITUDE"].max())],
    ]
    m.fit_bounds(bounds, padding=(20, 20))

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    m.save(OUTPUT_HTML)
    print(f"Map saved: {OUTPUT_HTML}")
    print(f"Antenna rows mapped: {len(antennas):,}")
    print(f"Tower locations: {len(tower_groups):,}")


if __name__ == "__main__":
    main()
