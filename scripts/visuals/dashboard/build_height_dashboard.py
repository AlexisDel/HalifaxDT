"""
Build a static HTML dashboard from the selected Halifax building heights.

Input:
  data/processed_data/building_heights_selected.gpkg

Output:
  data/visuals/dashboard/building_heights_dashboard.html
"""

from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd


HEIGHT_BINS = [0, 4, 7, 10, 15, 25, 40, np.inf]
HEIGHT_LABELS = ["< 4 m", "4-7 m", "7-10 m", "10-15 m", "15-25 m", "25-40 m", ">= 40 m"]
HEIGHT_COLORS = ["#2c7bb6", "#00a6ca", "#00ccbc", "#90eb9d", "#ffff8c", "#f9d057", "#d7191c"]

SOURCE_ORDER = [
    "DSM_P95",
    "DSM_DOMINANT",
    "CONSENSUS_DOMINANT",
    "OSM_NSTDB_CONSENSUS",
    "NSTDB",
    "OSM",
    "DEFAULT_FCODE",
    "missing",
]

CONFIDENCE_ORDER = ["high", "medium_high", "medium", "low", "missing"]

CONFLICT_ORDER = ["none", "moderate", "strong", "missing"]

FLAG_ORDER = [
    "dsm_stable",
    "dsm_selected",
    "source_conflict_keep_stable_dsm",
    "strong_source_conflict_keep_stable_dsm",
    "multi_volume_dsm_dominant",
    "multi_volume_consensus_dominant",
    "dsm_roof_confirmed",
    "external_replaces_unstable_dsm",
    "external_replaces_weak_dsm",
    "external_replaces_invalid_dsm",
    "fallback",
    "missing",
]

OSM_MATCH_QUALITY_ORDER = ["good", "multi_hrm_match", "broad_osm_polygon", "missing"]

NSTDB_MATCH_METHOD_ORDER = ["polygon_intersection", "nearest", "missing"]

SOURCE_COLORS = {
    "DSM_P95": "#2563eb",
    "DSM_DOMINANT": "#0891b2",
    "CONSENSUS_DOMINANT": "#16a34a",
    "OSM_NSTDB_CONSENSUS": "#65a30d",
    "NSTDB": "#ca8a04",
    "OSM": "#ea580c",
    "DEFAULT_FCODE": "#6b7280",
}

CONFIDENCE_COLORS = {
    "high": "#15803d",
    "medium_high": "#65a30d",
    "medium": "#ca8a04",
    "low": "#dc2626",
}

CONFLICT_COLORS = {
    "none": "#16a34a",
    "moderate": "#f59e0b",
    "strong": "#dc2626",
}


def data_root() -> Path:
    return Path(__file__).parents[3] / "data"


def parse_args() -> argparse.Namespace:
    root = data_root()
    parser = argparse.ArgumentParser(description="Build a dashboard of selected building heights.")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "processed_data" / "building_heights_selected.gpkg",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "visuals" / "dashboard" / "building_heights_dashboard.html",
    )
    parser.add_argument("--layer", default="buildings")
    return parser.parse_args()


def count_values(frame: pd.DataFrame, column: str, include_empty: bool = False) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype="int64")
    values = frame[column].fillna("").astype(str).str.strip()
    if include_empty:
        values = values.replace("", "missing")
    else:
        values = values[values != ""]
    return values.value_counts()


def with_expected_categories(counts: pd.Series, expected: list[str], keep_extra: bool = True) -> pd.Series:
    """Add expected categories with zero counts while preserving unexpected labels."""

    expected_counts = counts.reindex(expected, fill_value=0)
    if not keep_extra:
        return expected_counts
    extras = counts[~counts.index.isin(expected)]
    return pd.concat([expected_counts, extras])


def sort_counts_desc(counts: pd.Series) -> pd.Series:
    """Sort categories by count descending, keeping equal values in their current order."""

    return counts.sort_values(ascending=False, kind="stable")


def fmt_int(value: float | int) -> str:
    return f"{int(value):,}".replace(",", " ")


def fmt_float(value: float, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def percent(count: int, total: int) -> str:
    return "0.0%" if total == 0 else f"{count / total * 100:.1f}%"


def chart_color(label: str, palette: dict[str, str], index: int) -> str:
    fallback = ["#2563eb", "#0891b2", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed", "#4b5563"]
    return palette.get(label, fallback[index % len(fallback)])


def render_metric(label: str, value: str, note: str = "") -> str:
    return f"""
      <div class="metric">
        <div class="metric-label">{escape(label)}</div>
        <div class="metric-value">{escape(value)}</div>
        <div class="metric-note">{escape(note)}</div>
      </div>
    """


def render_bar_chart(title: str, counts: pd.Series, total: int, palette: dict[str, str] | None = None) -> str:
    if counts.empty:
        rows = '<div class="empty">No data</div>'
    else:
        max_count = int(counts.max())
        rows_list = []
        for index, (label, count) in enumerate(counts.items()):
            label_text = str(label)
            width = 0 if max_count == 0 else max(2.0, count / max_count * 100.0)
            color = chart_color(label_text, palette or {}, index)
            rows_list.append(
                f"""
                <div class="bar-row">
                  <div class="bar-label" title="{escape(label_text)}">{escape(label_text)}</div>
                  <div class="bar-track">
                    <div class="bar-fill" style="width:{width:.2f}%; background:{color};"></div>
                  </div>
                  <div class="bar-count">{fmt_int(count)} <span>{percent(int(count), total)}</span></div>
                </div>
                """
            )
        rows = "\n".join(rows_list)
    return f"""
      <section class="card">
        <h2>{escape(title)}</h2>
        {rows}
      </section>
    """


def render_stacked_bar(title: str, counts: pd.Series, total: int, palette: dict[str, str]) -> str:
    if counts.empty or total == 0:
        segments = '<div class="empty">No data</div>'
        legend = ""
    else:
        segment_items = []
        legend_items = []
        for index, (label, count) in enumerate(counts.items()):
            label_text = str(label)
            width = count / total * 100.0
            color = chart_color(label_text, palette, index)
            segment_items.append(
                f'<div class="stack-segment" title="{escape(label_text)}: {fmt_int(count)}" '
                f'style="width:{width:.2f}%; background:{color};"></div>'
            )
            legend_items.append(
                f"""
                <div class="legend-item">
                  <span class="dot" style="background:{color};"></span>
                  <span>{escape(label_text)}</span>
                  <b>{fmt_int(count)}</b>
                  <em>{percent(int(count), total)}</em>
                </div>
                """
            )
        segments = f'<div class="stacked">{"".join(segment_items)}</div>'
        legend = f'<div class="stack-legend">{"".join(legend_items)}</div>'
    return f"""
      <section class="card">
        <h2>{escape(title)}</h2>
        {segments}
        {legend}
      </section>
    """


def render_histogram(frame: pd.DataFrame, total: int) -> str:
    if "final_height_m" not in frame.columns:
        counts = pd.Series(dtype="int64")
    else:
        heights = pd.to_numeric(frame["final_height_m"], errors="coerce")
        buckets = pd.cut(heights, HEIGHT_BINS, labels=HEIGHT_LABELS, right=False)
        counts = buckets.value_counts(sort=False)
    max_count = int(counts.max()) if not counts.empty else 0
    bars = []
    for index, label in enumerate(HEIGHT_LABELS):
        count = int(counts.get(label, 0))
        height = 0 if max_count == 0 else max(3.0, count / max_count * 100.0)
        bars.append(
            f"""
            <div class="hist-col">
              <div class="hist-value">{fmt_int(count)}</div>
              <div class="hist-bar" style="height:{height:.2f}%; background:{HEIGHT_COLORS[index]};"></div>
              <div class="hist-label">{escape(label)}</div>
              <div class="hist-percent">{percent(count, total)}</div>
            </div>
            """
        )
    return f"""
      <section class="card wide">
        <h2>Final Height Distribution</h2>
        <div class="histogram">
          {"".join(bars)}
        </div>
      </section>
    """


def render_dashboard(frame: pd.DataFrame) -> str:
    total = int(len(frame))
    heights = pd.to_numeric(frame.get("final_height_m", pd.Series(dtype="float64")), errors="coerce")
    confidence_counts = with_expected_categories(
        count_values(frame, "final_height_confidence", include_empty=True),
        CONFIDENCE_ORDER,
    )
    confidence_counts = sort_counts_desc(confidence_counts)
    source_counts = with_expected_categories(
        count_values(frame, "final_height_source", include_empty=True),
        SOURCE_ORDER,
    )
    source_counts = sort_counts_desc(source_counts)
    flag_counts = with_expected_categories(
        count_values(frame, "height_selection_flag", include_empty=True),
        FLAG_ORDER,
    )
    flag_counts = sort_counts_desc(flag_counts)
    conflict_counts = with_expected_categories(
        count_values(frame, "external_dsm_conflict", include_empty=True),
        CONFLICT_ORDER,
    )
    conflict_counts = sort_counts_desc(conflict_counts)
    osm_quality_counts = with_expected_categories(
        count_values(frame, "osm_match_quality", include_empty=True),
        OSM_MATCH_QUALITY_ORDER,
    )
    osm_quality_counts = sort_counts_desc(osm_quality_counts)
    nstdb_method_counts = with_expected_categories(
        count_values(frame, "nstdb_match_method", include_empty=True),
        NSTDB_MATCH_METHOD_ORDER,
    )
    nstdb_method_counts = sort_counts_desc(nstdb_method_counts)

    low_count = int(confidence_counts.get("low", 0))
    median_height = float(heights.median()) if not heights.dropna().empty else np.nan
    p95_height = float(heights.quantile(0.95)) if not heights.dropna().empty else np.nan
    max_height = float(heights.max()) if not heights.dropna().empty else np.nan

    metrics = "\n".join(
        [
            render_metric("Buildings", fmt_int(total), "processed HRM footprints"),
            render_metric("Median Height", f"{fmt_float(median_height)} m", "final_height_m"),
            render_metric("P95 Height", f"{fmt_float(p95_height)} m", "95% of buildings below"),
            render_metric("Low confidence", fmt_int(low_count), percent(low_count, total)),
            render_metric("Max Height", f"{fmt_float(max_height)} m", "extreme value check"),
        ]
    )

    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Halifax Building Heights Dashboard</title>
  <style>
    :root {{
      --bg: #f3f4f6;
      --card: #ffffff;
      --border: #d1d5db;
      --text: #111827;
      --muted: #6b7280;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: Arial, sans-serif;
      margin: 0;
    }}
    main {{
      margin: 0 auto;
      max-width: 1320px;
      padding: 24px;
    }}
    header {{
      margin-bottom: 18px;
    }}
    h1 {{
      font-size: 28px;
      margin: 0 0 6px;
    }}
    .subtitle {{
      color: var(--muted);
      margin: 0;
    }}
    .metrics {{
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      margin-bottom: 16px;
    }}
    .metric, .card {{
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      box-shadow: 0 1px 6px rgba(15, 23, 42, 0.08);
    }}
    .metric {{
      padding: 14px 16px;
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }}
    .metric-value {{
      font-size: 28px;
      font-weight: 700;
      margin-top: 5px;
    }}
    .metric-note {{
      color: var(--muted);
      font-size: 12px;
      min-height: 16px;
    }}
    .grid {{
      display: grid;
      gap: 16px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
    .card {{
      padding: 16px;
    }}
    .wide {{
      grid-column: 1 / -1;
    }}
    h2 {{
      font-size: 17px;
      margin: 0 0 14px;
    }}
    .bar-row {{
      align-items: center;
      display: grid;
      gap: 10px;
      grid-template-columns: minmax(150px, 230px) 1fr minmax(110px, auto);
      margin: 8px 0;
    }}
    .bar-label {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .bar-track {{
      background: #e5e7eb;
      border-radius: 4px;
      height: 16px;
      overflow: hidden;
    }}
    .bar-fill {{
      border-radius: 4px;
      height: 100%;
    }}
    .bar-count {{
      font-variant-numeric: tabular-nums;
      text-align: right;
      white-space: nowrap;
    }}
    .bar-count span, .hist-percent, .legend-item em {{
      color: var(--muted);
      font-style: normal;
    }}
    .stacked {{
      border-radius: 6px;
      display: flex;
      height: 30px;
      overflow: hidden;
      width: 100%;
    }}
    .stack-legend {{
      display: grid;
      gap: 7px;
      margin-top: 12px;
    }}
    .legend-item {{
      align-items: center;
      display: grid;
      gap: 8px;
      grid-template-columns: auto 1fr auto auto;
    }}
    .dot {{
      border-radius: 50%;
      display: inline-block;
      height: 10px;
      width: 10px;
    }}
    .histogram {{
      align-items: end;
      display: grid;
      gap: 14px;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      height: 280px;
      padding-top: 18px;
    }}
    .hist-col {{
      align-items: center;
      display: grid;
      grid-template-rows: auto 1fr auto auto;
      height: 100%;
      justify-items: center;
      min-width: 0;
    }}
    .hist-value {{
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      margin-bottom: 5px;
    }}
    .hist-bar {{
      align-self: end;
      border-radius: 5px 5px 0 0;
      width: min(54px, 82%);
    }}
    .hist-label {{
      font-size: 12px;
      margin-top: 7px;
      text-align: center;
    }}
    .hist-percent {{
      font-size: 11px;
    }}
    .empty {{
      color: var(--muted);
      padding: 20px 0;
    }}
    footer {{
      color: var(--muted);
      font-size: 12px;
      margin-top: 18px;
    }}
    @media (max-width: 950px) {{
      .metrics {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; gap: 5px; }}
      .bar-count {{ text-align: left; }}
      .histogram {{ gap: 8px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Halifax Building Heights Dashboard</h1>
      <p class="subtitle">Summary of <code>building_heights_selected.gpkg</code> generated from the final height selection layer.</p>
    </header>
    <section class="metrics">
      {metrics}
    </section>
    <section class="grid">
      {render_bar_chart("Final Height Sources", source_counts, total, SOURCE_COLORS)}
      {render_stacked_bar("Final Confidence", confidence_counts, total, CONFIDENCE_COLORS)}
      {render_bar_chart("Decision Flags", flag_counts, total)}
      {render_stacked_bar("DSM / External Source Conflict", conflict_counts, total, CONFLICT_COLORS)}
      {render_histogram(frame, total)}
      {render_bar_chart("OSM Match Quality", osm_quality_counts, total)}
      {render_bar_chart("NSTDB Match Method", nstdb_method_counts, total)}
    </section>
    <footer>
      Scores are relative confidence indicators. Low confidence categories identify buildings that should be checked first on the interactive map.
    </footer>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    buildings = gpd.read_file(args.input, layer=args.layer)
    html = render_dashboard(buildings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
