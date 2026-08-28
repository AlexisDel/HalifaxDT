"""
Prepare 2G-to-5G cellular antenna records for the Halifax peninsula.

The source file is ISED's Site_Data_Extract_FX.csv. The output keeps the
project's internal antenna column names, adds DEM-based ground elevation, and
computes the absolute antenna Z used by Sionna/notebooks.

Output:
  data/interim_data/peninsula_cellular_antennas_2g_to_5g.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import rasterio
from rasterio.warp import transform as transform_coords
from shapely.geometry import Point, Polygon


DATA_ROOT = Path(__file__).parents[2] / "data"
INPUT_CSV = DATA_ROOT / "raw_data" / "Site_Data_Extract_FX.csv"
DEM_PATH = DATA_ROOT / "interim_data" / "dem_peninsula_5m" / "HRM_LiDAR_DEM_2018_5m_peninsula.tif"
OUTPUT_CSV = DATA_ROOT / "interim_data" / "peninsula_cellular_antennas_2g_to_5g.csv"

CHUNK_SIZE = 100_000

# Cellular/mobile operators kept for the Halifax radio-propagation study.
CELLULAR_LICENSEE_KEYWORDS = [
    "Bell",
    "Rogers",
    "TELUS",
    "Bragg",
    "FIDO",
]

# ISED service codes associated with cellular/mobile broadband deployments.
CELLULAR_SERVICES = {
    "CELL",
    "PCS",
    "PCSG",
    "AWS",
    "AWS-3",
    "BRS",
    "MBS",
    "600B",
    "3500B",
}

# Explicit technologies expected in the newer Site_Data extract.
CELLULAR_TECHNOLOGIES = {
    "GSM",    # 2G
    "HSPA",   # 3G
    "LTE",    # 4G
    "5GNR",   # 5G New Radio
    "5GDSS",  # 5G Dynamic Spectrum Sharing
}

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

# Convert ISED Site_Data column names to the internal names already used by the
# map and notebooks. SITE_ELEV is added later from the DEM for compatibility.
COLUMN_MAP = {
    "account_number*": "NEW_ACCOUNT",
    "licence_number": "NEW_LICNO",
    "reference_number": "REFERENCE_NUMBER",
    "licensee_name*": "LICENSEE",
    "licence_category*": "SERVICE",
    "location": "LOCATION",
    "station_type": "STATION_TYPE",
    "technology": "TECHNOLOGY",
    "cell_id": "CELL_ID",
    "physical_id": "PHYSICAL_ID",
    "province_code": "PROV",
    "latitude": "LATITUDE",
    "longitude": "LONGITUDE",
    "site_type": "SITE_TYPE",
    "structure_height": "STUCT_HT",
    "structure_type": "STRUCTURE_TYPE",
    "date_last_changed": "LAST_MOD_DATE",
    "record_id": "RECORD_ID",
    "tx_frequency": "TRANSMIT_FREQ",
    "rx_frequency": "RECEIVE_FREQ",
    "tx_radio_model_number": "TX_MODEL",
    "tx_radio_manufacturer_code": "TX_MFR",
    "bandwidth": "TRANSMIT_BW",
    "class_emission": "BW_EMISSION",
    "tx_power": "TX_PWR",
    "tx_ant_model_no": "TX_ANT_MODEL",
    "tx_ant_manufacturer": "TX_ANT_MFR",
    "tx_ant_height": "TX_ANT_HT",
    "tx_ant_omni_indicator": "TX_ANT_DIRECTIONAL",
    "tx_ant_azimuth": "TX_ANT_AZIM",
    "tx_ant_elevation_angle": "TX_ANT_ELEV_ANGLE",
    "tx_ant_gain": "TX_ANT_GAIN",
    "tx_line_loss": "TX_LINE_LOSS",
    "upload_date*": "LAST_UPLOAD_DATE",
}

NUMERIC_COLUMNS = [
    "LATITUDE",
    "LONGITUDE",
    "STUCT_HT",
    "TRANSMIT_FREQ",
    "RECEIVE_FREQ",
    "TRANSMIT_BW",
    "TX_PWR",
    "TX_ANT_HT",
    "TX_ANT_AZIM",
    "TX_ANT_ELEV_ANGLE",
    "TX_ANT_GAIN",
    "TX_LINE_LOSS",
]


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Rename source columns and normalize numeric fields."""
    chunk = chunk.rename(columns=COLUMN_MAP)

    for column in NUMERIC_COLUMNS:
        if column in chunk.columns:
            chunk[column] = pd.to_numeric(chunk[column], errors="coerce")

    chunk = chunk.dropna(subset=["LATITUDE", "LONGITUDE"]).copy()
    chunk["SERVICE"] = chunk["SERVICE"].astype(str).str.strip()
    chunk["TECHNOLOGY"] = chunk["TECHNOLOGY"].astype(str).str.strip()
    return chunk


def in_peninsula_mask(df: pd.DataFrame) -> list[bool]:
    """Return True for rows whose lon/lat point is inside the study polygon."""
    minx, miny, maxx, maxy = PENINSULA.bounds
    candidates = df[
        df["LONGITUDE"].between(minx, maxx)
        & df["LATITUDE"].between(miny, maxy)
    ].copy()

    mask = pd.Series(False, index=df.index)
    if candidates.empty:
        return mask.tolist()

    inside = [
        PENINSULA.contains(Point(lon, lat))
        for lat, lon in zip(candidates["LATITUDE"], candidates["LONGITUDE"])
    ]
    mask.loc[candidates.index] = inside
    return mask.tolist()


def sample_dem_ground_elevation(antennas: pd.DataFrame) -> pd.Series:
    """Sample the prepared DEM at each antenna lon/lat coordinate."""
    if antennas.empty:
        return pd.Series(dtype="float64")

    with rasterio.open(DEM_PATH) as dem:
        if dem.crs is None:
            raise ValueError(f"DEM has no CRS: {DEM_PATH}")

        xs, ys = transform_coords(
            "EPSG:4326",
            dem.crs,
            antennas["LONGITUDE"].tolist(),
            antennas["LATITUDE"].tolist(),
        )
        samples = list(dem.sample(zip(xs, ys), masked=True))

        values: list[float | None] = []
        for sample in samples:
            value = sample[0]
            if getattr(value, "mask", False):
                values.append(None)
                continue
            value = float(value)
            if dem.nodata is not None and value == dem.nodata:
                values.append(None)
            else:
                values.append(value)

    return pd.Series(values, index=antennas.index, dtype="float64")


def prepare_antennas() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing antenna source file: {INPUT_CSV}")
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"Missing DEM file for antenna elevation sampling: {DEM_PATH}")

    operator_pattern = "|".join(CELLULAR_LICENSEE_KEYWORDS)
    usecols = list(COLUMN_MAP.keys())

    total_rows = 0
    cellular_rows = 0
    peninsula_chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(INPUT_CSV, usecols=usecols, chunksize=CHUNK_SIZE, low_memory=False):
        total_rows += len(chunk)
        chunk = normalize_chunk(chunk)

        cellular = chunk[
            chunk["LICENSEE"].str.contains(operator_pattern, case=False, na=False)
            & chunk["SERVICE"].isin(CELLULAR_SERVICES)
            & chunk["TECHNOLOGY"].isin(CELLULAR_TECHNOLOGIES)
        ].copy()
        cellular_rows += len(cellular)

        if cellular.empty:
            continue

        cellular = cellular.loc[in_peninsula_mask(cellular)].copy()
        if not cellular.empty:
            peninsula_chunks.append(cellular)

    if peninsula_chunks:
        antennas = pd.concat(peninsula_chunks, ignore_index=True)
    else:
        antennas = pd.DataFrame(columns=list(COLUMN_MAP.values()))

    antennas["DEM_GROUND_ELEV_M"] = sample_dem_ground_elevation(antennas)
    antennas["SITE_ELEV"] = antennas["DEM_GROUND_ELEV_M"]
    antennas["ANTENNA_Z_M"] = antennas["DEM_GROUND_ELEV_M"] + antennas["TX_ANT_HT"]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    antennas.to_csv(OUTPUT_CSV, index=False)

    tower_count = antennas[["LATITUDE", "LONGITUDE"]].drop_duplicates().shape[0]
    print(f"Input antenna CSV: {INPUT_CSV}")
    print(f"Total Site Data rows: {total_rows:,}")
    print(f"2G-to-5G cellular antenna rows: {cellular_rows:,}")
    print(f"Peninsula 2G-to-5G antenna rows: {len(antennas):,}")
    print(f"Peninsula tower locations: {tower_count:,}")
    print(f"CSV saved: {OUTPUT_CSV}")
    return antennas


def main() -> None:
    prepare_antennas()


if __name__ == "__main__":
    main()
