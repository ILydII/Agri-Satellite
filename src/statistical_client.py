"""Thin wrapper around the Sentinel Hub Statistical API (served via Copernicus Data Space Ecosystem).

Returns per-polygon, per-date zonal statistics without ever downloading a raster -- the
aggregation happens server-side. An output band literally named "dataMask" is treated
specially by the API: samples where it evaluates to 0 are excluded from the stats of
every other output, so the evalscripts route SCL cloud/shadow pixels (S2) straight into
dataMask=0 and they never pollute the NDVI/NDRE means.
"""
from pathlib import Path

import pandas as pd
from sentinelhub import CRS, DataCollection, Geometry, MosaickingOrder, SentinelHubStatistical

from sh_config import get_config

EVALSCRIPT_DIR = Path(__file__).parent / "evalscripts"


def _load_evalscript(filename: str) -> str:
    return (EVALSCRIPT_DIR / filename).read_text()


def run_statistical_request(
    evalscript_filename: str,
    data_collection: DataCollection,
    wkt_polygon: str,
    time_interval: tuple[str, str],
    output_ids: list[str],
    aggregation_interval: str = "P1D",
    resolution: tuple[float, float] = (10, 10),
) -> pd.DataFrame:
    """Runs one Statistical API request and returns a tidy DataFrame: one row per date bin."""
    evalscript = _load_evalscript(evalscript_filename)
    geometry = Geometry(wkt_polygon, crs=CRS.WGS84)

    request = SentinelHubStatistical(
        aggregation=SentinelHubStatistical.aggregation(
            evalscript=evalscript,
            time_interval=time_interval,
            aggregation_interval=aggregation_interval,
            resolution=resolution,
        ),
        input_data=[
            SentinelHubStatistical.input_data(
                data_collection,
                mosaicking_order=(
                    MosaickingOrder.LEAST_CC
                    if data_collection == DataCollection.SENTINEL2_L2A
                    else None
                ),
            )
        ],
        geometry=geometry,
        config=get_config(),
    )

    response = request.get_data()[0]
    rows = []
    for interval_result in response.get("data", []):
        date = interval_result["interval"]["from"][:10]
        row = {"date": date}
        outputs = interval_result.get("outputs", {})
        for output_id in output_ids:
            band_stats = outputs.get(output_id, {}).get("bands", {}).get("B0", {}).get("stats", {})
            row[f"{output_id}_mean"] = band_stats.get("mean")
            row[f"{output_id}_stdev"] = band_stats.get("stDev")
            row[f"{output_id}_valid_px"] = band_stats.get("sampleCount")
            row[f"{output_id}_nodata_px"] = band_stats.get("noDataCount")
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)
    return df
