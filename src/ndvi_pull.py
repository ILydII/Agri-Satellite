"""Sentinel-2 L2A NDVI/NDRE time series for one polygon."""
from sentinelhub import DataCollection

from statistical_client import run_statistical_request


def pull_s2_timeseries(wkt_polygon: str, start_date: str, end_date: str):
    df = run_statistical_request(
        evalscript_filename="s2_ndvi_ndre.js",
        data_collection=DataCollection.SENTINEL2_L2A,
        wkt_polygon=wkt_polygon,
        time_interval=(start_date, end_date),
        output_ids=["ndvi", "ndre"],
        aggregation_interval="P1D",
        resolution=(10, 10),
    )
    if not df.empty:
        # Drop bins where every pixel in the polygon was clouded out.
        df = df[df["ndvi_valid_px"].fillna(0) > 0].reset_index(drop=True)
    return df
