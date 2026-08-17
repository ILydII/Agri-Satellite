"""Sentinel-1 GRD VV/VH backscatter time series for one polygon."""
from sentinelhub import DataCollection

from statistical_client import run_statistical_request


def pull_s1_timeseries(wkt_polygon: str, start_date: str, end_date: str):
    df = run_statistical_request(
        evalscript_filename="s1_backscatter.js",
        data_collection=DataCollection.SENTINEL1_IW,
        wkt_polygon=wkt_polygon,
        time_interval=(start_date, end_date),
        output_ids=["vv_db", "vh_db"],
        aggregation_interval="P1D",
        resolution=(10, 10),
    )
    return df
