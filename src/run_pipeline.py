"""Pulls Sentinel-2 (NDVI/NDRE) and Sentinel-1 (VV/VH) time series for every plot in a CSV,
via Copernicus Data Space Ecosystem, and writes one CSV + one chart per plot.

Usage:
    python run_pipeline.py --plots ../config/test_plots.csv --start 2026-01-01 --end 2026-08-01

Requires SH_CLIENT_ID / SH_CLIENT_SECRET in a .env file (see .env.example) from a free
OAuth client created at https://dataspace.copernicus.eu -> Dashboard -> User Settings.
"""
import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from crop_classifier import classify_and_estimate
from ndvi_pull import pull_s2_timeseries
from sar_pull import pull_s1_timeseries

OUTPUT_DIR = Path(__file__).parent.parent / "output"


def process_plot(row: pd.Series, start_date: str, end_date: str) -> pd.DataFrame:
    print(f"[{row.plot_id}] pulling Sentinel-2 NDVI/NDRE ...")
    s2 = pull_s2_timeseries(row.wkt_polygon, start_date, end_date)
    print(f"[{row.plot_id}]   {len(s2)} usable (cloud-free) dates")

    print(f"[{row.plot_id}] pulling Sentinel-1 VV/VH ...")
    s1 = pull_s1_timeseries(row.wkt_polygon, start_date, end_date)
    print(f"[{row.plot_id}]   {len(s1)} radar passes")

    merged = pd.merge(s2, s1, on="date", how="outer").sort_values("date")
    merged.insert(0, "plot_id", row.plot_id)
    return merged


def plot_timeseries(plot_id: str, label: str, df: pd.DataFrame, outdir: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    ax1.plot(df["date"], df["ndvi_mean"], "o-", color="#2F7A50", label="NDVI")
    ax1.plot(df["date"], df["ndre_mean"], "o-", color="#4C7A3C", alpha=0.6, label="NDRE")
    ax1.set_ylabel("NDVI / NDRE")
    ax1.set_ylim(-0.1, 1.0)

    ax2 = ax1.twinx()
    ax2.plot(df["date"], df["vv_db_mean"], "s--", color="#2E4374", alpha=0.7, label="VV (dB)")
    ax2.plot(df["date"], df["vh_db_mean"], "s--", color="#8DA1D9", alpha=0.7, label="VH (dB)")
    ax2.set_ylabel("SAR backscatter (dB)")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax1.set_title(f"{plot_id} — {label}")
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = outdir / f"{plot_id}_timeseries.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[{plot_id}] chart saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots", required=True, help="CSV with plot_id,label,crop,wkt_polygon columns")
    parser.add_argument("--start", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="YYYY-MM-DD")
    parser.add_argument("--outdir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--asof", default=None,
        help="YYYY-MM-DD to compute crop age as of (default: today). Mainly useful for "
             "reproducible test runs against a fixed end date.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    as_of = date.fromisoformat(args.asof) if args.asof else date.today()

    plots = pd.read_csv(args.plots)
    all_rows = []
    crop_rows = []

    for _, row in plots.iterrows():
        df = process_plot(row, args.start, args.end)
        df.to_csv(outdir / f"{row.plot_id}_timeseries.csv", index=False)
        if not df.empty and df["ndvi_mean"].notna().any():
            plot_timeseries(row.plot_id, row.label, df, outdir)
        all_rows.append(df)

        estimate = classify_and_estimate(df, as_of=as_of)
        print(f"[{row.plot_id}] detected: {estimate.crop_type} "
              f"({estimate.crop_confidence} confidence) -- {estimate.notes}")
        crop_rows.append({
            "plot_id": row.plot_id,
            "label": getattr(row, "label", None),
            "region": getattr(row, "region", None),
            "crop_declared": getattr(row, "crop", None),
            **estimate.as_row(),
        })

    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(outdir / "all_plots_timeseries.csv", index=False)

    crop_summary = pd.DataFrame(crop_rows)
    crop_summary["crop_age_days"] = crop_summary["crop_age_days"].astype("Int64")
    crop_summary.to_csv(outdir / "crop_summary.csv", index=False)

    print(f"\nDone. {len(plots)} plot(s) processed. Combined CSV: {outdir / 'all_plots_timeseries.csv'}")
    print(f"Crop/age summary: {outdir / 'crop_summary.csv'}")


if __name__ == "__main__":
    main()
