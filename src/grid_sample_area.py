"""Coarse grid-sample rice/corn area estimate over an administrative boundary.

This is NOT a real land-cover survey. It lays a grid of small (100 m radius) sample
AOIs across a region's actual land boundary, classifies each with the existing
rice/corn detector (crop_classifier.py), and scales the classified fraction up to
the region's land area. No pixel-level classification, no accuracy validation --
treat the output as a rough first-pass number to sense-check against an authoritative
source (e.g. BPS "luas tanam padi/jagung" statistics), not a real area survey.

Usage (run from src/):
    python grid_sample_area.py --boundary ../config/jember_boundary_raw.json --spacing-km 8
"""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape

from crop_classifier import classify_and_estimate
from ndvi_pull import pull_s2_timeseries
from sar_pull import pull_s1_timeseries

KM_PER_DEG_LAT = 111.32


def load_boundary(path: str):
    data = json.loads(Path(path).read_text())
    return shape(data[0]["geojson"])


def circle_wkt(lat: float, lon: float, radius_m: float, n: int = 16) -> str:
    lat_deg_per_m = 1 / (KM_PER_DEG_LAT * 1000)
    lon_deg_per_m = 1 / (KM_PER_DEG_LAT * 1000 * math.cos(math.radians(lat)))
    pts = []
    for i in range(n + 1):
        theta = 2 * math.pi * i / n
        dlat = radius_m * math.sin(theta) * lat_deg_per_m
        dlon = radius_m * math.cos(theta) * lon_deg_per_m
        pts.append(f"{lon + dlon:.6f} {lat + dlat:.6f}")
    return "POLYGON((" + ", ".join(pts) + "))"


def build_grid(boundary, spacing_km: float):
    minx, miny, maxx, maxy = boundary.bounds
    mean_lat = (miny + maxy) / 2
    dlat = spacing_km / KM_PER_DEG_LAT
    dlon = spacing_km / (KM_PER_DEG_LAT * math.cos(math.radians(mean_lat)))

    pts = []
    y = miny + dlat / 2
    while y < maxy:
        x = minx + dlon / 2
        while x < maxx:
            if boundary.contains(Point(x, y)):
                pts.append((y, x))  # (lat, lon)
            x += dlon
        y += dlat
    return pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True, help="Nominatim search JSON (jsonv2, polygon_geojson=1)")
    ap.add_argument("--spacing-km", type=float, default=8.0)
    ap.add_argument("--radius-m", type=float, default=100.0)
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-08-17")
    ap.add_argument("--out", default="../output/grid_sample.csv")
    ap.add_argument("--dry-run", action="store_true", help="Only print the grid point count, don't hit the API")
    args = ap.parse_args()

    boundary = load_boundary(args.boundary)
    pts = build_grid(boundary, args.spacing_km)
    cell_area_km2 = args.spacing_km ** 2
    print(f"{len(pts)} sample points inside boundary "
          f"({args.spacing_km} km spacing, {cell_area_km2:.0f} km2/cell, "
          f"~{len(pts) * cell_area_km2:.0f} km2 represented)")
    if args.dry_run:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "lat", "lon", "crop_detected", "crop_confidence",
                          "planting_date_est", "crop_age_days", "growth_stage", "notes"])
        for i, (lat, lon) in enumerate(pts):
            wkt = circle_wkt(lat, lon, args.radius_m)
            t0 = time.time()
            try:
                s2 = pull_s2_timeseries(wkt, args.start, args.end)
                s1 = pull_s1_timeseries(wkt, args.start, args.end)
                merged = pd.merge(s2, s1, on="date", how="outer").sort_values("date")
                est = classify_and_estimate(merged)
                writer.writerow([i, lat, lon, est.crop_type, est.crop_confidence,
                                  est.planting_date_est, est.age_days, est.growth_stage, est.notes])
            except Exception as exc:
                writer.writerow([i, lat, lon, "error", "", "", "", "", str(exc)[:200]])
            f.flush()
            print(f"[{i + 1}/{len(pts)}] {lat:.4f},{lon:.4f} -> {time.time() - t0:.1f}s", flush=True)

    print(f"\ndone, wrote {out_path}")


if __name__ == "__main__":
    main()
