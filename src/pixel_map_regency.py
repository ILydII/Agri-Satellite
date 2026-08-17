"""Full-regency pixel-level (10m native) rice/corn classification map.

Uses the Process API (not the Statistical API used elsewhere in this pipeline) to fetch
actual NDVI/VV rasters, tiled to fit the API's 2500x2500px max output, across a sparse set
of dates, then classifies each pixel independently -- no point-sample extrapolation. Real
measured cost: ~10,000 PU for Jember Regency at 19 dates (derived empirically from small
test requests, not estimated from documentation).

Every tile/date/band download is cached to disk keyed by its request parameters, so a
killed or resumed run does not re-spend PU on tiles already fetched.

Classification here is a SIMPLIFIED, vectorized port of crop_classifier.py's episode-based
logic, adapted for a sparse ~19-point-per-pixel time series (vs. ~30-50 dense points in the
point-sample method) -- see _classify_stack for the specific simplifications and why they're
an acceptable tradeoff at this density.

Usage (run from src/):
    python pixel_map_regency.py --boundary ../config/jember_boundary_raw.json
"""
import argparse
import io
import json
import math
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import requests
import tifffile
from shapely.geometry import box, shape

from sh_config import CDSE_BASE_URL, get_config

TILE_PX = 2500
RES_M = 10
KM_PER_DEG = 111.32
MIN_LAND_FRACTION = 0.001  # drop tiles that are effectively 100% ocean/outside the boundary

EVALSCRIPT_DIR = Path(__file__).parent / "evalscripts"
CACHE_DIR = Path(__file__).parent.parent / "output" / "pixel_map_cache"
PROCESS_URL = f"{CDSE_BASE_URL}/api/v1/process"

FLOOD_VV_DB = -16.0
FLOOD_REBOUND_DB = 3.0
FLOOD_NDVI_MAX = 0.30
RICE_POST_DIP_NDVI_MIN = 0.40  # real canopy must follow the dip -- permanent water never shows this
GREEN_UP_NDVI = 0.30
MIN_BASELINE_NDVI = 0.25
CORN_STEEP_SLOPE = 0.012
GREEN_UP_LOOKBACK_DAYS = 150


def get_token(config) -> str:
    resp = requests.post(
        config.sh_token_url,
        data={"grant_type": "client_credentials", "client_id": config.sh_client_id,
              "client_secret": config.sh_client_secret},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def build_tiles(boundary_path: str):
    data = json.loads(Path(boundary_path).read_text())
    boundary = shape(data[0]["geojson"])
    minx, miny, maxx, maxy = boundary.bounds
    mean_lat = (miny + maxy) / 2
    tile_deg_x = (TILE_PX * RES_M / 1000) / (KM_PER_DEG * math.cos(math.radians(mean_lat)))
    tile_deg_y = (TILE_PX * RES_M / 1000) / KM_PER_DEG

    nx = math.ceil((maxx - minx) / tile_deg_x)
    ny = math.ceil((maxy - miny) / tile_deg_y)

    tiles = []
    for j in range(ny):
        for i in range(nx):
            tminx, tmaxx = minx + i * tile_deg_x, min(minx + (i + 1) * tile_deg_x, maxx)
            tminy, tmaxy = miny + j * tile_deg_y, min(miny + (j + 1) * tile_deg_y, maxy)
            tile_box = box(tminx, tminy, tmaxx, tmaxy)
            if not boundary.intersects(tile_box):
                continue
            frac = boundary.intersection(tile_box).area / tile_box.area
            if frac < MIN_LAND_FRACTION:
                continue
            width = max(round((tmaxx - tminx) * KM_PER_DEG * math.cos(math.radians(mean_lat)) * 1000 / RES_M), 1)
            height = max(round((tmaxy - tminy) * KM_PER_DEG * 1000 / RES_M), 1)
            tiles.append({
                "id": f"{i}_{j}", "bbox": [tminx, tminy, tmaxx, tmaxy],
                "width": width, "height": height, "land_fraction": frac,
            })
    return tiles


def build_dates(start: str, end: str, n_dates: int):
    d0, d1 = date.fromisoformat(start), date.fromisoformat(end)
    total_days = (d1 - d0).days
    step = total_days / (n_dates - 1)
    return [d0 + timedelta(days=round(i * step)) for i in range(n_dates)]


def fetch_raster(token: str, evalscript: str, collection_type: str, bbox, width: int, height: int,
                  center_date: date, window_days: int = 5, mosaicking_order: str = None):
    date_from = (center_date - timedelta(days=window_days)).isoformat() + "T00:00:00Z"
    date_to = (center_date + timedelta(days=window_days)).isoformat() + "T00:00:00Z"
    data_filter = {"timeRange": {"from": date_from, "to": date_to}}
    if mosaicking_order:
        data_filter["mosaickingOrder"] = mosaicking_order
    payload = {
        "input": {"bounds": {"bbox": bbox}, "data": [{"type": collection_type, "dataFilter": data_filter}]},
        "output": {"width": width, "height": height,
                   "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
        "evalscript": evalscript,
    }
    for attempt in range(4):
        try:
            r = requests.post(PROCESS_URL, headers={"Authorization": f"Bearer {token}"}, json=payload, timeout=180)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 401:
            raise PermissionError("token expired")
        if r.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"process API error {r.status_code}: {r.text[:300]}")
        pu = float(r.headers.get("x-processingunits-spent", 0))
        arr = tifffile.imread(io.BytesIO(r.content))
        return arr, pu
    raise RuntimeError("gave up after retries")


def cached_fetch(token_holder, evalscript, collection_type, tile, d, window_days, mosaicking_order, cache_subdir):
    cache_path = CACHE_DIR / cache_subdir / f"tile{tile['id']}_{d.isoformat()}.npy"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return np.load(cache_path), 0.0
    try:
        arr, pu = fetch_raster(token_holder["token"], evalscript, collection_type, tile["bbox"],
                                tile["width"], tile["height"], d, window_days, mosaicking_order)
    except PermissionError:
        token_holder["token"] = get_token(get_config())
        arr, pu = fetch_raster(token_holder["token"], evalscript, collection_type, tile["bbox"],
                                tile["width"], tile["height"], d, window_days, mosaicking_order)
    np.save(cache_path, arr.astype(np.float32))
    return arr, pu


def _classify_stack(ndvi, vv, day_offsets):
    """Vectorized rice/corn/unknown classification over a (T, H, W) pixel stack.

    Simplified relative to crop_classifier.py's per-point episode grouping: with only ~19
    sparse points (vs ~30-50 dense daily-binned points per plot), redundant same-episode
    re-matches are far less likely, so this tests each time index independently rather than
    grouping into contiguous episodes first. Still scans most-recent-first so a double-cropped
    pixel reports its current cycle. Documented simplification, not a silent behavior change --
    cross-checked against the point-sample method's results for known plots (see notes to user).

    Returns three (H, W) arrays: crop_code (0=unknown, 1=rice, 2=corn), planting_day_offset
    (NaN if unknown), confidence_code (0=low, 1=medium, 2=high).
    """
    T, H, W = vv.shape
    crop_code = np.zeros((H, W), dtype=np.uint8)
    planting_day = np.full((H, W), np.nan, dtype=np.float32)
    confidence = np.zeros((H, W), dtype=np.uint8)
    resolved = np.zeros((H, W), dtype=bool)

    # --- Rice: most-recent-first scan over candidate flood dips ---
    for t in range(T - 2, -1, -1):
        dip_val = vv[t]
        after_max = np.nanmax(vv[t + 1:], axis=0) if t + 1 < T else np.full((H, W), np.nan)
        rebound = after_max - dip_val
        ndvi_lo = t - 1 >= 0
        ndvi_hi = t + 2 <= T
        ndvi_near = ndvi[max(t - 1, 0):min(t + 2, T)]
        ndvi_near_mean = np.nanmean(ndvi_near, axis=0)
        ndvi_ok = np.isnan(ndvi_near_mean) | (ndvi_near_mean <= FLOOD_NDVI_MAX)

        # A real paddy grows a canopy after flooding; permanent water (ocean, lakes, rivers)
        # never does, but can otherwise trivially fake the dip+rebound VV pattern from
        # wave/wind-driven backscatter noise. Require NDVI to actually reach vegetation
        # levels at some point after the dip to rule that out. Confirmed with a real bug:
        # ocean tiles were showing up as 76% "rice" before this check existed.
        ndvi_after_max = np.nanmax(ndvi[t:], axis=0)
        ndvi_after_count = np.sum(~np.isnan(ndvi[t:]), axis=0)
        canopy_ok = (ndvi_after_count < 2) | (ndvi_after_max >= RICE_POST_DIP_NDVI_MIN)

        qualifies = (
            (~resolved) & (~np.isnan(dip_val)) & (dip_val <= FLOOD_VV_DB) &
            (~np.isnan(rebound)) & (rebound >= FLOOD_REBOUND_DB) & ndvi_ok & canopy_ok
        )
        crop_code[qualifies] = 1  # rice
        planting_day[qualifies] = day_offsets[t]
        confidence[qualifies] = np.where(rebound[qualifies] >= 5.0, 2, 1)
        resolved |= qualifies

    # --- Corn: most-recent-first scan over candidate green-up crossings ---
    bare = ndvi <= MIN_BASELINE_NDVI
    above = ndvi >= GREEN_UP_NDVI
    for t in range(T - 1, 0, -1):
        crossing = above[t] & (~resolved)
        if not crossing.any():
            continue
        lookback_start_day = day_offsets[t] - GREEN_UP_LOOKBACK_DAYS
        eligible_prior = np.array([bare_t for bare_t in range(t) if day_offsets[bare_t] >= lookback_start_day])
        if eligible_prior.size == 0:
            continue
        # Most recent bare point before t, per pixel, within the lookback window.
        bare_window = bare[eligible_prior]  # (n_prior, H, W)
        has_bare = bare_window.any(axis=0)
        last_bare_pos = np.where(has_bare, np.argmax(bare_window[::-1], axis=0), -1)
        last_bare_idx = np.where(has_bare, eligible_prior[-1] - last_bare_pos, -1)

        candidate = crossing & has_bare
        if not candidate.any():
            continue
        idx_flat = last_bare_idx
        baseline_ndvi = np.take_along_axis(ndvi, idx_flat[None, :, :].clip(min=0), axis=0)[0]
        baseline_day = np.array(day_offsets)[idx_flat.clip(min=0)]
        days_elapsed = np.clip(day_offsets[t] - baseline_day, 1, None)
        slope = (ndvi[t] - baseline_ndvi) / days_elapsed

        is_corn = candidate & (slope >= CORN_STEEP_SLOPE)
        crop_code[is_corn] = 2
        planting_day[is_corn] = day_offsets[t]
        confidence[is_corn] = 1
        resolved |= is_corn
        # slow-but-real green-up without flood signature: leave as unknown (code 0), matching
        # the point-sample method's "possibly direct-seeded rice / another crop" behavior.

    return crop_code, planting_day, confidence


def process_tile(token_holder, tile, dates_list, s2_script, s1_script, cost_log):
    day_offsets = [(d - dates_list[0]).days for d in dates_list]
    ndvi_stack = np.full((len(dates_list), tile["height"], tile["width"]), np.nan, dtype=np.float32)
    vv_stack = np.full((len(dates_list), tile["height"], tile["width"]), np.nan, dtype=np.float32)

    for ti, d in enumerate(dates_list):
        arr, pu = cached_fetch(token_holder, s2_script, "sentinel-2-l2a", tile, d, 5, "leastCC", "s2")
        cost_log["pu"] += pu
        mask = arr[..., 1] == 1
        ndvi_stack[ti][mask] = arr[..., 0][mask]

        arr, pu = cached_fetch(token_holder, s1_script, "sentinel-1-grd", tile, d, 5, None, "s1")
        cost_log["pu"] += pu
        mask = arr[..., 1] == 1
        vv_stack[ti][mask] = arr[..., 0][mask]

    crop_code, planting_day, confidence = _classify_stack(ndvi_stack, vv_stack, np.array(day_offsets, dtype=np.float32))
    return crop_code, planting_day, confidence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boundary", required=True)
    ap.add_argument("--start", default="2025-08-01")
    ap.add_argument("--end", default="2026-08-17")
    ap.add_argument("--n-dates", type=int, default=19)
    ap.add_argument("--out", default="../output/pixel_map_summary.json")
    ap.add_argument("--only-tile", default=None, help="process a single tile id (e.g. 1_1) for testing")
    args = ap.parse_args()

    config = get_config()
    token_holder = {"token": get_token(config)}
    s2_script = (EVALSCRIPT_DIR / "s2_ndvi_mask_raster.js").read_text()
    s1_script = (EVALSCRIPT_DIR / "s1_vv_mask_raster.js").read_text()

    tiles = build_tiles(args.boundary)
    if args.only_tile:
        tiles = [t for t in tiles if t["id"] == args.only_tile]
    dates_list = build_dates(args.start, args.end, args.n_dates)
    print(f"{len(tiles)} tiles, {len(dates_list)} dates, dates: {[d.isoformat() for d in dates_list]}")

    cost_log = {"pu": 0.0}
    totals = {"unknown": 0, "rice": 0, "corn": 0}
    per_tile_results = []

    t0 = time.time()
    for n, tile in enumerate(tiles):
        tt0 = time.time()
        crop_code, planting_day, confidence = process_tile(token_holder, tile, dates_list, s2_script, s1_script, cost_log)
        counts = {
            "unknown": int((crop_code == 0).sum()),
            "rice": int((crop_code == 1).sum()),
            "corn": int((crop_code == 2).sum()),
        }
        for k in totals:
            totals[k] += counts[k]

        out_dir = CACHE_DIR / "classified"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / f"tile{tile['id']}.npz", crop_code=crop_code,
                             planting_day=planting_day, confidence=confidence,
                             bbox=tile["bbox"], date0=dates_list[0].isoformat())

        per_tile_results.append({"id": tile["id"], "bbox": tile["bbox"], "counts": counts})
        print(f"[{n + 1}/{len(tiles)}] tile {tile['id']} ({tile['width']}x{tile['height']}) "
              f"-> rice={counts['rice']} corn={counts['corn']} unknown={counts['unknown']} "
              f"| {time.time() - tt0:.0f}s | cumulative PU={cost_log['pu']:.1f}", flush=True)

    pixel_area_m2 = RES_M * RES_M
    summary = {
        "pu_spent": cost_log["pu"],
        "elapsed_seconds": time.time() - t0,
        "n_tiles": len(tiles),
        "n_dates": len(dates_list),
        "dates": [d.isoformat() for d in dates_list],
        "pixel_counts": totals,
        "area_ha": {k: v * pixel_area_m2 / 10000 for k, v in totals.items()},
        "per_tile": per_tile_results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary["area_ha"], indent=2))
    print(f"PU spent: {cost_log['pu']:.1f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
