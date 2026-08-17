"""Heuristic crop-type (rice vs. corn) classification and planting-date/age estimation
from Sentinel-1/2 time series.

This is NOT a trained model -- there's no local ground-truth yet to train one against.
It pattern-matches the observed NDVI/VV curve against two published phenology signatures:

- Rice (transplanted paddy): a sharp VV backscatter dip at flooding (specular reflection
  off standing water) while NDVI is still low, followed by a rebound as the canopy fills
  in. This is the standard SAR flood/transplant-date signature (e.g. Bouvet & Le Toan
  2011). It's the strongest signal in this module because it doesn't depend on cloud-free
  optical passes.
- Corn (rainfed, no flooding): no flood dip. NDVI rises from a bare-soil baseline to
  canopy closure considerably faster than rice's post-transplant crawl.

crop_confidence is a heuristic strength-of-match label, not a statistical probability.
Validate against real labeled plots once ground truth exists -- see README.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

# Thresholds below are starting points from published rice-paddy SAR flood-signature and
# maize/rice phenology literature, not locally calibrated. Revisit once real plots with
# known planting dates are available.
FLOOD_VV_DB = -16.0          # VV at/below this, alongside low NDVI, reads as standing water
FLOOD_NDVI_MAX = 0.30        # NDVI must still be low (bare/flooded) at the VV minimum
FLOOD_REBOUND_DB = 3.0       # minimum dB rebound after the dip to confirm it's flooding, not noise
GREEN_UP_NDVI = 0.30         # NDVI threshold marking crop emergence
MIN_BASELINE_NDVI = 0.25     # pre-green-up baseline must sit below this to count as bare soil
CORN_STEEP_SLOPE = 0.012     # NDVI units/day; corn closes canopy faster than post-transplant rice
CORN_EMERGENCE_LAG_DAYS = 8  # typical maize planting-to-visible-emergence lag
RICE_CYCLE_DAYS = 125        # typical tropical transplanted rice, transplant to harvest
CORN_CYCLE_DAYS = 105        # typical tropical maize, planting to harvest


@dataclass
class CropEstimate:
    crop_type: str              # "rice" | "corn" | "unknown"
    crop_confidence: str        # "high" | "medium" | "low"
    planting_date_est: str | None
    age_days: int | None
    growth_stage: str | None
    notes: str

    def as_row(self) -> dict:
        return {
            "crop_detected": self.crop_type,
            "crop_confidence": self.crop_confidence,
            "planting_date_est": self.planting_date_est,
            "crop_age_days": self.age_days,
            "growth_stage": self.growth_stage,
            "notes": self.notes,
        }


def _smooth(series: pd.Series) -> pd.Series:
    return series.interpolate(limit_direction="both").rolling(3, center=True, min_periods=1).median()


def _find_flood_dip(dates: pd.Series, ndvi: pd.Series, vv: pd.Series):
    """Returns (dip_date, dip_depth_db) if a rice-like flood/transplant signature is found."""
    valid = vv.notna()
    if valid.sum() < 4:
        return None
    vv_s = _smooth(vv)
    idx_min = vv_s.idxmin()
    dip_value = vv_s.loc[idx_min]
    if dip_value > FLOOD_VV_DB:
        return None

    after = vv_s.loc[idx_min:]
    if len(after) < 3:
        return None
    rebound = after.iloc[1:].max() - dip_value
    if rebound < FLOOD_REBOUND_DB:
        return None

    # NDVI near the dip date should still be low (bare/flooded), when we have optical coverage.
    ndvi_near = ndvi.loc[max(idx_min - 2, ndvi.index.min()):min(idx_min + 2, ndvi.index.max())]
    if ndvi_near.notna().any() and ndvi_near.dropna().mean() > FLOOD_NDVI_MAX:
        return None

    return dates.loc[idx_min], (after.iloc[1:].max() - dip_value)


def _find_green_up(dates: pd.Series, ndvi: pd.Series):
    """Returns (green_up_date, rise_slope_per_day) marking the earliest sustained NDVI rise
    off a bare-soil baseline."""
    valid = ndvi.notna()
    if valid.sum() < 4:
        return None
    ndvi_s = _smooth(ndvi)

    baseline_window = max(int(len(ndvi_s) * 0.2), 3)
    baseline = ndvi_s.iloc[:baseline_window].median()
    if baseline > MIN_BASELINE_NDVI:
        return None  # already vegetated at query start -- can't see the onset

    above = ndvi_s >= GREEN_UP_NDVI
    still_bare = ndvi_s <= (baseline + 0.05)
    for i in range(baseline_window, len(ndvi_s) - 1):
        if above.iloc[i] and above.iloc[i + 1]:
            green_up_date = dates.iloc[i]
            # Measure the rise itself (last still-bare point -> crossing point), not diluted
            # by however long the fallow baseline period happened to last.
            bare_idx = [j for j in range(i) if still_bare.iloc[j]]
            j = bare_idx[-1] if bare_idx else max(baseline_window - 1, 0)
            days_elapsed = max((green_up_date - dates.iloc[j]).days, 1)
            slope = (ndvi_s.iloc[i] - ndvi_s.iloc[j]) / days_elapsed
            return green_up_date, slope
    return None


def _growth_stage(age_days: int, cycle_days: int) -> str:
    frac = age_days / cycle_days
    if frac < 0:
        return "not yet planted (estimated planting date is after as-of date)"
    if frac < 0.15:
        return "establishment"
    if frac < 0.45:
        return "vegetative"
    if frac < 0.65:
        return "reproductive"
    if frac < 0.95:
        return "ripening"
    return "mature / past typical harvest window"


def classify_and_estimate(df: pd.DataFrame, as_of: date | None = None) -> CropEstimate:
    """df must have a 'date' column plus 'ndvi_mean' and 'vv_db_mean' columns (as produced by
    run_pipeline's merged per-plot DataFrame). as_of defaults to today."""
    as_of = as_of or date.today()

    df = df.dropna(subset=["ndvi_mean", "vv_db_mean"], how="all").sort_values("date").reset_index(drop=True)
    if df.empty:
        return CropEstimate("unknown", "low", None, None, None,
                             "no usable NDVI or VV observations in the queried window")

    dates = df["date"]
    ndvi = df["ndvi_mean"]
    vv = df["vv_db_mean"]

    flood = _find_flood_dip(dates, ndvi, vv)
    if flood is not None:
        dip_date, rebound_db = flood
        confidence = "high" if rebound_db >= 5.0 else "medium"
        planting_date = pd.Timestamp(dip_date).date()
        age_days = (as_of - planting_date).days
        return CropEstimate(
            "rice", confidence, planting_date.isoformat(), age_days,
            _growth_stage(age_days, RICE_CYCLE_DAYS),
            f"flood/transplant VV dip detected ({rebound_db:.1f} dB rebound)",
        )

    green_up = _find_green_up(dates, ndvi)
    if green_up is not None:
        green_up_date, slope = green_up
        if slope >= CORN_STEEP_SLOPE:
            planting_date = pd.Timestamp(green_up_date).date() - timedelta(days=CORN_EMERGENCE_LAG_DAYS)
            age_days = (as_of - planting_date).days
            return CropEstimate(
                "corn", "medium", planting_date.isoformat(), age_days,
                _growth_stage(age_days, CORN_CYCLE_DAYS),
                f"no flood signature; fast NDVI green-up ({slope:.3f}/day) matches corn canopy closure",
            )
        return CropEstimate(
            "unknown", "low", None, None, None,
            f"NDVI green-up detected ({slope:.3f}/day) but too slow for corn and no flood "
            "signature for rice -- possibly direct-seeded rice, another crop, or a noisy signal",
        )

    return CropEstimate(
        "unknown", "low", None, None, None,
        "no flood dip or NDVI green-up onset in the queried window -- the crop was likely "
        "already established before --start, or cloud/gap coverage is too sparse; try "
        "extending --start further back",
    )
