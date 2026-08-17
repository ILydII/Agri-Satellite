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
RICE_POST_DIP_NDVI_MIN = 0.40  # real canopy must follow the dip -- permanent water never shows this
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
    """Returns (dip_date, dip_rebound_db) for the most recent rice-like flood/transplant
    signature found.

    Groups the series into contiguous below-threshold episodes and takes each episode's
    deepest point as its candidate dip date -- NOT a per-point scan. VV rises gradually
    over weeks after transplant (not an instant step), so a per-point scan picking "the
    most recent point still under threshold" locks onto the tail end of the recovery
    rather than the actual dip, overstating the transplant date by however long the
    rebound took. Episodes are checked most-recent-first so a double-cropped field
    reports the CURRENT cycle's transplant date, not a stale earlier one.
    """
    valid = vv.notna()
    if valid.sum() < 4:
        return None
    vv_s = _smooth(vv)
    below = vv_s <= FLOOD_VV_DB

    episodes = []
    start = None
    for i in range(len(vv_s)):
        if below.iloc[i] and start is None:
            start = i
        elif not below.iloc[i] and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(vv_s) - 1))

    for ep_start, ep_end in reversed(episodes):
        local = vv_s.iloc[ep_start:ep_end + 1]
        i = local.idxmin()
        dip_value = vv_s.iloc[i]

        after = vv_s.iloc[ep_end + 1:]
        if len(after) < 2:
            continue
        rebound = after.max() - dip_value
        if rebound < FLOOD_REBOUND_DB:
            continue
        # NDVI near the dip date should still be low (bare/flooded), when we have optical coverage.
        ndvi_near = ndvi.iloc[max(i - 2, 0): i + 3]
        if ndvi_near.notna().any() and ndvi_near.dropna().mean() > FLOOD_NDVI_MAX:
            continue
        # A real paddy grows a canopy after flooding; permanent water (ocean, lakes, rivers)
        # never does, but can otherwise trivially fake the dip+rebound VV pattern from
        # wave/wind-driven backscatter noise. Require NDVI to actually reach vegetation
        # levels at some point after the dip to rule that out.
        ndvi_after = ndvi.iloc[i:]
        if ndvi_after.notna().sum() >= 2 and ndvi_after.max() < RICE_POST_DIP_NDVI_MIN:
            continue
        return dates.iloc[i], rebound
    return None


GREEN_UP_LOOKBACK_DAYS = 150  # how far back to search for the bare-soil trough before a rise


def _find_green_up(dates: pd.Series, ndvi: pd.Series):
    """Returns (green_up_date, rise_slope_per_day) marking the most recent sustained NDVI rise
    off a bare-soil baseline.

    Anchored to genuine "above threshold" EPISODE STARTS, not any index where two consecutive
    points happen to be above threshold -- once NDVI has risen, every later point on the same
    plateau redundantly re-satisfies that pairwise check too. Scanning those naively from the
    end would just grab the last point of the plateau and pair it with the original bare
    baseline months earlier, producing a wildly diluted (near-zero) slope instead of the real
    one. Episode boundaries fix that structurally: each episode is one rise event.

    The baseline trough is searched for in a bounded lookback window immediately before each
    episode's start -- NOT a single fixed window at the start of the series. Over a record
    spanning more than one crop cycle (e.g. extending --start back a year to catch an earlier
    planting), the real pre-rise trough can sit anywhere in the middle of the series, not at
    its start; a fixed "first 20%" baseline misses it entirely on multi-cycle records.

    Episodes are checked most-recent-first so a double-cropped field reports the CURRENT
    cycle's green-up, not a stale earlier one.
    """
    valid = ndvi.notna()
    if valid.sum() < 4:
        return None
    ndvi_s = _smooth(ndvi)
    above = ndvi_s >= GREEN_UP_NDVI
    bare = ndvi_s <= MIN_BASELINE_NDVI  # absolute bare-soil test, not relative to a computed baseline

    episodes = []
    start = None
    for i in range(len(ndvi_s)):
        if above.iloc[i] and start is None:
            start = i
        elif not above.iloc[i] and start is not None:
            episodes.append((start, i - 1))
            start = None
    if start is not None:
        episodes.append((start, len(ndvi_s) - 1))

    for ep_start, ep_end in reversed(episodes):
        if ep_end - ep_start < 1:
            continue  # require at least 2 sustained points, same bar as before
        crossing_date = dates.iloc[ep_start]
        lookback_start = crossing_date - pd.Timedelta(days=GREEN_UP_LOOKBACK_DAYS)
        # Last bare-soil point strictly before the crossing, within the lookback window --
        # this measures the actual rise rate, not diluted by an older, unrelated low point.
        bare_before = [j for j in range(ep_start) if bare.iloc[j] and dates.iloc[j] >= lookback_start]
        if not bare_before:
            continue
        j = bare_before[-1]
        baseline = ndvi_s.iloc[j]
        baseline_date = dates.iloc[j]
        days_elapsed = max((crossing_date - baseline_date).days, 1)
        slope = (ndvi_s.iloc[ep_start] - baseline) / days_elapsed
        return crossing_date, slope
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
