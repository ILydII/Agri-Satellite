"""Sanity checks for crop_classifier against synthetic NDVI/VV curves shaped like the
published rice flood-transplant and corn green-up signatures. No pytest dependency --
run directly:

    cd src
    python test_crop_classifier.py

There's no live Sentinel Hub credential in this environment to test against real
imagery, so this is what stands in until real labeled plots are available.
"""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from crop_classifier import classify_and_estimate


def _dates(n_days, step=5):
    start = date(2026, 1, 1)
    return [start + timedelta(days=i) for i in range(0, n_days, step)]


def make_rice_series(transplant_day=20, n_days=150):
    """VV dips at transplant (flooding) then rebounds; NDVI stays low through flooding,
    then climbs to a mid-season peak and senesces."""
    dates = _dates(n_days)
    rows = []
    for d in dates:
        t = (d - dates[0]).days
        if t < transplant_day:
            vv = -11.0 + np.random.normal(0, 0.3)          # bare/prepared field
            ndvi = 0.15 + np.random.normal(0, 0.02)
        elif t < transplant_day + 10:
            vv = -19.0 + np.random.normal(0, 0.4)           # flood dip at transplant
            ndvi = 0.10 + np.random.normal(0, 0.02)
        else:
            days_since = t - transplant_day
            vv = -19.0 + min(days_since / 40.0, 1.0) * 10 + np.random.normal(0, 0.3)
            ndvi = 0.10 + min(days_since / 60.0, 1.0) * 0.65 + np.random.normal(0, 0.02)
            if days_since > 90:
                ndvi -= (days_since - 90) * 0.004  # senescence
        rows.append({"date": pd.Timestamp(d), "ndvi_mean": ndvi, "vv_db_mean": vv})
    return pd.DataFrame(rows)


def make_corn_series(planting_day=15, n_days=150):
    """No flood dip; NDVI rises quickly off a bare-soil baseline."""
    dates = _dates(n_days)
    rows = []
    for d in dates:
        t = (d - dates[0]).days
        vv = -11.0 + min(max(t - planting_day, 0) / 50.0, 1.0) * 4 + np.random.normal(0, 0.3)
        if t < planting_day + 8:
            ndvi = 0.18 + np.random.normal(0, 0.02)
        else:
            days_since = t - (planting_day + 8)
            ndvi = 0.18 + min(days_since / 25.0, 1.0) * 0.62 + np.random.normal(0, 0.02)
            if days_since > 70:
                ndvi -= (days_since - 70) * 0.006
        rows.append({"date": pd.Timestamp(d), "ndvi_mean": ndvi, "vv_db_mean": vv})
    return pd.DataFrame(rows)


def make_multi_cycle_series(n_days=500):
    """A prior crop already at peak when the record starts, harvested/fallow around day 110-150,
    then a second, later planting green-up around day 160 -- the trough that should anchor the
    age estimate sits in the MIDDLE of the record, not at its start. Regression fixture for the
    bug where a fixed first-20%-of-series baseline missed a mid-record replanting entirely."""
    dates = _dates(n_days)
    rows = []
    for d in dates:
        t = (d - dates[0]).days
        if t < 100:
            ndvi = 0.72 + np.random.normal(0, 0.02)          # prior crop already mature
        elif t < 150:
            ndvi = 0.72 - (t - 100) / 50.0 * 0.55 + np.random.normal(0, 0.02)  # senescence -> bare
        elif t < 170:
            ndvi = 0.17 + np.random.normal(0, 0.02)          # fallow / bare soil
        else:
            days_since = t - 170
            ndvi = 0.17 + min(days_since / 25.0, 1.0) * 0.6 + np.random.normal(0, 0.02)
            if days_since > 70:
                ndvi -= (days_since - 70) * 0.005
        vv = -10.0 + np.random.normal(0, 0.3)  # flat -- no flood signature anywhere
        rows.append({"date": pd.Timestamp(d), "ndvi_mean": ndvi, "vv_db_mean": vv})
    return pd.DataFrame(rows)


def run():
    np.random.seed(0)

    as_of = date(2026, 1, 1) + timedelta(days=100)

    rice_df = make_rice_series(transplant_day=20)
    est = classify_and_estimate(rice_df, as_of=as_of)
    print("rice fixture ->", est)
    assert est.crop_type == "rice", f"expected rice, got {est.crop_type}"
    assert est.crop_confidence in ("high", "medium")
    planted = date.fromisoformat(est.planting_date_est)
    assert abs((planted - date(2026, 1, 21)).days) <= 5, f"planting date off: {planted}"
    assert est.age_days == (as_of - planted).days

    corn_df = make_corn_series(planting_day=15)
    est = classify_and_estimate(corn_df, as_of=as_of)
    print("corn fixture ->", est)
    assert est.crop_type == "corn", f"expected corn, got {est.crop_type}"
    planted = date.fromisoformat(est.planting_date_est)
    assert abs((planted - date(2026, 1, 16)).days) <= 8, f"planting date off: {planted}"

    empty_df = pd.DataFrame(columns=["date", "ndvi_mean", "vv_db_mean"])
    est = classify_and_estimate(empty_df, as_of=as_of)
    print("empty fixture ->", est)
    assert est.crop_type == "unknown"
    assert est.age_days is None

    already_grown = pd.DataFrame({
        "date": [pd.Timestamp(date(2026, 1, 1) + timedelta(days=i)) for i in range(0, 30, 5)],
        "ndvi_mean": [0.75] * 6,
        "vv_db_mean": [-9.0] * 6,
    })
    est = classify_and_estimate(already_grown, as_of=as_of)
    print("already-grown fixture ->", est)
    assert est.crop_type == "unknown"
    assert est.planting_date_est is None

    multi_cycle_df = make_multi_cycle_series()
    est = classify_and_estimate(multi_cycle_df, as_of=date(2026, 1, 1) + timedelta(days=250))
    print("multi-cycle fixture ->", est)
    assert est.crop_type == "corn", f"expected corn (mid-record replant), got {est.crop_type}"
    planted = date.fromisoformat(est.planting_date_est)
    assert abs((planted - (date(2026, 1, 1) + timedelta(days=171))).days) <= 8, f"planting date off: {planted}"

    print("\nAll crop_classifier sanity checks passed.")


if __name__ == "__main__":
    run()
