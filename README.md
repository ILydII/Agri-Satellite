# Satellite input-timing pipeline (Week 1 test)

Pulls Sentinel-2 (NDVI/NDRE, optical) and Sentinel-1 (VV/VH, radar) time series per farmer
plot from the Copernicus Data Space Ecosystem's Statistical API — no imagery download, no
commercial-use licensing catch (unlike Google Earth Engine).

## 1. Get API credentials (one-time, ~2 minutes, you do this — not me)

1. Create a free account at https://dataspace.copernicus.eu
2. Go to Dashboard -> User Settings -> OAuth clients -> create a new client
3. Copy the client ID and client secret
4. Copy `.env.example` to `.env` and paste them in:
   ```
   copy .env.example .env
   ```
   then edit `.env`.

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run against the placeholder test plots

Three synthetic ~0.25 ha boxes over rice-paddy country near Karawang, West Java — **not real
farmers**, just there to prove the pipeline runs end to end before real data arrives.

```bash
cd src
python run_pipeline.py --plots ../config/test_plots.csv --start 2026-01-01 --end 2026-08-01
```

Outputs land in `output/`: one CSV + one PNG chart per plot, a combined
`all_plots_timeseries.csv`, and a `crop_summary.csv` with the detected crop type, an
estimated planting date, and current crop age per plot (see below).

## 4. Swap in real farmer data

Fill in `config/farmers_template.csv` (or a copy of it) with real plot polygons and known
purchase dates, then rerun with `--plots` pointed at that file. Plot boundaries matter more
than precision — a boundary traced by eye on a satellite basemap beats a GPS point + guessed
radius, which frequently grabs the wrong field. Google My Maps or geojson.io both export WKT/
GeoJSON polygons you can convert into the `wkt_polygon` column.

## What the output means

- `ndvi_mean` / `ndre_mean`: vegetation vigor, cloud-masked (see `src/evalscripts/s2_ndvi_ndre.js`)
- `*_valid_px`: how many cloud-free pixels contributed to that date's mean — treat any date
  with a low count as low-confidence, especially on the smallest plots
- `vv_db_mean` / `vh_db_mean`: radar backscatter in dB — fills the gaps between usable optical
  passes and independently tracks crop-stage timing (rice's flood/transplant signature shows
  up clearly in VV)

## Crop type & age (`crop_summary.csv`)

`src/crop_classifier.py` distinguishes **rice vs. corn** and estimates a planting date
and current age per plot from the NDVI/VV time series. This is a heuristic
phenology-signature matcher, not a trained model -- there's no local ground truth to
train one against yet:

- **Rice**: detected from the SAR flood/transplant signature -- a sharp VV backscatter
  dip at flooding (standing water reflects specularly) while NDVI is still low, followed
  by a rebound as the canopy fills in. Planting date = the date of that VV dip. This is
  the strongest signal here since it doesn't depend on cloud-free optical passes.
- **Corn**: no flood dip, but a fast NDVI rise off a bare-soil baseline (canopy closes
  quicker than rice's post-transplant crawl). Planting date = NDVI green-up onset minus
  a ~8-day emergence lag.
- If neither pattern is found -- e.g. the crop was already established before `--start`,
  or cloud/gap coverage is too sparse -- the row comes back `unknown` with a `notes`
  explanation instead of a guessed date.

`crop_confidence` is a strength-of-match label (`high`/`medium`/`low`), not a
statistical probability. Treat it as a lead to check in the field, especially until it's
validated against plots with a known planting date. `--asof YYYY-MM-DD` fixes the "today"
used for age math (defaults to the real today) -- useful for reproducible runs.

Sanity-checked against synthetic rice/corn curves (no live imagery needed):
```bash
cd src
python test_crop_classifier.py
```

## Known limits (see the feasibility study)

- Reliable per-plot signal degrades below ~0.3 ha or on irregular/intercropped fields — that's
  a resolution limit of free 10m imagery, not a bug in this code
- A vegetation dip is a lead, not a diagnosis — nutrient stress, water stress, pest pressure,
  and normal senescence all suppress NDVI/NDRE in overlapping ways
