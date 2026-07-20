# Concepts

This page explains the ideas that recur across all five modules. Read it
once and every module's code reads the same way.

## Area of interest (AOI)

Every analysis starts from a GeoJSON `Polygon`, `MultiPolygon`, `Feature`, or
`FeatureCollection` describing the region to analyze. Internally, this is
converted two ways depending on what the code needs next:

- to an `ee.Geometry`, for querying Google Earth Engine;
- to a list of raw geometry dicts, for `rasterio`-style pixel masking (e.g.
  clipping a numpy grid to the AOI boundary before exporting a raster).

Both conversions live in `core/base_use_case.py` and are shared by every
module — see [`core`](api/core.md).

## The `AnalysisConfig` → `AnalysisOutput` contract

Every module accepts the same input shape and returns the same output shape:

- **`AnalysisConfig`** — `module`, `aoi_geojson`, `start_date`, `end_date`,
  `country`, and a free-form `extra_params` dict for model/module-specific
  overrides (`model_type`, `scale`, `n_pixels`, …).
- **`AnalysisOutput`** — `geojson` (risk/severity features), `raster_path`
  (COG path(s)), `stats` (summary metrics), `shap` (feature importances, or
  `None`), `charts` (frontend-ready chart payloads), `metadata` (model name,
  features used, country, dates, and optional AI/report info).

`stats` and `charts` are intentionally untyped dicts — their contents are
module-specific (a flood risk distribution looks nothing like a drought
forecast), so treat them as such rather than assuming a fixed schema across
modules.

## The pipeline: fetch → preprocess → model → explain

Every domain module (`DroughtUseCase`, `FloodRiskUseCase`,
`FoodSecurityUseCase`, `DiseaseRiskUseCase`, `LandDegradationUseCase`)
subclasses `BaseUseCase` and implements the same three stages:

```text
fetch_data(config)       → raw EO data (bbox, GEE handles, etc.)
preprocess(raw, config)  → feature arrays/dataframes, ready for modeling
run_model(features, cfg) → AnalysisOutput
explain(features)        → optional SHAP payload, merged into the output
```

`BaseUseCase.execute()` (called by `core.runner.run_analysis`, which is what
the top-level `run_analysis()` function delegates to) runs these four stages
in order, offloading each synchronous stage to a thread so the event loop
stays free for concurrent requests. It also checks an in-process
[cache](#caching) first and skips the whole pipeline on a hit.

Read [`core.BaseUseCase`](api/core.md) for the exact contract, and any
module's `use_case.py` (e.g. [`disease`](api/disease.md)) to see it
implemented.

## Feature stacks: two representations of the same data

Every module builds its Earth Engine feature bands in **two parallel
forms**, because they serve different downstream needs:

- **`build_feature_datasets(aoi, config) -> dict[str, xr.Dataset]`** —
  downloads each band as a local `xarray` raster (used for full-grid
  inference and COG export).
- **`build_gee_feature_stack(aoi, config) -> ee.Image`** — assembles the same
  bands as a *server-side* multi-band image, used for `sample_training_data`
  to draw labeled training pixels without downloading the whole raster.

Both exclude water (and, for food security, built-up land) using a shared
`exclusion_mask` from [`core.landcover_mask`](api/core.md) — risk scores are
a property of inhabited/inhabitable land, not open water or cities.

Independent band downloads run concurrently via
`DaskEngine.run_io_parallel`, which fans a dict of thunks out across a
`ThreadPoolExecutor` sharing one authenticated GEE session — see
[`core.DaskEngine`](api/core.md).

## Composite risk scores and fixed thresholds

Disease, food security, and land degradation don't have abundant
ground-truth labels, so training labels come from a **composite suitability
score**: a weighted sum of normalized indicators (temperature suitability,
rainfall, standing water, vegetation trend, etc.), each scaled to 0–100, with
weights summing to 1.

The critical design rule: **classify that score against fixed absolute
cutoffs, not percentiles of the current sample.** For example, disease and
food security use `RISK_SCORE_THRESHOLDS = (33.0, 66.0)`; land degradation
uses `DEGRADED_SCORE_THRESHOLD = 50.0`. Using `np.percentile()` on the
sample's own score distribution instead would force the same class
proportions (e.g. exactly one-third Low/Medium/High) on *every* AOI
regardless of its actual severity — a uniformly low-risk region and a
uniformly high-risk region would both come out identical. Fixed cutoffs let
an AOI's label distribution actually reflect its real conditions.

Flood and drought differ: flood labels come from real JRC-observed
flood-year satellite imagery (genuine ground truth, no synthetic score
needed), and drought severity is classified against a standard published CDI
scale (also fixed, not sample-relative).

## Models

Each module trains one or more models and — where more than one is
available — an ensemble:

| Module | Models | `extra_params["model_type"]` |
| --- | --- | --- |
| `drought` | LSTM, or the `drought_monitoring` package's statistical forecaster | `lstm`, `drought_monitoring` |
| `flood` | Random Forest, XGBoost | `rf`, `xgboost`, `ensemble` |
| `food_security` | Random Forest, XGBoost | `rf`, `xgboost`, `ensemble` |
| `disease` | Gradient Boosting, XGBoost | `gbm`, `xgboost`, `ensemble` |
| `land_degradation` | Random Forest, LightGBM | `rf`, `lgbm`, `ensemble` |

Each module's `model.py` always trains all of its supported models (so an
ensemble is always available cheaply) and `model_type` just selects which
prediction(s) populate the final output.

## Explainability (SHAP)

Where a module's model supports it, `compute_shap_importance` produces
per-feature SHAP values, surfaced as `AnalysisOutput.shap`. This is what lets
the AI interpreter and PDF report explain *why* a region was scored the way
it was, not just what the score is.

## Cloud-Optimized GeoTIFF (COG) export

Full-grid model inference (every pixel in the AOI, not just sampled training
points) is written out as a COG — a GeoTIFF optimized for partial/streamed
reads, suitable for direct use in web maps. Each module's `cog_export.py`
handles this (e.g. [`export_disease_cog`](api/disease.md)); the resulting
path(s) populate `AnalysisOutput.raster_path`.

## Population exposure

Area-based risk (`risk_pct`, `area_ha`) doesn't tell you how many people are
actually affected — a polygon that's 63% high flood risk could be empty
rangeland or a market town. [`core.population`](api/core.md) turns each
module's risk-classified grid into a headcount by taking a **raster zonal
sum** of WorldPop gridded population (`WorldPop/GP/100m/pop`, people per
pixel — an additive count, not a density) against the classification, per
class.

Two things make this correct rather than merely plausible:

- **Population is never downsampled.** WorldPop is always fetched at its
  native ~100 m resolution; a coarser risk grid is *upsampled* onto
  population's grid via nearest-neighbour (`upsample_class_grid`) — the
  semantically correct direction for a categorical grid — instead of
  aggregating population server-side, which was verified against live GEE
  data to silently under/over-count by tens of percent.
- **It isn't an area-weighted estimate.** Unlike `_attach_risk_area`, which
  assumes uniform density within a class share, `population_exposure()`
  sums real gridded population per pixel, so it reflects actual
  concentration of people, not an assumption.

Each module attaches `stats["total_population"]` and
`stats["population_affected"]` (population within that module's at-risk
classes — e.g. Medium + High for disease/food security, Degraded for land
degradation, Extreme + Severe drought for drought) plus a parallel
`data_population` array on the risk-distribution chart. All of it is
best-effort: `fetch_population_count_safe` returns `None` on any failure
(no WorldPop coverage, network error), and downstream code simply omits
these fields rather than failing the analysis — see
[`core.population`](api/core.md#population-exposure).

## Caching

Two independent cache layers exist, for different lifetimes:

- **`analysis_cache`** (`core.cache.SimpleCache`) — in-process, TTL-based
  (default 1 hour), keyed by a hash of module + dates + rounded AOI
  coordinates. Skips the *entire* pipeline on a hit. Swap for Redis in a
  production API layer.
- **`feature_cache`** (`core.cache.feature_cache`, a `joblib.Memory`) —
  disk-backed, used to memoize expensive `preprocess()` stages (e.g. GEE
  downloads) across process restarts. Location configurable via
  `ARIN_CACHE_DIR`.

## Distributed execution (Dask)

`DaskEngine` is a per-process singleton wrapping a Dask `LocalCluster`. It
has two distinct uses: `run_io_parallel` for fanning out independent
*network* calls (GEE downloads) across threads, and `submit`/`gather` for
genuinely CPU-bound work (e.g. training multiple regions in parallel via
each module's `run_multi_regions`). Worker count/memory are configurable via
`DASK_WORKERS`, `DASK_THREADS_PER_WORKER`, `DASK_MEMORY_LIMIT`.

## The registry

`USE_CASE_REGISTRY` (in top-level `registry.py`) holds one `UseCaseInfo` per
module: display name, icon, description, the variables it models, its best
model, accuracy notes, and its list of selectable `ModelOption`s. This
single source of truth drives both frontend UI copy and the AI
interpreter's prompt context — see [`registry`](api/registry.md).

## AI interpretation and PDF reports

Both are optional, bolted onto `run_analysis` after the core pipeline:

- Passing `openai_api_key` runs `AIInterpreter.interpret()`, which builds a
  module-aware prompt via `build_prompt()` (registry description + stats +
  module-specific chart context) and returns a policymaker-friendly summary,
  stored at `output.metadata["ai_interpretation"]`.
- Passing `report_output_dir` runs `ReportBuilder.build()`, producing a PDF
  with a cover page, map, stats, module-specific sections, the AI
  interpretation (if present), and a glossary — path stored at
  `output.metadata["report_path"]`.

See [End-to-end example](end-to-end-example.md) for both in context.
