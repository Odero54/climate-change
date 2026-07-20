# End-to-end example: disease risk over a region in Kenya

This walks one analysis start to finish, twice: once the way most callers
should do it (`run_analysis`), then again by calling the underlying module
functions directly, so you can see exactly what `run_analysis` does on your
behalf. Both produce the same result.

We use the `disease` module (climate suitability for outbreak-prone disease,
e.g. malaria), but the shape is identical for `drought`, `flood`,
`food_security`, and `land_degradation` — only the module name and
`extra_params` change. Background on the concepts used below is in
[Concepts](concepts.md).

## 0. Prerequisites

Follow [Getting started](getting-started.md) first: package installed,
`earthengine authenticate` run once, and a Google Cloud project with the
Earth Engine API enabled.

```python
GEE_PROJECT = "your-google-cloud-project-id"

AOI = {
    "type": "Polygon",
    "coordinates": [[
        [34.20, -0.60], [34.60, -0.60],
        [34.60, -0.25], [34.20, -0.25],
        [34.20, -0.60],
    ]],
}  # Homa Bay area, Kenya
```

## 1. The one-call way

```python
import asyncio
from dataclasses import asdict
from climate_change import run_analysis

async def main():
    output = await run_analysis(
        module="disease",
        aoi_geojson=AOI,
        start_date="2024-01-01",
        end_date="2024-06-30",
        country="Kenya",
        gee_project=GEE_PROJECT,
        extra_params={
            "model_type": "gbm",     # "gbm" | "xgboost" | "ensemble"
            "scale": 1000,           # metres per pixel
            "n_pixels": 3000,        # training pixels to sample
            "output_dir": "outputs",
            "prefix": "disease_homabay",
        },
    )
    print(output.stats)
    print(output.raster_path)

asyncio.run(main())
```

`output` is an `AnalysisOutput`:

- **`output.geojson`** — risk-classified features for the AOI (what the map
  in a frontend renders).
- **`output.raster_path`** — dict of exported COG paths, one per model
  variant (`{"gbm": "outputs/disease_homabay_gbm.tif", ...}`).
- **`output.stats`** — e.g. per-class pixel percentages, hotspot count, and
  (best-effort) `total_population`/`population_affected` — see
  [Concepts § Population exposure](concepts.md#population-exposure).
- **`output.shap`** — SHAP feature importances for the primary model.
- **`output.charts`** — chart-ready payloads (e.g. monthly NDVI/rainfall
  trend, hotspot map data).
- **`output.metadata`** — model used, feature list, date range, country.

## 2. The same thing, one layer down

`run_analysis` → `core.runner.run_analysis` → `DiseaseRiskUseCase.execute()`
→ `fetch_data` → `preprocess` → `run_model`. Here's that pipeline called
directly, which is useful in a notebook when you want to inspect
intermediate results (e.g. the sampled training dataframe) rather than only
the final `AnalysisOutput`.

```python
import ee
from climate_change.core.gee_auth import ensure_gee
from climate_change.disease.features import (
    build_feature_datasets,
    build_gee_feature_stack,
    fetch_monthly_timeseries,
    sample_training_data,
)
from climate_change.disease.model import DiseaseModel
from climate_change.disease.cog_export import export_disease_cog

# Same authentication run_analysis performs internally.
ensure_gee(GEE_PROJECT)

aoi = ee.Geometry(AOI)
config = {"start_date": "2024-01-01", "end_date": "2024-06-30", "scale": 1000}

# Two parallel representations of the same 7 feature bands — see
# Concepts § Feature stacks. `run_analysis` fetches these three concurrently
# via a thread pool; sequential calls here for clarity.
datasets = build_feature_datasets(aoi, config)               # dict[str, xr.Dataset] — for COG export
feature_stack = build_gee_feature_stack(aoi, config)          # ee.Image — for sampling
timeseries = fetch_monthly_timeseries(aoi, config["start_date"], config["end_date"])

# Draw 3000 labeled training pixels from the server-side feature stack.
training_df = sample_training_data(feature_stack, aoi, n_pixels=3000, scale=1000)
print(training_df.columns)   # FEATURE_COLS + ['lon', 'lat', 'risk_score', 'label']
print(training_df["label"].value_counts())  # NOT forced into equal thirds — see Concepts

# Train both supported models; predict() returns the same stats/charts shape
# run_analysis would return inside AnalysisOutput.
model = DiseaseModel()
result = model.predict(training_df, timeseries=timeseries, config={"model_type": "gbm"})
print(result["stats"])

# Full-grid inference + COG export — same call use_case.run_model() makes.
raster_paths = export_disease_cog(
    gbm_model=model.gbm,
    xgb_model=model.xgb,
    scaler=model.scaler,
    datasets=datasets,
    output_dir="outputs",
    prefix="disease_homabay",
    model_type="gbm",
    aoi_geojson=AOI,
)
print(raster_paths)
```

`training_df["label"].value_counts()` is worth pausing on: because disease
risk labels come from fixed absolute score thresholds rather than sample
percentiles (see [Concepts § Composite risk scores](concepts.md#composite-risk-scores-and-fixed-thresholds)),
this AOI's actual class balance shows through instead of being forced to
~33/33/33 every time.

## 3. Optional: AI interpretation and a PDF report

Both bolt onto the one-call path — pass the extra arguments and re-run:

```python
output = await run_analysis(
    module="disease",
    aoi_geojson=AOI,
    start_date="2024-01-01",
    end_date="2024-06-30",
    country="Kenya",
    gee_project=GEE_PROJECT,
    extra_params={"model_type": "gbm", "output_dir": "outputs", "prefix": "disease_homabay"},
    openai_api_key="your-openai-api-key",   # -> output.metadata["ai_interpretation"]
    report_output_dir="reports",             # -> output.metadata["report_path"], a PDF
)

print(output.metadata["ai_interpretation"])
print(output.metadata["report_path"])
```

Or, using the same lower-level pieces from step 2:

```python
from climate_change import AIInterpreter, ReportBuilder

interpretation = AIInterpreter("your-openai-api-key").interpret(output)
report_path = ReportBuilder("reports/disease_homabay.pdf").build(
    output, ai_interpretation=interpretation
)
```

## Next steps

- Swap `module="disease"` for `"drought"`, `"flood"`, `"food_security"`, or
  `"land_degradation"` — the calling shape is identical; only
  `extra_params` and the domain module you import from change. See the
  runnable notebook per module under
  [`example-usage/`](https://github.com/Odero54/climate-change/tree/main/example-usage).
- Look up any function or class used above in the [API reference](api/top-level.md).
