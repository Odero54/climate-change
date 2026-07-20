# climate-change

`climate-change` is the core Python computation package behind the ARIN
Climate Resilience Decision Support System. It turns satellite and climate
data into decision-ready risk assessments for five hazards:

| Module | Question it answers | Default model |
| --- | --- | --- |
| `drought` | How severe is drought now, and over the next 6 months? | LSTM |
| `flood` | Which areas are at risk of flooding, and how severe? | RF + XGBoost ensemble |
| `food_security` | Where is vegetation/climate stress threatening food security? | Random Forest |
| `disease` | Where are conditions climatically suitable for outbreak-prone disease? | Gradient Boosting |
| `land_degradation` | Where is vegetation declining and rangeland degrading? | LightGBM |

It is a **library, not a service** — no CLI, no web server. You call it from
Python: a script, a notebook, a FastAPI route, or a background task worker.

## The shape of every analysis

All five modules share one calling convention and one result shape, so
learning one teaches you all five:

```python
output = await run_analysis(
    module="disease",           # "drought" | "flood" | "food_security" | "disease" | "land_degradation"
    aoi_geojson=my_polygon,     # GeoJSON Polygon/MultiPolygon/Feature/FeatureCollection
    start_date="2024-01-01",
    end_date="2024-06-30",
    country="Kenya",
    gee_project="your-gcp-project",
)
```

and every module returns the same `AnalysisOutput` dataclass — `geojson`,
`raster_path`, `stats`, `shap`, `charts`, `metadata` — regardless of which
hazard you asked about. See [Concepts](concepts.md) for why that's true (all
five modules implement the same `fetch → preprocess → model` pipeline) and
[End-to-end example](end-to-end-example.md) for a full worked run, including
the lower-level module/function calls that `run_analysis` makes on your
behalf.

## Sample output

`output.stats` is a flat dict of summary indicators — model performance,
risk-class shares, dominant drivers, and (best-effort) population exposure.
Exact keys/values depend on the AOI, date range, and model chosen; these are
representative examples.

=== "drought"

    ```python
    {
        "model_type": "lstm",
        "mean_cdi": 0.7675,
        "latest_mean_cdi": 0.7909,
        "extreme_pct": 0.0,
        "severe_pct": 7.7,
        "moderate_pct": 43.0,
        "near_normal_pct": 15.2,
        "lstm_rmse": 0.162656,
        "total_population": 214830.0,
        "population_affected": 16542.1,   # Extreme + Severe drought
        "country": "Kenya",
    }
    ```

=== "flood"

    ```python
    {
        "model_type": "ensemble",
        "flooded_pct": 71.1,
        "selected_f1": 0.8,
        "selected_auc": 0.83,
        "top_flood_driver": "dist_river",
        "very_high_risk_pct": 4.0,
        "high_risk_pct": 17.8,
        "medium_risk_pct": 54.3,
        "low_risk_pct": 23.9,
        "total_population": 58210.0,
        "population_affected": 43109.4,   # Medium + High + Very High
        "country": "Niger",
    }
    ```

=== "food_security"

    ```python
    {
        "model_type": "rf",
        "selected_f1": 0.9406,
        "top_driver": "vci",
        "vci_mean": 31.2,
        "tci_mean": 53.2,
        "vhi_mean": 42.2,
        "high_risk_pct": 33.1,
        "medium_risk_pct": 34.2,
        "low_risk_pct": 32.8,
        "total_area_ha": 7305844.8,
        "total_population": 892110.0,
        "population_affected": 610732.5,  # Medium + High Risk
        "country": "Kenya",
    }
    ```

=== "disease"

    ```python
    {
        "model_type": "gbm",
        "selected_f1": 0.9677,
        "high_risk_pct": 33.3,
        "n_hotspot_clusters": 6,
        "top_driver": "temp_mean",
        "total_population": 431200.0,
        "population_affected": 287640.9,  # Medium + High Risk
        "country": "Kenya",
    }
    ```

=== "land_degradation"

    ```python
    {
        "model_type": "lgbm",
        "selected_f1": 0.9597,
        "degraded_label_pct": 30.0,
        "top_degradation_driver": "ndvi_slope",
        "ndvi_trend_per_year": 0.00604,
        "mk_significant": True,
        "breakpoint_years": [2017, 2019, 2022],
        "total_population": 168450.0,
        "population_affected": 50535.0,   # Degraded only
        "country": "Burkina Faso",
    }
    ```

Each risk-distribution entry in `output.charts` carries a parallel
`data_population` array alongside the existing percentage `data` — e.g.
`output.charts["riskDist"]`:

```python
{
    "labels": ["Low Risk", "Medium Risk", "High Risk"],
    "data": [33.4, 33.3, 33.3],           # % of sampled pixels
    "data_population": [143560.0, 156210.4, 131429.6],  # people, per class
    "colors": ["#2ECC71", "#F1C40F", "#E74C3C"],
}
```

`total_population`/`population_affected`/`data_population` are best-effort —
see [Concepts § Population exposure](concepts.md#population-exposure) for
why they're sometimes absent and how the headcount is actually computed.

## Where to go next

- **New to the package?** Start with [Getting started](getting-started.md) —
  installing it and authenticating Google Earth Engine.
- **Want the mental model before writing code?** Read [Concepts](concepts.md)
  — AOIs, feature stacks, composite risk scoring, SHAP, COG exports, caching.
- **Want to see it run start to finish?** Read the
  [end-to-end example](end-to-end-example.md), which walks a disease-risk
  analysis both through the one-call API and through the underlying module
  calls.
- **Looking up a specific function or class?** Jump straight to the
  [API reference](api/top-level.md).

## Requirements

- Python 3.10–3.13
- A Google Cloud project with the Earth Engine API enabled, and Earth Engine
  credentials available on the machine running the analysis
- Enough memory/disk for geospatial + ML workloads — runtime scales with AOI
  size, date range, and spatial resolution

## Links

- Repository: <https://github.com/Odero54/climate-change>
- PyPI: <https://pypi.org/project/climate-change/>
- Issues: <https://github.com/Odero54/climate-change/issues>
