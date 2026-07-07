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
