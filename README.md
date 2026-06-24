# Climate Change

[![PyPI version](https://img.shields.io/pypi/v/climate-change?label=pypi)](https://pypi.org/project/climate-change/)
[![Python versions](https://img.shields.io/pypi/pyversions/climate-change?label=python)](https://pypi.org/project/climate-change/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](#license)

`climate-change` is the core Python package for the ARIN Climate Resilience
Decision Support System. It combines Earth observation data, geospatial
processing, machine-learning models, distributed execution, explainability,
AI-assisted interpretation, and PDF reporting in one analysis library.

The package supports five climate-risk use cases:

- drought monitoring and early warning;
- flood-risk monitoring and mitigation;
- food-security assessment;
- climate-driven disease surveillance; and
- rangeland dynamics and land-degradation assessment.

Each analysis accepts an area of interest (AOI), date range, country or region
name, and optional model settings. It returns a common `AnalysisOutput` object
containing GeoJSON, raster paths, statistics, SHAP explanations, chart data,
and metadata.

> [!IMPORTANT]
> This repository contains the computation package, not a web application or
> command-line interface. It is designed to be called from Python, notebooks,
> task workers, or an API backend.

## Features

- Google Earth Engine integration for satellite and climate datasets
- Five analysis modules behind one asynchronous Python API
- Random Forest, XGBoost, LightGBM, Gradient Boosting, and LSTM models
- Dask-based local or distributed processing
- Cloud-Optimized GeoTIFF (COG) exports
- GeoJSON risk and severity layers
- SHAP feature-importance payloads where supported
- Optional OpenAI interpretation for policymaker-friendly summaries
- Optional PDF reports with maps, charts, findings, and recommendations
- In-process caching for repeated analyses

## Use cases

| Module ID | Analysis | Default runtime model | Typical outputs | Suggested period |
| --- | --- | --- | --- | --- |
| `drought` | Composite Drought Index monitoring and forecasting | LSTM | CDI severity map, sub-index time series, six-month forecast | At least 5 years; 10–30 years is preferable |
| `flood` | Flood susceptibility and event-risk mapping | RF + XGBoost ensemble | Four-class risk map, COG, risk distribution, SHAP drivers | A rainy season or known flood event |
| `food_security` | Vegetation and climate stress assessment | Random Forest | Food-insecurity risk map, COG, vegetation/rainfall charts | At least one complete growing season |
| `disease` | Climate suitability for disease outbreaks | Gradient Boosting | Risk map, COG, hotspot and forecast charts | Recent 3–6 months; longer for trends |
| `land_degradation` | Vegetation decline and degradation detection | LightGBM | Degradation map, COG, NDVI trend and breakpoint statistics | At least 5 years; 10+ years is preferable |

Alternative models can be selected through `extra_params["model_type"]`:

| Module | Supported values |
| --- | --- |
| Drought | `lstm`, `drought_monitoring` |
| Flood | `rf`, `xgboost`, `ensemble` |
| Food security | `rf`, `xgboost`, `ensemble` |
| Disease | `gbm`, `xgboost`, `ensemble` |
| Land degradation | `rf`, `lgbm`, `ensemble` |

The registry exposes longer descriptions, input variables, date guidance, and
model notes:

```python
from climate_change import USE_CASE_REGISTRY

for module_id, info in USE_CASE_REGISTRY.items():
    print(module_id, info.name, info.default_model)
```

## Requirements

- Python 3.10–3.13
- A Google Cloud project with the Earth Engine API enabled
- Google Earth Engine credentials available on the machine
- Sufficient memory and disk space for geospatial and ML workloads

Some analyses download and process substantial satellite datasets. Runtime
depends on AOI size, date range, Earth Engine quotas, spatial resolution, and
the selected model.

## Installation

Install the published package from PyPI:

```bash
python -m pip install climate-change
```

With `uv`:

```bash
uv add climate-change
```

The distribution name uses a hyphen, while the Python import uses an
underscore:

```python
from climate_change import run_analysis
```

### Install from source with `uv`

```bash
git clone https://github.com/Odero54/climate-change.git
cd climate-change
uv sync --all-extras
```

Run Python inside the managed environment:

```bash
uv run python
```

### Install from source with `pip`

```bash
git clone https://github.com/Odero54/climate-change.git
cd climate-change
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development tools and all optional integrations:

```bash
python -m pip install -e ".[all,dev]"
```

The repository root is mapped to the import package `climate_change` by
Setuptools. An editable installation is therefore recommended when working
from a clone.

## Google Earth Engine setup

Authenticate once on the machine that will run the analyses:

```bash
earthengine authenticate
```

Set the Google Cloud project ID in your shell:

```bash
export GEE_PROJECT="your-google-cloud-project-id"
```

Alternatively, add it to a local `.env` file:

```dotenv
GEE_PROJECT=your-google-cloud-project-id
```

The project must exist, have the Earth Engine API enabled, and be accessible
to the authenticated Google account or service account.

You may validate a project before accepting analysis requests:

```python
from climate_change import validate_gee_project

validate_gee_project("your-google-cloud-project-id")
```

For backend deployments, authenticate non-interactively before starting the
application. Pass each user's validated project explicitly as `gee_project`;
do not rely on an interactive prompt in workers or API processes.

## Quick start

`run_analysis()` is asynchronous:

```python
import asyncio
from dataclasses import asdict

from climate_change import run_analysis

NAIROBI_AOI = {
    "type": "Polygon",
    "coordinates": [
        [
            [36.65, -1.45],
            [37.10, -1.45],
            [37.10, -1.10],
            [36.65, -1.10],
            [36.65, -1.45],
        ]
    ],
}


async def main() -> None:
    output = await run_analysis(
        module="flood",
        aoi_geojson=NAIROBI_AOI,
        start_date="2024-03-01",
        end_date="2024-05-31",
        country="Kenya",
        gee_project="your-google-cloud-project-id",
        extra_params={
            "model_type": "rf",
            "scale": 90,
            "n_pixels": 5000,
            "output_dir": "outputs",
        },
    )

    print(output.stats)
    print(output.raster_path)
    print(asdict(output))


asyncio.run(main())
```

Inside Jupyter, FastAPI, or another running event loop, call it directly:

```python
output = await run_analysis(...)
```

## Example notebooks

The
[`example-usage`](https://github.com/Odero54/climate-change/tree/main/example-usage)
directory contains a runnable notebook
for each analysis module:

- [Drought monitoring and early warning](https://github.com/Odero54/climate-change/blob/main/example-usage/01_drought_analysis.ipynb)
- [Flood-risk monitoring and mitigation](https://github.com/Odero54/climate-change/blob/main/example-usage/02_flood_risk_analysis.ipynb)
- [Food-security assessment](https://github.com/Odero54/climate-change/blob/main/example-usage/03_food_security_analysis.ipynb)
- [Climate-driven disease surveillance](https://github.com/Odero54/climate-change/blob/main/example-usage/04_disease_risk_analysis.ipynb)
- [Land-degradation and rangeland dynamics](https://github.com/Odero54/climate-change/blob/main/example-usage/05_land_degradation_analysis.ipynb)

Launch Jupyter from the repository:

```bash
uv run jupyter lab
```

Each notebook reads `GEE_PROJECT` from the environment, uses a representative
African AOI, leaves cells unexecuted, and includes optional chart and PDF
report examples.

## Public API

```python
async def run_analysis(
    module: str,
    aoi_geojson: dict,
    start_date: str,
    end_date: str,
    country: str,
    gee_project: str = "",
    extra_params: dict | None = None,
    openai_api_key: str | None = None,
    report_output_dir: str | None = None,
    map_png_bytes: bytes | None = None,
) -> AnalysisOutput:
    ...
```

### Required arguments

- `module`: one of `drought`, `flood`, `food_security`, `disease`, or
  `land_degradation`;
- `aoi_geojson`: a GeoJSON `Polygon`, `MultiPolygon`, `Feature`, or
  `FeatureCollection`;
- `start_date` and `end_date`: ISO dates in `YYYY-MM-DD` format;
- `country`: display label included in statistics, metadata, and reports.

### Optional arguments

- `gee_project`: Google Cloud project used for Earth Engine. If omitted,
  `GEE_PROJECT` is used.
- `extra_params`: model and module-specific overrides.
- `openai_api_key`: generates an optional plain-language interpretation.
- `report_output_dir`: writes a timestamped PDF report to this directory.
- `map_png_bytes`: embeds a supplied PNG map image in the PDF report.

Common `extra_params` include:

| Parameter | Meaning |
| --- | --- |
| `model_type` | Selects one of the module's supported models |
| `scale` | Earth Engine sampling/export resolution in metres |
| `n_pixels` | Approximate number of training pixels to sample |
| `output_dir` | Directory for generated raster artifacts |
| `prefix` | Prefix used for output filenames |

Flood analyses also accept event-specific pre-flood, post-flood, rainfall, and
surface-water date windows. See
[`FloodRiskUseCase`](https://github.com/Odero54/climate-change/blob/main/flood/use_case.py)
for the complete configuration schema.

## Result structure

Every module returns `AnalysisOutput`:

```python
@dataclass
class AnalysisOutput:
    module: str
    geojson: dict
    raster_path: str | dict[str, str] | None
    stats: dict
    shap: dict | None
    charts: dict
    metadata: dict
```

- `geojson` contains spatial risk or severity features.
- `raster_path` points to one or more generated COG files when export
  succeeds.
- `stats` contains module-specific summary metrics.
- `shap` contains feature importance for supported ML modules.
- `charts` contains frontend-ready chart payloads.
- `metadata` records the model, features, country, date range, and optional
  report or AI interpretation information.

Because fields differ by module, consumers should treat `stats` and `charts`
as module-specific payloads.

## AI interpretation and PDF reports

Pass an OpenAI API key to request a concise climate-risk interpretation:

```python
output = await run_analysis(
    module="land_degradation",
    aoi_geojson=aoi,
    start_date="2015-01-01",
    end_date="2024-12-31",
    country="Kenya",
    gee_project="your-google-cloud-project-id",
    openai_api_key="your-openai-api-key",
)

print(output.metadata.get("ai_interpretation"))
```

The key is supplied to the OpenAI client for that analysis and is not managed
or persisted by this package.

To generate a PDF:

```python
from pathlib import Path

map_bytes = Path("map.png").read_bytes()

output = await run_analysis(
    module="drought",
    aoi_geojson=aoi,
    start_date="2010-01-01",
    end_date="2024-12-31",
    country="Kenya",
    gee_project="your-google-cloud-project-id",
    report_output_dir="reports",
    map_png_bytes=map_bytes,
)

print(output.metadata["report_path"])
```

`map_png_bytes` is optional. The report can still include generated charts
when no map screenshot is supplied.

## Backend integration

The package can be called from a FastAPI route or background task:

```python
from fastapi import APIRouter

from climate_change import run_analysis

router = APIRouter()


@router.post("/analyses")
async def create_analysis(body: dict):
    output = await run_analysis(
        module=body["module"],
        aoi_geojson=body["aoi_geojson"],
        start_date=body["start_date"],
        end_date=body["end_date"],
        country=body["country"],
        gee_project=body["gee_project"],
        extra_params=body.get("extra_params"),
    )
    return output
```

In a production service:

- validate the GEE project during user registration;
- keep OAuth or service-account credentials on the server, not in the user
  database;
- scope project IDs to their owners;
- run long analyses in a task queue;
- configure Dask workers before accepting jobs; and
- persist generated artifacts outside ephemeral worker storage.

## Development

Install all development dependencies:

```bash
uv sync --all-extras
```

Run the test suite:

```bash
uv run pytest
```

Run linting and type checks:

```bash
uv run ruff check .
uv run mypy .
```

Format Python files:

```bash
uv run ruff format .
```

## Package layout

```text
.
├── ai_interpreter/       # OpenAI result interpretation
├── core/                 # orchestration, caching, GEE auth, and Dask
├── disease/              # disease-risk features, models, and exports
├── drought/              # CDI processing and drought forecasting
├── flood/                # flood features, models, and exports
├── food_security/        # food-security features, models, and exports
├── land_degradation/     # degradation features, models, and exports
├── reporting/            # PDF report generation
├── tests/                # unit tests
├── registry.py           # use-case metadata and model options
└── pyproject.toml        # package and tool configuration
```

## Limitations

- Outputs are decision-support indicators, not operational warnings or
  substitutes for field observations.
- Model performance depends on AOI, period, source-data availability, labels,
  and Earth Engine processing limits.
- Synthetic or proxy labels used by individual workflows may not represent
  locally observed outcomes.
- Large AOIs, long periods, or fine spatial resolutions may exceed Earth
  Engine memory or quota limits.
- AI-generated interpretations should be reviewed by a domain expert before
  informing policy or emergency action.

## License

The package metadata declares the project under the MIT License.

## Links

- Repository: <https://github.com/Odero54/climate-change>
- Issues: <https://github.com/Odero54/climate-change/issues>
