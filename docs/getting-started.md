# Getting started

## 1. Install the package

From PyPI:

```bash
python -m pip install climate-change
```

Or with `uv`:

```bash
uv add climate-change
```

!!! note "Distribution name vs. import name"
    The PyPI distribution is `climate-change` (hyphen); the Python import is
    `climate_change` (underscore):

    ```python
    from climate_change import run_analysis
    ```

### Installing from source

Useful if you want to run the example notebooks, contribute, or build these
docs locally.

```bash
git clone https://github.com/Odero54/climate-change.git
cd climate-change
uv sync --all-extras   # or: python -m pip install -e ".[all,dev]"
```

The repository root **is** the `climate_change` package — there's no nested
`climate_change/` source directory. Setuptools maps the root to that import
name, so an editable install (`-e .` / `uv sync`) is required when working
from a clone.

### Optional dependency groups

Install only what you need instead of everything:

| Extra | Adds | Needed for |
| --- | --- | --- |
| `gee` | `earthengine-api`, `xee` | Fetching any satellite data (all modules) |
| `cog` | `rasterio`, `rioxarray` | Reading/writing Cloud-Optimized GeoTIFFs |
| `ml` | `scikit-learn`, `xgboost`, `lightgbm`, `shap`, `torch` | Training/running the risk models |
| `ai` | `openai` | AI-generated plain-language interpretations |
| `report` | `reportlab`, `Pillow` | PDF report generation |
| `dev` | `pytest`, `ruff`, `mypy`, `jupyterlab`, … | Running tests/lint/notebooks |
| `docs` | `mkdocs`, `mkdocs-material`, `mkdocstrings` | Building this documentation site |
| `all` | everything above except `dev`/`docs` | A full runtime install |

```bash
python -m pip install "climate-change[gee,ml,cog]"
```

## 2. Authenticate Google Earth Engine

Every module fetches at least some data from Google Earth Engine, so this
step is required no matter which module you use.

**One-time, on the machine that will run analyses:**

```bash
earthengine authenticate
```

This opens a browser flow (or prints a URL for headless machines) and stores
credentials locally — the package never manages or stores these itself.

**Tell the package which Google Cloud project to bill/run against**, either
as an environment variable:

```bash
export GEE_PROJECT="your-google-cloud-project-id"
```

or in a `.env` file in your working directory:

```dotenv
GEE_PROJECT=your-google-cloud-project-id
```

The project must exist, have the Earth Engine API enabled, and be accessible
to whichever account ran `earthengine authenticate`.

**Validate a project id before trusting it** (useful when accepting a
project id from a user, e.g. at account registration in a backend service):

```python
from climate_change import validate_gee_project

validate_gee_project("your-google-cloud-project-id")  # raises on failure
```

!!! warning "Backend / worker deployments"
    Don't rely on the interactive `earthengine authenticate` prompt inside a
    server process or task worker — authenticate the machine ahead of time,
    and pass each user's already-validated project explicitly as
    `gee_project=...` to `run_analysis`. See
    [`ensure_gee`](api/core.md) for the exact resolution order (explicit
    argument → `GEE_PROJECT` env → interactive prompt) and how it behaves
    inside Dask worker processes.

## 3. Verify the setup

```python
import asyncio
from climate_change import run_analysis

NAIROBI_AOI = {
    "type": "Polygon",
    "coordinates": [[
        [36.65, -1.45], [37.10, -1.45],
        [37.10, -1.10], [36.65, -1.10],
        [36.65, -1.45],
    ]],
}

async def main():
    output = await run_analysis(
        module="flood",
        aoi_geojson=NAIROBI_AOI,
        start_date="2024-03-01",
        end_date="2024-05-31",
        country="Kenya",
        gee_project="your-google-cloud-project-id",
    )
    print(output.stats)

asyncio.run(main())
```

If this prints a `stats` dict without raising, GEE auth and the package
install are both working. Continue to [Concepts](concepts.md) to understand
what just happened, or jump to the
[end-to-end example](end-to-end-example.md) for a fuller walkthrough.

## Running the example notebooks

The repository ships a runnable notebook per module under `example-usage/`:

```bash
git clone https://github.com/Odero54/climate-change.git
cd climate-change
uv sync --all-extras
uv run jupyter lab
```

Each notebook reads `GEE_PROJECT` from the environment and uses a
representative African AOI. Cells are left unexecuted so you run them
against your own GEE project.
