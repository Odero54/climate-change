# Top-level API

Everything most callers need is importable directly from `climate_change`.

```python
from climate_change import run_analysis, USE_CASE_REGISTRY, validate_gee_project
```

::: climate_change
    options:
      members:
        - run_analysis

## Domain use cases

Lazily importable from the top-level package (`climate_change.DroughtUseCase`,
etc.) so importing `climate_change` doesn't eagerly pull in every module's
heavy dependencies (torch, xgboost, lightgbm, …). Documented in full on their
own pages: [drought](drought.md), [flood](flood.md),
[food_security](food_security.md), [disease](disease.md),
[land_degradation](land_degradation.md).

## Other re-exports

`AnalysisConfig`, `AnalysisOutput`, `BaseUseCase`, `DaskEngine`,
`analysis_cache`, `feature_cache`, `ensure_gee`, `validate_gee_project`,
`register_module`, `MODULE_MAP` — see [core](core.md).

`USE_CASE_REGISTRY`, `UseCaseInfo`, `ModelOption`, `get_use_case_info` — see
[registry](registry.md).

`ReportBuilder` — see [reporting](reporting.md).

`AIInterpreter`, `build_interpretation_prompt` — see
[ai_interpreter](ai_interpreter.md).
