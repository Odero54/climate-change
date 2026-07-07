# core

Orchestration, GEE authentication, caching, and distributed execution shared
by every domain module. See [Concepts](../concepts.md) for how these fit
together.

## Analysis contract & base use case

::: climate_change.core.base_use_case
    options:
      members:
        - AnalysisConfig
        - AnalysisOutput
        - BaseUseCase

## Orchestrator

::: climate_change.core.runner

## Google Earth Engine authentication

::: climate_change.core.gee_auth
    options:
      members:
        - ensure_gee
        - validate_gee_project
        - startup_init_gee

## Caching

::: climate_change.core.cache

## Distributed execution

::: climate_change.core.dask_engine

## Land-cover masking

::: climate_change.core.landcover_mask
