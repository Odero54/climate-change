"""Shared fixtures for climate_change test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def simple_polygon_geojson():
    return {
        "type": "Polygon",
        "coordinates": [
            [[36.0, -1.0], [37.0, -1.0], [37.0, 0.0], [36.0, 0.0], [36.0, -1.0]]
        ],
    }


@pytest.fixture()
def feature_geojson(simple_polygon_geojson):
    return {"type": "Feature", "geometry": simple_polygon_geojson, "properties": {}}


@pytest.fixture()
def feature_collection_geojson(simple_polygon_geojson):
    return {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": simple_polygon_geojson, "properties": {}}
        ],
    }


@pytest.fixture()
def cdi_dataframe():
    """Monthly CDI time series spanning 10 years."""
    idx = pd.date_range("2010-01-01", periods=120, freq="MS")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "PDI": rng.uniform(0.6, 1.2, 120),
            "TDI": rng.uniform(0.6, 1.2, 120),
            "VDI": rng.uniform(0.6, 1.2, 120),
            "CDI": rng.uniform(0.5, 1.3, 120),
        },
        index=idx,
    )


@pytest.fixture()
def tiny_binary_xy():
    """Tiny balanced binary classification dataset (60 samples, 10 features)."""
    rng = np.random.default_rng(1)
    X = rng.standard_normal((60, 10))
    y = (rng.random(60) > 0.5).astype(int)
    # Guarantee at least one of each class
    y[:30] = 0
    y[30:] = 1
    return X, y


@pytest.fixture()
def tiny_multiclass_xy():
    """Tiny balanced 3-class dataset (90 samples, 7 features)."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((90, 7))
    y = np.repeat([0, 1, 2], 30)
    return X, y
