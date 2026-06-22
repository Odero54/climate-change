"""Tests for core/base_use_case.py."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.base_use_case import (
    AnalysisConfig,
    AnalysisOutput,
    BaseUseCase,
    _aoi_geometries,
    _ee_geometry_from_geojson,
    _lons_lats,
    _round_floats,
)


# ── _round_floats ─────────────────────────────────────────────────────────────

class TestRoundFloats:
    def test_rounds_plain_float(self):
        assert _round_floats(1.123456789) == 1.123457

    def test_respects_decimals_param(self):
        assert _round_floats(1.123456789, decimals=2) == 1.12

    def test_passes_through_int(self):
        assert _round_floats(5) == 5

    def test_passes_through_string(self):
        assert _round_floats("hello") == "hello"

    def test_recurses_into_list(self):
        result = _round_floats([1.111111, 2.222222], decimals=3)
        assert result == [1.111, 2.222]

    def test_recurses_into_dict(self):
        result = _round_floats({"x": 1.111111}, decimals=3)
        assert result == {"x": 1.111}

    def test_nested_structure(self):
        data = {"coords": [1.123456, 2.654321]}
        result = _round_floats(data, decimals=2)
        assert result == {"coords": [1.12, 2.65]}


# ── _lons_lats ────────────────────────────────────────────────────────────────

class TestLonsLats:
    def test_polygon_geojson(self, simple_polygon_geojson):
        lons, lats = _lons_lats(simple_polygon_geojson)
        assert lons == [36.0, 37.0, 37.0, 36.0, 36.0]
        assert lats == [-1.0, -1.0, 0.0, 0.0, -1.0]

    def test_feature_geojson(self, feature_geojson):
        lons, lats = _lons_lats(feature_geojson)
        assert len(lons) == 5

    def test_feature_collection_geojson(self, feature_collection_geojson):
        lons, lats = _lons_lats(feature_collection_geojson)
        assert len(lons) == 5

    def test_multipolygon(self):
        mp = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[36.0, -1.0], [37.0, -1.0], [37.0, 0.0], [36.0, 0.0], [36.0, -1.0]]]
            ],
        }
        lons, lats = _lons_lats(mp)
        assert len(lons) == 5

    def test_empty_coordinates_raises(self):
        with pytest.raises(ValueError):
            _lons_lats({"type": "Polygon", "coordinates": []})

    def test_empty_feature_collection_raises(self):
        with pytest.raises(ValueError):
            _lons_lats({"type": "FeatureCollection", "features": []})


# ── _aoi_geometries ───────────────────────────────────────────────────────────

class TestAoiGeometries:
    def test_polygon_returns_list_of_one(self, simple_polygon_geojson):
        result = _aoi_geometries(simple_polygon_geojson)
        assert len(result) == 1
        assert result[0]["type"] == "Polygon"

    def test_feature_unwraps_geometry(self, feature_geojson):
        result = _aoi_geometries(feature_geojson)
        assert len(result) == 1

    def test_feature_collection_returns_all_geometries(self, feature_collection_geojson):
        result = _aoi_geometries(feature_collection_geojson)
        assert len(result) == 1

    def test_none_returns_empty_list(self):
        assert _aoi_geometries(None) == []

    def test_unknown_type_returns_empty_list(self):
        assert _aoi_geometries({"type": "Point", "coordinates": [0, 0]}) == []

    def test_multipolygon_returned(self):
        mp = {
            "type": "MultiPolygon",
            "coordinates": [
                [[[36.0, -1.0], [37.0, -1.0], [37.0, 0.0], [36.0, -1.0]]]
            ],
        }
        result = _aoi_geometries(mp)
        assert len(result) == 1
        assert result[0]["type"] == "MultiPolygon"


# ── _ee_geometry_from_geojson ─────────────────────────────────────────────────

class TestEeGeometryFromGeojson:
    def test_polygon_calls_ee_geometry(self, simple_polygon_geojson):
        mock_ee = MagicMock()
        with patch.dict("sys.modules", {"ee": mock_ee}):
            _ee_geometry_from_geojson(simple_polygon_geojson)
        mock_ee.Geometry.assert_called_once_with(simple_polygon_geojson)

    def test_feature_unwraps_then_calls_ee(self, feature_geojson, simple_polygon_geojson):
        mock_ee = MagicMock()
        with patch.dict("sys.modules", {"ee": mock_ee}):
            _ee_geometry_from_geojson(feature_geojson)
        mock_ee.Geometry.assert_called_once_with(simple_polygon_geojson)


# ── AnalysisConfig ─────────────────────────────────────────────────────────────

class TestAnalysisConfig:
    def test_default_extra_params_empty(self, simple_polygon_geojson):
        cfg = AnalysisConfig(
            module="drought",
            aoi_geojson=simple_polygon_geojson,
            start_date="2010-01-01",
            end_date="2023-12-31",
            country="Kenya",
        )
        assert cfg.extra_params == {}

    def test_fields_stored_correctly(self, simple_polygon_geojson):
        cfg = AnalysisConfig(
            module="flood",
            aoi_geojson=simple_polygon_geojson,
            start_date="2020-01-01",
            end_date="2021-01-01",
            country="Uganda",
            extra_params={"model_type": "rf"},
        )
        assert cfg.module == "flood"
        assert cfg.country == "Uganda"
        assert cfg.extra_params["model_type"] == "rf"


# ── AnalysisOutput ────────────────────────────────────────────────────────────

class TestAnalysisOutput:
    def test_stores_all_fields(self):
        out = AnalysisOutput(
            module="drought",
            geojson={"type": "FeatureCollection", "features": []},
            raster_path=None,
            stats={"mean_cdi": 0.9},
            shap=None,
            charts={},
            metadata={"country": "Kenya"},
        )
        assert out.module == "drought"
        assert out.stats["mean_cdi"] == 0.9
        assert out.shap is None


# ── BaseUseCase._cache_key ────────────────────────────────────────────────────

class TestCacheKey:
    def _make_config(self, module="drought", start="2010-01-01", end="2023-12-31"):
        return AnalysisConfig(
            module=module,
            aoi_geojson={
                "type": "Polygon",
                "coordinates": [[[36.0, -1.0], [37.0, 0.0], [36.0, -1.0]]],
            },
            start_date=start,
            end_date=end,
            country="Kenya",
        )

    def test_key_is_16_chars(self):
        cfg = self._make_config()

        class _Impl(BaseUseCase):
            def fetch_data(self, c): ...
            def preprocess(self, r, c): ...
            def run_model(self, f, c): ...

        key = _Impl._cache_key(cfg)
        assert len(key) == 16

    def test_same_config_same_key(self):
        cfg1 = self._make_config()
        cfg2 = self._make_config()
        from core.base_use_case import BaseUseCase as BUC
        assert BUC._cache_key(cfg1) == BUC._cache_key(cfg2)

    def test_different_module_different_key(self):
        cfg1 = self._make_config(module="drought")
        cfg2 = self._make_config(module="flood")
        from core.base_use_case import BaseUseCase as BUC
        assert BUC._cache_key(cfg1) != BUC._cache_key(cfg2)

    def test_different_dates_different_key(self):
        cfg1 = self._make_config(start="2010-01-01")
        cfg2 = self._make_config(start="2015-01-01")
        from core.base_use_case import BaseUseCase as BUC
        assert BUC._cache_key(cfg1) != BUC._cache_key(cfg2)


# ── BaseUseCase.execute ───────────────────────────────────────────────────────

class TestBaseUseCaseExecute:
    def _make_use_case(self, output):
        class _Impl(BaseUseCase):
            def fetch_data(self, config):
                return {"raw": True}

            def preprocess(self, raw, config):
                return {"features": True}

            def run_model(self, features, config):
                return output

        mock_dask = MagicMock()
        return _Impl(mock_dask)

    def test_execute_returns_output(self, simple_polygon_geojson):
        expected = AnalysisOutput(
            module="drought", geojson={}, raster_path=None,
            stats={}, shap=None, charts={}, metadata={},
        )
        uc = self._make_use_case(expected)
        cfg = AnalysisConfig(
            module="drought", aoi_geojson=simple_polygon_geojson,
            start_date="2010-01-01", end_date="2020-01-01", country="Kenya",
        )
        result = asyncio.run(uc.execute(cfg))
        assert result is expected

    def test_execute_uses_cache_on_second_call(self, simple_polygon_geojson):
        call_count = 0

        class _CountingImpl(BaseUseCase):
            def fetch_data(self, config):
                nonlocal call_count
                call_count += 1
                return {}

            def preprocess(self, raw, config):
                return {}

            def run_model(self, features, config):
                return AnalysisOutput(
                    module="drought", geojson={}, raster_path=None,
                    stats={}, shap=None, charts={}, metadata={},
                )

        uc = _CountingImpl(MagicMock())
        cfg = AnalysisConfig(
            module="drought", aoi_geojson=simple_polygon_geojson,
            start_date="2010-01-01", end_date="2020-01-01", country="Kenya",
        )
        asyncio.run(uc.execute(cfg))
        asyncio.run(uc.execute(cfg))
        assert call_count == 1
