"""Tests for drought/use_case.py — DroughtUseCase.run_date_ranges/run_multi_regions."""

from unittest.mock import MagicMock

import pytest

from climate_change.drought.use_case import DroughtUseCase


@pytest.fixture()
def use_case():
    return DroughtUseCase(MagicMock())


class TestSingleAreaOnly:
    def test_run_date_ranges_raises(self, use_case, simple_polygon_geojson):
        config = {"aoi_geojson": simple_polygon_geojson, "start_date": "2010-01-01"}
        date_ranges = [{"end_date": "2015-01-01"}, {"end_date": "2020-01-01"}]
        with pytest.raises(ValueError, match="single AOI/date-range"):
            use_case.run_date_ranges(config, date_ranges)

    def test_run_multi_regions_raises(self, use_case, simple_polygon_geojson):
        configs = [
            {"aoi_geojson": simple_polygon_geojson, "start_date": "2010-01-01"},
            {"aoi_geojson": simple_polygon_geojson, "start_date": "2010-01-01"},
        ]
        with pytest.raises(ValueError, match="single AOI/date-range"):
            use_case.run_multi_regions(configs)
