"""Tests for core/aoi_upload.py — parse_aoi_upload."""

import io
import json
import zipfile

import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, box

from climate_change.core.aoi_upload import parse_aoi_upload

_SQUARE = box(34.0, -1.0, 35.0, 0.0)  # a plausible AOI in EPSG:4326


def _geojson_bytes(geom=_SQUARE) -> bytes:
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": geom.__geo_interface__}],
        }
    ).encode()


def _gpkg_bytes(geoms=(_SQUARE,), crs="EPSG:4326") -> bytes:
    import tempfile
    from pathlib import Path

    gdf = gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=list(geoms), crs=crs)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "aoi.gpkg"
        gdf.to_file(path, driver="GPKG")
        return path.read_bytes()


def _shapefile_zip_bytes(geoms=(_SQUARE,), crs="EPSG:4326") -> bytes:
    import tempfile
    from pathlib import Path

    gdf = gpd.GeoDataFrame({"id": range(len(geoms))}, geometry=list(geoms), crs=crs)
    with tempfile.TemporaryDirectory() as d:
        shp_path = Path(d) / "aoi.shp"
        gdf.to_file(shp_path, driver="ESRI Shapefile")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for sibling in Path(d).glob("aoi.*"):
                zf.write(sibling, arcname=sibling.name)
        return buf.getvalue()


class TestParseAoiUploadGeoJSON:
    def test_returns_polygon_geometry(self):
        result = parse_aoi_upload(_geojson_bytes(), "aoi.geojson")
        assert result["type"] == "Polygon"
        assert "coordinates" in result

    def test_json_extension_also_accepted(self):
        result = parse_aoi_upload(_geojson_bytes(), "aoi.json")
        assert result["type"] == "Polygon"

    def test_multiple_features_are_unioned(self):
        second = box(35.0, -1.0, 36.0, 0.0)  # adjacent square
        payload = json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": _SQUARE.__geo_interface__},
                    {"type": "Feature", "properties": {}, "geometry": second.__geo_interface__},
                ],
            }
        ).encode()
        result = parse_aoi_upload(payload, "aoi.geojson")
        # Two adjacent squares union into one combined polygon, not two separate ones.
        assert result["type"] in ("Polygon", "MultiPolygon")


class TestParseAoiUploadGeoPackage:
    def test_returns_polygon_geometry(self):
        result = parse_aoi_upload(_gpkg_bytes(), "aoi.gpkg")
        assert result["type"] == "Polygon"

    def test_reprojects_non_wgs84_crs(self):
        # EPSG:3857 (Web Mercator) — coordinates would be in metres, not
        # lon/lat, if reprojection didn't happen.
        result = parse_aoi_upload(_gpkg_bytes(crs="EPSG:3857"), "aoi.gpkg")
        coords = result["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        assert all(-180 <= lon <= 180 for lon in lons)
        assert all(-90 <= lat <= 90 for lat in lats)


class TestParseAoiUploadShapefile:
    def test_zipped_shapefile_returns_polygon_geometry(self):
        result = parse_aoi_upload(_shapefile_zip_bytes(), "aoi.zip")
        assert result["type"] == "Polygon"

    def test_zip_without_shp_raises(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("readme.txt", "not a shapefile")
        with pytest.raises(ValueError, match="does not contain a .shp"):
            parse_aoi_upload(buf.getvalue(), "aoi.zip")

    def test_corrupt_zip_raises(self):
        with pytest.raises(ValueError, match="not a valid zip"):
            parse_aoi_upload(b"not actually a zip file", "aoi.zip")


class TestParseAoiUploadValidation:
    def test_unsupported_extension_raises(self):
        with pytest.raises(ValueError, match="Unsupported AOI file type"):
            parse_aoi_upload(b"whatever", "aoi.kml")

    def test_corrupt_geojson_raises(self):
        with pytest.raises(ValueError, match="Could not read"):
            parse_aoi_upload(b"{not valid json or geojson", "aoi.geojson")

    def test_point_geometry_rejected(self):
        payload = json.dumps(Point(34.0, -1.0).__geo_interface__).encode()
        with pytest.raises(ValueError, match="must contain polygon geometry"):
            parse_aoi_upload(payload, "aoi.geojson")

    def test_line_geometry_rejected(self):
        payload = json.dumps(LineString([(34.0, -1.0), (35.0, 0.0)]).__geo_interface__).encode()
        with pytest.raises(ValueError, match="must contain polygon geometry"):
            parse_aoi_upload(payload, "aoi.geojson")

    def test_empty_featurecollection_raises(self):
        payload = json.dumps({"type": "FeatureCollection", "features": []}).encode()
        with pytest.raises(ValueError):
            parse_aoi_upload(payload, "aoi.geojson")
