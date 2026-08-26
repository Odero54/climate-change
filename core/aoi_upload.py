from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
from shapely.geometry import mapping

# Formats a caller may upload as an alternative to drawing/typing an AOI
# (e.g. an admin boundary or field polygon exported from QGIS/ArcGIS).
# Shapefiles are inherently multi-file (.shp needs its .shx/.dbf sidecars,
# often also .prj) so they must arrive as a .zip containing the bundle —
# a bare .shp alone is not self-sufficient.
_GEOJSON_EXTENSIONS = {".geojson", ".json"}
_SINGLE_FILE_EXTENSIONS = {".gpkg", *_GEOJSON_EXTENSIONS}
_ZIP_EXTENSIONS = {".zip"}
SUPPORTED_EXTENSIONS = _SINGLE_FILE_EXTENSIONS | _ZIP_EXTENSIONS


def parse_aoi_upload(file_bytes: bytes, filename: str) -> dict:
    """
    Parse an uploaded vector file (GeoJSON, GeoPackage, or a zipped
    Shapefile) into a single AOI geometry, reprojected to EPSG:4326 —
    the same GeoJSON-geometry shape (Polygon/MultiPolygon) that
    AnalysisConfig.aoi_geojson expects everywhere else in this package.

    Multiple features in the upload (e.g. several parcels, or an admin
    layer with more than one boundary) are unioned into one combined
    geometry rather than kept as a FeatureCollection — the rest of the
    pipeline (core.base_use_case._lons_lats, _aoi_geometries, etc.) treats
    the AOI as a single area, not a set of disjoint candidate areas, so
    returning a FeatureCollection here would just push this same union
    decision downstream to code that isn't set up to make it.

    Raises ValueError for an unsupported extension, a corrupt/unreadable
    file, a zip with no .shp inside, or a file with no polygonal geometry.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported AOI file type '{suffix or filename}'. "
            f"Supported: .geojson, .json, .gpkg, or a .zip containing a Shapefile."
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        read_path = _write_upload(file_bytes, suffix, tmp_path)
        try:
            gdf = gpd.read_file(read_path)
        except Exception as exc:
            raise ValueError(f"Could not read '{filename}' as a vector file: {exc}") from exc

    return _to_aoi_geometry(gdf, filename)


def _write_upload(file_bytes: bytes, suffix: str, tmp_path: Path) -> Path:
    """Write the upload to disk and return the path gpd.read_file should open."""
    if suffix in _ZIP_EXTENSIONS:
        zip_path = tmp_path / "upload.zip"
        zip_path.write_bytes(file_bytes)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp_path)
                shp_names = [n for n in zf.namelist() if n.lower().endswith(".shp")]
        except zipfile.BadZipFile as exc:
            raise ValueError(f"'{zip_path.name}' is not a valid zip file: {exc}") from exc
        if not shp_names:
            raise ValueError("The uploaded zip does not contain a .shp Shapefile.")
        return tmp_path / shp_names[0]

    upload_path = tmp_path / f"upload{suffix}"
    upload_path.write_bytes(file_bytes)
    return upload_path


def _to_aoi_geometry(gdf: gpd.GeoDataFrame, filename: str) -> dict:
    if gdf.empty:
        raise ValueError(f"'{filename}' contains no features.")

    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")

    geoms = [g for g in gdf.geometry if g is not None and not g.is_empty]
    if not geoms:
        raise ValueError(f"'{filename}' contains no usable geometry.")

    non_polygonal = [g.geom_type for g in geoms if g.geom_type not in ("Polygon", "MultiPolygon")]
    if non_polygonal:
        raise ValueError(
            f"'{filename}' must contain polygon geometry for an area of interest — "
            f"found {sorted(set(non_polygonal))} instead."
        )

    union = gpd.GeoSeries(geoms, crs="EPSG:4326").union_all()
    return dict(mapping(union))
