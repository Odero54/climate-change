"""
PDF report builder using reportlab.
Assembles analysis metadata, AI interpretation, embedded map PNG,
charts, and recommendations into a structured PDF.
"""

from __future__ import annotations

import io
import logging
import math
import re
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

import matplotlib
import numpy as np
import requests
from matplotlib.colors import ListedColormap
from PIL import Image as PILImage

# PDF report generation runs inside API worker threads, so force a
# non-interactive renderer before pyplot can initialize a GUI backend.
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # isort: skip

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from climate_change.core.base_use_case import AnalysisOutput
from climate_change.registry import USE_CASE_REGISTRY

# Brand colours
NAVY = colors.HexColor("#1B2A4A")
TEAL = colors.HexColor("#00897B")
GOLD = colors.HexColor("#F9A825")
LIGHT_GREY = colors.HexColor("#F5F5F5")
MID_GREY = colors.HexColor("#9E9E9E")

# CDI severity colours (matches frontend)
CDI_COLORS = {
    "Extreme drought": "#990000",
    "Severe drought": "#E65100",
    "Mild drought": "#F57C00",
    "Moderate drought": "#FFB74D",
    "Near normal": "#E0E0E0",
    "Mild wet": "#80CBC4",
    "Very wet": "#4DB6AC",
    "Moderately wet": "#00897B",
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "cover_title",
            parent=base["Title"],
            fontSize=26,
            textColor=NAVY,
            leading=32,
            alignment=TA_CENTER,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            parent=base["Normal"],
            fontSize=13,
            textColor=TEAL,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "section_heading": ParagraphStyle(
            "section_heading",
            parent=base["Heading1"],
            fontSize=14,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=6,
            borderPad=4,
        ),
        "body": ParagraphStyle(
            "body",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=base["Normal"],
            fontSize=10,
            leading=14,
            leftIndent=16,
            bulletIndent=6,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "caption",
            parent=base["Normal"],
            fontSize=8,
            textColor=MID_GREY,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "interpretation": ParagraphStyle(
            "interpretation",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
            spaceBefore=4,
            leftIndent=10,
            textColor=colors.HexColor("#2C3E50"),
            fontName="Helvetica-Oblique",
        ),
        "meta_key": ParagraphStyle(
            "meta_key",
            parent=base["Normal"],
            fontSize=9,
            textColor=MID_GREY,
        ),
        "meta_val": ParagraphStyle(
            "meta_val",
            parent=base["Normal"],
            fontSize=9,
            textColor=NAVY,
        ),
    }


class ReportBuilder:
    """
    Builds a PDF report for any completed AnalysisOutput.
    Drought module gets additional CDI sub-index and forecast sections.
    """

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._styles = _styles()

    def build(
        self,
        output: AnalysisOutput,
        ai_interpretation: str | None = None,
        map_png_bytes: bytes | None = None,
    ) -> Path:
        """
        Assemble and write the PDF.

        Args:
            output: completed AnalysisOutput from any module.
            ai_interpretation: human-centered AI text (optional; section omitted if None).
            map_png_bytes: PNG screenshot of the Leaflet result map (optional).

        Returns:
            Path to the written PDF file.
        """
        doc = SimpleDocTemplate(
            str(self.output_path),
            pagesize=A4,
            leftMargin=2.5 * cm,
            rightMargin=2.5 * cm,
            topMargin=2.5 * cm,
            bottomMargin=2.5 * cm,
        )
        story: list[Any] = []

        self._add_cover(story, output)
        story.append(PageBreak())
        self._add_metadata_table(story, output)
        self._add_rule(story)

        map_bytes = (
            self._render_raster_bytes(output)
            or self._render_choropleth_bytes(output)
            or map_png_bytes
        )
        if map_bytes:
            self._add_map_section(story, output, map_bytes)

        self._add_statistics_section(story, output)

        if output.module == "drought":
            self._add_drought_sections(story, output)
        elif output.module == "disease":
            self._add_disease_sections(story, output)
        elif output.module == "food_security":
            self._add_food_security_sections(story, output)
        elif output.module == "flood":
            self._add_flood_sections(story, output)
        elif output.module == "land_degradation":
            self._add_land_degradation_sections(story, output)

        if ai_interpretation:
            self._add_ai_section(story, ai_interpretation)

        self._add_glossary(story, output)
        self._add_appendix(story, output)

        doc.build(story)
        return self.output_path

    # Section builders

    @staticmethod
    def _location_label(output: AnalysisOutput) -> str:
        metadata = output.metadata or {}
        aoi_name = metadata.get("admin_name") or metadata.get("aoi_name")
        country = metadata.get("country")
        if aoi_name and country and str(aoi_name).strip() != str(country).strip():
            return f"{aoi_name}, {country}"
        return str(aoi_name or country or "Selected AOI")

    def _add_cover(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        info = USE_CASE_REGISTRY.get(output.module)
        module_name = info.name if info else output.module.replace("_", " ").title()

        story.append(Spacer(1, 3 * cm))
        story.append(Paragraph("ARIN Climate Resilience DSS", S["cover_sub"]))
        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph(module_name, S["cover_title"]))
        story.append(Spacer(1, 0.6 * cm))
        story.append(Paragraph("Analysis Report", S["cover_sub"]))
        story.append(Spacer(1, 1.5 * cm))

        country = self._location_label(output)
        start = output.metadata.get("start_date", "—")
        end = output.metadata.get("end_date", "—")
        generated = datetime.utcnow().strftime("%d %B %Y, %H:%M UTC")

        cover_data = [
            ["AOI / Region", country],
            ["Analysis Period", f"{start}  →  {end}"],
            ["Model", output.metadata.get("model", "—")],
            ["Generated", generated],
        ]
        tbl = Table(cover_data, colWidths=[5 * cm, 10 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
                    ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
                    ("FONTSIZE", (0, 0), (-1, -1), 10),
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
                    ("GRID", (0, 0), (-1, -1), 0.5, MID_GREY),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 6),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(tbl)

    def _add_metadata_table(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        story.append(Paragraph("1. Analysis Metadata", S["section_heading"]))
        info = USE_CASE_REGISTRY.get(output.module)
        if info:
            story.append(Paragraph(info.description, S["body"]))
            story.append(Spacer(1, 0.3 * cm))
            story.append(
                Paragraph(
                    f"<b>Dependent variable:</b> {info.dependent_variable}", S["body"]
                )
            )
            story.append(
                Paragraph(
                    f"<b>Best model:</b> {info.best_model}  |  "
                    f"<b>Accuracy:</b> {info.model_accuracy}",
                    S["body"],
                )
            )

    def _add_map_section(
        self, story: list, output: AnalysisOutput, map_png_bytes: bytes
    ) -> None:
        S = self._styles
        self._add_rule(story)
        story.append(Paragraph("2. Result Map", S["section_heading"]))
        img_buf = io.BytesIO(map_png_bytes)
        img = Image(img_buf, width=15 * cm, height=10 * cm)
        story.append(img)
        info = USE_CASE_REGISTRY.get(output.module)
        raster_source = self._raster_source(output)
        if raster_source:
            map_kind = "CDI map" if output.module == "drought" else "risk map"
            map_caption = (
                f"Figure 1. {info.name if info else output.module.replace('_', ' ').title()} "
                f"{map_kind} rendered from the exported Cloud-Optimised GeoTIFF "
                "over an OpenStreetMap basemap where tile access is available."
            )
        else:
            map_caption = (
                f"Figure 1. {info.name} risk choropleth — sampled pixels coloured by predicted risk class "
                f"with the AOI boundary outline."
                if info
                else "Figure 1. Analysis result map."
            )
        story.append(Paragraph(map_caption, S["caption"]))
        self._interpret(story, self._map_interpretation(output))
        story.append(Spacer(1, 0.4 * cm))

    def _map_interpretation(self, output: AnalysisOutput) -> str:
        """Generate a module-aware interpretation paragraph for the result map."""
        features = (output.geojson or {}).get("features", [])
        location = self._location_label(output)
        point_feats = [
            f for f in features if (f.get("properties") or {}).get("type") != "boundary"
        ]
        n_points = len(point_feats)
        stats = output.stats or {}

        # Count risk-class distribution from GeoJSON points
        class_counts: dict[str, int] = {}
        for f in point_feats:
            rc = (f.get("properties") or {}).get("risk_class", "Unknown")
            class_counts[rc] = class_counts.get(rc, 0) + 1

        def _pct(label: str) -> float:
            return round(class_counts.get(label, 0) / max(n_points, 1) * 100, 1)

        src = "sampled pixels" if n_points > 0 else "pixels"
        n_desc = f"{n_points:,} {src}" if n_points > 0 else "sampled pixels"

        if output.module == "flood":
            very_high = stats.get("very_high_risk_pct", _pct("Very High"))
            high = stats.get("high_risk_pct", _pct("High"))
            top_driver = stats.get("top_flood_driver", "elevation and terrain wetness")
            severity = "significant" if very_high + high > 30 else "moderate"
            if self._raster_source(output):
                mapped_pixels = stats.get("mapped_pixel_count")
                pixel_desc = (
                    f"{int(mapped_pixels):,} mapped raster pixels"
                    if isinstance(mapped_pixels, int | float)
                    else "the exported raster pixels"
                )
                return (
                    f"The map is rendered from the exported flood-risk COG, clipped to "
                    f"{location}. Across {pixel_desc}, Very High risk accounts for "
                    f"{very_high}% and High risk for {high}%, indicating {severity} "
                    f"flood exposure. The primary spatial driver is '{top_driver}', "
                    f"reflecting areas where topographic wetness, proximity to rivers, "
                    f"and SAR-detected backscatter change converge."
                )
            return (
                f"The choropleth shows {n_desc} across the AOI, each coloured by predicted "
                f"flood risk class (Low → Very High). Very High risk pixels account for "
                f"{very_high}% and High risk for {high}% of the sampled area, indicating "
                f"{severity} flood exposure. The primary spatial driver is '{top_driver}', "
                f"reflecting areas where topographic wetness, proximity to rivers, and "
                f"SAR-detected backscatter change converge. The navy outline delimits the "
                f"true AOI polygon — not a bounding box."
            )

        if output.module == "food_security":
            high_risk = stats.get("high_risk_pct", _pct("High Risk"))
            top_driver = stats.get("top_driver", "VCI and rainfall deficit")
            vhi = stats.get("vhi_mean")
            vhi_note = f"Area-mean VHI = {vhi:.1f}/100. " if vhi is not None else ""
            pressure = (
                "widespread food insecurity pressure"
                if high_risk > 33
                else "localised or moderate food insecurity pressure"
            )
            if self._raster_source(output):
                return (
                    f"The map is rendered from the exported food-security risk COG. "
                    f"For {location}, {vhi_note}{high_risk:.1f}% of mapped pixels are classified High "
                    f"Risk, indicating {pressure}. The dominant driver is "
                    f"'{top_driver}'. Red zones should be prioritised for food "
                    f"security programming."
                )
            return (
                f"The choropleth shows {n_desc} at 1 km resolution, coloured by composite "
                f"food stress risk class. {vhi_note}{high_risk:.1f}% of pixels are "
                f"classified High Risk, indicating {pressure}. The dominant driver is "
                f"'{top_driver}'. Red zones should be prioritised for food security "
                f"programming. The navy outline shows the exact AOI boundary."
            )

        if output.module == "disease":
            high_risk = stats.get("high_risk_pct", _pct("High Risk"))
            top_driver = stats.get(
                "top_driver", "rainfall and temperature co-occurrence"
            )
            n_clusters = stats.get("n_hotspot_clusters", 0)
            cluster_note = (
                f"DBSCAN identified {n_clusters} spatial hotspot cluster(s) within the "
                f"High Risk layer. "
                if n_clusters
                else ""
            )
            if self._raster_source(output):
                return (
                    f"The map is rendered from the exported disease-risk COG. "
                    f"For {location}, {high_risk:.1f}% of mapped pixels fall in the High Risk class. "
                    f"{cluster_note}The primary climate driver is '{top_driver}', "
                    f"reflecting conditions most conducive to disease vector activity."
                )
            return (
                f"The choropleth shows {n_desc}, coloured by climate-driven disease risk "
                f"class (Low / Medium / High). {high_risk:.1f}% of pixels fall in the High "
                f"Risk class. {cluster_note}The primary climate driver is '{top_driver}', "
                f"reflecting conditions most conducive to disease vector activity. "
                f"The navy boundary outline delimits the study AOI."
            )

        if output.module == "land_degradation":
            deg_pct = stats.get("degraded_label_pct", _pct("Degraded"))
            top_driver = stats.get("top_degradation_driver", "NDVI decline trend")
            mk_sig = stats.get("mk_significant", False)
            trend_note = (
                "Mann-Kendall testing confirms a statistically significant NDVI decline. "
                if mk_sig
                else ""
            )
            if self._raster_source(output):
                return (
                    f"The map is rendered from the exported land-degradation COG. "
                    f"For {location}, {deg_pct:.1f}% of mapped pixels are labelled Degraded based on "
                    f"the composite land degradation score. {trend_note}The leading "
                    f"driver is '{top_driver}'. Red zones indicate areas requiring "
                    f"restoration intervention."
                )
            return (
                f"The choropleth shows {n_desc} coloured as Not Degraded (green) or "
                f"Degraded (red). {deg_pct:.1f}% of pixels are labelled Degraded based on "
                f"the composite land degradation score. {trend_note}The leading driver is "
                f"'{top_driver}'. Red zones indicate areas requiring restoration "
                f"intervention. The navy outline marks the exact AOI boundary."
            )

        if output.module == "drought" and self._raster_source(output):
            latest_cdi = stats.get("latest_mean_cdi", stats.get("mean_cdi"))
            cdi_note = (
                f" The reported CDI summary value is {latest_cdi:.3f}."
                if isinstance(latest_cdi, int | float)
                else ""
            )
            return (
                "The map is rendered from the exported drought CDI COG, showing the "
                f"spatial drought index surface across {location}.{cdi_note} Refer to "
                "the CDI time-series and severity distribution sections for temporal "
                "drought characterisation."
            )

        return (
            "The map shows the spatial extent of the analysis area. "
            "Refer to the CDI time-series and severity distribution sections for "
            "quantitative drought characterisation across the study period."
        )

    # Module-specific risk-class color maps (matches frontend palette)
    _RISK_COLORS: dict[str, dict[str, str]] = {
        "flood": {
            "Low": "#2ECC71",
            "Medium": "#F1C40F",
            "High": "#E67E22",
            "Very High": "#E74C3C",
        },
        "food_security": {
            "Low Risk": "#184c09",
            "Medium Risk": "#ffcc36",
            "High Risk": "#f22d06",
        },
        "disease": {
            "Low Risk": "#2ECC71",
            "Medium Risk": "#F1C40F",
            "High Risk": "#E74C3C",
        },
        "land_degradation": {
            "Not Degraded": "#27AE60",
            "Degraded": "#E74C3C",
        },
        "drought": {
            "Extreme drought": "#990000",
            "Severe drought": "#E65100",
            "Mild drought": "#F57C00",
            "Moderate drought": "#FFB74D",
            "Near normal": "#E0E0E0",
            "Mild wet": "#80CBC4",
            "Very wet": "#4DB6AC",
            "Moderately wet": "#00897B",
        },
    }

    _RASTER_KEYS: dict[str, tuple[str, ...]] = {
        "flood": ("flood_risk",),
        "food_security": ("food_security_risk",),
        "drought": ("cdi", "CDI", "drought", "drought_risk"),
        "disease": ("disease_risk",),
        "land_degradation": ("degradation_risk",),
    }

    _RASTER_CLASS_SPECS: dict[str, list[tuple[int, str]]] = {
        "flood": [
            (1, "Low"),
            (2, "Medium"),
            (3, "High"),
            (4, "Very High"),
        ],
        "food_security": [
            (1, "Low Risk"),
            (2, "Medium Risk"),
            (3, "High Risk"),
        ],
        "disease": [
            (1, "Low Risk"),
            (2, "Medium Risk"),
            (3, "High Risk"),
        ],
        "land_degradation": [
            (0, "Not Degraded"),
            (1, "Degraded"),
        ],
    }

    def _raster_source(self, output: AnalysisOutput) -> str | None:
        raster = output.raster_path or output.metadata.get("raster")
        if not raster:
            return None
        if isinstance(raster, str):
            return raster
        if not isinstance(raster, dict):
            return None
        for key in self._RASTER_KEYS.get(output.module, ()):
            if raster.get(key):
                return str(raster[key])
        first = next(iter(raster.values()), None)
        return str(first) if first else None

    def _render_raster_bytes(self, output: AnalysisOutput) -> bytes | None:
        source = self._raster_source(output)
        if not source:
            return None

        is_url = source.startswith(("http://", "https://", "s3://", "gs://", "/vs"))
        if not is_url and not Path(source).exists():
            return None

        try:
            import rasterio
            from rasterio.enums import Resampling
            from rasterio.transform import array_bounds
            from rasterio.vrt import WarpedVRT
        except ImportError:
            return None

        try:
            with rasterio.open(source) as src:
                band_idx, band_label = self._select_raster_band(src, output)
                with WarpedVRT(
                    src,
                    crs="EPSG:3857",
                    resampling=Resampling.nearest,
                    nodata=src.nodata,
                ) as vrt:
                    max_dim = 1600
                    scale_factor = min(1.0, max_dim / max(vrt.width, vrt.height))
                    width = max(1, int(vrt.width * scale_factor))
                    height = max(1, int(vrt.height * scale_factor))
                    data = vrt.read(
                        band_idx,
                        out_shape=(height, width),
                        resampling=Resampling.nearest,
                    )
                    nodata = vrt.nodata
                    transform = vrt.transform * vrt.transform.scale(
                        vrt.width / width,
                        vrt.height / height,
                    )
                    bounds = array_bounds(height, width, transform)
        except Exception:
            return None

        mask = np.zeros(data.shape, dtype=bool)
        if nodata is not None:
            mask |= data == nodata
        mask |= ~np.isfinite(data)
        masked = np.ma.array(data, mask=mask)
        if masked.count() == 0:
            return None

        color_map = self._RISK_COLORS.get(output.module)
        class_spec = self._RASTER_CLASS_SPECS.get(output.module, [])
        cmap: str | ListedColormap
        vmin: float | None
        vmax: float | None
        if class_spec and color_map:
            labels = [label for _, label in class_spec]
            values = [value for value, _ in class_spec]
            value_to_index = {value: idx for idx, value in enumerate(values)}
            indexed = np.full(data.shape, -1, dtype=np.int16)
            for value, idx in value_to_index.items():
                indexed[data == value] = idx
            plot_data = np.ma.array(indexed, mask=mask | (indexed < 0))
            cmap = ListedColormap([color_map[label] for label in labels])
            vmin, vmax = -0.5, len(labels) - 0.5
        elif output.module == "drought":
            labels = [
                "Extreme drought",
                "Severe drought",
                "Moderate drought",
                "Mild drought",
                "Near normal",
                "Mild wet",
                "Moderately wet",
                "Very wet",
            ]
            indexed = np.full(data.shape, -1, dtype=np.int16)
            valid = ~mask
            indexed[valid & (data < 0.50)] = 0
            indexed[valid & (data >= 0.50) & (data < 0.65)] = 1
            indexed[valid & (data >= 0.65) & (data < 0.80)] = 2
            indexed[valid & (data >= 0.80) & (data < 0.90)] = 3
            indexed[valid & (data >= 0.90) & (data < 1.10)] = 4
            indexed[valid & (data >= 1.10) & (data < 1.20)] = 5
            indexed[valid & (data >= 1.20) & (data < 1.30)] = 6
            indexed[valid & (data >= 1.30)] = 7
            plot_data = np.ma.array(indexed, mask=mask | (indexed < 0))
            color_map = self._RISK_COLORS["drought"]
            cmap = ListedColormap([color_map[label] for label in labels])
            vmin, vmax = -0.5, len(labels) - 0.5
        else:
            labels = []
            cmap = "viridis"
            plot_data = masked
            vmin = vmax = None

        west, south, east, north = bounds
        fig, ax = plt.subplots(figsize=(10, 7))
        basemap = self._osm_basemap(bounds)
        if basemap:
            base_img, base_extent = basemap
            ax.imshow(base_img, extent=base_extent, origin="upper", zorder=0)
        ax.imshow(
            plot_data,
            extent=(west, east, south, north),
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            alpha=0.72 if basemap else 1.0,
            zorder=2,
        )

        if labels and color_map:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="s",
                    linestyle="",
                    color=color_map[label],
                    label=label,
                    markersize=9,
                )
                for label in labels
            ]
            ax.legend(
                handles=handles,
                loc="lower right",
                title="Risk Class",
                fontsize=9,
                title_fontsize=10,
                framealpha=0.9,
            )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(west, east)
        ax.set_ylim(south, north)
        self._add_scale_bar(ax, bounds)
        info = USE_CASE_REGISTRY.get(output.module)
        title_kind = "CDI Map" if output.module == "drought" else "Risk Map"
        location = self._location_label(output)
        map_title = (
            f"{info.name} {title_kind} - {location}"
            if info
            else f"{output.module.replace('_', ' ').title()} {title_kind} - {location}"
        )
        if band_label:
            map_title = f"{map_title} ({band_label})"
        ax.set_title(map_title, fontsize=12, color="#1B2A4A", pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    @staticmethod
    def _complete_year_from_end_date(end_date: str | None) -> int | None:
        if not end_date:
            return None
        try:
            dt = datetime.fromisoformat(str(end_date)[:10])
        except ValueError:
            return None
        if dt.month == 12 and dt.day == 31:
            return dt.year
        return dt.year - 1

    def _select_raster_band(
        self, src: Any, output: AnalysisOutput
    ) -> tuple[int, str | None]:
        if src.count <= 1:
            return 1, src.descriptions[0] if src.descriptions else None
        if output.module != "drought":
            return 1, src.descriptions[0] if src.descriptions else None

        target_year = self._complete_year_from_end_date(output.metadata.get("end_date"))
        candidates: list[tuple[int, int | None, str | None]] = []
        for band_idx in range(1, src.count + 1):
            description = src.descriptions[band_idx - 1] or src.tags(band_idx).get(
                "date"
            )
            year = None
            if description:
                try:
                    year = datetime.fromisoformat(str(description)[:10]).year
                except ValueError:
                    try:
                        year = int(str(description)[:4])
                    except ValueError:
                        year = None
            candidates.append((band_idx, year, description))

        if target_year is not None:
            eligible = [
                item
                for item in candidates
                if item[1] is not None
                and item[1] <= target_year
                and self._band_has_valid_pixels(src, item[0])
            ]
            if eligible:
                band_idx, _, label = max(eligible, key=lambda item: item[1] or -9999)
                return band_idx, label

        for band_idx in range(src.count, 0, -1):
            if self._band_has_valid_pixels(src, band_idx):
                label = src.descriptions[band_idx - 1] or src.tags(band_idx).get("date")
                return band_idx, label

        label = src.descriptions[-1] if src.descriptions else None
        return src.count, label

    @staticmethod
    def _band_has_valid_pixels(src: Any, band_idx: int) -> bool:
        data = src.read(band_idx, masked=True)
        if not np.ma.count(data):
            return False
        return bool(np.isfinite(np.ma.filled(data, np.nan)).any())

    @staticmethod
    def _osm_zoom(
        bounds: tuple[float, float, float, float], target_px: int = 900
    ) -> int:
        west, _, east, _ = bounds
        width_m = max(1.0, east - west)
        initial_resolution = 2 * math.pi * 6378137 / 256
        desired_resolution = width_m / target_px
        zoom = int(math.floor(math.log2(initial_resolution / desired_resolution)))
        return max(4, min(13, zoom))

    @staticmethod
    def _tile_xy(x_m: float, y_m: float, z: int) -> tuple[int, int]:
        origin_shift = math.pi * 6378137
        n = 2**z
        x = int((x_m + origin_shift) / (2 * origin_shift) * n)
        y = int((origin_shift - y_m) / (2 * origin_shift) * n)
        return x, y

    @staticmethod
    def _tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
        origin_shift = math.pi * 6378137
        n = 2**z
        tile_size = 2 * origin_shift / n
        west = x * tile_size - origin_shift
        east = (x + 1) * tile_size - origin_shift
        north = origin_shift - y * tile_size
        south = origin_shift - (y + 1) * tile_size
        return west, south, east, north

    def _osm_basemap(
        self,
        bounds: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
        west, south, east, north = bounds
        zoom = self._osm_zoom(bounds)
        min_x, max_y = self._tile_xy(west, south, zoom)
        max_x, min_y = self._tile_xy(east, north, zoom)
        min_x, max_x = sorted((min_x, max_x))
        min_y, max_y = sorted((min_y, max_y))

        tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)
        while tile_count > 36 and zoom > 4:
            zoom -= 1
            min_x, max_y = self._tile_xy(west, south, zoom)
            max_x, min_y = self._tile_xy(east, north, zoom)
            min_x, max_x = sorted((min_x, max_x))
            min_y, max_y = sorted((min_y, max_y))
            tile_count = (max_x - min_x + 1) * (max_y - min_y + 1)

        try:
            canvas = PILImage.new(
                "RGB",
                ((max_x - min_x + 1) * 256, (max_y - min_y + 1) * 256),
                (245, 245, 245),
            )
            headers = {"User-Agent": "ARIN-Climate-DSS/1.0 report-map"}
            for x in range(min_x, max_x + 1):
                for y in range(min_y, max_y + 1):
                    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                    resp = requests.get(url, headers=headers, timeout=4)
                    resp.raise_for_status()
                    tile = PILImage.open(io.BytesIO(resp.content)).convert("RGB")
                    canvas.paste(tile, ((x - min_x) * 256, (y - min_y) * 256))
        except Exception as exc:
            _log.warning(
                "OSM basemap fetch failed; map will render without basemap: %s", exc
            )
            return None

        base_west, _, _, base_north = self._tile_bounds(min_x, min_y, zoom)
        _, base_south, base_east, _ = self._tile_bounds(max_x, max_y, zoom)
        return np.asarray(canvas), (base_west, base_east, base_south, base_north)

    @staticmethod
    def _nice_distance(distance_m: float) -> float:
        if distance_m <= 0:
            return 1_000
        magnitude = 10 ** math.floor(math.log10(distance_m))
        for multiplier in (1, 2, 5, 10):
            candidate = multiplier * magnitude
            if candidate >= distance_m:
                return candidate
        return 10 * magnitude

    def _add_scale_bar(
        self,
        ax: Any,
        bounds: tuple[float, float, float, float],
    ) -> None:
        west, south, east, north = bounds
        width = east - west
        height = north - south
        bar_len = self._nice_distance(width / 5)
        if bar_len > width * 0.45:
            bar_len = self._nice_distance(width / 8)
        x0 = west + width * 0.06
        y0 = south + height * 0.06
        ax.plot([x0, x0 + bar_len], [y0, y0], color="#111827", linewidth=3, zorder=5)
        ax.plot(
            [x0, x0],
            [y0 - height * 0.008, y0 + height * 0.008],
            color="#111827",
            linewidth=2,
            zorder=5,
        )
        ax.plot(
            [x0 + bar_len, x0 + bar_len],
            [y0 - height * 0.008, y0 + height * 0.008],
            color="#111827",
            linewidth=2,
            zorder=5,
        )
        label = f"{bar_len / 1000:g} km" if bar_len >= 1000 else f"{bar_len:g} m"
        ax.text(
            x0 + bar_len / 2,
            y0 + height * 0.018,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#111827",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
            zorder=6,
        )

    def _render_choropleth_bytes(self, output: AnalysisOutput) -> bytes | None:
        """
        Render a choropleth map from output.geojson and return PNG bytes.

        Expects the FeatureCollection to contain:
          - One Feature with properties.type == "boundary" (the AOI polygon)
          - Point Features with properties.risk_class for sampled pixels

        Returns None if the geojson is empty or geopandas is unavailable.
        """
        try:
            import geopandas as gpd
            from shapely.geometry import shape
        except ImportError:
            return None

        features = (output.geojson or {}).get("features", [])
        if not features:
            return None

        boundary_geom = None
        point_feats = []
        for f in features:
            props = f.get("properties") or {}
            if props.get("type") == "boundary":
                try:
                    boundary_geom = shape(f["geometry"])
                except Exception:
                    pass
            else:
                point_feats.append(f)

        if not point_feats:
            return None

        try:
            gdf = gpd.GeoDataFrame.from_features(point_feats, crs="EPSG:4326")
        except Exception:
            return None

        color_map = self._RISK_COLORS.get(output.module, {})
        if not color_map:
            return None

        fig, ax = plt.subplots(figsize=(10, 7))

        if "risk_class" in gdf.columns:
            for risk_class, hex_color in color_map.items():
                subset = gdf[gdf["risk_class"] == risk_class]
                if not subset.empty:
                    subset.plot(
                        ax=ax,
                        color=hex_color,
                        markersize=4,
                        label=risk_class,
                        alpha=0.75,
                        zorder=2,
                    )

        if boundary_geom is not None:
            gpd.GeoSeries([boundary_geom], crs="EPSG:4326").plot(
                ax=ax,
                facecolor="none",
                edgecolor="#1B2A4A",
                linewidth=2.5,
                zorder=3,
            )

        ax.legend(
            loc="lower right",
            title="Risk Class",
            fontsize=9,
            title_fontsize=10,
            framealpha=0.85,
        )
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        ax.tick_params(labelsize=8)
        info = USE_CASE_REGISTRY.get(output.module)
        map_title = (
            f"{info.name} Risk Map"
            if info
            else f"{output.module.replace('_', ' ').title()} Risk Map"
        )
        ax.set_title(map_title, fontsize=12, color="#1B2A4A", pad=10)
        plt.tight_layout()

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    def _add_statistics_section(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        self._add_rule(story)
        story.append(Paragraph("3. Key Statistics", S["section_heading"]))

        stats = output.stats
        rows = [[k.replace("_", " ").title(), str(v)] for k, v in stats.items()]
        tbl = Table(rows, colWidths=[8 * cm, 7 * cm])
        tbl.setStyle(
            TableStyle(
                [
                    ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, LIGHT_GREY]),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    @staticmethod
    def _fig_to_image(fig: plt.Figure, width_cm: float = 15) -> Image:
        fig_w, fig_h = fig.get_size_inches()
        aspect = fig_h / fig_w
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        w = width_cm * cm
        return Image(buf, width=w, height=w * aspect)

    def _chart_timeseries(self, charts: dict) -> Image | None:
        ts = charts.get("timeseries", {})
        if not ts.get("labels") or not ts.get("datasets"):
            return None
        labels = ts["labels"]
        ds_map = {d["label"]: d["data"] for d in ts["datasets"]}
        colors_map = {
            "CDI": "#C0392B",
            "PDI": "#2980B9",
            "TDI": "#E67E22",
            "VDI": "#27AE60",
        }

        fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
        idx = range(len(labels))

        ax = axes[0]
        for key in ("PDI", "TDI", "VDI"):
            if key in ds_map:
                ax.plot(
                    idx,
                    ds_map[key],
                    label=key,
                    color=colors_map[key],
                    alpha=0.8,
                    linewidth=1.2,
                )
        ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax.set_ylabel("Sub-index value")
        ax.set_title("Drought Sub-Index Time Series (PDI · TDI · VDI)", fontsize=11)
        ax.legend(loc="upper right", fontsize=8)

        ax2 = axes[1]
        if "CDI" in ds_map:
            cdi_vals = ds_map["CDI"]
            ax2.fill_between(
                idx,
                cdi_vals,
                1.0,
                where=[v < 1.0 for v in cdi_vals],
                alpha=0.25,
                color="#E74C3C",
            )
            ax2.plot(idx, cdi_vals, color="#C0392B", linewidth=1.5, label="CDI")
        ax2.axhline(1.0, color="grey", linestyle="--", linewidth=0.8)
        ax2.set_ylabel("CDI")
        ax2.set_title("Composite Drought Index (CDI)", fontsize=11)
        step = max(1, len(labels) // 12)
        ax2.set_xticks(range(0, len(labels), step))
        ax2.set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
        ax2.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_forecast(
        self, charts: dict, cdi_series_tail: list | None = None
    ) -> Image | None:
        forecast = charts.get("forecast", {})
        if not forecast.get("dates") or not forecast.get("mean"):
            return None
        fc_dates = forecast["dates"]
        fc_mean = forecast["mean"]
        fc_lo = forecast.get("ci_lower", fc_mean)
        fc_hi = forecast.get("ci_upper", fc_mean)

        # use last 24 points of timeseries as historical context
        ts = charts.get("timeseries", {})
        hist_vals: list = []
        hist_labels: list = []
        if ts.get("datasets"):
            cdi_ds = next((d for d in ts["datasets"] if d["label"] == "CDI"), None)
            if cdi_ds:
                hist_vals = cdi_ds["data"][-24:]
                hist_labels = (ts.get("labels") or [])[-24:]

        fig, ax = plt.subplots(figsize=(12, 4))
        hist_idx = range(len(hist_vals))
        if hist_vals:
            ax.plot(
                list(hist_idx),
                hist_vals,
                color="#C0392B",
                linewidth=1.5,
                label="Historical CDI",
            )
        fc_idx = range(len(hist_vals), len(hist_vals) + len(fc_dates))
        ax.plot(
            list(fc_idx),
            fc_mean,
            color="#E67E22",
            linewidth=2,
            linestyle="--",
            marker="o",
            markersize=5,
            label="Forecast (mean)",
        )
        ax.fill_between(
            list(fc_idx), fc_lo, fc_hi, alpha=0.25, color="#E67E22", label="95% CI"
        )
        ax.axhline(
            0.90, color="gold", linestyle=":", linewidth=1, label="Mild drought (0.90)"
        )
        ax.axhline(
            0.65,
            color="orange",
            linestyle=":",
            linewidth=1,
            label="Moderate drought (0.65)",
        )
        all_labels = hist_labels + fc_dates
        step = max(1, len(all_labels) // 10)
        ax.set_xticks(range(0, len(all_labels), step))
        ax.set_xticklabels(all_labels[::step], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("CDI")
        ci_summary = forecast.get("ci_summary", "")
        ax.set_title(f"LSTM 6-Month CDI Forecast  {ci_summary}", fontsize=11)
        ax.legend(loc="upper left", fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_anomaly(self, charts: dict) -> Image | None:
        anomaly = charts.get("anomaly", {})
        if not anomaly.get("labels") or not anomaly.get("data"):
            return None
        years = anomaly["labels"]
        data = anomaly["data"]
        bar_colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in data]

        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.bar(years, data, color=bar_colors, edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel("Year")
        ax.set_ylabel("CDI anomaly")
        ax.set_title(
            f"Annual CDI Anomaly (mean baseline = {anomaly.get('mean', 0):.3f})",
            fontsize=11,
        )
        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_seasonal(self, charts: dict) -> Image | None:
        seasonal = charts.get("seasonal", {})
        if not seasonal.get("labels") or not seasonal.get("data"):
            return None
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        data = seasonal["data"]

        fig, ax = plt.subplots(figsize=(9, 3))
        bar_colors = ["#E74C3C" if v < 0.9 else "#2ECC71" for v in data]
        ax.bar(months[: len(data)], data, color=bar_colors, edgecolor="white")
        ax.axhline(
            0.9,
            color="gold",
            linestyle="--",
            linewidth=1,
            label="Mild drought threshold",
        )
        ax.axhline(
            1.0, color="grey", linestyle="--", linewidth=0.8, label="Normal (CDI=1)"
        )
        ax.set_ylabel("Mean CDI")
        ax.set_title("Seasonal CDI Pattern (climatological monthly mean)", fontsize=11)
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=13)

    def _chart_severity_dist(self, charts: dict) -> Image | None:
        sev_dist = charts.get("severity_distribution", {})
        if not sev_dist.get("labels") or not sev_dist.get("data"):
            return None
        CLASS_COLORS = {
            "Extreme drought": "#990000",
            "Severe drought": "#E65100",
            "Mild drought": "#F57C00",
            "Moderate drought": "#FFB74D",
            "Near normal": "#E0E0E0",
            "Mild wet": "#80CBC4",
            "Very wet": "#4DB6AC",
            "Moderately wet": "#00897B",
        }
        labels = sev_dist["labels"]
        data = sev_dist["data"]
        bar_colors = [
            CLASS_COLORS.get(lbl)
            or CLASS_COLORS.get(str(lbl).strip().capitalize())
            or "#95A5A6"
            for lbl in labels
        ]

        fig, ax = plt.subplots(figsize=(10, 3.5))
        bars = ax.bar(labels, data, color=bar_colors, edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_ylabel("% of time periods")
        ax.set_title("CDI Severity Distribution", fontsize=11)
        ax.tick_params(axis="x", rotation=20)
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=13)

    def _chart_typology(self, charts: dict) -> Image | None:
        typology = charts.get("typology", {})
        if not typology.get("label_map") or not typology.get("lons"):
            return None
        lons = typology["lons"]
        lats = typology["lats"]
        label_map = np.array(typology["label_map"])
        n_clusters = typology["n_clusters"]
        clusters = typology.get("clusters", [])

        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        im = ax.imshow(
            label_map.T,
            origin="lower",
            aspect="auto",
            extent=[min(lons), max(lons), min(lats), max(lats)],
            cmap="Set1",
            vmin=0,
            vmax=n_clusters - 1,
        )
        plt.colorbar(im, ax=ax, label="Cluster")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"KMeans Drought Typology ({n_clusters} clusters)", fontsize=10)

        ax2 = axes[1]
        if clusters:
            cluster_ids = [c["cluster_id"] for c in clusters]
            mean_cdis = [c["mean_cdi"] for c in clusters]
            pixel_pcts = [c["pixel_pct"] for c in clusters]
            cmap = plt.get_cmap("Set1")
            bar_cols = [cmap(i / max(n_clusters - 1, 1)) for i in cluster_ids]
            bars = ax2.bar(
                [f"C{i}" for i in cluster_ids],
                mean_cdis,
                color=bar_cols,
                edgecolor="white",
            )
            for bar, pct in zip(bars, pixel_pcts):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{pct:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
            ax2.axhline(
                0.9,
                color="gold",
                linestyle="--",
                linewidth=1,
                label="Mild drought (0.90)",
            )
            ax2.set_ylabel("Mean CDI")
            ax2.set_title("Cluster Mean CDI (label = % pixels)", fontsize=10)
            ax2.legend(fontsize=8)

        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_training(self, charts: dict) -> Image | None:
        training = charts.get("training", {})
        if not training.get("train_losses") or not training.get("val_losses"):
            return None
        train_losses = training["train_losses"]
        val_losses = training["val_losses"]

        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(train_losses, label="Train loss", color="#2980B9", linewidth=1.2)
        ax.plot(val_losses, label="Val loss", color="#E74C3C", linewidth=1.2)
        stopped = training.get("stopped_epoch")
        if stopped:
            ax.axvline(
                stopped,
                color="grey",
                linestyle="--",
                linewidth=0.8,
                label=f"Early stop (epoch {stopped})",
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE loss")
        ax.set_title("LSTM Training History", fontsize=11)
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=13)

    def _add_drought_sections(self, story: list, output: AnalysisOutput) -> None:
        """Drought-specific: CDI time-series, forecast, anomaly, seasonal, severity, typology."""
        S = self._styles
        charts = output.charts or {}

        # 4. CDI Time-Series
        self._add_rule(story)
        story.append(Paragraph("4. CDI Time-Series", S["section_heading"]))
        story.append(
            Paragraph(
                "The Composite Drought Index (CDI) combines PDI (50%), TDI (25%), and VDI (25%). "
                "Values below 1.0 indicate drought; shaded area shows months below normal.",
                S["body"],
            )
        )
        img = self._chart_timeseries(charts)
        if img:
            story.append(img)
            story.append(
                Paragraph("Figure 2. CDI and sub-index time series.", S["caption"])
            )
        story.append(Spacer(1, 0.4 * cm))

        # 5. LSTM Forecast
        forecast = charts.get("forecast", {})
        if forecast.get("dates") and forecast.get("mean"):
            self._add_rule(story)
            story.append(Paragraph("5. LSTM 6-Month Forecast", S["section_heading"]))
            story.append(
                Paragraph(
                    "6-month ahead CDI forecast with 95% confidence interval "
                    "(Monte Carlo Dropout, 500 samples).",
                    S["body"],
                )
            )
            img = self._chart_forecast(charts)
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 3. LSTM forecast with 95% CI bands.", S["caption"]
                    )
                )
            story.append(Spacer(1, 0.3 * cm))
            # Compact forecast table below the chart
            fc_rows = [["Month", "CDI", "CI Low", "CI High"]]
            for i, date in enumerate(forecast["dates"]):
                fc_rows.append(
                    [
                        date,
                        f"{forecast['mean'][i]:.3f}",
                        f"{forecast['ci_lower'][i]:.3f}"
                        if forecast.get("ci_lower")
                        else "—",
                        f"{forecast['ci_upper'][i]:.3f}"
                        if forecast.get("ci_upper")
                        else "—",
                    ]
                )
            fc_tbl = Table(fc_rows, colWidths=[4 * cm, 3.5 * cm, 3.5 * cm, 4 * cm])
            fc_tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, LIGHT_GREY],
                        ),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(fc_tbl)
            story.append(Spacer(1, 0.4 * cm))

        # 6. Severity Distribution
        self._add_rule(story)
        story.append(Paragraph("6. Severity Distribution", S["section_heading"]))
        img = self._chart_severity_dist(charts)
        if img:
            story.append(img)
            story.append(
                Paragraph(
                    "Figure 4. CDI severity class distribution across all time periods.",
                    S["caption"],
                )
            )
        story.append(Spacer(1, 0.4 * cm))

        # 7. Annual Anomaly
        if charts.get("anomaly", {}).get("data"):
            self._add_rule(story)
            story.append(Paragraph("7. Annual CDI Anomaly", S["section_heading"]))
            story.append(
                Paragraph(
                    "Green bars indicate above-baseline (wetter) years; red bars indicate below-baseline (drier) years.",
                    S["body"],
                )
            )
            img = self._chart_anomaly(charts)
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 5. Annual CDI anomaly relative to long-term mean.",
                        S["caption"],
                    )
                )
            story.append(Spacer(1, 0.4 * cm))

        # 8. Seasonal Pattern
        if charts.get("seasonal", {}).get("data"):
            self._add_rule(story)
            story.append(Paragraph("8. Seasonal CDI Pattern", S["section_heading"]))
            img = self._chart_seasonal(charts)
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 6. Climatological monthly mean CDI.", S["caption"]
                    )
                )
            story.append(Spacer(1, 0.4 * cm))

        # 9. Drought Typology
        if charts.get("typology", {}).get("label_map"):
            self._add_rule(story)
            story.append(Paragraph("9. Spatial Drought Typology", S["section_heading"]))
            story.append(
                Paragraph(
                    "KMeans clustering of pixel CDI trajectories identifies spatially coherent drought typologies. "
                    "Cluster 0 is the most drought-prone.",
                    S["body"],
                )
            )
            img = self._chart_typology(charts)
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 7. Left: typology cluster map. Right: mean CDI per cluster (label = % of pixels).",
                        S["caption"],
                    )
                )
            story.append(Spacer(1, 0.4 * cm))

        # 10. LSTM Training History
        if charts.get("training", {}).get("train_losses"):
            self._add_rule(story)
            story.append(Paragraph("10. LSTM Training History", S["section_heading"]))
            img = self._chart_training(charts)
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 8. Train vs. validation MSE loss per epoch.",
                        S["caption"],
                    )
                )
            story.append(Spacer(1, 0.4 * cm))

        # CDI severity reference legend
        self._add_rule(story)
        story.append(Paragraph("CDI Severity Reference", S["section_heading"]))
        legend_data = [["Severity Class", "CDI Range", "Colour"]]
        severity_rows = [
            ("Extreme drought", "< 0.50", CDI_COLORS["Extreme drought"]),
            ("Severe drought", "0.50 - 0.65", CDI_COLORS["Severe drought"]),
            ("Moderate drought", "0.65 - 0.80", CDI_COLORS["Moderate drought"]),
            ("Mild drought", "0.80 - 0.90", CDI_COLORS["Mild drought"]),
            ("Near normal", "0.90 - 1.10", CDI_COLORS["Near normal"]),
            ("Mild wet", "1.10 - 1.20", CDI_COLORS["Mild wet"]),
            ("Moderately wet", "1.20 - 1.30", CDI_COLORS["Moderately wet"]),
            ("Very wet", "> 1.30", CDI_COLORS["Very wet"]),
        ]
        for label, rng, _ in severity_rows:
            legend_data.append([label, rng, ""])
        tbl = Table(legend_data, colWidths=[6 * cm, 4 * cm, 5 * cm])
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]
        for i, (_, _, hex_col) in enumerate(severity_rows, start=1):
            style_cmds.append(("BACKGROUND", (2, i), (2, i), colors.HexColor(hex_col)))
        tbl.setStyle(TableStyle(style_cmds))
        story.append(tbl)
        story.append(Spacer(1, 0.4 * cm))

    # ── Shared chart renderers (used by disease / food_security / flood / land_degradation) ──

    def _chart_risk_dist_generic(
        self, labels: list, data: list, bar_colors: list, title: str
    ) -> Image | None:
        if not labels or not data:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        ax, ax2 = axes
        bars = ax.bar(labels, data, color=bar_colors, edgecolor="white", linewidth=0.5)
        for bar, pct in zip(bars, data):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{pct:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )
        ax.set_ylabel("% of sampled pixels")
        ax.set_title(title, fontsize=11)
        ax.set_ylim(0, max(data) + 8)
        ax.tick_params(axis="x", rotation=15)
        ax2.pie(
            data,
            labels=labels,
            colors=bar_colors,
            autopct="%1.1f%%",
            startangle=140,
            pctdistance=0.75,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax2.set_title("Proportions", fontsize=11)
        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_timeseries_multi(self, ts_data: dict, title: str) -> Image | None:
        datasets = ts_data.get("datasets", [])
        labels = ts_data.get("labels", [])
        if not datasets or not labels:
            return None
        n = len(datasets)
        fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, ds in zip(axes, datasets):
            vals = [v if v is not None else float("nan") for v in ds["data"]]
            col = ds.get("color", "#2980B9")
            ax.plot(
                range(len(labels)), vals, color=col, linewidth=1.5, label=ds["label"]
            )
            ax.fill_between(range(len(labels)), vals, alpha=0.12, color=col)
            ax.set_ylabel(ds["label"], fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
        step = max(1, len(labels) // 12)
        axes[-1].set_xticks(range(0, len(labels), step))
        axes[-1].set_xticklabels(labels[::step], rotation=45, ha="right", fontsize=7)
        axes[0].set_title(title, fontsize=11)
        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_shap(
        self, shap_data: dict, title: str = "Feature Importance (SHAP)"
    ) -> Image | None:
        features = shap_data.get("features", [])
        values = shap_data.get("mean_abs_shap", [])
        if not features or not values:
            return None
        cmap = plt.get_cmap("RdYlGn_r")
        bar_colors = [cmap(i / max(len(features) - 1, 1)) for i in range(len(features))]
        fig, ax = plt.subplots(figsize=(10, max(3, len(features) * 0.45)))
        y_pos = range(len(features))
        bars = ax.barh(
            list(y_pos), values, color=bar_colors, edgecolor="white", linewidth=0.4
        )
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(title, fontsize=11)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=8,
            )
        ax.invert_yaxis()
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=13)

    def _chart_model_perf(
        self, perf_data: dict, metrics: list[tuple[str, str]]
    ) -> Image | None:
        """
        perf_data: {model_key: {metric1: v, metric2: v, ...}, ..., 'selected': key}
        metrics: [(key, label), ...] — which metrics to plot as grouped bars
        """
        model_keys = [k for k in perf_data if k != "selected"]
        if not model_keys or not metrics:
            return None
        x = np.arange(len(model_keys))
        width = 0.8 / len(metrics)
        fig, ax = plt.subplots(figsize=(9, 4))
        palette = ["#1B2A4A", "#00897B", "#F9A825", "#E74C3C"]
        for i, (metric_key, metric_label) in enumerate(metrics):
            vals = [perf_data[k].get(metric_key, 0) for k in model_keys]
            offset = (i - len(metrics) / 2 + 0.5) * width
            bars = ax.bar(
                x + offset,
                vals,
                width * 0.9,
                label=metric_label,
                color=palette[i % len(palette)],
                alpha=0.85,
            )
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        selected = perf_data.get("selected")
        if selected and selected in model_keys:
            sel_idx = model_keys.index(selected)
            ax.axvline(
                sel_idx,
                color="gold",
                linewidth=2,
                linestyle="--",
                alpha=0.7,
                label=f"Selected: {selected}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([k.upper() for k in model_keys])
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title("Model Performance Comparison (held-out test set)", fontsize=11)
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=13)

    def _chart_ndvi_trend(self, ts_data: dict, trend_stats: dict) -> Image | None:
        datasets = ts_data.get("datasets", [])
        labels = ts_data.get("labels", [])
        if not datasets or not labels:
            return None
        ndvi_ds = next(
            (d for d in datasets if "ndvi" in d["label"].lower()), datasets[0]
        )
        vals = np.array([v if v is not None else float("nan") for v in ndvi_ds["data"]])
        years = np.array(labels, dtype=float)

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(
            years,
            vals,
            color="#27AE60",
            linewidth=1.5,
            label="Annual NDVI",
            marker="o",
            markersize=4,
        )

        # OLS trend line
        slope = trend_stats.get("ndvi_trend_per_year")
        if slope is not None:
            valid = ~np.isnan(vals)
            if valid.sum() >= 2:
                intercept = np.nanmean(vals) - slope * np.nanmean(years[valid])
                trend_line = slope * years + intercept
                ax.plot(
                    years,
                    trend_line,
                    color="#E74C3C",
                    linewidth=1.5,
                    linestyle="--",
                    label=f"Trend: {slope:+.4f}/yr (p={trend_stats.get('ndvi_trend_p', 'n/a'):.3f})",
                )

        for bkp in trend_stats.get("breakpoint_years", []):
            ax.axvline(
                bkp,
                color="orange",
                linewidth=1.2,
                linestyle=":",
                label=f"Breakpoint {bkp}",
            )

        ax.set_xlabel("Year")
        ax.set_ylabel("NDVI")
        ax.set_title(
            "Annual NDVI Trend with OLS Regression and Breakpoints", fontsize=11
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig)

    def _chart_flood_uncertainty(self, uncertainty: dict) -> Image | None:
        spread_stats = uncertainty.get("spread_stats", {})
        if not spread_stats:
            return None
        keys = ["min", "p25", "mean", "p75", "max"]
        labels = ["Min", "P25", "Mean", "P75", "Max"]
        vals = [spread_stats.get(k, 0) for k in keys]

        fig, ax = plt.subplots(figsize=(8, 3))
        bar_colors = [
            "#2ECC71" if v < 0.10 else "#F1C40F" if v < 0.20 else "#E74C3C"
            for v in vals
        ]
        bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white")
        ax.axhline(
            0.20,
            color="#E74C3C",
            linestyle="--",
            linewidth=1,
            label="High uncertainty threshold (0.20)",
        )
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        ax.set_ylabel("RF–XGBoost probability spread")
        high_pct = uncertainty.get("high_uncertainty_pct", 0)
        ax.set_title(
            f"Model Uncertainty (epistemic) — {high_pct:.1f}% pixels above threshold",
            fontsize=11,
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=12)

    def _chart_vhi_indices(self, indices: dict) -> Image | None:
        vci = indices.get("vci_mean")
        tci = indices.get("tci_mean")
        vhi = indices.get("vhi_mean")
        if all(v is None for v in [vci, tci, vhi]):
            return None
        labels = []
        values = []
        bar_colors = []
        for name, val, col in [
            ("VCI", vci, "#27AE60"),
            ("TCI", tci, "#E74C3C"),
            ("VHI", vhi, "#2980B9"),
        ]:
            if val is not None:
                labels.append(name)
                values.append(val)
                bar_colors.append(col)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        bars = ax.bar(
            labels,
            values,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
            width=0.5,
        )
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )
        ax.axhline(
            40,
            color="#E74C3C",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label="Drought threshold (40)",
        )
        ax.axhline(
            50,
            color="gray",
            linestyle=":",
            linewidth=1,
            alpha=0.6,
            label="Moderate stress (50)",
        )
        ax.set_ylim(0, max(values) + 15)
        ax.set_ylabel("Index value (0–100)")
        ax.set_title("Vegetation Health Index Summary", fontsize=11)
        ax.legend(fontsize=8)
        plt.tight_layout()
        return self._fig_to_image(fig, width_cm=11)

    def _interpret(self, story: list, text: str) -> None:
        """Append an italic interpretation paragraph to story."""
        story.append(Paragraph(text, self._styles["interpretation"]))
        story.append(Spacer(1, 0.2 * cm))

    # ── Module section builders ───────────────────────────────────────────────

    def _add_disease_sections(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        charts = output.charts or {}
        stats = output.stats or {}
        high_risk_pct = stats.get("high_risk_pct", 0)
        selected_f1 = stats.get("selected_f1", 0)
        model_type = stats.get("model_type", "ensemble")

        # 4. Risk Distribution
        rd = charts.get("riskDist", {})
        if rd.get("labels") and rd.get("data"):
            self._add_rule(story)
            story.append(
                Paragraph("4. Disease Risk Distribution", S["section_heading"])
            )
            story.append(
                Paragraph(
                    "Risk classes are derived from the GBM / XGBoost ensemble probability: "
                    "Low (< 0.33), Medium (0.33–0.67), High (≥ 0.67).",
                    S["body"],
                )
            )
            img = self._chart_risk_dist_generic(
                rd["labels"],
                rd["data"],
                rd.get("colors", ["#2ECC71", "#F1C40F", "#E74C3C"]),
                "Disease Risk Distribution",
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 2. Risk class distribution across sampled pixels.",
                        S["caption"],
                    )
                )
            severity = "widespread" if high_risk_pct > 30 else "localised"
            action = (
                "immediate public health surveillance is warranted"
                if high_risk_pct > 30
                else "continued monitoring is recommended to detect emerging clusters"
            )
            self._interpret(
                story,
                f"High-risk pixels account for {high_risk_pct:.1f}% of the sampled area, indicating "
                f"{severity} climate-driven disease exposure. {action.capitalize()}. "
                f"Risk classes are equally distributed at tercile thresholds of the composite "
                f"climate-suitability score.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 5. Monthly Environmental Indicators
        ts = charts.get("timeSeries", {})
        if ts.get("datasets"):
            self._add_rule(story)
            story.append(
                Paragraph("5. Monthly Environmental Indicators", S["section_heading"])
            )
            img = self._chart_timeseries_multi(ts, "Monthly NDVI / Rainfall / LST")
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 3. Monthly NDVI, rainfall, and land surface temperature.",
                        S["caption"],
                    )
                )
            n_series = len(ts.get("datasets", []))
            series_names = ", ".join(d["label"] for d in ts.get("datasets", []))
            self._interpret(
                story,
                f"The chart shows {n_series} environmental indicator(s): {series_names}. "
                f"Disease transmission risk typically peaks 2–4 weeks after elevated rainfall "
                f"and temperature, consistent with vector breeding cycles. "
                f"Periods of co-occurring high rainfall and high LST are particularly high-risk windows.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 6. SHAP Feature Importance
        shap_data = charts.get("shap", {})
        if shap_data.get("features"):
            self._add_rule(story)
            story.append(Paragraph("6. SHAP Feature Importance", S["section_heading"]))
            story.append(
                Paragraph(
                    "Mean absolute SHAP values from XGBoost TreeExplainer averaged across all risk classes.",
                    S["body"],
                )
            )
            features = shap_data["features"]
            values = shap_data.get("mean_abs_shap", [])
            img = self._chart_shap(
                shap_data,
                "Feature Importance (XGBoost SHAP — mean |SHAP| across classes)",
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        f"Figure 4. Top risk driver: {features[0]}.", S["caption"]
                    )
                )
            self._interpret(
                story,
                f"The primary driver of disease risk is '{features[0]}' "
                f"(mean |SHAP| = {values[0]:.4f}), followed by '{features[1] if len(features) > 1 else 'N/A'}'. "
                f"These climate variables dominate model predictions and represent the most actionable "
                f"targets for early-warning system design. "
                f"Features with near-zero SHAP values contribute negligible discriminative power "
                f"and may be candidates for removal in future model iterations.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 7. Model Performance
        perf = charts.get("model_performance", {})
        if perf:
            self._add_rule(story)
            story.append(
                Paragraph("7. Model Performance Comparison", S["section_heading"])
            )
            img = self._chart_model_perf(
                perf, [("f1", "F1-macro"), ("accuracy", "Accuracy")]
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 5. GBM, XGBoost, and ensemble F1 (macro) and accuracy on held-out test set.",
                        S["caption"],
                    )
                )
            perf_desc = (
                "strong"
                if selected_f1 > 0.70
                else "moderate"
                if selected_f1 > 0.50
                else "fair"
            )
            self._interpret(
                story,
                f"The selected '{model_type}' model achieved F1 = {selected_f1:.3f} ({perf_desc} performance "
                f"for 3-class climate-driven disease risk classification). "
                f"Cross-validated F1 scores confirm model stability across data folds. "
                f"The ensemble typically smooths class boundaries and reduces overfitting relative to "
                f"any single classifier.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 8. Hotspot cluster summary table
        n_clusters = stats.get("n_hotspot_clusters")
        if n_clusters is not None:
            self._add_rule(story)
            story.append(
                Paragraph("8. DBSCAN Hotspot Cluster Summary", S["section_heading"])
            )
            hotspot_rows: list[list[str]] = [["Metric", "Value"]]
            hotspot_rows.append(["Number of hotspot clusters", str(n_clusters)])
            if stats.get("hotspot_population") is not None:
                hotspot_rows.append(
                    [
                        "Estimated population at risk",
                        f"{stats['hotspot_population']:,.0f}",
                    ]
                )
            if stats.get("high_risk_pct") is not None:
                hotspot_rows.append(
                    ["High-risk pixel %", f"{stats['high_risk_pct']:.1f}%"]
                )
            hotspot_list = charts.get("hotspots", [])
            if hotspot_list:
                hotspot_rows.append(
                    ["Largest cluster size (pixels)", str(hotspot_list[0]["size"])]
                )
            tbl = Table(hotspot_rows, colWidths=[9 * cm, 6 * cm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), TEAL),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, LIGHT_GREY],
                        ),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(tbl)
            cluster_desc = (
                "no spatially coherent hotspots"
                if n_clusters == 0
                else f"{n_clusters} spatial hotspot cluster(s)"
            )
            self._interpret(
                story,
                f"DBSCAN clustering identified {cluster_desc} within the high-risk pixel layer. "
                f"{'Clusters represent areas of concentrated climate suitability and should be prioritised for field-level surveillance.' if n_clusters > 0 else 'The absence of clusters suggests diffuse, non-concentrated risk across the AOI.'}",
            )
            story.append(Spacer(1, 0.4 * cm))

    def _add_food_security_sections(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        charts = output.charts or {}
        stats = output.stats or {}
        high_risk_pct = stats.get("high_risk_pct", 0)
        selected_f1 = stats.get("selected_f1", 0)
        model_type = stats.get("model_type", "rf")

        def _vhi_label(v: float) -> str:
            if v < 10:
                return "Extreme stress"
            if v < 20:
                return "Severe stress"
            if v < 35:
                return "Moderate stress"
            if v < 50:
                return "Mild stress"
            return "No significant stress"

        # 4. Vegetation Health Indices — bar chart + summary table
        indices = charts.get("indices", {})
        vci = indices.get("vci_mean", stats.get("vci_mean"))
        tci = indices.get("tci_mean", stats.get("tci_mean"))
        vhi = indices.get("vhi_mean", stats.get("vhi_mean"))
        if any(v is not None for v in [vci, tci, vhi]):
            self._add_rule(story)
            story.append(
                Paragraph("4. Vegetation Health Indices", S["section_heading"])
            )
            story.append(
                Paragraph(
                    "VCI (Vegetation Condition Index) and TCI (Temperature Condition Index) are "
                    "normalised 0–100 indices derived from MODIS NDVI and LST relative to the "
                    "long-term baseline. VHI = 0.5×VCI + 0.5×TCI. "
                    "Values below 40 indicate moderate-to-severe agricultural drought stress.",
                    S["body"],
                )
            )
            img = self._chart_vhi_indices(
                {"vci_mean": vci, "tci_mean": tci, "vhi_mean": vhi}
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 2. VCI, TCI, and VHI area-mean values.", S["caption"]
                    )
                )
            # Summary table
            idx_rows = [["Index", "Area Mean", "Interpretation"]]
            for name, val in [("VCI", vci), ("TCI", tci), ("VHI", vhi)]:
                if val is not None:
                    idx_rows.append([name, f"{val:.1f}", _vhi_label(val)])
            tbl = Table(idx_rows, colWidths=[4 * cm, 5 * cm, 6 * cm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, LIGHT_GREY],
                        ),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(tbl)
            vhi_label = _vhi_label(vhi) if vhi is not None else "unavailable"
            vci_cond = (
                "below the drought threshold"
                if (vci or 100) < 40
                else "within acceptable range"
            )
            tci_cond = (
                "indicating elevated heat stress"
                if (tci or 100) < 50
                else "indicating manageable temperature conditions"
            )
            vhi_str = f"{vhi:.1f}/100" if vhi is not None else "N/A"
            vci_str = f"{vci:.1f}" if vci is not None else "N/A"
            tci_str = f"{tci:.1f}" if tci is not None else "N/A"
            self._interpret(
                story,
                f"The mean VHI of {vhi_str} indicates '{vhi_label}' across the study area. "
                f"VCI ({vci_str}) is {vci_cond}, reflecting vegetation condition relative to the "
                f"long-term MODIS baseline. TCI ({tci_str}) — {tci_cond}. "
                f"VHI values below 40 warrant food security alerts and targeted intervention.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 5. Risk Distribution
        rd = charts.get("riskDist", {})
        if rd.get("labels") and rd.get("data"):
            self._add_rule(story)
            story.append(
                Paragraph("5. Food Security Risk Distribution", S["section_heading"])
            )
            story.append(
                Paragraph(
                    "Three risk classes are assigned by tercile thresholds on the composite food "
                    "stress score: Low (bottom 33%), Medium (middle 33%), High (top 33%). "
                    "Score = 0.40×VCI-stress + 0.25×TCI-stress + 0.20×rainfall-deficit + 0.15×NDVI-slope-inv.",
                    S["body"],
                )
            )
            img = self._chart_risk_dist_generic(
                rd["labels"],
                rd["data"],
                rd.get("colors", ["#2ECC71", "#F1C40F", "#E74C3C"]),
                "Food Security Risk Distribution",
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 3. Risk class distribution across sampled pixels.",
                        S["caption"],
                    )
                )
            severity = (
                "widespread food insecurity stress"
                if high_risk_pct > 33
                else "moderate and localised food insecurity pressure"
            )
            self._interpret(
                story,
                f"{high_risk_pct:.1f}% of sampled pixels are classified as High Risk, indicating "
                f"{severity}. "
                f"{'Immediate food security programming — including emergency food assistance and livelihood support — should be considered in high-risk zones.' if high_risk_pct > 33 else 'Conditions warrant heightened surveillance and preparedness planning to prevent deterioration.'}",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 6. Monthly NDVI and Rainfall Time Series
        ts = charts.get("timeSeries", {})
        if ts.get("datasets"):
            self._add_rule(story)
            story.append(
                Paragraph("6. Monthly NDVI and Rainfall", S["section_heading"])
            )
            img = self._chart_timeseries_multi(ts, "Monthly NDVI and Rainfall")
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 4. Monthly area-mean NDVI (MODIS) and cumulative rainfall (CHIRPS) "
                        "over the study period.",
                        S["caption"],
                    )
                )
            ds_names = [d["label"] for d in ts.get("datasets", [])]
            self._interpret(
                story,
                f"The time series captures {'NDVI and monthly rainfall dynamics' if len(ds_names) >= 2 else ds_names[0] + ' dynamics'} "
                f"over the study period. Seasonal troughs in NDVI correspond to dry-season periods "
                f"when food production is most constrained. "
                f"Years with below-average rainfall peaks tend to lag into persistent low NDVI — "
                f"a key early-warning signal for food insecurity accumulation.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 7. SHAP Feature Importance
        shap_data = charts.get("shap", {})
        if shap_data.get("features"):
            self._add_rule(story)
            story.append(Paragraph("7. SHAP Feature Importance", S["section_heading"]))
            story.append(
                Paragraph(
                    "Mean absolute SHAP values from XGBoost TreeExplainer, averaged across all three "
                    "risk classes. Higher values indicate greater influence on the food stress prediction.",
                    S["body"],
                )
            )
            features = shap_data["features"]
            values = shap_data.get("mean_abs_shap", [])
            img = self._chart_shap(
                shap_data, "Feature Importance (XGBoost SHAP — Food Security Drivers)"
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        f"Figure 5. Top food security driver: '{features[0]}' "
                        f"(mean |SHAP| = {values[0]:.4f}).",
                        S["caption"],
                    )
                )
            second = features[1] if len(features) > 1 else "N/A"
            self._interpret(
                story,
                f"The dominant driver of food insecurity risk is '{features[0]}' "
                f"(mean |SHAP| = {values[0]:.4f}), followed by '{second}'. "
                f"Interventions targeting '{features[0]}' — such as irrigation, drought-resistant "
                f"crop varieties, or temperature-adapted farming calendars — are likely to have the "
                f"greatest impact on reducing food insecurity in this AOI.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 8. Model Performance
        perf = charts.get("model_performance", {})
        if perf:
            self._add_rule(story)
            story.append(
                Paragraph("8. Model Performance Comparison", S["section_heading"])
            )
            img = self._chart_model_perf(
                perf, [("f1", "F1-macro"), ("accuracy", "Accuracy")]
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 6. Random Forest, XGBoost, and ensemble F1-macro and accuracy "
                        "on the held-out test set (20% stratified split).",
                        S["caption"],
                    )
                )
            perf_desc = (
                "strong"
                if selected_f1 > 0.70
                else "moderate"
                if selected_f1 > 0.50
                else "fair"
            )
            self._interpret(
                story,
                f"The selected '{model_type}' model achieved macro F1 = {selected_f1:.3f} "
                f"({perf_desc} three-class performance). "
                f"The 20% held-out test set was stratified to preserve class balance. "
                f"Cross-validated F1 scores confirm that results are robust and not driven "
                f"by a favourable random split.",
            )
            story.append(Spacer(1, 0.4 * cm))

    def _add_flood_sections(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        charts = output.charts or {}
        stats = output.stats or {}
        very_high_pct = stats.get("very_high_risk_pct", 0)
        high_pct = stats.get("high_risk_pct", 0)
        selected_f1 = stats.get("selected_f1", 0)
        selected_auc = stats.get("selected_auc", 0)
        model_type = stats.get("model_type", "ensemble")
        threshold = stats.get("selected_threshold", 0.5)

        # 4. Risk Distribution
        rd = charts.get("risk_distribution", {})
        if rd.get("labels") and rd.get("data"):
            self._add_rule(story)
            story.append(Paragraph("4. Flood Risk Distribution", S["section_heading"]))
            story.append(
                Paragraph(
                    "Flood probability is thresholded per-model via precision-recall optimisation. "
                    "Risk classes: Very High (≥ 0.75), High (0.50–0.75), Medium (0.25–0.50), Low (< 0.25).",
                    S["body"],
                )
            )
            img = self._chart_risk_dist_generic(
                rd["labels"],
                rd["data"],
                rd.get("colors", ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71"]),
                "Flood Risk Distribution",
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 2. Flood risk class distribution across sampled pixels.",
                        S["caption"],
                    )
                )
            combined_high = very_high_pct + high_pct
            hazard = (
                "a significant flood hazard requiring immediate response"
                if combined_high > 25
                else "a moderate flood hazard"
                if combined_high > 10
                else "a relatively low flood hazard"
            )
            self._interpret(
                story,
                f"Very High risk accounts for {very_high_pct:.1f}% and High risk for {high_pct:.1f}% "
                f"of sampled pixels (combined {combined_high:.1f}%), indicating {hazard}. "
                f"Risk classification uses the '{model_type}' model probability at optimised "
                f"threshold {threshold:.3f} (precision-recall F1 maximisation). "
                f"Pixels in the Very High class should be prioritised for early-warning dissemination "
                f"and evacuation planning.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 5. SHAP Feature Importance
        shap_data = charts.get("shap", {})
        if shap_data.get("features"):
            self._add_rule(story)
            story.append(Paragraph("5. SHAP Feature Importance", S["section_heading"]))
            story.append(
                Paragraph(
                    "TreeExplainer SHAP values computed on the XGBoost model. "
                    "Mean absolute SHAP indicates how much each feature shifts the flood probability "
                    "prediction from the base rate.",
                    S["body"],
                )
            )
            features = shap_data["features"]
            values = shap_data.get("mean_abs_shap", [])
            img = self._chart_shap(
                shap_data, "Feature Importance (XGBoost SHAP — Flood Drivers)"
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        f"Figure 3. Top flood driver: '{features[0]}' (mean |SHAP| = {values[0]:.4f}).",
                        S["caption"],
                    )
                )
            second = features[1] if len(features) > 1 else "N/A"
            topo_drivers = [
                f for f in features[:3] if f in ("elevation", "twi", "dist_river")
            ]
            topo_note = (
                f"Topographic features ({', '.join(topo_drivers)}) dominate, confirming "
                f"terrain-controlled inundation patterns."
                if topo_drivers
                else "Dynamic weather signals (SAR change, rainfall) are the primary drivers, "
                "suggesting event-intensity rather than chronic topographic exposure."
            )
            self._interpret(
                story,
                f"The primary flood driver is '{features[0]}' (mean |SHAP| = {values[0]:.4f}), "
                f"followed by '{second}'. {topo_note} "
                f"Features with near-zero SHAP values are redundant for this event and location.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 6. Model Performance
        perf = charts.get("model_performance", {})
        if perf:
            self._add_rule(story)
            story.append(
                Paragraph("6. Model Performance Comparison", S["section_heading"])
            )
            img = self._chart_model_perf(perf, [("f1", "F1"), ("auc", "AUC-ROC")])
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 4. RF, XGBoost, and ensemble F1 and AUC-ROC on the held-out test set.",
                        S["caption"],
                    )
                )
            auc_qual = (
                "excellent"
                if selected_auc > 0.85
                else "strong"
                if selected_auc > 0.75
                else "acceptable"
            )
            self._interpret(
                story,
                f"The selected '{model_type}' model achieved F1 = {selected_f1:.3f} and "
                f"AUC-ROC = {selected_auc:.3f} ({auc_qual} discriminative ability). "
                f"AUC-ROC measures the model's ability to separate flooded from non-flooded pixels "
                f"across all classification thresholds. "
                f"The ensemble mean probability smooths RF and XGBoost disagreements and typically "
                f"reduces both false positives and missed detections.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 7. Model Uncertainty
        uncertainty = charts.get("uncertainty", {})
        if uncertainty.get("spread_stats"):
            self._add_rule(story)
            story.append(
                Paragraph("7. Model Uncertainty (Epistemic)", S["section_heading"])
            )
            story.append(
                Paragraph(
                    "Uncertainty is estimated from the absolute spread between RF and XGBoost flood "
                    "probabilities for each pixel. Spread > 0.20 flags pixels where the two classifiers "
                    "disagree substantially — these warrant field validation before operational use.",
                    S["body"],
                )
            )
            img = self._chart_flood_uncertainty(uncertainty)
            high_pct_unc = uncertainty.get("high_uncertainty_pct", 0)
            mean_spread = uncertainty.get("mean_spread", 0)
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        f"Figure 5. RF–XGBoost probability spread statistics. "
                        f"{high_pct_unc:.1f}% of pixels exceed the 0.20 threshold.",
                        S["caption"],
                    )
                )
            unc_level = (
                "elevated"
                if high_pct_unc > 15
                else "moderate"
                if high_pct_unc > 5
                else "low"
            )
            self._interpret(
                story,
                f"Mean RF–XGBoost probability spread is {mean_spread:.4f}; {high_pct_unc:.1f}% of pixels "
                f"show {unc_level} epistemic uncertainty (spread > 0.20). "
                f"{'High uncertainty may arise from sparse SAR coverage, cloud masking, or class imbalance — field validation is strongly recommended in these areas.' if high_pct_unc > 15 else 'Model agreement is sufficient for operational flood risk mapping at this scale.'}",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 8. Key Flood Statistics table
        flooded_pct = stats.get("flooded_pct")
        if flooded_pct is not None:
            self._add_rule(story)
            story.append(Paragraph("8. Key Flood Statistics", S["section_heading"]))
            rows = [["Metric", "Value"]]
            for key, label in [
                ("flooded_pct", "Flooded pixels — JRC label (%)"),
                ("very_high_risk_pct", "Very High risk (%)"),
                ("high_risk_pct", "High risk (%)"),
                ("medium_risk_pct", "Medium risk (%)"),
                ("low_risk_pct", "Low risk (%)"),
                ("selected_auc", "Selected model AUC"),
                ("selected_f1", "Selected model F1"),
                ("selected_threshold", "Optimised threshold"),
                ("top_flood_driver", "Top driver (SHAP)"),
                ("mean_spread", "Mean RF–XGBoost spread"),
                ("high_uncertainty_pct", "High-uncertainty pixels (%)"),
            ]:
                if stats.get(key) is not None:
                    val = stats[key]
                    rows.append(
                        [label, f"{val:.3f}" if isinstance(val, float) else str(val)]
                    )
            tbl = Table(rows, colWidths=[9 * cm, 6 * cm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, LIGHT_GREY],
                        ),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 0.4 * cm))

    def _add_land_degradation_sections(
        self, story: list, output: AnalysisOutput
    ) -> None:
        S = self._styles
        charts = output.charts or {}
        stats = output.stats or {}
        degraded_pct = stats.get("degraded_label_pct", 0)
        selected_f1 = stats.get("selected_f1", 0)
        model_type = stats.get("model_type", "ensemble")

        # 4. Risk Distribution
        rd = charts.get("riskDist", {})
        if rd.get("labels") and rd.get("data"):
            self._add_rule(story)
            story.append(
                Paragraph("4. Land Degradation Classification", S["section_heading"])
            )
            story.append(
                Paragraph(
                    "Binary classification: Degraded vs. Not Degraded. "
                    "Degradation is driven by persistent NDVI decline (OLS + Mann-Kendall), "
                    "soil organic carbon loss, and rainfall erosivity.",
                    S["body"],
                )
            )
            img = self._chart_risk_dist_generic(
                rd["labels"],
                rd["data"],
                rd.get("colors", ["#27AE60", "#E74C3C"]),
                "Land Degradation Distribution",
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 2. Degraded vs. non-degraded pixel distribution.",
                        S["caption"],
                    )
                )
            severity = (
                "widespread degradation"
                if degraded_pct > 40
                else "moderate and localised degradation"
                if degraded_pct > 20
                else "low overall degradation"
            )
            action = (
                "Urgent land restoration — including reforestation, soil conservation measures, "
                "and reduced grazing pressure — is recommended."
                if degraded_pct > 40
                else "Targeted monitoring and preventive land management in degraded zones is recommended."
            )
            self._interpret(
                story,
                f"{degraded_pct:.1f}% of sampled pixels are classified as degraded, indicating "
                f"{severity} across the AOI. {action}",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 5. Annual NDVI Trend
        ts = charts.get("timeSeries", {})
        trend = charts.get("trend", {})
        if ts.get("datasets"):
            self._add_rule(story)
            story.append(Paragraph("5. Annual NDVI Trend", S["section_heading"]))
            slope = trend.get("ndvi_trend_per_year") or stats.get("ndvi_trend_per_year")
            mk_sig = trend.get("mk_significant") or stats.get("mk_significant")
            r2 = trend.get("ndvi_trend_r2") or stats.get("ndvi_trend_r2")
            if slope is not None:
                direction = "declining" if slope < 0 else "improving"
                sig_text = (
                    "Mann-Kendall significant (p < 0.05)"
                    if mk_sig
                    else "Mann-Kendall not significant"
                )
                story.append(
                    Paragraph(
                        f"NDVI trend: {slope:+.4f} per year ({direction}). {sig_text}.",
                        S["body"],
                    )
                )
            img = self._chart_ndvi_trend(ts, trend if trend else stats)
            if img:
                story.append(img)
                bkps = trend.get("breakpoint_years") or stats.get(
                    "breakpoint_years", []
                )
                caption = "Figure 3. Annual NDVI with OLS trend line"
                if bkps:
                    caption += (
                        f" and breakpoint(s) at {', '.join(str(y) for y in bkps)}"
                    )
                story.append(Paragraph(caption + ".", S["caption"]))
            if slope is not None:
                r2_text = f" (R² = {r2:.3f})" if r2 is not None else ""
                trend_meaning = (
                    "persistently declining vegetation cover consistent with ongoing land degradation"
                    if slope < 0 and mk_sig
                    else "non-significant NDVI decline — monitoring is recommended but urgent intervention may not be warranted"
                    if slope < 0 and not mk_sig
                    else "stable or recovering vegetation cover — land management practices may be effective"
                )
                bkps = trend.get("breakpoint_years") or stats.get(
                    "breakpoint_years", []
                )
                bkp_note = (
                    f" A structural break was detected at {bkps[0]}, potentially coinciding with "
                    f"a land-use change event or major drought."
                    if bkps
                    else ""
                )
                self._interpret(
                    story,
                    f"The OLS trend of {slope:+.4f} NDVI/year{r2_text} indicates "
                    f"{trend_meaning}.{bkp_note}",
                )
            story.append(Spacer(1, 0.2 * cm))

        # 6. SHAP Feature Importance
        shap_data = charts.get("shap", {})
        if shap_data.get("features"):
            self._add_rule(story)
            story.append(Paragraph("6. SHAP Feature Importance", S["section_heading"]))
            story.append(
                Paragraph(
                    "Mean absolute SHAP values from TreeExplainer on the best-performing model. "
                    "Features are ranked by contribution to the degraded/non-degraded classification.",
                    S["body"],
                )
            )
            features = shap_data["features"]
            values = shap_data.get("mean_abs_shap", [])
            img = self._chart_shap(
                shap_data, "Feature Importance (SHAP — Land Degradation Drivers)"
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        f"Figure 4. Top degradation driver: '{features[0]}' "
                        f"(mean |SHAP| = {values[0]:.4f}).",
                        S["caption"],
                    )
                )
            second = features[1] if len(features) > 1 else "N/A"
            self._interpret(
                story,
                f"Land degradation in this AOI is primarily driven by '{features[0]}' "
                f"(mean |SHAP| = {values[0]:.4f}), followed by '{second}'. "
                f"SHAP values indicate correlative importance — causal mechanisms should be "
                f"verified with ground-truth surveys. "
                f"'{features[0]}' should be prioritised in monitoring frameworks and "
                f"restoration programme design.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 7. Model Performance
        perf = (
            charts.get("model_performance", {}) if "model_performance" in charts else {}
        )
        if perf:
            self._add_rule(story)
            story.append(
                Paragraph("7. Model Performance Comparison", S["section_heading"])
            )
            img = self._chart_model_perf(
                perf, [("f1", "F1-weighted"), ("accuracy", "Accuracy")]
            )
            if img:
                story.append(img)
                story.append(
                    Paragraph(
                        "Figure 5. RF, LightGBM, and ensemble weighted F1 and accuracy "
                        "on the held-out test set.",
                        S["caption"],
                    )
                )
            perf_desc = (
                "strong"
                if selected_f1 > 0.70
                else "moderate"
                if selected_f1 > 0.55
                else "fair"
            )
            self._interpret(
                story,
                f"The '{model_type}' model achieved F1 = {selected_f1:.3f} ({perf_desc} binary "
                f"classification performance). "
                f"Binary land degradation labelling is inherently noisy — OLS and Mann-Kendall "
                f"thresholds involve subjective cutoffs, so performance metrics should be interpreted "
                f"relative to the labelling assumptions rather than as absolute accuracy.",
            )
            story.append(Spacer(1, 0.2 * cm))

        # 8. NDVI Trend Statistics table
        if trend or stats.get("ndvi_trend_per_year") is not None:
            self._add_rule(story)
            story.append(Paragraph("8. NDVI Trend Statistics", S["section_heading"]))
            src = trend if trend else stats
            trend_rows = [["Statistic", "Value"]]
            for key, label in [
                ("ndvi_trend_per_year", "NDVI slope (per year)"),
                ("ndvi_trend_r2", "R² (OLS)"),
                ("ndvi_trend_p", "p-value (OLS)"),
                ("mk_tau", "Mann-Kendall τ"),
                ("mk_p", "Mann-Kendall p"),
                ("mk_significant", "Trend significant?"),
                ("breakpoint_year", "Primary breakpoint year"),
            ]:
                v = src.get(key)
                if v is not None:
                    trend_rows.append(
                        [label, f"{v:.4f}" if isinstance(v, float) else str(v)]
                    )
            tbl = Table(trend_rows, colWidths=[9 * cm, 6 * cm])
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        (
                            "ROWBACKGROUNDS",
                            (0, 1),
                            (-1, -1),
                            [colors.white, LIGHT_GREY],
                        ),
                        ("FONTSIZE", (0, 0), (-1, -1), 9),
                        ("GRID", (0, 0), (-1, -1), 0.4, MID_GREY),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, 0.4 * cm))

    def _add_ai_section(self, story: list, interpretation: str) -> None:
        S = self._styles
        self._add_rule(story)
        story.append(
            Paragraph("Human-centered AI Interpretation", S["section_heading"])
        )
        story.append(
            Paragraph(
                "The following human-centered AI interpretation uses the analysis "
                "statistics above. It is intended to support — not replace — expert review.",
                S["body"],
            )
        )
        story.append(Spacer(1, 0.3 * cm))
        for line in interpretation.split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.2 * cm))
            elif line.startswith("•") or line.startswith("-"):
                story.append(Paragraph(self._format_ai_line(line), S["bullet"]))
            else:
                story.append(Paragraph(self._format_ai_line(line), S["body"]))

    @staticmethod
    def _format_ai_line(line: str) -> str:
        """Convert simple GPT markdown into ReportLab-safe styled text."""
        text = escape(line)
        text = re.sub(
            r"^\*\*([^*:\n]+):\*\*\s*",
            r'<b><font color="#1B2A4A">\1:</font></b> ',
            text,
        )
        text = re.sub(
            r"\*\*([^*]+)\*\*",
            r'<b><font color="#1B2A4A">\1</font></b>',
            text,
        )
        return text.replace("*", "")

    def _add_appendix(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        self._add_rule(story)
        story.append(
            Paragraph("Appendix: Data Sources & Limitations", S["section_heading"])
        )

        sources = {
            "drought": [
                "ERA5-Land (ECMWF) — monthly precipitation and 2m temperature via GEE",
                "MODIS MOD13A3 — monthly 1km NDVI via GEE",
                "drought-monitoring v0.1.7 — CDI computation (PDI, TDI, VDI)",
            ],
            "flood": [
                "SRTM DEM (NASA) — terrain elevation and slope",
                "Sentinel-1 SAR — surface water extent via GEE",
                "CHIRPS v2.0 — daily rainfall",
            ],
            "food_security": [
                "MODIS MOD13A3 — 1 km monthly NDVI for VCI, TCI, and NDVI slope (vci, tci, ndvi_slope)",
                "CHIRPS v2.0 — pixel-wise rainfall anomaly vs long-term baseline (rainfall_anom_pct)",
                "Sentinel-2 MNDWI — surface / standing water availability (mndwi)",
                "USGS SRTM 30 m — terrain slope (slope_terrain)",
                "ESA WorldCover 2021 — land-use class (land_cover)",
            ],
            "disease": [
                "CHIRPS v2.0 — 28-day cumulative rainfall (rainfall_4w)",
                "MODIS Terra LST (MOD11A2) — daytime land surface temperature (temp_mean)",
                "Sentinel-2 MNDWI — surface water availability (ndwi)",
                "USGS SRTM 30 m — elevation",
                "WorldPop GP 100 m — log(1 + population density)",
                "MODIS MOD13A3 — monthly NDVI (ndvi)",
                "ESA WorldCover 2021 — land cover class (land_cover)",
            ],
            "land_degradation": [
                "MODIS MOD13A1 — 500m NDVI time series",
                "SoilGrids v2 — soil organic carbon",
                "CHIRPS v2.0 — rainfall erosivity",
            ],
        }

        for src in sources.get(output.module, ["See PLAN.md for module data sources."]):
            story.append(Paragraph(f"• {src}", S["bullet"]))

        story.append(Spacer(1, 0.4 * cm))
        story.append(Paragraph("<b>Known limitations:</b>", S["body"]))
        limitations = (
            [
                "Model is retrained for each AOI — results are not globally comparable without re-running.",
                "GEE data latency: ERA5-Land and MODIS typically lag 2–4 weeks behind real time.",
                "CDI weights (PDI 50%, TDI 25%, VDI 25%) follow Burchardt et al. — regional recalibration "
                "may improve accuracy for specific agro-ecological zones.",
                "LSTM forecast uncertainty is epistemic only (model uncertainty via MC Dropout); "
                "data uncertainty and structural model error are not quantified.",
            ]
            if output.module == "drought"
            else [
                "Model is retrained per AOI — results may vary with area and date range.",
                "GEE data is subject to cloud cover, orbit gaps, and latency.",
                "SHAP values indicate correlation, not causation.",
            ]
        )
        for lim in limitations:
            story.append(Paragraph(f"• {lim}", S["bullet"]))

        story.append(Spacer(1, 0.3 * cm))
        story.append(
            Paragraph(
                "Report generated by climate_change v1.0.0 — "
                "Africa Research & Impact Network (ARIN) · arin-africa.org",
                S["caption"],
            )
        )

    def _add_glossary(self, story: list, output: AnalysisOutput) -> None:
        S = self._styles
        glossary_items = [
            ("AOI", "Area of Interest", "Spatial boundary used for the analysis."),
            (
                "GEE",
                "Google Earth Engine",
                "Satellite data and raster processing platform used by the notebooks.",
            ),
            (
                "API",
                "Application Programming Interface",
                "Backend endpoints used by the frontend and report workflow.",
            ),
            (
                "TOC",
                "Table of Contents",
                "Quick navigation for the report and documentation tab.",
            ),
            (
                "NDVI",
                "Normalized Difference Vegetation Index",
                "Vegetation greenness and crop vigor indicator.",
            ),
            (
                "EVI",
                "Enhanced Vegetation Index",
                "Vegetation index that reduces saturation in dense canopies.",
            ),
            (
                "LST",
                "Land Surface Temperature",
                "Surface temperature derived from MODIS thermal observations.",
            ),
            (
                "CHIRPS",
                "Climate Hazards Group InfraRed Precipitation with Station data",
                "Rainfall estimates used for climate risk modeling.",
            ),
            (
                "SRTM",
                "Shuttle Radar Topography Mission",
                "Elevation and terrain source used in hydrology and erosion features.",
            ),
            (
                "MNDWI",
                "Modified Normalized Difference Water Index",
                "Open-water detection index used in flood mapping.",
            ),
            (
                "NDWI",
                "Normalized Difference Water Index",
                "Water and wet-surface indicator used in disease and moisture analysis.",
            ),
            (
                "TWI",
                "Topographic Wetness Index",
                "Terrain-derived wetness proxy used in flood modeling.",
            ),
            (
                "BSI",
                "Bare Soil Index",
                "Exposed-soil indicator used in land degradation assessment.",
            ),
            (
                "NDTI",
                "Normalized Difference Tillage Index",
                "Soil disturbance indicator used in land degradation analysis.",
            ),
            (
                "CDI",
                "Composite Drought Index",
                "Combined drought signal used in drought reporting.",
            ),
            (
                "vv_change",
                "Change in VV radar backscatter",
                "SAR change feature used to identify inundation and moisture shifts.",
            ),
        ]

        module_items = {
            "drought": [
                ("PDI", "Precipitation Deficit Index", "Measures rainfall shortfall."),
                (
                    "TDI",
                    "Temperature Drought Index",
                    "Captures heat-driven drought stress.",
                ),
                (
                    "VDI",
                    "Vegetation Drought Index",
                    "Tracks vegetation stress from moisture limitation.",
                ),
            ],
            "flood": [
                (
                    "rain_7d",
                    "7-day cumulative rainfall",
                    "Short rainfall window used to capture recent saturation.",
                ),
                (
                    "rain_30d",
                    "30-day cumulative rainfall",
                    "Longer rainfall window used to capture seasonal wetness.",
                ),
                (
                    "dist_river",
                    "Distance to river",
                    "Proximity-to-river feature used in flood exposure analysis.",
                ),
                (
                    "elevation",
                    "Height above sea level",
                    "Terrain height feature used to estimate runoff and exposure.",
                ),
            ],
            "food_security": [
                (
                    "NDVI",
                    "Normalized Difference Vegetation Index",
                    "Primary vegetation vigor index used in the food-security model.",
                ),
                (
                    "EVI",
                    "Enhanced Vegetation Index",
                    "Adds canopy-sensitive vegetation information.",
                ),
                (
                    "WorldCover",
                    "ESA WorldCover land cover product",
                    "Land cover context used for crop and land-use interpretation.",
                ),
            ],
            "disease": [
                (
                    "rainfall_4w",
                    "4-week cumulative rainfall",
                    "Rainfall window used to screen disease suitability.",
                ),
                (
                    "temp_mean",
                    "Mean temperature",
                    "Average temperature feature used in disease suitability modeling.",
                ),
                (
                    "pop_density",
                    "Population density",
                    "People-per-area feature used to estimate exposure.",
                ),
                (
                    "land_cover",
                    "Land cover class",
                    "Surface class used to add environmental context.",
                ),
            ],
            "land_degradation": [
                (
                    "ndvi_slope",
                    "NDVI trend slope",
                    "Vegetation trend feature used to detect decline or recovery.",
                ),
                (
                    "ndvi_mean",
                    "Average NDVI",
                    "Baseline vegetation productivity feature.",
                ),
                (
                    "ndvi_cv",
                    "NDVI coefficient of variation",
                    "Vegetation instability feature across time.",
                ),
                (
                    "slope_terrain",
                    "Terrain slope feature",
                    "Erosion-risk terrain input used by the model.",
                ),
                (
                    "rainfall_anom",
                    "Rainfall anomaly",
                    "Departure from normal rainfall used to explain degradation.",
                ),
            ],
        }

        story.append(Spacer(1, 0.15 * cm))
        self._add_rule(story)
        story.append(Paragraph("Glossary", S["section_heading"]))
        story.append(
            Paragraph(
                "The glossary below covers the shared project vocabulary and the module-specific indices used in this report.",
                S["body"],
            )
        )

        header_style = ParagraphStyle(
            "glossary_header",
            parent=S["body"],
            fontName="Helvetica-Bold",
            fontSize=7.6,
            leading=9,
            textColor=colors.white,
            alignment=0,
        )
        cell_style = ParagraphStyle(
            "glossary_cell",
            parent=S["body"],
            fontSize=7.4,
            leading=9,
            alignment=0,
            wordWrap="CJK",
            spaceAfter=0,
        )

        def cell(value: str) -> Paragraph:
            return Paragraph(escape(str(value)), cell_style)

        rows = [
            [Paragraph(label, header_style) for label in ("Term", "Meaning", "Purpose")]
        ]
        rows.extend(
            [
                [cell(term), cell(meaning), cell(purpose)]
                for term, meaning, purpose in glossary_items
            ]
        )
        rows.extend(
            [
                [cell(term), cell(meaning), cell(purpose)]
                for term, meaning, purpose in module_items.get(output.module, [])
            ]
        )

        tbl = Table(rows, colWidths=[2.15 * cm, 5.65 * cm, 7.75 * cm], repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("GRID", (0, 0), (-1, -1), 0.35, MID_GREY),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_GREY]),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 4),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                    ("TOPPADDING", (0, 0), (-1, -1), 3),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ]
            )
        )
        story.append(tbl)

    @staticmethod
    def _add_rule(story: list) -> None:
        story.append(Spacer(1, 0.3 * cm))
        story.append(HRFlowable(width="100%", thickness=1, color=TEAL))
        story.append(Spacer(1, 0.2 * cm))
