"""
AI result interpretation using OpenAI GPT-4o.
User supplies their own OPENAI_API_KEY — ARIN stores nothing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from openai import OpenAI, OpenAIError
from openai.types.chat import ChatCompletionMessageParam

from climate_change.registry import USE_CASE_REGISTRY

if TYPE_CHECKING:
    from climate_change.core.base_use_case import AnalysisOutput

# ── Module-level context strings ──────────────────────────────────────────────

_CDI_SCALE = """
CDI (Composite Drought Index) is a ratio index — lower means more drought:
  < 0.50  → Extreme Drought
  0.50–0.65 → Severe Drought
  0.65–0.80 → Moderate Drought
  0.80–0.90 → Mild Drought
  0.90–1.10 → Near Normal
  > 1.10  → Wet conditions
PDI = Precipitation Drought Index (ERA5 rainfall anomaly)
TDI = Temperature Drought Index (ERA5 heat stress anomaly)
VDI = Vegetation Drought Index (MODIS NDVI anomaly)
CDI = weighted combination (PDI×0.50 + TDI×0.25 + VDI×0.25)
"""

_FLOOD_CONTEXT = """
Risk classes derived from Random Forest / XGBoost flood probability:
  Low      (prob < 0.25) — terrain and rainfall unlikely to produce inundation
  Medium   (0.25–0.50)  — moderate susceptibility; monitor during heavy rain
  High     (0.50–0.75)  — likely to flood given the observed conditions
  Very High (≥ 0.75)   — high inundation probability; evacuation/intervention needed
Features: elevation, slope, TWI, SAR backscatter change, 7-day and 30-day rainfall,
MNDWI (surface water), distance to river, land cover.
SHAP values indicate which features drove the risk score for each pixel.
Model uncertainty = spread between RF and XGBoost probability estimates.
"""

_LAND_DEGRADATION_CONTEXT = """
Land degradation is assessed by a Random Forest / LightGBM classifier trained on:
  ndvi_slope    — direction and rate of vegetation change over the period
  ndvi_mean     — average greenness (low = sparse vegetation)
  ndvi_cv       — variability (high = unstable cover)
  bsi           — Bare Soil Index (high = exposed soil)
  ndti          — Normalised Difference Tillage Index
  slope_terrain — terrain steepness (erosion risk)
  rainfall_anom — rainfall departure from long-term mean
  land_cover    — ESA land-use class
Degradation class: 0 = Not Degraded, 1 = Degraded (threshold at 70th-percentile score).
NDVI trend (Mann-Kendall test) confirms whether change is statistically significant.
Breakpoints mark years when the NDVI time series experienced a structural shift.
"""

_FOOD_SECURITY_CONTEXT = """
Food security risk is assessed by a Random Forest / XGBoost classifier trained on:
  vci               — Vegetation Condition Index (MODIS NDVI; high = better vegetation condition)
  tci               — Temperature Condition Index (MODIS LST; high = cooler, less heat stress)
  rainfall_anom_pct — CHIRPS pixel-wise rainfall anomaly vs long-term baseline (2001–present), as %
  ndvi_slope        — pixel-wise MODIS NDVI linear trend (NDVI units per year; negative = declining)
  mndwi             — Sentinel-2 Modified NDWI — surface / standing water availability
  slope_terrain     — SRTM terrain slope in degrees (proxy for runoff and erosion risk)
  land_cover        — ESA WorldCover 2021 land-use class, normalised to [0, 1]
Risk classes: Low / Medium / High assigned by tercile thresholds on a composite stress score
(0.40 × VCI stress + 0.25 × TCI stress + 0.20 × rainfall deficit + 0.15 × inverted NDVI slope).
A declining NDVI slope combined with low VCI is the primary indicator of acute food insecurity.
"""

_DISEASE_CONTEXT = """
Disease outbreak risk (malaria-proxy) is assessed by a GBM / XGBoost ensemble classifier trained on:
  rainfall_4w   — CHIRPS 28-day cumulative rainfall (mm) ending on the study end date
  temp_mean     — MODIS Terra daytime Land Surface Temperature (°C), period mean
  ndwi          — Sentinel-2 MNDWI — surface / standing water availability
  elevation     — USGS SRTM 30 m elevation (m)
  pop_density   — WorldPop GP 100 m log(1 + population density)
  ndvi          — MODIS MOD13A3 monthly NDVI mean (vegetation / vector-habitat proxy)
  land_cover    — ESA WorldCover 2021 land-use class, normalised to [0, 1]
Risk classes: Low / Medium / High assigned by tercile thresholds on a composite suitability score
(0.40 × temperature suitability + 0.35 × rainfall suitability + 0.25 × NDWI score).
Hotspots are identified by DBSCAN spatial clustering of High-Risk pixel centroids.
"""

_MODULE_CONTEXT: dict[str, str] = {
    "drought": _CDI_SCALE,
    "flood": _FLOOD_CONTEXT,
    "land_degradation": _LAND_DEGRADATION_CONTEXT,
    "food_security": _FOOD_SECURITY_CONTEXT,
    "disease": _DISEASE_CONTEXT,
}


# Base prompt builder (all modules)


def build_interpretation_prompt(output: AnalysisOutput) -> str:
    info = USE_CASE_REGISTRY[output.module]
    module_ctx = _MODULE_CONTEXT.get(output.module, "")
    stats_str = "\n".join(f"  {k}: {v}" for k, v in output.stats.items())

    return f"""
You are a senior climate scientist interpreting results from an AI-powered Decision
Support System developed by the Africa Research & Impact Network (ARIN).

MODULE: {info.name}
COUNTRY / REGION: {output.metadata.get("country", "unknown")}
ANALYSIS PERIOD: {output.metadata.get("start_date", "?")} to {output.metadata.get("end_date", "?")}
MODEL USED: {output.metadata.get("model", info.best_model)}

MODULE CONTEXT:
{module_ctx}

WHAT THIS TOOL DOES:
{info.description}

WHAT WAS PREDICTED: {info.dependent_variable}

KEY FINDINGS (statistical summary):
{stats_str}

Please provide ALL of the following, labelled exactly as shown:
1. SUMMARY (3-4 sentences for a non-technical policymaker)
2. KEY DRIVERS (3 bullet points on the dominant risk factors or patterns)
3. RECOMMENDATIONS (3 specific, actionable adaptation actions for the region)
4. CAVEATS (data quality issues or model limitations the user should know)

Respond in clear, professional English. Avoid jargon. Be concise.
""".strip()


# Module-specific prompt builders


def build_drought_prompt(output: AnalysisOutput) -> str:
    """Appends CDI forecast values and severity distribution to the base prompt."""
    base = build_interpretation_prompt(output)
    charts = output.charts or {}

    forecast = charts.get("forecast", {})
    forecast_note = ""
    if forecast.get("mean"):
        fc_vals = forecast["mean"]
        fc_dates = forecast.get("dates", [])
        pairs = ", ".join(f"{d}: {v:.2f}" for d, v in zip(fc_dates[:6], fc_vals[:6]))
        forecast_note = f"\nFORECAST (next {len(fc_vals)} months): {pairs}"

    severity = charts.get("severity_distribution", {})
    sev_note = ""
    if severity.get("labels"):
        sev_note = "\nSEVERITY DISTRIBUTION: " + ", ".join(
            f"{lbl}: {pct}%" for lbl, pct in zip(severity["labels"], severity["data"])
        )

    return base + forecast_note + sev_note


def build_flood_prompt(output: AnalysisOutput) -> str:
    """Appends flood risk distribution, top driver, and model uncertainty to the base prompt."""
    base = build_interpretation_prompt(output)
    charts = output.charts or {}
    stats = output.stats or {}

    # Risk class breakdown
    risk_dist = charts.get("risk_distribution", {})
    risk_note = ""
    if risk_dist.get("labels") and risk_dist.get("data"):
        risk_note = "\nRISK CLASS DISTRIBUTION: " + ", ".join(
            f"{lbl}: {pct:.1f}%" for lbl, pct in zip(risk_dist["labels"], risk_dist["data"])
        )

    # Top SHAP driver
    shap = charts.get("shap", {})
    driver_note = ""
    if shap.get("features"):
        driver_note = f"\nTOP FLOOD DRIVER: {shap['features'][0]}"
        if shap.get("mean_abs_shap"):
            top3 = list(zip(shap["features"][:3], shap["mean_abs_shap"][:3]))
            driver_note += " | Top-3 SHAP: " + ", ".join(f"{f} ({v:.3f})" for f, v in top3)

    # Model uncertainty
    uncertainty_note = ""
    mean_spread = stats.get("mean_spread")
    high_unc_pct = stats.get("high_uncertainty_pct")
    if mean_spread is not None:
        uncertainty_note = f"\nMODEL UNCERTAINTY: mean RF–XGBoost spread = {mean_spread:.3f}"
        if high_unc_pct is not None:
            uncertainty_note += f"; {high_unc_pct:.1f}% of pixels are high-uncertainty"

    # Selected model performance
    perf_note = ""
    model_type = stats.get("model_type", "ensemble")
    selected_f1 = stats.get("selected_f1")
    selected_auc = stats.get("selected_auc")
    if selected_f1 is not None:
        perf_note = f"\nSELECTED MODEL ({model_type}): F1 = {selected_f1:.3f}" + (
            f", AUC = {selected_auc:.3f}" if selected_auc is not None else ""
        )

    return base + risk_note + driver_note + uncertainty_note + perf_note


def build_land_degradation_prompt(output: AnalysisOutput) -> str:
    """Appends degradation distribution, NDVI trend statistics, and top driver to the base prompt."""
    base = build_interpretation_prompt(output)
    charts = output.charts or {}
    stats = output.stats or {}

    # Degradation class breakdown
    risk_dist = charts.get("riskDist", {})
    dist_note = ""
    if risk_dist.get("labels") and risk_dist.get("data"):
        dist_note = "\nDEGRADATION DISTRIBUTION: " + ", ".join(
            f"{lbl}: {pct:.1f}%" for lbl, pct in zip(risk_dist["labels"], risk_dist["data"])
        )

    # NDVI trend
    trend = charts.get("trend", {})
    trend_note = ""
    slope = trend.get("ndvi_trend_per_year") or stats.get("ndvi_trend_per_year")
    mk_sig = trend.get("mk_significant") or stats.get("mk_significant")
    if slope is not None:
        direction = "declining" if slope < 0 else "improving"
        significance = "statistically significant" if mk_sig else "not statistically significant"
        trend_note = f"\nNDVI TREND: {slope:+.4f} NDVI units/year ({direction}; {significance})"

    # Breakpoint years
    bkp_years = trend.get("breakpoint_years") or stats.get("breakpoint_years", [])
    bkp_note = ""
    if bkp_years:
        bkp_note = f"\nBREAKPOINT YEARS: {', '.join(str(y) for y in bkp_years)}"

    # Top SHAP driver
    shap = charts.get("shap", {})
    driver_note = ""
    if shap.get("features"):
        driver_note = f"\nTOP DEGRADATION DRIVER: {shap['features'][0]}"
        if shap.get("mean_abs_shap"):
            top3 = list(zip(shap["features"][:3], shap["mean_abs_shap"][:3]))
            driver_note += " | Top-3 SHAP: " + ", ".join(f"{f} ({v:.3f})" for f, v in top3)

    # Selected model performance
    perf_note = ""
    model_type = stats.get("model_type", "lgbm")
    selected_f1 = stats.get("selected_f1")
    if selected_f1 is not None:
        perf_note = f"\nSELECTED MODEL ({model_type}): F1 = {selected_f1:.3f}"

    return base + dist_note + trend_note + bkp_note + driver_note + perf_note


def build_disease_prompt(output: AnalysisOutput) -> str:
    """Appends risk distribution, hotspot cluster info, and model performance to the base prompt."""
    base = build_interpretation_prompt(output)
    charts = output.charts or {}
    stats = output.stats or {}

    risk_dist = charts.get("riskDist", {})
    risk_note = ""
    if risk_dist.get("labels") and risk_dist.get("data"):
        risk_note = "\nRISK CLASS DISTRIBUTION: " + ", ".join(
            f"{lbl}: {pct:.1f}%" for lbl, pct in zip(risk_dist["labels"], risk_dist["data"])
        )

    cluster_note = ""
    n_clusters = stats.get("n_hotspot_clusters")
    if n_clusters is not None:
        cluster_note = f"\nHOTSPOT CLUSTERS (DBSCAN): {n_clusters} clusters identified"
        hotspot_pop = stats.get("hotspot_population")
        if hotspot_pop is not None:
            cluster_note += f"; estimated population at risk: {hotspot_pop:,.0f}"

    shap = charts.get("shap", {})
    driver_note = ""
    if shap.get("features"):
        driver_note = f"\nTOP DISEASE DRIVER: {shap['features'][0]}"
        if shap.get("mean_abs_shap"):
            top3 = list(zip(shap["features"][:3], shap["mean_abs_shap"][:3]))
            driver_note += " | Top-3 SHAP: " + ", ".join(f"{f} ({v:.3f})" for f, v in top3)

    perf_note = ""
    model_type = stats.get("model_type", "gbm")
    selected_f1 = stats.get("selected_f1")
    selected_auc = stats.get("selected_auc")
    if selected_f1 is not None:
        perf_note = f"\nSELECTED MODEL ({model_type}): F1 = {selected_f1:.3f}" + (
            f", AUC = {selected_auc:.3f}" if selected_auc is not None else ""
        )

    return base + risk_note + cluster_note + driver_note + perf_note


def build_food_security_prompt(output: AnalysisOutput) -> str:
    """Appends VCI/TCI/VHI indices, risk distribution, NDVI trend, and top driver to the base prompt."""
    base = build_interpretation_prompt(output)
    charts = output.charts or {}
    stats = output.stats or {}

    index_note = ""
    vci = stats.get("vci_mean")
    tci = stats.get("tci_mean")
    vhi = stats.get("vhi_mean")
    if any(v is not None for v in [vci, tci, vhi]):
        parts = []
        if vci is not None:
            parts.append(f"VCI = {vci:.2f}")
        if tci is not None:
            parts.append(f"TCI = {tci:.2f}")
        if vhi is not None:
            parts.append(f"VHI = {vhi:.2f}")
        index_note = "\nVEGETATION HEALTH INDICES (area mean): " + ", ".join(parts)
        index_note += "\n(VHI = 0.5×VCI + 0.5×TCI; < 0.35 indicates food insecurity stress)"

    risk_dist = charts.get("riskDist", {})
    risk_note = ""
    if risk_dist.get("labels") and risk_dist.get("data"):
        risk_note = "\nRISK CLASS DISTRIBUTION: " + ", ".join(
            f"{lbl}: {pct:.1f}%" for lbl, pct in zip(risk_dist["labels"], risk_dist["data"])
        )

    trend_note = ""
    ndvi_slope = stats.get("ndvi_slope_mean") or stats.get("ndvi_trend_per_year")
    if ndvi_slope is not None:
        direction = "declining" if ndvi_slope < 0 else "improving"
        trend_note = f"\nNDVI TREND: {ndvi_slope:+.4f} NDVI units/year ({direction})"

    shap = charts.get("shap", {})
    driver_note = ""
    if shap.get("features"):
        driver_note = f"\nTOP FOOD SECURITY DRIVER: {shap['features'][0]}"
        if shap.get("mean_abs_shap"):
            top3 = list(zip(shap["features"][:3], shap["mean_abs_shap"][:3]))
            driver_note += " | Top-3 SHAP: " + ", ".join(f"{f} ({v:.3f})" for f, v in top3)

    perf_note = ""
    model_type = stats.get("model_type", "rf")
    selected_f1 = stats.get("selected_f1")
    selected_auc = stats.get("selected_auc")
    if selected_f1 is not None:
        perf_note = f"\nSELECTED MODEL ({model_type}): F1 = {selected_f1:.3f}" + (
            f", AUC = {selected_auc:.3f}" if selected_auc is not None else ""
        )

    return base + index_note + risk_note + trend_note + driver_note + perf_note


# Dispatch helper

_PROMPT_BUILDERS = {
    "drought": build_drought_prompt,
    "flood": build_flood_prompt,
    "land_degradation": build_land_degradation_prompt,
    "disease": build_disease_prompt,
    "food_security": build_food_security_prompt,
}


def build_prompt(output: AnalysisOutput) -> str:
    """Return the richest prompt available for the given module."""
    builder = _PROMPT_BUILDERS.get(output.module, build_interpretation_prompt)
    return builder(output)


# AIInterpreter
class AIInterpreter:
    """
    Stateless GPT-4o interpreter. Instantiated per request with user's API key.
    The key is never logged or stored beyond the request lifecycle.
    """

    def __init__(self, api_key: str) -> None:
        self._client = OpenAI(api_key=api_key)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(client=<redacted>)"

    def __str__(self) -> str:
        return repr(self)

    def interpret(self, output: AnalysisOutput) -> str:
        """Generate a one-shot interpretation of the analysis results."""
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a climate adaptation expert working for ARIN. "
                            "You interpret geospatial analysis results and provide "
                            "actionable, evidence-based recommendations."
                        ),
                    },
                    {"role": "user", "content": build_prompt(output)},
                ],
                temperature=0.3,
                max_tokens=900,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI interpretation failed: {exc}") from exc
        return response.choices[0].message.content or ""

    def chat(
        self,
        output: AnalysisOutput,
        history: list[dict],
        user_message: str,
    ) -> str:
        """Follow-up questions after initial interpretation."""
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a climate adaptation expert discussing analysis results "
                    "from ARIN's Decision Support System. Answer questions about the "
                    "results clearly and concisely. Refer back to the findings when relevant."
                ),
            },
            {"role": "user", "content": build_prompt(output)},
            {"role": "assistant", "content": "I have reviewed the analysis results."},
            *history,
            {"role": "user", "content": user_message},
        ]
        try:
            response = self._client.chat.completions.create(
                model="gpt-4o",
                messages=cast(list[ChatCompletionMessageParam], messages),
                temperature=0.4,
                max_tokens=600,
            )
        except OpenAIError as exc:
            raise RuntimeError(f"OpenAI chat failed: {exc}") from exc
        return response.choices[0].message.content or ""
