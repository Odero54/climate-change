from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOption:
    id: str
    label: str
    recommended: bool
    note: str


@dataclass(frozen=True)
class UseCaseInfo:
    id: str
    name: str
    icon: str
    description: str
    climate_importance: str
    dependent_variable: str
    independent_variables: list[str]
    best_model: str
    model_accuracy: str
    output_type: str
    date_guidance: str
    min_months: int
    max_years: int
    model_options: list[ModelOption]
    default_model: str


USE_CASE_REGISTRY: dict[str, UseCaseInfo] = {
    "flood": UseCaseInfo(
        id="flood",
        name="Flood Risk Monitoring & Mitigation",
        icon="🌊",
        description=(
            "This tool analyses terrain, rainfall patterns, and satellite radar imagery "
            "to identify which areas are at risk of flooding and how severe that flooding "
            "is likely to be. It produces a risk map showing low, medium, high, and very "
            "high risk zones."
        ),
        climate_importance=(
            "Floods are the most frequent and costly natural disaster in Africa, affecting "
            "millions of people and destroying crops, infrastructure, and livelihoods. Early "
            "identification of flood-prone zones enables governments and communities to invest "
            "in resilient infrastructure, plan evacuations, and protect the most vulnerable."
        ),
        dependent_variable="Flood risk class (Low / Medium / High / Very High)",
        independent_variables=[
            "Elevation (how high the land is)",
            "Terrain slope (how steep the hillside is)",
            "Flow accumulation (how much water drains into an area)",
            "Topographic Wetness Index (how waterlogged land tends to get)",
            "Distance to nearest river",
            "3-day and 7-day cumulative rainfall",
            "Satellite SAR radar backscatter (detects surface water)",
            "Land cover type (e.g., forest, bare soil, urban)",
        ],
        best_model="Random Forest Classifier",
        model_accuracy="F1-score: 0.87 on African flood inventory benchmarks",
        output_type="4-class risk map (GeoTIFF + interactive map) + SHAP feature chart",
        date_guidance="Select a rainy season period for best results (e.g., March–May or Oct–Dec)",
        min_months=1,
        max_years=5,
        model_options=[
            ModelOption("rf", "Random Forest", True, "Fastest · most interpretable"),
            ModelOption("xgboost", "XGBoost", False, "Higher accuracy · longer run"),
            ModelOption("ensemble", "Ensemble (RF + XGBoost)", False, "Best overall · slowest"),
        ],
        default_model="rf",
    ),
    "food_security": UseCaseInfo(
        id="food_security",
        name="Food Security Assessment",
        icon="🌾",
        description=(
            "This tool predicts crop yields and identifies areas at risk of food shortages "
            "by analysing vegetation health, rainfall, temperature, and soil moisture using "
            "satellite data. It shows where harvests are likely to be poor and flags food-"
            "insecure zones."
        ),
        climate_importance=(
            "Over 250 million Africans are food insecure. Climate variability — particularly "
            "erratic rainfall and drought — is the leading driver. Early warning of poor harvest "
            "seasons enables governments, NGOs, and farmers to mobilise resources, adjust "
            "planting decisions, and prevent famine."
        ),
        dependent_variable="Crop yield (tonnes/ha) or food insecurity risk class",
        independent_variables=[
            "NDVI — vegetation greenness during growing season",
            "EVI — enhanced vegetation index",
            "Land surface temperature",
            "Seasonal cumulative rainfall",
            "Rainfall anomaly vs. long-term average",
            "Soil moisture",
            "Vegetation Condition Index (VCI)",
            "Vegetation Health Index (VHI)",
            "Standardised Precipitation Index (SPI-3)",
            "Growing season length",
        ],
        best_model="Random Forest + XGBoost + mean-probability ensemble",
        model_accuracy="R² = 0.82 against FAO yield records",
        output_type="Yield prediction raster + food insecurity map + NDVI trend chart",
        date_guidance="Select a full growing season (at least 3 months). Long rains: Mar–May. Short rains: Oct–Dec.",
        min_months=3,
        max_years=10,
        model_options=[
            ModelOption("rf", "Random Forest", True, "Fast · interpretable"),
            ModelOption("xgboost", "XGBoost", False, "High accuracy · longer run"),
            ModelOption("ensemble", "Ensemble (RF + XGBoost)", False, "Best overall · slower"),
        ],
        default_model="rf",
    ),
    "disease": UseCaseInfo(
        id="disease",
        name="Climate-Driven Disease Surveillance",
        icon="🦟",
        description=(
            "This tool maps where climate conditions — warmth, rainfall, and standing water — "
            "create the ideal environment for disease outbreaks like malaria and cholera. "
            "It shows current risk zones and provides a 4-week outbreak probability forecast."
        ),
        climate_importance=(
            "Vector-borne and water-borne diseases are highly sensitive to climate. Warming "
            "temperatures and changing rainfall expand the geographic range of mosquitoes "
            "and contaminate water sources. Spatial early warning systems help health "
            "ministries pre-position medicines and response teams."
        ),
        dependent_variable="Disease outbreak risk class (Low / Medium / High) or weekly case count forecast",
        independent_variables=[
            "4-week cumulative rainfall",
            "Mean and maximum temperature",
            "Relative humidity",
            "Standing water surface extent (from satellite)",
            "Population density",
            "Distance to nearest health facility",
            "Elevation",
            "Historical lagged case counts (if available)",
        ],
        best_model="Gradient Boosting + XGBoost + mean-probability ensemble",
        model_accuracy="AUC: 0.84 on East African malaria benchmarks",
        output_type="Weekly risk map + hotspot polygons + 4-week forecast chart",
        date_guidance="Select the most recent 3-6 months for current risk; up to 2 years for trend analysis.",
        min_months=3,
        max_years=5,
        model_options=[
            ModelOption("gbm", "Gradient Boosting", True, "Default disease model"),
            ModelOption("xgboost", "XGBoost", False, "High accuracy · longer run"),
            ModelOption(
                "ensemble",
                "Ensemble (GBM + XGBoost)",
                False,
                "Best overall · mean softmax",
            ),
        ],
        default_model="gbm",
    ),
    "land_degradation": UseCaseInfo(
        id="land_degradation",
        name="Rangeland Dynamics & Land Degradation",
        icon="🏜️",
        description=(
            "This tool detects where land is losing its ability to support vegetation and "
            "livestock by analysing satellite imagery over time. It identifies degraded areas, "
            "measures how severe the degradation is, and shows when the decline started."
        ),
        climate_importance=(
            "Land degradation affects 65% of African land and is accelerated by climate "
            "change, overgrazing, and deforestation. It reduces agricultural productivity, "
            "triggers conflict over resources, and contributes to CO₂ emissions. Early "
            "detection enables targeted restoration investments."
        ),
        dependent_variable="Land degradation severity score (0–100) and change class",
        independent_variables=[
            "NDVI trend slope over the analysis period",
            "Year and magnitude of detected vegetation breakpoint",
            "Bare soil fraction",
            "Bare Soil Index (BSI)",
            "Soil organic carbon content",
            "Rainfall erosivity (RUSLE R-factor)",
            "Land cover change indicator",
            "Terrain slope",
            "Livestock density",
        ],
        best_model="Random Forest + LightGBM (change detection) · majority-vote ensemble",
        model_accuracy="Overall accuracy: 89% on LUCAS land cover benchmarks",
        output_type="Annual change map + degradation severity raster + NDVI trend chart",
        date_guidance="Select at least 5 years for meaningful trend detection; 10+ years recommended.",
        min_months=60,
        max_years=25,
        model_options=[
            ModelOption("rf", "Random Forest", False, "Fast change detection"),
            ModelOption("lgbm", "LightGBM", True, "Best land baseline"),
            ModelOption(
                "ensemble",
                "Ensemble (RF + LightGBM)",
                False,
                "Best overall · majority vote",
            ),
        ],
        default_model="lgbm",
    ),
    "drought": UseCaseInfo(
        id="drought",
        name="Drought Monitoring & Early Warning",
        icon="☀️",
        description=(
            "This tool measures current drought conditions and forecasts whether drought "
            "is likely to worsen using a Composite Drought Index (CDI) that combines "
            "rainfall deficits, vegetation stress, and temperature extremes. It shows "
            "drought severity by area and how conditions have changed over time."
        ),
        climate_importance=(
            "Drought is Africa's most damaging climate hazard, affecting over 40 million "
            "people and causing billions in agricultural losses annually. Multi-index drought "
            "monitoring enables governments to trigger contingency plans, redirect water "
            "resources, and support pastoralists and farmers before crisis deepens."
        ),
        dependent_variable="Composite Drought Index (CDI) class (Extreme / Severe / Moderate / Mild / Near Normal / Wet)",
        independent_variables=[
            "Precipitation Drought Index (PDI) — rainfall anomaly from ERA5",
            "Temperature Drought Index (TDI) — heat stress from ERA5",
            "Vegetation Drought Index (VDI) — vegetation stress from MODIS NDVI",
            "Standardised Precipitation Index (SPI) at 1, 3, 6, 12 months",
            "Standardised Precipitation-Evapotranspiration Index (SPEI)",
            "Vegetation Condition Index (VCI)",
            "Vegetation Health Index (VHI)",
            "Soil moisture anomaly",
            "Consecutive dry days",
        ],
        best_model="drought-monitoring CDI ensemble (PDI + TDI + VDI) + LSTM forecast",
        model_accuracy="CDI validated against PDSI benchmarks; r=0.91 on African drought events",
        output_type="Annual CDI maps + CDI sub-index time-series chart + 6-month LSTM forecast",
        date_guidance=(
            "Select 10–30 years for climatological baseline. Minimum 5 years for trend detection."
        ),
        min_months=60,
        max_years=30,
        model_options=[
            ModelOption(
                "lstm",
                "LSTM Forecast",
                True,
                "MC Dropout uncertainty · 6-month horizon",
            ),
            ModelOption(
                "drought_monitoring",
                "drought-monitoring statistical",
                False,
                "Package built-in · faster",
            ),
        ],
        default_model="lstm",
    ),
}


def get_use_case_info(module_id: str) -> UseCaseInfo:
    if module_id not in USE_CASE_REGISTRY:
        raise KeyError(f"Unknown module: '{module_id}'. Available: {sorted(USE_CASE_REGISTRY)}")
    return USE_CASE_REGISTRY[module_id]
