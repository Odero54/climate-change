# MESI (Malaria Environmental Suitability Index) scale reference:
#   < 30   — Low environmental suitability
#   30–60  — Moderate suitability
#   > 60   — High suitability (perennial transmission zone)
#
# Disease risk classes (tercile thresholds on composite risk score):
#   0 = Low Risk   (score < 33rd pct)
#   1 = Medium Risk (33rd–66th pct)
#   2 = High Risk  (score > 66th pct)

from .use_case import (
    DiseaseRiskUseCase,  # noqa: F401 — triggers register_module("disease")
)

__all__ = ["DiseaseRiskUseCase"]
