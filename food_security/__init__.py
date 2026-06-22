# VHI (Vegetation Health Index) scale reference:
#   < 20   — Extreme food stress
#   20–35  — Severe stress
#   35–50  — Moderate stress
#   50–70  — Good vegetation health
#   > 70   — Excellent
#
# Food insecurity risk classes (tercile thresholds on composite stress score):
#   0 = Low Risk   (score < 33rd pct)
#   1 = Medium Risk (33rd–66th pct)
#   2 = High Risk  (score > 66th pct)

from .use_case import (
    FoodSecurityUseCase,  # noqa: F401 — triggers register_module("food_security")
)

__all__ = ["FoodSecurityUseCase"]
