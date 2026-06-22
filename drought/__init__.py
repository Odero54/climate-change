"""
Drought monitoring module — wraps the drought-monitoring package.

CDI sub-index scale (all indices are ratio-based, not anomalies):
  < 0.50  Extreme Drought
  0.50 – 0.65  Severe Drought
  0.65 – 0.80  Moderate Drought
  0.80 – 0.90  Mild Drought
  0.90 – 1.10  Near Normal
  1.10 – 1.20  Mild Wet
  1.20 – 1.30  Moderately Wet
  > 1.30  Very Wet
"""

from .use_case import DroughtUseCase

__all__ = ["DroughtUseCase"]
