"""CV bloklarinda SOTA ve v83 harman agirliklarini test et."""

from __future__ import annotations

import numpy as np

# yaz25 blok skorlari:
# v83 benzeri temel model: Sicak = 0.81360, Test-Agirlikli = 1.04529
# SOTA model: Sicak = 0.70258, Test-Agirlikli = 0.99424

# Iki modelin kovaryansi rho = 0.92 civarinda.
mse_v83_sicak = 0.81360**2  # 0.6619
mse_sota_sicak = 0.70258**2  # 0.4936
rho = 0.92
cov = rho * 0.81360 * 0.70258  # 0.5259

print("=" * 60)
print("SICAK TRAFOLAR ICIN HARMAN OPTIMIZASYONU (yaz25)")
print("=" * 60)
for w_sota in [1.0, 0.9, 0.85, 0.80, 0.75, 0.70, 0.60, 0.50, 0.0]:
    w_v83 = 1.0 - w_sota
    mse_blend = (
        (w_sota**2 * mse_sota_sicak) + (w_v83**2 * mse_v83_sicak) + (2 * w_sota * w_v83 * cov)
    )
    rmsle_blend = np.sqrt(mse_blend)
    print(
        f"  SOTA: %{w_sota * 100:4.0f} | v83: %{w_v83 * 100:4.0f} -> Tahmini RMSLE: {rmsle_blend:.5f}"  # noqa: E501
    )

# Optimum agirlik analitik olarak:
w_opt = (mse_v83_sicak - cov) / (mse_sota_sicak + mse_v83_sicak - 2 * cov)
w_opt = np.clip(w_opt, 0.0, 1.0)
print(f"\nAnalitik Optimum SOTA Agirligi: %{w_opt * 100:.1f}")
