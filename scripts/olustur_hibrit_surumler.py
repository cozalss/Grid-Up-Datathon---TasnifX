"""Yarin icin gonderime hazir tum SOTA ve hibrit surumleri uret ve dogrula."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
GONDERIM = KOK / "submissions"
DATA = KOK / "data" / "raw"


def main():
    print("=" * 80)
    print("GRID UP -- YARINKI 1.LIK & 2.LIK GONDERIM PAKETI URETIMI")
    print("=" * 80)

    sota_yol = GONDERIM / "tuketim_sota_v1.csv"
    v83_yol = GONDERIM / "tuketim_v83_sicak_optimum.csv"
    v85_yol = GONDERIM / "tuketim_v85_gram_rank2.csv"

    if not sota_yol.exists():
        print(f"HATA: {sota_yol} bulunamadi!")
        return 1

    df_sota = pd.read_csv(sota_yol)
    df_v83 = pd.read_csv(v83_yol) if v83_yol.exists() else None
    df_v85 = pd.read_csv(v85_yol) if v85_yol.exists() else None

    # Log1p donusumleri
    log_sota = np.log1p(np.clip(df_sota["tuketim"].to_numpy(), 0, None))
    log_v83 = (
        np.log1p(np.clip(df_v83["tuketim"].to_numpy(), 0, None)) if df_v83 is not None else None
    )
    log_v85 = (
        np.log1p(np.clip(df_v85["tuketim"].to_numpy(), 0, None)) if df_v85 is not None else None
    )

    # Train yukle (sicak/soguk ayrimi icin)
    tr = pd.read_csv(DATA / "train.csv", usecols=["tanim"], dtype={"tanim": str})
    sicak_set = set(tr["tanim"].unique())
    tanimlar = df_sota["id"].str.rsplit("_", n=1).str[0]
    sicak_mask = tanimlar.isin(sicak_set).to_numpy()

    # 1. Aday: tuketim_sota_v1.csv (Saf SOTA) - zaten mevcut

    # 2. Aday: tuketim_sota_v2_hibrit_v83 (70% SOTA + 30% v83)
    if log_v83 is not None:
        log_v2 = 0.70 * log_sota + 0.30 * log_v83
        tuketim_v2 = np.clip(np.expm1(log_v2), 0, None)
        # Olu trafo sifirlamasini koru
        tuketim_v2[df_sota["tuketim"] == 0] = 0.0
        yol_v2 = GONDERIM / "tuketim_sota_v2_hibrit_v83.csv"
        pd.DataFrame({"id": df_sota["id"], "tuketim": tuketim_v2}).to_csv(yol_v2, index=False)
        print(f"[URETILDI] {yol_v2.name} (70% SOTA + 30% v83)")

    # 3. Aday: tuketim_sota_v3_hibrit_v85 (70% SOTA + 30% v85)
    if log_v85 is not None:
        log_v3 = 0.70 * log_sota + 0.30 * log_v85
        tuketim_v3 = np.clip(np.expm1(log_v3), 0, None)
        tuketim_v3[df_sota["tuketim"] == 0] = 0.0
        yol_v3 = GONDERIM / "tuketim_sota_v3_hibrit_v85.csv"
        pd.DataFrame({"id": df_sota["id"], "tuketim": tuketim_v3}).to_csv(yol_v3, index=False)
        print(f"[URETILDI] {yol_v3.name} (70% SOTA + 30% v85)")

    # 4. Aday: tuketim_sota_v4_kohort_optimum (Sicak: 75% SOTA + 25% v83, Soguk: 65% SOTA + 35% v85)
    if log_v83 is not None and log_v85 is not None:
        log_v4 = np.zeros_like(log_sota)
        log_v4[sicak_mask] = 0.75 * log_sota[sicak_mask] + 0.25 * log_v83[sicak_mask]
        log_v4[~sicak_mask] = 0.65 * log_sota[~sicak_mask] + 0.35 * log_v85[~sicak_mask]
        tuketim_v4 = np.clip(np.expm1(log_v4), 0, None)
        tuketim_v4[df_sota["tuketim"] == 0] = 0.0
        yol_v4 = GONDERIM / "tuketim_sota_v4_kohort_optimum.csv"
        pd.DataFrame({"id": df_sota["id"], "tuketim": tuketim_v4}).to_csv(yol_v4, index=False)
        print(f"[URETILDI] {yol_v4.name} (Kohort bazli optimum hibrit)")

    print("\nTum dosyalar basariyla hazirlandi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
