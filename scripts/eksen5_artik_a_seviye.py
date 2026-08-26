# ruff: noqa
"""EKSEN 5 (a) -- ARTIK HEDEFI icin SEVIYE tanimi ve DOLULUK denetimi.

seviye_i = kesme oncesi son W gunun ortalama OFSETI (ofs = log1p(tuketim) - log1p(guc)),
YALNIZCA POZITIF satirlar uzerinden. W in {90, 180, 365, tum}.

Kalici kural 2: her aday kolonun EGITIM/TEST doluluk deseni karsilastirilir.
Kalici kural 7: as-of -- pencere hedefin kesme aninda BITER.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import tuketim_model as tm  # noqa: E402

CIK = KOK / "data" / "interim" / "eksen5"
PENCERELER = (90, 180, 365, 9999)
KESME = {
    "yaz25": "2025-03-31",
    "guz25": "2025-07-31",
    "kis26": "2025-11-30",
    "TEST": "2026-03-31",
}


def seviyeleri_cikar(ham: pd.DataFrame, kesme: str) -> pd.DataFrame:
    """Bir kesme tarihi icin trafo bazinda seviye tablosu (as-of)."""
    k = pd.Timestamp(kesme)
    alt = ham[(ham["tarih"] <= k) & (ham["tarih"] >= pd.Timestamp(tm.EGITIM_BASI))]
    poz = alt[alt["tuketim"] > 0].copy()
    poz["_ofs"] = np.log1p(poz["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        poz["guc"].to_numpy(dtype="float64")
    )
    cikti = None
    for W in PENCERELER:
        bas = k - pd.Timedelta(days=W - 1)
        p = poz[poz["tarih"] >= bas]
        g = p.groupby("tanim")["_ofs"]
        tab = pd.DataFrame({f"sev{W}": g.mean(), f"n{W}": g.size()})
        cikti = tab if cikti is None else cikti.join(tab, how="outer")
    return cikti


def main() -> int:
    CIK.mkdir(parents=True, exist_ok=True)
    ham, _ = tm.yukle()
    ham = ham[["tanim", "guc", "tarih", "tuketim"]].copy()
    ham["tarih"] = pd.to_datetime(ham["tarih"])
    print(f"  ham {ham.shape}  {ham['tarih'].min().date()} .. {ham['tarih'].max().date()}")

    egitim, test = d.cerceveleri_kur()

    tablolar = {}
    for ad, kes in KESME.items():
        tablolar[ad] = seviyeleri_cikar(ham, kes)
        print(f"  {ad:6} kesme {kes}  trafo(sev90 dolu) {tablolar[ad]['sev90'].notna().sum():,}")

    print("\n" + "=" * 96)
    print("DOLULUK DESENI -- satir bazinda (kalici kural 2)")
    print(
        f"  {'kume':7}{'satir':>10}{'sicak%':>8}"
        + "".join(f"{'sev' + str(W):>9}" for W in PENCERELER)
        + f"{'SICAKTA sev90%':>16}"
    )
    kayit = {}
    for ad in KESME:
        if ad == "TEST":
            cer = test
        else:
            cer = egitim[egitim["_blok"] == ad]
        tab = tablolar[ad]
        sicak = (cer["soguk_mu"] == 0).to_numpy()
        satir = []
        for W in PENCERELER:
            v = cer["tanim"].map(tab[f"sev{W}"]).to_numpy(dtype="float64")
            satir.append(np.isfinite(v).mean() * 100)
        v90 = cer["tanim"].map(tab["sev90"]).to_numpy(dtype="float64")
        sic90 = np.isfinite(v90[sicak]).mean() * 100
        kayit[ad] = (len(cer), sicak.mean() * 100, satir, sic90)
        print(
            f"  {ad:7}{len(cer):>10,}{sicak.mean() * 100:8.1f}"
            + "".join(f"{s:9.1f}" for s in satir)
            + f"{sic90:16.2f}"
        )

    print("\n" + "=" * 96)
    print("SEVIYE DAGILIMI (sicak satirlarda, satir agirlikli)")
    print(f"  {'kume':7}{'ort':>9}{'std':>9}{'p10':>9}{'p50':>9}{'p90':>9}{'n_ort':>9}")
    for ad in KESME:
        cer = test if ad == "TEST" else egitim[egitim["_blok"] == ad]
        tab = tablolar[ad]
        s = cer["tanim"].map(tab["sev90"]).to_numpy(dtype="float64")
        n = cer["tanim"].map(tab["n90"]).to_numpy(dtype="float64")
        ok = np.isfinite(s)
        print(
            f"  {ad:7}{s[ok].mean():9.4f}{s[ok].std():9.4f}"
            f"{np.percentile(s[ok], 10):9.4f}{np.percentile(s[ok], 50):9.4f}"
            f"{np.percentile(s[ok], 90):9.4f}{n[ok].mean():9.1f}"
        )

    # ---- geri cekilme merdiveni: sev90 -> sev180 -> sev365 -> tum -> NaN
    print("\n" + "=" * 96)
    print("GERI CEKILME MERDIVENI -- hangi basamak kac satiri kurtariyor (sicak satirlar)")
    print(f"  {'kume':7}{'sicak':>10}{'sev90':>9}{'+180':>9}{'+365':>9}{'+tum':>9}{'KALAN':>9}")
    for ad in KESME:
        cer = test if ad == "TEST" else egitim[egitim["_blok"] == ad]
        tab = tablolar[ad]
        sicak = (cer["soguk_mu"] == 0).to_numpy()
        c = cer[sicak]
        kalan = np.ones(len(c), dtype=bool)
        adim = []
        for W in PENCERELER:
            v = c["tanim"].map(tab[f"sev{W}"]).to_numpy(dtype="float64")
            yeni = kalan & np.isfinite(v)
            adim.append(yeni.mean() * 100)
            kalan = kalan & ~np.isfinite(v)
        print(
            f"  {ad:7}{len(c):>10,}"
            + "".join(f"{a:9.2f}" for a in adim)
            + f"{kalan.mean() * 100:9.2f}"
        )

    for ad, tab in tablolar.items():
        tab.to_parquet(CIK / f"seviye_{ad}.parquet")
    print(f"\n  yazildi -> {CIK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
