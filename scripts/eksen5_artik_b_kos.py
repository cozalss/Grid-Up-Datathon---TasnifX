# ruff: noqa
"""EKSEN 5 (b) -- ARTIK HEDEFI vs OFS HEDEFI, ayni satirlarda temiz A/B.

UC KOL (hepsi AYNI satir kumesi, AYNI aile, AYNI tohum):
  A   hedef = ofs                       kolonlar = taban
  A+  hedef = ofs                       kolonlar = taban + [sev, sev_n, sev_kaynak]
  B   hedef = u = ofs - sev             kolonlar = taban + [sev, sev_n, sev_kaynak]

A vs A+  : seviyeyi OZNITELIK olarak vermek ise yariyor mu?
A+ vs B  : hedefi merkezlemek (parametrelendirme) ise yariyor mu?  <-- ASIL SORU

Satir kumesi: SICAK ve seviyesi TANIMLI satirlar (geri cekilme merdiveni
90 -> 180 -> 365 -> tum gecmis). Soguk rejimde artik hedefi TANIMSIZ (sart d).

Cikti: her (blok, tohum, kol) icin LOG UZAYINDA tahmin -> npz.
Cozumleme ayri betikte (eksen5_artik_c_coz.py), fit tekrar etmeden.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

CIK = KOK / "data" / "interim" / "eksen5"
TOHUMLAR = (1000, 1001, 1002)
PENCERELER = (90, 180, 365, 9999)
SEV_KOLON = ["sev", "sev_n", "sev_kaynak"]


def seviye_ekle(cerceve: pd.DataFrame, blok_adi: str) -> pd.DataFrame:
    """Blogun KENDI kesme tarihinden cikarilmis seviyeyi merdivenle baglar."""
    tab = pd.read_parquet(CIK / f"seviye_{blok_adi}.parquet")
    tn = cerceve["tanim"]
    sev = np.full(len(cerceve), np.nan)
    sev_n = np.full(len(cerceve), np.nan)
    kaynak = np.full(len(cerceve), -1.0)
    for adim, W in enumerate(PENCERELER):
        v = tn.map(tab[f"sev{W}"]).to_numpy(dtype="float64")
        n = tn.map(tab[f"n{W}"]).to_numpy(dtype="float64")
        yeni = ~np.isfinite(sev) & np.isfinite(v)
        sev[yeni] = v[yeni]
        sev_n[yeni] = n[yeni]
        kaynak[yeni] = adim
    out = cerceve.copy()
    out["sev"] = sev
    out["sev_n"] = np.log1p(sev_n)
    out["sev_kaynak"] = kaynak
    return out


def kume_kur(egitim: pd.DataFrame) -> pd.DataFrame:
    """Tum egitim satirlarina blok bazinda seviye ekler."""
    parcalar = []
    for b in tm.BLOKLAR:
        parcalar.append(seviye_ekle(egitim[egitim["_blok"] == b.ad], b.ad))
    return pd.concat(parcalar, ignore_index=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aile", default="lgbm")
    ap.add_argument("--tohum-sayisi", type=int, default=3)
    ar = ap.parse_args()
    t00 = time.time()

    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    egitim = kume_kur(egitim)
    print(f"  egitim {len(egitim):,}  taban kolon {len(kolonlar)}")

    # UYGUN satir: sicak VE seviyesi tanimli
    uygun = ((egitim["soguk_mu"] == 0) & egitim["sev"].notna()).to_numpy()
    print(f"  uygun satir {uygun.sum():,} / {len(egitim):,}  (%{uygun.mean() * 100:.1f})")
    ege = egitim[uygun].reset_index(drop=True)
    for b in tm.BLOKLAR:
        n = int((ege["_blok"] == b.ad).sum())
        print(f"    {b.ad}: {n:,}")

    kol_a = kolonlar
    kol_b = kolonlar + SEV_KOLON
    tohumlar = TOHUMLAR[: ar.tohum_sayisi]

    for b in tm.BLOKLAR:
        dog = ege[ege["_blok"] == b.ad].reset_index(drop=True)
        kalan = ege[ege["_blok"] != b.ad].reset_index(drop=True)
        lg_d = np.log1p(dog["guc"].to_numpy(dtype="float64"))
        sev_d = dog["sev"].to_numpy(dtype="float64")
        sev_k = kalan["sev"].to_numpy(dtype="float64")
        # artik hedef: tuketim kolonunu gecici olarak degistirmek yerine
        # egit_tahmin'i baypas edip dogrudan model kuruyoruz.
        y_ofs = np.log1p(kalan[tm.HEDEF].clip(lower=0.0).to_numpy(dtype="float64")) - np.log1p(
            kalan["guc"].to_numpy(dtype="float64")
        )
        y_u = y_ofs - sev_k
        print(
            f"\n=== {b.ad}  egitim {len(kalan):,}  dogrulama {len(dog):,}"
            f"  hedef std: ofs {y_ofs.std():.4f}  u {y_u.std():.4f}"
        )
        paket = {
            "gercek": dog[tm.HEDEF].to_numpy(dtype="float64"),
            "lg": lg_d,
            "sev": sev_d,
            "tanim": dog["tanim"].to_numpy().astype(str),
            "tarih": dog["tarih"].to_numpy().astype("datetime64[D]").astype("int64"),
            "guc": dog["guc"].to_numpy(dtype="float64"),
            "ufuk_gun": dog["ufuk_gun"].to_numpy(dtype="float64"),
            "t_son_kayit_yasi": dog["t_son_kayit_yasi"].to_numpy(dtype="float64"),
            "sev_kaynak": dog["sev_kaynak"].to_numpy(dtype="float64"),
        }
        for tohum in tohumlar:
            for kol_ad, kols, hedef in (
                ("A", kol_a, y_ofs),
                ("Aplus", kol_b, y_ofs),
                ("B", kol_b, y_u),
            ):
                t0 = time.time()
                model = di.aile_modeli(ar.aile, tohum)
                x_e, x_h = kalan[kols], dog[kols]
                if ar.aile == "cat":
                    x_e, x_h = x_e.copy(), x_h.copy()
                    kat = [k for k in tm.KATEGORIK if k in x_e.columns]
                    for k in kat:
                        x_e[k] = x_e[k].astype(str)
                        x_h[k] = x_h[k].astype(str)
                    model.fit(x_e, hedef, cat_features=kat)
                else:
                    model.fit(x_e, hedef)
                ham = np.asarray(model.predict(x_h), dtype="float64")
                ofs_t = ham + sev_d if kol_ad == "B" else ham
                paket[f"{kol_ad}_{tohum}"] = ofs_t
                r = tm.rmsle(paket["gercek"], np.expm1(lg_d + ofs_t))
                print(f"    {kol_ad:6} t{tohum}  RMSLE {r:.5f}  ({time.time() - t0:.0f} sn)")
        np.savez_compressed(CIK / f"kos_{ar.aile}_{b.ad}.npz", **paket)
        print(f"  -> {CIK / f'kos_{ar.aile}_{b.ad}.npz'}")

    print(f"\nTAMAM {(time.time() - t00) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
