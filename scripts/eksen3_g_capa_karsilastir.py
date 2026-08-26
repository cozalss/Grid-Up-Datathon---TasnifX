# ruff: noqa
"""EKSEN 3 (sonuc): v55'in gun-olcegi CAPASI yanlis nicelige baglanmis mi?

v55'in c=1,604'u STD ORANINDAN geliyor:  sigma_gercek(2025 Nis-Tem) / sigma_tahmin(2026 Nis-Tem).
Ama gun ekseni std'si O PENCERENIN HAVA DEGISKENLIGINE baglidir. 2026 Nis-Tem
2025'ten SERIN; CDD'nin yayilimi daha kucuk. Ayni std'yi zorlamak, birim CDD
basina tepkiyi ASIRI buyutur.

Fiziksel nicelik EGIMDIR (birim CDD basina log tuketim), std degil.

CAPRAZ BLOK SINAMASI (yontemin kendisi):
    yaz25 gercekten olculen capayi guz25'e tasi.
        EGIM capasi:  +0,0717 -> guz25 gercegi +0,0693   (hata %3,5)
        STD  capasi:  0,2794  -> guz25 gercegi 0,2150    (hata %30)

    python scripts/eksen3_g_capa_karsilastir.py
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
import deney_ileri as di  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"


def iki_yonlu(e, trafo, gun, tur: int = 60):
    s = pd.Series(np.asarray(e, dtype="float64"))
    ti, gi = pd.Series(np.asarray(trafo)), pd.Series(np.asarray(gun))
    mu = float(s.mean())
    a = pd.Series(0.0, index=s.index)
    dd = pd.Series(0.0, index=s.index)
    for _ in range(tur):
        a = (s - mu - dd).groupby(ti).transform("mean")
        a = a - a.mean()
        dd = (s - mu - a).groupby(gi).transform("mean")
        dd = dd - dd.mean()
    return mu, dd.to_numpy()


def hava_serisi(ilce_agirlik: pd.Series) -> pd.DataFrame:
    h = pd.read_parquet(
        KOK / "data" / "external" / "hava_gunluk.parquet",
        columns=["ilce_key", "tarih", "sicaklik_ort"],
    )
    h["tarih"] = pd.to_datetime(h["tarih"]).dt.normalize()
    h = h[h["ilce_key"].isin(ilce_agirlik.index)].copy()
    h["w"] = h["ilce_key"].map(ilce_agirlik).astype("float64")
    h["sw"] = h["sicaklik_ort"] * h["w"]
    g = h.groupby("tarih", as_index=False)[["w", "sw"]].sum()
    g["sicaklik_ort"] = g["sw"] / g["w"]
    g["cdd22"] = (g["sicaklik_ort"] - 22.0).clip(lower=0)
    return g[["tarih", "sicaklik_ort", "cdd22"]]


def gun_serisi(frame: pd.DataFrame, deger: np.ndarray, hava: pd.DataFrame) -> pd.DataFrame:
    _, dd = iki_yonlu(deger, frame["tanim"].to_numpy(), frame["tarih"].to_numpy())
    t = (
        pd.DataFrame({"tarih": frame["tarih"].to_numpy(), "d": dd})
        .groupby("tarih", as_index=False)
        .mean()
        .merge(hava, on="tarih", how="left")
    )
    return t


def ozet(t: pd.DataFrame) -> dict:
    b, a = np.polyfit(t["cdd22"], t["d"], 1)
    kal = t["d"] - (a + b * t["cdd22"])
    return {
        "egim": float(b),
        "kor": float(t["d"].corr(t["cdd22"])),
        "std_gun": float(t["d"].std(ddof=0)),
        "std_dik": float(kal.std(ddof=0)),
        "std_cdd": float(t["cdd22"].std(ddof=0)),
        "ort_cdd": float(t["cdd22"].mean()),
    }


def main() -> int:
    z = np.load(ONBELLEK)
    egitim, test = d.cerceveleri_kur()
    O = {}

    for blok in ("yaz25", "guz25"):
        _, dv, gr, sk = di.blok_parcalari(egitim, blok)
        dg = dv[~sk].reset_index(drop=True)
        lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
        f = pd.DataFrame(
            {"tanim": dg["tanim"].to_numpy(), "tarih": pd.to_datetime(dg["tarih"]).to_numpy()}
        )
        hv = hava_serisi(dg["ilce_key"].value_counts(normalize=True))
        O[f"{blok} GERCEK"] = ozet(gun_serisi(f, np.log1p(np.clip(gr[~sk], 0, None)) - lg, hv))
        pay = sum(AGIRLIK)
        r = (
            np.mean(
                [
                    sum(AGIRLIK[i] * z[f"{blok}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
                    for t in di.TOHUMLAR
                ],
                axis=0,
            )
            - lg
        )
        O[f"{blok} FOLD tahmini"] = ozet(gun_serisi(f, r, hv))

    te = test[test["soguk_mu"] == 0].reset_index(drop=True)
    hv26 = hava_serisi(te["ilce_key"].value_counts(normalize=True))
    for isim, dosya in (
        ("v50 TEST", "tuketim_v50_nihai30.csv"),
        ("v55 TEST", "tuketim_v55_gunolcek.csv"),
    ):
        s = pd.read_csv(KOK / "submissions" / dosya)
        m = te[["id", "tanim", "tarih", "guc"]].merge(s, on="id", how="left")
        ofs = np.log1p(np.clip(m["tuketim"].to_numpy(dtype="float64"), 0, None)) - np.log1p(
            m["guc"].to_numpy(dtype="float64")
        )
        O[isim] = ozet(gun_serisi(m, ofs, hv26))

    print("=" * 104)
    print("GUN EKSENI PROFILLERI  (trafo etkisi cikarilmis, SICAK satirlar)")
    print("=" * 104)
    print(
        f"  {'seri':<24}{'egim/CDD':>10}{'kor':>8}{'std(gun)':>10}{'std(CDD-dik)':>14}{'std(CDD)':>10}{'ort CDD':>9}"
    )
    for k, v in O.items():
        print(
            f"  {k:<24}{v['egim']:>+10.5f}{v['kor']:>+8.3f}{v['std_gun']:>10.4f}"
            f"{v['std_dik']:>14.4f}{v['std_cdd']:>10.3f}{v['ort_cdd']:>9.3f}"
        )

    print()
    print("=" * 104)
    print("CAPA YONTEMI SINAMASI: yaz25'ten guz25'e tasi -- EGIM mi STD mi tasiniyor?")
    print("=" * 104)
    ye, ge = O["yaz25 GERCEK"], O["guz25 GERCEK"]
    print(
        f"  EGIM capasi : yaz25 {ye['egim']:+.5f} -> guz25 gercegi {ge['egim']:+.5f}"
        f"   hata {abs(ye['egim'] / ge['egim'] - 1) * 100:5.1f}%"
    )
    print(
        f"  STD  capasi : yaz25 {ye['std_gun']:.4f}  -> guz25 gercegi {ge['std_gun']:.4f}"
        f"    hata {abs(ye['std_gun'] / ge['std_gun'] - 1) * 100:5.1f}%"
    )

    print()
    print("=" * 104)
    print("v55'in CAPASI NEREYE OTURDU?  (gercek egim capasi = iki blogun ortalamasi)")
    print("=" * 104)
    capa = (ye["egim"] + ge["egim"]) / 2
    print(f"  EGIM CAPASI (yaz25 & guz25 ortalamasi) = {capa:+.5f}")
    for k in ("v50 TEST", "v55 TEST"):
        v = O[k]
        print(
            f"  {k:<10} egim {v['egim']:+.5f}   capaya gore {v['egim'] / capa:+.3f}x"
            f"   ({(v['egim'] / capa - 1) * 100:+.0f}%)"
        )
    k50 = capa / O["v50 TEST"]["egim"]
    print(f"\n  >> EGIM capasini tutturan olcek: c = {k50:.3f}   (v55 kullandigi: 1,604)")

    print()
    print("=" * 104)
    print("CDD KANALINDA dMSE  (gun ekseni, CDD'ye izdusum -- capa ile arasindaki fark)")
    print("=" * 104)
    varc = O["v50 TEST"]["std_cdd"] ** 2
    print(
        f"  Var(CDD) 2026 Nis-Tem = {varc:.4f}   (2025 Nis-Tem {O['yaz25 GERCEK']['std_cdd'] ** 2:.4f})"
    )
    print(f"  {'gonderim':<28}{'egim':>10}{'(egim-capa)^2*Var':>20}{'v55 e gore dMSE':>18}")
    taban = None
    kayit = {}
    for k in ("v50 TEST", "v55 TEST"):
        v = (O[k]["egim"] - capa) ** 2 * varc
        kayit[k] = v
        print(f"  {k:<28}{O[k]['egim']:>+10.5f}{v:>20.5f}{'':>18}")
    for c in (1.0, 1.195, 1.30, 1.604):
        e = O["v50 TEST"]["egim"] * c
        v = (e - capa) ** 2 * varc
        print(
            f"  {'v50 x c=' + f'{c:.3f}':<28}{e:>+10.5f}{v:>20.5f}{v - kayit['v55 TEST']:>+18.5f}"
        )
    print()
    print(
        f"  v55 -> capa-uyumlu olcek:  dMSE(CDD kanali, SICAK) = "
        f"{(capa - capa) ** 2 * varc - kayit['v55 TEST']:+.5f}"
    )
    print("  NOT: bu YALNIZ CDD kanali; dik kanal (hava disi gun degisimi) ayrica olcek yer.")

    # dik kanal: v55 dik std'yi 1,604 ile buyuttu; gercek dik std capasi 2025'ten
    print()
    print("=" * 104)
    print("DIK KANAL (CDD-disi gun degisimi): v55 gurultuyu de 1,604 ile buyuttu mu?")
    print("=" * 104)
    print(f"  2025 Nis-Tem GERCEK dik std   = {O['yaz25 GERCEK']['std_dik']:.4f}")
    print(f"  2025 Agu-Kas GERCEK dik std   = {O['guz25 GERCEK']['std_dik']:.4f}")
    print(f"  v50 TEST dik std              = {O['v50 TEST']['std_dik']:.4f}")
    print(f"  v55 TEST dik std              = {O['v55 TEST']['std_dik']:.4f}")
    print(
        "\n  Dik kanalda model ile gercegin korelasyonu BILINMIYOR (etiket yok)."
        "\n  Korelasyon r ise, dik kanali k ile olceklemenin dMSE'si:"
        "\n     (k^2 - 1)*var(dik_model) - 2*(k-1)*r*sd(dik_gercek)*sd(dik_model)"
    )
    sm50, sm55 = O["v50 TEST"]["std_dik"], O["v55 TEST"]["std_dik"]
    sg = (O["yaz25 GERCEK"]["std_dik"] + O["guz25 GERCEK"]["std_dik"]) / 2
    print(f"\n  {'r':>6}{'v55 dik dMSE (v50 tabanli)':>30}")
    for r in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        k = sm55 / sm50
        v = (k**2 - 1) * sm50**2 - 2 * (k - 1) * r * sg * sm50
        print(f"  {r:>6.1f}{v:>30.5f}")
    print("\n  (pozitif = v55 dik kanalda ZARAR etti; LB kazanci CDD kanalindan gelmis olmali)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
