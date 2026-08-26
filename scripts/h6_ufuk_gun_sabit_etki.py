# ruff: noqa
"""H6 (b) -- UFKU MEVSIMDEN AYIRAN TEK KURGU: ORTUSEN KOKENLER + GUN SABIT ETKISI.

SORUN
-----
Tek blok icinde ufuk = takvim gunu. Blok icinde r(ufuk) egrisi olcmek
MEVSIMI olcmektir; ikisi %100 karisik. Iki blok karsilastirmasi zayif bir
ayrimdir cunku bloklar farkli MODEL ve farkli tutulan-disarida mevsim
tasir.

KESIN AYRIM
-----------
AYNI TAKVIM GUNU birden fazla KESME'den tahmin edilir:
    2025-07-15  ->  kesme 2025-03-31'den ufuk 106
                    kesme 2025-04-30'dan ufuk  76
                    kesme 2025-05-31'den ufuk  45
                    kesme 2025-06-30'dan ufuk  15
Gun sabit etkisi konunca (mu_d) MEVSIM tamamen sogurulur. Geriye kalan
egim SAF UFUKTUR.

MODELDEN BAGIMSIZ. Tahminci = trafonun kesmeden onceki 90 gunluk log
ortalamasi (capa). Bu, "seviye capasi bayatladikca hata birikir"
mekanizmasinin EN GUCLU halidir: agac modeli capayi daha da yumusatir.
Burada sinyal yoksa modelde hic yok.

    uv run python scripts/h6_ufuk_gun_sabit_etki.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
CIKTI = KOK / "reports" / "h6_ufuk"

#: Aylik kesmeler. Her birinin ardindan 122 gunluk hedef penceresi var
#: (veri 2026-03-31'de bitiyor; 2025-11-30 son tam kesme).
KESMELER = [
    "2025-03-31",
    "2025-04-30",
    "2025-05-31",
    "2025-06-30",
    "2025-07-31",
    "2025-08-31",
    "2025-09-30",
    "2025-10-31",
    "2025-11-30",
]
CAPA_PENCERE = 90
EN_AZ_KAYIT = 30
UFUK_MAKS = 122


def iki_yonlu_ici(
    y: np.ndarray, x: np.ndarray, g1: np.ndarray, g2: np.ndarray, tur: int = 30
) -> tuple[np.ndarray, np.ndarray]:
    """Iki yonlu (g1, g2) sabit etki icin ardisik ortalama cikarma."""
    y = y.astype("float64").copy()
    x = x.astype("float64").copy()
    s1 = pd.Series(g1)
    s2 = pd.Series(g2)
    for _ in range(tur):
        for s in (s1, s2):
            y -= pd.Series(y).groupby(s).transform("mean").to_numpy()
            x -= pd.Series(x).groupby(s).transform("mean").to_numpy()
    return y, x


def kumeli_egim(y: np.ndarray, x: np.ndarray, kume: np.ndarray) -> tuple[float, float]:
    """Sabitsiz tek degiskenli OLS + kume-saglam SE."""
    sxx = float(np.dot(x, x))
    b = float(np.dot(x, y) / sxx)
    e = y - b * x
    s = pd.Series(x * e).groupby(pd.Series(kume)).sum().to_numpy()
    v = float(np.dot(s, s) / sxx**2)
    return b, float(np.sqrt(v))


def main() -> int:
    t0 = time.time()
    CIKTI.mkdir(parents=True, exist_ok=True)
    print("=" * 96)
    print("H6 (b) -- ORTUSEN KOKENLER + GUN SABIT ETKISI  (ufuk vs mevsim ayrimi)")
    print("=" * 96)

    ham = pd.read_csv(KOK / "data" / "raw" / "train.csv", parse_dates=["tarih"])
    ham["ly"] = np.log1p(ham["tuketim"].clip(lower=0.0))
    ham = ham.sort_values(["tanim", "tarih"]).reset_index(drop=True)
    print(f"  train {len(ham):,} satir  {ham['tanim'].nunique():,} trafo")

    parcalar = []
    print(f"\n  {'kesme':12}{'trafo':>8}{'hedef satir':>13}{'capa n ort':>12}")
    for kes in KESMELER:
        C = pd.Timestamp(kes)
        gec = ham[(ham["tarih"] > C - pd.Timedelta(days=CAPA_PENCERE)) & (ham["tarih"] <= C)]
        capa = gec.groupby("tanim")["ly"].agg(["mean", "size"])
        capa = capa[capa["size"] >= EN_AZ_KAYIT]
        hed = ham[(ham["tarih"] > C) & (ham["tarih"] <= C + pd.Timedelta(days=UFUK_MAKS))]
        hed = hed[hed["tanim"].isin(capa.index)].copy()
        hed["capa"] = hed["tanim"].map(capa["mean"]).to_numpy()
        hed["r"] = hed["ly"] - hed["capa"]
        hed["ufuk"] = (hed["tarih"] - C).dt.days.astype("int64")
        hed["kesme"] = kes
        parcalar.append(hed[["tanim", "tarih", "r", "ufuk", "kesme"]])
        print(f"  {kes:12}{len(capa):8,}{len(hed):13,}{capa['size'].mean():12.1f}")
    p = pd.concat(parcalar, ignore_index=True)
    print(f"\n  TOPLAM {len(p):,} (trafo,gun,kesme) uclusu")

    gun_kod = p["tarih"].astype("int64").to_numpy()
    tr_kod = pd.factorize(p["tanim"])[0]
    trkes_kod = pd.factorize(p["tanim"] + "|" + p["kesme"])[0]
    gun_sayisi = p.groupby(gun_kod)["kesme"].nunique()
    print(
        f"  kesme cesitliligi/gun: ort {gun_sayisi.mean():.2f}  "
        f"maks {gun_sayisi.max()}  >=2 kesmeli gun {int((gun_sayisi >= 2).sum())}"
    )

    # ------------------------------------------------------------------ 1
    print("\n" + "-" * 96)
    print("1) HAVUZLANMIS (mevsim SOGURULMAMIS) -- blok-ici egrinin karsiligi")
    print("-" * 96)
    b0, se0 = kumeli_egim(
        p["r"].to_numpy() - p["r"].mean(), p["ufuk"].to_numpy() - p["ufuk"].mean(), gun_kod
    )
    print(f"  b = {b0:+.6f}  SE {se0:.6f}  t {b0 / se0:+.2f}   122 gunde {b0 * 121:+.4f}")

    # ------------------------------------------------------------------ 2
    print("\n" + "-" * 96)
    print("2) GUN SABIT ETKISI + TRAFO SABIT ETKISI  --  MEVSIM SOGURULDU")
    print("   geriye kalan egim = SAF UFUK ETKISI")
    print("-" * 96)
    yv, xv = iki_yonlu_ici(
        p["r"].to_numpy(), p["ufuk"].to_numpy().astype("float64"), gun_kod, tr_kod
    )
    b1, se1 = kumeli_egim(yv, xv, gun_kod)
    print(f"  b = {b1:+.6f}  SE {se1:.6f}  t {b1 / se1:+.2f}   122 gunde {b1 * 121:+.4f}")
    print(f"  ufuk varyansinin sabit etkilerden SONRA kalani: {xv.var() / p['ufuk'].var():.4f}")

    # ------------------------------------------------------------------ 3
    print("\n" + "-" * 96)
    print("3) GUN SABIT ETKISI + (TRAFOxKESME) SABIT ETKISI")
    print("   capa seviyesindeki kesmeye ozgu farki da sogurur -- en muhafazakar")
    print("-" * 96)
    yv2, xv2 = iki_yonlu_ici(
        p["r"].to_numpy(), p["ufuk"].to_numpy().astype("float64"), gun_kod, trkes_kod
    )
    b2, se2 = kumeli_egim(yv2, xv2, gun_kod)
    print(f"  b = {b2:+.6f}  SE {se2:.6f}  t {b2 / se2:+.2f}   122 gunde {b2 * 121:+.4f}")
    print(f"  kalan ufuk varyansi: {xv2.var() / p['ufuk'].var():.4f}")

    # ------------------------------------------------------------------ 4
    print("\n" + "-" * 96)
    print("4) GUN SABIT ETKILI UFUK PROFILI (10 gunluk kova, referans kova 1-10)")
    print("-" * 96)
    kova = np.clip(p["ufuk"].to_numpy() // 10, 0, 12)
    D = pd.get_dummies(kova, prefix="k", drop_first=True).to_numpy(dtype="float64")
    yv3 = p["r"].to_numpy().astype("float64").copy()
    Dv = D.copy()
    s1 = pd.Series(gun_kod)
    s2 = pd.Series(tr_kod)
    for _ in range(30):
        for s in (s1, s2):
            yv3 -= pd.Series(yv3).groupby(s).transform("mean").to_numpy()
            Dv -= np.column_stack(
                [
                    pd.Series(Dv[:, j]).groupby(s).transform("mean").to_numpy()
                    for j in range(Dv.shape[1])
                ]
            )
    kat = np.linalg.lstsq(Dv, yv3, rcond=None)[0]
    print(f"  {'kova':>9}{'ham fark':>12}{'gunFE fark':>13}{'n':>10}")
    ham_ort = pd.Series(p["r"].to_numpy()).groupby(kova).mean()
    n_kova = pd.Series(kova).value_counts().sort_index()
    satirlar = []
    for j in range(13):
        etiket = f"{j * 10 + 1}-{min((j + 1) * 10, 122)}"
        hf = ham_ort[j] - ham_ort[0]
        gf = 0.0 if j == 0 else kat[j - 1]
        print(f"  {etiket:>9}{hf:+12.4f}{gf:+13.4f}{int(n_kova[j]):10,}")
        satirlar.append({"kova": etiket, "ham_fark": hf, "gunFE_fark": gf, "n": int(n_kova[j])})
    pd.DataFrame(satirlar).to_csv(CIKTI / "ufuk_gunFE_profil.csv", index=False)

    # ------------------------------------------------------------------ 5
    print("\n" + "-" * 96)
    print("5) KESME BAZLI EGIM (her kesme kendi icinde -- mevsim karisik)")
    print("-" * 96)
    print(f"  {'kesme':12}{'b':>11}{'122g':>10}{'n':>12}")
    for kes in KESMELER:
        m = (p["kesme"] == kes).to_numpy()
        b, _ = np.polyfit(p["ufuk"].to_numpy()[m], p["r"].to_numpy()[m], 1)
        print(f"  {kes:12}{b:+11.6f}{b * 121:+10.4f}{int(m.sum()):12,}")

    # ------------------------------------------------------------------ 6
    print("\n" + "-" * 96)
    print("6) YALNIZ YAZ HEDEFLERI (hedef gun Haziran-Temmuz)  --  yaz25 rampasinin kaynagi")
    print("-" * 96)
    yaz = p["tarih"].dt.month.isin([6, 7]).to_numpy()
    pz = p[yaz]
    print(
        f"  n {len(pz):,}  kesme sayisi {pz['kesme'].nunique()}  ufuk {pz['ufuk'].min()}-{pz['ufuk'].max()}"
    )
    b3, _ = np.polyfit(pz["ufuk"], pz["r"], 1)
    print(f"  havuzlanmis b = {b3:+.6f}   122g {b3 * 121:+.4f}")
    gk = pz["tarih"].astype("int64").to_numpy()
    tk = pd.factorize(pz["tanim"])[0]
    yv4, xv4 = iki_yonlu_ici(pz["r"].to_numpy(), pz["ufuk"].to_numpy().astype("float64"), gk, tk)
    b4, se4 = kumeli_egim(yv4, xv4, gk)
    print(f"  gunFE+trafoFE b = {b4:+.6f}  SE {se4:.6f}  t {b4 / se4:+.2f}   122g {b4 * 121:+.4f}")

    ozet = {
        "havuz_b": b0,
        "havuz_se": se0,
        "gunFE_trafoFE_b": b1,
        "gunFE_trafoFE_se": se1,
        "gunFE_trafokesmeFE_b": b2,
        "gunFE_trafokesmeFE_se": se2,
        "yaz_havuz_b": b3,
        "yaz_gunFE_b": b4,
        "yaz_gunFE_se": se4,
    }
    (CIKTI / "gunFE_ozet.json").write_text(
        json.dumps(ozet, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika  -> {CIKTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
