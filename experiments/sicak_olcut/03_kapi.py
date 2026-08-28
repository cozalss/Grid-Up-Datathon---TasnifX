"""ONARILMIS OLCUTLE ADAY YENIDEN TARAMASI -- kis26 ILERI kapisi.

Geometri (01_geometri.py): kis26 tek temiz blok (ileri kat payi %0, TEST gibi).
Karar kis26 ICINDE ileri kurulur:

    OGREN = 2025-12-01..2026-01-31   -> ayar/ofset buradan
    SINA  = 2026-02-01..2026-03-31   -> dMSE burada, trafo-kumeli bootstrap

Kuresel seviye hem tabanda hem adayda SINA uzerinde yeniden cozulur; boylece
karsilastirma seviyeden arindirilir (kuresel seviye zaten LB ile cozulmus,
docs/52 4: kappa*=0,31075 -> sicak cekirdekte ortalama artik SIFIR).

KABUL KAPISI: bootstrap %95 GA sifiri icermeyecek VE kazanan trafo payi >= %60.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
sys.path.insert(0, str(BURA))
sys.path.insert(0, str(BURA.parent / "sicak_kaldirac"))
from olcut import bootstrap, delta_coz, grup_ofseti, hazirla, mse_alt, zincir  # noqa: E402
from ortak import KOK, SICAK_PAY, bloklari_kur  # noqa: E402

BOLME = pd.Timestamp("2026-02-01")
YAZ = ("2025-04-01", "2025-07-31")


# ---------------------------------------------------------------- donusumler
def olcek(lam):
    def f(b, r, ham):
        m = float(r.mean())
        return m + lam * (r - m)

    return f


def trafo_buzme(beta):
    def f(b, r, ham):
        t = pd.Series(b.cerceve["tanim"].to_numpy())
        tort = pd.Series(r).groupby(t).transform("mean").to_numpy()
        m = float(r.mean())
        return m + beta * (tort - m) + (r - tort)

    return f


def ici_buzme(beta):
    def f(b, r, ham):
        t = pd.Series(b.cerceve["tanim"].to_numpy())
        tort = pd.Series(r).groupby(t).transform("mean").to_numpy()
        return tort + beta * (r - tort)

    return f


def dusuk_hedge(kappa, esik):
    def f(b, r, ham):
        tah = np.expm1(np.maximum(r + b.lgc, 0.0))
        out = r.copy()
        out[tah <= esik] -= kappa
        return out

    return f


def sifir_hedge(kappa, se, ke):
    def f(b, r, ham):
        c = b.cerceve
        m = (c["t_sifir_orani"].to_numpy() >= se) & (c["t_kuyruk_sifir"].to_numpy() >= ke)
        out = r.copy()
        out[m] -= kappa
        return out

    return f


def ofset(harita, anah, kat=1.0):
    def f(b, r, ham):
        d = pd.Series(b.cerceve[anah].to_numpy()).map(harita).fillna(0.0).to_numpy("float64")
        return r + kat * d

    return f


def sabit(R):
    def f(b, r, ham):
        return R

    return f


# ---------------------------------------------------------------- degerlendirme
def dmse(b, r_taban, don, m):
    r1 = don(b, r_taban, None)
    r1 = r1 + delta_coz(b, r1, m)
    r0 = r_taban + delta_coz(b, r_taban, m)
    return mse_alt(b, r1, m) - mse_alt(b, r0, m), r0, r1


def main() -> int:
    bl = bloklari_kur()
    b = bl["kis26"]
    hazirla(b)

    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    yazmask = (tr["tarih"] >= YAZ[0]) & (tr["tarih"] <= YAZ[1]) & (tr["tuketim"] > 0)
    g1 = set(tr.loc[yazmask, "tanim"])
    b.cerceve["koh"] = np.where(b.cerceve["tanim"].isin(g1), "G1", "G2")

    tar = pd.to_datetime(b.cerceve["tarih"]).to_numpy()
    kes = np.datetime64(BOLME)
    m_og = tar < kes
    m_si = tar >= kes
    r_ham = zincir(b)
    print(f"kis26 sicak {b.n:,} satir | OGREN {int(m_og.sum()):,} | SINA {int(m_si.sum()):,}")
    r0_si = r_ham + delta_coz(b, r_ham, m_si)
    taban_si = mse_alt(b, r0_si, m_si)
    r0_og = r_ham + delta_coz(b, r_ham, m_og)
    taban_og = mse_alt(b, r0_og, m_og)
    print(f"SINA taban MSE {taban_si:.6f}   OGREN taban MSE {taban_og:.6f}")

    adaylar = []

    # ---- I. OGRENILEN GRUP OFSETLERI (OGREN'den)
    for anah in (
        "hg",
        "kova",
        "ilce",
        "seviye_d",
        "seviye_d10",
        "gecmis_k",
        "sifir_k",
        "ilce_kova",
        "koh",
        "tanim",
    ):
        h = grup_ofseti(b, r0_og, m_og, anah, 200.0)
        for kat in (1.0, 0.5):
            adaylar.append((f"OFS {anah} n0=200 kat={kat}", ofset(h, anah, kat)))

    # ---- II. SKALER AYARLAR
    aileler = {
        "olcek lam": [(f"{v}", olcek(v)) for v in (0.94, 0.97, 1.03, 1.06, 1.10, 1.15)],
        "trafo-arasi buzme b": [(f"{v}", trafo_buzme(v)) for v in (0.90, 0.95, 1.05, 1.10, 1.15)],
        "trafo-ici buzme b": [(f"{v}", ici_buzme(v)) for v in (0.85, 0.90, 0.95, 1.05, 1.10)],
        "dusuk hedge": [
            (f"k={k} T={t}", dusuk_hedge(k, t))
            for k in (0.10, 0.15, 0.30, 0.50)
            for t in (5.0, 20.0, 50.0)
        ],
        "sifir hedge": [
            (f"k={k} s>={s} q>={q}", sifir_hedge(k, s, q))
            for k in (0.25, 0.5, 1.0)
            for s in (0.5, 0.9)
            for q in (7, 14)
        ],
    }
    print()
    print("=" * 108)
    print("SKALER AILELER -- OGREN taramasi (en iyi 4), sonra SINA'da kapi")
    print("=" * 108)
    for aile, uyeler in aileler.items():
        satir = []
        for et, fn in uyeler:
            d, _, _ = dmse(b, r_ham, fn, m_og)
            satir.append((d, et, fn))
        satir.sort(key=lambda z: z[0])
        print(f"{aile:24} " + "  ".join(f"{e}:{d:+.5f}" for d, e, _ in satir[:4]))
        d_best, et_best, fn_best = satir[0]
        if d_best < 0:
            adaylar.append((f"{aile}={et_best} (OGRENopt {d_best:+.5f})", fn_best))

    # ---- III. HARMAN AGIRLIKLARI ve AG AGIRLIGI
    print()
    print("=" * 108)
    print("HARMAN AGIRLIKLARI cat/xgb/lgbm/sinir_agi")
    print("=" * 108)
    harmanlar = {
        "uretim 3/1/1/1.4": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4},
        "ag YOK 3/1/1/0": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 0.0},
        "ag 0.7": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 0.7},
        "ag 2.1": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 2.1},
        "ag 3.0": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 3.0},
        "xgb agir 2/3/1/1.4": {"cat": 2.0, "xgb": 3.0, "lgbm": 1.0, "sinir_agi": 1.4},
        "2/2/1/1.4": {"cat": 2.0, "xgb": 2.0, "lgbm": 1.0, "sinir_agi": 1.4},
        "3/2/2/1.4": {"cat": 3.0, "xgb": 2.0, "lgbm": 2.0, "sinir_agi": 1.4},
        "2/3/0/1.4": {"cat": 2.0, "xgb": 3.0, "lgbm": 0.0, "sinir_agi": 1.4},
        "2/3/0/0": {"cat": 2.0, "xgb": 3.0, "lgbm": 0.0, "sinir_agi": 0.0},
        "esit 1/1/1/1": {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.0},
    }
    print(f"{'harman':24}{'dMSE_OGREN':>13}{'dMSE_SINA':>13}")
    harman_sonuc = []
    for ad, w in harmanlar.items():
        rr = zincir(b, agirlik=w)
        d_og = mse_alt(b, rr + delta_coz(b, rr, m_og), m_og) - taban_og
        d_si = mse_alt(b, rr + delta_coz(b, rr, m_si), m_si) - taban_si
        harman_sonuc.append((d_og, d_si, ad, rr))
        print(f"{ad:24}{d_og:>+13.5f}{d_si:>+13.5f}")
    harman_sonuc.sort(key=lambda z: z[0])
    if harman_sonuc[0][0] < -1e-9:
        adaylar.append(
            (
                f"HARMAN {harman_sonuc[0][2]} (OGRENopt {harman_sonuc[0][0]:+.5f})",
                sabit(harman_sonuc[0][3]),
            )
        )

    # ---- IV. GUN EKSENI ve KUYRUK
    print()
    print("=" * 108)
    print("URETIM SABITLERI -- gun ekseni c ve kuyruk deltasi")
    print("=" * 108)
    en_iyi_c = None
    for c in (1.0, 1.15, 1.3301, 1.5, 1.75, 2.0):
        rr = zincir(b, gun_olcek=c)
        d_og = mse_alt(b, rr + delta_coz(b, rr, m_og), m_og) - taban_og
        d_si = mse_alt(b, rr + delta_coz(b, rr, m_si), m_si) - taban_si
        print(f"  c={c:<8.4f} OGREN {d_og:+.5f}  SINA {d_si:+.5f}")
        if d_og < -1e-9 and (en_iyi_c is None or d_og < en_iyi_c[0]):
            en_iyi_c = (d_og, c, rr)
    if en_iyi_c:
        adaylar.append(
            (f"GUN EKSENI c={en_iyi_c[1]} (OGRENopt {en_iyi_c[0]:+.5f})", sabit(en_iyi_c[2]))
        )
    en_iyi_k = None
    for kd in (0.0, 0.08, 0.1664, 0.25, 0.35):
        rr = zincir(b, kuyruk=kd)
        d_og = mse_alt(b, rr + delta_coz(b, rr, m_og), m_og) - taban_og
        d_si = mse_alt(b, rr + delta_coz(b, rr, m_si), m_si) - taban_si
        print(f"  kuyruk={kd:<6.4f} OGREN {d_og:+.5f}  SINA {d_si:+.5f}")
        if d_og < -1e-9 and (en_iyi_k is None or d_og < en_iyi_k[0]):
            en_iyi_k = (d_og, kd, rr)
    if en_iyi_k:
        adaylar.append(
            (f"KUYRUK d={en_iyi_k[1]} (OGRENopt {en_iyi_k[0]:+.5f})", sabit(en_iyi_k[2]))
        )

    # ---- SINA'da kapi
    print()
    print("=" * 108)
    print("ILERI KAPI -- SINA (2026-02-01..2026-03-31), trafo-kumeli bootstrap B=1000")
    print("=" * 108)
    print(
        f"{'aday':48}{'dMSE_SINA':>11}{'GAalt':>10}{'GAust':>10}{'kazanan':>9}{'testdMSE':>10}  karar"
    )
    print("-" * 108)
    sonuclar = []
    for ad, fn in adaylar:
        d, r0, r1 = dmse(b, r_ham, fn, m_si)
        dm, lo, hi, kaz, nt = bootstrap(b, r0, r1, m_si, B=1000)
        if dm >= 0:
            karar = "RED(zararli)"
        elif hi >= 0:
            karar = "red(GA sifir)"
        elif kaz < 0.60:
            karar = f"red(kazanan %{100 * kaz:.0f})"
        else:
            karar = "GECTI"
        test_d = dm * SICAK_PAY
        print(
            f"{ad[:48]:48}{dm:>+11.5f}{lo:>+10.5f}{hi:>+10.5f}{100 * kaz:>8.1f}%"
            f"{test_d:>+10.5f}  {karar}"
        )
        sonuclar.append(
            {
                "aday": ad,
                "dMSE_SINA": dm,
                "GA_alt": lo,
                "GA_ust": hi,
                "kazanan_pay": kaz,
                "n_trafo": nt,
                "testdMSE": test_d,
                "gecti": karar == "GECTI",
            }
        )
    yol = BURA / "03_kapi.jsonl"
    with yol.open("w", encoding="utf-8") as f:
        for s in sonuclar:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nyazildi: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
