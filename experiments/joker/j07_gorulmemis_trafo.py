"""J7 - (a) ufuk yonunun aydan arindirilmis hali, (b) GECMISI OLMAYAN trafo grubu.

TEST'te 2.024 trafo (%28,8) train'de HIC gecmiyor -> 158.369 satir = %22,2.
Tezgah bloklarinda ayni tanimi kur: blok basindan ONCE hic satiri olmayan trafo.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

KOK = r"C:\Users\Cem\Desktop\Datahon_Laptop\Grid-Up-Datathon---TasnifX"
sys.path.insert(0, KOK + r"\experiments\joker")
from j05_tezgah import bloklari_kur  # noqa: E402

BLOKLAR = ("yaz25", "guz25", "kis26")


def p(*a):
    print(*a, flush=True)


def artik(d, lp=None):
    v = d["lp"].to_numpy() if lp is None else lp
    return np.log1p(np.clip(np.expm1(v), 0.0, None)) - d["ly"].to_numpy()


def yon_kazanc(e, v):
    """dMSE(k) = k^2 Q + 2k E[ev];  k* = -E[ev]/Q; en iyi = -(E[ev])^2/Q."""
    Q = float((v * v).mean())
    Lev = float((e * v).mean())
    if Q <= 0:
        return 0.0, 0.0, 0.0
    return Q, -Lev / Q, -(Lev**2) / Q


bl = bloklari_kur()
ham = pd.read_csv(
    KOK + r"\data\raw\train.csv",
    encoding="utf-8",
    dtype={"tanim": str},
    usecols=["tanim", "tarih", "tuketim"],
)
ham["t"] = pd.to_datetime(ham["tarih"])

p("=" * 100)
p("A) UFUK YONU -- AY KUKLA DEGISKENLERINDEN ARINDIRILMIS")
p("=" * 100)
p(f"{'blok':8}{'ham k*':>10}{'ham dMSE':>12}{'aydan arin k*':>16}{'arin dMSE':>12}")
for b in BLOKLAR:
    d = bl[b]
    e = artik(d)
    u = d["ufuk_gun"].to_numpy(dtype="float64")
    v = (u - u.mean()) / u.std()
    Q0, k0, g0 = yon_kazanc(e, v)
    ay = pd.to_datetime(d["tarih"]).dt.to_period("M").astype(str).to_numpy()
    # v'yi ay ortalamalarindan arindir
    v2 = v - pd.Series(v).groupby(ay).transform("mean").to_numpy()
    sd = v2.std()
    v2 = v2 / sd if sd > 0 else v2
    Q1, k1, g1 = yon_kazanc(e, v2)
    p(f"{b:8}{k0:>10.4f}{g0:>12.6f}{k1:>16.4f}{g1:>12.6f}")

p()
p("=" * 100)
p("B) GECMISI OLMAYAN TRAFO GRUBU")
p("=" * 100)
p(
    f"{'blok':8}{'n':>10}{'yeni satir':>12}{'pay %':>8}{'ort artik(yeni)':>17}"
    f"{'ort artik(eski)':>17}{'MSE yeni':>10}{'MSE eski':>10}"
)
ozet = {}
for b in BLOKLAR:
    d = bl[b]
    t0 = pd.to_datetime(d["tarih"]).min()
    onceki = set(ham.loc[ham["t"] < t0, "tanim"].unique())
    yeni = ~d["tanim"].astype(str).isin(onceki).to_numpy()
    e = artik(d)
    ozet[b] = (e, yeni)
    p(
        f"{b:8}{len(d):10,}{int(yeni.sum()):12,}{100 * yeni.mean():8.2f}"
        f"{e[yeni].mean():>17.5f}{e[~yeni].mean():>17.5f}"
        f"{(e[yeni] ** 2).mean():>10.4f}{(e[~yeni] ** 2).mean():>10.4f}"
    )

p()
p("1[yeni] YONU (merkezlenmemis gosterge; uretimde tam olarak uygulanabilir):")
p(f"{'blok':8}{'Q':>10}{'k* (log kaydirma)':>20}{'dMSE':>12}")
for b in BLOKLAR:
    e, yeni = ozet[b]
    v = yeni.astype(float)
    Q, k, g = yon_kazanc(e, v)
    p(f"{b:8}{Q:>10.4f}{k:>20.5f}{g:>12.6f}")

p()
p("KURESEL SEVIYEYE DIK HALI (1[yeni] - ort):")
p(f"{'blok':8}{'Q':>10}{'k*':>12}{'dMSE':>12}")
for b in BLOKLAR:
    e, yeni = ozet[b]
    v = yeni.astype(float) - yeni.mean()
    Q, k, g = yon_kazanc(e, v)
    p(f"{b:8}{Q:>10.4f}{k:>12.5f}{g:>12.6f}")

p()
p("=" * 100)
p("C) YENI TRAFOLARIN ANATOMISI (blok bazinda)")
p("=" * 100)
for b in BLOKLAR:
    d = bl[b]
    e, yeni = ozet[b]
    sub = d[yeni]
    p(f"--- {b}: {int(yeni.sum()):,} satir, {sub['tanim'].nunique():,} trafo ---")
    p(
        "   soguk pay %.3f   gercek sifir pay %.3f   ort ly %.3f   ort lp %.3f"
        % (sub["soguk"].mean(), (sub["tuketim"] == 0).mean(), sub["ly"].mean(), sub["lp"].mean())
    )
    p(
        "   eski taraf: soguk pay %.3f  sifir pay %.3f  ort ly %.3f  ort lp %.3f"
        % (
            d[~yeni]["soguk"].mean(),
            (d[~yeni]["tuketim"] == 0).mean(),
            d[~yeni]["ly"].mean(),
            d[~yeni]["lp"].mean(),
        )
    )
    # sicak/soguk ayri
    for ad, m in [("sicak", ~sub["soguk"].to_numpy()), ("soguk", sub["soguk"].to_numpy())]:
        if m.sum():
            p("     yeni-%s n=%d ort artik %+0.5f" % (ad, m.sum(), e[yeni][m].mean()))

p()
p("=" * 100)
p("D) TEST TARAFINDA GRUBUN BUYUKLUGU")
p("=" * 100)
te = pd.read_csv(KOK + r"\data\raw\test.csv", dtype={"tanim": str})
tr_set = set(ham["tanim"].unique())
yeni_te = ~te["tanim"].isin(tr_set)
p("test yeni satir: %d / %d = %.4f" % (yeni_te.sum(), len(te), yeni_te.mean()))
v83 = pd.read_csv(KOK + r"\submissions\tuketim_v83_sicak_optimum.csv")
lp83 = np.log1p(v83["tuketim"].to_numpy())
p("v83 ort log1p: yeni %.4f  eski %.4f" % (lp83[yeni_te].mean(), lp83[~yeni_te].mean()))
p("Q(1[yeni]) test = %.5f" % (yeni_te.mean()))
