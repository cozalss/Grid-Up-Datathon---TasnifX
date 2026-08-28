"""Hata KUTLESININ tam ayristirmasi: olu/diri x soguk/sicak."""

import numpy as np
import pandas as pd
from m1_geriteste import kes, yukle

tr = yukle()
for kesim in ["2025-10-31", "2025-11-30"]:
    gec, hed = kes(tr, kesim)
    gec = gec.copy()
    gec["ly"] = np.log1p(gec.tuketim)
    hed = hed.copy()
    hed["ly"] = np.log1p(hed.tuketim)
    g0 = gec.ly.mean()
    tm28 = gec[gec.tarih > pd.Timestamp(kesim) - pd.Timedelta(days=28)].groupby("tanim").ly.mean()
    tmall = gec.groupby("tanim").ly.mean()
    gucm = gec.groupby("guc").ly.mean()
    p = np.where(
        hed.soguk.values,
        hed.guc.map(gucm).fillna(g0).values,
        hed.tanim.map(tm28).fillna(hed.tanim.map(tmall)).fillna(g0).values,
    )
    L = (p - hed.ly.values) ** 2
    tot = L.mean()
    # trafo duzeyinde olu tanimi: hedef penceredeki ortalama log < 0.5
    lvl = hed.groupby("tanim").ly.mean()
    hed["olu_hedef"] = hed.tanim.map(lvl) < 0.5
    print(f"\n===== {kesim}  taban RMSLE {np.sqrt(tot):.4f} =====")
    print(f"{'dilim':28s} {'satir':>9s} {'%satir':>7s} {'RMSLE':>8s} {'%kutle':>7s}")
    for ad, m in [
        ("SOGUK x OLU", hed.soguk & hed.olu_hedef),
        ("SOGUK x DIRI", hed.soguk & ~hed.olu_hedef),
        ("SICAK x OLU", ~hed.soguk & hed.olu_hedef),
        ("SICAK x DIRI", ~hed.soguk & ~hed.olu_hedef),
    ]:
        m = m.values
        print(
            f"{ad:28s} {m.sum():9,d} {100 * m.mean():6.1f}% {np.sqrt(L[m].mean()):8.4f} {100 * L[m].sum() / L.sum():6.1f}%"
        )
    # sicak tarafta: gecmiste diri ama hedefte olu (= olen trafolar)
    gecl = gec.groupby("tanim").ly.mean()
    hed["gec_seviye"] = hed.tanim.map(gecl)
    sc = ~hed.soguk.values
    olen = sc & (hed.olu_hedef.values) & (hed.gec_seviye.values > 0.5)
    uyanan = sc & (~hed.olu_hedef.values) & (hed.gec_seviye.values < 0.5)
    print(
        f"  -> SICAK 'olen' (gecmis diri, hedef olu): {olen.sum():,} satir, kutle %{100 * L[olen].sum() / L.sum():.1f}"
    )
    print(
        f"  -> SICAK 'uyanan' (gecmis olu, hedef diri): {uyanan.sum():,} satir, kutle %{100 * L[uyanan].sum() / L.sum():.1f}"
    )
