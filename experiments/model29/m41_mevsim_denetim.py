"""Model AY etkisini ogrenmis mi? Iki denetim:
(A) TEST: modelin ay bazinda ongordugu seviye kaymasi (son7 tabanina gore)
(B) DOGRULAMA: modelin hedef-ay bazinda ARTIK yanliligi (ogrenememisse burada gorunur)"""

import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK, yukle_ham

tr, te = yukle_ham()
sub = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m1_ileri_huber.csv"))
v102 = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
te = te.copy()
te["lp"] = np.log1p(sub.tuketim.values)
te["lp102"] = np.log1p(v102.tuketim.values)
te["soguk"] = ~te.tanim.isin(set(tr.tanim))
# son7 tabani (2026-03-25..03-31)
son7 = tr[tr.tarih > pd.Timestamp("2026-03-31") - pd.Timedelta(days=7)].groupby("tanim").ly.mean()
te["k7"] = te.tanim.map(son7)
sic = te[(~te.soguk) & te.k7.notna()].copy()
sic["kayma"] = sic.lp - sic.k7
sic["kayma102"] = sic.lp102 - sic.k7
print("=== (A) TEST: sicak trafolarda modelin son7'ye gore AY KAYMASI ===")
print("  (ajanin 2025 olcumu, Mart sonuna gore: Nis +0.007 / May -0.025 / Haz +0.302 / Tem +0.625)")
print(f"  {'ay':>4s} {'satir':>8s} {'YENI MODEL':>11s} {'v102':>9s}")
for m, g in sic.groupby(sic.tarih.dt.month):
    print(f"  {m:4d} {len(g):8,d} {g.kayma.mean():+11.4f} {g.kayma102.mean():+9.4f}")
print("\n  ayni olcum 2025 GERCEK (Mart sonu son7 tabanina gore, egitim verisinden):")
b25 = (
    tr[
        (tr.tarih > pd.Timestamp("2025-03-31") - pd.Timedelta(days=7))
        & (tr.tarih <= pd.Timestamp("2025-03-31"))
    ]
    .groupby("tanim")
    .ly.mean()
)
h25 = tr[(tr.tarih > "2025-03-31") & (tr.tarih <= "2025-07-31")].copy()
h25["k7"] = h25.tanim.map(b25)
h25 = h25[h25.k7.notna()]
for m, g in h25.groupby(h25.tarih.dt.month):
    print(f"  {m:4d} {len(g):8,d} {(g.ly - g.k7).mean():+11.4f}")
