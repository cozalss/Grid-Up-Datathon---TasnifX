# -*- coding: utf-8 -*-
"""SPAN-DISI PAY: bir cep yonunun 30-gonderim span'i (+3 olculmus dik yon) disinda
   kalan norm payi. rho_LB(u) = rho_cv_beklentisi * sqrt(span_disi_pay) (en iyi hal).
   Span icindeki her sey ZATEN fiyatlanmis -> YENI kazanc SIFIR."""
import os, sys, json
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
CIK = r"C:/Users/Cem/AppData/Local/Temp/claude/c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/d8509f77-6f9b-4e1d-b980-62e299ed4fc5/scratchpad"

V = np.load(os.path.join(PK, "p34_V30.npy"))          # N x 30
BAZ = np.load(os.path.join(PK, "p34_dik_baz.npy"))    # k x N
a0 = np.load(os.path.join(PK, "p34_a0.npy"))
N = V.shape[0]
print("V30", V.shape, "BAZ", BAZ.shape, "N", N)
A = np.hstack([V, BAZ.T]).astype(np.float64)
print("A", A.shape)
G = (A.T @ A) / N
# sayisal kararlilik: kucuk ridge
lam = 1e-10 * np.trace(G) / G.shape[0]
Ginv = np.linalg.pinv(G + lam * np.eye(G.shape[0]), rcond=1e-12)

te = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"))
assert len(te) == N, (len(te), N)
print("test okundu", te.shape)


def span_disi(u, ad):
    u = np.asarray(u, dtype=np.float64)
    u = u - 0.0
    n2 = float((u * u).mean())
    if n2 <= 0:
        return None
    c = Ginv @ ((A.T @ u) / N)
    up = u - A @ c
    p2 = float((up * up).mean())
    return dict(ad=ad, norm2=n2, span_disi_norm2=p2, pay=p2 / n2,
                carpan=float(np.sqrt(max(p2 / n2, 0.0))))


yonler = []
one = np.ones(N)
yonler.append((one, "SABIT (global seviye)"))
yonler.append((te.soguk_mu.values.astype(float), "soguk kohort gostergesi"))
yonler.append((1.0 - te.soguk_mu.values.astype(float), "sicak kohort gostergesi"))
for c in ("t_olu_mu",):
    if c in te.columns:
        yonler.append((te[c].fillna(0).values.astype(float), c + " gostergesi"))
sf = te.t_sifir_orani.fillna(0).values.astype(float)
for e in (0.5, 0.8, 0.9, 0.99):
    yonler.append(((sf > e).astype(float), "t_sifir_orani>%.2f" % e))
yonler.append((sf, "t_sifir_orani (surekli)"))
if "t_kuyruk_sifir" in te.columns:
    kq = te.t_kuyruk_sifir.fillna(0).values.astype(float)
    yonler.append((kq, "t_kuyruk_sifir (surekli)"))
    yonler.append(((kq > 30).astype(float), "t_kuyruk_sifir>30"))
# olu VE soguk / olu VE sicak
olu = (sf > 0.9).astype(float)
yonler.append((olu * te.soguk_mu.values, "sifir-adayi & soguk"))
yonler.append((olu * (1 - te.soguk_mu.values), "sifir-adayi & sicak"))
# takvim
ay = pd.to_datetime(te.tarih).dt.month.values
for m in (4, 5, 6, 7):
    yonler.append(((ay == m).astype(float), "ay=%d" % m))
hg = pd.to_datetime(te.tarih).dt.dayofweek.values
yonler.append(((hg >= 5).astype(float), "hafta sonu"))
uf = te.ufuk_gun.values.astype(float)
yonler.append(((uf > 90).astype(float), "ufuk>90"))
yonler.append((uf, "ufuk (surekli)"))
# guc / seviye
gq = pd.qcut(te.guc.values.astype(float), 5, labels=False, duplicates="drop")
for i in range(int(np.nanmax(gq)) + 1):
    yonler.append(((gq == i).astype(float), "guc bandi %d/5" % (i + 1)))
yonler.append((a0, "a0 (taban tahmin seviyesi)"))
yonler.append(((a0 < 1.0).astype(float), "a0<1 (model dusuk diyor)"))
yonler.append(((a0 > 6.0).astype(float), "a0>6 (model yuksek diyor)"))
# bolge / ilce (en buyuk 5 ilce)
for il in pd.Series(te.ilce_key.astype(str)).value_counts().index[:3]:
    yonler.append(((te.ilce_key.astype(str).values == il).astype(float), "ilce=" + str(il)))

sonuc = []
for u, ad in yonler:
    s = span_disi(u, ad)
    if s:
        sonuc.append(s)
        print("%-34s norm2=%9.4f  span_disi_pay=%7.4f  carpan=%6.4f" % (
            ad[:34], s["norm2"], s["pay"], s["carpan"]))
with open(os.path.join(CIK, "k04_span_disi.json"), "w", encoding="utf-8") as f:
    json.dump(sonuc, f, indent=1)
print("YAZILDI k04_span_disi.json")
