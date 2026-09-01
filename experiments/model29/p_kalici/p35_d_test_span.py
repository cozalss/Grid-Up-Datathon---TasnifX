"""p35-d: SIFIR YONUNUN TEST TARAFINDAKI OLCEGI ve SPAN-DISI PAYI.

CV rho'yu olctuk. Burada sorulan: eger bu yonu gercekten uygulasaydik
  (a) kappa (uygulama olcegi) ne olurdu,
  (b) yonun ne kadari 30 dosyalik SPAN'in ICINDE (yani zaten fiyatlanmis),
  (c) uretim modeli TEST'te bayrakli satirlara ne tahmin ediyor
      (CV'deki "zaten ~0 tahmin ediyor" olgusu testte de gecerli mi).
KAGGLE GONDERIMI YOK; hicbir dosya yazilmiyor.
"""
import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
GEC = os.path.dirname(os.path.abspath(__file__))
DN = os.path.join(KOK, "data/interim/deney")
R = {}

te_raw = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te_raw.id.values
sub = pd.read_csv(os.path.join(S, "tuketim_YP_seviye.csv"))
assert np.array_equal(sub.id.values, IDS), "id sirasi farkli"
p = np.log1p(sub.tuketim.values.astype(np.float64))

T = pd.read_parquet(os.path.join(DN, "test.parquet"))
assert np.array_equal(T.id.values, IDS), "test.parquet id sirasi farkli"
ks = np.nan_to_num(T.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
sg = T.soguk_mu.values == 1
print("TEST soguk pay:", round(float(sg.mean()), 4))

a0 = np.load(os.path.join(GEC, "p34_a0.npy"))
V = np.load(os.path.join(GEC, "p34_V30.npy"))
N = len(a0)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G)

tab = []
for e in (1, 3, 7, 14, 30):
    f = ks >= e
    dl = -p * f
    kap = float(np.sqrt(np.mean(dl * dl)))
    c = Gi @ ((V.T @ dl) / N)
    dperp = dl - V @ c
    kperp = float(np.sqrt(np.mean(dperp * dperp)))
    tab.append(dict(
        esik=e, pay=round(float(f.mean()), 5),
        soguk_pay_bayrakta=round(float(sg[f].mean()), 4),
        ort_p_bayrak=round(float(p[f].mean()), 4),
        rms_p_bayrak=round(float(np.sqrt(np.mean(p[f] ** 2))), 4),
        kappa_tam_sifirlama=round(kap, 5),
        kappa_span_disi=round(kperp, 5),
        span_disi_pay=round(kperp / kap, 4),
    ))
    print(tab[-1])
R["01_test_kuyruk"] = tab

# t_olu_mu
f = np.nan_to_num(T.t_olu_mu.values.astype(np.float64), nan=0.0) > 0.5
dl = -p * f
kap = float(np.sqrt(np.mean(dl * dl)))
c = Gi @ ((V.T @ dl) / N)
dperp = dl - V @ c
R["02_t_olu_mu"] = dict(pay=round(float(f.mean()), 5),
                        ort_p=round(float(p[f].mean()), 4),
                        kappa_tam=round(kap, 5),
                        kappa_span_disi=round(float(np.sqrt(np.mean(dperp ** 2))), 5),
                        span_disi_pay=round(float(np.sqrt(np.mean(dperp ** 2))) / kap, 4))
print("t_olu_mu:", R["02_t_olu_mu"])

# TEST vs yaz25 karsilastirmasi icin: yaz25 blogunda ayni bayragin payi
E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
for b in ("yaz25", "guz25", "kis26"):
    d = E[E._blok == b]
    k = np.nan_to_num(d.t_kuyruk_sifir.values.astype(np.float64), nan=-1.0)
    R[f"03_pay_{b}"] = dict(kuyruk1=round(float((k >= 1).mean()), 5),
                            olu=round(float((np.nan_to_num(
                                d.t_olu_mu.values.astype(float), nan=0.0) > 0.5).mean()), 5),
                            soguk=round(float((d.soguk_mu.values == 1).mean()), 4))
    print(b, R[f"03_pay_{b}"])
R["03_pay_TEST"] = dict(kuyruk1=round(float((ks >= 1).mean()), 5),
                        olu=round(float(f.mean()), 5), soguk=round(float(sg.mean()), 4))
print("TEST", R["03_pay_TEST"])

with open(os.path.join(GEC, "p35_d.json"), "w", encoding="utf-8") as fh:
    json.dump(R, fh, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p35_d.json")
