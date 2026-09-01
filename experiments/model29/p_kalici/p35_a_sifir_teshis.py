"""p35-a: SIFIR CEBI TESHISI -- docs/82 bolum 4 ve bolum 6 celiskisini olc.

Soru: bolum 6 "sifir TUM cep MSE'nin %41-53'unu tasiyor" derken
bolum 4 "dogru yakalananlari sifirlamanin kahin degeri +0.0005..+0.0052"
diyor.  Bunlar bagdasmiyor.  Hangisi dogru?

Hicbir sey egitilmiyor: yalniz muhasebe + hazir t_* ozetleri (hepsi
hedef blogun BASINDAN ONCEKI ozet penceresinden geliyor, sizinti yok).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from p27_ortak import HEDEF_SOGUK, agirlik, blok, rmsle  # noqa: E402

CIK = os.path.dirname(os.path.abspath(__file__))
BLOKLAR = ("yaz25", "guz25", "kis26")
R = {}


def wmse(r, w):
    return float(np.sum(w * r * r) / np.sum(w))


print("bloklar kuruluyor (URETIM harmani, soguk=cat, son_islem=True)...")
D = {b: blok(b, soguk_harman="cat", son_islem=True) for b in BLOKLAR}

tab = []
for b in BLOKLAR:
    d = D[b]
    w = agirlik(d)
    r = d.r.values
    p = d.p.values
    y = d.y.values
    tuk = d.tuketim.values.astype(np.float64)
    z = tuk <= 0.0
    sg = d.soguk_mu.values == 1
    M = wmse(r, w)

    # --- olu trafo: blokta HIC uretmeyen
    gz = pd.DataFrame(dict(t=d.tanim.values, z=z))
    tam_olu = gz.groupby("t").z.mean() == 1.0
    olu_satir = tam_olu.reindex(d.tanim.values).to_numpy()

    # --- kahin: TUM sifir satirlari mukemmel bilsek
    p0 = p.copy()
    p0[z] = 0.0
    # --- kahin: yalniz OLU TRAFO satirlari
    p1 = p.copy()
    p1[olu_satir] = 0.0

    tab.append(dict(
        blok=b, n=len(d),
        rmsle_agirlikli=round(float(rmsle(y, p, w)), 5),
        sifir_satir_pay_ham=round(float(z.mean()), 5),
        sifir_satir_pay_agir=round(float(np.sum(w * z) / np.sum(w)), 5),
        sifir_MSE_payi=round(float(np.sum(w[z] * r[z] ** 2) / np.sum(w * r * r)), 4),
        sifir_ort_p=round(float(np.average(p[z], weights=w[z])), 4),
        sifir_ort_r=round(float(np.average(r[z], weights=w[z])), 4),
        sifir_rms_r=round(float(np.sqrt(np.average(r[z] ** 2, weights=w[z]))), 4),
        sifir_soguk_payi=round(float(np.average(sg[z], weights=w[z])), 4),
        # blokta hic uretmeyen trafolar
        olu_trafo_n=int(tam_olu.sum()), trafo_n=int(len(tam_olu)),
        olu_satir_pay_agir=round(float(np.sum(w * olu_satir) / np.sum(w)), 5),
        olu_satir_ort_p=round(float(np.average(p[olu_satir], weights=w[olu_satir])), 4),
        olu_MSE_payi=round(float(np.sum(w[olu_satir] * r[olu_satir] ** 2)
                                 / np.sum(w * r * r)), 4),
        sifirlarin_olu_payi=round(float(np.sum(w[z & olu_satir]) / np.sum(w[z])), 4),
        # kahin
        kahin_tum_sifir=round(float(rmsle(y, p0, w)), 5),
        kahin_tum_sifir_kazanc=round(float(rmsle(y, p, w) - rmsle(y, p0, w)), 5),
        kahin_olu_trafo=round(float(rmsle(y, p1, w)), 5),
        kahin_olu_trafo_kazanc=round(float(rmsle(y, p, w) - rmsle(y, p1, w)), 5),
    ))
    print(tab[-1])

R["01_taban"] = tab
print("\nORT rmsle:", np.mean([x["rmsle_agirlikli"] for x in tab]))

# ------------------------------------------------------------------
# 2) BASIT OLU BAYRAK (hicbir sey egitilmiyor): t_kuyruk_sifir esigi
#    docs/82 bolum 4 "dogru yakalananlari sifirlamanin kahin degeri
#    yalnizca +0.0005..+0.0052" iddiasini SINA.
# ------------------------------------------------------------------
print("\n--- 2) t_kuyruk_sifir bayragi: dogru-yakalananlari sifirlamanin kahin degeri")
tab2 = []
for b in BLOKLAR:
    d = D[b]
    w = agirlik(d)
    p, y, r = d.p.values, d.y.values, d.r.values
    z = d.tuketim.values.astype(np.float64) <= 0.0
    ks = d.t_kuyruk_sifir.values.astype(np.float64)
    for esik in (1, 3, 7, 14, 30):
        f = np.nan_to_num(ks, nan=-1.0) >= esik
        if f.sum() == 0:
            continue
        # yalniz DOGRU yakalananlari sifirla (kahin)
        p2 = p.copy()
        p2[f & z] = 0.0
        # bayragin TAMAMINI sifirla (gercekci uygulama)
        p3 = p.copy()
        p3[f] = 0.0
        tab2.append(dict(
            blok=b, esik=esik,
            bayrak_pay=round(float(np.sum(w * f) / np.sum(w)), 5),
            kesinlik=round(float(np.sum(w[f] * z[f]) / np.sum(w[f])), 4),
            duyarlilik=round(float(np.sum(w[f & z]) / np.sum(w[z])), 4),
            ort_p_bayrak=round(float(np.average(p[f], weights=w[f])), 3),
            ort_y_yanlis_poz=round(float(np.average(y[f & ~z], weights=w[f & ~z]))
                                   if (f & ~z).sum() else float("nan"), 3),
            kahin_dogruyu_sifirla=round(float(rmsle(y, p, w) - rmsle(y, p2, w)), 5),
            hepsini_sifirla=round(float(rmsle(y, p, w) - rmsle(y, p3, w)), 5),
        ))
        print("  ", tab2[-1])
R["02_kuyruk_bayrak"] = tab2

with open(os.path.join(CIK, "p35_a.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1, default=str)
print("\nyazildi p35_a.json")
