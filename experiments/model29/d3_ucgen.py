"""DUSMANCA DENETIM -- 3. asama: harman ozdesliginin GECMIS uzerinde sinanmasi.
Olculmus skoru olan tum dosyalar arasinda "c = a + k(b-a)" ucgenlerini bulur,
her biri icin ongorulen optimum ile GERCEKLESEN LB skorunu karsilastirir.
Boylece "ongoru hatasinin" gercek dagilimi olculur (public/private, yuvarlama, hepsi dahil).
Ayrica m4'un kirpma kaybini ve stale Q'yu belgeler. Sadece OKUR.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(KOK, "submissions")
SK = json.load(open(os.path.join(BURA, "olculmus_skorlar.json")))
R = {}

var = {k: v for k, v in SK.items() if os.path.exists(os.path.join(SUB, k))}
print(f"olculmus {len(SK)}, diskte {len(var)}")
L = {}
for f in var:
    _df = pd.read_csv(os.path.join(SUB, f))
    if len(_df) != 714688:
        continue
    L[f] = np.log1p(_df.iloc[:, 1].values)
N = len(next(iter(L.values())))
var = {k: v for k, v in var.items() if k in L}
ad = sorted(var)
idx = np.random.default_rng(1).choice(N, 4000, replace=False)

ucgen = []
for c in ad:
    for a_, b_ in itertools.combinations([x for x in ad if x != c], 2):
        A, B, C = L[a_][idx], L[b_][idx], L[c][idx]
        db = B - A
        m = np.abs(db) > 0.05
        if m.sum() < 500:
            continue
        k = (C - A)[m] / db[m]
        if not (np.nanstd(k) < 1e-9 and 0.0 < np.nanmean(k) < 1.0):
            continue
        kk = float(np.nanmean(k))
        # tam dogrulama
        fark = float(np.abs(L[a_] + kk * (L[b_] - L[a_]) - L[c]).max())
        if fark > 1e-9:
            continue
        Q = float(((L[b_] - L[a_]) ** 2).mean())
        m0 = var[a_] ** 2
        m1 = var[b_] ** 2
        Lv = (m0 + Q - m1) / 2
        kyildiz = Lv / Q
        ong = float(np.sqrt(max(m0 - Lv**2 / Q, 0)))
        ger = var[c]
        ucgen.append(
            dict(
                taban=a_,
                yon=b_,
                harman=c,
                k_dosyada=kk,
                k_optimum=float(kyildiz),
                optimumda_mi=bool(abs(kk - kyildiz) < 5e-5),
                Q=Q,
                ongorulen_optimum=ong,
                gerceklesen=ger,
                sapma=float(ger - ong),
                maks_log_fark=fark,
            )
        )
print(f"ucgen sayisi: {len(ucgen)}")
for u in ucgen:
    print(
        f"  {u['harman'][:34]:34s} = {u['taban'][:26]:26s} + {u['k_dosyada']:.5f}*({u['yon'][:26]:26s})"
        f"  ongoru {u['ongorulen_optimum']:.5f} gercek {u['gerceklesen']:.5f} sapma {u['sapma']:+.6f}"
        f"  {'(optimum k)' if u['optimumda_mi'] else '(k optimum DEGIL)'}"
    )
opt_u = [u for u in ucgen if u["optimumda_mi"]]
R["ucgenler"] = ucgen
R["ozet"] = dict(
    toplam=len(ucgen),
    optimum_k_ile=len(opt_u),
    sapmalar=[u["sapma"] for u in opt_u],
    sapma_mutlak_maks=float(max([abs(u["sapma"]) for u in opt_u], default=0)),
    NOT="k optimum degilse ongoru zaten optimum icin, kiyas anlamsiz",
)

# ---- m4'un KIRPMA kaybi: dosyadaki Q vs kirpilmamis Q
V = pd.read_csv(os.path.join(SUB, "tuketim_v102_kappa_optimum.csv")).tuketim.values
M4 = pd.read_csv(os.path.join(SUB, "tuketim_m4_hava_capali.csv")).tuketim.values
a = np.log1p(V)
b = np.log1p(M4)
kirp = M4 <= 0
R["m4_kirpma"] = dict(
    sifir_satir=int(kirp.sum()),
    yuzde=float(100 * kirp.mean()),
    Q_dosyadan=float(((b - a) ** 2).mean()),
    Q_belgede_yazan=0.121581,
    fark=float(0.121581 - ((b - a) ** 2).mean()),
    kirpilan_satirlarin_Q_payi=float(((b - a)[kirp] ** 2).sum() / len(a)),
    kirpilan_v102_ort=float(V[kirp].mean()),
    kirpilan_v102_maks=float(V[kirp].max()),
    NOT=(
        "m71 Q'yu KIRPMADAN ONCEKI log uzayinda olcmus (0.121581); diske yazilan "
        "dosyanin Q'su 0.121396. docs/53 ve docs/54 ESKI degeri tasiyor. "
        "m92/m93/m94/m95 dosyadan yeniden hesapladigi icin ZINCIR TEMIZ."
    ),
)
if 0.121581 != 0:
    m0 = 1.00553**2
    for Q, et in ((float(((b - a) ** 2).mean()), "dosya"), (0.121581, "belge")):
        Lv = (m0 + Q - 1.043**2) / 2
        R["m4_kirpma"][f"L_{et}"] = Lv
        R["m4_kirpma"][f"kappa_{et}"] = Lv / Q

json.dump(R, open(os.path.join(BURA, "d3_ucgen.json"), "w", encoding="utf-8"), indent=1)
print("\nYAZILDI d3_ucgen.json")
print(json.dumps(R["m4_kirpma"], indent=1))
print(json.dumps(R["ozet"], indent=1))
