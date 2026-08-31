"""p06: RAMPA -- yeni devreye alinan trafonun yuku ilk gunlerde oturuyor mu?

kendi_gun = ufuk_gun - p_ilk_ofset  = satirin, o trafonun ISTENEN ilk
gununden kac gun sonrasi oldugu. Test.csv'den turetilebilir (SIZINTI DEGIL);
uretim cercevesinde AYRI bir kolon olarak YOK -- model iki kolonu cikarmak
zorunda.

Ayrica TOPLU GIRIS: ayni gun devreye giren trafo gruplari.
TANI + dis-blok kestirimli duzeltme.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN, blok, skor  # noqa: E402

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
KEN = [0, 1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90]


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def hz(bad):
    d = blok(bad)
    d = d[d.soguk_mu == 1].copy()
    d["kendi"] = E.loc[d.index, "ufuk_gun"].values - E.loc[d.index, "p_ilk_ofset"].values
    return d


def tablo(d):
    k = np.digitize(d.kendi.values, KEN)
    out = []
    for kk in sorted(set(k)):
        m = k == kk
        out.append(dict(kova=int(kk), n=int(m.sum()),
                        kendi_ort=round(float(d.kendi.values[m].mean()), 1),
                        yanlilik=round(float(d.r.values[m].mean()), 4),
                        rmsle=round(rmsle(d.r.values[m]), 4),
                        sifir=round(float((d.tuketim.values[m] <= 0).mean()), 4),
                        kare_pay=round(float((d.r.values[m] ** 2).sum() / (d.r.values**2).sum()), 4)))
    return out


def main():
    R = {"kenarlar": KEN}
    ys, gs, ks = hz("yaz25"), hz("guz25"), hz("kis26")
    for ad, d in (("yaz25", ys), ("guz25", gs), ("kis26", ks)):
        R[ad] = tablo(d)
        print("---", ad)
        print(pd.DataFrame(R[ad]).to_string(index=False), flush=True)

    # --- TOPLU GIRIS: ilk gun dagilimi
    for ad, d in (("yaz25", ys), ("guz25", gs), ("kis26", ks)):
        ilk = d.groupby("tanim", observed=True).tarih.min()
        vc = ilk.value_counts().sort_values(ascending=False).head(5)
        R[f"{ad}_toplu_giris"] = [dict(tarih=str(pd.Timestamp(t).date()), trafo=int(n)) for t, n in vc.items()]
        print(ad, "en kalabalik giris gunleri:", R[f"{ad}_toplu_giris"])

    tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
    ts = tp[tp.soguk_mu == 1]
    ilk = ts.groupby("tanim", observed=True).tarih.min()
    vc = ilk.value_counts().sort_values(ascending=False).head(5)
    R["test_toplu_giris"] = [dict(tarih=str(pd.Timestamp(t).date()), trafo=int(n)) for t, n in vc.items()]
    print("TEST en kalabalik giris gunleri:", R["test_toplu_giris"])

    # --- DUZELTME: kendi_gun kovasi ofseti, DIS bloklardan kestirilir
    dis = pd.concat([gs, ks])
    kd = np.digitize(dis.kendi.values, KEN)
    ky = np.digitize(ys.kendi.values, KEN)
    ofs = np.zeros(len(KEN) + 1)
    for kk in range(len(ofs)):
        m = kd == kk
        if m.sum() >= 200:
            ofs[kk] = float(dis.r.values[m].mean())
    ofs -= float(dis.r.mean())  # kuresel yanliligi ayir; yalnizca SEKIL tasinir
    R["dis_ofset_sekli"] = [round(v, 4) for v in ofs]
    R["duzeltme"] = {}
    for lam in (0.25, 0.5, 1.0):
        p2 = ys.p.values + lam * ofs[ky]
        R["duzeltme"][f"lam={lam}"] = round(rmsle(ys.y.values - p2), 5)
    R["duzeltme"]["taban"] = round(rmsle(ys.r), 5)
    print(json.dumps(R["duzeltme"], indent=1))

    with open(os.path.join(BURA, "p06_soguk_rampa.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
