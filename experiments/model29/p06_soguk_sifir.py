"""p06: SOGUK satirlarda SIFIR yapisi -- kare hatanin %57'si burada.

TANI. yaz25 hedefi yalnizca olcum icin okunur.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN  # noqa: E402
from p06_soguk_tani import HAM, META, OZET, hazirla  # noqa: E402


def rmsle(r):
    return float(np.sqrt(np.mean(np.asarray(r) ** 2)))


def main():
    R = {}
    for bad in ("yaz25", "guz25", "kis26"):
        d = hazirla(bad)
        s = d[d.soguk_mu == 1].copy()
        s["sf"] = (s.tuketim <= 0).astype(int)
        tb = s.groupby("tanim", observed=True).agg(n=("sf", "size"), q=("sf", "mean"))
        R[bad] = dict(
            trafo=int(len(tb)),
            sifir_satir=int(s.sf.sum()),
            trafo_tam_sifir=int((tb.q == 1).sum()),
            trafo_hic_sifir_yok=int((tb.q == 0).sum()),
            trafo_kismi=int(((tb.q > 0) & (tb.q < 1)).sum()),
            tam_sifir_satir=int(tb.loc[tb.q == 1, "n"].sum()),
            kismi_sifir_satir=int(s[s.tanim.isin(tb.index[(tb.q > 0) & (tb.q < 1)])].sf.sum()),
        )
        print(bad, R[bad], flush=True)

    # --- yaz25 detay: tam-sifir trafolar ne kadar hataya sebep?
    d = hazirla("yaz25")
    s = d[d.soguk_mu == 1].copy()
    s["sf"] = (s.tuketim <= 0).astype(int)
    tb = s.groupby("tanim", observed=True).agg(n=("sf", "size"), q=("sf", "mean"))
    tam = set(tb.index[tb.q == 1])
    m = s.tanim.isin(tam)
    tk = float((s.r**2).sum())
    R["yaz25_detay"] = dict(
        tam_sifir_trafo=len(tam),
        tam_sifir_satir=int(m.sum()),
        kare_pay=round(float((s.loc[m, "r"] ** 2).sum()) / tk, 4),
        tam_sifir_p_ort=round(float(s.loc[m, "p"].mean()), 4),
    )
    # tam-sifir trafolari MUKEMMEL bilsek
    p2 = s.p.values.copy()
    p2[m.values] = 0.0
    R["yaz25_detay"]["tam_sifir_mukemmel_rmsle"] = round(rmsle(s.y.values - p2), 5)
    # tam-sifir trafolari 'yumusak' bassak (p *= c)
    for c in (0.0, 0.2, 0.4, 0.6):
        p3 = s.p.values.copy()
        p3[m.values] *= c
        R["yaz25_detay"][f"tam_sifir_carpan_{c}"] = round(rmsle(s.y.values - p3), 5)
    print(json.dumps(R["yaz25_detay"], ensure_ascii=False, indent=1))

    # --- ONGORULEBILIRLIK: tam-sifir trafolar neye benziyor?
    E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    kol = [
        "guc",
        "ilce_key",
        "tanim_num",
        "p_gun_sayisi",
        "p_ilk_ofset",
        "p_son_ofset",
        "p_yayilma",
        "p_doluluk",
        "g_ilce_log_ort",
        "g_ilce_kova_ort",
        "yas",
        "ilk_gun_mu",
    ]
    kol = [c for c in kol if c in E.columns]
    tmeta = E.loc[s.index, kol].assign(tanim=s.tanim.values, sf=s.sf.values)
    g = tmeta.groupby("tanim", observed=True).agg(
        {**{c: "first" for c in kol if c != "ilce_key"}, "sf": "mean"}
    )
    R["tam_sifir_profil"] = {
        c: dict(
            tam_sifir=round(float(g.loc[g.sf == 1, c].median()), 3),
            digerleri=round(float(g.loc[g.sf < 1, c].median()), 3),
        )
        for c in g.columns
        if c != "sf" and g[c].notna().any()
    }
    print(json.dumps(R["tam_sifir_profil"], ensure_ascii=False, indent=1))

    # --- KOMSU SIFIR ORANI: ayni ilcede numaraca yakin trafolarin gecmisteki sifir orani
    from p06_soguk_tani import komsu_capa

    HAM["sfr"] = (HAM.tuketim <= 0).astype(float)
    for bad in ("yaz25",):
        cp = komsu_capa(bad, k=8, sut="sfr")
        gq = s.tanim.map(cp)
        ok = gq.notna().values
        R["komsu_sifir_orani"] = dict(
            kapsam=round(float(ok.mean()), 4),
            korr_sf=round(float(np.corrcoef(gq[ok], s.sf.values[ok])[0, 1]), 4),
            ort_tam_sifir=round(float(gq[(s.sf.values == 1) & ok].mean()), 4),
            ort_diger=round(float(gq[(s.sf.values == 0) & ok].mean()), 4),
        )
    print(json.dumps(R["komsu_sifir_orani"], ensure_ascii=False, indent=1))

    with open(os.path.join(BURA, "p06_soguk_sifir.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
