# -*- coding: utf-8 -*-
"""YON 4 s6: ETIKETSIZ PANEL yonleri -- soguk trafonun KENDI tahmin panelinden
turetilen yonler (test'te de birebir hesaplanabilir; etiket gerektirmez).
  - trafo-ici gunluk sapmayi trafo ortalamasina buzme / yayma
  - gun profilini (tum soguk trafolarin o gunku ortalamasi) buzme
  - trafo x haftanin gunu sekli
"""
import json
import os
import numpy as np
import pandas as pd
from ortak import blok, ezber_maskesi, rho_olc, BLOKLAR, KOK
import p27_ortak as P

SP = os.path.dirname(os.path.abspath(__file__))


def panel_yonleri(tanim, tarih, hg, p, sg, w):
    """soguk-only delta sozlugu; hepsi yalniz p ve kimlik/takvimden turetilir."""
    n = len(p)
    out = {}
    df = pd.DataFrame({"t": tanim, "g": tarih, "hg": hg, "p": p, "w": w, "s": sg})
    dc = df[df.s]
    # trafo ortalamasi
    tm = dc.groupby("t").p.transform("mean")
    dev_t = np.zeros(n); dev_t[sg.nonzero()[0]] = (dc.p - tm).values
    out["panel_trafoici_BUZME"] = -dev_t
    out["panel_trafoici_YAYILMA"] = dev_t
    # gun ortalamasi (soguk kohortun o gunku seviyesi)
    gm = dc.groupby("g").p.transform("mean")
    dev_g = np.zeros(n); dev_g[sg.nonzero()[0]] = (dc.p - gm).values
    out["panel_gunici_BUZME"] = -dev_g
    # iki yonlu: trafo ve gun etkileri cikarildiktan sonra kalan etkilesim
    genel = dc.p.mean()
    art = np.zeros(n)
    art[sg.nonzero()[0]] = (dc.p - tm - gm + genel).values
    out["panel_etkilesim_BUZME"] = -art
    # trafo x haftanin gunu sekli
    thm = dc.groupby(["t", "hg"]).p.transform("mean")
    sek = np.zeros(n); sek[sg.nonzero()[0]] = (thm - tm).values
    out["panel_hg_sekli_BUZME"] = -sek
    # gunluk seviye profili (trafo ortalamasina gore gunun ortak kaymasi)
    gk = np.zeros(n); gk[sg.nonzero()[0]] = (gm - genel).values
    out["panel_gun_profili_BUZME"] = -gk
    return out


def main():
    R = {}
    EZ = {}
    print("=" * 100)
    print("YON 4 s6 -- ETIKETSIZ PANEL YONLERI (soguk-only, merkezsiz cunku hepsi zaten sifir ortalamali)")
    print("%-28s | %-21s | %-21s | %-21s" % ("YON", "yaz25", "guz25", "kis26"))
    print("-" * 100)
    kesif = {}
    for b in BLOKLAR:
        d = blok(b)
        w = P.agirlik(d)
        sg = (d.soguk_mu.values == 1)
        A = panel_yonleri(d.tanim.values, pd.to_datetime(d.tarih).values,
                          d.hg.values, d.p.values.astype(np.float64), sg, w)
        ez = ezber_maskesi(b)
        tmz = sg & (~ez)
        kesif[b] = {}
        for ad, delta in A.items():
            o = rho_olc(d, delta, w)
            ot = rho_olc(d, np.where(tmz, delta, 0.0), w)
            kesif[b][ad] = dict(rho=round(o["rho"], 5), se=round(o["se"], 5),
                                t=round(o["t"], 2), rho_temiz=round(ot["rho"], 5),
                                yon_rms=round(o["norm"], 5))
        R[b] = kesif[b]
    adlar = list(R["yaz25"].keys())
    for ad in adlar:
        h = ["%+.4f+-%.4f t%+.1f" % (R[b][ad]["rho"], R[b][ad]["se"], R[b][ad]["t"]) for b in BLOKLAR]
        print("%-28s | %-21s | %-21s | %-21s" % (ad, h[0], h[1], h[2]))
    print("\nTEMIZ alt kume (p30):")
    for ad in adlar:
        h = ["%+.4f" % R[b][ad]["rho_temiz"] for b in BLOKLAR]
        print("  %-28s yaz25 %s  guz25 %s  kis26 %s" % (ad, h[0], h[1], h[2]))
    print("\nYON RMS (buyukluk, log birimi):")
    for ad in adlar:
        print("  %-28s %s" % (ad, "  ".join("%s %.4f" % (b, R[b][ad]["yon_rms"]) for b in BLOKLAR)))

    with open(os.path.join(SP, "s6_panel.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1)
    print("\nyazildi: s6_panel.json")


if __name__ == "__main__":
    main()
