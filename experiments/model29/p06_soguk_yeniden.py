"""p06: URETIM soguk uzmanini KOMSU SEVIYESI ozelligiyle YENIDEN EGIT.

Fikir 1'in NIHAI sinavi. p06_soguk_tani.py capayi tahmin uzerine afin olarak
ekleyip ~0 kazanc buldu, ama bu dogrusal bir sinavdi. Burada ozellik
modelin ICINE konuyor ve uretim yolu (deney.soguk_maskele oran=1,00 +
deney_ileri.egit_tahmin, kapasite ofseti) BIREBIR kullaniliyor.

Maliyeti dusuk tutmak icin yalnizca lgbm ailesi, 3 tohum. Karsilastirma
KENDI tabaniyla yapilir (ayni aile/tohum, ozellik YOK) -- uretimin
15 uyeli harmaniyla degil.

SIZINTI: egitim = yaz25 DISI bloklar; capa yaz25 icin 2025-01-01..03-31
ozet penceresinden, her blok icin kendi ozet penceresinden hesaplanir.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
sys.path.insert(0, os.path.join(KOK, "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402
from p06_soguk_tani import OZET, komsu_capa  # noqa: E402

t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


def rmsle(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def main():
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    log(f"cerceve hazir {egitim.shape}  kolon {len(kol)}")

    # --- capa kolonu: her blok kendi ozet penceresinden
    cp = {b: komsu_capa(b, k=8) for b in OZET}
    kk = np.full(len(egitim), np.nan)
    for b, m in cp.items():
        msk = (egitim._blok == b).to_numpy()
        kk[msk] = egitim.loc[msk, "tanim"].map(m).to_numpy(dtype="float64")
    egitim = egitim.assign(k_komsu=kk)
    log(f"capa hazir, kapsam {np.isfinite(kk).mean():.4f}")

    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, "yaz25")
    y = np.log1p(dogrulama[tm.HEDEF].to_numpy(dtype="float64").clip(0))[soguk]
    R = {}
    for ad, kols in (("taban", kol), ("komsu", kol + ["k_komsu"])):
        ps = []
        for tohum in (1000, 1001, 1002):
            maskeli = d.soguk_maskele(parca, kols, 1.00, tohum)
            ps.append(di.egit_tahmin("lgbm", maskeli, dogrulama, kols, tohum)[soguk])
            log(f"{ad} tohum {tohum}: RMSLE {rmsle(y - ps[-1]):.5f}")
        p = np.mean(ps, axis=0)
        R[ad] = dict(soguk_rmsle=round(rmsle(y - p), 5),
                     tohumlar=[round(rmsle(y - q), 5) for q in ps])
        np.save(os.path.join(BURA, f"p06_yeniden_{ad}.npy"), p)
        log(f"{ad} HARMAN soguk RMSLE {R[ad]['soguk_rmsle']:.5f}")
    R["kazanc"] = round(R["taban"]["soguk_rmsle"] - R["komsu"]["soguk_rmsle"], 5)
    with open(os.path.join(BURA, "p06_soguk_yeniden.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
