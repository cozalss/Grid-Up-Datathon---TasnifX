"""p06 tani: SOGUK satirlarda KOMSU SEVIYESI capasinin uretim hattinda degeri var mi?

FIKIR (p03'ten): bir soguk trafonun gecmisi yok, ama AYNI ILCEDE numarasi
(tanim) yakin olan trafolarin gecmisi var. Onlarin ortalama log seviyesi
soguk trafonun seviyesi icin bir capadir.

SIZINTI KONTROLU
  - capa, o BLOGUN ozet penceresinden hesaplanir (yaz25 icin 2025-01-01..03-31);
    blogun kendi etiket satirlari capaya GIRMEZ, trafonun kendisi de girmez.
  - bu betik yalnizca TANI yapar: yaz25 hedefi sadece olcum icin okunur.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p02_duzeltme import DN, blok, skor  # noqa: E402

KOK = os.path.dirname(os.path.dirname(BURA))

#: blok -> ozet penceresi (tuketim_model.BLOKLAR / Blok.ozet_* ile birebir)
OZET = {
    "yaz25": ("2025-01-01", "2025-03-31"),
    "guz25": ("2025-01-01", "2025-07-31"),
    "kis26": ("2025-01-01", "2025-11-30"),
}

HAM = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), encoding="utf-8")
HAM["tarih"] = pd.to_datetime(HAM.tarih)
HAM["ly"] = np.log1p(HAM.tuketim.clip(lower=0).astype("float64"))
_ad = HAM.tanim.astype(str)
HAM["ilce"] = HAM.lokasyon.astype(str)
HAM["num"] = pd.to_numeric(_ad.where(_ad.str.fullmatch(r"\d+")), errors="coerce")
META = HAM.drop_duplicates("tanim")[["tanim", "ilce", "num"]].reset_index(drop=True)


def komsu_capa(bad, k=8, sut="ly"):
    """{tanim -> ayni ilcede numaraca en yakin k komsunun ortalama log seviyesi}."""
    b, s = OZET[bad]
    gec = HAM[(HAM.tarih >= b) & (HAM.tarih <= s)]
    lv = gec.groupby("tanim", observed=True)[sut].mean().rename("lv").reset_index()
    lv = lv.merge(META, on="tanim").dropna(subset=["num"])
    out = {}
    hedef = META.dropna(subset=["num"])
    for ilce, gh in hedef.groupby("ilce", observed=True):
        gl = lv[lv.ilce == ilce]
        if len(gl) < 2:
            continue
        o = np.argsort(gl.num.to_numpy())
        xs = gl.num.to_numpy()[o]
        vs = gl.lv.to_numpy()[o]
        ids = gl.tanim.to_numpy()[o]
        for tn, xv in zip(gh.tanim.to_numpy(), gh.num.to_numpy()):
            j = np.searchsorted(xs, xv)
            lo, hi = max(0, j - k), min(len(xs), j + k)
            sel = np.arange(lo, hi)
            sel = sel[ids[sel] != tn]
            if len(sel):
                out[tn] = float(vs[sel].mean())
    return out


def hazirla(bad, k=8):
    d = blok(bad)
    cp = komsu_capa(bad, k=k)
    d["capa"] = d.tanim.map(cp).astype("float64")
    return d


def rmsle(r):
    return float(np.sqrt(np.mean(np.asarray(r) ** 2)))


def main():
    R = {"aciklama": __doc__.strip().splitlines()[0]}
    yaz = hazirla("yaz25")
    sog = yaz[yaz.soguk_mu == 1]
    sic = yaz[yaz.soguk_mu == 0]
    t0, t0w = skor(yaz, yaz.p.values)
    R["taban"] = dict(
        yaz25_rmsle=round(t0, 5),
        test_bilesimi=round(t0w, 5),
        soguk_rmsle=round(rmsle(sog.r), 5),
        sicak_rmsle=round(rmsle(sic.r), 5),
        soguk_satir=int(len(sog)),
        soguk_trafo=int(sog.tanim.nunique()),
    )
    print(json.dumps(R["taban"], indent=1))

    # --- kapsam
    kap = {}
    for bad in ("yaz25", "guz25", "kis26"):
        d = hazirla(bad)
        s = d[d.soguk_mu == 1]
        kap[bad] = dict(
            soguk_satir=int(len(s)),
            capa_kapsami=round(float(s.capa.notna().mean()), 4),
            soguk_rmsle=round(rmsle(s.r), 5),
        )
        print(bad, kap[bad], flush=True)
    R["kapsam"] = kap

    # --- capa ile artik/hedef iliskisi (TANI; hedef okunur)
    s = sog[sog.capa.notna()].copy()
    R["iliski"] = dict(
        n=int(len(s)),
        korr_capa_y=round(float(np.corrcoef(s.capa, s.y)[0, 1]), 4),
        korr_capa_p=round(float(np.corrcoef(s.capa, s.p)[0, 1]), 4),
        korr_capa_artik=round(float(np.corrcoef(s.capa, s.r)[0, 1]), 4),
        korr_p_y=round(float(np.corrcoef(s.p, s.y)[0, 1]), 4),
        capa_ort=round(float(s.capa.mean()), 4),
        p_ort=round(float(s.p.mean()), 4),
        y_ort=round(float(s.y.mean()), 4),
    )
    print(json.dumps(R["iliski"], indent=1))

    # --- ORACLE ust sinir: capa uzerine en iyi afin donusum (yaz25'te FIT -- sizinti,
    #     yalnizca "ne kadar bilgi var" sorusuna ust sinir)
    X = np.c_[np.ones(len(s)), s.p.values, s.capa.values]
    cf = np.linalg.lstsq(X, s.y.values, rcond=None)[0]
    R["oracle_afin"] = dict(
        katsayi=[round(float(v), 4) for v in cf],
        soguk_rmsle=round(rmsle(s.y.values - X @ cf), 5),
        yalniz_p_rmsle=round(
            rmsle(
                s.y.values
                - np.c_[np.ones(len(s)), s.p.values]
                @ np.linalg.lstsq(np.c_[np.ones(len(s)), s.p.values], s.y.values, rcond=None)[0]
            ),
            5,
        ),
        uyari="yaz25 uzerinde fit -- UST SINIR, uretimde kullanilamaz",
    )
    print(json.dumps(R["oracle_afin"], indent=1))

    with open(os.path.join(BURA, "p06_soguk_tani.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, ensure_ascii=False, indent=1)
    print("yazildi p06_soguk_tani.json")


if __name__ == "__main__":
    main()
