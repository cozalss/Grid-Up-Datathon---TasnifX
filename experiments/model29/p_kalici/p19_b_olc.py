"""p19-B: SOGUK CAT adaylarinin URETIM olcutunde degerlendirmesi.

URETIM SOGUK HARMANI = YALNIZ CAT. Esit harman KULLANILMIYOR.
On-kayit: experiments/model29/p_kalici/p19_soguk_cat.json anahtari 00_ON_KAYIT.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
from p11_agirlik import (
    PG_HAKIM,
    agirlik,
    ess,
    kova,
    onyukleme_w,
    rmsle,  # noqa: E402
    test_dagilimi,
    wrmsle,
)
from p11_ortak import BLOKLAR, DN, HEDEF_SOGUK, egitim, kirp, sicak_rmsle, toplam  # noqa: E402

TOHUMLAR = (1000, 1001, 1002)


def npy(b, t, a):
    return os.path.join(BURA, f"p19_{b}_{t}_{a}.npy")


def tohumlari(b, a):
    return [t for t in TOHUMLAR if os.path.exists(npy(b, t, a))]


def ort_log(b, a, ts):
    """URETIM gibi: LOG UZAYINDA tohum ortalamasi."""
    return np.mean([np.load(npy(b, t, a)).astype(np.float64) for t in ts], axis=0)


def soguk_satirlar(b):
    E = egitim()
    blk = E[E._blok == b]
    return blk[blk.soguk_mu == 1].reset_index(drop=True)


def main():
    adaylar = sys.argv[1:] or None
    T = pd.read_parquet(os.path.join(DN, "test.parquet"))
    q = test_dagilimi(T[T.soguk_mu == 1])
    del T

    R = {
        "harman": "URETIM = YALNIZ CAT",
        "hedef_soguk_pay": HEDEF_SOGUK,
        "bloklar": {},
        "kazanclar": {},
        "npz_cat_tohum_gurultusu": {},
    }
    D, MW = {}, {}

    for b in BLOKLAR:
        sog = soguk_satirlar(b)
        y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
        w = agirlik(sog, q)
        m = kova(sog)[0] == PG_HAKIM
        sic = sicak_rmsle(b)
        D[b] = dict(sog=sog, y=y, w=w, m=m, sic=sic)

        # mevcut adaylar
        var = sorted(
            {
                f[len(f"p19_{b}_") :].split("_", 1)[1][:-4]
                for f in os.listdir(BURA)
                if f.startswith(f"p19_{b}_") and f.endswith(".npy")
            }
        )
        if adaylar:
            var = [a for a in var if a in adaylar or a == "TABAN"]

        def olc(lg, y=y, w=w, m=m, sic=sic):
            r = y - kirp(lg)
            return dict(
                ham=round(rmsle(r), 5),
                agr=round(wrmsle(r, w), 5),
                pg=round(rmsle(r[m]), 5),
                pg_agr=round(wrmsle(r[m], w[m]), 5),
                bilesim_ham=round(toplam(rmsle(r), sic), 5),
                bilesim_agr=round(toplam(wrmsle(r, w), sic), 5),
            )

        R["bloklar"][b] = {
            "n_soguk": int(len(y)),
            "n_trafo": int(sog.tanim.nunique()),
            "sicak_rmsle": round(sic, 5),
            "agirlik_ess": round(ess(w), 1),
            "agirlik_max": round(float(w.max()), 2),
            "pg_hakim_satir": int(m.sum()),
            "pg_hakim_trafo": int(sog.loc[m, "tanim"].nunique()),
            "aday_seviye": {},
            "tohumlar": {},
        }
        MW[b] = {}
        for a in var:
            ts = tohumlari(b, a)
            if not ts:
                continue
            R["bloklar"][b]["tohumlar"][a] = ts
            lg = ort_log(b, a, ts)
            R["bloklar"][b]["aday_seviye"][a] = dict(tohum=ts, **olc(lg))

        # npz cat tohum gurultusu (uretim onbellegi, TABAN'in gurultu tabani)
        z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
        tek = {}
        for k in sorted(z.files):
            if not k.endswith("_cat"):
                continue
            r = y - kirp(z[k].astype(np.float64))
            tek[k.split("_")[0]] = dict(
                ham=round(rmsle(r), 5), agr=round(wrmsle(r, w), 5), pg=round(rmsle(r[m]), 5)
            )
        va = np.array([v["agr"] for v in tek.values()])
        vp = np.array([v["pg"] for v in tek.values()])
        R["npz_cat_tohum_gurultusu"][b] = dict(
            tekil=tek,
            n=len(va),
            agr_std=round(float(va.std(ddof=1)), 5),
            agr_menzil=round(float(va.max() - va.min()), 5),
            pg_std=round(float(vp.std(ddof=1)), 5),
            pg_menzil=round(float(vp.max() - vp.min()), 5),
        )

    # --- KAZANC TABLOSU (ORTAK tohum kumesi)
    for b in BLOKLAR:
        d = D[b]
        y, w, m, sic = d["y"], d["w"], d["m"], d["sic"]
        t0 = set(tohumlari(b, "TABAN"))
        if not t0:
            continue
        for a in R["bloklar"][b]["aday_seviye"]:
            if a == "TABAN":
                continue
            ortak = sorted(t0 & set(tohumlari(b, a)))
            if not ortak:
                continue
            r0 = y - kirp(ort_log(b, "TABAN", ortak))
            r1 = y - kirp(ort_log(b, a, ortak))
            MW[b][a] = float(np.sum(w * r1 * r1) / w.sum())
            MW[b].setdefault("TABAN", float(np.sum(w * r0 * r0) / w.sum()))
            R["kazanclar"].setdefault(a, {})[b] = dict(
                ortak_tohum=ortak,
                n_tohum=len(ortak),
                ham=round(rmsle(r0) - rmsle(r1), 5),
                agr=round(wrmsle(r0, w) - wrmsle(r1, w), 5),
                pg=round(rmsle(r0[m]) - rmsle(r1[m]), 5),
                bilesim_ham=round(toplam(rmsle(r0), sic) - toplam(rmsle(r1), sic), 5),
                bilesim_agr=round(toplam(wrmsle(r0, w), sic) - toplam(wrmsle(r1, w), sic), 5),
                tohum_bazli_agr={
                    str(t): round(
                        wrmsle(y - kirp(np.load(npy(b, t, "TABAN"))), w)
                        - wrmsle(y - kirp(np.load(npy(b, t, a))), w),
                        5,
                    )
                    for t in ortak
                },
                onyukleme_agr=onyukleme_w(d["sog"].tanim.values, r0, r1, w, 500),
                onyukleme_pg=onyukleme_w(
                    d["sog"].tanim.values[m], r0[m], r1[m], np.ones(int(m.sum())), 500
                ),
            )

    # --- BLOK-DISI SECIM (on-kayitta sabitlenmis prosedur)
    sec = {}
    for h in BLOKLAR:
        dis = [b for b in BLOKLAR if b != h and MW.get(b)]
        if len(dis) < 2:
            continue
        hepsi = set.intersection(*[set(MW[b]) for b in dis])
        puan = {a: sum(MW[b][a] / MW[b]["TABAN"] for b in dis) for a in hepsi}
        if len(puan) < 2:
            continue
        s = min(puan, key=puan.get)
        sec[h] = dict(
            secilen=s,
            puan={k: round(v, 5) for k, v in sorted(puan.items())},
            hedefte_kazanc=(None if s == "TABAN" else R["kazanclar"].get(s, {}).get(h)),
        )
    R["blok_disi_secim"] = sec

    yol = os.path.join(BURA, "p19_b_olc.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=1, ensure_ascii=False)

    print("\n=== SEVIYE (URETIM = YALNIZ CAT) ===")
    print(f"{'aday':16}{'blok':7}{'nt':>3}{'ham':>10}{'agr':>10}{'pg':>10}{'bil_agr':>10}")
    for b in BLOKLAR:
        for a, v in R["bloklar"][b]["aday_seviye"].items():
            print(
                f"{a:16}{b:7}{len(v['tohum']):>3}{v['ham']:>10.5f}{v['agr']:>10.5f}"
                f"{v['pg']:>10.5f}{v['bilesim_agr']:>10.5f}"
            )
    print("\n=== KAZANC (pozitif = IYI) ===")
    print(
        f"{'aday':16}{'blok':7}{'nt':>3}{'ham':>10}{'agr':>10}{'pg':>10}"
        f"{'bil_agr':>10}{'oy_agr+':>9}"
    )
    for a, bb in R["kazanclar"].items():
        for b, v in bb.items():
            print(
                f"{a:16}{b:7}{v['n_tohum']:>3}{v['ham']:>+10.5f}{v['agr']:>+10.5f}"
                f"{v['pg']:>+10.5f}{v['bilesim_agr']:>+10.5f}"
                f"{v['onyukleme_agr']['pozitif_oran']:>9.3f}"
            )
    print("\n=== TOHUM GURULTUSU (npz cat) ===")
    for b, v in R["npz_cat_tohum_gurultusu"].items():
        print(
            f"  {b:6} n={v['n']} agr std={v['agr_std']:.5f} menzil={v['agr_menzil']:.5f}"
            f" | pg std={v['pg_std']:.5f} menzil={v['pg_menzil']:.5f}"
        )
    print("\n=== BLOK-DISI SECIM ===")
    print(json.dumps(sec, indent=1, ensure_ascii=False))
    print("\nyazildi", yol)


if __name__ == "__main__":
    main()
