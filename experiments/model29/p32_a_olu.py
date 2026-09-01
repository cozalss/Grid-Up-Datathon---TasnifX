"""p32-A KATMAN 1: olu trafo kurali -- pencere x carpan izgarasi, 3 blok x 3 tohum.

Tezgah: p24_b_olc (URETIM sicak harmani, kohort agirlikli).
Kural W,c: son W kayitta max<=0 VE gecmis sifir orani>=%99 VE kuyruk
kesintisiz sifir serisi >=W  ->  tahmin exp uzayinda xc.

Cikti: p_kalici/p32_katmanlar.json ["K1_olu"]
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_ortak import BLOKLAR, PK, TOHUMLAR, gecmis_ozet, olu_maske

PENCERELER: tuple = (30, 90, 180, 270, 365, "tum")
CARPANLAR = (0.0, 0.25, 0.5, 0.75)
SICAK_PAY = PB.SICAK_PAY
N_ONY = 500


def uygula(p_log: np.ndarray, m: np.ndarray, c: float) -> np.ndarray:
    q = p_log.copy()
    q[m] = np.log1p(c * np.expm1(np.maximum(p_log[m], 0.0)))
    return np.maximum(q, 0.0)


def main() -> None:
    W_SAY = tuple(w for w in PENCERELER if w != "tum")
    OZ = gecmis_ozet(W_SAY)
    B = PB.veri_kur()

    R: dict = {
        "00_kural": "son W kayitta max<=0 VE gecmis sifir orani>=0.99 VE kuyruk sifir serisi>=W -> exp uzayinda xc",
        "01_tezgah": "p24_b_olc: URETIM sicak harmani (3,1,1,1.4), kohort agirlikli (agr), soguk taraf sabit",
        "02_gecmis_uzunlugu": {
            k: {
                "medyan_gecmis_gun": float(v["gecmis_gun"].median()),
                "maks_gecmis_gun": float(v["gecmis_gun"].max()),
            }
            for k, v in OZ.items()
        },
        "03_UYARI": (
            "train.csv 2025-01-01'de basliyor. yaz25 gecmisi ~90 gun, guz25 ~212, "
            "kis26 ~334, TEST ~455. W=365 YALNIZ test'te tanimli; W>gecmis olan "
            "bloklarda maske BOS kalir (kural gecmis_gun>=W sarti tasir)."
        ),
    }

    # --- maske sayilari
    kaps: dict = {}
    for W in PENCERELER:
        kaps[str(W)] = {}
        for b in BLOKLAR:
            tn = B[b]["tanim"]
            m = olu_maske(OZ[b], tn, W)
            kaps[str(W)][b] = {
                "n_satir": int(m.sum()),
                "n_trafo": int(pd.Series(tn[m]).nunique()) if m.sum() else 0,
                "pay": round(float(m.mean()), 5),
                "kesinlik_gercek_sifir": (
                    round(float((B[b]["y"][m] <= 0).mean()), 4) if m.sum() else None
                ),
                "ort_gercek": round(float(np.expm1(B[b]["y"][m]).mean()), 2) if m.sum() else None,
            }
    R["04_maske_kapsami"] = kaps

    # --- izgara: dMSE (pozitif = kural IYI), ham/agr + tohum hucreleri
    izgara: dict = {}
    for W in PENCERELER:
        for c in CARPANLAR:
            ad = f"W{W}_c{c}"
            hucre: dict = {}
            for b in BLOKLAR:
                bb = B[b]
                m = olu_maske(OZ[b], bb["tanim"], W)
                if m.sum() == 0:
                    hucre[b] = {"ham": 0.0, "agr": 0.0, "n": 0, "tohum": {}}
                    continue
                p0 = PB.harman(bb, PB.ADAYLAR["URETIM"])
                e0 = bb["y"] - p0
                e1 = bb["y"] - uygula(p0, m, c)
                th = {}
                for t in TOHUMLAR:
                    pt = PB.harman(bb, PB.ADAYLAR["URETIM"], t)
                    et0 = bb["y"] - pt
                    et1 = bb["y"] - uygula(pt, m, c)
                    th[str(t)] = round(
                        float((bb["w"] * (et0 * et0 - et1 * et1)).mean()), 6
                    )
                hucre[b] = {
                    "ham": round(float((e0 * e0 - e1 * e1).mean()), 6),
                    "agr": round(float((bb["w"] * (e0 * e0 - e1 * e1)).mean()), 6),
                    "n": int(m.sum()),
                    "tohum": th,
                }
            agr = [hucre[b]["agr"] for b in BLOKLAR]
            n9 = sum(
                1
                for b in BLOKLAR
                for t in TOHUMLAR
                if hucre[b]["tohum"].get(str(t), 0.0) > 0
            )
            izgara[ad] = {
                **{b: hucre[b]["agr"] for b in BLOKLAR},
                "ham": {b: hucre[b]["ham"] for b in BLOKLAR},
                "n": {b: hucre[b]["n"] for b in BLOKLAR},
                "ORT_agr": round(float(np.mean(agr)), 6),
                "isaret": f"{sum(v > 0 for v in agr)}/3",
                "tohum_isareti": f"{n9}/9",
                "test_dMSE": round(float(np.mean(agr)) * SICAK_PAY, 6),
                "tohum": {b: hucre[b]["tohum"] for b in BLOKLAR},
            }
    R["05_izgara"] = izgara

    # --- BLOK-DISI SECIM: her blok icin diger iki blokta en iyi (W,c), sonra o blokta olc
    bd: dict = {}
    for b in BLOKLAR:
        dis = [x for x in BLOKLAR if x != b]
        en_iyi, en_deg = None, -9e9
        for ad, v in izgara.items():
            s = float(np.mean([v[d] for d in dis]))
            if s > en_deg:
                en_deg, en_iyi = s, ad
        bd[b] = {
            "secilen": en_iyi,
            "dis_ort": round(en_deg, 6),
            "tutulan_blokta_agr": izgara[en_iyi][b],
        }
    R["06_blok_disi_secim"] = bd

    # --- KAZANAN: 3/3 isaret + en yuksek ORT_agr
    uygun = {k: v for k, v in izgara.items() if v["isaret"] == "3/3" and min(v["n"].values()) > 0}
    kaz = max(uygun, key=lambda k: uygun[k]["ORT_agr"]) if uygun else None
    R["07_KAZANAN"] = {
        "3_3_isaretli_aday_sayisi": len(uygun),
        "kazanan": kaz,
        "detay": izgara[kaz] if kaz else None,
    }

    # --- onyukleme (trafo kumeli) kazanan + referans W30_c0.5
    ony: dict = {}
    for ad in {x for x in (kaz, "W30_c0.5") if x}:
        W, c = ad[1:].split("_c")
        W = "tum" if W == "tum" else int(W)
        c = float(c)
        ony[ad] = {}
        for b in BLOKLAR:
            bb = B[b]
            m = olu_maske(OZ[b], bb["tanim"], W)
            if m.sum() == 0:
                ony[ad][b] = None
                continue
            p0 = PB.harman(bb, PB.ADAYLAR["URETIM"])
            e0 = bb["y"] - p0
            e1 = bb["y"] - uygula(p0, m, c)
            d2 = (e0 * e0 - e1 * e1) * bb["w"]
            s = pd.Series(d2).groupby(bb["tanim"])
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(320 + BLOKLAR.index(b))
            k = len(top)
            idx = rng.integers(0, k, size=(N_ONY, k))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            ony[ad][b] = {
                "ort": round(float(o.mean()), 6),
                "GA95": [round(float(np.quantile(o, 0.025)), 6), round(float(np.quantile(o, 0.975)), 6)],
                "P_pozitif": round(float((o > 0).mean()), 3),
            }
    R["08_onyukleme"] = ony

    # --- TEST maskesi
    T = pd.read_parquet(os.path.join(PB.DN, "test.parquet"))
    te_tanim = T["tanim"].astype(str).to_numpy()
    te_soguk = T["soguk_mu"].to_numpy() == 1
    tm: dict = {}
    for W in PENCERELER:
        m = olu_maske(OZ["TEST"], te_tanim, W)
        tm[str(W)] = {
            "n_satir": int(m.sum()),
            "n_trafo": int(pd.Series(te_tanim[m]).nunique()) if m.sum() else 0,
            "soguk_kesisim": int((m & te_soguk).sum()),
            "pay": round(float(m.mean()), 5),
        }
    R["09_test_maskesi"] = tm

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K1_olu"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "05_izgara"}, ensure_ascii=False, indent=1))
    print("--- izgara ORT_agr / isaret ---")
    for ad, v in izgara.items():
        print(f"{ad:14s} ORT={v['ORT_agr']:+.6f} {v['isaret']} tohum {v['tohum_isareti']} n={v['n']}")


if __name__ == "__main__":
    main()
