"""p32-G: KUYRUK KAPAGI -- H-capa'nin buyuk-k ucu (yalnizca ucucu satirlar).

BULGU (p32-F/E)
---------------
H-capa kazanci k=12'de bile (satirlarin YALNIZ %0.35'i kirpilirken) duruyor:
kis26 +0.0183. Yani kazanc ince ayardan degil, BIR AVUC UC DEGERden geliyor.
Kucuk k'de (agresif kirpma) katman guz25'te test kohortuna
agirliklandirilinca ISARET DONDURUYOR (-0.034). Buyuk k'de ise UC BLOKTA DA
pozitif olabiliyor.

Bu betik k in {6,8,10,12,16,20,30} icin dort kapiyi da uygular:
 (a) gercek blok CV, URETIM harmani
 (b) blok-disi secim
 (c) uc blok isaret tutarliligi -- HEM ham HEM test-kohortlu
 (d) 3x3 tohum
+ trafo kumeli onyukleme, P(LB>=0.00628).

Cikti: p_kalici/p32_katmanlar.json ["K3e_kuyruk_kapagi"]
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_d_hcapa import wins_h
from p32_e_gecmis_uzunlugu import capa_ve_kapsam
from p32_f_test_kohort import hucre_gen
from p32_ortak import BLOKLAR, PK, TOHUMLAR

KLER = (6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 30.0)
YONLER = ("cift", "ust")
N_ONY = 4000
SICAK_PAY = PB.SICAK_PAY
HEDEF = 0.00628
BOLEN = 2 * 1.00115


def main() -> None:
    B = PB.veri_kur()
    E = pd.read_parquet(os.path.join(PB.DN, "egitim.parquet"))
    T = pd.read_parquet(os.path.join(PB.DN, "test.parquet"))
    ts = T[T["soguk_mu"] == 0]
    _, _, tn, tay = capa_ve_kapsam("TEST", ts["tanim"].astype(str).to_numpy())
    th = hucre_gen(
        ts["p_gun_sayisi"].to_numpy("float64"), ts["guc"].to_numpy("float64"),
        ts["ufuk_gun"].to_numpy("float64"), tn, tay,
    )
    test_pay = pd.Series(th).value_counts(normalize=True)

    CAPA, P0, PT, H = {}, {}, {}, {}
    for b in BLOKLAR:
        bb = B[b]
        mu, sd, n, ay = capa_ve_kapsam(b, bb["tanim"])
        CAPA[b] = (mu, sd)
        P0[b] = PB.harman(bb, PB.ADAYLAR["URETIM"])
        PT[b] = {t: PB.harman(bb, PB.ADAYLAR["URETIM"], t) for t in TOHUMLAR}
        ds = E[(E["_blok"] == b) & (E["soguk_mu"] == 0)]
        H[b] = hucre_gen(
            ds["p_gun_sayisi"].to_numpy("float64"), ds["guc"].to_numpy("float64"),
            ds["ufuk_gun"].to_numpy("float64"), n, ay,
        )
    HH = np.concatenate([H[b] for b in BLOKLAR])
    TN = np.concatenate([B[b]["tanim"] for b in BLOKLAR])
    BL = np.concatenate([np.full(len(H[b]), b) for b in BLOKLAR])
    havuz_pay = pd.Series(HH).value_counts(normalize=True)
    wk = np.array([test_pay.get(k, 0.0) / havuz_pay[k] for k in HH], dtype="float64")
    wk = wk / wk.mean()

    izgara: dict = {}
    for yon in YONLER:
        for k in KLER:
            ad = f"H{yon}_{k}"
            hu, d2l = {}, []
            for b in BLOKLAR:
                bb = B[b]
                p1 = wins_h(P0[b], k, yon, CAPA[b])
                e0, e1 = bb["y"] - P0[b], bb["y"] - p1
                d2l.append(e0 * e0 - e1 * e1)
                th_ = {}
                for t in TOHUMLAR:
                    pw = wins_h(PT[b][t], k, yon, CAPA[b])
                    a, c = bb["y"] - PT[b][t], bb["y"] - pw
                    th_[str(t)] = round(float((bb["w"] * (a * a - c * c)).mean()), 6)
                hu[b] = {
                    "agr": round(float((bb["w"] * (e0 * e0 - e1 * e1)).mean()), 6),
                    "budanan_pay": round(float((np.abs(p1 - P0[b]) > 1e-12).mean()), 5),
                    "budanan_satir": int((np.abs(p1 - P0[b]) > 1e-12).sum()),
                    "tohum": th_,
                }
            d2h = np.concatenate(d2l)
            hav = d2h * wk
            s = pd.Series(hav).groupby(TN)
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(3500)
            n_ = len(top)
            idx = rng.integers(0, n_, size=(N_ONY, n_))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            tas = np.random.default_rng(3501).uniform(0.28, 0.76, size=N_ONY)
            lb_u = tas * (o * SICAK_PAY) / BOLEN
            agr = [hu[b]["agr"] for b in BLOKLAR]
            kohort_blok = {b: round(float(hav[BL == b].mean()), 6) for b in BLOKLAR}
            n9 = sum(1 for b in BLOKLAR for t in TOHUMLAR if hu[b]["tohum"][str(t)] > 0)
            izgara[ad] = {
                "blok_agr": {b: hu[b]["agr"] for b in BLOKLAR},
                "isaret_blok_agr": f"{sum(v > 0 for v in agr)}/3",
                "kohortlu_blok": kohort_blok,
                "isaret_kohortlu": f"{sum(v > 0 for v in kohort_blok.values())}/3",
                "tohum": {b: hu[b]["tohum"] for b in BLOKLAR},
                "tohum_isareti": f"{n9}/9",
                "budanan": {b: hu[b]["budanan_satir"] for b in BLOKLAR},
                "budanan_pay": {b: hu[b]["budanan_pay"] for b in BLOKLAR},
                "HAVUZ_kohortlu_dMSE": round(float(hav.mean()), 6),
                "onyukleme_kohortlu": {
                    "ort": round(float(o.mean()), 6),
                    "GA95": [round(float(np.quantile(o, 0.025)), 6),
                             round(float(np.quantile(o, 0.975)), 6)],
                    "P_pozitif": round(float((o > 0).mean()), 3),
                },
                "beklenen_LB_tasima_0.5": round(
                    float(0.5 * (o.mean() * SICAK_PAY) / BOLEN), 5
                ),
                "P_LB_>=_0.00628": round(float((lb_u >= HEDEF).mean()), 3),
                "P_LB_>0": round(float((lb_u > 0).mean()), 3),
            }

    # blok-disi secim (kohortlu olcut uzerinden)
    bd = {}
    for b in BLOKLAR:
        dis = [x for x in BLOKLAR if x != b]
        en = max(izgara, key=lambda a: float(np.mean([izgara[a]["kohortlu_blok"][d] for d in dis])))
        bd[b] = {"secilen": en, "TUTULAN_BLOKTA_kohortlu": izgara[en]["kohortlu_blok"][b]}
    bd["DURUST_ORT"] = round(
        float(np.mean([bd[b]["TUTULAN_BLOKTA_kohortlu"] for b in BLOKLAR])), 6
    )
    R = {"00_izgara": izgara, "01_blok_disi_secim_kohortlu": bd}

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K3e_kuyruk_kapagi"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)

    print(f"{'aday':12s} {'ham3':>5s} {'koh3':>5s} {'toh':>4s} {'HAVUZ':>10s} "
          f"{'GA95':>24s} {'P>0':>6s} {'LB@.5':>8s} {'P>=hed':>7s} budanan")
    for ad, v in izgara.items():
        g = v["onyukleme_kohortlu"]["GA95"]
        print(f"{ad:12s} {v['isaret_blok_agr']:>5s} {v['isaret_kohortlu']:>5s} "
              f"{v['tohum_isareti']:>4s} {v['HAVUZ_kohortlu_dMSE']:+10.6f} "
              f"[{g[0]:+.5f},{g[1]:+.5f}] {v['onyukleme_kohortlu']['P_pozitif']:>6.3f} "
              f"{v['beklenen_LB_tasima_0.5']:+8.5f} {v['P_LB_>=_0.00628']:>7.3f} {v['budanan']}")
    print(json.dumps(bd, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
