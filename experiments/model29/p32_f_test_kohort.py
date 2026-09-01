"""p32-F: H-capa katmanini TEST KOHORTUNA agirliklandirilmis HAVUZDA olc.

GEREKCE
-------
p32-E, kazancin ne mevsimle ne gecmis uzunluguyla tekduze acikladigini
gosterdi: kis26'da kisa gecmisli trafolarda +0.093, orta kovalarda -0.13.
Yani katman YUKSEK VARYANSLI ve kohorta cok duyarli. Blok-esit ortalama
bu durumda yaniltici (MEMORY: "CV blogunu test kohortuna agirliklandir").

OLCUM
-----
Uc blogun butun SICAK satirlari havuzlanir; her satira, TEST sicak
satirlarinin (gecmis_gun kovasi x farkli_ay kovasi x p_gun_sayisi kovasi
x kVA kovasi x ufuk kovasi) hucre dagilimina gore agirlik verilir.
Hucre kapsami ve TV uzakligi raporlanir. Onyukleme trafo kumeli.

Ayrica ONCEKI kohort (p24'un pg x kVA x ufuk) ile karsilastirilir.

Cikti: p_kalici/p32_katmanlar.json ["K3d_test_kohort"]
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_d_hcapa import wins_h
from p32_e_gecmis_uzunlugu import capa_ve_kapsam
from p32_ortak import BLOKLAR, PK, TOHUMLAR

KLER = (1.5, 1.75, 2.0, 2.4, 3.0)
YONLER = ("cift", "ust")
N_ONY = 4000
SICAK_PAY = PB.SICAK_PAY
MEVCUT_LB = 1.00115
HEDEF = 0.00628
BOLEN = 2 * MEVCUT_LB

GG_KENAR = [90, 150, 210, 270, 330]
AY_KENAR = [3, 6, 9, 12]


def hucre_gen(pg, guc, ufuk, gg, ay):
    a = np.digitize(pg, PB.PG_KENAR)
    b = np.digitize(guc, PB.KVA_KENAR) - 1
    c = np.digitize(ufuk, PB.UFUK_KENAR)
    d = np.digitize(np.nan_to_num(gg, nan=-1), GG_KENAR)
    e = np.digitize(np.nan_to_num(ay, nan=-1), AY_KENAR)
    return (((a * 10 + b) * 10 + c) * 10 + d) * 10 + e


def main() -> None:
    B = PB.veri_kur()
    E = pd.read_parquet(os.path.join(PB.DN, "egitim.parquet"))
    T = pd.read_parquet(os.path.join(PB.DN, "test.parquet"))
    ts = T[T["soguk_mu"] == 0]
    _, _, tn, tay = capa_ve_kapsam("TEST", ts["tanim"].astype(str).to_numpy())
    th = hucre_gen(
        ts["p_gun_sayisi"].to_numpy("float64"),
        ts["guc"].to_numpy("float64"),
        ts["ufuk_gun"].to_numpy("float64"),
        tn,
        tay,
    )
    test_pay = pd.Series(th).value_counts(normalize=True)

    # havuz kur
    hh, ww, tanim_h, blok_h = [], [], [], []
    P0, CAPA = {}, {}
    for b in BLOKLAR:
        bb = B[b]
        d = E[E["_blok"] == b]
        ds = d[d["soguk_mu"] == 0]
        mu, sd, n, ay = capa_ve_kapsam(b, bb["tanim"])
        CAPA[b] = (mu, sd)
        P0[b] = PB.harman(bb, PB.ADAYLAR["URETIM"])
        h = hucre_gen(
            ds["p_gun_sayisi"].to_numpy("float64"),
            ds["guc"].to_numpy("float64"),
            ds["ufuk_gun"].to_numpy("float64"),
            n,
            ay,
        )
        hh.append(h)
        tanim_h.append(bb["tanim"])
        blok_h.append(np.full(len(h), b))
    H = np.concatenate(hh)
    TN = np.concatenate(tanim_h)
    BL = np.concatenate(blok_h)
    havuz_pay = pd.Series(H).value_counts(normalize=True)
    w = np.array([test_pay.get(k, 0.0) / havuz_pay[k] for k in H], dtype="float64")
    kapsam = float(test_pay[test_pay.index.isin(set(H))].sum())
    ortak = set(test_pay.index) | set(havuz_pay.index)
    tv = 0.5 * sum(abs(test_pay.get(k, 0.0) - havuz_pay.get(k, 0.0)) for k in ortak)
    w = w / w.mean()
    etkin = float(w.sum() ** 2 / (w * w).sum())

    R: dict = {
        "00_kohort": {
            "hucre": "p_gun_sayisi x kVA x ufuk x gecmis_gun x farkli_ay",
            "test_hucre_kapsami": round(kapsam, 4),
            "TV_uzakligi": round(tv, 4),
            "n_havuz": int(len(H)),
            "etkin_ornek": round(etkin, 1),
            "agirlik_maks": round(float(w.max()), 2),
            "UYARI": (
                "kapsam <1 ise test'in o kadarlik bolumu CV'de HIC temsil edilmiyor; "
                "olcum o bolum icin sessiz kaliyor."
            ),
        }
    }

    izgara: dict = {}
    for yon in YONLER:
        for k in KLER:
            ad = f"H{yon}_{k}"
            d2l = []
            for b in BLOKLAR:
                bb = B[b]
                p1 = wins_h(P0[b], k, yon, CAPA[b])
                e0, e1 = bb["y"] - P0[b], bb["y"] - p1
                d2l.append(e0 * e0 - e1 * e1)
            d2 = np.concatenate(d2l) * w
            # onyukleme (trafo kumeli, havuzda)
            s = pd.Series(d2).groupby(TN)
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(3400)
            n = len(top)
            idx = rng.integers(0, n, size=(N_ONY, n))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            rng2 = np.random.default_rng(3401)
            tas = rng2.uniform(0.28, 0.76, size=N_ONY)
            lb_u = tas * (o * SICAK_PAY) / BOLEN
            lb05 = 0.5 * (o * SICAK_PAY) / BOLEN
            izgara[ad] = {
                "HAVUZ_dMSE_agr": round(float(d2.mean()), 6),
                "blok_ici_agr": {
                    b: round(float(d2[BL == b].mean()), 6) for b in BLOKLAR
                },
                "onyukleme": {
                    "ort": round(float(o.mean()), 6),
                    "GA95": [round(float(np.quantile(o, 0.025)), 6),
                             round(float(np.quantile(o, 0.975)), 6)],
                    "P_pozitif": round(float((o > 0).mean()), 3),
                },
                "beklenen_LB": {
                    "tasima_0.5": round(float(lb05.mean()), 5),
                    "tasima_belirsiz": round(float(lb_u.mean()), 5),
                },
                "P_LB_>=_0.00628": round(float((lb_u >= HEDEF).mean()), 3),
                "P_LB_>0": round(float((lb_u > 0).mean()), 3),
            }
    R["01_izgara_test_kohortlu"] = izgara

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K3d_test_kohort"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)

    print(json.dumps(R["00_kohort"], ensure_ascii=False, indent=1))
    print(f"{'aday':12s} {'HAVUZ':>10s} {'GA95':>26s} {'P>0':>6s} {'LB@.5':>8s} {'P>=hed':>7s}")
    for ad, v in izgara.items():
        g = v["onyukleme"]["GA95"]
        print(f"{ad:12s} {v['HAVUZ_dMSE_agr']:+10.6f} [{g[0]:+.5f},{g[1]:+.5f}] "
              f"{v['onyukleme']['P_pozitif']:>6.3f} {v['beklenen_LB']['tasima_0.5']:+8.5f} "
              f"{v['P_LB_>=_0.00628']:>7.3f}  blok={v['blok_ici_agr']}")


if __name__ == "__main__":
    main()
