"""p32-D: GECMIS-CAPALI WINSORIZATION (H) inceleme + onyukleme + P(LB>=hedef).

p32-C'de tek gercek sinyal H ailesi cikti (SICAK taraf):
  H_2.0 ORT_agr +0.0119 (2/3, tohum 8/9), H_2.4 +0.0109 (2/3, tohum 9/9).
Burada H ailesi icinde k BLOK-DISI secilir, tek/cift yonlu kirpma ayrilir,
trafo-kumeli onyukleme ile kazancin DAGILIMI cikarilir ve her aday icin
P(LB kazanci >= HEDEF) hesaplanir.

TASIMA MODELI (docs/80 §7): LB kazanci = tasima_orani * SICAK_PAY * dMSE_agr.
tasima_orani belirsiz; muhafazakar nokta 0.5, gercek blok CV kanitindan
gozlenen bant 0.28-0.76. Belirsizligi de dagilima katiyoruz:
  oran ~ Uniform(0.28, 0.76)  (rekor.jsonl'daki uc gozlem araligi)
ve ayrica sabit 0.5 / 0.7 icin nokta degerler raporlanir.

UYARI: LB olcutu RMSLE'dir, dMSE degil. dRMSLE ~ dMSE / (2*RMSLE).
Mevcut LB 1.00115 -> bolen ~2.0023. Bu donusum de uygulanir.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_c_winsor import gecmis_capa, wins
from p32_ortak import BLOKLAR, PK, TOHUMLAR

KLER = (1.0, 1.25, 1.5, 1.75, 2.0, 2.4, 3.0)
YONLER = ("cift", "ust")  # cift = alt+ust kirpma, ust = yalniz yukari kirpma
N_ONY = 4000
SICAK_PAY = PB.SICAK_PAY
MEVCUT_LB = 1.00115
HEDEF = 0.00628  # ilk 3 icin gereken LB kazanci
BOLEN = 2 * MEVCUT_LB


def wins_h(p, k, yon, capa):
    mu, sd = capa
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    q = p.copy()
    ust = mu[ok] + k * sd[ok]
    alt = mu[ok] - k * sd[ok]
    if yon == "cift":
        q[ok] = np.clip(p[ok], alt, ust)
    else:
        q[ok] = np.minimum(p[ok], ust)
    return np.maximum(q, 0.0)


def main() -> None:
    B = PB.veri_kur()
    capa = {b: gecmis_capa(b, B[b]["tanim"]) for b in BLOKLAR}
    P0 = {b: PB.harman(B[b], PB.ADAYLAR["URETIM"]) for b in BLOKLAR}
    PT = {b: {t: PB.harman(B[b], PB.ADAYLAR["URETIM"], t) for t in TOHUMLAR} for b in BLOKLAR}

    izgara: dict = {}
    d2_sakla: dict = {}
    for yon in YONLER:
        for k in KLER:
            ad = f"H{yon}_{k}"
            hu, d2s = {}, {}
            for b in BLOKLAR:
                bb = B[b]
                p1 = wins_h(P0[b], k, yon, capa[b])
                e0, e1 = bb["y"] - P0[b], bb["y"] - p1
                d2 = (e0 * e0 - e1 * e1) * bb["w"]
                d2s[b] = d2
                th = {}
                for t in TOHUMLAR:
                    pw = wins_h(PT[b][t], k, yon, capa[b])
                    a, c = bb["y"] - PT[b][t], bb["y"] - pw
                    th[str(t)] = round(float((bb["w"] * (a * a - c * c)).mean()), 6)
                hu[b] = {
                    "agr": round(float(d2.mean()), 6),
                    "ham": round(float((e0 * e0 - e1 * e1).mean()), 6),
                    "budanan_pay": round(float((np.abs(p1 - P0[b]) > 1e-12).mean()), 5),
                    "tohum": th,
                }
            agr = [hu[b]["agr"] for b in BLOKLAR]
            n9 = sum(1 for b in BLOKLAR for t in TOHUMLAR if hu[b]["tohum"][str(t)] > 0)
            izgara[ad] = {
                **{b: hu[b]["agr"] for b in BLOKLAR},
                "budanan_pay": {b: hu[b]["budanan_pay"] for b in BLOKLAR},
                "tohum": {b: hu[b]["tohum"] for b in BLOKLAR},
                "ORT_agr": round(float(np.mean(agr)), 6),
                "isaret": f"{sum(v > 0 for v in agr)}/3",
                "tohum_isareti": f"{n9}/9",
            }
            d2_sakla[ad] = d2s

    R: dict = {
        "00_mekanizma": (
            "Her trafonun EGITIM GECMISINDEKI log1p(tuketim) ortalamasi mu_i ve "
            "std sd_i (kesim oncesi, SIZINTISIZ) capasi. Tahmin mu_i +- k*sd_i "
            "bandina kirpilir. Yalniz SICAK satirlar (soguk trafonun gecmisi YOK)."
        ),
        "01_izgara": izgara,
    }

    # --- BLOK-DISI SECIM (H ailesi ICINDE, tek serbestlik k ve yon)
    bd = {}
    for b in BLOKLAR:
        dis = [x for x in BLOKLAR if x != b]
        en = max(izgara, key=lambda a: float(np.mean([izgara[a][d] for d in dis])))
        bd[b] = {
            "secilen": en,
            "dis_ort": round(float(np.mean([izgara[en][d] for d in dis])), 6),
            "TUTULAN_BLOKTA": izgara[en][b],
        }
    bd["DURUST_BLOK_DISI_ORT"] = round(
        float(np.mean([bd[b]["TUTULAN_BLOKTA"] for b in BLOKLAR])), 6
    )
    R["02_blok_disi_secim"] = bd

    # --- ONYUKLEME + P(LB >= HEDEF)
    rng_ana = np.random.default_rng(3201)
    tasima_u = rng_ana.uniform(0.28, 0.76, size=N_ONY)
    ony = {}
    for ad in izgara:
        hepsi = []
        blok = {}
        for i, b in enumerate(BLOKLAR):
            d2 = d2_sakla[ad][b]
            s = pd.Series(d2).groupby(B[b]["tanim"])
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(3300 + i)
            n = len(top)
            idx = rng.integers(0, n, size=(N_ONY, n))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            hepsi.append(o)
            blok[b] = {
                "ort": round(float(o.mean()), 6),
                "GA95": [round(float(np.quantile(o, 0.025)), 6),
                         round(float(np.quantile(o, 0.975)), 6)],
                "P_pozitif": round(float((o > 0).mean()), 3),
            }
        hv = np.mean(hepsi, axis=0)  # blok esit ortalamasi
        # LB kazanci: dMSE -> test dMSE -> dRMSLE -> tasima
        lb_u = tasima_u * (hv * SICAK_PAY) / BOLEN
        lb05 = 0.5 * (hv * SICAK_PAY) / BOLEN
        lb07 = 0.7 * (hv * SICAK_PAY) / BOLEN
        ony[ad] = {
            "blok": blok,
            "BLOK_ORT_dMSE": {
                "ort": round(float(hv.mean()), 6),
                "GA95": [round(float(np.quantile(hv, 0.025)), 6),
                         round(float(np.quantile(hv, 0.975)), 6)],
                "P_pozitif": round(float((hv > 0).mean()), 3),
            },
            "beklenen_LB_kazanci": {
                "tasima_0.5": round(float(lb05.mean()), 5),
                "tasima_0.7": round(float(lb07.mean()), 5),
                "tasima_belirsiz_ort": round(float(lb_u.mean()), 5),
            },
            "P_LB_kazanci_>=_0.00628": {
                "tasima_0.5": round(float((lb05 >= HEDEF).mean()), 3),
                "tasima_0.7": round(float((lb07 >= HEDEF).mean()), 3),
                "tasima_belirsiz": round(float((lb_u >= HEDEF).mean()), 3),
            },
            "P_LB_kazanci_>0": round(float((lb_u > 0).mean()), 3),
        }
    R["03_onyukleme_ve_P_hedef"] = ony
    R["04_TASIMA_MODELI"] = {
        "formul": "LB_kazanci = oran * SICAK_PAY(0.7784) * dMSE_agr / (2*1.00115)",
        "oran_dagilimi": "Uniform(0.28, 0.76) -- rekor.jsonl gercek blok CV gozlemleri (docs/80 §7)",
        "HEDEF": HEDEF,
        "not": "dMSE -> dRMSLE donusumu tek terimli Taylor; buyuk dMSE'de kazanci HAFIF abartir.",
    }

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K3b_H_capa"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)

    print(f"{'aday':14s} {'yaz25':>10s} {'guz25':>10s} {'kis26':>10s} {'ORT':>10s} "
          f"{'isaret':>6s} {'tohum':>6s} {'LB@.5':>8s} {'P>=hedef':>9s} {'P>0':>6s}")
    for ad, v in izgara.items():
        o = ony[ad]
        print(f"{ad:14s} {v['yaz25']:+10.6f} {v['guz25']:+10.6f} {v['kis26']:+10.6f} "
              f"{v['ORT_agr']:+10.6f} {v['isaret']:>6s} {v['tohum_isareti']:>6s} "
              f"{o['beklenen_LB_kazanci']['tasima_0.5']:+8.5f} "
              f"{o['P_LB_kazanci_>=_0.00628']['tasima_belirsiz']:>9.3f} "
              f"{o['P_LB_kazanci_>0']:>6.3f}")
    print(json.dumps(bd, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
