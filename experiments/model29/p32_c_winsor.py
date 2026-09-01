"""p32-C KATMAN 3: WINSORIZATION (aykiri-deger budamasi) son islem olarak.

Recruit 8. sirasi son islem olarak 2.4 sigma winsorization kullanmis. Bizde
son islem olarak aykiri-deger budamasi HIC denenmedi.

VARYANTLAR (hepsi 0-parametreli disinda k; k blok-disi secilir)
  G_<k>   GLOBAL: log-tahmin dagiliminin ortalamasi m ve std s;
          p <- clip(p, m-k*s, m+k*s)
  T_<k>   TRAFO-ICI: her trafonun KENDI log-tahmin ortalamasi m_i ve
          global std s; p <- clip(p, m_i-k*s, m_i+k*s)
  H_<k>   GECMIS-CAPALI: trafonun EGITIM GECMISINDEKI log1p ortalamasi
          ve std'si (sizintisiz, kesim oncesi); p <- clip(p, mu_i-k*sd_i,
          mu_i+k*sd_i). sd_i=0 olanlarda budama YOK.

TARAFLAR AYRI:
  SICAK -- p24_b_olc tezgahi (URETIM harmani, kohort agirlikli), 3 blok.
  SOGUK -- EZBER KANALI yuzunden YALNIZ kis26 gecerli (p30_ezber, docs/36 §3).
           Taban = uretim sogugu (yalniz cat, tohum 1000-1002 ort, beta=0.60).

Cikti: p_kalici/p32_katmanlar.json ["K3_winsor"]
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import p24_b_olc as PB
from p32_ortak import BLOKLAR, KESIM, PK, TOHUMLAR, _ham

KLER = (1.5, 2.0, 2.4, 3.0)
N_ONY = 500
SICAK_PAY = PB.SICAK_PAY
SOGUK_PAY = PB.SOGUK_PAY


def gecmis_capa(kesim_ad: str, tanimlar: np.ndarray):
    tr = _ham()
    g = tr[tr["tarih"] < pd.Timestamp(KESIM[kesim_ad])]
    lg = np.log1p(np.clip(g["tuketim"].to_numpy("float64"), 0, None))
    df = pd.DataFrame({"tanim": g["tanim"].to_numpy(), "l": lg})
    gb = df.groupby("tanim")["l"]
    mu, sd = gb.mean(), gb.std()
    idx = pd.Index(tanimlar)
    return mu.reindex(idx).to_numpy("float64"), sd.reindex(idx).to_numpy("float64")


def wins(p: np.ndarray, tip: str, k: float, tanim=None, capa=None) -> np.ndarray:
    if tip == "G":
        m, s = float(p.mean()), float(p.std())
        return np.clip(p, m - k * s, m + k * s)
    if tip == "T":
        s = float(p.std())
        mi = pd.Series(p).groupby(tanim).transform("mean").to_numpy()
        return np.clip(p, mi - k * s, mi + k * s)
    if tip == "H":
        mu, sd = capa
        ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
        q = p.copy()
        q[ok] = np.clip(p[ok], mu[ok] - k * sd[ok], mu[ok] + k * sd[ok])
        return q
    raise ValueError(tip)


def main() -> None:
    B = PB.veri_kur()
    R: dict = {
        "00_fikir": "Recruit 8.'si son islem olarak 2.4 sigma winsorization kullanmis; bizde hic denenmedi.",
        "01_kapi_soguk": "SOGUK kararlar YALNIZ kis26 (ezber kanali, p30_ezber). yaz25/guz25 soguk olcumu GECERSIZ.",
    }

    # =============== SICAK
    capa_s = {b: gecmis_capa(b, B[b]["tanim"]) for b in BLOKLAR}
    sicak: dict = {}
    for tip in ("G", "T", "H"):
        for k in KLER:
            ad = f"{tip}_{k}"
            hu: dict = {}
            for b in BLOKLAR:
                bb = B[b]
                p0 = PB.harman(bb, PB.ADAYLAR["URETIM"])
                p1 = np.maximum(wins(p0, tip, k, bb["tanim"], capa_s[b]), 0.0)
                e0, e1 = bb["y"] - p0, bb["y"] - p1
                th = {}
                for t in TOHUMLAR:
                    pt = PB.harman(bb, PB.ADAYLAR["URETIM"], t)
                    pw = np.maximum(wins(pt, tip, k, bb["tanim"], capa_s[b]), 0.0)
                    a, c = bb["y"] - pt, bb["y"] - pw
                    th[str(t)] = round(float((bb["w"] * (a * a - c * c)).mean()), 6)
                hu[b] = {
                    "agr": round(float((bb["w"] * (e0 * e0 - e1 * e1)).mean()), 6),
                    "ham": round(float((e0 * e0 - e1 * e1).mean()), 6),
                    "budanan_pay": round(float((np.abs(p1 - p0) > 1e-12).mean()), 5),
                    "tohum": th,
                }
            agr = [hu[b]["agr"] for b in BLOKLAR]
            n9 = sum(1 for b in BLOKLAR for t in TOHUMLAR if hu[b]["tohum"][str(t)] > 0)
            sicak[ad] = {
                **{b: hu[b]["agr"] for b in BLOKLAR},
                "budanan_pay": {b: hu[b]["budanan_pay"] for b in BLOKLAR},
                "ORT_agr": round(float(np.mean(agr)), 6),
                "isaret": f"{sum(v > 0 for v in agr)}/3",
                "tohum_isareti": f"{n9}/9",
                "test_dMSE": round(float(np.mean(agr)) * SICAK_PAY, 6),
            }
    R["02_SICAK"] = sicak

    # blok-disi secim (sicak)
    bd: dict = {}
    for b in BLOKLAR:
        dis = [x for x in BLOKLAR if x != b]
        en = max(sicak, key=lambda a: float(np.mean([sicak[a][d] for d in dis])))
        bd[b] = {"secilen": en, "dis_ort": round(float(np.mean([sicak[en][d] for d in dis])), 6),
                 "tutulan_blokta": sicak[en][b]}
    R["03_SICAK_blok_disi"] = bd

    # =============== SOGUK (YALNIZ kis26 gecerli; digerleri bilgi icin)
    E = pd.read_parquet(os.path.join(PB.DN, "egitim.parquet"))
    soguk: dict = {}
    SOG: dict = {}
    for b in BLOKLAR:
        d = E[E["_blok"] == b]
        dg = d[d["soguk_mu"] == 1]
        z = np.load(os.path.join(PB.DN, f"soguk_tahmin_{b}.npz"))
        pc = np.mean([z[f"{t}_cat"].astype("float64") for t in TOHUMLAR], axis=0)
        lguc = np.log1p(dg["guc"].to_numpy("float64"))
        r = pc - lguc
        r = float(r.mean()) + PB.BETA_SOGUK * (r - float(r.mean()))
        p0 = np.maximum(r + lguc, 0.0)
        yg = np.log1p(np.clip(dg["tuketim"].to_numpy("float64"), 0, None))
        SOG[b] = dict(p0=p0, y=yg, tanim=dg["tanim"].to_numpy())
    for tip in ("G", "T"):  # soguk trafolarin gecmisi YOK -> H yok
        for k in KLER:
            ad = f"{tip}_{k}"
            hu = {}
            for b in BLOKLAR:
                s = SOG[b]
                p1 = np.maximum(wins(s["p0"], tip, k, s["tanim"]), 0.0)
                e0, e1 = s["y"] - s["p0"], s["y"] - p1
                hu[b] = {
                    "dMSE": round(float((e0 * e0 - e1 * e1).mean()), 6),
                    "budanan_pay": round(float((np.abs(p1 - s["p0"]) > 1e-12).mean()), 5),
                }
            soguk[ad] = {
                **{b: hu[b]["dMSE"] for b in BLOKLAR},
                "budanan_pay": {b: hu[b]["budanan_pay"] for b in BLOKLAR},
                "GECERLI_kis26": hu["kis26"]["dMSE"],
                "test_dMSE_kis26": round(hu["kis26"]["dMSE"] * SOGUK_PAY, 6),
            }
    R["04_SOGUK_yalniz_kis26_gecerli"] = soguk

    # =============== onyukleme: en iyi sicak aday(lar) + 2.4 referans
    uygun = {a: v for a, v in sicak.items() if v["isaret"] == "3/3"}
    kaz = max(uygun, key=lambda a: uygun[a]["ORT_agr"]) if uygun else None
    ony: dict = {}
    for ad in {x for x in (kaz, "G_2.4", "T_2.4") if x}:
        tip, k = ad.split("_")
        k = float(k)
        ony[ad] = {}
        hepsi = []
        for b in BLOKLAR:
            bb = B[b]
            p0 = PB.harman(bb, PB.ADAYLAR["URETIM"])
            p1 = np.maximum(wins(p0, tip, k, bb["tanim"], capa_s[b]), 0.0)
            e0, e1 = bb["y"] - p0, bb["y"] - p1
            d2 = (e0 * e0 - e1 * e1) * bb["w"]
            s = pd.Series(d2).groupby(bb["tanim"])
            top, adet = s.sum().to_numpy(), s.size().to_numpy().astype("float64")
            rng = np.random.default_rng(330 + BLOKLAR.index(b))
            n = len(top)
            idx = rng.integers(0, n, size=(N_ONY, n))
            o = top[idx].sum(axis=1) / adet[idx].sum(axis=1)
            hepsi.append(o)
            ony[ad][b] = {
                "ort": round(float(o.mean()), 6),
                "GA95": [round(float(np.quantile(o, 0.025)), 6), round(float(np.quantile(o, 0.975)), 6)],
                "P_pozitif": round(float((o > 0).mean()), 3),
            }
        hv = np.mean(hepsi, axis=0)
        ony[ad]["BLOK_ORT"] = {
            "ort": round(float(hv.mean()), 6),
            "GA95": [round(float(np.quantile(hv, 0.025)), 6), round(float(np.quantile(hv, 0.975)), 6)],
            "P_pozitif": round(float((hv > 0).mean()), 3),
        }
    R["05_onyukleme_SICAK"] = ony
    R["06_KAZANAN_SICAK"] = {"3_3": sorted(uygun), "kazanan": kaz,
                             "detay": sicak[kaz] if kaz else None}

    yol = os.path.join(PK, "p32_katmanlar.json")
    mevcut = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            mevcut = json.load(fh)
    mevcut["K3_winsor"] = R
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(mevcut, fh, ensure_ascii=False, indent=1)

    print("=== SICAK (agr dMSE, + = kazanc) ===")
    for a, v in sicak.items():
        print(f"{a:8s} {v['yaz25']:+.6f} {v['guz25']:+.6f} {v['kis26']:+.6f} "
              f"ORT={v['ORT_agr']:+.6f} {v['isaret']} tohum {v['tohum_isareti']} "
              f"budanan={v['budanan_pay']}")
    print("=== SOGUK (yalniz kis26 GECERLI) ===")
    for a, v in soguk.items():
        print(f"{a:8s} kis26={v['GECERLI_kis26']:+.6f} (yaz25 {v['yaz25']:+.6f} "
              f"guz25 {v['guz25']:+.6f}) budanan={v['budanan_pay']}")
    print(json.dumps({k: R[k] for k in ("03_SICAK_blok_disi", "05_onyukleme_SICAK", "06_KAZANAN_SICAK")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
