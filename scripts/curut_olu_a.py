"""CURUTUCU A -- olu/kuyruk-sifir satirlarinda kova/grup sabitine cekme ekseni.

BAGIMSIZ yeniden olcum. Onbelleklenmis SICAK tahminler (cat/xgb/lgbm harmani,
3 tohum) uzerinde calisir; fit yok.

Sorular:
  1. Model olu satirlarda TEK SABITTEN gercekten iyi mi? (in-sample orakul)
  2. LOO (blok-disi) sabitine cekmek uc blokta da kazandiriyor mu?
  3. Kazanc trafo bazinda dagilmis mi? (K=0,1,5,10,25,50 kirpma)
  4. (blok,tohum) ciftleri uzerinde eslenik standart hata ne?
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
TOHUMLAR = (1000, 1001, 1002)
BLOKLAR = ("yaz25", "guz25", "kis26")
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
EGITIM = KOK / "data" / "interim" / "deney" / "egitim.parquet"
WLER = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))
KLER = (0, 1, 5, 10, 25, 50)


def rmse(e: np.ndarray) -> float:
    return float(np.sqrt(e.mean()))


def kuyruk_kovasi(k: np.ndarray) -> np.ndarray:
    """Kuyruk uzunlugu kovasi (olu_hedge.py KOVALAR ile ayni sinirlar)."""
    return np.digitize(k, [15, 30, 60, 90])


def ufuk_kovasi(u: np.ndarray) -> np.ndarray:
    return np.digitize(u, [31, 61, 91])


def veri_yukle() -> dict[str, dict]:
    z = np.load(ONBELLEK)
    eg = pd.read_parquet(
        EGITIM,
        columns=["_blok", "soguk_mu", "tanim", "tuketim", "guc", "t_kuyruk_sifir", "ufuk_gun"],
    )
    veri: dict[str, dict] = {}
    for b in BLOKLAR:
        dog = eg[eg["_blok"] == b]
        sicak = (dog["soguk_mu"] == 1).to_numpy()
        dg = dog[~sicak]
        pay = sum(AGIRLIK)
        tohum_log = [
            sum(AGIRLIK[i] * z[f"{b}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
            for t in TOHUMLAR
        ]
        log_t = np.mean(tohum_log, axis=0)
        ly = np.log1p(dg["tuketim"].to_numpy(dtype="float64").clip(min=0.0))
        kuy = np.nan_to_num(dg["t_kuyruk_sifir"].to_numpy(dtype="float64"), nan=0.0)
        veri[b] = {
            "log_t": log_t,
            "tohum_log": tohum_log,
            "ly": ly,
            "kuyruk": kuy,
            "ufuk": dg["ufuk_gun"].to_numpy(dtype="float64"),
            "tanim": dg["tanim"].to_numpy(),
            "olu": kuy >= 1.0,
        }
    return veri


def main() -> int:
    t0 = time.time()
    veri = veri_yukle()

    print("=" * 100)
    print("1) TABAN: olu satirlarin payi ve tasidigi kare hata")
    print("=" * 100)
    print(
        f"  {'blok':7}{'sicak satir':>13}{'olu satir':>11}{'pay %':>8}{'olu MSE':>10}"
        f"{'blok MSE':>10}{'hata payi %':>13}{'trafo':>8}"
    )
    for b in BLOKLAR:
        v = veri[b]
        e0 = (v["log_t"] - v["ly"]) ** 2
        m = v["olu"]
        print(
            f"  {b:7}{len(e0):13,}{int(m.sum()):11,}{100 * m.mean():8.2f}"
            f"{e0[m].mean():10.5f}{e0.mean():10.5f}{100 * e0[m].sum() / e0.sum():13.2f}"
            f"{len(np.unique(v['tanim'][m])):8,}"
        )

    print()
    print("=" * 100)
    print("2) IN-SAMPLE ORAKUL TAVANLARI (ulasilamaz -- eksenin GENISLIGI)")
    print("=" * 100)
    print(
        f"  {'blok':7}{'mevcut olu MSE':>16}{'tek sabit':>12}{'kuyruk kova':>13}"
        f"{'trafo sabiti':>14}{'tam blok dRMSLE(trafo)':>24}"
    )
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        ly, lt = v["ly"][m], v["log_t"][m]
        e0 = (lt - ly) ** 2
        c = ly.mean()
        e_tek = (c - ly) ** 2
        kv = kuyruk_kovasi(v["kuyruk"][m])
        e_kova = np.empty_like(ly)
        for k in np.unique(kv):
            s = kv == k
            e_kova[s] = (ly[s].mean() - ly[s]) ** 2
        tn = v["tanim"][m]
        s = pd.Series(ly).groupby(pd.Series(tn)).transform("mean").to_numpy()
        e_trafo = (s - ly) ** 2
        # tam blok dRMSLE, trafo-sabiti orakulu
        tam0 = (v["log_t"] - v["ly"]) ** 2
        tam1 = tam0.copy()
        tam1[m] = e_trafo
        print(
            f"  {b:7}{e0.mean():16.5f}{e_tek.mean():12.5f}{e_kova.mean():13.5f}"
            f"{e_trafo.mean():14.5f}{rmse(tam1) - rmse(tam0):24.5f}"
        )

    print()
    print("=" * 100)
    print("3) LOO (blok-disi) SABITINE CEKME -- p' = (1-w)p + w*c")
    print("=" * 100)

    semalar = {
        "tek": lambda v, m: np.zeros(int(m.sum()), dtype=int),
        "kuyruk5": lambda v, m: kuyruk_kovasi(v["kuyruk"][m]),
        "ikili60": lambda v, m: (v["kuyruk"][m] >= 60).astype(int),
        "ikili30": lambda v, m: (v["kuyruk"][m] >= 30).astype(int),
        "ufuk4": lambda v, m: ufuk_kovasi(v["ufuk"][m]),
        "kuyrukXufuk": lambda v, m: kuyruk_kovasi(v["kuyruk"][m]) * 4 + ufuk_kovasi(v["ufuk"][m]),
    }

    sonuc: dict[str, dict] = {}
    for ad, fn in semalar.items():
        print(f"\n  --- sema: {ad} ---")
        print(f"  {'w':>5}" + "".join(f"{b:>12}" for b in BLOKLAR) + f"{'3/3':>6}{'ortalama':>11}")
        # her blok icin LOO kova sabitleri
        kova_deger: dict[str, dict[int, float]] = {}
        for b in BLOKLAR:
            v = veri[b]
            kova_deger[b] = {}
            digerleri = [x for x in BLOKLAR if x != b]
            ly_d = np.concatenate([veri[x]["ly"][veri[x]["olu"]] for x in digerleri])
            kv_d = np.concatenate([fn(veri[x], veri[x]["olu"]) for x in digerleri])
            genel = float(ly_d.mean())
            for k in np.unique(kv_d):
                s = kv_d == k
                kova_deger[b][int(k)] = float(ly_d[s].mean()) if s.sum() >= 30 else genel
            kova_deger[b][-1] = genel
        sonuc[ad] = {"kova_deger": kova_deger, "w": {}}
        for w in WLER:
            farklar = []
            for b in BLOKLAR:
                v = veri[b]
                m = v["olu"]
                kv = fn(v, m)
                c = np.array([kova_deger[b].get(int(k), kova_deger[b][-1]) for k in kv])
                tam0 = (v["log_t"] - v["ly"]) ** 2
                yeni = v["log_t"].copy()
                yeni[m] = (1 - w) * yeni[m] + w * c
                tam1 = (yeni - v["ly"]) ** 2
                farklar.append(rmse(tam1) - rmse(tam0))
            kaz = sum(1 for f in farklar if f < 0)
            sonuc[ad]["w"][float(w)] = (farklar, kaz)
            print(
                f"  {w:5.2f}"
                + "".join(f"{f:+12.5f}" for f in farklar)
                + f"{kaz:>4}/3{np.mean(farklar):+11.5f}"
            )

    print()
    print("=" * 100)
    print("4) KIRPMA TABLOSU (KURAL 1) -- her sema icin 3/3 varsa en iyi w'de")
    print("=" * 100)
    for ad, fn in semalar.items():
        uygun = [(w, np.mean(f)) for w, (f, k) in sonuc[ad]["w"].items() if k == 3 and w > 0]
        if not uygun:
            en_iyi = min(
                ((w, np.mean(f), k) for w, (f, k) in sonuc[ad]["w"].items() if w > 0),
                key=lambda t: t[1],
            )
            print(
                f"\n  sema {ad}: 3/3 YOK (en iyi w={en_iyi[0]:.2f}, {en_iyi[2]}/3, "
                f"ort {en_iyi[1]:+.5f}) -- yine de kirpma tablosu:"
            )
            w = en_iyi[0]
        else:
            w = min(uygun, key=lambda t: t[1])[0]
            print(f"\n  sema {ad}: 3/3 w={w:.2f}, ort {min(uygun, key=lambda t: t[1])[1]:+.5f}")
        print(f"  {'blok':7}{'trafo':>8}" + "".join(f"{'K=' + str(k):>10}" for k in KLER))
        kova_deger = sonuc[ad]["kova_deger"]
        for b in BLOKLAR:
            v = veri[b]
            m = v["olu"]
            kv = fn(v, m)
            c = np.array([kova_deger[b].get(int(k), kova_deger[b][-1]) for k in kv])
            yeni = v["log_t"].copy()
            yeni[m] = (1 - w) * yeni[m] + w * c
            e0 = (v["log_t"] - v["ly"]) ** 2
            e1 = (yeni - v["ly"]) ** 2
            d = e1 - e0
            tn = v["tanim"]
            df = pd.DataFrame({"tanim": tn[m], "d": d[m]})
            katki = df.groupby("tanim")["d"].sum().sort_values()
            satir = f"  {b:7}{katki.size:8,}"
            for K in KLER:
                at = set(katki.index[:K])
                tut = ~pd.Series(tn).isin(at).to_numpy()
                satir += f"{rmse(e1[tut]) - rmse(e0[tut]):+10.5f}"
            print(satir)

    print()
    print("=" * 100)
    print("5) ESLENIK STANDART HATA -- (blok,tohum) ciftleri, en umut verici sema")
    print("=" * 100)
    for ad, fn in semalar.items():
        uygun = [(w, np.mean(f)) for w, (f, k) in sonuc[ad]["w"].items() if k == 3 and w > 0]
        if not uygun:
            continue
        w = min(uygun, key=lambda t: t[1])[0]
        kova_deger = sonuc[ad]["kova_deger"]
        d9 = []
        for b in BLOKLAR:
            v = veri[b]
            m = v["olu"]
            kv = fn(v, m)
            c = np.array([kova_deger[b].get(int(k), kova_deger[b][-1]) for k in kv])
            for i, tl in enumerate(v["tohum_log"]):
                e0 = (tl - v["ly"]) ** 2
                yeni = tl.copy()
                yeni[m] = (1 - w) * yeni[m] + w * c
                e1 = (yeni - v["ly"]) ** 2
                d9.append(rmse(e1) - rmse(e0))
        d9 = np.array(d9)
        print(
            f"  sema {ad} w={w:.2f}: ort {d9.mean():+.5f}  sh {d9.std(ddof=1) / np.sqrt(len(d9)):.5f}"
            f"  (n={len(d9)}, negatif {int((d9 < 0).sum())}/{len(d9)})"
        )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
