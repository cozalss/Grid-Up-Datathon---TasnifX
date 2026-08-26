"""CURUTUCU C -- 'hangi trafo dirilecek bilgisi test aninda YOK' iddiasini sinar.

Bulgu, trafo-bazinda sabit orakulunun (-0,073..-0,108 dMSE) ULASILAMAZ oldugunu
soyluyor. Buradaki test: KESME ANINDA BILINEN trafo ozelliklerinden (kuyruk,
epizot sayisi, olum oncesi seviye, guc, doluluk, ilce) o sabitin ULASILABILIR
bir tahmini kurulabilir mi?

Tasarim: trafo duzeyinde LOO regresyon (blok-disi egitim), sonra
p' = (1-w)*p + w*c_hat(trafo). Uc blok, 3/3 sarti, K kirpma tablosu.
Hicbir yerde hedef blogun etiketi kullanilmaz.
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

from curut_olu_a import BLOKLAR, KLER, rmse, veri_yukle  # noqa: E402

KESME = {"yaz25": "2025-03-31", "guz25": "2025-07-31", "kis26": "2025-11-30"}
WLER = tuple(np.round(np.arange(0.0, 1.01, 0.1), 2))


def trafo_ozellikleri(tr: pd.DataFrame, kesme: str) -> pd.DataFrame:
    """Kesme tarihine kadarki veriden trafo duzeyi ozellikler. SIZINTI YOK."""
    k = pd.Timestamp(kesme)
    g = tr[tr["tarih"] <= k].sort_values(["tanim", "tarih"])
    y = np.log1p(g["tuketim"].to_numpy(dtype="float64").clip(min=0.0))
    g = g.assign(_ly=y, _sifir=(g["tuketim"].to_numpy() <= 0).astype(int))
    ozet = []
    for ad, s in g.groupby("tanim", observed=True, sort=False):
        sf = s["_sifir"].to_numpy()
        ly = s["_ly"].to_numpy()
        n = len(sf)
        # kuyruk sifir
        poz = np.flatnonzero(sf == 0)
        kuyruk = n if poz.size == 0 else n - poz[-1] - 1
        # epizot: sifir -> pozitif gecis sayisi (DIRILME gecmisi)
        gecis = int(((sf[:-1] == 1) & (sf[1:] == 0)).sum()) if n > 1 else 0
        canli = ly[sf == 0]
        # olum oncesi son 14 canli gunun seviyesi
        son_canli = float(canli[-14:].mean()) if canli.size else 0.0
        ozet.append(
            {
                "tanim": ad,
                "kuyruk": float(kuyruk),
                "log_kuyruk": float(np.log1p(kuyruk)),
                "epizot": float(gecis),
                "sifir_orani": float(sf.mean()),
                "canli_gun": float(canli.size),
                "son_canli": son_canli,
                "toplam_gun": float(n),
                "log_guc": float(np.log1p(s["guc"].iloc[0])),
                "olu_pay": float(kuyruk) / float(n),
            }
        )
    return pd.DataFrame(ozet).set_index("tanim")


def main() -> int:
    t0 = time.time()
    veri = veri_yukle()
    tr = pd.read_csv(KOK / "data/raw/train.csv", encoding="utf-8", dtype={"tanim": str})
    tr["tarih"] = pd.to_datetime(tr["tarih"])

    ozellik = {b: trafo_ozellikleri(tr, KESME[b]) for b in BLOKLAR}
    kolonlar = [
        "log_kuyruk",
        "epizot",
        "sifir_orani",
        "canli_gun",
        "son_canli",
        "toplam_gun",
        "log_guc",
        "olu_pay",
    ]

    # trafo duzeyi hedef: blogun etiket penceresinde ortalama log1p
    tablo: dict[str, pd.DataFrame] = {}
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        df = pd.DataFrame({"tanim": v["tanim"][m], "ly": v["ly"][m]})
        h = df.groupby("tanim")["ly"].agg(["mean", "size"])
        oz = ozellik[b].reindex(h.index)
        tablo[b] = pd.concat([h, oz], axis=1).dropna()
        print(f"  {b}: {len(tablo[b])} olu trafo, {int(tablo[b]['size'].sum()):,} satir")

    print()
    print("=" * 100)
    print("C1) TRAFO DUZEYI LOO REGRESYON -- kesme aninda bilinen ozelliklerden c_hat")
    print("=" * 100)
    from sklearn.linear_model import Ridge

    chat: dict[str, pd.Series] = {}
    for b in BLOKLAR:
        dig = [x for x in BLOKLAR if x != b]
        eg = pd.concat([tablo[x] for x in dig])
        X = eg[kolonlar].to_numpy()
        mu, sd = X.mean(0), X.std(0) + 1e-9
        mdl = Ridge(alpha=3.0)
        mdl.fit((X - mu) / sd, eg["mean"].to_numpy(), sample_weight=eg["size"].to_numpy())
        Xt = (tablo[b][kolonlar].to_numpy() - mu) / sd
        p = np.clip(mdl.predict(Xt), 0.0, None)
        chat[b] = pd.Series(p, index=tablo[b].index)
        gercek = tablo[b]["mean"].to_numpy()
        w = tablo[b]["size"].to_numpy()
        sabit = float(np.average(eg["mean"], weights=eg["size"]))
        r2 = 1 - np.average((p - gercek) ** 2, weights=w) / np.average(
            (sabit - gercek) ** 2, weights=w
        )
        print(
            f"  {b}: agirlikli R^2 (tek sabite gore) {r2:+.4f}  "
            f"c_hat ort {np.average(p, weights=w):.3f}  gercek ort {np.average(gercek, weights=w):.3f}"
        )

    print()
    print("=" * 100)
    print("C2) p' = (1-w)p + w*c_hat(trafo)  -- tam blok dRMSLE (sicak payda)")
    print("=" * 100)
    print(f"  {'w':>5}" + "".join(f"{b:>12}" for b in BLOKLAR) + f"{'3/3':>6}{'ortalama':>11}")
    tut_w = []
    for w in WLER:
        farklar = []
        for b in BLOKLAR:
            v = veri[b]
            m = v["olu"]
            c = pd.Series(v["tanim"][m]).map(chat[b]).to_numpy()
            c = np.where(np.isnan(c), np.nanmean(c), c)
            e0 = (v["log_t"] - v["ly"]) ** 2
            yeni = v["log_t"].copy()
            yeni[m] = (1 - w) * yeni[m] + w * c
            e1 = (yeni - v["ly"]) ** 2
            farklar.append(rmse(e1) - rmse(e0))
        kaz = sum(1 for f in farklar if f < 0)
        tut_w.append((float(w), farklar, kaz))
        print(
            f"  {w:5.2f}"
            + "".join(f"{f:+12.5f}" for f in farklar)
            + f"{kaz:>4}/3{np.mean(farklar):+11.5f}"
        )

    uygun = [(w, np.mean(f)) for w, f, k in tut_w if k == 3 and w > 0]
    print()
    print("=" * 100)
    print("C3) KIRPMA TABLOSU")
    print("=" * 100)
    if uygun:
        w = min(uygun, key=lambda t: t[1])[0]
        print(f"  3/3 VAR: w={w:.2f} ort {min(uygun, key=lambda t: t[1])[1]:+.5f}")
    else:
        en = min(((w, np.mean(f), k) for w, f, k in tut_w if w > 0), key=lambda t: t[1])
        w = en[0]
        print(f"  3/3 YOK. en iyi w={w:.2f} ({en[2]}/3, ort {en[1]:+.5f}) -- yine de tablo:")
    print(f"  {'blok':7}{'trafo':>8}" + "".join(f"{'K=' + str(k):>10}" for k in KLER))
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        c = pd.Series(v["tanim"][m]).map(chat[b]).to_numpy()
        c = np.where(np.isnan(c), np.nanmean(c), c)
        e0 = (v["log_t"] - v["ly"]) ** 2
        yeni = v["log_t"].copy()
        yeni[m] = (1 - w) * yeni[m] + w * c
        e1 = (yeni - v["ly"]) ** 2
        d = e1 - e0
        tn = v["tanim"]
        katki = pd.DataFrame({"tanim": tn[m], "d": d[m]}).groupby("tanim")["d"].sum().sort_values()
        satir = f"  {b:7}{katki.size:8,}"
        for K in KLER:
            at = set(katki.index[:K])
            keep = ~pd.Series(tn).isin(at).to_numpy()
            satir += f"{rmse(e1[keep]) - rmse(e0[keep]):+10.5f}"
        print(satir)

    print()
    print("=" * 100)
    print("C4) EPIZOT (dirilme gecmisi) sinyal tasiyor mu? -- olu trafolarda")
    print("=" * 100)
    for b in BLOKLAR:
        t = tablo[b]
        e = t["epizot"].to_numpy()
        kova = np.digitize(e, [1, 3, 10])
        print(f"  {b}:")
        for k, et in enumerate(["0", "1-2", "3-9", "10+"]):
            s = kova == k
            if s.sum() == 0:
                continue
            print(
                f"    epizot {et:>4}  trafo {int(s.sum()):4}  satir {int(t['size'][s].sum()):7,}"
                f"  hedef ort log1p {np.average(t['mean'][s], weights=t['size'][s]):.4f}"
            )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
