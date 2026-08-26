"""H15 -- SIFIRLIK, TOPLU-KATILIM nufusunda ogrenilebilir mi?

NEDEN YENIDEN SORULUYOR (yasak bolge ihlali DEGIL, DENETIM)
------------------------------------------------------------
docs/41 §3 "hurdle CURUDU" hukmunu **kis26 soguk** uzerinde verdi:
sifir siniflayici AUC 0,5728 (trafo duzeyinde 0,5648).

H14 olctu ki kis26 soguk ile TEST soguk **ayni nufus DEGIL**:

    nufus payi (satir)     kis26 soguk     TEST soguk
    TOPLU >=100                 1,1%          80,7%
    orta 20-99                 39,6%           7,4%
    tekil/kucuk <20            59,2%          11,9%

Yani hukum, testin %81'ini olusturan siniftan neredeyse HIC ornek gormeden
verildi (701 satir). Ve mekanizma farkli olmali: toplu katilim bir
enerjilendirme dalgasi degil, ZATEN CALISAN trafolarin veri setine geriye
dolgusu. Boyle bir trafonun sifiri "olu trafo" degil, baska bir sey olabilir
(mevsimsel duruş, bakim, olcum kesintisi) -- ve baska bir sey OGRENILEBILIR
olabilir.

SORU
----
Toplu-katilim nufusunda sifirlik, statik ve AS-OF ozniteliklerle
kis26'da olculdugunden DAHA IYI kestirilebiliyor mu?

TASARIM -- IKI ORTUSMEYEN ZAMAN KESMESI (kural 9)
-------------------------------------------------
    EGIT : 2025-01-01 kohortu (1.902 trafo)
    SINA : 2025-11-25 kohortu (153 trafo)   <- farkli zaman, farkli parti
Ozellikler YALNIZCA dogum aninda bilinenler (kural 8 AS-OF):
    guc (kVA), lokasyon, tanim onekleri, parti buyuklugu, dogum ayi
Trafonun KENDI tuketim gecmisi KULLANILMAZ -- test soguk trafolarinin
tahmin aninda gecmisi yoktur.

Ayrica sifirligin YOGUNLASMASI olculur: birkac olu trafoda mi, yoksa
tum kohorta yayilmis mi? Yayilmissa "olu trafo" cercevesi yanlistir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ILK_GUN_AT = 7


def auc(y: np.ndarray, s: np.ndarray) -> float:
    y = np.asarray(y, bool)
    if y.all() or not y.any():
        return float("nan")
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def kohort_cerceve(
    tr: pd.DataFrame, ilk: pd.Series, parti: pd.Series, gun: pd.Timestamp, ufuk: int = 127
) -> pd.DataFrame:
    bit = min(gun + pd.Timedelta(days=ufuk), tr["tarih"].max())
    t = set(ilk[ilk == gun].index)
    a = tr[tr["tanim"].isin(t) & (tr["tarih"] >= gun) & (tr["tarih"] <= bit)].copy()
    a = a[a["yas"] >= ILK_GUN_AT]
    return a


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim", "lokasyon"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    parti = ilk.groupby(ilk).size()
    tr["ilk"] = tr["tanim"].map(ilk)
    tr["yas"] = (tr["tarih"] - tr["ilk"]).dt.days
    tr["sifir"] = tr["tuketim"].eq(0)

    egit = kohort_cerceve(tr, ilk, parti, pd.Timestamp("2025-01-01"))
    sina = kohort_cerceve(tr, ilk, parti, pd.Timestamp("2025-11-25"))
    print("=" * 92)
    print("1. YOGUNLASMA -- sifirlik birkac OLU trafoda mi, yayilmis mi?")
    print("=" * 92)
    for ad, a in (("2025-01-01 kohortu", egit), ("2025-11-25 kohortu", sina)):
        pt = a.groupby("tanim")["sifir"].agg(["mean", "size"])
        pt = pt[pt["size"] >= 30]
        print(
            f"\n  {ad}: {len(a):,} satir, {a.tanim.nunique():,} trafo, "
            f"sifir orani {a['sifir'].mean():.4f}"
        )
        print(f"    trafo bazinda sifir orani dagilimi (>=30 gunluk trafolar, n={len(pt)}):")
        for q in (0.50, 0.75, 0.90, 0.95, 0.99):
            print(f"      q{int(q * 100):>2} {pt['mean'].quantile(q):.4f}", end="")
        print()
        tam_olu = int((pt["mean"] > 0.95).sum())
        hic = int((pt["mean"] == 0).sum())
        print(
            f"    TAM OLU (>%95 sifir) {tam_olu:>4} / {len(pt)}  ({tam_olu / max(len(pt), 1):.1%})"
        )
        print(f"    HIC SIFIRI YOK       {hic:>4} / {len(pt)}  ({hic / max(len(pt), 1):.1%})")
        # sifir kutlesinin ne kadari TAM OLU trafolardan?
        olu_t = set(pt[pt["mean"] > 0.95].index)
        pay = float(a[a["tanim"].isin(olu_t)]["sifir"].sum() / max(a["sifir"].sum(), 1))
        print(f"    sifir KUTLESININ  {pay:.1%}'i TAM OLU trafolardan geliyor")

    # ---------------- 2. AS-OF siniflayici ----------------
    print("\n" + "=" * 92)
    print("2. AS-OF SIFIR SINIFLAYICI -- egit 2025-01-01, sina 2025-11-25")
    print("=" * 92)
    print("  (ozellikler yalnizca dogumda bilinenler; trafonun kendi gecmisi YOK)")

    def oznitelik(a: pd.DataFrame) -> pd.DataFrame:
        d = pd.DataFrame(index=a.index)
        d["lg_guc"] = np.log1p(a["guc"].to_numpy(dtype="float64"))
        d["ilce"] = a["lokasyon"].astype(str)
        d["on2"] = a["tanim"].str[:2]
        d["on3"] = a["tanim"].str[:3]
        d["ay"] = a["tarih"].dt.month
        d["haftagunu"] = a["tarih"].dt.dayofweek
        d["yas"] = a["yas"].to_numpy()
        return d

    xe, ye = oznitelik(egit), egit["sifir"].to_numpy()
    xs, ys = oznitelik(sina), sina["sifir"].to_numpy()

    # kategorik hedef kodlamasi -- YALNIZCA egitim kohortundan (sizinti yok)
    skor_s = np.zeros(len(xs))
    skor_e = np.zeros(len(xe))
    taban = float(ye.mean())
    print(f"\n  egitim taban sifir orani {taban:.4f}   sinama {float(ys.mean()):.4f}")
    for kol, duzgun in (("ilce", 50.0), ("on2", 50.0), ("on3", 50.0)):
        g = pd.DataFrame({"k": xe[kol].to_numpy(), "y": ye}).groupby("k")["y"]
        ort, n = g.mean(), g.size()
        kod = (ort * n + taban * duzgun) / (n + duzgun)
        se = xe[kol].map(kod).fillna(taban).to_numpy()
        ss = xs[kol].map(kod).fillna(taban).to_numpy()
        a_e, a_s = auc(ye, se), auc(ys, ss)
        print(f"  {kol:<6} tek basina  AUC egit {a_e:.4f}   AUC SINA {a_s:.4f}")
        skor_e += np.log(np.clip(se, 1e-6, 1) / taban)
        skor_s += np.log(np.clip(ss, 1e-6, 1) / taban)

    for kol in ("lg_guc", "yas"):
        a_s = auc(ys, xs[kol].to_numpy())
        print(f"  {kol:<6} tek basina  AUC SINA {a_s:.4f} (yon: {'+' if a_s > 0.5 else '-'})")

    print(
        f"\n  >>> BIRLESIK (ilce+on2+on3)  AUC egit {auc(ye, skor_e):.4f}   "
        f"AUC SINA {auc(ys, skor_s):.4f}"
    )
    print("  docs/41 §3 referansi (kis26 soguk, farkli nufus): AUC 0,5728")

    # trafo duzeyi
    ts = pd.DataFrame({"t": sina["tanim"].to_numpy(), "y": ys, "s": skor_s})
    tg = ts.groupby("t").agg(y=("y", "mean"), s=("s", "mean"), n=("y", "size"))
    tg = tg[tg["n"] >= 30]
    a_t = auc(tg["y"].to_numpy() > 0.5, tg["s"].to_numpy())
    print(f"  >>> TRAFO duzeyinde (>%50 sifir = olu) AUC SINA {a_t:.4f}   (docs/41: 0,5648)")
    ust = tg.nlargest(10, "s")
    print(f"  en yuksek 10 trafonun gercek sifir orani: {ust['y'].round(3).tolist()}")

    print("\n" + "=" * 92)
    print("HUKUM")
    print("=" * 92)
    print("  AUC ~0,57 civarindaysa hurdle hukmu YANLIS NUFUSTA verilmis olsa da")
    print("  SONUCU DEGISMIYOR -> denetim temiz, eksen kapali kalir.")
    print("  AUC belirgin yuksekse (>0,65) eksen TOPLU nufusunda YENIDEN ACILIR.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
