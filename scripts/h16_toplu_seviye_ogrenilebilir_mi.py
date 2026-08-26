"""H16 -- SOGUK TRAFO SEVIYESI, TOPLU-KATILIM nufusunda ogrenilebilir mi?

NEDEN (denetim, yasak bolge ihlali degil)
------------------------------------------
"soguk trafo seviyesi" ekseni UC KEZ kapatildi:
    docs/42 §5   ofset std 1,9357; en iyi kestirici (ilce) R2 0,016
    docs/43      as-of OOF R2 0,015  "kapali (ucuncu kez dogrulandi)"
    kimlik komsulugu   R2 0,019
Ama H14 olctu ki bu hukumlerin verildigi nufus (kis26 soguk: %59 tekil,
%1 toplu) TEST sogugunun (%12 tekil, %81 TOPLU) ikizi DEGIL.

Ve mekanizma farkli olmali: toplu katilim, ZATEN CALISAN trafolarin veri
setine geriye dolgusu. Boyle bir trafonun seviyesi komsulariyla/kVA'siyla
daha tutarli olabilir -- cunku gercekten yeni enerjilendirilmis bir trafo
gibi "musteri baglanana kadar rastgele" degil.

ODUL: soguk trafo seviyesi, b_soguk'un (0,16 -> -0,00567) yasadigi eksen.
Trafo BAZINDA kestirilebiliyorsa kuresel kaymadan cok daha degerli.

TASARIM -- H15 ile AYNI, IKI ORTUSMEYEN ZAMAN KESMESI (kural 9)
----------------------------------------------------------------
    EGIT : 2025-01-01 toplu kohortu
    SINA : 2025-11-25 toplu kohortu
Hedef  : trafonun ORTALAMA log-ofseti  r = log1p(tuketim) - log1p(guc)
         (sifir gunler DISLANIR -- onlar ayri eksen, H15'te olculdu)
Ozellik: YALNIZCA dogumda bilinenler -- kVA, lokasyon, tanim onekleri,
         parti buyuklugu. Trafonun KENDI gecmisi YOK.

KARSILASTIRMA TABANI: kohortun kendi ortalamasi (kuresel kayma). Bir kestirici
ancak bunun USTUNE cikarsa deger uretir -- b_soguk zaten kuresel kaymayi
yakaliyor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ILK_GUN_AT = 7
EGIT_GUN = pd.Timestamp("2025-01-01")
SINA_GUN = pd.Timestamp("2025-11-25")


def kohort(tr: pd.DataFrame, ilk: pd.Series, gun: pd.Timestamp, ufuk: int = 127) -> pd.DataFrame:
    bit = min(gun + pd.Timedelta(days=ufuk), tr["tarih"].max())
    t = set(ilk[ilk == gun].index)
    a = tr[tr["tanim"].isin(t) & (tr["tarih"] >= gun) & (tr["tarih"] <= bit)]
    return a[(a["yas"] >= ILK_GUN_AT) & (a["tuketim"] > 0)].copy()


def trafo_seviye(a: pd.DataFrame) -> pd.DataFrame:
    a = a.copy()
    a["r"] = np.log1p(a["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        a["guc"].to_numpy(dtype="float64")
    )
    g = a.groupby("tanim").agg(
        r=("r", "mean"), n=("r", "size"), guc=("guc", "first"), lokasyon=("lokasyon", "first")
    )
    return g[g["n"] >= 30]


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim", "lokasyon"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    tr["ilk"] = tr["tanim"].map(ilk)
    tr["yas"] = (tr["tarih"] - tr["ilk"]).dt.days

    ge = trafo_seviye(kohort(tr, ilk, EGIT_GUN))
    gs = trafo_seviye(kohort(tr, ilk, SINA_GUN))
    print("=" * 90)
    print("SOGUK TRAFO SEVIYESI -- TOPLU kohortlarda, iki ortusmeyen kesme")
    print("=" * 90)
    print(
        f"\n  EGIT {EGIT_GUN.date()}  {len(ge):,} trafo   "
        f"ofset ort {ge.r.mean():+.4f}  std {ge.r.std():.4f}"
    )
    print(
        f"  SINA {SINA_GUN.date()}  {len(gs):,} trafo   "
        f"ofset ort {gs.r.mean():+.4f}  std {gs.r.std():.4f}"
    )
    print("\n  docs/42 referansi: 'ofset std 1,9357; en iyi kestirici R2 0,016'")

    for d in (ge, gs):
        d["on2"] = d.index.str[:2]
        d["on3"] = d.index.str[:3]
        d["lg_guc"] = np.log1p(d["guc"].to_numpy(dtype="float64"))

    taban = float(ge.r.mean())
    y = gs.r.to_numpy()
    # TABAN 1: kohortun KENDI ortalamasi (kuresel kayma -- b_soguk'un yaptigi)
    sse_kendi = float(((y - y.mean()) ** 2).sum())
    # TABAN 0: egitim kohortunun ortalamasi (hicbir sey bilmeden)
    sse_egit = float(((y - taban) ** 2).sum())
    print(f"\n  SSE(sina kendi ortalamasi)  {sse_kendi:,.1f}   <- ASILACAK TABAN")
    print(f"  SSE(egit ortalamasi)        {sse_egit:,.1f}")
    print(
        f"  kuresel kaymanin degeri     R2 = "
        f"{1 - sse_kendi / max(sse_egit, 1e-9):.4f}  (b_soguk bunu yakaliyor)"
    )

    print(f"\n  {'kestirici':<26} {'R2 (kendi ort. uzerine)':>24} {'kor':>8}")
    for ad, kol, duzgun in (
        ("lokasyon (ilce)", "lokasyon", 5.0),
        ("tanim on2", "on2", 5.0),
        ("tanim on3", "on3", 5.0),
    ):
        g = pd.DataFrame({"k": ge[kol].to_numpy(), "y": ge.r.to_numpy()}).groupby("k")["y"]
        ort, n = g.mean(), g.size()
        kod = (ort * n + taban * duzgun) / (n + duzgun)
        p = gs[kol].map(kod).fillna(taban).to_numpy()
        # kuresel kaymayi ADIL kilmak icin: kestiriciyi sina ortalamasina merkezle
        p_m = p - p.mean() + y.mean()
        r2 = 1 - float(((y - p_m) ** 2).sum()) / max(sse_kendi, 1e-9)
        kor = float(np.corrcoef(p, y)[0, 1]) if np.std(p) > 1e-12 else float("nan")
        print(f"  {ad:<26} {r2:>24.4f} {kor:>8.4f}")

    # kVA
    from numpy.polynomial import polynomial as P

    kf = P.polyfit(ge.lg_guc.to_numpy(), ge.r.to_numpy(), 1)
    p = P.polyval(gs.lg_guc.to_numpy(), kf)
    p_m = p - p.mean() + y.mean()
    r2 = 1 - float(((y - p_m) ** 2).sum()) / max(sse_kendi, 1e-9)
    print(f"  {'kVA (dogrusal)':<26} {r2:>24.4f} {float(np.corrcoef(p, y)[0, 1]):>8.4f}")

    # hepsi birlikte: lokasyon + kVA
    g = pd.DataFrame({"k": ge["lokasyon"].to_numpy(), "y": ge.r.to_numpy()}).groupby("k")["y"]
    ort, n = g.mean(), g.size()
    kod = (ort * n + taban * 5.0) / (n + 5.0)
    p1 = gs["lokasyon"].map(kod).fillna(taban).to_numpy()
    p2 = P.polyval(gs.lg_guc.to_numpy(), kf)
    p = 0.5 * p1 + 0.5 * p2
    p_m = p - p.mean() + y.mean()
    r2 = 1 - float(((y - p_m) ** 2).sum()) / max(sse_kendi, 1e-9)
    print(f"  {'lokasyon + kVA harman':<26} {r2:>24.4f} {float(np.corrcoef(p, y)[0, 1]):>8.4f}")

    print("\n" + "=" * 90)
    print("HUKUM")
    print("=" * 90)
    print("  R2 ~0 veya NEGATIF ise eksen TOPLU nufusunda da kapali -> DENETIM TEMIZ.")
    print("  R2 > 0,05 ise trafo bazinda seviye kestirimi TOPLU nufusunda")
    print("  ogrenilebilir demektir -> 27 Agustos icin CANLI eksen.")
    print("\n  Not: R2 kohortun KENDI ortalamasi uzerine olculuyor, yani kuresel")
    print("  kayma (b_soguk) zaten cikarilmis. Sadece EK deger raporlaniyor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
