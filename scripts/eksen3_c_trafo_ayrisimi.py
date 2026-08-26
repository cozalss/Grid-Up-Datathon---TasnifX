# ruff: noqa
"""EKSEN 3 (c)+(d)+(e): trafo bazinda yanliligin ayrisimi -- HAREKET mi BUZME mi?

    b_i = ort(gercek ofs) - ort(tahmin ofs)
        = [gercek_i - kesme_oncesi_seviye_i] - [tahmin_i - kesme_oncesi_seviye_i]
        =        GERCEK HAREKET            -        MODELIN ONGORDUGU HAREKET

Uc blokta ayni ayrisim; hangi bilesen ayni isaret ve buyuklukte?

    python scripts/eksen3_c_trafo_ayrisimi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

AILELER = ("cat", "xgb", "lgbm")
AGIRLIK = (3.0, 1.0, 1.0)
ONBELLEK = KOK / "data" / "interim" / "deney" / "sicak_tahmin.npz"
CIKTI = KOK / "reports" / "_eksen3_trafo.csv"


def blok_verisi(egitim: pd.DataFrame, blok: str, z) -> pd.DataFrame:
    _, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
    pay = sum(AGIRLIK)
    loglar = [
        sum(AGIRLIK[i] * z[f"{blok}_{t}_{a}"] for i, a in enumerate(AILELER)) / pay
        for t in di.TOHUMLAR
    ]
    log_t = np.mean(loglar, axis=0)
    dg = dogrulama[~soguk].reset_index(drop=True)
    lg = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    y = np.clip(gercek[~soguk], 0, None)
    return pd.DataFrame(
        {
            "tanim": dg["tanim"].to_numpy(),
            "tarih": pd.to_datetime(dg["tarih"]).to_numpy(),
            "guc": dg["guc"].to_numpy(dtype="float64"),
            "r": log_t - lg,
            "g": np.log1p(y) - lg,
            "sifir": (y <= 0).astype("float64"),
        }
    )


def agirlikli(x: pd.Series, w: pd.Series) -> float:
    return float((x * w).sum() / w.sum())


def main() -> int:
    z = np.load(ONBELLEK)
    egitim, _ = d.cerceveleri_kur()
    ham, _te = tm.yukle()
    ham = ham.copy()
    ham["tarih"] = pd.to_datetime(ham["tarih"])
    ham["ofs"] = np.log1p(np.clip(ham["tuketim"].to_numpy(dtype="float64"), 0, None)) - np.log1p(
        ham["guc"].to_numpy(dtype="float64")
    )
    ham["sifir"] = (ham["tuketim"] <= 0).astype("float64")

    tum = []
    for b in tm.BLOKLAR:
        kesme = pd.Timestamp(b.etiket_basi)
        v = blok_verisi(egitim, b.ad, z)
        gr = v.groupby("tanim")
        T = pd.DataFrame(
            {
                "n": gr.size(),
                "g": gr["g"].mean(),
                "r": gr["r"].mean(),
                "sifir_hedef": gr["sifir"].mean(),
                "guc": gr["guc"].first(),
            }
        )
        T["b"] = T["g"] - T["r"]

        # --- kesme oncesi (AS-OF, kural 7) ozetler
        gec = ham[ham["tarih"] < kesme]
        p90 = gec[gec["tarih"] >= kesme - pd.Timedelta(days=90)]
        p180 = gec[
            (gec["tarih"] >= kesme - pd.Timedelta(days=180))
            & (gec["tarih"] < kesme - pd.Timedelta(days=90))
        ]
        T["s90"] = p90.groupby("tanim")["ofs"].mean()
        T["s180"] = p180.groupby("tanim")["ofs"].mean()
        T["s_uzun"] = gec.groupby("tanim")["ofs"].mean()
        T["sifir_gec"] = gec.groupby("tanim")["sifir"].mean()
        T["omur"] = gec.groupby("tanim")["tarih"].size()
        T["bayat"] = (kesme - gec.groupby("tanim")["tarih"].max()).dt.days
        T["egim_gec"] = T["s90"] - T["s180"]

        # --- gecen yil ayni takvim aylari (YALNIZ kis26'da mumkun: Oca-Mar)
        if b.ad == "kis26":
            gy = ham[(ham["tarih"] >= "2025-01-01") & (ham["tarih"] <= "2025-03-31")]
            T["gy_ayni_ay"] = gy.groupby("tanim")["ofs"].mean()
        else:
            T["gy_ayni_ay"] = np.nan

        T["blok"] = b.ad
        T = T.dropna(subset=["s90"])
        T["hareket"] = T["g"] - T["s90"]  # GERCEK hareket (kesme sonrasi)
        T["ongoru"] = T["r"] - T["s90"]  # MODELIN ongordugu hareket
        tum.append(T.reset_index())

    A = pd.concat(tum, ignore_index=True)
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    A.to_csv(CIKTI, index=False)

    print("=" * 100)
    print("(c)+(e)  ORTALAMA AYRISIMI:  b = GERCEK HAREKET - MODELIN ONGORDUGU HAREKET")
    print("        (kesme oncesi son 90 gun seviyesi taban alinarak; satir agirlikli)")
    print("=" * 100)
    print(
        f"  {'blok':<7}{'trafo':>7}{'b':>9}{'HAREKET':>10}{'ONGORU':>10}"
        f"{'|':>3}{'std(b)':>8}{'std(har)':>10}{'std(ong)':>10}{'kor':>8}"
    )
    for ad in ("yaz25", "guz25", "kis26"):
        T = A[A["blok"] == ad]
        w = T["n"]
        print(
            f"  {ad:<7}{len(T):>7,}{agirlikli(T['b'], w):>+9.4f}"
            f"{agirlikli(T['hareket'], w):>+10.4f}{agirlikli(T['ongoru'], w):>+10.4f}"
            f"{'|':>3}{T['b'].std():>8.4f}{T['hareket'].std():>10.4f}"
            f"{T['ongoru'].std():>10.4f}{T['hareket'].corr(T['ongoru']):>+8.3f}"
        )
    print(
        "\n  OKUMA: ONGORU ~ 0 ise model 'son bilinen seviyede kal' diyor;"
        " HAREKET buyukse yanlilik GERCEK MEVSIMSEL HAREKETTEN geliyor."
    )

    print()
    print("=" * 100)
    print("(c)  BUZME TESTI: hareket ve ongoru, KESME ONCESI SEVIYEYE gore nasil degisiyor?")
    print("     (yuksek seviyeli trafo daha mi asagi cekiliyor?)")
    print("=" * 100)
    for ad in ("yaz25", "guz25", "kis26"):
        T = A[A["blok"] == ad].dropna(subset=["s90", "hareket", "ongoru"])
        x = T["s90"].to_numpy()
        xc = x - x.mean()
        for isim, y in (
            ("GERCEK hareket", T["hareket"].to_numpy()),
            ("MODEL ongoru", T["ongoru"].to_numpy()),
        ):
            eg = float(np.polyfit(xc, y, 1)[0])
            print(
                f"  {ad:<7}{isim:<16} egim(seviye) = {eg:+.4f}   kor = {np.corrcoef(xc, y)[0, 1]:+.3f}"
            )
        eg_b = float(np.polyfit(xc, T["b"].to_numpy(), 1)[0])
        print(
            f"  {ad:<7}{'-> b uzerinde':<16} egim(seviye) = {eg_b:+.4f}"
            f"   (negatif = model YUKSEK seviyeyi ASIRI asagi cekiyor = BUZME)\n"
        )

    print("=" * 100)
    print("(c)  kis26'da KENDI YoY BUYUMESI mi, BUZME mi?  (Oca-Mar 2026 vs Oca-Mar 2025)")
    print("=" * 100)
    T = A[(A["blok"] == "kis26")].dropna(subset=["gy_ayni_ay", "s90"]).copy()
    T["yoy"] = T["g"] - T["gy_ayni_ay"]  # SIZINTILI (hedef penceresini kullanir)
    y = T["b"].to_numpy()
    reg = {
        "yoy (kendi buyumesi, SIZINTILI)": T["yoy"].to_numpy(),
        "s90 (kesme oncesi seviye)": T["s90"].to_numpy(),
        "egim_gec (kesme oncesi egim)": T["egim_gec"].fillna(0).to_numpy(),
        "sifir_gec": T["sifir_gec"].to_numpy(),
        "log_guc": np.log1p(T["guc"].to_numpy()),
    }
    print(f"  n={len(T):,}  b ort {agirlikli(T['b'], T['n']):+.4f}  b std {y.std():.4f}")
    print(f"  {'regresor':<34}{'tek basina R2':>15}{'std beta':>11}")
    for k, v in reg.items():
        v = np.nan_to_num(v, nan=float(np.nanmean(v)))
        c = np.corrcoef(v, y)[0, 1]
        print(f"  {k:<34}{c**2:>15.4f}{c:>11.3f}")
    # Yildiz ifadesi indis icinde Python 3.11+ sozdizimi; proje 3.10'u hedefliyor.
    # Listeyi onceden kurup np.c_'ye tek dizi olarak vermek her surumde calisir.
    sutunlar = [np.ones(len(T))]
    sutunlar += [np.nan_to_num(v, nan=float(np.nanmean(v))) for v in reg.values()]
    X = np.c_[tuple(sutunlar)]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2_tam = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    print(f"\n  TAM MODEL R2 = {r2_tam:.4f}")
    for i, k in enumerate(reg, start=1):
        sut = [j for j in range(X.shape[1]) if j != i]
        bb, *_ = np.linalg.lstsq(X[:, sut], y, rcond=None)
        r2e = 1 - ((y - X[:, sut] @ bb) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        print(f"    KISMI R2 [{k:<34}] {r2_tam - r2e:+.4f}   beta={beta[i]:+.4f}")
    print(
        "\n  NOT: 'yoy' hedef penceresinin kendisini icerir -- ACIKLAYICI, kestirici DEGIL."
        "\n  Kesme oncesi regresorlerin (s90, egim_gec, sifir_gec, log_guc) toplam R2'si:"
    )
    sut = [0, 2, 3, 4, 5]
    bb, *_ = np.linalg.lstsq(X[:, sut], y, rcond=None)
    print(f"    {1 - ((y - X[:, sut] @ bb) ** 2).sum() / ((y - y.mean()) ** 2).sum():.4f}")

    print()
    print("=" * 100)
    print("(d)  SIFIR / OLU ETKISI:  b, sifir orani kovalarinda")
    print("=" * 100)
    for ad in ("yaz25", "guz25", "kis26"):
        T = A[A["blok"] == ad].copy()
        T["kova"] = pd.cut(
            T["sifir_hedef"],
            [-0.01, 1e-9, 0.05, 0.25, 0.75, 1.01],
            labels=["hedefte hic sifir yok", "<=5%", "5-25%", "25-75%", ">75%"],
        )
        print(f"\n  --- {ad}   (satir agirlikli genel b = {agirlikli(T['b'], T['n']):+.4f})")
        print(f"    {'hedef sifir kovasi':<24}{'trafo':>7}{'satir pay':>11}{'b':>9}{'katki':>9}")
        toplam_w = T["n"].sum()
        for k in T["kova"].cat.categories:
            s = T[T["kova"] == k]
            if not len(s):
                continue
            pay = s["n"].sum() / toplam_w
            bk = agirlikli(s["b"], s["n"])
            print(f"    {k:<24}{len(s):>7,}{pay:>11.4f}{bk:>+9.4f}{pay * bk:>+9.4f}")
        temiz = T[T["sifir_hedef"] <= 1e-9]
        print(
            f"    >> SIFIRSIZ trafolarla b = {agirlikli(temiz['b'], temiz['n']):+.4f}"
            f"  (tumu {agirlikli(T['b'], T['n']):+.4f})"
        )

    print()
    print("=" * 100)
    print("(kural 1)  EN BUYUK K TRAFO ATILINCA blok ortalama b")
    print("=" * 100)
    print(f"  {'blok':<7}" + "".join(f"{'K=' + str(k):>11}" for k in (0, 1, 5, 10, 25, 50)))
    for ad in ("yaz25", "guz25", "kis26"):
        T = A[A["blok"] == ad].copy()
        T["katki"] = (T["b"] * T["n"]).abs()
        T = T.sort_values("katki", ascending=False)
        satir = f"  {ad:<7}"
        for k in (0, 1, 5, 10, 25, 50):
            s = T.iloc[k:]
            satir += f"{agirlikli(s['b'], s['n']):>+11.4f}"
        print(satir)

    print()
    print("=" * 100)
    print("(e)  BLOKLARARASI TASINABILIRLIK:  ortak trafolarda b_i korelasyonu")
    print("=" * 100)
    P = A.pivot_table(index="tanim", columns="blok", values="b")
    H = A.pivot_table(index="tanim", columns="blok", values="hareket")
    O = A.pivot_table(index="tanim", columns="blok", values="ongoru")
    for isim, M in (("b_i", P), ("hareket_i", H), ("ongoru_i", O)):
        print(f"\n  --- {isim}")
        for a1, a2 in (("yaz25", "guz25"), ("yaz25", "kis26"), ("guz25", "kis26")):
            s = M[[a1, a2]].dropna()
            print(f"    {a1} x {a2}:  n={len(s):,}  kor={s[a1].corr(s[a2]):+.3f}")
    print(f"\n  trafo tablosu yazildi: {CIKTI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
