"""H8 -- GUN EKSENI OLCEKLEMESI SOGUK SATIRLARA UYGULANIYOR MU?

KAPI DENETIMINDEN CIKAN IPUCU
-----------------------------
v50_nihai30 -> v55_gunolcek -> v66_c1335 zincirinde SOGUK satirlarin
ortalama log1p farki TAM SIFIR. Yani gun ekseni olcekleme (LB'de dogrulanmis
TEK yapisal kazanc) yalnizca SICAK satirlara uygulanmis olabilir.

Soguk satirlar test satirlarinin %22'si ama MSE'nin %63'u. Gun ekseni genligi
soguk tarafta da AZ yayilmissa, orada uygulanmamis bir kazanc duruyor.

BU BETIK NE YAPAR
-----------------
1. Gonderim dosyalari arasinda sicak/soguk kirilimli DEGISIM sayar --
   olcekleme gercekten soguga dokunmamis mi, kesinlestirir.
2. Soguk tarafin gun ekseni genligini olcer (trafo etkisi cikarilmis):
   - 2026 test tahmininde (sampiyon dosyadan)
   - 2025 Nis-Tem GERCEK'te, soguk-BENZERI trafolarda (2025 Nis-Tem'de
     dogmus, yani o pencerede "yeni" olan trafolar) -- ETIKETSIZ CAPA
3. c_soguk = kor * sigma_gercek / sigma_model oranini turetir.
4. Beklenen dMSE'yi p_soguk = 0,22159 payiyla verir.

KURAL 6: gun ekseni olcumu TRAFO ETKISI CIKARILMADAN yapilmaz.
KURAL 7: capa 2025 Nis-Tem'den (testin mevsimsel ikizi) alinir.
Test etiketi KULLANILMAZ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
P_SOGUK = 0.22159
P_SICAK = 0.77841


def iki_yonlu(
    df: pd.DataFrame, deger: str, birim: str = "tanim", eksen: str = "tarih", tur: int = 40
) -> pd.Series:
    """Iki yonlu ortalama cikarma (trafo + gun sabit etkileri). Gun etkisini dondurur."""
    v = df[deger].to_numpy(dtype="float64").copy()
    bi = pd.factorize(df[birim])[0]
    gi, gun_deg = pd.factorize(df[eksen])
    nb, ng = bi.max() + 1, gi.max() + 1
    a = np.zeros(nb)
    b = np.zeros(ng)
    mu = v.mean()
    for _ in range(tur):
        r = v - mu - b[gi]
        a = np.bincount(bi, r, minlength=nb) / np.maximum(np.bincount(bi, minlength=nb), 1)
        r = v - mu - a[bi]
        b = np.bincount(gi, r, minlength=ng) / np.maximum(np.bincount(gi, minlength=ng), 1)
        b -= b.mean()
    return pd.Series(b, index=pd.Index(gun_deg, name=eksen)).sort_index()


def main() -> int:
    print("=" * 74)
    print("1. GONDERIMLER ARASI DEGISIM -- sicak/soguk kirilimli")
    print("=" * 74)

    tr_tanim = set(
        pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})[
            "tanim"
        ].unique()
    )
    ss = pd.read_csv(KOK / "data/raw/sample_submission.csv", usecols=["id"])
    tanim = ss["id"].str.rsplit("_", n=1).str[0]
    sicak = tanim.isin(tr_tanim).to_numpy()

    def oku(ad: str) -> np.ndarray:
        d = pd.read_csv(KOK / "submissions" / ad)
        assert (d["id"].values == ss["id"].values).all(), ad
        return np.log1p(np.clip(d["tuketim"].to_numpy(dtype="float64"), 0, None))

    dosyalar = [
        "tuketim_v50_nihai30.csv",
        "tuketim_v55_gunolcek.csv",
        "tuketim_v66_c1335.csv",
        "tuketim_v67_c1335_olay.csv",
    ]
    lg = {a: oku(a) for a in dosyalar}

    taban = lg["tuketim_v50_nihai30.csv"]
    for ad in dosyalar[1:]:
        f = lg[ad] - taban
        ds, dc = f[sicak], f[~sicak]
        print(f"\n{ad}  (v50'ye gore)")
        print(
            f"  SICAK  degisen {int((np.abs(ds) > 1e-9).sum()):>7,}/{sicak.sum():,}"
            f"  ort {ds.mean():+.6f}  std {ds.std():.6f}  maxabs {np.abs(ds).max():.6f}"
        )
        print(
            f"  SOGUK  degisen {int((np.abs(dc) > 1e-9).sum()):>7,}/{(~sicak).sum():,}"
            f"  ort {dc.mean():+.6f}  std {dc.std():.6f}  maxabs {np.abs(dc).max():.6f}"
        )

    print()
    print("=" * 74)
    print("2. SOGUK TARAFIN GUN EKSENI GENLIGI -- 2026 TAHMIN (sampiyon)")
    print("=" * 74)

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    sam = pd.read_csv(KOK / "submissions/tuketim_v67_c1335_olay.csv")
    assert (sam["id"].values == te["id"].values).all(), "test.csv id sirasi gonderimle ayni degil"
    te["lg"] = np.log1p(np.clip(sam["tuketim"].to_numpy(dtype="float64"), 0, None))
    te["sicak"] = te["tanim"].isin(tr_tanim)

    for etiket, alt in (("SICAK", te[te["sicak"]]), ("SOGUK", te[~te["sicak"]])):
        # sifir tahminler gun ekseni olcumunu bozar -> ayri raporla, disla
        n0 = int((alt["lg"] == 0).sum())
        a2 = alt[alt["lg"] > 0]
        b = iki_yonlu(a2, "lg")
        print(f"\n{etiket}  n={len(alt):,} (sifir {n0:,} dislandi)  trafo {alt.tanim.nunique():,}")
        print(f"  gun ekseni std (trafo etkisi cikarilmis) = {b.std():.4f}")
        print(f"  ilk 5 gun {b.head(5).round(3).to_dict()}")
        # ay bazinda genlik
        aylik = b.groupby(b.index.to_period("M")).std()
        print(f"  ay bazinda std: {aylik.round(4).to_dict()}")

    print()
    print("=" * 74)
    print("3. ETIKETSIZ CAPA -- 2025 Nis-Tem GERCEK, 'yeni dogmus' trafolarda")
    print("=" * 74)

    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()

    pen = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")].copy()
    pen["lg"] = np.log1p(np.clip(pen["tuketim"].to_numpy(dtype="float64"), 0, None))
    pen["ilk"] = pen["tanim"].map(ilk)

    # SOGUK-BENZERI: pencere icinde dogmus (testteki soguk trafolarin durumu)
    yeni = pen[pen["ilk"] >= "2025-04-01"]
    eski = pen[pen["ilk"] < "2025-04-01"]

    for etiket, alt in (("2025 ESKI (sicak ikizi)", eski), ("2025 YENI (soguk ikizi)", yeni)):
        a2 = alt[alt["lg"] > 0]
        if a2["tanim"].nunique() < 20 or len(a2) < 500:
            print(f"\n{etiket}: yetersiz n ({len(a2):,} satir, {a2.tanim.nunique()} trafo)")
            continue
        b = iki_yonlu(a2, "lg")
        print(f"\n{etiket}  n={len(alt):,}  trafo {alt.tanim.nunique():,}")
        print(f"  GERCEK gun ekseni std = {b.std():.4f}")
        aylik = b.groupby(b.index.to_period("M")).std()
        print(f"  ay bazinda std: {aylik.round(4).to_dict()}")

    print()
    print("=" * 74)
    print("4. HUKUM ICIN GEREKEN ORAN")
    print("=" * 74)
    print("  c_soguk = kor * (sigma_gercek_2025yeni / sigma_model_2026soguk)")
    print("  kor: gun-of-year hizasinda 2025 GERCEK ile 2026 TAHMIN profil korelasyonu")
    print("  (asagida)")

    # gun-of-year hizasi: 2025 yeni-dogmus GERCEK vs 2026 soguk TAHMIN
    a2 = yeni[yeni["lg"] > 0]
    if a2["tanim"].nunique() >= 20 and len(a2) >= 500:
        b25 = iki_yonlu(a2, "lg")
        c2 = te[(~te["sicak"]) & (te["lg"] > 0)]
        b26 = iki_yonlu(c2, "lg")
        i25 = b25.index.dayofyear
        i26 = b26.index.dayofyear
        s25 = pd.Series(b25.to_numpy(), index=i25)
        s26 = pd.Series(b26.to_numpy(), index=i26)
        ortak = s25.index.intersection(s26.index)
        if len(ortak) >= 30:
            kor = float(np.corrcoef(s25.loc[ortak], s26.loc[ortak])[0, 1])
            oran = float(b25.std() / b26.std())
            c = kor * oran
            print(f"\n  ortak gun sayisi        {len(ortak)}")
            print(f"  sigma_gercek (2025 yeni) {b25.std():.4f}")
            print(f"  sigma_model  (2026 soguk){b26.std():.4f}")
            print(f"  oran                     {oran:.4f}")
            print(f"  korelasyon               {kor:+.4f}")
            print(f"  >>> c_soguk = kor * oran = {c:.4f}")
            print("\n  NOT: c_soguk ~ 1 ise yapacak bir sey YOK (hukum CURUDU).")
            print("       c_soguk belirgin > 1 ise soguk gun ekseni AZ yayilmis.")
        else:
            print(f"\n  ortak gun yetersiz ({len(ortak)}) -- hukum verilemez")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
