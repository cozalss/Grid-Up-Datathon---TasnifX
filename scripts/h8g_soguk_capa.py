"""H8g -- SOGUK GUN EKSENI ICIN ETIKETSIZ CAPA (son_islem_gunolcek.py yolu).

NEDEN BU YOL
------------
h8f: (c_dusuk, c_yuksek) izgarasinda yaz25 ile guz25'in ORTAK negatif bolgesi
YOK. Tek genlik de, frekans ayrimi da iki blokta ZIT yon soyluyor
(yaz25 c*=2,78 | guz25 c*=0,67).

Bu bir CURUME DEGIL, gun ekseni genliginin MEVSIMSEL oldugunun kanitidir --
ve SICAK tarafta da ayni desen olculmustu (yaz25 2,65 | guz25 0,75 | kis26 0,70,
son_islem_gunolcek.py). Orada karar bloklar arasi tasima ile DEGIL, ETIKETSIZ
CAPA ile verildi ve LB'de KAZANDI (1,01750 -> 1,01591).

Mevsime bagli bir parametre icin "iki ortusmeyen kesme ayni isareti versin"
kapisi MANTIKEN saglanamaz: parametrenin kendisi mevsimle degisiyor.
O sinifta gecerli kapi sudur:
    (a) capa ETIKETSIZ olmali (test etiketi kullanilmaz -- kural 5),
    (b) capa formulu, ETIKETLI optimumu URETEBILDIGI ispatlanmis olmali.

BU BETIK (b)'yi SINAR, sonra (a)'yi UYGULAR.

ADIM 1 -- KALIBRASYON. yaz25 SOGUK ikiz panelinde (T3 temiz) capa formulu
    c_capa = kor * sigma_gercek / sigma_model
etiketli kesin optimumu (c* = 2,78) UREYEBILIYOR mu? Ayni sinama guz25'te de
(c* = 0,67). Formul iki blokta da optimumu tutturuyorsa GUVENILIR.

ADIM 2 -- UYGULAMA. Test soguk satirlari icin:
    sigma_gercek <- 2025 Nis-Tem, pencerede DOGMUS trafolar, GERCEK (train, etiketli)
    sigma_model  <- 2026 Nis-Tem, test soguk satirlari, SAMPIYON tahmini
    kor          <- gun-of-year hizasinda iki profilin korelasyonu
Test etiketi HIC kullanilmaz.

PROTOKOL HER IKI TARAFTA DA AYNI (yoksa oran anlamsiz):
    ilk 7 gun atilir (dogum/olay eseri) + >=60 gunluk trafolar (kucuk-n)
    iki yonlu sabit etki (kural 6: trafo etkisi cikarilmadan olculmez)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"
SAMPIYON = "tuketim_v67_c1335_olay.csv"
P_SOGUK = 0.22159
MIN_YAS, MIN_GUN = 7, 60


def iki_yonlu(v, bi, gi, nb, ng, tur=400):
    mu = float(v.mean())
    a = np.zeros(nb)
    b = np.zeros(ng)
    cb = np.maximum(np.bincount(bi, minlength=nb), 1)
    cg = np.maximum(np.bincount(gi, minlength=ng), 1)
    for _ in range(tur):
        a = np.bincount(bi, v - mu - b[gi], minlength=nb) / cb
        b = np.bincount(gi, v - mu - a[bi], minlength=ng) / cg
        b -= b.mean()
    return a, b, mu


def t3_maske(d: pd.DataFrame) -> np.ndarray:
    ilk = d.groupby("tanim")["tarih"].transform("min")
    yas = (d["tarih"] - ilk).dt.days.to_numpy()
    say = d.groupby("tanim")["tanim"].transform("size").to_numpy()
    return (yas >= MIN_YAS) & (say >= MIN_GUN)


def profil(
    d: pd.DataFrame, deger: np.ndarray
) -> tuple[pd.Series, np.ndarray, np.ndarray, int, int]:
    bi, _ = pd.factorize(d["tanim"])
    gi, gun = pd.factorize(d["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    a, b, mu = iki_yonlu(deger, bi, gi, nb, ng)
    return pd.Series(b, index=pd.Index(gun, name="tarih")).sort_index(), bi, gi, nb, ng


def kesin_c(r, bm, gi, ng):
    n_d = np.bincount(gi, minlength=ng).astype(float)
    pay = float(np.dot(np.bincount(gi, r, minlength=ng), bm))
    payda = float(np.dot(n_d, bm**2))
    return 1.0 + pay / payda if payda > 0 else 1.0


def kalibrasyon(ad: str) -> None:
    m = pd.read_parquet(ONBELLEK / f"{ad}_meta.parquet").reset_index(drop=True)
    mask = t3_maske(m)
    a = m.loc[mask].reset_index(drop=True)
    lgy = np.log1p(np.clip(a["y"].to_numpy(dtype="float64"), 0, None))
    pg, bi, gi, nb, ng = profil(a, lgy)
    n_d = np.bincount(gi, minlength=ng).astype(float)

    print(f"\n--- {ad}  T3 panel {len(a):,} satir, {a.tanim.nunique()} trafo, {ng} gun")
    sat = []
    for p in sorted(ONBELLEK.glob(f"{ad}_*_taban.npy")):
        pr = np.load(p).astype("float64")[mask]
        _, bm, _ = iki_yonlu(pr, bi, gi, nb, ng)
        bg = pg.to_numpy()
        # gun sirasi pg'de sirali; bm ise factorize sirasinda -> hizala
        sira = np.argsort(pd.factorize(a["tarih"])[1].values)
        bm_s = bm[sira]
        kor = float(np.corrcoef(bg, bm_s)[0, 1])
        c_capa = kor * float(bg.std()) / float(bm_s.std())
        c_yildiz = kesin_c(lgy - pr, bm, gi, ng)
        sat.append(
            {
                "tohum": p.stem.split("_")[1],
                "sig_g": bg.std(),
                "sig_m": bm_s.std(),
                "kor": kor,
                "c_capa": c_capa,
                "c_ETIKETLI": c_yildiz,
                "oran": c_capa / c_yildiz,
            }
        )
    d = pd.DataFrame(sat)
    print(d.to_string(index=False, float_format=lambda x: f"{x:9.4f}"))
    print(
        f"  ort  c_capa {d.c_capa.mean():.3f}   c_ETIKETLI {d.c_ETIKETLI.mean():.3f}"
        f"   ORAN capa/etiketli = {d.oran.mean():.3f}  (1,00 olmali)"
    )


def main() -> int:
    print("=" * 88)
    print("ADIM 1 -- CAPA FORMULU ETIKETLI OPTIMUMU UREYEBILIYOR MU?")
    print("=" * 88)
    for ad in ("yaz25", "guz25"):
        kalibrasyon(ad)

    print("\n" + "=" * 88)
    print("ADIM 2 -- TEST SOGUK ICIN ETIKETSIZ CAPA")
    print("=" * 88)

    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk_tum = tr.groupby("tanim")["tarih"].min()

    # --- 2025 GERCEK: Nis-Tem penceresinde DOGMUS trafolar (soguk ikiz)
    g25 = tr[(tr["tarih"] >= "2025-04-01") & (tr["tarih"] <= "2025-07-31")].copy()
    g25["ilk_tum"] = g25["tanim"].map(ilk_tum)
    g25 = g25[g25["ilk_tum"] >= pd.Timestamp("2025-04-01")].reset_index(drop=True)
    m25 = t3_maske(g25)
    a25 = g25.loc[m25].reset_index(drop=True)
    lg25 = np.log1p(np.clip(a25["tuketim"].to_numpy(dtype="float64"), 0, None))
    p25, *_ = profil(a25, lg25)
    print(
        f"\n2025 GERCEK (soguk ikiz, T3): {len(a25):,} satir, "
        f"{a25.tanim.nunique()} trafo, {p25.size} gun"
    )
    print(f"  sigma_gercek = {p25.std():.4f}")

    # --- 2026 MODEL: test soguk satirlari, SAMPIYON tahmini
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    sam = pd.read_csv(KOK / "submissions" / SAMPIYON)
    assert (sam["id"].values == te["id"].values).all()
    tr_set = set(tr["tanim"].unique())
    te["lg"] = np.log1p(np.clip(sam["tuketim"].to_numpy(dtype="float64"), 0, None))
    tc = te[~te["tanim"].isin(tr_set)].reset_index(drop=True)
    m26 = t3_maske(tc)
    a26 = tc.loc[m26].reset_index(drop=True)
    p26, *_ = profil(a26, a26["lg"].to_numpy(dtype="float64"))
    print(
        f"\n2026 MODEL (test soguk, T3, {SAMPIYON}): {len(a26):,} satir, "
        f"{a26.tanim.nunique()} trafo, {p26.size} gun"
    )
    print(f"  sigma_model  = {p26.std():.4f}")

    # --- gun-of-year hizasi
    s25 = pd.Series(p25.to_numpy(), index=p25.index.dayofyear)
    s26 = pd.Series(p26.to_numpy(), index=p26.index.dayofyear)
    ortak = s25.index.intersection(s26.index)
    kor = float(np.corrcoef(s25.loc[ortak], s26.loc[ortak])[0, 1])
    oran = float(p25.std() / p26.std())
    c_capa = kor * oran

    print(f"\n  ortak gun (gun-of-year)  {len(ortak)}")
    print(f"  oran   sigma_g/sigma_m    {oran:.4f}")
    print(f"  korelasyon               {kor:+.4f}")
    print(f"  >>> c_soguk_capa = kor * oran = {c_capa:.4f}")

    print("\n" + "=" * 88)
    print("HUKUM")
    print("=" * 88)
    print("  ADIM 1'deki ORAN (capa/etiketli) 1,00'e yakinsa capa GUVENILIR ve")
    print("  c_soguk_capa uretime BUZULEREK yazilabilir.")
    print("  Uzaksa formul bu rejimde kalibre DEGIL -> hukum CURUDU.")
    print("\n  Referans: SICAK tarafta ayni formul c*=1,335 verdi ve LB'de KAZANDI.")
    print(f"  Soguk paydasi p_soguk = {P_SOGUK}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
