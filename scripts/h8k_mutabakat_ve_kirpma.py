"""H8k -- (1) BUYUKLUK MUTABAKATI ve (2) TAM KIRPMA TABLOSU.

SORU 1 -- MUTABAKAT
-------------------
Bu knob repoda ZATEN biliniyordu:
    docs/42 §2 (b): son_islem_gunolcek.py --yalniz-soguk --lb-kalibre 0.893
    docs/44 §3 sira 2: "soguk gun ekseni lb-kalibre 0,893"
Ikisinde de beklenti ~-0,0006. H8 -0,01487 diyor. 25 kat fark.

Uc aday: (a) eski c 1'e yakindi, ben 2,20 kullaniyorum; (b) eski beklenti
affin kalibre (BUG 2) ile hesaplandi; (c) eski beklenti farkli bir sey olcuyordu.

Betik ucunu de SAYIYLA ayirir. Kritik supheli docs/41'in su cumlesi:

    "v50 ham soguk gun ekseni std = 0,1626/0,60 = 0,2710 ve 2025 Nis-Tem
     GERCEK referansi da 0,2710. Ham model soguk gun genligini ZATEN DOGRU
     biliyor; onu bozan uretim buzmesidir."

Bu cumle 0,2710'u referans aliyor. Ama 0,2710, son_islem_gunolcek.py'nin capa
kodunda gorulecegi gibi, pencerenin >=%90'inda VAR OLAN trafolardan geliyor --
yani YERLESIK (sicak-benzeri) nufus. SOGUK satirlar ise pencerede YENI DOGMUS
trafolar. Ikisinin gun ekseni genligi ayni olmak zorunda degil: yeni trafoda
mevsimsel rampanin USTUNE musteri baglanma rampasi biniyor.

BU AYRIM MUTABAKATIN TAMAMI OLABILIR. Betik iki nufusun sigma'sini AYNI
pencerede, AYNI protokolle olcup karsilastirir.

SORU 2 -- TAM KIRPMA TABLOSU (kalici kural 1, pazarlik yok)
------------------------------------------------------------
K = 0,1,5,10,25,50 icin: kalan dMSE, kalan trafo, kalan satir, kazanan/kaybeden.
Hem TRAFO hem GUN bazli (gun ekseni genlik duzeltmesi icin trafo dogal birim
olmayabilir). Karsilastirma icin ESKI c de ayni tabloyla kosulur.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"
P_SOGUK = 0.22159
CAPA_BASI, CAPA_SONU = "2025-04-01", "2025-07-31"


def gun_etkisi_eski(tanim, gun, r) -> pd.Series:
    """son_islem_gunolcek.py'nin TAM AYNI fonksiyonu (tek turlu, esit-agirlik)."""
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


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
    return b


def bolum1() -> float:
    print("=" * 94)
    print("SORU 1 -- BUYUKLIK MUTABAKATI")
    print("=" * 94)

    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    g = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)]
    rg = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        g["guc"].to_numpy(dtype="float64")
    )
    tan_g = g["tanim"].to_numpy()
    gun_g = g["tarih"].to_numpy()

    # --- (A) ESKI CAPA NUFUSU: pencerenin >=%90'inda var olan trafolar
    x = pd.DataFrame({"t": tan_g, "g": gun_g})
    say = x.groupby("t")["g"].nunique()
    ngun = x["g"].nunique()
    yerlesik = set(say[say >= 0.9 * ngun].index)
    sec = np.isin(tan_g, list(yerlesik))
    b_yerlesik = gun_etkisi_eski(tan_g[sec], gun_g[sec], rg[sec])

    # --- (B) H8 CAPA NUFUSU: pencerede DOGMUS trafolar (soguk ikizi)
    ilk_tum = tr.groupby("tanim")["tarih"].min()
    dogmus = set(ilk_tum[ilk_tum >= pd.Timestamp(CAPA_BASI)].index)
    sec2 = np.isin(tan_g, list(dogmus))
    b_dogmus = gun_etkisi_eski(tan_g[sec2], gun_g[sec2], rg[sec2])

    print("\n2025-04-01..07-31 GERCEK gun ekseni std -- AYNI pencere, AYNI protokol")
    print(
        f"  (A) YERLESIK nufus (>=%90 gun var)  {len(yerlesik):>5} trafo  "
        f"{int(sec.sum()):>7,} satir   sigma = {b_yerlesik.std():.4f}"
    )
    print(
        f"  (B) DOGMUS  nufus (soguk ikizi)     {len(dogmus):>5} trafo  "
        f"{int(sec2.sum()):>7,} satir   sigma = {b_dogmus.std():.4f}"
    )
    print(f"      ORAN B/A = {b_dogmus.std() / b_yerlesik.std():.3f}")
    print("\n  docs/41: 'v50 ham soguk gun std 0,2710 ve 2025 Nis-Tem GERCEK")
    print("  referansi da 0,2710 -> ham model genligi ZATEN DOGRU biliyor'")
    print(f"  -> o referans (A)'dir: {b_yerlesik.std():.4f}. Ama SOGUK satirlar (B).")

    # --- ESKI RECETEYI GERCEKTEN KOS
    print("\n" + "-" * 94)
    print("ESKI RECETE KOSULUYOR: --yalniz-soguk --lb-kalibre 0.893")
    print("-" * 94)
    gecici = KOK / "submissions" / "_gecici_eski_soguk.csv"
    p = subprocess.run(
        [
            sys.executable,
            str(KOK / "scripts/son_islem_gunolcek.py"),
            "--giris",
            "submissions/tuketim_v67_c1335_olay.csv",
            "--cikis",
            str(gecici),
            "--yalniz-soguk",
            "--lb-kalibre",
            "0.893",
        ],
        capture_output=True,
        encoding="utf-8",
        cwd=str(KOK),
    )
    print(p.stdout.strip()[-1600:] if p.stdout else "")
    if p.returncode != 0:
        print("HATA:", (p.stderr or "")[-800:])
    c_eski = None
    for satir in (p.stdout or "").splitlines():
        for anahtar in ("c =", "c_kullan", "c formul", "kullanilan c", "c:"):
            if anahtar in satir:
                for parca in satir.replace("=", " ").replace(",", " ").split():
                    try:
                        v = float(parca)
                    except ValueError:
                        continue
                    if 0.3 <= v <= 3.0:
                        c_eski = v
                        break
            if c_eski:
                break
        if c_eski:
            break
    if c_eski is None:
        c_eski = 1.411
        print("  (c ciktidan okunamadi; docs/41'in belgeledigi 1,411 kullanilacak)")
    print(f"\n  >>> ESKI RECETENIN SECTIGI c = {c_eski:.4f}")
    print("  >>> H8'in sectigi          c = 2.2000")
    if gecici.exists():
        gecici.unlink()
    return float(c_eski)


def dmse_izgara(mask_ad: str, c_listesi) -> None:
    m = pd.read_parquet(ONBELLEK / "yaz25_meta.parquet").reset_index(drop=True)
    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(m["tanim"])
    gi, _ = pd.factorize(m["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng).astype(float)
    tohumlar = sorted(ONBELLEK.glob("yaz25_*_taban.npy"))

    print(
        f"\n{mask_ad}: yaz25 T0 ham panel, {len(m):,} satir, {nb} trafo, "
        f"{ng} gun, {len(tohumlar)} tohum -- SEVIYE-NOTR"
    )
    print(f"  {'c':>6} {'dMSE_panel':>12} {'SH':>9} {'t':>8} {'test etkisi':>13}")
    for c in c_listesi:
        per = []
        for p in tohumlar:
            pr = np.load(p).astype("float64")
            b = iki_yonlu(pr, bi, gi, nb, ng)
            bc = b - float(np.dot(n_d, b) / n_d.sum())
            r = lgy - pr
            per.append(float(((r - (c - 1) * bc[gi]) ** 2).mean()) - float((r**2).mean()))
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v))
        print(
            f"  {c:6.3f} {v.mean():+12.5f} {sh:9.5f} {v.mean() / sh:+8.2f} "
            f"{P_SOGUK * v.mean():+13.6f}"
        )


def kirpma_tam(c: float, birim: str) -> None:
    """birim = 'trafo' veya 'gun'."""
    m = pd.read_parquet(ONBELLEK / "yaz25_meta.parquet").reset_index(drop=True)
    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(m["tanim"])
    gi, _ = pd.factorize(m["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng).astype(float)
    tohumlar = sorted(ONBELLEK.glob("yaz25_*_taban.npy"))
    grup = bi if birim == "trafo" else gi
    ngrup = nb if birim == "trafo" else ng

    print(f"\n  KIRPMA -- {birim.upper()} bazli, c={c:.2f}, SEVIYE-NOTR, {len(tohumlar)} tohum")
    print(
        f"    {'K':>4} {'kalan dMSE':>12} {'SH':>9} {'t':>8} "
        f"{'kalan ' + birim:>12} {'kalan satir':>12} {'kazanan':>9} {'kaybeden':>9}"
    )
    for K in (0, 1, 5, 10, 25, 50):
        if ngrup <= K:
            break
        per, kaz, kyb, kalan_g, kalan_s = [], None, None, None, None
        for p in tohumlar:
            pr = np.load(p).astype("float64")
            b = iki_yonlu(pr, bi, gi, nb, ng)
            bc = b - float(np.dot(n_d, b) / n_d.sum())
            r = lgy - pr
            d = (r - (c - 1) * bc[gi]) ** 2 - r**2
            katki = np.bincount(grup, d, minlength=ngrup)
            at = np.argsort(katki)[:K]
            tut = ~np.isin(grup, at) if K else np.ones(len(d), bool)
            per.append(float(d[tut].mean()))
            if kaz is None:
                kt = katki[~np.isin(np.arange(ngrup), at)] if K else katki
                kaz = int((kt < 0).sum())
                kyb = int((kt > 0).sum())
                kalan_g = ngrup - K
                kalan_s = int(tut.sum())
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v))
        print(
            f"    {K:>4} {v.mean():+12.5f} {sh:9.5f} {v.mean() / sh:+8.2f} "
            f"{kalan_g:>12,} {kalan_s:>12,} "
            f"{kaz:>5,} ({kaz / max(kalan_g, 1):>4.0%}) {kyb:>5,}"
        )


def main() -> int:
    c_eski = bolum1()

    print("\n" + "=" * 94)
    print("MUTABAKAT ARITMETIGI -- IKI c AYNI PANELDE, AYNI OLCUTLE")
    print("=" * 94)
    dmse_izgara("KARSILASTIRMA", [1.00, c_eski, 1.50, 2.20, 2.60, 3.03])

    print("\n\n" + "=" * 94)
    print("SORU 2 -- TAM KIRPMA TABLOSU")
    print("=" * 94)
    kirpma_tam(2.20, "trafo")
    kirpma_tam(2.20, "gun")
    print("\n  (karsilastirma icin ESKI c)")
    kirpma_tam(c_eski, "trafo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
