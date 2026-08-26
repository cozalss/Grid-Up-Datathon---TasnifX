"""H8c -- SOGUK GUN EKSENI OLCEGI, MEVSIMSEL IKIZDE ETIKETLE OLCULUR.

SORU
----
Gun ekseni olceklemesi (LB'de dogrulanmis TEK yapisal kazanc, c*=1,335)
SOGUK satirlara HIC uygulanmiyor: v50 -> v55 -> v66 zincirinde soguk
satirlarin degisen sayisi TAM SIFIR (h8_soguk_gun_ekseni.py, bolum 1).

Soguk satirlar test satirlarinin %22'si ama MSE'nin %63'u
(RMSE_soguk ~ 1,713 vs RMSE_sicak ~ 0,700).

IKIZ PANEL
----------
data/interim/gun_ekseni/yaz25_meta.parquet -- 2025-04-06..07-31 penceresinde
DOGMUS 678 trafo, 20.633 satir. Bu tam olarak testin soguk rejiminin
mevsimsel ikizi: pencerede yeni, gecmissiz, ayni mevsim.
Dogrulama: bu panelde model RMSE 1,58; test soguk tarafinda 1,71. Uyusuyor.
yaz25_{1000..1005}_taban.npy -- ALTI tohum (kural 3: soguk tarafta uc yetmez).

YONTEM (son_islem_gunolcek.py ile AYNI cebir)
---------------------------------------------
    r = mu + a_trafo + b_gun + e            (iki yonlu, kural 6)
    b_gun_model  : TAHMINI ayristirinca cikan gun bileseni
    b_gun_gercek : GERCEGI ayristirinca cikan gun bileseni
    c* = kor(b_gercek, b_model) * sigma_gercek / sigma_model
    duzeltme: tahmin' = tahmin + (c - 1) * b_gun_model[gun]

GURULTU DUZELTMESI
------------------
b_gun kestirimi orneklem gurultusu tasir: var_gurultu[d] ~ sigma_e^2 / n_d.
Ham sigma bu yuzden SISMIS olur ve c*'i yukari cekerek YANLIS karar verdirir.
Betik hem HAM hem GURULTU-DUZELTILMIS sigma'yi raporlar; hukum duzeltilmise
gore verilir.

KAPILAR
-------
- ALTI tohum, eslenik SH (kural 3, 4)
- KIRPMA TABLOSU K = 0,1,5,10,25,50 (kural 1)
- Dengeli alt panel saglamlik kosusu (dengesiz panel giris etkisi yaratir)
- guz25'te de kosulur; isaret AYNI olmali (kural 7)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
ONBELLEK = KOK / "data/interim/gun_ekseni"


def iki_yonlu(
    v: np.ndarray, bi: np.ndarray, gi: np.ndarray, nb: int, ng: int, tur: int = 60
) -> tuple[np.ndarray, np.ndarray, float]:
    """Iki yonlu sabit etki. (a_birim, b_gun, mu) dondurur; b ortalamasi 0."""
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


def gurultu_duzelt(sig2: float, artik: np.ndarray, gi: np.ndarray, ng: int) -> float:
    """sigma^2_b'den orneklem gurultusunu cikar. Negatife duserse 0."""
    n_d = np.maximum(np.bincount(gi, minlength=ng), 1)
    s2e = float((artik**2).mean())
    gur = float(np.mean(s2e / n_d))
    return max(sig2 - gur, 0.0)


def blok_olc(ad: str, etiket: str) -> dict | None:
    mp = ONBELLEK / f"{ad}_meta.parquet"
    if not mp.exists():
        print(f"{etiket}: meta yok")
        return None
    m = pd.read_parquet(mp).reset_index(drop=True)
    tohumlar = sorted(ONBELLEK.glob(f"{ad}_*_taban.npy"))
    if not tohumlar:
        print(f"{etiket}: tohum yok")
        return None

    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    bi, _ = pd.factorize(m["tanim"])
    gi, gun_deg = pd.factorize(m["tarih"])
    nb, ng = int(bi.max()) + 1, int(gi.max()) + 1
    n_d = np.bincount(gi, minlength=ng)

    print("=" * 78)
    print(
        f"{etiket}   n={len(m):,}  trafo={nb:,}  gun={ng}  "
        f"gun basi ort n={n_d.mean():.1f} (min {n_d.min()}, max {n_d.max()})"
    )
    print("=" * 78)

    # GERCEK gun ekseni (tohumdan bagimsiz)
    ag, bg, mug = iki_yonlu(lgy, bi, gi, nb, ng)
    art_g = lgy - mug - ag[bi] - bg[gi]
    s2g_ham = float(bg.var())
    s2g_duz = gurultu_duzelt(s2g_ham, art_g, gi, ng)
    print(
        f"GERCEK gun ekseni  sigma_ham {np.sqrt(s2g_ham):.4f}   "
        f"sigma_duzeltilmis {np.sqrt(s2g_duz):.4f}   (artik sigma {art_g.std():.4f})"
    )

    satirlar = []
    c_izgara = np.round(np.arange(0.8, 3.61, 0.1), 2)
    dmse_izgara = {c: [] for c in c_izgara}
    per_tohum_bmodel = []

    for p in tohumlar:
        pr = np.load(p).astype("float64")
        am, bm, mum = iki_yonlu(pr, bi, gi, nb, ng)
        art_m = pr - mum - am[bi] - bm[gi]
        s2m_ham = float(bm.var())
        s2m_duz = gurultu_duzelt(s2m_ham, art_m, gi, ng)
        kor = float(np.corrcoef(bg, bm)[0, 1])
        # gurultu-duzeltilmis korelasyon: kovaryans gurultuden ETKILENMEZ
        # (iki tarafin gurultusu bagimsiz), sadece paydalar sisiyor
        kor_duz = float(np.cov(bg, bm)[0, 1] / np.sqrt(max(s2g_duz * s2m_duz, 1e-12)))
        kor_duz = float(np.clip(kor_duz, -1.0, 1.0))
        c_ham = kor * np.sqrt(s2g_ham / max(s2m_ham, 1e-12))
        c_duz = kor_duz * np.sqrt(s2g_duz / max(s2m_duz, 1e-12))
        # OLS egimi (gercek ~ a + c*model) -- ucuncu bagimsiz yol
        egim = float(np.cov(bg, bm)[0, 1] / max(np.var(bm), 1e-12))

        mse0 = float(((lgy - pr) ** 2).mean())
        for c in c_izgara:
            yeni = pr + (c - 1.0) * bm[gi]
            dmse_izgara[c].append(float(((lgy - yeni) ** 2).mean()) - mse0)

        satirlar.append(
            {
                "tohum": p.stem.split("_")[1],
                "mse0": mse0,
                "sig_m_ham": np.sqrt(s2m_ham),
                "sig_m_duz": np.sqrt(s2m_duz),
                "kor": kor,
                "kor_duz": kor_duz,
                "c_ham": c_ham,
                "c_duz": c_duz,
                "egim": egim,
            }
        )
        per_tohum_bmodel.append(bm)

    df = pd.DataFrame(satirlar)
    print("\nTOHUM BAZINDA")
    print(df.to_string(index=False, float_format=lambda x: f"{x:8.4f}"))
    print(f"\n  c_ham  ort {df.c_ham.mean():.4f} (std {df.c_ham.std():.4f})")
    print(f"  c_duz  ort {df.c_duz.mean():.4f} (std {df.c_duz.std():.4f})")
    print(f"  egim   ort {df.egim.mean():.4f} (std {df.egim.std():.4f})")

    print("\nc IZGARASI -- dMSE (log uzayi, bu panelde), 6 tohum eslenik")
    print(f"  {'c':>5} {'dMSE_ort':>11} {'eslenik_SH':>11} {'t':>7}  {'pozitif tohum':>13}")
    en_iyi_c, en_iyi_d = None, 0.0
    for c in c_izgara:
        v = np.array(dmse_izgara[c])
        sh = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        t = v.mean() / sh if sh and sh > 0 else float("nan")
        yildiz = ""
        if v.mean() < en_iyi_d:
            en_iyi_d, en_iyi_c = v.mean(), c
        if abs(c - round(df.c_duz.mean(), 1)) < 0.051:
            yildiz = "  <- c_duz"
        print(
            f"  {c:5.2f} {v.mean():+11.5f} {sh:11.5f} {t:+7.2f}  "
            f"{int((v < 0).sum()):>4}/{len(v)}{yildiz}"
        )
    print(f"\n  IZGARA OPTIMUMU c={en_iyi_c}  dMSE={en_iyi_d:+.5f}")

    return {
        "ad": ad,
        "etiket": etiket,
        "n": len(m),
        "nb": nb,
        "ng": ng,
        "sig_gercek_ham": float(np.sqrt(s2g_ham)),
        "sig_gercek_duz": float(np.sqrt(s2g_duz)),
        "c_ham": float(df.c_ham.mean()),
        "c_duz": float(df.c_duz.mean()),
        "egim": float(df.egim.mean()),
        "izgara_c": en_iyi_c,
        "izgara_dmse": en_iyi_d,
        "dmse_izgara": {float(k): np.array(v) for k, v in dmse_izgara.items()},
        "bi": bi,
        "gi": gi,
        "nb": nb,
        "ng": ng,
        "lgy": lgy,
        "meta": m,
        "tohumlar": tohumlar,
        "bm_list": per_tohum_bmodel,
    }


def kirpma_tablosu(sonuc: dict, c: float) -> None:
    """Kazanc birkac trafodan mi geliyor? En buyuk K katkiyi at (kural 1)."""
    m, bi, gi, lgy = sonuc["meta"], sonuc["bi"], sonuc["gi"], sonuc["lgy"]
    nb = sonuc["nb"]
    print(f"\nKIRPMA TABLOSU  (c={c})  -- en buyuk K katkili TRAFO atilinca")
    print(f"  {'K':>4} {'dMSE_ort':>11} {'eslenik_SH':>11} {'t':>7}  {'kazanan trafo':>15}")
    for K in (0, 1, 5, 10, 25, 50):
        per = []
        kazanan = None
        for p, bm in zip(sonuc["tohumlar"], sonuc["bm_list"]):
            pr = np.load(p).astype("float64")
            e0 = (lgy - pr) ** 2
            e1 = (lgy - (pr + (c - 1.0) * bm[gi])) ** 2
            d = e1 - e0
            katki = np.bincount(bi, d, minlength=nb)
            if kazanan is None:
                kazanan = (int((katki < 0).sum()), nb)
            sira = np.argsort(katki)  # en cok IYILESTIREN basta
            at = set(sira[:K].tolist())
            tut = ~np.isin(bi, list(at)) if K else np.ones(len(d), bool)
            per.append(float(d[tut].mean()))
        v = np.array(per)
        sh = v.std(ddof=1) / np.sqrt(len(v))
        t = v.mean() / sh if sh > 0 else float("nan")
        ek = f"  {kazanan[0]}/{kazanan[1]} ({kazanan[0] / kazanan[1]:.1%})" if K == 0 else ""
        print(f"  {K:>4} {v.mean():+11.5f} {sh:11.5f} {t:+7.2f}{ek}")


def dengeli_saglamlik(sonuc: dict, c: float) -> None:
    """Yalnizca pencerenin TAMAMINDA var olan trafolarla tekrarla."""
    m, gi, lgy = sonuc["meta"], sonuc["gi"], sonuc["lgy"]
    ng = sonuc["ng"]
    say = m.groupby("tanim")["tarih"].nunique()
    tam = set(say[say == say.max()].index)
    mask = m["tanim"].isin(tam).to_numpy()
    if mask.sum() < 2000:
        print(f"\nDENGELI ALT PANEL: yetersiz ({mask.sum()} satir, {len(tam)} trafo)")
        return
    bi2, _ = pd.factorize(m.loc[mask, "tanim"])
    gi2, _ = pd.factorize(m.loc[mask, "tarih"])
    nb2, ng2 = int(bi2.max()) + 1, int(gi2.max()) + 1
    lgy2 = lgy[mask]
    ag, bg, mug = iki_yonlu(lgy2, bi2, gi2, nb2, ng2)
    print(
        f"\nDENGELI ALT PANEL  {mask.sum():,} satir  {len(tam)} trafo  "
        f"{say.max()} gun   GERCEK sigma {bg.std():.4f}"
    )
    per = []
    for p in sonuc["tohumlar"]:
        pr = np.load(p).astype("float64")[mask]
        am, bm, mum = iki_yonlu(pr, bi2, gi2, nb2, ng2)
        e0 = float(((lgy2 - pr) ** 2).mean())
        e1 = float(((lgy2 - (pr + (c - 1.0) * bm[gi2])) ** 2).mean())
        per.append(e1 - e0)
    v = np.array(per)
    sh = v.std(ddof=1) / np.sqrt(len(v))
    print(
        f"  c={c}  dMSE {v.mean():+.5f}  eslenik_SH {sh:.5f}  "
        f"t {v.mean() / sh:+.2f}  pozitif {int((v < 0).sum())}/{len(v)}"
    )


def main() -> int:
    yaz = blok_olc("yaz25", "yaz25 SOGUK IKIZ (2025 Nis-Tem'de dogmus trafolar)")
    if yaz is None:
        return 1
    c_sec = round(float(yaz["c_duz"]), 1)
    kirpma_tablosu(yaz, c_sec)
    dengeli_saglamlik(yaz, c_sec)
    if yaz["izgara_c"] and abs(yaz["izgara_c"] - c_sec) > 0.15:
        print(f"\n  (izgara optimumu {yaz['izgara_c']} icin de kirpma:)")
        kirpma_tablosu(yaz, float(yaz["izgara_c"]))

    print("\n\n")
    guz = blok_olc("guz25", "guz25 (ikinci blok -- ISARET TUTARLILIGI kapisi)")
    if guz is not None:
        kirpma_tablosu(guz, c_sec)

    print("\n" + "=" * 78)
    print("HUKUM ICIN")
    print("=" * 78)
    print(
        f"  yaz25  c_duz {yaz['c_duz']:.3f}  egim {yaz['egim']:.3f}  "
        f"izgara {yaz['izgara_c']} -> dMSE {yaz['izgara_dmse']:+.5f}"
    )
    if guz is not None:
        print(
            f"  guz25  c_duz {guz['c_duz']:.3f}  egim {guz['egim']:.3f}  "
            f"izgara {guz['izgara_c']} -> dMSE {guz['izgara_dmse']:+.5f}"
        )
        print("\n  ISARET KAPISI: iki blokta da c > 1 ve dMSE < 0 mi?")
    print("\n  NOT: bu panel dMSE'si SOGUK REJIMIN KENDI icinde.")
    print("  Test etkisi = p_soguk * dMSE_panel = 0.22159 * dMSE")
    print(f"  yaz25 izgara optimumunda test etkisi ~ {0.22159 * yaz['izgara_dmse']:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
