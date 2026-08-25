# ruff: noqa
"""AJAN URETIMI DENEY KAYDI -- 2026-08-25 fan-out (skor-atagi-25agustos).

Bu betik bir alt-ajan tarafindan tek bir olcum icin yazildi; sonucu
docs/41 ve docs/42'ye islendi. Yeniden uretilebilirlik icin duruyor,
uretim yolunun parcasi DEGILDIR ve depo lint standardindan muaftir.
"""

"""GUN EKSENI IDDIASINI DUSMANCA SINA -- son_islem_gunsade.py hukmu.

IDDIA (docs/41 §6c): son_islem beta=0,60 hem trafo hem gun eksenini eziyor,
ama gun ekseni OLS egimi +1,828 (az yayilmis) => buzme ZARARLI.

BU BETIGIN BULDUGU
------------------
1. §6c tablosu KENDI ICINDE TUTARSIZ: std'ler n-AGIRLIKLI, korelasyon ve egim
   AGIRLIKSIZ hesaplanmis. Tutarli n-agirlikli okuma: TRAFO egim +0,453
   (kor +0,102), GUN egim +1,449 (kor +0,633). Agirliksiz okuma: +0,795 /
   +1,828. Kayip satir-basi oldugu icin n-agirlikli olan dogru olcuttur.
   Iki hukum de ayni yone gidiyor, ama sayilar birbirine karistirilmamali.

2. son_islem r'yi TEK bir genel sabite buzer: r' = m + beta*(r-m). Bu, genel
   ortalamadan HER sapmayi -- trafo etkisini, gun etkisini ve artigi -- ayni
   beta ile carpar. Gun ortalamasinin sapmasi

       D_d = C_d (KOMPOZISYON: o gun hangi trafolar var) + b_d (ZAMAN)

   ve gunsade D_d'nin katsayisini ~1,0 yapar. Yani son_islem'in gun ekseninde
   dokundugu nicelik HAM D_d'dir, trafo ARINDIRILMIS b_d degil. Arindirilmis
   egim (kis26: iki yonlu FE ile 0,610 n-agirlikli / 0,767 agirliksiz) DOGRU
   olculmus ama YANLIS niceliktir -- hicbir son islem ona tek basina dokunmaz.

3. KIRILMA NOKTASI cebirle: MSE(c) = var(D)*(c-s)^2 + sabit, s = gercek egim.
   c=0,60 ile c=1,00 esit MSE verir <=> s = (0,60+1,00)/2 = 0,80. Yani gunsade
   ancak GERCEK gun ekseni egimi s > 0,80 ise kazandirir.

4. KOMPOZISYON TUZAGI olculdu ve TESTTE TEHLIKESIZ: test soguk gun ortalamasi
   varyansinin yalnizca %1,0'i (n-agirlikli) kompozisyondan geliyor; satirlarin
   %98,7'si n>=1.834 olan gunlerde ve o gunlerde butun trafolar mevcut. Seyrek
   Nisan gunleri EB agirligiyla (min %24,5) zaten uretim davranisina dusuyor.
   Agirliksiz bakista pay %22,5 -- ama agirliksiz bakis kaybi olcmez.

5. RIG/URETIM YAPISAL UYUMSUZLUGU (en zayif halka): egitim TAM BIR YIL
   (2025-04-01..2026-03-31) ve bloklar orusmez, dolayisiyla HER blogun kendi
   mevsimi kendi egitim parcasinda TAMAMEN yoktur; uretimde ise test mevsimi
   (Nis-Tem) egitimde 2025 kopyasiyla VARDIR. Bu betik ``gun_uzunlugu_saat``
   destek disiligini de olcer: yaz25 %92,8 / kis26 %12,7 / guz25 %0,0 /
   URETIM %0,0.

    python scripts/deney_gun_ekseni_dogrula.py --uret   # yaz25/guz25 onbellegi
    python scripts/deney_gun_ekseni_dogrula.py          # olcumler
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import olcut as ol  # noqa: E402
import tuketim_model as tm  # noqa: E402

DIZIN = KOK / "data" / "interim" / "gun_ekseni"
BETA = 0.60
SOGUK_MASKE = 1.00
SOGUK_CAT: dict[str, object] = {"depth": 7}
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907

#: yaz25 test penceresinin MEVSIMSEL IKIZI oldugu icin alti tohum (kalici kural 3:
#: soguk tarafta uc tohum yetmez). Digerleri uc tohumla kalir -- ikisi de
#: kirpma altinda zaten hukmunu degistirmiyor.
TOHUMLAR: dict[str, tuple[int, ...]] = {
    "yaz25": (1000, 1001, 1002, 1003, 1004, 1005),
    "guz25": (1000, 1001, 1002),
    "kis26": (1000, 1001, 1002),
}


def _yol(blok: str, tohum: int) -> Path:
    """kis26 zaten ``deney_soguk_kolon_temizle.py`` onbelleginde -- AYNI uretim yolu."""
    if blok == "kis26":
        return KOK / "data/interim/soguk_temiz" / f"kis26_{tohum}_taban.npy"
    return DIZIN / f"{blok}_{tohum}_taban.npy"


def uret() -> None:
    """yaz25/guz25 icin URETIM ESLI soguk uzman tahminlerini onbellege yaz."""
    DIZIN.mkdir(parents=True, exist_ok=True)
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    for blok in ("yaz25", "guz25"):
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
        dg = dogrulama[soguk]
        meta = DIZIN / f"{blok}_meta.parquet"
        if not meta.exists():
            pd.DataFrame(
                {
                    "tanim": dg["tanim"].astype(str).to_numpy(),
                    "tarih": pd.to_datetime(dg["tarih"]).astype("datetime64[ns]").to_numpy(),
                    "guc": dg["guc"].to_numpy(dtype="float64"),
                    "y": gercek[soguk].astype("float64"),
                }
            ).to_parquet(meta)
        for t in TOHUMLAR[blok]:
            if _yol(blok, t).exists():
                continue
            t0 = time.time()
            maskeli = d.soguk_maskele(parca, kol, SOGUK_MASKE, t)
            log_t = di.egit_tahmin("cat", maskeli, dogrulama, kol, t, **SOGUK_CAT)
            v = log_t[soguk] if log_t.shape[0] == soguk.size else log_t
            np.save(_yol(blok, t), v.astype("float32"))
            print(f"  {blok} tohum {t} yazildi ({time.time() - t0:.0f} sn)")


def _meta(blok: str) -> pd.DataFrame:
    if blok == "kis26":
        return pd.read_parquet(KOK / "data/interim/kis26_soguk_meta.parquet")
    return pd.read_parquet(DIZIN / f"{blok}_meta.parquet")


def ols(x, y, w=None) -> dict:  # noqa: ANN001
    """``w`` verilmezse gun/trafo basina esit agirlik; verilirse satir sayisi."""
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    w = np.ones_like(x) if w is None else np.asarray(w, dtype="float64")
    w = w / w.sum()
    mx, my = w @ x, w @ y
    vx, vy = w @ (x - mx) ** 2, w @ (y - my) ** 2
    cv = w @ ((x - mx) * (y - my))
    return {"sx": np.sqrt(vx), "sy": np.sqrt(vy), "kor": cv / np.sqrt(vx * vy), "egim": cv / vx}


def gunsade_ofset(rh: np.ndarray, tarih: pd.Series) -> np.ndarray:
    """``son_islem_gunsade.py``nin ofset uzayindaki BIREBIR kopyasi (M etiketsiz)."""
    dd = pd.DataFrame({"g": tarih.to_numpy(), "r": rh})
    sayim = dd.groupby("g")["r"].size()
    gort = dd.groupby("g")["r"].mean()
    genel = float(rh.mean())
    s2a = float(np.average((gort.to_numpy() - genel) ** 2, weights=sayim.to_numpy(dtype="float64")))
    s2i = float(dd.groupby("g")["r"].transform("mean").rsub(dd["r"]).pow(2).mean())
    agir = sayim / (sayim + s2i / s2a)
    hedef = tarih.map(agir * gort + (1.0 - agir) * genel).to_numpy(dtype="float64")
    return hedef + BETA * (rh - hedef)


def egim_tablosu(egitim: pd.DataFrame) -> None:
    print("=" * 100)
    print("HAM GUN EKSENI EGIMI -- son_islem'in FIILEN dokundugu nicelik")
    print("  KIRILMA NOKTASI s=0,80.  TEST penceresi Nis-Tem => mevsimsel ikiz yaz25.")
    print("=" * 100)
    print(
        f"{'blok':>7}{'n':>9}{'gun':>5}{'model sd':>10}{'gercek sd':>11}"
        f"{'kor':>7}{'EGIM(n)':>9}{'EGIM(duz)':>11}{'destek disi':>13}"
    )
    for blok in ("yaz25", "guz25", "kis26"):
        meta = _meta(blok)
        lg = np.log1p(meta["guc"].to_numpy(dtype="float64"))
        rt = np.log1p(np.clip(meta["y"].to_numpy(), 0.0, None)) - lg
        rh = np.mean(
            [np.load(_yol(blok, t)).astype("float64") - lg for t in TOHUMLAR[blok]], axis=0
        )
        tarih = pd.Series(meta["tarih"].to_numpy())
        g = (
            pd.DataFrame({"k": tarih.to_numpy(), "rh": rh, "rt": rt})
            .groupby("k")
            .agg(rh=("rh", "mean"), rt=("rt", "mean"), n=("rh", "size"))
        )
        o = ols(g["rh"], g["rt"], g["n"].to_numpy())
        o2 = ols(g["rh"], g["rt"])
        # kural 2: mevsim surucusunun egitim destegi
        eg = egitim.loc[egitim["_blok"] != blok, "gun_uzunlugu_saat"].to_numpy(dtype="float64")
        dg = egitim.loc[
            (egitim["_blok"] == blok) & (egitim["soguk_mu"] == 1), "gun_uzunlugu_saat"
        ].to_numpy(dtype="float64")
        dis = float(np.nanmean((dg < np.nanmin(eg)) | (dg > np.nanmax(eg))))
        print(
            f"{blok:>7}{len(meta):>9,}{len(g):>5}{o['sx']:>10.4f}{o['sy']:>11.4f}"
            f"{o['kor']:>+7.3f}{o['egim']:>+9.3f}{o2['egim']:>+11.3f}{100 * dis:>12.1f}%"
        )


def hukum_tablosu(egitim: pd.DataFrame, te_c: pd.DataFrame, guc_kenar: np.ndarray) -> None:
    print("\n" + "=" * 100)
    print("GUNSADE HUKMU -- kVA agirlikli, eslenik SH, KIRPILMIS (kalici kural 1/3/4)")
    print("=" * 100)
    for blok in ("yaz25", "guz25", "kis26"):
        meta = _meta(blok)
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, blok)
        w, tani = ol.test_agirliklari(dogrulama[soguk], te_c, guc_kenar, eksenler=("guc",))
        lg = np.log1p(meta["guc"].to_numpy(dtype="float64"))
        rt = np.log1p(np.clip(meta["y"].to_numpy(), 0.0, None)) - lg
        tarih = pd.Series(meta["tarih"].to_numpy())
        tanim = pd.Series(meta["tanim"].to_numpy())
        rp, rg = [], []
        for t in TOHUMLAR[blok]:
            rh = np.load(_yol(blok, t)).astype("float64") - lg
            rp.append(rh.mean() + BETA * (rh - rh.mean()))
            rg.append(gunsade_ofset(rh, tarih))

        def fark(msk: np.ndarray) -> np.ndarray:
            ww, yy = w[msk], rt[msk]
            return np.array(
                [
                    float(
                        np.sqrt(np.dot(ww, (yy - a[msk]) ** 2) / ww.sum())
                        - np.sqrt(np.dot(ww, (yy - b[msk]) ** 2) / ww.sum())
                    )
                    for a, b in zip(rp, rg, strict=True)
                ]
            )

        dm = (
            np.mean([(rt - b) ** 2 - (rt - a) ** 2 for a, b in zip(rp, rg, strict=True)], axis=0)
            * w
        )
        ser = pd.Series(dm).groupby(tanim.to_numpy()).sum().sort_values()
        top = float(dm.sum())
        print(
            f"\n  {blok}  tohum {len(rp)}  ESS %{100 * tani['ess_orani']:.0f}"
            f"  toplam d(MSE)*w {top:+.1f}  EN BUYUK %{100 * ser.iloc[0] / top:.1f}"
            f"  ilk5 %{100 * ser.iloc[:5].sum() / top:.1f}"
        )
        mut = ser.abs().sort_values(ascending=False)
        print(f"    {'K':>4}{'kalan':>7}{'fark':>11}{'SH':>10}{'t':>8}{'tohum':>9}{'genele':>10}")
        for K in (0, 1, 5, 10, 25, 50):
            msk = ~tanim.isin(set(mut.index[:K])).to_numpy()
            f = fark(msk)
            sh = float(f.std(ddof=1) / np.sqrt(len(f)))
            print(
                f"    {K:>4}{ser.size - K:>7}{f.mean():>+11.5f}{sh:>10.5f}"
                f"{f.mean() / sh:>+8.2f}{(f > 0).sum():>6}/{len(f)}"
                f"{-f.mean() * SOGUK_KATSAYI:>10.5f}"
            )
    print("\n  (fark>0 = gunsade IYI)")


def main() -> int:
    ap = argparse.ArgumentParser(description="gun ekseni iddiasini dusmanca sina")
    ap.add_argument("--uret", action="store_true", help="onbellegi tamamla, olcme")
    ar = ap.parse_args()
    if ar.uret:
        uret()
        return 0
    egitim, test = d.cerceveleri_kur()
    te_c = test[test["soguk_mu"] == 1]
    egim_tablosu(egitim)
    hukum_tablosu(egitim, te_c, ol.guc_kenarlari(te_c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
