"""SOGUK UZMANDAN KAYMIS KOLONLARI AT -- ``yas`` ve ``ozet_pencere_gun``.

BULGU (2026-08-25 denetimi)
---------------------------
Soguk uzman ``maske=1,00`` ile YAPAY sogutulmus satirlarda egitilir. Maske
gecmisi siler ama ``yas``i silmez: 400 gunluk bir trafo maskelenince
"gecmisi olmayan 400 gunluk trafo" olur. GERCEK soguk satir boyle bir sey
degildir -- gercek soguk trafo YENIDIR.

Olculdu:

    egitim (maskeli) satirlarin  %69,9'u   yas > 121
    DOGAL soguk yas maksimumu:   yaz25 115 | guz25 121 | kis26 120 | TEST 121
    dort kumede de yas > 121 payi TAM SIFIR

Yani egitim kutlesinin ucte ikisi, hicbir gercek soguk satirin
BULUNAMAYACAGI bir bolgede. Ustune ``ozet_pencere_gun`` soguk testte %100
DESTEK DISI (blok pencereleri 90/212/334 gun, uretim 455).

Ikisi de soguk uzman icin NUISANCE degiskenidir: tahmin edilecek seyle degil,
SATIRIN HANGI RIGDEN GELDIGIYLE ilgililer.

NEDEN OLCULEBILIR (ek_kolon denemelerinden farki)
-------------------------------------------------
Dogrulamanin soguk satirlari da DOGALdir (maskeli degil) ve onlarda da
yas <= 121. Yani rig, uretimdeki kaymanin AYNISINI uretiyor. Bu gece curuyen
dort adayin hicbirinde bu yoktu -- onlar dogrulamanin OLCEMEDIGI seylerdi.

KALICI KURAL 1 uygulanir: kazanc trafo bazinda ayristirilir ve hukum
KIRPILMIS tabloya bakilarak verilir.

    python scripts/deney_soguk_kayma_kolon.py
"""

from __future__ import annotations

import argparse
import json
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

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

DIZIN = KOK / "data" / "interim" / "soguk_kayma"
KAYIT = KOK / "experiments" / "soguk_kayma_kolon.jsonl"
BLOK = "kis26"
TOHUMLAR = (1000, 1001, 1002)
EK_TOHUMLAR = (1003, 1004, 1005)
SOGUK_MASKE = 1.00
SOGUK_CAT: dict[str, object] = {"depth": 7}
BETA = 0.60
SOGUK_KATSAYI = 0.2216 * 1.82133 / 1.07907
ADAYLAR: dict[str, tuple[str, ...]] = {
    "taban": (),
    "-yas": ("yas",),
    "-pencere": ("ozet_pencere_gun",),
    "-ikisi": ("yas", "ozet_pencere_gun"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--genis", action="store_true", help="6 tohum, taban ve -ikisi")
    ar = ap.parse_args()
    # 6 tohumlu kip AYRI bir kosudur: uc tohumluk hukum yaniltti (bkz. dosya
    # basligi), o yuzden genisletilmis kip yalnizca ayakta kalan adayi tasir.
    tohumlar = TOHUMLAR + EK_TOHUMLAR if ar.genis else TOHUMLAR
    adaylar = {"taban": (), "-ikisi": ("yas", "ozet_pencere_gun")} if ar.genis else ADAYLAR
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 104)
    print(f"SOGUK UZMAN -- KAYMIS KOLON ABLASYONU ({BLOK}, son islem sonrasi, kVA duzeltilmis)")
    print("=" * 104)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    dg = dogrulama[soguk]
    y = gercek[soguk]
    te_c = test[test["soguk_mu"] == 1]
    w, tani = ol.test_agirliklari(dg, te_c, ol.guc_kenarlari(te_c), eksenler=("guc",))
    log_guc = np.log1p(dg["guc"].to_numpy(dtype="float64"))
    tanim = dg["tanim"].astype(str).to_numpy()
    g = np.log1p(np.clip(y, 0.0, None))

    egt_ust = 100.0 * (parca["yas"] > 121).mean()
    dg_ust = 100.0 * (dg["yas"] > 121).mean()
    te_ust = 100.0 * (te_c["yas"] > 121).mean()
    print(f"  egitim {len(parca):,} satir   yas>121 payi %{egt_ust:.1f}")
    dg_maks, te_maks = int(dg["yas"].max()), int(te_c["yas"].max())
    print(f"  dogrulama soguk {len(y):,} satir  yas maks {dg_maks}  >121 %{dg_ust:.1f}")
    print(f"  TEST soguk {len(te_c):,} satir  yas maks {te_maks}  >121 %{te_ust:.1f}")
    print(f"  ozet_pencere_gun egitim {sorted(parca['ozet_pencere_gun'].unique())[:6]}")
    print(f"  ozet_pencere_gun test   {sorted(te_c['ozet_pencere_gun'].unique())[:3]}")
    print(f"  kVA agirlik ESS %{100 * tani['ess_orani']:.1f}")

    def buz(log_t: np.ndarray) -> np.ndarray:
        r = log_t - log_guc
        return np.clip(np.expm1(r.mean() + BETA * (r - r.mean()) + log_guc), 0.0, None)

    DIZIN.mkdir(parents=True, exist_ok=True)
    tahmin: dict[tuple[str, int], np.ndarray] = {}
    for ad, cikar in adaylar.items():
        kk = [k for k in kol if k not in cikar]
        for t in tohumlar:
            yol = DIZIN / f"{BLOK}_{t}_{ad}.npy"
            if yol.exists():
                tahmin[(ad, t)] = np.load(yol).astype("float64")
                continue
            t1 = time.time()
            maskeli = d.soguk_maskele(parca, kk, SOGUK_MASKE, t)
            log_t = di.egit_tahmin("cat", maskeli, dogrulama, kk, t, **SOGUK_CAT)
            v = log_t[soguk] if log_t.shape[0] == soguk.size else log_t
            np.save(yol, v.astype("float32"))
            tahmin[(ad, t)] = v.astype("float64")
            print(f"    {ad:9} tohum {t}  ({len(kk)} kolon, {time.time() - t1:.0f} sn)")

    def puan(ad: str, t: int, alt: np.ndarray | None = None) -> float:
        p = buz(tahmin[(ad, t)])
        if alt is None:
            return ol.agirlikli_rmsle(y, p, w)
        return ol.agirlikli_rmsle(y[alt], p[alt], w[alt])

    def torba(ad: str) -> float:
        ort = np.mean([tahmin[(ad, t)] for t in tohumlar], axis=0)
        return ol.agirlikli_rmsle(y, buz(ort), w)

    print("\n" + "-" * 104)
    print("HUKUM (taban - aday; POZITIF = aday IYI)")
    print("-" * 104)
    print(f"  {'taban':10} torbalanmis {torba('taban'):.5f}")
    sonuc: dict[str, dict[str, float | str]] = {}
    for ad in list(adaylar)[1:]:
        f = np.array([puan("taban", t) - puan(ad, t) for t in tohumlar])
        sh = float(f.std(ddof=1) / np.sqrt(len(tohumlar)))
        td = float(f.mean() / sh) if sh > 0 else 0.0
        hukum = "AL" if td >= 2 else ("REDDET" if td <= -2 else "esik alti")
        print(
            f"  {ad:10} torbalanmis {torba(ad):.5f}   fark {f.mean():+.5f}  SH {sh:.5f}"
            f"  t {td:+6.2f}  {(f > 0).sum()}/{len(tohumlar)}"
            f"  genele {-f.mean() * SOGUK_KATSAYI:+.5f}  {hukum}"
        )
        sonuc[ad] = {"fark": float(f.mean()), "sh": sh, "t": td, "hukum": hukum}

    print("\n" + "-" * 104)
    print("KIRPILMIS HUKUM -- en buyuk K trafo atildi (kalici kural 1)")
    print("-" * 104)
    du = sum((g - tahmin[("taban", t)]) ** 2 for t in tohumlar) / len(tohumlar)
    for ad in list(adaylar)[1:]:
        ds = sum((g - tahmin[(ad, t)]) ** 2 for t in tohumlar) / len(tohumlar)
        seri = pd.Series((du - ds) * w).groupby(tanim).sum().sort_values(ascending=False)
        top = float(seri.sum())
        pay = 100.0 * seri.iloc[0] / top if top else float("nan")
        ilk5 = 100.0 * float(seri.iloc[:5].sum()) / top if top else float("nan")
        poz = 100.0 * float((seri > 0).mean())
        print(f"  {ad}:  EN BUYUK %{pay:.1f}  ilk5 %{ilk5:.1f}  pozitif trafo %{poz:.1f}")
        print(f"    {'K':>4} {'fark':>10} {'SH':>9} {'t':>7} {'tohum':>6} {'genele':>10}")
        for K in (0, 1, 5, 10, 25, 50):
            at = list(seri.head(K).index) if K else []
            alt = ~np.isin(tanim, at) if K else np.ones(len(tanim), dtype=bool)
            f = np.array([puan("taban", t, alt) - puan(ad, t, alt) for t in tohumlar])
            sh = float(f.std(ddof=1) / np.sqrt(len(tohumlar)))
            td = f.mean() / sh if sh > 0 else 0.0
            print(
                f"    {K:>4} {f.mean():>+10.5f} {sh:>9.5f} {td:>+7.2f}"
                f" {(f > 0).sum()}/{len(tohumlar)}   {-f.mean() * SOGUK_KATSAYI:>+9.5f}"
            )

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(sonuc, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
