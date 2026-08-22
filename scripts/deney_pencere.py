"""ozet_pencere_gun GIZLI BLOK KIMLIGI mi?  Cikarilmasi kazandiriyor mu?

BULGU (2026-08-22 gece, veri hatti denetimi)
-------------------------------------------
``ozet_pencere_gun`` egitimde YALNIZCA UC deger aliyor ve her biri bir
blogu birebir tanimliyor::

    yaz25 -> 90      guz25 -> 212      kis26 -> 334      TEST -> 455

Testte 455, yani egitim araliginin TAMAMEN disinda (%100 satir). Agaclar
ekstrapolasyon yapamaz: butun test satirlari ">334" dalina, yani KIS
blogunun dalina gider. Oysa test Nisan-Temmuz.

Dahasi kolon, mevsimi dolayli olarak etiketliyor. Blok-CV'de bu, modelin
"hangi mevsimdeyim" bilgisini bedavaya almasi demek -- ve tuttugu blok
dogrulamada disarida oldugu icin yanlis dala gidiyor.

Iki aday, iki rejim, uretim ayarlariyla::

    TABAN      uretim 105 kolon
    -PENCERE   ozet_pencere_gun cikarilmis (104)

Fit: 2 aday x 2 rejim x 3 blok x 3 tohum = 36 CatBoost.

    python scripts/deney_pencere.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

HAFTA = ("tk_haftanin_gunu", "tk_hafta_sonu")
KURULUM = (
    ("SICAK", 0.15, {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}, ()),
    ("SOGUK", 1.00, {"depth": 7}, HAFTA),
)
CIKAR = "ozet_pencere_gun"
KAYIT = KOK / "experiments" / "pencere.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("ozet_pencere_gun -- gizli blok kimligi mi?")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    if CIKAR not in uretim:
        raise RuntimeError(f"{CIKAR} uretim kolonlarinda yok")
    tm.kategorik_kodla(egitim, test)
    print(f"  uretim {len(uretim)} kolon | -{CIKAR} -> {len(uretim) - 1}")
    print(f"  egitimdeki degerler: {sorted(egitim[CIKAR].dropna().unique())}")
    print(f"  testteki degerler:   {sorted(test[CIKAR].dropna().unique())}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    kayitlar = []
    for rejim, maske_orani, ustyazim, ek in KURULUM:
        taban = uretim + [k for k in ek if k not in uretim]
        adaylar = (("TABAN", taban), (f"-{CIKAR}", [k for k in taban if k != CIKAR]))
        maskeli = {
            (b.ad, t): d.soguk_maskele(parcalar[b.ad][0], taban, maske_orani, t)
            for b in tm.BLOKLAR
            for t in di.TOHUMLAR
        }
        tekil: dict[str, dict[tuple[str, int], float]] = {}
        print(f"\n--- {rejim} rejimi (maske {maske_orani:.2f}) ---")
        for ad, kol in adaylar:
            t0 = time.time()
            tekil[ad] = {}
            blok = {}
            for b in tm.BLOKLAR:
                _, dogrulama, gercek, soguk = parcalar[b.ad]
                secim = soguk if rejim == "SOGUK" else ~soguk
                loglar = []
                for tohum in di.TOHUMLAR:
                    log_t = di.egit_tahmin(
                        "cat", maskeli[(b.ad, tohum)], dogrulama, kol, tohum, **ustyazim
                    )
                    loglar.append(log_t)
                    tek = np.clip(np.expm1(log_t), 0.0, None)
                    tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[secim], tek[secim])
                harman = np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)
                blok[b.ad] = tm.rmsle(gercek[secim], harman[secim])
            ort = float(np.mean(list(blok.values())))
            detay = "  ".join(f"{k} {v:.5f}" for k, v in blok.items())
            print(f"  {ad:20} {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

        f = np.array([tekil["TABAN"][k] - tekil[f"-{CIKAR}"][k] for k in tekil["TABAN"]])
        o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = o / sh if sh > 0 else 0.0
        hukum = "CIKAR" if t_d >= 2 else ("TUT" if t_d <= -2 else "esik alti")
        print(f"  ESLENIK FARK {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
        for b in tm.BLOKLAR:
            bb = np.array(
                [tekil["TABAN"][(b.ad, t)] - tekil[f"-{CIKAR}"][(b.ad, t)] for t in di.TOHUMLAR]
            )
            print(f"     {b.ad:6} {bb.mean():+.5f}  ({(bb > 0).sum()}/{len(bb)} tohum pozitif)")
        kayitlar.append({"rejim": rejim, "fark": o, "sh": sh, "t": t_d, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
