"""SOGUK UZMANINA HAFTA GUNU -- sinirdaki adayi bes tohumla karara baglar.

DURUM
-----
``deney_takvim.py`` (3 tohum, 2026-08-22) takvimin REJIME GORE ayrildigini
buldu::

    SICAK   +HAFTA           -0,00270  t=-0,83   zararli
    SOGUK   +HAFTA           +0,00292  t=+1,96   3/3 blok POZITIF
    SOGUK   +HAFTA+TAKVIM    -0,02323  t=-1,78   zararli (kis26 -0,074)

Hafta gunu yalnizca soguk uzmanina yariyor -- mantikli, cunku maske 1,00'da
onun elinde trafoyu ayirt eden baska hicbir sey yok; hafta gunu nadir
bulunan gercek bir sinyal.

Ama t=1,96 esigin (2,0) hemen ALTINDA. Uc blokta da pozitif olmasi guclu
bir isaret, yine de bugun dort kez "cok mantikli" gorunen fikir olcumde
coktu. Iki tohum daha ekleyip karari netlestiriyoruz: etki gercekse t
yaklasik sqrt(5/3) = 1,29 kat buyur, yani ~2,5 cikar.

Yalnizca iki aday, yalnizca soguk rejim, bes tohum::

    TABAN    uretim 105 kolon, maske 1,00, d7
    +HAFTA   tk_haftanin_gunu, tk_hafta_sonu

Fit: 2 aday x 3 blok x 5 tohum = 30 CatBoost ~ 25 dakika.

    python scripts/deney_soguk_hafta.py
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
TOHUMLAR = (1000, 1001, 1002, 1003, 1004)
USTYAZIM: dict[str, object] = {"depth": 7}
SOGUK_MASKE = 1.00

KAYIT = KOK / "experiments" / "soguk_hafta.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("SOGUK UZMANINA HAFTA GUNU -- 5 tohum, soguk satirlarda")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    print(f"  taban {len(taban)} kolon | +{len(HAFTA)} = {len(taban) + len(HAFTA)}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], taban + list(HAFTA), SOGUK_MASKE, tohum
            )

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for ad, kol in (("TABAN", taban), ("+HAFTA", taban + list(HAFTA))):
        t0 = time.time()
        tekil[ad] = {}
        blok_skor = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            log_tahminler = []
            for tohum in TOHUMLAR:
                log_t = di.egit_tahmin(
                    "cat", maskeli[(b.ad, tohum)], dogrulama, kol, tohum, **USTYAZIM
                )
                log_tahminler.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[soguk], tek[soguk])
            harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
            blok_skor[b.ad] = tm.rmsle(gercek[soguk], harman[soguk])
        ort = float(np.mean(list(blok_skor.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok_skor.items())
        print(f"  {ad:10} SOGUK {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    f = np.array([tekil["TABAN"][k] - tekil["+HAFTA"][k] for k in tekil["TABAN"]])
    o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
    t_d = o / sh if sh > 0 else 0.0
    hukum = "KAZANDIRIYOR -- AL" if t_d >= 2 else ("ZARARLI" if t_d <= -2 else "esik alti -- ALMA")
    print(f"\n  ESLENIK FARK  {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}  ({len(f)} hucre)   {hukum}")
    for b in tm.BLOKLAR:
        bb = np.array([tekil["TABAN"][(b.ad, t)] - tekil["+HAFTA"][(b.ad, t)] for t in TOHUMLAR])
        artis = (bb > 0).sum()
        print(f"    {b.ad:6} {bb.mean():+.5f}   ({artis}/{len(TOHUMLAR)} tohumda pozitif)")
    print(f"\n  genel skora etkisi: {o * 0.350:+.5f}  (d(genel)/d(soguk) = 0,350)")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"fark": o, "sh": sh, "t": t_d, "hukum": hukum,
                             "tohum": len(TOHUMLAR)}, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
