"""HIC OLCULMEMIS KOLON GRUPLARI -- URETIM ESLI rig, pg10 tabani uzerine.

NEDEN
-----
``pg10`` kazanci (t=+3,59, uc blokta da pozitif, genele -0,00545) eski bir
kararin URETIM ESLI rig'de yeniden olculmesinden cikti. ``YALIN_CIKARILAN``
docstring'i (``tuketim_model.py:972``) ayni sinifta, ustelik **hic
olculmemis** kolonlar oldugunu ACIKCA yaziyor:

    "nufus ailesi (5): onbellekten SONRA hatta girdi. ... t_mevsim_ (2): bu
    gece eklendi, HIC olculmedi. Ikisi de 'kotu oldugu icin' degil, 'olcumun
    disinda oldugu icin' cikariliyor. Onbellek yenilenip yeniden olculunce
    karar gozden gecirilmeli -- ozellikle t_mevsim_, ki fikir umut verici."

Bu, koda birakilmis bir yapilacak notudur ve bugune kadar yapilmadi.

UC KOL
------
    mevsim   2 kolon   t_mevsim_genlik, t_mevsim_gun     HIC OLCULMEDI
    nufus    5 kolon   nufus, alan_km2, yogunluk, ...    HIC OLCULMEDI
    takvim  26 kolon   tk_*, tatil_*, ramazan_*          olculdu ama YANLIS rig'de

``takvim`` daha once ``deney_takvim.py`` ile -0,00428 (zararli) olculdu, ama
o rig de ek kokensizdi ve maske/rs uretimden farkliydi -- ``pg10``i yanlis
gosteren rig'in aynisi. Yeniden sorulmayi hak ediyor.

DIKKAT: ``t_mevsim_*`` ``t_`` onekli, yani GECMIS MASKESINE tabi (%15).
Bu dogru davranis -- onlar gecmisten turetilen kolonlar.

TABAN BEDAVA: ``deney_pg_maske.py``nin ``pg10`` kolu artik URETIM
yapilandirmasidir (105 + 10 = 115 kolon) ve onbellekte duruyor.

    python scripts/deney_kolon_adaylari.py
    python scripts/deney_kolon_adaylari.py --kollar mevsim,nufus
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

DIZIN = KOK / "data" / "interim" / "aile_onbellek"
KAYIT = KOK / "experiments" / "kolon_adaylari.jsonl"
TOHUMLAR = (1000, 1001, 1002)
GBDT_AGIRLIK = (3.0, 1.0, 1.0)
AG_AGIRLIK = 1.4
MASKE = 0.15
URETIM_CAT: dict[str, object] = {
    "random_strength": 4.0,
    "l2_leaf_reg": 1.0,
    "depth": 6,
    "iterations": 250,
}
TABAN_SONEK = "cat_pg10"

#: Kol -> kolon secici. Taban ("pg10") egitilmez, onbellekten okunur.
SECICILER: dict[str, object] = {
    "mevsim": ("t_mevsim_",),
    "nufus": (
        "nufus",
        "alan_km2",
        "ilce_nufus_yogunlugu",
        "trafo_basina_nufus",
        "kva_basina_nufus",
    ),
    "takvim": ("tk_", "tatil", "ramazan"),
}
SICAK_KATSAYI = 0.7784 * 0.74263 / 1.07907


def _aday_kolonlar(cerceve: pd.DataFrame, secici) -> list[str]:  # noqa: ANN001
    if isinstance(secici, tuple) and all(s.endswith("_") for s in secici):
        return sorted(c for c in cerceve.columns if c.startswith(secici))
    return sorted(c for c in cerceve.columns if c.startswith(tuple(secici)))


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--kollar", default=",".join(SECICILER))
    ar = ap.parse_args()
    istenen = [k.strip() for k in ar.kollar.split(",") if k.strip() in SECICILER]

    t0 = time.time()
    print("=" * 100)
    print("HIC OLCULMEMIS KOLON GRUPLARI -- pg10 tabani uzerine, uretim esli")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    taban_kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    ek10 = list(tm.REJIM_AYARLARI["sicak"]["ek_kolon"])  # type: ignore[index]
    kol = taban_kol + ek10
    tm.kategorik_kodla(egitim, test)
    guc_kenar = ol.guc_kenarlari(test)
    te_s = test[test["soguk_mu"] != 1]
    print(f"  TABAN (pg10) {len(kol)} kolon = {len(taban_kol)} yalin + {len(ek10)} ek_kolon")

    ek = d._ek_kokenler_kur(False)
    ek = ek[ek["_blok"].isin([a for a, _, _ in tm.EK_KOKENLER])]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)

    adaylar = {}
    for ad in istenen:
        c = [k for k in _aday_kolonlar(genis, SECICILER[ad]) if k in test.columns and k not in kol]
        adaylar[ad] = c
        print(f"    {ad:8} +{len(c):2d} kolon  {c[:4]}{' ...' if len(c) > 4 else ''}")

    veri = {}
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = di.blok_parcalari(egitim, b.ad)
        sicak = ~soguk
        parca = tm.kokenleri_ayikla(genis, b.ad)
        w, _ = ol.test_agirliklari(dogrulama[~soguk], te_s, guc_kenar, eksenler=("bayatlik",))
        blok = {"gercek": np.load(DIZIN / f"{b.ad}_gercek.npy"), "w": w}
        for t in TOHUMLAR:
            for a in ("xgb", "lgbm", "sinir_agi"):
                blok[(t, a)] = np.load(DIZIN / f"{b.ad}_{t}_{a}_uretim.npy").astype("float64")
            tb = DIZIN / f"{b.ad}_{t}_{TABAN_SONEK}.npy"
            if not tb.exists():
                raise SystemExit(f"taban eksik: {tb}\n  once: python scripts/deney_pg_maske.py")
            blok[(t, "taban")] = np.load(tb).astype("float64")
        print(f"\n  {b.ad}  egitim {len(parca):,}  sicak {int(sicak.sum()):,}")
        for t in TOHUMLAR:
            for ad in istenen:
                yol = DIZIN / f"{b.ad}_{t}_cat_ka_{ad}.npy"
                if yol.exists():
                    blok[(t, ad)] = np.load(yol).astype("float64")
                    continue
                t1 = time.time()
                kk = kol + adaylar[ad]
                maskeli = d.soguk_maskele(parca, kk, MASKE, t)
                log_t = di.egit_tahmin("cat", maskeli, dogrulama, kk, t, **URETIM_CAT)
                v = log_t[sicak] if log_t.shape[0] == soguk.size else log_t
                np.save(yol, v.astype("float32"))
                blok[(t, ad)] = v.astype("float64")
                print(f"    tohum {t}  {ad:8} ({time.time() - t1:.0f} sn)")
        veri[b.ad] = blok

    def skor(bad: str, tohum: int, ad: str) -> float:
        v = veri[bad]
        top = sum(GBDT_AGIRLIK) + AG_AGIRLIK
        pay = (
            GBDT_AGIRLIK[0] * v[(tohum, ad)]
            + GBDT_AGIRLIK[1] * v[(tohum, "xgb")]
            + GBDT_AGIRLIK[2] * v[(tohum, "lgbm")]
            + AG_AGIRLIK * v[(tohum, "sinir_agi")]
        )
        return ol.agirlikli_rmsle(v["gercek"], np.clip(np.expm1(pay / top), 0.0, None), v["w"])

    ciftler = [(b.ad, t) for b in tm.BLOKLAR for t in TOHUMLAR]
    u = {c: skor(*c, "taban") for c in ciftler}
    print("\n" + "-" * 100)
    print("ESLENIK FARK (pg10 tabani - aday; POZITIF = aday IYI)")
    print("-" * 100)
    print(f"  {'aday':>10}{'fark':>10}{'SH':>9}{'t':>7}{'kazanan':>9}{'genel etki':>12}  hukum")
    kayitlar = []
    for ad in istenen:
        s = {c: skor(*c, ad) for c in ciftler}
        f = np.array([u[c] - s[c] for c in ciftler])
        ort, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = ort / sh if sh > 0 else 0.0
        blok_ort = {
            b.ad: float(np.mean([u[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR]))
            for b in tm.BLOKLAR
        }
        uc_pozitif = all(v > 0 for v in blok_ort.values())
        hukum = "AL" if (t_d >= 2 and uc_pozitif) else ("REDDET" if t_d <= -2 else "esik alti")
        print(
            f"  {ad:>10}{ort:+10.5f}{sh:9.5f}{t_d:+7.2f}"
            f"{(f > 0).sum():5d}/{len(f):<3d}{-ort * SICAK_KATSAYI:+12.5f}  {hukum}"
        )
        for b in tm.BLOKLAR:
            bf = np.array([u[(b.ad, t)] - s[(b.ad, t)] for t in TOHUMLAR])
            print(f"       {b.ad:8} {bf.mean():+.5f}  ({(bf > 0).sum()}/{len(bf)} tohum)")
        kayitlar.append(
            {
                "aday": ad,
                "n_kolon": len(adaylar[ad]),
                "fark": ort,
                "sh": sh,
                "t": t_d,
                "blok": blok_ort,
                "hukum": hukum,
            }
        )

    print("\n  KARAR KURALI: t >= 2 VE uc blokta da pozitif ortalama.")
    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        for k in kayitlar:
            fh.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
