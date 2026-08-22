"""DAHA COK KOKEN ise yariyor mu -- ve ozet merdiveni test'e uzanmali mi.

DURUM
-----
``v18`` LB'de 1,03370 aldi, birinci 1,03170 -- fark 0,0020. Kalibrasyon
iki bagimsiz noktada 0,0006 sapmayla tuttu::

    v15  yaz25 CV 0,99715 -> LB 1,03910   +0,04195
    v18  yaz25 CV 0,99115 -> LB 1,03370   +0,04255

Yani hedef bulanik degil: **yaz25 CV <= 0,98940**. Su an 0,99115, gereken
-0,00175.

NEDEN DAHA COK KOKEN
--------------------
Ek kokenler v18'de sicak uzmanina +0,00946 kazandirdi (soguga -0,03273,
o yuzden yalnizca sicaga veriliyor). Alti koken vardi. Recruit
yarismasinin birincisi ayni yapiyi 63 kokenle kurmustu.

Ama korlemesine cogaltmak yerine OLCULMUS bir bosluga nisan aliniyor::

    eski merdiven   31  90 120 181 212 243 304 334 365   | TEST 455
    yeni merdiven   31  59  90 120 151 181 212 243 273
                   304 334 365 396 424                   | TEST 455

Test'in ozet penceresi 455 gun ve ``ozet_pencere_gun`` modele ACIK bir
kolon. Eski merdivenin ust ucu 365'ti, yani model test'te hic gormedigi
bir degere DISDEGERLEME yapiyordu. Yeni ``sub26`` (396) ve ``mar26`` (424)
kisa etiketli/uzun ozetli: amaclari satir sayisi degil, merdivenin ust
ucunu test'e yaklastirmak.

TASARIM
-------
Uc kol, hepsi SICAK uzmani (maske 0,15, rs=4, l2=1, d6), sicak satirlarda::

    ANA        ek koken YOK                 -- capa; onceki olcumde 0,80675
    V18 (6)    v18'in gonderdigi kokenler   -- taban
    TUM (11)   bes yeni koken eklenmis      -- aday

ANA kolu capa olarak duruyor: onbellek yeniden kuruldugu icin, onceki
olcumle ayni sayiyi vermesi kurulumun bozulmadiginin kanitidir.

Dogrulamada ``kokenleri_ayikla`` -- ortusme sizintidir.

Fit: 3 kol x 3 blok x 3 tohum = 27 CatBoost ~ 30 dakika (+ onbellek kurulumu).

    python scripts/deney_koken_yogun.py
"""

from __future__ import annotations

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
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SICAK_MASKE = 0.15
USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}

KAYIT = KOK / "experiments" / "koken_yogun.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("DAHA COK KOKEN -- sicak uzmani, ozet merdiveni test'e uzatiliyor")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum_kol = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum_kol if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    print("\n  ek koken onbellegi YENIDEN kuruluyor (11 koken)...")
    t0 = time.time()
    ek = d._ek_kokenler_kur(True)
    print(f"  hazir: {len(ek):,} satir ({time.time() - t0:.0f} sn)")
    eksik = [k for k in kolonlar if k not in ek.columns]
    if eksik:
        raise RuntimeError(f"ek koken cercevesinde {len(eksik)} kolon eksik: {eksik[:5]}")

    ortak = [k for k in egitim.columns if k in ek.columns]
    ek6 = ek[ek["_blok"].isin(tm.KOKENLER_V18)]
    kollar: tuple[tuple[str, pd.DataFrame, bool], ...] = (
        ("ANA (koken yok)", egitim, False),
        (f"V18 ({len(tm.KOKENLER_V18)} koken)", None, True),  # type: ignore[arg-type]
        (f"TUM ({len(tm.EK_KOKENLER)} koken)", None, True),  # type: ignore[arg-type]
    )
    cerceveler = {}
    for ad, parca in ((kollar[1][0], ek6), (kollar[2][0], ek)):
        genis = pd.concat([egitim[ortak], parca[ortak]], ignore_index=True)
        for k in tm.KATEGORIK:
            genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
        cerceveler[ad] = genis
        print(f"  {ad:16} {len(genis):,} satir")

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for ad, sabit, ayikla in kollar:
        t0 = time.time()
        kaynak = sabit if sabit is not None else cerceveler[ad]
        tekil[ad] = {}
        blok_skor = {}
        for b in tm.BLOKLAR:
            dogrulama = egitim[egitim["_blok"] == b.ad]
            kalan = tm.kokenleri_ayikla(kaynak, b.ad) if ayikla else kaynak[kaynak["_blok"] != b.ad]
            gercek = dogrulama[tm.HEDEF].to_numpy()
            sic = (dogrulama["soguk_mu"] == 0).to_numpy()
            log_tahminler = []
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, kolonlar, SICAK_MASKE, tohum)
                log_t = di.egit_tahmin("cat", maskeli, dogrulama, kolonlar, tohum, **USTYAZIM)
                log_tahminler.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[sic], tek[sic])
            harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
            blok_skor[b.ad] = tm.rmsle(gercek[sic], harman[sic])
        ort = float(np.mean(list(blok_skor.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok_skor.items())
        print(f"\n  {ad:16} SICAK {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    taban_ad = kollar[1][0]
    kayitlar = []
    for ad, _, _ in (kollar[0], kollar[2]):
        farklar = np.array([tekil[taban_ad][k] - tekil[ad][k] for k in tekil[taban_ad]])
        o = float(farklar.mean())
        sh = float(farklar.std(ddof=1) / np.sqrt(len(farklar)))
        t_deger = o / sh if sh > 0 else 0.0
        print(f"\n  {ad} vs {taban_ad}:  {o:+.5f}  SH {sh:.5f}  t {t_deger:+.2f}")
        for b in tm.BLOKLAR:
            f = np.array([tekil[taban_ad][(b.ad, t)] - tekil[ad][(b.ad, t)] for t in di.TOHUMLAR])
            print(f"      {b.ad:6} {f.mean():+.5f}")
        kayitlar.append({"kol": ad, "taban": taban_ad, "fark": o, "sh": sh, "t": t_deger})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")

    print("\n  ANA kolu CAPA: onceki olcumde 0,80675 idi. Sapma varsa onbellek")
    print("  yeniden kurulurken bir sey degismis demektir -- once onu arastir.")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
