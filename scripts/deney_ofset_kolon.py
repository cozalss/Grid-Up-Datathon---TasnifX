"""SICAK UZMAN: seviye kolonlarini OFSET uzayinda da ver.

NEDEN
-----
Hedef ofsetli (``log1p(y) - log1p(guc)``) ama butun seviye kolonlari HAM.
Ofsetin hedefe uygulanmasi -0,0352 kazandirmisti ve gerekcesi acikti:
"agaclarin artik olcegi merdivenlerle yaklastirmasi gerekmesin"
(``tuketim_model.py`` ofsetli_hedef). Ayni argüman OZNITELIKLERE hic
uygulanmadi.

Olculdu (kis26 sicak, n=366.552, dogrusal artik std):

    t_log_son30 tek                     0,9351
    (t_log_son30 - log1p(guc)) tek      0,6698   <- tek eksen
    t_log_son30 + log1p(guc) (2 kolon)  0,6660   <- iki kolonlu optimum

Fark oznitelagi iki kolonlu optimumun 0,004 uzaginda: bilginin tamamini TEK
eksende tasiyor. Agacin bunu iki eksenden kurma bedeli (blok-disi, esit
hucre butcesi):

     64 hucre:  IZGARA(s30,lg) 1,00266  vs  TEK EKSEN 0,76915   +0,23351
    256 hucre:  IZGARA(s30,lg) 0,83952  vs  TEK EKSEN 0,76371   +0,07581

CatBoost sicak uzmani derinlik 6 OBLIVIOUS: agac basina 64 yaprak ve her
seviyede TEK bolme. Bu bedel her agacta odeniyor.

ONEMLI: yeni kolonlar ``t_`` onekiyle baslamak ZORUNDA -- ``soguk_maskele``
gecmis kolonlarini bu onekten tanir. Onek olmazsa kolonlar maskelemeden sag
cikar ve soguk satirlara trafo gecmisi sizar (docs/35'teki ezber kanalinin
aynisi).

    python scripts/deney_ofset_kolon.py
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

#: Ofset uzayina cevrilecek seviye kolonlari. Yeni ad ``t_dofs_*``.
KAYNAK = ("t_log_son7", "t_log_son14", "t_log_son30", "t_log_son60", "t_log_ort", "t_log_medyan")
SICAK_USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
SICAK_MASKE = 0.15
KAYIT = KOK / "experiments" / "ofset_kolon.jsonl"


def ofset_kolonlari_ekle(cerceve: pd.DataFrame) -> list[str]:
    """``t_dofs_*`` kolonlarini yerinde ekler, adlarini dondurur."""
    log_guc = np.log1p(cerceve["guc"].to_numpy(dtype="float64"))
    yeni = []
    for k in KAYNAK:
        ad = k.replace("t_log_", "t_dofs_")
        cerceve[ad] = cerceve[k].to_numpy(dtype="float64") - log_guc
        yeni.append(ad)
    return yeni


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("SICAK UZMAN: seviye kolonlari ofset uzayinda da  --  uc blok, 3 tohum")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    eksik = [k for k in KAYNAK if k not in egitim.columns or k not in test.columns]
    if eksik:
        raise RuntimeError(f"kaynak kolonlar yok: {eksik}")
    yeni_kol = ofset_kolonlari_ekle(egitim)
    ofset_kolonlari_ekle(test)
    onek = tm.GECMIS_ONEKI if hasattr(tm, "GECMIS_ONEKI") else "t_"
    if not all(k.startswith(onek) for k in yeni_kol):
        raise RuntimeError("yeni kolonlar maskelenmeyecek onekte")

    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN) and k not in yeni_kol]
    tm.kategorik_kodla(egitim, test)
    print(f"  uretim {len(uretim)} kolon  |  +{len(yeni_kol)} -> {len(uretim) + len(yeni_kol)}")
    print(f"  yeni: {yeni_kol}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    adaylar = (("TABAN", uretim), ("+ofset_kolon", uretim + yeni_kol))
    maskeli = {
        (b.ad, t): d.soguk_maskele(parcalar[b.ad][0], uretim + yeni_kol, SICAK_MASKE, t)
        for b in tm.BLOKLAR
        for t in di.TOHUMLAR
    }
    # maskeleme dogrulamasi: yeni kolonlar da NaN olmali
    ornek = maskeli[(tm.BLOKLAR[0].ad, di.TOHUMLAR[0])]
    m = ornek["soguk_mu"] == 1
    dolu = [k for k in yeni_kol if ornek.loc[m, k].notna().any()]
    if dolu:
        raise RuntimeError(f"maskelenmemis yeni kolon: {dolu}")
    print("  maskeleme dogrulandi: yeni kolonlar soguk satirlarda NaN")

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for ad, kol in adaylar:
        t0 = time.time()
        tekil[ad] = {}
        blok = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            sicak = ~soguk
            loglar = []
            for tohum in di.TOHUMLAR:
                log_t = di.egit_tahmin(
                    "cat", maskeli[(b.ad, tohum)], dogrulama, kol, tohum, **SICAK_USTYAZIM
                )
                loglar.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[sicak], tek[sicak])
            harman = np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)
            blok[b.ad] = tm.rmsle(gercek[sicak], harman[sicak])
        ort = float(np.mean(list(blok.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok.items())
        print(f"  {ad:16} {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    f = np.array([tekil["TABAN"][k] - tekil["+ofset_kolon"][k] for k in tekil["TABAN"]])
    o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
    t_d = o / sh if sh > 0 else 0.0
    hukum = "EKLE" if t_d >= 2 else ("EKLEME" if t_d <= -2 else "esik alti")
    print(f"\n  ESLENIK FARK {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}   {hukum}")
    for b in tm.BLOKLAR:
        bb = np.array(
            [tekil["TABAN"][(b.ad, t)] - tekil["+ofset_kolon"][(b.ad, t)] for t in di.TOHUMLAR]
        )
        print(f"     {b.ad:6} {bb.mean():+.5f}  ({(bb > 0).sum()}/{len(bb)} tohum kazanc)")
    print(f"  genel skora tahmini etki {-o * 0.528:+.5f}   (d(genel)/d(sicak) = 0,528)")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        kayit = {"fark": o, "sh": sh, "t": t_d, "hukum": hukum}
        fh.write(json.dumps(kayit, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
