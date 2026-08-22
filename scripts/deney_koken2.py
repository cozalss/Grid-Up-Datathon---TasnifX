"""EK KOKENLER -- eslenik, uc tohum, uretim yapilandirmasiyla.

BIRINCI OLCUM (deney.py --deney koken, tek tohum, 2026-08-22)::

    ANA (3 blok)   GENEL 1,10449   yaz25 1,08137  guz25 1,11481  kis26 1,11730
    EK KOKENLI     GENEL 1,09941   yaz25 1,08426  guz25 1,10957  kis26 1,10441
                        -0,00508         +0,0029       -0,0052       -0,0129

Genelde kazandiriyor ama TEST IKIZI olan yaz25'te kaybettiriyor. Tek
tohumda blok gurultusu ~0,007-0,010, yani yaz25'in +0,0029'u gurultunun
icinde; kis26'nin -0,0129'u degil. Hukum icin eslenik olcum gerekiyor.

NEDEN YAZ25 FARKLI OLABILIR
``kokenleri_ayikla`` hedef blokla kesisen her kokeni atar. yaz25 (Nis-Tem)
dogrulanirken ``bah25`` (May-Agu) ve ``yaz25b`` (Tem-Eki) DUSER; geriye
kalan kokenler kis ve sonbahar agirlikli. Yani yaz25 icin ek kokenler
"yaz olmayan mevsimi daha da pekistir" anlamina geliyor -- ve yaz25 zaten
gorulmemis-mevsim yanliligi tasiyan blok. Uretimde bu kisit YOK: son model
butun kokenleri gorur.

Bu yuzden karar yalnizca yaz25'e bakarak verilemez; ama yaz25 test ikizi
oldugu icin gormezden de gelinemez. Olcum uc tohumla tekrarlanip
eslenik farkin t degeri okunacak.

Fit: 2 aday x 3 blok x 3 tohum = 18 CatBoost ~ 15 dakika (ek kokenli
fitler ~2 kat uzun).

    python scripts/deney_koken2.py
"""

from __future__ import annotations

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
USTYAZIM: dict[str, object] = {"random_strength": 4.0}


def _test_agirlikli(gercek: np.ndarray, soguk: np.ndarray, tahmin: np.ndarray) -> float:
    """Test'in soguk payina (%22,2) agirliklandirilmis RMSLE."""
    s = tm.rmsle(gercek[soguk], tahmin[soguk]) if soguk.any() else 0.0
    w = tm.rmsle(gercek[~soguk], tahmin[~soguk]) if (~soguk).any() else 0.0
    return float(np.sqrt((1 - tm.TEST_SOGUK_PAYI) * w**2 + tm.TEST_SOGUK_PAYI * s**2))


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("EK KOKENLER -- eslenik, 3 tohum, uretim seti")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    ek = d._ek_kokenler_kur(False)
    eksik = [k for k in kolonlar if k not in ek.columns]
    if eksik:
        print(f"  UYARI: ek koken onbellegi BAYAT, {len(eksik)} kolon eksik -- dusuruluyor")
        kolonlar = [k for k in kolonlar if k in ek.columns]
    ortak = [k for k in egitim.columns if k in ek.columns]
    genis = pd.concat([egitim[ortak], ek[ortak]], ignore_index=True)
    for k in tm.KATEGORIK:
        genis[k] = pd.Categorical(genis[k], categories=egitim[k].cat.categories)
    print(f"  {len(kolonlar)} kolon | ana {len(egitim):,} satir -> ek kokenli {len(genis):,}")

    tekil: dict[str, dict[tuple[str, int], float]] = {}
    torbali: dict[str, dict[str, float]] = {}
    for ad, kaynak, ayikla in (("ANA (3 blok)", egitim, False), ("EK KOKENLI", genis, True)):
        t0 = time.time()
        tekil[ad], torbali[ad] = {}, {}
        for b in tm.BLOKLAR:
            dogrulama = kaynak[kaynak["_blok"] == b.ad]
            kalan = tm.kokenleri_ayikla(kaynak, b.ad) if ayikla else kaynak[kaynak["_blok"] != b.ad]
            gercek = dogrulama[tm.HEDEF].to_numpy()
            soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
            log_tahminler = []
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, kolonlar, SICAK_MASKE, tohum)
                log_t = di.egit_tahmin(
                    "cat", maskeli, dogrulama, kolonlar, tohum, **USTYAZIM
                )
                log_tahminler.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = _test_agirlikli(gercek, soguk, tek)
            harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
            torbali[ad][b.ad] = _test_agirlikli(gercek, soguk, harman)
        ort = float(np.mean(list(torbali[ad].values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in torbali[ad].items())
        print(f"  {ad:14} GENEL {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    a, b_ = "ANA (3 blok)", "EK KOKENLI"
    farklar = np.array([tekil[a][k] - tekil[b_][k] for k in tekil[a]])
    ort = float(farklar.mean())
    sh = float(farklar.std(ddof=1) / np.sqrt(len(farklar)))
    print(f"\n  ESLENIK FARK (ANA - EK)  {ort:+.5f}  SH {sh:.5f}  t {ort / sh:+.2f}")
    print(f"  hukum: {'OLCULDU -- ek kokenler kazandiriyor' if ort / sh >= 2 else 'esik alti'}")
    print("\n  BLOK BAZINDA eslenik fark:")
    for b in tm.BLOKLAR:
        f = np.array([tekil[a][(b.ad, t)] - tekil[b_][(b.ad, t)] for t in di.TOHUMLAR])
        print(f"    {b.ad:6} {f.mean():+.5f}  (tohumlar: " + " ".join(f"{x:+.4f}" for x in f) + ")")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
