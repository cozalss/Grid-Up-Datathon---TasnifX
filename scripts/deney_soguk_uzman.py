"""SOGUK UZMANI -- hatanin %59'unu tasiyan rejime odaklanmis deney.

NEDEN AYRI BIR DENEY
--------------------
Test satirlarinin %22,2'si gecmisi hic olmayan trafolar ve bu dilim toplam
KARESEL hatanin %59'unu tasiyor. 2026-08-21 gecesi olculdu: maskeleme
orani arttikca soguk rejim TEKDUZE iyilesiyor --

    maske   0,00    0,15    0,22    0,35    0,50    0,70    1,00
    soguk  1,8215  1,7851  1,7792  1,7852  1,7733  1,7688  1,7595

-- yani en iyi soguk tahminci, hicbir trafo gecmisi GORMEYEN modeldir.

Ama maske=1,0'da 19 ``t_*`` kolonunun tamami her satirda NaN. Yani model
144 kolonluk bir cerceve tasiyor, 19'u sabit-NaN. ``rsm=0,75`` her bolmede
108 kolon orneklediginden, orneklerin ~%13'u hicbir bilgi tasimayan
kolonlara gidiyor. Ayni sey ``soguk_mu`` icin de gecerli: sabit 1.

Bu betik soguk uzmanini kendi hakkiyla kurar ve YALNIZCA SOGUK SATIRLARDA
olcer. Genel skor bilerek raporlanmiyor -- burada aranan sey soguk
rejimdeki en iyi tahminci; yonlendirme onu zaten sicak uzmaniyla
birlestirecek.

ADAYLAR
    1  maske=1,0, 144 kolon            mevcut soguk uzmani (kiyas tabani)
    2  maske=1,0, t_* ATILDI           olu kolonlar cikarilmis
    3  maske=1,0, t_* atildi, derinlik 6   kolon azalinca kapasite artabilir
    4  maske=1,0, t_* atildi, OFSETSIZ     ham log1p(y) hedefi -- cesitlilik
    5  maske=0,85                       1,0'in gercekten uc nokta oldugunu sina

Fit: 3 blok x 3 tohum x 5 aday = 45 CatBoost ~ 26 dakika.

Calistirma::

    python scripts/deney_soguk_uzman.py
"""

from __future__ import annotations

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

#: ``t_*`` kolonlari + ``soguk_mu``: maske=1,0'da hepsi sabit, yani olu.
OLU_ONEK = ("t_",)
OLU_KOLON = ("soguk_mu",)


def canli_kolonlar(kolonlar: list[str]) -> list[str]:
    return [k for k in kolonlar if not k.startswith(OLU_ONEK) and k not in OLU_KOLON]


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("SOGUK UZMANI -- yalnizca soguk satirlarda olculuyor")
    print("=" * 96)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    canli = canli_kolonlar(kolonlar)
    print(f"  egitim {len(egitim):,} satir")
    print(f"  tam kolon {len(kolonlar)} | olu kolon cikarilinca {len(canli)}")
    print(f"  atilanlar: {len(kolonlar) - len(canli)} adet (t_* + soguk_mu)")

    adaylar: tuple[tuple[str, float, list[str], dict[str, object], bool], ...] = (
        ("1 maske=1.0  tam kolon", 1.0, kolonlar, {}, True),
        ("2 maske=1.0  t_* atildi", 1.0, canli, {}, True),
        ("3 maske=1.0  t_* atildi d6", 1.0, canli, {"depth": 6}, True),
        ("4 maske=1.0  t_* atildi OFSETSIZ", 1.0, canli, {}, False),
        ("5 maske=0.85 tam kolon", 0.85, kolonlar, {}, True),
    )

    parcalar = {}
    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
        parcalar[b.ad] = (kalan, dogrulama, gercek, soguk)
        print(f"  {b.ad:6} dogrulama {len(dogrulama):,} satir, soguk {int(soguk.sum()):,}")

    # aday -> blok -> soguk satirlarin log tahmini (torbalanmis)
    saklanan: dict[str, dict[str, np.ndarray]] = {}
    print()
    for ad, maske_orani, alt, ustyazim, ofset in adaylar:
        t0 = time.time()
        skorlar = {}
        saklanan[ad] = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = parcalar[b.ad]
            tahminler = []
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, kolonlar, maske_orani, tohum)
                tahminler.append(
                    di.egit_tahmin("cat", maskeli, dogrulama, alt, tohum, ofset=ofset, **ustyazim)
                )
                del maskeli
            torbali = np.mean(tahminler, axis=0)
            saklanan[ad][b.ad] = torbali
            t = np.clip(np.expm1(torbali), 0.0, None)
            skorlar[b.ad] = tm.rmsle(gercek[soguk], t[soguk])
        genel = float(np.mean(list(skorlar.values())))
        detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        print(f"  {ad:36} SOGUK {genel:.5f}   {detay}   ({time.time() - t0:.0f} sn)")

    print("\n  --- ADAYLARIN HARMANI (soguk satirlar, log uzayinda) ---")
    print("  farkli oznitelik kumesi = ilintisiz hata; tek basina kotu olan")
    print("  bir aday harmanda yine de kazandirabilir.")
    adlar = [a[0] for a in adaylar]
    for i, ilk in enumerate(adlar):
        for ikinci in adlar[i + 1 :]:
            skorlar = {}
            for b in tm.BLOKLAR:
                _, _, gercek, soguk = parcalar[b.ad]
                karisim = 0.5 * (saklanan[ilk][b.ad] + saklanan[ikinci][b.ad])
                t = np.clip(np.expm1(karisim), 0.0, None)
                skorlar[b.ad] = tm.rmsle(gercek[soguk], t[soguk])
            genel = float(np.mean(list(skorlar.values())))
            detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
            print(f"  {ilk[0]}+{ikinci[0]} yari yariya{'':<21} SOGUK {genel:.5f}   {detay}")

    print("\n  --- KIYAS TABANLARI ---")
    for b in tm.BLOKLAR:
        _, dogrulama, gercek, soguk = parcalar[b.ad]
        kaynak = egitim[egitim["_blok"] != b.ad]
        kuresel = float(
            (np.log1p(kaynak[tm.HEDEF].clip(lower=0.0)) - np.log1p(kaynak["guc"])).mean()
        )
        sabit = np.clip(np.expm1(np.log1p(dogrulama["guc"].to_numpy()) + kuresel), 0.0, None)
        print(f"  {b.ad:6} kapasite x sabit  {tm.rmsle(gercek[soguk], sabit[soguk]):.4f}")

    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
