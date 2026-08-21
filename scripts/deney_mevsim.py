"""``t_mevsim_*`` OLCUMU -- mevsimsel genlik ise yariyor mu.

FIKIR
-----
Butun seviye olcumlerimiz KISI olcuyor: guncellik pencereleri ozet
penceresinin sonundan geriye bakiyor, o da Mart 2026. Ama tahmin edilen
donem Nisan-Temmuz. Kistan yaza gecerken herkese AYNI carpani uygulamak
yanlis, cunku carpan herkes icin ayni degil.

Olculdu (2026-08-22, 2.270 trafo, yaz ve kista >=30 gun): yaz/kis orani
p5 0,62 / p50 1,18 / p95 4,92 -- yani p95/p05 SEKIZ KAT. Sulama trafosu
yazin katlaniyor, sokak aydinlatmasi DUSUYOR (gun uzuyor), klimali ticari
trafo ikiye katliyor. Mevsimsel genlik, elimizde olmayan ``trafo_tipi``
kolonunun veriden okunabilen en iyi vekili.

NEDEN SICAK REJIMDE OLCULUYOR
``t_mevsim_genlik`` ``t_`` onekiyle basliyor, yani soguk maskeleme onu da
NaN yapiyor. Soguk uzmani maske %100'de calistigi icin kolon orada
SABIT-NaN, yani olu. Etkisi yalnizca SICAK uzmaninda olabilir.

KAPSAM SINIRI -- sonuc okunurken UNUTULMAMALI
    blok    ozet penceresi          yaz ayi (6,7) var mi
    yaz25   2025-01-01..03-31       YOK   <- kolon tamamen bos
    guz25   2025-01-01..07-31       VAR
    kis26   2025-01-01..11-30       VAR
    TEST    2025-01-01..2026-03-31  VAR
yaz25 bizim en iyi test vekilimiz ve orada kolon BOS. Yani beklenen sonuc:
yaz25'te FARK YOK, guz25 ve kis26'da varsa gercek.

Ustelik olculebilen iki blokta iliski TEST'inkinden farkli: guz25'te "yaz
oraniyla sonbahari tahmin et", kis26'da "yaz oraniyla kisi tahmin et";
test'in ihtiyaci ise "yaz oraniyla YAZI tahmin et". Yani bu olcum, ozelligin
test'teki degerinin ALT SINIRI sayilmali.

Fit: 2 aday x 3 blok x 3 tohum = 18 CatBoost ~ 11 dakika.

ONCE ONBELLEK YENILENMELI: kolon 2026-08-22'de eklendi, onbellek dun
17:10'da kuruldu.

    python scripts/deney.py --yenile
    python scripts/deney_mevsim.py
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

MEVSIM_ONEK = "t_mevsim_"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("t_mevsim_* OLCUMU -- sicak uzmani (maske %15), SICAK satirlarda")
    print("=" * 100)
    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    mevsim = [k for k in tum if k.startswith(MEVSIM_ONEK)]
    if not mevsim:
        print("  HATA: t_mevsim_* kolonu onbellekte YOK.")
        print("  Once calistir: python scripts/deney.py --yenile")
        return 1
    tm.kategorik_kodla(egitim, test)

    # Uretim yapilandirmasi: YALIN_CIKARILAN uygulanmis 105 kolon.
    # t_mevsim_ o listede oldugu icin taban ondan YOKSUN -- dogru kiyas bu.
    taban_kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    ile_kol = taban_kol + mevsim
    print(f"  onbellek {len(tum)} kolon | uretim tabani {len(taban_kol)} | +mevsim {len(ile_kol)}")
    print(f"  mevsim kolonlari: {mevsim}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}

    print("\n  --- KOLON DOLULUGU (sicak satirlar) ---")
    for b in tm.BLOKLAR:
        _, dogrulama, _, soguk = parcalar[b.ad]
        sic = dogrulama.loc[~soguk]
        parca = "  ".join(f"{k} %{sic[k].notna().mean() * 100:5.1f}" for k in mevsim)
        print(f"    {b.ad:6} {parca}")
    ts = test[test["soguk_mu"] == 0]
    print("    TEST   " + "  ".join(f"{k} %{ts[k].notna().mean() * 100:5.1f}" for k in mevsim))

    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(parcalar[b.ad][0], ile_kol, 0.15, tohum)

    print("\n  --- MODELLER (CatBoost d5, 3 tohum torbalanmis) ---")
    taban = None
    for ad, alt in (("TABAN (uretim, mevsimsiz)", taban_kol), ("+ t_mevsim_*", ile_kol)):
        t0 = time.time()
        skorlar = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            tahminler = [
                di.egit_tahmin("cat", maskeli[(b.ad, tohum)], dogrulama, alt, tohum)
                for tohum in di.TOHUMLAR
            ]
            t = np.clip(np.expm1(np.mean(tahminler, axis=0)), 0.0, None)
            skorlar[b.ad] = tm.rmsle(gercek[~soguk], t[~soguk])
        genel = float(np.mean(list(skorlar.values())))
        if taban is None:
            taban = genel
            isaret = "TABAN"
        else:
            fark = taban - genel
            isaret = f"{fark:+.5f} " + ("GECTI" if fark > di.ESIK else "esik alti")
        detay = "  ".join(f"{k} {v:.4f}" for k, v in skorlar.items())
        sure = f"({time.time() - t0:.0f} sn)"
        print(f"  {ad:28} [{len(alt):3d}] SICAK {genel:.5f}  {isaret:20} {detay}  {sure}")

    print("\n  BEKLENEN DESEN: yaz25'te fark YOK (kolon orada bos),")
    print("  guz25 ve kis26'da varsa gercek. Aksi cikarsa gurultuye bakiyoruz.")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
