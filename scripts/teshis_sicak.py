"""SICAK HATANIN HARITASI -- optimize etmeden ONCE teshis.

NEDEN
-----
Soguk tarafta hatanin haritasi cikarilmisti ve bir sey buldu: soguk
satirlarin %5,16'si tam SIFIR ve KARESEL hatanin %52,3'unu tasiyor. Bu,
"sifir yigilmasi" adiyla kayda gecti ve sonraki butun soguk kararlarini
yonlendirdi.

Sicak tarafta boyle bir harita HIC cikarilmadi. Satirlarin %78'i orada ve
uc turdur parametre taramasi plato veriyor (en iyi aday +0,002, t<2).
Parametre aramaya devam etmek yerine hatanin NEREDE oldugunu sormak
gerekiyor: yogun bir cep varsa hedefli bir mudahale, korlemesine aramadan
cok daha fazla kazandirir.

TASARIM
-------
Tek blok (yaz25 -- test doneminin mevsimsel ikizi), tek tohum, uretim
sicak uzmani. Teshis icin torbalama gereksiz: aranan sey skorun son
hanesi degil, hatanin dagilimi.

Her kirilimda uc sayi raporlanir::

    satir payi        o kovada satirlarin yuzdesi
    HATA PAYI         toplam karesel log hatanin yuzdesi   <- asil sayi
    yogunluk          hata payi / satir payi               <- 1'den buyukse cep

Ayrica her kovada ORTALAMA YANLILIK (log uzayinda tahmin eksi gercek)
veriliyor: yanlilik sistematikse duzeltilebilir, sacilimsa duzeltilemez.

    python scripts/teshis_sicak.py
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

BLOK = "yaz25"
TOHUM = 1000
SICAK_MASKE = 0.15
USTYAZIM: dict[str, object] = {"random_strength": 4.0}


def _kir(ad: str, kova: pd.Series, kare: np.ndarray, yanlilik: np.ndarray) -> None:
    """Bir kirilimda satir payi, hata payi, yogunluk ve yanliligi yazar."""
    toplam = kare.sum()
    print(f"\n  --- {ad} ---")
    print(f"  {'kova':>22} {'satir':>9} {'satir%':>7} {'HATA%':>7} {'yogunluk':>9} {'yanlilik':>9}")
    for k, idx in kova.groupby(kova, observed=True).groups.items():
        yer = kova.index.get_indexer(idx)
        s_pay = len(yer) / len(kare) * 100
        h_pay = kare[yer].sum() / toplam * 100
        print(
            f"  {str(k):>22} {len(yer):>9,} {s_pay:>7.1f} {h_pay:>7.1f} "
            f"{h_pay / s_pay if s_pay else 0:>9.2f} {yanlilik[yer].mean():>+9.3f}"
        )


def main() -> int:
    satir_tamponlu_cikti()
    t0 = time.time()
    print("=" * 96)
    print(f"SICAK HATANIN HARITASI -- blok {BLOK}, tek tohum, uretim sicak uzmani")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, BLOK)
    maskeli = d.soguk_maskele(kalan, kolonlar, SICAK_MASKE, TOHUM)
    log_t = di.egit_tahmin("cat", maskeli, dogrulama, kolonlar, TOHUM, **USTYAZIM)

    sic = ~soguk
    tahmin = np.clip(np.expm1(log_t), 0.0, None)[sic]
    y = gercek[sic]
    ln_t, ln_y = np.log1p(tahmin), np.log1p(y)
    kare = (ln_t - ln_y) ** 2
    yanlilik = ln_t - ln_y  # + = FAZLA tahmin
    print(f"\n  sicak satir {sic.sum():,} | RMSLE {np.sqrt(kare.mean()):.5f}")
    print(f"  ortalama yanlilik {yanlilik.mean():+.4f} (+ = fazla tahmin)")

    dv = dogrulama.loc[sic].reset_index(drop=True)
    dv["_y"] = y

    def kovala(seri: pd.Series, kenar: list[float], etiket: list[str]) -> pd.Series:
        return pd.cut(seri, bins=kenar, labels=etiket, include_lowest=True)

    _kir(
        "GERCEK TUKETIM",
        kovala(dv["_y"], [-1, 0, 1, 10, 100, 1000, 1e12],
               ["tam sifir", "0-1", "1-10", "10-100", "100-1k", "1k+"]),
        kare, yanlilik,
    )
    _kir(
        "TRAFONUN SIFIR ORANI (gecmis)",
        kovala(dv["t_sifir_orani"], [-0.01, 0.001, 0.05, 0.2, 0.5, 1.0],
               ["0", "0-5%", "5-20%", "20-50%", "50%+"]),
        kare, yanlilik,
    )
    _kir(
        "TRAFONUN OYNAKLIGI t_log_std",
        kovala(dv["t_log_std"], [-0.01, 0.3, 0.6, 1.0, 1.5, 99],
               ["<0,3", "0,3-0,6", "0,6-1,0", "1,0-1,5", "1,5+"]),
        kare, yanlilik,
    )
    _kir(
        "GECMIS UZUNLUGU t_gun_sayisi",
        kovala(dv["t_gun_sayisi"], [-1, 7, 30, 60, 90, 9999],
               ["<=7 gun", "8-30", "31-60", "61-90", "90+"]),
        kare, yanlilik,
    )
    _kir(
        "UFUK (tahmin mesafesi)",
        kovala(dv["ufuk_gun"], [0, 30, 60, 90, 200], ["1-30", "31-60", "61-90", "90+"]),
        kare, yanlilik,
    )
    _kir(
        "KAPASITE guc",
        kovala(dv["guc"], [-1, 50, 160, 400, 1000, 1e9],
               ["<=50", "51-160", "161-400", "401-1000", "1000+"]),
        kare, yanlilik,
    )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
