"""GERCEKLESMIS HAVA vs IKLIM ORTALAMASI -- "veri sizintisi" itirazinin olcumu.

NEDEN BU DENEY VAR
------------------
Yarisma forumunda su itiraz yapildi (2026-08-21): tahmin donemi
Nisan-Temmuz 2026 ve bu doneme ait GERCEKLESMIS hava verisine erisimimiz
var; gercek bir sistemde olmazdi, dolayisiyla kullanmak sizinti sayilir.

Itiraz gercekcilik acisindan hakli. Kural acisindan degil: organizatorun
"Kullanilabilecek Ek Veriler" slaydi hava durumunu acikca listeliyor.

Ama asil soru sudur ve tartisilarak degil OLCULEREK cevaplanir:
**gerceklesmis hava ile iklim ortalamasi arasindaki fark kac puan?**

Cunku havanin degerinin buyuk kismi sizinti DEGIL. "Temmuz sicaktir"
iklimsel bilgidir, herkesin elindedir, gelecekten okuma degildir. Sizinti
olan kisim yalnizca "27 Haziran 2026 tam olarak 34,2 dereceydi" ile
"27 Haziran'da ortalama 31 derece olur" arasindaki farktir.

Bu betik o farki olcer. Iki sey birden verir:
  1. Organizator "test donemi dis verisi yasak" derse KAC PUAN kaybederiz.
  2. Juri sunumu icin: gerceklesmis veriye bagimliligimizi olctuk.

YONTEM
------
Iklim ortalamasi 2020-2024'ten kurulur -- yani egitim verimizin (2025-01)
tamamen ONCESINDEN. Her ilce x yilin-gunu icin ortalama. Sonra dogrulama
bloklarinin ETIKET PENCERESINDEKI hava kolonlari bu ortalamayla degistirilir
(ozet penceresi dokunulmaz: gecmis hava zaten mesru bilgidir).

Fit: 3 blok x 3 tohum x 2 aday = 18 CatBoost ~ 11 dakika.

Calistirma::

    python scripts/deney_iklim.py
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

#: Iklim ortalamasinin kuruldugu yillar. Egitim verimiz 2025-01'de basliyor,
#: yani bu pencere tamamen ONCESINDE -- hicbir sekilde gelecekten okuma yok.
IKLIM_YILLARI = (2020, 2021, 2022, 2023, 2024)


def iklim_tablosu() -> pd.DataFrame:
    """Ilce x yilin-gunu iklim ortalamasi. Sayisal hava kolonlarinin hepsi."""
    hava = tm.hava_yukle()
    gecmis = hava[hava["tarih"].dt.year.isin(IKLIM_YILLARI)].copy()
    gecmis["_yg"] = gecmis["tarih"].dt.dayofyear
    sayisal = [
        k
        for k in gecmis.columns
        if k not in ("tarih", "ilce_key", "_yg") and gecmis[k].dtype.kind in "if"
    ]
    return gecmis.groupby(["ilce_key", "_yg"], observed=True)[sayisal].mean().reset_index()


def hava_kolonlari(kolonlar: list[str]) -> list[str]:
    return [k for k in kolonlar if k.startswith(d.AILELER["hava"])]


def iklimle_degistir(
    cerceve: pd.DataFrame, iklim: pd.DataFrame, degistir: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Cercevenin hava kolonlarini iklim ortalamasiyla degistirir.

    Turetilmis kolonlar (cdd*, hareketli ortalamalar) ham kolonlardan
    yeniden hesaplanmiyor -- iklim tablosunda ZATEN varlar, cunku ayni
    turetme hava tablosunun tamamina uygulanmis durumda. Yani cdd22'nin
    iklim ortalamasi, gunluk cdd22'lerin ortalamasidir; bu, "ortalama
    sicakliktan hesaplanan cdd"den biraz farklidir ama daha dogrudur:
    beklenen sogutma yuku, beklenen sicakligin sogutma yuku degildir
    (Jensen esitsizligi).
    """
    sonuc = cerceve.copy()
    anahtar = pd.DataFrame(
        {
            "ilce_key": cerceve["ilce_key"].to_numpy(),
            "_yg": cerceve[tm.ZAMAN].dt.dayofyear.to_numpy(),
        }
    )
    var = [k for k in degistir if k in iklim.columns]
    yeni = anahtar.merge(iklim[["ilce_key", "_yg", *var]], on=["ilce_key", "_yg"], how="left")
    for k in var:
        v = yeni[k].to_numpy()
        # Iklimde bulunamayan gun (29 Subat) icin gerceklesmisi birak.
        sonuc[k] = np.where(np.isnan(v), cerceve[k].to_numpy(), v)
    return sonuc, var


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("GERCEKLESMIS HAVA vs IKLIM ORTALAMASI")
    print("=" * 96)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    hk = hava_kolonlari(kolonlar)
    print(f"  egitim {len(egitim):,} satir | hava kolonu {len(hk)}")
    iklim = iklim_tablosu()
    print(f"  iklim tablosu {len(iklim):,} satir ({IKLIM_YILLARI[0]}-{IKLIM_YILLARI[-1]})")

    for ad in ("GERCEKLESMIS hava (mevcut)", "IKLIM ortalamasi"):
        t0 = time.time()
        blok_skorlari = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = di.blok_parcalari(egitim, b.ad)
            hedef = dogrulama
            if ad.startswith("IKLIM"):
                # YALNIZCA dogrulama (etiket) penceresi degisir. Egitim
                # penceresindeki hava GECMIS bilgidir; mesru.
                hedef, var = iklimle_degistir(dogrulama, iklim, hk)
                if b.ad == tm.BLOKLAR[0].ad:
                    print(f"  degistirilen kolon: {len(var)} / {len(hk)}")
            tahminler = []
            for tohum in di.TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, kolonlar, tm.SOGUK_MASKE_ORANI, tohum)
                tahminler.append(di.egit_tahmin("cat", maskeli, hedef, kolonlar, tohum))
                del maskeli
            blok_skorlari[b.ad] = [di.skorla(gercek, soguk, np.mean(tahminler, axis=0))]
        di.yazdir(ad, blok_skorlari)
        di.kaydet(ad, blok_skorlari, {"deney": "iklim"})
        print(f"      ({time.time() - t0:.0f} sn)")

    print("\n  Fark = gerceklesmis hava bilmenin DEGERI.")
    print("  Kucukse: 'sizinti' itirazi bizim icin maddi degil, ve organizator")
    print("  yasaklarsa kaybimiz o kadar. Buyukse: bagimliligimizi bilerek tasiyoruz.")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
