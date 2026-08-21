"""Maruziyet etkilesimleri: ruzgar TEK BASINA degil, AGACLA birlikte keser.

NEDEN BU MODUL VAR
------------------
Kesinti literaturunde en tekrarlanan bulgu, etkinin CARPIMSAL oldugudur:

  * NHESS 2023 (10.5194/nhess-23-1665-2023) -- AYNI ruzgar hizinda kesinti
    olasiligi yaprakli mevsimde 3-4x, islak toprakta 2-3x, ikisi birlikteyken
    4-5x. Yalnizca-ruzgar modeli buyuk kesintileri 2-5x EKSIK tahmin ediyor.
  * NHESS 2021 (10.5194/nhess-21-607-2021) -- 7 gunluk ONCUL yagis, ruzgardan
    sonra 2. en onemli degisken (onem 0,33; ruzgar 1,00).
  * UConn/Eversource OPM (WO2018013148A1) -- esik-ustu maruziyet SURESI anlik
    degerden ustun; medyan APE %130 -> %59.

Fizik tek cumleyle: ruzgar hattin kendisini nadiren koparir; AGACI devirir,
agac hatta duser. Agac ortusu bu yuzden bir CARPANDIR, ayri bir toplanan
degil. Yaprakli agac daha cok ruzgar tutar (yelken etkisi); islak toprak
kokun tutunmasini zayiflatir.

NE ZATEN VARDI, NE EKSIKTI
--------------------------
Depoda esik-ustu saatler ve kantiller ZATEN vardi (``hava_saatlik_turev``:
``ruzgar_8ms_saat`` ... ``hamle_25ms_saat``, ``ruzgar_q90``), ESA WorldCover
agac ortusu orani da vardi. Eksik olan ikisinin CARPIMIYDI. GBDT etkilesim
ogrenebilir, ama sinirli agac derinliginde ve ornegin seyrek oldugu bolgede
(siddetli firtina gunleri -- tam da onem tasiyan gunler) zorlanir. Fizik
biliniyorsa etkilesimi acikca vermek ucuzdur ve orneklem verimlidir.

SIZINTI GUVENCESI
-----------------
Uretilen hicbir kolon HEDEFE dokunmaz; fonksiyon hedef adini parametre
olarak dahi almaz. Girdilerin tamami disaridan gelen hava/arazi verisidir --
bunlar tahmin aninda da bilinir (hava tahmini koprusu tam bunun icin var).
Tek GECMISE bakan hesap ``oncul_yagis_7g``'dir ve bilerek ONCEKI gunlerden
toplanir; ayni gunun yagisi "oncul" degildir.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "MaruziyetSonucu",
    "ONCUL_PENCERE_GUN",
    "SICAK_GUN_ESIGI",
    "add_maruziyet_etkilesimleri",
    "yaprak_mevsimi_orani",
]

#: Oncul islaklik penceresi (gun). NHESS 2021 7 gunluk toplami 2. en onemli
#: degisken olarak olctu; ayni pencereyi kullaniyoruz.
ONCUL_PENCERE_GUN = 7

#: "Sicak gun" esigi (C, gunluk maksimum).
#:
#: 32 C secildi: EPIAS panelinde (2022-2026, 96 ilce) temmuz-agustos gunluk
#: maksimum ortalamasi bu civardadir, yani esik yaz rejimini ikiye boler.
#: Daha yuksek bir esik (or. 38) yalnizca birkac gunu isaretler ve
#: sureklilik sayaci neredeyse hep 0 kalir; daha dusuk bir esik (or. 28)
#: butun yazi tek bir blok yapar ve ayirt etme gucunu kaybeder.
SICAK_GUN_ESIGI = 32.0

#: Ege'de yaprakli mevsimin baslangic/bitis gunleri (yilin gunu).
#:
#: Ege Bolgesi yaprak dokenlerinde (cinar, kavak, dut, meyve agaclari) yaprak
#: acilimi mart sonu-nisan, doku kasim. Gecisler ANI degildir, bu yuzden
#: pencereler yumusatilir: kesin bir esik, sinir gunlerinde modele sahte bir
#: siçrama ogretirdi.
YAPRAK_ACILIM_GUN = 90  # ~31 Mart
YAPRAK_DOKUM_GUN = 320  # ~16 Kasim
YAPRAK_GECIS_GUN = 25.0

#: Girdi TAKMA ADLARI: kanonik ad -> denenecek kolon adlari, SIRAYLA.
#:
#: NEDEN GEREKLI (2026-08-21 provasi olctu): ilk surum ``agac_ortusu_orani``
#: bekliyordu, gercek kolon adi ``agac_orani``. Bes etkilesimin BESI birden
#: atlandi -- betik cokmedi ve nedenini soyledi, ama uretilen deger sifirdi.
#: Yarisma verisinde ayni sey daha da olasidir; tek bir ad varsaymak kirilgan.
#:
#: ``toprak_nem_ort`` ile ``nem_ort`` AYRI seylerdir ve karistirmak fizigi
#: bozar: ilki TOPRAK nemi (koku gevsetir, mekanizma budur), ikincisi HAVA
#: nemi. Sira bu yuzden toprakla baslar.
_TAKMA_ADLAR: dict[str, tuple[str, ...]] = {
    "agac": ("agac_orani", "agac_ortusu_orani", "tree_cover_fraction"),
    "bitki": ("bitki_ortusu_orani", "agac_orani"),
    "ruzgar": ("ruzgar_max", "wind_speed_10m_max"),
    "hamle_saat": ("hamle_20ms_saat", "hamle_15ms_saat", "hamle_25ms_saat"),
    "toprak_nem": ("toprak_nem_ort", "toprak_islak_saat", "nem_ort"),
    "yaprak": ("yaprak_mevsimi",),
    "oncul": ("oncul_yagis_7g",),
    # SEBEKE BUYUKLUGU vekilleri -- bkz. asagidaki olcum notu.
    "yerlesim": ("yerlesim_orani", "built_up_fraction"),
    "sogutma": ("sogutma_derece_gun", "sicaklik_max"),
    "sebeke": ("osm_dagitim_hat_km", "osm_iletim_hat_km", "osm_trafo"),
    "sicaklik": ("sicaklik_max", "temperature_2m_max"),
    "sicak_sureklilik": ("sicak_sureklilik",),
}

#: Etkilesim tanimlari: (uretilecek ad, KANONIK girdiler, aciklama).
#:
#: Sira ONEMSIZ; her biri bagimsizdir ve girdisi eksikse yalnizca kendisi
#: atlanir. Adlar ``maruziyet_`` on ekiyle baslar ki feature listesinde
#: kokenleri tek bakista gorunsun.
_ETKILESIMLER: tuple[tuple[str, tuple[str, ...], str], ...] = (
    (
        "maruziyet_ruzgar_agac",
        ("ruzgar", "agac"),
        "Ruzgar hattin kendisini degil AGACI devirir; agac ortusu carpandir.",
    ),
    (
        "maruziyet_hamle_agac",
        ("hamle_saat", "agac"),
        "Esik-ustu hamle SURESI x agac ortusu -- OPM cekirdegi (APE %130->%59).",
    ),
    (
        "maruziyet_hamle_agac_yaprak",
        ("hamle_saat", "agac", "yaprak"),
        "Yaprakli agac daha cok ruzgar tutar; NHESS 2023: ayni ruzgarda 3-4x.",
    ),
    (
        "maruziyet_ruzgar_agac_islak",
        ("ruzgar", "agac", "oncul"),
        "Islak toprak koku gevsetir; NHESS 2023: 2-3x, yaprakla birlikte 4-5x.",
    ),
    (
        "maruziyet_ruzgar_agac_toprak",
        ("ruzgar", "agac", "toprak_nem"),
        "Ayni mekanizma, yagis vekili yerine dogrudan TOPRAK nemi olcumuyle.",
    ),
    # ---------------------------------------------------------------- KENTSEL
    # Yukaridaki bes etkilesim ABD/Ingiltere firtina literaturunden gelir ve
    # KIRSAL havai hattin ormandan gectigi sebekeler icin yazilmistir. Ege
    # icin OLCTUK (2026-08-21, 162.240 satirlik EPIAS paneli x hava):
    #
    #     yerlesim_orani -> kesinti   rho = +0,155   <- en guclu TEK sinyal
    #     agac_orani     -> kesinti   rho = -0,058   <- NEGATIF
    #     ruzgar x agac  -> kesinti   rho = -0,031   (yaz)
    #     sicak x yerlesim -> kesinti rho = +0,136   (yaz)
    #
    # Yani burada kesinti SAYISINI belirleyen sey firtina degil, SEBEKENIN
    # BUYUKLUGU. Sebep tablosu da bunu soyluyor: en sik arizalar 'Ic Tesisat'
    # (78.621), 'OG Fider Acmasi' (72.269), 'AG Pano Kol Sigorta Atisi'
    # (28.727) -- ekipman arizasi, firtina hasari degil.
    #
    # Yaz aylari (7-8) hem EN YUKSEK kesintiyi (3,82 / 3,64 vs yil ort. 2,48)
    # hem EN DUSUK ruzgar korelasyonunu (+0,081 / +0,048 vs ocak +0,264)
    # tasiyor: yaz kesintileri termal/yuk kaynakli, kis kesintileri firtina
    # kaynakli. Iki rejim, iki ayri etkilesim kumesi.
    (
        "maruziyet_sicak_yerlesim",
        ("sogutma", "yerlesim"),
        "Sogutma yuku x kentlesme: klima yuku dagitim trafosunu zorlar (rho +0,136).",
    ),
    (
        "maruziyet_sicak_sureklilik",
        ("sicak_sureklilik",),
        "Ust uste sicak gun sayisi: trafo ilk gunde degil, SUREKLI yukte arizalanir.",
    ),
    (
        "maruziyet_sicak_sureklilik_yerlesim",
        ("sicak_sureklilik", "yerlesim"),
        "Sicak dalgasinin kentsel sebekedeki birikimli etkisi.",
    ),
    (
        "maruziyet_sebeke_sicak",
        ("sebeke", "sogutma"),
        "Sebeke uzunlugu x sogutma yuku: maruziyet x zorlanma.",
    ),
)


def _kolonu_coz(kanonik: str, frame: pd.DataFrame) -> str | None:
    """Kanonik girdi adini frame'deki gercek kolona cozer; yoksa ``None``."""
    for aday in _TAKMA_ADLAR.get(kanonik, (kanonik,)):
        if aday in frame.columns:
            return aday
    return None


@dataclass(frozen=True)
class MaruziyetSonucu:
    """Uretilen etkilesimler ve ATLANANLARIN gerekcesi.

    Attributes:
        frame: Yeni kolonlarin eklendigi kopya. Girdi mutasyona ugramaz.
        uretilen: Basariyla eklenen kolon adlari.
        atlanan: ``kolon adi -> neden atlandigi``. Sessiz atlama YOKTUR.
    """

    frame: pd.DataFrame
    uretilen: list[str] = field(default_factory=list)
    atlanan: dict[str, str] = field(default_factory=dict)

    def ozet(self) -> str:
        satirlar = [f"maruziyet: {len(self.uretilen)} etkilesim uretildi"]
        if self.uretilen:
            satirlar.append("  " + ", ".join(self.uretilen))
        for ad, neden in self.atlanan.items():
            satirlar.append(f"  ATLANDI {ad}: {neden}")
        return "\n".join(satirlar)


def yaprak_mevsimi_orani(zaman: pd.Series) -> pd.Series:
    """Yilin gunune gore yaprakli mevsim orani (0..1).

    Ani esik yerine iki yumusak gecis kullanilir. Sebep: 31 Mart'ta 0'dan
    1'e siçrayan bir degisken, modele takvimde var olmayan bir kirilma
    ogretir; agaclar bir gunde yapraklanmaz.

    Args:
        zaman: Tarih serisi.

    Returns:
        0 (yaprak yok) ile 1 (tam yaprakli) arasi oran, girdinin indeksiyle.
    """
    gun = pd.to_datetime(zaman).dt.dayofyear.to_numpy(dtype=float)
    acilim = 1.0 / (1.0 + np.exp(-(gun - YAPRAK_ACILIM_GUN) / YAPRAK_GECIS_GUN))
    dokum = 1.0 / (1.0 + np.exp((gun - YAPRAK_DOKUM_GUN) / YAPRAK_GECIS_GUN))
    return pd.Series(np.clip(acilim * dokum, 0.0, 1.0), index=zaman.index)


def _oncul_yagis(
    frame: pd.DataFrame, *, time_column: str, key_column: str | None, pencere: int
) -> pd.Series:
    """ONCEKI ``pencere`` gunun yagis toplami -- ayni gun HARIC.

    ``shift(1)`` bilerek ``rolling``den ONCE uygulanir: ayni gunun yagisi
    "oncul islaklik" degildir, o gunun kendi olayidir. Sirasi ters olsa
    degisken kendi gunuyle kirlenir.
    """
    sirali = frame.sort_values([*([key_column] if key_column else []), time_column])
    yagis = sirali["yagis_toplam"]
    if key_column:
        gruplu = yagis.groupby(sirali[key_column], sort=False)
        toplam = gruplu.shift(1).rolling(pencere, min_periods=1).sum()
    else:
        toplam = yagis.shift(1).rolling(pencere, min_periods=1).sum()
    return toplam.reindex(frame.index).fillna(0.0)


def _sicak_sureklilik(
    frame: pd.DataFrame,
    *,
    time_column: str,
    key_column: str | None,
    sicaklik_kolonu: str,
    esik: float,
) -> pd.Series:
    """Ust uste kacinci sicak gun -- BUGUN DAHIL.

    Trafo, sicak dalgasinin ilk gununde degil, sogumaya firsat bulamadigi
    ucuncu-dorduncu gununde arizalanir. Tek gunluk sicaklik bu birikimi
    tasimaz; ust uste gun sayisi tasir.

    Bugun DAHILDIR (``oncul_yagis``in aksine) cunku bu bir GECMIS olcumu
    degil, o gunun rejimini tarif eden bir durum degiskenidir -- ve o gunun
    sicakligi tahmin aninda hava tahmininden zaten bilinir.
    """
    sirali = frame.sort_values([*([key_column] if key_column else []), time_column])
    sicak = (sirali[sicaklik_kolonu].astype(float) > esik).astype(int)
    if key_column:
        gruplar = sirali[key_column]
        # Her "sicak degil" gunu yeni bir blok baslatir; blok icinde kumulatif
        # sayim ust uste kacinci gun oldugunu verir.
        blok = (sicak == 0).groupby(gruplar, sort=False).cumsum()
        seri = sicak.groupby([gruplar, blok], sort=False).cumsum()
    else:
        blok = (sicak == 0).cumsum()
        seri = sicak.groupby(blok, sort=False).cumsum()
    return seri.reindex(frame.index).fillna(0).astype(float)


def add_maruziyet_etkilesimleri(
    frame: pd.DataFrame,
    *,
    time_column: str,
    key_column: str | None = None,
    pencere: int = ONCUL_PENCERE_GUN,
    sicak_esigi: float = SICAK_GUN_ESIGI,
    etkilesimler: Sequence[tuple[str, tuple[str, ...], str]] | None = None,
) -> MaruziyetSonucu:
    """Fizik temelli carpimsal maruziyet etkilesimlerini ekler.

    Eksik girdi HATTI DURDURMAZ. Yarisma verisinin semasini bilmiyoruz; bir
    etkilesimin girdisi yoksa dogru davranis o etkilesimi atlayip nedenini
    raporlamaktir. Sessiz atlama da kabul degildir -- ``atlanan`` sozlugu
    her atlamanin sebebini tasir.

    Args:
        frame: Dis veri BAGLANMIS panel (hava + arazi ortusu kolonlari burada
            olmali; ``attach_external`` sonrasi cagrilir).
        time_column: Tarih kolonu.
        key_column: Varlik kolonu (ilce). Verilirse oncul yagis grup ICINDE
            hesaplanir -- verilmezse ilceler birbirine karisir.
        pencere: Oncul islaklik penceresi (gun).
        etkilesimler: Tanim listesini ezer (test ve arama icin).

    Returns:
        Uretilen kolonlari ve atlama gerekcelerini tasiyan kayit.
    """
    sonuc = frame.copy()
    uretilen: list[str] = []
    atlanan: dict[str, str] = {}

    # Yardimci degiskenler once: etkilesimlerin bazilari bunlara dayanir.
    if time_column in sonuc.columns:
        sonuc["yaprak_mevsimi"] = yaprak_mevsimi_orani(sonuc[time_column])
        uretilen.append("yaprak_mevsimi")
    else:
        atlanan["yaprak_mevsimi"] = f"zaman kolonu {time_column!r} yok"

    if "yagis_toplam" in sonuc.columns and time_column in sonuc.columns:
        sonuc[f"oncul_yagis_{pencere}g"] = _oncul_yagis(
            sonuc, time_column=time_column, key_column=key_column, pencere=pencere
        )
        uretilen.append(f"oncul_yagis_{pencere}g")
        if pencere != ONCUL_PENCERE_GUN:
            sonuc["oncul_yagis_7g"] = sonuc[f"oncul_yagis_{pencere}g"]
    else:
        atlanan[f"oncul_yagis_{pencere}g"] = "'yagis_toplam' veya zaman kolonu yok"

    sicaklik_kolonu = _kolonu_coz("sicaklik", sonuc)
    if sicaklik_kolonu and time_column in sonuc.columns:
        sonuc["sicak_sureklilik"] = _sicak_sureklilik(
            sonuc,
            time_column=time_column,
            key_column=key_column,
            sicaklik_kolonu=sicaklik_kolonu,
            esik=sicak_esigi,
        )
        uretilen.append("sicak_sureklilik")
    else:
        atlanan["sicak_sureklilik"] = "'sicaklik_max' veya zaman kolonu yok"

    for ad, girdiler, _gerekce in etkilesimler or _ETKILESIMLER:
        cozulen = {k: _kolonu_coz(k, sonuc) for k in girdiler}
        eksik = [k for k, v in cozulen.items() if v is None]
        if eksik:
            # Hangi ADLARIN denendigi de yazilir: "agac yok" demek yetmez,
            # yarisma verisinde kolon baska bir adla duruyor olabilir ve
            # okuyanin takma ad listesine ne ekleyecegini bilmesi gerekir.
            detay = "; ".join(f"{k} ({'/'.join(_TAKMA_ADLAR.get(k, (k,)))})" for k in eksik)
            atlanan[ad] = f"eksik girdi: {detay}"
            continue
        kolonlar = [cozulen[k] for k in girdiler]
        carpim = sonuc[kolonlar[0]].astype(float)
        for kolon in kolonlar[1:]:
            carpim = carpim * sonuc[kolon].astype(float)
        sonuc[ad] = carpim
        uretilen.append(ad)

    return MaruziyetSonucu(frame=sonuc, uretilen=uretilen, atlanan=atlanan)
