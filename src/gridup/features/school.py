"""MEB okul takvimi feature'lari: donem, ara tatil, yariyil, yaz.

NEDEN AYRI BIR AILE
-------------------
Elektrik tuketim ve kesinti deseni okul donemlerinde degisir:

  * okul binalarinin yuku (isitma, aydinlatma, yemekhane) yalnizca donem
    icinde vardir; bir ilcedeki okul yogunluguna gore sabah pik profili kayar,
  * servis/veli trafigi ve konut ritmi ders gunlerinde farklidir,
  * yaz tatili Ege'de nufusu sahil ilcelerine tasir -- yani YUK COGRAFYASI
    degisir; ilce bazli tahminde dogrudan sinyaldir.

Resmi tatil kolonlari (``temporal.add_turkish_holiday_features``) bunu
YAKALAYAMAZ: okul ara tatili resmi tatil DEGILDIR. Ornek: 15-19 Kasim 2021'de
okullar bes is gunu kapaliydi ama kamu ve sanayi acikti -- tatil kolonlari o
haftayi siradan bir hafta sanir.

SIZINTI RISKI YOK (bir istisnayla): MEB takvimi ders yili baslamadan aylar
once ilan edilir ve tahmin aninda bilinir; lag/horizon gerekmez --
``ramadan_calendar`` ile ayni sinif. ISTISNA: olaganustu degisiklikler kisa
surede duyurulabilir (deprem sonrasi 20 Sub 2023 acilisi ~9 gun once ilan
edildi, meb.gov.tr haber/29112). Yani gecmis icin takvim kesindir; GELECEK
31 gunluk bir tahmin penceresi olaganustu bir donemi kapsiyorsa takvimin
degisebilecegi akilda tutulmalidir.

TARIH DOGRULAMASI (2026-08-15'te MEB duyurulari + basin arsiviyle teyit edildi)
-------------------------------------------------------------------------------
Kapsanan alti kaydin TAMAMI dogrulandi; bu modulde TAHMINI tarih YOKTUR:

  2020-2021  yalnizca ikinci donem KUYRUGU (AA duyurusu, karne 18 Haziran):
             8 Sub - 18 Haz 2021. Gercek GDZ verisi 2021-05'te basladigi icin
             kapsama alindi. DIKKAT: bahar 2021 COVID donemidir -- takvim
             "acik" dese de egitim buyuk olcude UZAKTAN yurudu (29 Nis-17 May
             tam kapanma). Kolon RESMI takvimi kodlar, fiziksel devami degil.
  2021-2022  OGM calisma takvimi PDF'i: 6 Eyl 2021 - 21 Oca 2022 /
             7 Sub - 17 Haz 2022; aralar 15-19 Kas 2021 ve 11-15 Nis 2022.
  2022-2023  resmi takvim: 12 Eyl 2022 baslangic, 20 Oca 2023 donem sonu,
             16 Haz 2023 bitis. 6 Subat 2023 depremi ikinci donemi ERTELEDI:
             deprem disi illerde (bes Ege ili dahil) okullar 20 SUBAT 2023'te
             acildi; afet illerinde 1-27 Mart arasi kademeliydi. Nisan arasi
             IPTAL EDILMEDI: 17-21 Nis 2023 uygulandi, Ramazan Bayrami
             (21-23 Nis) ile birlesti.
  2023-2024  meb.gov.tr haber/30337: 11 Eyl 2023 - 19 Oca 2024 /
             5 Sub - 14 Haz 2024; aralar 13-17 Kas 2023, 8-12 Nis 2024.
  2024-2025  meb.gov.tr haber/33888: 9 Eyl 2024 - 17 Oca 2025 /
             3 Sub - 20 Haz 2025; aralar 11-15 Kas 2024, 31 Mar - 4 Nis 2025
             (Ramazan Bayrami ile birlesik bahar arasi).
  2025-2026  meb.gov.tr haber/37198: 8 Eyl 2025 - 16 Oca 2026 /
             2 Sub - 26 Haz 2026; aralar 10-14 Kas 2025, 16-20 Mar 2026
             (Ramazan Bayrami 20-22 Mar ile birlesik).

ETIKETLEME KARARLARI
--------------------
* Iki donem arasindaki TUM bosluk (aradaki hafta sonlari dahil) ``yariyil``
  sayilir: aileler icin kesintisiz tek bir kapali bloktur, seyahat ve tuketim
  davranisi butun blokta degisir. 2022-2023'te bu blok deprem ertelemesiyle
  30 gune uzar (21 Oca - 19 Sub 2023) -- Ege illeri icin dogru olan budur.
* ``ara`` tatiller resmi takvimdeki bes is gunudur; bitisik hafta sonlari
  donem ici ``hafta_sonu`` olarak kalir (okul zaten kapalidir, ayrim
  ``okul_acilisa_gun`` kolonunda korunur).
* Ders yillari arasindaki her gun (eylul baslangicindan onceki gunler dahil)
  ``yaz`` sayilir.

OLCULDU (bu makinede, tam kapsam 2021-05-01..2026-08-31 = 1949 gun)
-------------------------------------------------------------------
    yok (okul acik) : 945 gun
    yaz             : 482 gun
    hafta_sonu      : 378 gun
    yariyil         :  94 gun  (deprem yili 30 gun, digerleri 16'sar)
    ara             :  50 gun  (10 ara tatil x 5 is gunu)
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import pandas as pd

DateRange = tuple[str, str]
TimestampRange = tuple[pd.Timestamp, pd.Timestamp]


class _SchoolYear(TypedDict):
    ad: str
    donemler: tuple[DateRange, ...]
    aralar: tuple[DateRange, ...]


class _ResolvedSchoolYear(TypedDict):
    ad: str
    donemler: tuple[TimestampRange, ...]
    aralar: tuple[TimestampRange, ...]


__all__ = [
    "COVERAGE_END",
    "COVERAGE_START",
    "MISSING_OPENING_DISTANCE",
    "SCHOOL_DAY_TYPES",
    "SCHOOL_YEARS",
    "add_school_calendar_features",
    "school_calendar",
]

#: Kapsam disi / bozuk tarihli satirlar icin "acilisa uzak" sentineli.
#: ``temporal.MISSING_HOLIDAY_DISTANCE`` ile ayni konvansiyon (999).
MISSING_OPENING_DISTANCE = 999

#: Takvim kapsami. Baslangic, gercek GDZ verisinin ilk ayina (2021-05) gore
#: secildi ve 2020-2021 ikinci doneminin dogrulanmis araligine denk duser.
#: Bitis, 2025-2026 ders yili sonrasi yazin sonu -- 2026-2027 takvimi bu
#: modulde YOKTUR, o yuzden Eylul 2026 ve sonrasi kapsam disidir.
COVERAGE_START = pd.Timestamp("2021-05-01")
COVERAGE_END = pd.Timestamp("2026-08-31")

#: Uretilen kategorik kolonun SABIT kategori kumesi. "bilinmiyor" yalnizca
#: kapsam disi veya bozuk tarihli satirlarda gorulur; sabit kumede tutuyoruz
#: ki dtype, girdi ne olursa olsun ayni kalsin (train/test tutarliligi).
SCHOOL_DAY_TYPES = ("yok", "hafta_sonu", "ara", "yariyil", "yaz", "bilinmiyor")

#: MEB resmi takvim kayitlari -- moduldeki tek veri kaynagi.
#: ``donemler``: okulun ACIK sayildigi araliklar (hafta ici). Iki donem
#: arasindaki bosluk yariyil, kayitlar arasi bosluk yaz olarak TURETILIR.
#: ``aralar``: donem ICINDEKI bes is gunluk ara tatiller (2019'dan beri var).
#: Tarih dogrulama kaynaklari modul docstring'inde.
SCHOOL_YEARS: tuple[_SchoolYear, ...] = (
    {
        "ad": "2020-2021",
        # Yalnizca dogrulanmis ikinci donem -- kapsam 2021-05-01'de basladigi
        # icin onceki bolumu yazmiyoruz. COVID uyarisi: docstring.
        "donemler": (("2021-02-08", "2021-06-18"),),
        # 12-16 Nisan 2021 ikinci donem ara tatili (web'den dogrulandi).
        # Kapsam 2021-05-01'de basladigi icin bugun etiket uretmez; kayit,
        # COVERAGE_START ileride geri cekilirse dogru kalsin diye tam yazildi.
        "aralar": (("2021-04-12", "2021-04-16"),),
    },
    {
        "ad": "2021-2022",
        "donemler": (("2021-09-06", "2022-01-21"), ("2022-02-07", "2022-06-17")),
        "aralar": (("2021-11-15", "2021-11-19"), ("2022-04-11", "2022-04-15")),
    },
    {
        "ad": "2022-2023",
        # Ikinci donem baslangici resmi takvimde 6 Subat'ti; 6 Subat 2023
        # depremi nedeniyle deprem disi illerde (bizim bes il dahil)
        # 20 Subat'a ertelendi. Afet illeri farkliydi -- bu modul GDZ/Ege
        # bolgesini kodlar.
        "donemler": (("2022-09-12", "2023-01-20"), ("2023-02-20", "2023-06-16")),
        "aralar": (("2022-11-14", "2022-11-18"), ("2023-04-17", "2023-04-21")),
    },
    {
        "ad": "2023-2024",
        "donemler": (("2023-09-11", "2024-01-19"), ("2024-02-05", "2024-06-14")),
        "aralar": (("2023-11-13", "2023-11-17"), ("2024-04-08", "2024-04-12")),
    },
    {
        "ad": "2024-2025",
        "donemler": (("2024-09-09", "2025-01-17"), ("2025-02-03", "2025-06-20")),
        "aralar": (("2024-11-11", "2024-11-15"), ("2025-03-31", "2025-04-04")),
    },
    {
        "ad": "2025-2026",
        "donemler": (("2025-09-08", "2026-01-16"), ("2026-02-02", "2026-06-26")),
        "aralar": (("2025-11-10", "2025-11-14"), ("2026-03-16", "2026-03-20")),
    },
)


def _yillari_coz() -> tuple[_ResolvedSchoolYear, ...]:
    """``SCHOOL_YEARS``i Timestamp'lere cevirir ve tutarliligi dogrular.

    Import aninda calisir: bir yazim hatasi (ters aralik, string bozuklugu)
    modul yuklenirken PATLAR -- sessizce yanlis takvim uretmek yerine.
    """
    cozulmus: list[_ResolvedSchoolYear] = []
    for kayit in SCHOOL_YEARS:
        donemler = tuple((pd.Timestamp(b), pd.Timestamp(s)) for b, s in kayit["donemler"])
        aralar = tuple((pd.Timestamp(b), pd.Timestamp(s)) for b, s in kayit["aralar"])
        for baslangic, bitis in (*donemler, *aralar):
            if baslangic > bitis:
                raise ValueError(
                    f"SCHOOL_YEARS kaydi bozuk ({kayit['ad']}): {baslangic.date()} > {bitis.date()}"
                )
        cozulmus.append({"ad": kayit["ad"], "donemler": donemler, "aralar": aralar})
    return tuple(cozulmus)


_YILLAR = _yillari_coz()


def _gun_etiketleri(gunler: pd.DatetimeIndex) -> np.ndarray:
    """Her gun icin tatil turu etiketi uretir.

    Oncelik sirasi (sonra yazilan kazanir): donem ici hafta ici ``yok`` /
    hafta sonu ``hafta_sonu`` -> iki donem arasi ``yariyil`` -> ``ara``.
    Hicbir kayda dusmayen gun ``yaz``dir -- ders yillari arasindaki butun
    bosluklar (eylul baslangicindan onceki gunler dahil) boyle etiketlenir.
    """
    etiket = np.full(len(gunler), "yaz", dtype=object)
    hafta_sonu = np.asarray(gunler.dayofweek >= 5)

    for kayit in _YILLAR:
        donemler = kayit["donemler"]
        for baslangic, bitis in donemler:
            maske = (gunler >= baslangic) & (gunler <= bitis)
            etiket[maske & ~hafta_sonu] = "yok"
            etiket[maske & hafta_sonu] = "hafta_sonu"
        if len(donemler) == 2:
            # Iki donem arasindaki TUM gunler yariyil -- aradaki hafta
            # sonlari dahil (gerekce: modul docstring'i).
            (_, birinci_bitis), (ikinci_baslangic, _) = donemler
            yariyil = (gunler > birinci_bitis) & (gunler < ikinci_baslangic)
            etiket[yariyil] = "yariyil"
        for baslangic, bitis in kayit["aralar"]:
            maske = (gunler >= baslangic) & (gunler <= bitis)
            etiket[maske] = "ara"
    return etiket


def school_calendar(start: object, end: object) -> pd.DataFrame:
    """MEB okul takvimini gunluk cozunurlukte dondurur.

    Args:
        start: Baslangic gunu (str/Timestamp/date -- ``pd.Timestamp``in
            kabul ettigi her sey). Gune normalize edilir.
        end: Bitis gunu (dahil).

    Returns:
        Su kolonlarla YENI bir DataFrame (gunde bir satir):
          * ``tarih`` -- gun (datetime64, saat 00:00)
          * ``okul_acik`` -- fiilen ders yapilan gun mu (0/1, int8)
          * ``tatil_turu`` -- kategorik: ``SCHOOL_DAY_TYPES``ten biri
            (``bilinmiyor`` bu fonksiyonda hic uretilmez; kategori kumesi
            ``add_school_calendar_features`` ile ayni kalsin diye durur)

    Raises:
        ValueError: Aralik ters ise, tarih cozulmuyorsa veya kapsam
            (``COVERAGE_START``..``COVERAGE_END``) disina tasiyorsa.
            Kapsam disini SESSIZCE "yaz" saymak, 2026-2027 gibi HENUZ ILAN
            EDILMEMIS bir yili uydurmak olurdu -- o yuzden hata.
    """
    baslangic = pd.Timestamp(start)
    bitis = pd.Timestamp(end)
    if pd.isna(baslangic) or pd.isna(bitis):
        raise ValueError("school_calendar: start/end gecerli bir tarih olmali, NaT verildi.")
    baslangic = baslangic.normalize()
    bitis = bitis.normalize()
    if baslangic > bitis:
        raise ValueError(
            f"school_calendar: baslangic ({baslangic.date()}) bitisten ({bitis.date()}) sonra."
        )
    if baslangic < COVERAGE_START or bitis > COVERAGE_END:
        raise ValueError(
            f"school_calendar: istenen aralik {baslangic.date()}..{bitis.date()}, "
            f"takvim kapsami {COVERAGE_START.date()}..{COVERAGE_END.date()}. "
            "Kapsam disi yillar icin SCHOOL_YEARS'a DOGRULANMIS kayit ekle -- "
            "tahmini tarih eklersen bunu acikca isaretle (repo kurali)."
        )

    gunler = pd.date_range(baslangic, bitis, freq="D")
    etiketler = _gun_etiketleri(gunler)
    return pd.DataFrame(
        {
            "tarih": gunler,
            "okul_acik": (etiketler == "yok").astype("int8"),
            "tatil_turu": pd.Categorical(etiketler, categories=list(SCHOOL_DAY_TYPES)),
        }
    )


def _acilisa_gun(takvim: pd.DataFrame) -> np.ndarray:
    """Her takvim gunu icin bir sonraki ACIK gune kalan gun sayisi.

    Acik gunde 0. Kapsam icinde bir sonraki acilis yoksa (2026 yazinin
    kuyrugu: sonraki acilis Eylul 2026'da, kapsam disinda) sentinel doner.
    """
    tum_gunler = takvim["tarih"].to_numpy(dtype="datetime64[D]")
    acik_gunler = tum_gunler[takvim["okul_acik"].to_numpy(dtype=bool)]

    mesafe = np.full(len(tum_gunler), MISSING_OPENING_DISTANCE, dtype=np.int16)
    if acik_gunler.size == 0:
        return mesafe

    konum = np.searchsorted(acik_gunler, tum_gunler, side="left")
    bulunan = konum < acik_gunler.size
    fark = (acik_gunler[konum[bulunan]] - tum_gunler[bulunan]).astype("int64")
    mesafe[bulunan] = np.minimum(fark, MISSING_OPENING_DISTANCE).astype(np.int16)
    return mesafe


def add_school_calendar_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    prefix: str = "okul",
) -> pd.DataFrame:
    """Okul takvimi feature'lari ekler. YENI frame dondurur.

    Args:
        frame: Girdi frame'i (degistirilmez).
        time_column: Zaman kolonu adi.
        prefix: Uretilen kolonlarin oneki.

    Returns:
        Su kolonlar eklenmis yeni frame:
          * ``{prefix}_acik_mi`` -- fiilen ders yapilan gun mu (0/1, int8)
          * ``{prefix}_tatil_turu`` -- kategorik (``SCHOOL_DAY_TYPES``)
          * ``{prefix}_acilisa_gun`` -- bir sonraki acik gune kalan gun
            (int16; acikken 0, hafta sonu 1-2, yazda onlarca). Agac modeli
            icin "tatilin neresindeyiz" sinyali: yazin son haftasi ile
            ortasi ayni "yaz" etiketini tasir ama farkli davranir.

    Kapsam disi (``COVERAGE_START`` oncesi / ``COVERAGE_END`` sonrasi) veya
    bozuk tarihli satirlar SESSIZCE atlanmaz: kac satir oldugu yazdirilir ve
    ``tatil_turu='bilinmiyor'``, ``acik_mi=0``,
    ``acilisa_gun=MISSING_OPENING_DISTANCE`` verilir. Boylece model "takvimi
    bilmiyoruz" durumunu "yaz tatili" sanmaz.

    Raises:
        KeyError: ``time_column`` frame'de yoksa.
    """
    if time_column not in frame.columns:
        raise KeyError(f"Zaman kolonu '{time_column}' frame'de yok.")

    times = pd.to_datetime(frame[time_column], errors="coerce")
    if getattr(times.dt, "tz", None) is not None:
        # Takvim gun cozunurlugundedir; tz-aware girdiyi naive'e indiriyoruz
        # ki kapsam karsilastirmasi TypeError ile cokmesin.
        times = times.dt.tz_localize(None)
    times = times.dt.normalize()

    kapsam_ici = times.notna() & (times >= COVERAGE_START) & (times <= COVERAGE_END)
    dis_sayisi = int((~kapsam_ici).sum())
    if dis_sayisi:
        print(
            f"[add_school_calendar_features] UYARI: {dis_sayisi:,} satir okul takvimi "
            f"kapsami ({COVERAGE_START.date()}..{COVERAGE_END.date()}) disinda veya "
            f"tarihi bozuk (%{dis_sayisi / max(len(frame), 1) * 100:.2f}). Bu satirlar "
            f"tatil_turu='bilinmiyor', acik_mi=0, acilisa_gun={MISSING_OPENING_DISTANCE} alir."
        )

    takvim = school_calendar(COVERAGE_START, COVERAGE_END)
    indeks = takvim["tarih"]
    tur_tablosu = pd.Series(takvim["tatil_turu"].astype(object).to_numpy(), index=indeks)
    acik_tablosu = pd.Series(takvim["okul_acik"].to_numpy(), index=indeks)
    mesafe_tablosu = pd.Series(_acilisa_gun(takvim), index=indeks)

    # Series.map deger bazlidir -- frame indeksi tekrarli olsa bile (concat
    # sonrasi tipik durum) hizalama kaymasi yasanmaz.
    tur = times.map(tur_tablosu).where(kapsam_ici, "bilinmiyor")
    acik_mi = times.map(acik_tablosu).where(kapsam_ici, 0).astype("int8")
    acilisa = times.map(mesafe_tablosu).where(kapsam_ici, MISSING_OPENING_DISTANCE).astype("int16")

    return frame.assign(
        **{
            f"{prefix}_acik_mi": acik_mi,
            f"{prefix}_tatil_turu": pd.Categorical(tur, categories=list(SCHOOL_DAY_TYPES)),
            f"{prefix}_acilisa_gun": acilisa,
        }
    )
