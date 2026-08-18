"""Zaman tabanli feature'lar: takvim, dongusel kodlama, gecikme (lag), kayan pencere.

Elektrik yuku ve ariza problemlerinde sinyalin buyuk kismi ZAMANDADIR:
gunun saati, haftanin gunu, mevsim, tatil, sicaklik gecmisi. Bu modul o
sinyali cikarir.

SIZINTI UYARISI
---------------
``add_lag_features`` ve ``add_rolling_features`` GECMISE bakar ve bu dogrudur.
Ama iki kural ihlal edilirse sizinti olur:

  1. Veri ZAMANA GORE SIRALI olmali. Sirali degilse ``shift(1)`` rastgele bir
     satiri alir ve gelecegi sizdirir. Bu fonksiyonlar sirayi KENDILERI garanti
     eder ve girdiyi degistirmez.

  2. Kayan pencere ``closed="left"`` olmali -- yani icinde bulunulan satirin
     kendi degeri pencereye GIRMEMELI. pandas varsayilani mevcut satiri dahil
     eder; bu, hedef turevli bir kolonda dogrudan hedef sizintisidir.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..turkish import tr_lower
from ..validation import parse_time_series

__all__ = [
    "ADMINISTRATIVE_LEAVE",
    "HOLIDAY_CODES",
    "LAST_TEN_NIGHTS",
    "MISSING_HOLIDAY_DISTANCE",
    "add_calendar_features",
    "add_cyclical_features",
    "add_ramadan_features",
    "add_turkish_holiday_features",
    "ramadan_calendar",
    "add_lag_features",
    "add_rolling_features",
    "add_expanding_features",
    "add_mass_event_features",
    "add_event_decay_features",
    "add_days_since_event_features",
    "add_previous_month_features",
    "add_upcoming_holiday_features",
    "shared_origin",
    "TURKISH_SEASONS",
]

# Gecerli bir tarihi olmayan (veya hic tatil bulunmayan) satirlar icin "tatile
# uzak" sentineli. Gercek bir mesafe olamayacak kadar buyuk secildi.
MISSING_HOLIDAY_DISTANCE = 999

# Tatil KIMLIKLERI. Serbest metin yerine sabit kod kullaniyoruz cunku
# kutuphanenin dondurdugu adlar uc sorun tasiyor (bu makinede OLCULDU):
#
#   1. Cakismada ';' ile birlesiyorlar:
#      2022-05-01 -> "Emek ve Dayanisma Gunu; Ramazan Bayrami (saat 13.00'ten)"
#      Turkce CSV'de ';' ALAN AYIRICISIDIR -- bu ad bir kolona yazilirsa
#      dosya kayar. Ayrica frekans=1 olan sahte bir "nadir kategori" yaratir.
#   2. ASCII disi karakter iceriyorlar: i (U+0131), s (U+015F), i (U+00EE),
#      u (U+00FC), c (U+00E7) ve kesme isareti. Kodlama zincirinin her
#      halkasinda bozulma riski.
#   3. Kutuphane surumleri arasinda metin degisebilir; sabit kod degismez.
#: ``holidays`` kutuphanesi tatil adlarinin DILINI sistem yerelinden secer.
#: ``_holiday_code`` ise ad icinde TURKCE anahtar kelime arar ("ramazan",
#: "cumhuriyet"...). Dil Ingilizce olursa ad "Eid al-Fitr" / "Republic Day"
#: gelir, hicbir anahtar eslesmez ve HER TATIL SESSIZCE 0 (tatil yok) olur.
#:
#: OLCULDU 2026-08-18: yerelde (Turkce Windows yereli) kodlar dogru
#: uretiliyordu; GitHub'in Ubuntu runner'inda (Ingilizce yerel) dort test
#: birden ``tatil_kod == 0`` ile dustu. KAGGLE DA LINUX/INGILIZCE yereldir --
#: yani bu hata fark edilmeseydi yarisma notebook'unda tum tatil ailesi
#: olu olacakti ve hicbir hata mesaji cikmayacakti.
#:
#: Bu yuzden dil ACIKCA sabitlenir; ortamdan miras ALINMAZ.
HOLIDAY_LANGUAGE = "tr"

HOLIDAY_NONE = 0
HOLIDAY_CODES: dict[str, int] = {
    "ramazan": 1,
    "kurban": 2,
    "cumhuriyet": 3,
    "egemenlik": 4,  # 23 Nisan Ulusal Egemenlik ve Cocuk Bayrami
    "genclik": 5,  # 19 Mayis Ataturk'u Anma, Genclik ve Spor Bayrami
    "zafer": 6,  # 30 Agustos
    "emek": 7,  # 1 Mayis
    "demokrasi": 8,  # 15 Temmuz
    "yilbasi": 9,
}

# Cakismada hangi tatil kazanir. Dini bayramlar once: elektrik tuketimi ve
# isgucu davranisi acisindan baskin olan onlardir (uc gunluk tatil, seyahat,
# sanayinin durmasi). Milli bayram tek gundur ve etkisi daha zayiftir.
_HOLIDAY_PRIORITY = (
    "ramazan",
    "kurban",
    "cumhuriyet",
    "egemenlik",
    "genclik",
    "zafer",
    "demokrasi",
    "emek",
    "yilbasi",
)

# IDARI IZIN gunleri -- hukumetin bayram oncesi/sonrasi verdigi EK tatil.
# holidays kutuphanesinde YOKTUR ama kamu kapanir, okullar kapanir, kazi ve
# planli saha isi durur. Kutuphane bu gunleri normal is gunu sayar.
#
# KAYNAK (duzeltildi)
# -------------------
# Bayram idari izinleri Resmi Gazete'de numarali "Cumhurbaskani Karari" olarak
# YAYIMLANMAZ. Dogru kaynak, Cumhurbaskanligi Idari Isler Baskanligi Personel
# ve Prensipler Genel Mudurlugu'nun kamu kurumlarina gonderdigi YAZI'dir;
# kamuoyuna Iletisim Baskanligi/AA duyurusu veya Cumhurbaskani aciklamasiyla
# bildirilir. Onceki surumdeki "Resmi Gazete arsivinden teyit et" talimati
# yanlis kaynagi gosteriyordu.
#
# KONVANSIYON (tek anlam)
# -----------------------
# Deger = YASAL TATILIN USTUNE EKLENEN izin.
#   1.0 = normalde tam is gunu olan gun tamamen izinli
#   0.5 = arife gunu; yasal tatil 13.00'te basliyor, SABAHI da izinli
# Bu ayrim onemli: arifede holiday_agirligi zaten 0.5'tir; buraya 1.0 yazmak
# ayni yarim gunu iki kez saymak olur.
ADMINISTRATIVE_LEAVE: dict[str, float] = {
    "2019-06-03": 0.5,
    "2019-06-07": 1.0,
    "2021-05-10": 1.0,
    "2021-05-11": 1.0,
    "2021-05-12": 0.5,
    "2022-07-13": 1.0,
    "2022-07-14": 1.0,
    # 2023 Kurban arifesi: 27 Haziran yasal olarak zaten yarim gun ->
    # ek izin de yarim gundur. Onceki 1.0 degeri konvansiyonu bozuyordu.
    "2023-06-26": 1.0,
    "2023-06-27": 0.5,
    "2024-04-08": 1.0,
    "2024-04-09": 0.5,
    "2024-06-20": 1.0,
    "2024-06-21": 1.0,
    # 2025 Ramazan: 26.03.2025 Cumhurbaskani aciklamasi -- "2, 3 ve 4 Nisan'da
    # kamu calisanlarimiz idari izinli sayilacak" (toplam 9 gunluk tatil).
    # Onceki surumde bu UC GUN TAMAMEN EKSIKTI.
    "2025-04-02": 1.0,
    "2025-04-03": 1.0,
    "2025-04-04": 1.0,
    # 2025 Kurban: EK idari izin VERILMEDI. Iletisim Baskanligi aciklamasi
    # yalnizca YASAL tatili teyit ediyordu (5 Haziran ogleden sonra + 6-9
    # Haziran). Onceki surumdeki "2025-06-05": 0.5 girdisi HATALIYDI --
    # zaten yasal olan yarim gunu ek izin sayiyordu; kaldirildi.
    # 2026 Kurban: 04.05.2026 aciklamasi -- resmi tatile "1,5 gun daha"
    # eklendi; 26 Mayis Sali OGLEDEN ONCE de izinli.
    "2026-05-25": 1.0,
    "2026-05-26": 0.5,
}

# Ege bolgesi icin mevsimsellik: yaz turizmi ve tarimsal sulama yuku belirleyicidir.
TURKISH_SEASONS = {
    12: "kis",
    1: "kis",
    2: "kis",
    3: "ilkbahar",
    4: "ilkbahar",
    5: "ilkbahar",
    6: "yaz",
    7: "yaz",
    8: "yaz",
    9: "sonbahar",
    10: "sonbahar",
    11: "sonbahar",
}


def _zaman_ayristir(frame: pd.DataFrame, time_column: str, *, cagiran: str) -> pd.Series:
    """Zaman kolonunu TEK kuralla ayristirir: ``validation.parse_time_series``.

    NEDEN TEK KAYNAK (olculdu)
    --------------------------
    Bu modulde AYNI kolon iki farkli sekilde cozuluyordu::

        shared_origin        : pd.to_datetime(..., format="mixed")
        add_calendar_features: pd.to_datetime(...)              # format YOK

    TR bicimli ``gg.aa.yyyy`` kolonunda ikisi ayri takvim uretiyordu::

        ham                     ['01.10.2025', ..., '15.10.2025', '20.10.2025']
        shared_origin  ONCE     -> 2025-01-10   (yanlis: ay-once okundu)
        add_calendar   ONCE     -> gun>12 olan 2 satir NaT (%40 gecersiz uyarisi)
        ikisi de       SONRA    -> 2025-10-01   (dogru; bicim VERIDEN kanitlandi)

    Yani origin hesabinda GORULEN satir, feature uretiminde GORULMUYORDU.

    ``parse_time_series`` bicimi tahmin etmez, KANITLAR: bir kayitta ilk
    bilesen 12'yi asiyorsa gun-once, ikinci bilesen asiyorsa ay-once. Hicbiri
    asmiyorsa iki okuma da hatasiz calisir ama farkli takvim uretir -- o
    durumda ``strict=True`` hata firlatir ve biz de firlatiriz.

    Bozuk tarihler ise TOLERE edilir (NaT olarak doner): ``strict`` kapisi
    yalnizca bicim kaniti icin, ayristirilabilen satirlar uzerinde calisir.
    Cagiranin bozuk satiri nasil ele alacagi kendi sorumlulugundadir.
    """
    if time_column not in frame.columns:
        raise KeyError(f"[{cagiran}] Zaman kolonu '{time_column}' frame'de yok.")
    ham = frame[time_column]
    gevsek = parse_time_series(ham, strict=False)
    try:
        # Bicim kapisi: NaT orani kapisini devre disi birakmak icin yalnizca
        # ayristirilabilen satirlarla cagriyoruz.
        parse_time_series(ham[gevsek.notna()], strict=True)
    except ValueError as hata:
        raise ValueError(f"[{cagiran}] '{time_column}' kolonu: {hata}") from hata
    return gevsek


def shared_origin(*frames: pd.DataFrame, time_column: str) -> pd.Timestamp:
    """Birden fazla frame icin ORTAK zaman baslangici hesaplar.

    ``add_calendar_features``in ``origin`` parametresine verilmek uzere tasarlandi::

        origin = shared_origin(train, test, time_column="tarih")
        train_f = add_calendar_features(train, "tarih", origin=origin)
        test_f  = add_calendar_features(test,  "tarih", origin=origin)

    Neden gerekli: gun sayaci monoton bir trend kolonudur ve train ile test
    AYNI sifir noktasindan olculmezse test satirlari train'in gecmisine
    kaymis gorunur. Bkz. ``add_calendar_features`` docstring'i.

    Raises:
        ValueError: Kolonun bicimi veriden kanitlanamiyorsa, hicbir frame'de
            gecerli tarih yoksa veya frame'ler arasinda saat dilimi
            (tz-aware/tz-naive) TUTARSIZSA.
    """
    minimums = [
        _zaman_ayristir(frame, time_column, cagiran=f"shared_origin frame#{sira}").min()
        for sira, frame in enumerate(frames)
    ]
    valid = [value for value in minimums if pd.notna(value)]
    if not valid:
        raise ValueError(f"'{time_column}' kolonunda gecerli tarih bulunamadi.")

    # tz KARISIMI: min() ham TypeError ("Cannot compare tz-naive and tz-aware
    # timestamps") firlatiyordu ve mesaj HANGI frame'in suclu oldugunu
    # soylemiyordu. Karsilastirmadan once kendimiz soyluyoruz.
    tz_durumlari = {value.tz is not None for value in valid}
    if len(tz_durumlari) > 1:
        etiket = [
            f"frame#{sira}={'tz-aware' if v.tz else 'tz-naive'}" for sira, v in enumerate(valid)
        ]
        raise ValueError(
            f"'{time_column}' kolonu frame'ler arasinda TUTARSIZ saat dilimi tasiyor: "
            f"{', '.join(etiket)}. Ortak origin hesaplanamaz -- once hepsini ayni "
            "hale getir: df['tarih'] = df['tarih'].dt.tz_localize(None)"
        )
    return min(valid)


def add_calendar_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    prefix: str | None = None,
    include_year: bool = True,
    origin: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Zaman damgasindan takvim feature'lari cikarir. YENI frame dondurur.

    Args:
        frame: Girdi (degistirilmez).
        time_column: Datetime kolonu adi.
        prefix: Uretilen kolon oneki. Varsayilan: ``time_column``.
        include_year: Yil kolonu eklensin mi? DIKKAT: test donemi train'den
            sonraysa yil bir sizinti/ekstrapolasyon riskidir -- agac modelleri
            gorulmemis yil degerini ele alamaz. Zaman ayrimi varsa ``False`` yap.
        origin: Gun sayaci icin sifir noktasi. **Verilmezse gun sayaci
            URETILMEZ.**

    Returns:
        Yeni DataFrame.

    ``origin`` NEDEN ZORUNLU TUTULUYOR
    ----------------------------------
    Gun sayaci (``{prefix}_gun_sayaci``) monoton bir trend kolonudur ve agac
    modellerinin zaman trendini yakalamasi icin degerlidir. Ama bu fonksiyon
    train ve test'e AYRI AYRI cagrildiginda -- ki normal kullanim budur --
    her cagri kendi ``times.min()`` degerini sifir kabul ederse **test sayaci
    yeniden 0'dan baslar**.

    Olculen ornek (train 2024-01-01..2025-09-30, test 2025-10-01..2025-12-31)::

        train gun_sayaci: 0 -> 638
        test  gun_sayaci: 0 ->  91      <-- test'in EN GUNCEL satiri, train'in
                                            EN ESKI donemi gibi gorunur

    Bu hata **lokal CV'de GORUNMEZ**: capraz dogrulama yalnizca train uzerinde,
    tek ve tutarli bir origin ile calisir. Yalnizca gercek submission bozulur --
    yani "CV mukemmel, leaderboard cokmus" tablosunun tam kendisi.

    Bu yuzden varsayilan davranis, sessizce yanlis bir sayac uretmek yerine
    **sayaci hic uretmemektir**. Sayaci istiyorsan ortak origin'i acikca ver::

        from gridup.features.temporal import shared_origin
        origin = shared_origin(train, test, time_column="tarih")
    """
    prefix = prefix or time_column
    # shared_origin ile AYNI kural: bicim veriden kanitlanir. Duz
    # ``pd.to_datetime(errors="coerce")`` TR ``gg.aa.yyyy`` kolonunda gun>12
    # olan satirlari sessizce NaT yapiyordu -- yani origin hesabinda gorulen
    # satir burada gorulmuyordu.
    times = _zaman_ayristir(frame, time_column, cagiran="add_calendar_features")

    # Bozuk/eksik tarihler: gercek veride kacinilmaz (bos hucre, '00.00.0000',
    # yanlis bicim). NaT'yi int8'e cast etmek IntCastingNaNError ile coker --
    # yani veri geldigi gun pipeline tek bir kotu hucre yuzunden durur.
    # Cozum: NaT varsa kompakt int yerine float kullan (NaN tasiyabilir ve
    # LightGBM/XGBoost/CatBoost NaN'i YERLI olarak ele alir), ve durumu
    # SESSIZ birakma -- kac satirin bozuk oldugunu yazdir.
    invalid_count = int(times.isna().sum())
    has_invalid = invalid_count > 0
    if has_invalid:
        print(
            f"[add_calendar_features] UYARI: '{time_column}' kolonunda {invalid_count:,} "
            f"gecersiz tarih var (%{invalid_count / max(len(frame), 1) * 100:.2f}). "
            "Takvim kolonlari NaN tasiyabilmesi icin float olarak uretiliyor."
        )

    def _int(series: pd.Series, dtype: str) -> pd.Series:
        """NaT yoksa kompakt int, varsa NaN tasiyabilen float dondurur."""
        return series.astype("float32") if has_invalid else series.astype(dtype)

    new_columns = {
        f"{prefix}_ay": _int(times.dt.month, "int8"),
        f"{prefix}_gun": _int(times.dt.day, "int8"),
        f"{prefix}_haftanin_gunu": _int(times.dt.dayofweek, "int8"),
        f"{prefix}_yilin_gunu": _int(times.dt.dayofyear, "int16"),
        f"{prefix}_hafta": _int(times.dt.isocalendar().week, "int8"),
        f"{prefix}_ceyrek": _int(times.dt.quarter, "int8"),
        f"{prefix}_ayin_ilk_gunu": _int(times.dt.is_month_start, "int8"),
        f"{prefix}_ayin_son_gunu": _int(times.dt.is_month_end, "int8"),
        f"{prefix}_hafta_sonu": _int(times.dt.dayofweek >= 5, "int8"),
        f"{prefix}_mevsim": times.dt.month.map(TURKISH_SEASONS).astype("category"),
    }

    # Saat bilgisi yalnizca gercekten varsa eklenir; hepsi 00:00 ise gurultudur.
    if times.dt.hour.nunique(dropna=True) > 1:
        new_columns[f"{prefix}_saat"] = _int(times.dt.hour, "int8")
        new_columns[f"{prefix}_mesai_saati"] = _int(
            times.dt.hour.between(8, 18) & (times.dt.dayofweek < 5), "int8"
        )
        # Elektrik yukunde puant (pik) saatler: aksam 17-22 arasi tuketim zirvesi.
        new_columns[f"{prefix}_puant_saat"] = _int(times.dt.hour.between(17, 22), "int8")

    if include_year:
        new_columns[f"{prefix}_yil"] = _int(times.dt.year, "int16")

    # Mutlak zaman: agac modellerinin trendi yakalamasi icin tek monoton kolon.
    # ORIGIN VERILMEDIYSE URETILMEZ -- bkz. docstring. Sessizce yanlis bir
    # sayac uretmek, hic uretmemekten cok daha pahaliya patlar.
    if origin is not None:
        elapsed = (times - pd.Timestamp(origin)).dt.days
        new_columns[f"{prefix}_gun_sayaci"] = (
            elapsed.astype("float32") if has_invalid else elapsed.astype("int32")
        )

    return frame.assign(**new_columns)


def add_cyclical_features(
    frame: pd.DataFrame,
    columns: dict[str, int],
    *,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Dongusel degiskenleri sin/cos ciftine cevirir. YENI frame dondurur.

    NEDEN: Saat 23 ile saat 0 zamansal olarak KOMSUDUR ama sayisal olarak 23
    birim uzaktir. Agac modelleri bunu bolme yaparak asabilir, ama dogrusal
    modeller ve sinir aglari asamaz. sin/cos kodlamasi komsulugu korur.

    Args:
        columns: ``{kolon_adi: periyot}`` -- or. ``{"saat": 24, "ay": 12}``.

    >>> import pandas as pd
    >>> df = pd.DataFrame({"saat": [0, 6, 12, 18, 23]})
    >>> out = add_cyclical_features(df, {"saat": 24})
    >>> sorted(c for c in out.columns if c != "saat")
    ['saat_cos', 'saat_sin']
    """
    new_columns = {}
    for column, period in columns.items():
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        values = frame[column].astype(float)
        angle = 2.0 * np.pi * values / period
        new_columns[f"{column}_sin"] = np.sin(angle).astype("float32")
        new_columns[f"{column}_cos"] = np.cos(angle).astype("float32")

    result = frame.assign(**new_columns)
    if drop_original:
        result = result.drop(columns=list(columns))
    return result


def _holiday_code(name: str) -> int:
    """Tatil adini SABIT bir koda cevirir.

    Ad, cakismada ``"A; B"`` bicimindedir ve ``"(saat 13.00'ten)"`` gibi ekler
    tasir. Anahtar kelime aramasi ikisine de dayaniklidir. Cakismada dini
    bayram kazanir -- elektrik tuketimi ve isgucu davranisi acisindan baskin
    olan odur (uc gunluk tatil, seyahat, sanayinin durmasi).
    """
    lowered = tr_lower(name)
    for keyword in _HOLIDAY_PRIORITY:
        if keyword in lowered:
            return HOLIDAY_CODES[keyword]
    return HOLIDAY_NONE


#: Ramazan hicri takvimin 9. ayidir. Sonraki ayin (Sevval) 1'i bayramin
#: birinci gunudur; Ramazan onun bir onceki gunu biter.
_RAMADAN_MONTH = 9
_SHAWWAL_MONTH = 10

#: Ramazanin son on gunu Kadir Gecesi'ni icerir; camii, carsi ve gece
#: hareketliligi belirgin sekilde artar. Ayri bayrak olarak veriyoruz.
LAST_TEN_NIGHTS = 10


def ramadan_calendar(years: Sequence[int]) -> dict[int, tuple[pd.Timestamp, pd.Timestamp]]:
    """Verilen miladi yillar icin Ramazan ayi baslangic/bitis tarihleri.

    Args:
        years: Miladi yillar.

    Returns:
        ``{miladi_yil: (baslangic, bitis)}``. Bir miladi yila iki Ramazan
        dusebilir (nadiren, ~33 yilda bir); o durumda **ilki** dondurulur --
        veri araligimiz icin bu dal hic calismaz.

    Raises:
        ImportError: ``hijridate`` kurulu degilse.

    DOGRULAMA (bu makinede olculdu)
    -------------------------------
    ``frtgnn/turkish-calendar`` veri setinin ``RAMADAN_FLAG`` kolonuyla
    2020-2024 arasi **birebir** ortusuyor -- 2023'un 29 gunluk Ramazani dahil::

        1441 AH: 2020-04-24 -> 2020-05-23  (30 gun)
        1442 AH: 2021-04-13 -> 2021-05-12  (30 gun)
        1443 AH: 2022-04-02 -> 2022-05-01  (30 gun)
        1444 AH: 2023-03-23 -> 2023-04-20  (29 gun)
        1445 AH: 2024-03-11 -> 2024-04-09  (30 gun)
        1446 AH: 2025-03-01 -> 2025-03-29  (29 gun)
        1447 AH: 2026-02-18 -> 2026-03-19  (30 gun)
    """
    try:
        from hijridate import Gregorian, Hijri
    except ImportError as exc:  # pragma: no cover - kurulum yolu
        raise ImportError(
            "Ramazan feature'lari icin hijridate gerekli: pip install hijridate\n"
            "Kaggle'da internet kapaliysa ramadan_calendar ciktisini sabit bir "
            "sozluk olarak notebook'a gom -- yilda iki satir."
        ) from exc

    calendar: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for year in sorted(set(years)):
        # Miladi yilin ortasindan hicri yila gecip komsu hicri yillari tara:
        # Ramazan miladi takvimde yilda ~11 gun kaydigi icin tek bir hicri yil
        # her zaman dogru olmaz.
        pivot = Gregorian(year, 7, 1).to_hijri().year
        for hijri_year in (pivot - 1, pivot, pivot + 1):
            try:
                start = Hijri(hijri_year, _RAMADAN_MONTH, 1).to_gregorian()
                end = Hijri(hijri_year, _SHAWWAL_MONTH, 1).to_gregorian()
            except (ValueError, OverflowError):
                continue
            if start.year != year:
                continue
            calendar[year] = (
                pd.Timestamp(start.year, start.month, start.day),
                pd.Timestamp(end.year, end.month, end.day) - pd.Timedelta(days=1),
            )
            break
    return calendar


def add_ramadan_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    prefix: str = "ramazan",
) -> pd.DataFrame:
    """Ramazan ayi feature'lari ekler. YENI frame dondurur.

    NEDEN AYRI BIR KOLON SART
    -------------------------
    Ramazan **ayi**, Ramazan **Bayrami**ndan farklidir: bayram 3 gundur,
    Ramazan 29-30 gundur ve tuketim profilini butunuyle degistirir:

      * **Sahur** (~03:00-05:00): normalde olu olan saatte ulke capinda
        es zamanli pisirme ve aydinlatma tepesi.
      * **Iftar** (gun batimi): tum hanelerde ayni dakikada baslayan pisirme.
        Saati Ramazan boyunca ~30 dakika kayar -- ``solar`` modulundeki
        ``gun_uzunlugu_saat`` ile birlikte kullanildiginda cok gucludur.
      * **Teravih ve gece hareketliligi**: aksam yuku uzar.
      * **Gunduz sanayi/ticaret** dususu.

    Ve kritik olan su: Ramazan hicri takvime gore **her yil ~11 gun geriye
    kayar**. 2020'de Nisan'da, 2026'da Subat'ta. Bu yuzden ``ay``,
    ``dayofyear`` veya sinus/kosinus mevsimsellik kolonlari onu **hicbir
    sekilde ogrenemez** -- agac modeli "Nisan" ile "Ramazan"i baglar, ertesi
    yil o bag yanlis olur. Ayri kolon olmadan bu sinyal tamamen kaybolur.

    Args:
        frame: Girdi frame'i (degistirilmez).
        time_column: Zaman kolonu adi.
        prefix: Uretilen kolonlarin oneki.

    Returns:
        Su kolonlar eklenmis yeni frame:
          * ``{prefix}_ayi`` -- Ramazan icinde mi (0/1)
          * ``{prefix}_gunu`` -- ayin kacinci gunu (Ramazan disinda 0)
          * ``{prefix}_ilerleme`` -- 0..1 arasi konum (Ramazan disinda 0.0)
          * ``{prefix}_son_on_gun`` -- Kadir Gecesi donemi (0/1)
          * ``{prefix}_bayrama_kalan`` -- bayramin ilk gunune kac gun kaldigi
            (Ramazan disinda ``MISSING_HOLIDAY_DISTANCE``)

    Raises:
        KeyError: ``time_column`` frame'de yoksa.
        ImportError: ``hijridate`` kurulu degilse.
    """
    if time_column not in frame.columns:
        raise KeyError(f"Zaman kolonu '{time_column}' frame'de yok.")

    result = frame.copy()
    times = pd.to_datetime(result[time_column], errors="coerce").dt.normalize()

    valid = times.dropna()
    if valid.empty:
        result[f"{prefix}_ayi"] = 0
        result[f"{prefix}_gunu"] = 0
        result[f"{prefix}_ilerleme"] = 0.0
        result[f"{prefix}_son_on_gun"] = 0
        result[f"{prefix}_bayrama_kalan"] = MISSING_HOLIDAY_DISTANCE
        return result

    years = range(int(valid.dt.year.min()) - 1, int(valid.dt.year.max()) + 2)
    calendar = ramadan_calendar(list(years))

    day_number = np.zeros(len(result), dtype=np.int16)
    length = np.zeros(len(result), dtype=np.int16)
    days_to_eid = np.full(len(result), MISSING_HOLIDAY_DISTANCE, dtype=np.int16)

    values = times.to_numpy()
    for start, end in calendar.values():
        inside = (values >= start.to_datetime64()) & (values <= end.to_datetime64())
        if not inside.any():
            continue
        offset = (times[inside] - start).dt.days.to_numpy(dtype=np.int16)
        day_number[inside] = offset + 1
        length[inside] = int((end - start).days) + 1
        # Bayramin 1. gunu Ramazanin bitiminden bir sonraki gundur.
        days_to_eid[inside] = ((end - times[inside]).dt.days + 1).to_numpy(dtype=np.int16)

    in_ramadan = day_number > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        progress = np.where(in_ramadan, (day_number - 1) / np.maximum(length - 1, 1), 0.0)

    result[f"{prefix}_ayi"] = in_ramadan.astype(np.int8)
    result[f"{prefix}_gunu"] = day_number
    result[f"{prefix}_ilerleme"] = progress.astype(np.float32)
    result[f"{prefix}_son_on_gun"] = (in_ramadan & (days_to_eid <= LAST_TEN_NIGHTS)).astype(np.int8)
    result[f"{prefix}_bayrama_kalan"] = days_to_eid
    return result


def add_turkish_holiday_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    prefix: str = "tatil",
    window_days: int = 3,
    include_half_days: bool = True,
    administrative_leave: dict[str, float] | None = None,
) -> pd.DataFrame:
    """TR resmi tatil feature'lari ekler. YENI frame dondurur.

    NEDEN ONEMLI: Elektrik tuketimi tatillerde ciddi degisir -- sanayi durur,
    konut tuketimi artar, tatil bolgelerinde (Mugla, Aydin, Izmir sahili) nufus
    patlar. Dini bayramlar HICRI takvime gore her yil ~11 gun KAYAR, bu yuzden
    ``ay + gun`` kolonlari onlari YAKALAYAMAZ. Ayri bir kolon sart.

    2023 GDZ Elektrik Datathon birincisi tatil bayraklarini KATEGORIK feature
    olarak modele verdi (``PUBLIC_HOLIDAY_FLAG``, ``RELIGIOUS_DAY_FLAG_SK``,
    ``NATIONAL_DAY_FLAG_SK``, ``WEEKEND_FLAG``, ``RAMADAN_FLAG``) -- yani bu
    aile bu problemde kanitlanmis sekilde gucludur.

    DOGRULAMA (bu makinede olculdu, 2020-2024 kesisimi)
    ---------------------------------------------------
    Birincinin kullandigi Kaggle takvim veri seti (``frtgnn/turkish-calendar``)
    ile ``holidays`` kutuphanesi karsilastirildi. Sonuc: kutuphanenin hafta ici
    resmi tatil listesi Kaggle setinin **tam ust kumesi** (60'a 17; Kaggle'da
    olup kutuphanede olmayan TEK gun yok). Kaggle setinin
    ``PUBLIC_HOLIDAY_FLAG``i butun DINI bayramlari kaciriyor -- birinci onlari
    ayri ``RELIGIOUS_DAY_FLAG_SK`` kolonundan aldigi icin sorun yasamamis.
    Bizim kaynagimiz daha iyi; degistirmeye gerek yok.

    Args:
        window_days: Bayram oncesi/sonrasi kopru gunlerini yakalamak icin
            tatile olan gun mesafesi de uretilir.
        include_half_days: **Varsayilan True ve oyle kalmali.**
            ``holidays.country_holidays("TR")`` varsayilan cagrisi ARIFE
            gunlerini ATLAR. Bu makinede olculdu: 2026 icin varsayilan 14 gun
            dondururken ``categories=("public", "half_day")`` 17 gun donduruyor.
            Kacan uc gun: Ramazan, Kurban ve Cumhuriyet arifeleri (13.00'ten).
            Arifede ogleden sonra isgucu sahayi terk eder, kazi ve planli bakim
            durur -- yani plansiz/planli kesinti dagilimi degisir. Yilda uc
            yuksek sinyalli gun, bedava.
        administrative_leave: ``{"YYYY-AA-GG": agirlik}`` -- hukumetin bayram
            cevresinde verdigi ek izin gunleri. ``None`` ise modul icindeki
            ``ADMINISTRATIVE_LEAVE`` kullanilir. Bos sozluk ``{}`` vererek
            kapatabilirsin.

    Uretilen kolonlar:
        ``{prefix}_mi``            tam gun tatil mi
        ``{prefix}_yarim_gun``     arife mi (13.00'ten sonra)
        ``{prefix}_agirligi``      1.0 tam / 0.5 yarim / 0.0 degil
        ``{prefix}_kod``           sabit tatil kimligi (bkz. HOLIDAY_CODES)
        ``{prefix}_cakisma``       ayni gune iki tatil dusmus mu
        ``{prefix}_mesafe``        en yakin tatile gun mesafesi
        ``{prefix}_yakininda``     mesafe <= window_days
        ``{prefix}_veya_haftasonu``
        ``{prefix}_idari_izin``    idari izin agirligi
        ``{prefix}_isgucu_kaybi``  tatil + idari izin birlesik agirligi (0..1)
    """
    try:
        import holidays as holidays_lib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TR tatil feature'lari icin 'holidays' paketi gerekli: pip install holidays"
        ) from exc

    times = pd.to_datetime(frame[time_column], errors="coerce")
    valid_years = times.dt.year.dropna()
    if valid_years.empty:
        return frame.assign(**{f"{prefix}_mi": np.zeros(len(frame), dtype="int8")})

    years = list(range(int(valid_years.min()) - 1, int(valid_years.max()) + 2))
    categories = ("public", "half_day") if include_half_days else ("public",)
    calendar = holidays_lib.TR(years=years, categories=categories, language=HOLIDAY_LANGUAGE)

    # Yarim gunleri ayirt edebilmek icin yalnizca tam gunlerin ayri bir kumesi.
    full_day_calendar = holidays_lib.TR(
        years=years, categories=("public",), language=HOLIDAY_LANGUAGE
    )

    dates = times.dt.date
    in_calendar = dates.map(lambda day: day in calendar if pd.notna(day) else False)
    is_full_day = dates.map(lambda day: day in full_day_calendar if pd.notna(day) else False)
    is_half_day = in_calendar.to_numpy() & ~is_full_day.to_numpy()

    names = dates.map(lambda day: calendar.get(day, "") if pd.notna(day) else "")
    codes = names.map(_holiday_code).astype("int8")
    collision = names.map(lambda value: 1 if ";" in str(value) else 0).astype("int8")

    leave_table = ADMINISTRATIVE_LEAVE if administrative_leave is None else administrative_leave
    leave_lookup = {pd.Timestamp(day).date(): weight for day, weight in leave_table.items()}
    leave_weight = dates.map(
        lambda day: leave_lookup.get(day, 0.0) if pd.notna(day) else 0.0
    ).astype("float32")

    holiday_weight = np.where(is_full_day.to_numpy(), 1.0, np.where(is_half_day, 0.5, 0.0)).astype(
        "float32"
    )

    holiday_dates = np.array(sorted(calendar.keys()), dtype="datetime64[D]")
    is_holiday = is_full_day

    # GECERSIZ TARIHLERI MASKELE. NaT'nin int64 sentinel'i (-9223372036854775808)
    # cikarma sonrasi tasar ve np.abs() bile onu duzeltemez; int16'ya cast
    # edilince bit kirpmasiyla 0'a duser. Sonuc: bozuk tarihli bir satir
    # "mesafe=0, tatil_yakininda=1" alir -- yani TAM BAYRAM GUNUNDE gorunur.
    # Hicbir hata firlamaz. Bu yuzden maskeleme cast'ten ONCE yapilmali.
    valid_mask = times.notna().to_numpy()
    distances = np.full(len(frame), MISSING_HOLIDAY_DISTANCE, dtype="int16")

    if holiday_dates.size and valid_mask.any():
        valid_days = times[valid_mask].dt.normalize().to_numpy(dtype="datetime64[D]")
        # Buyuk veri setlerinde NxM matris bellek yer; benzersiz gunler uzerinden hesapla.
        unique_days, inverse = np.unique(valid_days, return_inverse=True)
        differences = np.abs(
            (unique_days[:, None] - holiday_dates[None, :]).astype("timedelta64[D]").astype("int64")
        )
        nearest = differences.min(axis=1)
        distances[valid_mask] = np.clip(nearest[inverse], 0, MISSING_HOLIDAY_DISTANCE).astype(
            "int16"
        )

    weekend = (times.dt.dayofweek >= 5).fillna(False).to_numpy()

    new_columns = {
        f"{prefix}_mi": is_holiday.astype("int8"),
        f"{prefix}_yarim_gun": is_half_day.astype("int8"),
        f"{prefix}_agirligi": holiday_weight,
        f"{prefix}_kod": codes,
        f"{prefix}_cakisma": collision,
        f"{prefix}_mesafe": distances,
        # Gecersiz tarihli satir "tatile yakin" SAYILMAZ -- maskeye bagli.
        f"{prefix}_yakininda": ((distances <= window_days) & valid_mask).astype("int8"),
        f"{prefix}_veya_haftasonu": (is_holiday.to_numpy() | weekend).astype("int8"),
        f"{prefix}_idari_izin": leave_weight,
        # Birlesik isgucu kaybi: "bugun saha ekibi ne kadar calisiyor?"
        #
        # Tatil ve idari izin TOPLANIR, max ALINMAZ. Sebep: arife gunu ikisi
        # AYNI GUNUN FARKLI YARILARIDIR -- yasal tatil 13.00'te baslar
        # (holiday_weight=0.5), idari izin ise sabahi kapatir
        # (leave_weight=0.5). max() bunu 0.5 gosterir ve gunun tamamen
        # kapali oldugunu KACIRIR; toplam 1.0 verir, dogrusu budur.
        # Hafta sonu ise ayri bir kaynak degil ayni gunun baska bir sebebi
        # oldugu icin onunla max aliriz.
        f"{prefix}_isgucu_kaybi": np.maximum(
            np.minimum(1.0, holiday_weight + leave_weight.to_numpy()),
            weekend.astype("float32"),
        ).astype("float32"),
    }
    return frame.assign(**new_columns)


def _sort_key(series: pd.Series, *, is_time: bool) -> np.ndarray:
    """Bir kolonu ``np.lexsort`` icin guvenli bir sayisal anahtara cevirir.

    IKI SESSIZ HATAYI birden onler:

    1. **Karisik tipli grup kolonu.** ``np.lexsort`` ham object dizisi uzerinde
       ``<`` karsilastirmasi yapar; kolon ``["A", None, "A"]`` gibiyse
       ``TypeError: '<' not supported between 'str' and 'NoneType'`` ile coker.
       Gercek sebeke verisinde ``trafo_id``/``fider_no`` eksik olabilir (yeni
       tesis, kayit hatasi). ``pd.factorize`` NaN'i ``-1`` kodlar ve coker degil.

    2. **Metin olarak saklanmis tarih.** Kolon datetime DEGILSE siralama
       SOZLUKSEL olur: ``"2024-1-10" < "2024-1-2" < "2024-1-9"``. Bu, lag ve
       kayan pencere degerlerinin YANLIS satirlardan gelmesi demektir -- hicbir
       hata firlamaz, yalnizca feature'lar sessizce anlamsizlasir.
    """
    if is_time:
        converted = pd.to_datetime(series, errors="coerce")
        if converted.isna().all() and series.notna().any():
            raise ValueError(
                f"'{series.name}' zaman kolonu olarak ayristirilamadi. "
                "Siralama yapilamazsa lag/kayan pencere degerleri yanlis satirlardan gelir."
            )
        # NaT'ler dizinin SONUNA gitsin: gecersiz zamanli satirlar gecerli
        # olanlarin gecmisine karismamali.
        values = converted.to_numpy(dtype="datetime64[ns]").astype("int64")
        return np.where(converted.isna().to_numpy(), np.iinfo(np.int64).max, values)

    codes, _ = pd.factorize(series, use_na_sentinel=True)
    return codes


def _tek_satir_dogrula(frame: pd.DataFrame, sort_by: Sequence[str], *, time_column: str) -> None:
    """``(grup..., gun)`` bazinda tek satir degilse hata firlatir.

    Bu moduldeki her gecmis-hedef feature'i ``shift(horizon)`` ile SATIR
    kaydirir, GUN degil. Grup basina gunde birden cok satir (olay-duzeyi
    kayit) varsa "horizon satir once" cogu zaman ayni gun ya da birkac gun
    oncesidir ve ufuk duvari SESSIZCE yok olur. Olculdu (2026-08-18
    denetimi): 1 ilce x 40 gun x 3 olay/gun, horizon=7 -> shift7 feature'inin
    en kucuk gun farki 2, satirlarin %94'u <7 gun oncesinden. Ayni guard
    spatial.py'de vardi, buraya tasindi. Once gunluk topla (build_panel).
    """
    tekrarli = int(frame.duplicated(list(sort_by)).sum())
    if tekrarli:
        ornek = frame.loc[frame.duplicated(list(sort_by)), list(sort_by)].head(2)
        raise ValueError(
            f"{tuple(sort_by)} ikilisi {tekrarli} satirda tekrarliyor "
            f"(ornek: {ornek.to_dict('records')}). "
            "shift(horizon) SATIR kaydirir, GUN degil: tekrarli satirlarda ufuk duvari "
            "sessizce delinir (olculdu: 3 satir/gun + horizon=7 -> satirlarin %94'u <7 gun "
            f"oncesinden). Once ('{time_column}' bazinda) gunluk topla: gridup.build_panel(...) "
            "ya da frame.groupby([grup, gun])[hedef].sum().reset_index()."
        )


def _sorted_view(
    frame: pd.DataFrame,
    sort_by: Sequence[str],
    *,
    time_column: str,
    require_unique: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sirali bir kopya ve orijinal siraya donmek icin permutasyon dondurur.

    ``require_unique`` (varsayilan) satir kaydirmali feature'lar icin
    ``sort_by`` bazinda tekillik ister; bkz. ``_tek_satir_dogrula``.
    """
    if require_unique:
        _tek_satir_dogrula(frame, sort_by, time_column=time_column)
    keys = [
        _sort_key(frame[column], is_time=(column == time_column))
        for column in reversed(list(sort_by))
    ]
    order = np.lexsort(keys)
    return frame.iloc[order], order


def _integer_offset(value: int, *, parameter: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{parameter} tam sayi olmali, verilen: {value!r}")
    return int(value)


def _lag_specifications(
    *,
    lags: Sequence[int] | None,
    shifts: Sequence[int] | None,
    horizon: int,
    prefix: str,
) -> list[tuple[int, str]]:
    """Mutlak shift degeri ile cikti adini tek, dogrulanmis sozlesmeye cevir."""
    if lags is not None and shifts is not None:
        raise ValueError("lags ve shifts birlikte verilemez; yalnizca birini secin.")
    if lags is None and shifts is None:
        raise ValueError("Gecikme icin shifts (onerilen) veya legacy lags verilmelidir.")

    if shifts is not None:
        if len(shifts) == 0:
            raise ValueError("shifts bos olamaz.")
        specifications = []
        for shift in shifts:
            absolute_shift = _integer_offset(shift, parameter="shift")
            if absolute_shift < horizon:
                raise ValueError(
                    f"shift ({absolute_shift}) horizon ({horizon}) kadar veya daha buyuk "
                    "olmali; aksi halde tahmin aninda bilinmeyen veri kullanilir."
                )
            specifications.append((absolute_shift, f"{prefix}_shift{absolute_shift}"))
        return specifications

    assert lags is not None
    if len(lags) == 0:
        raise ValueError("lags bos olamaz.")
    warnings.warn(
        "lags= origin-relative semantigi kullanimdan kalkiyor; gercek mutlak "
        "ofsetleri shifts= ile verin.",
        DeprecationWarning,
        stacklevel=3,
    )
    specifications = []
    for lag in lags:
        legacy_lag = _integer_offset(lag, parameter="lag")
        if legacy_lag < 1:
            raise ValueError(f"legacy lag >= 1 olmali, verilen: {legacy_lag}")
        absolute_shift = horizon + legacy_lag - 1
        name = (
            f"{prefix}_lag{legacy_lag}"
            if horizon == 1
            else f"{prefix}_ufuk{horizon}_lag{legacy_lag}"
        )
        specifications.append((absolute_shift, name))
    return specifications


def add_lag_features(
    frame: pd.DataFrame,
    value_column: str,
    lags: Sequence[int] | None = None,
    *,
    shifts: Sequence[int] | None = None,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Gecikmeli (lag) feature'lar ekler. YENI frame dondurur, sirayi korur.

    Zaman serisi problemlerinde en guclu feature ailesi genellikle budur:
    "bir onceki gunun tuketimi", "gecen haftanin ayni gunu".

    Args:
        value_column: Gecikmesi alinacak kolon (genellikle hedef veya bir olcum).
        lags: KULLANIMDAN KALKIYOR. Tahmin origin'ine gore gecikme adimlari;
            gercek ofset ``horizon + lag - 1`` olur. Bir gecis surumu icin eski
            davranis ve kolon adlari korunur. Yeni kod ``shifts`` kullanmalidir.
        shifts: Kaynak seriye uygulanacak GERCEK, mutlak satir ofsetleri.
            Ornegin ``shifts=[31, 62, 93]`` dogrudan ``shift(31)``,
            ``shift(62)``, ``shift(93)`` uretir. Her ofset ``horizon`` kadar
            veya daha buyuk olmalidir.
        time_column: Siralama icin zaman kolonu -- ZORUNLU. Sirasiz veride
            ``shift`` rastgele satir alir ve gelecegi sizdirir.
        group_columns: Varsa her varlik icin ayri gecikme (or. trafo bazinda).
        horizon: **Tahmin ufku** -- tahmin anindan geriye dogru kac adim veri
            YOK. Varsayilan 1 (bir sonraki adimi tahmin ediyorsun, dun elinde).

    Returns:
        Yeni DataFrame (girdi sirasinda).

    ``horizon`` NEDEN VAR -- yarismanin en pahali sessiz hatasi
    ----------------------------------------------------------
    Test kumesi tek bir gun degil, ILERIDEKI BIR BLOK ise (or. bir sonraki ay),
    o blogun 28. gununu tahmin ederken elindeki en taze veri 28 gun eskidir.
    ``shift(1)`` ile hesaplanan bir lag, o gun icin **var olmayan** bir bilgiyi
    kullanir.

    Bu, CV'de GORUNMEZ: capraz dogrulamada her satir icin bir onceki gun
    train icinde mevcuttur. Yalnizca gercek submission bozulur -- model,
    uretimde asla sahip olmayacagi bir sinyale bagimli hale gelmistir.

    Kural: **``horizon``, test bloğunun uzunlugu kadar olmali.** Bir aylik
    blok tahmin ediyorsan ``horizon=30``; lag ``k`` o zaman
    ``shift(horizon + k - 1)`` olarak hesaplanir ve ``lags=[1]`` "tahmin
    anindaki en taze mevcut deger" anlamina gelir.

    Yeni ``shifts`` API'sinde ofset dogrudan ifade edilir: bir aylik blok ve
    en taze kullanilabilir deger icin ``horizon=30, shifts=[30]``. Tahmin
    aninda bilinmeyen veri kullanmamak icin ``shift >= horizon`` zorunludur.
    Eski ``lags`` semantigi bir gecis surumu boyunca korunur ve uyari verir.
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")
    if horizon < 1:
        raise ValueError(f"horizon >= 1 olmali, verilen: {horizon}")

    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys, time_column=time_column)

    source = (
        ordered.groupby(list(group_columns), observed=True)[value_column]
        if group_columns
        else ordered[value_column]
    )

    specifications = _lag_specifications(lags=lags, shifts=shifts, horizon=horizon, prefix=prefix)

    lagged = {}
    for absolute_shift, name in specifications:
        shifted = source.shift(absolute_shift)
        lagged[name] = np.asarray(shifted)

    # Orijinal siraya geri dondur.
    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{name: values[restore] for name, values in lagged.items()})


def add_rolling_features(
    frame: pd.DataFrame,
    value_column: str,
    windows: Sequence[int],
    *,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str] | None = None,
    aggregations: Sequence[str] = ("mean", "std", "min", "max"),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Kayan pencere istatistikleri ekler. YENI frame dondurur.

    KRITIK: Pencere ``shift(horizon)`` sonrasi hesaplanir, yani MEVCUT SATIR
    DAHIL DEGILDIR. Bu, hedef turevli bir kolonda dogrudan hedef sizintisini
    onler. pandas'in varsayilan davranisi mevcut satiri DAHIL EDER -- bu
    fonksiyon onu bilerek degistirir.

    Args:
        windows: Pencere boylari, or. ``[7, 14, 30]``.
        horizon: Tahmin ufku. Test ILERIDEKI bir blok ise, o bloğun uzunlugu
            kadar olmali -- aksi halde pencere, tahmin aninda var olmayan
            veriyi kullanir. Ayrintili gerekce: ``add_lag_features``.
        aggregations: ``mean``, ``std``, ``min``, ``max``, ``median``, ``sum``.
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")
    if horizon < 1:
        raise ValueError(f"horizon >= 1 olmali, verilen: {horizon}")

    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys, time_column=time_column)

    if group_columns:
        grouped = ordered.groupby(list(group_columns), observed=True)[value_column]
        shifted = grouped.shift(horizon)
        # sort=False ZORUNLU. ``_sorted_view`` satirlari pd.factorize ile
        # GORUNUM sirasina gore dizer; buradaki ikinci groupby varsayilan
        # sort=True ile gruplari ALFABETIK sirlar. Iki sira uyusmazsa kayan
        # pencere degerleri BASKA GRUBA yazilir -- hatasiz, ayni satir
        # sayisiyla, tamamen sessizce.
        #
        # OLCULDU: gorunum sirasi [bornova, aliaga] olan bir panelde
        # bornova'nin y=1000..1004 degerleri icin uretilen pencere
        # [nan, 1.0, 1.5, 2.0, 3.0] cikiyordu -- yani ALIAGA'nin degerleri.
        # 96 ilcelik bir panelde bu, tum kayan feature'larin yanlis ilceye
        # yazilmasi demektir.
        rolling_source = shifted.groupby(
            [ordered[column].to_numpy() for column in group_columns], sort=False
        )
    else:
        shifted = ordered[value_column].shift(horizon)
        rolling_source = shifted

    label = "kayan" if horizon == 1 else f"ufuk{horizon}_kayan"
    computed = {}
    for window in windows:
        roller = rolling_source.rolling(window=window, min_periods=1)
        for aggregation in aggregations:
            values = getattr(roller, aggregation)()
            computed[f"{prefix}_{label}{window}_{aggregation}"] = np.asarray(
                values, dtype="float32"
            )

    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{name: values[restore] for name, values in computed.items()})


def add_expanding_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str] | None = None,
    aggregations: Sequence[str] = ("mean", "std"),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Genisleyen pencere (tum gecmis) istatistikleri. YENI frame dondurur.

    Kayan pencerenin aksine tum gecmisi kullanir; bir varligin "genel
    seviyesini" yakalar. Mevcut satir ve tahmin ufku icindeki satirlar
    ``shift(horizon)`` ile DISLANIR.

    ``horizon`` ZORUNLUDUR -- NEDEN
    -------------------------------
    Onceki surum ``shift(1)`` SABIT KODLUYDU ve ufuk parametresi hic yoktu.
    Bu, bir gunden uzun her tahmin ufkunda dogrudan sizintiydi.

    OLCULDU (60 gunluk seri, hedef = 0..59, ufuk 30 gun)::

        genisleyen_max son satirda      = 58.0
        tahmin aninda gorulebilir EN BUYUK = 29.0
        -> 29 GUNLUK SIZINTI

    CV skoru mukemmel gorunur, leaderboard coker. Bu yuzden artik deger
    VERILMEK ZORUNDA: ``embargo``da oldugu gibi, sessiz bir varsayilan
    uretmektense bilincli bir karar istiyoruz.
    """
    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys, time_column=time_column)

    if horizon < 1:
        raise ValueError(
            f"horizon en az 1 olmali, {horizon} verildi. horizon=0 mevcut satirin "
            "KENDI degerini pencereye sokar -- dogrudan hedef sizintisi."
        )

    if group_columns:
        shifted = ordered.groupby(list(group_columns), observed=True)[value_column].shift(horizon)
        # sort=False ZORUNLU -- ayni sebep: bkz. add_rolling_features.
        source = shifted.groupby(
            [ordered[column].to_numpy() for column in group_columns], sort=False
        )
    else:
        source = ordered[value_column].shift(horizon)

    expander = source.expanding(min_periods=1)
    computed = {
        f"{prefix}_genisleyen_{aggregation}": np.asarray(
            getattr(expander, aggregation)(), dtype="float32"
        )
        for aggregation in aggregations
    }

    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{name: values[restore] for name, values in computed.items()})


def add_previous_month_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Son TAM gozlemlenmis takvim ayinin istatistikleri. YENI frame dondurur.

    NEREDEN GELDI (2024 GDZ birincisi Pikachow, final sunumu s.18 + s.26)
    --------------------------------------------------------------------
    Kazanan cozumun feature listesinde uc kalip vardi ve bu fonksiyon ucunu
    de uretir:

      * "Gecen aya ait toplam bildirimsiz kesinti sayisi"
      * "Ilce bazinda son ay kesintili ve kesintisiz gun sayisi"
      * ``bildirimsiz_sum_last_month_same_day_max/skew`` -- feature importance
        listesinde EN USTLERDEYDI (s.26): gecen ayin AYNI GUNUNUN degeri.

    Kayan pencereden (``add_rolling_features``) farki: pencere satirla
    birlikte kaymaz, TAKVIM AYINA sabitlenir. "Subat'in her gunu icin Ocak
    toplami AYNI degerdir" -- model bunu ay-seviyesi bir rejim sinyali olarak
    okur; kayan pencere ise her gun degisir. Ikisi farkli bilgidir.

    SIZINTI DISIPLINI -- "gecen ay" DEGIL "son tam gozlemlenmis ay"
    ---------------------------------------------------------------
    2024'te test TAM BIR AYDI (Subat); tahmin aninda Ocak'in tamami
    gorulmustu, "gecen ay" guvenliydi. Ama test blogu ay SINIRINI ASARSA
    (or. 21 Agustos - 1 Eylul), Eylul satirlari icin "gecen ay" Agustos'tur
    ve Agustos'un 21-31'i TEST ICINDEDIR -- dogrudan sizinti, CV'de gorunmez.

    Bu yuzden kural repodaki lag konvansiyonunun aynisidir: ``d`` gunundeki
    satir icin kullanilabilir en taze gozlem ``d - horizon`` gunudur; feature,
    SONU ``d - horizon``'u gecmeyen son takvim ayindan hesaplanir.
    ``horizon=1`` bunu Pikachow'un "gecen ay"ina indirger. Test blogu ay
    basinda basliyorsa ``horizon=1`` guvenlidir; degilse ufku ver.

    NOT: ``olaysiz_gun`` sayimi yogun panel ister -- once ``build_panel``
    calistir; seyrek olay kaydinda sifir gunler SATIR OLARAK YOKTUR ve
    sayim yaniltici cikar.

    Uretilen kolonlar (``horizon=1`` icin; degilse ``ufuk{h}_`` on eki):
        ``{prefix}_sontamay_toplam``      ay toplami
        ``{prefix}_sontamay_ortalama``    gunluk ortalama
        ``{prefix}_sontamay_max``         ayin tepe gunu
        ``{prefix}_sontamay_olayli_gun``  deger > 0 olan gun sayisi
        ``{prefix}_sontamay_olaysiz_gun`` deger == 0 olan gun sayisi
        ``{prefix}_sontamay_ayni_gun``    ayni ay-gununun degeri (31 -> Subat
                                          gibi olmayan gunlerde NaN)
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")
    if horizon < 1:
        raise ValueError(f"horizon >= 1 olmali, verilen: {horizon}")

    prefix = prefix or value_column
    label = "sontamay" if horizon == 1 else f"ufuk{horizon}_sontamay"
    groups = list(group_columns or [])

    times = parse_time_series(frame[time_column], strict=False)
    degerler = pd.to_numeric(frame[value_column], errors="coerce")

    # Ay-seviyesi ozet tablosu: her (grup, ay) icin bir satir.
    kaynak = pd.DataFrame(
        {
            "_ay": times.dt.to_period("M"),
            "_gun_no": times.dt.day,
            "_deger": degerler,
            "_olayli": (degerler > 0).astype("float64"),
            "_olaysiz": (degerler == 0).astype("float64"),
        }
    )
    for column in groups:
        kaynak[column] = frame[column].to_numpy()

    aylik = (
        kaynak.dropna(subset=["_ay"])
        .groupby([*groups, "_ay"], observed=True)
        .agg(
            toplam=("_deger", "sum"),
            ortalama=("_deger", "mean"),
            max=("_deger", "max"),
            olayli_gun=("_olayli", "sum"),
            olaysiz_gun=("_olaysiz", "sum"),
        )
        .reset_index()
    )

    # Ayni ay-gunu tablosu: (grup, ay, gun_no) -> gunun toplami. Panel gunde
    # tek satir garanti eder; yine de sum kullaniyoruz ki olasi cift kayit
    # sessizce ilkini secmek yerine gunluk toplama katlansin.
    gunluk = (
        kaynak.dropna(subset=["_ay"])
        .groupby([*groups, "_ay", "_gun_no"], observed=True)["_deger"]
        .sum(min_count=1)
        .rename("ayni_gun")
        .reset_index()
    )

    # Satir basina kullanilabilir ay: sonu (d - horizon)'u gecmeyen SON ay.
    # (d - horizon + 1 gun)'un ayindan bir onceki ay tam olarak budur:
    # d - horizon ayin son gunuyse o ayin kendisi, degilse bir oncesi.
    cutoff = times - pd.to_timedelta(horizon - 1, unit="D")
    kullanilabilir_ay = cutoff.dt.to_period("M") - 1

    anahtar = pd.DataFrame({"_ay": kullanilabilir_ay, "_gun_no": times.dt.day})
    for column in groups:
        anahtar[column] = frame[column].to_numpy()

    ozet = anahtar.merge(aylik, on=[*groups, "_ay"], how="left")
    ozet = ozet.merge(gunluk, on=[*groups, "_ay", "_gun_no"], how="left")

    yeni = {
        f"{prefix}_{label}_toplam": ozet["toplam"].to_numpy(dtype="float32"),
        f"{prefix}_{label}_ortalama": ozet["ortalama"].to_numpy(dtype="float32"),
        f"{prefix}_{label}_max": ozet["max"].to_numpy(dtype="float32"),
        f"{prefix}_{label}_olayli_gun": ozet["olayli_gun"].to_numpy(dtype="float32"),
        f"{prefix}_{label}_olaysiz_gun": ozet["olaysiz_gun"].to_numpy(dtype="float32"),
        f"{prefix}_{label}_ayni_gun": ozet["ayni_gun"].to_numpy(dtype="float32"),
    }
    return frame.assign(**yeni)


def add_mass_event_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str],
    threshold: float = 0.5,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Toplu-olay (bolge geneli) gun feature'lari -- YALNIZCA ufuk-kaydirilmis.

    NEREDEN GELDI (M5 out-of-stock analogu, docs/09 bolum 2.1)
    ----------------------------------------------------------
    M5'te "magazalarin cogunda ayni gun satis sifir" gunleri ayri isaretlemek
    hurdle modellerin p(0) asamasini keskinlestirdi. Bizim analogumuz: bir
    firtina gununde ilcelerin buyuk kismi AYNI GUN kesintilidir. Gun bazinda
    "kesintili grup payi" bu bolgesel rejimi tek sayida ozetler.

    SIZINTI DISIPLINI -- AYNI GUNUN PAYI ASLA YAYINLANMAZ
    -----------------------------------------------------
    Gunun payi HEDEFTEN turetilir: satirin kendi hedefi o gunun payina
    katilir. Ayni-gun payi feature olursa dogrudan hedef sizintisidir ve
    CV'de gorunmez. Bu yuzden ``horizon >= 1`` ZORUNLUDUR ve yalnizca
    kaydirilmis degerler uretilir: ``d`` gunundeki satirin gordugu en taze
    pay ``d - horizon`` gununun payidir -- ``add_lag_features``taki lag1
    konvansiyonunun aynisi (lag1 = shift(horizon)).

    Kaydirma TAKVIM gunu uzerinden yapilir (satir uzerinden degil): pay gunun
    ozelligidir, gun eksikse feature NaN kalir -- yanlis gunden okunmaz.

    Args:
        frame: Girdi (degistirilmez).
        value_column: Hedef benzeri kolon; ``> 0`` olan gun "olayli" sayilir.
        time_column: Zaman kolonu.
        horizon: Tahmin ufku -- test blogu kadar (bkz. ``add_lag_features``).
        group_columns: Payin paydasini olusturan varlik kolonlari (or. ilce).
        threshold: Gunun "toplu olay" bayragi icin pay esigi (0..1].
        prefix: Kolon oneki. Varsayilan: ``value_column``.

    Returns:
        Su kolonlar eklenmis YENI frame (``horizon != 1`` ise ``ufuk{h}_`` ile):
          * ``{prefix}_topluolay_pay_lag1``    ``d - horizon`` gununun payi
          * ``{prefix}_topluolay_pay_kayan7``  payin 7 gunluk ortalamasi
            (sonu ``d - horizon`` olan pencere)
          * ``{prefix}_topluolay_bayrak_lag1`` pay >= threshold (0/1; pay
            bilinmiyorsa NaN -- "olay yok" demek degildir)
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")
    if horizon < 1:
        raise ValueError(
            f"horizon >= 1 olmali, verilen: {horizon}. horizon=0 AYNI GUNUN "
            "payini yayinlar -- satirin kendi hedefi paya katildigi icin "
            "dogrudan hedef sizintisidir."
        )
    if not group_columns:
        raise ValueError("group_columns bos olamaz: pay, gruplarin orani uzerinden tanimli.")
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold (0, 1] araliginda olmali, verilen: {threshold}")

    prefix = prefix or value_column
    groups = list(group_columns)
    times = parse_time_series(frame[time_column], strict=False).dt.normalize()
    degerler = pd.to_numeric(frame[value_column], errors="coerce")

    kaynak = pd.DataFrame({"_gun": times, "_olayli": (degerler > 0).astype("float64")})
    for column in groups:
        kaynak[column] = frame[column].to_numpy()

    # Once (gun, grup) -> "o gun o grupta olay var mi", sonra gun -> olayli
    # grup payi. Cift kayitli/gevrek panellerde ayni grubun ayni gununu iki
    # kez saymamak icin iki adimli.
    gecerli = kaynak.dropna(subset=["_gun"])
    if gecerli.empty:
        bos = np.full(len(frame), np.nan, dtype="float32")
        etiket = "" if horizon == 1 else f"ufuk{horizon}_"
        return frame.assign(
            **{
                f"{prefix}_{etiket}topluolay_pay_lag1": bos,
                f"{prefix}_{etiket}topluolay_pay_kayan7": bos,
                f"{prefix}_{etiket}topluolay_bayrak_lag1": bos,
            }
        )

    grup_gun = gecerli.groupby(["_gun", *groups], observed=True)["_olayli"].max()
    pay = grup_gun.groupby(level=0).mean()

    # Tam takvim araligina yay: eksik gunler NaN kalir ve kaydirma yanlis
    # gunden OKUMAZ (satir bazli shift eksik gunu sessizce atlardi).
    tum_gunler = pd.date_range(pay.index.min(), pay.index.max(), freq="D")
    gunluk = pay.reindex(tum_gunler)
    kaydirilmis = gunluk.shift(horizon)
    kayan7 = kaydirilmis.rolling(7, min_periods=1).mean()
    # Bayrakta NaN korunur: "pay bilinmiyor" ile "toplu olay yok" ayni sey degil.
    bayrak = pd.Series(
        np.where(kaydirilmis.isna(), np.nan, (kaydirilmis >= threshold).astype("float64")),
        index=tum_gunler,
    )

    etiket = "" if horizon == 1 else f"ufuk{horizon}_"
    return frame.assign(
        **{
            f"{prefix}_{etiket}topluolay_pay_lag1": times.map(kaydirilmis).astype("float32"),
            f"{prefix}_{etiket}topluolay_pay_kayan7": times.map(kayan7).astype("float32"),
            f"{prefix}_{etiket}topluolay_bayrak_lag1": times.map(bayrak).astype("float32"),
        }
    )


def _grup_kimlikleri(ordered: pd.DataFrame, group_columns: Sequence[str] | None) -> np.ndarray:
    """Sirali gorunumde satir basina grup kodu (grup yoksa hepsi 0).

    ``_sorted_view`` gruplari bitisik dizer; kodlar gorunum sirasinda artar.
    """
    if not group_columns:
        return np.zeros(len(ordered), dtype=np.int64)
    anahtar = pd.MultiIndex.from_frame(ordered[list(group_columns)])
    return pd.factorize(anahtar)[0].astype(np.int64)


def _grup_dilimleri(kimlikler: np.ndarray) -> list[tuple[int, int]]:
    """Bitisik grup kodlarindan ``(baslangic, bitis)`` dilimleri cikarir."""
    if len(kimlikler) == 0:
        return []
    sinirlar = np.flatnonzero(np.diff(kimlikler) != 0) + 1
    kenarlar = [0, *sinirlar.tolist(), len(kimlikler)]
    return [(kenarlar[i], kenarlar[i + 1]) for i in range(len(kenarlar) - 1)]


def add_event_decay_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str] | None = None,
    half_lives: Sequence[float] = (3.0, 14.0),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Hawkes-esinli ustel-bozunumlu gecmis toplami. YENI frame dondurur.

    NEREDEN GELDI (docs/10 bolum 3): art arda ariza KUMELENIR -- dun ariza
    yasayan sebeke bugun de kirilgandir (agac dallari sarkmis, ekip sahada,
    gecici onarim). Tam Hawkes sureci kanitsiz; ucuz feature hali su::

        bozunum[d] = toplam_{s <= d - horizon}  2^(-((d - horizon) - s) / yari_omur) * x[s]

    Yani gecmisteki her gozlem, yari omru ``half_life`` gun olan bir agirlikla
    toplama katilir. Iki seri uretilir: olay GOSTERGESI (deger > 0) ve degerin
    kendisi -- gosterge "kac gundur ariza kumesi icindeyiz"i, deger "ne kadar
    siddetli"yi tasir.

    SIZINTI DISIPLINI -- repodaki lag konvansiyonunun AYNISI
    --------------------------------------------------------
    ``d`` gunundeki satirin gordugu en taze gozlem ``d - horizon`` gunudur
    (``add_lag_features``taki lag1 = shift(horizon)). Bozunum once HAM seri
    uzerinde ozyinelemeli hesaplanir (``D[j] = x[j] + alpha * D[j-1]``),
    sonra ``horizon`` satir kaydirilarak yayinlanir -- boylece ``d``
    satirinin degeri yalnizca ``<= d - horizon`` gozlemlerinden gelir.
    Gecmis hic yokken (grubun ilk ``horizon`` satiri) deger NaN'dir.

    IZGARA VARSAYIMI: ``build_panel`` sonrasi DUZENLI GUNLUK izgara varsayilir
    (grup basina gunde tek satir, eksik gun yok) -- ``shift(horizon)`` ancak o
    zaman "horizon gun once" demektir. ``alpha`` yine de ardil satirlar arasi
    GERCEK takvim-gun farkiyla hesaplanir; izgarada delik varsa bozunum orani
    dogru kalir ama kaydirma ``horizon`` satir sayar, gun degil.

    Args:
        frame: Girdi (degistirilmez).
        value_column: Hedef benzeri kolon; ``> 0`` olan gun "olayli" sayilir.
        time_column: Zaman kolonu.
        horizon: Tahmin ufku -- test blogu kadar (bkz. ``add_lag_features``).
        group_columns: Varsa bozunum her grup icinde AYRI hesaplanir.
        half_lives: Yari omurler (gun). 3 = "gecen haftanin izleri",
            14 = "bu ayin rejimi".
        prefix: Kolon oneki. Varsayilan: ``value_column``.

    Returns:
        Su kolonlar eklenmis YENI frame (``horizon != 1`` ise ``ufuk{h}_``):
          * ``{prefix}_bozunum{yari_omur:g}g_olay``  gosterge bozunumu
          * ``{prefix}_bozunum{yari_omur:g}g_deger`` deger bozunumu

    Raises:
        KeyError: ``value_column`` frame'de yoksa.
        ValueError: ``horizon < 1``; ``half_lives`` bos veya pozitif degilse.
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")
    if horizon < 1:
        raise ValueError(
            f"horizon >= 1 olmali, verilen: {horizon}. horizon=0 AYNI GUNUN "
            "degerini toplama sokar -- dogrudan hedef sizintisi."
        )
    if not half_lives:
        raise ValueError("En az bir yari omur ver, or. half_lives=(3.0, 14.0).")
    kotu = [h for h in half_lives if not h > 0]
    if kotu:
        raise ValueError(f"Yari omurler pozitif olmali, verilen: {kotu}")

    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys, time_column=time_column)

    times = parse_time_series(ordered[time_column], strict=False)
    gecerli = times.notna().to_numpy()
    gunler = times.to_numpy(dtype="datetime64[D]").astype("int64")
    degerler = pd.to_numeric(ordered[value_column], errors="coerce").to_numpy(dtype="float64")
    olaylar = (degerler > 0).astype("float64")  # NaN > 0 -> False -> 0
    degerler = np.nan_to_num(degerler, nan=0.0)

    kimlikler = _grup_kimlikleri(ordered, group_columns)
    etiket = "" if horizon == 1 else f"ufuk{horizon}_"
    ciktilar = {
        f"{prefix}_{etiket}bozunum{yari_omur:g}g_{tur}": np.full(len(ordered), np.nan)
        for yari_omur in half_lives
        for tur in ("olay", "deger")
    }

    for bas, son in _grup_dilimleri(kimlikler):
        # NaT satirlar _sort_key ile grubun SONUNA gider -- gecerli on-ek.
        n_gecerli = int(gecerli[bas:son].sum())
        if n_gecerli <= horizon:
            continue  # gorulebilir gecmis yok; NaN kalir
        g_gun = gunler[bas : bas + n_gecerli]
        for yari_omur in half_lives:
            for tur, seri in (("olay", olaylar), ("deger", degerler)):
                x = seri[bas : bas + n_gecerli]
                bozunum = np.empty(n_gecerli, dtype="float64")
                bozunum[0] = x[0]
                for j in range(1, n_gecerli):
                    fark = float(g_gun[j] - g_gun[j - 1])
                    alpha = 2.0 ** (-fark / yari_omur)
                    bozunum[j] = x[j] + alpha * bozunum[j - 1]
                kolon = f"{prefix}_{etiket}bozunum{yari_omur:g}g_{tur}"
                ciktilar[kolon][bas + horizon : bas + n_gecerli] = bozunum[:-horizon]

    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    return frame.assign(**{ad: dizi[restore].astype("float32") for ad, dizi in ciktilar.items()})


def add_days_since_event_features(
    frame: pd.DataFrame,
    value_column: str,
    *,
    time_column: str,
    horizon: int,
    group_columns: Sequence[str] | None = None,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Son olaydan (deger > 0) gecen gun sayisi. YENI frame dondurur.

    NEREDEN GELDI (docs/10 bolum 1): Sivas tezi (2025, No 972166) gunluk
    kesinti sayisi tahmininde 3. en onemli feature'i "son bakimdan gecen gun
    sayisi" olctu. Bakim kaydimiz yok; en yakin vekili "son ariza olayindan
    gecen gun"dur -- uzun sessizlik ya saglam sebeke ya birikmis bakim acigi
    demektir, ikisi de duz lag'lerin tasimadigi bir sinyaldir.

    SIZINTI DISIPLINI: son olay, ufuk-kaydirilmis seride aranir -- ``d``
    gunundeki satir yalnizca ``<= d - horizon`` gunlerinin olaylarini gorur
    (``add_lag_features`` konvansiyonu). Henuz hic olay gorunmuyorsa deger
    NaN'dir ve ``_hic_olay_yok`` bayragi 1'dir: "olay yok" ile "0 gun once
    olay oldu" ayni sey degildir ve ayni sayiya dusurulmez.

    IZGARA VARSAYIMI: ``add_event_decay_features`` ile ayni -- ``build_panel``
    sonrasi duzenli gunluk izgara; kaydirma satir sayar, mesafe takvim gunuyle
    olculur.

    Args:
        frame: Girdi (degistirilmez).
        value_column: Hedef benzeri kolon; ``> 0`` olan gun "olayli" sayilir.
        time_column: Zaman kolonu.
        horizon: Tahmin ufku -- test blogu kadar (bkz. ``add_lag_features``).
        group_columns: Varsa sayim her grup icinde AYRI tutulur.
        prefix: Kolon oneki. Varsayilan: ``value_column``.

    Returns:
        Su kolonlar eklenmis YENI frame (``horizon != 1`` ise ``ufuk{h}_``):
          * ``{prefix}_son_olaydan_gun``  satirin gunu ile gorulebilir son
            olay gunu arasindaki takvim-gun farki (float32; olay yoksa NaN)
          * ``{prefix}_hic_olay_yok``     gorulebilir gecmiste hic olay yoksa
            1, varsa 0 (int8)

    Raises:
        KeyError: ``value_column`` frame'de yoksa.
        ValueError: ``horizon < 1`` ise.
    """
    if value_column not in frame.columns:
        raise KeyError(f"Kolon '{value_column}' frame icinde yok.")
    if horizon < 1:
        raise ValueError(
            f"horizon >= 1 olmali, verilen: {horizon}. horizon=0 satirin KENDI "
            "gununun olayini gorunur kilar -- dogrudan hedef sizintisi."
        )

    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys, time_column=time_column)

    times = parse_time_series(ordered[time_column], strict=False)
    gecerli = times.notna().to_numpy()
    gunler = times.to_numpy(dtype="datetime64[D]").astype("int64")
    degerler = pd.to_numeric(ordered[value_column], errors="coerce")
    olayli = (degerler > 0).to_numpy()

    kimlikler = _grup_kimlikleri(ordered, group_columns)
    gecen_gun = np.full(len(ordered), np.nan)
    hic_yok = np.ones(len(ordered), dtype="int64")

    #: "Henuz olay yok" sentineli -- gecerli hicbir takvim gununden buyuk
    #: olamayacak kadar kucuk.
    SENTINEL = np.iinfo(np.int64).min // 4

    for bas, son in _grup_dilimleri(kimlikler):
        n_gecerli = int(gecerli[bas:son].sum())
        if n_gecerli <= horizon:
            continue
        g_gun = gunler[bas : bas + n_gecerli]
        olay_gunu = np.where(olayli[bas : bas + n_gecerli], g_gun, SENTINEL)
        son_olay = np.maximum.accumulate(olay_gunu)
        # Ufuk kaydirmasi: i satiri, i - horizon'a kadarki son olayi gorur.
        gorunen = son_olay[:-horizon]
        hedef_gunler = g_gun[horizon:]
        var = gorunen > SENTINEL
        dilim = np.full(n_gecerli - horizon, np.nan)
        dilim[var] = (hedef_gunler[var] - gorunen[var]).astype("float64")
        gecen_gun[bas + horizon : bas + n_gecerli] = dilim
        hic_yok[bas + horizon : bas + n_gecerli] = (~var).astype("int64")

    restore = np.empty_like(order)
    restore[order] = np.arange(len(order))
    etiket = "" if horizon == 1 else f"ufuk{horizon}_"
    return frame.assign(
        **{
            f"{prefix}_{etiket}son_olaydan_gun": gecen_gun[restore].astype("float32"),
            f"{prefix}_{etiket}hic_olay_yok": hic_yok[restore].astype("int8"),
        }
    )


def add_upcoming_holiday_features(
    frame: pd.DataFrame,
    time_column: str,
    *,
    windows: Sequence[int] = (3, 7, 15),
    prefix: str = "tatil",
    include_half_days: bool = True,
) -> pd.DataFrame:
    """ILERI bakisli tatil feature'lari: "onumuzdeki N gun icinde bayram var mi".

    NEREDEN GELDI
    -------------
    2024 GDZ birincisi Pikachow (final sunumu s.18): "ileriki 3-7-15 gun
    icerisinde bayram olma durumu". Rohlik Orders 8.'si de "days-to-next-
    holiday" kullandi. Mekanizma gercek: bayram ONCESI davranis degisir --
    seyahat baslar, sanayi vardiya kapatir, planli kesintiler bayram oncesine
    cekilir. ``add_turkish_holiday_features``in ``{prefix}_mesafe``si EN YAKIN
    tatile mutlak mesafedir; "dun bayramdi" ile "yarin bayram" ayni gorunur.
    Bu fonksiyon yalnizca ILERIYE bakar ve o ayrimi modele verir.

    SIZINTI YOK -- NEDEN GUVENLI
    ----------------------------
    Takvim BILINEN-GELECEK kovaryattir: 2026'nin butun resmi tatilleri bugun
    bellidir. Ileri bakmak burada sizinti degildir; hedef turevli kolonlarda
    olurdu.

    Uretilen kolonlar:
        ``{prefix}_sonraki_mesafe``      bir SONRAKI tatile gun (bugun haric;
                                         tatil bulunamazsa MISSING_HOLIDAY_DISTANCE)
        ``{prefix}_onumuzdeki_{w}g``     sonraki_mesafe <= w (her pencere icin)
    """
    try:
        import holidays as holidays_lib
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TR tatil feature'lari icin 'holidays' paketi gerekli: pip install holidays"
        ) from exc

    if not windows:
        raise ValueError("En az bir pencere ver, or. windows=(3, 7, 15).")

    times = parse_time_series(frame[time_column], strict=False)
    valid_years = times.dt.year.dropna()
    if valid_years.empty:
        sentinel = np.full(len(frame), MISSING_HOLIDAY_DISTANCE, dtype="int16")
        yeni = {f"{prefix}_sonraki_mesafe": sentinel}
        for window in windows:
            yeni[f"{prefix}_onumuzdeki_{window}g"] = np.zeros(len(frame), dtype="int8")
        return frame.assign(**yeni)

    # +2 yil ileri: serinin son gunlerinde bile "sonraki tatil" bulunsun.
    years = list(range(int(valid_years.min()) - 1, int(valid_years.max()) + 2))
    categories = ("public", "half_day") if include_half_days else ("public",)
    calendar = holidays_lib.TR(years=years, categories=categories, language=HOLIDAY_LANGUAGE)
    holiday_dates = np.array(sorted(calendar.keys()), dtype="datetime64[D]")

    # NaT maskesi CAST'TEN ONCE -- gerekce add_turkish_holiday_features'taki
    # sentinel tasmasi notuyla ayni.
    valid_mask = times.notna().to_numpy()
    distances = np.full(len(frame), MISSING_HOLIDAY_DISTANCE, dtype="int16")

    if holiday_dates.size and valid_mask.any():
        valid_days = times[valid_mask].dt.normalize().to_numpy(dtype="datetime64[D]")
        unique_days, inverse = np.unique(valid_days, return_inverse=True)
        # side="right": bugun tatilse bile bir SONRAKI tatili isaret et --
        # "bugun bayram" bilgisini {prefix}_mi zaten tasiyor.
        # asarray(): np.searchsorted dizi girdiyle dizi dondurur ama eski
        # numpy stub'lari skaler asiri yuklemesini seciyor ve konum[maske]
        # indekslemesi tip hatasi veriyor. Davranis degismez.
        konum = np.asarray(np.searchsorted(holiday_dates, unique_days, side="right"))
        ileri = np.full(len(unique_days), MISSING_HOLIDAY_DISTANCE, dtype="int64")
        bulunan = konum < len(holiday_dates)
        ileri[bulunan] = (
            (holiday_dates[konum[bulunan]] - unique_days[bulunan])
            .astype("timedelta64[D]")
            .astype("int64")
        )
        distances[valid_mask] = np.clip(ileri[inverse], 0, MISSING_HOLIDAY_DISTANCE).astype("int16")

    yeni = {f"{prefix}_sonraki_mesafe": distances}
    for window in windows:
        yeni[f"{prefix}_onumuzdeki_{window}g"] = ((distances <= window) & valid_mask).astype("int8")
    return frame.assign(**yeni)
