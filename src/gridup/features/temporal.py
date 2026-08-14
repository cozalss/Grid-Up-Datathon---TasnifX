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

from collections.abc import Sequence

import numpy as np
import pandas as pd

from ..turkish import tr_lower

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
HOLIDAY_NONE = 0
HOLIDAY_CODES: dict[str, int] = {
    "ramazan": 1,
    "kurban": 2,
    "cumhuriyet": 3,
    "egemenlik": 4,      # 23 Nisan Ulusal Egemenlik ve Cocuk Bayrami
    "genclik": 5,        # 19 Mayis Ataturk'u Anma, Genclik ve Spor Bayrami
    "zafer": 6,          # 30 Agustos
    "emek": 7,           # 1 Mayis
    "demokrasi": 8,      # 15 Temmuz
    "yilbasi": 9,
}

# Cakismada hangi tatil kazanir. Dini bayramlar once: elektrik tuketimi ve
# isgucu davranisi acisindan baskin olan onlardir (uc gunluk tatil, seyahat,
# sanayinin durmasi). Milli bayram tek gundur ve etkisi daha zayiftir.
_HOLIDAY_PRIORITY = ("ramazan", "kurban", "cumhuriyet", "egemenlik",
                     "genclik", "zafer", "demokrasi", "emek", "yilbasi")

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
    "2019-06-03": 0.5, "2019-06-07": 1.0,
    "2021-05-10": 1.0, "2021-05-11": 1.0, "2021-05-12": 0.5,
    "2022-07-13": 1.0, "2022-07-14": 1.0,
    # 2023 Kurban arifesi: 27 Haziran yasal olarak zaten yarim gun ->
    # ek izin de yarim gundur. Onceki 1.0 degeri konvansiyonu bozuyordu.
    "2023-06-26": 1.0, "2023-06-27": 0.5,
    "2024-04-08": 1.0, "2024-04-09": 0.5,
    "2024-06-20": 1.0, "2024-06-21": 1.0,
    # 2025 Ramazan: 26.03.2025 Cumhurbaskani aciklamasi -- "2, 3 ve 4 Nisan'da
    # kamu calisanlarimiz idari izinli sayilacak" (toplam 9 gunluk tatil).
    # Onceki surumde bu UC GUN TAMAMEN EKSIKTI.
    "2025-04-02": 1.0, "2025-04-03": 1.0, "2025-04-04": 1.0,
    # 2025 Kurban: EK idari izin VERILMEDI. Iletisim Baskanligi aciklamasi
    # yalnizca YASAL tatili teyit ediyordu (5 Haziran ogleden sonra + 6-9
    # Haziran). Onceki surumdeki "2025-06-05": 0.5 girdisi HATALIYDI --
    # zaten yasal olan yarim gunu ek izin sayiyordu; kaldirildi.
    # 2026 Kurban: 04.05.2026 aciklamasi -- resmi tatile "1,5 gun daha"
    # eklendi; 26 Mayis Sali OGLEDEN ONCE de izinli.
    "2026-05-25": 1.0, "2026-05-26": 0.5,
}

# Ege bolgesi icin mevsimsellik: yaz turizmi ve tarimsal sulama yuku belirleyicidir.
TURKISH_SEASONS = {
    12: "kis", 1: "kis", 2: "kis",
    3: "ilkbahar", 4: "ilkbahar", 5: "ilkbahar",
    6: "yaz", 7: "yaz", 8: "yaz",
    9: "sonbahar", 10: "sonbahar", 11: "sonbahar",
}


def shared_origin(*frames: pd.DataFrame, time_column: str) -> pd.Timestamp:
    """Birden fazla frame icin ORTAK zaman baslangici hesaplar.

    ``add_calendar_features``in ``origin`` parametresine verilmek uzere tasarlandi::

        origin = shared_origin(train, test, time_column="tarih")
        train_f = add_calendar_features(train, "tarih", origin=origin)
        test_f  = add_calendar_features(test,  "tarih", origin=origin)

    Neden gerekli: gun sayaci monoton bir trend kolonudur ve train ile test
    AYNI sifir noktasindan olculmezse test satirlari train'in gecmisine
    kaymis gorunur. Bkz. ``add_calendar_features`` docstring'i.
    """
    minimums = [
        pd.to_datetime(frame[time_column], errors="coerce", format="mixed").min()
        for frame in frames
    ]
    valid = [value for value in minimums if pd.notna(value)]
    if not valid:
        raise ValueError(f"'{time_column}' kolonunda gecerli tarih bulunamadi.")
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
    times = pd.to_datetime(frame[time_column], errors="coerce")

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
    result[f"{prefix}_son_on_gun"] = (
        in_ramadan & (days_to_eid <= LAST_TEN_NIGHTS)
    ).astype(np.int8)
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
    calendar = holidays_lib.TR(years=years, categories=categories)

    # Yarim gunleri ayirt edebilmek icin yalnizca tam gunlerin ayri bir kumesi.
    full_day_calendar = holidays_lib.TR(years=years, categories=("public",))

    dates = times.dt.date
    in_calendar = dates.map(lambda day: day in calendar if pd.notna(day) else False)
    is_full_day = dates.map(
        lambda day: day in full_day_calendar if pd.notna(day) else False
    )
    is_half_day = in_calendar.to_numpy() & ~is_full_day.to_numpy()

    names = dates.map(lambda day: calendar.get(day, "") if pd.notna(day) else "")
    codes = names.map(_holiday_code).astype("int8")
    collision = names.map(lambda value: 1 if ";" in str(value) else 0).astype("int8")

    leave_table = ADMINISTRATIVE_LEAVE if administrative_leave is None else administrative_leave
    leave_lookup = {pd.Timestamp(day).date(): weight for day, weight in leave_table.items()}
    leave_weight = dates.map(
        lambda day: leave_lookup.get(day, 0.0) if pd.notna(day) else 0.0
    ).astype("float32")

    holiday_weight = np.where(
        is_full_day.to_numpy(), 1.0, np.where(is_half_day, 0.5, 0.0)
    ).astype("float32")

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
            (unique_days[:, None] - holiday_dates[None, :])
            .astype("timedelta64[D]")
            .astype("int64")
        )
        nearest = differences.min(axis=1)
        distances[valid_mask] = np.clip(
            nearest[inverse], 0, MISSING_HOLIDAY_DISTANCE
        ).astype("int16")

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


def _sorted_view(
    frame: pd.DataFrame, sort_by: Sequence[str], *, time_column: str
) -> tuple[pd.DataFrame, np.ndarray]:
    """Sirali bir kopya ve orijinal siraya donmek icin permutasyon dondurur."""
    keys = [
        _sort_key(frame[column], is_time=(column == time_column))
        for column in reversed(list(sort_by))
    ]
    order = np.lexsort(keys)
    return frame.iloc[order], order


def add_lag_features(
    frame: pd.DataFrame,
    value_column: str,
    lags: Sequence[int],
    *,
    time_column: str,
    group_columns: Sequence[str] | None = None,
    horizon: int = 1,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Gecikmeli (lag) feature'lar ekler. YENI frame dondurur, sirayi korur.

    Zaman serisi problemlerinde en guclu feature ailesi genellikle budur:
    "bir onceki gunun tuketimi", "gecen haftanin ayni gunu".

    Args:
        value_column: Gecikmesi alinacak kolon (genellikle hedef veya bir olcum).
        lags: Gecikme adimlari, or. ``[1, 7, 14, 28]``. Tahmin anindan GERIYE
            sayilir, satirdan degil -- bkz. ``horizon``.
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

    lagged = {}
    for lag in lags:
        # Lag tahmin ANINDAN geriye sayilir: ufuk 1 iken shift(lag),
        # ufuk 30 iken shift(29 + lag).
        shifted = source.shift(horizon + lag - 1)
        name = f"{prefix}_lag{lag}" if horizon == 1 else f"{prefix}_ufuk{horizon}_lag{lag}"
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
    group_columns: Sequence[str] | None = None,
    horizon: int = 1,
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
        rolling_source = shifted.groupby(
            [ordered[column].to_numpy() for column in group_columns]
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
    group_columns: Sequence[str] | None = None,
    aggregations: Sequence[str] = ("mean", "std"),
    prefix: str | None = None,
) -> pd.DataFrame:
    """Genisleyen pencere (tum gecmis) istatistikleri. YENI frame dondurur.

    Kayan pencerenin aksine tum gecmisi kullanir; bir varligin "genel seviyesini"
    yakalar. Yine ``shift(1)`` ile mevcut satir haric tutulur.
    """
    prefix = prefix or value_column
    sort_keys = list(group_columns or []) + [time_column]
    ordered, order = _sorted_view(frame, sort_keys, time_column=time_column)

    if group_columns:
        shifted = ordered.groupby(list(group_columns), observed=True)[value_column].shift(1)
        source = shifted.groupby([ordered[column].to_numpy() for column in group_columns])
    else:
        source = ordered[value_column].shift(1)

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
