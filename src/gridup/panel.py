"""Panel (varlik x zaman) yapisi kurma.

NEDEN BU MODUL VAR -- arastirmanin en sessiz tuzagi
---------------------------------------------------
Kesinti/ariza veri setleri genellikle **olay kayitlaridir**: bir satir = bir
kesinti. "O gun o ilcede kesinti olmadi" bilgisi veri setinde **HIC BULUNMAZ**.

Bu, iki sekilde oldurur:

1. **Lag/rolling kayar.** ``shift(1)`` "bir onceki SATIR"i alir, "bir onceki
   GUN"u degil. Kayitlar seyrekse bir onceki satir 3 hafta oncesine ait olabilir.
   Feature'lar sessizce anlamsizlasir.

2. **Model sifir tahmin etmeyi ogrenemez.** Egitim setinde hic sifir yoksa,
   model asla sifir uretmez -- ama gercek gunlerin cogu sifirdir.

Cozum: tam kartezyen carpim (her varlik x her tarih) uzerinde yeniden indeksle
ve eksik gunleri sifirla doldur.

DOGRULANMIS ONCEKI YARISMA (atif iki kez duzeltildi -- ders: 404 != yok)
------------------------------------------------------------------------
2024 GDZ Datathon'unun hedefi gercekten bir SAYIM idi: ilce basina gunluk
``bildirimsiz`` (plansiz) kesinti ADEDI, metrik MAE. Kaynak: birincinin
(Pikachow) final sunumu, s.4 -- anilozturk.net'te halka acik; ayrica
coderspace.io/etkinlikler/gdz-elektrik-datathon-2024 etkinlik sayfasi.
Bir onceki denetim ``kaggle competitions list -s gdz`` ciktisinda 2024'u
goremedigi icin bu atifi "yarisma yok" diye silmisti -- in-class yarismalar
o aramada gorunmuyor; 404/bos arama, yokluk kaniti DEGILDIR.
2023 yarismasinin hedefi ise ``Dagitilan Enerji (MWh)`` idi -- bir OLCUM.

Bu ayrim tam da bu modulun konusudur:

    olcum hedefi (MWh)      -> kayit yoksa deger BILINMIYOR  -> np.nan
    sayim hedefi (kesinti)  -> kayit yoksa olay OLMAMIS      -> 0.0

2026 Grid Up'in hangisi oldugu HENUZ BILINMIYOR (2024'un devami olarak
sayim olmasi muhtemel). Veri geldiginde ilk kararlardan biri budur; yanlis
secim veriyi sessizce bozar.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .validation import parse_time_series

__all__ = ["PANEL_FLAG_COLUMN", "build_panel", "panel_coverage"]


#: Doldurma orani bunu asarsa fill_value semantigini ACIKCA hatirlatiriz.
#: %5 esigi kasitli dusuk: birkac yuz sentetik satir bile bir varligin
#: ortalamasini belirgin kaydirabilir (olculdu: %40 doldurmada 50.0 -> 10.0).
FILL_NOTICE_SHARE = 5.0

#: Bu oranin ustunde hedef pratikte sifir-siskindir; iki asamali model oner.
HIGH_FILL_SHARE = 60.0

#: Sentetik satir bayragi. FEATURE OLARAK KULLANILMAZ -- fill_value=0 iken
#: hedefin sifir olmasiyla BIREBIR ayni seydir (olculdu: %100 ortusme,
#: Spearman -0.9810). Modelleyen kod bu adi feature listesinden cikarmalidir.
PANEL_FLAG_COLUMN = "_dolduruldu"


def _izgara_basi(damga: pd.Timestamp, freq: str) -> pd.Timestamp:
    """Zaman damgasini freq adiminin basina indirir; yapamazsa oldugu gibi birakir."""
    try:
        return damga.floor(freq)
    except ValueError:
        # "ME", "W" gibi takvim tabanli adimlarda floor tanimsizdir; izgaraya
        # oturtma isini _izgaraya_otur yapar, burada zarar vermeden geciyoruz.
        return damga


def _izgaraya_otur(times: pd.Series, timeline: pd.DatetimeIndex) -> pd.Series:
    """Her damgayi kendisinden kucuk-esit SON izgara noktasina oturtur.

    NEDEN ZORUNLU (olculdu)
    -----------------------
    Kesinti kayitlari saat damgalidir (``2024-01-03 14:23``). ``freq="D"``
    izgarasi gece yarilarindan olusur, dolayisiyla ``merge`` neredeyse hicbir
    kaydi bulamaz ve gercek gozlemler paneli hic gormeden dusup gider:

        ham 332 kayit, sure_dk toplami 35.576
        panel 1.200 satir, sure_dk toplami  3.416   -> hedefin %90.4'u YOK

        gunde ~3 olayli veride kayip %99.8

    Hicbir hata, hicbir uyari cikmiyordu; model tamamen sifir ogreniyordu.
    """
    if len(timeline) == 0:
        return times
    konum = np.searchsorted(timeline.to_numpy(), times.to_numpy(), side="right") - 1
    konum = np.clip(konum, 0, len(timeline) - 1)
    return pd.Series(timeline.to_numpy()[konum], index=times.index, name=times.name)


def _panel_hazirla(
    frame: pd.DataFrame,
    entity_list: list[str],
    time_column: str,
    *,
    freq: str,
    start: str | pd.Timestamp | None,
    end: str | pd.Timestamp | None,
) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Kayitlari temizler ve izgaraya oturtur; (calisma cercevesi, izgara) doner.

    Uc sessiz kayip burada kapaniyor -- ucu de olculdu:
      1. gecersiz tarih
      2. varlik anahtarinda NaN (hem kutle kaybi hem hayalet varlik)
      3. saat damgasinin gunluk izgaraya oturmamasi (%90-99 kutle kaybi)
    """
    working = frame.copy()
    # parse_time_series ile: duz pd.to_datetime, TR "gg.aa.yyyy" kolonunda
    # gun>12 olan her kaydi NaT yapar. GERCEK bir gun-1 kosusunda olculdu:
    #     [build_panel] UYARI: 641 satirda gecersiz tarih var  (1.091 satirin)
    #     20 varlik x 339 zaman adimi                          (120 gunluk veri)
    # Yani panelin ucte ikisi atiliyor, kalani yanlis takvime yayiliyordu.
    # strict=False: bicim yine KANITLANARAK secilir, ama gercekten eksik olan
    # tarihler asagidaki uyari+dusurme yolundan gecmeye devam eder.
    working[time_column] = parse_time_series(working[time_column], strict=False)

    invalid = int(working[time_column].isna().sum())
    if invalid:
        print(
            f"[build_panel] UYARI: {invalid:,} satirda gecersiz tarih var, "
            "panel disinda birakiliyor."
        )
        working = working[working[time_column].notna()]
    if working.empty:
        raise ValueError("Gecerli tarihli satir kalmadi; panel kurulamaz.")

    # Varlik anahtarinda NaN olan satirlar iki yonde birden zarar veriyordu
    # (olculdu, 50 satir / 5.483 dakika):
    #   * groupby(dropna=True) onlarin hedefini SESSIZCE dusuruyordu
    #   * ama drop_duplicates() NaN'i bir "varlik" sayip 60 satirlik HAYALET
    #     bir varlik uretiyordu -- hepsi sifir dolgulu
    # Yani hem gercek kutle kayboluyor hem yerine uydurma satir geliyordu.
    anahtar_bos = working[entity_list].isna().any(axis=1)
    if bool(anahtar_bos.any()):
        print(
            f"[build_panel] UYARI: {int(anahtar_bos.sum()):,} satirin varlik "
            f"anahtarinda ({', '.join(entity_list)}) eksik deger var. Bu satirlar "
            "panel disinda birakiliyor -- aksi halde hedefleri sessizce kaybolur "
            "ve yerlerine NaN anahtarli hayalet varlik satirlari uretilirdi."
        )
        working = working[~anahtar_bos]
        if working.empty:
            raise ValueError("Varlik anahtari dolu satir kalmadi; panel kurulamaz.")

    grid_start = (
        _izgara_basi(pd.Timestamp(start), freq)
        if start
        else _izgara_basi(working[time_column].min(), freq)
    )
    grid_end = pd.Timestamp(end) if end else working[time_column].max()
    timeline = pd.date_range(grid_start, grid_end, freq=freq, name=time_column)
    if len(timeline) == 0:
        raise ValueError(
            f"Zaman izgarasi bos: start={grid_start}, end={grid_end}, freq={freq!r}."
        )

    # Acik start/end verildiyse disarida kalan kayitlar izgara kenarina
    # YAPISTIRILMAZ -- oraya ait olmadiklari halde kutle tasirlardi.
    disarida = (working[time_column] < timeline[0]) | (working[time_column] > grid_end)
    if bool(disarida.any()):
        print(
            f"[build_panel] UYARI: {int(disarida.sum()):,} kayit "
            f"[{timeline[0]}, {grid_end}] araligi disinda, panele alinmadi."
        )
        working = working[~disarida]
        if working.empty:
            raise ValueError("Izgara araliginda kayit kalmadi; panel kurulamaz.")

    # Saat damgalarini gunluk izgaraya oturt: yoksa merge hicbir sey bulamaz.
    working[time_column] = _izgaraya_otur(working[time_column], timeline)
    return working, timeline


def build_panel(
    frame: pd.DataFrame,
    *,
    entity_columns: Sequence[str],
    time_column: str,
    value_columns: Sequence[str] | None = None,
    freq: str = "D",
    fill_value: float = 0.0,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Eksik varlik-zaman kombinasyonlarini doldurarak tam panel uretir.

    Args:
        frame: Olay kayitlari (degistirilmez).
        entity_columns: Varlik anahtari, or. ``["il", "ilce"]`` veya ``["trafo_id"]``.
        time_column: Tarih kolonu.
        value_columns: Sifirla doldurulacak sayisal kolonlar. ``None`` ise
            entity/time disindaki tum sayisal kolonlar.
        freq: Zaman izgarasi (``D`` gunluk, ``h`` saatlik, ``ME`` ay sonu).
        fill_value: Eksik kombinasyonlara yazilacak deger. Sayim hedefinde
            ``0.0`` dogrudur; **olcum** hedefinde (tuketim gibi) ``np.nan``
            kullan -- orada "kayit yok" ile "deger sifir" farkli seylerdir.
        start / end: Izgara sinirlari. ``None`` ise veriden alinir.
        verbose: Kac satirin eklendigini yazdirir.

    Returns:
        Yeni DataFrame, ``entity_columns + [time_column]`` siralı.
        ``_dolduruldu`` kolonu, satirin sentetik olup olmadigini isaretler --
        bu bayragi feature olarak KULLANMA (test'te anlamsizdir) ama hata
        analizinde ise yarar.

    Raises:
        KeyError: Beklenen kolon yoksa.
    """
    missing = [
        column
        for column in list(entity_columns) + [time_column]
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(f"Panel icin gereken kolonlar eksik: {missing}")

    entity_list = list(entity_columns)
    working, timeline = _panel_hazirla(
        frame, entity_list, time_column, freq=freq, start=start, end=end
    )
    entities = working[entity_list].drop_duplicates().reset_index(drop=True)

    # Kartezyen carpim: her varlik x her zaman adimi.
    grid = entities.merge(pd.DataFrame({time_column: timeline}), how="cross")

    if value_columns is None:
        value_columns = [
            column
            for column in working.columns
            if column not in entity_list + [time_column]
            and pd.api.types.is_numeric_dtype(working[column])
        ]

    value_list = list(value_columns)
    # Sayisal olmayan kolonlar eskiden panele HIC alinmiyordu: girdi
    # ['ilce','tarih','sure_dk','sebep'] iken cikti ['ilce','tarih','sure_dk',
    # '_dolduruldu'] oluyor ve 'sebep' sessizce yok oluyordu (olculdu). Artik
    # tasiniyorlar; doldurulan satirlarda degerleri NaN kalir -- cunku o
    # satirlarda gercekten bir gozlem YOKTUR, sifir da degildir.
    diger_kolonlar = [
        column
        for column in working.columns
        if column not in entity_list + [time_column] + value_list
    ]

    # Ayni varlik-zaman icin birden fazla kayit olabilir (bir gunde iki kesinti):
    # once topla, sonra izgaraya otur.
    toplama: dict[str, str] = {column: "sum" for column in value_list}
    toplama.update({column: "first" for column in diger_kolonlar})
    aggregated = (
        working.groupby(entity_list + [time_column], observed=True)
        .agg(toplama)
        .reset_index()
    )

    panel = grid.merge(
        aggregated,
        on=entity_list + [time_column],
        how="left",
        validate="one_to_one",
        indicator="_kaynak",
    )

    # Sentetik satir bayragi: hata analizinde ise yarar, FEATURE OLARAK KULLANMA.
    # merge gostergesinden turetiyoruz; eskiden ilk deger kolonunun NaN olmasina
    # bakiliyordu ve o kolonu bos olan GERCEK bir satir da "sentetik" sayiliyordu.
    panel[PANEL_FLAG_COLUMN] = (panel["_kaynak"] == "left_only").astype("int8")
    panel = panel.drop(columns="_kaynak")

    for column in value_list:
        panel[column] = panel[column].fillna(fill_value)

    panel = panel.sort_values(entity_list + [time_column]).reset_index(drop=True)

    if verbose:
        added = len(panel) - len(aggregated)
        share = added / max(len(panel), 1) * 100
        print(
            f"[build_panel] {len(entities):,} varlik x {len(timeline):,} zaman adimi "
            f"= {len(panel):,} satir. {added:,} satir eklendi (%{share:.1f})."
        )
        # Doldurulan her satir UYDURULMUS bir gozlemdir. Hangi degerle
        # dolduruldugu ve bunun ne ANLAMA geldigi, oran anlamli oldugunda
        # acikca soylenmeli -- aksi halde kullanici olcum hedefinde sahte
        # sifirlarla model egitir ve bunu hicbir yerde gormez.
        #
        # OLCULDU: yalnizca ariza gunlerinde kayit ureten bir trafo icin
        # panel %40 sentetik satir uretti ve o varligin ortalamasi
        # 50.0 -> 10.0'a dustu. Sayim hedefinde bu DOGRUDUR; tuketim gibi
        # bir OLCUM hedefinde ise veriyi bozar.
        if added and share >= FILL_NOTICE_SHARE:
            print(
                f"  DOLDURMA: {added:,} satira fill_value={fill_value!r} yazildi.\n"
                "    sayim hedefi   (ariza adedi, kesinti suresi) -> 0.0 DOGRU\n"
                "    olcum hedefi   (tuketim, gerilim)            -> np.nan KULLAN\n"
                "    'kayit yok' ile 'deger sifir' ayni sey degildir."
            )
        if share > HIGH_FILL_SHARE:
            print(
                "  NOT: satirlarin cogu sentetik. Hedef sifir-siskin -- iki asamali "
                "model (once 'olay var mi', sonra 'kac tane') dusun."
            )
        if diger_kolonlar:
            print(
                f"  Sayisal olmayan {len(diger_kolonlar)} kolon 'first' ile tasindi "
                f"({diger_kolonlar[:5]}); doldurulan satirlarda NaN kalir."
            )

    return panel


def panel_coverage(
    frame: pd.DataFrame, *, entity_columns: Sequence[str], time_column: str, freq: str = "D"
) -> dict[str, float]:
    """Panelin ne kadar seyrek oldugunu olcer -- ``build_panel`` oncesi tanı.

    Returns:
        ``entity_count``, ``time_steps``, ``expected_rows``, ``actual_rows``,
        ``coverage`` (0-1 arasi doluluk orani).

    ZAMANI IZGARAYA OTURTMAK NEDEN SART (olculdu)
    ---------------------------------------------
    ``actual`` eskiden ham damgalarla gruplaniyordu. Saat damgali kayitta ayni
    gunun 03:12 ve 17:40 kayitlari IKI ayri "varlik-zaman" sayiliyor, dolayisiyla
    ``actual`` ``expected``i asabiliyordu:

        beklenen 1.200, gercek 3.658  ->  DOLULUK %304.8

    ``day_one`` bu sayiya bakip "doluluk >= %95, panel gerekmez" diyor ve panel
    kurulmuyordu -- oysa ayni veride panel kurulsa hedefin %99.8'i kaybolacakti.
    Yani hatali olcum, hatali kurulumu GIZLIYORDU.
    """
    entity_list = list(entity_columns)
    times = parse_time_series(frame[time_column], strict=False)
    gecerli = times.notna() & ~frame[entity_list].isna().any(axis=1)
    if not bool(gecerli.any()):
        return {"coverage": float("nan"), "note": "gecerli tarih yok"}  # type: ignore[dict-item]

    times = times[gecerli]
    entity_count = len(frame.loc[gecerli, entity_list].drop_duplicates())
    timeline = pd.date_range(_izgara_basi(times.min(), freq), times.max(), freq=freq)
    if len(timeline) == 0:
        return {"coverage": float("nan"), "note": "izgara bos"}  # type: ignore[dict-item]

    oturtulmus = _izgaraya_otur(times, timeline)
    expected = entity_count * len(timeline)
    actual = len(
        frame.loc[gecerli, entity_list]
        .assign(**{time_column: oturtulmus.to_numpy()})
        .drop_duplicates()
    )

    return {
        "entity_count": float(entity_count),
        "time_steps": float(len(timeline)),
        "expected_rows": float(expected),
        "actual_rows": float(actual),
        "coverage": actual / expected if expected else float("nan"),
    }
