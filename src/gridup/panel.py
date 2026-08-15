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

DOGRULANMIS ONCEKI YARISMA
--------------------------
Onceki surumde burada "2024 GDZ Datathon'unda hedef ``bildirimsiz_sum``"
yaziyordu. Bu atif **DOGRULANAMADI ve iki yonden yanlisti**: Kaggle'da
GDZ'nin 2024 yarismasi YOKTUR (bkz. ``kaggle competitions list -s gdz``),
ve gercek olan 2023 yarismasinin hedefi ``Dagitilan Enerji (MWh)`` idi --
bir SAYIM degil, bir OLCUM.

Bu ayrim tam da bu modulun konusudur:

    olcum hedefi (MWh)      -> kayit yoksa deger BILINMIYOR  -> np.nan
    sayim hedefi (kesinti)  -> kayit yoksa olay OLMAMIS      -> 0.0

2026 Grid Up'in hangisi oldugu HENUZ BILINMIYOR. Veri geldiginde ilk
kararlardan biri budur; yanlis secim veriyi sessizce bozar.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["build_panel", "panel_coverage"]


#: Doldurma orani bunu asarsa fill_value semantigini ACIKCA hatirlatiriz.
#: %5 esigi kasitli dusuk: birkac yuz sentetik satir bile bir varligin
#: ortalamasini belirgin kaydirabilir (olculdu: %40 doldurmada 50.0 -> 10.0).
FILL_NOTICE_SHARE = 5.0

#: Bu oranin ustunde hedef pratikte sifir-siskindir; iki asamali model oner.
HIGH_FILL_SHARE = 60.0


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
    working = frame.copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce")

    invalid = int(working[time_column].isna().sum())
    if invalid:
        print(
            f"[build_panel] UYARI: {invalid:,} satirda gecersiz tarih var, "
            "panel disinda birakiliyor."
        )
        working = working[working[time_column].notna()]

    if working.empty:
        raise ValueError("Gecerli tarihli satir kalmadi; panel kurulamaz.")

    grid_start = pd.Timestamp(start) if start else working[time_column].min()
    grid_end = pd.Timestamp(end) if end else working[time_column].max()
    timeline = pd.date_range(grid_start, grid_end, freq=freq, name=time_column)

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

    # Ayni varlik-zaman icin birden fazla kayit olabilir (bir gunde iki kesinti):
    # once topla, sonra izgaraya otur.
    aggregated = (
        working.groupby(entity_list + [time_column], observed=True)[list(value_columns)]
        .sum()
        .reset_index()
    )

    panel = grid.merge(
        aggregated, on=entity_list + [time_column], how="left", validate="one_to_one"
    )

    # Sentetik satir bayragi: hata analizinde ise yarar, FEATURE OLARAK KULLANMA
    # (test'te her satir "dolduruldu" olacagi icin anlamsizdir).
    value_list = list(value_columns)
    if value_list:
        panel["_dolduruldu"] = panel[value_list[0]].isna().astype("int8")
    else:
        panel["_dolduruldu"] = 0

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

    return panel


def panel_coverage(
    frame: pd.DataFrame, *, entity_columns: Sequence[str], time_column: str, freq: str = "D"
) -> dict[str, float]:
    """Panelin ne kadar seyrek oldugunu olcer -- ``build_panel`` oncesi tanı.

    Returns:
        ``entity_count``, ``time_steps``, ``expected_rows``, ``actual_rows``,
        ``coverage`` (0-1 arasi doluluk orani).
    """
    entity_list = list(entity_columns)
    times = pd.to_datetime(frame[time_column], errors="coerce").dropna()
    if times.empty:
        return {"coverage": float("nan"), "note": "gecerli tarih yok"}  # type: ignore[dict-item]

    entity_count = len(frame[entity_list].drop_duplicates())
    timeline = pd.date_range(times.min(), times.max(), freq=freq)
    expected = entity_count * len(timeline)
    actual = len(frame.groupby(entity_list + [time_column], observed=True).size())

    return {
        "entity_count": float(entity_count),
        "time_steps": float(len(timeline)),
        "expected_rows": float(expected),
        "actual_rows": float(actual),
        "coverage": actual / expected if expected else float("nan"),
    }
