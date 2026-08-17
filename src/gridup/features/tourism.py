"""AYLIK turizm serisi (KTB il bultenleri) ve ilce-ay tahmini.

NEDEN AYRI MODUL
----------------
``national.add_annual_district_attribute`` yillik ilce ozniteligini yil
gecikmesiyle baglar: "bu ilce ne kadar turistik". Aylik seri ise "yilin
HANGI ayinda ne kadar" sorusuna cevaptir -- Mugla'da Temmuz gecelemesi
Ocak'in 30 kati (olculdu: 2025-07 4.9M, 2026-02 0.5M). Bu mevsim profili
sabit takvim feature'larindan farklidir: il'e gore degisir (Denizli'de
yaz sicramasi yok, Mugla'da devasa) ve yildan yila kayar.

SIZINTI DISIPLINI -- neden ``lag_months >= 2``
----------------------------------------------
KTB ay M'nin bultenini M+2'nin 3-11'i arasi yayimlar (fetch_turizm_aylik.py
basliginda olculen tarihler; 2026 Ocak istisnasi M+3). Bir tarih icin:
  * lag 12 (varsayilan) -- gecen yilin ayni ayi: HER ZAMAN elde, mevsim
    profilini tasir. Uzun ufuklu tahminde tek guvenli secim.
  * lag 3 -- her gun icin guvenle yayimlanmis en son ay.
  * lag 2 -- ayin ~11'inden sonra guvenli; oncesinde ELDE OLMAYABILIR.
    Izin verilir ama bilincli secilmelidir.
  * lag 0-1 -- yayimlanmamis veriyi kullanmak demektir; REDDEDILIR.

KAPSAM SICRAMALARI
------------------
2022 Kasim'da bulten kapsami "isletme belgeli" -> "isletme ve basit
belgeli" oldu; 2025 Temmuz civarinda basliksiz ikinci bir genisleme daha
var (doluluk sabitken geceleme +%25-50; bkz. fetch_turizm_aylik.py). Ham
geceleme/gelis yillar arasi kiyaslanamaz. Iki kapsamdan bagimsiz olcu:
  * ``doluluk`` -- oran, tesis sayisindan etkilenmez; TERCIH EDILEN.
  * ``{prefix}_{kolon}_yil_payi`` -- o ayin, kaynak yilin 12 ayindaki
    payi; kapsam yil ICINDE degismediyse temizdir (2022 ve 2025 icin
    kismen bozuktur, o yillarin payina guvenilmemeli).
Ay payi yalnizca 12 ayin tamami varsa hesaplanir; eksikse NaN birakilir
(yanlis payda ile "dogru gorunen" oran uretmek yerine).
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

__all__ = ["add_monthly_attribute", "district_monthly_estimate", "MIN_LAG_MONTHS"]

#: Yayin gecikmesinin altina inen lag reddedilir (bkz. modul basligi).
MIN_LAG_MONTHS = 2
AY_SAYISI = 12


def _ay_indeksi(yil: pd.Series, ay: pd.Series) -> pd.Series:
    """(yil, ay) -> tam sayi ay indeksi; ay kaydirmasini toplama cevirir."""
    return yil.astype(int) * AY_SAYISI + (ay.astype(int) - 1)


def add_monthly_attribute(
    frame: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    value_columns: Sequence[str],
    monthly_key_column: str | None = None,
    year_column: str = "yil",
    month_column: str = "ay",
    lag_months: int = 12,
    add_year_share: bool = True,
    prefix: str = "aylik",
) -> pd.DataFrame:
    """Aylik anahtar x donem ozniteligini panele ``lag_months`` gecikmeyle baglar.

    Panel satirinin (yil, ay) donemi ``lag_months`` geri kaydirilir ve o
    donemin degeri alinir. ``lag_months=12`` "gecen yilin ayni ayi"dir.
    YENI frame dondurur; girdi degismez.

    ``add_year_share`` acikken her deger kolonu icin ``_yil_payi`` turevi de
    uretilir: kaynak donemin degeri / kaynak YILIN 12 aylik toplami. Kapsam
    sicramasindan bagimsiz mevsim profili budur.

    Raises:
        ValueError: ``lag_months < MIN_LAG_MONTHS`` (yayimlanmamis veri).
        KeyError: Gerekli kolon eksikse.
    """
    if lag_months < MIN_LAG_MONTHS:
        raise ValueError(
            f"lag_months={lag_months} yayin gecikmesinin altinda; en az {MIN_LAG_MONTHS} olmali. "
            "KTB ay M'yi M+2'de yayimlar -- daha kucuk lag tahmin aninda elde olmayan veridir."
        )
    for kolon in (key_column, time_column):
        if kolon not in frame.columns:
            raise KeyError(f"frame icinde '{kolon}' kolonu yok.")
    aylik_anahtar = monthly_key_column or key_column
    for kolon in (aylik_anahtar, year_column, month_column, *value_columns):
        if kolon not in monthly.columns:
            raise KeyError(f"monthly icinde '{kolon}' kolonu yok.")

    tablo = monthly[[aylik_anahtar, year_column, month_column, *value_columns]].copy()
    tablo["_anahtar"] = tablo[aylik_anahtar].astype(str)
    yeniden = {k: f"{prefix}_{k}" for k in value_columns}
    tablo = tablo.rename(columns=yeniden)

    if add_year_share:
        tablo = _yil_payi_ekle(tablo, list(yeniden.values()), year_column, month_column)

    # Gecikme KAYNAK donemine eklenir: 2025-07 verisi lag 12 ile 2026-07'de gorunur.
    tablo["_eslesme"] = _ay_indeksi(tablo[year_column], tablo[month_column]) + int(lag_months)
    tablo = tablo.drop(columns=[aylik_anahtar, year_column, month_column])

    cikti = frame.copy()
    zaman = pd.to_datetime(cikti[time_column])
    cikti["_anahtar"] = cikti[key_column].astype(str)
    cikti["_eslesme"] = _ay_indeksi(zaman.dt.year, zaman.dt.month)
    cikti = cikti.merge(tablo, on=["_anahtar", "_eslesme"], how="left")
    return cikti.drop(columns=["_anahtar", "_eslesme"])


def _yil_payi_ekle(
    tablo: pd.DataFrame, deger_kolonlari: list[str], year_column: str, month_column: str
) -> pd.DataFrame:
    """Her deger icin ``<kolon>_yil_payi`` ekler; 12 ayi eksik yillarda NaN."""
    grup = tablo.groupby(["_anahtar", year_column], sort=False)
    ay_sayisi = grup[month_column].transform("nunique")
    tam_yil = ay_sayisi.eq(AY_SAYISI)
    cikti = tablo.copy()
    for kolon in deger_kolonlari:
        degerler = pd.to_numeric(cikti[kolon], errors="coerce")
        toplam = degerler.groupby([cikti["_anahtar"], cikti[year_column]]).transform("sum")
        gecerli = tam_yil & toplam.gt(0)
        cikti[f"{kolon}_yil_payi"] = (degerler / toplam.where(gecerli)).where(gecerli)
    return cikti


def district_monthly_estimate(
    annual: pd.DataFrame,
    monthly: pd.DataFrame,
    *,
    district_column: str = "ilce_key",
    province_column: str = "il_key",
    year_column: str = "yil",
    month_column: str = "ay",
    value_column: str = "geceleme",
) -> pd.DataFrame:
    """Ilce x yil x ay tahmini: ilcenin il icindeki YILLIK payi x ilin AYLIK degeri.

    KTB aylik bulten yalnizca il kirilimi verir; ilce kirilimi yilliktir.
    Ilcenin mevsim profilinin ilinkiyle ayni oldugu varsayilir (Bodrum'un
    Temmuz payi Mugla'nin Temmuz payina esit). Bu bir YAKLASIMDIR ve kolon
    adi bunu soyler (``_tahmini``); dogrudan olcum gibi sunulmamalidir.

    Yalnizca HER IKI tabloda da bulunan yillar uretilir. Ilcenin il payi
    yillik tablodaki il toplamina gore hesaplanir (il alt-toplam satirlari
    onceden ayiklanmis olmalidir; fetch_turizm.py bunu yapar).

    Returns:
        Kolonlar: district_column, province_column, year_column,
        month_column, ``{value_column}_tahmini``, ``ilce_il_payi``.

    Raises:
        KeyError: Gerekli kolon eksikse.
        ValueError: Ortak yil yoksa.
    """
    for kolon in (district_column, province_column, year_column, value_column):
        if kolon not in annual.columns:
            raise KeyError(f"annual icinde '{kolon}' kolonu yok.")
    for kolon in (province_column, year_column, month_column, value_column):
        if kolon not in monthly.columns:
            raise KeyError(f"monthly icinde '{kolon}' kolonu yok.")

    ortak = sorted(set(annual[year_column].astype(int)) & set(monthly[year_column].astype(int)))
    if not ortak:
        raise ValueError("annual ve monthly tablolarinda ortak yil yok; tahmin uretilemez.")

    yillik = annual[[district_column, province_column, year_column, value_column]].copy()
    yillik[year_column] = yillik[year_column].astype(int)
    yillik = yillik[yillik[year_column].isin(ortak)]
    yillik[value_column] = pd.to_numeric(yillik[value_column], errors="coerce")
    il_toplam = yillik.groupby([province_column, year_column])[value_column].transform("sum")
    yillik["ilce_il_payi"] = (yillik[value_column] / il_toplam.where(il_toplam.gt(0))).where(
        il_toplam.gt(0)
    )
    yillik = yillik.drop(columns=[value_column])

    aylik = monthly[[province_column, year_column, month_column, value_column]].copy()
    aylik[year_column] = aylik[year_column].astype(int)
    aylik = aylik[aylik[year_column].isin(ortak)]
    aylik = aylik.rename(columns={value_column: "_il_aylik"})

    birlesik = yillik.merge(aylik, on=[province_column, year_column], how="inner")
    birlesik[f"{value_column}_tahmini"] = birlesik["ilce_il_payi"] * pd.to_numeric(
        birlesik["_il_aylik"], errors="coerce"
    )
    kolonlar = [
        district_column,
        province_column,
        year_column,
        month_column,
        f"{value_column}_tahmini",
        "ilce_il_payi",
    ]
    return (
        birlesik[kolonlar]
        .sort_values([district_column, year_column, month_column])
        .reset_index(drop=True)
    )
