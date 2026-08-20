"""Harici veri kaynaklarini TEK cagriyla panele baglar (aile bazli, olculebilir).

NEDEN BU MODUL VAR
------------------
2026-08-18 denetimi olctu: on bir harici kaynagin SEKIZI yalnizca kutuphane
fonksiyonuydu -- ``tests/`` disinda hicbir cagirani yoktu. ``hava_saatlik_turev``
parquet'ini hicbir kod OKUMUYORDU. Yani veri toplanmis, dogrulanmis, Kaggle
paketine konmus ama panele hic girmemisti. Veri gununde her birini elle,
bilinmeyen bir semaya karsi, saat baskisi altinda baglamak gerekecekti.

Bu modul dagitik bilgiyi tek yere toplar:

  * her kaynagin JOIN ANAHTARI ve zaman cozunurlugu,
  * hangi feature fonksiyonuyla baglandigi,
  * YAYIN GECIKMESI (ufuk kaydirmasi gerektiren kaynaklar),
  * eksikse ne olacagi (sessiz NaN degil, raporlanan atlama).

TASARIM: AILE = ABLASYON BIRIMI
-------------------------------
Her kaynak bir "aile"dir ve ``families=`` ile tek tek acilip kapatilabilir.
Cikti ``(frame, aileler)`` ciftidir; ``aileler`` her ailenin URETTIGI kolon
adlarini verir. ``scripts/ablation_gercek.py`` bunu dogrudan tuketir: bir
ailenin katkisi = o ailenin kolonlari cikarilinca MAE'nin degisimi. Boylece
"hangi harici veri ise yariyor" sorusu fikirle degil olcumle yanitlanir.

SESSIZ BOSA DUSME YASAK
-----------------------
Her join sonrasi eslesme orani olculur: %0 ise ``ValueError`` (anahtar
bozuk), esik altiysa uyari. Denetimde tam bu sinif iki kez yakalandi
(docs/07'nin il_key<->konum_key ornegi %0 esliyordu). Kaynak dosya yoksa
aile ATLANIR ve nedeni ``atlanan`` sozlugunde raporlanir -- feature sessizce
NaN olmaz.

UFUK DISIPLINI
--------------
Gozlenen hava/hava-kalitesi/CAPE ayni gun joinlenir (yarisma kurali: tahmin
aninda meteorolojik TAHMIN elde olur; 2024 birincisi de boyle kullandi).
Sonradan yayimlanan kaynaklar zorunlu gecikmeyle baglanir: KTB aylik
``lag_months>=2``, KTB yillik ``year_lag=1``, EPIAS ulusal seri
``shift(horizon)``, nokta olaylar (yangin/deprem) ``horizon`` kaydirmali
yogunluk. Bu kurallar ilgili fonksiyonlarin ICINDE zorunludur; burasi
yalnizca dogru parametreyi gecer.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .national import (
    add_annual_district_attribute,
    add_national_series,
    add_seasonal_district_profile,
    daily_from_hourly,
)
from .point_events import add_point_event_features
from .tourism import add_monthly_attribute, district_monthly_estimate
from .weather import add_physical_derivatives

__all__ = [
    "ExternalAttachment",
    "EXTERNAL_FAMILIES",
    "KapsamBoslugu",
    "attach_external",
]


class KapsamBoslugu(ValueError):  # noqa: N818 -- Turkce ad; "Error" eki depo diline aykiri
    """Kaynak dogru ama BASKA bir donemi kapsiyor -- aile atlanir, hat durmaz.

    NEDEN AYRI BIR SINIF (2026-08-21, gercek veri provasi yakaladi)
    ---------------------------------------------------------------
    ``%0 eslesme`` iki tamamen farkli arizadan gelebilir ve ikisinin dogru
    cevabi ZITTIR:

      * **Anahtar bozuk** -- ilce adlari uyusmuyor. Durmak DOGRUDUR; devam
        etmek sessiz bir NaN sutunu uretir ve model onu "bilgi yok" diye
        degil "bu ilcede turizm sifir" diye okur.
      * **Zaman kapsami ortusmuyor** -- anahtarlar gayet dogru, kaynak sadece
        baska yillari kapsiyor. Durmak YANLISTIR; eksik olan tek bir ailedir.

    Prova bunu ayna verisinde gosterdi: panel 2021-2022, ``turizm_geceleme``
    2023-2025. ``year_lag=1`` ile istenen yillar 2020-2021 -- hic ortusmuyor.
    Eski davranista TUM ``attach_external`` cagrisi ValueError ile duruyordu.
    Yarisma gunu bu cok muhtemeldir (yarismanin donemi bizim tablolarimizinkiyle
    ayni olmak zorunda degil) ve tek aile yuzunden gun-1 hattinin komple durmasi
    kabul edilemez.

    ValueError'dan turer ki bu istisnayi bilmeyen eski yakalamalar kirilmasin.
    """


#: Bir join'in gecerli sayilmasi icin gereken en dusuk eslesme orani.
#: Altinda uyari, sifirda hata (anahtar tamamen bozuk demektir).
MIN_MATCH_RATE = 0.5

#: Panelin ilce anahtarindan il anahtarina cevrim icin referans tablo.
REFERENCE_RELATIVE = Path("data/reference/ilceler_gdz_adm.parquet")


@dataclass
class ExternalAttachment:
    """``attach_external`` sonucu: frame + aile->kolon haritasi + atlananlar."""

    frame: pd.DataFrame
    families: dict[str, list[str]] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    match_rates: dict[str, float] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        """Tum ailelerin urettigi kolonlar (aile sirasiyla)."""
        return [kolon for kolonlar in self.families.values() for kolon in kolonlar]

    def summary(self) -> str:
        satirlar = [
            f"attach_external: {len(self.families)} aile, {len(self.feature_columns)} kolon"
        ]
        for ad, kolonlar in self.families.items():
            oran = self.match_rates.get(ad)
            oran_metni = f"  eslesme %{100 * oran:.1f}" if oran is not None else ""
            satirlar.append(f"  + {ad:<22} {len(kolonlar):>3} kolon{oran_metni}")
        for ad, neden in self.skipped.items():
            satirlar.append(f"  - {ad:<22} ATLANDI: {neden}")
        return "\n".join(satirlar)


#: Ailelerin KANONIK sirasi. Sira deterministik cikti icin onemlidir
#: (kolon sirasi -> model girdisi -> tekrar uretilebilirlik).
EXTERNAL_FAMILIES: tuple[str, ...] = (
    "hava",
    "hava_saatlik",
    "hava_kalitesi",
    "konvektif",
    "nem_toprak",
    "gunes",
    "yangin",
    "deprem",
    "turizm_yillik",
    "turizm_aylik",
    "turizm_il_aylik",
    "izsu",
    "epias",
    # STATIK ILCE OZELLIKLERI (docs/18 bolum A). Zaman boyutu yoktur; ufuk
    # kaydirmasi ve ambargo bunlara UYGULANMAZ cunku gelecege ait bir bilgi
    # tasimazlar -- arazi ortusu ve sebeke altyapisi gun icinde degismez.
    # Sirasi sonda: once zamanli aileler, sonra statikler.
    "arazi_ortusu",
    "osm_altyapi",
)

#: Statik ilce tablolari: (aile, gorece yol, feature olmayan kolonlar).
#: ``ilce_key`` join anahtaridir; ``il_key`` ve olcum-usulu kolonlar (kac
#: piksel okundu, hangi yaricap kullanildi) KOKEN bilgisidir, feature degil --
#: panele girerse model "olcum penceresi buyuklugunu" ogrenmeye calisir.
_STATIK_ILCE_TABLOLARI: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "arazi_ortusu",
        "data/external/arazi_ortusu_ilce.parquet",
        ("il_key", "ortu_piksel", "ortu_yaricap_km"),
    ),
    (
        "osm_altyapi",
        "data/external/osm_altyapi_ilce.parquet",
        ("il_key", "osm_yaricap_km"),
    ),
)

#: ilce_key x tarih anahtariyla dogrudan birlesen gunluk tablolar.
#: (aile, gorece yol, atlanacak kolonlar)
_GUNLUK_ILCE_TABLOLARI: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # ``hava_tahmin`` KOKEN BILGISIDIR, feature degil: satirin arsivden mi
    # tahmin API'sinden mi geldigini soyler. Feature olarak verilirse zararli
    # bir sey olur -- egitim satirlarinin TAMAMI arsivdir, yani kolon egitimde
    # sabittir ve model onu hic ogrenemez; buna karsilik gelecege tahmin
    # uretilirken TAMAMI 1 olur. Egitimde sabit / testte sabit ama BASKA bir
    # sabit: tasidigi tek sey "bu satir test blogunda" bilgisidir.
    # Tabloda KALIR (denetim ve hizalama kapisi onu kullanir), panele girmez.
    (
        "hava",
        "data/external/hava_gunluk.parquet",
        ("konum", "konum_key", "il_key", "hava_tahmin"),
    ),
    # ``tahmin`` bayragi -- ``hava_tahmin`` ile AYNI gerekce: satirin arsivden
    # mi forecast API'sinden mi geldigini soyleyen KOKEN bilgisidir. Egitim
    # satirlarinin tamami arsiv oldugu icin egitimde sabittir; feature olarak
    # tasidigi tek sey "bu satir gelecekte" bilgisi olur.
    # scripts/kopru_saatlik.py bu kolonu ekler.
    ("hava_saatlik", "data/external/hava_saatlik_turev.parquet", ("tahmin",)),
    ("hava_kalitesi", "data/external/hava_kalitesi_gunluk.parquet", ("tahmin",)),
    ("konvektif", "data/external/konvektif_gunluk.parquet", ("tahmin",)),
    ("nem_toprak", "data/external/nem_toprak_gunluk.parquet", ("tahmin",)),
)

#: Nokta olay kataloglari: (aile, yol, agirlik kolonu, yaricaplar km).
_NOKTA_OLAYLAR: tuple[tuple[str, str, str | None, tuple[float, ...]], ...] = (
    # Uc yaricap UC AYRI MEKANIZMA olcer, ayni seyin uc olcegi degil:
    #   10 km -- dogrudan hasar: alev/isi hattin kendisine ulasir, direk ve
    #            iletken zarar gorur. En seyrek ama en siddetli yol.
    #   25 km -- duman/aerosol: iyonize duman izolator yuzeyinde ATLAMAYA
    #            (flashover) yol acar; yangina temas etmeden kesinti olur.
    #   50 km -- bolgesel yangin havasi vekili: sicak+kuru+ruzgarli rejimin
    #            kendisi zaten yuk ve ariza artirir.
    # 10 km OLCULEREK eklendi (2026-08-20): yaricap basina maliyet ~0.4 sn
    # (232.608 satirlik panelde toplam 1.17 sn olculdu), yani ucuz.
    ("yangin", "data/external/yanginlar.parquet", "frp", (10.0, 25.0, 50.0)),
    # Agirlik "buyukluk" DEGIL "enerji": Richter logaritmiktir, dolayisiyla
    # buyuklukleri toplamak otuz kucuk sarsintiyi bir buyuk depremden onemli
    # gosterirdi. ``enerji`` = 10^(1.5*(M-4)), bkz. scripts/fetch_deprem.py.
    ("deprem", "data/external/depremler.parquet", "enerji", (50.0, 100.0)),
)


def _eslesme_orani(frame: pd.DataFrame, oncesi: pd.DataFrame, kolon: str, aile: str) -> float:
    """Yeni kolonun dolu satir oranini olcer; %0 ise hata, dusukse uyari."""
    del oncesi
    oran = float(frame[kolon].notna().mean()) if len(frame) else 0.0
    if oran == 0.0:
        raise ValueError(
            f"'{aile}' ailesi panele HIC eslesmedi (%0). Join anahtari bozuk: "
            "panelin ilce anahtari ile kaynagin ilce anahtari ayni bicimde mi "
            "(gridup.turkish.join_key)? Sessiz NaN yerine durduruldu."
        )
    if oran < MIN_MATCH_RATE:
        warnings.warn(
            f"'{aile}' ailesi panelin yalnizca %{100 * oran:.1f}'ine eslesti; "
            "anahtar/tarih kapsamini kontrol et.",
            UserWarning,
            stacklevel=3,
        )
    return oran


def _gunluk_ilce_ekle(
    frame: pd.DataFrame,
    yol: Path,
    *,
    key_column: str,
    time_column: str,
    drop_columns: Sequence[str],
    aile: str,
) -> tuple[pd.DataFrame, list[str], float]:
    """ilce_key x tarih tablosunu panele birlestirir; satir sayisini korur."""
    tablo = pd.read_parquet(yol)
    eksik = [k for k in ("ilce_key", "tarih") if k not in tablo.columns]
    if eksik:
        raise KeyError(f"{yol.name}: {eksik} kolonlari yok; beklenen sema ilce_key + tarih.")
    tablo = tablo.drop(columns=[k for k in drop_columns if k in tablo.columns])
    tablo = tablo.rename(columns={"ilce_key": "_dis_anahtar", "tarih": "_dis_tarih"})
    tablo["_dis_tarih"] = pd.to_datetime(tablo["_dis_tarih"]).dt.normalize()
    tablo = tablo.drop_duplicates(subset=["_dis_anahtar", "_dis_tarih"])

    cikti = frame.copy()
    cikti["_dis_anahtar"] = cikti[key_column].astype(str)
    cikti["_dis_tarih"] = pd.to_datetime(cikti[time_column]).dt.normalize()
    oncesi = len(cikti)
    yeni_kolonlar = [k for k in tablo.columns if k not in ("_dis_anahtar", "_dis_tarih")]
    cakisan = [k for k in yeni_kolonlar if k in cikti.columns]
    if cakisan:
        raise ValueError(f"'{aile}' ailesinin kolonlari panelde zaten var: {cakisan}")
    cikti = cikti.merge(
        tablo, on=["_dis_anahtar", "_dis_tarih"], how="left", validate="many_to_one"
    )
    if len(cikti) != oncesi:
        raise RuntimeError(
            f"'{aile}' birlestirmesi satir sayisini degistirdi ({oncesi} -> {len(cikti)})."
        )
    oran = _eslesme_orani(cikti, frame, yeni_kolonlar[0], aile)
    return cikti.drop(columns=["_dis_anahtar", "_dis_tarih"]), yeni_kolonlar, oran


def _statik_ilce_ekle(
    sonuc: ExternalAttachment,
    yol: Path,
    *,
    key_column: str,
    drop_columns: Sequence[str],
    aile: str,
) -> None:
    """Ilce basina TEK satirlik, zaman boyutsuz tabloyu panele baglar.

    NEDEN AYRI BIR YOL: zamanli ailelerde ufuk kaydirmasi ve ambargo kapisi
    yanlis bir join'i er ge yakalar. Statik tabloda oyle bir kapi YOKTUR --
    zaman ekseni olmadigi icin sizinti denetimi devreye girmez. Bu yuzden
    burada uc kontrol elle yapilir:

      1. **Tekillik**: kaynakta ilce basina birden fazla satir varsa merge
         paneli COKLAR ve hedef sessizce tekrarlanir. ``validate="many_to_one"``
         bunu hata yapar.
      2. **Satir sayisi**: birlestirme sonrasi satir sayisi degismemeli.
      3. **Eslesme orani**: %0 ise ValueError (bkz. ``_eslesme_orani``).
         Statikte bu ozellikle sinsidir: tum ilceler ayni NaN'i alir ve tablo
         "dolu ama etkisiz" gorunur.
    """
    tablo = pd.read_parquet(yol)
    if "ilce_key" not in tablo.columns:
        raise KeyError(
            f"{yol.name}: 'ilce_key' kolonu yok; statik tablo semasi ilce_key + feature."
        )
    tablo = tablo.drop(columns=[k for k in drop_columns if k in tablo.columns])
    tablo = tablo.rename(columns={"ilce_key": "_dis_anahtar"})
    tablo["_dis_anahtar"] = tablo["_dis_anahtar"].astype(str)

    cikti = sonuc.frame.copy()
    cikti["_dis_anahtar"] = cikti[key_column].astype(str)
    oncesi = len(cikti)
    yeni_kolonlar = [k for k in tablo.columns if k != "_dis_anahtar"]
    if not yeni_kolonlar:
        raise ValueError(f"'{aile}': dusulen kolonlardan sonra feature kalmadi ({yol.name}).")
    cakisan = [k for k in yeni_kolonlar if k in cikti.columns]
    if cakisan:
        raise ValueError(f"'{aile}' ailesinin kolonlari panelde zaten var: {cakisan}")

    cikti = cikti.merge(tablo, on="_dis_anahtar", how="left", validate="many_to_one")
    if len(cikti) != oncesi:
        raise RuntimeError(
            f"'{aile}' birlestirmesi satir sayisini degistirdi ({oncesi} -> {len(cikti)})."
        )
    oran = _eslesme_orani(cikti, sonuc.frame, yeni_kolonlar[0], aile)
    sonuc.frame = cikti.drop(columns=["_dis_anahtar"])
    sonuc.families[aile] = yeni_kolonlar
    sonuc.match_rates[aile] = oran


def _il_anahtari_ekle(frame: pd.DataFrame, key_column: str, root: Path) -> pd.Series | None:
    """Panelin ilce anahtarindan il anahtarini referans tablodan turetir."""
    ref_yolu = root / REFERENCE_RELATIVE
    if not ref_yolu.exists():
        return None
    ref = pd.read_parquet(ref_yolu)
    if not {"ilce_key", "il_key"} <= set(ref.columns):
        return None
    esleme = dict(zip(ref["ilce_key"].astype(str), ref["il_key"].astype(str), strict=True))
    return frame[key_column].astype(str).map(esleme)


def attach_external(
    panel: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    horizon: int,
    families: Sequence[str] | None = None,
    root: Path | str = ".",
    coordinates: pd.DataFrame | None = None,
    population: pd.DataFrame | None = None,
) -> ExternalAttachment:
    """Harici kaynaklari panele baglar. YENI frame dondurur; girdi degismez.

    Args:
        panel: ``key_column`` (ilce anahtari) ve ``time_column`` (gun) tasiyan
            panel. Satir sirasi KORUNUR -- fold indeksleri gecerli kalir.
        horizon: Tahmin ufku (gun). Sonradan yayimlanan kaynaklar bu kadar
            kaydirilir; ufuk 1'den kucuk olamaz.
        families: Yalnizca bu aileler baglanir (varsayilan: hepsi).
        root: Repo koku (parquet yollari buna gore cozulur).
        coordinates: Nokta olaylar icin ``ilce_key, lat, lon``; verilmezse
            referans tablodan okunur.
        population: Yillik turizm normalizasyonu icin ``ilce_key, nufus``.

    Returns:
        ``ExternalAttachment``: frame, aile->kolon haritasi, atlananlar,
        eslesme oranlari.

    Raises:
        ValueError: ``horizon < 1``, bir aile %0 eslesti veya kolon cakismasi.
        KeyError: Panelde anahtar/zaman kolonu yok.
    """
    if horizon < 1:
        raise ValueError(f"horizon >= 1 olmali, verilen: {horizon}")
    for kolon in (key_column, time_column):
        if kolon not in panel.columns:
            raise KeyError(f"panel icinde '{kolon}' kolonu yok.")

    kok = Path(root)
    secilen = tuple(families) if families is not None else EXTERNAL_FAMILIES
    bilinmeyen = [ad for ad in secilen if ad not in EXTERNAL_FAMILIES]
    if bilinmeyen:
        raise ValueError(f"Bilinmeyen aile: {bilinmeyen}. Secenekler: {list(EXTERNAL_FAMILIES)}")

    sonuc = ExternalAttachment(frame=panel.copy())
    for aile in EXTERNAL_FAMILIES:
        if aile not in secilen:
            continue
        oncesi = list(sonuc.frame.columns)
        try:
            _aile_ekle(
                sonuc,
                aile,
                kok=kok,
                key_column=key_column,
                time_column=time_column,
                horizon=horizon,
                coordinates=coordinates,
                population=population,
            )
        except FileNotFoundError as hata:
            sonuc.skipped[aile] = f"kaynak dosya yok ({hata})"
            continue
        except KapsamBoslugu as hata:
            # Kaynak saglam, anahtar saglam -- yalnizca donem ortusmuyor.
            # Tek aile yuzunden tum hatti durdurmak yanlis olur; atla ve BILDIR.
            sonuc.skipped[aile] = f"zaman kapsami ortusmuyor ({hata})"
            continue
        yeni = [k for k in sonuc.frame.columns if k not in oncesi]
        if yeni and aile not in sonuc.families:
            sonuc.families[aile] = yeni
    return sonuc


def _aile_ekle(
    sonuc: ExternalAttachment,
    aile: str,
    *,
    kok: Path,
    key_column: str,
    time_column: str,
    horizon: int,
    coordinates: pd.DataFrame | None,
    population: pd.DataFrame | None,
) -> None:
    """Tek aileyi ``sonuc.frame``e ekler (yerinde gunceller)."""
    gunluk = {ad: (yol, dus) for ad, yol, dus in _GUNLUK_ILCE_TABLOLARI}
    if aile in gunluk:
        gorece, dus = gunluk[aile]
        yol = kok / gorece
        if not yol.exists():
            raise FileNotFoundError(str(yol))
        frame, kolonlar, oran = _gunluk_ilce_ekle(
            sonuc.frame,
            yol,
            key_column=key_column,
            time_column=time_column,
            drop_columns=dus,
            aile=aile,
        )
        if aile == "hava" and "sicaklik_ort" in frame.columns:
            # Fiziksel turevler (derece-gun birikimi, sicaklik degisimi, ruzgar
            # x yagis etkilesimi) hava ailesinin PARCASIDIR: ayni kaynaktan
            # turer ve ablasyonda ayri sayilmamalidir. ablation_gercek'in eski
            # yerel ``_hava`` adimi bunu yapiyordu; orkestratore tasindi.
            oncesi_kolonlar = set(frame.columns)
            frame = add_physical_derivatives(
                frame, group_columns=[key_column], time_column=time_column
            )
            kolonlar = kolonlar + [k for k in frame.columns if k not in oncesi_kolonlar]
        sonuc.frame, sonuc.families[aile], sonuc.match_rates[aile] = frame, kolonlar, oran
        return

    statik = {ad: (yol, dus) for ad, yol, dus in _STATIK_ILCE_TABLOLARI}
    if aile in statik:
        gorece, dus = statik[aile]
        yol = kok / gorece
        if not yol.exists():
            raise FileNotFoundError(str(yol))
        _statik_ilce_ekle(sonuc, yol, key_column=key_column, drop_columns=dus, aile=aile)
        return

    if aile in {"yangin", "deprem"}:
        _nokta_olay_ekle(
            sonuc,
            aile,
            kok=kok,
            key_column=key_column,
            time_column=time_column,
            horizon=horizon,
            coordinates=coordinates,
        )
        return

    if aile == "gunes":
        _gunes_ekle(sonuc, kok=kok, key_column=key_column, time_column=time_column)
    elif aile == "turizm_yillik":
        _turizm_yillik_ekle(
            sonuc, kok=kok, key_column=key_column, time_column=time_column, population=population
        )
    elif aile == "turizm_aylik":
        _turizm_aylik_ekle(sonuc, kok=kok, key_column=key_column, time_column=time_column)
    elif aile == "turizm_il_aylik":
        _turizm_il_aylik_ekle(sonuc, kok=kok, key_column=key_column, time_column=time_column)
    elif aile == "izsu":
        _izsu_ekle(sonuc, kok=kok, key_column=key_column, time_column=time_column)
    elif aile == "epias":
        _epias_ekle(sonuc, kok=kok, time_column=time_column, horizon=horizon)


def _gunes_ekle(sonuc: ExternalAttachment, *, kok: Path, key_column: str, time_column: str) -> None:
    """Gunes geometrisi: anahtar "il|ilce" bicimindedir, ilce parcasi alinir."""
    yol = kok / "data/external/gunes_gunluk.parquet"
    if not yol.exists():
        raise FileNotFoundError(str(yol))
    tablo = pd.read_parquet(yol)
    tablo["ilce_key"] = tablo["anahtar"].astype(str).str.split("|").str[-1]
    tablo = tablo.drop(columns=["anahtar"])
    frame, kolonlar, oran = _gunluk_ilce_ekle_tablo(
        sonuc.frame, tablo, key_column=key_column, time_column=time_column, aile="gunes"
    )
    sonuc.frame, sonuc.families["gunes"], sonuc.match_rates["gunes"] = frame, kolonlar, oran


def _gunluk_ilce_ekle_tablo(
    frame: pd.DataFrame,
    tablo: pd.DataFrame,
    *,
    key_column: str,
    time_column: str,
    aile: str,
) -> tuple[pd.DataFrame, list[str], float]:
    """``_gunluk_ilce_ekle``in bellekteki tablo alan surumu."""
    tablo = tablo.rename(columns={"ilce_key": "_dis_anahtar", "tarih": "_dis_tarih"})
    tablo["_dis_tarih"] = pd.to_datetime(tablo["_dis_tarih"]).dt.normalize()
    tablo = tablo.drop_duplicates(subset=["_dis_anahtar", "_dis_tarih"])
    cikti = frame.copy()
    cikti["_dis_anahtar"] = cikti[key_column].astype(str)
    cikti["_dis_tarih"] = pd.to_datetime(cikti[time_column]).dt.normalize()
    yeni_kolonlar = [k for k in tablo.columns if k not in ("_dis_anahtar", "_dis_tarih")]
    cakisan = [k for k in yeni_kolonlar if k in cikti.columns]
    if cakisan:
        raise ValueError(f"'{aile}' ailesinin kolonlari panelde zaten var: {cakisan}")
    oncesi = len(cikti)
    cikti = cikti.merge(
        tablo, on=["_dis_anahtar", "_dis_tarih"], how="left", validate="many_to_one"
    )
    if len(cikti) != oncesi:
        raise RuntimeError(f"'{aile}' birlestirmesi satir sayisini degistirdi.")
    oran = _eslesme_orani(cikti, frame, yeni_kolonlar[0], aile)
    return cikti.drop(columns=["_dis_anahtar", "_dis_tarih"]), yeni_kolonlar, oran


def _koordinatlari_coz(kok: Path, coordinates: pd.DataFrame | None) -> pd.DataFrame:
    if coordinates is not None:
        return coordinates
    ref_yolu = kok / REFERENCE_RELATIVE
    if not ref_yolu.exists():
        raise FileNotFoundError(str(ref_yolu))
    ref = pd.read_parquet(ref_yolu)
    return ref[["ilce_key", "lat", "lon"]]


def _nokta_olay_ekle(
    sonuc: ExternalAttachment,
    aile: str,
    *,
    kok: Path,
    key_column: str,
    time_column: str,
    horizon: int,
    coordinates: pd.DataFrame | None,
) -> None:
    """Yangin/deprem katalogunu yaricap filtreli, ufuk kaydirmali baglar."""
    kayit = {ad: (yol, agirlik, yaricaplar) for ad, yol, agirlik, yaricaplar in _NOKTA_OLAYLAR}
    gorece, agirlik_kolonu, yaricaplar = kayit[aile]
    yol = kok / gorece
    if not yol.exists():
        raise FileNotFoundError(str(yol))
    olaylar = pd.read_parquet(yol)
    olaylar = olaylar.rename(columns={"tarih": "_olay_tarih"})
    olaylar["_olay_tarih"] = pd.to_datetime(olaylar["_olay_tarih"])
    koordinatlar = _koordinatlari_coz(kok, coordinates)
    koordinatlar = koordinatlar.rename(columns={"ilce_key": key_column})

    oncesi = list(sonuc.frame.columns)
    sonuc.frame = add_point_event_features(
        sonuc.frame,
        olaylar,
        koordinatlar,
        key_column=key_column,
        time_column=time_column,
        event_time_column="_olay_tarih",
        horizon=horizon,
        radii_km=yaricaplar,
        weight_column=agirlik_kolonu if agirlik_kolonu in olaylar.columns else None,
        prefix=aile,
    )
    sonuc.families[aile] = [k for k in sonuc.frame.columns if k not in oncesi]


def _yil_kapsamini_dogrula(
    zaman: pd.Series,
    kaynak_tablo: pd.DataFrame,
    *,
    yil_lag: int,
    aile: str,
    kaynak: str,
) -> None:
    """Panelin ihtiyac duydugu yillar kaynakta VAR MI -- merge'den once bakar.

    Raises:
        KapsamBoslugu: hic ortusen yil yok. Anahtar sorunu DEGIL, donem sorunu.
    """
    if "yil" not in kaynak_tablo.columns:
        return
    # int()'e cevir: numpy skalerleri repr'de "np.int32(2020)" olarak basilir
    # ve hata mesajini okunmaz hale getirir.
    gereken = {int(y) - yil_lag for y in pd.to_datetime(zaman).dt.year.unique()}
    mevcut = {int(y) for y in kaynak_tablo["yil"].dropna().unique()}
    if gereken & mevcut:
        return
    raise KapsamBoslugu(
        f"'{aile}': panel {sorted(gereken)} yillarini istiyor (yil_lag={yil_lag}), "
        f"{kaynak} yalnizca {sorted(mevcut)} tasiyor. Anahtarlar degil DONEM ortusmuyor."
    )


def _turizm_yillik_ekle(
    sonuc: ExternalAttachment,
    *,
    kok: Path,
    key_column: str,
    time_column: str,
    population: pd.DataFrame | None,
) -> None:
    yol = kok / "data/external/turizm_geceleme.parquet"
    if not yol.exists():
        raise FileNotFoundError(str(yol))
    yillik = pd.read_parquet(yol)
    # KAPSAM KONTROLU MERGE'DEN ONCE. ``add_annual_district_attribute``
    # year_lag=1 kullanir: panelin YIL-1 degeri kaynakta olmali. Ortusme
    # yoksa merge %0 doner ve "anahtar bozuk" hatasi verirdi -- oysa anahtar
    # saglam, kapsam farkli. Ayrimi burada, iki tarafi da gorurken kuruyoruz.
    _yil_kapsamini_dogrula(
        sonuc.frame[time_column], yillik, yil_lag=1, aile="turizm_yillik", kaynak=yol.name
    )
    nufus = population
    if nufus is None:
        ref_yolu = kok / REFERENCE_RELATIVE
        if ref_yolu.exists():
            ref = pd.read_parquet(ref_yolu)
            if {"ilce_key", "nufus"} <= set(ref.columns):
                nufus = ref[["ilce_key", "nufus"]]
    oncesi = list(sonuc.frame.columns)
    sonuc.frame = add_annual_district_attribute(
        sonuc.frame,
        yillik,
        key_column=key_column,
        annual_key_column="ilce_key",
        time_column=time_column,
        value_columns=["geceleme", "tesise_gelis"],
        population=nufus,
        population_key_column="ilce_key" if nufus is not None else None,
        prefix="turizm",
    )
    sonuc.families["turizm_yillik"] = [k for k in sonuc.frame.columns if k not in oncesi]


def _turizm_aylik_ekle(
    sonuc: ExternalAttachment, *, kok: Path, key_column: str, time_column: str
) -> None:
    """Aylik il serisini ILCE tahminine cevirip lag-12 ile baglar."""
    aylik_yolu = kok / "data/external/turizm_aylik_il.parquet"
    yillik_yolu = kok / "data/external/turizm_geceleme.parquet"
    if not aylik_yolu.exists() or not yillik_yolu.exists():
        raise FileNotFoundError(str(aylik_yolu if not aylik_yolu.exists() else yillik_yolu))
    aylik = pd.read_parquet(aylik_yolu)
    yillik = pd.read_parquet(yillik_yolu)
    # Ilce tahmini YILLIK tablodan turedigi icin onun kapsamina hapsolur;
    # ayrica lag_months=12 bir yil geriye bakar. Iki kisit da yil duzeyinde
    # ayni kontrolle yakalanir.
    _yil_kapsamini_dogrula(
        sonuc.frame[time_column], yillik, yil_lag=1, aile="turizm_aylik", kaynak=yillik_yolu.name
    )
    ref_yolu = kok / REFERENCE_RELATIVE
    ilceler = pd.read_parquet(ref_yolu)[["il_key", "ilce_key"]] if ref_yolu.exists() else None
    tahmin = district_monthly_estimate(yillik, aylik, districts=ilceler)
    oncesi = list(sonuc.frame.columns)
    sonuc.frame = add_monthly_attribute(
        sonuc.frame,
        tahmin,
        key_column=key_column,
        monthly_key_column="ilce_key",
        time_column=time_column,
        value_columns=["geceleme_tahmini"],
        lag_months=12,
        prefix="turizm_ay",
    )
    sonuc.families["turizm_aylik"] = [k for k in sonuc.frame.columns if k not in oncesi]


#: Il aylik turizminden panele TASINAN tek olcu: DOLULUK ORANI.
#:
#: NEDEN SADECE BU, ham "geceleme" degil (OLCULDU 2026-08-20):
#: ``turizm_aylik_il`` uc farkli KAPSAM REJIMI tasiyor -- KTB'nin hangi tesis
#: turlerini saydigi 2022 Eylul'de ve 2025 Temmuz'da degisti:
#:
#:     rejim 1: 2019-2022/08  (isletme belgeli)
#:     rejim 2: 2022/09-2025/06 (isletme + isletme_basit)
#:     rejim 3: 2025/07-2026   (isletme_basit)
#:
#: Rejim sinirinda seviye TANIMSAL olarak ziplar. Ege 5 ilinde ay bazinda
#: yillik oranla olculdu (rejim degismeyen aylar kontrol grubu):
#:
#:     ham geceleme : 1.31x kirilma
#:     yil_payi     : 1.31x kirilma  (gecis yilinda yil toplami iki rejimi karistiriyor)
#:     DOLULUK      : 0.92x          -- pratikte kirilma yok
#:
#: 1.31x'lik bir siçrama tam 2025/07'de, yani bir yarismada test blogunun
#: oturdugu yerde baslar. Model onu "turizm patladi" diye okur; gercekte
#: yalnizca sayim tanimi genisledi. Doluluk ORAN oldugu icin pay ve payda
#: birlikte genisler ve kirilma buyuk olcude sadelesir.
#:
#: Bu yuzden ``add_year_share=False``: yil payi turevi de kirilmayi tasiyor.
IL_AYLIK_TURIZM_KOLONLARI = ["doluluk"]


def _turizm_il_aylik_ekle(
    sonuc: ExternalAttachment, *, kok: Path, key_column: str, time_column: str
) -> None:
    """Il aylik turizm dolulugunu panele baglar -- ILCE tablosuna BAGIMSIZ.

    NEDEN AYRI BIR AILE: ``turizm_aylik`` ailesi, il aylik serisini
    ``turizm_geceleme`` (ILCE, yalnizca 2023-2025) ile carparak ilce tahmini
    uretir; bu yuzden ilce tablosunun dar kapsamina HAPSOLUR. Olculdu
    (2026-08-20): turizm feature'lari 2020-2023 panel satirlarinin %0'inda,
    2024-2026'nin %100'unde doluydu.

    Bu, kapsam boslugunun TEHLIKELI cesididir: egitimin ilk dort yili
    tamamen bos, test blogu tamamen dolu. Model feature'i yalnizca son iki
    yildan ogrenir ve o iki yilin rejimine asiri uyum saglar.

    Il serisi (81 il, 2019-2026, Ege 5 ilinde %0 NaN) bu kisiti tasimaz:
    ilce tablosuna hic dokunmadan panelin TAMAMINI kapsar. Ilce cozunurlugu
    kaybi bilincli takas -- bos bir feature'in cozunurlugu zaten yoktur.
    """
    aylik_yolu = kok / "data/external/turizm_aylik_il.parquet"
    if not aylik_yolu.exists():
        raise FileNotFoundError(str(aylik_yolu))
    aylik = pd.read_parquet(aylik_yolu)

    ref_yolu = kok / REFERENCE_RELATIVE
    if not ref_yolu.exists():
        raise FileNotFoundError(str(ref_yolu))
    ilceler = pd.read_parquet(ref_yolu)[["ilce_key", "il_key"]]

    calisma = sonuc.frame.copy()
    calisma["_il_key"] = (
        calisma[key_column].astype(str).map(ilceler.set_index("ilce_key")["il_key"].astype(str))
    )
    eslesmeyen = int(calisma["_il_key"].isna().sum())
    if eslesmeyen:
        raise ValueError(
            f"{eslesmeyen} panel satirinin ilce anahtari referans tablosunda yok; "
            "il eslemesi yapilamadi. Sessiz NaN yerine durduruldu."
        )

    oncesi = list(sonuc.frame.columns)
    calisma = add_monthly_attribute(
        calisma,
        aylik,
        key_column="_il_key",
        monthly_key_column="il_key",
        time_column=time_column,
        value_columns=IL_AYLIK_TURIZM_KOLONLARI,
        lag_months=12,
        add_year_share=False,
        prefix="turizm_il",
    )
    sonuc.frame = calisma.drop(columns=["_il_key"])
    sonuc.families["turizm_il_aylik"] = [k for k in sonuc.frame.columns if k not in oncesi]


def _izsu_ekle(sonuc: ExternalAttachment, *, kok: Path, key_column: str, time_column: str) -> None:
    yol = kok / "data/external/izsu_su_profili.parquet"
    if not yol.exists():
        raise FileNotFoundError(str(yol))
    profil = pd.read_parquet(yol)
    oncesi = list(sonuc.frame.columns)
    sonuc.frame = add_seasonal_district_profile(
        sonuc.frame,
        profil,
        key_column=key_column,
        profile_key_column="ilce_key",
        time_column=time_column,
        prefix="su",
    )
    sonuc.families["izsu"] = [k for k in sonuc.frame.columns if k not in oncesi]


def _epias_ekle(sonuc: ExternalAttachment, *, kok: Path, time_column: str, horizon: int) -> None:
    """Ulusal saatlik tuketimi gunluge indirip ufuk kaydirmali baglar."""
    yol = kok / "data/external/epias/tuketim_saatlik.parquet"
    if not yol.exists():
        raise FileNotFoundError(str(yol))
    saatlik = pd.read_parquet(yol)
    zaman_kolonu = "zaman" if "zaman" in saatlik.columns else saatlik.columns[0]
    deger_kolonlari = [k for k in saatlik.columns if k != zaman_kolonu]
    gunluk = daily_from_hourly(
        saatlik,
        time_column=zaman_kolonu,
        value_columns=deger_kolonlari,
        aggregations=("mean", "max"),
    )
    oncesi = list(sonuc.frame.columns)
    sonuc.frame = add_national_series(
        sonuc.frame,
        gunluk,
        time_column=time_column,
        horizon=horizon,
        prefix="tr",
    )
    sonuc.families["epias"] = [k for k in sonuc.frame.columns if k not in oncesi]
