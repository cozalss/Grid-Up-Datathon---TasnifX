"""Saatlik koprünün sozlesmeleri.

``scripts/kopru_saatlik.py`` uc panel tablosunu forecast API'siyle gelecege
uzatir. Koprünün iki sessiz bozulma yolu vardir ve buradaki testler ikisini
de kapatir:

  1. Tahmin satirlari arsiv satirlarini EZERSE veri bilerek kotulesir
     (ERA5 tahminden dogrudur).
  2. Dikis kontrolu yalnizca gorunuste calisirsa -- her zaman gecen bir
     kontrol, kontrolsuzlukla aynidir.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]


def _modul():
    sys.path.insert(0, str(KOK / "scripts"))
    sys.path.insert(0, str(KOK / "src"))
    yol = KOK / "scripts" / "kopru_saatlik.py"
    spec = importlib.util.spec_from_file_location("kopru_saatlik", yol)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    sys.modules["kopru_saatlik"] = modul
    spec.loader.exec_module(modul)
    return modul


def _tablo(bas: str, son: str, deger: float, ilceler=("a", "b")) -> pd.DataFrame:
    gunler = pd.date_range(bas, son, freq="D")
    satirlar = [
        {"ilce_key": k, "tarih": g, "basinc_ort": deger, "nem_ort": deger}
        for k in ilceler
        for g in gunler
    ]
    return pd.DataFrame(satirlar)


def test_arsiv_satirlari_tahminle_ezilmez() -> None:
    """Ortusen gunlerde ARSIV kazanir.

    ERA5 yeniden analizi tahminden dogrudur; ortusen bir gunde tahmini
    tercih etmek veriyi bilerek kotulestirmek olurdu. Bu test degerin
    arsivinki oldugunu birebir dogrular.
    """
    modul = _modul()
    arsiv = _tablo("2026-08-01", "2026-08-10", deger=1000.0)
    tahmin = _tablo("2026-08-08", "2026-08-20", deger=999.0)

    birlesik = modul.kopruyu_birlestir(arsiv, tahmin)

    ortusen = birlesik[(birlesik["tarih"] >= "2026-08-08") & (birlesik["tarih"] <= "2026-08-10")]
    assert (ortusen["basinc_ort"] == 1000.0).all(), (
        "Ortusen gunlerde tahmin degeri arsivi ezdi -- veri bilerek kotulesti."
    )
    assert (ortusen[modul.TAHMIN_KOLONU] == 0).all(), "Arsiv satiri tahmin diye isaretlendi."


def test_yalnizca_eksik_gunler_ekleniyor() -> None:
    """Satir sayisi tam olarak eklenen GUN kadar artmali -- ne fazla ne az."""
    modul = _modul()
    arsiv = _tablo("2026-08-01", "2026-08-10", deger=1000.0)
    tahmin = _tablo("2026-08-08", "2026-08-20", deger=999.0)

    birlesik = modul.kopruyu_birlestir(arsiv, tahmin)

    # 08-11 .. 08-20 = 10 gun x 2 ilce
    assert len(birlesik) == len(arsiv) + 20
    assert not birlesik.duplicated(subset=["ilce_key", "tarih"]).any()
    yeni = birlesik[birlesik[modul.TAHMIN_KOLONU] == 1]
    assert yeni["tarih"].min() == pd.Timestamp("2026-08-11")
    assert (yeni["basinc_ort"] == 999.0).all()


def test_dikis_farki_ayrisan_veriyi_buyuk_gosteriyor() -> None:
    """Kontrolun ISE YARADIGININ kaniti: uyusan veri kucuk, ayrisan buyuk fark.

    Her zaman 0 donen bir fonksiyon da "dikis gecti" derdi.
    """
    modul = _modul()
    kopru = modul.Kopru(
        ad="test",
        yol="yok",
        degiskenler=(),
        topla=lambda f, k: f,
        dikis_kolonu="basinc_ort",
        dikis_toleransi=2.0,
    )
    arsiv = _tablo("2026-08-01", "2026-08-10", deger=1000.0)

    uyusan = _tablo("2026-08-08", "2026-08-20", deger=1000.5)
    assert modul.dikis_farki(arsiv, uyusan, kopru) == pytest.approx(0.5)

    ayrisan = _tablo("2026-08-08", "2026-08-20", deger=1050.0)
    assert modul.dikis_farki(arsiv, ayrisan, kopru) == pytest.approx(50.0)


def test_ortusme_yoksa_dikis_kontrolu_hata_verir() -> None:
    """Ortusmeyen bir koprü, kontrolsuz birlestirmedir -- sessizce gecmemeli."""
    modul = _modul()
    kopru = modul.Kopru(
        ad="test",
        yol="yok",
        degiskenler=(),
        topla=lambda f, k: f,
        dikis_kolonu="basinc_ort",
        dikis_toleransi=2.0,
    )
    arsiv = _tablo("2026-08-01", "2026-08-10", deger=1000.0)
    kopuk = _tablo("2026-08-15", "2026-08-20", deger=1000.0)

    with pytest.raises(ValueError, match="ortusmuyor"):
        modul.dikis_farki(arsiv, kopuk, kopru)


def test_degisken_birlesimi_tekil_ve_tam() -> None:
    """Ilce basina TEK istek: uc koprünün degiskenleri tekillestirilmeli.

    Kopyali bir birlesim istegi buyutur ve kotayi gereksiz yakar; eksik bir
    birlesim ise bir tablonun toplama fonksiyonunu KeyError ile dusurur.
    """
    modul = _modul()
    kopruler = modul._kopruleri_kur()
    birlesim = modul.tum_degiskenler(kopruler)

    assert len(birlesim) == len(set(birlesim)), "Birlesimde tekrar var."
    for kopru in kopruler:
        eksik = [d for d in kopru.degiskenler if d not in birlesim]
        assert not eksik, f"{kopru.ad} icin birlesimde eksik degisken: {eksik}"


def test_kopruler_arsiv_cekicilerinin_kendi_fonksiyonunu_kullanir() -> None:
    """Toplama mantigi KOPYALANMAMALI.

    Kopyalanmis bir toplama, esikler degistiginde sessizce ayrisir ve koprü
    satirlari arsiv satirlarindan BASKA bir sey olcmeye baslar. Bu depoda
    2026-08-20'de tam olarak bu hata bulundu: geri cekilme merdiveninin alti
    kopyasi vardi ve duzeltme hicbirine ulasmadi (docs/17, madde 3.4).
    """
    modul = _modul()
    saatlik = modul._modul("fetch_hourly_weather")
    konvektif = modul._modul("fetch_konvektif")
    nem = modul._modul("fetch_nem_toprak")
    hava_kalitesi = modul._modul("fetch_hava_kalitesi")
    beklenen = {
        "hava_saatlik_turev": saatlik.aggregate_daily,
        "konvektif_gunluk": konvektif.gunluge_indir,
        "nem_toprak_gunluk": nem.gunluge_indir,
        "hava_kalitesi_gunluk": hava_kalitesi.gunluge_indir,
    }
    for kopru in modul._kopruleri_kur():
        assert kopru.topla is beklenen[kopru.ad], (
            f"{kopru.ad} kendi toplama fonksiyonunu kullanmiyor -- kopya mantik riski."
        )


def test_ileri_ufuk_en_zayif_kaynaga_gore_kirpiliyor() -> None:
    """Panel EN ZAYIF kaynagi kadar uzar -- daha fazla degil.

    Hava kalitesi API'si ileriye yalnizca 7 gun verir (olculdu 2026-08-20),
    hava/toprak/konvektif 16 gun. Havayi +16'ya uzatip hava kalitesini +7'de
    birakmak, 8..16. gunlerde tam olarak kacinmaya calistigimiz asimetriyi
    yeniden kurardi: o araliktaki her satirda hava ailesi DOLU, hava kalitesi
    ailesi BOS olurdu.
    """
    modul = _modul()
    kopruler = modul._kopruleri_kur()

    ufuk, not_ = modul.ileri_ufuk(kopruler, 16)
    assert ufuk == modul.MAX_AIR_QUALITY_DAYS, f"Ufuk en zayif kaynaga kirpilmadi: {ufuk} gun"
    assert "hava_kalitesi" in not_, "Kirpmayi yapan kaynak raporlanmiyor."

    # Tavanin altindaki bir istek KIRPILMAZ ve gereksiz not uretmez.
    ufuk, not_ = modul.ileri_ufuk(kopruler, 3)
    assert (ufuk, not_) == (3, "")


def test_ayni_uc_nokta_tek_istekte_gruplaniyor() -> None:
    """Ilce basina uc nokta SAYISI kadar istek atilmali, koprü sayisi kadar degil."""
    modul = _modul()
    gruplar = modul.uc_noktaya_gore(modul._kopruleri_kur())

    assert len(gruplar) == 2, f"Beklenen iki uc nokta, gelen: {list(gruplar)}"
    assert len(gruplar[modul.FORECAST_URL]) == 3, "Hava/toprak/konvektif ayni istekte olmali."
    assert len(gruplar[modul.AIR_QUALITY_URL]) == 1


def test_arsiv_ucu_tahmin_satirlarini_saymiyor() -> None:
    """``past_days`` referansi TAHMIN OLMAYAN son gun olmali.

    OLCULDU 2026-08-20: koprü IKINCI kez kosunca tablolar zaten gelecege
    uzaniyordu (2026-08-26) ve ``bugun - tablo_ucu`` NEGATIF cikti ->
    ``past_days=-3`` -> API her ilce icin HTTP 400.

    Yani betik ILK kosuda calisip IKINCI kosuda tamamen bozuluyordu. Yarisma
    gunu yapilacak sey tam olarak ikinci kosudur: veri tazelenir, koprü
    yeniden kurulur. Bu test o senaryoyu birebir kurar.
    """
    modul = _modul()
    arsiv = _tablo("2026-08-01", "2026-08-10", deger=1000.0)
    tahmin = _tablo("2026-08-11", "2026-08-26", deger=999.0)
    ikinci_kosu = modul.kopruyu_birlestir(arsiv, tahmin)

    assert ikinci_kosu["tarih"].max() == pd.Timestamp("2026-08-26"), "test kurgusu hatali"
    assert modul._arsiv_ucu(ikinci_kosu) == pd.Timestamp("2026-08-10"), (
        "Arsiv ucu, tahmin satirlarindan etkilendi -- past_days negatife duser."
    )
    # Bayrak yoksa tum tablonun son gunu kullanilir (ilk kosu durumu).
    assert modul._arsiv_ucu(arsiv) == pd.Timestamp("2026-08-10")


def test_dikis_kontrolu_tahmin_satirlarini_kendisiyle_kiyaslamaz() -> None:
    """Dikis ARSIVLE olculur; onceki tahmini yeni tahminle kiyaslamak
    farki yapay olarak kucultur ve kontrolu anlamsizlastirir.

    Burada arsiv 1000, ESKI tahmin 999, YENI tahmin 1050. Kontrol arsivle
    yapilirsa fark 50 gorunur (ve tolerans asilir); eski tahminle yapilirsa
    ortusme sadece tahmin satirlarina duser ve gercek ayrisma gizlenir.
    """
    modul = _modul()
    kopru = modul.Kopru(
        ad="test",
        yol="yok",
        degiskenler=(),
        topla=lambda f, k: f,
        dikis_kolonu="basinc_ort",
        dikis_toleransi=2.0,
    )
    arsiv = _tablo("2026-08-01", "2026-08-10", deger=1000.0)
    eski_tahmin = _tablo("2026-08-11", "2026-08-20", deger=999.0)
    onceki_kosu = modul.kopruyu_birlestir(arsiv, eski_tahmin)

    yeni_tahmin = _tablo("2026-08-08", "2026-08-26", deger=1050.0)
    fark = modul.dikis_farki(onceki_kosu, yeni_tahmin, kopru)

    assert fark == pytest.approx(50.0), (
        f"Dikis farki {fark}; arsiv satirlariyla olculmemis olabilir."
    )
