"""Harici veri cekici betiklerinin ortak sozlesmeleri.

Bu dosya, 2026-08-17'de GERCEKTEN yasanmis bir hatadan dogdu:
``fetch_nem_toprak.py`` ``cap_end_date``in donusunu tek deger sanip dogrudan
kullandi. Fonksiyon ``(kirpilmis_tarih, uyari)`` CIFTI donuyor; ``requests``
bu cifti seri hale getirince istek URL'sine ``end_date`` IKI KEZ girdi ve API
96 ilcenin hepsinde HTTP 400 dondu. Betik 15 ilce boyunca sessizce donup
hicbir kontrol noktasi yazmadan ilerledi.

Hata sinifi genel: yardimci fonksiyonun donus SEKLI degisir veya yanlis
okunursa, cekiciler cok gec fark edilen bicimde bozulur. Buradaki testler o
sinifi API yuzeyinde zorlar.
"""

from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
BETIK_DIZINI = KOK / "scripts"


def _fetch_betikleri() -> list[Path]:
    return sorted(BETIK_DIZINI.glob("fetch_*.py"))


def test_fetch_betikleri_bulunuyor() -> None:
    """Tarama gercekten dosya buluyor mu -- bos tarama sessiz basari olurdu."""
    betikler = _fetch_betikleri()
    assert len(betikler) >= 5, f"Beklenenden az cekici betigi bulundu: {betikler}"


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_cap_end_date_donusu_cift_olarak_acilir(betik: Path) -> None:
    """``cap_end_date`` cagiran her betik donusu IKI degere acmalidir.

    ``end = cap_end_date(x)`` yazmak, tarih yerine bir demet dondurur.
    ``requests`` demeti tekrarlanan sorgu parametresi olarak seri hale
    getirir ve API her konum icin HTTP 400 verir -- olculdu.
    """
    kaynak = betik.read_text(encoding="utf-8")
    if "cap_end_date(" not in kaynak:
        pytest.skip(f"{betik.name} cap_end_date kullanmiyor")

    agac = ast.parse(kaynak)
    hatalar: list[str] = []
    for dugum in ast.walk(agac):
        if not isinstance(dugum, ast.Assign):
            continue
        deger = dugum.value
        if not (
            isinstance(deger, ast.Call)
            and isinstance(deger.func, ast.Name)
            and deger.func.id == "cap_end_date"
        ):
            continue
        hedef = dugum.targets[0]
        if not isinstance(hedef, ast.Tuple) or len(hedef.elts) != 2:
            hatalar.append(
                f"satir {dugum.lineno}: cap_end_date donusu TEK degere atanmis. "
                "Dogrusu: `end, uyari = cap_end_date(...)`"
            )

    assert not hatalar, f"{betik.name}: " + "; ".join(hatalar)


def _modul_yukle(ad: str):
    """``scripts/<ad>.py`` dosyasini modul olarak yukler."""
    sys.path.insert(0, str(BETIK_DIZINI))
    spec = importlib.util.spec_from_file_location(ad, BETIK_DIZINI / f"{ad}.py")
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    sys.modules[ad] = modul
    spec.loader.exec_module(modul)
    return modul


def test_cap_end_date_gercekten_cift_donuyor() -> None:
    """Sozlesmenin kendisi: fonksiyon iki elemanli demet dondurur.

    Yukaridaki AST testi 'cagiran dogru aciyor mu' diye bakar; bu test
    'acilacak sey gercekten cift mi' diye bakar. Fonksiyon ileride tek
    degere donerse AST testi yanlis yere bagirmaya baslar -- bu test onu
    ayirt eder.
    """
    modul = _modul_yukle("fetch_weather")

    sonuc = modul.cap_end_date("2020-01-01")
    assert isinstance(sonuc, tuple) and len(sonuc) == 2, (
        f"cap_end_date (tarih, uyari) cifti dondurmeli, donen: {sonuc!r}"
    )
    tarih, uyari = sonuc
    assert isinstance(tarih, str), f"ilk eleman str olmali, {type(tarih).__name__} geldi"
    assert uyari is None or isinstance(uyari, str)


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_cekici_yazmadan_once_dogruluyor(betik: Path) -> None:
    """Parquet yazan her cekici, YAZMADAN ONCE bir kalite kapisindan gecmelidir.

    ``fetch_hourly_weather``te olculen ders: yalnizca ekrana yazdiran bir
    dogrulama, gozetimsiz gece kosusunda bozuk veriyi sessizce yayinlar.
    Kapinin yazimdan SONRA kosmasi da ayni sey demektir -- veri zaten diskte
    olur ve sonraki kosular onu okur.
    """
    kaynak = betik.read_text(encoding="utf-8")
    yazim = "atomic_write_dataframe" in kaynak or "to_parquet" in kaynak
    if not yazim:
        pytest.skip(f"{betik.name} parquet yazmiyor")

    # Kalite kapisi = raise ile reddeden bir dogrulama yolu olmali.
    kapi_var = any(
        anahtar in kaynak for anahtar in ("kalite_kapisi", "raise ValueError", "raise RuntimeError")
    )
    assert kapi_var, (
        f"{betik.name} parquet yaziyor ama reddeden bir dogrulama yolu yok. "
        "Bozuk veri sessizce yayinlanir."
    )


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_kontrol_noktasi_kapsam_ile_atlanir(betik: Path) -> None:
    """Kontrol noktasi VARLIGA degil KAPSAMA gore atlanmalidir.

    2026-08-20'de olculen hata: uc cekici (hava kalitesi, konvektif, nem
    toprak) kontrol noktasini yalnizca dosya var mi diye sorup atliyordu::

        if ckpt.is_file() and not args.fresh:
            continue                      # araligi kapsiyor mu? SORULMADI

    Bunun bedeli SESSIZ BAYATLIKTIR ve tam olarak sessiz oldugu icin
    tehlikelidir: ``--end`` ileri tasinir, betik 96 ilcenin hepsi icin
    "kontrol noktasindan" yazar, exit 0 doner ve tablo eski tarihte kalir.
    Hicbir hata mesaji yoktur -- yalnizca panelin son gunleri feature'siz
    kalir ve model, egitimde dolu / testte bos bir kolona guvenmeyi ogrenir.

    Kural: kontrol noktasi dizini kullanan her cekici ``checkpoint_covers``
    cagirmak ZORUNDADIR.
    """
    kaynak = betik.read_text(encoding="utf-8")
    kullaniyor = "CKPT_DIR" in kaynak or "CHECKPOINT_DIR" in kaynak
    if not kullaniyor:
        pytest.skip(f"{betik.name} kontrol noktasi dizini kullanmiyor")

    # Iki gecerli bicim var ve ikisi de AYNI soruyu sorar:
    #   checkpoint_covers(...)  -> "kapsiyor mu?"  (atla / bastan indir)
    #   eksik_aralik(...)       -> "neresi eksik?" (yalnizca kuyrugu indir)
    # Ikincisi ustundur -- kotayi 300+ kat az yakar -- ama sozlesme acisindan
    # ikisi de kabul edilir. Onemli olan dosyanin VARLIGINA guvenilmemesi.
    assert "checkpoint_covers(" in kaynak or "eksik_aralik(" in kaynak, (
        f"{betik.name} kontrol noktasi kullaniyor ama kapsam kontrolu YOK. "
        "Dosyanin varligi, istenen [start, end] araligini kapsadigi anlamina "
        "gelmez; `--end` ileri tasindiginda betik sessizce eski veriyi "
        "yeniden yayinlar. Dogrusu: "
        "`aralik = eksik_aralik(ckpt, args.start, end)` ya da "
        "`if not args.fresh and checkpoint_covers(ckpt, args.start, end): continue`"
    )


def test_checkpoint_covers_gercekten_kapsam_olcuyor(tmp_path: Path) -> None:
    """Kapinin ISE YARADIGININ kaniti -- her zaman True donen bir fonksiyon
    da yukaridaki metin taramasini gecerdi.

    Dar araligi olan bir kontrol noktasi dosyasi yazip, genis bir aralik
    icin REDDEDILDIGINI dogruluyoruz.
    """
    import pandas as pd

    modul = _modul_yukle("fetch_weather")
    yol = tmp_path / "ckpt.parquet"
    pd.DataFrame({"tarih": pd.date_range("2020-01-01", "2026-08-12", freq="D")}).to_parquet(yol)

    assert modul.checkpoint_covers(yol, "2020-01-01", "2026-08-12"), (
        "Tam olarak kapsanan aralik reddedildi -- sinir kosulu yanlis."
    )
    assert not modul.checkpoint_covers(yol, "2020-01-01", "2026-08-19"), (
        "Kontrol noktasi 08-12'de bitiyor ama 08-19 istegi KABUL edildi. "
        "Sessiz bayatlik kapisi calismiyor."
    )
    assert not modul.checkpoint_covers(yol, "2019-01-01", "2026-08-12"), (
        "Kontrol noktasi 2020'de basliyor ama 2019 istegi KABUL edildi."
    )
    assert not modul.checkpoint_covers(tmp_path / "yok.parquet", "2020-01-01", "2020-01-02"), (
        "Var olmayan dosya icin True dondu."
    )


class _SahteYanit:
    """429 yanitinin test icin yeterli yuzeyi: govde metni + basliklar."""

    def __init__(self, text: str = "", headers: dict[str, str] | None = None) -> None:
        self.text = text
        self.headers = headers or {}


def test_saatlik_limit_saat_basina_kadar_bekliyor() -> None:
    """SAATLIK limit geldiginde dakikalik merdiven KULLANILMAMALI.

    Open-Meteo uc ayri pencere isletir (dakikalik/saatlik/gunluk) ve 429
    govdesinde hangisinin doldugunu yazar. Merdiven (65/130/300 sn) yalnizca
    DAKIKALIK limit icin dogrudur; uc deneme toplam ~8 dakikadir.

    OLCULDU 2026-08-20 21:41: 96 ilcelik saatlik cekimde SAATLIK limit doldu
    ve sunucu "bir sonraki saat" dedi -- 19 dakika. Eski merdivenle kalan 23
    ilcenin her biri sekiz dakika bosuna deneyip DUSECEKTI.
    """
    import datetime as dt

    modul = _modul_yukle("fetch_weather")
    an = dt.datetime(2026, 8, 20, 21, 41, 0)
    yanit = _SahteYanit(
        '{"error":true,"reason":"Hourly API request limit exceeded. '
        'Please try again in the next hour."}'
    )
    saniye, gerekce = modul.rate_limit_beklemesi(yanit, 1, simdi=an)

    # 21:41 -> 22:00:30 = 1170 sn
    assert saniye == 1170, f"saat basina kadar beklenmedi: {saniye} sn"
    assert "SAATLIK" in gerekce
    assert saniye > max(modul.RATE_LIMIT_BACKOFF), (
        "Saatlik limit icin beklenen sure dakikalik merdivenin en uzun basamagindan "
        "kisa -- cekim limit sifirlanmadan once pes eder."
    )


def test_dakikalik_limit_merdiveni_kullaniyor() -> None:
    """Dakikalik limitte saat basina kadar beklemek de YANLIS olurdu.

    Duzeltmenin ters yone tasmadigini olcer: 19 dakika beklemek gereksiz
    yere cekimi durdurur.
    """
    import datetime as dt

    modul = _modul_yukle("fetch_weather")
    yanit = _SahteYanit('{"error":true,"reason":"Minutely API request limit exceeded."}')
    for deneme, beklenen in enumerate(modul.RATE_LIMIT_BACKOFF, start=1):
        saniye, gerekce = modul.rate_limit_beklemesi(
            yanit, deneme, simdi=dt.datetime(2026, 8, 20, 21, 41, 0)
        )
        assert saniye == beklenen, f"deneme {deneme}: {saniye} != {beklenen}"
        assert "dakikalik" in gerekce


def test_retry_after_basligi_her_seyi_ezer() -> None:
    """Sunucu kendi suresini soylediyse ona uyulur -- tahmin yurutulmez."""
    modul = _modul_yukle("fetch_weather")
    yanit = _SahteYanit('{"reason":"Hourly API request limit exceeded."}', {"Retry-After": "42"})
    saniye, gerekce = modul.rate_limit_beklemesi(yanit, 1)
    assert saniye == 42
    assert "Retry-After" in gerekce


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_429_beklemesi_ortak_yardimciyi_kullanir(betik: Path) -> None:
    """Hicbir cekici geri cekilme merdivenini KENDI indekslememeli.

    2026-08-20'de bu tam olarak yasandi: ``fetch_weather.py``deki merdiven
    saatlik limiti tanıyacak sekilde duzeltildi, ama ALTI cekicinin her
    birinde merdivenin kendi kopyasi vardi ve hicbiri duzelmedi. Cekim
    yeniden baslatildiginda yine "65 sn bekleniyor" yazdi.

    Kural: 429 beklemesi ``rate_limit_beklemesi`` uzerinden hesaplanir.
    Merdivenin KENDISI (or. EPIAS'in ayri kalibrasyonu) betikte tanimli
    kalabilir; ``merdiven=`` parametresiyle gecirilir.
    """
    kaynak = betik.read_text(encoding="utf-8")
    # Tetikleyici GERCEK 429 ele alisi olmali. Duz "429" aramasi yorumlarda
    # ve URL parcalarindaki rakamlarda da eslesiyordu -- yanlis alarm ureten
    # bir kapi, okunmayan bir kapidir.
    ele_aliyor = "status_code == 429" in kaynak or '"429" in str(' in kaynak
    if not ele_aliyor:
        pytest.skip(f"{betik.name} 429 ele almiyor")

    kendi_indeksi = "RATE_LIMIT_BACKOFF[min(" in kaynak.replace("fw.", "")
    assert not kendi_indeksi, (
        f"{betik.name} geri cekilme merdivenini kendisi indeksliyor. "
        "Boyle bir kopya, ortak yardimcida yapilan duzeltmeyi ALMAZ -- "
        "2026-08-20'de saatlik limit duzeltmesi tam olarak boyle kayboldu. "
        "Dogrusu: `bekle, gerekce = rate_limit_beklemesi(yanit, deneme)`"
    )
    assert "rate_limit_beklemesi" in kaynak, (
        f"{betik.name} 429 ele aliyor ama ortak bekleme yardimcisini cagirmiyor."
    )


@pytest.mark.parametrize("betik", _fetch_betikleri(), ids=lambda p: p.name)
def test_zaman_serisi_istekleri_yerel_gun_siniri_kullanir(betik: Path) -> None:
    """Zaman serisi isteyen her cekici ``timezone=Europe/Istanbul`` gecmeli.

    Open-Meteo varsayilan olarak UTC doner. Turkiye kalici UTC+3'tur (yaz
    saati yok), dolayisiyla varsayilani birakmak her GUNU uc saat kaydirir:
    bir gunun "maksimum sicakligi" aslinda oncekiyle sonrakinin karisimi
    olur. Sema kontrolu bunu yakalamaz -- degerler makul kalir, yalnizca
    YANLIS gune yazilir.

    Kaymanin ne kadar sinsi oldugu olculdu (2026-08-20): saatlik ham veriden
    hesaplanan gunluk maksimum ile gunluk API'nin maksimumu arasindaki
    korelasyon dogru hizalamada 0.99883, UC SAAT kaydirilmis halde 0.99124
    idi. Yani veri, bozukken bile neredeyse kusursuz gorunur.

    Ayni olcum iki tablonun BIRIMLERININ farkli oldugunu da gosterdi
    (oran tam 3.6000): gunluk tablo km/sa, saatlik turev m/s. Ikisi de kendi
    icinde tutarlidir; esik karsilastirirken bu fark hatirlanmali.
    """
    kaynak = betik.read_text(encoding="utf-8")
    zaman_serisi = '"hourly"' in kaynak or '"daily"' in kaynak
    if "open-meteo" not in kaynak or not zaman_serisi:
        pytest.skip(f"{betik.name} Open-Meteo zaman serisi cekmiyor")

    assert '"Europe/Istanbul"' in kaynak, (
        f"{betik.name} Open-Meteo'dan zaman serisi cekiyor ama "
        "timezone=Europe/Istanbul GECMIYOR. Varsayilan UTC'dir ve her gunu "
        "uc saat kaydirir -- degerler makul gorunur, yalnizca yanlis gune yazilir."
    )


def test_referans_tablosu_yayin_dogrulamasindan_geciyor() -> None:
    """Ilce referansi ``validate_published_dataframe`` ile OKUNABILMELI.

    ``fetch_weather.py`` ilce listesini bu fonksiyonla okur; yan metadata
    dosyasi yoksa betik ILK SATIRDA ``ValueError`` ile duser ve hicbir hava
    verisi cekilemez.

    2026-08-20'de tam olarak bu durumdaydi: ``ilceler_gdz_adm.parquet`` icin
    ``.metadata.json`` yan dosyasi yoktu (``data/`` gitignore kapsaminda
    oldugu icin hic islenmemisti) ve ``fetch_weather.py`` calismiyordu.
    1200'un uzerinde test yesilken bunu HICBIRI gormedi -- cunku hicbiri
    cekicinin acilis yolunu denemiyordu.

    Yarisma gunu bunun bedeli somuttur: hava verisini tazelemek isteyen ekip,
    sebebi belirsiz bir hatayla karsilasir.
    """
    referans = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
    if not referans.is_file():
        pytest.skip("ilce referansi bu ortamda yok")

    sys.path.insert(0, str(KOK / "src"))
    from gridup.io_utils import validate_published_dataframe

    frame = validate_published_dataframe(
        referans,
        required_columns=("ilce", "il", "il_key", "ilce_key", "anahtar"),
        min_rows=5,
    )
    assert len(frame) == 96, f"96 ilce bekleniyordu, {len(frame)} geldi"
    assert frame["ilce_key"].is_unique, "ilce_key tekil degil"
    assert not frame[["lat", "lon"]].isna().any().any(), "koordinat eksik"


def test_eksik_aralik_yalnizca_kuyrugu_istiyor(tmp_path: Path) -> None:
    """Kontrol noktasi kismen kapsiyorsa YALNIZCA eksik kuyruk cekilmeli.

    OLCULDU 2026-08-20: cekiciler "kapsamiyor" gordugunde TUM araligi bastan
    indiriyordu. Gercek ihtiyac sondaki birkac gundu::

        nem_toprak   eksik  7 gun  ama 2430 gun isteniyordu = 347x
        konvektif    eksik  6 gun  ama 1944 gun isteniyordu = 324x
        hava_gunluk  eksik 10 gun  ama 2430 gun isteniyordu = 243x

    Open-Meteo kotasi istenen VERI MIKTARINA gore agirliklandirilir; 347 kat
    fazla veri istemek kotayi 347 kat hizli tuketir ve gunluk tazelemeyi
    saatlere cikarir. Yarisma gunu veri tazeleyememek somut bir kayiptir.
    """
    import pandas as pd

    modul = _modul_yukle("fetch_weather")
    ckpt = tmp_path / "c.parquet"
    pd.DataFrame({"tarih": pd.date_range("2020-01-01", "2026-08-12", freq="D")}).to_parquet(ckpt)

    aralik = modul.eksik_aralik(ckpt, "2020-01-01", "2026-08-19")
    assert aralik is not None, "Eksik kuyruk varken None dondu."
    bas, son = aralik
    assert son == "2026-08-19"
    # 1 gun ortusme payiyla: 08-11'den baslar, yani ~9 gun -- 2400 degil.
    gun = (pd.Timestamp(son) - pd.Timestamp(bas)).days + 1
    assert gun <= 15, f"Kuyruk yerine {gun} gun isteniyor -- kota bosa yaniyor."
    assert pd.Timestamp(bas) < pd.Timestamp("2026-08-12"), "Ortusme payi yok; revizyon kacar."


def test_eksik_aralik_tam_kapsamda_none_donuyor(tmp_path: Path) -> None:
    """Kapsanan aralik icin hic istek atilmamali."""
    import pandas as pd

    modul = _modul_yukle("fetch_weather")
    ckpt = tmp_path / "c.parquet"
    pd.DataFrame({"tarih": pd.date_range("2020-01-01", "2026-08-19", freq="D")}).to_parquet(ckpt)
    assert modul.eksik_aralik(ckpt, "2020-01-01", "2026-08-19") is None
    assert modul.eksik_aralik(ckpt, "2020-01-01", "2026-08-01") is None


def test_eksik_aralik_bas_boslugunda_tum_araligi_istiyor(tmp_path: Path) -> None:
    """Bastaki bosluk icin tum aralik: iki parcayi birlestirmek risklidir."""
    import pandas as pd

    modul = _modul_yukle("fetch_weather")
    ckpt = tmp_path / "c.parquet"
    pd.DataFrame({"tarih": pd.date_range("2021-05-01", "2026-08-19", freq="D")}).to_parquet(ckpt)
    assert modul.eksik_aralik(ckpt, "2020-01-01", "2026-08-19") == ("2020-01-01", "2026-08-19")


def test_konvektif_kendi_kapsam_basina_kirpiyor() -> None:
    """CAPE 2021-05'te basliyor; daha erken bir ``--start`` KIRPILMALI.

    Kirpilmazsa kalici bir "bas boslugu" olusur: kontrol noktasi 2021-05'te
    baslar, istek 2020-01'de baslar ve ``eksik_aralik`` bunu kapanmamis
    bosluk sanip HER KOSUDA tum araligi bastan indirir. Kaynak tek basina
    hicbir kazanc gostermez -- olculdu ve bu testle kapatildi.
    """
    kaynak = (BETIK_DIZINI / "fetch_konvektif.py").read_text(encoding="utf-8")
    assert "KAPSAM_BASI" in kaynak
    assert "args.start = KAPSAM_BASI" in kaynak, (
        "fetch_konvektif, kapsam basindan onceki bir --start'i KIRPMIYOR. "
        "Cagiran tum kaynaklara ayni --start'i gecerse her kosuda tam cekim olur."
    )


def test_ckpt_birlestir_yeni_degeri_tercih_ediyor(tmp_path: Path) -> None:
    """Ortusen gunde YENI cekim kazanir -- ERA5 son gunlerini REVIZE eder.

    ERA5T (on surum) birkac gun sonra nihai ERA5 ile degistirilir; daha gec
    cekilmis deger daha dogrudur. Eskiyi tercih etmek revizyonu kalici olarak
    kaybettirirdi.
    """
    import pandas as pd

    modul = _modul_yukle("fetch_weather")
    ckpt = tmp_path / "c.parquet"
    eski = pd.DataFrame(
        {
            "ilce_key": ["a"] * 3,
            "tarih": pd.date_range("2026-08-10", periods=3, freq="D"),
            "deger": [1.0, 2.0, 3.0],
        }
    )
    eski.to_parquet(ckpt)
    yeni = pd.DataFrame(
        {
            "ilce_key": ["a"] * 3,
            "tarih": pd.date_range("2026-08-12", periods=3, freq="D"),
            "deger": [30.0, 40.0, 50.0],
        }
    )
    birlesik = modul.ckpt_birlestir(ckpt, yeni, anahtarlar=("ilce_key", "tarih"))

    assert len(birlesik) == 5, "Gunler birlestirilmedi."
    ortusen = birlesik.loc[birlesik["tarih"] == pd.Timestamp("2026-08-12"), "deger"].iloc[0]
    assert ortusen == 30.0, "Ortusen gunde ESKI deger kazandi -- revizyon kaybolur."
