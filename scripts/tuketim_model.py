"""Grid Up Datathon -- trafo bazli gunluk tuketim tahmini.

NEDEN AYRI BIR HAT
------------------
``day_one.py`` gorevden bagimsiz bir hattir ve isini yapti: 3,1 dakikada
gecerli bir gonderim uretti (public LB 1,22670). Ama uc noktada bu problemin
sekline uymuyor ve UCUNU DE KENDISI RAPOR ETTI:

1. ``lokasyon``u ham kategorik olarak aldi; ilce anahtarini ``tanim``dan
   (trafo numarasi) turetmeye calisti ve 7.368 adin 7.368'i tutmadi. Yani
   232 dis kolonun HICBIRI baglanmadi.
2. Seyrek paneli 0,0 ile doldurdu -- 1.205.283 uydurma satir. Kendi uyarisi:
   "olcum hedefi (tuketim) -> np.nan KULLAN; 'kayit yok' ile 'deger sifir'
   ayni sey degildir." Tahmin ortalamasi 1.990'a dustu (gercek 3.252).
3. Trafo gecmisini hic kullanmadi -- oysa olculdu: trafo seviyesi tek basina
   log-varyansin %90,1'ini acikliyor.

Bu betik ucunu de duzeltir.

EGITIM KURGUSU: yuvarlanan koken (rolling origin)
-------------------------------------------------
Trafo ozetleri hedefi okur. Bir satirin ozeti kendi degerini iceriyorsa model
cevabi kopyalar. Gercek test'te ozetler GECMISTEN, etiketler GELECEKTEN gelir;
egitim de ayni sekilde kurulmalidir. Bu yuzden egitim bloklara ayrilir --
her blokta ozet penceresi etiket penceresinden ONCE biter:

    blok      ozet penceresi              etiket penceresi           gun
    ------    --------------------------  -------------------------  ---
    yaz25     2025-01-01 .. 2025-03-31    2025-04-01 .. 2025-07-31   122
    guz25     2025-01-01 .. 2025-07-31    2025-08-01 .. 2025-11-30   122
    kis26     2025-01-01 .. 2025-11-30    2025-12-01 .. 2026-03-31   121
    TEST      2025-01-01 .. 2026-03-31    2026-04-01 .. 2026-07-31   122

``yaz25`` blogu test doneminin MEVSIMSEL IKIZIDIR -- ayni aylar, ayni ufuk
uzunlugu. Dogrulama once orada yapilir; baska hicbir blok Nisan-Temmuz
davranisini olcemez.

Ozet penceresi uzunlugu bloklar arasinda degisiyor (90/212/334/455 gun). Bu
bilerek boyle: test'in penceresi de hepsinden uzun. Modelin bu kaymayi
gorebilmesi icin ``ozet_pencere_gun`` ve ``t_doluluk`` (gun sayisi / pencere)
acikca oznitelik olarak veriliyor -- ham ``t_gun_sayisi`` pencereyle birlikte
buyudugu icin tek basina yaniltir.

Calistirma::

    python scripts/tuketim_model.py                 # dogrula + gonderim uret
    python scripts/tuketim_model.py --hizli         # az agac, hizli deneme
    python scripts/tuketim_model.py --gonder "not"  # ureti + Kaggle'a yolla
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.features.temporal import (  # noqa: E402
    add_calendar_features,
    add_ramadan_features,
    add_turkish_holiday_features,
)
from gridup.features.trafo import (  # noqa: E402
    grup_seviyeleri_ekle,
    guc_kovasi,
    trafo_ozetleri_cikar,
    trafo_ozetleri_uygula,
)
from gridup.reporting import satir_tamponlu_cikti  # noqa: E402
from gridup.turkish import join_key  # noqa: E402

HAM = KOK / "data" / "raw"
DIS = KOK / "data" / "external"
GONDERIM = KOK / "submissions"

HEDEF = "tuketim"
ZAMAN = "tarih"
GRUP = "tanim"


@dataclass(frozen=True)
class Blok:
    """Bir etiket penceresi. Ozet penceresi ``OZET_PENCERE_GUN``den turer."""

    ad: str
    etiket_basi: str
    etiket_sonu: str

    @property
    def ozet_basi(self) -> pd.Timestamp:
        return pd.Timestamp(EGITIM_BASI)

    @property
    def ozet_bitis(self) -> pd.Timestamp:
        return pd.Timestamp(self.etiket_basi) - pd.Timedelta(days=1)


#: Ozet penceresi TUM GECMIS: egitimin basindan etiket blogunun basina kadar.
#:
#: SABIT UZUNLUK DENENDI VE OLCULDU (2026-08-21) -- daha kotu.
#: Gerekce makuldu: pencere uzunlugu bloklar arasi 90/212/334, test'te 455
#: oldugu icin model her blokta baska bir dagilim goruyor ve erken durdurma
#: 22 ile 376 agac arasinda zipliyordu. Hepsini 90 gune sabitlemek bu
#: tutarsizligi gercekten cozdu, ama beklemedigim bir bedeli vardi:
#:
#:     ozet penceresi        TEST'te soguk pay
#:     455 gun (tum gecmis)  %22,2   (158.369 satir)
#:      90 gun (sabit)       %27,3   (195.233 satir)
#:
#: Kisa pencere, yalnizca 2025'in basinda gorulup sonra susan trafolari da
#: SOGUGA itiyor. Soguk rejimin RMSLE'si sicagin iki katindan fazla oldugu
#: icin bu takas net zarar: yaz ikizi 1,0677 -> 1,0727, kis blogu
#: 1,3125 -> 1,4065.
#:
#: Dagilim tutarsizligi ise ``ozet_pencere_gun`` ve ``t_doluluk``
#: kolonlariyla modele ACIKCA bildiriliyor; guncellik pencereleri
#: (``t_log_son7`` ... ``t_log_son90``) zaten pencere uzunlugundan
#: bagimsiz oldugu icin taze seviye bilgisi her blokta ayni bicimde
#: geliyor. Yani asil derdi cozen sey pencereyi kisaltmak degil,
#: guncellik pencerelerini eklemekmis.

#: Egitim bloklari. ``yaz25`` test doneminin mevsimsel ikizi -- ayni aylar,
#: ayni ufuk uzunlugu. Dogrulama once orada okunur.
BLOKLAR: tuple[Blok, ...] = (
    Blok("yaz25", "2025-04-01", "2025-07-31"),
    Blok("guz25", "2025-08-01", "2025-11-30"),
    Blok("kis26", "2025-12-01", "2026-03-31"),
)

EGITIM_BASI = "2025-01-01"

#: EK KOKENLER -- yuvarlanan kokenin daha sik ornekleri.
#:
#: Ana bloklar (yaz25/guz25/kis26) egitimin yalnizca %84,7'sini etiket
#: olarak kullaniyor; 2025-01-01..03-31 arasindaki 187.500 satir hicbir
#: zaman etiket olmuyor, sadece ozet penceresi oluyor. Ustelik uc blok,
#: modele "gecmisten gelecege esleme"nin yalnizca UC ornegini gosteriyor.
#:
#: Recruit yarismasinin birincisi ayni yapiyi 63 farkli baslangic
#: noktasiyla yigmisti. Buradaki ek kokenler ana bloklarla ORTUSUR --
#: bu bilerek: ayni etiket satiri farkli ozet pencereleriyle birden
#: fazla kez gorulur ve model "ozet ne kadar eskiyse tahmin o kadar
#: belirsiz" iliskisini ogrenir. Dogrulamada ortusme YASAKTIR; ilgili
#: fonksiyon hedef blokla kesisen her kokeni atar.
#: Her koken kendi ozet penceresini EGITIM_BASI'ndan etiketin bir gun
#: oncesine kadar kurar. Yani etiket ne kadar gec baslarsa ozet o kadar uzun.
#:
#: OZET PENCERESI MERDIVENI -- eklemelerin gerekcesi bu. Test'in ozet
#: penceresi 455 GUN ve ``ozet_pencere_gun`` modele ACIK bir kolon olarak
#: veriliyor; yani model test'te hic gormedigi bir degere disdegerleme
#: yapmak zorunda kaliyordu::
#:
#:     eski merdiven   31  90  120  181  212  243  304  334  365   | TEST 455
#:     yeni merdiven   31  59  90  120  151  181  212  243  273
#:                     304  334  365  396  424                     | TEST 455
#:
#: Bes yeni koken hem aradaki bosluklari dolduruyor hem de 455'e uzaniyor.
#: ``sub26`` ve ``mar26`` kisa etiketli/uzun ozetli: amaclari satir sayisi
#: degil, merdivenin ust ucunu test'e yaklastirmak.
#: YOGUNLASTIRMA DENENDI VE ALINMADI (2026-08-22, ``deney_koken_yogun.py``).
#: Bes koken daha eklenip merdiven 31/59/90/120/151/181/212/243/273/304/
#: 334/365/396/424'e cikarildi (eskisi 365'te bitiyordu, TEST 455)::
#:
#:     V18 (6 koken)   SICAK 0,79848
#:     TUM (11 koken)  SICAK 0,79703   fark +0,00184  t=+0,24
#:       yaz25 +0,0018   guz25 +0,0276   kis26 -0,0238
#:
#: Bloklar birbirini yiyor -- klasik blok yapayligi deseni. Kazanc alti
#: kokende DOYMUS. Merdiven argumani makuldu ama veri desteklemedi;
#: LB'de dogrulanmis olan altili korundu.
EK_KOKENLER: tuple[tuple[str, str, str], ...] = (
    ("sub25", "2025-02-01", "2025-03-31"),
    ("bah25", "2025-05-01", "2025-08-31"),
    ("yaz25b", "2025-07-01", "2025-10-31"),
    ("guz25b", "2025-09-01", "2025-12-31"),
    ("kis26b", "2025-11-01", "2026-02-28"),
    ("bah26", "2026-01-01", "2026-03-31"),
)

#: ``v18``in gonderdigi kokenler. Su an ``EK_KOKENLER`` ile ayni; yogunlastirma
#: yeniden denenirse olcum tabani bu kalsin diye ayri duruyor.
KOKENLER_V18: tuple[str, ...] = ("sub25", "bah25", "yaz25b", "guz25b", "kis26b", "bah26")

#: Gunluk hava tablolari: (dosya, kullanilacak kolonlar).
#: Konvektif ve hava kalitesi BILEREK yok -- ikisi de kesinti fizigi icin
#: cekilmisti; tuketimle fiziksel bir baglantilari yok ve 47 ilcede gunluk
#: cozunurlukte gurultuden baska bir sey katmalari beklenmiyor. Olculmeden
#: eklenmeleri "her sey feature'dir" hatasi olur.
HAVA_TABLOLARI: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "hava_gunluk.parquet",
        (
            "sicaklik_ort",
            "sicaklik_max",
            "sicaklik_min",
            "hissedilen_max",
            "yagis_toplam",
            "yagis_saati",
            "ruzgar_max",
            "gunes_radyasyon",
            "asiri_sicak",
            "asiri_soguk",
        ),
    ),
    (
        "nem_toprak_gunluk.parquet",
        ("nem_ort", "ciy_ort", "vpd_ort", "toprak_nem_ort", "bulut_dusuk_ort", "et0_toplam"),
    ),
)

#: CDD taban sicakliklari. 22 = MGM/Eurostat resmi Turkiye tanimi; 24
#: olculdugunde ayni bantta cikti (r 0,1966 vs 0,1968). Uc taban da
#: veriliyor, secim modele birakiliyor.
CDD_TABANLARI: tuple[int, ...] = (18, 22, 24)

#: Isil atalet pencereleri. Bina kutlesi sicakligi ANINDA takip etmez; klima
#: yuku birkac gunluk birikmis sicaga tepki verir. Bu yuzden CDD'nin
#: hareketli ortalamalari ham gunluk degerden daha bilgilidir.
ISIL_PENCERELER: tuple[int, ...] = (3, 7, 14)
ISIL_KOLONLAR: tuple[str, ...] = ("cdd22", "cdd24", "sicaklik_ort")

#: Modelden CIKARILACAK kolonlar. ``tanim`` bilerek disarida: 7.036 seviyeli
#: bir kimlik kategorik olarak verilirse model onu ezberler ve soguk
#: trafolarda (test satirlarinin %22,16'si) hicbir sey ogrenmemis olur.
#: Trafonun kimligi zaten ``t_*`` ozetleri uzerinden, GENELLENEBILIR
#: bicimde tasiniyor.
DISLANAN = {"id", HEDEF, ZAMAN, GRUP, "lokasyon", "_blok"}

KATEGORIK = ("il_key", "bolge", "ilce_key")


# ---------------------------------------------------------------- yukleme


def yukle() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(
        HAM / "train.csv", dtype={GRUP: "string"}, parse_dates=[ZAMAN], encoding="utf-8"
    )
    te = pd.read_csv(
        HAM / "test.csv", dtype={GRUP: "string"}, parse_dates=[ZAMAN], encoding="utf-8"
    )
    return tr, te


def lokasyon_ayristir(frame: pd.DataFrame) -> pd.DataFrame:
    """``lokasyon`` -> ``il_key`` / ``bolge`` / ``ilce_key``. YENI frame.

    Olculdu (2026-08-21, train+test birlesimi): 47 benzersiz deger, sifir NaN,
    iki bicim:

        İZMİR>METROPOL>KARABAĞLAR    3 parca  (IL > BOLGE > ILCE)  30 ilce
        MANİSA>SALİHLİ               2 parca  (IL > ILCE)          17 ilce

    Kural: **son parca her zaman ilcedir.** Orta parca yalnizca Izmir'de var
    ve isletme bolgesidir (METROPOL / GUNEY BOLGE / KUZEY BOLGE); Manisa'da
    yoktur, orada ``YOK`` yazilir -- bos string degil, cunku ``YOK`` gercek
    bir bilgi tasir ("bu il bolgelere ayrilmamis").

    ``join_key`` Turkce i-tuzagini ele alir: ``İZMİR`` -> ``izmir``,
    ``KIRKAĞAÇ`` -> ``kirkagac``. Ham ``.lower()`` burada U+0307 birlesik
    noktasi uretir ve eslesme sessizce sifira duser.
    """
    p = frame["lokasyon"].str.split(">")
    sonuc = frame.copy()
    sonuc["il_key"] = p.str[0].str.strip().map(join_key)
    sonuc["ilce_key"] = p.str[-1].str.strip().map(join_key)
    sonuc["bolge"] = np.where(p.str.len() >= 3, p.str[1].str.strip(), "YOK")
    return sonuc


def hava_yukle() -> pd.DataFrame:
    """Ilce x gun hava tablosu. Isil atalet pencereleri burada uretilir."""
    tablolar: list[pd.DataFrame] = []
    for dosya, kolonlar in HAVA_TABLOLARI:
        yol = DIS / dosya
        d = pd.read_parquet(yol, columns=["ilce_key", "tarih", *kolonlar])
        d["tarih"] = pd.to_datetime(d["tarih"]).dt.normalize()
        tablolar.append(d.set_index(["ilce_key", "tarih"]))
    hava = pd.concat(tablolar, axis=1).reset_index()

    gunes = pd.read_parquet(
        DIS / "gunes_gunluk.parquet",
        columns=["anahtar", "tarih", "gunes_ghi_gunluk", "gun_uzunlugu_saat"],
    )
    gunes["ilce_key"] = gunes["anahtar"].str.split("|").str[-1]
    gunes["tarih"] = pd.to_datetime(gunes["tarih"]).dt.normalize()
    hava = hava.merge(
        gunes.drop(columns=["anahtar"]), on=["ilce_key", "tarih"], how="left", validate="one_to_one"
    )

    # CDD'yi TURKIYE tanimiyla yeniden hesapla.
    #
    # Open-Meteo'nun ``sogutma_derece_gun``u taban 18 C kullaniyor (ABD
    # varsayilani) -- geri cikarildi: CDD>0 olan en dusuk sicaklik 18,10 C.
    # MGM'in (Eurostat uyumlu) resmi Turkiye tanimi ise taban 22 C.
    #
    # Olculdu (2026-08-21, 1,23M satir, trafo seviyesi arindirilmis tuketim
    # sapmasiyla korelasyon):
    #     CDD taban 18 (mevcut)  r = +0,1804
    #     CDD taban 22           r = +0,1966
    #     CDD taban 24           r = +0,1968
    #     sicaklik ortalamasi    r = +0,1001
    #
    # Ikisi de birakiliyor; model hangisini kullanacagina kendi karar verir.
    #
    # HDD ise ATILIYOR. Olculdu: taban 18 formunda r = -0,0038, MGM formunda
    # r = +0,0030 -- yani SIFIR. Bu bolgede isitma elektrikle degil dogalgazla
    # yapiliyor, dolayisiyla isitma-derece-gunu bu problemde bilgi tasimiyor.
    # Tasimayan bir kolonu birakmak, agaca gurultulu bir bolme adayi
    # vermekten baska bir sey degil.
    for taban in CDD_TABANLARI:
        hava[f"cdd{taban}"] = (hava["sicaklik_ort"] - taban).clip(lower=0.0)
    hava = hava.drop(columns=["isitma_derece_gun", "sogutma_derece_gun"], errors="ignore")

    hava = hava.sort_values(["ilce_key", "tarih"])
    g = hava.groupby("ilce_key", observed=True)
    for kolon in ISIL_KOLONLAR:
        for p in ISIL_PENCERELER:
            hava[f"{kolon}_ort{p}"] = g[kolon].transform(
                lambda s, _p=p: s.rolling(_p, min_periods=1).mean()
            )
    return hava


def hava_ekle(frame: pd.DataFrame, hava: pd.DataFrame) -> pd.DataFrame:
    oncesi = len(frame)
    sonuc = frame.merge(hava, on=["ilce_key", "tarih"], how="left", validate="many_to_one")
    if len(sonuc) != oncesi:
        raise RuntimeError(f"hava birlesimi satir sayisini degistirdi: {oncesi} -> {len(sonuc)}")
    return sonuc


#: Statik ilce tablolari: (dosya, kullanilacak kolonlar).
#: Ilceyi CIPLAK BIR ETIKET olmaktan cikarip VEKTOR yapmak icin. Soguk
#: trafolarda elimizdeki tek zengin bilgi kaynagi konum; 47 seviyeli bir
#: kategorik ise konumun tasidigi seyin ancak kucuk bir kismini modele
#: gecirir. Arazi ortusu ve sebeke altyapisi, ilceler arasi farki
#: SUREKLI degiskenler olarak tasir -- ve model bunlari daha once hic
#: gormedigi bir ilceye bile genelleyebilir.
STATIK_ILCE_TABLOLARI: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "arazi_ortusu_ilce.parquet",
        (
            "agac_orani",
            "calilik_orani",
            "otlak_orani",
            "tarim_orani",
            "yerlesim_orani",
            "ciplak_orani",
            "su_orani",
            "bitki_ortusu_orani",
        ),
    ),
    (
        "osm_altyapi_ilce.parquet",
        (
            "osm_trafo",
            "osm_direk",
            "osm_dagitim_hat_km",
            "osm_iletim_hat_km",
            "osm_kablo_km",
            "osm_direk_yogunlugu",
            "osm_hat_yogunlugu",
        ),
    ),
)


def statik_ilce_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    """Arazi ortusu ve OSM sebeke altyapisi. YENI frameler.

    ``yerlesim_orani`` bu veri kumesinde daha once olculmustu (kesinti
    calismasinda, 162.240 satir): ilceler arasi en guclu tek ayrim
    degiskeniydi. Tuketim icin fiziksel gerekcesi daha da dogrudan --
    kentsel yogunluk, trafo basina dusen abone sayisinin vekilidir.

    ``osm_dagitim_hat_km`` / ``osm_direk`` ise sebekenin ilcedeki
    yayginligini olcer: ayni kVA'lik bir trafo, seyrek kirsal bir agda
    ile yogun kentsel bir agda farkli yuk tasir.

    Kaynaklar ``data/sources.yml``de kayitli ve ``model_girdisi`` bayragi
    serbest -- yarisma dis veri kullanimina acik.
    """
    tablolar = []
    for dosya, kolonlar in STATIK_ILCE_TABLOLARI:
        d = pd.read_parquet(DIS / dosya, columns=["ilce_key", *kolonlar])
        tablolar.append(d.set_index("ilce_key"))
    statik = pd.concat(tablolar, axis=1).reset_index()

    sonuclar = []
    for c in cerceveler:
        yeni = c.merge(statik, on="ilce_key", how="left", validate="many_to_one")
        if len(yeni) != len(c):
            raise RuntimeError("statik ilce birlesimi satir sayisini degistirdi")
        sonuclar.append(yeni)
    return sonuclar


def ilce_yapisi_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    """Trafonun KENDI ILCESI icindeki kapasite konumu. YENI frameler.

    Hedefi degil, VARLIK LISTESINI kullanir: train ve test birlikte
    7.368 trafonun tamamini ve her birinin ``guc``unu aciklar. Yani bir
    ilcede kac trafo oldugu, toplam kurulu gucun ne oldugu ve bu trafonun
    o dagilimda nerede durdugu -- hepsi SOGUK trafolar icin de bilinir.

    Neden onemli: ham ``guc`` mutlak bir buyukluktur. 630 kVA, Kiraz'da
    ilcenin en buyuk trafolarindan biri, Konak'ta ortalamanin altidir.
    ``guc_yuzdelik`` bu farki tek kolonda tasir ve ilceler arasi
    karsilastirilabilir kilar.
    """
    # Ilce nufusu ve alani -- ``data/reference/ilceler_gdz_adm.parquet``.
    #
    # Bu tablo depoda BASTAN BERI vardi ve gozden kacmisti; TUIK'ten ayrica
    # indirmeye gerek yok. Onemi su: elimizde ilcedeki TRAFO SAYISI da var
    # (train+test birlesimi 7.368 trafonun tamamini acikliyor), dolayisiyla
    #
    #     nufus / trafo_sayisi  ~  trafo basina ortalama abone
    #
    # hesaplanabiliyor. Literaturdeki tek olculmus calisma (Energies 18(7):
    # 1832, 2025, 16.696 trafo) trafo yukunu tahmin eden asil degiskenin
    # anma gucu DEGIL, tarife tipine gore ABONE SAYISI oldugunu buluyor
    # (SVR ile R2 = 0,90). Abone sayisini alamayiz; nufus/trafo onun kamuya
    # acik en yakin vekili.
    ref = pd.read_parquet(
        KOK / "data" / "reference" / "ilceler_gdz_adm.parquet",
        columns=["ilce_key", "nufus", "alan_km2"],
    ).drop_duplicates("ilce_key")

    hepsi = pd.concat([c[[GRUP, "guc", "ilce_key"]] for c in cerceveler]).drop_duplicates(GRUP)
    g = hepsi.groupby("ilce_key", observed=True)["guc"]
    ilce = pd.DataFrame(
        {
            "ilce_trafo_sayisi": g.size().astype("float64"),
            "ilce_toplam_guc": g.sum(),
            "ilce_guc_medyan": g.median(),
        }
    ).reset_index()
    hepsi["guc_yuzdelik"] = g.rank(pct=True)
    yuzdelik = hepsi.set_index(GRUP)["guc_yuzdelik"]

    sonuclar = []
    for c in cerceveler:
        yeni = c.merge(ilce, on="ilce_key", how="left", validate="many_to_one")
        yeni = yeni.merge(ref, on="ilce_key", how="left", validate="many_to_one")
        yeni["ilce_nufus_yogunlugu"] = yeni["nufus"] / yeni["alan_km2"]
        yeni["trafo_basina_nufus"] = yeni["nufus"] / yeni["ilce_trafo_sayisi"]
        yeni["kva_basina_nufus"] = yeni["nufus"] / yeni["ilce_toplam_guc"]
        yeni["guc_yuzdelik"] = yeni[GRUP].map(yuzdelik).astype("float64")
        yeni["guc_payi"] = yeni["guc"] / yeni["ilce_toplam_guc"]
        yeni["guc_medyan_orani"] = yeni["guc"] / yeni["ilce_guc_medyan"]
        # Sebeke yogunlugu: ilcedeki trafo basina dusen dagitim hatti.
        if "osm_dagitim_hat_km" in yeni.columns:
            yeni["trafo_basina_hat"] = yeni["osm_dagitim_hat_km"] / yeni["ilce_trafo_sayisi"]
        sonuclar.append(yeni)
    return sonuclar


def kimlik_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    """``tanim`` numarasindan turetilen STATIK kolonlar. YENI frameler.

    ``tanim`` modele kategorik olarak VERILMEZ (7.036 seviye, %28,8'i test'e
    ozgu -- ezberlenir). Ama sayisal degerinin kendisi bilgi tasiyor: trafo
    numaralari isletme ve fider bazinda bloklar halinde verilmis, yani yakin
    numaralar cogu zaman yakin sebeke parcalari.

    Olculdu (2026-08-21, 5.344 trafo, 5-katmanli CV, hedef = trafonun
    ortalama ``log1p(tuketim)``i):

        yalniz log_guc                     RMSE(log) 1,9707
        log_guc + ilce                               1,9739
        log_guc + ilce + panel yapisi                1,7343
        + ID kolonlari (bu fonksiyon)                1,7055

    Onemli ayrinti: kaba denemeler BASARISIZ olmustu -- onek ortalamasi
    (2,21-2,30) ve ID komsulugu (2,20) duz ``guc`` kovasindan (2,04) kotuydu.
    Ham sayiyi agac modeline vermek ise calisiyor, cunku model onek
    sinirlarini kendisi buluyor; elle secilen 2-3-4 haneli kesme noktalari
    dogru yerlerde degildi.

    12 trafonun numarasi sayisal degil (``202917T``, ``Iskele DM``,
    ``M-3115`` ...). Onlarda ``tanim_num`` NaN kalir -- 0 yazmak onlari
    numara uzayinin en basina koyardi ve bu uydurma bir komsuluk olurdu.
    """
    sonuclar = []
    for c in cerceveler:
        yeni = c.copy()
        ad = yeni[GRUP].astype("string")
        yeni["tanim_num"] = pd.to_numeric(ad.where(ad.str.fullmatch(r"\d+")), errors="coerce")
        yeni["tanim_uzunluk"] = ad.str.len().astype("float64")
        for n in (2, 3, 4, 5):
            yeni[f"tanim_on{n}"] = pd.to_numeric(ad.str[:n], errors="coerce")
        sonuclar.append(yeni)
    return sonuclar


def panel_yapisi_ekle(etiket: pd.DataFrame) -> pd.DataFrame:
    """Trafonun ETIKET penceresindeki satir yapisi. YENI frame.

    Hedefi degil, SORULAN satirlarin desenini kullanir: test.csv hangi
    (trafo, gun) ciftinin istendigini zaten acikca soyluyor. Bir trafonun
    122 gunun kacinda soruldugu, ilk ve son gununun nerede oldugu, araligin
    ne kadar dolu oldugu -- hepsi mesru bicimde bilinebilir.

    Neden guclu: olculdu, bu bes kolon trafo seviyesi tahmininde RMSE(log)'u
    1,9707'den 1,7343'e indiriyor -- ID ve ilce dahil butun diger statik
    bilgilerden daha buyuk bir katki. Sezgisi acik: surekli olculen bir
    trafo ile ayda birkac gun gorunen bir trafo ayni sey degildir; seyrek
    gorunme, dusuk ve duzensiz yuke isaret eder.

    Soguk trafolar icin bu kolonlar TAM DOLUDUR -- gecmisi olmayan 2.024
    trafo hakkinda sahip oldugumuz neredeyse tek gozlemsel bilgi budur.
    """
    sonuc = etiket.copy()
    pencere_basi = sonuc[ZAMAN].min()
    pencere_gun = float((sonuc[ZAMAN].max() - pencere_basi).days) + 1.0
    g = sonuc.groupby(GRUP, observed=True)[ZAMAN]
    ilk, son, adet = g.transform("min"), g.transform("max"), g.transform("size")
    sonuc["p_gun_sayisi"] = adet.astype("float64")
    sonuc["p_ilk_ofset"] = (ilk - pencere_basi).dt.days.astype("float64")
    sonuc["p_son_ofset"] = (son - pencere_basi).dt.days.astype("float64")
    sonuc["p_yayilma"] = (son - ilk).dt.days.astype("float64") + 1.0
    sonuc["p_doluluk"] = sonuc["p_gun_sayisi"] / sonuc["p_yayilma"]
    sonuc["p_pencere_payi"] = sonuc["p_gun_sayisi"] / pencere_gun
    return sonuc


def ulusal_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    """EPIAS ulusal SAATLIK tuketimden gunluk endeks. YENI frameler.

    Bu, disaridan hazirladigimiz veriler icinde bu goreve en dogrudan
    bagli olani. Turkiye toplam tuketimi, tek bir trafonun goremedigi ama
    hepsini birden etkileyen seyleri tasir: ulke capinda sicak dalgasi,
    bayram tatilinin gercek isgucu etkisi, ekonomik aktivite seviyesi.

    Kapsam olculdu (2026-08-21): 2020-01-01 -> 2026-08-20 saatlik, yani
    test penceresinin (2026-04-01 .. 07-31) 122 gununun TAMAMI elimizde --
    tahmin degil, GERCEKLESMIS deger. Yarisma dis veriye acik oldugu icin
    bu mesru; ve bir yil sonra ayni modeli kurmak isteyen biri icin de
    gecerli, cunku EPIAS bu seriyi gecmise donuk yayimliyor.

    Uretilen kolonlar:
        ``ulusal_gunluk``     o gunun ulusal toplam tuketimi (MWh)
        ``ulusal_tepe``       gunun en yuksek saatlik degeri
        ``ulusal_tepe_orani`` tepe / ortalama -- gun ici yayilma olcusu
        ``ulusal_yil_once``   364 gun onceki ayni haftagunu (yillik taban)
        ``ulusal_yillik_buyume`` log oran -- mevsimden arinmis buyume
    """
    yol = DIS / "epias" / "tuketim_saatlik.parquet"
    saatlik = pd.read_parquet(yol)
    saatlik["_g"] = pd.to_datetime(saatlik["zaman"]).dt.normalize()
    g = saatlik.groupby("_g")["consumption"]
    u = pd.DataFrame({"ulusal_gunluk": g.sum(), "ulusal_tepe": g.max(), "_ort": g.mean()})
    u["ulusal_tepe_orani"] = u["ulusal_tepe"] / u["_ort"].replace(0.0, np.nan)
    # 364 gun = 52 tam hafta: haftagunu hizasi KORUNUR. 365 kullanmak
    # pazartesiyi pazara denk getirir ve yillik orani haftagunu etkisiyle
    # kirletir.
    u["ulusal_yil_once"] = u["ulusal_gunluk"].reindex(u.index - pd.Timedelta(days=364)).to_numpy()
    u["ulusal_yillik_buyume"] = np.log(
        u["ulusal_gunluk"] / u["ulusal_yil_once"].replace(0.0, np.nan)
    )
    u = u.drop(columns=["_ort"]).reset_index().rename(columns={"_g": ZAMAN})

    sonuclar = []
    for c in cerceveler:
        yeni = c.merge(u, on=ZAMAN, how="left", validate="many_to_one")
        if len(yeni) != len(c):
            raise RuntimeError("ulusal birlesim satir sayisini degistirdi")
        sonuclar.append(yeni)
    return sonuclar


def grup_profilleri_ekle(uygula: pd.DataFrame, profil_kaynak: pd.DataFrame) -> pd.DataFrame:
    """Ilce ve guc-kovasi duzeyinde MEVSIM/HAFTAGUNU profilleri. YENI frame.

    Trafo bazli ``t_ay_sapma`` gucludur ama iki durumda yoktur: soguk
    trafolarda (test satirlarinin %22,16'si) ve o ayda hic gozlenmemis
    sicak trafolarda. Grup duzeyindeki profil bu bosluklarda devreye girer
    -- Bornova'daki trafolar temmuzda ortalama ne kadar yukseliyor sorusu,
    trafo bilinmese bile cevaplanabilir.

    Profil kaynagi, trafo profilleriyle AYNI cerceve: hedef blogun disi.
    """
    p = profil_kaynak[[GRUP, ZAMAN, HEDEF, "guc", "ilce_key"]].copy()
    p["_y"] = np.log1p(p[HEDEF].clip(lower=0.0))
    p["_sapma"] = p["_y"] - p.groupby(GRUP, observed=True)["_y"].transform("mean")
    p["_ay"] = p[ZAMAN].dt.month
    p["_hg"] = p[ZAMAN].dt.dayofweek
    p["_kova"] = guc_kovasi(p["guc"])

    sonuc = uygula.copy()
    sonuc["_ay"] = sonuc[ZAMAN].dt.month
    sonuc["_hg"] = sonuc[ZAMAN].dt.dayofweek
    sonuc["_kova"] = guc_kovasi(sonuc["guc"])
    for ad, anahtar in (
        ("gp_ilce_ay", ["ilce_key", "_ay"]),
        ("gp_ilce_hg", ["ilce_key", "_hg"]),
        ("gp_kova_ay", ["_kova", "_ay"]),
    ):
        tablo = p.groupby(anahtar, observed=True)["_sapma"].mean().rename(ad).reset_index()
        sonuc = sonuc.merge(tablo, on=anahtar, how="left", validate="many_to_one")
    return sonuc.drop(columns=["_ay", "_hg", "_kova"])


def yas_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    """Trafonun veri setinde ILK gorulmesinden bu yana gecen gun. YENI frameler.

    Hedefi DEGIL, satir YAPISINI kullanir: hangi (trafo, gun) ciftlerinin
    soruldugu test.csv'de zaten aciktir. Dolayisiyla test satirlarinin ilk
    gorulme tarihi mesru bicimde bilinebilir.

    Neden onemli: olculdu (2026-08-21, train'e 2025-02-01'den sonra katilan
    3.147 trafo, 347.606 satir) -- trafonun ILK gununde log-tuketim, kendi
    ortalamasindan **-0,562** sapiyor. Sonraki gunlerde sapma +-0,05'e
    duşuyor. Yani etki tek gune yogunlasmis ve keskin; enerjilendirme gunu
    kismi bir gundur.

    Test'te soguk trafolarin 1.666'si 2026 Mayis'inda birden beliriyor --
    yani fiziksel bir devreye alma dalgasi degil, veri sistemine toplu
    katilim. Bu yuzden "yeni tesis rampasi" beklenmiyor; yalnizca ilk gun
    etkisi bekleniyor ve ``yas`` onu yakalamaya yeter.
    """
    ilk = (
        pd.concat([c[[GRUP, ZAMAN]] for c in cerceveler]).groupby(GRUP, observed=True)[ZAMAN].min()
    )
    sonuclar = []
    for c in cerceveler:
        yeni = c.copy()
        yeni["yas"] = (yeni[ZAMAN] - yeni[GRUP].map(ilk)).dt.days.astype("float64")
        yeni["ilk_gun_mu"] = (yeni["yas"] == 0).astype("int8")
        sonuclar.append(yeni)
    return sonuclar


def takvim_ekle(frame: pd.DataFrame) -> pd.DataFrame:
    sonuc = add_calendar_features(frame, ZAMAN, prefix="tk")
    sonuc = add_turkish_holiday_features(sonuc, ZAMAN, prefix="tatil")
    return add_ramadan_features(sonuc, ZAMAN, prefix="ramazan")


# ---------------------------------------------------------------- bloklar


def blok_kur(tam_egitim: pd.DataFrame, blok: Blok) -> pd.DataFrame:
    """Bir blogun etiket satirlarini, ozet penceresinden turetilmis
    ozniteliklerle birlikte dondurur.

    Ozet penceresi ile etiket penceresi KESISMEZ -- kesisirse hedef sizar.
    Bu, fonksiyonun tek kritik guvencesi ve altta acikca denetleniyor.
    """
    ozet = tam_egitim[
        (tam_egitim[ZAMAN] >= blok.ozet_basi) & (tam_egitim[ZAMAN] <= blok.ozet_bitis)
    ]
    etiket_maske = (tam_egitim[ZAMAN] >= blok.etiket_basi) & (tam_egitim[ZAMAN] <= blok.etiket_sonu)
    etiket = tam_egitim[etiket_maske]
    if ozet.empty or etiket.empty:
        raise RuntimeError(f"blok {blok.ad}: ozet {len(ozet)} / etiket {len(etiket)} satir")
    if ozet[ZAMAN].max() >= etiket[ZAMAN].min():
        raise RuntimeError(
            f"blok {blok.ad}: ozet penceresi ({ozet[ZAMAN].max().date()}) etiket "
            f"penceresine ({etiket[ZAMAN].min().date()}) TASIYOR -- hedef sizardi"
        )
    # Profil kaynagi: egitimin tamami EKSI bu blogun etiket penceresi.
    # Satirin kendi etiketi yine hic okunmaz, ama ay kapsami test'inkiyle
    # ayni mertebeye cikar (bkz. trafo.trafo_ozetleri_cikar aciklamasi).
    return _ozet_tasi(ozet, etiket, blok.ad, profil_kaynak=tam_egitim[~etiket_maske])


def _ozet_tasi(
    ozet: pd.DataFrame, etiket: pd.DataFrame, ad: str, *, profil_kaynak: pd.DataFrame
) -> pd.DataFrame:
    ozetler = trafo_ozetleri_cikar(
        ozet,
        profil_kaynak=profil_kaynak,
        hedef_penceresi=(etiket[ZAMAN].min(), etiket[ZAMAN].max()),
        isil_kolonlar=("cdd22", "sicaklik_ort"),
    )
    sonuc = trafo_ozetleri_uygula(etiket, ozetler)
    sonuc = grup_seviyeleri_ekle(sonuc, ozet)
    sonuc = grup_profilleri_ekle(sonuc, profil_kaynak)
    sonuc = panel_yapisi_ekle(sonuc)
    pencere = int((ozet[ZAMAN].max() - ozet[ZAMAN].min()).days) + 1
    sonuc["ozet_pencere_gun"] = float(pencere)
    sonuc["t_doluluk"] = sonuc["t_gun_sayisi"] / float(pencere)
    # UFUK: ozet penceresinin son gununden bu satira kac gun var.
    #
    # Modelin en temel hizalama degiskeni ve simdiye kadar YOKTU. Bir blok
    # icinde takvim tarihiyle esdogrusal oldugu icin gorunmez kaliyordu; ama
    # bloklar arasinda anlami tam olarak "bu tahmin ne kadar uzaga bakiyor".
    # Trafo ozetlerinin degeri ufukla birlikte azalir: 1 gun sonrasi icin
    # ``t_log_son14`` neredeyse kesindir, 122 gun sonrasi icin bir tahmindir.
    # Model bu azalmayi ancak ufku GORURSE ogrenebilir.
    sonuc["ufuk_gun"] = (sonuc[ZAMAN] - ozet[ZAMAN].max()).dt.days.astype("float64")
    sonuc["_blok"] = ad
    return sonuc


def ek_kokenleri_kur(tam_egitim: pd.DataFrame) -> pd.DataFrame:
    """``EK_KOKENLER``den ek etiket bloklari uretir. YENI frame.

    Her koken kendi ozet penceresini (basindan etiketin bir gun oncesine)
    ve kendi profil kaynagini (egitim eksi kendi etiket penceresi) kullanir
    -- yani ana bloklarla ayni fold-guvenlik kurali.
    """
    parcalar = []
    for ad, bas, son in EK_KOKENLER:
        blok = Blok(ad, bas, son)
        if pd.Timestamp(bas) <= pd.Timestamp(EGITIM_BASI):
            raise RuntimeError(f"koken {ad}: etiket egitim basindan once basliyor")
        parcalar.append(blok_kur(tam_egitim, blok))
        p = parcalar[-1]
        soguk = int(p["soguk_mu"].sum())
        print(
            f"  koken {ad:7} etiket {len(p):>7,} satir  "
            f"ozet {p['ozet_pencere_gun'].iloc[0]:>4.0f} gun  "
            f"soguk {soguk:>6,} (%{100 * soguk / len(p):.1f})"
        )
    return pd.concat(parcalar, ignore_index=True)


def kokenleri_ayikla(egitim: pd.DataFrame, hedef_blok: str) -> pd.DataFrame:
    """Hedef blogun etiket penceresiyle KESISEN her kokeni atar.

    Ek kokenler bilerek ortusuyor; dogrulamada ortusme sizintidir. Bir
    koken hedef blokla tek gun bile kesisiyorsa, o blogun etiketlerini
    egitimde gormus olur.
    """
    hedef = next(b for b in BLOKLAR if b.ad == hedef_blok)
    h_bas, h_son = pd.Timestamp(hedef.etiket_basi), pd.Timestamp(hedef.etiket_sonu)
    pencereler = {b.ad: (b.etiket_basi, b.etiket_sonu) for b in BLOKLAR}
    pencereler.update({ad: (bas, son) for ad, bas, son in EK_KOKENLER})
    tutulacak = {
        ad
        for ad, (bas, son) in pencereler.items()
        if pd.Timestamp(son) < h_bas or pd.Timestamp(bas) > h_son
    }
    return egitim[egitim["_blok"].isin(tutulacak)]


def egitim_kur(tam_egitim: pd.DataFrame) -> pd.DataFrame:
    parcalar = [blok_kur(tam_egitim, b) for b in BLOKLAR]
    for b, p in zip(BLOKLAR, parcalar, strict=True):
        soguk = int(p["soguk_mu"].sum())
        print(
            f"  blok {b.ad:6} etiket {len(p):>7,} satir  "
            f"ozet penceresi {p['ozet_pencere_gun'].iloc[0]:>4.0f} gun  "
            f"soguk {soguk:>6,} (%{100 * soguk / len(p):.1f})"
        )
    return pd.concat(parcalar, ignore_index=True)


def test_kur(tam_egitim: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Test satirlari. Ozet penceresi egitimin TAMAMI.

    Profil kaynagi da egitimin tamami: test'ten once bittigi icin "fold-disi"
    ile "gecmis" burada ayni sey.
    """
    sonuc = _ozet_tasi(tam_egitim, test, "TEST", profil_kaynak=tam_egitim)
    soguk = int(sonuc["soguk_mu"].sum())
    print(
        f"  TEST         {len(sonuc):>7,} satir  "
        f"ozet penceresi {sonuc['ozet_pencere_gun'].iloc[0]:>4.0f} gun  "
        f"soguk {soguk:>6,} (%{100 * soguk / len(sonuc):.1f})"
    )
    return sonuc


# ---------------------------------------------------------------- model


def oznitelikler(frame: pd.DataFrame) -> list[str]:
    adaylar = [k for k in frame.columns if k not in DISLANAN]
    return [k for k in adaylar if frame[k].dtype.kind in "ifbu" or k in KATEGORIK]


def kategorik_kodla(egitim: pd.DataFrame, *digerleri: pd.DataFrame) -> None:
    """Kategorikleri YERINDE ortak bir sozlukle kodlar.

    Ayri ayri kodlamak sessiz bir hata kaynagidir: egitimde 3 numara olan
    bolge, test'te 1 numaraya dusebilir ve model yanlis seviyeyi okur.
    """
    for kolon in KATEGORIK:
        seviyeler = pd.Index(sorted(egitim[kolon].dropna().unique()))
        for f in (egitim, *digerleri):
            f[kolon] = pd.Categorical(f[kolon], categories=seviyeler)


#: Egitimde YAPAY olarak sogutulacak trafo orani. Test'in gercek soguk
#: payina (%22,16) esitlendi ve tezgahta olculdu -- %10 ve %35 daha kotu,
#: yani tepe tam test oraninda. DropoutNet'in (NeurIPS 2017) mekanizmasi:
#: model servis anindaki girdi dagilimini EGITIMDE gorur.
SOGUK_MASKE_ORANI = 0.2216  # = TEST_SOGUK_PAYI (asagida ayni deger)

#: Sabit agac sayisi -- erken durdurma yerine. Olculdu: 200-1000 arasi duz.
SABIT_AGAC = 400

#: ``t_*`` = trafo gecmisinden turemis kolonlar; maskeleme bunlari siler.
_GECMIS_ONEKI = ("t_",)


#: REJIM UZMANLARI. ``None`` ise yonlendirme kapali (tek model, eski hal).
#:
#: Olculdu (2026-08-21 gece, 63 CatBoost fit): maske orani sicak ve soguk
#: satirlarda TERS yonde calisiyor ve iki egri de TEKDUZE --
#:
#:     maske   0,00    0,15    0,22    0,35    0,50    0,70    1,00
#:     sicak  0,8136  0,8128  0,8219  0,8239  0,8181  0,8830  1,6299
#:     soguk  1,8215  1,7851  1,7792  1,7852  1,7733  1,7688  1,7595
#:
#: Tek bir oran ikisini birden en iyi yapamaz; %22,16 bir uzlasmaydi.
#: Satiri rejimine gore yonlendirmek mesru, cunku test aninda bir trafonun
#: gecmisi olup olmadigini BILIYORUZ (``soguk_mu``). CatBoost'ta olculdu:
#: 1,10805 -> 1,09608, ucu blokta da ayni yonde.
#:
#: Bagimsiz dayanak: DropoutNet'in kendi oran taramasi (NeurIPS 2017,
#: Sekil 2) soguk baslangic icin TEKDUZE artiyor (0,378 -> 0,659, oran
#: 0 -> 0,9) ve ic optimum yok. NeurIPS hakemi tam bu soruyu sormus
#: ("neden maskeli tek model yerine ayri bir soguk model?"); yazarlar
#: cevap vermemis.
# GERI ALINDI 2026-08-23: soguk uzmanina eklenen ``ek_kolon`` (hafta gunu)
# ve harman agirliginin 3/1/1 -> 1/1/1 degistirilmesi LB'de OLCULDU ve
# ZARARLI cikti: v18 1,03370 -> v23 1,04820 (+0,0145). Sebep docs/35:
# ``tanim_num`` bire-bir trafo kimligi ve maskelemeden sag ciktigi icin
# CV soguk satirlarinin %48'i EZBERLENEBILIR (testte %0). Bu yuzden
# ``yaz25`` (ezber %97,2) her iki degisikligi de ONAYLADI, ``kis26``
# (ezber %0,0) ikisine de HAYIR dedi -- ve LB kis26'yi hakli cikardi.
# SOGUK REJIM KARARLARI BUNDAN SONRA ``kis26`` ILE VERILIR.
REJIM_AYARLARI: dict[str, dict[str, object]] | None = {
    # l2_leaf_reg=1 + depth=6 (2026-08-22): MERKEZLI olcumde +0,00628, uc
    # blokta da pozitif. Ham olcumde yalnizca +0,00317 gorunuyordu -- cunku
    # dogrulama kurgumuz kapasiteyi ARTIRAN degisiklikleri haksiz yere
    # cezalandiriyor (bkz. docs/27, §14). Uretimde o kurgusal sapma yok.
    # ``ek_koken``: bu uzman EK KOKENLI egitim setini gorsun mu. Olculdu
    # (2026-08-22, ``deney_koken_rejim.py``, eslenik, 3 tohum):
    #
    #     SICAK  ANA 0,80675 -> EK 0,79848   +0,00946  t=+1,46
    #     SOGUK  ANA 1,70349 -> EK 1,73612   -0,03273  t=-2,59  ZARARLI
    #
    # Neden ters yonde: ek kokenler AYNI (trafo, gun) satirini farkli ozet
    # pencereleriyle tekrar gosteriyor. Sicak uzmani icin bu gercek veri
    # artirmadir -- ``t_*`` ozetleri gercekten farkli geliyor. Soguk uzmani
    # maske 1,00'da calisiyor, yani butun ``t_*`` NaN; kopyalar arasinda
    # geriye yalnizca ``ozet_pencere_gun``, ``t_doluluk``, ``ufuk_gun``
    # farki kaliyor ve hedef BIREBIR ayni. Veri artirma degil, kopya
    # cogaltma -- ve etiketleri tarih boyunca yeniden agirliklandiriyor.
    #
    # v17 kosusu ikisini birden acmisti: genel skor berabere kaldi
    # (1,05818 -> 1,06049) ama test ikizi yaz25 +0,028 kotulesti, cunku
    # orada soguk kaybi (-0,077) sicak kazancini (+0,008) ezdi.
    # ``agirlik``: rejime ozel harman. Olculdu (2026-08-22,
    # ``deney_agirlik*.py``, 3 tohum torbalanmis, AYNI onbelleklenmis
    # tahminler uzerinde -- yani agirliklar arasi fark tohum gurultusu
    # TASIMIYOR, belirlenimci).
    #
    #   SICAK                          SOGUK
    #   (0,1,0) xgb   0,78953  EN IYI  (1,0,0) cat  1,70044  EN KOTU
    #   (1,0,0) cat   0,79848          (0,1,0) xgb  1,66614
    #   (0,0,1) lgbm  0,80177          (0,0,1) lgbm 1,66851
    #   (3,3,1)       0,77958          (1,1,1)      1,64308  EN IYI
    #   (2,2,1)       0,78007  SECILEN (2,2,2)      1,64308  (ayni oran)
    #   (3,1,1)       0,78280  ESKI    (3,1,1)      1,65354  ESKI
    #   (0,1,1)       0,79111          (0,1,1)      1,65441
    #
    # Iki uzman TERS yonde. Sicakta xgb en iyi aile, sogukta cat EN KOTU
    # aile -- ve eski 3/1/1 ikisinde de cat'e agirlik veriyordu.
    #
    # SICAK AGIRLIK DEGISIKLIGI DENENDI VE REDDEDILDI. Izgara (2,2,1)'in
    # (3,1,1)'den 0,0027 iyi oldugunu soyluyordu; URETIM DOGRULAMASINDA uc
    # blokta da KOTU cikti (yaz25 +0,0027, guz25 +0,0006, kis26 +0,0040).
    #
    # Neden yaniltti: izgara 3 tohum TORBALANMIS tahminler uzerindeydi,
    # uretimin dogrulama adimi ise TEK tohum (42). Iki farkli tahminci icin
    # optimum agirlik farkli. Gunun tekrarlayan dersi: olcum duzenegi
    # uretimden bir adim bile ayrilirsa, olctugun sey gonderdigin sey degil.
    #
    # Soguk degisikligi AYRISTIRILARAK olculdu (--sadece-dogrulama):
    #             yaz25     guz25     kis26    ORTALAMA
    #   v20     0,98863   1,05023   1,09700   1,04529
    #   A soguk 0,97559   1,04238   1,11354   1,04384  <- ALINDI
    #   B sicak 0,99129   1,05079   1,10104   1,04771  <- reddedildi
    #   ikisi   0,97828   1,04294   1,11752   1,04625
    #
    # A'da bloklar AYRISIYOR (yaz25 -0,0130, kis26 +0,0165). CV'den hangisinin
    # hakli oldugu soylenemez: yaz25 test'in mevsimsel ikizi ve kalibrasyon
    # iki kez tuttu, ama kis26 ozet penceresi uzunlugu ve soguk payi
    # bakimindan test'e daha yakin. Belirsizlik LB'de cozulecek.
    #
    # Sogukta egri TEKDUZE: cat agirligi 1->2->3->4->6 boyunca her adimda
    # kotulesiyor. Izgaradan rastgele secim degil, duz bir egilim. Ama
    # (0,1,1) kotu: cat katki veriyor, yalnizca baskin olmamali.
    # ``ek_kolon`` (2026-08-24 aksami): YALIN_CIKARILAN'in attigi p_/g_/gp_
    # ailesinden ONU sicak uzmana GERI VERILIYOR.
    #
    # NEDEN: yalin set 144->105 karari ``deney_ileri.py:731`` ile alindi ve o
    # rig uretimden DORT eksende ayriydi -- ek kokensiz (1,04M vs 2,86M),
    # maske 0,2216 (uretim 0,15), random_strength 1 (uretim 4), depth 5 /
    # l2 3 (uretim 6/1). Uretim esli rig'de yeniden olculdu
    # (``deney_pg_maske.py``, 3 blok x 3 tohum, teste agirliklandirilmis):
    #
    #     kol      yaz25     guz25     kis26    fark      SH      t    blok
    #     taban   0,80608   0,99674   0,87742  +0,0000
    #     +14     0,80122   0,98670   0,87307  +0,0059  0,0032  +1,88  3/3
    #     +10     0,80180   0,98278   0,86480  +0,0102  0,0028  +3,59  3/3  <- SECILEN
    #
    # Genel skora etki -0,00545. Bugun karar kuralini (t>=2 VE uc blok pozitif)
    # gecen TEK aday.
    #
    # NEDEN 14 DEGIL 10: ``p_gun_sayisi``, ``p_ilk_ofset``, ``p_son_ofset``,
    # ``p_yayilma`` panel penceresine gore HAM GUN. Ana bloklar 121-122 gun ve
    # TEST 122 gun, ama EK_KOKENLER icinde sub25=59 ve bah26=90 gun var --
    # o dort kolon ek satirlarin bir kisminda testte HIC gorulmeyen sikistirilmis
    # bir olcekte geliyor. ``p_doluluk``/``p_pencere_payi`` normalize, ``g_``/
    # ``gp_`` grup istatistigi; onlarda bu sorun yok. Olcum bunu dogruladi:
    # dortunu atmak +0,0059'u +0,0102'ye cikariyor.
    #
    # Maskeleme etkilenmez: bu kolonlar ``t_`` onekli DEGIL, yani gecmis
    # maskesinin disindalar -- deneyde de oyle olculdu.
    "sicak": {
        "maske": 0.15,
        "cat": {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6},
        "ek_koken": True,
        "ek_kolon": (
            "g_guc_kova",
            "g_ilce_kova_n",
            "g_ilce_kova_ort",
            "g_ilce_log_ort",
            "g_kova_log_ort",
            "gp_ilce_ay",
            "gp_ilce_hg",
            "gp_kova_ay",
            "p_doluluk",
            "p_pencere_payi",
        ),
        "agirlik": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4},
    },
    # ``ek_kolon``: bu uzmana YALIN_CIKARILAN'a ragmen geri verilen kolonlar.
    # Olculdu (2026-08-22, ``deney_soguk_hafta.py``, eslenik, 5 tohum, 15 hucre):
    #
    #     SOGUK  TABAN 1,70391 -> +HAFTA 1,70120   +0,00274  SH 0,00115  t=+2,39
    #       yaz25 +0,00214 (3/5)  guz25 +0,00439 (5/5)  kis26 +0,00169 (4/5)
    #     SICAK  +HAFTA  -0,00270  t=-0,83  ZARARLI (deney_takvim.py)
    #
    # Ikinci rejim ayrimi: ek kokenler gibi hafta gunu de yalnizca BIR uzmana
    # yariyor. Mekanizma: soguk uzmani maske 1,00'da calisiyor ve elinde
    # trafoyu ayirt eden HICBIR sey yok -- butun t_* kolonlari NaN. Hafta
    # gunu onun icin nadir bulunan gercek bir sinyal. Sicak uzmaninin ise
    # gecmis ozetleri (t_hg_genligi, t_hg_sapma) haftalik deseni zaten
    # tasiyor, ustelik 105 kolonun icinde seyreltme maliyeti agir basiyor.
    #
    # tk_ oneki _GECMIS_ONEKI ("t_") ile eslesmiyor, yani maskelenmiyor.
    # +HAFTA+TAKVIM (tk_ay, tk_yilin_gunu, tatil_mi eklenmis) ZARARLI
    # cikti (soguk -0,02323, kis26 -0,074) -- yalnizca bu iki kolon.
    #
    # SOGUK HARMAN -> YALNIZ cat (2026-08-23, ``deney_soguk_taban.py``).
    # Eski 3/1/1 hukmu KIRLI bloklarin ortalamasindan geliyordu. kis26 tek
    # basina bakildiginda ve YENI son islem (gun korumali buzme, beta=0,25)
    # altinda olculdu -- ayni onbelleklenmis tahminler, saf aritmetik:
    #
    #     harman     beta=1,00   0,30      0,25      0,20
    #     3/1/1        1,86931  1,83083  1,83041  1,83031
    #     5/1/1        1,85717  1,82767  1,82750  1,82758
    #     YALNIZ cat   1,84106  1,82245  1,82250  1,82274   <- SECILEN
    #
    # beta ne olursa olsun siralama ayni: cat tek basina, harmandan iyi.
    # kis26 soguk kazanci -0,0079; d(genel)/d(soguk)=0,377 ile genel -0,0030.
    # ``ek_kolon`` (2026-08-24 aksami): GRUP SEVIYELERI soguk uzmana verildi.
    #
    # NEDEN: soguk uzman maske 1,00'da calisir, butun ``t_*`` NaN'dir ve
    # elinde trafoyu ayirt eden HICBIR gecmis yoktur. ``g_*``/``gp_*`` ise
    # ozet penceresinden hesaplanan GRUP istatistikleridir -- ilce x kVA
    # kovasi ortalamalari. Gecmisi olmayan bir model icin tam olarak eksik
    # olan sey budur: bir SEVIYE kestirimi. docs/30 bunu bagimsiz olarak
    # olcmustu: ilce x kova, soguk seviye kestiricileri arasinda EN IYISI
    # (1,9867; tanim onekleri 2,10-2,13). Modele verilmemesinin tek sebebi
    # ``YALIN_CIKARILAN``in ``g_`` onekini global silmesiydi.
    #
    # OLCULDU (``deney_soguk_grup_kolon.py``, kis26, son islem sonrasi,
    # testin kVA karisimina agirliklandirilmis, 3 tohum eslenik):
    #
    #     kol          HAM kis26   kVA duzeltilmis   fark     SH      t
    #     taban         1,82605      1,98505        +0,0000
    #     +grup (8)     1,81837      1,95269        +0,0318  0,0023  +13,71  3/3
    #     +grup+panel   1,85064      2,01490        -0,0306  0,0027  -11,17  0/3
    #
    # Genel skora etki -0,01189. Panel doluluk kolonlari (``p_doluluk``,
    # ``p_pencere_payi``) AGIR ZARARLI ve DISARIDA birakildi: gecmisi olmayan
    # bir trafo icin kendi panel dolulugu dejenere bir sayidir.
    #
    # SIZINTI ELENDI: ``g_*`` ``ozet`` penceresinden gelir ve ``blok_kur``
    # icinde sert bir kontrol var (ozet.max() >= etiket.min() ise
    # RuntimeError). Ortalama BASKA trafolardan, etiket oncesi donemden.
    # kis26 soguk trafolarinin %0'i baska katlarda mevcut (docs/35).
    #
    # IKI UYARI, kayda geciyor:
    #   1. 2026-08-23'te soguga ``ek_kolon`` (HAFTA GUNU) + harman 3/1/1->1/1/1
    #      BIRLIKTE gonderildi ve LB'de yandi (v18 1,03370 -> v23 1,04820).
    #      O paket grup istatistigi DEGILDI, ama "soguga kolon ekle" bir kez
    #      LB'de yanmis durumda.
    #   2. ``son_islem_gun.py`` AYNI bilgi kanalini (ilce x kova hucre
    #      tablosu) SON ISLEMDE kullanip LB'de curudu (+0,00414). Fark: orada
    #      sabit 0,40 agirlik DAYATILIYORDU; burada kolon modele veriliyor ve
    #      model ona ne kadar guvenecegini kosullu ogreniyor.
    # Bu yuzden IZOLE bir LB gonderimiyle sinanmalidir.
    "soguk": {
        "maske": 1.00,
        "cat": {"depth": 7},
        "ek_koken": False,
        "ek_kolon": (
            "g_guc_kova",
            "g_ilce_kova_n",
            "g_ilce_kova_ort",
            "g_ilce_log_ort",
            "g_kova_log_ort",
            "gp_ilce_ay",
            "gp_ilce_hg",
            "gp_kova_ay",
        ),
        "agirlik": {"cat": 1.0},
    },
}

#: EK KOKENLERI EGITIME KAT. Olculdu (2026-08-22, ``deney_koken2.py``):
#: eslenik fark +0,00782, SH 0,00260, **t = +3,01** -- gunun esigi gecen
#: TEK degisikligi. Uc blokta da pozitif, test ikizi yaz25 dahil (+0,00686).
#:
#: Egitim 1.038.737 -> 2.855.584 satir. Satir sayisi kadar BILGI artmaz
#: (ortusen bloklar korele); kazanc, ayni etiketin farkli tazelikteki
#: ozetlerle tekrar gorulmesinden geliyor -- model "eski ozete ne kadar
#: guvenmeli" sorusunu uc ornek yerine dokuz ornekten ogreniyor.
#:
#: Olculen deger bir ALT SINIR: dogrulamada hedef blokla kesisen kokenler
#: ``kokenleri_ayikla`` ile ATILIYOR, uretimde ise hepsi kullanilabiliyor
#: cunku test butun egitim verisinden sonra geliyor.
EK_KOKEN_KULLAN = True

#: YALIN OZNITELIK SETI -- bu onekli kolonlar CIKARILIR (144 -> 105).
#:
#: Olculdu (2026-08-22 gece, yigin deneyi): yalin set + soguk uzmani d7 +
#: sicak uzmani random_strength=4, BIRLIKTE 1,08143 -> 1,05194, yani
#: -0,0295. Esik 0,01995.
#:
#: Kazanc PARCALARIN TOPLAMINDAN buyuk (beklenen 0,017) ve tamami SOGUK
#: tarafta: sicak 0,7979 -> 0,7962 (hicbir sey), soguk 1,7404 -> 1,6576.
#: Izole olculdugunde soguk derinlik kazanci 0,015'ti; birlikte 0,083.
#:
#: Mekanizma: soguk uzmani gecmis gormuyor, elinde kalan ~125 kolonun
#: cogu ilce duzeyinde hava/takvim -- yani trafolari birbirinden AYIRMAYAN
#: kolonlar. 39 gurultu kolonunu atmak kit sinyali yogunlastiriyor,
#: derinlik 7 de onu kullanabilmesini sagliyor. Tek basina ne budama ne
#: derinlik yetiyor; ikisi birlikte aciyor.
YALIN_CIKARILAN: tuple[str, ...] = (
    # OLCULEN yalin set: takvim + panel yapisi + grup seviye/profil.
    "tk_",
    "tatil",
    "ramazan",
    "p_",
    "g_",
    "gp_",
    # OLCUMUN DISINDA KALANLAR -- yigin 144 kolonluk bir ONBELLEK uzerinde
    # olculdu (dun 17:10'da kurulmus), uretim ise 151 kolon kuruyor. Aradaki
    # 7 kolon olcume HIC girmedi:
    #   * nufus ailesi (5): onbellekten SONRA hatta girdi. Daha once
    #     olculmustu ve DEGERSIZ cikti (artikla korelasyon 0,038). Yiginin
    #     kazanci tam olarak 39 gurultu kolonunu atmaktan geliyor; geri
    #     5 gurultu kolonu koymak o kazanci kismen geri alirdi.
    #   * t_mevsim_* (2): bu gece eklendi, HIC olculmedi.
    # Ikisi de "kotu oldugu icin" degil, "olcumun disinda oldugu icin"
    # cikariliyor. Onbellek yenilenip yeniden olculunce karar gozden
    # gecirilmeli -- ozellikle t_mevsim_*, ki fikir umut verici.
    "nufus",
    "alan_km2",
    "ilce_nufus_yogunlugu",
    "trafo_basina_nufus",
    "kva_basina_nufus",
    "t_mevsim_",
)


def soguk_maskele(
    cerceve: pd.DataFrame, kolonlar: list[str], tohum: int, oran: float | None = None
) -> pd.DataFrame:
    """Trafolarin ``oran`` kadarini yapay olarak sogutur.

    Trafo BAZINDA maskelenir: bir trafonun bazi gunlerinde gecmisi olup
    bazilarinda olmamasi diye bir sey yoktur.

    ``oran=None`` ise ``SOGUK_MASKE_ORANI`` kullanilir (yonlendirmesiz hal).
    """
    if oran is None:
        oran = SOGUK_MASKE_ORANI
    rng = np.random.default_rng(tohum)
    trafolar = cerceve[GRUP].unique()
    secilen = set(rng.choice(trafolar, size=int(len(trafolar) * oran), replace=False))
    maske = cerceve[GRUP].isin(secilen).to_numpy()
    sonuc = cerceve.copy()
    sonuc.loc[maske, [k for k in kolonlar if k.startswith(_GECMIS_ONEKI)]] = np.nan
    sonuc.loc[maske, "soguk_mu"] = 1
    return sonuc


def ofsetli_hedef(cerceve: pd.DataFrame) -> np.ndarray:
    """``log1p(tuketim) - log1p(guc)`` -- kapasite ofsetli hedef.

    Log uzayinda satir-basi SABIT bir kaydirma oldugu icin L2 optimumu
    degismez; metrik acisindan birebir ayni problem. Ama agaclarin artik
    olcegi merdivenlerle yaklastirmasi gerekmez -- kapasiteyi bedava ve TAM
    alirlar. Esneklik olculdu: satir duzeyinde 1,0630, trafo duzeyinde
    1,0727, yani ofset formu (esneklik = 1 varsayimi) hakli.

    Tezgahta olculdu (3 tohum, 3 blok, test karisimina agirliklandirilmis):
    taban 1,19673 -> ofset 1,16155, delta -0,0352, esik 0,0183.
    """
    return (np.log1p(cerceve[HEDEF].clip(lower=0.0)) - np.log1p(cerceve["guc"])).to_numpy()


def ofseti_geri_ekle(log_tahmin: np.ndarray, cerceve: pd.DataFrame) -> np.ndarray:
    return log_tahmin + np.log1p(cerceve["guc"]).to_numpy()


def rmsle(gercek: np.ndarray, tahmin: np.ndarray) -> float:
    t = np.clip(np.asarray(tahmin, dtype="float64"), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(t) - np.log1p(np.asarray(gercek))) ** 2)))


#: Harman agirliklari. Olculdu (3 tohum, 3 blok, test karisimina agirlikli):
#:
#:     cat agirlikli (4/1/1)  1,10621
#:     cat + xgb (3/1)        1,10644
#:     cat agirlikli (3/1/1)  1,10665     <-- secilen
#:     cat agirlikli (2/1/1)  1,10886
#:     cat + xgb (1/1)        1,11039
#:     cat tek basina         1,11618
#:     esit ucu (1/1/1)       1,11666
#:
#: 4/1/1 ile 3/1/1 arasindaki 0,0004 fark gurultunun cok altinda; 3/1/1
#: secildi cunku zayif uyelere biraz daha agirlik verir. CatBoost gercek
#: test'te beklendigi kadar iyi cikmazsa, daha dengeli bir harman daha az
#: kaybettirir -- olculemeyen bir farkta daha DAYANIKLI olani secmek.
#:
#: Esit agirligin KOTU olmasi tesadufi degil: Krogh & Vedelsby ozdesligi
#: "harman ORTALAMA uyeden kotu olamaz" der, EN IYI uyeden degil. Uyeler
#: arasi fark buyudukce esit agirlik zarar verir. Zayif uyeleri tamamen
#: atmak da cozum degil (cat+xgb 3/1 = 1,10644 > 3/1/1); kucuk agirlikla
#: tutmak en iyisi.
AILE_AGIRLIKLARI: dict[str, float] = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}


def aile_modeli(aile: str, tohum: int, *, hizli: bool, cat_ustyazim=None):  # noqa: ANN001, ANN201
    """Uc GBDT ailesinden biri. Ayarlar ``scripts/deney.py`` ile BIREBIR ayni.

    ``cat_ustyazim`` YALNIZCA CatBoost'a uygulanir. Bu bilerek boyle:
    ``depth`` ve ``random_strength`` CatBoost parametreleri, ve olculdu
    (2026-08-22) -- LightGBM/XGBoost'a gecirildiginde saklaniyor ama
    ``max_depth``i DEGISTIRMIYOR (lgbm -1, xgb 8 kaliyor), yani atillar.
    Tezgahta olculen yigin da tam olarak bu davranisla olculdu; uretimde
    ACIKCA yazmak, o sessiz atilligi belgelenmis bir karara cevirir.
    """
    if aile == "lgbm":
        return model_kur(hizli=hizli, tohum=tohum)
    if aile == "xgb":
        import xgboost as xgb

        return xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=150 if hizli else SABIT_AGAC,
            learning_rate=0.05,
            max_depth=8,
            min_child_weight=20,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_lambda=2.0,
            random_state=tohum,
            n_jobs=-1,
            tree_method="hist",
            enable_categorical=True,
            verbosity=0,
        )
    if aile == "cat":
        import catboost as cb

        # Derinlik 5 / 250 iterasyon -- taranarak bulundu. Onceki ayar
        # (d8/400) yuzeyin en kotu kosesiydi: 1,12817 ye karsi 1,11075.
        p: dict[str, object] = dict(
            loss_function="RMSE",
            iterations=100 if hizli else 250,
            learning_rate=0.05,
            depth=5,
            l2_leaf_reg=3.0,
            rsm=0.75,
            random_seed=tohum,
            verbose=0,
            allow_writing_files=False,
        )
        p.update(cat_ustyazim or {})
        return cb.CatBoostRegressor(**p)
    raise ValueError(f"bilinmeyen aile: {aile}")


def aile_tahmini(
    aile: str,
    egitim: pd.DataFrame,
    hedef_cerceve: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    hizli: bool,
    maske_orani: float | None = None,
    cat_ustyazim: dict[str, object] | None = None,
) -> np.ndarray:
    """Bir aileyi egitip LOG UZAYINDA tahmin dondurur.

    Log uzayinda donmesi sart: harman ``expm1(mean(log1p(...)))`` seklinde
    yapiliyor. Krogh & Vedelsby ayrismasi (NeurIPS 1994) bir OZDESLIKTIR ve
    RMSLE log uzayinda kareli hata oldugu icin burada birebir gecerlidir --
    ama yalnizca birlestirici aritmetik ortalamaysa (Wood ve ark., JMLR
    2023). Ham uzayda ortalama almanin boyle bir garantisi YOK.
    """
    egitim = soguk_maskele(egitim, kolonlar, tohum, maske_orani)
    y = ofsetli_hedef(egitim)
    if aile == "sinir_agi":
        # 4. UYE -- agac degil. Neden: uc GBDT ailesinin hata korelasyonu
        # 0,914 ve cesitliligin payi yalnizca %5,6 (olculdu 2026-08-23,
        # scripts/teshis_cesitlilik.py). Farkli tumevarim onyargisi olan bir
        # uye, TEK BASINA daha kotu olsa bile harmani duzeltir.
        from sinir_agi import SinirAgi

        ag = SinirAgi(tohum=tohum, hizli=hizli)
        ag.fit(egitim[kolonlar], y)
        return ofseti_geri_ekle(ag.predict(hedef_cerceve[kolonlar]), hedef_cerceve)
    model = aile_modeli(aile, tohum, hizli=hizli, cat_ustyazim=cat_ustyazim)
    x_egitim, x_hedef = egitim[kolonlar], hedef_cerceve[kolonlar]
    if aile == "cat":
        # CatBoost kategorik dtype'i dogrudan yutmuyor; ayri bildirilmeli.
        x_egitim, x_hedef = x_egitim.copy(), x_hedef.copy()
        kategorik = [k for k in KATEGORIK if k in x_egitim.columns]
        for k in kategorik:
            x_egitim[k] = x_egitim[k].astype(str)
            x_hedef[k] = x_hedef[k].astype(str)
        model.fit(x_egitim, y, cat_features=kategorik)
    else:
        model.fit(x_egitim, y)
    return ofseti_geri_ekle(model.predict(x_hedef), hedef_cerceve)


def rejim_tahmini(
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    hizli: bool,
    dar_egitim: pd.DataFrame | None = None,
) -> np.ndarray:
    """Rejim uzmanlarini egitip satiri rejimine gore YONLENDIRIR.

    Neden mesru: test aninda bir trafonun gecmisi olup olmadigini biliyoruz
    (``soguk_mu``), yani yonlendirme etiket kullanmiyor. Neden gerekli:
    maskeleme orani sicak ve soguk satirlarda ters yonde calisiyor, bkz.
    ``REJIM_AYARLARI``.

    Uzman basina AILE HARMANI ayni kalir (``AILE_AGIRLIKLARI``); degisen
    her uzmanin gordugu maske orani, CatBoost ustyazimi ve EGITIM SETI.

    ``dar_egitim`` verilirse, ``ek_koken: False`` isaretli uzmanlar onu
    kullanir -- ek kokensiz, yalnizca ana bloklar. Olculdu: ek kokenler
    sicak uzmanina yariyor, soguk uzmanina zarar veriyor (bkz.
    ``REJIM_AYARLARI``). Verilmezse butun uzmanlar ``egitim``i gorur.

    Tahmin yalnizca ilgili satirlar icin uretilir -- geri kalanini
    hesaplamanin anlami yok.

    ``REJIM_AYARLARI is None`` ise eski tek-model davranisina duser.
    """
    if REJIM_AYARLARI is None:
        toplam = sum(AILE_AGIRLIKLARI.values())
        return (
            sum(
                w * aile_tahmini(a, egitim, hedef, kolonlar, tohum, hizli=hizli)
                for a, w in AILE_AGIRLIKLARI.items()
            )
            / toplam
        )

    soguk = (hedef["soguk_mu"] == 1).to_numpy()
    cikti = np.zeros(len(hedef), dtype="float64")
    for rejim, ayar in REJIM_AYARLARI.items():
        maske = soguk if rejim == "soguk" else ~soguk
        if not maske.any():
            continue
        alt = hedef.loc[maske]
        kaynak = egitim
        if not ayar.get("ek_koken", True) and dar_egitim is not None:
            kaynak = dar_egitim
        # Rejime ozel geri verilen kolonlar (bkz. ``REJIM_AYARLARI``).
        ek_kolon = [k for k in ayar.get("ek_kolon", ()) if k not in kolonlar]  # type: ignore[union-attr]
        eksik = [k for k in ek_kolon if k not in kaynak.columns or k not in alt.columns]
        if eksik:
            raise RuntimeError(f"{rejim} uzmaninin ek_kolon'u cercevede yok: {eksik}")
        kol = kolonlar + ek_kolon
        # Rejime ozel harman agirligi. Iki uzman FARKLI problemler cozuyor:
        # sicakta xgb en iyi aile, sogukta cat EN KOTU aile (bkz. REJIM_AYARLARI).
        agirlik = ayar.get("agirlik", AILE_AGIRLIKLARI)
        toplam = sum(agirlik.values())  # type: ignore[union-attr]
        cikti[maske] = (
            sum(
                w
                * aile_tahmini(
                    a,
                    kaynak,
                    alt,
                    kol,
                    tohum,
                    hizli=hizli,
                    maske_orani=float(ayar["maske"]),  # type: ignore[arg-type]
                    cat_ustyazim=ayar.get("cat"),  # type: ignore[arg-type]
                )
                for a, w in agirlik.items()  # type: ignore[union-attr]
            )
            / toplam
        )
    return cikti


def model_kur(*, hizli: bool, tohum: int):  # noqa: ANN201 - lgb tipi kosullu import
    """LightGBM. Ayarlar ``scripts/deney.py`` tezgahiyla BIREBIR ayni.

    Bu es olmazsa tezgahta olculen her sey buraya tasinamaz: bir kez
    hizalanmadigi icin (tezgah lr=0,05, betik lr=0,08) ayni yapilandirma
    1,15516 yerine 1,17813 verdi ve fark ozelliklerden saniliyordu.
    ``--hizli`` artik yalnizca AGAC SAYISINI dusurur, ogrenme hizini degil.
    """
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        objective="regression",  # log1p uzayinda RMSE == ham uzayda RMSLE
        n_estimators=150 if hizli else SABIT_AGAC,
        learning_rate=0.05,
        num_leaves=255,
        min_child_samples=40,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=tohum,
        n_jobs=-1,
        verbose=-1,
    )


#: Test kumesindeki SOGUK satir payi -- olculdu: 158.369 / 714.688.
#: Iki yerde kullaniliyor ve ikisi de zorunlu:
#:   1) egitim agirliklari: bloklarda soguk pay %7,5-13,9, yani model soguk
#:      rejimi test'tekinden AZ onemsiyor.
#:   2) dogrulama raporu: bir blogun ham RMSLE'si kendi soguk payini tasir,
#:      dolayisiyla test'i temsil etmez. Test payina yeniden agirliklandirilmis
#:      birlesim, karsilastirilabilir tek sayidir.
TEST_SOGUK_PAYI = 0.2216


def soguk_agirliklari(soguk_mu: pd.Series, *, hedef_pay: float = TEST_SOGUK_PAYI) -> np.ndarray:
    """Soguk satirlarin toplam agirliktaki payini ``hedef_pay``a cikarir.

    Sicak satirlarin agirligi 1,0'da sabit; yalnizca soguklar olceklenir.
    Boylece toplam agirlik olcegi kayarsa ogrenme hizi da kaymaz.
    """
    soguk = soguk_mu.to_numpy() == 1
    n_soguk, n_sicak = int(soguk.sum()), int((~soguk).sum())
    if n_soguk == 0 or n_sicak == 0:
        return np.ones(len(soguk_mu), dtype="float64")
    carpan = (hedef_pay / (1.0 - hedef_pay)) * (n_sicak / n_soguk)
    return np.where(soguk, carpan, 1.0).astype("float64")


@dataclass(frozen=True)
class Dogrulama:
    """Bir blogun dogrulama sonucu -- toplam ve REJIM BAZINDA."""

    genel: float
    sicak: float
    soguk: float

    @property
    def test_agirlikli(self) -> float:
        """Test'in soguk/sicak karisimina yeniden agirliklandirilmis RMSLE.

        RMSLE karesel bir ortalamanin karekoku oldugu icin rejimler kare
        uzayinda birlesir. Ham ``genel`` skoru blogun KENDI karisimini
        tasir (yaz25'te soguk yalnizca %7,5) ve bu yuzden test'i temsil
        etmez -- iki blogu birbiriyle kiyaslamak icin bile kullanilamaz.
        """
        return float(
            np.sqrt((1 - TEST_SOGUK_PAYI) * self.sicak**2 + TEST_SOGUK_PAYI * self.soguk**2)
        )

    def satir(self, ad: str, n: int) -> str:
        return (
            f"  {ad:6} RMSLE {self.genel:.5f}  |  sicak {self.sicak:.5f}  "
            f"soguk {self.soguk:.5f}  |  TEST-AGIRLIKLI {self.test_agirlikli:.5f}  "
            f"|  {n:>7,} satir"
        )


def egit_ve_olc(
    egitim: pd.DataFrame,
    dogrulama: pd.DataFrame,
    kolonlar: list[str],
    *,
    hizli: bool,
    dar_egitim: pd.DataFrame | None = None,
) -> Dogrulama:
    """Egitir, dogrular ve skoru IKI REJIME AYIRIR.

    Toplam skor tek basina yaniltici: test satirlarinin %22,16'si soguk
    baslangic ve o rejimde model bambaska bir bilgi kumesiyle calisiyor.
    Genel skor iyilesirken soguk rejim kotulesiyorsa bunu gormek sart --
    tek sayiya bakan biri bu takasi HIC fark etmez.
    """
    # Erken durdurma YOK. Olculdu: skor 200-1000 agac arasi duz, ama erken
    # durdurma her kosuda baska yerde durup GURULTU uretiyordu (22-382 agac).
    # Sabit agac, tezgahtaki olcumlerle ayni kosullari verir.
    tahmin = np.expm1(
        rejim_tahmini(egitim, dogrulama, kolonlar, 42, hizli=hizli, dar_egitim=dar_egitim)
    )
    gercek = dogrulama[HEDEF].to_numpy()
    soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
    return Dogrulama(
        genel=rmsle(gercek, tahmin),
        sicak=rmsle(gercek[~soguk], tahmin[~soguk]) if (~soguk).any() else float("nan"),
        soguk=rmsle(gercek[soguk], tahmin[soguk]) if soguk.any() else float("nan"),
    )


# ---------------------------------------------------------------- ana akis


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hizli", action="store_true", help="az agac, hizli deneme")
    ap.add_argument("--gonder", metavar="NOT", help="uret ve Kaggle'a gonder")
    ap.add_argument("--tohum", type=int, default=3, help="son egitimde ortalanacak tohum sayisi")
    ap.add_argument("--cikti", default="tuketim_v1.csv")
    ap.add_argument(
        "--tohum-baslangic",
        type=int,
        default=100,
        help="son egitimde ilk tohum. Var olan bir gonderime EK tohum uretip log "
        "uzayinda birlestirmek icin kullanilir: yanlilik degismez, varyans duser.",
    )
    ap.add_argument(
        "--dogrulama-atla",
        action="store_true",
        help="4/5 dogrulamayi atla -- yapilandirma degismediyse tekrar kosmak bos zaman",
    )
    ap.add_argument(
        "--sadece-dogrulama",
        action="store_true",
        help="4/5 dogrulamayi kosup dur -- yapilandirma karsilastirmasi icin",
    )
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 74)
    print("GRID UP -- TRAFO BAZLI GUNLUK TUKETIM")
    print("=" * 74)

    print("\n1/5  YUKLEME + LOKASYON")
    tr, te = yukle()
    tr, te = lokasyon_ayristir(tr), lokasyon_ayristir(te)
    print(f"  train {len(tr):,} satir | test {len(te):,} satir")
    print(f"  ilce  {tr['ilce_key'].nunique()} (train) / {te['ilce_key'].nunique()} (test)")

    print("\n2/5  HAVA + TAKVIM")
    hava = hava_yukle()
    print(f"  hava tablosu {hava.shape[0]:,} satir x {hava.shape[1]} kolon")
    tr, te = hava_ekle(tr, hava), hava_ekle(te, hava)
    bos = tr[ISIL_KOLONLAR[0]].isna().mean()
    if bos > 0.001:
        raise RuntimeError(f"hava eslesmedi: train'de %{100 * bos:.1f} NaN")
    print(f"  hava eslesmesi: train %{100 * (1 - bos):.2f}")
    tr, te = takvim_ekle(tr), takvim_ekle(te)
    tr, te = yas_ekle(tr, te)
    tr, te = kimlik_ekle(tr, te)
    tr, te = statik_ilce_ekle(tr, te)
    tr, te = ilce_yapisi_ekle(tr, te)
    tr, te = ulusal_ekle(tr, te)
    print(
        f"  ulusal endeks: train NaN %{100 * tr['ulusal_gunluk'].isna().mean():.2f}"
        f" | test NaN %{100 * te['ulusal_gunluk'].isna().mean():.2f}"
    )

    print("\n3/5  BLOKLAR (yuvarlanan koken)")
    egitim = egitim_kur(tr)
    dar = egitim  # ek kokensiz -- soguk uzmani bunu gorur
    if EK_KOKEN_KULLAN:
        ek = ek_kokenleri_kur(tr)
        fark = set(egitim.columns) ^ set(ek.columns)
        if fark:
            # Sessizce kesismek, olculenden BASKA bir model uretmek demek.
            raise RuntimeError(f"ek koken kolonlari ana bloklardan farkli: {sorted(fark)}")
        egitim = pd.concat([egitim, ek[egitim.columns]], ignore_index=True)
        print(f"  ek kokenlerle egitim {len(egitim):,} satir (dar set {len(dar):,})")
    test = test_kur(tr, te)

    kolonlar = oznitelikler(egitim)
    kolonlar = [k for k in kolonlar if k in test.columns]
    if YALIN_CIKARILAN:
        tam = len(kolonlar)
        kolonlar = [k for k in kolonlar if not k.startswith(YALIN_CIKARILAN)]
        print(f"  yalin set: {tam} -> {len(kolonlar)} kolon ({tam - len(kolonlar)} cikarildi)")
    if dar is egitim:
        kategorik_kodla(egitim, test)
    else:
        # ``dar`` concat ONCESI cerceveye isaret ediyor; ayri kodlanmali.
        # Seviyeler GENIS cerceveden geliyor, yani ikisi ayni sozlugu paylasir.
        kategorik_kodla(egitim, dar, test)
    print(f"\n  {len(kolonlar)} oznitelik")

    print("\n4/5  DOGRULAMA")
    sonuclar: dict[str, Dogrulama] = {}
    for b in BLOKLAR if not args.dogrulama_atla else ():
        dogrulama = egitim[egitim["_blok"] == b.ad]
        # Ek kokenler bilerek ortusuyor; dogrulamada ortusme SIZINTIDIR.
        # ``kokenleri_ayikla`` hedef blokla tek gun bile kesisen her kokeni
        # atar -- ana bloklarin kendisi dahil.
        kalan = (
            kokenleri_ayikla(egitim, b.ad) if EK_KOKEN_KULLAN else egitim[egitim["_blok"] != b.ad]
        )
        kalan_dar = dar[dar["_blok"] != b.ad]
        sonuclar[b.ad] = egit_ve_olc(
            kalan, dogrulama, kolonlar, hizli=args.hizli, dar_egitim=kalan_dar
        )
        print(sonuclar[b.ad].satir(b.ad, len(dogrulama)))
    if sonuclar:
        print(f"  ORTALAMA (ham)            {np.mean([s.genel for s in sonuclar.values()]):.5f}")
        print(
            "  ORTALAMA (test-agirlikli) "
            f"{np.mean([s.test_agirlikli for s in sonuclar.values()]):.5f}"
        )
        print(
            f"  YAZ IKIZI (yaz25)         {sonuclar['yaz25'].test_agirlikli:.5f}"
            f"  <-- test donemine en yakin, test karisimina agirliklandirilmis"
        )
    else:
        print("  --dogrulama-atla: bu adim atlandi")

    if args.sadece_dogrulama:
        print(f"\n--sadece-dogrulama: son egitim ATLANDI  ({(time.time() - t0) / 60:.1f} dakika)")
        return 0

    print("\n5/5  SON EGITIM + GONDERIM")

    # Son model butun bloklarla egitilir; ayrilmis dogrulama kumesi kalmaz.
    # Erken durdurma YOK -- kazandirmiyordu ama her kosuda baska yerde durup
    # (22-382 agac) olculebilirligi bozuyordu. Her ailenin agac/iterasyon
    # sayisi taranarak sabitlendi (bkz. ``aile_modeli``).
    print(f"  harman: {AILE_AGIRLIKLARI}  x {args.tohum} tohum (log uzayinda)")
    print(
        f"  rejim uzmanlari: {REJIM_AYARLARI}"
        if REJIM_AYARLARI
        else f"  yonlendirme KAPALI, tek maske {SOGUK_MASKE_ORANI}"
    )

    # Tohumlar LOG UZAYINDA ortalanir: expm1(mean(log1p(...))). Krogh &
    # Vedelsby ayrismasi (NeurIPS 1994) bir ozdesliktir ve RMSLE log uzayinda
    # kareli hata oldugu icin burada birebir gecerlidir -- ama Wood ve ark.
    # (JMLR 2023) uyariyor: birlestirici degisirse garanti kalkar. Ham uzayda
    # ortalama almanin boyle bir garantisi YOK.
    #
    # Her tohum KENDI soguk maskesini alir: maske de bir cesitlilik kaynagi,
    # ve hepsine ayni maskeyi vermek o cesitliligi bosa harcardi.
    birikim = np.zeros(len(test), dtype="float64")
    for i in range(args.tohum):
        t_tohum = time.time()
        tohum = args.tohum_baslangic + i
        birikim += rejim_tahmini(egitim, test, kolonlar, tohum, hizli=args.hizli, dar_egitim=dar)
        print(f"    tohum {tohum} ({i + 1}/{args.tohum}) bitti ({time.time() - t_tohum:.0f} sn)")
    tahmin = np.clip(np.expm1(birikim / args.tohum), 0.0, None)

    GONDERIM.mkdir(parents=True, exist_ok=True)
    yol = GONDERIM / args.cikti
    pd.DataFrame({"id": test["id"].to_numpy(), HEDEF: tahmin}).to_csv(yol, index=False)
    print(f"  yazildi: {yol}  ({len(tahmin):,} satir)")
    print(
        f"  tahmin: min={tahmin.min():.1f} medyan={np.median(tahmin):.1f} "
        f"ort={tahmin.mean():.1f} max={tahmin.max():.3e}"
    )

    if args.gonder:
        print("\n  Kaggle'a gonderiliyor...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "competitions",
                "submit",
                "-c",
                "grid-up-datathon",
                "-f",
                str(yol),
                "-m",
                args.gonder,
            ],
            check=False,
            cwd=KOK,
        )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
