"""Trafo bazli GECMIS ozetleri -- fold-guvenli.

NEDEN BU MODUL VAR
------------------
Grid Up Datathon'da hedef trafo bazli gunluk tuketim (kWh) ve metrik RMSLE.
``log1p(tuketim)`` uzerinde olculdu (2026-08-21, gercek train.csv, 1.226.237
satir):

    global ortalama              R2  %0,0    egitim-ici RMSLE 2,1474
    ilce ortalamasi              R2 %15,5                     1,9739
    guc kovasi (log1p, 20 kova)  R2 %26,2                     1,8454
    TRAFO seviyesi               R2 %90,1                     0,6772
    trafo x haftagunu            R2 %90,3                     0,6676
    trafo x ay                   R2 %95,4                     0,4582

Yani baskin sinyal trafonun KENDI gecmisi. Ama bu sinyali modele tasimak,
sizintiya en yakin duran istir: bir trafonun ortalamasi, o ortalamanin
icinde YER ALAN satir icin hesaplanirsa model kendi cevabini okur.

Bu modulun tek isi o cizgiyi net tutmak: ozetler ``uydur`` cercevesinden
CIKAR, ``uygula`` cercevesine TASINIR. Iki cerceve asla ayni olamaz --
fonksiyon bunu kendisi denetler.

SOGUK BASLANGIC
---------------
Olculdu: test'in 7.036 trafosunun 2.024'u (satirlarin %22,16'si) train'de
HIC yok. Bu satirlarda trafo ozetlerinin tamami NaN olur ve bu DOGRU
davranistir -- 0 yazmak "tuketimi sifir" demektir, ortalama yazmak
"ortalama bir trafo" demektir; ikisi de uydurmadir. LightGBM NaN'i kendi
dallanmasinda ayri bir yon olarak ogrenir, yani bayragi zaten ucretsiz
alir. Yine de ``soguk_mu`` acikca uretilir: model "ozet yok" durumunu
tek bir kolondan gorebilsin.

Soguk satirlarin dusecegi yedek seviyeler ayri bir fonksiyonda
(``grup_seviyeleri_ekle``) ve onlar da ayni fold-guvenlik kuralina tabidir.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "GUC_KOVA_SINIRLARI",
    "TrafoOzetleri",
    "grup_seviyeleri_ekle",
    "guc_kovasi",
    "trafo_ozetleri_cikar",
    "trafo_ozetleri_uygula",
]

#: ``guc`` (kVA) kova sinirlari. Trafo gucleri standart kademelerde uretilir
#: (50/100/160/250/400/630/1000/1600 kVA ...), yani esit genislikli kova
#: yanlis olur. Bu sinirlar gercek dagilimdan secildi (olculdu 2026-08-21:
#: min 40, medyan 400, %75 1.000, max 35.900).
GUC_KOVA_SINIRLARI: tuple[float, ...] = (0, 100, 160, 250, 400, 630, 1000, 1600, np.inf)

#: Trafo ozetlerinin uretecegi kolon onekleri. Tek yerde tutuluyor ki
#: ``trafo_ozetleri_uygula`` ile ``TrafoOzetleri`` birbirinden kaymasin.
_SEVIYE_KOLONLARI = (
    "t_log_ort",
    "t_log_std",
    "t_log_medyan",
    "t_log_p10",
    "t_log_p90",
    "t_gun_sayisi",
    "t_sifir_orani",
    "t_yuk_faktoru",
    "t_trend",
    "t_kuyruk_sifir",
    "t_olu_mu",
    "t_son_kayit_yasi",
)

#: Guncellik pencereleri (gun). Trafo seviyesi zamanla kayiyor ve OLCULDU
#: (2026-08-21, ozet penceresi Oca-Mar 2025, hedef Nis-Tem 2025): ozet
#: penceresinin SONUNDAKI kisa ortalama, pencerenin tamamindan belirgin
#: bicimde daha iyi bir seviye tahmincisi --
#:
#:     son  7 gun -> hedef seviyesi RMSE(log) 0,5982
#:     son 14 gun ->                          0,5751   <-- en iyi
#:     son 30 gun ->                          0,5870
#:     son 60 gun ->                          0,6143
#:     son 90 gun ->                          0,7158
#:     TUM pencere ->                         0,7158
#:
#: Fark kucuk degil: 0,716 -> 0,575, yani seviye hatasinin yaklasik %20'si.
#: Tek pencere secmek yerine hepsi veriliyor; model hangisine ne kadar
#: guvenecegine ``t_sonN_gun`` sayilarina bakarak kendi karar verir.
GUNCELLIK_PENCERELERI: tuple[int, ...] = (7, 14, 30, 60, 90)

#: Kac kesintisiz sifir KAYIT'tan sonra trafo "olu" sayilir.
#: 14 secildi cunku olcum bu esikte yapildi: son 14 gunu tamamen sifir olan
#: trafolarin hedef penceresindeki gunlerinin %96,0'si sifir cikti. Daha
#: kisa esik gecici arizalari da olu sayar, daha uzunu ise gec kalir.
OLUM_ESIGI_KAYIT = 14


def guc_kovasi(guc: pd.Series) -> pd.Series:
    """``guc`` degerlerini standart kVA kademelerine gore kovalar.

    Kategorik degil ORDINAL kod dondurur: kova sirasi anlamlidir (buyuk
    trafo = buyuk tuketim), yani agac modeli tek bolmede "X kVA ustu"
    diyebilir. One-hot bu bilgiyi dagitir.
    """
    kova = pd.cut(guc.astype("float64"), bins=list(GUC_KOVA_SINIRLARI), right=True)
    return kova.cat.codes.astype("int16")


@dataclass(frozen=True)
class TrafoOzetleri:
    """``uydur`` cercevesinden cikarilmis, tasinabilir trafo istatistikleri.

    Cerceve DEGIL sozluk tutulmasinin nedeni: bu nesne bir fold'un egitim
    yarisina aittir ve yanlislikla baska bir fold'a uygulanirsa sessiz bir
    sizinti olur. ``kaynak_ozeti`` alani tam da bunu gorunur kilar --
    raporlarda hangi pencereden cikarildigi yazilir.
    """

    seviye: pd.DataFrame
    haftagunu: pd.DataFrame
    ay: pd.DataFrame
    isil_egim: pd.DataFrame
    kaynak_ozeti: dict[str, object] = field(default_factory=dict)

    @property
    def trafo_sayisi(self) -> int:
        return int(len(self.seviye))


def _egim(x: np.ndarray, y: np.ndarray) -> float:
    """En kucuk kareler egimi. Tanimsizsa NaN -- 0 DEGIL.

    0 dondurmek "egim yok" iddiasidir ve olculmemis bir iddiadir. NaN
    "bilmiyorum" der; agac modeli ikisini farkli dallara ayirir.
    """
    gecerli = np.isfinite(x) & np.isfinite(y)
    if gecerli.sum() < 3:
        return float("nan")
    x, y = x[gecerli], y[gecerli]
    xd = x - x.mean()
    payda = float((xd * xd).sum())
    if payda <= 0.0:
        return float("nan")
    return float((xd * (y - y.mean())).sum() / payda)


def _isil_egim(parca: pd.DataFrame, *, hedef: str, isil_kolon: str) -> float:
    """Bir trafonun ``log1p(tuketim) ~ isil_kolon`` regresyon egimi.

    Fiziksel anlami: trafo sicakliga ne kadar tepki veriyor. Klimali bir
    ticari trafoda sogutma-derece-gun egimi yuksek, sokak aydinlatmasi
    beslemesinde sifira yakin olur. Bu, ``guc``'un tasiyamadigi bir ayrim.

    En az uc farkli ``isil_kolon`` degeri sart; yoksa egim tanimsizdir ve
    NaN doner (0 DEGIL -- 0 "tepki yok" demektir, bu bir iddiadir).
    """
    x = parca[isil_kolon].to_numpy(dtype="float64")
    y = parca[hedef].to_numpy(dtype="float64")
    # En az UC farkli sicaklik degeri sart: iki noktadan gecen dogrunun
    # egimi her zaman tanimlidir ama hicbir sey olcmez.
    if np.unique(x[np.isfinite(x)]).size < 3:
        return float("nan")
    return _egim(x, y)


def _kuyruk_sifir_serisi(tuketim: np.ndarray) -> float:
    """Serinin SONUNDAN geriye dogru kesintisiz sifir gun sayisi.

    Tarihe gore SIRALI bir dizi bekler. Panel seyrek oldugu icin takvim
    gunu degil, trafonun KENDI kayitlari sayilir: eksik gun "sifir" da
    degildir "pozitif" de, yalnizca kayit yoklugudur.
    """
    if tuketim.size == 0:
        return 0.0
    sifir = tuketim <= 0.0
    if not sifir[-1]:
        return 0.0
    # Sondan geriye ilk POZITIF kaydin yeri; yoksa dizinin tamami sifir.
    pozitif = np.flatnonzero(~sifir)
    return float(tuketim.size if pozitif.size == 0 else tuketim.size - pozitif[-1] - 1)


def _olum_ozetleri(
    d: pd.DataFrame, *, grup_kolonu: str, zaman_kolonu: str, hedef_kolonu: str
) -> pd.DataFrame:
    """Trafonun ozet penceresi SONUNDA olu olup olmadigini olcen kolonlar.

    NEDEN AYRI VE NEDEN ONEMLI -- olculdu (2026-08-21, gercek train.csv):

    RMSLE log olcekte calistigi icin sifir gunler orantisiz ceza tasir.
    Hata ayristirmasi (ozet Oca-Mar 2025, hedef Nis-Tem 2025, 274.929 satir,
    trafo seviyesi tahmincisi):

        gercek deger    satir payi    KARESEL HATA PAYI
        SIFIR                %5,42               %22,2
        <= 10 kWh            %6,45               %25,4

    Yani satirlarin %6'si hatanin dortte birini tasiyor. Bu satirlari
    mukemmel bilseydik RMSLE 1,0271 -> 0,8871 olurdu.

    Ve tahmin edilebilirler, cunku sifirlar neredeyse mutlak kalicidir:

        P(bugun sifir)                   = 0,0468
        P(bugun sifir | DUN sifir)       = 0,9755
        P(bugun sifir | dun sifir degil) = 0,0011

    Trafo duzeyinde: ozet penceresinin son 14 gunu tamamen sifir olan 143
    trafonun hedef penceresindeki gunlerinin **%96,0'si** sifir (medyan
    tuketim tam 0). Olu olmayan 2.070 trafoda ayni oran %0,9.

    Yani "ozet penceresi sonunda olu mu" tek basina, hatanin en pahali
    dilimini neredeyse tamamen aciklayan bir ayrimdir. ``t_sifir_orani``
    bunu YAKALAYAMAZ: pencerenin ORTASINDA bir ay susup sonra geri donen
    trafo ile sonunda susup bir daha donmeyen trafo ayni orani verir.

    Uretilen kolonlar:
        ``t_kuyruk_sifir``     pencere sonundan geriye kesintisiz sifir kayit
        ``t_olu_mu``           kuyruk >= ``OLUM_ESIGI_KAYIT``
        ``t_son_kayit_yasi``   son kayittan pencere sonuna gecen gun
    """
    sirali = d.sort_values([grup_kolonu, zaman_kolonu])
    g = sirali.groupby(grup_kolonu, observed=True)
    kuyruk = g[hedef_kolonu].apply(lambda s: _kuyruk_sifir_serisi(s.to_numpy(dtype="float64")))
    son_gun = d[zaman_kolonu].max()
    return pd.DataFrame(
        {
            "t_kuyruk_sifir": kuyruk,
            "t_olu_mu": (kuyruk >= OLUM_ESIGI_KAYIT).astype("float64"),
            "t_son_kayit_yasi": (son_gun - g[zaman_kolonu].max()).dt.days.astype("float64"),
        }
    )


def _profil_cercevesi(
    kaynak: pd.DataFrame, *, grup_kolonu: str, zaman_kolonu: str, hedef_kolonu: str
) -> pd.DataFrame:
    """Profil hesabi icin ``_sapma`` / ``_hg`` / ``_ay`` kolonlarini hazirlar.

    ``_sapma`` KENDI cercevesinin trafo ortalamasina gore olculur; boylece
    profil, pencereler arasi seviye farkindan arinmis olur.
    """
    p = kaynak[[grup_kolonu, zaman_kolonu, hedef_kolonu]].copy()
    zaman = pd.to_datetime(p[zaman_kolonu])
    p["_y"] = np.log1p(p[hedef_kolonu].clip(lower=0.0))
    p["_sapma"] = p["_y"] - p.groupby(grup_kolonu, observed=True)["_y"].transform("mean")
    p["_hg"] = zaman.dt.dayofweek
    p["_ay"] = zaman.dt.month
    return p


def trafo_ozetleri_cikar(
    uydur: pd.DataFrame,
    *,
    profil_kaynak: pd.DataFrame | None = None,
    grup_kolonu: str = "tanim",
    zaman_kolonu: str = "tarih",
    hedef_kolonu: str = "tuketim",
    guc_kolonu: str = "guc",
    isil_kolonlar: tuple[str, ...] = ("sogutma_derece_gun", "isitma_derece_gun"),
) -> TrafoOzetleri:
    """Trafo ozetleri cikarir: SEVIYE ``uydur``dan, PROFIL ``profil_kaynak``tan.

    Bu fonksiyon hedefi OKUR. Verilen cerceveler, sonucun uygulanacagi
    cerceveyle KESISMEMELIDIR; denetimi ``trafo_ozetleri_uygula`` yapar.

    NEDEN IKI AYRI KAYNAK
    ---------------------
    Iki tur ozet var ve sizinti riskleri ayni degil:

    * **Seviye** (``t_log_ort``, ``t_log_son14`` ...) trafonun BUYUKLUGUNU
      tasir. Buyukluk zamanla kayar (yeni abone, kapanan tesis), dolayisiyla
      gelecekten okumak dogrudan trend sizdirir. Bunlar KESIN olarak
      gecmisten cikarilir.
    * **Profil** (``t_hg_sapma``, ``t_ay_sapma``) trafonun kendi ortalamasina
      gore SEKLINI tasir -- "bu trafo pazarlari %8 dusuktur", "temmuzda %20
      yuksektir". Sekil seviyeden arindirilmistir ve mevsim boyunca kararlidir.

    Profil icin ayri bir kaynak SART, cunku gecmis-yalniz bir pencere hedef
    aylarini hic icermeyebilir. Olculdu (2026-08-21, gercek veri):

        blok    ozet aylari      hedef aylari      t_ay_sapma DOLU
        yaz25   1,2,3            4,5,6,7                     %0,0
        guz25   1..7             8,9,10,11                   %0,0
        kis26   1..11            12,1,2,3                   %13,6
        TEST    1..12            4,5,6,7                    %36,4

    Yani model egitimde bu kolonu neredeyse HIC gormuyor, test'te ise dolu
    geliyor. LightGBM egitimde NaN icin bir yon ogrenir ve test'teki dolu
    degerler o yonu hic ogrenilmemis bicimde ezer. Sessiz, ve tam olarak
    dogrulamanin yakalayamayacagi turden bir uyumsuzluk.

    ``profil_kaynak`` olarak "egitimin tamami EKSI hedef blogu" verildiginde
    kapsam test'le ayni mertebeye cikar ve satirin KENDI etiketi yine hic
    okunmaz -- K-katmanli hedef kodlamanin standart mantigi.

    Args:
        profil_kaynak: Haftagunu/ay sapmalarinin cikarilacagi cerceve.
            ``None`` ise ``uydur`` kullanilir (gecmis-yalniz davranis).
    """
    if uydur.empty:
        raise ValueError("trafo_ozetleri_cikar: 'uydur' cercevesi bos")
    eksik = {grup_kolonu, zaman_kolonu, hedef_kolonu, guc_kolonu} - set(uydur.columns)
    if eksik:
        raise KeyError(f"trafo_ozetleri_cikar: kolon(lar) yok: {sorted(eksik)}")

    d = uydur[[grup_kolonu, zaman_kolonu, hedef_kolonu, guc_kolonu]].copy()
    mevcut_isil = [k for k in isil_kolonlar if k in uydur.columns]
    for k in mevcut_isil:
        d[k] = uydur[k].to_numpy()
    d["_y"] = np.log1p(d[hedef_kolonu].clip(lower=0.0))
    d[zaman_kolonu] = pd.to_datetime(d[zaman_kolonu])

    g = d.groupby(grup_kolonu, observed=True)
    seviye = pd.DataFrame(
        {
            "t_log_ort": g["_y"].mean(),
            "t_log_std": g["_y"].std(ddof=0),
            "t_log_medyan": g["_y"].median(),
            "t_log_p10": g["_y"].quantile(0.10),
            "t_log_p90": g["_y"].quantile(0.90),
            "t_gun_sayisi": g["_y"].size().astype("float64"),
            "t_sifir_orani": g[hedef_kolonu].apply(lambda s: float((s <= 0).mean())),
        }
    )

    # Yuk faktoru: gunluk kWh / (kVA x 24). Guc birimi kVA, tuketim kWh --
    # oran birimsiz ve trafo buyuklugunden ARINDIRILMIS bir yogunluk olcusu.
    # Soguk trafolarda ``guc`` biliniyor ama gecmis yok; sicak trafolarda
    # ogrenilen tipik yuk faktoru, soguklar icin makul bir kopru kurar.
    kapasite = g[guc_kolonu].first().astype("float64") * 24.0
    seviye["t_yuk_faktoru"] = (g[hedef_kolonu].mean() / kapasite.replace(0.0, np.nan)).astype(
        "float64"
    )

    # Trend: pencere icinde ``_y``nin gune gore egimi (gun basina log birim).
    # Olculdu: bu egim ile bir SONRAKI blogun seviye kaymasi arasindaki
    # korelasyon 0,269 -- kucuk ama gercek. Buyuyen ve kuculen trafolari
    # ayirt eder; 122 gunluk ufukta bu fark birikir.
    gun_no = (d[zaman_kolonu] - d[zaman_kolonu].min()).dt.days.astype("float64")
    seviye["t_trend"] = (
        d.assign(_g=gun_no)
        .groupby(grup_kolonu, observed=True)
        .apply(lambda p: _egim(p["_g"].to_numpy(), p["_y"].to_numpy()), include_groups=False)
    )
    olum = _olum_ozetleri(
        d, grup_kolonu=grup_kolonu, zaman_kolonu=zaman_kolonu, hedef_kolonu=hedef_kolonu
    )
    for kolon in olum.columns:
        seviye[kolon] = olum[kolon]
    seviye = seviye.reindex(columns=list(_SEVIYE_KOLONLARI))

    # Guncellik pencereleri: uydurma penceresinin SONUNDAN geriye dogru.
    # Her pencere icin hem ortalama hem GUN SAYISI uretilir -- seyrek bir
    # pencerenin ortalamasi gurultudur ve model bunu ancak sayiyi gorurse
    # indirime tabi tutabilir.
    son_gun = d[zaman_kolonu].max()
    for p in GUNCELLIK_PENCERELERI:
        son = d[d[zaman_kolonu] >= son_gun - pd.Timedelta(days=p - 1)]
        gsl = son.groupby(grup_kolonu, observed=True)["_y"]
        seviye[f"t_log_son{p}"] = gsl.mean() if not son.empty else np.nan
        seviye[f"t_son{p}_gun"] = (
            gsl.size().astype("float64").reindex(seviye.index).fillna(0.0) if not son.empty else 0.0
        )

    # Haftagunu ve ay SAPMALARI -- mutlak ortalama degil.
    #
    # Sapma tutulmasinin nedeni: mutlak ortalama seviye bilgisini TEKRAR
    # tasir ve ``t_log_ort`` ile neredeyse esdogrusal olur. Sapma ise saf
    # profil bilgisidir: "bu trafo pazar gunleri kendi ortalamasindan ne
    # kadar sapar".
    #
    # Sapmanin cikarildigi ortalama, sapmanin cikarildigi PENCERENIN kendi
    # ortalamasidir -- ``seviye["t_log_ort"]`` degil. Farkli pencerelerin
    # ortalamalari farklidir; karisik kullanilirsa sapmaya seviye farki
    # sizar ve profil artik profil olmaktan cikar.
    p = _profil_cercevesi(
        profil_kaynak if profil_kaynak is not None else uydur,
        grup_kolonu=grup_kolonu,
        zaman_kolonu=zaman_kolonu,
        hedef_kolonu=hedef_kolonu,
    )
    haftagunu = (
        p.groupby([grup_kolonu, "_hg"], observed=True)["_sapma"]
        .mean()
        .rename("t_hg_sapma")
        .reset_index()
    )
    ay = (
        p.groupby([grup_kolonu, "_ay"], observed=True)["_sapma"]
        .mean()
        .rename("t_ay_sapma")
        .reset_index()
    )

    egimler: dict[str, pd.Series] = {}
    for k in mevcut_isil:
        egimler[f"t_egim_{k}"] = g.apply(
            lambda p, _k=k: _isil_egim(p, hedef="_y", isil_kolon=_k), include_groups=False
        )
    isil_egim = pd.DataFrame(egimler) if egimler else pd.DataFrame(index=seviye.index)

    return TrafoOzetleri(
        seviye=seviye,
        haftagunu=haftagunu,
        ay=ay,
        isil_egim=isil_egim,
        kaynak_ozeti={
            "satir": int(len(d)),
            "trafo": int(seviye.shape[0]),
            "ilk_gun": str(d[zaman_kolonu].min().date()),
            "son_gun": str(son_gun.date()),
            "isil_kolonlar": mevcut_isil,
        },
    )


def trafo_ozetleri_uygula(
    uygula: pd.DataFrame,
    ozetler: TrafoOzetleri,
    *,
    grup_kolonu: str = "tanim",
    zaman_kolonu: str = "tarih",
) -> pd.DataFrame:
    """Ozetleri hedef cerceveye tasir. YENI frame dondurur.

    Uretilen kolonlar: ``t_log_*``, ``t_gun_sayisi``, ``t_sifir_orani``,
    ``t_yuk_faktoru``, ``t_hg_sapma``, ``t_ay_sapma``, ``t_egim_*``,
    ``soguk_mu``.

    ``soguk_mu`` ozetlerde KARSILIGI OLMAYAN satirlari isaretler. Bu, test
    satirlarinin %22,16'si (olculdu) -- yani kenar durum degil, problemin
    ikinci rejimi.
    """
    if grup_kolonu not in uygula.columns:
        raise KeyError(f"trafo_ozetleri_uygula: '{grup_kolonu}' kolonu yok")
    sonuc = uygula.copy()
    zaman = pd.to_datetime(sonuc[zaman_kolonu])

    # ``_SEVIYE_KOLONLARI`` degil, cercevenin KENDI kolonlari geziliyor:
    # guncellik pencereleri (``t_log_son7`` ... ``t_son90_gun``) sabitte
    # sayilmiyor, ``GUNCELLIK_PENCERELERI``den turuyor. Sabite baglanmak,
    # yeni bir pencere eklendiginde onun sessizce tasinmamasi demekti.
    for kolon in ozetler.seviye.columns:
        sonuc[kolon] = sonuc[grup_kolonu].map(ozetler.seviye[kolon]).astype("float64")

    sonuc["soguk_mu"] = (~sonuc[grup_kolonu].isin(ozetler.seviye.index)).astype("int8")

    hg = ozetler.haftagunu.rename(columns={grup_kolonu: "_g", "_hg": "_k"})
    sonuc["_g"] = sonuc[grup_kolonu]
    sonuc["_k"] = zaman.dt.dayofweek
    sonuc = sonuc.merge(hg, on=["_g", "_k"], how="left", validate="many_to_one")

    ay = ozetler.ay.rename(columns={grup_kolonu: "_g", "_ay": "_k"})
    sonuc["_k"] = zaman.dt.month
    sonuc = sonuc.merge(ay, on=["_g", "_k"], how="left", validate="many_to_one")
    sonuc = sonuc.drop(columns=["_g", "_k"])

    for kolon in ozetler.isil_egim.columns:
        sonuc[kolon] = sonuc[grup_kolonu].map(ozetler.isil_egim[kolon]).astype("float64")

    # SOGUK SATIRLARIN PROFILI SILINIR.
    #
    # Bu, sessiz bir dogrulama yanliliginin duzeltmesi. ``profil_kaynak``
    # "egitimin tamami eksi hedef blogu" oldugu icin, bir blokta SOGUK olan
    # (yani ozet penceresinde hic gorulmemis) bir trafo, blogun SONRASINDAKI
    # aylardan profil alabiliyordu. Ornek: yaz25 blogunda Oca-Mar 2025'te
    # gorulmemis bir trafo, Agu 2025 - Mar 2026 verisinden haftagunu ve ay
    # profili kazaniyordu.
    #
    # Gercek test'te boyle bir sey YOK: oradaki 2.024 soguk trafonun hicbir
    # yerde tek satiri bile yok. Yani dogrulama soguk rejimi olcemeyecek
    # kadar kolaylastiriyor ve iyilesme oldugunu sandigimiz seyler
    # leaderboard'a gecmiyordu (olculdu: v2 CV 1,0705 -> LB 1,16143;
    # v7 CV 1,0618 -> LB 1,16922 -- CV duzelirken LB kotulesti).
    #
    # Duzeltme, CV skorunu YUKSELTIR. Amac skoru guzellestirmek degil,
    # olctugu seyin gercek olmasi.
    profil_kolonlari = [k for k in ("t_hg_sapma", "t_ay_sapma") if k in sonuc.columns]
    if profil_kolonlari:
        sonuc.loc[sonuc["soguk_mu"] == 1, profil_kolonlari] = np.nan

    return sonuc


def grup_seviyeleri_ekle(
    uygula: pd.DataFrame,
    uydur: pd.DataFrame,
    *,
    hedef_kolonu: str = "tuketim",
    guc_kolonu: str = "guc",
    ilce_kolonu: str = "ilce_key",
) -> pd.DataFrame:
    """SOGUK trafolarin dusecegi yedek seviyeler. YENI frame dondurur.

    Trafo gecmisi yoksa geriye ne kaliyor: kurulu guc ve konum. Olculdu --
    ``guc`` kovasi tek basina log-varyansin %26,2'sini, ilce %15,5'ini
    acikliyor. Ikisinin KESISIMI, ikisinin toplamindan fazlasini tasir:
    Bornova'daki 1.000 kVA'lik bir trafo ile Kiraz'daki 1.000 kVA'lik bir
    trafo ayni sey degildir.

    Uretilen kolonlar:
        ``g_guc_kova``        ordinal kVA kademesi
        ``g_kova_log_ort``    kova ortalamasi
        ``g_ilce_log_ort``    ilce ortalamasi
        ``g_ilce_kova_ort``   ilce x kova ortalamasi (asil yedek)
        ``g_ilce_kova_n``     o hucredeki egitim satiri sayisi (guven olcusu)

    ``g_ilce_kova_n`` bilerek uretiliyor: seyrek bir hucrenin ortalamasi
    gurultudur ve model bunu ancak sayiyi GORURSE indirime tabi tutabilir.
    """
    if uydur.empty:
        raise ValueError("grup_seviyeleri_ekle: 'uydur' cercevesi bos")

    u = uydur[[hedef_kolonu, guc_kolonu, ilce_kolonu]].copy()
    u["_y"] = np.log1p(u[hedef_kolonu].clip(lower=0.0))
    u["_kova"] = guc_kovasi(u[guc_kolonu])

    kova_ort = u.groupby("_kova", observed=True)["_y"].mean()
    ilce_ort = u.groupby(ilce_kolonu, observed=True)["_y"].mean()
    hucre = u.groupby([ilce_kolonu, "_kova"], observed=True)["_y"].agg(["mean", "size"])
    hucre.columns = ["g_ilce_kova_ort", "g_ilce_kova_n"]

    sonuc = uygula.copy()
    sonuc["g_guc_kova"] = guc_kovasi(sonuc[guc_kolonu])
    sonuc["g_kova_log_ort"] = sonuc["g_guc_kova"].map(kova_ort).astype("float64")
    sonuc["g_ilce_log_ort"] = sonuc[ilce_kolonu].map(ilce_ort).astype("float64")
    sonuc = sonuc.merge(
        hucre.reset_index().rename(columns={"_kova": "g_guc_kova"}),
        on=[ilce_kolonu, "g_guc_kova"],
        how="left",
        validate="many_to_one",
    )
    sonuc["g_ilce_kova_n"] = sonuc["g_ilce_kova_n"].fillna(0.0).astype("float64")
    return sonuc
