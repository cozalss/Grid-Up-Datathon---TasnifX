"""SINIR AGI -- GBM harmaninin 4. uyesi (SIFIR-AYRIK, IKI REJIMLI MLP).

NEDEN VAR
---------
Uretimdeki uc aile de AGAC tabanli: CatBoost, XGBoost, LightGBM. Ayni
tumevarim onyargisini paylastiklari icin ayni hatalari yapiyorlar -- yaz25
blogunda OLCULDU (``scripts/teshis_cesitlilik.py``, tohum 1000):

    aileler arasi hata korelasyonu   SICAK 0,914   SOGUK 0,913
    cesitliligin payi (Krogh-V.)     SICAK %5,58   SOGUK %6,06

Krogh & Vedelsby (NeurIPS 1994) ayrisimi log uzayinda ORTALAMA alan
topluluklar icin bir OZDESLIK:  E_topluluk = E_ortalama_uye - A_cesitlilik.
Yani bir uye TEK BASINA kotu olsa bile, hatasi digerleriyle ORTUSMUYORSA
harmani duzeltir. Olculen esikler (4. uye, ESIT agirlikli tabloya gore):

    uye esit kaliteliyse   korelasyon < 0,94
    uye %20 kotuyse        korelasyon < 0,72
    uye 2 kat kotuyse      korelasyon < 0,22

Bu modulun hedefi DUSUK KORELASYON; yuksek tekil kalite DEGIL. Prototip tam
bunu gosterdi: (256,128) agi TEKIL olarak daha iyi (0,95612 vs 0,97227) ama
korelasyonu yuksek (0,7876 vs 0,7607) ve HARMANDA DAHA KOTU (0,80744 vs
0,80533). Secim kriteri korelasyondur -- agi BUYUTMEYIN.

OLCULEN CEKIRDEK (yaz25, SICAK rejim, tek tohum, ek kokensiz)
-------------------------------------------------------------
    sinir agi tek basina RMSLE  0,97227   (harmandan %19,6 kotu)
    GBM'lerle korelasyon        0,7607
    3'lu harman                 0,81308
    4'lu harman (w = 0,15)      0,80533   -> -0,00775

UC OLCULMUS TASARIM KARARI (ucu de calismanin ON SARTI)
-------------------------------------------------------
1. KIMLIK KOLONLARI ATILIR (``tanim_num``, ``tanim_on2..on5``,
   ``tanim_uzunluk``). Tutulunca soguk RMSLE 1,92857; atilinca 1,69297.
   Mekanizma: ag ilce x kod x guc kovasindan trafo kimligini yeniden insa
   edip OLGUN tuketim seviyesini ezberliyor. Testte 2.035 YENI trafo var,
   yani ezber orada BOSA gidiyor.
2. SIFIR SATIRLARI REGRESYON EGITIMINDEN CIKARILIR. Sifir dahil 2,36535 ->
   sifir ayri 1,92857. Hedef bimodal: sifirda ofsetli hedef ort -5,85,
   sifir-disinda +0,93 -- 6,8 birimlik ucurum MSE gradyanini esir aliyor.
   Sifir kutlesi tahminde KOSULLU ORTALAMA olarak geri eklenir:
       E[log1p(Y)] = P(Y=0)*0 + P(Y>0)*E[log1p(Y) | Y>0]
   Bu formule uc bagimsiz tasarim ayri ayri ulasti. "p yuksekse tahmini
   sifira yapistir" gibi bir son-islem KESINLIKLE YOKTUR: RMSLE log uzayinda
   kareli hata oldugu icin kosullu ortalama ZATEN optimaldir, yapistirmak
   beklenen hatayi KOTULESTIRIR.
3. REJIM BASINA AYRI RECETE. Sicakta P(sifir) ogrenilebilir (dogrulama
   AUC 0,9817); sogukta ogrenilemez (AUC 0,5934) -- ogrenilmis p denendi,
   ortalama 0,1118 cikti (gercek taban 0,0378) ve RMSLE 1,93 -> 2,50 COKTU.
   Soguk kolda p SABIT taban oranidir.

NEDEN sklearn, NEDEN torch DEGIL
--------------------------------
Kutuphane kontrolu ``motor_sec()`` ile calisma aninda yapilir. Bu makinede
**torch KURULU DEGIL**, dolayisiyla sklearn'e dusuyoruz. torch kurulu OLSA
BILE bu modul sklearn kullanir ve bu bilincli bir karardir:

  * ``pyproject.toml`` torch'u ``neural`` extra'sina koymus; CI onu KURMUYOR
    ve offline Kaggle paketine GIRMIYOR. torch tabanli bir uye
    GONDERILEMEZ.
  * Gonderim hedge'i iki makinede uretiliyor; torch'un CPU yolu bit
    duzeyinde makineye baglidir ve olculmus tasinabilirligi riske atar.
  * sklearn>=1.3 CEKIRDEK bagimlilik, yani wheel'e kendiliginden girer.

Ek fayda: sklearn MLP'de BatchNorm YOKTUR. Egitim soguk payi %11,74, test
%22,16 -- BatchNorm hareketli ortalamalari servis aninda kayardi. Sorun
YAPISAL OLARAK dogmuyor.

SIZINTI DISIPLINI
-----------------
On-isleme sizintinin en sik girdigi yerdir. Su dort sey YALNIZCA ``fit``e
gelen egitim satirlarindan ogrenilir ve ``predict``te AYNEN uygulanir:
medyan doldurma degerleri, kuantil sinirlari, one-hot sozlugu, hedef
olcegi. Hicbiri hedef cerceveyi gormez; ``fit_transform`` yalnizca egitim
tarafinda cagrilir.

CALISTIRMA (bagimsiz olcum tezgahi)::

    python scripts/sinir_agi.py                 # yaz25, SICAK, tam olcum
    python scripts/sinir_agi.py --sadece-ag     # GBM'leri egitme, hizli bak
    python scripts/sinir_agi.py --hizli         # az epok, dumanaltı kosu

URETIME BAGLANMA: ``egit_tahmin`` fonksiyonu ``deney_ileri.egit_tahmin`` ile
BIREBIR ayni imzayi ve ayni donus sozlesmesini (LOG UZAYI) saglar; model
nesnesi ``sinir_agi_kur`` ile kurulur ve ``fit(DataFrame, ndarray)`` /
``predict(DataFrame) -> ndarray`` sozlesmesine uyar, yani
``aile_tahmini``nin ``else`` dalindan degisiklik gerektirmeden gecer.
Bu dosya HICBIR uretim dosyasini degistirmez.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    QuantileTransformer,
    StandardScaler,
)

KOK = Path(__file__).resolve().parents[1]

#: Bu modulun tanittigi aile adi. ``aile_modeli`` dallarinda kullanilir.
AILE_ADI = "sinir_agi"

# --------------------------------------------------------------- sabitler

#: AGDAN ATILAN KOLONLAR (12). Her biri OLCULMUS bir gerekceyle.
ATILAN: tuple[str, ...] = (
    # Kimlik tasiyicilari: sogukta tutulunca 1,92857 / atilinca 1,69297.
    # Egitim-ici RMSE 0,774 (egitim std 1,159) = sert asiri uydurma.
    "tanim_num",
    "tanim_on2",
    "tanim_on3",
    "tanim_on4",
    "tanim_on5",
    "tanim_uzunluk",
    # Aralik disi: egitimde {90, 212, 334}, testte TEK deger {455}. Testin
    # %100'u egitim araliginin disinda; ag dogrusal ekstrapole eder ve TUM
    # test tahminlerine ayni yonde sabit bir kayma bindirir.
    "ozet_pencere_gun",
    # DOGRULAMADA KOR kolonlar (yaz25 bos orani / test bos orani):
    "t_ay_sapma",  # 1,0000 / 0,6360
    "t_gy_log_ort",  # 1,0000 / 0,5903
    "t_gy_sifir_orani",  # 1,0000 / 0,5903
    "t_gy_gun",  # 1,0000 / 0,5903
    "t_egim_cdd22",  # 1,0000 / 0,5494 (sicak satirlarda yaz25 %100 bos)
)

#: One-hot'a girecek kolonlar. ``kategorik_kodla`` bunlari pandas
#: Categorical yapiyor; ``.astype(str)`` YAPILMAZSA sklearn
#: "Cannot cast str dtype to float64" verir.
KATEGORIK: tuple[str, ...] = ("il_key", "bolge", "ilce_key")

#: Zaten 0/1 olan kolonlar. OLCEKLENMEZ, kuantile SOKULMAZ:
#: ``ilk_gun_mu`` carpikligi +18,30 (satirlarin %0,3'u 1) -- standartlastirmak
#: 18 sigmalik uc deger uretip ilk katmani doyurur.
IKILI: tuple[str, ...] = (
    "soguk_mu",
    "asiri_sicak",
    "asiri_soguk",
    "ilk_gun_mu",
    "tk_hafta_sonu",
)

#: A3 augmentasyonunda NaN'a cekilecek "yakin gecmis" ailesi. Olculen
#: asimetri (sicak satirlarda bos orani): yaz25 %1,48 / guz25 %1,32 /
#: kis26 %1,22 ama TEST %7,77 -- yani 5-6 KAT. Uc blokta da ayni yonde,
#: dolayisiyla yaz25'te de OLCULEBILIR.
GUNCELLIK: tuple[str, ...] = (
    "t_log_son7",
    "t_log_son14",
    "t_log_son30",
    "t_log_son60",
    "t_log_son90",
    "t_son7_gun",
    "t_son14_gun",
    "t_son30_gun",
    "t_son60_gun",
    "t_son90_gun",
)

#: A2 ablasyonu -- es-dogrusal ikizin FAZLA olani (parantezde |r|).
ES_DOGRUSAL_IKIZ: tuple[str, ...] = (
    "t_log_son14",  # t_log_son7 ile +0,9968
    "t_log_son60",  # t_log_son90 ile +0,9958
    "hissedilen_max",  # sicaklik_max ile +0,9923
    "cdd24_ort14",  # cdd22_ort14 ile +0,9911
    "gunes_ghi_gunluk",  # gun_uzunlugu_saat ile +0,9948
    "trafo_basina_hat",  # osm_dagitim_hat_km ile +0,9887
)

#: A2 ablasyonu -- ILCE duzeyinde sabit kolonlar (ilce-ici essiz deger
#: sayisi 1,00 olculdu). ``ilce_key`` one-hot'u varken tam fazlalik.
ILCE_SABIT_ONEK: tuple[str, ...] = ("osm_",)
ILCE_SABIT_AD: tuple[str, ...] = (
    "agac_orani",
    "calilik_orani",
    "otlak_orani",
    "tarim_orani",
    "yerlesim_orani",
    "ciplak_orani",
    "su_orani",
    "bitki_ortusu_orani",
)

#: Ek olarak uretilen ve IKILI dala (olceklenmeden) giren skaler.
EKSIK_ORANI_KOLONU = "_eksik_orani"

#: Sicak rejim esigi: ``soguk_mu`` ortalamasi bunun uzerindeyse SOGUK.
#: Soguk uzman maske 1,00 ile cagrilir (ortalama 1,00); sicak uzman
#: maske 0,15 ile (ortalama ~0,25). Esik genis ama REJIM_AYARLARI'ndaki
#: maske degerleri degisirse sessizce yanlis mimari secer -- tek savunma
#: ``fit``in bastigi rejim satiridir, o yuzden ATLANMAZ.
SOGUK_REJIM_ESIGI = 0.95

#: alpha <-> batch_size GIZLI BAGIMLILIGI: sklearn L2 cezasini PARTI
#: boyutuna boler (``_backprop``: ceza once ``alpha * W``, sonra parti
#: buyuklugune bolunur). batch 200 -> 2048 gecisinde AYNI alpha 10 KAT
#: zayiflar. sklearn varsayilani 1e-4 REFERANS ALINMAZ; asagidaki alpha
#: degerleri YALNIZCA bu batch ile gecerlidir ve birlikte degistirilir.
PARTI = 2048
OGRENME_HIZI = 2e-3
DOGRULAMA_PAYI = 0.08

#: SICAK regresyon bası. (128,64) SECILDI -- bkz. dosya basi.
SICAK_KATMAN = (128, 64)
SICAK_ALPHA = 1e-3
SICAK_EPOK = 100
SICAK_EPOK_HIZLI = 20

#: SICAK sifir siniflandiricisi (dogrulama AUC 0,9817).
SINIF_KATMAN = (48, 24)
SINIF_ALPHA = 1e-3
SINIF_EPOK = 30
SINIF_EPOK_HIZLI = 8

#: SOGUK regresyon bası. Boyut taramasi olculdu: (96,48) 1,87 |
#: (64,32) a=1e-1 1,68 | (32,16) a=1e-2 1,693 | (48,) 1,78 |
#: Ridge(alpha=30) 1,720. Siniflandirici YOK.
SOGUK_KATMAN = (32, 16)
SOGUK_ALPHA = 1e-2
SOGUK_EPOK = 120
SOGUK_EPOK_HIZLI = 25

#: Sicak uzman ek kokenlerle 2.855.584 satir gorur; prototip 763.808'de
#: olculdu. Bu esigin uzerinde alt-orneklenir.
ALT_ORNEK_ESIGI = 1_200_000


# ------------------------------------------------------------------ motor


def motor_sec() -> tuple[str, str]:
    """Kullanilacak motoru ve GEREKCESINI dondurur.

    torch yoksa sklearn'e duseriz. torch VARSA da sklearn kullaniriz:
    ``neural`` extra'si CI'da kurulmuyor, offline Kaggle paketine girmiyor
    ve torch'un CPU yolu bit duzeyinde makineye bagli oldugu icin iki
    makinede uretilen hedge dosyalarinin ayniligini bozar.
    """
    if importlib.util.find_spec("torch") is None:
        return "sklearn", "torch KURULU DEGIL -> sklearn"
    return "sklearn", "torch var ama KULLANILMIYOR (offline paket + tasinabilirlik)"


# ------------------------------------------------- on-isleme yardimcilari
# NOT: Bu fonksiyonlar MODUL DUZEYINDE tanimli, lambda DEGIL -- aksi halde
# tahminci pickle edilemez (tohum torbalama ve tasima paketi bunu ister).


def _f32_matris(cerceve: pd.DataFrame) -> np.ndarray:
    """Sayisal kolonlari NaN-guvenli float32 matrise cevirir.

    ``to_numpy(dtype=..., na_value=...)`` sart: ``tanim_*`` disindaki bazi
    kolonlar da pandas nullable (``Int64``) gelebilir ve duz
    ``.astype("float32")`` bunlarda ``pd.NA`` uzerinde patlar; duz
    ``.to_numpy()`` ise sessizce ``dtype=object`` dondurur.

    float32 OPSIYONEL DEGIL: 2,86M x 181 float64 = 4,1 GB, float32 = 2,07 GB.
    """
    if cerceve.shape[1] == 0:
        return np.empty((len(cerceve), 0), dtype="float32")
    return np.column_stack(
        [cerceve[k].to_numpy(dtype="float32", na_value=np.nan) for k in cerceve.columns]
    )


def _metin_matris(cerceve: pd.DataFrame) -> np.ndarray:
    """Kategorikleri metne cevirir -- ``.lower()``/``.upper()`` UYGULANMAZ.

    Turkce i/I tuzagi: ``ilce_key`` seviyeleri zaten ASCII'ye katlanmis
    ('karabaglar'), yeniden katlamak bozar. ``bolge`` seviyelerinde Turkce
    karakter var ('GUNEY BOLGE') ve 'YOK' GERCEK bir seviyedir (egitimde
    276.445 satir) -- NaN'a CEVIRILMEZ.
    """
    if cerceve.shape[1] == 0:
        return np.empty((len(cerceve), 0), dtype=object)
    return np.column_stack([cerceve[k].astype(str).to_numpy(dtype=object) for k in cerceve.columns])


def _log_guc(cerceve: pd.DataFrame) -> np.ndarray:
    """``log1p(guc)`` -- kapasite ofseti. ``guc`` hicbir satirda bos degil."""
    return np.log1p(cerceve["guc"].to_numpy(dtype="float64", na_value=np.nan))


# -------------------------------------------------------------- tahminci


class SinirAgi(BaseEstimator, RegressorMixin):
    """Iki basli, rejim basina ayri receteli MLP tahmincisi.

    SOZLESME (``tuketim_model.aile_tahmini``nin ``else`` dalindan):

      * ``fit(X: DataFrame, y: ndarray)`` -- ``X`` MASKELENMIS gelir; bu
        sinif ``soguk_maskele``yi ASLA kendisi cagirmaz. (Iki modulde imza
        TERS: ``tm.soguk_maskele(cerceve, kolonlar, tohum, oran)`` ama
        ``deney.soguk_maskele(cerceve, kolonlar, oran, tohum)``. Cagirmayarak
        tuzaga girmiyoruz.)
      * ``y`` = ``ofsetli_hedef`` = ``log1p(tuketim.clip(0)) - log1p(guc)``.
      * ``predict(X) -> ndarray`` OFSETLI uzayda doner; ``log1p(guc)``i
        cagiran ekler.
      * ``X``te NaN VAR ve ``KATEGORIK`` kolonlar pandas Categorical'dir;
        ikisi de bu sinifin ICINDE cozulur, uretim dosyalarina dokunulmaz.

    Ablasyon anahtarlarinin HEPSI varsayilan olarak KAPALIDIR: olculmus
    cekirdek dokunulmazdir, graftlar uzerine ablasyon olarak biner.
    """

    def __init__(
        self,
        *,
        tohum: int = 0,
        hizli: bool = False,
        rejim: str | None = None,
        ofset: bool = True,
        alt_ornek: float = 0.40,
        aug_bayat: float = 0.0,
        eksik_orani_kolonu: bool = False,
        budama: bool = False,
        ayri_gosterge: bool = False,
        n_ag: int = 5,
        sessiz: bool = False,
    ) -> None:
        # sklearn sozlesmesi: __init__ YALNIZCA atar, hicbir hesap yapmaz.
        self.tohum = tohum
        self.hizli = hizli
        self.rejim = rejim
        self.ofset = ofset
        self.alt_ornek = alt_ornek
        self.aug_bayat = aug_bayat
        self.eksik_orani_kolonu = eksik_orani_kolonu
        self.budama = budama
        self.ayri_gosterge = ayri_gosterge
        self.n_ag = n_ag
        self.sessiz = sessiz

    # ------------------------------------------------------------ yardim

    def _yaz(self, mesaj: str) -> None:
        if not self.sessiz:
            print(mesaj, flush=True)

    def _rejim_sec(self, X: pd.DataFrame) -> bool:
        if self.rejim is not None:
            if self.rejim not in ("sicak", "soguk"):
                raise ValueError(f"bilinmeyen rejim: {self.rejim}")
            return self.rejim == "soguk"
        pay = float(X["soguk_mu"].to_numpy(dtype="float64", na_value=0.0).mean())
        return pay > SOGUK_REJIM_ESIGI

    def _kolonlari_sinifla(self, X: pd.DataFrame) -> None:
        """Kolon siniflarini ve olu kolon budamasini EGITIMDEN ogrenir."""
        atilacak = set(ATILAN)
        if self.budama:
            atilacak |= set(ES_DOGRUSAL_IKIZ) | set(ILCE_SABIT_AD)
            atilacak |= {k for k in X.columns if k.startswith(ILCE_SABIT_ONEK)}
        kalan = [k for k in X.columns if k not in atilacak]

        self.kategorik_ = [k for k in kalan if k in KATEGORIK]
        self.ikili_ = [k for k in kalan if k in IKILI]
        aday = [k for k in kalan if k not in self.kategorik_ and k not in self.ikili_]

        # OLU KOLON BUDAMASI: tam bos ya da sabit. SOGUK rejimde (maske 1,00)
        # bu, 33 t_* kolonunun tamamini kendiliginden dusurur -- ve medyani
        # NaN olan bir kolonun SimpleImputer'i patlatmasini onler.
        tutulan: list[str] = []
        for k in aday:
            sutun = X[k].to_numpy(dtype="float64", na_value=np.nan)
            if np.isnan(sutun).all():
                continue
            if float(np.nanstd(sutun)) <= 1e-12:
                continue
            tutulan.append(k)
        self.sayisal_ = tutulan

    def _cerceve_hazirla(self, X: pd.DataFrame) -> pd.DataFrame:
        """A1 skalerini ekler. fit ve predict'te BIREBIR ayni islem."""
        if not self.eksik_orani_kolonu:
            return X
        hazir = X.copy()
        if self.sayisal_:
            ham = _f32_matris(X[self.sayisal_])
            hazir[EKSIK_ORANI_KOLONU] = np.isnan(ham).mean(axis=1).astype("float32")
        else:
            hazir[EKSIK_ORANI_KOLONU] = np.float32(0.0)
        return hazir

    def _on_isleyici(self) -> ColumnTransformer:
        """Sayisal / kategorik / ikili dallar. SIRA: once DOLDUR, sonra KUANTIL.

        Bu sira OLCULEN siradir. Ters sira matematiksel olarak neredeyse
        ozdestir (QT-normal ciktisinda medyan tam 0'dir) ama olcumu bozmamak
        icin DEGISTIRILMEZ.

        QuantileTransformer dort tuzagi TEK hamlede kapatir:
          (a) 10 buyukluk mertebelik olcek yayilimi (std 0,0088 .. 2,945e8),
          (b) |carpiklik| > 2 olan 41 kolon,
          (c) ARALIK DISI DOYURMA -- ``yas`` %32,57, ``t_gun_sayisi`` %29,82
              testte egitim araligini asiyor; QT egitim kuantillerini
              kullandigi icin ust kuyruk OTOMATIK kelepcelenir (agacin son
              yaprakta yaptigi is). Bu yuzden ayrica log1p + p99 kirpma +
              "_asti" bayragi GEREKSIZDIR ve KONMAZ -- iki mekanizma ust uste
              bilgiyi iki kez kirpar.
          (d) StandardScaler burada +2,24 sigma otede sabit kayma bindirirdi.

        SIFIRLA DOLDURMA YASAK: ham uzayda ``t_log_ort = 0`` "tuketimi sifir
        trafo" demektir. HAM ORTALAMA da yasak: butun soguk trafolari tek
        noktaya yigar. Medyan + acik eksiklik gostergesi kullaniyoruz.
        """
        sayisal_dal = Pipeline(
            [
                ("f32", FunctionTransformer(_f32_matris, feature_names_out="one-to-one")),
                (
                    "doldur",
                    SimpleImputer(strategy="median", add_indicator=not self.ayri_gosterge),
                ),
                (
                    "kuantil",
                    QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=1000,
                        subsample=200_000,
                        random_state=self.tohum,
                        copy=False,
                    ),
                ),
            ]
        )
        kategorik_dal = Pipeline(
            [
                ("metin", FunctionTransformer(_metin_matris, feature_names_out="one-to-one")),
                (
                    "kod",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=30,
                        sparse_output=False,
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        # Ikili dal OLCEKLENMEZ. SimpleImputer(constant) burada sayisal olarak
        # passthrough'dur (bu kolonlarda NaN olculmedi), yalnizca servis
        # anindaki bir NaN'in agi patlatmasina karsi kalkandir.
        ikili_dal = Pipeline(
            [
                ("f32", FunctionTransformer(_f32_matris, feature_names_out="one-to-one")),
                ("doldur", SimpleImputer(strategy="constant", fill_value=0.0)),
            ]
        )
        dallar: list[tuple[str, Any, list[str]]] = [
            ("sayisal", sayisal_dal, self.sayisal_),
            ("kategorik", kategorik_dal, self.kategorik_),
            ("ikili", ikili_dal, self.ikili_ + self.ek_ikili_),
        ]
        if self.ayri_gosterge:
            # A5: gostergeler QT'den GECMEZ, ham 0/1 kalir.
            gosterge_dal = Pipeline(
                [
                    ("f32", FunctionTransformer(_f32_matris, feature_names_out="one-to-one")),
                    ("gosterge", MissingIndicator(features="missing-only")),
                ]
            )
            dallar.append(("gosterge", gosterge_dal, self.sayisal_))
        return ColumnTransformer(dallar, sparse_threshold=0, n_jobs=None)

    def _augmentasyon(self, X: pd.DataFrame) -> pd.DataFrame:
        """A3 -- "bayat gecmis" augmentasyonu (YALNIZCA SICAK).

        ``soguk_maskele``nin yumusak ve HEDEFLI ikizi: yalnizca guncellik
        ailesi NaN'a cekilir, ``soguk_mu`` 0 KALIR ve diger ``t_*``
        kolonlarina DOKUNULMAZ. Maskeleme TRAFO duzeyindedir, satir
        duzeyinde degil -- bir trafonun bazi gunlerinde gecmisi olup
        bazilarinda olmamasi diye bir sey yoktur.
        """
        sutunlar = [k for k in GUNCELLIK if k in X.columns]
        if not sutunlar or "tanim_num" not in X.columns:
            self._yaz("[sinir_agi] A3 atlandi: guncellik/kimlik kolonu yok")
            return X
        # T9: nullable Int64 -> once float64, yoksa np.unique/np.isin sessizce
        # yanlis calisir (dtype=object dizi).
        kimlik = X["tanim_num"].to_numpy(dtype="float64", na_value=np.nan)
        trafolar = np.unique(kimlik[~np.isnan(kimlik)])
        adet = int(len(trafolar) * float(self.aug_bayat))
        if adet <= 0:
            return X
        rng = np.random.default_rng(self.tohum + 31337)
        secilen = rng.choice(trafolar, size=adet, replace=False)
        maske = np.isin(kimlik, secilen)
        sonuc = X.copy()
        sonuc.loc[maske, sutunlar] = np.nan
        self._yaz(f"[sinir_agi] A3 bayat gecmis: {adet:,} trafo, {int(maske.sum()):,} satir")
        return sonuc

    # --------------------------------------------------------------- fit

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> SinirAgi:
        """Ofsetli hedefte egitir. ``X`` maskelenmis, ``y`` ofsetli gelir."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("SinirAgi bir pandas.DataFrame bekler (kolon adlari lazim)")
        if "guc" not in X.columns:
            raise KeyError("'guc' kolonu yok -- kapasite ofseti hesaplanamaz")
        t0 = time.time()
        y = np.asarray(y, dtype="float64")

        # T1: deney_ileri.egit_tahmin non-cat aileler icin sample_weight
        # geciriyor ve MLPRegressor bunu KABUL ETMEZ. Sessizce yok saymak en
        # kotusudur; agirlikla orantili yeniden ornekleme yapiyoruz.
        if sample_weight is not None:
            rng = np.random.default_rng(self.tohum)
            w = np.asarray(sample_weight, dtype="float64")
            secim = rng.choice(len(X), size=len(X), replace=True, p=w / w.sum())
            X, y = X.iloc[secim], y[secim]
            self._yaz("[sinir_agi] sample_weight -> agirlikli yeniden ornekleme")

        soguk = self._rejim_sec(X)

        # Alt-ornekleme YALNIZCA sicakta ve buyuk cercevede. Iki kazanc:
        # (a) bellek yariya iner, (b) her uretim tohumu FARKLI bir %40 gorur,
        # yani AILE ICI cesitlilik bedava gelir ve torbalama egrisi diklesir.
        if not soguk and len(X) > ALT_ORNEK_ESIGI and 0.0 < self.alt_ornek < 1.0:
            rng = np.random.default_rng(self.tohum)
            secim = rng.choice(len(X), size=int(len(X) * self.alt_ornek), replace=False)
            X, y = X.iloc[secim], y[secim]

        if not soguk and self.aug_bayat > 0.0:
            X = self._augmentasyon(X)

        self._kolonlari_sinifla(X)
        self.ek_ikili_ = [EKSIK_ORANI_KOLONU] if self.eksik_orani_kolonu else []
        hazir = self._cerceve_hazirla(X)

        self.on_isleyici_ = self._on_isleyici()
        # SIZINTI SINIRI: fit_transform YALNIZCA burada, egitim satirlarinda.
        Xt = self.on_isleyici_.fit_transform(hazir).astype(np.float32, copy=False)
        self.genislik_ = int(Xt.shape[1])
        self.soguk_ = bool(soguk)

        etiket = "SOGUK" if soguk else "SICAK"
        self._yaz(
            f"[sinir_agi] rejim={etiket}  n={len(X):,}  girdi={self.genislik_}"
            f"  tohum={self.tohum}  motor={motor_sec()[0]}"
        )

        log_guc = _log_guc(X) if self.ofset else np.zeros(len(X), dtype="float64")
        # Sifir etiketi ek argumana gerek kalmadan turetilir: ofsetli hedefe
        # log1p(guc) geri eklenince log1p(tuketim) cikar ve tuketim <= 0 ile
        # ozdestir. Egitimde satirlarin %4,2156'si tam sifir.
        z = (y + log_guc) <= 1e-12
        self.p_taban_ = float(z.mean())

        # HEDEF STANDARTLASTIRMA -- ON SART, iyilestirme degil. Olculdu:
        # standartlastirilmadan max_iter 60 ve 80'de n_iter_ HER SEFERINDE
        # tavana dayandi ve RMSLE 2,32-2,94 cikti (sabit-tahmin tavani 1,797).
        self.hedef_olcek_ = StandardScaler()
        regresyon_satiri = ~z
        if int(regresyon_satiri.sum()) < 100:
            raise ValueError("sifir-disi egitim satiri 100'den az -- regresyon egitilemez")
        y_reg = self.hedef_olcek_.fit_transform(y[regresyon_satiri].reshape(-1, 1)).ravel()

        epok = self._epok(soguk)
        # IC TORBALAMA -- OLCULMUS ZORUNLULUK, suslu degil.
        # Uc tohumla olculdu (2026-08-23, yaz25/SICAK): tek ag kararsiz --
        # tekil RMSLE 0,83763 / 0,86716 / 1,10609. Kotu tohum (1001) 4'lu
        # harmani w=1,4'te +0,01861 BOZDU, iyi tohumlar -0,01675 ve -0,02516
        # KAZANDIRDI. Eslenik ortalama -0,0078 ama SH 0,0133, t=-0,59 --
        # yani sinyal varyansin altinda kaldi. MLP'nin kararsizligi rastgele
        # baslangic + erken durdurma kaynakli ve standart carasi ORTALAMA
        # ALMAKTIR: k bagimsiz agin ortalamasinin hata varyansi ~1/k'ya iner,
        # yanlilik degismez. Torbalama BURADA yapilir (uretimin 7 tohumuna
        # birakilmaz), cunku uretim tohumu TUM hattı degistirir; burada
        # yalnizca agin baslangicini degistirip digerlerini sabit tutuyoruz.
        self.reg_listesi_ = []
        for _i in range(max(1, int(self.n_ag))):
            _reg = MLPRegressor(
                hidden_layer_sizes=SOGUK_KATMAN if soguk else SICAK_KATMAN,
                activation="relu",
                solver="adam",
                alpha=SOGUK_ALPHA if soguk else SICAK_ALPHA,
                batch_size=PARTI,
                learning_rate_init=OGRENME_HIZI,
                max_iter=epok,
                early_stopping=True,
                n_iter_no_change=10,
                validation_fraction=DOGRULAMA_PAYI,
                tol=1e-5,
                shuffle=True,
                random_state=self.tohum * 1000 + _i,
            )
            _reg.fit(Xt[regresyon_satiri], y_reg)
            self.reg_listesi_.append(_reg)
        self.reg_ = self.reg_listesi_[0]  # geriye donuk uyumluluk

        # SICAK: P(sifir) OGRENILIR (dogrulama AUC 0,9817).
        # SOGUK: OGRENILMEZ -- AUC 0,5934 ve denendiginde p ortalamasi 0,1118
        # cikti (gercek taban 0,0378), RMSLE 1,93 -> 2,50 COKTU. Sabit taban.
        self.sinif_ = None
        if not soguk and 0.0 < self.p_taban_ < 1.0 and int(z.sum()) >= 100:
            self.sinif_ = MLPClassifier(
                hidden_layer_sizes=SINIF_KATMAN,
                activation="relu",
                solver="adam",
                alpha=SINIF_ALPHA,
                batch_size=PARTI,
                learning_rate_init=OGRENME_HIZI,
                max_iter=SINIF_EPOK_HIZLI if self.hizli else SINIF_EPOK,
                early_stopping=True,
                n_iter_no_change=6,
                validation_fraction=DOGRULAMA_PAYI,
                shuffle=True,
                random_state=self.tohum,
            )
            self.sinif_.fit(Xt, z)

        self.sure_ = time.time() - t0
        self._yaz(
            f"[sinir_agi] egitildi {self.sure_:.0f} sn  epok reg={self.reg_.n_iter_}"
            f"  sifir_pay={self.p_taban_:.4f}"
            f"  p={'ogrenilmis' if self.sinif_ is not None else 'sabit'}"
        )
        return self

    def _epok(self, soguk: bool) -> int:
        """``hizli`` YALNIZCA epok sayisini duşurur.

        Ogrenme hizina, parti boyutuna, mimariye, alpha'ya ASLA dokunmaz.
        Deponun acı dersi (``model_kur`` docstring): tezgah lr=0,05 / betik
        lr=0,08 hizasizligi bir kez 1,15516 yerine 1,17813 uretti ve fark
        ozelliklerden sanildi.
        """
        if soguk:
            return SOGUK_EPOK_HIZLI if self.hizli else SOGUK_EPOK
        return SICAK_EPOK_HIZLI if self.hizli else SICAK_EPOK

    # ----------------------------------------------------------- predict

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """OFSETLI uzayda tahmin dondurur (cagiran ``log1p(guc)`` ekler)."""
        if not hasattr(self, "on_isleyici_"):
            raise RuntimeError("SinirAgi once fit edilmeli")
        hazir = self._cerceve_hazirla(X)
        Xt = self.on_isleyici_.transform(hazir).astype(np.float32, copy=False)
        if int(Xt.shape[1]) != self.genislik_:
            # T5: bolge'de 'YOK' gercek bir seviye ve seviye adlarinda Turkce
            # karakter var; herhangi bir ara adimda cp1254 okunursa seviye
            # IKIYE bolunur ve gomme SESSIZCE kayar. Sessiz kaymayi
            # gurultulu bir cokmeye ceviriyoruz.
            raise RuntimeError(
                f"one-hot/gosterge genisligi kaydi: fit {self.genislik_} -> "
                f"predict {int(Xt.shape[1])}. Kategorik seviye kumesi ya da "
                "eksiklik deseni degisti (kodlama? yeni seviye?)."
            )
        log_guc = _log_guc(X) if self.ofset else np.zeros(len(X), dtype="float64")
        p = (
            self.sinif_.predict_proba(Xt)[:, 1]
            if self.sinif_ is not None
            else np.full(len(X), self.p_taban_, dtype="float64")
        )
        _ham = np.mean([r.predict(Xt) for r in self.reg_listesi_], axis=0)
        m_ofs = self.hedef_olcek_.inverse_transform(_ham.reshape(-1, 1)).ravel()
        # KOSULLU ORTALAMA: E[log1p(Y)] = P(Y=0)*0 + P(Y>0)*E[log1p(Y)|Y>0].
        # Sifir kolunun hedefi TAHMIN EDILMEZ, ANALITIKTIR (log1p(0) = 0).
        # np.maximum(...,0) mesru: log1p(y) >= 0 her zaman.
        # "p yuksekse sifira yapistir" gibi bir son-islem KESINLIKLE YOK.
        log_tahmin = (1.0 - p) * np.maximum(m_ofs + log_guc, 0.0)
        return log_tahmin - log_guc


def sinir_agi_kur(*, hizli: bool, tohum: int) -> SinirAgi:
    """``model_kur`` kalibinda fabrika -- ``aile_modeli`` dalindan cagrilir."""
    return SinirAgi(tohum=tohum, hizli=hizli)


# ----------------------------------------------------- hat sozlesmesi


def egit_tahmin(
    aile: str,
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    agirlik_pay: float | None = None,
    ofset: bool = True,
    **ustyazim: object,
) -> np.ndarray:
    """``deney_ileri.egit_tahmin`` ile BIREBIR ayni imza ve donus sozlesmesi.

    DONUS: LOG UZAYINDA, yani ``log1p(tuketim)`` olceginde; uzunluk
    ``len(hedef)``. Cagiran ``expm1``i kendi yapar.

    ``egitim`` cercevesi ZATEN maskelenmis gelmeli -- maskeleme burada
    YAPILMAZ (uretim tarafiyla ayni disiplin; iki modulde ``soguk_maskele``
    imzasi TERS oldugu icin cagirmamak en guvenlisi).

    ``aile`` ``"sinir_agi"`` degilse cagri ``deney_ileri.egit_tahmin``e
    devredilir; boylece bu fonksiyon olcum donguleri icinde birebir yerine
    gecebilir.

    ``agirlik_pay`` verilirse ``tm.soguk_agirliklari`` ile ornek agirligi
    uretilir ve ``SinirAgi.fit`` bunu agirlikli yeniden ornekleme olarak
    isler (MLP ``sample_weight`` kabul etmez).

    ``ofset=False`` (cesitlilik kanali) desteklenir ama OLCULMEDI: sifir
    etiketinin turetilmesi ofsete bagli oldugu icin bu kanal uretimde
    KULLANILMAZ.
    """
    tm, di = _tezgah()
    if aile != AILE_ADI:
        return di.egit_tahmin(
            aile, egitim, hedef, kolonlar, tohum, agirlik_pay=agirlik_pay, ofset=ofset, **ustyazim
        )
    y = np.log1p(egitim[tm.HEDEF].clip(lower=0.0))
    if ofset:
        y = y - np.log1p(egitim["guc"])
    w = None
    if agirlik_pay is not None:
        w = tm.soguk_agirliklari(egitim["soguk_mu"], hedef_pay=agirlik_pay)
    model = SinirAgi(tohum=tohum, ofset=ofset, **ustyazim)
    model.fit(egitim[kolonlar], y.to_numpy(), sample_weight=w)
    ham = model.predict(hedef[kolonlar])
    return ham + np.log1p(hedef["guc"]).to_numpy() if ofset else ham


def _tezgah() -> tuple[Any, Any]:
    """``tuketim_model`` ve ``deney_ileri`` modullerini TEMBEL ithal eder.

    Tembellik kasitli: uretim tarafi bu modulu ``aile_modeli`` icinden
    ithal ediyor; modul basinda ``tuketim_model``i ithal etmek dairesel
    ithal riski yaratirdi.
    """
    for yol in (KOK / "src", KOK / "scripts"):
        if str(yol) not in sys.path:
            sys.path.insert(0, str(yol))
    import deney_ileri as di
    import tuketim_model as tm

    return tm, di


# ------------------------------------------------------- olcum tezgahi


def _harman(log_tahminler: dict[str, np.ndarray], agirlik: dict[str, float]) -> np.ndarray:
    """LOG UZAYINDA agirlikli aritmetik ortalama.

    Krogh-Vedelsby ozdesligi yalnizca ARITMETIK birlestiricide gecerli, ve
    uretim harmani ``expm1(mean(log1p(...)))`` seklinde. Bu yuzden uyeler
    log uzayinda birlestirilir.
    """
    toplam = sum(agirlik.values())
    birikim = np.zeros_like(next(iter(log_tahminler.values())))
    for ad, w in agirlik.items():
        birikim += w * log_tahminler[ad]
    return birikim / toplam


def _rmsle_log(hata: np.ndarray) -> float:
    """Log uzayindaki hatadan RMSLE. ``teshis_cesitlilik.py`` ile AYNI tanim."""
    return float(np.sqrt(np.mean(hata**2)))


def main(argv: list[str] | None = None) -> int:
    """Bagimsiz olcum: yaz25 blogunda SICAK rejim, 4. uye karari."""
    ayristirici = argparse.ArgumentParser(description="sinir agi -- 4. uye olcumu")
    ayristirici.add_argument("--blok", default="yaz25", help="dogrulama blogu")
    ayristirici.add_argument("--tohum", type=int, default=1000)
    ayristirici.add_argument("--rejim", default="SICAK", choices=("SICAK", "SOGUK"))
    ayristirici.add_argument("--hizli", action="store_true", help="az epok / az agac")
    ayristirici.add_argument("--sadece-ag", action="store_true", help="GBM'leri egitme")
    ayristirici.add_argument("--alt-ornek", type=float, default=0.40)
    ayristirici.add_argument("--aug-bayat", type=float, default=0.0)
    ayristirici.add_argument("--eksik-orani", action="store_true", help="A1 ablasyonu")
    ayristirici.add_argument("--budama", action="store_true", help="A2 ablasyonu")
    ayristirici.add_argument("--ayri-gosterge", action="store_true", help="A5 ablasyonu")
    args = ayristirici.parse_args(argv)

    tm, di = _tezgah()
    import deney as d

    from gridup.reporting import satir_tamponlu_cikti

    satir_tamponlu_cikti()
    t0 = time.time()
    motor, gerekce = motor_sec()
    print("=" * 92)
    print(f"SINIR AGI -- 4. UYE OLCUMU  ({args.blok} blogu, tohum {args.tohum})")
    print(f"motor: {motor}  --  {gerekce}")
    print("=" * 92)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, args.blok)
    log_gercek = np.log1p(gercek)

    # Kurulum uretimdeki REJIM_AYARLARI ile BIREBIR ayni olmali; olcum
    # duzenegi uretimden bir adim ayrilirsa olctugun sey gonderdigin sey degil.
    if args.rejim == "SOGUK":
        maske, cat_ust, ek_kolon = 1.00, {"depth": 7}, ("tk_haftanin_gunu", "tk_hafta_sonu")
        secim, agirlik3 = soguk, {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0}
    else:
        maske = 0.15
        cat_ust = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
        ek_kolon = ()
        secim, agirlik3 = ~soguk, {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
    kol = uretim + [k for k in ek_kolon if k not in uretim]

    print(f"\nkolon sayisi {len(kol)}  |  egitim {len(parca):,}  |  dogrulama {len(dogrulama):,}")
    print(f"{args.rejim} rejimi: {int(secim.sum()):,} satir, maske {maske:.2f}")
    maskeli = d.soguk_maskele(parca, kol, maske, args.tohum)

    hata: dict[str, np.ndarray] = {}
    log_tahmin: dict[str, np.ndarray] = {}
    aileler = () if args.sadece_ag else ("cat", "xgb", "lgbm")

    print("\n--- TEKIL RMSLE ---")
    for aile in aileler:
        t1 = time.time()
        ek_ust = cat_ust if aile == "cat" else {}
        log_t = di.egit_tahmin(aile, maskeli, dogrulama, kol, args.tohum, **ek_ust)
        log_tahmin[aile] = log_t
        hata[aile] = (log_t - log_gercek)[secim]
        print(f"  {aile:9} RMSLE {_rmsle_log(hata[aile]):.5f}   ({time.time() - t1:.0f} sn)")

    t1 = time.time()
    log_t = egit_tahmin(
        AILE_ADI,
        maskeli,
        dogrulama,
        kol,
        args.tohum,
        hizli=args.hizli,
        alt_ornek=args.alt_ornek,
        aug_bayat=args.aug_bayat,
        eksik_orani_kolonu=args.eksik_orani,
        budama=args.budama,
        ayri_gosterge=args.ayri_gosterge,
    )
    log_tahmin[AILE_ADI] = log_t
    hata[AILE_ADI] = (log_t - log_gercek)[secim]
    print(f"  {AILE_ADI:9} RMSLE {_rmsle_log(hata[AILE_ADI]):.5f}   ({time.time() - t1:.0f} sn)")

    if args.sadece_ag:
        print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika  (--sadece-ag: harman olculmedi)")
        return 0

    # --- HATA KORELASYONU -------------------------------------------------
    tum_aile = (*aileler, AILE_ADI)
    H = np.column_stack([hata[a] for a in tum_aile])
    K = np.corrcoef(H.T)
    print("\n--- HATA KORELASYON MATRISI (1,0 = ayni hatayi yapiyorlar) ---")
    print("             " + "  ".join(f"{a:>10}" for a in tum_aile))
    for i, a in enumerate(tum_aile):
        print(f"  {a:10} " + "  ".join(f"{K[i, j]:10.4f}" for j in range(len(tum_aile))))
    print(f"\n  sinir_agi <-> GBM ortalamasi: {float(K[3, :3].mean()):.4f}")
    print(f"  GBM'ler kendi arasinda      : {float((K[:3, :3].sum() - 3) / 6):.4f}")

    # --- KROGH-VEDELSBY ---------------------------------------------------
    ucu = H[:, :3]
    E_uye = float((ucu**2).mean())
    E_top = float((ucu.mean(axis=1) ** 2).mean())
    print("\n--- KROGH-VEDELSBY (uc GBM, esit agirlik) ---")
    print(f"  ortalama uye hatasi (MSE) {E_uye:.5f}  -> RMSLE {np.sqrt(E_uye):.5f}")
    A = E_uye - E_top
    print(f"  cesitlilik A              {A:.5f}  (%{100 * A / E_uye:.2f})")
    print(f"  TOPLULUK                  {E_top:.5f}  -> RMSLE {np.sqrt(E_top):.5f}")

    # --- 3'LU vs 4'LU HARMAN ----------------------------------------------
    # Karar kurali ESIK TABLOSU DEGIL, uretim agirliklariyla olculen izgara.
    # teshis_cesitlilik.py'deki tablo ESIT AGIRLIKLIDIR (her uye %25); uretim
    # sicak harmani 3/1/1 ve yeni uye ~%12 pay alir -- tablo o uye icin ASIRI
    # KATIDIR. Asagidaki izgara uretimin kendi agirliklariyla olculur.
    print(f"\n--- HARMAN: 3'LU (uretim {agirlik3}) vs 4'LU ---")
    taban_hata = (_harman(log_tahmin, agirlik3) - log_gercek)[secim]
    taban = _rmsle_log(taban_hata)
    print(f"  3'lu harman                       RMSLE {taban:.5f}")
    for w in (0.35, 0.7, 1.0, 1.4):
        dortlu = {**agirlik3, AILE_ADI: w}
        skor = _rmsle_log((_harman(log_tahmin, dortlu) - log_gercek)[secim])
        pay = w / sum(dortlu.values())
        print(
            f"  4'lu  w={w:<4}  (pay %{100 * pay:4.1f})       RMSLE {skor:.5f}"
            f"  ({skor - taban:+.5f})"
        )
    print("\n  NOT: karar S2 adiminda verilir -- UC BLOKTA, tek tohum 42, uretimin")
    print("  kendi dogrulama duzenegiyle. Buradaki tek blok bir ON ELEMEDIR.")
    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
