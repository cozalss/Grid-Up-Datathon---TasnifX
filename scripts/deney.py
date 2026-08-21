"""Tuketim modeli icin DENEY TEZGAHI -- once gurultu tabanini olcer.

NEDEN BU BETIK VAR
------------------
21 Agustos'ta ust uste dort degisiklik denendi ve dordu de "belki iyi
belki kotu" ciktisi verdi (yaz25 test-agirlikli):

    v4  guncellik + trend + agirlik      1,0677
    v5  + grup profilleri + ulusal       1,0686
    v7  + olu trafo ozellikleri          1,0618
    v10 + panel yapisi + ID              1,1497 (sizinti duzeltmesinden SONRA)

Ayni donemde erken durdurma 22 ile 382 agac arasinda zipladi. Yani
olculen fark, olcum aracinin kendi oynakligindan kucuktu. Boyle bir
tezgahta ilerleme "sansli tohum secmek"e donusur.

Bu betigin ilk isi bir SAYI uretmek: **ayni yapilandirmayi farkli
tohumlarla kosunca skor ne kadar oynuyor.** O sayi, bundan sonraki her
kararin esigidir. Esigin altindaki bir "iyilesme" iyilesme degildir.

TASARIM KARARLARI
    * Erken durdurma YOK. Agac sayisi sabit. Olculdu (2026-08-21): egri
      200-1000 agac arasinda duz (1,0683 - 1,0706), yani erken durdurma
      hicbir sey kazandirmiyor ama her kosuda baska bir yerde durarak
      gurultu URETIYOR.
    * Cok tohum. Tek tohum bir ornektir, uc tohum bir olcumdur.
    * Ozellik cercevesi bir kez kurulup diske yaziliyor. Kurulum ~90 sn;
      deney basina odenirse gunde yapilabilecek deney sayisi dusuyor.
    * Skor her zaman TEST KARISIMINA agirliklandirilmis olarak raporlanir
      (bkz. tuketim_model.TEST_SOGUK_PAYI): bloklarin kendi soguk paylari
      %7,5-%13,9, test'inki %22,2. Ham blok skoru bloklari birbiriyle bile
      kiyaslamaya elverisli degil.

Calistirma::

    python scripts/deney.py --taban            # gurultu tabanini olc
    python scripts/deney.py --deney oznitelik  # oznitelik ailesi ablasyonu
    python scripts/deney.py --deney model      # LightGBM/CatBoost/XGBoost
    python scripts/deney.py --yenile           # onbellegi yeniden kur
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

ONBELLEK = KOK / "data" / "interim" / "deney"
SONUC_DOSYASI = KOK / "experiments" / "deney_sonuclari.jsonl"

#: Sabit agac sayisi. Olculdu: 200-1000 arasi duz, en iyi ~400.
AGAC = 400

#: Tohum sayisi. Uc tohum, ortalamanin standart hatasini tek tohumun
#: yaklasik %58'ine indirir (1/sqrt(3)); bes tohumda %45. Uc, deney
#: hizini kabul edilebilir tutan en dusuk anlamli sayi.
TOHUM_SAYISI = 3


@dataclass(frozen=True)
class Sonuc:
    """Bir yapilandirmanin cok tohumlu olcumu."""

    ad: str
    blok_skorlari: dict[str, list[float]]

    def blok_ort(self, blok: str) -> float:
        return float(np.mean(self.blok_skorlari[blok]))

    def blok_std(self, blok: str) -> float:
        return float(np.std(self.blok_skorlari[blok], ddof=1))

    @property
    def genel(self) -> float:
        return float(np.mean([self.blok_ort(b) for b in self.blok_skorlari]))

    @property
    def tohum_yayilmasi(self) -> float:
        """Bloklar arasinda ortalanmis tohum-ici standart sapma."""
        return float(np.mean([self.blok_std(b) for b in self.blok_skorlari]))

    def yazdir(self) -> None:
        parcalar = [
            f"{b} {self.blok_ort(b):.4f}+-{self.blok_std(b):.4f}" for b in self.blok_skorlari
        ]
        print(f"  {self.ad:38} GENEL {self.genel:.5f}   " + "  ".join(parcalar))


# ------------------------------------------------------------- onbellek


def cerceveleri_kur(yenile: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Egitim ve test ozellik cercevelerini kurar; diske onbellekler."""
    ONBELLEK.mkdir(parents=True, exist_ok=True)
    e_yol, t_yol = ONBELLEK / "egitim.parquet", ONBELLEK / "test.parquet"
    if not yenile and e_yol.exists() and t_yol.exists():
        print(f"  onbellekten okunuyor: {e_yol.parent}")
        return pd.read_parquet(e_yol), pd.read_parquet(t_yol)

    print("  cerceveler kuruluyor (onbellek yok ya da --yenile verildi)...")
    t0 = time.time()
    tr, te = tm.yukle()
    tr, te = tm.lokasyon_ayristir(tr), tm.lokasyon_ayristir(te)
    hava = tm.hava_yukle()
    tr, te = tm.hava_ekle(tr, hava), tm.hava_ekle(te, hava)
    tr, te = tm.takvim_ekle(tr), tm.takvim_ekle(te)
    tr, te = tm.yas_ekle(tr, te)
    tr, te = tm.kimlik_ekle(tr, te)
    tr, te = tm.statik_ilce_ekle(tr, te)
    tr, te = tm.ilce_yapisi_ekle(tr, te)
    tr, te = tm.ulusal_ekle(tr, te)
    egitim, test = tm.egitim_kur(tr), tm.test_kur(tr, te)
    egitim.to_parquet(e_yol, index=False)
    test.to_parquet(t_yol, index=False)
    print(f"  kuruldu ve yazildi ({time.time() - t0:.0f} sn)")
    return egitim, test


# --------------------------------------------------------------- olcum


def _lgbm(tohum: int, **ustyazim: object):  # noqa: ANN202 - kosullu import
    import lightgbm as lgb

    parametreler: dict[str, object] = {
        "objective": "regression",
        "n_estimators": AGAC,
        "learning_rate": 0.05,
        "num_leaves": 255,
        "min_child_samples": 40,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_lambda": 2.0,
        "random_state": tohum,
        "n_jobs": -1,
        "verbose": -1,
    }
    parametreler.update(ustyazim)
    return lgb.LGBMRegressor(**parametreler)


#: ``t_*`` ozetlerinin silinecegi kolon oneki. Soguk maskeleme bunlari
#: NaN yapar -- yani "bu trafonun gecmisi yok" durumunu birebir taklit eder.
_GECMIS_ONEKI = ("t_",)


def soguk_maskele(
    cerceve: pd.DataFrame, kolonlar: list[str], oran: float, tohum: int
) -> pd.DataFrame:
    """Egitimde trafolarin bir kismini YAPAY OLARAK soguga cevirir.

    DropoutNet'in (Volkovs ve ark., NeurIPS 2017) mekanizmasinin agac
    modeline tasinmis hali: egitim sirasinda rastgele secilen varliklarin
    gecmisten turemis kolonlari sifirlanir, boylece model servis anindaki
    girdi dagilimini EGITIMDE gorur ve tek model hem sicak hem soguk
    rejimi ogrenir. Yazarlarin ifadesi: "neural network models can be
    explicitly trained for cold start through dropout."

    Neden bize lazim -- olculdu (2026-08-21):
    Egitim bloklarindaki soguk satirlar dogal olarak olusuyor ama YANLI bir
    altkume: 2025 ortasinda devreye alinmis trafolar. Test'teki soguk dilim
    baska bir kesim ve sistematik olarak DAHA BUYUK -- medyan ``guc``
    630 kVA, egitimdeki 400 kVA'ya karsi. Yani model, test'te karsilasacagi
    soguk girdi dagilimini egitimde hic gormuyor.

    Maskeleme trafo BAZINDA yapilir, satir bazinda degil: bir trafonun bazi
    gunlerinde gecmisi olup bazilarinda olmamasi diye bir sey yoktur, ve
    satir bazinda maskelemek modele o tutarsizligi ogretirdi.
    """
    if oran <= 0.0:
        return cerceve
    rng = np.random.default_rng(tohum)
    trafolar = cerceve[tm.GRUP].unique()
    secilen = set(rng.choice(trafolar, size=int(len(trafolar) * oran), replace=False))
    maske = cerceve[tm.GRUP].isin(secilen).to_numpy()
    sonuc = cerceve.copy()
    gecmis_kolonlari = [k for k in kolonlar if k.startswith(_GECMIS_ONEKI)]
    sonuc.loc[maske, gecmis_kolonlari] = np.nan
    if "soguk_mu" in sonuc.columns:
        sonuc.loc[maske, "soguk_mu"] = 1
    return sonuc


def blok_olc(
    egitim: pd.DataFrame,
    kolonlar: list[str],
    blok: str,
    tohum: int,
    *,
    soguk_maske: float = 0.0,
    ofset: bool = False,
    **ustyazim: object,
) -> float:
    """Bir blogu bir tohumla olcer. Erken durdurma YOK -- sabit agac.

    Args:
        soguk_maske: Egitimde yapay olarak sogutulacak trafo orani.
        ofset: ``log1p(guc)`` hedeften CIKARILIR, tahmine geri eklenir.
            Log uzayinda satir-basi sabit bir kaydirma oldugu icin L2
            optimumu DEGISMEZ -- yani metrik acisindan birebir ayni
            problem, ama agaclarin olcegi ogrenmesi gerekmez. Olculdu
            (varlik-disarida-birakmali, 356.218 soguk satir): tek
            parametreli global ofset 1,7746 -- 20 kovalik ``guc``
            kodlamasiyla (1,7696) ayni bantta, ve mutlak global
            ortalamadan (2,0331) 0,26 RMSLE iyi.
    """
    import lightgbm as lgb

    dogrulama = egitim[egitim["_blok"] == blok]
    kalan = egitim[egitim["_blok"] != blok]
    kalan = soguk_maskele(kalan, kolonlar, soguk_maske, tohum)

    y = np.log1p(kalan[tm.HEDEF].clip(lower=0.0))
    if ofset:
        y = y - np.log1p(kalan["guc"])
    model = _lgbm(tohum, **ustyazim)
    model.fit(kalan[kolonlar], y, callbacks=[lgb.log_evaluation(0)])
    ham = model.predict(dogrulama[kolonlar])
    if ofset:
        ham = ham + np.log1p(dogrulama["guc"]).to_numpy()
    tahmin = np.clip(np.expm1(ham), 0.0, None)
    gercek = dogrulama[tm.HEDEF].to_numpy()
    soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
    s = tm.rmsle(gercek[soguk], tahmin[soguk]) if soguk.any() else 0.0
    w = tm.rmsle(gercek[~soguk], tahmin[~soguk]) if (~soguk).any() else 0.0
    # Test karisimina yeniden agirliklandirilmis birlesim.
    return float(np.sqrt((1 - tm.TEST_SOGUK_PAYI) * w**2 + tm.TEST_SOGUK_PAYI * s**2))


def olc(
    egitim: pd.DataFrame,
    kolonlar: list[str],
    ad: str,
    *,
    tohum_sayisi: int = TOHUM_SAYISI,
    **ustyazim: object,
) -> Sonuc:
    skorlar = {
        b.ad: [blok_olc(egitim, kolonlar, b.ad, 1000 + i, **ustyazim) for i in range(tohum_sayisi)]
        for b in tm.BLOKLAR
    }
    sonuc = Sonuc(ad=ad, blok_skorlari=skorlar)
    sonuc.yazdir()
    _kaydet(sonuc, len(kolonlar), ustyazim)
    return sonuc


def _kaydet(sonuc: Sonuc, kolon_sayisi: int, ustyazim: dict[str, object]) -> None:
    SONUC_DOSYASI.parent.mkdir(parents=True, exist_ok=True)
    kayit = {
        "ad": sonuc.ad,
        "genel": sonuc.genel,
        "tohum_yayilmasi": sonuc.tohum_yayilmasi,
        "kolon_sayisi": kolon_sayisi,
        "agac": AGAC,
        "ustyazim": {k: str(v) for k, v in ustyazim.items()},
        "blok_skorlari": sonuc.blok_skorlari,
    }
    with SONUC_DOSYASI.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")


# ------------------------------------------------------------ deneyler


#: Oznitelik AILELERI -- ablasyon bunlari birer birer cikarir.
#: Onek eslesmesiyle tanimli; yeni bir kolon eklendiginde ilgili aileye
#: kendiliginden dahil olur.
AILELER: dict[str, tuple[str, ...]] = {
    "trafo_seviye": ("t_log_", "t_gun_sayisi", "t_sifir_orani", "t_yuk_faktoru", "t_son"),
    "trafo_profil": ("t_hg_sapma", "t_ay_sapma"),
    "trafo_olum": ("t_kuyruk_sifir", "t_olu_mu", "t_son_kayit_yasi"),
    "trafo_isil": ("t_egim_", "t_trend"),
    "grup_seviye": ("g_",),
    "grup_profil": ("gp_",),
    "panel_yapisi": ("p_",),
    "kimlik": ("tanim_",),
    "arazi_ortusu": (
        "agac_orani",
        "calilik_orani",
        "otlak_orani",
        "tarim_orani",
        "yerlesim_orani",
        "ciplak_orani",
        "su_orani",
        "bitki_ortusu_orani",
    ),
    "osm_altyapi": ("osm_", "trafo_basina_hat"),
    "ilce_yapisi": ("ilce_", "guc_yuzdelik", "guc_payi", "guc_medyan_orani"),
    "hava": (
        "sicaklik",
        "hissedilen",
        "isitma_derece",
        "sogutma_derece",
        "yagis",
        "ruzgar",
        "gunes",
        "asiri_",
        "nem_ort",
        "ciy_",
        "vpd_",
        "toprak_",
        "bulut_",
        "et0_",
        "gun_uzunlugu",
    ),
    "takvim": ("tk_", "tatil", "ramazan"),
    "ulusal": ("ulusal_",),
    "yas": ("yas", "ilk_gun_mu", "ufuk_gun"),
}


def _aile_kolonlari(kolonlar: list[str], aile: str) -> list[str]:
    onekler = AILELER[aile]
    return [k for k in kolonlar if k.startswith(onekler)]


def taban_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """GURULTU TABANI: ayni yapilandirma, bes tohum.

    Bu sayinin altindaki hicbir fark 'iyilesme' sayilmaz.
    """
    print("\n" + "=" * 78)
    print("GURULTU TABANI -- ayni yapilandirma, 5 tohum")
    print("=" * 78)
    sonuc = olc(egitim, kolonlar, "TABAN (tum oznitelikler)", tohum_sayisi=5)
    print()
    print(f"  TOHUM YAYILMASI (blok-ici std): {sonuc.tohum_yayilmasi:.5f}")
    print(f"  KARAR ESIGI (2 std)           : {2 * sonuc.tohum_yayilmasi:.5f}")
    print()
    print("  Bundan sonra: bu esikten kucuk her fark GURULTUDUR.")


def oznitelik_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Aile bazli ablasyon: her aileyi CIKAR, ne kaybediyoruz."""
    print("\n" + "=" * 78)
    print("OZNITELIK ABLASYONU -- her aile TEK TEK cikariliyor")
    print("=" * 78)
    taban = olc(egitim, kolonlar, "TABAN")
    print()
    satirlar = []
    for aile in AILELER:
        cikan = _aile_kolonlari(kolonlar, aile)
        if not cikan:
            print(f"  {aile:38} (bu ailede kolon yok -- atlandi)")
            continue
        kalan = [k for k in kolonlar if k not in set(cikan)]
        sonuc = olc(egitim, kalan, f"-{aile} ({len(cikan)} kolon)")
        satirlar.append((aile, len(cikan), sonuc.genel - taban.genel, sonuc.tohum_yayilmasi))
    print("\n  --- OZET: aile cikarilinca skor ne kadar KOTULESIYOR ---")
    print("  (pozitif = aile FAYDALI; negatif = aile ZARARLI)")
    esik = 2 * taban.tohum_yayilmasi
    for aile, n, fark, _ in sorted(satirlar, key=lambda x: -x[2]):
        hukum = "FAYDALI" if fark > esik else ("ZARARLI" if fark < -esik else "KARARSIZ")
        print(f"  {aile:16} {n:>3} kolon   delta {fark:+.5f}   {hukum}")
    print(f"\n  esik = 2 x tohum yayilmasi = {esik:.5f}")


def soguk_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Arastirmadan gelen dort adayin ESIGE karsi olcumu.

    Hepsi bagimsiz kaynaklarda olculmus tekniklerin bu veriye tasinmasi;
    hicbiri burada kanitlanmis sayilmaz. Karar esigi tohum yayilmasinin
    iki kati (bkz. ``taban_deneyi``).
    """
    print("\n" + "=" * 78)
    print("SOGUK REJIM ADAYLARI")
    print("=" * 78)
    taban = olc(egitim, kolonlar, "TABAN")
    esik = 2 * taban.tohum_yayilmasi
    print()

    adaylar: list[tuple[str, dict[str, object]]] = [
        ("ofset: log1p(guc) hedeften cikar", {"ofset": True}),
        ("soguk maske %10", {"soguk_maske": 0.10}),
        ("soguk maske %22 (test orani)", {"soguk_maske": 0.222}),
        ("soguk maske %35", {"soguk_maske": 0.35}),
        ("min_child=200", {"min_child_samples": 200}),
        ("min_child=500 + leaves=127", {"min_child_samples": 500, "num_leaves": 127}),
        ("leaves=63 + max_depth=8", {"num_leaves": 63, "max_depth": 8}),
        ("ofset + maske %22", {"ofset": True, "soguk_maske": 0.222}),
        (
            "ofset + maske %22 + min_child=200",
            {"ofset": True, "soguk_maske": 0.222, "min_child_samples": 200},
        ),
    ]
    sonuclar = [(ad, olc(egitim, kolonlar, ad, **kw)) for ad, kw in adaylar]

    print("\n  --- OZET (taban'a gore; negatif = IYI) ---")
    for ad, s in sorted(sonuclar, key=lambda x: x[1].genel):
        fark = s.genel - taban.genel
        hukum = "IYI" if fark < -esik else ("KOTU" if fark > esik else "kararsiz")
        print(f"  {ad:42} {s.genel:.5f}  delta {fark:+.5f}  {hukum}")
    print(f"\n  taban {taban.genel:.5f}   esik {esik:.5f}")


def model_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Model bicimi denemeleri -- literaturden gelen ucuz adaylar.

    ``linear_tree``: yapraklarda sabit yerine DOGRUSAL model. Shi ve ark.
    (IJCAI 2019, arXiv:1802.05640) yogun sayisal regresyonda hem daha hizli
    yakinsama hem daha iyi dogruluk olcmus. Buradaki gerekcesi ozel:
    ``log1p(tuketim)``in ``log1p(guc)``e egimi 1,063 olculdu -- yani hedef
    kapasiteye neredeyse DOGRUSAL bagli. Sabit yaprakli bir agac bu dogruyu
    merdivenlerle yaklastirmak zorunda ve egitimde gorulen kapasite
    araliginin DISINDA duz kaliyor. Soguk trafolarda tam olarak bu olur.

    ``varlik_esit``: satir degil VARLIK basina esit agirlik. Trafo seviyesi
    varliklar arasi bir iliski; 500 gunluk bir trafo su an 1 gunluk bir
    trafodan 500 kat agirlik tasiyor. Varliklar arasi problemde etkin
    orneklem 1.038.737 satir degil 5.344 trafo.
    """
    print("\n" + "=" * 78)
    print("MODEL BICIMI DENEYLERI")
    print("=" * 78)
    taban = olc(egitim, kolonlar, "TABAN")

    olc(egitim, kolonlar, "linear_tree=True", linear_tree=True)
    olc(egitim, kolonlar, "num_leaves=63", num_leaves=63)
    olc(egitim, kolonlar, "num_leaves=511 + min_child=100", num_leaves=511, min_child_samples=100)
    olc(egitim, kolonlar, "colsample=0.5", colsample_bytree=0.5)
    olc(egitim, kolonlar, "reg_lambda=20", reg_lambda=20.0)

    print(f"\n  esik = {2 * taban.tohum_yayilmasi:.5f} (taban tohum yayilmasinin iki kati)")


def kalibrasyon_olc(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Copas kalibrasyon egimi: tahminler fazla mi yayilmis?

    Copas (JRSS-B 1983): dogru kuculme carpani, fold-disi gercek degerin
    tahmine gore EGIMIDIR. Egim b < 1 ise tahminler asiri yayilmistir ve
    ``ort + b*(tahmin - ort)`` MSE'yi (1-b)^2 * Var(tahmin) kadar dusurur.

    Ama Van Calster ve ark. (2020) olcmus: TAHMIN EDILEN kuculme carpani,
    optimal olanla cogu zaman TERS korelasyonlu -- "en cok gerektiginde en
    az ise yariyor". Bu yuzden burada yalnizca OLCUYORUZ; uygulamak ancak
    b acikca 1'in altindaysa ve etki esigi asiyorsa anlamli.
    """
    import lightgbm as lgb

    print("\n" + "=" * 78)
    print("KALIBRASYON EGIMI (Copas)")
    print("=" * 78)
    for b in tm.BLOKLAR:
        dogrulama = egitim[egitim["_blok"] == b.ad]
        kalan = egitim[egitim["_blok"] != b.ad]
        model = _lgbm(1000)
        model.fit(
            kalan[kolonlar],
            np.log1p(kalan[tm.HEDEF].clip(lower=0.0)),
            callbacks=[lgb.log_evaluation(0)],
        )
        tahmin = model.predict(dogrulama[kolonlar])
        gercek = np.log1p(dogrulama[tm.HEDEF].clip(lower=0.0).to_numpy())
        soguk = (dogrulama["soguk_mu"] == 1).to_numpy()
        for ad, maske in (
            ("tumu", np.ones(len(tahmin), bool)),
            ("soguk", soguk),
            ("sicak", ~soguk),
        ):
            if maske.sum() < 10:
                continue
            x, y = tahmin[maske], gercek[maske]
            egim = float(np.polyfit(x, y, 1)[0])
            kazanc = (1 - egim) ** 2 * float(np.var(x)) if egim < 1 else 0.0
            print(
                f"  {b.ad:6} {ad:6} egim {egim:+.4f}  Var(tahmin) {np.var(x):.3f}"
                f"  potansiyel MSE kazanci {kazanc:.4f}"
            )
    print("\n  egim ~1,0 ise kalibrasyon zaten dogru -- yapilacak bir sey yok.")


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yenile", action="store_true", help="onbellegi yeniden kur")
    ap.add_argument("--taban", action="store_true", help="yalnizca gurultu tabanini olc")
    ap.add_argument(
        "--deney",
        choices=("oznitelik", "model", "kalibrasyon", "soguk"),
        help="calistirilacak deney",
    )
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 78)
    print("DENEY TEZGAHI")
    print("=" * 78)
    egitim, test = cerceveleri_kur(yenile=args.yenile)
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    print(f"  egitim {len(egitim):,} satir | {len(kolonlar)} oznitelik | {AGAC} sabit agac")

    if args.taban or not args.deney:
        taban_deneyi(egitim, kolonlar)
    if args.deney == "oznitelik":
        oznitelik_deneyi(egitim, kolonlar)
    if args.deney == "model":
        model_deneyi(egitim, kolonlar)
    if args.deney == "soguk":
        soguk_deneyi(egitim, kolonlar)
    if args.deney == "kalibrasyon":
        kalibrasyon_olc(egitim, kolonlar)

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika   -> {SONUC_DOSYASI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
