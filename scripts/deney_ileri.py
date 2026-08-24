"""Ileri deneyler -- ``deney.py`` tezgahinin uzantisi.

NEDEN AYRI DOSYA
----------------
``deney.py`` 950 satira geldi ve isini yapiyor: gurultu tabanini olcup
oznitelik/model/harman eksenlerini taradi. Buradaki deneyler onun
BULAMAYACAGI dort boslugu kapatiyor -- her biri, mevcut tezgahin kendi
kurgusundan dogan bir kor nokta:

1. SOGUK MASKE ORANI YANLIS AILE UZERINDE AYARLANDI.
   ``SOGUK_MASKE_ORANI = 0,2216`` LightGBM ile olculdu (%10 -0,004,
   %22 -0,019, %35 -0,012). Ama uretimdeki model CatBoost agirlikli
   harman. Bir hiperparametrenin optimumu model ailesine baglidir;
   LightGBM'in tepesi CatBoost'un tepesi degildir.

2. TOHUM TORBALAMA OLCULMEDI -- ama gonderimde ZATEN VAR.
   Dikkat: ``tuketim_model.main`` gonderimi uretirken uc tohumu log
   uzayinda biriktiriyor (``--tohum 3``), yani ``tuketim_v12.csv``
   torbalanmis. Torbalanmamis olan iki sey var ve ikisi de OLCUM tarafi:
     * ``deney.py`` uc tohumun SKORUNU ortaliyor, TAHMINLERINI degil;
     * ``egit_ve_olc`` dogrulama raporunu tek tohumla (42) uretiyor.
   Yani raporladigimiz CV, gonderdigimiz dosyayi OLDUGUNDAN KOTU
   gosteriyor. Krogh & Vedelsby ayrismasi (NeurIPS 1994) log uzayinda bir
   OZDESLIK:
       ortalama_uye_hatasi - ayrisma = harman_hatasi
   Ayrisma negatif olamaz, yani K tohumun log-uzayi ortalamasi ortalama
   tohumdan KOTU OLAMAZ. CatBoost'un tohum yayilmasi 0,004-0,011, yani
   ayrisma terimi sifir degil. Bu deney iki seyi soyler: dogru K, ve
   dogrulama raporunun ne kadar yaniltdigi.

3. ``soguk_agirliklari()`` YAZILMIS AMA HIC CAGRILMIYOR.
   ``tuketim_model.py:953``te duruyor, hicbir yerden cagrilmiyor. Ya olu
   kod ya olculmemis kaldirac; ikisi de bilinmeli. Soguk satirlar test
   karesel hatasinin %59'unu tasiyor ama egitim bloklarinda yalnizca
   %7,5-13,9 pay tutuyor -- yani model soguk rejimi test'tekinden AZ
   onemsiyor.

4. REJIM YONLENDIRMESI DENENMEDI.
   Tek model hem sicak hem soguk rejime hizmet ediyor. Test aninda bir
   trafonun gecmisi olup olmadigini BILIYORUZ (``soguk_mu``). Yani iki
   ayri uzman egitip satiri rejimine gore yonlendirmek mesru. Maske
   orani %100 olan bir model saf soguk uzmanidir: hicbir gecmis gormez.

TASARIM: BIR KEZ EGIT, COK KEZ DEGERLENDIR
------------------------------------------
Butun deneyler once tahminleri uretip bellekte tutar, sonra kombinasyonlar
uzerinde bedavaya gezer. Yonlendirme ve harman agirligi aritmetiktir --
hesaplama degil. Ayrica maskeleme (bir ``DataFrame.copy()``, ~800 MB)
blok x tohum basina BIR KEZ yapilip aileler arasinda paylasilir;
``deney.py`` bunu aile basina tekrarliyordu.

Calistirma::

    python scripts/deney_ileri.py --deney soguk_oran   # maske orani + yonlendirme
    python scripts/deney_ileri.py --deney torba        # tohum torbalama + harman
    python scripts/deney_ileri.py --deney yalin        # yalin oznitelik setleri
    python scripts/deney_ileri.py --deney agirlik      # soguk ornek agirligi
    python scripts/deney_ileri.py --deney ayar         # CatBoost duzenlilestirme
    python scripts/deney_ileri.py --deney naif         # ufka gore naif taban
    python scripts/deney_ileri.py --sure               # tek fit suresini olc
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

import deney as d  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SONUC_DOSYASI = KOK / "experiments" / "ileri_sonuclar.jsonl"

#: Tohumlar. Uc tohum bir olcum; tek tohum bir ornek.
TOHUMLAR = (1000, 1001, 1002)

#: Karar esigi = 2 x gurultu tabani (0,00998, ``deney.py --taban``).
ESIK = 0.01995


# ------------------------------------------------------------------ skor


@dataclass(frozen=True)
class Skor:
    """Bir tahminin REJIM BAZINDA skoru.

    Tek sayi yaniltici: soguk satirlar test'in %22,2'si ama karesel
    hatanin %59'u. Bir degisiklik geneli iyilestirip sogugu bozuyorsa
    bunu gormek sart -- yonlendirme fikri tam da bu ayrimdan dogdu.
    """

    sicak: float
    soguk: float

    @property
    def agirlikli(self) -> float:
        """Test'in soguk/sicak karisimina yeniden agirliklandirilmis RMSLE."""
        p = tm.TEST_SOGUK_PAYI
        return float(np.sqrt((1 - p) * self.sicak**2 + p * self.soguk**2))


def skorla(gercek: np.ndarray, soguk: np.ndarray, log_tahmin: np.ndarray) -> Skor:
    t = np.clip(np.expm1(log_tahmin), 0.0, None)
    return Skor(
        sicak=tm.rmsle(gercek[~soguk], t[~soguk]) if (~soguk).any() else 0.0,
        soguk=tm.rmsle(gercek[soguk], t[soguk]) if soguk.any() else 0.0,
    )


def kaydet(ad: str, blok_skorlari: dict[str, list[Skor]], ek: dict[str, object]) -> float:
    """Sonucu diske yazar ve GENEL skoru dondurur."""
    genel = float(np.mean([np.mean([s.agirlikli for s in v]) for v in blok_skorlari.values()]))
    SONUC_DOSYASI.parent.mkdir(parents=True, exist_ok=True)
    kayit = {
        "ad": ad,
        "genel": genel,
        "bloklar": {
            b: {
                "agirlikli": [s.agirlikli for s in v],
                "sicak": [s.sicak for s in v],
                "soguk": [s.soguk for s in v],
            }
            for b, v in blok_skorlari.items()
        },
        **{k: str(v) for k, v in ek.items()},
    }
    with SONUC_DOSYASI.open("a", encoding="utf-8") as f:
        f.write(json.dumps(kayit, ensure_ascii=False) + "\n")
    return genel


def yazdir(ad: str, blok_skorlari: dict[str, list[Skor]]) -> float:
    genel = float(np.mean([np.mean([s.agirlikli for s in v]) for v in blok_skorlari.values()]))
    parcalar = []
    for b, v in blok_skorlari.items():
        ort = np.mean([s.agirlikli for s in v])
        std = np.std([s.agirlikli for s in v], ddof=1) if len(v) > 1 else 0.0
        parcalar.append(f"{b} {ort:.4f}+-{std:.4f}")
    sicak = np.mean([s.sicak for v in blok_skorlari.values() for s in v])
    soguk = np.mean([s.soguk for v in blok_skorlari.values() for s in v])
    print(
        f"  {ad:36} GENEL {genel:.5f}  [sicak {sicak:.4f} soguk {soguk:.4f}]  " + " ".join(parcalar)
    )
    return genel


# ----------------------------------------------------------------- model


def aile_modeli(aile: str, tohum: int, **ustyazim: object):  # noqa: ANN201 - kosullu import
    """``deney._aile_modeli`` ile AYNI ayarlar, ama ustyazilabilir."""
    if aile == "cat":
        import catboost as cb

        p: dict[str, object] = {
            "loss_function": "RMSE",
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 5,
            "l2_leaf_reg": 3.0,
            "rsm": 0.75,
            "random_seed": tohum,
            "verbose": 0,
            "allow_writing_files": False,
        }
        p.update(ustyazim)
        return cb.CatBoostRegressor(**p)
    if aile == "xgb":
        import xgboost as xgb

        p = {
            "objective": "reg:squarederror",
            "n_estimators": d.AGAC,
            "learning_rate": 0.05,
            "max_depth": 8,
            "min_child_weight": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.75,
            "reg_lambda": 2.0,
            "random_state": tohum,
            "n_jobs": -1,
            "tree_method": "hist",
            "enable_categorical": True,
            "verbosity": 0,
        }
        p.update(ustyazim)
        return xgb.XGBRegressor(**p)
    if aile == "lgbm":
        return d._lgbm(tohum, **ustyazim)
    raise ValueError(f"bilinmeyen aile: {aile}")


def egit_tahmin(
    aile: str,
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    agirlik_pay: float | None = None,
    agirlik: np.ndarray | None = None,
    ofset: bool = True,
    **ustyazim: object,
) -> np.ndarray:
    """Bir aileyi egitip LOG UZAYINDA tahmin dondurur.

    ``egitim`` cercevesi ZATEN maskelenmis gelmeli -- maskeleme pahali bir
    kopyalama ve cagiran tarafta blok x tohum basina bir kez yapiliyor.

    Kapasite ofseti burada uygulanir: hedeften ``log1p(guc)`` cikarilir,
    tahmine geri eklenir. Log uzayinda satir-basi sabit kaydirma oldugu
    icin L2 optimumu degismez (olculdu: -0,0352).

    ``ofset=False`` ayni modeli HAM ``log1p(y)`` hedefiyle egitir. Tek
    basina daha kotu (olculdu: ofset -0,0352) ama harmanda CESITLILIK
    kaynagi: ASHRAE 1.'si tam bunu yapmis -- ``log1p(y/m2)`` ile egitilmis
    fazladan bir uye ekleyip "topluluga cesitlilik kattı ve skoru ~0,002
    iyilestirdi" demis. Uyeler ayni hedef donusumunu paylasmak zorunda
    degildir.
    """
    y = np.log1p(egitim[tm.HEDEF].clip(lower=0.0))
    if ofset:
        y = y - np.log1p(egitim["guc"])
    w = None
    if agirlik_pay is not None:
        w = tm.soguk_agirliklari(egitim["soguk_mu"], hedef_pay=agirlik_pay)
    if agirlik is not None:
        # Dogrudan verilen ornek agirligi. KOVARYAT KAYMA duzeltmesi icin:
        # egitim bayatlik dagilimi testinkinden 15-16 kat sapiyor
        # (``deney_bayatlik_agirlik.py``). ``agirlik_pay`` ile birlikte
        # verilirse ikisi CARPILIR -- farkli eksenler, birbirini ezmezler.
        a = np.asarray(agirlik, dtype="float64")
        if a.shape[0] != len(egitim):
            raise ValueError(f"agirlik {a.shape[0]} satir, egitim {len(egitim)} satir")
        w = a if w is None else np.asarray(w, dtype="float64") * a
    model = aile_modeli(aile, tohum, **ustyazim)
    x_e, x_h = egitim[kolonlar], hedef[kolonlar]
    if aile == "cat":
        # CatBoost kategorik dtype'i yutmuyor; ayri bildirmek istiyor.
        x_e, x_h = x_e.copy(), x_h.copy()
        kat = [k for k in tm.KATEGORIK if k in x_e.columns]
        for k in kat:
            x_e[k] = x_e[k].astype(str)
            x_h[k] = x_h[k].astype(str)
        model.fit(x_e, y, sample_weight=w, cat_features=kat)
    else:
        model.fit(x_e, y, sample_weight=w)
    ham = model.predict(x_h)
    return ham + np.log1p(hedef["guc"]).to_numpy() if ofset else ham


def blok_parcalari(
    egitim: pd.DataFrame, blok: str
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    dogrulama = egitim[egitim["_blok"] == blok]
    kalan = egitim[egitim["_blok"] != blok]
    return (
        kalan,
        dogrulama,
        dogrulama[tm.HEDEF].to_numpy(),
        (dogrulama["soguk_mu"] == 1).to_numpy(),
    )


# --------------------------------------------------- deney: soguk oran


#: Taranacak maske oranlari. 0,0 = maskeleme yok; 1,0 = SAF SOGUK UZMAN
#: (hicbir trafonun gecmisini gormez). Aradakiler tepeyi arar.
MASKE_ORANLARI = (0.0, 0.15, 0.2216, 0.35, 0.50, 0.70, 1.0)


def soguk_oran_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Maske oranini CatBoost uzerinde tarar ve REJIM YONLENDIRMESINI olcer.

    Iki soru birden:
      1) LightGBM'de %22 olan tepe, CatBoost'ta da %22 mi?
      2) Sicak satirlar icin en iyi oran ile soguk satirlar icin en iyi
         oran FARKLI mi? Farkliysa tek model yanlis kurgu -- satiri
         rejimine gore yonlendirmek mesru, cunku test aninda bir trafonun
         gecmisi olup olmadigini biliyoruz.
    """
    print("\n" + "=" * 92)
    print("SOGUK MASKE ORANI x CatBoost  +  REJIM YONLENDIRMESI")
    print("=" * 92)

    # blok -> oran -> tohum -> log tahmin
    tahmin: dict[str, dict[float, list[np.ndarray]]] = {}
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    log_guc: dict[str, np.ndarray] = {}

    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, b.ad)
        gercekler[b.ad] = (gercek, soguk)
        log_guc[b.ad] = np.log1p(dogrulama["guc"].to_numpy())
        tahmin[b.ad] = {}
        for oran in MASKE_ORANLARI:
            t0 = time.time()
            tahmin[b.ad][oran] = []
            for tohum in TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, kolonlar, oran, tohum)
                tahmin[b.ad][oran].append(egit_tahmin("cat", maskeli, dogrulama, kolonlar, tohum))
                del maskeli
            print(f"  {b.ad:6} oran {oran:.4f}  {len(TOHUMLAR)} tohum  ({time.time() - t0:.0f} sn)")

    def olc(ad: str, sec) -> float:  # noqa: ANN001 - ic yardimci
        """``sec(blok, tohum_index) -> log tahmin`` alir, skorlar."""
        blok_skorlari = {
            b.ad: [
                skorla(*gercekler[b.ad], sec(b.ad, i))  # type: ignore[arg-type]
                for i in range(len(TOHUMLAR))
            ]
            for b in tm.BLOKLAR
        }
        genel = yazdir(ad, blok_skorlari)
        kaydet(ad, blok_skorlari, {"deney": "soguk_oran"})
        return genel

    print("\n  --- TEK TOHUM (mevcut uretim bicimi) ---")
    tekil = {
        oran: olc(f"maske {oran:.4f}", lambda b, i, o=oran: tahmin[b][o][i])
        for oran in MASKE_ORANLARI
    }

    print("\n  --- TOHUM TORBALANMIS (3 tohumun log-uzayi ortalamasi) ---")
    torba = {}
    for oran in MASKE_ORANLARI:
        blok_skorlari = {
            b.ad: [skorla(*gercekler[b.ad], np.mean(tahmin[b.ad][oran], axis=0))]
            for b in tm.BLOKLAR
        }
        torba[oran] = yazdir(f"maske {oran:.4f} + torba", blok_skorlari)
        kaydet(f"maske {oran:.4f} + torba", blok_skorlari, {"deney": "soguk_oran_torba"})

    print("\n  --- REJIM YONLENDIRMESI (sicak oranA'dan, soguk oranB'den) ---")
    print("  torbalanmis tahminler uzerinde; yalnizca farkli oran ciftleri")
    en_iyi = (1e9, None)
    for a in MASKE_ORANLARI:
        for c in MASKE_ORANLARI:
            if a == c:
                continue
            blok_skorlari = {}
            for b in tm.BLOKLAR:
                _, soguk = gercekler[b.ad]
                birlesik = np.where(
                    soguk,
                    np.mean(tahmin[b.ad][c], axis=0),
                    np.mean(tahmin[b.ad][a], axis=0),
                )
                blok_skorlari[b.ad] = [skorla(*gercekler[b.ad], birlesik)]
            ad = f"sicak={a:.4f} soguk={c:.4f}"
            genel = yazdir(ad, blok_skorlari)
            kaydet(ad, blok_skorlari, {"deney": "yonlendirme"})
            if genel < en_iyi[0]:
                en_iyi = (genel, ad)

    taban = torba[tm.SOGUK_MASKE_ORANI]
    buzulme = buzulme_olc(tahmin, gercekler, log_guc)
    print("\n  --- OZET ---")
    print(f"  mevcut uretim (maske %22,16, tek tohum)   {tekil[tm.SOGUK_MASKE_ORANI]:.5f}")
    print(f"  ayni + tohum torbalama                    {taban:.5f}")
    print(f"  en iyi yonlendirme: {en_iyi[1]}   {en_iyi[0]:.5f}")
    print(f"  yonlendirme kazanci {taban - en_iyi[0]:+.5f}   (esik {ESIK:.5f})")
    print(f"  buzulme kazanci     {taban - buzulme:+.5f}")


def buzulme_olc(
    tahmin: dict[str, dict[float, list[np.ndarray]]],
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]],
    log_guc: dict[str, np.ndarray],
) -> float:
    """BUZULME: tahminleri rejim ortalamasina dogru cekmenin kazanci.

    NEDEN BU OLCUM VAR
    Soguk rejimde artigi hicbir bilgi aciklamiyor (olculdu: kuresel
    ortalama 2,0490, ilce ortalamasi 2,0444, ID oneki 2,0388 -- ucu de
    ayni). Bir model boyle bir rejimde yine de degisken tahmin uretir;
    o degiskenligin gercekle korelasyonu zayifsa, kareli hata altinda
    tahmini ortalamaya dogru cekmek MATEMATIKSEL OLARAK kazandirir:

        E[(y - (a + b*p))^2] en kucuk oldugunda b = Cov(y,p)/Var(p)

    ve b < 1 ise cekmek gerekir; kazanc (1-b)^2 * Var(p). Eger b ~ 1 ise
    yapilacak bir sey yoktur -- bu olcum onu da soyler.

    OLCUM ALANI: ofset uzayi, yani ``log1p(y) - log1p(guc)``. Ham log
    uzayinda cekmek tahmini KURESEL tuketim ortalamasina dogru cekerdi ve
    kapasite bilgisini yok ederdi; ofset uzayinda cekmek "kVA basina
    ortalama tuketim"e dogru ceker -- dogru onsel budur.

    SIZINTI YOK: her blogun egimi DIGER iki bloktan kestirilir.
    """
    print("\n  --- BUZULME (capraz-blok egim kestirimi, ofset uzayi) ---")
    oran = tm.SOGUK_MASKE_ORANI
    torbali = {b.ad: np.mean(tahmin[b.ad][oran], axis=0) for b in tm.BLOKLAR}
    # blok -> rejim -> (r_sapka, r)
    ciftler: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for b in tm.BLOKLAR:
        gercek, soguk = gercekler[b.ad]
        r = np.log1p(np.clip(gercek, 0.0, None)) - log_guc[b.ad]
        r_sapka = torbali[b.ad] - log_guc[b.ad]
        ciftler[b.ad] = {
            "soguk": (r_sapka[soguk], r[soguk]),
            "sicak": (r_sapka[~soguk], r[~soguk]),
        }

    blok_skorlari: dict[str, list[Skor]] = {}
    for b in tm.BLOKLAR:
        gercek, soguk = gercekler[b.ad]
        duzeltilmis = torbali[b.ad].copy()
        for rejim, maske in (("soguk", soguk), ("sicak", ~soguk)):
            digerleri = [ciftler[o.ad][rejim] for o in tm.BLOKLAR if o.ad != b.ad]
            x = np.concatenate([p[0] for p in digerleri])
            y = np.concatenate([p[1] for p in digerleri])
            if len(x) < 100:
                continue
            egim, kesim = np.polyfit(x, y, 1)
            r_sapka = torbali[b.ad][maske] - log_guc[b.ad][maske]
            duzeltilmis[maske] = (kesim + egim * r_sapka) + log_guc[b.ad][maske]
            print(f"    {b.ad:6} {rejim:6} egim {egim:+.4f}  kesim {kesim:+.4f}")
        blok_skorlari[b.ad] = [skorla(gercek, soguk, duzeltilmis)]
    genel = yazdir("buzulme (capraz-blok egim)", blok_skorlari)
    kaydet("buzulme", blok_skorlari, {"deney": "buzulme"})
    return genel


# ------------------------------------------------------------ deney: torba


def torba_deneyi(egitim: pd.DataFrame, kolonlar: list[str], tohum_sayisi: int = 5) -> None:
    """Tohum torbalama egrisi + torbalanmis uyelerle harman agirligi.

    Krogh & Vedelsby (NeurIPS 1994) ozdesligi log uzayinda birebir gecerli:
    K tohumun ortalamasi ORTALAMA tohumdan kotu olamaz. Soru "kazanc var
    mi" degil, "ne kadar ve nerede doyuyor".

    Maskeleme blok x tohum basina BIR KEZ yapilip uc aile arasinda
    paylasiliyor -- ``deney.harman_agirlik_deneyi`` bunu aile basina
    tekrarliyor ve her tekrar ~800 MB'lik bir kopyalama.
    """
    print("\n" + "=" * 92)
    print(f"TOHUM TORBALAMA ({tohum_sayisi} tohum) + HARMAN AGIRLIGI")
    print("=" * 92)
    aileler = ("cat", "xgb", "lgbm")
    tohumlar = tuple(1000 + i for i in range(tohum_sayisi))

    tahmin: dict[str, dict[str, list[np.ndarray]]] = {}
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, b.ad)
        gercekler[b.ad] = (gercek, soguk)
        tahmin[b.ad] = {a: [] for a in aileler}
        for tohum in tohumlar:
            t0 = time.time()
            maskeli = d.soguk_maskele(kalan, kolonlar, tm.SOGUK_MASKE_ORANI, tohum)
            for a in aileler:
                tahmin[b.ad][a].append(egit_tahmin(a, maskeli, dogrulama, kolonlar, tohum))
            del maskeli
            print(f"  {b.ad:6} tohum {tohum}  3 aile  ({time.time() - t0:.0f} sn)")

    print("\n  --- TORBALAMA EGRISI: K tohumun log-uzayi ortalamasi ---")
    for a in aileler:
        for k in range(1, tohum_sayisi + 1):
            blok_skorlari = {
                b.ad: [skorla(*gercekler[b.ad], np.mean(tahmin[b.ad][a][:k], axis=0))]
                for b in tm.BLOKLAR
            }
            yazdir(f"{a} K={k}", blok_skorlari)
            kaydet(f"{a} K={k}", blok_skorlari, {"deney": "torba", "aile": a, "K": k})

    print("\n  --- HARMAN (torbalanmis uyelerle) ---")
    torbali = {b.ad: {a: np.mean(tahmin[b.ad][a], axis=0) for a in aileler} for b in tm.BLOKLAR}
    en_iyi = (1e9, None)
    for ad, agirlik in (
        ("cat tek", {"cat": 1.0}),
        ("esit 1/1/1", {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0}),
        ("2/1/1", {"cat": 2.0, "xgb": 1.0, "lgbm": 1.0}),
        ("3/1/1 (mevcut)", {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}),
        ("4/1/1", {"cat": 4.0, "xgb": 1.0, "lgbm": 1.0}),
        ("6/1/1", {"cat": 6.0, "xgb": 1.0, "lgbm": 1.0}),
        ("cat+xgb 3/1", {"cat": 3.0, "xgb": 1.0}),
        ("cat+lgbm 3/1", {"cat": 3.0, "lgbm": 1.0}),
        ("4/2/1", {"cat": 4.0, "xgb": 2.0, "lgbm": 1.0}),
    ):
        toplam = sum(agirlik.values())
        blok_skorlari = {
            b.ad: [
                skorla(
                    *gercekler[b.ad],
                    sum(w * torbali[b.ad][a] for a, w in agirlik.items()) / toplam,
                )
            ]
            for b in tm.BLOKLAR
        }
        genel = yazdir(ad, blok_skorlari)
        kaydet(ad, blok_skorlari, {"deney": "harman_torbali", "agirlik": agirlik})
        if genel < en_iyi[0]:
            en_iyi = (genel, ad)
    print(f"\n  en iyi harman: {en_iyi[1]}   {en_iyi[0]:.5f}")

    print("\n  --- REJIM BAZINDA AGIRLIK ---")
    print("  soguk rejimde gecmis kolonlari NaN; aileler NaN'i farkli ele aliyor.")
    print("  Bu yuzden en iyi karisim iki rejimde ayni olmak zorunda degil.")
    _rejim_agirligi(torbali, gercekler, aileler)

    print("\n  --- BIRLESTIRICININ BICIMI (kuvvet ortalamasi / uyusmazlik kaymasi) ---")
    _birlestirici_bicimi(torbali, gercekler, aileler)


def _birlestirici_bicimi(
    torbali: dict[str, dict[str, np.ndarray]],
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]],
    aileler: tuple[str, ...],
) -> None:
    """Aritmetik-log-ortalama disindaki birlestiricileri olcer.

    NEDEN: Krogh & Vedelsby garantisi yalnizca metrigin kendi uzayindaki
    ARITMETIK ortalama icin gecerli ve o garanti "harman ORTALAMA uyeden
    kotu olamaz" der -- en iyi birlestirici oldugunu SOYLEMEZ. Iki
    yarismanin birincisi baska bicim kullanmis:
      * Rossmann 1. (Jacobusse): harmonik ortalama
      * ASHRAE 1.: Optuna ile ayarlanmis genellestirilmis agirlikli ortalama
    Ikisi de agir sag kuyruklu hedeflerde tek bir uyenin savrulmasini
    sonumler. Bizim hedefimizin maksimumu p99'un 3.700 katinda.

    Iki aile olculuyor:
      1. Kuvvet ortalamasi  PM_p = (ort((1+t)^p))^(1/p) - 1.
         p -> 0 mevcut harmanimizi TAM olarak verir; p=1 ham uzayda
         aritmetik, p=-1 harmonik.
      2. Uyusmazlik kaymasi  log_harman - c * std(uyeler).
         Uyeler ayristiginda tahmini asagi ceker. Tek parametreli ve log
         uzayinda tanimli oldugu icin kuvvet ortalamasindan daha kararli.
    """
    agirlik = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
    toplam = sum(agirlik.values())

    def skorla_hepsi(uret) -> tuple[float, list[str]]:  # noqa: ANN001 - ic yardimci
        blok_skorlari = {b.ad: [skorla(*gercekler[b.ad], uret(b.ad))] for b in tm.BLOKLAR}
        genel = float(np.mean([s[0].agirlikli for s in blok_skorlari.values()]))
        return genel, [f"{b} {v[0].agirlikli:.4f}" for b, v in blok_skorlari.items()]

    taban = None
    for p in (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0):

        def uret(b: str, _p=p) -> np.ndarray:
            if abs(_p) < 1e-9:  # p -> 0 : log uzayinda agirlikli aritmetik
                return sum(w * torbali[b][a] for a, w in agirlik.items()) / toplam
            # (1+t) = exp(log1p_tahmin); kuvvet ortalamasi log uzayinda kararli
            ust = sum(w * np.exp(_p * torbali[b][a]) for a, w in agirlik.items()) / toplam
            return np.log(np.clip(ust, 1e-12, None)) / _p

        genel, detay = skorla_hepsi(uret)
        if abs(p) < 1e-9:
            taban = genel
        etiket = "p=0 (MEVCUT)" if abs(p) < 1e-9 else f"p={p:+.2f}"
        print(f"    kuvvet ort {etiket:14} GENEL {genel:.5f}   " + "  ".join(detay))

    for c in (0.0, 0.1, 0.2, 0.35, 0.5):

        def uret(b: str, _c=c) -> np.ndarray:
            yigin = np.stack([torbali[b][a] for a in aileler])
            ort = sum(w * torbali[b][a] for a, w in agirlik.items()) / toplam
            return ort - _c * yigin.std(axis=0)

        genel, detay = skorla_hepsi(uret)
        print(f"    uyusmazlik c={c:.2f}         GENEL {genel:.5f}   " + "  ".join(detay))

    if taban is not None:
        print(f"    (taban = p=0, {taban:.5f}; esik {ESIK:.5f})")


def _rejim_agirligi(
    torbali: dict[str, dict[str, np.ndarray]],
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]],
    aileler: tuple[str, ...],
) -> None:
    """Sicak ve soguk satirlara AYRI harman agirligi arar.

    Mesru, cunku test aninda bir satirin rejimini biliyoruz. Ama iki kat
    daha fazla serbestlik demek: uc blogun kendisine uydurma riski de iki
    kat. Bu yuzden sonuc yalnizca HER UC BLOKTA ayni yonde ise ciddiye
    alinir -- rapor blok kirilimini basiyor.
    """
    adaylar = {
        "1/1/1": {"cat": 1.0, "xgb": 1.0, "lgbm": 1.0},
        "3/1/1": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0},
        "6/1/1": {"cat": 6.0, "xgb": 1.0, "lgbm": 1.0},
        "cat": {"cat": 1.0},
    }
    for rejim in ("sicak", "soguk"):
        print(f"    -- {rejim} satirlar --")
        for ad, agirlik in adaylar.items():
            toplam = sum(agirlik.values())
            parcalar = []
            for b in tm.BLOKLAR:
                gercek, soguk = gercekler[b.ad]
                maske = soguk if rejim == "soguk" else ~soguk
                if not maske.any():
                    parcalar.append(f"{b.ad} --")
                    continue
                karisim = sum(w * torbali[b.ad][a] for a, w in agirlik.items()) / toplam
                t = np.clip(np.expm1(karisim), 0.0, None)
                parcalar.append(f"{b.ad} {tm.rmsle(gercek[maske], t[maske]):.4f}")
            print(f"      {ad:10} " + "  ".join(parcalar))
    del aileler  # imza uyumu icin duruyor; agirliklar adaylarda tanimli


# ------------------------------------------------------------ deney: yalin


#: Yalin setler. Ablasyon 144 kolonun 125'inin OLCULEMEZ oldugunu soyledi
#: (esik 0,02214); ASHRAE kazananlari 10-35 oznitelikle calisti. Ama
#: "tek tek olculemez" ile "toplu cikarilabilir" ayni sey degil -- bu
#: deney farki olcer.
CEKIRDEK = (
    "guc",
    "il_key",
    "ilce_key",
    "bolge",
    "t_log_ort",
    "t_log_std",
    "t_log_medyan",
    "t_log_p10",
    "t_log_p90",
    "t_gun_sayisi",
    "t_sifir_orani",
    "t_yuk_faktoru",
    "t_doluluk",
    "t_log_son30",
    "t_son30_gun",
    "t_log_son90",
    "t_son90_gun",
    "t_hg_sapma",
    "t_ay_sapma",
    "soguk_mu",
    "tanim_num",
    "tanim_uzunluk",
    "tanim_on2",
    "yas",
    "ilk_gun_mu",
    "ufuk_gun",
    "sicaklik_ort",
    "cdd22",
    "cdd22_ort7",
    "gun_uzunlugu_saat",
    "tk_haftanin_gunu",
    "tk_ay",
    "tk_hafta_sonu",
    "tatil_mi",
    "ulusal_gunluk",
    "guc_yuzdelik",
)

MINI = (
    "guc",
    "ilce_key",
    "t_log_ort",
    "t_log_std",
    "t_log_medyan",
    "t_gun_sayisi",
    "t_sifir_orani",
    "t_log_son30",
    "t_log_son90",
    "t_hg_sapma",
    "soguk_mu",
    "tanim_num",
    "tanim_uzunluk",
    "yas",
    "ufuk_gun",
    "cdd22_ort7",
    "tk_haftanin_gunu",
)


def _cikar(kolonlar: list[str], *aileler: str) -> list[str]:
    at = {k for a in aileler for k in d._aile_kolonlari(kolonlar, a)}
    return [k for k in kolonlar if k not in at]


def yalin_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Yalin oznitelik setlerini CatBoost + tohum torbalamasiyla olcer."""
    print("\n" + "=" * 92)
    print("YALIN OZNITELIK SETLERI (CatBoost, 3 tohum torbalanmis)")
    print("=" * 92)

    setler: dict[str, list[str]] = {
        "tam (144)": kolonlar,
        "-takvim": _cikar(kolonlar, "takvim"),
        "-takvim -panel -grup": _cikar(
            kolonlar, "takvim", "panel_yapisi", "grup_seviye", "grup_profil"
        ),
        "-takvim -panel -grup -osm -arazi": _cikar(
            kolonlar,
            "takvim",
            "panel_yapisi",
            "grup_seviye",
            "grup_profil",
            "osm_altyapi",
            "arazi_ortusu",
        ),
        "cekirdek": [k for k in CEKIRDEK if k in kolonlar],
        "mini": [k for k in MINI if k in kolonlar],
    }

    # set -> blok -> torbalanmis log tahmin. Harman icin saklaniyor:
    # bir set TEK BASINA daha kotu olsa bile harmanda daha iyi olabilir.
    # Krogh & Vedelsby'de kazanc uyelerin FARKLILIGINDAN gelir, tek tek
    # iyi olmalarindan degil -- ve farkli oznitelik alt kumesi, tanim
    # geregi ilintisiz hata uretir.
    saklanan: dict[str, dict[str, np.ndarray]] = {}
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for ad, alt in setler.items():
        t0 = time.time()
        blok_skorlari = {}
        saklanan[ad] = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, b.ad)
            gercekler[b.ad] = (gercek, soguk)
            tahminler = []
            for tohum in TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, alt, tm.SOGUK_MASKE_ORANI, tohum)
                tahminler.append(egit_tahmin("cat", maskeli, dogrulama, alt, tohum))
                del maskeli
            saklanan[ad][b.ad] = np.mean(tahminler, axis=0)
            blok_skorlari[b.ad] = [skorla(gercek, soguk, saklanan[ad][b.ad])]
        yazdir(f"{ad} [{len(alt)}]", blok_skorlari)
        kaydet(ad, blok_skorlari, {"deney": "yalin", "kolon_sayisi": len(alt)})
        print(f"      ({time.time() - t0:.0f} sn)")

    print("\n  --- SET HARMANI (tam + yalin, log uzayinda) ---")
    tam = "tam (144)"
    for digeri in setler:
        if digeri == tam:
            continue
        for w in (0.25, 0.5):
            blok_skorlari = {
                b.ad: [
                    skorla(
                        *gercekler[b.ad],
                        (1 - w) * saklanan[tam][b.ad] + w * saklanan[digeri][b.ad],
                    )
                ]
                for b in tm.BLOKLAR
            }
            ad = f"tam + {w:.2f}x {digeri}"
            yazdir(ad, blok_skorlari)
            kaydet(ad, blok_skorlari, {"deney": "yalin_harman", "agirlik": w})


# --------------------------------------------------------- deney: agirlik


def agirlik_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """``soguk_agirliklari`` -- yazilmis ama hic cagrilmamis kaldirac.

    Egitim bloklarinda soguk pay %7,5-13,9; test'te %22,2. Yani model
    soguk rejimi test'tekinden AZ onemsiyor. Agirlik bu payi hedeflenen
    degere cikarir. Maskeleme zaten YAPAY soguk satir uretiyor, dolayisiyla
    etkin pay maskelemeden sonra olculmeli -- burada olculuyor.
    """
    print("\n" + "=" * 92)
    print("SOGUK ORNEK AGIRLIGI (CatBoost, 3 tohum torbalanmis)")
    print("=" * 92)
    etkin = float("nan")
    for pay in (None, 0.2216, 0.35, 0.50):
        t0 = time.time()
        blok_skorlari = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, b.ad)
            tahminler = []
            for tohum in TOHUMLAR:
                maskeli = d.soguk_maskele(kalan, kolonlar, tm.SOGUK_MASKE_ORANI, tohum)
                etkin = float((maskeli["soguk_mu"] == 1).mean())
                tahminler.append(
                    egit_tahmin("cat", maskeli, dogrulama, kolonlar, tohum, agirlik_pay=pay)
                )
                del maskeli
            blok_skorlari[b.ad] = [skorla(gercek, soguk, np.mean(tahminler, axis=0))]
        ad = "agirliksiz" if pay is None else f"soguk pay -> {pay:.4f}"
        if pay is None:
            ad += f" (maskeden sonra etkin pay {etkin:.4f})"
        yazdir(ad, blok_skorlari)
        kaydet(ad, blok_skorlari, {"deney": "agirlik", "pay": pay})
        print(f"      ({time.time() - t0:.0f} sn)")


# ----------------------------------------------------------- deney: naif


#: Ufuk kovalari. Olculdu (2026-08-21 gece): en iyi son30/uzun-ortalama
#: karisim agirligi ufuk boyunca 0,956'dan 0,503'e DUZGUN biçimde kayiyor.
UFUK_KOVALARI = (0, 15, 30, 45, 60, 75, 90, 105, 123)


def _naif_agirliklar(kaynak: pd.DataFrame) -> np.ndarray:
    """Her ufuk kovasi icin en iyi ``w``yi kapali formla cozer.

    ``min_w ||y - (w*a + (1-w)*b)||^2`` -> ``w = (a-b).(y-b) / (a-b).(a-b)``
    Tek parametre, kova basina 100 binden fazla satir; asiri uydurma yok.
    """
    y = np.log1p(kaynak[tm.HEDEF].clip(lower=0.0).to_numpy())
    a = kaynak["t_log_son30"].to_numpy()
    b = kaynak["t_log_ort"].to_numpy()
    u = kaynak["ufuk_gun"].to_numpy()
    gecerli = ~(np.isnan(a) | np.isnan(b))
    w = np.zeros(len(UFUK_KOVALARI) - 1)
    for i in range(len(w)):
        m = gecerli & (u >= UFUK_KOVALARI[i]) & (u < UFUK_KOVALARI[i + 1])
        if m.sum() < 500:
            w[i] = 0.5
            continue
        dd = a[m] - b[m]
        payda = float(dd @ dd)
        w[i] = float(dd @ (y[m] - b[m]) / payda) if payda > 0 else 0.5
    return np.clip(w, 0.0, 1.0)


def _naif_uygula(cerceve: pd.DataFrame, w: np.ndarray) -> np.ndarray:
    a = cerceve["t_log_son30"].to_numpy()
    b = cerceve["t_log_ort"].to_numpy()
    u = cerceve["ufuk_gun"].to_numpy()
    kova = np.clip(np.searchsorted(np.array(UFUK_KOVALARI[1:-1]), u, side="right"), 0, len(w) - 1)
    ww = w[kova]
    return ww * a + (1.0 - ww) * b


def naif_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Ufka gore agirliklandirilmis naif tahmini modele VERMEYI olcer.

    NEDEN: olculdu (bu gece) -- naif karisim sicak satirlarda 0,7792,
    modelin sicak skoru 0,80-0,84. Yani iki oznitelik ve sekiz parametre,
    144 oznitelikli uc aileli toplulugu geciyor. Model ``t_log_son30``,
    ``t_log_ort`` ve ``ufuk_gun``un ucune de sahip ama bu etkilesimi
    ogrenmiyor: agaclar surekli-x-surekli carpimlari merdivenle
    yaklastirir ve burada merdiven yetmiyor.

    Iki bicim olculuyor:
      A) OZNITELIK  -- ``t_naif`` kolonu eklenir. Guvenli: model kullanmak
         zorunda degil. ``t_`` onekiyle basladigi icin soguk maskeleme
         onu da NaN yapar; bu DOGRU, cunku gecmisten turuyor.
      B) TABAN      -- hedef ``log1p(y) - t_naif`` olur, tahmine geri
         eklenir. Guclu ama riskli: model artik yalnizca ARTIGI modelliyor.
         Soguk satirlarda ``t_naif`` NaN oldugu icin taban
         ``log1p(guc) + kuresel_ofset``e duser -- yani mevcut kurgu.

    SIZINTI YOK: ``w`` her blok icin DIGER bloklardan kestirilir.
    """
    print("\n" + "=" * 92)
    print("NAIF TABAN: ufka gore agirliklandirilmis karisim (CatBoost, 3 tohum torbalanmis)")
    print("=" * 92)

    for ad in ("TABAN (mevcut)", "A: t_naif OZNITELIK", "B: t_naif TABAN"):
        t0 = time.time()
        blok_skorlari = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, b.ad)
            w = _naif_agirliklar(kalan)
            if b.ad == tm.BLOKLAR[0].ad and ad.startswith("A"):
                print("    w(ufuk) = " + " ".join(f"{x:.3f}" for x in w))
            alt = list(kolonlar)
            kalan_x, dog_x = kalan, dogrulama
            if ad.startswith(("A", "B")):
                kalan_x = kalan.assign(t_naif=_naif_uygula(kalan, w))
                dog_x = dogrulama.assign(t_naif=_naif_uygula(dogrulama, w))
                if ad.startswith("A"):
                    alt = [*kolonlar, "t_naif"]
            tahminler = []
            for tohum in TOHUMLAR:
                maskeli = d.soguk_maskele(
                    kalan_x,
                    alt if ad.startswith("A") else [*alt, "t_naif"],
                    tm.SOGUK_MASKE_ORANI,
                    tohum,
                )
                if ad.startswith("B"):
                    kuresel = float(
                        (
                            np.log1p(maskeli[tm.HEDEF].clip(lower=0.0)) - np.log1p(maskeli["guc"])
                        ).mean()
                    )
                    e_taban = np.where(
                        np.isnan(maskeli["t_naif"].to_numpy()),
                        np.log1p(maskeli["guc"].to_numpy()) + kuresel,
                        maskeli["t_naif"].to_numpy(),
                    )
                    h_taban = np.where(
                        np.isnan(dog_x["t_naif"].to_numpy()),
                        np.log1p(dog_x["guc"].to_numpy()) + kuresel,
                        dog_x["t_naif"].to_numpy(),
                    )
                    y = np.log1p(maskeli[tm.HEDEF].clip(lower=0.0).to_numpy()) - e_taban
                    model = aile_modeli("cat", tohum)
                    x_e, x_h = maskeli[alt].copy(), dog_x[alt].copy()
                    kat = [k for k in tm.KATEGORIK if k in x_e.columns]
                    for k in kat:
                        x_e[k] = x_e[k].astype(str)
                        x_h[k] = x_h[k].astype(str)
                    model.fit(x_e, y, cat_features=kat)
                    tahminler.append(model.predict(x_h) + h_taban)
                else:
                    tahminler.append(egit_tahmin("cat", maskeli, dog_x, alt, tohum))
                del maskeli
            blok_skorlari[b.ad] = [skorla(gercek, soguk, np.mean(tahminler, axis=0))]
        yazdir(f"{ad} [{len(alt)}]", blok_skorlari)
        kaydet(ad, blok_skorlari, {"deney": "naif"})
        print(f"      ({time.time() - t0:.0f} sn)")


# ----------------------------------------------------------- deney: ayar


#: CatBoost'ta HIC DOKUNULMAMIS parametreler. Uretim cagrisi yalnizca alti
#: anahtar veriyor; gerisi kutuphane varsayilani:
#:   * ``l2_leaf_reg=3.0`` CatBoost'un KENDI varsayilani -- "ayarlandi"
#:     denmesine ragmen hic oynatilmamis.
#:   * ``bootstrap_type`` sessizce ``MVS``; ``Bernoulli``/``Bayesian``
#:     hicbir arama uzayina girmemis.
#:   * ``random_strength`` bolme SECIMINI duzenler, hic denenmemis.
#: Derinlik ve iterasyon BILEREK yok: o yuzey tarandi ve duz cikti
#: (d4-d6 / 150-300 arasi 1,109-1,114). Olculmus duz bir bolgeyi yeniden
#: taramak butce israfidir.
AYARLAR: tuple[tuple[str, dict[str, object]], ...] = (
    ("TABAN (mevcut)", {}),
    ("l2_leaf_reg=1", {"l2_leaf_reg": 1.0}),
    ("l2_leaf_reg=8", {"l2_leaf_reg": 8.0}),
    ("l2_leaf_reg=20", {"l2_leaf_reg": 20.0}),
    ("bootstrap=Bayesian", {"bootstrap_type": "Bayesian", "bagging_temperature": 1.0}),
    ("bootstrap=Bernoulli 0.7", {"bootstrap_type": "Bernoulli", "subsample": 0.7}),
    ("bootstrap=MVS 0.6", {"bootstrap_type": "MVS", "subsample": 0.6}),
    ("random_strength=0.2", {"random_strength": 0.2}),
    ("random_strength=4", {"random_strength": 4.0}),
    ("random_strength=10", {"random_strength": 10.0}),
    ("lr=0.025 x 500", {"learning_rate": 0.025, "iterations": 500}),
    ("rsm=0.4", {"rsm": 0.4}),
    ("rsm=1.0", {"rsm": 1.0}),
)


def ayar_deneyi(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """CatBoost duzenlilestirme eksenlerini ESLESTIRILMIS olarak tarar.

    ESLESTIRILMIS: her aday TABAN ile AYNI tohumlari VE ayni maskelenmis
    cerceveleri kullanir. Bagimsiz tohumlarla kiyaslamak, farki tohum
    gurultusuyle (std 0,010) karistirir; esli kiyasta o gurultunun buyuk
    kismi ikisinde de ayni yonde hareket eder ve fark uzerinden dusor.
    """
    print("\n" + "=" * 92)
    print("CatBoost AYAR TARAMASI (eslestirilmis, 3 tohum, tohum-torbalanmis)")
    print("=" * 92)
    gercekler: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    parcalar: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    maskeli: dict[tuple[str, int], pd.DataFrame] = {}
    for b in tm.BLOKLAR:
        kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, b.ad)
        gercekler[b.ad] = (gercek, soguk)
        parcalar[b.ad] = (kalan, dogrulama)

    taban = None
    for ad, ustyazim in AYARLAR:
        t0 = time.time()
        blok_skorlari = {}
        for b in tm.BLOKLAR:
            kalan, dogrulama = parcalar[b.ad]
            tahminler = []
            for tohum in TOHUMLAR:
                anahtar = (b.ad, tohum)
                if anahtar not in maskeli:
                    maskeli[anahtar] = d.soguk_maskele(kalan, kolonlar, tm.SOGUK_MASKE_ORANI, tohum)
                tahminler.append(
                    egit_tahmin("cat", maskeli[anahtar], dogrulama, kolonlar, tohum, **ustyazim)
                )
            blok_skorlari[b.ad] = [skorla(*gercekler[b.ad], np.mean(tahminler, axis=0))]
        genel = yazdir(ad, blok_skorlari)
        kaydet(ad, blok_skorlari, {"deney": "ayar", "ustyazim": ustyazim})
        if taban is None:
            taban = genel
        else:
            isaret = "GECTI" if taban - genel > ESIK else "esik alti"
            print(f"      taban farki {taban - genel:+.5f}  {isaret}   ({time.time() - t0:.0f} sn)")
        # bellek: maskeli cerceveler blok x tohum basina 800 MB
        if len(maskeli) > len(tm.BLOKLAR) * len(TOHUMLAR):
            maskeli.clear()


# ------------------------------------------------------------------- sure


def sure_olc(egitim: pd.DataFrame, kolonlar: list[str]) -> None:
    """Tek fit suresini olcer -- butce tahmine degil OLCUME dayansin."""
    print("\n" + "=" * 92)
    print("SURE OLCUMU -- blok yaz25, tek tohum")
    print("=" * 92)
    kalan, dogrulama, gercek, soguk = blok_parcalari(egitim, "yaz25")
    t0 = time.time()
    maskeli = d.soguk_maskele(kalan, kolonlar, tm.SOGUK_MASKE_ORANI, 1000)
    t_maske = time.time() - t0
    print(f"  maskeleme       {t_maske:6.1f} sn   ({len(kalan):,} satir kopyalandi)")
    for a in ("cat", "xgb", "lgbm"):
        t0 = time.time()
        log_t = egit_tahmin(a, maskeli, dogrulama, kolonlar, 1000)
        s = skorla(gercek, soguk, log_t)
        print(f"  {a:5} fit+tahmin {time.time() - t0:6.1f} sn   agirlikli {s.agirlikli:.5f}")


# ------------------------------------------------------------------- ana


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--deney",
        choices=("soguk_oran", "torba", "yalin", "agirlik", "ayar", "naif"),
        help="calistirilacak deney",
    )
    ap.add_argument("--sure", action="store_true", help="tek fit suresini olc")
    ap.add_argument("--tohum-sayisi", type=int, default=5, help="torba deneyi tohum sayisi")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 92)
    print("ILERI DENEY TEZGAHI")
    print("=" * 92)
    egitim, test = d.cerceveleri_kur()
    kolonlar = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    tm.kategorik_kodla(egitim, test)
    print(f"  egitim {len(egitim):,} satir | {len(kolonlar)} oznitelik | esik {ESIK:.5f}")

    if args.sure:
        sure_olc(egitim, kolonlar)
    if args.deney == "soguk_oran":
        soguk_oran_deneyi(egitim, kolonlar)
    if args.deney == "torba":
        torba_deneyi(egitim, kolonlar, args.tohum_sayisi)
    if args.deney == "yalin":
        yalin_deneyi(egitim, kolonlar)
    if args.deney == "agirlik":
        agirlik_deneyi(egitim, kolonlar)
    if args.deney == "ayar":
        ayar_deneyi(egitim, kolonlar)
    if args.deney == "naif":
        naif_deneyi(egitim, kolonlar)

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika   -> {SONUC_DOSYASI}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
