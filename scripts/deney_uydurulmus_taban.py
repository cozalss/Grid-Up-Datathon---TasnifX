r"""UYDURULMUS TABAN -- ofsetin katsayi bedelini ortadan kaldirir.

BULGU (2026-08-22)
------------------
Sicak hata tam ikiye ayriliyor (yaz25, 254.296 satir)::

    SEVIYE (trafo bazinda ort artik)   MSE 0,33336   %49,5
    SEKIL  (trafo ici sapma)           MSE 0,34006   %50,5

Ve modelin SEVIYE tahmini, naif bir kuraldan DAHA KOTU::

    t_log_son30 (ham)                 hata std 0,5651
    blok disi uydurulmus dogrusal     hata std 0,5459
    105 KOLONLUK MODEL                hata std 0,6010

Mekanizma: karar agaclari ``t_log_son30``u esiklerle bolerek merdiven
seklinde yaklasiyor; ``tahmin ~= son30 + duzeltme`` gibi DOGRUSAL bir
ozdesligi basamaklarla kurmak zorunda kaliyor. Kapasite ofseti dun tam bu
yuzden -0,0352 kazandirmisti.

NEDEN ONCEKI TABAN DENEMELERI BASARISIZ OLDU
--------------------------------------------
``docs/23-olcumler`` §17 uc HAM taban denedi ve hepsi reddedildi::

    taban            regresyon b    dayatmanin MSE bedeli
    log1p(guc)         0,97-1,20         0,0113   <- ALINDI
    t_log_ort          0,86-0,98         0,0434   <- reddedildi
    t_log_son30        0,91-1,02         0,0199   <- reddedildi

Bir ofset iki sey yapar: kosullandirmayi iyilestirir (kazanc) ve
katsayiyi 1'e CIVILER (bedel). Ham tabanlarin hicbirinin katsayisi 1
degildi, o yuzden bedel odendi.

EN KUCUK KARELE UYDURULMUS bir taban icin katsayi TANIM GEREGI 1'dir.
O bedel ortadan kalkar. Ayrica uydurma, tek bir gurultulu pencereye
(son30, 30 gunluk ortalama) dayanmak yerine pencereleri agirliklandirip
gurultuyu de azaltir -- ham son30'un ofset olarak kotu calismasinin ikinci
sebebi buydu: ofset, tabanin GURULTUSUNU katsayi 1 ile tahmine gecirir.

TASARIM
-------
Blok basina: seviye modeli DIGER bloklarin sicak trafolarindan uydurulur
(ozet ozellikleri -> etiket penceresi ortalama log1p seviyesi), hedef blogun
trafolarina uygulanir, ve hedef ondan CIKARILIR.

Soguk uzmanina DOKUNULMUYOR: gecmisi olmayan trafo icin seviye tahmini
uretilemez, tabani ``log1p(guc)`` kalir.

Adaylar::

    1  TABAN            ofset = log1p(guc)              [uretim]
    2  UYDURULMUS       ofset = uydurulmus seviye
    3  UYDURULMUS+OZ    ofset = uydurulmus seviye, AYRICA oznitelik olarak da

Fit: 3 aday x 3 blok x 3 tohum = 27 CatBoost ~ 30 dakika.

    python scripts/deney_uydurulmus_taban.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SICAK_MASKE = 0.15
USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}

#: Seviye modelinin girdileri -- hepsi blok icinde trafo basina SABIT.
#:
#: MINIMAL TUTULUYOR ve bu OLCULDU (2026-08-22, blok disi uydurup yaz25'te
#: sinayarak). Zengin set TRANSFERI BOZUYOR::
#:
#:     yalniz son30                  hata std 0,5459
#:     son30 + ort                            0,5503
#:     butun pencereler                       0,5476
#:     pencereler + yapi (10 kolon)           0,5776
#:     on bir kolon + guc                     0,6370   <- ilk denemem
#:     (kiyas) 105 kolonluk MODEL             0,6010
#:
#: Iki kolon secildi: ``t_log_son30`` en iyisi, ``t_log_ort`` ise son30
#: NaN oldugunda dolgu gorevi goruyor (kisa gecmisli trafolar).
SEVIYE_GIRDI: tuple[str, ...] = ("t_log_son30", "t_log_ort")

KAYIT = KOK / "experiments" / "uydurulmus_taban.jsonl"


def _trafo_tablosu(cerceve: pd.DataFrame) -> pd.DataFrame:
    """(blok, trafo) basina: ozet ozellikleri + etiket penceresi seviyesi.

    BLOGA GORE DE gruplaniyor. Yalnizca ``tanim``a gore gruplamak, iki
    blokta birden gorunen bir trafonun IKI FARKLI (ozet, etiket) ciftini
    tek satira karistirir -- ilk denememde tam bunu yapmis ve seviye
    modelinin hatasini 0,60'a sismisti (ham son30 0,5651 iken).
    """
    alt = cerceve[cerceve["soguk_mu"] == 0]
    g = alt.groupby(["_blok", "tanim"], observed=True)
    t = g[list(SEVIYE_GIRDI) + ["guc"]].first()
    t["hedef"] = g[tm.HEDEF].apply(lambda s: float(np.log1p(s.clip(lower=0)).mean()))
    return t


def _seviye_modeli(egit: pd.DataFrame) -> tuple[np.ndarray, pd.Series]:
    """En kucuk kareler. NaN'lar uydurma kumesinin MEDYANIYLA dolduruluyor."""
    girdi = list(SEVIYE_GIRDI)
    x = egit[girdi].copy()
    medyan = x.median()
    x = x.fillna(medyan)
    tasarim = np.c_[np.ones(len(x)), x.to_numpy()]
    katsayi, *_ = np.linalg.lstsq(tasarim, egit["hedef"].to_numpy(), rcond=None)
    return katsayi, medyan


def _seviye_uygula(katsayi: np.ndarray, medyan: pd.Series, cerceve: pd.DataFrame) -> np.ndarray:
    x = cerceve[list(SEVIYE_GIRDI)].copy().fillna(medyan)
    return np.c_[np.ones(len(x)), x.to_numpy()] @ katsayi


def _egit_tahmin(
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    ofset_e: np.ndarray,
    ofset_h: np.ndarray,
) -> np.ndarray:
    """CatBoost, VERILEN ofsetle. Log uzayinda tahmin dondurur."""
    import catboost as cb

    y = np.log1p(egitim[tm.HEDEF].clip(lower=0).to_numpy()) - ofset_e
    p: dict[str, object] = {
        "loss_function": "RMSE", "iterations": 250, "learning_rate": 0.05,
        "rsm": 0.75, "random_seed": tohum, "verbose": 0, "allow_writing_files": False,
        **USTYAZIM,
    }
    model = cb.CatBoostRegressor(**p)
    x_e, x_h = egitim[kolonlar].copy(), hedef[kolonlar].copy()
    kat = [k for k in tm.KATEGORIK if k in x_e.columns]
    for k in kat:
        x_e[k] = x_e[k].astype(str)
        x_h[k] = x_h[k].astype(str)
    model.fit(x_e, y, cat_features=kat)
    return model.predict(x_h) + ofset_h


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 100)
    print("UYDURULMUS TABAN -- ofsetin katsayi bedelini kaldirir")
    print("=" * 100)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kolonlar = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    maskeli = {}
    for b in tm.BLOKLAR:
        for tohum in di.TOHUMLAR:
            maskeli[(b.ad, tohum)] = d.soguk_maskele(
                parcalar[b.ad][0], kolonlar, SICAK_MASKE, tohum
            )

    # Blok basina seviye modeli -- DIGER bloklarin trafolarindan uyduruluyor.
    seviye: dict[str, tuple[np.ndarray, pd.Series]] = {}
    for b in tm.BLOKLAR:
        egit_t = _trafo_tablosu(egitim[egitim["_blok"] != b.ad])
        seviye[b.ad] = _seviye_modeli(egit_t.dropna(subset=["hedef"]))
        sina_t = _trafo_tablosu(egitim[egitim["_blok"] == b.ad]).dropna(subset=["hedef"])
        tahmin = _seviye_uygula(*seviye[b.ad], sina_t)
        hata = sina_t["hedef"].to_numpy() - tahmin
        print(f"  {b.ad:6} seviye modeli: {len(egit_t):,} trafodan uydu -> "
              f"{len(sina_t):,} trafoda hata std {hata.std():.4f} ort {hata.mean():+.4f}")

    adaylar = ("TABAN (guc ofseti)", "UYDURULMUS", "MERKEZLI UYDURULMUS")
    tekil: dict[str, dict[tuple[str, int], float]] = {a: {} for a in adaylar}
    torbali: dict[str, dict[str, float]] = {a: {} for a in adaylar}

    for ad in adaylar:
        t0 = time.time()
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            sic = ~soguk
            log_tahminler = []
            for tohum in di.TOHUMLAR:
                m, dv, kol = maskeli[(b.ad, tohum)], dogrulama, list(kolonlar)
                if ad == "TABAN (guc ofseti)":
                    oe = np.log1p(m["guc"].to_numpy())
                    oh = np.log1p(dv["guc"].to_numpy())
                else:
                    oe = _seviye_uygula(*seviye[b.ad], m)
                    oh = _seviye_uygula(*seviye[b.ad], dv)
                    if ad.startswith("MERKEZLI"):
                        # Kesisimi KALDIR. Uydurulmus tabanin blok bazinda
                        # buyuk bir yanliligi var (yaz25 +0,1747, guz25
                        # -0,2875, kis26 +0,0934) ve ofset onu katsayi 1 ile
                        # dogrudan tahmine geciriyordu -- ilk denemenin
                        # -0,043 batmasinin sebebi buydu.
                        #
                        # Merkezlenince geriye yalnizca TRAFOLAR ARASI
                        # goreli yapi kaliyor; aradigimiz da zaten o, cunku
                        # seviye hatasi trafolar arasi bir hata. Genel
                        # seviyeyi model kendisi ogreniyor (iyi yaptigi is).
                        #
                        # Egitim ve hedef AYNI sabitle kaydiriliyor, yoksa
                        # ofset egitimle hedef arasinda kayar.
                        ort = float(oe.mean())
                        oe, oh = oe - ort, oh - ort
                log_t = _egit_tahmin(m, dv, kol, tohum, oe, oh)
                log_tahminler.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[sic], tek[sic])
            harman = np.clip(np.expm1(np.mean(log_tahminler, axis=0)), 0.0, None)
            torbali[ad][b.ad] = tm.rmsle(gercek[sic], harman[sic])
        ort = float(np.mean(list(torbali[ad].values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in torbali[ad].items())
        print(f"\n  {ad:22} SICAK {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    taban = adaylar[0]
    kayitlar = []
    for ad in adaylar[1:]:
        f = np.array([tekil[taban][k] - tekil[ad][k] for k in tekil[taban]])
        o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
        t_d = o / sh if sh > 0 else 0.0
        hukum = "OLCULDU" if abs(t_d) >= 2 else "esik alti"
        print(f"\n  {ad} vs TABAN: {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}  {hukum}")
        for b in tm.BLOKLAR:
            bb = np.array([tekil[taban][(b.ad, t)] - tekil[ad][(b.ad, t)] for t in di.TOHUMLAR])
            print(f"      {b.ad:6} {bb.mean():+.5f}")
        kayitlar.append({"aday": ad, "fark": o, "sh": sh, "t": t_d, "hukum": hukum})

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
