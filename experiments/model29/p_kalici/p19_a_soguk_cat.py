"""p19-A: SOGUK CATBOOST aday uretici.

URETIM REJIMI ile BIREBIR (scripts/uret_soguk_tahmin.py ile ayni yol):
  * dar egitim cercevesi (data/interim/deney/egitim.parquet), hedef blok disi
  * d.soguk_maskele(parca, kol, 1.00, tohum)  -- SAF soguk uzman (t_* hepsi NaN)
  * cat ustyazimi {"depth": 7}
  * kapasite ofseti: y = log1p(tuketim) - log1p(guc), tahmine geri eklenir
  * di.egit_tahmin ile ayni kod yolu

TEK DEGISEN: kayip fonksiyonu VEYA yakinlik agirligi (tau).

TABAN diskte ZATEN VAR: data/interim/deney/soguk_tahmin_{blok}.npz anahtari
"{tohum}_cat". Bu betik --aday TABAN ile onu YENIDEN uretip birebir dogrular.

Cikti: p19_{blok}_{tohum}_{aday}.npy  (SOGUK satirlarin log tahmini, float64)

    python p19_a_soguk_cat.py --blok yaz25 --tohum 1000 --aday huber_a10 l1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(KOK, "scripts"))
sys.path.insert(0, os.path.join(KOK, "src"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

URETIM_CAT = {"depth": 7}
DN = os.path.join(KOK, "data/interim/deney")

#: Blok etiket penceresinin BASLANGICI -- tau icin "kesim".
BLOK_KESIM = {"yaz25": "2025-04-01", "guz25": "2025-08-01", "kis26": "2025-12-01"}

T0 = time.time()


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f}dk]", *a, flush=True)


#: aday -> (hedef_modu, cat ustyazimi)   URETIM_CAT her zaman ONCE uygulanir.
#: CatBoost'ta Huber/Quantile/MAE icin leaf_estimation_method=Gradient sart.
ADAYLAR = {
    "TABAN": ("ofset", {}),
    "huber_a02": (
        "ofset",
        {"loss_function": "Huber:delta=0.2", "leaf_estimation_method": "Gradient"},
    ),
    "huber_a05": (
        "ofset",
        {"loss_function": "Huber:delta=0.5", "leaf_estimation_method": "Gradient"},
    ),
    "huber_a10": (
        "ofset",
        {"loss_function": "Huber:delta=1.0", "leaf_estimation_method": "Gradient"},
    ),
    "huber_a20": (
        "ofset",
        {"loss_function": "Huber:delta=2.0", "leaf_estimation_method": "Gradient"},
    ),
    "huber_a40": (
        "ofset",
        {"loss_function": "Huber:delta=4.0", "leaf_estimation_method": "Gradient"},
    ),
    "l1": ("ofset", {"loss_function": "MAE", "leaf_estimation_method": "Gradient"}),
    "quantile_05": (
        "ofset",
        {"loss_function": "Quantile:alpha=0.5", "leaf_estimation_method": "Gradient"},
    ),
    "mape": ("ofset", {"loss_function": "MAPE", "leaf_estimation_method": "Gradient"}),
    # --- hiperparametre (oncelik 3)
    "hp_l2r10": ("ofset", {"l2_leaf_reg": 10.0}),
    "hp_rs4": ("ofset", {"random_strength": 4.0}),
    "hp_lr03_it400": ("ofset", {"learning_rate": 0.03, "iterations": 400}),
    "hp_mdl50": ("ofset", {"min_data_in_leaf": 50}),
    "hp_bt10": ("ofset", {"bagging_temperature": 1.0}),
    "hp_derin6": ("ofset", {"depth": 6}),
    "hp_derin8": ("ofset", {"depth": 8}),
}
#: tau adaylari: kayip TABAN kalir, egitim satirlarina yakinlik agirligi verilir.
TAU_ADAYLAR = ("tau240", "tau480", "tau960", "tau1920")


def tau_agirlik(parca, kesim, tau):
    gun = (pd.Timestamp(kesim) - pd.to_datetime(parca["tarih"])).dt.days
    gun = np.maximum(gun.to_numpy(dtype="float64"), 0.0)
    w = np.exp(-gun / float(tau))
    return w / w.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blok", nargs="+", default=["yaz25"])
    ap.add_argument("--tohum", type=int, nargs="+", default=[1000])
    ap.add_argument("--aday", nargs="+", default=["TABAN"])
    ar = ap.parse_args()

    for a in ar.aday:
        if a not in ADAYLAR and a not in TAU_ADAYLAR:
            raise SystemExit(f"bilinmeyen aday: {a}")

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    kol = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    tm.kategorik_kodla(egitim, test)
    del test
    log(f"egitim {egitim.shape}  kolon {len(kol)}")

    yol = os.path.join(SP, "p19_a_skor_" + "-".join(ar.blok) + ".json")
    R = {}
    if os.path.exists(yol):
        with open(yol, encoding="utf-8") as fh:
            R = json.load(fh)

    for blok in ar.blok:
        parca, dogrulama, gercek, soguk = di.blok_parcalari(egitim, blok)
        log(f"{blok}: parca {parca.shape} dogrulama {len(dogrulama):,} soguk {int(soguk.sum()):,}")
        for tohum in ar.tohum:
            maskeli = None
            for ad in ar.aday:
                cikti = os.path.join(SP, f"p19_{blok}_{tohum}_{ad}.npy")
                if os.path.exists(cikti):
                    log(f"  {blok} t={tohum} {ad:16} zaten var, atlandi")
                    continue
                if maskeli is None:
                    maskeli = d.soguk_maskele(parca, kol, 1.00, tohum)
                    tk = [k for k in kol if k.startswith("t_")]
                    assert all(maskeli[k].isna().all() for k in tk), "maske 1.00 bozuk"
                ust = dict(URETIM_CAT)
                agir = None
                if ad in TAU_ADAYLAR:
                    agir = tau_agirlik(maskeli, BLOK_KESIM[blok], float(ad[3:]))
                else:
                    ust.update(ADAYLAR[ad][1])
                t = time.time()
                try:
                    lg = di.egit_tahmin(
                        "cat", maskeli, dogrulama, kol, tohum, agirlik=agir, ofset=True, **ust
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"  {blok} t={tohum} {ad:16} COKTU: {type(e).__name__} {e}")
                    R.setdefault(ad, {}).setdefault(blok, {})[str(tohum)] = "COKTU"
                    with open(yol, "w", encoding="utf-8") as fh:
                        json.dump(R, fh, indent=1, ensure_ascii=False)
                    continue
                lgs = lg[soguk].astype(np.float64)
                s = float(tm.rmsle(gercek[soguk], np.clip(np.expm1(lgs), 0.0, None)))
                np.save(cikti, lgs)
                R.setdefault(ad, {}).setdefault(blok, {})[str(tohum)] = round(s, 6)
                ek = ""
                if ad == "TABAN":  # BIREBIR DOGRULAMA
                    z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
                    k = f"{tohum}_cat"
                    if k in z.files:
                        mx = float(np.max(np.abs(z[k].astype(np.float64) - lgs)))
                        ek = f"  [birebir maxabs={mx:.3e}]"
                        R.setdefault("_dogrulama", {})[f"{blok}_{tohum}"] = mx
                log(f"  {blok} t={tohum} {ad:16} RMSLE={s:.6f} ({time.time() - t:.0f}sn){ek}")
                with open(yol, "w", encoding="utf-8") as fh:
                    json.dump(R, fh, indent=1, ensure_ascii=False)
            del maskeli
        del parca, dogrulama
    log("TAMAM")


if __name__ == "__main__":
    main()
