"""Grid Up Datathon -- SOTA Fast Multi-Model Pipeline.

Optimized for fast execution (~5 min) with 2 bagged seeds and full feature engineering.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import sota_tuketim_pipeline as sota

HAM = KOK / "data" / "raw"
DIS = KOK / "data" / "external"
GONDERIM = KOK / "submissions"
HEDEF = "tuketim"


def egit_ve_tahmin_et_fast(
    egitim: pd.DataFrame,
    hedef_cerceve: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    rejim: str,
) -> np.ndarray:
    maske_orani = 1.00 if rejim == "soguk" else 0.15
    e = sota.soguk_maskele(egitim, kolonlar, tohum, maske_orani)
    y = sota.ofsetli_hedef(e)

    x_e = e[kolonlar].copy()
    x_h = hedef_cerceve[kolonlar].copy()

    # 1. CatBoost (250 iter)
    import catboost as cb

    if rejim == "sicak":
        cb_params = dict(
            loss_function="RMSE",
            iterations=250,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=1.0,
            random_strength=4.0,
            rsm=0.75,
            random_seed=tohum,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        )
    else:
        cb_params = dict(
            loss_function="RMSE",
            iterations=220,
            learning_rate=0.05,
            depth=7,
            l2_leaf_reg=3.0,
            random_strength=2.0,
            rsm=0.75,
            random_seed=tohum,
            verbose=0,
            thread_count=-1,
            allow_writing_files=False,
        )

    kat_cols = [k for k in sota.KATEGORIK if k in x_e.columns]
    x_e_cb, x_h_cb = x_e.copy(), x_h.copy()
    for k in kat_cols:
        x_e_cb[k] = x_e_cb[k].astype(str)
        x_h_cb[k] = x_h_cb[k].astype(str)

    cb_model = cb.CatBoostRegressor(**cb_params)
    cb_model.fit(x_e_cb, y, cat_features=kat_cols)
    cb_pred = sota.ofseti_geri_ekle(cb_model.predict(x_h_cb), hedef_cerceve)

    # 2. LightGBM (250 trees)
    import lightgbm as lgb

    lgb_params = dict(
        objective="regression",
        n_estimators=250,
        learning_rate=0.05,
        num_leaves=127 if rejim == "sicak" else 63,
        min_child_samples=30 if rejim == "sicak" else 50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=tohum,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(x_e, y)
    lgb_pred = sota.ofseti_geri_ekle(lgb_model.predict(x_h), hedef_cerceve)

    # 3. XGBoost (250 trees)
    import xgboost as xgb

    xgb_params = dict(
        objective="reg:squarederror",
        n_estimators=250,
        learning_rate=0.05,
        max_depth=7 if rejim == "sicak" else 6,
        min_child_weight=15 if rejim == "sicak" else 25,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=tohum,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
    )
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(x_e, y)
    xgb_pred = sota.ofseti_geri_ekle(xgb_model.predict(x_h), hedef_cerceve)

    if rejim == "sicak":
        return (3.0 * cb_pred + 1.5 * xgb_pred + 1.5 * lgb_pred) / 6.0
    return (3.0 * cb_pred + 1.0 * xgb_pred + 1.0 * lgb_pred) / 5.0


def fast_tahmin_uret(
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    dar_egitim: pd.DataFrame | None = None,
) -> np.ndarray:
    soguk = (hedef["soguk_mu"] == 1).to_numpy()
    cikti = np.zeros(len(hedef), dtype="float64")

    if (~soguk).any():
        alt = hedef.loc[~soguk]
        cikti[~soguk] = egit_ve_tahmin_et_fast(egitim, alt, kolonlar, tohum, rejim="sicak")

    if soguk.any():
        alt = hedef.loc[soguk]
        kaynak = dar_egitim if dar_egitim is not None else egitim
        cikti[soguk] = egit_ve_tahmin_et_fast(kaynak, alt, kolonlar, tohum, rejim="soguk")

    return cikti


def main() -> int:
    sota.satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tohum", type=int, default=2, help="harmanlanacak tohum sayisi")
    ap.add_argument("--tohum-baslangic", type=int, default=42, help="ilk tohum")
    ap.add_argument("--cikti", default="tuketim_sota_v1.csv")
    ap.add_argument("--gonder", metavar="NOT", help="Kaggle gonderimi yap")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 80)
    print("GRID UP -- SOTA FAST PIPELINE")
    print("=" * 80)

    print("\n1/4  YUKLEME + LOKASYON")
    tr, te = sota.yukle()
    tr, te = sota.lokasyon_ayristir(tr), sota.lokasyon_ayristir(te)
    print(f"  train {len(tr):,} satir | test {len(te):,} satir")

    print("\n2/4  GELISMIS HAVA + TAKVIM + STATIK ILCE + ULUSAL")
    hava = sota.hava_yukle()
    tr, te = sota.hava_ekle(tr, hava), sota.hava_ekle(te, hava)
    tr, te = sota.gelismis_takvim_ekle(tr), sota.gelismis_takvim_ekle(te)
    tr, te = sota.yas_ekle(tr, te)
    tr, te = sota.kimlik_ekle(tr, te)
    tr, te = sota.statik_ilce_ekle(tr, te)
    tr, te = sota.ilce_yapisi_ekle(tr, te)
    tr, te = sota.ulusal_ekle(tr, te)

    print("\n3/4  BLOKLAR (SOTA yuvarlanan koken)")
    egitim = sota.sota_egitim_kur(tr)
    dar = egitim.copy()
    ek = sota.sota_ek_kokenleri_kur(tr)
    egitim = pd.concat([egitim, ek[egitim.columns]], ignore_index=True)
    print(f"  ek kokenlerle egitim {len(egitim):,} satir (dar set {len(dar):,})")

    test = sota.sota_test_kur(tr, te)

    sota.kategorik_kodla(egitim, dar, test)
    kolonlar = sota.oznitelikler(egitim)
    kolonlar = [k for k in kolonlar if k in test.columns]
    print(f"\n  Kullanilan toplam oznitelik sayisi: {len(kolonlar)}")

    print(f"\n4/4  SON EGITIM ({args.tohum} tohum x 3 model ailesi)")
    birikim = np.zeros(len(test), dtype="float64")
    for i in range(args.tohum):
        t_tohum = time.time()
        tohum = args.tohum_baslangic + i
        birikim += fast_tahmin_uret(egitim, test, kolonlar, tohum, dar_egitim=dar)
        print(
            f"    tohum {tohum} ({i + 1}/{args.tohum}) tamamlandi ({time.time() - t_tohum:.0f} sn)"
        )

    tahmin_final = np.clip(np.expm1(birikim / args.tohum), 0.0, None)
    tahmin_final = sota.olu_trafo_sifirla(tahmin_final, test)

    soguk_test = (test["soguk_mu"] == 1).to_numpy()
    if soguk_test.any():
        r = np.log1p(tahmin_final[soguk_test]) - np.log1p(test.loc[soguk_test, "guc"].to_numpy())
        r_buzulmus = r.mean() + 0.60 * (r - r.mean()) + 0.1046
        tahmin_final[soguk_test] = np.clip(
            np.expm1(r_buzulmus + np.log1p(test.loc[soguk_test, "guc"].to_numpy())), 0.0, None
        )
        print(
            "  [SON ISLEM] Soguk satirlara James-Stein buzulmesi ve seviye kalibrasyonu uygulandi."
        )

    GONDERIM.mkdir(parents=True, exist_ok=True)
    yol = GONDERIM / args.cikti
    pd.DataFrame({"id": test["id"].to_numpy(), HEDEF: tahmin_final}).to_csv(yol, index=False)
    print(f"\n  YAZILDI: {yol} ({len(tahmin_final):,} satir)")
    print(
        f"  Tahmin dagilimi: min={tahmin_final.min():.2f}, medyan={np.median(tahmin_final):.2f}, ort={tahmin_final.mean():.2f}, max={tahmin_final.max():.2f}"  # noqa: E501
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

    print(f"\nTAMAMLANDI -- Toplam sure: {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
