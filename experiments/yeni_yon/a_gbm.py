"""ADAY AILESI G -- ayni ozelliklerle FARKLI HEDEF / FARKLI KAYIP.

Uretim hatti butunuyle ``log1p(tuketim) - log1p(guc)`` hedefi ve L2 kaybi
uzerinde calisiyor. Ayni bilgiyi baska bir aciyla kodlayan model, span'in
disina cikar. Burada uc yon uretiliyor:

    G1  ham log1p hedefi (kapasite ofseti YOK)
    G2  kuantil kaybi (alpha=0.40) -- MSLE ortalamayi hedefler, bu medyan alti
    H1  hurdle: P(sifir) siniflandirici x pozitif-kosullu seviye
    R1  rejim sinirini BASKA yerden cizmek: soguk/sicak yerine
        "panel gecmisi kisa mi" (t_gun_sayisi) kohortlari, ayri modeller

Protokol: bir blok olculurken o blok egitimden CIKARILIR (uretimin K-katmanli
duzeni). Test icin uc blogun tamami egitim olur.

Kaggle'a hicbir sey gondermez.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import ortak

N_JOBS = 3
AYAR = dict(
    n_estimators=600,
    learning_rate=0.06,
    num_leaves=96,
    min_child_samples=60,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    n_jobs=N_JOBS,
    verbose=-1,
)


def _matris() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """(X_egitim, X_test, y_log, log1p(guc)_egitim, blok kodu)."""
    kol = ortak.SAYISAL + ortak.KATEGORIK + ["tuketim", "_blok"]
    e = ortak.egitim(kol)
    t = ortak.test(ortak.SAYISAL + ortak.KATEGORIK)
    Xe = e[ortak.SAYISAL + ortak.KATEGORIK].copy()
    Xt = t[ortak.SAYISAL + ortak.KATEGORIK].copy()
    for c in ortak.KATEGORIK:
        kat = pd.Categorical(pd.concat([Xe[c], Xt[c]], ignore_index=True).astype(str)).categories
        Xe[c] = pd.Categorical(Xe[c].astype(str), categories=kat)
        Xt[c] = pd.Categorical(Xt[c].astype(str), categories=kat)
    for c in ortak.SAYISAL:
        Xe[c] = Xe[c].astype("float32")
        Xt[c] = Xt[c].astype("float32")
    y = np.log1p(np.clip(e["tuketim"].to_numpy("float64"), 0.0, None))
    lgc = np.log1p(e["guc"].to_numpy("float64"))
    blok = e["_blok"].to_numpy()
    return Xe, Xt, y, lgc, blok


def _kos(
    ad: str,
    Xe: pd.DataFrame,
    Xt: pd.DataFrame,
    hedef: np.ndarray,
    blok: np.ndarray,
    geri: "callable",  # noqa: UP037
    **ustyaz,
) -> None:
    """Bir hedef tanimi icin 4 fit (3 blok + test). ``geri(pred, idx)`` -> log1p."""
    if ortak.var_mi(ad):
        print(f"{ad}: onbellekte, atlaniyor")
        return
    ay = {**AYAR, **ustyaz}
    cv: dict[str, np.ndarray] = {}
    for b in ortak.BLOKLAR:
        t0 = time.time()
        egt = blok != b
        m = lgb.LGBMRegressor(**ay)
        m.fit(Xe[egt], hedef[egt], categorical_feature=ortak.KATEGORIK)
        p = m.predict(Xe[~egt])
        cv[b] = np.clip(np.expm1(np.maximum(geri(p, ~egt), 0.0)), 0.0, None)
        print(f"  {ad}/{b}: {time.time() - t0:.0f}s")
    t0 = time.time()
    m = lgb.LGBMRegressor(**ay)
    m.fit(Xe, hedef, categorical_feature=ortak.KATEGORIK)
    p = m.predict(Xt)
    test_tahmin = np.clip(np.expm1(np.maximum(geri(p, None), 0.0)), 0.0, None)
    print(f"  {ad}/TEST: {time.time() - t0:.0f}s")
    ortak.kaydet(ad, cv, test_tahmin)


def main() -> None:
    Xe, Xt, y, lgc, blok = _matris()
    lgc_t = np.log1p(ortak.test(["guc"])["guc"].to_numpy("float64"))
    print(f"matris: egitim {Xe.shape}, test {Xt.shape}")

    def duz(p, idx):  # noqa: ANN001, ARG001
        return p

    def ofsetli(p, idx):  # noqa: ANN001
        return p + (lgc_t if idx is None else lgc[idx])

    # --- G1: HAM log1p hedefi (kapasite ofseti yok)
    _kos("G1_ham_hedef", Xe, Xt, y, blok, duz)

    # --- G2: kuantil kaybi, ofsetli hedef (uretimin hedefi, farkli KAYIP)
    _kos(
        "G2_kuantil40",
        Xe,
        Xt,
        y - lgc,
        blok,
        ofsetli,
        objective="quantile",
        alpha=0.40,
    )

    # --- R1: rejim sinirini panel gecmisi uzunlugundan cizmek
    _rejim(Xe, Xt, y, lgc, lgc_t, blok)

    # --- H1: hurdle
    _hurdle(Xe, Xt, y, lgc, lgc_t, blok)


def _rejim(Xe, Xt, y, lgc, lgc_t, blok) -> None:  # noqa: ANN001
    """R1 -- uretimin sicak/soguk/kuyruk uclusunun YERINE gecmis-uzunlugu kohortu.

    Kohort kenarlari ``t_gun_sayisi``: yok / <=60 / <=200 / >200. Her kohort
    KENDI modelini egitir; boylece ortak modelin havuzladigi etkiler ayrisir.
    """
    ad = "R1_kohort_gecmis"
    if ortak.var_mi(ad):
        print(f"{ad}: onbellekte, atlaniyor")
        return
    gs_e = Xe["t_gun_sayisi"].to_numpy("float64")
    gs_t = Xt["t_gun_sayisi"].to_numpy("float64")

    def koh(g):  # noqa: ANN001
        k = np.zeros(len(g), dtype="int64")
        k[np.isnan(g) | (g <= 0)] = 0
        k[(~np.isnan(g)) & (g > 0) & (g <= 60)] = 1
        k[(~np.isnan(g)) & (g > 60) & (g <= 200)] = 2
        k[(~np.isnan(g)) & (g > 200)] = 3
        return k

    ke, kt = koh(gs_e), koh(gs_t)
    hedef = y - lgc
    cv: dict[str, np.ndarray] = {}
    for b in ortak.BLOKLAR:
        t0 = time.time()
        egt = blok != b
        p = np.zeros(int((~egt).sum()), dtype="float64")
        alt_k = ke[~egt]
        for c in range(4):
            me = egt & (ke == c)
            mc = alt_k == c
            if mc.sum() == 0:
                continue
            if me.sum() < 5000:
                me = egt
            m = lgb.LGBMRegressor(**{**AYAR, "n_estimators": 350})
            m.fit(Xe[me], hedef[me], categorical_feature=ortak.KATEGORIK)
            p[mc] = m.predict(Xe[~egt][mc])
        cv[b] = np.clip(np.expm1(np.maximum(p + lgc[~egt], 0.0)), 0.0, None)
        print(f"  {ad}/{b}: {time.time() - t0:.0f}s")
    t0 = time.time()
    p = np.zeros(len(Xt), dtype="float64")
    for c in range(4):
        me = ke == c
        mc = kt == c
        if mc.sum() == 0:
            continue
        if me.sum() < 5000:
            me = np.ones(len(Xe), dtype=bool)
        m = lgb.LGBMRegressor(**{**AYAR, "n_estimators": 350})
        m.fit(Xe[me], hedef[me], categorical_feature=ortak.KATEGORIK)
        p[mc] = m.predict(Xt[mc])
    test_tahmin = np.clip(np.expm1(np.maximum(p + lgc_t, 0.0)), 0.0, None)
    print(f"  {ad}/TEST: {time.time() - t0:.0f}s")
    ortak.kaydet(ad, cv, test_tahmin)


def _hurdle(Xe, Xt, y, lgc, lgc_t, blok) -> None:  # noqa: ANN001
    """H1 -- sifir kutlesi ayri modellenir.

    MSLE altinda optimum tahmin ``E[log1p(y)]``. Iki parcali kurgu:
        E[log1p(y)] = (1-p) * E[log1p(y) | y>0]
    ``p`` = P(y==0) siniflandiricidan, ikinci carpan yalniz POZITIF satirlarda
    egitilmis regresyondan. Uretimin tek-parcali modeli bu ayrimi yapmiyor.
    """
    ad = "H1_hurdle_sifir"
    if ortak.var_mi(ad):
        print(f"{ad}: onbellekte, atlaniyor")
        return
    sifir = (y <= 0).astype("int64")
    poz = y > 0
    hedef = y - lgc
    cv: dict[str, np.ndarray] = {}
    for b in ortak.BLOKLAR:
        t0 = time.time()
        egt = blok != b
        c = lgb.LGBMClassifier(**{**AYAR, "n_estimators": 300})
        c.fit(Xe[egt], sifir[egt], categorical_feature=ortak.KATEGORIK)
        p0 = c.predict_proba(Xe[~egt])[:, 1]
        r = lgb.LGBMRegressor(**{**AYAR, "n_estimators": 350})
        r.fit(Xe[egt & poz], hedef[egt & poz], categorical_feature=ortak.KATEGORIK)
        mu = r.predict(Xe[~egt]) + lgc[~egt]
        cv[b] = np.clip(np.expm1(np.maximum((1.0 - p0) * mu, 0.0)), 0.0, None)
        print(f"  {ad}/{b}: {time.time() - t0:.0f}s")
    t0 = time.time()
    c = lgb.LGBMClassifier(**{**AYAR, "n_estimators": 300})
    c.fit(Xe, sifir, categorical_feature=ortak.KATEGORIK)
    p0 = c.predict_proba(Xt)[:, 1]
    r = lgb.LGBMRegressor(**{**AYAR, "n_estimators": 350})
    r.fit(Xe[poz], hedef[poz], categorical_feature=ortak.KATEGORIK)
    mu = r.predict(Xt) + lgc_t
    test_tahmin = np.clip(np.expm1(np.maximum((1.0 - p0) * mu, 0.0)), 0.0, None)
    print(f"  {ad}/TEST: {time.time() - t0:.0f}s")
    ortak.kaydet(ad, cv, test_tahmin)


if __name__ == "__main__":
    main()
