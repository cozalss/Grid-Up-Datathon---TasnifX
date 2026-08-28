"""SICAK KOHORT ADAYLARI -- her aday UC blokta, BLOK-DISI ogrenilmis.

KURAL: bir duzeltme hedef blogun kendi etiketinden ASLA ogrenilmez.
Grup ofsetleri / kalibrasyon egrileri / yigin modelleri DIGER IKI bloktan
ogrenilir, hedef bloga uygulanir. Uygulanan duzeltme hedef blokta yeniden
MERKEZLENIR: kuresel seviye zaten LB ile cozulmus (v83'te +0,02486), o
yuzden bir adayin kazanci yalnizca YAPIDAN gelmelidir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, bloklari_kur, mse, rapor, taban_r, tablo_yaz  # noqa: E402

# Blok konumundan (ufuk / ozet penceresi uzunlugu) gelen yapay eksenler.
# Bunlar CV'de blok kimligini ele veriyor, testte karsiligi yok -> YASAK.
YASAK = (
    "ufuk_gun",
    "ozet_pencere_gun",
    "t_son_kayit_yasi",
    "t_gy_gun",
    "tk_yil",
    "tanim_num",
    "ulusal_yil_once",
)


def _artik(b, r0: np.ndarray) -> np.ndarray:
    return b.lgy - (r0 + b.lgc)


def _merkezle(v: np.ndarray) -> np.ndarray:
    return v - float(np.mean(v))


def grup_ofseti(anahtar: str, *, n0: float = 200.0, kirp: float = 0.5):
    """Blok-disi ogrenilmis grup ofseti (James-Stein tipi buzmeli)."""

    def yap(bl, taban):
        cikti = {}
        for k in BLOKLAR:
            pay: dict[object, list[float]] = {}
            for j in BLOKLAR:
                if j == k:
                    continue
                b = bl[j]
                e = _merkezle(_artik(b, taban[j]))
                g = pd.Series(b.cerceve[anahtar].to_numpy())
                agg = pd.DataFrame({"e": e}).groupby(g)["e"].agg(["mean", "size"])
                for idx, satir in agg.iterrows():
                    pay.setdefault(idx, [0.0, 0.0])
                    pay[idx][0] += satir["mean"] * satir["size"]
                    pay[idx][1] += satir["size"]
            harita = {i: (s / n) * (n / (n + n0)) for i, (s, n) in pay.items()}
            b = bl[k]
            d = pd.Series(b.cerceve[anahtar].to_numpy()).map(harita).fillna(0.0).to_numpy()
            cikti[k] = np.clip(_merkezle(d), -kirp, kirp)
        return cikti

    return yap


def kesikli(v: np.ndarray, kenar: np.ndarray) -> np.ndarray:
    return np.digitize(v, kenar)


def kalibrasyon_egrisi(kova_sayisi: int = 40, *, n0: float = 200.0):
    """Tahmin edilen r'nin desiline gore blok-disi ofset (sekil kalibrasyonu)."""

    def yap(bl, taban):
        cikti = {}
        for k in BLOKLAR:
            # kenarlar HEDEF blogun tahmin dagilimindan (etiketsiz -> mesru)
            kenar = np.quantile(taban[k], np.linspace(0, 1, kova_sayisi + 1)[1:-1])
            pay: dict[int, list[float]] = {}
            for j in BLOKLAR:
                if j == k:
                    continue
                b = bl[j]
                e = _merkezle(_artik(b, taban[j]))
                idx = kesikli(taban[j], kenar)
                agg = pd.DataFrame({"e": e}).groupby(idx)["e"].agg(["mean", "size"])
                for i, satir in agg.iterrows():
                    pay.setdefault(int(i), [0.0, 0.0])
                    pay[int(i)][0] += satir["mean"] * satir["size"]
                    pay[int(i)][1] += satir["size"]
            harita = {i: (s / n) * (n / (n + n0)) for i, (s, n) in pay.items()}
            d = pd.Series(kesikli(taban[k], kenar)).map(harita).fillna(0.0).to_numpy()
            cikti[k] = _merkezle(d)
        return cikti

    return yap


def olcek(lam: float):
    """Kuresel genlik: r' = ort + lam*(r - ort)."""

    def aday(b, r0):
        m = float(r0.mean())
        return m + lam * (r0 - m)

    return aday


def trafo_buzme(beta: float):
    """Trafo-arasi bilesenin buzulmesi (sicak James-Stein)."""

    def aday(b, r0):
        t = pd.Series(b.cerceve["tanim"].to_numpy())
        s = pd.Series(r0)
        tort = s.groupby(t).transform("mean").to_numpy()
        m = float(r0.mean())
        return m + beta * (tort - m) + (r0 - tort)

    return aday


def yigin_ridge(kolonlar: list[str], *, alpha: float = 30.0, kirp: float = 0.6):
    """Blok-disi ogrenilmis DOGRUSAL artik yigini (standardize + ridge)."""

    def yap(bl, taban):
        cikti = {}
        for k in BLOKLAR:
            xs, ys = [], []
            for j in BLOKLAR:
                if j == k:
                    continue
                b = bl[j]
                xs.append(_x(b, taban[j], kolonlar))
                ys.append(_merkezle(_artik(b, taban[j])))
            X = np.vstack(xs)
            y = np.concatenate(ys)
            mu, sd = X.mean(0), X.std(0) + 1e-9
            Z = (X - mu) / sd
            A = Z.T @ Z + alpha * np.eye(Z.shape[1])
            w = np.linalg.solve(A, Z.T @ y)
            Xk = (_x(bl[k], taban[k], kolonlar) - mu) / sd
            cikti[k] = np.clip(_merkezle(Xk @ w), -kirp, kirp)
        return cikti

    return yap


def _x(b, r0: np.ndarray, kolonlar: list[str]) -> np.ndarray:
    c = b.cerceve
    sut = [r0]
    for k in kolonlar:
        v = pd.to_numeric(c[k], errors="coerce").to_numpy(dtype="float64")
        sut.append(np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    sut.append(r0 * r0)
    hg = c["hg"].to_numpy()
    sut.append((hg >= 5).astype("float64"))
    sut.append((hg == 6).astype("float64"))
    return np.column_stack(sut)


def yigin_lgbm(kolonlar: list[str], *, kirp: float = 0.6, yaprak: int = 31, agac: int = 250):
    """Blok-disi ogrenilmis LightGBM artik yigini."""

    def yap(bl, taban):
        import lightgbm as lgb

        cikti = {}
        for k in BLOKLAR:
            xs, ys = [], []
            for j in BLOKLAR:
                if j == k:
                    continue
                b = bl[j]
                xs.append(_x(b, taban[j], kolonlar))
                ys.append(_merkezle(_artik(b, taban[j])))
            X = np.vstack(xs)
            y = np.concatenate(ys)
            m = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=agac,
                learning_rate=0.05,
                num_leaves=yaprak,
                min_child_samples=200,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                reg_lambda=5.0,
                random_state=7,
                n_jobs=-1,
                verbose=-1,
            )
            m.fit(X, y)
            p = m.predict(_x(bl[k], taban[k], kolonlar))
            cikti[k] = np.clip(_merkezle(p), -kirp, kirp)
        return cikti

    return yap


def toplayici(uretici, kat: float = 1.0):
    """Onceden hesaplanmis duzeltme sozlugunu adaya cevirir."""
    onbellek: dict[int, dict] = {}

    def kur(bl, taban):
        anahtar = id(taban)
        if anahtar not in onbellek:
            onbellek[anahtar] = uretici(bl, taban)
        return onbellek[anahtar]

    def aday_yap(bl, taban):
        d = kur(bl, taban)

        def aday(b, r0):
            return r0 + kat * d[b.ad]

        return aday

    return aday_yap


def main() -> int:
    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}
    print("TABAN sicak MSE:", {k: round(mse(bl[k], taban[k]), 5) for k in BLOKLAR})

    ozet_kol = [
        "guc",
        "t_log_ort",
        "t_log_std",
        "t_log_medyan",
        "t_log_p10",
        "t_log_p90",
        "t_sifir_orani",
        "t_yuk_faktoru",
        "t_trend",
        "t_yayilma",
        "t_kayma",
        "t_hg_genligi",
        "t_log_son7",
        "t_log_son30",
        "t_log_son90",
        "t_kuyruk_sifir",
        "gecmis_gun",
        "t_doluluk",
        "p_doluluk",
        "hg",
        "sicaklik_ort",
        "cdd22_ort7",
        "tarim_orani",
        "yerlesim_orani",
        "guc_yuzdelik",
        "t_hg_sapma",
    ]
    ozet_kol = [k for k in ozet_kol if k in bl["yaz25"].cerceve.columns and k not in YASAK]
    print(f"yigin oznitelikleri ({len(ozet_kol)}):", ozet_kol)

    satirlar = []

    # --- A1..A5: blok-disi grup ofsetleri
    for ad, anah in (
        ("A1 hafta gunu ofseti", "hg"),
        ("A2 kVA kovasi ofseti", "kova"),
        ("A3 ilce ofseti", "ilce"),
    ):
        f = toplayici(grup_ofseti(anah))(bl, taban)
        satirlar.append(rapor(bl, f, ad, taban))

    # seviye desili + sifir orani kovalari
    for k in BLOKLAR:
        c = bl[k].cerceve
        c["seviye_d"] = pd.qcut(c["t_log_ort"].to_numpy(), 20, labels=False, duplicates="drop")
        c["sifir_k"] = np.digitize(
            np.nan_to_num(c["t_sifir_orani"].to_numpy(), nan=0.0),
            [0.001, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9],
        )
        c["gecmis_k"] = np.digitize(c["gecmis_gun"].to_numpy(), [7, 31, 91, 181, 366])
    for ad, anah in (
        ("A4 seviye desili (20) ofseti", "seviye_d"),
        ("A5 sifir orani kovasi ofseti", "sifir_k"),
        ("A6 gecmis uzunlugu ofseti", "gecmis_k"),
    ):
        f = toplayici(grup_ofseti(anah))(bl, taban)
        satirlar.append(rapor(bl, f, ad, taban))

    # --- A7: tahmin sekli kalibrasyonu
    f = toplayici(kalibrasyon_egrisi(40))(bl, taban)
    satirlar.append(rapor(bl, f, "A7 r-desili sekil kalibrasyonu", taban))

    # --- A8: kuresel olcek
    for lam in (0.94, 0.97, 1.03, 1.06):
        satirlar.append(rapor(bl, olcek(lam), f"A8 kuresel olcek lam={lam}", taban))

    # --- A9: trafo-arasi buzme
    for beta in (0.90, 0.95, 1.05):
        satirlar.append(rapor(bl, trafo_buzme(beta), f"A9 trafo-arasi buzme b={beta}", taban))

    # --- A10: dogrusal yigin
    for alpha in (30.0, 300.0):
        f = toplayici(yigin_ridge(ozet_kol, alpha=alpha))(bl, taban)
        satirlar.append(rapor(bl, f, f"A10 ridge yigin a={alpha:.0f}", taban))

    # --- A11: LightGBM yigin
    for yaprak, agac in ((15, 200), (63, 400)):
        f = toplayici(yigin_lgbm(ozet_kol, yaprak=yaprak, agac=agac))(bl, taban)
        satirlar.append(rapor(bl, f, f"A11 lgbm yigin yaprak={yaprak}", taban))

    tablo_yaz(satirlar)
    yol = Path(__file__).resolve().parent / "adaylar.jsonl"
    with yol.open("w", encoding="utf-8") as f:
        for s in satirlar:
            f.write(pd.Series(s).to_json() + "\n")
    print(f"\nyazildi: {yol}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
