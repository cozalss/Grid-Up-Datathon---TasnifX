"""YENI YON URETIMI -- ortak veri, olcut ve GEOMETRI.

AMAC
----
Olculmus 20 gonderimlik span'a DIK, yeni enerji tasiyan tahmin yonleri uretmek.
Prob kampanyasi var olan yonu olcer, YENI yon uretmez; buradaki is yon uretimidir.

Kaggle'a HICBIR SEY GONDERMEZ. Yalnizca yerel dosyalari okur.

OLCUT
-----
RMSLE, uretimle BIREBIR ayni kirpma ile:
    log1p(np.clip(np.expm1(log_tahmin), 0, None)) == np.maximum(log_tahmin, 0)
Kirpma onemsiz degil (docs/40 §3): guz25'te binden fazla satirda log tahmin
negatife dusuyor.

CV PROTOKOLU
------------
``data/interim/deney/egitim.parquet`` uretimle ayni uc blogu ve blok-disi
ozetleri tasiyor (``t_*`` ozetleri "egitimin tamami eksi hedef blogu"ndan).
Bir aday bir blokta olculurken o blok EGITIMDEN CIKARILIR. Test icin uc blogun
tamami egitim, ``test.parquet`` hedef olur -- uretimin protokolu.

GEOMETRI
--------
Span = 21 olculmus gonderimin v83'e gore 20 fark yonu (``gram2/g01_havuz.py``).
Bir aday yonu ``u`` icin:
    u_span = span'a izdusum,  u_dik = u - u_span,  q_perp = |u_dik|^2 / n
Mevcut dik envanteri (v90, v99, P1, P3, v96, ...) ile kosinus de olculur;
YENI enerji, envanterden de arindirildiktan sonra kalan paydir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent
ONB = CIK / "onbellek"
ONB.mkdir(exist_ok=True)

sys.path.insert(0, str(KOK / "experiments" / "gram2"))

BLOKLAR = ("yaz25", "guz25", "kis26")

#: docs/52 §2 -- uretim modelinin blok bazli RMSLE'si (sicak / soguk ayri).
URETIM_CV = {
    "yaz25": (0.81224, 1.4359),
    "guz25": (0.83436, 1.6082),
    "kis26": (0.77826, 1.9061),
}

#: Ozellik seti -- uretimin 157 kolonundan yalin bir alt kume.
SAYISAL = [
    "guc",
    "sicaklik_ort",
    "sicaklik_max",
    "sicaklik_min",
    "hissedilen_max",
    "yagis_toplam",
    "ruzgar_max",
    "gunes_radyasyon",
    "nem_ort",
    "cdd22",
    "cdd24",
    "cdd22_ort7",
    "sicaklik_ort_ort7",
    "gun_uzunlugu_saat",
    "tk_ay",
    "tk_gun",
    "tk_haftanin_gunu",
    "tk_yilin_gunu",
    "tk_hafta_sonu",
    "tatil_mi",
    "tatil_mesafe",
    "yas",
    "ufuk_gun",
    "ilce_trafo_sayisi",
    "ilce_guc_medyan",
    "nufus",
    "ilce_nufus_yogunlugu",
    "guc_yuzdelik",
    "guc_payi",
    "guc_medyan_orani",
    "ulusal_gunluk",
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
    "t_son_kayit_yasi",
    "t_gy_log_ort",
    "t_gy_sifir_orani",
    "t_yayilma",
    "t_kayma",
    "t_hg_genligi",
    "t_log_son7",
    "t_log_son30",
    "t_log_son90",
    "t_son90_gun",
    "t_mevsim_genlik",
    "t_hg_sapma",
    "t_ay_sapma",
    "t_egim_cdd22",
    "t_doluluk",
    "g_guc_kova",
    "g_kova_log_ort",
    "g_ilce_log_ort",
    "g_ilce_kova_ort",
    "g_ilce_kova_n",
    "gp_ilce_ay",
    "gp_ilce_hg",
    "gp_kova_ay",
    "p_gun_sayisi",
    "p_doluluk",
]
KATEGORIK = ["il_key", "ilce_key", "bolge"]

#: Klasik (agac-disi) adaylar icin yeterli olan yalin kolonlar.
YALIN = [
    "tanim",
    "guc",
    "tarih",
    "tuketim",
    "il_key",
    "ilce_key",
    "soguk_mu",
    "g_guc_kova",
    "ufuk_gun",
    "_blok",
]
YALIN_TEST = [c for c in YALIN if c not in ("tuketim", "_blok")] + ["id"]


# ----------------------------------------------------------------------------- veri
_ONBELLEK: dict[str, object] = {}


def egitim(kolonlar: list[str] | None = None) -> pd.DataFrame:
    anahtar = "egitim:" + (",".join(sorted(kolonlar)) if kolonlar else "*")
    if anahtar not in _ONBELLEK:
        _ONBELLEK[anahtar] = pd.read_parquet(
            KOK / "data/interim/deney/egitim.parquet", columns=kolonlar
        )
    return _ONBELLEK[anahtar]  # type: ignore[return-value]


def test(kolonlar: list[str] | None = None) -> pd.DataFrame:
    anahtar = "test:" + (",".join(sorted(kolonlar)) if kolonlar else "*")
    if anahtar not in _ONBELLEK:
        _ONBELLEK[anahtar] = pd.read_parquet(
            KOK / "data/interim/deney/test.parquet", columns=kolonlar
        )
    return _ONBELLEK[anahtar]  # type: ignore[return-value]


def ham_train() -> pd.DataFrame:
    """Ham gunluk panel -- klasik zaman serisi adaylari icin."""
    if "ham" not in _ONBELLEK:
        d = pd.read_csv(
            KOK / "data/raw/train.csv",
            usecols=["tanim", "guc", "tarih", "tuketim"],
            dtype={"tanim": str},
        )
        d["tarih"] = pd.to_datetime(d["tarih"])
        d["lg"] = np.log1p(np.clip(d["tuketim"].to_numpy("float64"), 0.0, None))
        _ONBELLEK["ham"] = d
    return _ONBELLEK["ham"]  # type: ignore[return-value]


# ----------------------------------------------------------------------------- olcut
def rmsle_log(lgy: np.ndarray, log_tahmin: np.ndarray) -> float:
    """Uretim kirpmasi dahil RMSLE. ``log_tahmin`` log1p uzayinda."""
    e = lgy - np.maximum(log_tahmin, 0.0)
    return float(np.sqrt(np.mean(e * e)))


def rmsle(y_kwh: np.ndarray, tahmin_kwh: np.ndarray) -> float:
    lgy = np.log1p(np.clip(np.asarray(y_kwh, dtype="float64"), 0.0, None))
    lgp = np.log1p(np.clip(np.asarray(tahmin_kwh, dtype="float64"), 0.0, None))
    e = lgy - lgp
    return float(np.sqrt(np.mean(e * e)))


def uretim_referansi() -> dict[str, float]:
    """Uretim modelinin blok bazli havuzlanmis RMSLE'si (docs/52 §2 sayilari).

    Kapi esigi bunun IKI KATI: aday bunun altinda kalirsa "felaket degil".
    """
    e = egitim(["_blok", "soguk_mu"])
    ref = {}
    for b in BLOKLAR:
        m = e["_blok"].to_numpy() == b
        sg = e.loc[m, "soguk_mu"].to_numpy().astype(bool)
        h, c = URETIM_CV[b]
        n = int(m.sum())
        ref[b] = float(np.sqrt(((~sg).sum() * h * h + sg.sum() * c * c) / n))
    return ref


# ----------------------------------------------------------------------------- geometri
class Geometri:
    """Span izdusumu ve dik envanter."""

    def __init__(self) -> None:
        from g01_havuz import yukle  # type: ignore[import-not-found]

        adlar, X, skorlar, ids = yukle()
        self.adlar = adlar
        self.ids = ids
        self.n = X.shape[1]
        i83 = adlar.index("v83")
        self.m0 = float(skorlar[i83] ** 2)
        self.v83 = X[i83].copy()
        self.v102 = X[adlar.index("v102")].copy()
        idx = [k for k in range(len(adlar)) if k != i83]
        self.D = X[idx] - self.v83
        G = (self.D @ self.D.T) / self.n
        mm = skorlar[idx] ** 2
        self.b = (self.m0 + np.diag(G) - mm) / 2.0
        lam, V = np.linalg.eigh(G)
        kes = lam > lam.max() * 1e-10
        self.Vk, self.lk = V[:, kes], lam[kes]
        self._E: list[np.ndarray] | None = None
        self._E_ad: list[str] = []

    # --- izdusum
    def perp(self, u: np.ndarray) -> tuple[np.ndarray, float]:
        """(span'a dik bilesen, span'in ongordugu L)."""
        c = (self.D @ u) / self.n
        al = self.Vk @ ((self.Vk.T @ c) / self.lk)
        return u - al @ self.D, float(al @ self.b)

    def q(self, u: np.ndarray) -> float:
        return float(u @ u) / self.n

    # --- mevcut dik envanter (g07_butce2.py ile ayni acgozlu sira)
    def envanter(self) -> tuple[list[np.ndarray], list[str]]:
        if self._E is not None:
            return self._E, self._E_ad
        N = {
            "v93": "tuketim_v93_gram_optimum.csv",
            "v90": "tuketim_v90_temiz_sota.csv",
            "P1": "tuketim_p1_sicak_ilce.csv",
            "P2": "tuketim_p2_sicak_seviye.csv",
            "P3": "tuketim_p3_soguk_seviye.csv",
            "P4": "tuketim_p4_sicak_ay.csv",
            "P5": "tuketim_p5_soguk_kva.csv",
            "yas": "tuketim_prob_yas790.csv",
            "v82": "tuketim_v82_ayirici.csv",
            "v99": "tuketim_v99_mimari_sekil.csv",
            "B": "tuketim_v96_grupb_optimum.csv",
            "bos": "tuketim_v94_bosluk_oncesi.csv",
        }
        F = {k: log1p_gonderim(v) for k, v in N.items()}
        sira = [
            ("P1", F["P1"] - F["v93"]),
            ("P3", F["P3"] - F["v93"]),
            ("v96", F["B"] - F["v93"]),
            ("bos", F["bos"] - F["v93"]),
            ("v90", F["v90"] - self.v83),
            ("P2", F["P2"] - F["v93"]),
            ("yas", F["yas"] - F["v93"]),
            ("v99", F["v99"] - F["v90"]),
            ("P4", F["P4"] - F["v93"]),
            ("v82", F["v82"] - self.v83),
            ("P5", F["P5"] - F["v93"]),
        ]
        E: list[np.ndarray] = []
        ad: list[str] = []
        for a, u in sira:
            up, _ = self.perp(u)
            v = up.copy()
            for e in E:
                v -= (float(v @ e) / self.n) * e
            qy = float(v @ v) / self.n
            if qy > 5e-6:  # gorev tanimindaki 9 boyutluk envanteri verir
                E.append(v / np.sqrt(qy))
                ad.append(a)
        self._E, self._E_ad = E, ad
        return E, ad

    def olc(self, ad: str, u: np.ndarray) -> dict:
        """Bir aday yonunun geometrik karnesi."""
        Q = self.q(u)
        up, L_span = self.perp(u)
        Qp = self.q(up)
        E, E_ad = self.envanter()
        kos = {}
        v = up.copy()
        for e, a in zip(E, E_ad, strict=True):
            ic = float(up @ e) / self.n
            kos[a] = ic / np.sqrt(Qp) if Qp > 0 else 0.0
            v -= (float(v @ e) / self.n) * e
        Qy = self.q(v)
        maks = max(kos.items(), key=lambda kv: abs(kv[1])) if kos else ("-", 0.0)
        return {
            "ad": ad,
            "Q": Q,
            "q_perp": Qp,
            "span_pay": 1.0 - Qp / Q if Q > 0 else 0.0,
            "q_yeni": Qy,
            "maks_kos_ad": maks[0],
            "maks_kos": maks[1],
            "kos": kos,
        }


def log1p_gonderim(dosya: str) -> np.ndarray:
    d = pd.read_csv(GON / dosya)
    return np.log1p(np.clip(d["tuketim"].to_numpy("float64"), 0.0, None))


_GEO: Geometri | None = None


def geo() -> Geometri:
    global _GEO
    if _GEO is None:
        _GEO = Geometri()
    return _GEO


# ----------------------------------------------------------------------------- aday kaydi
def kaydet(ad: str, cv: dict[str, np.ndarray], test_tahmin: np.ndarray) -> None:
    np.savez_compressed(
        ONB / f"{ad}.npz",
        test=test_tahmin.astype("float32"),
        **{f"cv_{k}": v.astype("float32") for k, v in cv.items()},
    )


def yukle_aday(ad: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    z = np.load(ONB / f"{ad}.npz")
    cv = {b: z[f"cv_{b}"].astype("float64") for b in BLOKLAR if f"cv_{b}" in z}
    return cv, z["test"].astype("float64")


def var_mi(ad: str) -> bool:
    return (ONB / f"{ad}.npz").exists()
