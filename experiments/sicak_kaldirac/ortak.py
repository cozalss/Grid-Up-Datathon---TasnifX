"""SICAK KOHORT (G1+G2) KALDIRACI -- ortak veri kurulumu ve olcut.

Uc CV blogunun (yaz25 / guz25 / kis26) onbelleklenmis SICAK UZMAN
tahminlerini yukler. Onbellek ``scripts/aile_onbellegi.py`` tarafindan
URETIM ESLI uretildi:

  * maske 0,15, CatBoost ustyazim {random_strength 4, l2_leaf_reg 1, depth 6}
  * ek kokenli egitim seti, ``kokenleri_ayikla`` ile SIZINTISIZ
  * dosyalar YALNIZ sicak satirlari tasiyor

Uretim harmani ``REJIM_AYARLARI['sicak']['agirlik']``:
    cat 3,0 | xgb 1,0 | lgbm 1,0 | sinir_agi 1,4

URETIM SON ISLEM ZINCIRI (sicak tarafta), v83 dosyalarindan OLCULDU:

    v66_c1335 -> v67 : "olay" duzeltmesi, 1.832 sicak satir
    v80_b -> v80_opt : KUYRUK rejimi  +0,16640  (29.873 satir)
    v80_opt -> v83   : SICAK CEKIRDEK +0,02486  (526.446 satir)

ve bundan once gun ekseni genligi (``son_islem_gunolcek.py``, c=1,3301,
varsayilan hedef = SICAK satirlar).

TABAN SECIMI
------------
+0,02486 LB'den cozulmus KURESEL bir seviye kaymasidir. CV blogunda o
sabiti aynen uygulamak anlamsiz -- her blogun kendi yanliligi var. Bu
yuzden taban her blokta KURESEL OPTIMUM seviyeye kalibre edilir
(delta = ortalama artik). Boylece bir aday ancak YAPI ekleyerek
kazanabilir; kuresel seviye zaten LB ile cozulmus sayilir.

Olcut: SICAK satirlarda log1p uzayinda MSE. Genel (test) MSE'ye cevrim
carpani SICAK_PAY = 0,77841 (556.319 / 714.688).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

ONB = KOK / "data" / "interim" / "aile_onbellek"
BLOKLAR = ("yaz25", "guz25", "kis26")
TOHUMLAR = (1000, 1001, 1002)
AGIRLIK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}
SICAK_PAY = 556_319 / 714_688
KUYRUK_DELTA = 0.16640
C_GUN = 1.3301

BLOK_PENCERE = {
    "yaz25": ("2025-04-01", "2025-07-31"),
    "guz25": ("2025-08-01", "2025-11-30"),
    "kis26": ("2025-12-01", "2026-03-31"),
}


def _lokasyon_parcala(lok: pd.Series) -> tuple[pd.Series, pd.Series]:
    """``IL>...>ILCE`` -- parca sayisi DEGISKEN (Izmir 3, Manisa 2)."""
    p = lok.fillna("").astype(str).str.split(">")
    return p.str[0].str.strip(), p.str[-1].str.strip()


def kova(guc: np.ndarray) -> np.ndarray:
    kenar = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
    et = ["<=50", "100", "160", "250", "400", "630", "1000", ">1000"]
    return np.asarray(et)[np.digitize(guc, kenar) - 1]


@dataclass
class Blok:
    ad: str
    cerceve: pd.DataFrame
    y: np.ndarray
    lgy: np.ndarray
    lgc: np.ndarray
    ham: dict[str, np.ndarray]
    tohum_harman: list[np.ndarray]

    @property
    def n(self) -> int:
        return len(self.y)


def _egitim() -> pd.DataFrame:
    return pd.read_parquet(KOK / "data/interim/deney/egitim.parquet")


def _train_ham() -> pd.DataFrame:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih", "tuketim"],
        encoding="utf-8",
        dtype={"tanim": str},
    )
    tr["tarih"] = pd.to_datetime(tr["tarih"])
    return tr


def bloklari_kur() -> dict[str, Blok]:
    egitim = _egitim()
    trh = _train_ham()
    ilk_gun = trh.groupby("tanim")["tarih"].min()

    bloklar: dict[str, Blok] = {}
    for ad in BLOKLAR:
        dg = egitim[egitim["_blok"] == ad]
        sicak = (dg["soguk_mu"] == 0).to_numpy()
        dg = dg[sicak].reset_index(drop=True)

        harmanlar = []
        aile_top: dict[str, list[np.ndarray]] = {a: [] for a in AGIRLIK}
        pay = sum(AGIRLIK.values())
        for t in TOHUMLAR:
            s = np.zeros(len(dg), dtype="float64")
            for a, w in AGIRLIK.items():
                v = np.load(ONB / f"{ad}_{t}_{a}_uretim.npy").astype("float64")
                if len(v) != len(dg):
                    raise RuntimeError(f"{ad}/{t}/{a}: {len(v)} != {len(dg)}")
                aile_top[a].append(v)
                s += w * v
            harmanlar.append(s / pay)
        ham = {a: np.mean(v, axis=0) for a, v in aile_top.items()}

        y = np.clip(dg["tuketim"].to_numpy(dtype="float64"), 0.0, None)
        gercek = np.load(ONB / f"{ad}_gercek.npy").astype("float64")
        if not np.allclose(np.clip(gercek, 0, None), y):
            raise RuntimeError(f"{ad}: onbellek gercegi cerceveyle uyusmuyor")

        il, ilce = _lokasyon_parcala(dg["lokasyon"])
        tarih = pd.to_datetime(dg["tarih"])
        dg = dg.assign(
            il=il.to_numpy(),
            ilce=ilce.to_numpy(),
            kova=kova(dg["guc"].to_numpy()),
            ay=tarih.dt.month.to_numpy(),
            hg=tarih.dt.dayofweek.to_numpy(),
        )
        bas = pd.Timestamp(BLOK_PENCERE[ad][0])
        ig = dg["tanim"].map(ilk_gun)
        dg["gecmis_gun"] = (bas - ig).dt.days.to_numpy()
        dg["kuyruk"] = (dg["gecmis_gun"] <= 6).to_numpy()

        bloklar[ad] = Blok(
            ad=ad,
            cerceve=dg,
            y=y,
            lgy=np.log1p(y),
            lgc=np.log1p(dg["guc"].to_numpy(dtype="float64")),
            ham=ham,
            tohum_harman=harmanlar,
        )
    return bloklar


def gun_etkisi(tanim: np.ndarray, gun: np.ndarray, r: np.ndarray, tur: int = 8) -> pd.Series:
    """Trafo etkisi cikarilmis GUN sabit etkisi (dengesiz panel)."""
    s = pd.Series(r - r.mean())
    t = pd.Series(tanim)
    g = pd.Series(gun)
    for _ in range(tur):
        s = s - s.groupby(t).transform("mean")
        s = s - s.groupby(g).transform("mean")
    kal = pd.Series(r - r.mean()) - s
    return kal.groupby(g).mean()


def taban_r(b: Blok, *, gun_olcek: float = C_GUN, kuyruk: float = KUYRUK_DELTA) -> np.ndarray:
    """URETIM TABANI (v83 sicak zinciri) + blok kuresel seviye kalibrasyonu."""
    r = np.mean(b.tohum_harman, axis=0) - b.lgc
    tan = b.cerceve["tanim"].to_numpy()
    gun = b.cerceve["tarih"].to_numpy()
    be = gun_etkisi(tan, gun, r)
    etki = pd.Series(gun).map(be).to_numpy(dtype="float64")
    etki = etki - etki.mean()
    r = r + (gun_olcek - 1.0) * etki
    r = r + kuyruk * b.cerceve["kuyruk"].to_numpy(dtype="float64")
    return r + kuresel_delta(b, r)


def kuresel_delta(b: Blok, r: np.ndarray) -> float:
    """Kirpma altinda OPTIMUM kuresel seviye kaymasi (1 boyutlu arama).

    Kirpmasiz cozum analitik ortalama artiktir; kirpma altinda degil.
    Kaba tarama + incelme, 1e-4 cozunurlukte.
    """
    d0 = float((b.lgy - b.lgc - r).mean())
    en_iyi, en_iyi_m = d0, mse(b, r + d0)
    adim = 0.08
    for _ in range(6):
        for d in np.arange(en_iyi - 4 * adim, en_iyi + 4.001 * adim, adim):
            m = mse(b, r + float(d))
            if m < en_iyi_m:
                en_iyi, en_iyi_m = float(d), m
        adim /= 4.0
    return en_iyi


def mse(b: Blok, r: np.ndarray) -> float:
    """URETIM OLCUTU: tahmin ``np.clip(expm1(.), 0, None)`` ile kirpiliyor.

    Kirpma log uzayinda ``max(log_tahmin, 0)`` demektir ve ONEMSIZ DEGIL:
    guz25'te 1.000'den fazla satirda log tahmin negatife dusuyor ve kirpma
    tek basina o blokta -0,0068 MSE getiriyor. Kirpmasiz olcen bir tezgah
    uretimden ayrilir (bkz. docs/40 §3).
    """
    e = b.lgy - np.maximum(r + b.lgc, 0.0)
    return float((e * e).mean())


def rapor(bloklar: dict[str, Blok], aday, ad: str, taban: dict[str, np.ndarray]) -> dict:
    """Adayi uc blokta olcer. ``aday(b, r_taban) -> r_yeni``."""
    sat: dict = {"aday": ad}
    tn = td = 0.0
    for k in BLOKLAR:
        b = bloklar[k]
        r0 = taban[k]
        r1 = aday(b, r0)
        d = mse(b, r1) - mse(b, r0)
        sat[k] = d
        tn += b.n
        td += d * b.n
    sat["GENEL"] = td / tn
    sat["testMSE"] = sat["GENEL"] * SICAK_PAY
    sat["ayni_isaret"] = all(sat[k] < 0 for k in BLOKLAR) or all(sat[k] > 0 for k in BLOKLAR)
    return sat


def tablo_yaz(satirlar: list[dict]) -> None:
    print(
        f"\n{'aday':38}{'yaz25':>11}{'guz25':>11}{'kis26':>11}{'GENEL':>11}{'testdMSE':>11}  karar"
    )
    print("-" * 104)
    for s in satirlar:
        if not s["ayni_isaret"]:
            karar = "RED(ters isaret)"
        elif s["testMSE"] <= -0.002:
            karar = "KABUL"
        elif s["testMSE"] < 0:
            karar = "red(kucuk)"
        else:
            karar = "RED(zararli)"
        print(
            f"{s['aday'][:38]:38}{s['yaz25']:>+11.5f}{s['guz25']:>+11.5f}"
            f"{s['kis26']:>+11.5f}{s['GENEL']:>+11.5f}{s['testMSE']:>+11.5f}  {karar}"
        )
