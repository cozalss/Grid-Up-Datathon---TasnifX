"""ONARILMIS SOGUK OLCUT TEZGAHI.

KIRLILIK: CV blogu ``b`` icin egitim parcasi = diger bloklar. ``b``de SOGUK
sayilan trafolar o parcada ETIKETLERIYLE bulunuyor (yaz25 %97,2 / guz25 %97,7
satir); ``tanim_num`` (ayirt gucu 0,9985) maskelemeden sag ciktigi icin model
onlarin seviyesini EZBERLIYOR. kis26 ve TEST'te bu kanal %0.

ONARIM: blok ``b`` icin, ``b``de soguk olan trafolarin BUTUN satirlari egitim
parcasindan atilir. Ezber uc blokta da %0 olur -- TEST ile ayni.

Tahminler LOG uzayinda onbelleklenir: onbellek/{setup}_{blok}_{ayar}.npz
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import deney as d, deney_ileri as di, tuketim_model as tm  # noqa

BURA = Path(__file__).resolve().parent
ONBELLEK = BURA / "onbellek"
ONBELLEK.mkdir(parents=True, exist_ok=True)
BLOKLAR = ("yaz25", "guz25", "kis26")
TOHUMLAR = (1000, 1001, 1002)

#: ayar adi -> (aile, ustyazim)
AYARLAR: dict[str, tuple[str, dict]] = {
    "cat_d5": ("cat", {"depth": 5}),
    "cat_d6": ("cat", {"depth": 6}),
    "cat_d7": ("cat", {"depth": 7}),  # URETIM
    "cat_d8": ("cat", {"depth": 8}),
    "xgb": ("xgb", {}),
    "lgbm": ("lgbm", {}),
    "cat_d7_rs1": ("cat", {"depth": 7, "random_strength": 1.0}),
    "cat_d7_rs4": ("cat", {"depth": 7, "random_strength": 4.0}),
    "cat_d7_lr03": ("cat", {"depth": 7, "learning_rate": 0.03, "iterations": 420}),
    "cat_d7_l2_1": ("cat", {"depth": 7, "l2_leaf_reg": 1.0}),
    "cat_d6_rs4": ("cat", {"depth": 6, "random_strength": 4.0}),
    "cat_d7_lr03_rs4": (
        "cat",
        {"depth": 7, "learning_rate": 0.03, "iterations": 400, "random_strength": 4.0},
    ),
    "cat_d5_rs4": ("cat", {"depth": 5, "random_strength": 4.0}),
}

#: ayar adi -> cikarilacak kolonlar (kimlik kanali deneyleri)
CIKARIM: dict[str, tuple[str, ...]] = {
    "cat_d7_nokimlik": (
        "tanim_num",
        "tanim_uzunluk",
        "tanim_on2",
        "tanim_on3",
        "tanim_on4",
        "tanim_on5",
    ),
    "cat_d7_notanimnum": ("tanim_num",),
}
for _a in CIKARIM:
    AYARLAR[_a] = ("cat", {"depth": 7})

#: ek kokenli egitim seti kullanan ayarlar
EKKOKEN = {
    "cat_d7_ekkoken": ("cat", {"depth": 7}),
    "xgb_ekkoken": ("xgb", {}),
    "lgbm_ekkoken": ("lgbm", {}),
}
AYARLAR.update(EKKOKEN)

_veri = None


def veri():
    global _veri
    if _veri is None:
        ONB = KOK / "data" / "interim" / "deney"
        egitim = pd.read_parquet(ONB / "egitim.parquet")
        test = pd.read_parquet(ONB / "test.parquet")
        tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
        uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
        tm.kategorik_kodla(egitim, test)
        _veri = (egitim, uretim)
    return _veri


_ek = None


def parca(blok: str, setup: str, ek_koken: bool = False):
    """(egitim_parcasi, dogrulama, soguk_maske) dondurur."""
    global _ek
    egitim, _uretim = veri()
    dog = egitim[egitim["_blok"] == blok]
    if ek_koken:
        if _ek is None:
            import deney as _d

            e = _d._ek_kokenler_kur(False)
            tm.kategorik_kodla(egitim, e)  # AYNI seviyeler
            _ek = pd.concat([egitim, e[egitim.columns]], ignore_index=True)
        kalan = tm.kokenleri_ayikla(_ek, blok)
    else:
        kalan = egitim[egitim["_blok"] != blok]
    soguk_t = set(dog.loc[dog["soguk_mu"] == 1, tm.GRUP].unique())
    if setup == "onarilmis":
        kalan = kalan[~kalan[tm.GRUP].isin(soguk_t)]
    elif setup == "rastgele":
        # KONTROL: ayni SATIR sayisini, soguk OLMAYAN trafolardan rastgele at.
        # Onarimin etkisi "veri azaldi" mi yoksa "ezber kesildi" mi?
        hedef_satir = int(kalan[tm.GRUP].isin(soguk_t).sum())
        aday = np.array(sorted(set(kalan[tm.GRUP].unique()) - soguk_t))
        rng = np.random.default_rng(12345)
        rng.shuffle(aday)
        boy = kalan.groupby(tm.GRUP, observed=True).size()
        atilan, birikim = [], 0
        for t in aday:
            if birikim >= hedef_satir:
                break
            atilan.append(t)
            birikim += int(boy.get(t, 0))
        kalan = kalan[~kalan[tm.GRUP].isin(set(atilan))]
    elif setup != "kirli":
        raise ValueError(setup)
    return kalan, dog, (dog["soguk_mu"] == 1).to_numpy()


def meta(blok: str) -> pd.DataFrame:
    """Soguk dogrulama satirlarinin meta bilgisi (tahmin dizileriyle hizali)."""
    yol = ONBELLEK / f"meta_{blok}.parquet"
    if yol.exists():
        return pd.read_parquet(yol)
    _, dog, soguk = parca(blok, "kirli")
    m = dog.loc[soguk, [tm.GRUP, "tarih", "guc", tm.HEDEF]].reset_index(drop=True)
    m = m.rename(columns={tm.GRUP: "tanim", tm.HEDEF: "y"})
    m.to_parquet(yol, index=False)
    return m


def tahmin(setup: str, blok: str, ayar: str) -> np.ndarray:
    """(tohum, n_soguk) LOG uzayinda tahmin. Onbellekli."""
    yol = ONBELLEK / f"{setup}_{blok}_{ayar}.npy"
    if yol.exists():
        return np.load(yol)
    _, uretim = veri()
    kalan, dog, soguk = parca(blok, setup, ek_koken=ayar in EKKOKEN)
    aile, ust = AYARLAR[ayar]
    cik = set(CIKARIM.get(ayar, ()))
    if cik:
        uretim = [k for k in uretim if k not in cik]
    ciktilar = []
    for tohum in TOHUMLAR:
        t0 = time.time()
        maskeli = d.soguk_maskele(kalan, uretim, 1.00, tohum)
        lg = di.egit_tahmin(aile, maskeli, dog, uretim, tohum, **ust)
        del maskeli
        ciktilar.append(lg[soguk])
        print(
            f"    {setup:10} {blok:6} {ayar:12} tohum {tohum} ({time.time() - t0:.0f}s)", flush=True
        )
    a = np.asarray(ciktilar, dtype="float64")
    np.save(yol, a)
    return a


# ------------------------------------------------------------- olcut
def rmsle(y, lg):
    t = np.clip(np.expm1(lg), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(t) - np.log1p(y)) ** 2)))


def mse(y, lg):
    return rmsle(y, lg) ** 2


def kare_hatalar(y, lg):
    t = np.clip(np.expm1(lg), 0.0, None)
    return (np.log1p(t) - np.log1p(y)) ** 2


if __name__ == "__main__":
    hedef = sys.argv[1] if len(sys.argv) > 1 else "temel"
    if hedef == "temel":
        ayarlar = ["cat_d7", "cat_d5", "cat_d6", "cat_d8", "xgb", "lgbm"]
        setuplar = ["onarilmis", "kirli"]
    elif hedef == "kontrol":
        ayarlar = ["cat_d7", "lgbm"]
        setuplar = ["rastgele"]
    elif hedef == "ekkoken":
        ayarlar = ["cat_d7_ekkoken"]
        setuplar = ["onarilmis"]
    elif hedef == "kimlik":
        ayarlar = ["cat_d7_nokimlik"]
        setuplar = ["kirli", "onarilmis"]
    elif hedef == "ayar":
        ayarlar = ["cat_d7_rs4", "cat_d7_lr03_rs4", "cat_d7_l2_1", "cat_d6_rs4"]
        setuplar = ["onarilmis"]
    else:
        raise SystemExit(f"bilinmeyen hedef {hedef}")
    t0 = time.time()
    for setup in setuplar:
        for blok in BLOKLAR:
            if setup in ("kirli", "rastgele") and blok == "kis26":
                continue  # onarim kis26'da hicbir satir atmiyor -- ayni model
            for ayar in ayarlar:
                tahmin(setup, blok, ayar)
    print(f"TAMAM {(time.time() - t0) / 60:.1f} dk")
