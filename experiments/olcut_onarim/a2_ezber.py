"""A2: soguk ezberlenebilirlik -- mevcut hal, kolon cikarma, blok onarimi."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))
import tuketim_model as tm  # noqa

ONB = KOK / "data" / "interim" / "deney"
egitim = pd.read_parquet(ONB / "egitim.parquet")
test = pd.read_parquet(ONB / "test.parquet")
G = tm.GRUP

print("=" * 96)
print("1) EZBERLENEBILIRLIK -- soguk satirin trafosu EGITIM PARCASINDA var mi?")
print("=" * 96)
print(f"{'blok':8}{'soguk satir':>13}{'soguk trafo':>13}{'ezber satir%':>14}{'ezber trafo%':>14}")
ezber_trafolar = {}
for b in tm.BLOKLAR:
    dog = egitim[egitim["_blok"] == b.ad]
    kalan = egitim[egitim["_blok"] != b.ad]
    gorulen = set(kalan[G].unique())
    soguk = dog[dog["soguk_mu"] == 1]
    st = soguk[G].unique()
    ezb = soguk[G].isin(gorulen)
    ezber_trafolar[b.ad] = {t for t in st if t in gorulen}
    print(
        f"{b.ad:8}{len(soguk):>13,}{len(st):>13,}{100 * ezb.mean():>13.1f}%"
        f"{100 * np.mean([t in gorulen for t in st]):>13.1f}%"
    )
# TEST
gorulen_tum = set(egitim[G].unique())
ts = test[test["soguk_mu"] == 1]
tst = ts[G].unique()
print(
    f"{'TEST':8}{len(ts):>13,}{len(tst):>13,}"
    f"{100 * ts[G].isin(gorulen_tum).mean():>13.1f}%"
    f"{100 * np.mean([t in gorulen_tum for t in tst]):>13.1f}%"
)

print("\n" + "=" * 96)
print("2) ONARIM: blok b icin, b'de SOGUK olan trafolarin TUM satirlarini egitimden at")
print("=" * 96)
print(
    f"{'blok':8}{'egitim satir':>14}{'onarilmis':>12}{'atilan':>10}{'atilan%':>10}"
    f"{'ezber sonrasi':>15}"
)
for b in tm.BLOKLAR:
    dog = egitim[egitim["_blok"] == b.ad]
    kalan = egitim[egitim["_blok"] != b.ad]
    soguk_t = set(dog.loc[dog["soguk_mu"] == 1, G].unique())
    tut = ~kalan[G].isin(soguk_t)
    onar = kalan[tut]
    gor = set(onar[G].unique())
    soguk = dog[dog["soguk_mu"] == 1]
    print(
        f"{b.ad:8}{len(kalan):>14,}{len(onar):>12,}{len(kalan) - len(onar):>10,}"
        f"{100 * (1 - tut.mean()):>9.1f}%{100 * soguk[G].isin(gor).mean():>14.1f}%"
    )

print("\n" + "=" * 96)
print("3) KIMLIKSIZ KANAL: tanim_* cikarilirsa soguk trafo statik anahtarla ayirt edilir mi?")
print("=" * 96)
# statik trafo-sabit kolonlar (tanim_* haric): kac soguk trafo egitimde
# BIREBIR ayni anahtara sahip bir trafoyla eslesir?
STATIK = [
    "guc",
    "ilce_key",
    "il_key",
    "bolge",
    "guc_payi",
    "guc_yuzdelik",
    "guc_medyan_orani",
    "ilce_toplam_guc",
    "ilce_trafo_sayisi",
    "osm_direk",
    "osm_trafo",
    "osm_dagitim_hat_km",
    "yerlesim_orani",
]


def anahtar(df, kols):
    s = pd.Series("", index=df.index)
    for k in kols:
        v = df[k]
        if str(v.dtype) == "category":
            v = v.astype(str)
        s = s + "|" + v.astype(str)
    return s


for ad, kols in (
    ("tanim_num tek", ["tanim_num"]),
    (
        "tanim_* hepsi",
        ["tanim_num", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5", "tanim_uzunluk"],
    ),
    ("STATIK (tanimsiz)", STATIK),
    ("STATIK + tanim_on5", STATIK + ["tanim_on5"]),
):
    satir = []
    for b in tm.BLOKLAR:
        dog = egitim[egitim["_blok"] == b.ad]
        kalan = egitim[egitim["_blok"] != b.ad]
        soguk = dog[dog["soguk_mu"] == 1]
        st = soguk.drop_duplicates(G)
        ka = kalan.drop_duplicates(G)
        # egitimde gorulmeyen trafolarla sinirla (yani ezber DISI kanal)
        gor = set(ka[G])
        yeni = st[~st[G].isin(gor)]
        if len(yeni) == 0:
            satir.append(float("nan"))
            continue
        ha = set(anahtar(ka, kols))
        satir.append(100 * anahtar(yeni, kols).isin(ha).mean())
    print(
        f"  {ad:20} yeni-trafo anahtar carpismasi %: "
        + "  ".join(f"{b.ad}={v:.1f}" for b, v in zip(tm.BLOKLAR, satir))
    )
