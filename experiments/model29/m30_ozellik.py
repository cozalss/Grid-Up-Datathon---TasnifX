"""Ileri-pencere DOGRUDAN TAHMIN icin ozellik insasi.
Bir kesim tarihinde gecmisten ozellik cikar, hedef penceredeki her satiri etiketle.
Test'in yapisini birebir taklit eder: soguk trafolarda gecmis ozellikleri NaN kalir."""

import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TATIL = set(
    pd.to_datetime(
        [
            "2025-01-01",
            "2025-03-30",
            "2025-03-31",
            "2025-04-01",
            "2025-04-23",
            "2025-05-01",
            "2025-05-19",
            "2025-06-06",
            "2025-06-07",
            "2025-06-08",
            "2025-06-09",
            "2025-07-15",
            "2025-08-30",
            "2025-10-29",
            "2026-01-01",
            "2026-03-20",
            "2026-03-21",
            "2026-03-22",
            "2026-04-23",
            "2026-05-01",
            "2026-05-19",
            "2026-05-27",
            "2026-05-28",
            "2026-05-29",
            "2026-06-06",
            "2026-07-15",
            "2026-08-30",
            "2026-10-29",
        ]
    )
)


def yukle_ham():
    tr = pd.read_csv(
        os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
    )
    te = pd.read_csv(
        os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
    )
    for d in (tr, te):
        p = d.lokasyon.str.split(">")
        d["il"] = p.str[0]
        d["bolge"] = p.str[1]
        d["ilce"] = p.str[2]
        d["idnum"] = pd.to_numeric(d.tanim, errors="coerce")
    tr["ly"] = np.log1p(tr.tuketim)
    return tr, te


def gecmis_ozellik(gec, kesim):
    """Trafo duzeyinde gecmis ozetleri (kesim tarihinde bilinen her sey)."""
    kesim = pd.Timestamp(kesim)
    g = gec.groupby("tanim")
    f = pd.DataFrame(index=g.size().index)
    f["h_n"] = g.size()
    f["h_ilk"] = (kesim - g.tarih.min()).dt.days
    f["h_son"] = (kesim - g.tarih.max()).dt.days
    f["h_ort"] = g.ly.mean()
    f["h_med"] = g.ly.median()
    f["h_std"] = g.ly.std()
    f["h_sifir"] = g.tuketim.apply(lambda v: float((v == 0).mean()))
    for w in (7, 14, 28, 56, 91, 182):
        sub = gec[gec.tarih > kesim - pd.Timedelta(days=w)]
        sg = sub.groupby("tanim")
        f[f"h_ort{w}"] = sg.ly.mean()
        f[f"h_n{w}"] = sg.size()
        if w in (28, 91):
            f[f"h_sifir{w}"] = sg.tuketim.apply(lambda v: float((v == 0).mean()))
    # egim (son 90 gun, gun basina log degisim)
    s90 = gec[gec.tarih > kesim - pd.Timedelta(days=90)].copy()
    s90["x"] = (s90.tarih - kesim).dt.days.astype(float)

    def egim(d):
        if len(d) < 10 or d.x.std() == 0:
            return np.nan
        return float(np.cov(d.x, d.ly, bias=True)[0, 1] / d.x.var())

    f["h_egim"] = s90.groupby("tanim")[["x", "ly"]].apply(egim)
    # gecen yil ayni takvim penceresi (kesim+1 .. kesim+4ay, bir yil once)
    a = kesim - pd.DateOffset(years=1)
    b = a + pd.DateOffset(months=4)
    gy = gec[(gec.tarih > a) & (gec.tarih <= b)]
    if len(gy):
        f["h_gecenyil"] = gy.groupby("tanim").ly.mean()
        f["h_gecenyil_n"] = gy.groupby("tanim").size()
    else:
        f["h_gecenyil"] = np.nan
        f["h_gecenyil_n"] = np.nan
    return f


def grup_ozellik(gec):
    """Soguk trafolar icin geri dusus: grup duzeyinde seviyeler."""
    o = {}
    o["guc"] = gec.groupby("guc").ly.mean()
    o["ilce"] = gec.groupby("ilce").ly.mean()
    o["bolge"] = gec.groupby("bolge").ly.mean()
    o["guc_bolge"] = gec.groupby(["guc", "bolge"]).ly.mean()
    o["guc_ilce"] = gec.groupby(["guc", "ilce"]).ly.mean()
    o["genel"] = gec.ly.mean()
    # guc-grubu olu orani
    o["guc_sifir"] = gec.assign(z=(gec.tuketim == 0)).groupby("guc").z.mean()
    o["ilce_sifir"] = gec.assign(z=(gec.tuketim == 0)).groupby("ilce").z.mean()
    return o


def varlik_ozellik(hed, kesim):
    """Hedef penceredeki VARLIK deseni -- test zamaninda bilinir (sizinti degil)."""
    kesim = pd.Timestamp(kesim)
    g = hed.groupby("tanim")
    v = pd.DataFrame(index=g.size().index)
    v["v_n"] = g.size()
    v["v_ilk"] = (g.tarih.min() - kesim).dt.days
    v["v_son"] = (g.tarih.max() - kesim).dt.days
    v["v_aralik"] = v.v_son - v.v_ilk + 1
    v["v_yogunluk"] = v.v_n / v.v_aralik
    # ayni gun devreye giren trafo sayisi (toplu dalga gostergesi)
    dalga = v.groupby("v_ilk").size()
    v["v_dalga"] = v.v_ilk.map(dalga)
    return v


def kur(gec, hed, kesim, sicak_kume):
    """hed satirlarindan model matrisi uret."""
    hf = gecmis_ozellik(gec, kesim)
    gf = grup_ozellik(gec)
    vf = varlik_ozellik(hed, kesim)
    X = hed[["tanim", "guc", "il", "bolge", "ilce", "idnum", "tarih"]].copy()
    X = X.join(hf, on="tanim").join(vf, on="tanim")
    X["soguk"] = (~X.tanim.isin(sicak_kume)).astype(np.int8)
    X["g_guc"] = X.guc.map(gf["guc"])
    X["g_ilce"] = X.ilce.map(gf["ilce"])
    X["g_bolge"] = X.bolge.map(gf["bolge"])
    X["g_guc_bolge"] = (
        pd.MultiIndex.from_frame(X[["guc", "bolge"]]).map(gf["guc_bolge"]).to_numpy(dtype=float)
    )
    X["g_guc_ilce"] = (
        pd.MultiIndex.from_frame(X[["guc", "ilce"]]).map(gf["guc_ilce"]).to_numpy(dtype=float)
    )
    X["g_guc_sifir"] = X.guc.map(gf["guc_sifir"])
    X["g_ilce_sifir"] = X.ilce.map(gf["ilce_sifir"])
    X["log_guc"] = np.log(X.guc.clip(lower=1))
    t = X.tarih
    X["ufuk"] = (t - pd.Timestamp(kesim)).dt.days
    X["ay"] = t.dt.month
    X["hgun"] = t.dt.dayofweek
    X["hafta_sonu"] = (X.hgun >= 5).astype(np.int8)
    X["tatil"] = t.isin(TATIL).astype(np.int8)
    doy = t.dt.dayofyear
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["ayin_gunu"] = t.dt.day
    for c in ("il", "bolge", "ilce"):
        X[c] = X[c].astype("category")
    return X.drop(columns=["tanim", "tarih"])


OZELLIKLER = None


def ozellik_listesi(X):
    return [c for c in X.columns if c not in ("tuketim", "ly")]
