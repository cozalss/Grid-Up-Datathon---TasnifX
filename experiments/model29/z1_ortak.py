"""Ortak tezgah: rejim maskeleri, OLCULMUS-CURUK maske, capa, kirpma, kapi denetimi.

Yeni adaylarin hepsi ayni son islemden gecer ki OLCUMLERI karsilastirilabilir olsun:
  1) f = L_aday - L_m6                     (log uzayinda yon)
  2) f[CURUK] = 0                          (docs/52 s1: olu-trafo tezi LB'de curudu)
  3) f = clip(f, -c, +c)                   (kaldirac sinirlamasi -- y2_kirp ile ayni)
  4) rejim bazinda ortalama sifirlanir     (seviye yalniz LB'de olculur)
  5) kapi denetimi + yazim
"""

import os

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))
S = os.path.join(KOK, "submissions")


def yukle():
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
    tr["L"] = np.log1p(tr.tuketim)
    return tr, te


def maskeler(tr, te):
    """SOGUK / KUYRUK / CEKIRDEK rejimleri + CURUK (olculmus-curutulmus) satirlar."""
    ilk = tr.groupby("tanim").tarih.min()
    son = tr.groupby("tanim").tarih.max()
    mx = tr.groupby("tanim").tuketim.max()
    s28 = tr[tr.tarih > pd.Timestamp("2026-03-03")].groupby("tanim").tuketim.max()
    olu = (
        set(mx[mx == 0].index)
        | set(son[son < pd.Timestamp("2026-02-01")].index)
        | set(s28[s28 == 0].index)
    )
    curuk = te.tanim.isin(olu).to_numpy()
    soguk = (~te.tanim.isin(set(tr.tanim))).to_numpy()
    kuyruk = (~soguk) & (te.tanim.map(ilk) >= pd.Timestamp("2026-03-26")).to_numpy()
    cek = (~soguk) & (~kuyruk)
    return dict(soguk=soguk, kuyruk=kuyruk, cekirdek=cek, curuk=curuk, ilk=ilk)


def taban():
    return np.log1p(pd.read_csv(os.path.join(S, "tuketim_m6_ikiyon.csv")).tuketim.values)


def olcut(f):
    """Yon kalitesi olcutleri: Q, Pearson kurtoz, en kotu %1 satirin Q payi."""
    Q = float((f**2).mean())
    if Q <= 0:
        return dict(Q=0.0, kurtoz=float("nan"), en_kotu_yuzde1_pay=float("nan"))
    k = float((f**4).mean() / Q**2)
    a = np.sort(f**2)
    n1 = max(1, int(round(0.01 * len(f))))
    return dict(Q=Q, kurtoz=k, en_kotu_yuzde1_pay=float(a[-n1:].sum() / a.sum()))


def bitir(L_hat, te, msk, A6, dosya, kirp=2.0):
    """Cizgi: curuk temizligi -> kirpma -> rejim capasi -> kapi denetimi -> yazim."""
    L_hat = np.asarray(L_hat, dtype=float)
    kotu = ~np.isfinite(L_hat)
    L_hat = np.where(kotu, A6, L_hat)
    f0 = L_hat - A6
    rap = dict(dosya=dosya, finite_olmayan=int(kotu.sum()), ham=olcut(f0))
    rap["curuk_Q_payi"] = float((f0[msk["curuk"]] ** 2).sum() / max(1e-30, (f0**2).sum()))
    f = np.where(msk["curuk"], 0.0, f0)
    rap["temiz"] = olcut(f)
    if kirp is not None:
        f = np.clip(f, -kirp, kirp)
    rap["kirpma"] = kirp
    capa = {}
    for nm in ("soguk", "kuyruk", "cekirdek"):
        mm = msk[nm] & ~msk["curuk"]
        if mm.sum():
            d = float(f[mm].mean())
            f[mm] -= d
            capa[nm] = dict(satir=int(mm.sum()), kaydirma=-d)
    f[msk["curuk"]] = 0.0
    rap["capa"] = capa
    rap["son"] = olcut(f)
    y = np.clip(np.expm1(A6 + f), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    yol = os.path.join(S, dosya)
    out.to_csv(yol, index=False)
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    kapi = dict(
        satir=int(len(out)),
        id_birebir=bool(len(out) == len(ss) and (out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        baslik=list(out.columns) == ["id", "tuketim"],
    )
    assert kapi["satir"] == 714688 and kapi["id_birebir"] and not kapi["nan"]
    assert not kapi["negatif"] and kapi["baslik"]
    rap["kapi"] = kapi
    b = np.log1p(out.tuketim.values)
    rap["dagilim"] = dict(
        log_ort=float(b.mean()),
        log_std=float(b.std()),
        maks=float(out.tuketim.max()),
        alt1kwh=float((out.tuketim.values < 1).mean()),
    )
    print(
        f"  YAZILDI {dosya}  Q={rap['son']['Q']:.5f} kurtoz={rap['son']['kurtoz']:.1f} "
        f"%1pay={100 * rap['son']['en_kotu_yuzde1_pay']:.1f} curukpay={rap['curuk_Q_payi']:.3f} "
        f"kapi=OK",
        flush=True,
    )
    return rap
