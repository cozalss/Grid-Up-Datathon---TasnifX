"""Aday tahmin dosyalarini m6 tabanina gore olcer.

SECIM OLCUTU: aci / diklik. Q yalnizca sayisal hassasiyet icin raporlanir.
Olculmus span = {m4 - v102, m6 - v102}. Aday yonunun bu span'a DIK bileseni
gercekten yeni bilgidir; span icinde kalan bileseni ikinci kez sayilmis olur.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
S = os.path.join(KOK, "submissions")
BURA = os.path.dirname(os.path.abspath(__file__))
M0 = 1.005688  # = 1.00284^2, m6'nin LB MSE'si
R_KAL = 0.0640  # m4'ten kalibre L/sqrt(Q) -- yon KALITESI

_TE = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"),
    dtype={"tanim": str},
    parse_dates=["tarih"],
    usecols=["id", "tanim", "tarih"],
)
IDS = _TE.id.values
N = len(IDS)
_TR = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"),
    dtype={"tanim": str},
    parse_dates=["tarih"],
    usecols=["tanim", "tarih"],
)
_ILK = _TR.groupby("tanim").tarih.min()
SOGUK = (~_TE.tanim.isin(set(_TR.tanim))).to_numpy()
KUYRUK = (~SOGUK) & (_TE.tanim.map(_ILK) >= pd.Timestamp("2026-03-26")).to_numpy()
CEK = (~SOGUK) & (~KUYRUK)
REJIM = [("soguk", SOGUK), ("kuyruk", KUYRUK), ("cekirdek", CEK)]


def logoku(ad):
    return np.log1p(pd.read_csv(os.path.join(S, ad)).tuketim.values)


A6 = logoku("tuketim_m6_ikiyon.csv")
V102 = logoku("tuketim_v102_kappa_optimum.csv")
D4 = logoku("tuketim_m4_hava_capali.csv") - V102
D6 = A6 - V102


# span{D4, D6} icin ortonormal taban (Gram-Schmidt)
def _nrm(v):
    return v / np.sqrt((v**2).mean())


E1 = _nrm(D4)
_e2 = D6 - (D6 * E1).mean() * E1
E2 = _nrm(_e2)
BAZ = [E1, E2]


def olc(ad):
    d = pd.read_csv(os.path.join(S, ad))
    kapi = dict(
        satir=int(len(d)),
        id_birebir=bool(len(d) == N and (d.id.values == IDS).all()),
        nan=int(d.tuketim.isna().sum()),
        negatif=int((d.tuketim < 0).sum()),
        baslik=list(d.columns) == ["id", "tuketim"],
    )
    b = np.log1p(d.tuketim.values)
    f = b - A6
    Q = float((f**2).mean())
    if Q <= 0:
        f = np.ones(N) * 1e-12
        Q = float((f**2).mean())
    kos4 = float((f * E1).mean() / np.sqrt(Q))
    kosD6 = float((f * E2).mean() / np.sqrt(Q))
    dik = f - (f * E1).mean() * E1 - (f * E2).mean() * E2
    Q_yeni = float((dik**2).mean())
    yeni_pay = Q_yeni / Q
    rej = {nm: float((f[m] ** 2).sum() / (f**2).sum()) for nm, m in REJIM}
    return dict(
        dosya=ad,
        Q=Q,
        Q_rejim_payi=rej,
        maks_fark=float(np.abs(f).max()),
        p999_fark=float(np.quantile(np.abs(f), 0.999)),
        Q_yeni=Q_yeni,
        yeni_pay=yeni_pay,
        kos_m4=kos4,
        kos_m6dik=kosD6,
        span_ortusme=float(np.sqrt(max(0.0, 1 - yeni_pay))),
        korelasyon=float(np.corrcoef(b, A6)[0, 1]),
        kalibre_kazanc=R_KAL**2,
        kalibre_kazanc_yeni=R_KAL**2 * yeni_pay,
        basabas_skor=float(np.sqrt(M0 + Q)),
        log_ort=float(b.mean()),
        log_std=float(b.std()),
        maks=float(d.tuketim.max()),
        alt1kwh_orani=float((d.tuketim.values < 1).mean()),
        kapi=kapi,
    )


def yaz(r):
    k = r["kapi"]
    ok = k["id_birebir"] and k["nan"] == 0 and k["negatif"] == 0 and k["baslik"]
    print(
        f"{r['dosya']:42s} Q={r['Q']:.5f} Qyeni={r['Q_yeni']:.5f} "
        f"yeni%={100 * r['yeni_pay']:5.1f} kos_m4={r['kos_m4']:+.3f} "
        f"kos_m6d={r['kos_m6dik']:+.3f} kor={r['korelasyon']:.5f} "
        f"logort={r['log_ort']:.4f} std={r['log_std']:.4f} <1kWh={r['alt1kwh_orani']:.4f} "
        f"kapi={'OK' if ok else 'HATA'} "
        f"Qpay(s/k/c)={r['Q_rejim_payi']['soguk']:.2f}/{r['Q_rejim_payi']['kuyruk']:.2f}/{r['Q_rejim_payi']['cekirdek']:.2f} "
        f"p999={r['p999_fark']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    hedefler = sys.argv[1:] or sorted(x for x in os.listdir(S) if x.endswith(".csv"))
    r6 = olc("tuketim_m6_ikiyon.csv")
    print(
        f"REF m6: logort={r6['log_ort']:.4f} std={r6['log_std']:.4f} "
        f"maks={r6['maks']:.0f} <1kWh={r6['alt1kwh_orani']:.4f}"
    )
    sonuc = []
    for ad in hedefler:
        try:
            r = olc(ad)
        except Exception as e:  # noqa: BLE001
            print(f"  ATLANDI {ad}: {e}")
            continue
        sonuc.append(r)
        yaz(r)
    json.dump(sonuc, open(os.path.join(BURA, "y1_olcum_ham.json"), "w"), indent=1)
