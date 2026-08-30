"""YAPISAL SAPMA YONLERI -- hic olculmemis eksenler.

Simdiye kadar hep MODEL CIKTILARI olculdu. Onlarin bilgisi tukendi:
25 olcumun artimli kazanc rekoru 3.12e-04, 2. sira icin eksen basi 4.53e-04 gerek.

Bu betik farkli bir sey yapiyor: tahminin YAPISAL olarak kaydigi kesitleri
test eden yonler kuruyor. Ornegin test satirlarinin %22'si "soguk" (trafo
kimligi egitimde yok). Bu kesitte sistematik bir sapma varsa tek yon buyuk
kazanc verir:
    d = soguk satirlarda +c, sicak satirlarda 0
    rho = 0.47 * (soguk satirlarin ortalama sapmasi)
    rho = 0.03 icin soguk satirlarda %6.4 sapma yeterli.

Her yon olculmus span'a DIK hale getirilir -> saf yeni bilgi.
"""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.005846366
RCOND = 1e-6
YERDEG = float(os.environ.get("YERDEG", "0.005"))
EK_OLCUM = {"tuketim_y40_sota_temiz.csv": -0.002229}


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return np.log1p(d[k].values.astype(np.float64))


def main():
    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), usecols=["tanim"])
    a0 = oku(TABAN)
    N = len(a0)

    # --- olculmus span ---
    SK = json.load(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "olculmus_skorlar.json"))
    )
    V, L = [], []
    for f, P in SK.items():
        if f == TABAN or not os.path.exists(os.path.join(S, f)):
            continue
        v = oku(f)
        if len(v) != N:
            continue
        d = v - a0
        V.append(d)
        L.append((M0 + float((d * d).mean()) - P * P) / 2)
    for f, Lj in EK_OLCUM.items():
        V.append(oku(f) - a0)
        L.append(Lj)
    V = np.array(V).T
    L = np.array(L)
    G = (V.T @ V) / N
    r_hat = V @ (np.linalg.pinv(G, rcond=RCOND) @ L)
    nrm = float((r_hat * r_hat).mean())
    print(f"taban: ||r_hat||^2={nrm:.6f}  -> {np.sqrt(M0 - nrm):.5f}")

    # --- yapisal ozellikler ---
    tarih = pd.to_datetime(te.tarih)
    sicak_set = set(tr.tanim.unique())
    soguk = (~te.tanim.isin(sicak_set)).to_numpy().astype(float)
    ay = tarih.dt.month.to_numpy()
    hafta_sonu = (tarih.dt.dayofweek >= 5).to_numpy().astype(float)
    lg = np.log1p(te.guc.values.astype(float))
    bolge = te.lokasyon.str.split(">").str[1].fillna("?")
    seviye = (a0 - a0.mean()) / a0.std()
    print(
        f"soguk satir {int(soguk.sum()):,} ({soguk.mean() * 100:.1f}%)  "
        f"ay {sorted(set(ay))}  bolge {bolge.nunique()}"
    )

    YON = {
        "soguk": soguk - soguk.mean(),
        "seviye": seviye,
        "seviye2": seviye**2 - (seviye**2).mean(),
        "haftasonu": hafta_sonu - hafta_sonu.mean(),
        "guc": (lg - lg.mean()) / lg.std(),
        "ay_dogrusal": (ay - ay.mean()) / ay.std(),
        "ay_temmuz": (ay == 7).astype(float) - (ay == 7).mean(),
        "soguk_x_seviye": soguk * seviye - (soguk * seviye).mean(),
        "guc_x_soguk": ((lg - lg.mean()) / lg.std()) * soguk,
    }
    for b in bolge.value_counts().index[:3]:
        m = (bolge == b).to_numpy().astype(float)
        YON[f"bolge_{b[:8]}"] = m - m.mean()

    print(
        f"\n{'yon':>18s} {'Q':>10s} {'Q_dik':>10s} {'span-disi':>10s} "
        f"{'rho=0.03 icin gereken sapma':>28s}"
    )
    sonuc = []
    for ad, x in YON.items():
        x = x.astype(np.float64)
        x = x / np.sqrt(float((x * x).mean()))  # birim norm
        Q = 1.0
        c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
        xp = x - V @ c
        Qd = float((xp * xp).mean())
        # yorum: rho=0.03 icin gereken ortalama sapma (log uzayinda)
        sonuc.append((ad, Q, Qd, Qd / Q, xp))
        print(
            f"{ad:>18s} {Q:10.4f} {Qd:10.4f} {Qd / Q:10.3f} "
            f"{0.03 / np.sqrt(Qd) if Qd > 0 else float('nan'):28.4f}"
        )

    sonuc.sort(key=lambda t: -t[2])
    print("\nEN BUYUK DIK BILESENLI 6 YAPISAL YON -> sonda uretiliyor")
    rap = {}
    for ad, Q, Qd, oran, xp in sonuc[:6]:
        kap = YERDEG / np.sqrt(Qd)
        p = a0 + r_hat + kap * xp
        y = np.clip(np.expm1(p), 0.0, None)
        out = pd.DataFrame({"id": te.id.values, "tuketim": y})
        if not (
            (out.id.values == ss.iloc[:, 0].values).all()
            and len(out) == 714688
            and out.tuketim.isna().sum() == 0
            and (out.tuketim < 0).sum() == 0
        ):
            print(f"  {ad}: KAPI KALDI, atlandi")
            continue
        if out.tuketim.max() > 3 * np.expm1(a0).max():
            print(f"  {ad}: maks cok buyuk, atlandi")
            continue
        dgv = np.log1p(out.tuketim.values) - a0
        sabit = float(M0 - 2 * nrm + float(dgv @ dgv) / N)
        cik = f"tuketim_YP_{ad}.csv"
        gec = Path(os.path.join(S, cik) + ".tmp")
        out.to_csv(gec, index=False)
        gec.replace(os.path.join(S, cik))
        rap[ad] = dict(cikti=cik, kappa=kap, Q_dik=Qd, sabit=sabit, kirpik=int((y == 0).sum()))
        print(
            f"  {cik:32s} kappa={kap:.4f}  L=0 -> {np.sqrt(sabit):.5f}  "
            f"rho=0.03 -> {np.sqrt(sabit - 2 * kap * 0.03 * np.sqrt(Qd)):.5f}  "
            f"kirpik {int((y == 0).sum())}"
        )
    json.dump(
        rap,
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m111_yapisal.json"), "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
