"""Z3b -- SINIR AGI (varlik gomulu MLP). GBM'den FONKSIYON SINIFI olarak farkli.

Neden bu aday, hata haritasina gore umutlu:
  * GBM eksen-hizali basamak fonksiyonu kurar; SEVIYE GRADYANI ve GUC BANDI
    U SEKLI gibi PURUZSUZ, monoton olmayan egrilikleri basamaklarla taklit eder
    ve uclarda (en dusuk desil, <=50 kVA, >1000 kVA) sistematik olarak sasar --
    olculen hata deseni tam olarak budur.
  * MLP surekli ve global bir fonksiyon ogrenir; ilce/bolge gomuleri kesitsel
    benzerligi paylasimli bir uzayda tasir, dolayisiyla az veri goren
    (soguk, dusuk seviyeli) kohortta komsularindan odunc alir.
  * Kayip DOGRUDAN log uzayinda L2 -- yani metrigin ta kendisi (Huber degil),
    dolayisiyla carpik kosullu dagilimda ORTALAMAYA nisan alir.

Egitim kurulumu uretim hattiyla birebir ayni (ayni 10 kesim, ayni ozellikler,
ayni test matrisi) -- degisen tek sey OGRENICI. Boylece aday farki saf model
sinifi farkidir. Gurultu eklenmez; iki tohum ortalanir.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402
import z2_tezgah as T  # noqa: E402

TOHUMLAR = (7, 17)
DEVIR = 4
YIGIN = 16384
KAT = ["il", "bolge", "ilce", "ay", "hgun", "gb"]
torch.set_num_threads(14)


def hazirla(Xtr, Xte):
    """Kategorik -> tamsayi kod, sayisal -> saglam olcekleme + eksik gostergesi."""
    n1 = len(Xtr)
    X = pd.concat([Xtr, Xte], ignore_index=True)
    X["gb"] = pd.cut(np.log1p(X.guc), bins=[-1, 5.5, 6.3, 6.9, 7.4, 99], labels=False)
    kodlar, boyut = [], []
    for c in KAT:
        k = pd.Categorical(X[c].astype(str)).codes.astype(np.int64)
        kodlar.append(k)
        boyut.append(int(k.max()) + 1)
    C = np.stack(kodlar, axis=1)
    say = [c for c in X.columns if c not in KAT and X[c].dtype.name != "category"]
    V = X[say].to_numpy(dtype=np.float32)
    eksik = ~np.isfinite(V)
    med = np.nanmedian(np.where(eksik, np.nan, V), axis=0)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    V = np.where(eksik, med[None, :], V)
    q1, q3 = np.percentile(V, [25, 75], axis=0)
    olc = (q3 - q1).astype(np.float32)
    olc[olc < 1e-6] = 1.0
    V = np.clip((V - med[None, :]) / olc[None, :], -8.0, 8.0).astype(np.float32)
    V = np.c_[V, eksik.astype(np.float32)]
    return (
        (V[:n1], C[:n1]),
        (V[n1:], C[n1:]),
        boyut,
        say,
    )


class Ag(nn.Module):
    def __init__(self, nsay, boyut):
        super().__init__()
        gom = [min(16, (b + 1) // 2 + 1) for b in boyut]
        self.g = nn.ModuleList([nn.Embedding(b, d) for b, d in zip(boyut, gom)])
        gir = nsay + sum(gom)
        self.f = nn.Sequential(
            nn.Linear(gir, 384),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(384, 192),
            nn.ReLU(),
            nn.Linear(192, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, v, c):
        e = [g(c[:, i]) for i, g in enumerate(self.g)]
        return self.f(torch.cat([v] + e, dim=1)).squeeze(1)


def main():
    t0 = time.time()
    Xtr, ytr, Xte, tr, te = T.matrisler()
    tr["L"] = tr.ly
    msk = Z.maskeler(tr, te)
    A6 = Z.taban()
    (Vtr, Ctr), (Vte, Cte), boyut, say = hazirla(Xtr, Xte)
    del Xtr, Xte
    y = np.log1p(ytr).astype(np.float32)
    ym, ys = float(y.mean()), float(y.std())
    yn = (y - ym) / ys
    print(
        f"sayisal {Vtr.shape[1]} kolon, kategorik boyutlar {boyut}, "
        f"egitim {Vtr.shape[0]:,} ({time.time() - t0:.0f}s)",
        flush=True,
    )

    Vt = torch.from_numpy(Vtr)
    Ct = torch.from_numpy(Ctr)
    Yt = torch.from_numpy(yn)
    Vp = torch.from_numpy(Vte)
    Cp = torch.from_numpy(Cte)
    n = len(Yt)
    tahminler = []
    for s in TOHUMLAR:
        torch.manual_seed(s)
        np.random.seed(s)
        ag = Ag(Vtr.shape[1], boyut)
        opt = torch.optim.AdamW(ag.parameters(), lr=3e-3, weight_decay=1e-5)
        adim = DEVIR * ((n + YIGIN - 1) // YIGIN)
        plan = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=adim)
        kayip_fn = nn.MSELoss()
        for d in range(DEVIR):
            ag.train()
            perm = torch.randperm(n)
            top, adet = 0.0, 0
            for i in range(0, n, YIGIN):
                j = perm[i : i + YIGIN]
                opt.zero_grad()
                kayip = kayip_fn(ag(Vt[j], Ct[j]), Yt[j])
                kayip.backward()
                nn.utils.clip_grad_norm_(ag.parameters(), 5.0)
                opt.step()
                plan.step()
                top += float(kayip) * len(j)
                adet += len(j)
            print(
                f"  tohum {s} devir {d + 1}: MSE {top / adet:.5f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
        ag.eval()
        cik = []
        with torch.no_grad():
            for i in range(0, len(Vp), 65536):
                cik.append(ag(Vp[i : i + 65536], Cp[i : i + 65536]).numpy())
        tahminler.append(np.concatenate(cik) * ys + ym)
        print(f"  tohum {s} tahmin bitti ({time.time() - t0:.0f}s)", flush=True)

    L = np.clip(np.mean(tahminler, axis=0), 0.0, 14.0)
    tohum_std = float(np.std(tahminler, axis=0).mean())
    print(f"tohumlar arasi ortalama sapma {tohum_std:.4f}", flush=True)
    rap = Z.bitir(L, te, msk, A6, "tuketim_z3_sinir.csv", kirp=2.0)
    rap["parametreler"] = dict(
        devir=DEVIR, yigin=YIGIN, tohumlar=list(TOHUMLAR), tohum_std=tohum_std, kategorik=KAT
    )
    np.save(os.path.join(BURA, "z3_sinir_L.npy"), L)
    json.dump(rap, open(os.path.join(BURA, "z3_sinir.json"), "w"), indent=1)
    print(f"TAMAM ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
