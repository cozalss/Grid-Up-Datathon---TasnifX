"""Z2 -- ANALOG (en yakin komsu) profil aktarimi. HIC MODEL UYDURULMAZ.

Hata haritasi: gorulebilen hatanin %22 kadari EN DUSUK SEVIYE desilinde,
%21,3 kadari 50 kVA trafolarda (cogu SOGUK), ve uc guc bantlarinda sistematik
fazla tahmin var. Bu kohortta parametrik model (GBM ya da ayristirma) ortalamaya
cekiyor. Analog yaklasim tam tersini yapar: hicbir sey OGRENMEZ, sadece
"bu trafoya en cok benzeyen trafolar gecen yil ayni gunlerde ne yapti"
sorusunu sorar ve o yorungeyi AYNEN aktarir.

Tasarim -- donor ve alici AYNI GORELI KONUMDA olculur:
  * alici  : kesim 2026-03-31, ozellikler kesim oncesinden, hedef 2026-04..07
  * donor  : kesim 2025-04-01, ozellikler kesim oncesinden, yorunge 2025-04..08
  * eslesme 364 GUN (52 hafta) kaydirmali -> haftanin gunu korunur

SICAK alici : yorunge = donorun kendi referans seviyesinden SAPMASI,
              alicinin kendi seviyesine bindirilir.
SOGUK alici : donor havuzu = egitimde 2025-02 sonrasi DOGAN trafolar,
              aktarilan sey YASA gore kapasite kullanim orani u(yas) =
              L - log1p(guc). Yani yeni baglantinin devreye girme yorungesi
              medyan degil, BENZERLIK AGIRLIKLI komsulardan gelir.

Uzaklik: standartlastirilmis ozellik uzayinda Oklid + ilce/bolge cezasi;
agirlik Gauss cekirdegi (K=30). Hicbir yerde gurultu yoktur.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402

LAG = pd.Timedelta(days=364)
K = 30
KES_TE = pd.Timestamp("2026-03-31")
KES_DN = KES_TE - LAG  # 2025-04-01
UFUK = 122

t0 = time.time()
tr, te = Z.yukle()
msk = Z.maskeler(tr, te)
A6 = Z.taban()
ILK = msk["ilk"]
print(f"yuklendi ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------- ozellik cikarimi
def ozellik(gec, kesim):
    """Bir KESIM tarihinde, kesim oncesi gecmisten trafo duzeyi ozellikler."""
    kesim = pd.Timestamp(kesim)
    g = gec.groupby("tanim")
    f = pd.DataFrame(index=g.size().index)
    f["n"] = g.size()
    f["yas"] = np.log1p((kesim - g.tarih.min()).dt.days)
    f["guc"] = np.log1p(g.guc.first())
    s45 = gec[gec.tarih > kesim - pd.Timedelta(days=45)]
    s91 = gec[gec.tarih > kesim - pd.Timedelta(days=91)]
    onc = gec[gec.tarih <= kesim - pd.Timedelta(days=45)]
    f["ref"] = s45.groupby("tanim").L.mean()
    f["ref"] = f.ref.fillna(g.L.mean())
    f["u"] = f.ref - f.guc
    f["std"] = s91.groupby("tanim").L.std()
    f["std"] = f["std"].fillna(0.0)
    z91 = s91.assign(z=(s91.tuketim < 1).astype(float))
    f["sifir"] = z91.groupby("tanim").z.mean()
    f["sifir"] = f["sifir"].fillna(0.0)
    hs = s91.assign(w=(s91.tarih.dt.dayofweek >= 5))
    f["hsonu"] = hs[hs.w].groupby("tanim").L.mean() - hs[~hs.w].groupby("tanim").L.mean()
    f["hsonu"] = f["hsonu"].fillna(0.0)
    f["egim"] = s45.groupby("tanim").L.mean() - onc.groupby("tanim").L.mean()
    f["egim"] = f["egim"].fillna(0.0)
    f["ilce"] = g.ilce.first()
    f["bolge"] = g.bolge.first()
    return f


F_AL = ozellik(tr, KES_TE)
F_DN = ozellik(tr[tr.tarih <= KES_DN], KES_DN)
print(
    f"ozellikler: alici {len(F_AL)}, donor havuzu {len(F_DN)} ({time.time() - t0:.0f}s)", flush=True
)

# --------------------------------------------------------------- donor yorungeleri
KAY0 = KES_DN + pd.Timedelta(days=1)
KAY1 = KES_DN + pd.Timedelta(days=UFUK)
KAY = tr[(tr.tarih >= KAY0) & (tr.tarih <= KAY1)].copy()
KAY["tau"] = (KAY.tarih - KES_DN).dt.days - 1
YOR = KAY.pivot_table(index="tanim", columns="tau", values="L", aggfunc="mean")
YOR = YOR.reindex(columns=range(UFUK))
kapsam = YOR.notna().sum(axis=1)
UYGUN = kapsam[kapsam >= 90].index
print(f"yorunge matrisi {YOR.shape}, >=90 gun kapsayan donor {len(UYGUN)}", flush=True)

D_SICAK = F_DN.index[F_DN.n >= 60].intersection(UYGUN)
Y_SICAK = YOR.loc[D_SICAK]
REF_D = F_DN.loc[D_SICAK, "ref"].to_numpy(dtype=float)[:, None]
DELTA = Y_SICAK.to_numpy(dtype=float) - REF_D
sat_ort = np.nanmean(DELTA, axis=1, keepdims=True)
DELTA = np.where(np.isfinite(DELTA), DELTA, sat_ort)
DELTA = np.nan_to_num(DELTA)
print(f"SICAK donor {len(D_SICAK)}, delta {DELTA.shape}", flush=True)

# SOGUK donorlar: 2025-02-01 sonrasi dogmus trafolar, YAS eksenli u(yas)
YENI = ILK[pd.Timestamp("2025-02-01") <= ILK].index
kh = tr[tr.tanim.isin(YENI)].copy()
kh["yas"] = (kh.tarih - kh.tanim.map(ILK)).dt.days
kh["u"] = kh.L - np.log1p(kh.guc)
kh = kh[kh.yas < 190]
UY = kh.pivot_table(index="tanim", columns="yas", values="u", aggfunc="mean")
UY = UY.reindex(columns=range(190))
kap2 = UY.notna().sum(axis=1)
D_SOGUK = kap2[kap2 >= 60].index
UY = UY.loc[D_SOGUK]
UYA = UY.to_numpy(dtype=float)
med_yas = np.nanmedian(UYA, axis=0)
med_yas = np.where(np.isfinite(med_yas), med_yas, float(np.nanmedian(UYA)))
UYA = np.where(np.isfinite(UYA), UYA, med_yas[None, :])
G_SOGUK = pd.DataFrame(index=D_SOGUK)
G_SOGUK["guc"] = np.log1p(kh.groupby("tanim").guc.first())
G_SOGUK["ilce"] = kh.groupby("tanim").ilce.first()
G_SOGUK["bolge"] = kh.groupby("tanim").bolge.first()
G_SOGUK["dogum_ay"] = kh.groupby("tanim").tarih.min().dt.month
print(f"SOGUK donor {len(D_SOGUK)}, u(yas) {UYA.shape} ({time.time() - t0:.0f}s)", flush=True)


# --------------------------------------------------------------- komsuluk
def komsu(Xa, Xd, ceza_a, ceza_d, k=K):
    """standartlastirilmis uzaklik + kategorik ceza -> Gauss agirlikli K komsu."""
    mu = Xd.mean(0)
    sd = Xd.std(0) + 1e-9
    A = (Xa - mu) / sd
    D = (Xd - mu) / sd
    d2 = (A**2).sum(1)[:, None] + (D**2).sum(1)[None, :] - 2 * A @ D.T
    for ca, cd, w in zip(ceza_a, ceza_d, (2.0, 0.7)):
        d2 = d2 + w * (ca[:, None] != cd[None, :])
    d2 = np.maximum(d2, 0.0)
    k = min(k, D.shape[0] - 1)
    idx = np.argpartition(d2, k, axis=1)[:, :k]
    dk = np.take_along_axis(d2, idx, 1)
    h = np.median(dk, axis=1, keepdims=True) + 1e-6
    w = np.exp(-dk / h)
    w /= w.sum(1, keepdims=True)
    return idx, w


te_tanim = te.tanim.to_numpy()
KOL = ["guc", "u", "std", "sifir", "hsonu", "egim", "yas"]

# ---- SICAK aliciler
sicak_tr = np.array(sorted(set(te_tanim[~msk["soguk"]])))
Xa = np.nan_to_num(F_AL.loc[sicak_tr, KOL].to_numpy(dtype=float))
Xd = np.nan_to_num(F_DN.loc[D_SICAK, KOL].to_numpy(dtype=float))
ia, wa = komsu(
    Xa,
    Xd,
    (F_AL.loc[sicak_tr, "ilce"].to_numpy(), F_AL.loc[sicak_tr, "bolge"].to_numpy()),
    (F_DN.loc[D_SICAK, "ilce"].to_numpy(), F_DN.loc[D_SICAK, "bolge"].to_numpy()),
)
PROF_S = np.einsum("ik,ikt->it", wa, DELTA[ia])
print(f"SICAK profiller {PROF_S.shape} ({time.time() - t0:.0f}s)", flush=True)

# ---- SOGUK aliciler
soguk_tr = np.array(sorted(set(te_tanim[msk["soguk"]])))
te_g = te.groupby("tanim")
gucs = np.log1p(te_g.guc.first())
ilce_te = te_g.ilce.first()
bolge_te = te_g.bolge.first()
dogum = te_g.tarih.min()
Xa2 = np.c_[
    gucs.loc[soguk_tr].to_numpy(dtype=float),
    dogum.loc[soguk_tr].dt.month.to_numpy(dtype=float),
]
Xd2 = np.c_[
    G_SOGUK.guc.to_numpy(dtype=float),
    G_SOGUK.dogum_ay.to_numpy(dtype=float),
]
ib, wb = komsu(
    Xa2,
    Xd2,
    (ilce_te.loc[soguk_tr].to_numpy(), bolge_te.loc[soguk_tr].to_numpy()),
    (G_SOGUK.ilce.to_numpy(), G_SOGUK.bolge.to_numpy()),
    k=40,
)
PROF_C = np.einsum("ik,ikt->it", wb, UYA[ib])
print(f"SOGUK profiller {PROF_C.shape} ({time.time() - t0:.0f}s)", flush=True)

# --------------------------------------------------------------- birlestir
L = np.full(len(te), np.nan)
tau = np.clip((te.tarih - KES_TE).dt.days.to_numpy() - 1, 0, UFUK - 1)

sm = ~msk["soguk"]
ri = pd.Series(np.arange(len(sicak_tr)), index=sicak_tr)
r = ri.reindex(te_tanim[sm]).to_numpy()
L[sm] = F_AL.ref.reindex(te_tanim[sm]).to_numpy(dtype=float) + PROF_S[r, tau[sm]]

cm = msk["soguk"]
ci = pd.Series(np.arange(len(soguk_tr)), index=soguk_tr)
rc = ci.reindex(te_tanim[cm]).to_numpy()
ilk_te = te_g.tarih.min()
yas_te = np.clip((te.tarih - te.tanim.map(ilk_te)).dt.days.to_numpy(), 0, UYA.shape[1] - 1)
L[cm] = gucs.reindex(te_tanim[cm]).to_numpy(dtype=float) + PROF_C[rc, yas_te[cm]]
L = np.clip(L, 0.0, 14.0)

rap = Z.bitir(L, te, msk, A6, "tuketim_z2_analog.csv", kirp=2.0)
rap["parametreler"] = dict(
    K_sicak=K,
    K_soguk=40,
    lag_gun=364,
    donor_sicak=int(len(D_SICAK)),
    donor_soguk=int(len(D_SOGUK)),
)
json.dump(rap, open(os.path.join(BURA, "z2_analog.json"), "w"), indent=1)
print(f"TAMAM ({time.time() - t0:.0f}s)")
