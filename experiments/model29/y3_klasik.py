"""GBM OLMAYAN klasik zaman serisi aday -- tamamen farkli tumevarim yanliligi.

SICAK trafo (gecmisi var):
    L_hat = EWMA_seviye_i + mevsim_endeksi(doy) + haftagunu_endeksi
    mevsim endeksi 2025'in AYNI takvim gunlerinden; trafo kendi gecen yili ile
    ilce endeksi arasinda n-agirlikli buzulme ile harmanlanir (James-Stein).
SOGUK trafo (gecmisi yok):
    YAS KOHORT profili: egitimde 2025-02-01 sonrasi DOGAN trafolarin
    kapasite kullanim orani u = log1p(tuketim) - log1p(guc); yasa gore medyan
    profil U(yas), ustune ilce ve guc bandi ofsetleri.
    Uretim hatti burada GBM grup ortalamalarini kullaniyor -- mekanizma farkli.

Hicbir yerde gurultu eklenmez. Seviye rejim bazinda m6 ile ayni yapilir.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
KOK = os.path.dirname(os.path.dirname(BURA))
KESIM = pd.Timestamp("2026-03-31")

t0 = time.time()
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
for d in (tr, te):
    p = d.lokasyon.str.split(">")
    d["ilce"] = p.str[2]
tr["L"] = np.log1p(tr.tuketim)
tr["doy"] = tr.tarih.dt.dayofyear
tr["dow"] = tr.tarih.dt.dayofweek
te["doy"] = te.tarih.dt.dayofyear
te["dow"] = te.tarih.dt.dayofweek
GENEL = float(tr.L.mean())
print(f"yuklendi ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------------ 1) EWMA seviye
ARD = tr[tr.tarih > KESIM - pd.Timedelta(days=91)].copy()
ARD["w"] = np.exp(-(KESIM - ARD.tarih).dt.days / 20.0)  # yari-omur ~14 gun
ARD["wl"] = ARD.w * ARD.L
g = ARD.groupby("tanim")[["w", "wl"]].sum()
seviye = (g.wl / g.w).rename("taban")
tam_ort = tr.groupby("tanim").L.mean()
seviye = seviye.reindex(tam_ort.index).fillna(tam_ort)
ilk_gun = tr.groupby("tanim").tarih.min()
print(f"seviye {len(seviye):,} trafo ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------- 2) mevsim endeksleri
G25 = tr[(tr.tarih >= "2025-02-15") & (tr.tarih <= "2025-08-15")]
capa25 = G25[(G25.tarih >= "2025-03-01") & (G25.tarih <= "2025-03-31")]

ic_capa = capa25.groupby("ilce").L.mean()
ic_gun = G25.groupby(["ilce", "doy"]).L.mean().rename("m").reset_index()
ic_gun["s"] = ic_gun.m - ic_gun.ilce.map(ic_capa)
ic_gun = ic_gun.sort_values(["ilce", "doy"])
ic_gun["s"] = ic_gun.groupby("ilce").s.transform(
    lambda v: v.rolling(7, center=True, min_periods=1).mean()
)
IC = ic_gun.set_index(["ilce", "doy"]).s

gn_capa = float(capa25.L.mean())
gn = G25.groupby("doy").L.mean() - gn_capa
GN = gn.rolling(7, center=True, min_periods=1).mean()

tf_capa = capa25.groupby("tanim").L.mean()
tf_n = G25[G25.doy >= 91].groupby("tanim").size()
tf_gun = G25.groupby(["tanim", "doy"]).L.mean().rename("m").reset_index()
tf_gun["s"] = tf_gun.m - tf_gun.tanim.map(tf_capa)
tf_gun = tf_gun.sort_values(["tanim", "doy"])
tf_gun["s"] = tf_gun.groupby("tanim").s.transform(
    lambda v: v.rolling(15, center=True, min_periods=3).mean()
)
TF = tf_gun.dropna(subset=["s"]).set_index(["tanim", "doy"]).s
print(f"mevsim endeksleri ({time.time() - t0:.0f}s)", flush=True)

# ---------------------------------------------------------- 3) haftagunu
SON91 = tr[tr.tarih > KESIM - pd.Timedelta(days=91)]
dw_genel = SON91.groupby("dow").L.mean() - SON91.L.mean()
tf_ort91 = SON91.groupby("tanim").L.mean()
DW = SON91.assign(r=SON91.L - SON91.tanim.map(tf_ort91)).groupby(["tanim", "dow"]).r.mean()
print(f"haftagunu ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------------- 4) yas kohort (soguk)
YENI = ilk_gun[ilk_gun >= pd.Timestamp("2025-02-01")].index
kh = tr[tr.tanim.isin(YENI)].copy()
kh["yas"] = (kh.tarih - kh.tanim.map(ilk_gun)).dt.days
kh["u"] = kh.L - np.log1p(kh.guc)
kh = kh[kh.yas <= 130]
kh["yas_k"] = np.minimum(kh.yas // 7, 18)
U_YAS = kh.groupby("yas_k").u.median()
U_GENEL = float(kh.u.median())
_n = kh.groupby("ilce").size()
U_ILCE = (kh.groupby("ilce").u.median() - U_GENEL) * (_n / (_n + 200.0))
kh["gb"] = pd.cut(np.log1p(kh.guc), bins=[-1, 5.5, 6.3, 6.9, 7.4, 99], labels=False)
U_GUC = kh.groupby("gb").u.median() - U_GENEL
print(
    f"kohort: {len(YENI)} yeni trafo, {len(kh):,} satir, U_genel={U_GENEL:.3f} "
    f"({time.time() - t0:.0f}s)",
    flush=True,
)

# ================================================================= TAHMIN
sicak_m = te.tanim.isin(seviye.index).to_numpy()
L_hat = np.full(len(te), np.nan)

sc = te[sicak_m]
taban = sc.tanim.map(seviye).to_numpy()
s_ic = pd.MultiIndex.from_arrays([sc.ilce, sc.doy]).map(IC).to_numpy(dtype=float)
s_gn = sc.doy.map(GN).to_numpy(dtype=float)
s_ic = np.where(np.isnan(s_ic), s_gn, s_ic)
s_tf = pd.MultiIndex.from_arrays([sc.tanim, sc.doy]).map(TF).to_numpy(dtype=float)
n_tf = np.nan_to_num(sc.tanim.map(tf_n).to_numpy(dtype=float))
w = n_tf / (n_tf + 40.0)
w = np.where(np.isnan(s_tf), 0.0, w)
s_tf = np.nan_to_num(s_tf)
mevsim = w * s_tf + (1 - w) * np.nan_to_num(s_ic)
d_tf = pd.MultiIndex.from_arrays([sc.tanim, sc.dow]).map(DW).to_numpy(dtype=float)
d_gn = sc.dow.map(dw_genel).to_numpy(dtype=float)
haftagunu = np.where(np.isnan(d_tf), d_gn, 0.5 * np.nan_to_num(d_tf) + 0.5 * d_gn)
L_hat[sicak_m] = taban + mevsim + haftagunu

sg = te[~sicak_m]
ilk_te = te.groupby("tanim").tarih.min()
yas = (sg.tarih - sg.tanim.map(ilk_te)).dt.days.to_numpy()
yk = np.minimum(yas // 7, 18)
u_yas = pd.Series(yk).map(U_YAS).to_numpy(dtype=float)
u_yas = np.where(np.isnan(u_yas), U_GENEL, u_yas)
o_ilce = sg.ilce.map(U_ILCE).fillna(0.0).to_numpy(dtype=float)
gb = pd.cut(np.log1p(sg.guc), bins=[-1, 5.5, 6.3, 6.9, 7.4, 99], labels=False)
o_guc = pd.Series(np.asarray(gb)).map(U_GUC).fillna(0.0).to_numpy(dtype=float)
s_ic_sg = pd.MultiIndex.from_arrays([sg.ilce, sg.doy]).map(IC).to_numpy(dtype=float)
s_ic_sg = np.where(np.isnan(s_ic_sg), sg.doy.map(GN).to_numpy(dtype=float), s_ic_sg)
d_sg = sg.dow.map(dw_genel).to_numpy(dtype=float)
L_hat[~sicak_m] = (
    np.log1p(sg.guc.to_numpy(dtype=float))
    + u_yas
    + 0.5 * o_ilce
    + 0.5 * o_guc
    + np.nan_to_num(s_ic_sg)
    + d_sg
)

kotu = ~np.isfinite(L_hat)
print(f"finite olmayan {int(kotu.sum())} -> genel ortalama ile dolduruluyor")
L_hat[kotu] = GENEL
L_hat = np.clip(L_hat, 0.0, 14.0)

# ------------------------------------------------------- rejim capasi (m6)
A6 = np.log1p(pd.read_csv(os.path.join(KOK, "submissions/tuketim_m6_ikiyon.csv")).tuketim.values)
soguk = ~sicak_m
kuyruk = sicak_m & (te.tanim.map(ilk_gun) >= pd.Timestamp("2026-03-26")).to_numpy()
cek = sicak_m & ~kuyruk
rap = {}
L2 = L_hat.copy()
for nm, m in [("soguk", soguk), ("kuyruk", kuyruk), ("cekirdek", cek)]:
    d = float(A6[m].mean() - L_hat[m].mean())
    L2[m] = L_hat[m] + d
    rap[nm] = dict(satir=int(m.sum()), kaydirma=d)
    print(f"  capa {nm}: {d:+.4f}  (satir {int(m.sum()):,})")

y = np.clip(np.expm1(L2), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
yol = os.path.join(KOK, "submissions", "tuketim_y33_klasik.csv")
out.to_csv(yol, index=False)
SS = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == SS.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
)
assert kapi["satir"] == 714688 and kapi["id_birebir"] and not kapi["nan"] and not kapi["negatif"]
json.dump(dict(capa=rap, kapi=kapi), open(os.path.join(BURA, "y3_klasik.json"), "w"), indent=1)
print(f"YAZILDI {yol} kapi={kapi} ({time.time() - t0:.0f}s)")
