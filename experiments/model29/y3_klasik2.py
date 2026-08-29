"""Klasik aday, 2. surum -- MEVSIMSEL NAIF + YoY SURUKLENME (GBM yok).

y3_klasik.py'nin zayifligi: taban son 91 gunun (kis) EWMA seviyesiydi; sulama gibi
MEVSIMSEL UYANAN trafolari sistematik olarak dusuk tahmin ediyordu (Q'nun %57'si
201 trafodan geliyordu). Bu surum dogru klasik kurgu:

    L_hat(i, t) = gecen_yil_i(doy(t))  +  yoy_i        (gecen yili olan trafo)
    yoy_i       = simdiki_seviye_i - gecen_yil_ayni_mevsim_i   (buzulmus)
    L_hat(i, t) = simdiki_seviye_i + ilce_mevsim_endeksi(doy)  (gecen yili yok)
    ikisi arasinda gecen-yil gozlem sayisina gore agirlikli harman

SOGUK trafo: y3_klasik ile ayni yas-kohort profili (mekanizma GBM'den farkli).
Gurultu eklenmez; seviye rejim bazinda m6'ya capalanir.
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
    d["ilce"] = d.lokasyon.str.split(">").str[2]
tr["L"] = np.log1p(tr.tuketim)
tr["doy"] = tr.tarih.dt.dayofyear
tr["dow"] = tr.tarih.dt.dayofweek
te["doy"] = te.tarih.dt.dayofyear
te["dow"] = te.tarih.dt.dayofweek
GENEL = float(tr.L.mean())
ilk_gun = tr.groupby("tanim").tarih.min()
print(f"yuklendi ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------- 1) simdiki seviye (EWMA)
ARD = tr[tr.tarih > KESIM - pd.Timedelta(days=91)].copy()
ARD["w"] = np.exp(-(KESIM - ARD.tarih).dt.days / 20.0)
ARD["wl"] = ARD.w * ARD.L
_g = ARD.groupby("tanim")[["w", "wl"]].sum()
simdi = _g.wl / _g.w
tam_ort = tr.groupby("tanim").L.mean()
simdi = simdi.reindex(tam_ort.index).fillna(tam_ort)

# ------------------------------- 2) gecen yil ayni takvim penceresi (doy profili)
# test doy araligi 91..212 -> 2025'in ayni doy'lari (iki yil da artik degil)
GY = tr[(tr.tarih >= "2025-03-15") & (tr.tarih <= "2025-08-15")].copy()
prof = GY.groupby(["tanim", "doy"]).L.mean().rename("m").reset_index().sort_values(["tanim", "doy"])
# +-10 gunluk merkezli yumusatma (mevsimsel naif, gurultuyu bastirir)
prof["s"] = prof.groupby("tanim").m.transform(
    lambda v: v.rolling(21, center=True, min_periods=4).mean()
)
PROF = prof.dropna(subset=["s"]).set_index(["tanim", "doy"]).s
gy_n = GY[GY.doy >= 91].groupby("tanim").size()

# gecen yilin AYNI MEVSIMI (kis-bahar gecisi) -> yoy suruklenmesinin capasi
CAPA = tr[(tr.tarih >= "2025-01-01") & (tr.tarih <= "2025-03-31")]
gy_capa = CAPA.groupby("tanim").L.mean()
capa_n = CAPA.groupby("tanim").size()

yoy_ham = (simdi - gy_capa).dropna()
ilce_of = tr.groupby("tanim").ilce.first()
yoy_ilce = yoy_ham.groupby(ilce_of.reindex(yoy_ham.index).values).median()
yoy_genel = float(yoy_ham.median())
wn = capa_n.reindex(yoy_ham.index).fillna(0)
w_yoy = wn / (wn + 25.0)
yoy_gr = pd.Series(
    ilce_of.reindex(yoy_ham.index).map(yoy_ilce).fillna(yoy_genel).values, index=yoy_ham.index
)
YOY = (w_yoy * yoy_ham + (1 - w_yoy) * yoy_gr).clip(-1.5, 1.5)  # suruklenme sinirli
print(
    f"gecen yil profili {PROF.index.get_level_values(0).nunique()} trafo, "
    f"yoy medyan {yoy_genel:+.3f} ({time.time() - t0:.0f}s)",
    flush=True,
)

# ------------------------------------- 3) ilce mevsim endeksi (yedek yol icin)
G25 = tr[(tr.tarih >= "2025-01-01") & (tr.tarih <= "2025-08-15")]
capa25_ic = G25[(G25.tarih >= "2025-01-01") & (G25.tarih <= "2025-03-31")].groupby("ilce").L.mean()
ic = G25.groupby(["ilce", "doy"]).L.mean().rename("m").reset_index()
ic["s"] = ic.m - ic.ilce.map(capa25_ic)
ic = ic.sort_values(["ilce", "doy"])
ic["s"] = ic.groupby("ilce").s.transform(lambda v: v.rolling(15, center=True, min_periods=2).mean())
IC = ic.set_index(["ilce", "doy"]).s
gn_capa = float(G25[(G25.tarih <= "2025-03-31")].L.mean())
GN = (G25.groupby("doy").L.mean() - gn_capa).rolling(15, center=True, min_periods=2).mean()

# ---------------------------------------------------------- 4) haftagunu
SON91 = tr[tr.tarih > KESIM - pd.Timedelta(days=91)]
dw_genel = SON91.groupby("dow").L.mean() - SON91.L.mean()
tf_ort91 = SON91.groupby("tanim").L.mean()
DW = SON91.assign(r=SON91.L - SON91.tanim.map(tf_ort91)).groupby(["tanim", "dow"]).r.mean()

# ------------------------------------------------------- 5) yas kohort (soguk)
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
print(f"kohort {len(YENI)} trafo, U_genel={U_GENEL:.3f} ({time.time() - t0:.0f}s)", flush=True)

# ================================================================= TAHMIN
sicak_m = te.tanim.isin(simdi.index).to_numpy()
L_hat = np.full(len(te), np.nan)

sc = te[sicak_m]
now = sc.tanim.map(simdi).to_numpy(dtype=float)
# yol A: mevsimsel naif + yoy
p_gy = pd.MultiIndex.from_arrays([sc.tanim, sc.doy]).map(PROF).to_numpy(dtype=float)
yoy = sc.tanim.map(YOY).to_numpy(dtype=float)
yoy = np.where(np.isnan(yoy), yoy_genel, yoy)
yolA = p_gy + yoy
# yol B: simdiki seviye + ilce mevsim endeksi
s_ic = pd.MultiIndex.from_arrays([sc.ilce, sc.doy]).map(IC).to_numpy(dtype=float)
s_ic = np.where(np.isnan(s_ic), sc.doy.map(GN).to_numpy(dtype=float), s_ic)
yolB = now + np.nan_to_num(s_ic)
# harman: gecen yil gozlem sayisina gore
n_gy = np.nan_to_num(sc.tanim.map(gy_n).to_numpy(dtype=float))
wA = n_gy / (n_gy + 15.0)
wA = np.where(np.isnan(yolA), 0.0, wA)
yolA = np.where(np.isnan(yolA), 0.0, yolA)
L_sc = wA * yolA + (1 - wA) * yolB
# haftagunu
d_tf = pd.MultiIndex.from_arrays([sc.tanim, sc.dow]).map(DW).to_numpy(dtype=float)
d_gn = sc.dow.map(dw_genel).to_numpy(dtype=float)
L_sc = L_sc + np.where(np.isnan(d_tf), d_gn, 0.5 * np.nan_to_num(d_tf) + 0.5 * d_gn)
# klasik tahminci icin standart makuliyet bandi: trafonun kendi son 365 gunluk
# log1p dagiliminin [q05-0.5, q95+0.5] araligi disina cikmaz (kacak onleme)
YIL = tr[tr.tarih > KESIM - pd.Timedelta(days=365)]
_q = YIL.groupby("tanim").L.quantile([0.05, 0.95]).unstack()
alt = sc.tanim.map(_q[0.05]).to_numpy(dtype=float) - 0.5
ust = sc.tanim.map(_q[0.95]).to_numpy(dtype=float) + 0.5
kac = np.isfinite(alt) & np.isfinite(ust)
L_sc[kac] = np.clip(L_sc[kac], alt[kac], ust[kac])
print(f"  makuliyet bandi uygulandi: {int(kac.sum()):,} satir")
L_hat[sicak_m] = L_sc
print(f"sicak: yol A agirligi ort {wA.mean():.3f} ({time.time() - t0:.0f}s)", flush=True)

sg = te[~sicak_m]
ilk_te = te.groupby("tanim").tarih.min()
yk = np.minimum((sg.tarih - sg.tanim.map(ilk_te)).dt.days.to_numpy() // 7, 18)
u_yas = pd.Series(yk).map(U_YAS).to_numpy(dtype=float)
u_yas = np.where(np.isnan(u_yas), U_GENEL, u_yas)
o_ilce = sg.ilce.map(U_ILCE).fillna(0.0).to_numpy(dtype=float)
gb = pd.cut(np.log1p(sg.guc), bins=[-1, 5.5, 6.3, 6.9, 7.4, 99], labels=False)
o_guc = pd.Series(np.asarray(gb)).map(U_GUC).fillna(0.0).to_numpy(dtype=float)
s_sg = pd.MultiIndex.from_arrays([sg.ilce, sg.doy]).map(IC).to_numpy(dtype=float)
s_sg = np.where(np.isnan(s_sg), sg.doy.map(GN).to_numpy(dtype=float), s_sg)
L_hat[~sicak_m] = (
    np.log1p(sg.guc.to_numpy(dtype=float))
    + u_yas
    + 0.5 * o_ilce
    + 0.5 * o_guc
    + np.nan_to_num(s_sg)
    + sg.dow.map(dw_genel).to_numpy(dtype=float)
)

kotu = ~np.isfinite(L_hat)
print(f"finite olmayan {int(kotu.sum())}")
L_hat[kotu] = GENEL
L_hat = np.clip(L_hat, 0.0, 14.0)

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
yol = os.path.join(KOK, "submissions", "tuketim_y34_mevsimsel.csv")
out.to_csv(yol, index=False)
SS = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
kapi = dict(
    satir=len(out),
    id_birebir=bool((out.id.values == SS.iloc[:, 0].values).all()),
    nan=int(out.tuketim.isna().sum()),
    negatif=int((out.tuketim < 0).sum()),
)
assert kapi["satir"] == 714688 and kapi["id_birebir"] and not kapi["nan"] and not kapi["negatif"]
json.dump(dict(capa=rap, kapi=kapi), open(os.path.join(BURA, "y3_klasik2.json"), "w"), indent=1)
print(f"YAZILDI {yol} kapi={kapi} ({time.time() - t0:.0f}s)")
