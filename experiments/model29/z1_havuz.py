"""Z1 -- HIYERARSIK / HAVUZLANMIS (top-down) tahmin.

Bilgi kullanim bicimi mevcut adaylarin HIC BIRINDE yok:

  * Mevsim sekli TRAFO duzeyinde degil, (ilce x guc-kovasi) HAVUZUNDA ve
    DOGRUSAL (kWh) uzayda olculur -- yani havuzun sekli KUTLE AGIRLIKLI:
    buyuk trafolar sekli belirler. y3/y45 log uzayinda ilce ORTALAMASI kullanir
    (esit agirlikli); ikisi Jensen farkiyla sistematik olarak ayrisir.
  * Takvim eslemesi 364 GUN (52 tam hafta) geriye -- haftanin gunu korunur.
    Uretim hattinin doy tabanli eslemesi haftagununu kaydirir.
  * Sekil ucgen buzulme ile havuz -> ilce -> genel yonunde regularize edilir
    (James-Stein); seviye trafonun KENDI son 45 gunundeki log ortalamasidir,
    yani MSLE'nin istedigi uzayda.
  * Soguk trafolarda seviye guc x (yeni-baglanti kohortunun yasa gore
    kapasite kullanim orani) ile kurulur, sekil havuzdan gelir.

Buyume carpani R_g bilerek yoktur: trafo kendi guncel seviyesine capalandigi
icin R_g oranda sadelesir. Kalan sey saf MEVSIM ORANIDIR.
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
KESIM = pd.Timestamp("2026-03-31")
REF0, REF1 = pd.Timestamp("2026-02-15"), KESIM  # seviye/capa penceresi (2026)
GB = [-1, 5.5, 6.3, 6.9, 7.4, 99]  # log1p(guc) kovalari (y3 ile ayni)

t0 = time.time()
tr, te = Z.yukle()
msk = Z.maskeler(tr, te)
A6 = Z.taban()
for d in (tr, te):
    d["gb"] = pd.cut(np.log1p(d.guc), bins=GB, labels=False).astype(int)
    d["havuz"] = d.ilce.astype(str) + "|" + d.gb.astype(str)
print(f"yuklendi, {tr.havuz.nunique()} havuz ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------- 1) havuz gunluk kisi-basi (kWh)
KAYNAK0 = (te.tarih.min() - LAG).normalize()
KAYNAK1 = (te.tarih.max() - LAG).normalize()
G = tr[(tr.tarih >= KAYNAK0) & (tr.tarih <= KAYNAK1)]
R0, R1 = REF0 - LAG, REF1 - LAG  # 2025'teki referans penceresi
REFP = tr[(tr.tarih >= R0) & (tr.tarih <= R1)]
print(f"kaynak penceresi {KAYNAK0.date()}..{KAYNAK1.date()} ({len(G):,} satir)", flush=True)


def oran(df, ref, anahtar):
    """anahtar duzeyinde gunluk kisi-basi kWh / referans kisi-basi kWh."""
    m = df.groupby([anahtar, "tarih"]).tuketim.mean().rename("m").reset_index()
    r = ref.groupby(anahtar).tuketim.mean().rename("r")
    m["r"] = m[anahtar].map(r)
    m = m[m.r > 0]
    m["o"] = np.log(np.maximum(m.m, 1.0) / np.maximum(m.r, 1.0))
    n = df.groupby(anahtar).size().rename("n")
    return m, n


def ayristir(m, anahtar):
    """log-orani PURUZSUZ TREND + HAFTAGUNU carpanina ayir."""
    m = m.sort_values([anahtar, "tarih"]).copy()
    m["trend"] = m.groupby(anahtar).o.transform(
        lambda v: v.rolling(7, center=True, min_periods=3).mean()
    )
    m["kal"] = m.o - m.trend
    m["dow"] = m.tarih.dt.dayofweek
    dw = m.groupby([anahtar, "dow"]).kal.mean()
    dw = dw - dw.groupby(level=0).transform("mean")
    return m.set_index([anahtar, "tarih"]).trend, dw


mh, nh = oran(G, REFP, "havuz")
mi, ni = oran(G, REFP, "ilce")
TR_H, DW_H = ayristir(mh, "havuz")
TR_I, DW_I = ayristir(mi, "ilce")
gg = G.groupby("tarih").tuketim.mean()
gr = float(REFP.tuketim.mean())
og = np.log(np.maximum(gg, 1.0) / max(gr, 1.0))
TR_G = og.rolling(7, center=True, min_periods=3).mean()
kal = og - TR_G
DW_G = kal.groupby(kal.index.dayofweek).mean()
DW_G = DW_G - DW_G.mean()
print(f"sekiller hazir ({time.time() - t0:.0f}s)", flush=True)

# ------------------------------------------------- 2) test satirlarina sekil
kt = (te.tarih - LAG).dt.normalize()
dow = te.tarih.dt.dayofweek


def cek(idx, seri):
    return pd.Series(seri.reindex(idx).to_numpy(dtype=float), index=te.index)


s_h = cek(pd.MultiIndex.from_arrays([te.havuz, kt]), TR_H)
s_i = cek(pd.MultiIndex.from_arrays([te.ilce, kt]), TR_I)
s_g = cek(pd.Index(kt), TR_G)
d_h = cek(pd.MultiIndex.from_arrays([te.havuz, dow]), DW_H)
d_i = cek(pd.MultiIndex.from_arrays([te.ilce, dow]), DW_I)
d_g = cek(pd.Index(dow), DW_G)
s_g = s_g.fillna(0.0)
d_g = d_g.fillna(0.0)
s_i = s_i.fillna(s_g)
d_i = d_i.fillna(d_g)
s_h = s_h.fillna(s_i)
d_h = d_h.fillna(d_i)

# ucgen buzulme: havuz -> ilce -> genel (gozlem sayisiyla)
K1, K2 = 3000.0, 20000.0
nh_v = te.havuz.map(nh).fillna(0).to_numpy(dtype=float)
ni_v = te.ilce.map(ni).fillna(0).to_numpy(dtype=float)
w_h = nh_v / (nh_v + K1)
w_i = (1 - w_h) * (ni_v / (ni_v + K2))
w_g = 1 - w_h - w_i
SEKIL = w_h * (s_h + d_h) + w_i * (s_i + d_i) + w_g * (s_g + d_g)
SEKIL = np.asarray(SEKIL, dtype=float)
print(
    f"sekil: ort agirlik havuz {w_h.mean():.2f} ilce {w_i.mean():.2f} genel {w_g.mean():.2f}, "
    f"sekil std {SEKIL.std():.3f} ({time.time() - t0:.0f}s)",
    flush=True,
)

# ------------------------------------------------- 3) SICAK seviye (log uzayi)
SON = tr[(tr.tarih >= REF0) & (tr.tarih <= REF1)]
lv = SON.groupby("tanim").L.mean()
nlv = SON.groupby("tanim").size()
tam = tr.groupby("tanim").L.mean()
lv = lv.reindex(tam.index)
nlv = nlv.reindex(tam.index).fillna(0)
lv = lv.fillna(tam)
# az gozlemli trafolar kendi tam gecmis ortalamasina buzulur
wv = nlv / (nlv + 7.0)
SEVIYE = wv * lv + (1 - wv) * tam

# ------------------------------------------------- 4) SOGUK seviye: kohort x guc
ilk_tr = msk["ilk"]
YENI = ilk_tr[ilk_tr >= pd.Timestamp("2025-02-01")].index
kh = tr[tr.tanim.isin(YENI)].copy()
kh["yas"] = (kh.tarih - kh.tanim.map(ilk_tr)).dt.days
kh["u"] = kh.L - np.log1p(kh.guc)
kh = kh[kh.yas <= 140]
kh["yk"] = np.minimum(kh.yas // 10, 13)
U_YAS = kh.groupby("yk").u.median()
U_GEN = float(kh.u.median())
_n = kh.groupby(["ilce", "gb"]).size()
U_HAVUZ = (kh.groupby(["ilce", "gb"]).u.median() - U_GEN) * (_n / (_n + 300.0))
print(f"kohort: {len(YENI)} yeni trafo, {len(kh):,} satir, U={U_GEN:.3f}", flush=True)

# ------------------------------------------------- 5) birlestir
L = np.full(len(te), np.nan)
sicak = ~msk["soguk"]
L[sicak] = te.tanim[sicak].map(SEVIYE).to_numpy(dtype=float) + SEKIL[sicak]

sg = te[msk["soguk"]]
ilk_te = te.groupby("tanim").tarih.min()
yas = (sg.tarih - sg.tanim.map(ilk_te)).dt.days.to_numpy()
yk = np.minimum(yas // 10, 13)
u = pd.Series(yk).map(U_YAS).to_numpy(dtype=float)
u = np.where(np.isfinite(u), u, U_GEN)
oh = pd.MultiIndex.from_arrays([sg.ilce, sg.gb]).map(U_HAVUZ).to_numpy(dtype=float)
oh = np.nan_to_num(oh)
# soguk trafoda yas profili zaten buyumeyi tasir; sekilin pencere ortalamasi cikarilir
sk = SEKIL[msk["soguk"]]
L[msk["soguk"]] = np.log1p(sg.guc.to_numpy(dtype=float)) + u + oh + (sk - sk.mean())
L = np.clip(L, 0.0, 14.0)

rap = Z.bitir(L, te, msk, A6, "tuketim_z1_havuz.csv", kirp=2.0)
rap["parametreler"] = dict(lag_gun=364, kovalar=GB, K1=K1, K2=K2, U_GEN=U_GEN)
json.dump(rap, open(os.path.join(BURA, "z1_havuz.json"), "w"), indent=1)
print(f"TAMAM ({time.time() - t0:.0f}s)")
