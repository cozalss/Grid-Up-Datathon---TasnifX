"""p03 fikirler: yaz25 tezgahinda SIZINTISIZ olculmus kazanclar.

Butun kalibrasyon sabitleri EGITIM blogunda (2025-12..2026-03 hedefi) ogrenilir,
yaz25'te yalnizca UYGULANIR. yaz25 hedefi hicbir yerde parametre uydurmaz.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import p03_tezgah as T  # noqa: E402

BURA = os.path.dirname(os.path.abspath(__file__))
PK = dict(
    objective="l2",
    metric="l2",
    learning_rate=0.04,
    num_leaves=63,
    min_data_in_leaf=200,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5.0,
    num_threads=8,
    verbose=-1,
    seed=7,
)
PKC = dict(PK, objective="binary", metric="binary_logloss")
TUR = 600
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


tr, te = T.ortam()
Xe, ye, Xd, yd, hd, d_soguk = T.veri(tr)
soguk_d = hd.tanim.isin(d_soguk).to_numpy()
log(f"egitim {Xe.shape} deger {Xd.shape}, deger soguk %{100 * soguk_d.mean():.1f}")

R = {
    "tezgah": {
        "egitim_satir": int(len(Xe)),
        "deger_satir": int(len(Xd)),
        "deger_soguk_orani": float(soguk_d.mean()),
    }
}

# ---------- TABAN ----------
m0 = lgb.train(PK, lgb.Dataset(Xe, ye), num_boost_round=TUR)
p0 = m0.predict(Xd)
taban = T.rmsle(yd, p0)
R["taban"] = taban
log(f"TABAN {taban:.5f}")

# egitim blogu icin kendi-dis tahmini (kalibrasyon uydurmasin diye 2 katli)
kat = (np.arange(len(Xe)) % 2).astype(bool)
pe = np.empty(len(Xe))
for b in (False, True):
    mm = lgb.train(
        PK, lgb.Dataset(Xe[~np.equal(kat, b)], ye[~np.equal(kat, b)]), num_boost_round=TUR
    )
    pe[np.equal(kat, b)] = mm.predict(Xe[np.equal(kat, b)])
log("egitim blogu katli tahmin hazir")

# ================= FIKIR 1: sifir olasiligi ile iki asamali =================
z_e = (ye == 0).astype(int)
mc = lgb.train(PKC, lgb.Dataset(Xe, z_e), num_boost_round=TUR)
P0d = mc.predict(Xd)
poz = ye > 0
mp = lgb.train(PK, lgb.Dataset(Xe[poz], ye[poz]), num_boost_round=TUR)
ppos_d = mp.predict(Xd)
# RMSLE optimal nokta tahmini E[log1p] = (1-P0)*E[log1p | y>0]
p1 = (1 - P0d) * ppos_d
R["f1_iki_asamali"] = {"rmsle": T.rmsle(yd, p1), "kazanc": taban - T.rmsle(yd, p1)}
log(
    f"F1 iki-asamali {R['f1_iki_asamali']['rmsle']:.5f} (kazanc {R['f1_iki_asamali']['kazanc']:+.5f})"
)
# varyant: taban regresyonu ile carpim
p1b = (1 - P0d) * p0
R["f1b_carpim"] = {"rmsle": T.rmsle(yd, p1b), "kazanc": taban - T.rmsle(yd, p1b)}
log(f"F1b carpim {R['f1b_carpim']['rmsle']:.5f} ({R['f1b_carpim']['kazanc']:+.5f})")
# varyant: yalnizca yuksek olasilikta sifirla (esik EGITIM blogunda secilir)
P0e = None
mce = np.empty(len(Xe))
for b in (False, True):
    mm = lgb.train(
        PKC, lgb.Dataset(Xe[~np.equal(kat, b)], z_e[~np.equal(kat, b)]), num_boost_round=TUR
    )
    mce[np.equal(kat, b)] = mm.predict(Xe[np.equal(kat, b)])
en, esik = 1e9, None
for th in np.arange(0.3, 0.96, 0.05):
    q = np.where(mce > th, 0.0, pe)
    v = float(np.sqrt(np.mean((q - ye) ** 2)))
    if v < en:
        en, esik = v, float(th)
p1c = np.where(P0d > esik, 0.0, p0)
R["f1c_esik"] = {
    "esik": esik,
    "egitim_rmsle": en,
    "rmsle": T.rmsle(yd, p1c),
    "kazanc": taban - T.rmsle(yd, p1c),
}
log(f"F1c esik={esik:.2f} {R['f1c_esik']['rmsle']:.5f} ({R['f1c_esik']['kazanc']:+.5f})")

# ================= FIKIR 2: soguk satirlara ayri muamele =================
# EGITIM blogunda soguk olan satirlarda en iyi sabit kaydirma/buzme aranir
e_gec = tr[(tr.tarih <= T.E_KESIM) & (tr.tarih >= T.E_GEC_BAS)]
e_hed = tr[(tr.tarih > T.E_KESIM) & (tr.tarih <= T.E_HED_SON)]
e_hed = e_hed[~e_hed.tanim.isin(d_soguk)].reset_index(drop=True)
soguk_e = (~e_hed.tanim.isin(set(e_gec.tanim))).to_numpy()
log(f"egitim blogu soguk orani %{100 * soguk_e.mean():.1f} ({soguk_e.sum()} satir)")
if soguk_e.sum() > 500:
    m_e, s_e = pe[soguk_e], ye[soguk_e]
    en2, par = 1e9, None
    for a in np.arange(-0.6, 0.61, 0.05):
        for lam in np.arange(0.5, 1.31, 0.1):
            q = lam * (m_e - m_e.mean()) + m_e.mean() + a
            v = float(np.sqrt(np.mean((q - s_e) ** 2)))
            if v < en2:
                en2, par = v, (float(a), float(lam))
    a, lam = par
    p2 = p0.copy()
    mu = p0[soguk_d].mean()
    p2[soguk_d] = lam * (p0[soguk_d] - mu) + mu + a
    R["f2_soguk_kalibrasyon"] = {
        "kaydirma": a,
        "buzme": lam,
        "rmsle": T.rmsle(yd, p2),
        "kazanc": taban - T.rmsle(yd, p2),
    }
    log(
        f"F2 soguk a={a:+.2f} lam={lam:.2f} {R['f2_soguk_kalibrasyon']['rmsle']:.5f} "
        f"({R['f2_soguk_kalibrasyon']['kazanc']:+.5f})"
    )

# ================= FIKIR 3: genel sapma duzeltmesi =================
a3 = float((ye - pe).mean())
p3 = p0 + a3
R["f3_genel_kaydirma"] = {
    "kaydirma": a3,
    "rmsle": T.rmsle(yd, p3),
    "kazanc": taban - T.rmsle(yd, p3),
}
log(
    f"F3 genel kaydirma {a3:+.4f} -> {R['f3_genel_kaydirma']['rmsle']:.5f} "
    f"({R['f3_genel_kaydirma']['kazanc']:+.5f})"
)


# ================= FIKIR 4: idnum-komsu seviyesi (soguk trafo onceligi) =================
def komsu_ozellik(gec, hed_tanim_df, k=8):
    """Her trafo icin: ayni ilcede, idnum'a en yakin k komsunun ortalama log seviyesi."""
    lv = gec.groupby("tanim").agg(ly=("ly", "mean")).reset_index()
    meta = gec.drop_duplicates("tanim")[["tanim", "ilce", "idnum", "guc"]]
    lv = lv.merge(meta, on="tanim")
    lv = lv.dropna(subset=["idnum"])
    out = {}
    hedmeta = hed_tanim_df.dropna(subset=["idnum"])
    for ilce, gh in hedmeta.groupby("ilce", observed=True):
        gl = lv[lv.ilce == ilce]
        if len(gl) < 2:
            continue
        xs = np.sort(gl.idnum.to_numpy())
        order = np.argsort(gl.idnum.to_numpy())
        vs = gl.ly.to_numpy()[order]
        ids = gl.tanim.to_numpy()[order]
        for tn, xv in zip(gh.tanim.to_numpy(), gh.idnum.to_numpy()):
            j = np.searchsorted(xs, xv)
            lo, hi = max(0, j - k), min(len(xs), j + k)
            sel = np.arange(lo, hi)
            sel = sel[ids[sel] != tn]
            if len(sel):
                out[tn] = float(vs[sel].mean())
    return out


def ekle_komsu(X, hed, gec):
    hm = hed.drop_duplicates("tanim")[["tanim", "ilce", "idnum", "guc"]]
    kom = komsu_ozellik(gec, hm)
    X = X.copy()
    X["k_komsu"] = hed.tanim.map(kom).to_numpy()
    return X


d_gec = tr[(tr.tarih <= T.D_KESIM) & (tr.tarih >= T.D_GEC_BAS)]
Xe2 = ekle_komsu(Xe, e_hed, e_gec[~e_gec.tanim.isin(d_soguk)])
Xd2 = ekle_komsu(Xd, hd, d_gec)
log(
    f"komsu ozelligi: egitim NaN %{100 * Xe2.k_komsu.isna().mean():.1f} "
    f"deger NaN %{100 * Xd2.k_komsu.isna().mean():.1f}"
)
if len(Xe2) == len(ye):
    m4 = lgb.train(PK, lgb.Dataset(Xe2, ye), num_boost_round=TUR)
    p4 = m4.predict(Xd2)
    R["f4_idnum_komsu"] = {
        "rmsle": T.rmsle(yd, p4),
        "kazanc": taban - T.rmsle(yd, p4),
        "soguk_rmsle_once": float(np.sqrt(np.mean((p0[soguk_d] - yd[soguk_d]) ** 2))),
        "soguk_rmsle_sonra": float(np.sqrt(np.mean((p4[soguk_d] - yd[soguk_d]) ** 2))),
    }
    log(f"F4 idnum-komsu {R['f4_idnum_komsu']['rmsle']:.5f} ({R['f4_idnum_komsu']['kazanc']:+.5f})")
else:
    log("F4 ATLANDI: satir sayisi uyusmuyor", len(Xe2), len(ye))

# ================= FIKIR 5: F1c + F2 + F4 birlesik =================
try:
    pk = p4.copy()
    P0d4 = P0d
    if "f2_soguk_kalibrasyon" in R:
        mu = pk[soguk_d].mean()
        pk[soguk_d] = (
            R["f2_soguk_kalibrasyon"]["buzme"] * (pk[soguk_d] - mu)
            + mu
            + R["f2_soguk_kalibrasyon"]["kaydirma"]
        )
    pk = np.where(P0d4 > esik, 0.0, pk)
    R["f5_birlesik"] = {"rmsle": T.rmsle(yd, pk), "kazanc": taban - T.rmsle(yd, pk)}
    log(f"F5 birlesik {R['f5_birlesik']['rmsle']:.5f} ({R['f5_birlesik']['kazanc']:+.5f})")
except NameError:
    pass

with open(os.path.join(BURA, "p03_fikir.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
