"""p30 -- KIRMIZI TAKIM: ezber kanali ve KIRLILIK TESTI.

Soru: p21 (soguk harman cat-tekil -> 3/1/1) kazanci yaz25/guz25 bloklarinin
EZBERLENEBILIR soguk satirlarindan mi geliyor?

Egitim YOK. Gonderim YOK. submissions/ altina yazma YOK.
"""
import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
BLOKLAR = ("yaz25", "guz25", "kis26")
AILE = ("cat", "xgb", "lgbm")
W311 = np.array([0.6, 0.2, 0.2])
WCAT = np.array([1.0, 0.0, 0.0])

R = {"00_BASLIK": "p30 kirmizi takim -- ezber kanali nicelemesi + KIRLILIK TESTI"}

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
T = pd.read_parquet(os.path.join(DN, "test.parquet"))

# ---------------------------------------------------------------- 1. KANAL
tn = E[["tanim", "tanim_num"]].drop_duplicates()
R["01_kimlik_kanali"] = {
    "tanim_num_benzersiz": int(tn.tanim_num.nunique()),
    "tanim_benzersiz": int(tn.tanim.nunique()),
    "tanim_num_basina_max_trafo": int(tn.groupby("tanim_num").tanim.nunique().max()),
    "tanim_onekli_kolonlar": sorted(c for c in E.columns if c.startswith("tanim")),
    "not": "maske yalnizca 't_' onekli kolonlari siler; tanim* kolonlari SAG KALIR",
}

# ---------------------------------------------------------- 2. EZBER ORANI
ezber = {}
maske = {}
for b in BLOKLAR:
    blk = E[E._blok == b]
    sog = blk[blk.soguk_mu == 1]
    egit_tanim = set(E.loc[E._blok != b, "tanim"].unique())  # tuketim_model.py:1543
    m = sog.tanim.isin(egit_tanim).to_numpy()
    maske[b] = m
    ezber[b] = {
        "n_soguk_satir": int(len(sog)),
        "n_soguk_trafo": int(sog.tanim.nunique()),
        "ezberlenebilir_satir_orani": round(float(m.mean()), 4),
        "ezberlenebilir_trafo_orani": round(
            float(sog.loc[m, "tanim"].nunique() / max(sog.tanim.nunique(), 1)), 4
        ),
        "ezber_satir_medyan_egitim_satiri": float(
            E[E.tanim.isin(sog.loc[m, "tanim"].unique()) & (E._blok != b)]
            .groupby("tanim").size().median()
        )
        if m.any()
        else 0.0,
    }
# TEST tarafi
test_soguk_tanim = set(T.loc[T.soguk_mu == 1, "tanim"].unique())
egit_tum = set(E.tanim.unique())
ezber["TEST"] = {
    "n_soguk_trafo": len(test_soguk_tanim),
    "egitimde_gorunen_trafo": len(test_soguk_tanim & egit_tum),
    "ezberlenebilir_satir_orani": round(
        float(T.loc[T.soguk_mu == 1, "tanim"].isin(egit_tum).mean()), 4
    ),
}
R["02_ezber_orani"] = ezber

# ------------------------------------------------- 3. KIRLILIK TESTI (ANA)
def yukle(b):
    z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    return np.c_[[np.mean([z[k] for k in z.files if k.endswith("_" + a)], axis=0)
                  for a in AILE]].T.astype(np.float64)


def kirp(x):
    return np.log1p(np.clip(np.expm1(x), 0, None))


def mse(r, w=None):
    if w is None:
        return float(np.mean(r * r))
    return float(np.sum(w * r * r) / np.sum(w))


def rmsle(r, w=None):
    return float(np.sqrt(mse(r, w)))


# test kohort agirligi: soguk test satirlarinin (kVA kovasi x ay) bilesimi
def kova_guc(g):
    kenar = [0, 50, 100, 160, 250, 400, 630, 1000, 1600, np.inf]
    return np.digitize(g, kenar) - 1


q_t = pd.Series(kova_guc(T.loc[T.soguk_mu == 1, "guc"].to_numpy())).value_counts(normalize=True)

kir = {}
detay = {}
for b in BLOKLAR:
    blk = E[E._blok == b]
    sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    P = yukle(b)
    assert len(P) == len(y), (b, P.shape, y.shape)
    kb = kova_guc(sog.guc.to_numpy())
    pay = pd.Series(kb).value_counts(normalize=True)
    w = np.array([q_t.get(k, 0.0) / pay.get(k, 1.0) for k in kb])
    w = w / w.mean()

    lg_cat = kirp(P @ WCAT)
    lg_311 = kirp(P @ W311)
    d = lg_311 - lg_cat
    m = maske[b]

    def olc(sel, wsel_kaynak):
        """sel: satir maskesi. Merkezleme SEL ICINDE yapilir (dogru kurgu)."""
        yy, cc, dd = y[sel], lg_cat[sel], d[sel]
        ww = w[sel] / w[sel].mean() if sel.sum() else w[sel]
        out = {}
        for ad, wt in (("ham", None), ("agr", ww)):
            mrk = dd.mean() if wt is None else float(np.sum(wt * dd) / np.sum(wt))
            r0 = yy - cc
            r1 = yy - kirp(cc + (dd - mrk))
            r2 = yy - lg_311[sel]  # merkezlemesiz (ham 3/1/1)
            out[ad] = {
                "n": int(sel.sum()),
                "rmsle_cat": round(rmsle(r0, wt), 6),
                "rmsle_311_seviyesiz": round(rmsle(r1, wt), 6),
                "rmsle_311_seviyeli": round(rmsle(r2, wt), 6),
                "KAZANC_yapi": round(rmsle(r0, wt) - rmsle(r1, wt), 6),
                "KAZANC_ham311": round(rmsle(r0, wt) - rmsle(r2, wt), 6),
            }
        return out

    tum = np.ones(len(y), dtype=bool)
    kir[b] = {
        "TUMU": olc(tum, w),
        "EZBERLENEBILIR": olc(m, w),
        "TEMIZ_ezberlenemez": olc(~m, w),
    }
    # ayrica: TUMU uzerinde merkezleme yapip alt kumelerde OLC (uretimdeki hal)
    mrk_tum = d.mean()
    r0 = y - lg_cat
    r1 = y - kirp(lg_cat + (d - mrk_tum))
    alt = {}
    for ad, sel in (("TUMU", tum), ("EZBERLENEBILIR", m), ("TEMIZ_ezberlenemez", ~m)):
        if sel.sum() == 0:
            alt[ad] = None
            continue
        ww = w[sel] / w[sel].mean()
        alt[ad] = {
            "n": int(sel.sum()),
            "ham_kazanc": round(rmsle(r0[sel]) - rmsle(r1[sel]), 6),
            "agr_kazanc": round(rmsle(r0[sel], ww) - rmsle(r1[sel], ww), 6),
        }
    detay[b] = {"GLOBAL_merkezleme_ile_altkume_kazanci": alt}

R["03_KIRLILIK_TESTI"] = kir
R["04_global_merkezlemeli_altkume"] = detay

# ---------------------------------------------- 5. tohum bazinda temiz alt kume
tohum = {}
for b in BLOKLAR:
    z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    tohumlar = sorted({k.rsplit("_", 1)[0] for k in z.files})
    blk = E[E._blok == b]
    sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    m = maske[b]
    tt = {}
    for t in tohumlar:
        P = np.c_[[z[f"{t}_{a}"] for a in AILE]].T.astype(np.float64)
        lg_cat, lg_311 = kirp(P @ WCAT), kirp(P @ W311)
        d = lg_311 - lg_cat
        cell = {}
        for ad, sel in (("TUMU", np.ones(len(y), bool)), ("EZBER", m), ("TEMIZ", ~m)):
            if sel.sum() == 0:
                cell[ad] = None
                continue
            dd = d[sel] - d[sel].mean()
            r0 = y[sel] - lg_cat[sel]
            r1 = y[sel] - kirp(lg_cat[sel] + dd)
            cell[ad] = round(rmsle(r0) - rmsle(r1), 6)
        tt[t] = cell
    tohum[b] = tt
R["05_tohum_bazinda_ham_kazanc"] = tohum

# ---------------------------------------------- 6. onyukleme (trafo kumeli)
rng = np.random.default_rng(7)
oy = {}
for b in BLOKLAR:
    blk = E[E._blok == b]
    sog = blk[blk.soguk_mu == 1].reset_index(drop=True)
    y = np.log1p(sog.tuketim.to_numpy(dtype="float64").clip(0))
    P = yukle(b)
    lg_cat, lg_311 = kirp(P @ WCAT), kirp(P @ W311)
    d = lg_311 - lg_cat
    m = maske[b]
    tan = sog.tanim.to_numpy()
    oy[b] = {}
    for ad, sel in (("EZBERLENEBILIR", m), ("TEMIZ_ezberlenemez", ~m)):
        if sel.sum() < 50:
            oy[b][ad] = None
            continue
        yy, cc, dd0, tt = y[sel], lg_cat[sel], d[sel], tan[sel]
        grup = pd.Series(np.arange(len(tt))).groupby(tt).apply(lambda s: s.to_numpy())
        gl = list(grup.values)
        vals = []
        for _ in range(500):
            pick = np.concatenate([gl[i] for i in rng.integers(0, len(gl), len(gl))])
            dd = dd0[pick] - dd0[pick].mean()
            r0 = yy[pick] - cc[pick]
            r1 = yy[pick] - kirp(cc[pick] + dd)
            vals.append(rmsle(r0) - rmsle(r1))
        vals = np.array(vals)
        oy[b][ad] = {
            "n_trafo": len(gl),
            "ort": round(float(vals.mean()), 6),
            "GA95": [round(float(np.percentile(vals, 2.5)), 6),
                     round(float(np.percentile(vals, 97.5)), 6)],
            "P_pozitif": round(float((vals > 0).mean()), 4),
        }
R["06_onyukleme_trafo_kumeli"] = oy

yol = os.path.join(KOK, "experiments/model29/p_kalici/p30_ezber.json")
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(R, fh, indent=1, ensure_ascii=False)
print(json.dumps({k: R[k] for k in ("01_kimlik_kanali", "02_ezber_orani")},
                 indent=1, ensure_ascii=False))
print("\n=== KIRLILIK TESTI (kazanc, + = 3/1/1 iyi) ===")
for b in BLOKLAR:
    for g in ("TUMU", "EZBERLENEBILIR", "TEMIZ_ezberlenemez"):
        v = kir[b][g]
        print(f"{b:7}{g:22}n={v['ham']['n']:>7}  ham_yapi={v['ham']['KAZANC_yapi']:+.5f}"
              f"  agr_yapi={v['agr']['KAZANC_yapi']:+.5f}"
              f"  ham311={v['ham']['KAZANC_ham311']:+.5f}")
print("\n=== ONYUKLEME ===")
print(json.dumps(oy, indent=1, ensure_ascii=False))
print("\n=== TOHUM ===")
print(json.dumps(tohum, indent=1, ensure_ascii=False))
