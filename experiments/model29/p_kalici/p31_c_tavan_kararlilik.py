"""p31_c -- (1) KAHIN TAVANI: mukemmel ilce x ay duzeltmesi ne kazandirir?
           (2) YIL-UZERI KARARLILIK: Oca-Mar 2025 vs Oca-Mar 2026 (ham vekil)
           (3) test bilesimi aritmetigi

Egitim YOK. Gonderim YOK. submissions/ yazma YOK.
"""
import json
import os
import unicodedata

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
CIK = os.path.join(KOK, "experiments/model29/p_kalici")
BLOKLAR = ("yaz25", "guz25", "kis26")
AILE = ("cat", "xgb", "lgbm")
TOHUM = (1000, 1001, 1002)
W_SICAK = np.array([0.6, 0.2, 0.2])
W_SOGUK = np.array([1.0, 0.0, 0.0])

TR = str.maketrans({"İ": "I", "I": "I", "ı": "i", "Ş": "S", "ş": "s", "Ğ": "G",
                    "ğ": "g", "Ü": "U", "ü": "u", "Ö": "O", "ö": "o", "Ç": "C",
                    "ç": "c", "Â": "A", "â": "a"})


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).translate(TR))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip().replace(" ", "")


def anahtar(lok):
    p = str(lok).split(">")
    return norm(p[0]) + "|" + norm(p[-1])


def kova_guc(g):
    return np.digitize(g, [0, 50, 100, 160, 250, 400, 630, 1000, 1600, np.inf]) - 1


def kirp(x):
    return np.log1p(np.clip(np.expm1(x), 0, None))


def mse(r, ww):
    return float(np.sum(ww * r * r) / np.sum(ww))


E = pd.read_parquet(os.path.join(DN, "egitim.parquet"),
                    columns=["tanim", "tarih", "lokasyon", "guc", "tuketim",
                             "soguk_mu", "_blok"])
T = pd.read_parquet(os.path.join(DN, "test.parquet"),
                    columns=["tanim", "tarih", "lokasyon", "guc", "soguk_mu"])
arz = pd.read_parquet(os.path.join(KOK, "data/external/arazi_ortusu_ilce.parquet"))
arz["k"] = arz.il_key.map(norm) + "|" + arz.ilce_key.map(norm)
arz_i = arz.set_index("k")
for df in (E, T):
    df["k"] = df.lokasyon.map(anahtar)
    df["ay"] = df.tarih.dt.month
    df["kova"] = kova_guc(df.guc.to_numpy())
E["y"] = np.log1p(E.tuketim.clip(lower=0).to_numpy(dtype="float64"))

pay_t = T.groupby(["soguk_mu", "kova"]).size() / len(T)
w = np.ones(len(E))
for b in BLOKLAR:
    sel = (E._blok == b).to_numpy()
    pe = E.loc[sel].groupby(["soguk_mu", "kova"]).size() / sel.sum()
    key = pd.MultiIndex.from_arrays([E.loc[sel, "soguk_mu"], E.loc[sel, "kova"]])
    ww = np.array([pay_t.get(kk, 0.0) / max(pe.get(kk, 1e-12), 1e-12) for kk in key])
    w[sel] = ww / ww.mean()
E["w"] = w

zs = np.load(os.path.join(DN, "sicak_tahmin.npz"))
ZC = {b: np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz")) for b in BLOKLAR}
YH = {}
for t in TOHUM:
    yh = np.full(len(E), np.nan)
    for b in BLOKLAR:
        sel = (E._blok == b).to_numpy()
        sog = (E.loc[sel, "soguk_mu"] == 1).to_numpy()
        v = np.full(sel.sum(), np.nan)
        v[~sog] = np.c_[[zs[f"{b}_{t}_{a}"] for a in AILE]].T @ W_SICAK
        v[sog] = np.c_[[ZC[b][f"{t}_{a}"] for a in AILE]].T @ W_SOGUK
        yh[sel] = v
    YH[t] = yh

kser, blok, ayv = E.k.to_numpy(), E._blok.to_numpy(), E.ay.to_numpy()
sicak_m = (E.soguk_mu == 0).to_numpy()
yv, wv = E.y.to_numpy(), E.w.to_numpy()
R = {}

# ------------------------------------------------------- 1. KAHIN TAVANI
tav = {}
for b in BLOKLAR:
    hedef = blok == b
    for ad, m2 in (("TUM", np.ones(hedef.sum(), bool)),
                   ("SICAK", sicak_m[hedef]), ("SOGUK", ~sicak_m[hedef])):
        gz, gs = [], []
        for t in TOHUM:
            yh = YH[t][hedef][m2]
            yy, ww = yv[hedef][m2], wv[hedef][m2]
            res = yy - kirp(yh)
            key = pd.Series(kser[hedef][m2] + "_" + ayv[hedef][m2].astype(str))
            keyi = pd.Series(kser[hedef][m2])
            b_dm = key.map(pd.DataFrame({"k": key, "r": res}).groupby("k").r.mean()).to_numpy()
            b_d = keyi.map(pd.DataFrame({"k": keyi, "r": res}).groupby("k").r.mean()).to_numpy()
            t0 = mse(res, ww)
            gz.append(t0 - mse(yy - kirp(yh + b_dm), ww))
            gs.append(t0 - mse(yy - kirp(yh + b_d), ww))
        tav[f"{b}_{ad}"] = {"kahin_ilce_x_ay": round(float(np.mean(gz)), 6),
                            "kahin_ilce_sabiti": round(float(np.mean(gs)), 6),
                            "n": int(m2.sum())}
R["09_KAHIN_TAVANI_blok_ICI"] = tav
R["09_KAHIN_TAVANI_blok_ICI"]["_not"] = ("Blok ICI kahin -- ulasilamaz ust sinir. "
                                         "Gercek kazanc bunun kucuk bir kesri olur.")

# ---------------------------------------------- 2. YIL-UZERI KARARLILIK
# Ham vekil: log1p(tuketim) - trafo ortalamasi - ortak gun etkisi
ham = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"))
ham["tarih"] = pd.to_datetime(ham.tarih)
ham["k"] = ham.lokasyon.map(anahtar)
ham["y"] = np.log1p(ham.tuketim.clip(lower=0))
ham = ham[ham.tuketim > 0].copy()
ham["r"] = ham.y - ham.groupby("tanim").y.transform("mean")
ham["r"] = ham.r - ham.groupby("tarih").r.transform("mean")
ham["ym"] = ham.tarih.dt.to_period("M").astype(str)
ham["ay"] = ham.tarih.dt.month
ham["tar"] = ham.k.map(arz_i.tarim_orani)

g = ham.groupby(["k", "ym"]).r.agg(["mean", "size"]).reset_index()
g = g[g["size"] >= 30]
g["b"] = g["mean"] - g.ym.map(ham.groupby("ym").r.mean())
piv = g.pivot(index="k", columns="ym", values="b")

kar = []
for m, a, bb in ((1, "2025-01", "2026-01"), (2, "2025-02", "2026-02"),
                 (3, "2025-03", "2026-03")):
    s = piv[[a, bb]].dropna()
    x, yy = s[a].to_numpy(), s[bb].to_numpy()
    kar.append({"ay": m, "n_ilce": len(s),
                "korelasyon": round(float(np.corrcoef(x, yy)[0, 1]), 3),
                "egim_2026_uzerine_2025": round(float(np.cov(x, yy)[0, 1] / np.var(x)), 3),
                "std_2025": round(float(x.std()), 3), "std_2026": round(float(yy.std()), 3)})
R["10_YIL_UZERI_ILCE_VEKTORU"] = {"aylik": kar,
    "ortalama_egim": round(float(np.mean([k["egim_2026_uzerine_2025"] for k in kar])), 3),
    "yontem": "ham vekil (trafo + ortak gun cikarilmis), 2025 Oca-Mar deney penceresinde YOK"}

# tarim egiminin yil-uzeri kararliligi (parametrik nesnenin kendisi)
be = {}
for ym in sorted(ham.ym.unique()):
    s = g[g.ym == ym].merge(arz[["k", "tarim_orani"]], on="k")
    if len(s) < 20:
        continue
    x = s.tarim_orani.to_numpy() - s.tarim_orani.mean()
    be[ym] = round(float(np.cov(x, s.b)[0, 1] / np.var(x)), 4)
R["11_YIL_UZERI_TARIM_EGIMI_beta"] = {
    "aylik_beta": be,
    "cift_2025_vs_2026": {m: [be.get(f"2025-{m}"), be.get(f"2026-{m}")]
                          for m in ("01", "02", "03")},
    "not": "Nisan-Temmuz beta'si YALNIZ 2025'te var -- yil-uzeri dogrulanamaz.",
}

with open(os.path.join(CIK, "p31_c_ara.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
