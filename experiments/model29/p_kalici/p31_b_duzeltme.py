"""p31_b -- ilce yanliligi duzeltmesi: blok-disi olcum.

(a) ILCE SABITI: bias_d = diger bloklarin ortalama artigi -> lambda ile uygula
(b) PARAMETRIK HARMONIK: bias(d,m) = beta(m) * tarim_c_d,
    beta(m) = a + b*sin(2pi m/12) + c*cos(2pi m/12), kaynak bloklarin
    aylarindan fit, hedef blogun aylarina EKSTRAPOLE. 3 serbestlik.
(c) beta(m) tablosu: sulama rampasinin dogrudan olcumu.

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
LAM = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)

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
    kenar = [0, 50, 100, 160, 250, 400, 630, 1000, 1600, np.inf]
    return np.digitize(g, kenar) - 1


def kirp(x):
    return np.log1p(np.clip(np.expm1(x), 0, None))


def mse(r, ww):
    return float(np.sum(ww * r * r) / np.sum(ww))


# ------------------------------------------------------------------ VERI
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
    df["tar"] = df.k.map(arz_i.tarim_orani)
    assert df.tar.notna().all()
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
        Q = np.c_[[ZC[b][f"{t}_{a}"] for a in AILE]].T
        assert len(Q) == sog.sum()
        v[sog] = Q @ W_SOGUK
        yh[sel] = v
    assert np.isfinite(yh).all()
    YH[t] = yh

kser = E.k.to_numpy()
blok = E._blok.to_numpy()
ayv = E.ay.to_numpy()
sicak_m = (E.soguk_mu == 0).to_numpy()
yv, wv, tarv = E.y.to_numpy(), E.w.to_numpy(), E.tar.to_numpy()
tar_c = tarv - float(np.average(T.tar.to_numpy()))

R = {}


def ilce_sabiti(res, sel, min_n=200):
    d = pd.DataFrame({"k": kser[sel], "r": res[sel]})
    g = d.groupby("k").r.agg(["mean", "size"])
    g["mean"] -= d.r.mean()
    g.loc[g["size"] < min_n, "mean"] = 0.0
    return g["mean"]


def harmonik(m):
    m = np.asarray(m, dtype=float)
    return np.c_[np.ones_like(m), np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)]


def olc(hedef, duz, yh, etiket, kutu):
    for ad, m2 in (("TUM", np.ones(hedef.sum(), bool)),
                   ("SICAK", sicak_m[hedef]), ("SOGUK", ~sicak_m[hedef])):
        yy, hh, dd, ww = yv[hedef][m2], yh[hedef][m2], duz[m2], wv[hedef][m2]
        taban = mse(yy - kirp(hh), ww)
        for L in LAM:
            kutu.setdefault(f"{etiket}|{ad}|{L}", []).append(
                taban - mse(yy - kirp(hh + L * dd), ww))


KA, KB, BETA, GAM = {}, {}, {}, {}
for t in TOHUM:
    yh = YH[t]
    res = yv - kirp(yh)
    for b in BLOKLAR:
        hedef = blok == b
        kaynak = (~hedef) & sicak_m          # ezber kanali yok: yalniz SICAK
        # ---- (a) ilce sabiti
        bd = ilce_sabiti(res, kaynak)
        duz_a = pd.Series(kser[hedef]).map(bd).fillna(0.0).to_numpy()
        olc(hedef, duz_a, yh, b, KA)
        # ---- (b) harmonik parametrik: ay bazli tarim egimi
        rr = res - pd.Series(kser).map(bd).fillna(0.0).to_numpy()
        dfk = pd.DataFrame({"ay": ayv[kaynak], "x": tar_c[kaynak], "r": rr[kaynak]})
        eg = dfk.groupby("ay").apply(
            lambda g: float(np.cov(g.x, g.r)[0, 1] / max(np.var(g.x), 1e-12)))
        aylar = eg.index.to_numpy()
        X = harmonik(aylar)
        cf = np.linalg.lstsq(X, eg.to_numpy(), rcond=None)[0]
        beta_hedef = harmonik(np.arange(1, 13)) @ cf
        duz_b = duz_a + beta_hedef[ayv[hedef] - 1] * tar_c[hedef]
        olc(hedef, duz_b, yh, b, KB)
        BETA.setdefault(b, []).append({int(a): round(float(v), 4) for a, v in eg.items()})
        GAM.setdefault(b, []).append([round(float(x), 4) for x in cf])
        GAM.setdefault(b + "_ekstrapole", []).append(
            {int(m): round(float(beta_hedef[m - 1]), 4)
             for m in sorted(set(ayv[hedef]))})

for ad, kutu in (("06_a_ILCE_SABITI_blokdisi_dMSE", KA),
                 ("07_b_ILCE_SABITI_ARTI_HARMONIK_TARIM_dMSE", KB)):
    R[ad] = {f"{b}_{s}": {str(L): round(float(np.mean(kutu[f"{b}|{s}|{L}"])), 6)
                          for L in LAM}
             for b in BLOKLAR for s in ("TUM", "SICAK", "SOGUK")}

R["08_beta_ay_tarim_egimi"] = {
    b: {"kaynak_aylar_gozlenen": BETA[b][0],
        "harmonik_katsayi_a_sin_cos": GAM[b][0],
        "hedef_aylara_ekstrapole": GAM[b + "_ekstrapole"][0]} for b in BLOKLAR}

with open(os.path.join(CIK, "p31_b_ara.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
