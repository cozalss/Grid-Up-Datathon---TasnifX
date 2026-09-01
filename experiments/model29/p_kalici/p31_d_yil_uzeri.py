"""p31_d -- ASIL DENEY: YIL-UZERI AYNI AY tasimasi.

Onerinin gercek kurgusu:
  CAPA(d,m) = 2025 yilinin m ayindan olculen ilce x ay ham-vekil etkisi
              (trafo ortalamasi + ortak gun etkisi cikarilmis, ay-ici merkezli).
  DUZELTME : yhat_2026 += lambda * CAPA(d, m)

DOGRULAMA (tamamen durust, blok-disi + yil-disi):
  hedef = kis26 blogunun Oca/Sub/Mar 2026 satirlari (GERCEK CV tahminleri)
  capa  = 2025 Oca/Sub/Mar (yalniz 2025 verisi -- 2026 hic kullanilmadi)

TAVAN/TASIMA ORANI:
  G_ic  = ayni yilin capasi (blok ici, ulasilamaz)
  G_dis = onceki yilin capasi (gercek kosul)
  rho   = G_dis / G_ic  -> yaz25'te olculen G_ic ile carpilarak TEST beklentisi

Iki capa ailesi:
  (a) DOGRUDAN ilce x ay
  (b) PARAMETRIK  beta(m) * tarim_c_d   (1 serbestlik/ay)

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
LAM = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0)
NBOOT = 500
RNG = np.random.default_rng(31)

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
    df["yil"] = df.tarih.dt.year
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

# ------------------------------------------------------- HAM VEKIL CAPALAR
ham = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"))
ham["tarih"] = pd.to_datetime(ham.tarih)
ham["k"] = ham.lokasyon.map(anahtar)
ham["yil"] = ham.tarih.dt.year
ham["ay"] = ham.tarih.dt.month
ham = ham[ham.tuketim > 0].copy()
ham["yl"] = np.log1p(ham.tuketim)


def capa_uret(yil, aylar, min_n=30):
    """Yalniz verilen yil+aylarin verisinden ilce x ay etkisi ve tarim egimi."""
    d = ham[(ham.yil == yil) & (ham.ay.isin(aylar))].copy()
    d["r"] = d.yl - d.groupby("tanim").yl.transform("mean")
    d["r"] = d.r - d.groupby("tarih").r.transform("mean")
    g = d.groupby(["k", "ay"]).r.agg(["mean", "size"]).reset_index()
    aym = d.groupby("ay").r.mean()
    g["b"] = g["mean"] - g.ay.map(aym)
    g.loc[g["size"] < min_n, "b"] = 0.0
    dogrudan = {(r.k, int(r.ay)): float(r.b) for r in g.itertuples()}
    beta = {}
    for a in aylar:
        s = g[g.ay == a].merge(arz[["k", "tarim_orani"]], on="k")
        x = s.tarim_orani.to_numpy() - s.tarim_orani.mean()
        beta[a] = float(np.cov(x, s.b)[0, 1] / np.var(x)) if len(s) > 10 else 0.0
    return dogrudan, beta


tar_ort = float(arz_i.reindex(sorted(set(E.k))).tarim_orani.mean())
E["tar_c"] = E.k.map(arz_i.tarim_orani) - tar_ort
T["tar_c"] = T.k.map(arz_i.tarim_orani) - tar_ort


def duzeltme_vek(alt, dogrudan, beta, tip):
    if tip == "dogrudan":
        return np.array([dogrudan.get((k, a), 0.0)
                         for k, a in zip(alt.k.to_numpy(), alt.ay.to_numpy())])
    return np.array([beta.get(a, 0.0) for a in alt.ay.to_numpy()]) * alt.tar_c.to_numpy()


R = {}

# ================================================ 1. DOGRULAMA: kis26 Oca-Mar
AY_V = [1, 2, 3]
hedef = ((E._blok == "kis26") & (E.ay.isin(AY_V))).to_numpy()
alt = E[hedef]
c25 = capa_uret(2025, AY_V)
c26 = capa_uret(2026, AY_V)

dog = {}
for tip in ("dogrudan", "parametrik"):
    for kaynak, cc in (("YIL_DISI_2025", c25), ("AYNI_YIL_2026_kahin", c26)):
        duz = duzeltme_vek(alt, cc[0], cc[1], tip)
        for ad, m2 in (("TUM", np.ones(len(alt), bool)),
                       ("SICAK", (alt.soguk_mu == 0).to_numpy()),
                       ("SOGUK", (alt.soguk_mu == 1).to_numpy())):
            ww = alt.w.to_numpy()[m2]
            yy = alt.y.to_numpy()[m2]
            per = {}
            for L in LAM:
                gz = []
                for t in TOHUM:
                    hh = YH[t][hedef][m2]
                    gz.append(mse(yy - kirp(hh), ww) - mse(yy - kirp(hh + L * duz[m2]), ww))
                per[str(L)] = round(float(np.mean(gz)), 6)
            dog[f"{tip}|{kaynak}|{ad}"] = per
R["12_DOGRULAMA_kis26_Oca_Mar"] = dog
R["12_DOGRULAMA_kis26_Oca_Mar"]["_kurulum"] = {
    "hedef_satir": int(hedef.sum()),
    "hedef_sicak": int((alt.soguk_mu == 0).sum()),
    "hedef_soguk": int((alt.soguk_mu == 1).sum()),
    "capa_kaynagi": "YALNIZ 2025 Oca-Mar ham verisi; 2026 hic kullanilmadi",
    "not": "AYNI_YIL_2026 satiri KAHIN (ulasilamaz ust sinir), tasima orani icin",
}

# ================================== 2. yaz25 blogunda ayni-yil kahin (test vekili)
AY_T = [4, 5, 6, 7]
hy = ((E._blok == "yaz25") & (E.ay.isin(AY_T))).to_numpy()
alty = E[hy]
cy = capa_uret(2025, AY_T)
yz = {}
for tip in ("dogrudan", "parametrik"):
    duz = duzeltme_vek(alty, cy[0], cy[1], tip)
    for ad, m2 in (("TUM", np.ones(len(alty), bool)),
                   ("SICAK", (alty.soguk_mu == 0).to_numpy()),
                   ("SOGUK", (alty.soguk_mu == 1).to_numpy())):
        ww, yy = alty.w.to_numpy()[m2], alty.y.to_numpy()[m2]
        per = {}
        for L in LAM:
            gz = [mse(yy - kirp(YH[t][hy][m2]), ww)
                  - mse(yy - kirp(YH[t][hy][m2] + L * duz[m2]), ww) for t in TOHUM]
            per[str(L)] = round(float(np.mean(gz)), 6)
        yz[f"{tip}|{ad}"] = per
R["13_yaz25_AYNI_YIL_KAHIN"] = yz
R["13_yaz25_AYNI_YIL_KAHIN"]["_not"] = (
    "Capa yaz25'in KENDI yilindan -- test'te bu MUMKUN DEGIL (2026 Nis-Tem yok). "
    "Yalniz odulun buyuklugunu ve tasima orani carpani icin taban.")

# ============================================== 3. ONYUKLEME (trafo kumeli)
def onyukleme(alt_df, hedef_maske, duz, L, m2, n=NBOOT):
    tan = alt_df.tanim.to_numpy()[m2]
    yy, ww = alt_df.y.to_numpy()[m2], alt_df.w.to_numpy()[m2]
    hh = np.mean([YH[t][hedef_maske][m2] for t in TOHUM], axis=0)
    r0 = (yy - kirp(hh)) ** 2
    r1 = (yy - kirp(hh + L * duz[m2])) ** 2
    u, inv = np.unique(tan, return_inverse=True)
    idx = [np.flatnonzero(inv == i) for i in range(len(u))]
    out = np.empty(n)
    for i in range(n):
        pick = RNG.integers(0, len(u), len(u))
        ii = np.concatenate([idx[p] for p in pick])
        sw = ww[ii]
        out[i] = (np.sum(sw * r0[ii]) - np.sum(sw * r1[ii])) / np.sum(sw)
    return out


ob = {}
for tip in ("dogrudan", "parametrik"):
    duz = duzeltme_vek(alt, c25[0], c25[1], tip)
    for ad, m2 in (("TUM", np.ones(len(alt), bool)),
                   ("SICAK", (alt.soguk_mu == 0).to_numpy()),
                   ("SOGUK", (alt.soguk_mu == 1).to_numpy())):
        for L in (0.3, 0.5, 0.7, 1.0):
            s = onyukleme(alt, hedef, duz, L, m2)
            ob[f"{tip}|{ad}|{L}"] = {
                "ort": round(float(s.mean()), 6),
                "GA95": [round(float(np.percentile(s, 2.5)), 6),
                         round(float(np.percentile(s, 97.5)), 6)],
                "P_pozitif": round(float((s > 0).mean()), 3)}
R["14_ONYUKLEME_dogrulama_blogu"] = ob

with open(os.path.join(CIK, "p31_d_ara.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
