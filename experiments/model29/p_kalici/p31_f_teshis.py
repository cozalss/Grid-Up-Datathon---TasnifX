"""p31_f -- TESHIS: ham-vekil ilce x ay etkisi, modelin ARTIK yanliligiyla
ayni sey mi? (Sulama hikayesinin can damari.)

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


def kirp(x):
    return np.log1p(np.clip(np.expm1(x), 0, None))


E = pd.read_parquet(os.path.join(DN, "egitim.parquet"),
                    columns=["tanim", "tarih", "lokasyon", "tuketim", "soguk_mu", "_blok"])
E["k"] = E.lokasyon.map(anahtar)
E["ay"] = E.tarih.dt.month
E["yil"] = E.tarih.dt.year
E["y"] = np.log1p(E.tuketim.clip(lower=0).to_numpy(dtype="float64"))

zs = np.load(os.path.join(DN, "sicak_tahmin.npz"))
yh = np.full(len(E), np.nan)
for b in BLOKLAR:
    sel = (E._blok == b).to_numpy()
    sog = (E.loc[sel, "soguk_mu"] == 1).to_numpy()
    zc = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    v = np.full(sel.sum(), np.nan)
    v[~sog] = np.mean([np.c_[[zs[f"{b}_{t}_{a}"] for a in AILE]].T @ W_SICAK
                       for t in TOHUM], axis=0)
    v[sog] = np.mean([np.c_[[zc[f"{t}_{a}"] for a in AILE]].T @ W_SOGUK
                      for t in TOHUM], axis=0)
    yh[sel] = v
E["res"] = E.y - kirp(yh)

# model artik yanliligi (ilce x ay, ay-ici merkezli)
gm = E.groupby(["k", "ay"]).res.mean()
aym = E.groupby("ay").res.mean()
model_b = (gm - gm.index.get_level_values("ay").map(aym)).rename("model")

# ham vekil etkisi, AYNI aylardan (2025-04..2026-03)
ham = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"))
ham["tarih"] = pd.to_datetime(ham.tarih)
ham = ham[(ham.tuketim > 0) & (ham.tarih >= "2025-04-01")].copy()
ham["k"] = ham.lokasyon.map(anahtar)
ham["ay"] = ham.tarih.dt.month
ham["yl"] = np.log1p(ham.tuketim)
ham["r"] = ham.yl - ham.groupby("tanim").yl.transform("mean")
ham["r"] = ham.r - ham.groupby("tarih").r.transform("mean")
gh = ham.groupby(["k", "ay"]).r.mean()
ham_b = (gh - gh.index.get_level_values("ay").map(ham.groupby("ay").r.mean())).rename("ham")

s = pd.concat([model_b, ham_b], axis=1).dropna()
R = {
    "15_TESHIS_ham_vekil_vs_model_artigi": {
        "n_hucre": int(len(s)),
        "std_model_artigi": round(float(s.model.std()), 4),
        "std_ham_vekil": round(float(s.ham.std()), 4),
        "korelasyon": round(float(np.corrcoef(s.model, s.ham)[0, 1]), 3),
        "R2": round(float(np.corrcoef(s.model, s.ham)[0, 1] ** 2), 4),
        "egim_model_uzerine_ham": round(float(np.cov(s.ham, s.model)[0, 1]
                                              / np.var(s.ham)), 3),
        "ANLAM": ("Ham-vekil ilce x ay etkisi (p29'un sulama imzasi) ile MODELIN "
                  "artik yanliligi ayni nesne DEGILSE, sulama capasi modelin "
                  "hatasini duzeltemez."),
    }
}
with open(os.path.join(CIK, "p31_f_ara.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
