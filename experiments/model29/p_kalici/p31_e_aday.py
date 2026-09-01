"""p31_e -- ADAY URETIMI: 2025 Nis-Tem capasi -> test (2026 Nis-Tem).

Taban: submissions/tuketim_YP_seviye.csv (olculmus LB 1.00115)
Delta: log1p uzayinda  lambda * CAPA(ilce, ay)
Varyantlar: dogrudan / parametrik  x  lambda in {0.3, 0.5, 0.7, 1.0}
Ek: p08 olu trafo x0.50 deltasi bindirilmis surumler.

Gonderim YOK. submissions/ yazma YOK. Cikti: p_kalici/aday_csv/p31_sulama_*.csv
"""
import json
import os
import unicodedata

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
CIK = os.path.join(KOK, "experiments/model29/p_kalici")
ADAY = os.path.join(CIK, "aday_csv")
AY_T = [4, 5, 6, 7]
LAM = (0.3, 0.5, 0.7, 1.0)

TR = str.maketrans({"İ": "I", "I": "I", "ı": "i", "Ş": "S", "ş": "s", "Ğ": "G",
                    "ğ": "g", "Ü": "U", "ü": "u", "Ö": "O", "ö": "o", "Ç": "C",
                    "ç": "c", "Â": "A", "â": "a"})


def norm(s):
    s = unicodedata.normalize("NFKD", str(s).translate(TR))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip().replace(" ", "")


def anahtar(lok):
    p = str(lok).split(">")
    return norm(p[0]) + "|" + norm(p[-1])


arz = pd.read_parquet(os.path.join(KOK, "data/external/arazi_ortusu_ilce.parquet"))
arz["k"] = arz.il_key.map(norm) + "|" + arz.ilce_key.map(norm)
arz_i = arz.set_index("k")

ham = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"))
ham["tarih"] = pd.to_datetime(ham.tarih)
ham = ham[(ham.tuketim > 0) & (ham.tarih.dt.year == 2025)
          & (ham.tarih.dt.month.isin(AY_T))].copy()
ham["k"] = ham.lokasyon.map(anahtar)
ham["ay"] = ham.tarih.dt.month
ham["yl"] = np.log1p(ham.tuketim)
ham["r"] = ham.yl - ham.groupby("tanim").yl.transform("mean")
ham["r"] = ham.r - ham.groupby("tarih").r.transform("mean")
g = ham.groupby(["k", "ay"]).r.agg(["mean", "size"]).reset_index()
g["b"] = g["mean"] - g.ay.map(ham.groupby("ay").r.mean())
g.loc[g["size"] < 30, "b"] = 0.0
dogrudan = {(r.k, int(r.ay)): float(r.b) for r in g.itertuples()}
beta = {}
for a in AY_T:
    s = g[g.ay == a].merge(arz[["k", "tarim_orani"]], on="k")
    x = s.tarim_orani.to_numpy() - s.tarim_orani.mean()
    beta[a] = float(np.cov(x, s.b)[0, 1] / np.var(x))

T = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
T["tarih"] = pd.to_datetime(T.tarih)
T["k"] = T.lokasyon.map(anahtar)
T["ay"] = T.tarih.dt.month
tar_ort = float(pd.Series(sorted(set(T.k))).map(arz_i.tarim_orani).mean())
T["tar_c"] = T.k.map(arz_i.tarim_orani) - tar_ort
assert T.tar_c.notna().all()

D = {
    "dogrudan": np.array([dogrudan.get((k, a), 0.0)
                          for k, a in zip(T.k.to_numpy(), T.ay.to_numpy())]),
    "parametrik": np.array([beta[a] for a in T.ay.to_numpy()]) * T.tar_c.to_numpy(),
}

taban = pd.read_csv(os.path.join(KOK, "submissions/tuketim_YP_seviye.csv"))
assert (taban.id.to_numpy() == T.id.to_numpy()).all()
lg0 = np.log1p(taban.tuketim.to_numpy(dtype="float64"))
p08 = np.load(os.path.join(ADAY, "p08_olu_delta_log_c050.npy"))
assert len(p08) == len(lg0)

os.makedirs(ADAY, exist_ok=True)
rap = {"capa_beta_2025": {str(a): round(beta[a], 4) for a in AY_T},
       "capa_std": {str(a): round(float(g[g.ay == a].b.std()), 4) for a in AY_T},
       "dosyalar": {}}
for tip, dv in D.items():
    for L in LAM:
        for olu in (False, True):
            lg = np.maximum(lg0 + L * dv + (p08 if olu else 0.0), 0.0)
            v = np.clip(np.expm1(lg), 0, None)
            assert np.isfinite(v).all() and (v >= 0).all()
            ad = f"p31_sulama_{tip}_l{int(L * 100):03d}" + ("_olu50" if olu else "")
            yol = os.path.join(ADAY, ad + ".csv")
            pd.DataFrame({"id": taban.id, "tuketim": v}).to_csv(yol, index=False)
            rap["dosyalar"][ad] = {
                "satir": int(len(v)),
                "degisen_satir": int((np.abs(lg - lg0) > 1e-12).sum()),
                "log_kayma_ort": round(float((lg - lg0).mean()), 6),
                "log_kayma_std": round(float((lg - lg0).std()), 6),
                "log_kayma_aralik": [round(float((lg - lg0).min()), 4),
                                     round(float((lg - lg0).max()), 4)],
                "nan": int(np.isnan(v).sum()), "negatif": int((v < 0).sum()),
            }
            print(ad, rap["dosyalar"][ad])

with open(os.path.join(CIK, "p31_e_ara.json"), "w", encoding="utf-8") as f:
    json.dump(rap, f, ensure_ascii=False, indent=1)
