"""p31_a -- arazi_ortusu ilce anahtarini duzelt + ilce x ay yanliligini olc.

Egitim YOK. Gonderim YOK. submissions/ altina yazma YOK.
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
W_SICAK = np.array([0.6, 0.2, 0.2])   # uretim: cat 3 / xgb 1 / lgbm 1 (sinir_agi yok)
W_SOGUK = np.array([1.0, 0.0, 0.0])   # uretim: soguk = yalniz cat

TR = str.maketrans({"İ": "I", "I": "I", "ı": "i", "Ş": "S", "ş": "s", "Ğ": "G",
                    "ğ": "g", "Ü": "U", "ü": "u", "Ö": "O", "ö": "o", "Ç": "C",
                    "ç": "c", "Â": "A", "â": "a"})


def norm(s):
    s = str(s).translate(TR)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.lower().strip().replace(" ", "")


def ilce_ayikla(lok):
    return norm(str(lok).split(">")[-1])


def il_ayikla(lok):
    return norm(str(lok).split(">")[0])


R = {"00_BASLIK": "p31 sulama / ilce x ay ekseni",
     "_meta": {"kural": "Kaggle gonderimi YOK, submissions/ yazilmadi, commit yok"}}

# ------------------------------------------------------------- 1. ANAHTAR
arz = pd.read_parquet(os.path.join(KOK, "data/external/arazi_ortusu_ilce.parquet"))
arz["k"] = arz.il_key.map(norm) + "|" + arz.ilce_key.map(norm)

E = pd.read_parquet(os.path.join(DN, "egitim.parquet"),
                    columns=["tanim", "tarih", "lokasyon", "guc", "tuketim",
                             "soguk_mu", "_blok"])
T = pd.read_parquet(os.path.join(DN, "test.parquet"),
                    columns=["tanim", "tarih", "lokasyon", "guc", "soguk_mu"])
for df in (E, T):
    df["ilce"] = df.lokasyon.map(ilce_ayikla)
    df["il"] = df.lokasyon.map(il_ayikla)
    df["k"] = df.il + "|" + df.ilce
    df["ay"] = df.tarih.dt.month

ilceler = sorted(set(E.k.unique()) | set(T.k.unique()))
eslesen = [k for k in ilceler if k in set(arz.k)]
R["01_ANAHTAR"] = {
    "veri_ilce_sayisi": len(ilceler),
    "arazi_ortusu_satir": int(len(arz)),
    "eslesen": len(eslesen),
    "eslesmeyen": sorted(set(ilceler) - set(eslesen)),
    "p29_eslesen": 18,
    "yontem": "lokasyon son segmenti + il oneki, TR harf ceviri + NFKD",
}
assert len(eslesen) == len(ilceler), R["01_ANAHTAR"]

arz_i = arz.set_index("k")
for df in (E, T):
    for c in ("tarim_orani", "yerlesim_orani", "agac_orani", "otlak_orani",
              "bitki_ortusu_orani"):
        df[c] = df.k.map(arz_i[c])
    assert df[["tarim_orani"]].notna().all().all()

# --------------------------------------------- 2. TAHMINLER (uretim harmani)
def tahmin(b):
    """blok b icin uretim harmani log1p tahmini, egitim satir sirasinda."""
    blk = E._blok.to_numpy() == b
    out = np.full(blk.sum(), np.nan)
    sog = E.loc[blk, "soguk_mu"].to_numpy() == 1
    zs = np.load(os.path.join(DN, "sicak_tahmin.npz"))
    P = np.c_[[np.mean([zs[k] for k in zs.files
                        if k.startswith(b + "_") and k.endswith("_" + a)], axis=0)
               for a in AILE]].T
    out[~sog] = P @ W_SICAK
    zc = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    Q = np.c_[[np.mean([zc[k] for k in zc.files if k.endswith("_" + a)], axis=0)
               for a in AILE]].T
    assert len(Q) == sog.sum(), (b, Q.shape, sog.sum())
    out[sog] = Q @ W_SOGUK
    assert np.isfinite(out).all()
    return out


E["yhat"] = np.nan
for b in BLOKLAR:
    E.loc[E._blok == b, "yhat"] = tahmin(b)
E["y"] = np.log1p(E.tuketim.clip(lower=0).to_numpy(dtype="float64"))
E["res"] = E.y - E.yhat

# ------------------------------------- 3. ILCE x AY YANLILIGI (model artigi)
def yanlilik(alt, min_n=30):
    """(ilce, ay) ortalama artik; ay-ici global ortalama cikarilmis."""
    g = alt.groupby(["k", "ay"]).res.agg(["mean", "size"]).reset_index()
    g = g[g["size"] >= min_n]
    aym = alt.groupby("ay").res.mean()
    g["b"] = g["mean"] - g.ay.map(aym)
    return g


sicak = E[E.soguk_mu == 0]
etki = {}
for b in BLOKLAR:
    for ad, alt in (("TUM", E[E._blok == b]), ("SICAK", sicak[sicak._blok == b])):
        g = yanlilik(alt)
        gg = g.merge(arz[["k", "tarim_orani", "yerlesim_orani"]], on="k")
        # ilce basina yaz/kis genligi yerine: blok icinde ilce ortalamasi
        d = gg.groupby("k").agg(b=("b", "mean"), tar=("tarim_orani", "first"),
                                yer=("yerlesim_orani", "first"))
        etki[f"{b}_{ad}"] = {
            "n_hucre": int(len(g)),
            "n_ilce": int(g.k.nunique()),
            "ilce_x_ay_std": round(float(g.b.std()), 4),
            "ilce_ort_std": round(float(d.b.std()), 4),
            "kor_tarim_orani": round(float(np.corrcoef(d.b, d.tar)[0, 1]), 3),
            "kor_yerlesim_orani": round(float(np.corrcoef(d.b, d.yer)[0, 1]), 3),
        }
R["02_ILCE_x_AY_MODEL_ARTIGI"] = etki

# ------------------------- 4. SULAMA IMZASI: yaz-kis farki, 47 ilce, ARTIK
# yaz = 6,7,8 ; kis = 12,1,2 (p29 ile ayni tanim), model artigi uzerinde
gm = E.groupby(["k", "ay"]).res.mean().unstack()
aym = E.groupby("ay").res.mean()
gm = gm.sub(aym, axis=1)
yazk = gm[[6, 7, 8]].mean(axis=1) - gm[[12, 1, 2]].mean(axis=1)
gms = sicak.groupby(["k", "ay"]).res.mean().unstack()
gms = gms.sub(sicak.groupby("ay").res.mean(), axis=1)
yazk_s = gms[[6, 7, 8]].mean(axis=1) - gms[[12, 1, 2]].mean(axis=1)

tar = arz_i.reindex(yazk.index)
kor = {}
for ad, v in (("TUM", yazk), ("SICAK", yazk_s)):
    kor[ad] = {"n_ilce": int(v.notna().sum()), "std": round(float(v.std()), 4)}
    for c in ("tarim_orani", "yerlesim_orani", "agac_orani", "agac_yerlesim_orani",
              "otlak_orani", "ciplak_orani", "bitki_ortusu_orani"):
        m = v.notna() & tar[c].notna()
        kor[ad]["kor_" + c] = round(float(np.corrcoef(v[m], tar.loc[m, c])[0, 1]), 3)
    s = v.dropna().sort_values()
    kor[ad]["en_negatif"] = [f"{i.split('|')[1]} {x:+.3f}" for i, x in s.head(5).items()]
    kor[ad]["en_pozitif"] = [f"{i.split('|')[1]} {x:+.3f}" for i, x in s.tail(5).items()]
R["03_SULAMA_IMZASI_47_ILCE"] = kor
R["03_SULAMA_IMZASI_47_ILCE"]["p29_karsilastirma"] = {
    "p29_n_ilce": 18, "p29_kor_tarim": 0.541, "p29_std": 0.241,
    "p29_taban": "trafo+ortak gun cikarilmis HAM artik (model tahmini DEGIL)",
    "p31_taban": "URETIM harman tahminine gore model artigi",
}

os.makedirs(CIK, exist_ok=True)
with open(os.path.join(CIK, "p31_sulama.json"), "w", encoding="utf-8") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
