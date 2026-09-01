"""p27 ortak: URETIM harmanli blok kurucu (sicak 3/1/1/1.4 + sinir_agi, soguk cat-tekil,
son islem beta=0.60 soguk buzme).  Kalibrasyon: docs/80.
"""
import os
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
AO = os.path.join(KOK, "data/interim/aile_onbellek")
DN = os.path.join(KOK, "data/interim/deney")
HEDEF_SOGUK = 0.2216
BETA = 0.60

# uretim harman agirliklari (scripts/tuketim_model.py REJIM_AYARLARI "sicak")
W_SICAK = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}

_E = None


def egitim():
    global _E
    if _E is None:
        _E = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
    return _E


def blok(bad, soguk_harman="cat", son_islem=True):
    """soguk_harman: 'cat' (URETIM) veya 'esit' (tezgah) veya '311'."""
    E = egitim()
    blk = E[E._blok == bad]
    sic = blk[blk.soguk_mu == 0]
    sog = blk[blk.soguk_mu == 1]

    # --- SICAK: agirlikli aile harmani, tohum ortalamasi
    tohumlar = [1000, 1001, 1002]
    num = np.zeros(len(sic), dtype=np.float64)
    den = 0.0
    for a, w in W_SICAK.items():
        akum, n = np.zeros(len(sic)), 0
        for t in tohumlar:
            yol = os.path.join(AO, f"{bad}_{t}_{a}_uretim.npy")
            if os.path.exists(yol):
                akum += np.load(yol).astype(np.float64)
                n += 1
        assert n > 0, (bad, a)
        num += w * (akum / n)
        den += w
    p_sic = num / den

    # --- SOGUK: npz
    z = np.load(os.path.join(DN, f"soguk_tahmin_{bad}.npz"))
    aile = {}
    for a in ("cat", "xgb", "lgbm"):
        ks = [k for k in z.files if k.endswith("_" + a)]
        aile[a] = np.mean([z[k] for k in ks], axis=0)
    if soguk_harman == "cat":
        p_sog = aile["cat"]
    elif soguk_harman == "esit":
        p_sog = np.mean([aile["cat"], aile["xgb"], aile["lgbm"]], axis=0)
    elif soguk_harman == "311":
        p_sog = (3 * aile["cat"] + aile["xgb"] + aile["lgbm"]) / 5.0
    else:
        raise ValueError(soguk_harman)

    if son_islem:  # kapasite ofset uzayinda buzme
        lg = np.log1p(sog.guc.values.astype(np.float64))
        r = p_sog - lg
        p_sog = r.mean() + BETA * (r - r.mean()) + lg

    idx = np.concatenate([sic.index.values, sog.index.values])
    d = E.loc[idx].copy()
    d["p"] = np.concatenate([p_sic, p_sog])
    d["y"] = np.log1p(d.tuketim.values.astype(np.float64))
    d["r"] = d.y - d.p
    d["ay"] = pd.to_datetime(d.tarih).dt.month
    d["hg"] = pd.to_datetime(d.tarih).dt.dayofweek
    for a in ("cat", "xgb", "lgbm"):
        col = np.full(len(d), np.nan)
        col[len(sic):] = aile[a]
        d["sog_" + a] = col
    return d


def agirlik(d):
    """test kohort bilesimine (soguk %22.16) agirliklandirma."""
    sg = d.soguk_mu.values.astype(np.float64)
    pay = sg.mean()
    w = np.where(sg == 1, HEDEF_SOGUK / pay, (1 - HEDEF_SOGUK) / (1 - pay))
    return w / w.mean()


def rmsle(y, p, w=None):
    r = np.asarray(y) - np.asarray(p)
    if w is None:
        return float(np.sqrt(np.mean(r * r)))
    return float(np.sqrt(np.sum(w * r * r) / np.sum(w)))
