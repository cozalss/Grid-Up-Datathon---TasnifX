"""p25-B KIRMIZI TAKIM: afin kopru s=0.8132'yi BAGIMSIZ yeniden olc.

Sorular:
 1. s farkli yontemlerle tutarli mi? (kova-merkezli 3-kolon, tek-kolon,
    TRAFO-ici, trafolar-arasi, Theil-Sen benzeri saglam egim)
 2. Zincir soguk satirlarda gercekten AFIN mi? (kova basina egim sabit mi,
    trafo-ici R2 ne)
 3. Uygulanan delta dd'nin varyansi trafo-ICI mi trafolar-ARASI mi?
    (hangi s'nin gecerli oldugunu bu belirler)
 4. cat kolonundaki tohum gurultusu egimi ne kadar SONDURUYOR (attenuation)?
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
AC = os.path.join(PK, "aday_csv")
KVA_KENAR = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
W311 = np.array([0.6, 0.2, 0.2])

R = {}

A = np.load(os.path.join(AC, "p06_test_soguk_aile.npy")).astype("float64")
soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))
te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), usecols=["id", "tanim", "guc"], dtype={"tanim": str}
)
guc_s = te["guc"].to_numpy("float64")[soguk]
tanim_s = te["tanim"].to_numpy()[soguk]
sub = pd.read_csv(os.path.join(KOK, "submissions/tuketim_YP_seviye.csv"))
L = np.log1p(sub["tuketim"].to_numpy("float64"))
L_s = L[soguk]
kova = np.digitize(guc_s, KVA_KENAR) - 1

lguc = np.log1p(guc_s)
dd = A @ W311 - A[:, 0]  # uygulanan delta yonu (merkezlenmemis)


def grup_merkez(v, g):
    s = pd.Series(v)
    return (s - s.groupby(g).transform("mean")).to_numpy()


def egim_tek(x, y):
    return float(np.dot(x, y) / np.dot(x, x))


# ---- 1) yontem cesitlemesi
Xk = np.c_[[grup_merkez(A[:, i], kova) for i in range(3)]].T
yk = grup_merkez(L_s, kova)
bk, *_ = np.linalg.lstsq(Xk, yk, rcond=None)
R["s_kova_3kolon_toplam"] = round(float(bk.sum()), 4)
R["s_kova_3kolon_katsayilar"] = [round(float(x), 4) for x in bk]
R["s_kova_tek_cat"] = round(egim_tek(Xk[:, 0], yk), 4)

Xt = np.c_[[grup_merkez(A[:, i], tanim_s) for i in range(3)]].T
yt = grup_merkez(L_s, tanim_s)
bt, *_ = np.linalg.lstsq(Xt, yt, rcond=None)
ss_res = float(((yt - Xt @ bt) ** 2).sum())
ss_tot = float((yt**2).sum())
R["s_TRAFO_ici_3kolon_toplam"] = round(float(bt.sum()), 4)
R["s_TRAFO_ici_3kolon_katsayilar"] = [round(float(x), 4) for x in bt]
R["s_TRAFO_ici_tek_cat"] = round(egim_tek(Xt[:, 0], yt), 4)
R["TRAFO_ici_R2"] = round(1 - ss_res / ss_tot, 4)

# trafolar-ARASI: trafo ortalamalari, kova-merkezli
df = pd.DataFrame(
    {"L": L_s, "cat": A[:, 0], "x": A[:, 1], "g": A[:, 2], "kova": kova, "tanim": tanim_s, "dd": dd}
)
tr = df.groupby("tanim").agg(
    L=("L", "mean"),
    cat=("cat", "mean"),
    x=("x", "mean"),
    g=("g", "mean"),
    dd=("dd", "mean"),
    kova=("kova", "first"),
    n=("L", "size"),
)
Xa = np.c_[[grup_merkez(tr[c].to_numpy(), tr["kova"].to_numpy()) for c in ("cat", "x", "g")]].T
ya = grup_merkez(tr["L"].to_numpy(), tr["kova"].to_numpy())
ba, *_ = np.linalg.lstsq(Xa, ya, rcond=None)
R["s_TRAFOLAR_arasi_3kolon_toplam"] = round(float(ba.sum()), 4)
R["s_TRAFOLAR_arasi_tek_cat"] = round(egim_tek(Xa[:, 0], ya), 4)
R["s_TRAFOLAR_arasi_R2"] = round(float(1 - ((ya - Xa @ ba) ** 2).sum() / (ya**2).sum()), 4)

# saglam egim (alt-orneklem medyan egimleri, tek-cat, trafo-ici)
rng = np.random.default_rng(7)
egimler = []
for _ in range(200):
    ix = rng.choice(len(yt), 20000, replace=False)
    egimler.append(egim_tek(Xt[ix, 0], yt[ix]))
R["s_TRAFO_ici_tek_cat_altorneklem"] = {
    "medyan": round(float(np.median(egimler)), 4),
    "q05": round(float(np.quantile(egimler, 0.05)), 4),
    "q95": round(float(np.quantile(egimler, 0.95)), 4),
}

# ---- 2) afinlik: kova basina trafo-ici tek-cat egimi
kova_egim = {}
for kv in sorted(set(kova)):
    m = kova == kv
    x = grup_merkez(A[m, 0], tanim_s[m])
    y = grup_merkez(L_s[m], tanim_s[m])
    kova_egim[int(kv)] = {"n": int(m.sum()), "egim": round(egim_tek(x, y), 4)}
R["kova_basina_TRAFO_ici_egim"] = kova_egim

# ---- 3) dd'nin varyans ayrisimi
dd_c = dd - dd.mean()
ici = grup_merkez(dd, tanim_s)
arasi_var = float(dd_c.var() - ici.var())
R["dd_varyans_ayrisimi"] = {
    "toplam_var": round(float(dd_c.var()), 5),
    "trafo_ici_var": round(float(ici.var()), 5),
    "trafolar_arasi_var": round(arasi_var, 5),
    "arasi_pay": round(arasi_var / float(dd_c.var()), 4),
}

# ---- 4) tohum gurultusu sonmesi (blok npz'lerinden kestirim)
DN = os.path.join(KOK, "data/interim/deney")
att = {}
for b in ("yaz25", "guz25", "kis26"):
    z = np.load(os.path.join(DN, f"soguk_tahmin_{b}.npz"))
    toh = [1000, 1001, 1002]
    C = np.stack([z[f"{t}_cat"].astype("float64") for t in toh])
    gurultu_var_ort3 = float(C.var(axis=0, ddof=1).mean()) / 3  # 3-tohum ort.nun gurultusu
    sinyal_var = float(C.mean(axis=0).var())
    att[b] = {
        "cat3ort_gurultu_var": round(gurultu_var_ort3, 5),
        "cat_sinyal_var": round(sinyal_var, 5),
        "sonme_katsayisi(1/(1+g/s))": round(1 / (1 + gurultu_var_ort3 / sinyal_var), 4),
    }
    # dd yonunun tohum gurultusu
    X_ = np.stack(
        [
            np.c_[z[f"{t}_cat"], z[f"{t}_xgb"], z[f"{t}_lgbm"]].astype("float64") @ W311
            - z[f"{t}_cat"].astype("float64")
            for t in toh
        ]
    )
    att[b]["dd_tohum_sd_satirbasina"] = round(float(np.sqrt(X_.var(axis=0, ddof=1).mean() / 3)), 5)
R["tohum_sonme"] = att

yol = os.path.join(PK, "p25_kirmizi.json")
mevcut = {}
if os.path.exists(yol):
    with open(yol, encoding="utf-8") as fh:
        mevcut = json.load(fh)
mevcut["B_afin_kopru"] = R
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(mevcut, fh, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
