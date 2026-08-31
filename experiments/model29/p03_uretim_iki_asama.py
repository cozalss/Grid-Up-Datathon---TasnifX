"""p03: IKI ASAMALI AYRISTIRMAYI URETIM BORU HATTINA tasi ve yaz25'te olc.

SIZINTI KONTROLU
  * yaz25 satirlarinin hedefi (tuketim) HICBIR yerde etiket olarak
    kullanilmiyor. Siniflandirici ve pozitif-altkume regresyonu YALNIZCA
    guz25 + kis26 bloklarinda egitiliyor.
  * yaz25 satirlarinin OZELLIKLERI zaten kesim-oncesi (2025-03-31) gecmisten
    uretilmis (uretim hattinin kendi kurulusu).
  * KIMLIK EZBERI kanali: guz25/kis26 satirlarinin t_* ozetleri yaz25
    donemini de kapsadigi icin, tanim_num/tanim_on* ile birlestiginde model
    "su trafo yaz25'te oluydu" ezberleyebilir. Bu yuzden KIMLIK sutunlari
    (tanim_num, tanim_on2..5) ELENIYOR. Elenmemis surum de ayrica olculup
    farki raporlaniyor (KIMLIK_ILE).
  * Esik/harman agirligi gibi her sabit, EGITIM bloklarinda (guz25+kis26)
    kendi-dis (2 katli) tahminle secilir; yaz25'te yalnizca UYGULANIR.
"""

import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
BURA = os.path.dirname(os.path.abspath(__file__))
ARA = os.environ.get(
    "ARA",
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
HEDEF_SOGUK = 0.222
KIMLIK = ["tanim_num", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5"]
ATLA = ["tanim", "tarih", "tuketim", "lokasyon", "_blok"]
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


# ---------------- veri ----------------
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [
    np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
    for t in (1000, 1001, 1002)
    for aa in ("cat", "xgb", "lgbm")
    if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))
]
z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
idx = np.concatenate([sic.index.values, sog.index.values])
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
bf = e.loc[idx].copy()
yv = np.log1p(bf.tuketim.to_numpy(dtype=np.float64))
sgm = bf.soguk_mu.to_numpy(dtype=np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
log(f"uretim yaz25: {len(pb)} satir, {len(P)} sicak model, soguk pay {sgm.mean():.4f}")


def olc(p):
    r = np.asarray(p, dtype=np.float64) - yv
    duz = float(np.sqrt(np.mean(r * r)))
    agir = float(np.sqrt(np.mean(ww * r * r)))
    s = sgm == 1
    return {
        "duz": duz,
        "test_agirlikli": agir,
        "soguk": float(np.sqrt(np.mean(r[s] ** 2))),
        "sicak": float(np.sqrt(np.mean(r[~s] ** 2))),
    }


TABAN = olc(pb)
log("URETIM TABANI", json.dumps(TABAN))

# ---------------- ozellik matrisi ----------------
say = [c for c in e.columns if c not in ATLA and pd.api.types.is_numeric_dtype(e[c])]
say_temiz = [c for c in say if c not in KIMLIK]
egt = e[e._blok.isin(["guz25", "kis26"])]
ye = np.log1p(egt.tuketim.to_numpy(dtype=np.float64))
ze = (egt.tuketim.to_numpy() == 0).astype(int)
log(f"egitim bloklari {len(egt)} satir, ozellik {len(say_temiz)} (kimliksiz) "
    f"/ {len(say)} (kimlikli); sifir orani {ze.mean():.4f}")

ORT = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=5.0, num_threads=8, verbose=-1)
PC = dict(ORT, objective="binary", metric="binary_logloss")
PR_HUB = dict(ORT, objective="huber", alpha=2.0, lambda_l2=20.0, metric="l2")
PR_L2 = dict(ORT, objective="l2", metric="l2")
TUR = 700
TOHUM = [7, 17, 27]
R = {"uretim_tabani": TABAN, "sizinti_kontrolu": __doc__.strip()}


def egit_tahmin(pk, X, y, Xh, tohumlar=TOHUM, tur=TUR):
    return np.mean(
        [lgb.train(dict(pk, seed=s), lgb.Dataset(X, y), num_boost_round=tur).predict(Xh)
         for s in tohumlar], axis=0)


Xe = egt[say_temiz].astype(np.float32)
Xh = bf[say_temiz].astype(np.float32)
poz = ye > 0

# ---------------- P0 siniflandiricisi ----------------
P0 = egit_tahmin(PC, Xe, ze, Xh)
log(f"P0 hazir: yaz25 ort {P0.mean():.4f}, gercek sifir orani "
    f"{float((bf.tuketim.to_numpy() == 0).mean()):.4f}")
R["P0"] = {"yaz25_ortalama": float(P0.mean()),
           "yaz25_gercek_sifir_orani": float((bf.tuketim.to_numpy() == 0).mean()),
           "soguk_ortalama": float(P0[sgm == 1].mean()),
           "sicak_ortalama": float(P0[sgm == 0].mean())}

# ---------------- (a) uretim tahmini x (1-P0) ----------------
R["a_uretim_carpim"] = olc((1 - P0) * pb)
log("a)", json.dumps(R["a_uretim_carpim"]))

# ---------------- (b) siniflandirici + pozitif altkumede yeniden egitim ----
ppos_hub = egit_tahmin(PR_HUB, Xe[poz], ye[poz], Xh)
ppos_l2 = egit_tahmin(PR_L2, Xe[poz], ye[poz], Xh)
R["b_yeniden_egitim_huber"] = olc((1 - P0) * ppos_hub)
R["b_yeniden_egitim_l2"] = olc((1 - P0) * ppos_l2)
R["b_ham_pozitif_regresyon_huber_carpansiz"] = olc(ppos_hub)
log("b huber)", json.dumps(R["b_yeniden_egitim_huber"]))
log("b l2)", json.dumps(R["b_yeniden_egitim_l2"]))

# ---------------- (c) harman: uretim tahmini ile yeni pozitif regresyon ----
# harman agirligi EGITIM bloklarinda secilemez (uretim tahmini orada yok),
# bu yuzden birkac sabit agirlik ACIKCA raporlanir; secim LB'ye birakilir.
R["c_harman"] = {}
for w in (0.25, 0.5, 0.75):
    R["c_harman"][f"w_pozreg_{w}"] = olc((1 - P0) * (w * ppos_hub + (1 - w) * pb))
log("c)", json.dumps(R["c_harman"]))

# ---------------- (d) esikle sifirlama (sabit EGITIM blogunda secilir) ----
kat = (np.arange(len(Xe)) % 2).astype(bool)
p0e = np.empty(len(Xe))
pbe = np.empty(len(Xe))  # egitim blogunda "uretim tahmini" yerine l2 regresyon
for b in (False, True):
    m = np.equal(kat, b)
    p0e[m] = lgb.train(dict(PC, seed=7), lgb.Dataset(Xe[~m], ze[~m]),
                       num_boost_round=TUR).predict(Xe[m])
    pbe[m] = lgb.train(dict(PR_L2, seed=7), lgb.Dataset(Xe[~m], ye[~m]),
                       num_boost_round=TUR).predict(Xe[m])
en, esik = 1e9, None
for th in np.arange(0.30, 0.96, 0.05):
    q = np.where(p0e > th, 0.0, pbe)
    v = float(np.sqrt(np.mean((q - ye) ** 2)))
    if v < en:
        en, esik = v, float(th)
R["d_esik"] = {"esik": esik, "egitim_blogu_rmsle": en,
               "yaz25": olc(np.where(P0 > esik, 0.0, pb))}
log("d)", json.dumps(R["d_esik"]))

# ---------------- (e) KIMLIK sutunlariyla (sizinti riski) ----------------
Xe2 = egt[say].astype(np.float32)
Xh2 = bf[say].astype(np.float32)
P0k = egit_tahmin(PC, Xe2, ze, Xh2, tohumlar=[7])
R["e_KIMLIK_ILE"] = {
    "uyari": "tanim_num/tanim_on* ile -- yaz25 kimlik ezberi riski, ANA SONUC DEGIL",
    "a_carpim": olc((1 - P0k) * pb),
}
log("e)", json.dumps(R["e_KIMLIK_ILE"]["a_carpim"]))

# ---------------- kazanclar ----------------
def kazanc(d):
    return {"duz": TABAN["duz"] - d["duz"],
            "test_agirlikli": TABAN["test_agirlikli"] - d["test_agirlikli"],
            "soguk": TABAN["soguk"] - d["soguk"],
            "sicak": TABAN["sicak"] - d["sicak"]}


R["kazanclar"] = {
    "a_uretim_carpim": kazanc(R["a_uretim_carpim"]),
    "b_yeniden_egitim_huber": kazanc(R["b_yeniden_egitim_huber"]),
    "b_yeniden_egitim_l2": kazanc(R["b_yeniden_egitim_l2"]),
    "d_esik": kazanc(R["d_esik"]["yaz25"]),
    **{f"c_{k}": kazanc(v) for k, v in R["c_harman"].items()},
}
json.dump(R, open(os.path.join(BURA, "p03_uretim_iki_asama.json"), "w",
                  encoding="utf-8"), indent=1, ensure_ascii=False)
np.save(os.path.join(ARA, "p03_P0_yaz25.npy"), P0)
np.save(os.path.join(ARA, "p03_ppos_hub_yaz25.npy"), ppos_hub)
np.save(os.path.join(ARA, "p03_pb_yaz25.npy"), pb)
print(json.dumps(R["kazanclar"], indent=1, ensure_ascii=False))
log("bitti")
