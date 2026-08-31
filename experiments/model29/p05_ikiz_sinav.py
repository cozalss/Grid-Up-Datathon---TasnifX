"""p05 IKIZ SINAV: fikir URETIM OZNITELIKLERINDE hic tekrarliyor mu?

p03'un tezgahinda kiyas "tek asamali TEK lgbm" idi. Uretim hatti ise 9
modelli cat/xgb/lgbm demeti. Iki-asamalinin uretimde kaybetmesi iki ayri
sebepten olabilir:
  (i) fikir uretim ozniteliklerinde HIC calismiyor, ya da
  (ii) fikir calisiyor ama tek-lgbm seviyesi demetin cok altinda kaliyor.
Ayirt etmek icin AYNI model sinifi ile ikiz kiyas yapilir:
    tek asamali  : tum satirlarda huber lgbm            (pall)
    iki asamali  : (1-P0) * yalniz-pozitif huber lgbm   (P0, ppos)
Ikisi de ayni oznitelik matrisinde, ayni tohumda, guz25+kis26'da egitilmis;
yaz25 hedefi kullanilmamistir. P0/ppos p05_uretim_iki_asama.py'den gelir.
"""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
BURA = os.path.dirname(os.path.abspath(__file__))
ARA = os.environ.get(
    "ARA",
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
HEDEF_SOGUK = 0.222
KIMLIK = ["tanim_num", "tanim_on2", "tanim_on3", "tanim_on4", "tanim_on5"]
ATLA = ["tanim", "tarih", "tuketim", "lokasyon", "_blok", "id"]
ORT = dict(learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
           lambda_l2=5.0, num_threads=8, verbose=-1, seed=7)
PR_HUB = dict(ORT, objective="huber", alpha=2.0, lambda_l2=20.0, metric="l2")
PR_L1 = dict(ORT, objective="l1", metric="l2")
TUR = 500

e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
SAY = [c for c in e.columns
       if c not in ATLA and c not in KIMLIK and pd.api.types.is_numeric_dtype(e[c])]
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
idx = np.concatenate([sic.index.values, sog.index.values])
bf = e.loc[idx]
yv = np.log1p(bf.tuketim.to_numpy(dtype=np.float64))
sgm = bf.soguk_mu.to_numpy(dtype=np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
s = sgm == 1


def olc(p):
    r = np.asarray(p, dtype=np.float64) - yv
    return {"duz": float(np.sqrt(np.mean(r * r))),
            "test_agirlikli": float(np.sqrt(np.mean(ww * r * r))),
            "soguk": float(np.sqrt(np.mean(r[s] ** 2))),
            "sicak": float(np.sqrt(np.mean(r[~s] ** 2)))}


egt = e[~e._blok.eq("yaz25")]
ye = np.log1p(egt.tuketim.to_numpy(dtype=np.float64))
Xe = egt[SAY].astype(np.float32)
Xh = bf[SAY].astype(np.float32)
P0 = np.load(os.path.join(ARA, "p05_P0_yaz25.npy"))
ppos = np.load(os.path.join(ARA, "p05_ppos_yaz25.npy"))

R = {"aciklama": __doc__.strip()}
for ad, pk in (("huber", PR_HUB), ("l1", PR_L1)):
    pall = lgb.train(pk, lgb.Dataset(Xe, ye), num_boost_round=TUR).predict(Xh)
    R[f"tek_asamali_{ad}"] = olc(pall)
    print(ad, json.dumps(R[f"tek_asamali_{ad}"]), flush=True)
    if ad == "huber":
        np.save(os.path.join(ARA, "p05_pall_yaz25.npy"), pall)
R["iki_asamali_huber"] = olc((1 - P0) * ppos)
R["fikrin_ikiz_kazanci"] = {
    k: R["tek_asamali_huber"][k] - R["iki_asamali_huber"][k] for k in R["iki_asamali_huber"]}
R["fikrin_ikiz_kazanci_l1_tabana_gore"] = {
    k: R["tek_asamali_l1"][k] - R["iki_asamali_huber"][k] for k in R["iki_asamali_huber"]}
json.dump(R, open(os.path.join(BURA, "p05_ikiz_sinav.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print(json.dumps(R, indent=1, ensure_ascii=False))
