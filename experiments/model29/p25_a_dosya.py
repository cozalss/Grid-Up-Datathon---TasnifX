"""p25-A KIRMIZI TAKIM: aday dosyalarin mekanik denetimi.

Hat 5: seviyesiz merkezleme -- kayma dagilimi (ort/medyan/kuyruk), hangi
kumede merkezlendi, exp uzayinda seviye etkisi.
Hat 6: p21 = p20 + p08(x0.5) katmani -- birebir mi, p08 deltasi YP_seviye
tabaninda gercekten x0.5 mi (D1_demet tabaninda olculmustu).
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")
AC = os.path.join(PK, "aday_csv")

R = {}

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), usecols=["id"], dtype={"id": str})
IDS = te["id"].to_numpy()
soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))

yp = pd.read_csv(os.path.join(KOK, "submissions/tuketim_YP_seviye.csv"))
p20 = pd.read_csv(os.path.join(AC, "p20_harman_ESKI_3_1_1_V1_seviyesiz.csv"))
p21 = pd.read_csv(os.path.join(AC, "p21_harman311_olu50.csv"))
d08 = np.load(os.path.join(AC, "p08_olu_delta_log_c050.npy"))

for ad, df in (("yp", yp), ("p20", p20), ("p21", p21)):
    assert np.array_equal(df["id"].to_numpy(), IDS), f"{ad} id sirasi"

Lyp = np.log1p(yp["tuketim"].to_numpy("float64"))
L20 = np.log1p(p20["tuketim"].to_numpy("float64"))
L21 = np.log1p(p21["tuketim"].to_numpy("float64"))

# ---------------- HAT 5: merkezleme / kayma dagilimi
k = L20 - Lyp  # p20'nin YP'ye gore log kaymasi
ks = k[soguk]
R["hat5_kayma_dagilimi"] = {
    "soguk_ort": float(ks.mean()),
    "soguk_medyan": float(np.median(ks)),
    "soguk_std": float(ks.std()),
    "q01": float(np.quantile(ks, 0.01)),
    "q05": float(np.quantile(ks, 0.05)),
    "q95": float(np.quantile(ks, 0.95)),
    "q99": float(np.quantile(ks, 0.99)),
    "min": float(ks.min()),
    "max": float(ks.max()),
    "pozitif_pay": float((ks > 0).mean()),
    "sicak_maks_mutlak": float(np.abs(k[~soguk]).max()),
}
# exp uzayinda seviye: toplam tuketim degisimi
R["hat5_exp_seviye"] = {
    "soguk_toplam_tuketim_yp": float(yp["tuketim"].to_numpy("float64")[soguk].sum()),
    "soguk_toplam_tuketim_p20": float(p20["tuketim"].to_numpy("float64")[soguk].sum()),
    "oran": float(
        p20["tuketim"].to_numpy("float64")[soguk].sum()
        / yp["tuketim"].to_numpy("float64")[soguk].sum()
    ),
}

# trafo bazli merkezleme mi kontrol: kaymanin trafo ici mi genel mi merkezlendigi
tanim = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), usecols=["tanim"], dtype={"tanim": str}
)["tanim"].to_numpy()
s_tr = pd.Series(ks, index=tanim[soguk])
tr_ort = s_tr.groupby(level=0).mean()
R["hat5_trafo_bazli_kayma"] = {
    "trafo_ort_kaymalarin_std": float(tr_ort.std()),
    "trafo_ort_kaymalarin_ort": float(tr_ort.mean()),
    "en_negatif_trafo_ort": float(tr_ort.min()),
    "en_pozitif_trafo_ort": float(tr_ort.max()),
    "yorum": "merkezleme GENEL (tek sabit); trafo bazli degil -- trafo ort kaymalari genis daginiksa seviye trafolar arasi yeniden dagitiliyor",
}

# ---------------- HAT 6: p21 katmani
fark = L21 - L20
uyum = np.abs(fark - d08)
R["hat6_p21_esitligi"] = {
    "p21_eq_p20_plus_d08_maxabs": float(uyum.max()),
    "d08_nonzero": int((np.abs(d08) > 1e-12).sum()),
    "d08_soguk_kesisim": int(((np.abs(d08) > 1e-12) & soguk).sum()),
}

# p08 deltasi hangi tabana gore x0.5? YP tabaninda dogru delta:
m08 = np.abs(d08) > 1e-12
x_yp = yp["tuketim"].to_numpy("float64")[m08]
dogru_delta = np.log1p(0.5 * x_yp) - np.log1p(x_yp)
uyg = d08[m08]
R["hat6_carpan_tutarliligi"] = {
    "n": int(m08.sum()),
    "uygulanan_delta_ort": float(uyg.mean()),
    "yp_tabaninda_dogru_x05_delta_ort": float(dogru_delta.mean()),
    "fark_maxabs": float(np.abs(uyg - dogru_delta).max()),
    "fark_rms": float(np.sqrt(((uyg - dogru_delta) ** 2).mean())),
    "efektif_carpan_medyan": float(
        np.median((np.expm1(np.log1p(x_yp) + uyg)) / np.where(x_yp > 0, x_yp, np.nan))
    ),
}
# etkilenen satirlarin tahmin buyuklugu
R["hat6_etkilenen_satirlar"] = {
    "x_yp_medyan": float(np.median(x_yp)),
    "x_yp_q90": float(np.quantile(x_yp, 0.9)),
    "x_yp_maks": float(x_yp.max()),
}

yol = os.path.join(PK, "p25_kirmizi.json")
mevcut = {}
if os.path.exists(yol):
    with open(yol, encoding="utf-8") as fh:
        mevcut = json.load(fh)
mevcut["A_dosya_denetimi"] = R
with open(yol, "w", encoding="utf-8") as fh:
    json.dump(mevcut, fh, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
