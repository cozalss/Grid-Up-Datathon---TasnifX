"""y1_ihrac: URETIM ANALOGU -- ARTIK MODELI'ni MEVCUT TUM artik verisiyle (uc blok)
egit, TEST uzerinde delta uret, ISARETI CV'de POZITIF rho verecek sekilde cevir,
p34 SPAN'ina dik payini olc ve dogrulamalari yap.

Kullanim: python y1_ihrac.py <kurulum> <cikti_adi> <isaret>
"""
import json, os, sys
import numpy as np
import pandas as pd

GEC = os.path.dirname(os.path.abspath(__file__))
SCR = os.path.dirname(GEC)
AM = os.path.join(SCR, "am")
sys.path.insert(0, AM)
import b_ana as B

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
PK = os.path.join(KOK, "experiments/model29/p_kalici")

AD = sys.argv[1]
CIKTI_AD = sys.argv[2]
ISARET = float(sys.argv[3])

TAM = list(range(len(B.OZN)))
TAKVIMSIZ = [i for i in TAM if B.OZN[i] not in set(B.TAKVIM)]
KIMLIK = {"tanim", "tanim_num", "tanim_on2", "tanim_on3", "tanim_on4",
          "tanim_on5", "tanim_uzunluk"}
TANIMSIZ = [i for i in TAM if B.OZN[i] not in KIMLIK]
TANIMSIZ_TAKVIMSIZ = [i for i in TANIMSIZ if B.OZN[i] not in set(B.TAKVIM)]
KONF = {
    "ridge_merkezli_tamozn": dict(model="ridge", hedef_tip="merkezli", kol=TAM, par=None),
    "lgbm_ham_KIMLIKSIZ_takvimsiz": dict(model="lgbm", hedef_tip="ham",
                                         kol=TANIMSIZ_TAKVIMSIZ, par=None),
    "lgbm_merkezli_takvimsiz": dict(model="lgbm", hedef_tip="merkezli",
                                    kol=TAKVIMSIZ, par=None),
}[AD]

# --- 1) SIRA DOGRULAMASI: X_TEST (test.parquet) ile test_ids.npy ayni sirada mi
T = pd.read_parquet(os.path.join(KOK, "data/interim/deney/test.parquet"),
                    columns=["id"]) if True else None
tid = np.load(os.path.join(SCR, "test_ids.npy"), allow_pickle=False)
assert len(T) == len(tid) == 714688, (len(T), len(tid))
esit = bool((T["id"].astype(str).to_numpy() == tid.astype(str)).all())
print(f"SIRA DOGRULAMASI test.parquet id == test_ids.npy : {esit}")
assert esit

# --- 2) uc blokta egit (mevcut TUM artik verisi), TEST'te tahmin
eb = [(b, np.ones(len(B.yukle(b)[1]["r"]), bool)) for b in B.BLOKLAR]
_, _, _, ni, f = B.kur(eb, ("yaz25", np.ones(len(B.yukle("yaz25")[1]["r"]), bool)),
                       KONF["kol"], KONF["hedef_tip"], KONF["model"], par=KONF["par"])
n_egt = sum(len(B.yukle(b)[1]["r"]) for b in B.BLOKLAR)
print(f"{AD}: uc blokta egitildi n_egitim={n_egt} agac={ni}")

XT = np.load(os.path.join(AM, "X_TEST.npy"))
g = f(XT[:, KONF["kol"]]).astype(np.float64)
del XT
assert len(g) == 714688
delta = ISARET * g
assert np.isfinite(delta).all(), "NaN/inf VAR"
print(f"delta: n={len(delta)} rms={np.sqrt((delta*delta).mean()):.5f} "
      f"ort={delta.mean():+.5f} min={delta.min():+.4f} maks={delta.max():+.4f} "
      f"NaN={int(np.isnan(delta).sum())} inf={int(np.isinf(delta).sum())}")

# --- 3) p34 SPAN'a dik pay
a0 = np.load(os.path.join(PK, "p34_a0.npy"))
V = np.load(os.path.join(PK, "p34_V30.npy"))
r30 = np.load(os.path.join(PK, "p34_r30.npy"))
BAZ = np.load(os.path.join(PK, "p34_dik_baz.npy"))
with open(os.path.join(PK, "p34_b_capa.json"), encoding="utf-8") as fh:
    RHO_BAZ = np.array(json.load(fh)["ortonormal_baz"]["rho"])
taban_ref = a0 + r30 + BAZ.T @ RHO_BAZ
taban = np.load(os.path.join(SCR, "taban_log.npy"))
print(f"taban_log tutarlilik: maks fark {np.max(np.abs(taban - taban_ref)):.3e}")
N = len(a0)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
d_dik = delta - V @ (Gi @ ((V.T @ delta) / N))
for b in BAZ:
    d_dik = d_dik - float((d_dik * b).mean()) * b
nn = float(np.sqrt((delta * delta).mean()))
nd = float(np.sqrt((d_dik * d_dik).mean()))
print(f"SPAN: ||delta||={nn:.5f} ||dik||={nd:.5f} dik_pay={nd/nn:.4f} "
      f"span_ici_varyans={1-(nd/nn)**2:.4f}")

# --- 4) DOGRULAMA: taban + 0.15*u (u = birim rms) expm1 saglikli mi
u = delta / nn
for kap in (0.15, -0.15, 0.25, -0.25):
    ad_log = taban + kap * u
    v = np.expm1(ad_log)
    print(f"  kappa={kap:+.2f}: log min/maks {ad_log.min():.4f}/{ad_log.max():.4f}  "
          f"expm1 min={v.min():.6f} NaN={int(np.isnan(v).sum())} neg={int((v<0).sum())}")
u_dik = d_dik / nd
for kap in (0.15, -0.15):
    ad_log = taban + kap * u_dik
    v = np.expm1(ad_log)
    print(f"  DIK kappa={kap:+.2f}: log min={ad_log.min():.4f} expm1 min={v.min():.6f} "
          f"neg={int((v<0).sum())}")

yol = os.path.join(SCR, f"yon_{CIKTI_AD}.npy")
np.save(yol, delta)
geri = np.load(yol)
assert geri.shape == (714688,) and geri.dtype == np.float64
assert np.array_equal(geri, delta)
print(f"KAYDEDILDI -> {yol}  (geri okundu, birebir esit)")

with open(os.path.join(GEC, f"y1_ihrac_{CIKTI_AD}.json"), "w", encoding="utf-8") as fh:
    json.dump(dict(kurulum=AD, isaret=ISARET, agac=int(ni), n_egitim=int(n_egt),
                   n=714688, rms=nn, ort=float(delta.mean()),
                   dik_norm=nd, dik_pay=nd / nn,
                   span_ici_varyans=1 - (nd / nn) ** 2, dosya=yol),
              fh, ensure_ascii=False, indent=1)
print("TAMAM")
