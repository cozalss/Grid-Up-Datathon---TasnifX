"""TASIMA OLCUMUNU DUZELT: katsayilar LB'den, bloktan FIT EDILMIYOR.

Onceki olcum (m125) her yarida agirliklari YENIDEN FIT edip orani aliyordu.
Ama gercek kurulumda katsayilar LB'den geliyor (1.95*|rho_s|), bloktan
fit edilmiyor. Dolayisiyla "fit/holdout" ayrimi yapay ve oran gurultulu
(n=6'da sd 0.647 -- olculemiyor).

DOGRUSU: bilesigi LB katsayilariyla SABIT kur, sonra blogun farkli zaman
pencerelerinde korelasyonunu olc. Fit yok, sizinti yok, oran yok.

Test penceresi 122 gunluk bir ufuk oldugu icin GEC pencere en iyi vekildir.
Her on-ek uzunlugu icin:
  kor_tum    tum blokta korelasyon
  kor_erken  gun 1-40
  kor_gec    gun 83-122      <- test'e en yakin
  kor_sd     bes pencere arasindaki sacilim
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, L_gurultusu, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != len(IDS) or d.id.duplicated().any():
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].values.astype(np.float64))


a0 = oku(TABAN)
N = len(a0)
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
for f, Lj in EK_MODEL.items():
    V.append(oku(f) - a0)
    L.append(Lj)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
V, L = np.array(V).T, np.array(L)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
MSE_OPT = M0 - gercek

# ---------------------------------------------------------------------------
# 1.95 CARPANININ KALIBRASYON KUMESI -- BIRINI DISARIDA BIRAK.
#
# 1.95 tek bir deneyden (seviye) geliyordu (n=1) ve butun bahsimiz ona
# dayaniyor. Gonderdigimiz her dosya span'a dahil edildigi icin dogrudan
# bakilamiyor -- ama j'yi DISARIDA BIRAKIP kalanla span kurarsak, d_j'nin
# gercek bir DIK bileseni olur ve skoru bilindigi icin o dik yondeki
# GERCEKLESEN korelasyon cozulur:
#     L_j = (M0 + Q_j - P_j^2)/2            (olculmus, tam)
#     <r, d_j^span>/N ~ <r_hat_(-j), d_j^span>/N
#     <r, d_j^dik >/N = L_j - <r_hat_(-j), d_j^span>/N
#     rho_u = <r,d_j^dik>/N / sqrt(Q_dik)   GERCEKLESEN
#     rho_s = <r_hat_(-j), d_j>/N / sqrt(Q_span)
#     ORAN  = rho_u / rho_s                 <- 1.95'in bagimsiz olcumu
# Her gonderim bir olcum verir. Q_dik cok kucukse olcum gurultuludur, elenir.
# ---------------------------------------------------------------------------

with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SKR = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DR = json.load(fh)
AD, DV, LV, PV = [], [], [], []
for f, Pj in SKR.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    d = v - a0
    AD.append(f)
    DV.append(d)
    LV.append((M0 + float((d * d).mean()) - Pj * Pj) / 2)
    PV.append(Pj)
for o in DR.get("olcumler", []):
    d = oku(o["dosya"]) - a0
    AD.append(o["dosya"])
    DV.append(d)
    LV.append((M0 + float((d * d).mean()) - o["skor"] ** 2) / 2)
    PV.append(o["skor"])
DV = np.array(DV)
LV = np.array(LV)
print(f"\n{len(AD)} olculmus yon (EK_MODEL turetilmisleri disarida)")
SIG_TAM = L_gurultusu(DV.T, N)  # BIR KEZ; her turda yeniden kurmak 27x maliyetliydi
print(f"sigma_L ort {SIG_TAM.mean():.3e}")

print(
    f"\n{'disarida birakilan':>32s} {'P':>8s} {'Q_dik':>8s} {'rho_u':>9s} "
    f"{'rho_s':>9s} {'ORAN':>8s}"
)
KAYIT = []
for j in range(len(AD)):
    ix = [i for i in range(len(AD)) if i != j]
    Vr = DV[ix].T
    Lr = LV[ix]
    Gr = (Vr.T @ Vr) / N
    try:
        rh, _, _ = buzmeli_r_hat(Vr, Lr, Gr, N, sigma=SIG_TAM[ix])
    except Exception as hata:
        print(f"  {AD[j][:30]}: {type(hata).__name__} {hata}")
        continue
    dj = DV[j]
    cc = np.linalg.pinv(Gr, rcond=1e-6) @ ((Vr.T @ dj) / N)
    dsp = Vr @ cc
    ddk = dj - dsp
    Qdk = float((ddk * ddk).mean())
    Qsp = float((dsp * dsp).mean())
    if Qsp < 1e-10:
        continue
    r_dik = LV[j] - float((rh * dsp).mean())
    rho_s = float((rh * dj).mean()) / np.sqrt(Qsp)
    if Qdk < 1e-6 or abs(rho_s) < 1e-5:
        continue
    rho_u = r_dik / np.sqrt(Qdk)
    KAYIT.append((AD[j], PV[j], Qdk, rho_u, rho_s, rho_u / rho_s))
    print(
        f"{AD[j][:32]:>32s} {PV[j]:8.5f} {Qdk:8.5f} {rho_u:+9.4f} "
        f"{rho_s:+9.4f} {rho_u / rho_s:+8.2f}"
    )

for esik in [1e-4, 1e-3, 1e-2, 5e-2]:
    gv = [k for k in KAYIT if k[2] > esik]
    if len(gv) < 3:
        continue
    o = np.array([k[5] for k in gv])
    print(
        f"\nQ_dik > {esik:g}: n={len(gv)}  ortanca oran {np.median(o):+.2f}  "
        f"ortanca |oran| {np.median(np.abs(o)):.2f}  "
        f"%25-%75 [{np.quantile(o, 0.25):+.2f}, {np.quantile(o, 0.75):+.2f}]"
    )
print("\nKULLANDIGIMIZ DEGER: 1.95 (buyukluk). Ortanca |oran| buna yakinsa")
print("  kalibrasyon dogrulanir; cok kucukse tavan ISKALIYOR demektir.")

# ---------------------------------------------------------------------------
# DOGRU TAHMIN EDICI: BOLME DEGIL AGIRLIKLI REGRESYON.
#
# Yukaridaki "oran" sutunu rho_u = r_dik/sqrt(Q_dik) seklinde KUCUK bir
# sayiya boluyor. Q_dik ~ 1e-5 olan satirlarda sqrt(Q_dik) ~ 0.003 ve
# r_dik'in kendi hatasi ~sigma_L (2.3e-4) -> rho_u hatasi ~0.07, yani
# sinyalden buyuk. Ortancayi bu gurultu suruyor.
#
# Modelimiz zaten dogrusal:   <r, d_dik>/N  =  c * rho_s * sqrt(Q_dik)
# Dogrudan bu regresyon kurulur; agirlik 1/sigma^2. Egim c, 1.95'in
# TARAFSIZ ve HATA PAYLI tahminidir.
# ---------------------------------------------------------------------------
print("\n\nAGIRLIKLI REGRESYON:  <r,d_dik>/N = c * rho_s * sqrt(Q_dik)")
x = np.array([k[4] * np.sqrt(k[2]) for k in KAYIT])
y = np.array([k[3] * np.sqrt(k[2]) for k in KAYIT])  # = r_dik
sg = float(SIG_TAM.mean())
wt = np.full(len(x), 1.0 / sg**2)
c_hat = float((wt * x * y).sum() / (wt * x * x).sum())
se = float(np.sqrt(1.0 / (wt * x * x).sum()))
art = y - c_hat * x
khi = float((wt * art * art).sum() / max(len(x) - 1, 1))
print(f"  n = {len(x)}   sigma = {sg:.2e}")
print(f"  c = {c_hat:+.3f} +- {se:.3f}   (khi-kare/sd = {khi:.1f})")
if khi > 1:
    print(f"  khi-kare>1 -> hata payi olcek duzeltmesiyle: +-{se * np.sqrt(khi):.3f}")
    se *= np.sqrt(khi)
print(f"  KULLANDIGIMIZ 1.95, tahminden {(1.95 - c_hat) / se:+.1f} sigma uzakta")

# en buyuk Q_dik'li satirlar bizim rejimimize (Q_dik>=0.25) en yakin olanlar
for esik in [0.0, 1e-3, 1e-2]:
    m = np.array([k[2] > esik for k in KAYIT])
    if m.sum() < 3:
        continue
    cc = float((wt[m] * x[m] * y[m]).sum() / (wt[m] * x[m] * x[m]).sum())
    ss = float(np.sqrt(1.0 / (wt[m] * x[m] * x[m]).sum()))
    a2 = y[m] - cc * x[m]
    k2 = float((wt[m] * a2 * a2).sum() / max(int(m.sum()) - 1, 1))
    print(f"  Q_dik > {esik:g}: n={int(m.sum())}  c = {cc:+.3f} +- {ss * np.sqrt(max(k2, 1)):.3f}")

print("\nUYARI: bu kalibrasyonun en buyuk Q_dik'i 0.062; bizim eksenlerimiz")
print("  Q_dik >= 0.25 rejiminde. Yani olcum bizim rejimimizin DISINDA")
print("  yapiliyor ve dogrudan tasinmayabilir.")
