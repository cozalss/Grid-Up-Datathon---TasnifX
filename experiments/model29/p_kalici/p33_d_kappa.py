"""p33-d: |c| dagilimi, kappa izgarasi, adaylar (span + p32 kuyruk kapagi).

Cikti: p_kalici/aday_csv/p33_span_k*.csv  ve  p_kalici/p33_span.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
S = os.path.join(KOK, "submissions")
PK = os.path.join(M29, "p_kalici")
AC = os.path.join(PK, "aday_csv")
GEC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, M29)
from m112_kalibre import M0  # noqa: E402
from p32_ortak import KESIM, _ham  # noqa: E402

YEDEK = 1.00115
HEDEF3 = 0.99487

with open(os.path.join(GEC, "p33_a_cebir.json"), encoding="utf-8") as fh:
    A = json.load(fh)
with open(os.path.join(GEC, "p33_b_capa.json"), encoding="utf-8") as fh:
    B = json.load(fh)
with open(os.path.join(GEC, "p33_c_zincir.json"), encoding="utf-8") as fh:
    C = json.load(fh)
if not C["gecti"]:
    raise SystemExit("DUR: zincir testi GECMEDI.")

V = np.load(os.path.join(GEC, "p33_V.npy"))
a0 = np.load(os.path.join(GEC, "p33_a0.npy"))
r_hat = np.load(os.path.join(GEC, "p33_r_hat.npy"))
GD = np.load(os.path.join(GEC, "GD.npy"))
N = A["N"]
TABAN_MSE = A["TABAN_MSE"]  # cebirsel sabit (L gurultusuz)
DURUST_MSE = M0 - A["beklenen_kazanc"]  # buzmenin BEKLEDIGI gercek MSE

# --- Y1'in kendi dik yonu (OLCULMUS rho tasiyor) ----------------------------
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), usecols=["id", "tanim"],
                 dtype={"id": str, "tanim": str})
IDS = te["id"].to_numpy()
tanim = te["tanim"].to_numpy()


def oku(f):
    d = pd.read_csv(os.path.join(S, f), dtype={"id": str})
    assert np.array_equal(d["id"].to_numpy(), IDS), f"{f}: id sirasi"
    return np.log1p(d["tuketim"].to_numpy("float64"))


G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
dY = oku("tuketim_Y1_demet.csv") - a0
dY_perp = dY - V @ (Gi @ ((V.T @ dY) / N))
uY = dY_perp / np.sqrt(float((dY_perp * dY_perp).mean()))
rhoY = next(s["rho_dik_SAF"] for s in B["sondalar"] if s["sonda"] == "Y1")

# --- bahis yonu: bugunku bilesik BETA, uY'den arindirilmis ------------------
BETA = 0.1700 * GD[0] + 0.1302 * GD[1]  # ||BETA|| = 0.2141 (dokum2.log)
beta_norm = float(np.sqrt(float((BETA * BETA).mean())))
BETA_o = BETA - float((BETA * uY).mean()) * uY
BETA_o = BETA_o - V @ (Gi @ ((V.T @ BETA_o) / N))
kalan = float(np.sqrt(float((BETA_o * BETA_o).mean())))
u = BETA_o / kalan
RHO_PRED = kalan  # 1.95 tavanina dayanan ongoru (oran = 1)
print(f"||BETA|| = {beta_norm:.4f}, uY cikarildiktan sonra kalan = {kalan:.4f}")
print(f"rho_Y1 (OLCULMUS) = {rhoY:+.6f}")

# --- |c| / gerceklesme orani dagilimi --------------------------------------
ORANLAR = [(s["sonda"], s["gerceklesme_orani"]) for s in B["sondalar"]]
o = np.array([x[1] for x in ORANLAR])
n = len(o)
ort, sd = float(o.mean()), float(o.std(ddof=1))
# ongoru dagilimi: t_{n-1}, olcek sd*sqrt(1+1/n)
olcek = sd * np.sqrt(1 + 1 / n)
sd_ort = sd / np.sqrt(n)
print(f"gerceklesme orani: {[f'{x:+.3f}' for x in o]}  ort {ort:+.4f} sd {sd:.4f}")
print(f"  |c| = 1.95*oran -> ort {1.95 * ort:+.4f}, %95 GA "
      f"[{1.95 * (ort - stats.t.ppf(0.975, n - 1) * sd_ort):+.3f}, "
      f"{1.95 * (ort + stats.t.ppf(0.975, n - 1) * sd_ort):+.3f}]")

C0 = TABAN_MSE - rhoY**2  # olculmus demeti kilitledikten sonraki sabit
C0d = DURUST_MSE - rhoY**2


def P_hedef(kap):
    """P(skor <= HEDEF3) -- oran t-dagilimi uzerinden."""
    if kap <= 0:
        return float(np.sqrt(max(C0d, 0)) <= HEDEF3)
    # skor^2 = C0 - 2*kap*oran*RHO_PRED + kap^2 <= HEDEF3^2
    esik_oran = (C0d + kap * kap - HEDEF3**2) / (2 * kap * RHO_PRED)
    return float(1 - stats.t.cdf((esik_oran - ort) / olcek, n - 1))


# kappa adaylari
kap_ilk3 = float(np.sqrt(max(C0d - HEDEF3**2, 0.0)))  # rho = kappa ise tam tutar
IZGARA = []
adaylar_kappa = {
    "k0000_yalniz_olculmus": 0.0,
    "k_orta_nokta": max(ort, 0.0) * RHO_PRED,
    "k_ust_GA95": max(ort + stats.t.ppf(0.95, n - 1) * olcek, 0.0) * RHO_PRED,
    "k_eski_capa_c0434": (0.434 / 1.95) * RHO_PRED,
    "k_eski_capa_c1986": (1.986 / 1.95) * RHO_PRED,
    "k_ilk3": kap_ilk3,
}
for ad, kap in adaylar_kappa.items():
    bek = float(np.sqrt(max(C0d - 2 * kap * ort * RHO_PRED + kap * kap, 1e-12)))
    kotu = float(np.sqrt(max(C0d + kap * kap, 1e-12)))  # rho = 0
    iyi = float(np.sqrt(max(C0d - 2 * kap * RHO_PRED + kap * kap, 1e-12)))  # oran = 1
    IZGARA.append(
        {
            "ad": ad,
            "kappa": kap,
            "ima_edilen_c": 1.95 * kap / RHO_PRED if RHO_PRED else None,
            "beklenen_skor": bek,
            "en_kotu_rho0": kotu,
            "oran1_ise": iyi,
            "P_ilk3": P_hedef(kap),
            "yedekten_iyi_mi_beklenen": bool(bek < YEDEK),
            "en_kotu_yedekten_kotu_mu": bool(kotu > YEDEK),
        }
    )
    print(f"  {ad:24s} kappa={kap:.5f} |c|~{1.95 * kap / RHO_PRED:5.2f} "
          f"beklenen {bek:.5f}  en kotu(rho=0) {kotu:.5f}  P(ilk3)={P_hedef(kap):.4f}")

# --- ADAY CSV'ler ----------------------------------------------------------
soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy")) \
    if os.path.exists(os.path.join(AC, "p06_test_soguk_maske.npy")) \
    else np.load(os.path.join(PK, "aday_csv", "p06_test_soguk_maske.npy"))

g = _ham()
g = g[g["tarih"] < pd.Timestamp(KESIM["TEST"])]
lg = np.log1p(np.clip(g["tuketim"].to_numpy("float64"), 0, None))
gb = pd.DataFrame({"tanim": g["tanim"].to_numpy(), "l": lg}).groupby("tanim")["l"]
mu = gb.mean().reindex(pd.Index(tanim)).to_numpy("float64")
sdv = gb.std().reindex(pd.Index(tanim)).to_numpy("float64")
ok = (~soguk) & np.isfinite(mu) & np.isfinite(sdv) & (sdv > 0)
KCAP = 8.0

taban_log = a0 + r_hat + rhoY * uY  # OLCULMUS demet kilitli


def kapak(L):
    L = L.copy()
    L[ok] = np.clip(L[ok], mu[ok] - KCAP * sdv[ok], mu[ok] + KCAP * sdv[ok])
    return L


def yaz(ad, L):
    x = np.clip(np.expm1(np.maximum(L, 0.0)), 0, None)
    yol = os.path.join(AC, f"{ad}.csv")
    pd.DataFrame({"id": IDS, "tuketim": x}).to_csv(yol, index=False)
    geri = pd.read_csv(yol, dtype={"id": str})
    return {
        "yol": f"p_kalici/aday_csv/{ad}.csv",
        "n_satir": int(len(geri)),
        "satir_dogru": bool(len(geri) == 714688),
        "id_sirasi_birebir": bool(np.array_equal(geri["id"].to_numpy(), IDS)),
        "NaN": int(geri["tuketim"].isna().sum()),
        "negatif": int((geri["tuketim"] < 0).sum()),
        "sonsuz": int((~np.isfinite(geri["tuketim"].to_numpy("float64"))).sum()),
        "toplam_tuketim": float(x.sum()),
    }


URET = {}
SEC = ["k0000_yalniz_olculmus", "k_orta_nokta", "k_ust_GA95", "k_ilk3"]
DEN = {}
for ad in SEC:
    kap = adaylar_kappa[ad]
    L = taban_log + kap * u
    DEN[f"p33_span_{ad}"] = yaz(f"p33_span_{ad}", L)
    DEN[f"p33_span_{ad}_kuyruk8"] = yaz(f"p33_span_{ad}_kuyruk8", kapak(L))
    URET[ad] = kap

with open(os.path.join(PK, "p33_span.json"), "w", encoding="utf-8") as fh:
    json.dump(
        {
            "00_KURAL": "Kaggle gonderimi YOK, submissions/ yazilmadi, commit yok",
            "01_cebir_bagimsiz": A,
            "02_zincir_testi": C,
            "03_capalar": B["sondalar"],
            "04_oran_dagilimi": {
                "oranlar": {k: v for k, v in ORANLAR},
                "n": n,
                "ort": ort,
                "sd": sd,
                "ongoru_olcegi": olcek,
                "c_ort": 1.95 * ort,
                "c_GA95": [
                    1.95 * (ort - stats.t.ppf(0.975, n - 1) * sd_ort),
                    1.95 * (ort + stats.t.ppf(0.975, n - 1) * sd_ort),
                ],
                "eski_capalar": {"n10_gonderim_farklari": 0.434, "seviye_ekseni": 1.986},
            },
            "05_geometri": {
                "TABAN_MSE": TABAN_MSE,
                "saf_span_cebirsel": float(np.sqrt(TABAN_MSE)),
                "DURUST_MSE_buzme_beklentisi": DURUST_MSE,
                "saf_span_durust": float(np.sqrt(DURUST_MSE)),
                "rho_Y1_olculmus": rhoY,
                "C0_kilitli_durust": C0d,
                "skor_yalniz_olculmus": float(np.sqrt(C0d)),
                "BETA_norm": beta_norm,
                "kalan_bahis_yonu_normu": kalan,
                "RHO_PRED_oran1": RHO_PRED,
                "kappa_ilk3": kap_ilk3,
            },
            "06_kappa_izgarasi": IZGARA,
            "07_adaylar": DEN,
            "08_kuyruk_kapagi": {
                "k": KCAP,
                "kaynak": "p32 K3e (dort kapiyi da gecen tek katman)",
                "capasi_olan_sicak": int(ok.sum()),
                "beklenen_LB_kazanci_tasima0.5": 0.00395,
                "cakisma": "span yonu TUM satirlarda, kapak YALNIZ sicak; "
                           "kapak span'dan SONRA uygulanir",
            },
        },
        fh,
        ensure_ascii=False,
        indent=1,
    )
print("\n-> p_kalici/p33_span.json")
for k, v in DEN.items():
    print(f"  {k}: satir {v['n_satir']} id {v['id_sirasi_birebir']} "
          f"NaN {v['NaN']} neg {v['negatif']} inf {v['sonsuz']}")
