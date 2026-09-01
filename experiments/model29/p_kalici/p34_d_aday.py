"""p34-d: ekstrapolasyon izgarasi, P(ilk 3) ve SON HAK adaylari.

Cikti: p_kalici/aday_csv/p34_*.csv  ve  p_kalici/p34_son_hak.json
KURAL: submissions/ ALTINA YAZILMAZ, gonderim YOK.
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
HEDEF3_MSE = HEDEF3**2

with open(os.path.join(GEC, "p34_a_cebir.json"), encoding="utf-8") as fh:
    A = json.load(fh)
with open(os.path.join(GEC, "p34_b_capa.json"), encoding="utf-8") as fh:
    B = json.load(fh)
with open(os.path.join(GEC, "p34_c_zincir.json"), encoding="utf-8") as fh:
    C = json.load(fh)
if not C["gecti"]:
    raise SystemExit("DUR: zincir testi GECMEDI.")

a0 = np.load(os.path.join(GEC, "p34_a0.npy"))
V = np.load(os.path.join(GEC, "p34_V30.npy"))
L30 = np.load(os.path.join(GEC, "p34_L30.npy"))
r30 = np.load(os.path.join(GEC, "p34_r30.npy"))
r28 = np.load(os.path.join(GEC, "p34_r28.npy"))
BAZ = np.load(os.path.join(GEC, "p34_dik_baz.npy"))
N = A["N"]
kL30 = A["c30_yeni"]["kL"]
kL28 = A["c28_p33_yeniden_uretim"]["kL"]
nrm30 = A["c30_yeni"]["r_hat_norm2"]
kaz30 = A["c30_yeni"]["beklenen_kazanc"]
RHO_BAZ = np.array(B["ortonormal_baz"]["rho"])
EK_KAZANC = float((RHO_BAZ**2).sum())

# --- TABAN --------------------------------------------------------------
MSE_durust = M0 - kaz30 - EK_KAZANC          # buzmenin GERCEK kazanc beklentisi
MSE_cebirsel = M0 - 2 * kL30 + nrm30 - EK_KAZANC
print(f"TABAN (span30 + olculmus dik baz):")
print(f"  durust  MSE {MSE_durust:.7f} -> skor {np.sqrt(MSE_durust):.6f}")
print(f"  cebirsel MSE {MSE_cebirsel:.7f} -> skor {np.sqrt(MSE_cebirsel):.6f}")
print(f"  (p33 muadili: 1.0008685 / 1.0017377 tabanindan)")

taban_log = a0 + r30 + BAZ.T @ RHO_BAZ

# ==========================================================================
# 1) IC-SPAN EKSTRAPOLASYON -- olculmus L'lerle TAM hesaplanabilir
# ==========================================================================
EKS = {}
# (a) r30'u s ile olcekle:  MSE(s) = M0 - 2 s kL30 + s^2 nrm30
s_opt = kL30 / nrm30
EKS["olcek_s_r30"] = {
    "aciklama": "r_hat30 -> s*r_hat30 (ic-span global olcek)",
    "s_optimum": float(s_opt),
    "izgara": [
        {"s": float(s),
         "MSE": float(M0 - 2 * s * kL30 + s * s * nrm30 - EK_KAZANC),
         "skor": float(np.sqrt(max(M0 - 2 * s * kL30 + s * s * nrm30 - EK_KAZANC, 1e-12)))}
        for s in (1.0, s_opt, 1.5 * s_opt, 2 * s_opt, 3 * s_opt)
    ],
}
# (b) H1/H2 duzeltmesi yonunde:  r = r28 + t*(r30-r28)
D = r30 - r28
nD = float((D * D).mean())
LD = kL30 - kL28                       # <r, D> -- OLCULMUS L'lerden
t_opt = LD / nD
Q28 = float((r28 * r28).mean())
X28 = float((r28 * D).mean())


def mse_t(t):
    rr = Q28 + 2 * t * X28 + t * t * nD
    return M0 - 2 * (kL28 + t * LD) + rr - EK_KAZANC


EKS["H_duzeltme_yonu"] = {
    "aciklama": "r = r_hat28 + t*(r_hat30-r_hat28); t=1 YENI cozum",
    "ic_carpim_L_D": float(LD),
    "norm2_D": nD,
    "t_optimum": float(t_opt),
    "izgara": [{"t": float(t), "MSE": float(mse_t(t)),
                "skor": float(np.sqrt(max(mse_t(t), 1e-12)))}
               for t in (0.0, 1.0, float(t_opt), 1.5 * t_opt, 2 * t_opt, 3 * t_opt)],
}
print("\n=== IC-SPAN EKSTRAPOLASYON ===")
for ad, e in EKS.items():
    print(f"  {ad}: optimum {e.get('s_optimum', e.get('t_optimum')):.4f}")
    for g in e["izgara"]:
        k = "s" if "s" in g else "t"
        print(f"    {k}={g[k]:+.4f}  skor {g['skor']:.6f}")

# ==========================================================================
# 2) H1/H2 YONLERININ KENDI OPTIMUMU (gorevdeki ana fikir)
# ==========================================================================
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
H_ANA = {}
for j, ad in [(28, "H1"), (29, "H2")]:
    d = V[:, j]
    # span'in GERI KALANINA (o yon haric) dik bileseni
    idx = [i for i in range(V.shape[1]) if i != j]
    Vo = V[:, idx]
    Go = (Vo.T @ Vo) / N
    c = np.linalg.pinv(Go, rcond=1e-6) @ ((Vo.T @ d) / N)
    dp = d - Vo @ c
    npd = float(np.sqrt(float((dp * dp).mean())))
    rho = (L30[j] - float(c @ L30[idx])) / npd
    H_ANA[ad] = {
        "dik_norm": npd, "rho_dik": float(rho),
        "optimal_katsayi_t": float(rho / npd),
        "kazanc_MSE": float(rho * rho),
        "izgara_t_carpani": [
            {"carpan": float(m), "MSE_degisimi": float(-2 * m * rho * rho + m * m * rho * rho),
             "skor": float(np.sqrt(max(MSE_durust - 2 * m * rho * rho + m * m * rho * rho, 1e-12)))}
            for m in (1.0, 1.5, 2.0, 3.0)
        ],
    }
    print(f"\n  {ad} yonu (span'in geri kalanina dik): ||.||={npd:.5f} "
          f"rho={rho:+.6f} optimal t={rho / npd:+.4f} kazanc={rho * rho:.7f}")
    print("    NOT: kazanc t=optimum'da MAKSIMUM; carpan>1 ZARARLI:")
    for g in H_ANA[ad]["izgara_t_carpani"]:
        print(f"      carpan {g['carpan']:.1f} -> skor {g['skor']:.6f}")

# ==========================================================================
# 3) SPAN-DISI YON TARAMASI (hail-mary adaylari)
# ==========================================================================
te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"),
                 usecols=["id", "tanim"], dtype={"id": str, "tanim": str})
IDS = te["id"].to_numpy()
tanim = te["tanim"].to_numpy()


def oku_yol(yol):
    d = pd.read_csv(yol, dtype={"id": str})
    assert np.array_equal(d["id"].to_numpy(), IDS), yol
    return np.log1p(d["tuketim"].to_numpy("float64"))


soguk = np.load(os.path.join(AC, "p06_test_soguk_maske.npy"))
g = _ham()
g = g[g["tarih"] < pd.Timestamp(KESIM["TEST"])]
lg = np.log1p(np.clip(g["tuketim"].to_numpy("float64"), 0, None))
gb = pd.DataFrame({"tanim": g["tanim"].to_numpy(), "l": lg}).groupby("tanim")["l"]
mu = gb.mean().reindex(pd.Index(tanim)).to_numpy("float64")
sdv = gb.std().reindex(pd.Index(tanim)).to_numpy("float64")
ok = (~soguk) & np.isfinite(mu) & np.isfinite(sdv) & (sdv > 0)
print(f"\ncapasi olan sicak satir: {int(ok.sum())}")


def kapak(L, k):
    L = L.copy()
    L[ok] = np.clip(L[ok], mu[ok] - k * sdv[ok], mu[ok] + k * sdv[ok])
    return L


ADAY_YON = {}
# (i) kuyruk kapagi merdiveni -- H1 ekseni POZITIF olculdu (+0.0155), daha sert kap
for k in (6.0, 4.0, 3.0, 2.0):
    ADAY_YON[f"kapak_k{k:g}"] = kapak(taban_log, k) - taban_log
# (ii) yapisal dosyalar (hicbiri gonderilmedi)
for ad, yol in [
    ("harman_ESIT", os.path.join(AC, "p20_harman_ESIT_V1_seviyesiz.csv")),
    ("esit_olu50", os.path.join(AC, "p21_esit_olu50.csv")),
    ("sulama_l100", os.path.join(AC, "p31_sulama_dogrudan_l100.csv")),
    ("beta50", os.path.join(AC, "p28_beta50.csv")),
]:
    if os.path.exists(yol):
        ADAY_YON[ad] = oku_yol(yol) - a0

# hepsini S30'a VE olculmus dik baza dik yap
TARAMA = []
BIRIM = {}
for ad, x in ADAY_YON.items():
    y = x - V @ (Gi @ ((V.T @ x) / N))
    for b in BAZ:
        y = y - float((y * b).mean()) * b
    nn = float(np.sqrt(float((y * y).mean())))
    if nn < 1e-5:
        print(f"  {ad}: dik bilesen yok ({nn:.2e}), atlandi")
        continue
    u = y / nn
    BIRIM[ad] = u
    TARAMA.append({"ad": ad, "dik_norm": nn, "maks_abs_birim": float(np.abs(u).max()),
                   "yogunlasma_p99": float(np.quantile(np.abs(u), 0.99))})
    print(f"  {ad:16s} dik_norm={nn:.5f}  maks|u|={np.abs(u).max():8.2f}  "
          f"p99|u|={np.quantile(np.abs(u), 0.99):6.3f}")

# ==========================================================================
# 4) P(ilk 3)
# ==========================================================================
# Olculmus span-disi rho'lar (5 sonda): bunlar rho icin AMPIRIK ONSELI verir.
RHO_OLC = np.array([s["rho_dik"] for s in B["S28"]])
n = len(RHO_OLC)
ro, rs = float(RHO_OLC.mean()), float(RHO_OLC.std(ddof=1))
olcek = rs * np.sqrt(1 + 1 / n)
print(f"\nolculmus span-disi rho'lar (n={n}): {[f'{x:+.4f}' for x in RHO_OLC]}")
print(f"  ort {ro:+.5f}  sd {rs:.5f}  ongoru olcegi (t_{n - 1}) {olcek:.5f}")

# public/private bolunme gurultusu: delta = aday - taban icin
# g_i = delta_i*(delta_i + 2 r_i);  sd(ortalama farki) ~ 2*sd(g)/sqrt(N)
r_kalan_rms = float(np.sqrt(max(M0 - nrm30, 0.0)))


def sigma_bolunme(delta):
    yak = 2.0 * float(np.sqrt(float((delta * delta).mean()))) * r_kalan_rms
    return 2.0 * yak / np.sqrt(N)


kappa_ilk3 = float(np.sqrt(max(MSE_durust - HEDEF3_MSE, 0.0)))
print(f"\nGEREKEN kappa (rho=kappa ise tam tutar) = {kappa_ilk3:.5f}")
print(f"  gereken rho / en buyuk olculmus |rho| = {kappa_ilk3 / np.abs(RHO_OLC).max():.1f} kat")


def P_ilk3(kap, sig_ek=0.0):
    """P(skor <= HEDEF3). rho ~ t_{n-1}(ro, olcek); + bolunme gurultusu."""
    if kap <= 0:
        return float(np.sqrt(MSE_durust) <= HEDEF3)
    esik_rho = (MSE_durust + kap * kap - HEDEF3_MSE) / (2 * kap)
    olc = float(np.sqrt(olcek**2 + (sig_ek / (2 * kap)) ** 2))
    return float(1 - stats.t.cdf((esik_rho - ro) / olc, n - 1))


# ==========================================================================
# 5) ADAY IZGARASI
# ==========================================================================
# SECIM OLCUTU: yon NORMU onemsiz (birime cevriliyor). Onemli olan ISARETIN
# olculmus bir kanita dayanmasi. Tek boyle eksen KUYRUK KAPAGI: H1 (k=8 kapak)
# span-disi rho'su +0.0164 olculdu -> "daha sert kirp" ayni ailenin devami ve
# isareti ARTI. sulama/harman_ESIT eksenlerinde isaret bilgisi YOK (sulama CV
# kokenli, tasima orani negatif olculmus), harman ekseninin isareti H2'den
# NEGATIF cikti ama dik artigi kucuk.
SEC = "kapak_k2" if "kapak_k2" in BIRIM else max(TARAMA, key=lambda r: r["dik_norm"])["ad"]
SEC_GEREKCE = (
    "kuyruk kapagi ekseni: H1 (k=8) span-disi rho = +0.0164 OLCULDU; k=2 kapak "
    "ayni ailenin devami ve ISARETI bilinen tek span-disi yon. Birim yonun "
    "olcegi ||dik||=0.1058, gereken kappa 0.1077 -> carpan ~1.02, yani aday "
    "fiilen 'k=2 kuyruk kapagini uygula' demektir; yapay buyutme YOK."
)
print(f"\nSECILEN hail-mary yonu: {SEC}")
u_hm = BIRIM[SEC]
print(f"  yon isaret dagilimi: negatif kayma orani "
      f"{float((u_hm < 0).mean()):.4f}, pozitif {float((u_hm > 0).mean()):.4f}")

IZGARA = []
for ad, kap in [("A_muhafazakar_kappa0", 0.0),
                ("B_orta_yarim", 0.5 * kappa_ilk3),
                ("C_agresif_ilk3", kappa_ilk3),
                ("D_asiri_1.5x", 1.5 * kappa_ilk3)]:
    Lc = taban_log + kap * u_hm
    delta = Lc - taban_log
    sig = sigma_bolunme(Lc - np.log1p(np.expm1(a0)))  # tabana gore toplam fark
    bek = float(np.sqrt(max(MSE_durust - 2 * kap * ro + kap * kap, 1e-12)))
    kotu = float(np.sqrt(max(MSE_durust + kap * kap, 1e-12)))
    IZGARA.append({
        "ad": ad, "kappa": float(kap),
        "beklenen_skor": bek,
        "en_kotu_rho0": kotu,
        "rho_kappa_ise": float(np.sqrt(max(MSE_durust - kap * kap, 1e-12))),
        "P_ilk3": P_ilk3(kap),
        "P_ilk3_bolunme_dahil": P_ilk3(kap, sig),
        "sigma_bolunme_MSE": float(sig),
        "log_kayma_rms": float(np.sqrt(float((delta * delta).mean()))),
        "log_kayma_maks_abs": float(np.abs(delta).max()),
    })
    print(f"  {ad:22s} kappa={kap:.5f}  beklenen {bek:.5f}  en kotu {kotu:.5f}  "
          f"P(ilk3)={P_ilk3(kap):.5f} (bolunme dahil {P_ilk3(kap, sig):.5f})")

# ==========================================================================
# 6) CSV URETIMI + DOGRULAMA
# ==========================================================================
def yaz(ad, Lc):
    x = np.clip(np.expm1(np.maximum(Lc, 0.0)), 0, None)
    yol = os.path.join(AC, f"{ad}.csv")
    pd.DataFrame({"id": IDS, "tuketim": x}).to_csv(yol, index=False)
    geri = pd.read_csv(yol, dtype={"id": str})
    v = geri["tuketim"].to_numpy("float64")
    return {
        "yol": f"p_kalici/aday_csv/{ad}.csv",
        "n_satir": int(len(geri)),
        "satir_dogru": bool(len(geri) == 714688),
        "id_sirasi_birebir": bool(np.array_equal(geri["id"].to_numpy(), IDS)),
        "NaN": int(geri["tuketim"].isna().sum()),
        "negatif": int((v < 0).sum()),
        "sonsuz": int((~np.isfinite(v)).sum()),
        "maks": float(v.max()), "toplam_tuketim": float(v.sum()),
    }


DEN = {}
DEN["p34_A_saf_span"] = yaz("p34_A_saf_span", taban_log)
DEN["p34_B_orta"] = yaz("p34_B_orta", taban_log + 0.5 * kappa_ilk3 * u_hm)
DEN["p34_D_asiri"] = yaz("p34_D_asiri", taban_log + 1.5 * kappa_ilk3 * u_hm)
DEN["p34_son_hak"] = yaz("p34_son_hak", taban_log + kappa_ilk3 * u_hm)
# ALTERNATIF hail-mary: gorevdeki H2 fikri -- harman ekseninde NEGATIF yon
if "harman_ESIT" in BIRIM:
    DEN["p34_E_harman_negatif"] = yaz(
        "p34_E_harman_negatif", taban_log - kappa_ilk3 * BIRIM["harman_ESIT"])

print("\n=== ADAY DOGRULAMASI ===")
for k, v in DEN.items():
    print(f"  {k}: satir {v['n_satir']} id {v['id_sirasi_birebir']} NaN {v['NaN']} "
          f"neg {v['negatif']} inf {v['sonsuz']} maks {v['maks']:.1f} "
          f"toplam {v['toplam_tuketim']:.0f}")

CIK = {
    "00_KURAL": "Kaggle gonderimi YOK, submissions/ yazilmadi, commit yok",
    "01_cebir_30yon": A,
    "02_zincir": {k: C[k] for k in ("gecti", "en_buyuk_skor_farki",
                                    "en_buyuk_rho_hatasi", "u_dikligi_maks")},
    "03_capalar": B,
    "04_taban": {
        "MSE_durust": MSE_durust, "skor_durust": float(np.sqrt(MSE_durust)),
        "MSE_cebirsel": MSE_cebirsel, "skor_cebirsel": float(np.sqrt(MSE_cebirsel)),
        "ek_kazanc_olculmus_dik_baz": EK_KAZANC,
        "p33_karsiligi_skor": 1.0008684670312924,
    },
    "05_ic_span_ekstrapolasyon": EKS,
    "06_H1_H2_yon_optimumu": H_ANA,
    "07_yon_taramasi": TARAMA,
    "08_secilen_hail_mary_yonu": {"yon": SEC, "gerekce": SEC_GEREKCE},
    "09_rho_onseli": {"olculmus_rho": RHO_OLC.tolist(), "n": n, "ort": ro,
                      "sd": rs, "ongoru_olcegi_t": olcek,
                      "gereken_kappa_ilk3": kappa_ilk3,
                      "gereken_kat": float(kappa_ilk3 / np.abs(RHO_OLC).max())},
    "10_aday_izgarasi": IZGARA,
    "11_dosya_denetimi": DEN,
}
with open(os.path.join(PK, "p34_son_hak.json"), "w", encoding="utf-8") as fh:
    json.dump(CIK, fh, ensure_ascii=False, indent=1)
print("\n-> p_kalici/p34_son_hak.json")
