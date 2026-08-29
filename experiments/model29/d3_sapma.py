"""DUSMANCA DENETIM -- 4. asama: 0,00008 sapmasinin kaynagi + J-parca risk fiyati.
1) Gecmis 2 ucgenin sapmasi ile bugunku sapmayi ayni modelde karsilastir.
2) "public yarim" hipotezinin ongordugu sd'leri her yon icin AYRI olc.
3) J parca icin prob-coz sisteminin KOSUL SAYISINI ve yuvarlama gurultusunun
   buyutme carpanini simule et -> kac parca guvenli?
Sadece OKUR.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BURA = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(KOK, "submissions")
R = {}
oku = lambda n: np.log1p(pd.read_csv(os.path.join(SUB, n)).iloc[:, 1].values)

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
N = len(te)
soguk = (~te.tanim.isin(set(tr.tanim))).values
rg = np.random.default_rng(99)


def sd_Q_yarim(dv, msk=None, T=1500):
    u = dv**2 if msk is None else (dv**2) * msk
    return float(np.std([u[rg.random(N) < 0.5].mean() for _ in range(T)]))


# ---------- 1) GECMIS UCGENLER: public-yarim hipotezinin ongordugu sd
gec = []
for taban, yon, harman, k, s0, s1, sger in [
    (
        "tuketim_v101_hepsi.csv",
        "tuketim_v83_sicak_optimum.csv",
        "tuketim_v102_kappa_optimum.csv",
        0.54098,
        1.01614,
        1.01318,
        1.00553,
    ),
    (
        "tuketim_v80_optimum.csv",
        "tuketim_v81_sicak08.csv",
        "tuketim_v83_sicak_optimum.csv",
        0.31075,
        1.01341,
        1.01429,
        1.01318,
    ),
]:
    dv = oku(yon) - oku(taban)
    Q = float((dv**2).mean())
    m0 = s0**2
    Lv = (m0 + Q - s1**2) / 2
    kk = Lv / Q
    ong = float(np.sqrt(m0 - Lv**2 / Q))
    sdQ = sd_Q_yarim(dv)
    # Turetim: gerceklesen_MSE - ongorulen_MSE = k(1-k) * (Q_tum - Q_public)
    sd_sapma = abs(kk * (1 - kk)) * sdQ / (2 * ong)
    # yuvarlama bandi (m0, m1 5 hane)
    ban = []
    for _ in range(20000):
        e0, e1 = rg.uniform(-5e-6, 5e-6, 2)
        mm0 = (s0 + e0) ** 2
        LL = (mm0 + Q - (s1 + e1) ** 2) / 2
        ban.append(np.sqrt(mm0 - LL**2 / Q))
    ban = np.array(ban)
    gec.append(
        dict(
            harman=harman,
            kappa=kk,
            Q=Q,
            ongoru=ong,
            gerceklesen=sger,
            sapma=sger - ong,
            yuvarlama_bandi=[float(ban.min()), float(ban.max())],
            yuvarlama_yarim_genislik=float((ban.max() - ban.min()) / 2),
            bantta_mi=bool(ban.min() - 5e-6 <= sger <= ban.max() + 5e-6),
            sd_Q_yarim=sdQ,
            public_yarim_hipotezi_sd_sapma=sd_sapma,
            gozlenen_kac_sd=float(abs(sger - ong) / sd_sapma),
        )
    )
R["1_gecmis_ucgenler"] = gec

# ---------- 2) BUGUN: m6
V, M4 = oku("tuketim_v102_kappa_optimum.csv"), oku("tuketim_m4_hava_capali.csv")
d = M4 - V
Qw = float((d[~soguk] ** 2).sum() / N)
Qc = float((d[soguk] ** 2).sum() / N)
m0 = 1.00553**2
Ltot = (m0 + Qw + Qc - 1.043**2) / 2
TW, TC = 0.50, 0.022319 / (Qw + Qc)
Lw = (m0 + TW**2 * Qw + TC**2 * Qc - 2 * TC * 0.022319 - 1.00946**2) / (2 * (TW - TC))
Lc = 0.022319 - Lw
kw, kc = Lw / Qw, Lc / Qc
ong6 = float(np.sqrt(m0 - Lw**2 / Qw - Lc**2 / Qc))
# analitik duyarlilik: gerceklesen - ongoru = A*dw + B*dc  (dw=Qw_tum-Qw_pub)
ew_w = (TW**2 - TC) / (2 * (TW - TC))
ew_c = -TC * (1 - TC) / (2 * (TW - TC))
A = 2 * kw * ew_w + 2 * kc * (0.5 - ew_w) - kw**2
B = 2 * kw * ew_c + 2 * kc * (0.5 - ew_c) - kc**2
sdw, sdc = sd_Q_yarim(d, ~soguk), sd_Q_yarim(d, soguk)
sd6 = float(np.sqrt((A * sdw) ** 2 + (B * sdc) ** 2) / (2 * ong6))
R["2_bugun_m6"] = dict(
    ongoru=ong6,
    gerceklesen=1.00284,
    sapma=1.00284 - ong6,
    buyutme_carpani=float(1 / (2 * (TW - TC))),
    A_katsayisi=float(A),
    B_katsayisi=float(B),
    sd_Qw_yarim=sdw,
    sd_Qc_yarim=sdc,
    public_yarim_hipotezi_sd_sapma=sd6,
    gozlenen_kac_sd=float(abs(1.00284 - ong6) / sd6),
    gereken_bagil_Q_kaymasi=float(
        (1.00284**2 - ong6**2) / (A * Qw + B * Qc) if (A * Qw + B * Qc) else float("nan")
    ),
)

# ---------- 3) J PARCA: prob-coz sisteminin gurultu buyutmesi
ilk_tr = tr.groupby("tanim").tarih.min()
kuyruk = (~soguk) & (te.tanim.map(ilk_tr) >= pd.Timestamp("2026-03-26")).values
cekirdek = (~soguk) & ~kuyruk
ilk_te = te.groupby("tanim").tarih.min()
dalga = te.tanim.map(ilk_te).eq(pd.Timestamp("2026-05-11")).values
bolmeler = {
    2: [soguk, ~soguk],
    3: [soguk, kuyruk, cekirdek],
    4: [soguk & dalga, soguk & ~dalga, (~soguk) & dalga, (~soguk) & ~dalga],
}
riskler = {}
for J, parcalar in bolmeler.items():
    Qj = np.array([float((d[m] ** 2).sum() / N) for m in parcalar])
    if (Qj <= 0).any():
        continue
    # prob tasarimi: prob_r'de parca r'ye 0.5, digerlerine TC (m4 = hepsi 1)
    T = [np.ones(J)]
    for r in range(J - 1):
        t = np.full(J, TC)
        t[r] = 0.5
        T.append(t)
    T = np.array(T)  # (J, J) : her satir bir olcum
    # MSE_i = m0 - 2 t_i . Lvec + t_i^2 . Qvec  ->  2 T Lvec = m0 + (T^2)Q - MSE
    M = 2 * T
    ko = float(np.linalg.cond(M))
    Minv = np.linalg.inv(M)
    # her olcumun 5 hane yuvarlamasi -> MSE hatasi ~ 2*1.005*5e-6
    sMSE = 2 * 1.005 * 5e-6 / np.sqrt(3)  # uniform sd
    covL = Minv @ (np.eye(J) * sMSE**2) @ Minv.T
    sdL = np.sqrt(np.diag(covL))
    # ongoru MSE* = m0 - sum L_j^2/Q_j ; hata ~ sum 2 L_j/Q_j * dL_j = 2 k_j dL_j
    kvec = np.array([kw, kc]) if J == 2 else np.full(J, Ltot / (Qw + Qc))
    if J != 2:
        kvec = np.full(J, 0.23)  # tipik olcek
    sd_ong = float(np.sqrt(np.sum((2 * kvec * sdL) ** 2)) / (2 * 1.003))
    # public-yarim Q gurultusu
    sdQj = np.array([sd_Q_yarim(d, m, T=400) for m in parcalar])
    sd_pub = float(np.sqrt(np.sum((0.3 * sdQj) ** 2)) / (2 * 1.003))
    riskler[J] = dict(
        parca_satirlari=[int(m.sum()) for m in parcalar],
        Q_parca=[float(q) for q in Qj],
        kosul_sayisi=ko,
        sd_L_yuvarlamadan=[float(x) for x in sdL],
        sd_ongoru_yuvarlamadan_RMSLE=sd_ong,
        sd_ongoru_public_yarimdan_RMSLE=sd_pub,
        toplam_sd_RMSLE=float(np.sqrt(sd_ong**2 + sd_pub**2)),
        gereken_prob_sayisi=J - 1,
        asiri_uyum_MSE=float(J * 4 * m0 / N),
        asiri_uyum_RMSLE=float(J * 4 * m0 / N / (2 * 1.003)),
    )
R["3_J_parca_riski"] = riskler
R["3_yorum"] = (
    "Bagliyici kisit ASIRI UYUM DEGIL (parca basina ~2.8e-6 RMSLE), "
    "COZUM SISTEMININ KOSUL SAYISI ve her parcanin bir GONDERIM HAKKI yemesi."
)

json.dump(R, open(os.path.join(BURA, "d3_sapma.json"), "w", encoding="utf-8"), indent=1)
print(json.dumps(R, indent=1))
print("\nYAZILDI d3_sapma.json")
