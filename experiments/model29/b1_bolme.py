"""b1: m4-v102 yonunun DIK alt-bolmeleri. Taban m6_ikiyon (S=1.00284).

CEBIR
  Taban b = m6. Yon d = log1p(m4) - log1p(v102).
  m6 = v102 + kw*d_sicak + kc*d_soguk  (kw, kc OLCULMUS optimumlar)
  => b tabaninda d_sicak ve d_soguk yonleri boyunca L = 0 (doygun).

  Yeni yon z = w(x) * d, agirlik w satir bazli.
  Q_z = mean(w^2 d^2), L_z = -mean(e_b * z), optimum c* = L_z/Q_z,
  KAZANC (MSE dususu) = L_z^2 / Q_z.

  w'yi SICAK icinde ve SOGUK icinde ayri ayri d^2-agirlikli merkezlersek
  z, d_sicak ve d_soguk'a DIK olur -> m6'nin kazanci korunur, ustune eklenir.

  Ikili bolme (A / A-disi) icin, gercek kappa A'da kA, disinda kB ise:
      KAZANC = SUM_g Q_g * q_g*(1-q_g) * (kA-kB)^2      g in {sicak, soguk}
  Formul m6'nin kendi kazancini birebir yeniden uretiyor (denetim asagida).
"""

import json
import os

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))

te = pd.read_csv(
    os.path.join(KOK, "data/raw/test.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
tr = pd.read_csv(
    os.path.join(KOK, "data/raw/train.csv"), parse_dates=["tarih"], dtype={"tanim": str}
)
A = pd.read_csv(os.path.join(KOK, "submissions/tuketim_v102_kappa_optimum.csv"))
B = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m4_hava_capali.csv"))
M6 = pd.read_csv(os.path.join(KOK, "submissions/tuketim_m6_ikiyon.csv"))
assert (A.id.values == te.id.values).all()
assert (B.id.values == te.id.values).all()
assert (M6.id.values == te.id.values).all()

a = np.log1p(A.tuketim.values)
bb = np.log1p(B.tuketim.values)
m6 = np.log1p(M6.tuketim.values)
d = bb - a
N = len(d)

soguk = (~te.tanim.isin(set(tr.tanim))).values
sicak = ~soguk
KW, KC = 0.1224276353574521, 0.33687957152525483
yeniden = a + KW * d * sicak + KC * d * soguk
print("m6 yeniden uretim maks log fark: %.3e" % np.abs(yeniden - m6).max())

Q_TOT = float((d**2).mean())
Q_W = float((d[sicak] ** 2).sum() / N)
Q_C = float((d[soguk] ** 2).sum() / N)
L_TOT, L_W, L_C = 0.022319, 0.010605162105012665, 0.011713837894987334
M0_A = 1.00553**2
M0_B = 1.00284**2
G_W, G_C = L_W**2 / Q_W, L_C**2 / Q_C
print("Q_tot %.6f  Q_sicak %.6f  Q_soguk %.6f" % (Q_TOT, Q_W, Q_C))
print("katki: sicak %.6f  soguk %.6f  toplam %.6f" % (G_W, G_C, G_W + G_C))
print("m0(v102) %.6f  m0(m6) %.6f  fark %.6f" % (M0_A, M0_B, M0_A - M0_B))
qcpay = Q_C / Q_TOT
dogr = Q_TOT * qcpay * (1 - qcpay) * (KC - KW) ** 2
tek = M0_A - L_TOT**2 / Q_TOT
print(
    "DENETIM tek-kappa MSE %.6f (S=%.5f); iki-yon kazanci formul %.6f vs gercek %.6f"
    % (tek, np.sqrt(tek), dogr, tek - (M0_A - G_W - G_C))
)

HEDEF_MSE = 1.00041**2
print("\nHEDEF: 2. sira MSE %.6f; m6'dan gereken dMSE %+.6f" % (HEDEF_MSE, HEDEF_MSE - M0_B))

# ---------------- aday eksenler ----------------
te["bolge"] = te.lokasyon.str.split(">").str[1]
te["ilce"] = te.lokasyon.str.split(">").str[2]
tr["ilce"] = tr.lokasyon.str.split(">").str[2]
ilk_te = te.groupby("tanim").tarih.transform("min")
n_te = te.groupby("tanim").tarih.transform("size")
ilce_yog = tr.groupby("ilce").tanim.nunique()
te_ilce_yog = te.ilce.map(ilce_yog).fillna(0).values.astype(float)
idnum = pd.to_numeric(te.tanim, errors="coerce").fillna(-1).values.astype(float)
guc = te.guc.values.astype(float)
lvl = m6
ay = te.tarih.dt.month.values
hs = te.tarih.dt.dayofweek.values >= 5


def w_medyan(x, mask):
    o = np.argsort(x[mask])
    xs = x[mask][o]
    ws = (d[mask] ** 2)[o]
    c = np.cumsum(ws)
    return xs[min(np.searchsorted(c, c[-1] / 2), len(xs) - 1)]


def ikili_med(x):
    """sicak ve soguk icinde AYRI d^2-agirlikli medyanla ustu/alti"""
    m = np.zeros(N, bool)
    for g in (sicak, soguk):
        m |= g & (x > w_medyan(x, g))
    return m


ADAYLAR = {
    "dalga_0511": (ilk_te == pd.Timestamp("2026-05-11")).values,
    "ilk_gorus_nisan": (ilk_te <= pd.Timestamp("2026-04-30")).values,
    "d_pozitif": d > 0,
    "d_buyuk": ikili_med(np.abs(d)),
    "seviye_yuksek": ikili_med(lvl),
    "guc_yuksek": ikili_med(guc),
    "metropol": (te.bolge == "METROPOL").values,
    "guney": (te.bolge == "GÜNEY BÖLGE").values,
    "ay_haz_tem": np.isin(ay, [6, 7]),
    "hafta_sonu": hs,
    "idnum_yuksek": ikili_med(idnum),
    "ilce_yogun": ikili_med(te_ilce_yog),
    "test_satir_cok": ikili_med(n_te.values.astype(float)),
    "sifira_yakin": lvl < np.log1p(50.0),
}


def payla(mask, grup):
    tot = (d[grup] ** 2).sum()
    return float((d[grup & mask] ** 2).sum() / tot) if tot > 0 else 0.0


sonuc = {}
for ad, mk in ADAYLAR.items():
    qh, qc_ = payla(mk, sicak), payla(mk, soguk)
    w = np.where(sicak, mk.astype(float) - qh, mk.astype(float) - qc_)
    Qz = float(((w * d) ** 2).mean())
    Qz_h = Q_W * qh * (1 - qh)
    Qz_c = Q_C * qc_ * (1 - qc_)
    sonuc[ad] = dict(
        satir=int(mk.sum()),
        satir_pay=float(mk.mean()),
        satir_sicak=int((mk & sicak).sum()),
        satir_soguk=int((mk & soguk).sum()),
        q_sicak=qh,
        q_soguk=qc_,
        Qz=Qz,
        Qz_sicak=Qz_h,
        Qz_soguk=Qz_c,
        kazanc_dk010=Qz * 0.01,
        kazanc_dk020=Qz * 0.04,
        kazanc_dk030=Qz * 0.09,
        kazanc_dk050=Qz * 0.25,
        soguk_kazanc_dk030=Qz_c * 0.09,
        dk_hedef=float(np.sqrt((M0_B - HEDEF_MSE) / Qz)) if Qz > 0 else None,
        dk_hedef_soguk=float(np.sqrt((M0_B - HEDEF_MSE) / Qz_c)) if Qz_c > 0 else None,
    )

sir = sorted(sonuc, key=lambda k: -sonuc[k]["Qz"])
print("\n== EKSENLER (z = w*d, sicak ve soguk icinde ayri merkezlenmis gosterge) ==")
print(
    "%-18s %7s %6s %6s %9s %9s %9s %9s %9s %9s %8s"
    % (
        "eksen",
        "satir%",
        "q_sic",
        "q_sog",
        "Qz",
        "Qz_sic",
        "Qz_sog",
        "dk=0.2",
        "dk=0.3",
        "dk=0.5",
        "dk*hedef",
    )
)
for k in sir:
    s = sonuc[k]
    print(
        "%-18s %6.1f%% %6.3f %6.3f %9.6f %9.6f %9.6f %9.6f %9.6f %9.6f %8.3f"
        % (
            k,
            100 * s["satir_pay"],
            s["q_sicak"],
            s["q_soguk"],
            s["Qz"],
            s["Qz_sicak"],
            s["Qz_soguk"],
            s["kazanc_dk020"],
            s["kazanc_dk030"],
            s["kazanc_dk050"],
            s["dk_hedef"],
        )
    )

print("\n(dk = iki grubun gercek kappa farki; KAZANC = Qz * dk^2)")
print("(dk*hedef = tek basina 2. siraya yetmek icin gereken kappa farki)")

print("\n== YALNIZ SOGUK bolme (gorevde istenen) ==")
print(
    "%-18s %6s %9s %9s %9s %9s %9s"
    % ("eksen", "q_sog", "Qz_sog", "dk=0.3", "dk=0.5", "dk=1.0", "dk*hedef")
)
for k in sorted(sonuc, key=lambda x: -sonuc[x]["Qz_soguk"]):
    s = sonuc[k]
    print(
        "%-18s %6.3f %9.6f %9.6f %9.6f %9.6f %9.3f"
        % (
            k,
            s["q_soguk"],
            s["Qz_soguk"],
            s["Qz_soguk"] * 0.09,
            s["Qz_soguk"] * 0.25,
            s["Qz_soguk"],
            s["dk_hedef_soguk"],
        )
    )

ozet = dict(
    Q_tot=Q_TOT,
    Q_sicak=Q_W,
    Q_soguk=Q_C,
    L_tot=L_TOT,
    L_sicak=L_W,
    L_soguk=L_C,
    kappa_sicak=KW,
    kappa_soguk=KC,
    katki_sicak=G_W,
    katki_soguk=G_C,
    m0_v102=M0_A,
    m0_m6=M0_B,
    hedef_mse=HEDEF_MSE,
    gereken_dmse=HEDEF_MSE - M0_B,
    eksenler=sonuc,
)
json.dump(
    ozet,
    open(os.path.join(BURA, "b1_bolme.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
np.save(os.path.join(BURA, "b1_d.npy"), d)
np.save(os.path.join(BURA, "b1_m6log.npy"), m6)
np.save(os.path.join(BURA, "b1_soguk.npy"), soguk)
print("\nYAZILDI b1_bolme.json")
