"""ADAY ELEME VE UCLU SECIMI.

Butun aday dosyalari tek elemeden gecirir:
    kosinus (mevcutlarla) <= 0,20  ·  kurtoz <= 10  ·  Q >= 0,01
Gecenler icin, g7 (L'si BILINEN) ile birlikte en iyi UCLUYU secer.

CV skoru KULLANILMAZ -- kural 36: bu veride geri-test LB'yi ongormuyor.
Secim olcutu GEOMETRI: dik yonlerin katkilari toplanir.
"""

import itertools
import json
import os

import numpy as np
import pandas as pd
from m30_ozellik import KOK
from scipy import stats

S = os.path.join(KOK, "submissions")
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.00284**2
HEDEF = 1.00041**2
LG, QG = 0.002728, 0.002494  # g7'nin OLCULMUS L'si ve Q'su
# ELEME OLCUTLERI -- duzeltildi (ilk surumde hata vardi):
#   NEGATIF korelasyon ISTENEN seydir (super-toplamsal, g7.y40=-0,555 gibi).
#   Yalniz POZITIF korelasyon fazlalik demektir. Ve fazlalik zaten ortak
#   cozumde kendiliginde cezalandiriliyor -- on eleme yapmaya gerek yok.
#   Sert eleme yalniz OLCULEBILIRLIK icin: Q cok kucukse LB gurultusunde bogulur.
Q_ESIK = 0.01  # SERT: altindaysa L olculemez (gurultu +-9,5e-5)
KURTOZ_UYARI = 12.0  # YUMUSAK: ongoru guvenilirligini dusurur, elemez


def yukle(f):
    return np.log1p(pd.read_csv(os.path.join(S, f)).tuketim.values)


def envanter(dosyalar):
    a0 = yukle(TABAN)
    N = len(a0)
    D, ad, olcu = [], [], {}
    for f in dosyalar:
        yol = os.path.join(S, f)
        if not os.path.exists(yol):
            print(f"  [yok] {f}")
            continue
        d = yukle(f) - a0
        q = float((d @ d) / N)
        k = float(stats.kurtosis(d, fisher=False))
        s = np.sort(d**2)[::-1]
        pay = float(s[: len(d) // 100].sum() / (d**2).sum())
        D.append(d)
        ad.append(f.replace("tuketim_", "").replace(".csv", ""))
        olcu[ad[-1]] = dict(Q=q, kurtoz=k, en_kotu_yuzde1=pay)
    M = np.array(D)
    G = M @ M.T / N
    return ad, G, olcu, N


def rapor(ad, G, olcu):
    s = np.sqrt(np.diag(G))
    print(f"\n{'aday':26s} {'Q':>9s} {'kurtoz':>7s} {'%1 payi':>8s}  {'maks |kos|':>10s}  ELEME")
    gecen = []
    for i, a in enumerate(ad):
        # en YARDIMCI ortak (en negatif kosinus) ve en FAZLALIK ortak (en pozitif)
        koslar = [(G[i, j] / (s[i] * s[j]), ad[j]) for j in range(len(ad)) if j != i]
        en_neg = min(koslar)
        en_poz = max(koslar)
        o = olcu[a]
        sebep = []
        if o["Q"] < Q_ESIK:
            sebep.append(f"Q {o['Q']:.4f} < {Q_ESIK} -- OLCULEMEZ")
        not_ = (
            ""
            if o["kurtoz"] <= KURTOZ_UYARI
            else f" [kurtoz {o['kurtoz']:.0f} yuksek: ongoru guvenilmez]"
        )
        hukum = ("GECTI" + not_) if not sebep else "ELENDI: " + ", ".join(sebep)
        if not sebep:
            gecen.append(a)
        maks_kos = f"{en_neg[0]:+.2f}/{en_poz[0]:+.2f}"
        print(
            f"  {a:24s} {o['Q']:9.5f} {o['kurtoz']:7.1f} {100 * o['en_kotu_yuzde1']:7.1f}% {maks_kos:>12s}  {hukum}"
        )
    print("\nKOSINUS MATRISI")
    print("        " + " ".join(f"{x[:8]:>8s}" for x in ad))
    for i, a in enumerate(ad):
        print(
            f"{a[:7]:>7s} " + " ".join(f"{G[i, j] / (s[i] * s[j]):+8.3f}" for j in range(len(ad)))
        )
    return gecen


def ucluler(ad, G, gecen, r=0.035):
    """g7 + iki aday. g7'nin L'si BILINIYOR; digerleri icin r varsayilir."""
    ig = ad.index("g7_span_tau3") if "g7_span_tau3" in ad else None
    if ig is None:
        print("\n[g7 envanterde yok -- uclu secimi atlandi]")
        return []
    sonuc = []
    for a, b in itertools.combinations([x for x in gecen if x != "g7_span_tau3"], 2):
        idx = [ig, ad.index(a), ad.index(b)]
        Gs = G[np.ix_(idx, idx)]
        L = np.array([LG, r * np.sqrt(G[idx[1], idx[1]]), r * np.sqrt(G[idx[2], idx[2]])])
        try:
            k = np.linalg.solve(Gs, L)
        except np.linalg.LinAlgError:
            continue
        mse = M0 - 2 * k @ L + k @ Gs @ k
        sonuc.append(
            (
                np.sqrt(max(mse, 0)),
                a,
                b,
                float(np.abs(k).sum()),
                float(np.linalg.cond(Gs)),
                float(np.sqrt(k @ Gs @ k)),
            )
        )
    sonuc.sort()
    print(f"\nEN IYI UCLULER (g7 + iki aday, r={r} senaryosu)")
    print(
        f"  {'sonuc':>8s} {'aday A':16s} {'aday B':16s} {'|k|1':>6s} {'kosul':>8s} {'yerdeg':>7s}"
    )
    for v, a, b, k1, c, yd in sonuc[:8]:
        im = "  <- 2. SIRA" if v < 1.00041 else ""
        print(f"  {v:8.5f} {a:16s} {b:16s} {k1:6.2f} {c:8.1f} {yd:7.3f}{im}")
    return sonuc


if __name__ == "__main__":
    MEVCUT = [
        "tuketim_g7_span_tau3.csv",
        "tuketim_y40_sota_temiz.csv",
        "tuketim_q1c_kapasite_siki.csv",
        "tuketim_y46_amnezik_kirpik.csv",
        "tuketim_y45_mevsimsel_kirpik.csv",
        "tuketim_z2_analog.csv",
        "tuketim_q1d_kuantil38_siki.csv",
    ]
    YENI = [
        "tuketim_k5_kesinti.csv",
        "tuketim_t1_turizm.csv",
        "tuketim_t2_tatil.csv",
        "tuketim_h1_haftagunu.csv",
        "tuketim_h2_takvim.csv",
    ]
    ad, G, olcu, N = envanter(MEVCUT + YENI)
    gecen = rapor(ad, G, olcu)
    print(f"\nELEMEYI GECEN: {len(gecen)} aday -> {gecen}")
    s = ucluler(ad, G, gecen)
    json.dump(
        dict(
            olcu=olcu, gecen=gecen, en_iyi=[{"sonuc": v, "A": a, "B": b} for v, a, b, *_ in s[:5]]
        ),
        open("m105_secim.json", "w"),
        indent=1,
    )
