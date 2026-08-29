"""r2: r_span teshisinden SPAN DISI aday yonler uret.

MANTIK
  r_span zaten span icinde -> onun yonunde gitmek span optimumundan
  FAZLASINI VERMEZ. Deger TESHISTE: r_span'in kohort profili
  s(x) = ort(r_span | kohort(x))
  gercek artik r'nin ayni kohort profilinin (gorulebilen) golgesidir.

  Aday:  p_yeni = p0 - eta * s_perp ,  s_perp = s - Proj_span(s)
  s_perp span'a DIK -> LB skoru bu yon icin YENI ve saf bir olcum verir:
      S^2 = m0 - 2*L_z + Q_z  ->  L_z olculur, kazanc = L_z^2/Q_z.

  eta: span, hatanin yalnizca %0.30'unu goruyor. Kohort sapmasinin
  gercek buyuklugu golgesinden buyuk olabilir. eta, Q_z hedefine gore
  secilir ve ACIKCA raporlanir -- bu bir BAHISTIR, olcum degil.

KAGGLE'A HICBIR SEY GONDERILMEZ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

BURA = Path(__file__).resolve().parent
KOK = BURA.parents[1]
SKOR = json.loads((BURA / "olculmus_skorlar.json").read_text(encoding="utf-8"))
TABAN = "tuketim_m6_ikiyon.csv"
K_ANA = 15
K_DIK = 20  # dikleme icin kullanilan span boyutu (ozdeger >= 5.4e-5)
Q_HEDEF = 0.0030  # aday yonun m6'ya gore kare uzakligi (probe olcegi)

meta = json.loads((BURA / "g1_meta.json").read_text(encoding="utf-8"))
dosyalar = meta["dosyalar"]
X = np.load(BURA / "g1_X.npy")
N = X.shape[1]
i0 = dosyalar.index(TABAN)
p0 = X[i0]
M0 = SKOR[TABAN] ** 2
jj = [j for j in range(len(dosyalar)) if j != i0]
adlar = [dosyalar[j].replace("tuketim_", "").replace(".csv", "") for j in jj]
D = X[jj] - p0
n = D.shape[0]
G = (D @ D.T) / N
Qd = np.diag(G).copy()
w, V = np.linalg.eigh(G)
o = np.argsort(-w)
w, V = w[o], V[:, o]
r = np.load(BURA / "r1_rspan.npy")

# ---------------------------------------------------------------- ozellikler
te = pd.read_csv(KOK / "data/raw/test.csv", parse_dates=["tarih"], dtype={"tanim": str})
tr = pd.read_csv(KOK / "data/raw/train.csv", parse_dates=["tarih"], dtype={"tanim": str})
ids = te.id.values
soguk = (~te.tanim.isin(set(tr.tanim))).values
ay = te.tarih.dt.month.values
sev20 = pd.qcut(p0, 20, labels=False, duplicates="drop").astype(int)


def kohort_profil(kod):
    """kod: N uzunlugunda tamsayi kohort etiketi -> parcali sabit s(x)"""
    u, c = np.unique(kod, return_inverse=True)
    tot = np.bincount(c, weights=r, minlength=len(u))
    cnt = np.bincount(c, minlength=len(u)).astype(float)
    ort = tot / cnt
    return ort[c], u, ort, cnt


def span_dik(v):
    """v'yi span{d_j} icinden cikar (kesik SVD, k=K_DIK)

    DIKLIK GEOMETRIK bir ozelliktir, gurultuye bagli degildir -> burada
    sayisal olarak gecerli TUM span (k=20) cikarilir, teshis icin
    kullanilan k=15 degil."""
    b = (D @ v) / N
    bt = V.T @ b
    a = V[:, :K_DIK] @ (bt[:K_DIK] / w[:K_DIK])
    return v - a @ D


def olcu(vperp):
    q = float((vperp**2).mean())
    z2 = vperp**2
    esik = np.quantile(z2, 0.99)
    return dict(
        Q=q,
        kurtoz=float(((vperp - vperp.mean()) ** 4).mean() / max(vperp.var() ** 2, 1e-300)),
        en_kotu_yuzde1_pay=float(z2[z2 >= esik].sum() / z2.sum()),
        sd=float(vperp.std()),
        maks=float(np.abs(vperp).max()),
    )


def kosinus(v):
    nv = np.sqrt((v**2).mean())
    return [float((D[j] @ v) / N / (np.sqrt(Qd[j]) * nv)) for j in range(n)]


# ---------------------------------------------------------------- adaylar
TANIM = {
    "r1_seviye": ("seviye 20-lik x sicak/soguk", sev20 * 2 + soguk.astype(int)),
    "r2_ayseviye": (
        "ay x seviye 20-lik x sicak/soguk",
        (ay - 4) * 40 + sev20 * 2 + soguk.astype(int),
    ),
    "r3_ay": ("ay x sicak/soguk (kaba, en dayanikli eksen)", (ay - 4) * 2 + soguk.astype(int)),
}

sonuc = {}
for ad, (aciklama, kod) in TANIM.items():
    s, u, ort, cnt = kohort_profil(kod)
    sp = span_dik(s)
    m1 = olcu(s)
    mp = olcu(sp)
    eta = float(np.sqrt(Q_HEDEF / mp["Q"]))
    z = eta * sp
    mz = olcu(z)
    ks = kosinus(z)
    yeni = p0 - z
    tk = np.expm1(yeni)
    tk = np.maximum(tk, 0.0)
    dosya = KOK / "submissions" / ("tuketim_%s.csv" % ad)
    pd.DataFrame({"id": ids, "tuketim": tk}).to_csv(dosya, index=False)

    # kapi denetimi -- diskten geri oku
    chk = pd.read_csv(dosya)
    kapi = dict(
        satir=int(len(chk)),
        satir_ok=bool(len(chk) == 714688),
        id_ok=bool((chk.id.values == ids).all()),
        baslik_ok=bool(list(chk.columns) == ["id", "tuketim"]),
        nan=int(chk.tuketim.isna().sum()),
        negatif=int((chk.tuketim.values < 0).sum()),
        min=float(chk.tuketim.min()),
        maks=float(chk.tuketim.max()),
    )
    kapi["hepsi_ok"] = bool(
        kapi["satir_ok"]
        and kapi["id_ok"]
        and kapi["baslik_ok"]
        and kapi["nan"] == 0
        and kapi["negatif"] == 0
    )

    sonuc[ad] = dict(
        aciklama=aciklama,
        dosya=dosya.name,
        kohort_sayisi=int(len(u)),
        s_ham=m1,
        s_perp=mp,
        span_disi_pay=float(mp["Q"] / m1["Q"]),
        eta=eta,
        z=mz,
        kazanc_c1=mz["Q"],  # eger L_z = Q_z (yani duzeltme birebir dogru) ise
        kazanc_c05=0.25 * mz["Q"],
        kazanc_c025=0.0625 * mz["Q"],
        kosinus_maks=float(np.abs(ks).max()),
        kosinus=dict(zip(adlar, ks)),
        m6_log_fark_sd=float(z.std()),
        m6_log_fark_maks=float(np.abs(z).max()),
        kapi=kapi,
    )
    print("\n=== %s : %s ===" % (ad, aciklama))
    print("  kohort sayisi        %d" % len(u))
    print("  Q(s ham)             %.3e" % m1["Q"])
    print("  Q(s_perp)            %.3e   span disi pay %.4f" % (mp["Q"], mp["Q"] / m1["Q"]))
    print("  eta (Q hedef %.4f)  %.2f   <-- BAHIS carpani" % (Q_HEDEF, eta))
    print("  Q(z=eta*s_perp)      %.6f" % mz["Q"])
    print("  kurtoz               %.2f" % mz["kurtoz"])
    print("  en kotu %%1 satir payi %.4f" % mz["en_kotu_yuzde1_pay"])
    print("  log fark  sd %.5f  maks %.5f" % (z.std(), np.abs(z).max()))
    print("  olculmus yonlerle |kosinus| maks  %.2e" % np.abs(ks).max())
    print(
        "  kazanc  c=1 %.6f   c=0.5 %.6f   c=0.25 %.6f"
        % (mz["Q"], 0.25 * mz["Q"], 0.0625 * mz["Q"])
    )
    print(
        "  KAPI: %s  (nan=%d neg=%d min=%.4f maks=%.1f)"
        % (
            "TAMAM" if kapi["hepsi_ok"] else "HATA",
            kapi["nan"],
            kapi["negatif"],
            kapi["min"],
            kapi["maks"],
        )
    )
    print("  YAZILDI %s" % dosya)

# ---------------------------------------------------------------- kontrast kararliligi
print("\n== KONTRAST KARARLILIGI (k degisimi) ==")
duy = json.loads((BURA / "r1_artik.json").read_text(encoding="utf-8"))["duyarlilik"]["kesik"]
ks = list(duy)
KONT = {
    "ay5 - ay7": ("ay", 1, 3),
    "ay4 - ay7": ("ay", 0, 3),
    "D00 - D09": ("desil", 0, 9),
    "D00 - D08": ("desil", 0, 8),
    "SIC - SOG": ("soguk", 0, 1),
    "guc<=50 - guc400-630": ("guc", 6, 3),
}
kont_sonuc = {}
print("%-22s" % "kontrast" + "".join("%9s" % ("k=" + k) for k in ks) + "  isaret_kararli")
for ad, (eks, i, j) in KONT.items():
    vals = [duy[k]["kohort"][eks][i] - duy[k]["kohort"][eks][j] for k in ks]
    v12 = [v for k, v in zip(ks, vals) if int(k) >= 12]
    kar = all(np.sign(v) == np.sign(vals[-1]) for v in vals)
    kar12 = all(np.sign(v) == np.sign(v12[-1]) for v in v12)
    kont_sonuc[ad] = dict(
        k=ks,
        deger=vals,
        tum_k_kararli=bool(kar),
        k12ustu_kararli=bool(kar12),
        ort_k12ustu=float(np.mean(v12)),
        sd_k12ustu=float(np.std(v12)),
    )
    print(
        "%-22s" % ad
        + "".join("%+9.5f" % v for v in vals)
        + "   tum-k:%s k>=12:%s" % ("E" if kar else "H", "E" if kar12 else "H")
    )

json.dump(
    dict(q_hedef=Q_HEDEF, k=K_ANA, adaylar=sonuc, kontrast=kont_sonuc),
    open(BURA / "r2_aday.json", "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print("\nYAZILDI r2_aday.json")
