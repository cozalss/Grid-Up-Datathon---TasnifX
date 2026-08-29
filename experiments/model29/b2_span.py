"""b2: OLCULMUS BUTUN skorlarin span'i. Her bilinen LB skoru BEDAVA bir denklemdir.

taban a = log1p(v102),  m0 = 1.00553^2 = 1.011091
dosya F icin  u = log1p(F) - a
S_F^2 = m0 - 2*L_u + Q_u    =>   L_u = (m0 + Q_u - S_F^2)/2      (OLCULDU)

n dosya -> n olculmus dogrusal fonksiyonel. V = span{u_i} icindeki EN IYI
tahmin:  p = a + sum c_i u_i,  c = G^-1 L,  MSE* = m0 - L' G^-1 L
G_ij = mean(u_i u_j).

Skorlar 5 haneye yuvarlik -> dMSE ~ 1e-5, dL ~ 5e-6. G kotu kosullu ise
cozum patlar; bu yuzden KESIK SVD ile kademeli cozulur ve her kademede
yuvarlama gurultusunun buyutulmesi raporlanir.
"""

import json
import os

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
KOK = os.path.dirname(os.path.dirname(BURA))

SKOR = json.load(open(os.path.join(BURA, "olculmus_skorlar.json"), encoding="utf-8"))
TABAN = "tuketim_v102_kappa_optimum.csv"
M0 = SKOR[TABAN] ** 2

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"), dtype={"tanim": str})
tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), dtype={"tanim": str})
soguk = (~te.tanim.isin(set(tr.tanim))).values

a = np.log1p(pd.read_csv(os.path.join(KOK, "submissions", TABAN)).tuketim.values)
N = len(a)

adlar, U = [], []
for f, s in SKOR.items():
    if f == TABAN:
        continue
    yol = os.path.join(KOK, "submissions", f)
    if not os.path.exists(yol):
        print("YOK  %s" % f)
        continue
    df = pd.read_csv(yol)
    if len(df) != N or not (df.id.values == te.id.values).all():
        print("HIZALANMIYOR  %s" % f)
        continue
    adlar.append(f)
    U.append(np.log1p(df.tuketim.values) - a)
U = np.array(U)
n = len(adlar)
print("yon sayisi n = %d" % n)

G = (U @ U.T) / N
Qd = np.diag(G)
S = np.array([SKOR[f] for f in adlar])
L = (M0 + Qd - S**2) / 2.0
kap = L / Qd

print("\n%-38s %8s %9s %9s %8s %10s" % ("dosya", "skor", "Q", "L", "kappa", "tek-yon*"))
for i in np.argsort(-(L**2 / Qd)):
    print(
        "%-38s %8.5f %9.6f %+9.6f %+8.3f %10.5f"
        % (adlar[i], S[i], Qd[i], L[i], kap[i], np.sqrt(M0 - L[i] ** 2 / Qd[i]))
    )

# --- kesik SVD ile span optimumu, gurultu buyutmesi ile birlikte ---
w, V = np.linalg.eigh(G)
o = np.argsort(-w)
w, V = w[o], V[:, o]
Lt = V.T @ L
sigL = 5e-6 * np.sqrt(2.0)  # skor yuvarlamasindan L belirsizligi (kaba)
print("\n== KESIK SVD: k yon kullanildiginda ==")
print("%3s %11s %11s %11s %11s %9s" % ("k", "ozdeger", "kum.kazanc", "MSE*", "S*", "gurultu+-"))
kayit = []
for k in range(1, n + 1):
    if w[k - 1] <= 1e-14:
        break
    kaz = float((Lt[:k] ** 2 / w[:k]).sum())
    var = float((sigL**2 / w[:k]).sum())  # kazanc tahmininin gurultu payi (kaba ust sinir)
    mse = M0 - kaz
    kayit.append(
        dict(
            k=k,
            ozdeger=float(w[k - 1]),
            kazanc=kaz,
            mse=mse,
            S=float(np.sqrt(max(mse, 0))),
            gurultu=float(np.sqrt(var) * 2 * abs(Lt[:k] / w[:k]).max() * 0 + np.sqrt(var)),
        )
    )
    print(
        "%3d %11.3e %11.6f %11.6f %11.5f %9.6f"
        % (k, w[k - 1], kaz, mse, np.sqrt(max(mse, 0)), np.sqrt(var))
    )


# tam cozum katsayilari (kesik)
def coz(k):
    return V[:, :k] @ (Lt[:k] / w[:k])


print("\n== OZ-YONLER (z = sum v_i u_i) ==")
print(
    "%3s %11s %11s %10s %11s %9s %9s %9s"
    % ("i", "Q_z=w", "L_z", "kappa_z", "katki", "CS orani", "sigma kat", "|c|max")
)
for i in range(n):
    if w[i] <= 1e-12:
        print("%3d %11.3e  (sifir ozdeger -- dogrusal bagimli)" % (i + 1, w[i]))
        continue
    cs = abs(Lt[i]) / np.sqrt(M0 * w[i])
    c_i = V[:, i] * (Lt[i] / w[i])
    print(
        "%3d %11.3e %+11.6f %10.3f %11.6f %9.4f %9.0f %9.2f"
        % (
            i + 1,
            w[i],
            Lt[i],
            Lt[i] / w[i],
            Lt[i] ** 2 / w[i],
            cs,
            abs(Lt[i]) / 5e-6,
            np.abs(c_i).max(),
        )
    )

print("\n(CS orani = |L_z| / sqrt(m0*Q_z) <= 1 olmak ZORUNDA; 1'e yaklasmasi")
print(" 'bu yon hatanin neredeyse tamamini aciklyor' demektir -- supheli)")
print("(sigma kat = yuvarlama gurultusune (5e-6) gore kac katlik sinyal)")

# --- DEGENERELIK DENETIMI: u_j digerlerinin span'inda mi? ---
print("\n== DOGRUSAL BAGIMLILIK DENETIMI (blof kirici) ==")
print("u_j digerlerinden kestirilir. Artik norm r_j kucukse Cauchy-Schwarz")
print("|L_j - L_kestirim| <= sqrt(m0)*r_j SINIRINI cignemek IMKANSIZ.")
print(
    "%-38s %10s %11s %11s %11s %8s"
    % ("dosya", "r_j", "L_olculen", "L_kestirim", "fark", "fark/sinir")
)
ihlal = []
for j in range(n):
    idx = [i for i in range(n) if i != j]
    Gm = G[np.ix_(idx, idx)]
    gj = G[np.ix_(idx, [j])].ravel()
    alpha = np.linalg.pinv(Gm, rcond=1e-10) @ gj
    r2 = Qd[j] - gj @ alpha
    r = float(np.sqrt(max(r2, 0.0)))
    Lp = float(alpha @ L[idx])
    fark = L[j] - Lp
    sinir = np.sqrt(M0) * r
    oran = abs(fark) / sinir if sinir > 0 else np.inf
    ihlal.append((adlar[j], r, L[j], Lp, fark, oran))
for ad, r, lo, lp, f, o in sorted(ihlal, key=lambda t: -t[5])[:12]:
    print("%-38s %10.2e %+11.6f %+11.6f %+11.6f %8.3f" % (ad, r, lo, lp, f, o))
kotu = [t for t in ihlal if t[5] > 1.0]
print("\nIHLAL SAYISI (fark/sinir > 1): %d / %d" % (len(kotu), n))
if kotu:
    print("  -> olculen L'ler KENDI ICINDE TUTARSIZ: skor yuvarlamasi ve/veya")
    print("     public/private ayrimi yuzunden ince yapi GUVENILMEZ.")
else:
    print("  -> tutarli; ince yapinin gercek olma ihtimali var.")

print("\n== HEDEF ==")
print(
    "2. sira 1.00041 -> MSE 1.000820 ; taban v102 m0 = %.6f -> gereken kazanc %.6f"
    % (M0, M0 - 1.00041**2)
)
print("m6 (mevcut en iyi) kazanc = %.6f" % (M0 - 1.00284**2))

# --- tutarlilik denetimi: m6 span icinde mi? ---
m6 = (
    np.log1p(pd.read_csv(os.path.join(KOK, "submissions/tuketim_m6_ikiyon.csv")).tuketim.values) - a
)
i6 = adlar.index("tuketim_m6_ikiyon.csv")
print(
    "\nDENETIM m6 tek-yon optimumu %.5f (gercek skoru %.5f, L=%.6f Q=%.6f)"
    % (np.sqrt(M0 - L[i6] ** 2 / Qd[i6]), S[i6], L[i6], Qd[i6])
)

json.dump(
    dict(
        m0=M0,
        adlar=adlar,
        skor=S.tolist(),
        Q=Qd.tolist(),
        L=L.tolist(),
        kappa=kap.tolist(),
        ozdegerler=w.tolist(),
        kesik=kayit,
    ),
    open(os.path.join(BURA, "b2_span.json"), "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
np.save(os.path.join(BURA, "b2_U.npy"), U)
np.save(os.path.join(BURA, "b2_G.npy"), G)
np.save(os.path.join(BURA, "b2_L.npy"), L)
with open(os.path.join(BURA, "b2_adlar.json"), "w", encoding="utf-8") as fh:
    json.dump(adlar, fh, ensure_ascii=False, indent=1)
print("\nYAZILDI b2_span.json")
