# ruff: noqa: F821  -- kapanis degiskenleri; betikler kosup json uretti
"""n15'in KAZANANI SANSTAN AYIRT EDILEBILIYOR MU?  -- SANS DUZELTMESI

n15 iki yonu AYRI AYRI olctu ve yonler CELISTI (g_arama +0.0053 / -0.0081,
c_wyildiz +0.0021 / -0.0077). Yalniz d_pca iki yonde de POZITIF. Ama
"6 aday denedim, biri iki yonde de pozitif cikti" kendi basina delil
DEGILDIR -- rastgele bolmelerde de olur.

BU BETIK. Ayni 25 eksen uzerinde RASTGELE bolmeler uretir, HER IKI yonde
de olcer ve sorar:
  (1) rastgele bir bolme, satir-agirlikli ORTALAMA rho^2'de a_fit'i
      d_pca kadar gecme olasiligi nedir?   -> tek kuyruk p degeri
  (2) rastgele bir bolme IKI YONDE DE a_fit'i gecme olasiligi nedir?
  (3) esli onyukleme ile ORTALAMA farkin %90 GA'si (iki yonun onyukleme
      cekilisleri bagimsiz; agirlikli toplam alinir)
  (4) 6 aday denendigi icin AILE-BAZLI duzeltme: 6 rastgele bolmenin
      EN IYISININ dagilimi.
"""

import json
import os

import numpy as np

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
M29 = os.path.join(KOK, "experiments/model29")
SCR = os.path.join(
    r"C:/Users/Cem/AppData/Local/Temp/claude",
    "c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX",
    "e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad",
)
K = 25
NRAST = 20000

Z = {
    y: np.load(os.path.join(SCR, f"n15_yon_{y}.npz"), allow_pickle=True)
    for y in ("NisMay", "HazTem")
}
YY = list(Z)
NN = {y: float(Z[y]["no"]) for y in YY}
TY = sum(NN.values())
ADLAR = [str(a) for a in Z[YY[0]]["adlar"]]
print("yonler:", {y: (int(NN[y]), len(Z[y]["idx"])) for y in YY})
print("adaylar:", ADLAR)


def deger(et, kk, ri):
    s = 0.0
    for q in np.unique(et):
        m = et == q
        if not m.any():
            continue
        c = kk[m]
        nn = float((c * c).sum())
        if nn > 0:
            s += float((c @ ri[m]) ** 2) / nn
    return s


def boot_dizi(et, z):
    """esli onyukleme dizisi (NBOOT,) -- n15'teki olc_boot ile BIREBIR."""
    B1, B2, B3, kk = z["B1"], z["B2"], z["B3"], z["kk"]
    nk, nb = B1.shape
    acc = np.zeros(nb)
    for q in np.unique(et):
        m = et == q
        if not m.any():
            continue
        c = np.zeros(nk)
        c[m] = kk[m]
        n0 = np.sqrt(float((c * c).sum()))
        if n0 <= 0:
            continue
        c = c / n0
        t1 = c @ B1
        t2 = np.einsum("a,abp,b->p", c, B2, c)
        gec = (t2 > 0) & (B3 > 0)
        v = np.zeros(nb)
        v[gec] = t1[gec] / np.sqrt(t2[gec] * B3[gec])
        acc += v**2
    return acc


# --- adaylarin TAM-K (0..24) etiketleri: her yonun kendi alive kumesinde
# saklandi. Rastgele bolmeler TAM 25 eksen uzerinde uretilip her yone
# KISITLANIR (aday kurallari da oyle davranirdi).
ET = {y: {ad: np.array(Z[y]["et_ler"][i]) for i, ad in enumerate(ADLAR)} for y in YY}
IDX = {y: np.array(Z[y]["idx"]) for y in YY}

# gercek degerler
GERC = {}
for ad in ADLAR:
    GERC[ad] = {y: deger(ET[y][ad], Z[y]["kk"], Z[y]["ri"]) for y in YY}
    GERC[ad]["ort"] = sum(NN[y] * GERC[ad][y] for y in YY) / TY

TB = "a_fit"
print(f"\n{'aday':>12s} " + " ".join(f"{y:>10s}" for y in YY) + f" {'ORT':>10s} {'ORTfark':>10s}")
for ad in ADLAR:
    print(
        f"{ad:>12s} "
        + " ".join(f"{GERC[ad][y]:10.6f}" for y in YY)
        + f" {GERC[ad]['ort']:10.6f} {GERC[ad]['ort'] - GERC[TB]['ort']:+10.6f}"
    )

# --- esli onyukleme: ORTALAMA farkin GA'si (yonler bagimsiz cekilis)
BT = {ad: {y: boot_dizi(ET[y][ad], Z[y]) for y in YY} for ad in ADLAR}
BORT = {ad: sum(NN[y] * BT[ad][y] for y in YY) / TY for ad in ADLAR}
print(f"\nESLI ONYUKLEME -- ORTALAMA rho^2 farki (vs {TB}):")
for ad in ADLAR:
    if ad == TB:
        continue
    d = BORT[ad] - BORT[TB]
    d1 = BT[ad][YY[0]] - BT[TB][YY[0]]
    d2 = BT[ad][YY[1]] - BT[TB][YY[1]]
    print(
        f"  {ad:>12s}: {GERC[ad]['ort'] - GERC[TB]['ort']:+.6f} "
        f"[%90 GA {np.quantile(d, 0.05):+.6f},{np.quantile(d, 0.95):+.6f}] "
        f"P(>0)={float((d > 0).mean()):.2f}  "
        f"P(iki yonde de >0)={float(((d1 > 0) & (d2 > 0)).mean()):.2f}"
    )

# --- RASTGELE bolmeler: ayni bolme her iki yonde de olculur
rng = np.random.default_rng(2026)
RO, R1, R2 = np.zeros(NRAST), np.zeros(NRAST), np.zeros(NRAST)
for t in range(NRAST):
    while True:
        et_tam = rng.integers(0, 4, size=K)
        if len(np.unique(et_tam)) == 4:
            break
    d = {}
    for y in YY:
        et = et_tam[IDX[y]]
        d[y] = deger(et, Z[y]["kk"], Z[y]["ri"])
    R1[t], R2[t] = d[YY[0]], d[YY[1]]
    RO[t] = sum(NN[y] * d[y] for y in YY) / TY

IKI = (GERC[TB][YY[0]] < R1) & (GERC[TB][YY[1]] < R2)
print(f"\nRASTGELE {NRAST} bolme (TAM 25 eksende uretilip iki yonde de olculdu):")
print(
    f"  ORT rho^2: ort {RO.mean():.6f} medyan {np.median(RO):.6f} "
    f"%90 {np.quantile(RO, 0.9):.6f} %99 {np.quantile(RO, 0.99):.6f} maks {RO.max():.6f}"
)
print(
    f"  a_fit ORT rho^2 = {GERC[TB]['ort']:.6f} "
    f"-> rastgelenin %{100 * (GERC[TB]['ort'] > RO).mean():.1f} yuzdeligi"
)
print(f"  P(rastgele IKI YONDE de a_fit'i gecer) = {IKI.mean():.3f}")

print("\nADAY BASINA SANS DUZELTMESI (tek kuyruk p, 6 aday icin aile duzeltmeli):")
NA = len([a for a in ADLAR if a not in ("a_fit", "a_tam", "a9_aile")])
for ad in ADLAR:
    if ad in ("a_fit", "a_tam", "a9_aile"):
        continue
    p = float((GERC[ad]["ort"] <= RO).mean())
    p_aile = 1.0 - (1.0 - p) ** NA  # 6 bagimsiz denemenin en iyisi
    iki = bool(GERC[ad][YY[0]] > GERC[TB][YY[0]] and GERC[ad][YY[1]] > GERC[TB][YY[1]])
    print(
        f"  {ad:>12s}: ORT {GERC[ad]['ort']:.6f}  p_ham={p:.4f}  "
        f"p_aile({NA} aday)={p_aile:.4f}  iki_yonde_de_pozitif={iki}"
    )

# 6 rastgele adaydan EN IYISININ dagilimi (dogrudan simulasyon)
rng2 = np.random.default_rng(77)
EN = np.array([RO[rng2.choice(NRAST, NA, replace=False)].max() for _ in range(20000)])
print(
    f"\n{NA} RASTGELE adayin EN IYISI: medyan {np.median(EN):.6f} "
    f"%90 {np.quantile(EN, 0.9):.6f} %95 {np.quantile(EN, 0.95):.6f}"
)
for ad in ADLAR:
    if ad in ("a_fit", "a_tam", "a9_aile"):
        continue
    print(f"  {ad:>12s} bu dagilimin %{100 * (GERC[ad]['ort'] > EN).mean():.1f} yuzdeliginde")

J = dict(
    gercek={a: GERC[a] for a in ADLAR},
    NRAST=NRAST,
    rastgele_ort=dict(
        ort=float(RO.mean()),
        q50=float(np.median(RO)),
        q90=float(np.quantile(RO, 0.9)),
        q99=float(np.quantile(RO, 0.99)),
        maks=float(RO.max()),
    ),
    P_rastgele_iki_yonde=float(IKI.mean()),
    p={
        a: dict(p_ham=float((GERC[a]["ort"] <= RO).mean()))
        for a in ADLAR
        if a not in ("a_fit", "a_tam", "a9_aile")
    },
    en_iyi_6_rastgele=dict(
        q50=float(np.median(EN)), q90=float(np.quantile(EN, 0.9)), q95=float(np.quantile(EN, 0.95))
    ),
)
with open(os.path.join(M29, "n15b_sans.json"), "w", encoding="utf-8") as fh:
    json.dump(J, fh, indent=1, ensure_ascii=False)
print("\nyazildi: n15b_sans.json")
