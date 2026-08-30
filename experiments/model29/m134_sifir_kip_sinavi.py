"""PUBLIC/PRIVATE BOLUNMESI RASTGELE MI? -- M0'in kararliligiyla sinanir.

RISK. r_hat public %50'nin skorlarindan geliyor, sonuc private %50'de
belirlenecek. Bolunme tarihe ya da trafoya gore yapildiysa public'te olculen
yon private'a tasinmayabilir.

SINAV. Kurdugumuz cebir sunu varsayar:
    P_j^2 = M0 - 2 L_j + Q_j,   Q_j = TUM satirlarda ort(d_j^2)
ama P_j yalnizca PUBLIC alt kumede olculur. Yani gercekte
    P_j^2 = M0_S - 2 L_j + ort_S(d_j^2)
Eger public alt kume temsili degilse ort_S(d_j^2) ile ort_TUM(d_j^2) ayrisir
ve M0 capadan capaya KAYAR. Gozlenen kayma 9.1e-07.

Her aday bolunme icin oran = ort_S(d_j^2) / ort_TUM(d_j^2) hesaplanir.
Bu oranin d_j'ler arasindaki SACILIMI x Q, M0'da beklenen kaymadir.
Gozlenen 9.1e-07'yi hangi bolunme aciklar?
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
TABAN = "tuketim_m6_ikiyon.csv"
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0  # noqa: E402

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
D = []
for f in list(SK) + list(EK_MODEL):
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is not None and len(v) == N:
        D.append(v - a0)
print(f"{len(D)} fark yonu, {N} satir")

tar = pd.to_datetime(te.tarih)
trf = te.tanim.values
uq = pd.unique(trf)
rng = np.random.default_rng(3)

ADAYLAR = {}
ADAYLAR["rastgele satir"] = rng.random(N) < 0.5
srt = np.argsort(tar.values, kind="stable")
m = np.zeros(N, dtype=bool)
m[srt[: N // 2]] = True
ADAYLAR["tarihe gore (ilk yari)"] = m
ADAYLAR["tek/cift gun"] = (tar.dt.dayofyear.values % 2) == 0
sec = rng.permutation(len(uq))[: len(uq) // 2]
ADAYLAR["trafoya gore"] = pd.Series(np.isin(np.arange(len(uq)), sec), index=uq)[trf].to_numpy()

# ---------------------------------------------------------------------------
# VARSAYIMSIZ SINAV. m133'te aday bolunmeleri M0'in uc capadaki uyumuyla
# elemeye calistim; o kanit zayif cikti (uc capa bagimsiz degil). Bu betik
# HICBIR bolunme varsayimi yapmayan bir olcum kurar.
#
# FIKIR. G = V'V/N'in SIFIR ozvektorleri var (ozdegerler 1e-17, 1e-18):
# bunlar d_j'ler arasindaki TAM dogrusal bagimliliklardir, yani V u = 0.
# Boyle bir u icin gercek L de ayni bagintiya uymalidir:
#     u . L_gercek = <r, V u>/N = 0
# Ama biz L_j'yi  L_j = (M0 + Q_j^TUM - P_j^2)/2  ile kuruyoruz; P_j PUBLIC'te
# olculdugu icin hata (Q_j^TUM - Q_j^PUBLIC)/2 kadardir. Dolayisiyla
#     u . L_kurulan = (1/2) * sum_j u_j (Q_j^TUM - Q_j^PUBLIC)
# DOGRUDAN bolunme uyusmazligini olcer. Varsayim yok, cebir kapali.
#
# Sonra her aday bolunme icin ayni buyuklugu benzetip karsilastiririz.
# ---------------------------------------------------------------------------
V = np.array(D).T
G = (V.T @ V) / N
w, U = np.linalg.eigh(G)
sira = np.argsort(w)
w, U = w[sira], U[:, sira]
print(f"\nG ozdegerleri (kucukten): {', '.join(f'{x:.2e}' for x in w[:6])}")

with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK2 = json.load(fh)
DOSYA, Q_ALL, P2 = [], [], []
for f in list(SK2) + list(EK_MODEL):
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    d = v - a0
    DOSYA.append(f)
    Q_ALL.append(float((d * d).mean()))
    P2.append(SK2[f] ** 2 if f in SK2 else np.nan)
Q_ALL = np.array(Q_ALL)
P2 = np.array(P2)
L_KUR = (M0 + Q_ALL - P2) / 2.0

SIFIR = [i for i in range(len(w)) if w[i] < 1e-14]
print(f"TAM bagimli (ozdeger < 1e-14) kip sayisi: {len(SIFIR)}")

gecerli = np.isfinite(L_KUR)
print(f"skoru olculmus yon: {int(gecerli.sum())}/{len(L_KUR)}")

if SIFIR:
    print(f"\n{'kip':>4s} {'ozdeger':>10s} {'|V u| rms':>10s} {'u . L':>12s}")
    OLC = []
    for i in SIFIR:
        u = U[:, i]
        if not np.all(np.isfinite(L_KUR[np.abs(u) > 1e-8])):
            continue
        vu = float(np.sqrt(((V @ u) ** 2).mean()))
        ul = float(u @ np.nan_to_num(L_KUR))
        OLC.append(abs(ul))
        print(f"{i:4d} {w[i]:10.2e} {vu:10.2e} {ul:+12.3e}")
    if OLC:
        gozlenen = float(np.mean(OLC))
        print(f"\nGOZLENEN |u.L| ortalamasi: {gozlenen:.3e}")
        rng2 = np.random.default_rng(3)
        tar2 = pd.to_datetime(te.tarih)
        trf2 = te.tanim.values
        uq2 = pd.unique(trf2)
        srt2 = np.argsort(tar2.values, kind="stable")
        mm = np.zeros(N, dtype=bool)
        mm[srt2[: N // 2]] = True
        sec2 = rng2.permutation(len(uq2))[: len(uq2) // 2]
        AD2 = {
            "rastgele satir": rng2.random(N) < 0.5,
            "tarihe gore": mm,
            "tek/cift gun": (tar2.dt.dayofyear.values % 2) == 0,
            "trafoya gore": pd.Series(np.isin(np.arange(len(uq2)), sec2), index=uq2)[
                trf2
            ].to_numpy(),
        }
        print(f"\n{'aday':>18s} {'beklenen |u.L|':>15s} {'gozlenen/beklenen':>19s}")
        for ad, msk in AD2.items():
            QS = np.array([float((d[msk] ** 2).mean()) for d in D])
            tah = np.mean([abs(float(U[:, i] @ ((Q_ALL - QS) / 2))) for i in SIFIR])
            print(f"{ad:>18s} {tah:15.3e} {gozlenen / tah:19.2f}")
        print("\nORAN 1'e YAKIN olan aday, gercek bolunmeyle uyumludur.")
        print("Cok BUYUK oran: o bolunme gozlenenden fazla uyusmazlik uretirdi")
        print("  -> ELENIR. Cok KUCUK: o bolunme gozleneni aciklayamaz.")
else:
    print("\nTAM bagimli kip yok -- bu sinav uygulanamaz.")

# ---------------------------------------------------------------------------
# DONGUSELLIK DENETIMI. Gozlenen |u.L| = 2.87e-06 tam LB yuvarlama tabanina
# (5 ondalik -> sd 5e-6/sqrt(3) = 2.89e-06) oturuyor. Bu ya gercekten public
# alt kumenin TUM kume oldugunu soyler, ya da sifir kipler L'si TURETILMIS
# (olculmemis) yonlerden olusuyordur ve olcum kendi kendini dogruluyordur.
# g7 olayindaki hatayi tekrarlamamak icin kiplerin yuku dosya dosya yazilir.
# ---------------------------------------------------------------------------
TURETILMIS = set(EK_MODEL)
print("\n\nDONGUSELLIK DENETIMI -- sifir kiplerinde hangi dosyalar var?")
for i in SIFIR:
    u = U[:, i]
    ag = np.argsort(-np.abs(u))
    print(f"\n  kip {i} (ozdeger {w[i]:+.2e}), |u.L| = {abs(float(u @ np.nan_to_num(L_KUR))):.3e}")
    tur_yuk = 0.0
    for j in ag[:6]:
        if abs(u[j]) < 1e-6:
            break
        f = DOSYA[j]
        et = (
            "TURETILMIS"
            if f in TURETILMIS
            else ("SKOR YOK" if not np.isfinite(P2[j]) else "olculmus")
        )
        if et != "olculmus":
            tur_yuk += u[j] ** 2
        print(f"      {u[j]:+9.5f}  {f[:46]:46s} {et}")
    print(f"      -> olculmemis yuk payi: {tur_yuk:.1%}")

print("\nKARAR KURALI: bir kipte olculmemis yuk payi kayda degerse (>%1)")
print("  o kip DONGUSELDIR ve kanit sayilmaz.")

# ---------------------------------------------------------------------------
# IKINCI DONGUSELLIK: M0'IN KENDISI. u.L acilirsa
#     u.L = ( M0 * sum_j u_j  +  sum_j u_j (Q_j - P_j^2) ) / 2
# Eger sum_j u_j != 0 ise sonuc M0'a baglidir; M0 ise TAM BU dosyalardan
# (p51, m4, v102) belirlendi -> kip kucuk cikmak zorunda. Kanit degil.
#
# COZUM: sifir uzayi 3 boyutlu; "sum_j u_j = 0" tek bir dogrusal kosul.
# Geriye M0'DAN TAMAMEN BAGIMSIZ 2 boyutluk alt uzay kalir. Kanit orada.
# ---------------------------------------------------------------------------
print("\n\nM0-BAGIMSIZ ALT UZAY")
Un = U[:, SIFIR]
s_vek = Un.sum(axis=0)
print(f"  kip basina sum(u_j): {', '.join(f'{x:+.4f}' for x in s_vek)}")
# s_vek'e dik bilesim katsayilari -> M0 terimi yok olur
Qb, _ = np.linalg.qr(s_vek.reshape(-1, 1))
Pdik = np.eye(len(SIFIR)) - Qb @ Qb.T
wq, Vq = np.linalg.eigh(Pdik)
BAG = [Vq[:, i] for i in range(len(SIFIR)) if wq[i] > 0.5]
print(f"  M0-bagimsiz boyut: {len(BAG)}")

QP = np.nan_to_num(Q_ALL - P2)
gozler = []
for t, cvek in enumerate(BAG):
    u = Un @ cvek
    u = u / np.linalg.norm(u)
    su = float(u.sum())
    ul = float(u @ QP) / 2.0
    vu = float(np.sqrt(((V @ u) ** 2).mean()))
    gozler.append(abs(ul))
    print(
        f"  yon {t}: sum(u_j) = {su:+.2e} (M0 terimi yok), |V u| rms {vu:.2e}, "
        f"|u.L| = {abs(ul):.3e}"
    )

goz = float(np.mean(gozler))
YUV = 5e-6 / np.sqrt(3.0)  # 5 ondalik yuvarlamanin sd'si, dP^2 ~ 2P dP, /2
print(f"\n  GOZLENEN ortalama |u.L| = {goz:.3e}")
print(f"  LB YUVARLAMA TABANI      = {YUV:.3e}   (5 ondalik, P~1)")

rng3 = np.random.default_rng(3)
tar3 = pd.to_datetime(te.tarih)
trf3 = te.tanim.values
uq3 = pd.unique(trf3)
srt3 = np.argsort(tar3.values, kind="stable")
m3 = np.zeros(N, dtype=bool)
m3[srt3[: N // 2]] = True
sec3 = rng3.permutation(len(uq3))[: len(uq3) // 2]
AD3 = {
    "public = TUM kume": np.ones(N, dtype=bool),
    "rastgele %50": rng3.random(N) < 0.5,
    "tarihe gore %50": m3,
    "tek/cift gun": (tar3.dt.dayofyear.values % 2) == 0,
    "trafoya gore %50": pd.Series(np.isin(np.arange(len(uq3)), sec3), index=uq3)[trf3].to_numpy(),
}
print(
    f"\n{'aday bolunme':>20s} {'beklenen |u.L|':>15s} {'+yuvarlama':>12s} {'gozlenen/beklenen':>19s}"
)
for ad, msk in AD3.items():
    QS = np.array([float((d[msk] ** 2).mean()) for d in D])
    tah = np.mean([abs(float((Un @ c / np.linalg.norm(Un @ c)) @ (Q_ALL - QS)) / 2.0) for c in BAG])
    top = float(np.sqrt(tah**2 + YUV**2))
    print(f"{ad:>20s} {tah:15.3e} {top:12.3e} {goz / top:19.2f}")

print("\nORAN ~1 olan aday gercek bolunmeyle uyumludur.")

# ---------------------------------------------------------------------------
# KANITI GUCLENDIR. Yukarida yalnizca TAM sifir kipler (ozdeger < 1e-14)
# kullanildi -> M0 dusuldukten sonra 2 boyut kaldi, n=2 ince.
#
# YAKIN-SIFIR kipler de kullanilabilir: ozdeger lambda olan bir kip icin
# gercek kisit sifir degil ama SINIRLIDIR:
#     |u . L_gercek| = |<r, V u>/N| <= |r|_rms * sqrt(lambda)
# |r|_rms ~ 1.0 (M0 ~ 1.0058). Bu sinir yuvarlama tabaninin altinda kaldigi
# surece (sqrt(lambda) < 2.89e-06, yani lambda < 8.4e-12) kip kullanilabilir.
# ---------------------------------------------------------------------------
print("\n\nKANIT GUCLENDIRME -- yakin-sifir kipler de dahil")
RMS_R = float(np.sqrt(M0))
ESIK_L = (YUV / RMS_R) ** 2
KULLAN = [i for i in range(len(w)) if w[i] < ESIK_L]
print(f"  lambda esigi {ESIK_L:.2e} (sqrt = yuvarlama tabani)")
print(f"  kullanilabilir kip: {len(KULLAN)} (once {len(SIFIR)})")
for i in KULLAN:
    print(
        f"    kip {i}: lambda {w[i]:+.2e}, gercek kisit siniri {RMS_R * np.sqrt(max(w[i], 0)):.2e}"
    )

Un2 = U[:, KULLAN]
s2 = Un2.sum(axis=0)
Qb2, _ = np.linalg.qr(s2.reshape(-1, 1))
Pd2 = np.eye(len(KULLAN)) - Qb2 @ Qb2.T
wq2, Vq2 = np.linalg.eigh(Pd2)
BAG2 = [Vq2[:, i] for i in range(len(KULLAN)) if wq2[i] > 0.5]
print(f"  M0-bagimsiz boyut: {len(BAG2)} (once {len(BAG)})")

goz2 = []
SINIR = []
for t, c in enumerate(BAG2):
    u = Un2 @ c
    u = u / np.linalg.norm(u)
    ul = float(u @ QP) / 2.0
    vu = float(np.sqrt(((V @ u) ** 2).mean()))
    goz2.append(abs(ul))
    SINIR.append(RMS_R * vu)
    print(f"    yon {t}: |u.L| = {abs(ul):.3e}, gercek-kisit siniri {RMS_R * vu:.2e}")

print(f"\n  n = {len(goz2)} bagimsiz M0-siz olcum")
print(f"{'aday bolunme':>20s} {'sd (toplam)':>12s} {'-2 log L':>10s} {'goreli olasilik':>16s}")
SKOR = {}
for ad, msk in AD3.items():
    QS = np.array(
        [
            float(
                (Un2 @ c / np.linalg.norm(Un2 @ c))
                @ np.array([float((d[msk] ** 2).mean()) for d in D])
                / 2.0
            )
            for c in BAG2
        ]
    )
    tahler = []
    for t, c in enumerate(BAG2):
        u = Un2 @ c
        u = u / np.linalg.norm(u)
        QSv = np.array([float((d[msk] ** 2).mean()) for d in D])
        tahler.append(abs(float(u @ (Q_ALL - QSv)) / 2.0))
    sd = np.sqrt(np.mean(np.square(tahler)) + YUV**2 + np.mean(np.square(SINIR)) / 3)
    m2ll = float(np.sum((np.array(goz2) / sd) ** 2) + 2 * len(goz2) * np.log(sd))
    SKOR[ad] = m2ll
    print(f"{ad:>20s} {sd:12.3e} {m2ll:10.2f}")
en = min(SKOR, key=SKOR.get)
print(f"\n  EN OLASI: {en}")
for ad in SKOR:
    print(f"    {ad:>20s}  goreli olasilik {np.exp(-(SKOR[ad] - SKOR[en]) / 2):.4f}")
print("\n  Goreli olasilik ~1 olanlar ayirt edilemez; ~0 olanlar ELENIR.")
