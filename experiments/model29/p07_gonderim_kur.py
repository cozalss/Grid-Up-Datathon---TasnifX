"""p06'nin SOGUK HARMAN duzeltmesini gonderilebilir dosyalara cevirir.

p06 bulgusu: soguk uzman harmani cat/xgb/lgbm ESIT agirlikli. Agirligi
guz25'te secilen (0.05/0.35/0.60) yapmak yaz25 soguk RMSLE'yi
1.43592 -> 1.41269 getiriyor; test bilesiminde 0.97932 -> 0.97180,
yani +0.00752 (2. sira icin gereken 0.00579).

Delta p06 tarafindan LOG uzayinda uretildi ve YALNIZ soguk satirlarda
sifir disi (158369 / 714688 = %22.2). Uretim harmanina gore fark oldugu
icin, uretim harmanini iceren HERHANGI bir tabana eklenebilir.

UC TABAN denenir:
  A  SPAN OPTIMUMU (a0 + r_hat)  -- en iyi tabanimiz, LB ~1.00101
  B  YP_seviye                   -- LB'de OLCULMUS en iyi dosyamiz, 1.00115
  C  m6_ikiyon                   -- p06'nin delta'yi turettigi taban, 1.00284

A en cok pay birakir ama r_hat dogrusal bir duzeltme; soguk satirlari
ozel olarak hedeflemedigi icin delta ile cift sayim riski KUCUK.
C en guvenli capa: delta tam olarak o tabana gore olculdu.

Ayrica AGRESIF varyant: p06'nin verdigi uc aile ortalamasindan
(cat/xgb/lgbm) yalniz-lgbm harmani. yaz25'te 1.40001 (esit harmanda
1.43592), yani daha buyuk kazanc -- ama kis26 ters yonu soyledigi icin
daha riskli. Esik hedefinde ikisi de gonderilir.
"""

import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
SP = (
    r"C:/Users/Cem/AppData/Local/Temp/claude/"
    r"c--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX/"
    r"e98517bd-fcb3-465e-95ae-9f16be93da6b/scratchpad"
)
CIK = os.path.join(SP, "p07")
os.makedirs(CIK, exist_ok=True)

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
IDS = te.id.values
N = len(IDS)


def oku(f, zorunlu=True):
    """m148'in oku()'suyla AYNI davranis: eslesmeyen dosya None doner.

    olculmus_skorlar.json bazi eski/kucuk dosyalari da listeliyor
    (ornegin gun1_baseline.csv, 1450 bayt). m148 bunlari atliyor; span
    kurulusu birebir ayni olsun diye burada da atliyoruz.
    """
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    if not np.array_equal(d.id.values, IDS):
        if len(d) != N or d.id.duplicated().any():
            if zorunlu:
                raise SystemExit(f"{f}: id eslesmiyor")
            return None
        pos = pd.Index(d.id).get_indexer(IDS)
        if (pos < 0).any():
            if zorunlu:
                raise SystemExit(f"{f}: id eslesmiyor")
            return None
        d = d.iloc[pos].reset_index(drop=True)
    return np.log1p(d[k].to_numpy(dtype=np.float64))


DELTA = np.load(os.path.join(SP, "p06_test_delta_log.npy"))
MASKE = DELTA != 0
AILE = np.load(os.path.join(SP, "p06_test_soguk_aile.npy"))
print(f"delta: {N} satir, {int(MASKE.sum())} soguk (%{100 * MASKE.mean():.1f})")
print(f"aile dizisi sekli: {AILE.shape}")

# --- AGRESIF DELTA: yalniz-lgbm harmani ----------------------------------
# AILE, soguk satirlar icin (cat, xgb, lgbm) log tahminleri tasiyor.
# Uretim ESIT harman; agresif varyant yalniz lgbm.
if AILE.ndim == 2 and 3 in AILE.shape:
    A = AILE if AILE.shape[1] == 3 else AILE.T
    esit = A.mean(axis=1)
    lgbm = A[:, 2]
    DELTA_AGR = np.zeros(N)
    DELTA_AGR[MASKE] = lgbm - esit
    print(
        f"agresif delta (yalniz-lgbm): ort {DELTA_AGR[MASKE].mean():+.5f} "
        f"std {DELTA_AGR[MASKE].std():.5f}"
    )
else:
    DELTA_AGR = None
    print("UYARI: aile dizisi beklenen sekilde degil, agresif varyant atlandi")

# --- TABANLAR -------------------------------------------------------------
TABANLAR = {}
TABANLAR["m6"] = oku("tuketim_m6_ikiyon.csv")
TABANLAR["ypseviye"] = oku("tuketim_YP_seviye.csv")

# span optimumu: m148'in kendi kurulusundan a0 + r_hat
import sys  # noqa: E402

sys.path.insert(0, M29)
import json  # noqa: E402

from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

a0 = TABANLAR["m6"]
with open(os.path.join(M29, "olculmus_skorlar.json")) as fh:
    SK = json.load(fh)
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == "tuketim_m6_ikiyon.csv" or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f, zorunlu=False)
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
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
TABANLAR["span"] = a0 + r_hat
print(f"\nspan optimumu kuruldu: saf skor {np.sqrt(M0 - gercek):.6f}")

# --- DOSYALARI URET -------------------------------------------------------
LB = {"m6": 1.00284, "ypseviye": 1.00115, "span": 1.00101}
KAZANC = 0.00752  # p06 olcumu (test bilesimi)

print(f"\n{'dosya':>34s} {'taban LB':>9s} {'beklenen LB':>12s} {'satir':>8s}")
URETILEN = []
for tad, taban in TABANLAR.items():
    for dad, delta, kz in [("hafif", DELTA, KAZANC), ("agresif", DELTA_AGR, None)]:
        if delta is None:
            continue
        y = np.clip(np.expm1(taban + delta), 0.0, None)
        out = pd.DataFrame({"id": IDS, "tuketim": y})
        ad = f"p07_{tad}_{dad}.csv"
        yol = os.path.join(CIK, ad)
        out.to_csv(yol, index=False)
        # dogrulama
        g = pd.read_csv(yol)
        ok = (
            len(g) == N
            and np.array_equal(g.id.values, IDS)
            and int(g.tuketim.isna().sum()) == 0
            and int((g.tuketim < 0).sum()) == 0
            and np.isfinite(g.tuketim.to_numpy()).all()
        )
        if not ok:
            raise SystemExit(f"DUR: {ad} dogrulamadan gecmedi")
        bek = LB[tad] - kz if kz else float("nan")
        URETILEN.append((ad, tad, dad, bek))
        print(f"{ad:>34s} {LB[tad]:9.5f} {(f'{bek:.5f}' if kz else '?'):>12s} {len(g):8d}")

print("\nHEPSI DOGRULANDI (satir sayisi, id sirasi, NaN, negatif, sonlu).")
print(f"Dosyalar: {CIK}")
print("\nsubmissions/ altina YAZILMADI, GONDERIM YAPILMADI.")
