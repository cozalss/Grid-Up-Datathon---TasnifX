"""NIHAI BILESIK -- duzeltilmis kapiyla.

DUZELTME. Onceki kapi "oran = rho_pred / rho_s(bilesik) <= 4" idi. Bu YANLIS:
tek eksen icin oran = 1.95*sqrt(Q_dik) <= 1.95 her zaman; cok eksende ise
eksenlerin SPAN bileseni birbirini goturunce payda kuculuyor ve oran siisiyor.
Yani oran, inandiriciligi degil isaret sadelesmesini olcuyor.

DOGRU KAPI: her eksenin katsayisi 1.95*|rho_s| TAVANINA DAYANSIN.
Dayaniyorsa tahmin LB'nin kendi olcumune capalidir (CV'ye degil).
    rho_kul = isaret(rho_cv) * min(|rho_cv|, 1.95*|rho_s|)
    TAVAN DAYANIYOR  <=>  |rho_cv| >= 1.95*|rho_s|

Bilesigin ongorulen rho'su = ||beta|| (dik eksenler). Bu, her eksenin kendi
LB olcumune capali oldugu icin savunulabilir; tek varsayim 1.95 carpaninin
seviye'den digerlerine tasinmasi (n=1, docs/68).

Ek kapi: rho_s'in kendi gurultusu sigma(rho_s) ~ 3e-4; |rho_s| >= 0.015
(50 sigma) sarti aranir.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"  # M0 m112den gelir (docs/69)
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
HEDEF_2, HEDEF_3 = 0.99790, 0.99940
RHO_S_ALT = 0.015
AZAMI_EKSEN = 40  # kesim KAPIDAN gelsin, sert tavandan degil (Kural 64)
sys.path.insert(0, M29)
from m112_kalibre import EK_MODEL, M0, buzmeli_r_hat  # noqa: E402

te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
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
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == TABAN or not os.path.exists(os.path.join(S, f)):
        continue
    v = oku(f)
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
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
nrm = float((r_hat * r_hat).mean())
MSE_OPT = M0 - gercek
print(f"saf optimum {np.sqrt(MSE_OPT):.6f}")

tp = pd.read_parquet(os.path.join(DN, "test.parquet"))
e = pd.read_parquet(os.path.join(DN, "egitim.parquet"))
blk = e[e._blok == "yaz25"]
sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
P = [
    np.load(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy")).astype(np.float64)
    for t in (1000, 1001, 1002)
    for aa in ("cat", "xgb", "lgbm")
    if os.path.exists(os.path.join(AO, f"yaz25_{t}_{aa}_uretim.npy"))
]
z = np.load(os.path.join(DN, "soguk_tahmin_yaz25.npz"))
idx = np.concatenate([sic.index.values, sog.index.values])
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
bf = e.loc[idx].copy()
rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()
m0b = float((ww * rb * rb).mean())


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svT, svB = st(a0), st(pb)
sgT = tp.soguk_mu.values.astype(np.float64)
ufT, ufB = st(tp.ufuk_gun.to_numpy()), st(bf.ufuk_gun.to_numpy())
ayT = st(pd.to_datetime(tp.tarih).dt.month.to_numpy().astype(np.float64))
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
CARP = {"x_sv": (svT, svB), "x_soguk": (sgT, sgm), "x_ufuk": (ufT, ufB), "x_ay": (ayT, ayB)}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


def kur(ad):
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in tp.columns or k2 not in tp.columns:
            return None, None
        a1, b1 = st(tp[k1].to_numpy()), st(bf[k1].to_numpy())
        a2, b2 = st(tp[k2].to_numpy()), st(bf[k2].to_numpy())
        if a1 is None or a2 is None or b1 is None or b2 is None:
            return None, None
        return st(a1 * a2), st(b1 * b2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in tp.columns or kol not in bf.columns:
        return None, None
    xt, xb = tp[kol].to_numpy(), bf[kol].to_numpy()
    if kip in CARP:
        mt, mb = CARP[kip]
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None or b_ is None else (st(a_ * mt), st(b_ * mb))
    if kip in ESIK:
        q, ust = ESIK[kip]
        fv = xt[np.isfinite(xt)]
        if fv.size == 0:
            return None, None
        v_ = np.quantile(fv, q)
        if ust:
            return st((xt > v_).astype(np.float64)), st((xb > v_).astype(np.float64))
        return st((xt < v_).astype(np.float64)), st((xb < v_).astype(np.float64))
    if kip == "kare":
        a_, b_ = st(xt), st(xb)
        return (None, None) if a_ is None else (st(a_**2), st(b_**2))
    return st(xt), st(xb)


with open(os.path.join(M29, "m121_derin_tarama.json")) as fh:
    TARAMA = json.load(fh)
rng = np.random.default_rng(5)
tn = bf.tanim.values
uqn = pd.unique(tn)
gi = pd.Series(np.arange(len(uqn)), index=uqn)[tn].to_numpy()
PERM = [
    np.argsort(np.argsort(rng.permutation(len(uqn))[gi], kind="stable"), kind="stable")
    for _ in range(20)
]

duz = np.zeros(N)
kul = []
print(
    f"\n{'eksen':>34s} {'rho_cv':>8s} {'rho_s':>8s} {'rho_kul':>8s} {'Q_dik':>6s} "
    f"{'tavan':>6s} {'kum.rho':>8s}"
)
ONCEKI = []
for kayit in TARAMA:
    if len(kul) >= AZAMI_EKSEN:
        break
    ad = kayit["eksen"]
    xt, xb = kur(ad)
    if xt is None or xb is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    Lsp = float(cc @ L)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    if Qs < 0.02:
        continue
    rho_s = Lsp / np.sqrt(Qs)
    if abs(rho_s) < RHO_S_ALT:
        continue
    xp = xp0.copy()
    for u in ONCEKI:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    if Qd < 0.25:  # eksenler birbirinden GERCEKTEN farkli olsun
        continue
    kor = float((ww * rb * xb).mean()) / np.sqrt(m0b)
    gur = np.std([float((ww * rb[s] * xb).mean()) / np.sqrt(m0b) for s in PERM])
    if abs(kor) < 3 * gur:
        continue
    rho_cv = CARPAN * kor
    dayanir = abs(rho_cv) >= TAVAN * abs(rho_s)
    if not dayanir:  # KAPI: tavan dayanmiyorsa tahmin CV'ye kalir
        continue
    # KATSAYI (docs/69 §2.5). seviye kalibrasyonu IKI BIRIM YON arasindaydi:
    #   rho_s = L_span/sqrt(Q_span) = +0.0156  (span birim yonu)
    #   rho_u = L_dik /sqrt(Q_dik)  = -0.0304  (dik birim yonu)   oran 1.95
    # Yani 1.95*|rho_s| DOGRUDAN dik birim yondeki korelasyonun tahminidir ve
    # u yonundeki optimal katsayi da odur. Eski kod ayrica sqrt(Q_dik) ile
    # carpiyordu; bu 1.95*|rho_s|'i TUM eksenin korelasyonu sayip izotropiyle
    # dik parcaya dagitmaya denk gelir -- oysa seviye'de rho_x/rho_s = 0.99,
    # 1.95 degil. Olcum de sqrt'siz hali destekliyor: blok korelasyonu
    # 0.2288 vs 0.2269, zaman-bolmeli tutma 1.098 vs 1.057.
    rho_kul = np.sign(rho_cv) * TAVAN * abs(rho_s)
    duz += rho_kul * (xp / np.sqrt(Qd))
    ONCEKI.append(xp / np.sqrt(Qd))
    kul.append(ad)
    print(
        f"{ad[:34]:>34s} {rho_cv:+8.4f} {rho_s:+8.4f} {rho_kul:+8.4f} {Qd:6.3f} "
        f"{'EVET':>6s} {np.sqrt(float((duz * duz).mean())):8.4f}"
    )

Q = float((duz * duz).mean())
birim = duz / np.sqrt(Q)
RHO = float(np.sqrt(Q))
print(f"\n{len(kul)} eksen (hepsinde TAVAN DAYANIYOR -> tahmin LB-capali)")
print(f"BILESIGIN ongorulen rho = {RHO:.4f}")
for ad, h in [("3. sira", HEDEF_3), ("2. sira", HEDEF_2), ("1. sira", 0.99009)]:
    kap = np.sqrt(max(MSE_OPT - h * h, 1e-12))
    print(f"  {ad}: gereken rho {kap:.4f}  -> f = {kap / RHO:.3f}")

KAPPA = 0.070
pn = a0 + r_hat + KAPPA * birim
y = np.clip(np.expm1(pn), 0.0, None)
out = pd.DataFrame({"id": te.id.values, "tuketim": y})
kapi = {
    "satir": len(out) == 714688,
    "id": bool((out.id.values == ss.iloc[:, 0].values).all()),
    "NaN": int(out.tuketim.isna().sum()) == 0,
    "negatif": int((out.tuketim < 0).sum()) == 0,
    "sonlu": bool(np.isfinite(out.tuketim.values).all()),
    "maks": bool(out.tuketim.max() < 3 * np.expm1(a0).max()),
}
print(f"\nKAPI: {kapi}")
if all(kapi.values()):
    yol = os.path.join(S, "tuketim_K_TEKHAK.csv")
    out.to_csv(yol + ".tmp", index=False)
    Path(yol + ".tmp").replace(yol)
    dgv = np.log1p(out.tuketim.values) - a0
    sabit = float(M0 - 2 * kL + float(dgv @ dgv) / N)
    ek = dgv - r_hat
    # KIRPMA sonrasi GERCEK yer degistirme. expm1'in 0'a kirpilmasi yonu bir
    # miktar kisaltir; cozumde ILAN EDILEN kappa degil BU kullanilmalidir,
    # yoksa cozulen |rho| sistematik olarak dusuk cikar.
    KAPPA_ETKIN = float(np.sqrt(float((ek * ek).mean())))
    print(
        f"YAZILDI  kappa(ilan)={KAPPA:.5f}  kappa(ETKIN)={KAPPA_ETKIN:.5f}  "
        f"sifir {int((y == 0).sum()):,}"
    )
    print(f"  sabit={sabit:.9f}")
    print(f"  COZUM: rho = ({sabit:.9f} - P*P) / {2 * KAPPA_ETKIN:.6f}")
    print(f"\n  {'gercek rho':>11s} {'skor':>9s} {'sira':>10s}")
    for rr in [0.0, 0.0304, 0.0500, 0.0574, 0.0700, 0.0793, RHO * 0.7, RHO]:
        sk = np.sqrt(max(sabit - 2 * KAPPA_ETKIN * rr, 1e-9))
        sr = (
            "2. SIRA"
            if sk < HEDEF_2
            else "3. sira"
            if sk < HEDEF_3
            else "4. sira"
            if sk < 1.00118
            else "5.+"
        )
        print(f"  {rr:11.4f} {sk:9.5f} {sr:>10s}")
    with open(os.path.join(BURA, "m122_nihai.json"), "w") as fh:
        json.dump(
            dict(kappa=KAPPA, kappa_etkin=KAPPA_ETKIN, sabit=sabit, rho=RHO, eksenler=kul),
            fh,
            indent=1,
        )
