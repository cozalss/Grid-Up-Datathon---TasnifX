"""KESIM NOKTASINI OLCEREK BUL -- Kural 64'u gercekten uygula.

40 eksen rho'yu 0.1650 -> 0.2140 cikariyor AMA bilesigin zaman-bolmeli tutmasi
1.283 -> 0.620'ye dusuyor. Yani ekstra eksenler yaz25'in ilk yarisina uyuyor,
ikinci yarisina TASINMIYOR -- ve test tam olarak o ikinci-yari durumudur.

Bu betik her on-ek uzunlugu icin:
  rho_pred     : bilesigin ongorulen rho'su (ustunde durdugumuz tahmin)
  tutma_zaman  : gun1-61'de kurulup gun62-122'de olculen korelasyon orani
  tutma_kesit  : trafo-bolmeli capraz dogrulama orani
hesaplar. SECIM OLCUTU: rho_pred * tutma_zaman -- yani tahminin taşınan kismi.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95
sys.path.insert(0, M29)
from m112_kalibre import M0, buzmeli_r_hat  # noqa: E402

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
with open(os.path.join(M29, "m112_durum.json")) as fh:
    DUR = json.load(fh)
V, L = [], []
for f, Pj in SK.items():
    if f == TABAN:
        continue
    v = oku(f)
    if v is None or len(v) != N:
        continue
    dd = v - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - Pj * Pj) / 2)
for o in DUR.get("olcumler", []):
    dd = oku(o["dosya"]) - a0
    V.append(dd)
    L.append((M0 + float((dd * dd).mean()) - o["skor"] ** 2) / 2)
V, L = np.array(V).T, np.array(L)
G = (V.T @ V) / N
Gi = np.linalg.pinv(G, rcond=1e-6)
r_hat, gercek, kL = buzmeli_r_hat(V, L, G, N)
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
pb = np.concatenate([np.mean(P, axis=0), np.mean([z[k] for k in z.files], axis=0)])
bf = e.loc[idx].copy()
rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
sgm = bf.soguk_mu.values.astype(np.float64)
ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
ww = ww / ww.mean()


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


with open(os.path.join(M29, "m122_nihai.json")) as fh:
    EKSENLER = json.load(fh)["eksenler"]
print(f"{len(EKSENLER)} eksenli tam liste taraniyor")

XB, XT, BETA = [], [], []
ONC = []
for ad in EKSENLER:
    xt, xb = kur(ad)
    if xt is None:
        continue
    cc = Gi @ ((V.T @ xt) / N)
    xp0 = xt - V @ cc
    Qs = 1.0 - float((xp0 * xp0).mean())
    # m122 ile AYNI: L_span tahmini buzmeli r_hat'ten (docs/69 §2.6)
    rho_s = float((r_hat * xt).mean()) / np.sqrt(Qs)
    xp = xp0.copy()
    for u in ONC:
        xp -= float((xp * u).mean()) * u
    Qd = float((xp * xp).mean())
    kor = float((ww * rb * xb).mean()) / np.sqrt(float((ww * rb * rb).mean()))
    # m122 ile AYNI: sqrt(Q_dik) CARPANI YOK (docs/69 §2.5)
    beta = np.sign(CARPAN * kor) * TAVAN * abs(rho_s)
    XT.append(xp / np.sqrt(Qd))
    XB.append(xb)
    BETA.append(beta)
    ONC.append(xp / np.sqrt(Qd))

uf = bf.ufuk_gun.to_numpy()
# TEK bir zaman kesimi gurultulu; bes ayri kesimde ortalama alinir.
ZAMAN_KESIM = [40, 50, 61, 72, 85]
rng = np.random.default_rng(17)
tn = bf.tanim.values
uq = pd.unique(tn)


def kor_m(mask, x):
    w, r = ww[mask], rb[mask]
    return float((w * r * x[mask]).mean()) / np.sqrt(float((w * r * r).mean()))


def tutma(n, fit, olc):
    """ilk n ekseni fit maskesinde agirliklandirip olc maskesinde sina."""
    duz_f = np.zeros(int(fit.sum()))
    duz_o = np.zeros(int(olc.sum()))
    ONC_f, ONC_o = [], []
    wf, rf = ww[fit], rb[fit]
    m0f = float((wf * rf * rf).mean())
    for i in range(n):
        xf, xo = XB[i][fit].copy(), XB[i][olc].copy()
        for uf_, uo_ in zip(ONC_f, ONC_o):
            k = float((wf * xf * uf_).mean()) / float((wf * uf_ * uf_).mean())
            xf -= k * uf_
            xo -= k * uo_
        nf = np.sqrt(float((wf * xf * xf).mean()))
        if nf < 0.15:
            continue
        xf, xo = xf / nf, xo / nf
        b = CARPAN * float((wf * rf * xf).mean()) / np.sqrt(m0f)
        duz_f += b * xf
        duz_o += b * xo
        ONC_f.append(xf)
        ONC_o.append(xo)
    nf = np.sqrt(float((ww[fit] * duz_f * duz_f).mean()))
    no = np.sqrt(float((ww[olc] * duz_o * duz_o).mean()))
    if nf < 1e-12 or no < 1e-12:
        return 0.0, 0.0
    return kor_m(fit, np.zeros(len(rb))), 0.0  # yer tutucu


def tutma2(n, fit, olc):
    wf, rf = ww[fit], rb[fit]
    m0f = float((wf * rf * rf).mean())
    wo, ro = ww[olc], rb[olc]
    m0o = float((wo * ro * ro).mean())
    duz_f = np.zeros(int(fit.sum()))
    duz_o = np.zeros(int(olc.sum()))
    ONC_f, ONC_o = [], []
    for i in range(n):
        xf, xo = XB[i][fit].copy(), XB[i][olc].copy()
        for uf_, uo_ in zip(ONC_f, ONC_o):
            k = float((wf * xf * uf_).mean()) / float((wf * uf_ * uf_).mean())
            xf -= k * uf_
            xo -= k * uo_
        nf = np.sqrt(float((wf * xf * xf).mean()))
        if nf < 0.15:
            continue
        xf, xo = xf / nf, xo / nf
        b = CARPAN * float((wf * rf * xf).mean()) / np.sqrt(m0f)
        duz_f += b * xf
        duz_o += b * xo
        ONC_f.append(xf)
        ONC_o.append(xo)
    nf = np.sqrt(float((wf * duz_f * duz_f).mean()))
    no = np.sqrt(float((wo * duz_o * duz_o).mean()))
    if nf < 1e-12 or no < 1e-12:
        return 0.0, 0.0
    kf = float((wf * rf * (duz_f / nf)).mean()) / np.sqrt(m0f)
    ko = float((wo * ro * (duz_o / no)).mean()) / np.sqrt(m0o)
    return kf, ko


print(
    f"\n{'n':>4s} {'rho_pred':>9s} {'zaman fit':>10s} {'zaman sinav':>12s} "
    f"{'tutma':>7s} {'kesit tutma':>12s} {'tasinan rho':>12s} {'2.sira f':>9s}"
)
sonuc = []
for n in [4, 6, 8, 10, 12, 14, 16, 20, 24, 30, 40]:
    if n > len(BETA):
        break
    rho = float(np.sqrt(sum(b * b for b in BETA[:n])))
    zs = []
    for kes in ZAMAN_KESIM:
        kf_, ko_ = tutma2(n, uf <= kes, uf > kes)
        if abs(kf_) > 1e-12:
            zs.append(ko_ / kf_)
    t_zaman = float(np.median(zs)) if zs else 0.0
    t_sd = float(np.std(zs)) if zs else 0.0
    kf, ko = tutma2(n, uf <= 61, uf > 61)
    # kesit: trafo bolmeli, 4 tekrar
    ts = []
    for _ in range(6):
        sec = rng.random(len(uq)) < 0.5
        A = pd.Series(sec, index=uq)[tn].to_numpy()
        k1, k2 = tutma2(n, A, ~A)
        if abs(k1) > 1e-12:
            ts.append(k2 / k1)
    t_kesit = float(np.mean(ts)) if ts else 0.0
    tas = rho * min(t_zaman, 1.0)
    kap = np.sqrt(max(MSE_OPT - 0.99790**2, 1e-12))
    sonuc.append((n, rho, t_zaman, t_kesit, tas))
    print(
        f"{n:4d} {rho:9.4f} {t_zaman:10.3f} {t_sd:12.3f} {t_kesit:12.3f} "
        f"{tas:12.4f} {kap / rho:9.3f}"
    )

en = max(sonuc, key=lambda s: s[4])
print(
    f"\nEN IYI TASINAN rho: n={en[0]}  rho_pred={en[1]:.4f}  "
    f"zaman tutmasi={en[2]:.3f}  tasinan={en[4]:.4f}"
)
with open(os.path.join(BURA, "al_kesim.json"), "w") as fh:
    json.dump(
        [dict(n=s[0], rho=s[1], t_zaman=s[2], t_kesit=s[3], tasinan=s[4]) for s in sonuc],
        fh,
        indent=1,
    )
