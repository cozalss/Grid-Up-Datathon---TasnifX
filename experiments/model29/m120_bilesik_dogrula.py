"""BILESIGIN KENDISI ASIRI UYUM MU? -- trafo-bolmeli capraz dogrulama.

Bilesigin agirliklari yaz25'in TUM trafolarindan hesaplandi. Eger o
korelasyonlar gurultuye uyuyorsa, bilesik baska trafolarda calismaz.

SINAV: yaz25 trafolarini ikiye bol.
  A yarisinda kor_i'leri hesapla -> agirliklar -> bilesik yon
  B yarisinda o bilesigin artikla korelasyonunu olc
  ve tersi.
Elde edilen "capraz kor", tum-veriyle hesaplanan "ic kor"un ne kadarina
denk? Oran 1'e yakinsa asiri uyum yok; 0'a yakinsa agirliklar gurultu.

PLASEBO: trafo etiketlerini karistir -> capraz kor 0'a inmeli.
"""

import json
import os

import numpy as np
import pandas as pd

KOK = r"c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
DN = os.path.join(KOK, "data/interim/deney")
AO = os.path.join(KOK, "data/interim/aile_onbellek")
S = os.path.join(KOK, "submissions")
M29 = os.path.join(KOK, "experiments/model29")
BURA = os.path.dirname(os.path.abspath(__file__))
M0, TABAN = 1.005846366, "tuketim_m6_ikiyon.csv"
HEDEF_SOGUK, CARPAN, TAVAN = 0.222, 0.798, 1.95

with open(os.path.join(BURA, "m122_nihai.json")) as fh:
    V2 = json.load(fh)
EKSENLER = V2["eksenler"]
print(f"bilesikteki {len(EKSENLER)} eksen: {EKSENLER}")

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


def st(x):
    x = np.asarray(x, dtype=np.float64).copy()
    f = np.isfinite(x)
    if not f.any():
        return None
    x[~f] = np.median(x[f])
    x -= x.mean()
    s = np.sqrt(float((x * x).mean()))
    return x / s if s > 1e-12 else None


svB = st(pb)


ufB = st(bf.ufuk_gun.to_numpy())
ayB = st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64))
CARP = {"x_sv": svB, "x_soguk": sgm, "x_ufuk": ufB, "x_ay": ayB}
ESIK = {"ust10": (0.9, True), "ust25": (0.75, True), "alt10": (0.1, False)}


def kurB(ad):
    """m122 ad bicimleri: 'kol', 'kol:kip', 'kolA*kolB'."""
    if "*" in ad:
        k1, k2 = ad.split("*", 1)
        if k1 not in bf.columns or k2 not in bf.columns:
            return None
        b1, b2 = st(bf[k1].to_numpy()), st(bf[k2].to_numpy())
        return None if b1 is None or b2 is None else st(b1 * b2)
    kol, kip = ad.split(":", 1) if ":" in ad else (ad, "")
    if kol not in bf.columns:
        return None
    xb = bf[kol].to_numpy()
    if kip in CARP:
        b_ = st(xb)
        return None if b_ is None else st(b_ * CARP[kip])
    if kip in ESIK:
        q, ust = ESIK[kip]
        fv = xb[np.isfinite(xb)]
        if fv.size == 0:
            return None
        v_ = np.quantile(fv, q)
        return st(((xb > v_) if ust else (xb < v_)).astype(np.float64))
    if kip == "kare":
        b_ = st(xb)
        return None if b_ is None else st(b_**2)
    return st(xb)


X = {}
for ad in EKSENLER:
    v = kurB(ad)
    if v is not None:
        X[ad] = v
print(f"blok tarafinda kurulan eksen: {len(X)}")


def bilesik_kor(fit_maske, olc_maske):
    """fit_maske'de agirlik hesapla, olc_maske'de bilesigin korelasyonunu olc."""
    wf, rf = ww[fit_maske], rb[fit_maske]
    m0f = float((wf * rf * rf).mean())
    wo, ro = ww[olc_maske], rb[olc_maske]
    m0o = float((wo * ro * ro).mean())
    duz_o = np.zeros(int(olc_maske.sum()))
    ONC_f, ONC_o = [], []
    for ad, x in X.items():
        xf, xo = x[fit_maske].copy(), x[olc_maske].copy()
        for uf, uo in zip(ONC_f, ONC_o):
            k = float((wf * xf * uf).mean()) / float((wf * uf * uf).mean())
            xf -= k * uf
            xo -= k * uo
        nf = np.sqrt(float((wf * xf * xf).mean()))
        if nf < 0.15:
            continue
        xf, xo = xf / nf, xo / nf
        beta = CARPAN * float((wf * rf * xf).mean()) / np.sqrt(m0f)
        duz_o += beta * xo
        ONC_f.append(xf)
        ONC_o.append(xo)
    n = np.sqrt(float((wo * duz_o * duz_o).mean()))
    if n < 1e-12:
        return 0.0, 0.0
    return float((wo * ro * (duz_o / n)).mean()) / np.sqrt(m0o), n


tam = np.ones(len(rb), dtype=bool)
ic_kor, ic_norm = bilesik_kor(tam, tam)
print(f"\nIC (tum veri, fit=olc): kor={ic_kor:+.4f}  bilesik normu={ic_norm:.4f}")
print(f"  -> birim yon icin rho_cv = carpan * kor = {CARPAN * ic_kor:+.4f}")

rng = np.random.default_rng(17)
tn = bf.tanim.values
uq = pd.unique(tn)
print(f"\nTRAFO-BOLMELI CAPRAZ DOGRULAMA ({len(uq):,} trafo, 8 tekrar)")
capraz = []
for t in range(8):
    sec = rng.random(len(uq)) < 0.5
    kod = pd.Series(sec, index=uq)
    A = kod[tn].to_numpy()
    B = ~A
    if A.sum() < 1000 or B.sum() < 1000:
        continue
    k1, _ = bilesik_kor(A, B)
    k2, _ = bilesik_kor(B, A)
    capraz += [k1, k2]
    print(f"  tekrar {t + 1}: A->B kor={k1:+.4f}   B->A kor={k2:+.4f}")
capraz = np.array(capraz)
print(f"\n  capraz kor ortalamasi = {capraz.mean():+.4f}  sd={capraz.std():.4f}")
print(f"  ic kor                = {ic_kor:+.4f}")
print(f"  TUTMA ORANI           = {capraz.mean() / ic_kor:.3f}  (1'e yakin = asiri uyum yok)")

print("\nPLASEBO (trafo etiketleri karistirilmis)")
gi = pd.Series(np.arange(len(uq)), index=uq)[tn].to_numpy()
bos = []
for t in range(6):
    perm = rng.permutation(len(uq))
    sira = np.argsort(np.argsort(perm[gi], kind="stable"), kind="stable")
    rb_k = rb[sira]
    tut = rb.copy()
    rb = rb_k
    sec = rng.random(len(uq)) < 0.5
    A = pd.Series(sec, index=uq)[tn].to_numpy()
    k, _ = bilesik_kor(A, ~A)
    bos.append(k)
    rb = tut
bos = np.array(bos)
print(f"  bos capraz kor: ort={bos.mean():+.4f} sd={bos.std():.4f}")
print(f"  z = {(capraz.mean() - bos.mean()) / (bos.std() + 1e-12):+.1f}")
with open(os.path.join(BURA, "aa_dogrula.json"), "w") as fh:
    json.dump(
        dict(
            ic_kor=ic_kor,
            capraz_ort=float(capraz.mean()),
            tutma=float(capraz.mean() / ic_kor),
            plasebo_ort=float(bos.mean()),
            plasebo_sd=float(bos.std()),
        ),
        fh,
        indent=1,
    )
