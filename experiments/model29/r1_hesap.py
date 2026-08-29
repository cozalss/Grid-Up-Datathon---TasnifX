"""r1: OLCULMUS GONDERIMLERDEN ARTIK VEKTORUNUN SPAN BILESENI (r_span).

CEBIR (taban p0 = log1p(tuketim_m6_ikiyon.csv), m0 = 1.00284^2)
  r    = p0 - y                       (y = gercek hedef, log1p uzayinda)
  d_j  = p_j - p0                     (j = diger 24 olculmus gonderim)
  m_j  = LB_j^2 = ||p_j - y||^2 / N = m0 + 2<r,d_j>/N + Q_j
  =>  <r,d_j>/N = (m_j - m0 - Q_j)/2 = -L_j ,  L_j = (m0 + Q_j - m_j)/2

  Projeksiyon:  min_a ||r + sum a_j d_j||^2  ->  G a = L  ->  a = G^-1 L
  r_span = -sum_j a_j d_j = -a' D          (ISARET: gorevdekinin TERSI)
  ||r_span||^2/N = L' G^-1 L = m0 - MSE*   (span optimumunun kazanci)

  r_span > 0 olan satirda m6 FAZLA tahmin ediyor (p0 > y).

G'nin rank'i 21/24 (kosul sayisi ~1e12) -> KESIK SVD sart.
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
K_LIST = [5, 8, 10, 12, 13, 15, 17, 18, 20]
LAM_LIST = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]

# ---------------------------------------------------------------- veri
meta = json.loads((BURA / "g1_meta.json").read_text(encoding="utf-8"))
dosyalar = meta["dosyalar"]
X = np.load(BURA / "g1_X.npy")  # (25, N) log1p
N = X.shape[1]
assert N == meta["n"] == 714688
i0 = dosyalar.index(TABAN)
p0 = X[i0]
M0 = SKOR[TABAN] ** 2

jj = [j for j in range(len(dosyalar)) if j != i0]
adlar = [dosyalar[j].replace("tuketim_", "").replace(".csv", "") for j in jj]
D = X[jj] - p0  # (24, N)
n = D.shape[0]
G = (D @ D.T) / N
Qd = np.diag(G).copy()
mj = np.array([SKOR[dosyalar[j]] ** 2 for j in jj])
L = (M0 + Qd - mj) / 2.0

w, V = np.linalg.eigh(G)
o = np.argsort(-w)
w, V = w[o], V[:, o]
Lt = V.T @ L

print("N=%d  yon sayisi=%d  m0(m6)=%.6f" % (N, n, M0))
print("ozdegerler: " + " ".join("%.2e" % x for x in w))


def katsayi_kesik(k, Lv=L):
    Ltv = V.T @ Lv
    return V[:, :k] @ (Ltv[:k] / w[:k])


def katsayi_ridge(lam, Lv=L):
    return np.linalg.solve(G + lam * np.eye(n), Lv)


def rspan(a):
    return -(a @ D)


# ---------------------------------------------------------------- ana cozum
a_ana = katsayi_kesik(K_ANA)
r = rspan(a_ana)
kaz = float(a_ana @ L)  # = ||r_span||^2/N
print("\n== ANA COZUM k=%d ==" % K_ANA)
print("||r_span||^2/N = %.6f   ||r||^2/N = m0 = %.6f" % (kaz, M0))
print("GORULEBILEN PAY = %.4f%%" % (100 * kaz / M0))
print("MSE* = %.6f  -> S* = %.5f" % (M0 - kaz, np.sqrt(M0 - kaz)))
print("r_span: ort %+.6f  sd %.6f  min %+.4f  maks %+.4f" % (r.mean(), r.std(), r.min(), r.max()))
print("dogrulama ||r_span||^2/N (dogrudan) = %.6f" % float((r**2).mean()))

# ---------------------------------------------------------------- ozellikler
te = pd.read_csv(KOK / "data/raw/test.csv", parse_dates=["tarih"], dtype={"tanim": str})
tr = pd.read_csv(KOK / "data/raw/train.csv", parse_dates=["tarih"], dtype={"tanim": str})
assert len(te) == N
soguk = (~te.tanim.isin(set(tr.tanim))).values
_pl = te.lokasyon.str.split(">")
il = _pl.str[0].values.astype(object)
ilce = _pl.str[-1].values.astype(object)
bolge = np.where(
    _pl.str.len().values == 3,
    _pl.str[0].values.astype(object) + ">" + _pl.str[1].values.astype(object),
    _pl.str[0].values.astype(object),
)
ay = te.tarih.dt.month.values
ilk = te.groupby("tanim").tarih.transform("min").values
dalga = ilk == np.datetime64("2026-05-11")
guc = te.guc.values.astype(float)
tanim = te.tanim.values
seviye = p0  # m6 log1p tahmin seviyesi

gucband = pd.cut(guc, [-1, 50, 100, 160, 250, 400, 630, 1000, 1e9], labels=False).astype(int)
GUCAD = ["<=50", "50-100", "100-160", "160-250", "250-400", "400-630", "630-1000", ">1000"]
sevdes = pd.qcut(seviye, 10, labels=False, duplicates="drop").astype(int)

print(
    "\nsoguk satir %d (%.2f%%)  dalga satir %d (%.2f%%)"
    % (soguk.sum(), 100 * soguk.mean(), dalga.sum(), 100 * dalga.mean())
)

# ---------------------------------------------------------------- kirilim
TOPK2 = float((r**2).sum())


def kirilim(ad, etiketler, degerler=None):
    """etiketler: N uzunlugunda kategori dizisi"""
    s = pd.DataFrame({"g": etiketler, "r": r, "r2": r**2})
    ag = s.groupby("g").agg(n=("r", "size"), ort=("r", "mean"), kare=("r2", "sum"))
    ag["satir_pay"] = ag.n / N
    ag["kare_pay"] = ag.kare / TOPK2
    ag["rms"] = np.sqrt(ag.kare / ag.n)
    ag["zenginlik"] = ag.kare_pay / ag.satir_pay
    return ag.sort_values("kare_pay", ascending=False)


def bas(ad, ag, sinir=20):
    print("\n== KIRILIM: %s ==" % ad)
    print(
        "%-24s %9s %7s %8s %8s %9s %9s" % ("grup", "n", "satir%", "kare%", "zengin", "ort_r", "rms")
    )
    for g, rw in ag.head(sinir).iterrows():
        print(
            "%-24s %9d %6.2f%% %7.2f%% %8.2f %+9.5f %9.5f"
            % (
                str(g)[:24],
                rw.n,
                100 * rw.satir_pay,
                100 * rw.kare_pay,
                rw.zenginlik,
                rw.ort,
                rw.rms,
            )
        )


KIR = {}
tablolar = {
    "soguk": np.where(soguk, "SOGUK(train'de yok)", "SICAK"),
    "dalga_0511": np.where(dalga, "dalga 2026-05-11", "diger"),
    "ay": np.array(["ay%d" % a for a in ay]),
    "guc_bandi": np.array([GUCAD[b] for b in gucband]),
    "bolge": bolge,
    "seviye_desil": np.array(["D%02d" % d for d in sevdes]),
    "soguk_x_ay": np.array(["%s-ay%d" % ("SOG" if s else "SIC", a) for s, a in zip(soguk, ay)]),
}
for ad, et in tablolar.items():
    ag = kirilim(ad, et)
    bas(ad, ag)
    KIR[ad] = {
        str(g): dict(
            n=int(rw.n),
            satir_pay=float(rw.satir_pay),
            kare_pay=float(rw.kare_pay),
            zenginlik=float(rw.zenginlik),
            ort_r=float(rw.ort),
            rms=float(rw.rms),
        )
        for g, rw in ag.iterrows()
    }

ag_ilce = kirilim("ilce", ilce)
bas("ilce (ilk 20 / kare payina gore)", ag_ilce, 20)
KIR["ilce"] = {
    str(g): dict(
        n=int(rw.n),
        satir_pay=float(rw.satir_pay),
        kare_pay=float(rw.kare_pay),
        zenginlik=float(rw.zenginlik),
        ort_r=float(rw.ort),
        rms=float(rw.rms),
    )
    for g, rw in ag_ilce.head(30).iterrows()
}

# ---------------------------------------------------------------- trafo
tf = (
    pd.DataFrame({"t": tanim, "r": r, "r2": r**2})
    .groupby("t")
    .agg(n=("r", "size"), ort=("r", "mean"), kare=("r2", "sum"))
)
tf = tf.sort_values("kare", ascending=False)
top50 = tf.head(50)
print("\n== EN BUYUK 50 TRAFO (kare toplamina gore) ==")
print(
    "toplam kare payi: %.2f%% (satir payi %.3f%%)"
    % (100 * top50.kare.sum() / TOPK2, 100 * top50.n.sum() / N)
)
_tfd = pd.DataFrame(
    {"tanim": tanim, "guc": guc, "bolge": bolge, "ilce": ilce, "tarih": te.tarih.values}
)
ilk_tf = _tfd.groupby("tanim").agg(
    guc=("guc", "first"), bolge=("bolge", "first"), ilce=("ilce", "first"), ilk=("tarih", "min")
)
sog_tf = ~ilk_tf.index.isin(set(tr.tanim))
ilk_tf["soguk"] = sog_tf
sev_tf = pd.DataFrame({"t": tanim, "s": seviye}).groupby("t").s.mean()
print(
    "  soguk orani  ilk50 %.3f   tumu %.3f"
    % (ilk_tf.loc[top50.index, "soguk"].mean(), ilk_tf["soguk"].mean())
)
print(
    "  medyan guc   ilk50 %.0f     tumu %.0f"
    % (ilk_tf.loc[top50.index, "guc"].median(), ilk_tf["guc"].median())
)
print("  ort seviye   ilk50 %.3f    tumu %.3f" % (sev_tf.loc[top50.index].mean(), sev_tf.mean()))
print(
    "  dalga orani  ilk50 %.3f   tumu %.3f"
    % (
        (ilk_tf.loc[top50.index, "ilk"] == pd.Timestamp("2026-05-11")).mean(),
        (ilk_tf["ilk"] == pd.Timestamp("2026-05-11")).mean(),
    )
)
print(
    "\n%-12s %6s %5s %8s %9s %9s %-18s %s"
    % ("trafo", "guc", "n", "kare", "ort_r", "seviye", "ilce", "sog")
)
for t, rw in top50.head(25).iterrows():
    m = ilk_tf.loc[t]
    print(
        "%-12s %6.0f %5d %8.2f %+9.4f %9.3f %-18s %s"
        % (t, m.guc, rw.n, rw.kare, rw.ort, sev_tf.loc[t], m.ilce[:18], "SOG" if m.soguk else "sic")
    )

KIR["top50_trafo"] = dict(
    kare_pay=float(top50.kare.sum() / TOPK2),
    satir_pay=float(top50.n.sum() / N),
    soguk_orani=float(ilk_tf.loc[top50.index, "soguk"].mean()),
    soguk_orani_tumu=float(ilk_tf["soguk"].mean()),
    medyan_guc=float(ilk_tf.loc[top50.index, "guc"].median()),
    medyan_guc_tumu=float(ilk_tf["guc"].median()),
    ort_seviye=float(sev_tf.loc[top50.index].mean()),
    ort_seviye_tumu=float(sev_tf.mean()),
    dalga_orani=float((ilk_tf.loc[top50.index, "ilk"] == pd.Timestamp("2026-05-11")).mean()),
    dalga_orani_tumu=float((ilk_tf["ilk"] == pd.Timestamp("2026-05-11")).mean()),
    liste=[
        dict(
            trafo=str(t),
            guc=float(ilk_tf.loc[t, "guc"]),
            n=int(rw.n),
            kare=float(rw.kare),
            ort_r=float(rw.ort),
            seviye=float(sev_tf.loc[t]),
            ilce=str(ilk_tf.loc[t, "ilce"]),
            soguk=bool(ilk_tf.loc[t, "soguk"]),
        )
        for t, rw in top50.iterrows()
    ],
)

# ---------------------------------------------------------------- duyarlilik
# hizli kohort ortalamasi: cohort x yon toplam matrisi
KOHORT = {
    "soguk": np.where(soguk, "SOG", "SIC"),
    "dalga": np.where(dalga, "DLG", "yok"),
    "ay": np.array(["ay%d" % a for a in ay]),
    "guc": np.array([GUCAD[b] for b in gucband]),
    "bolge": bolge,
    "desil": np.array(["D%02d" % d for d in sevdes]),
}
kod, seviyeler, SUM = {}, {}, {}
for ad, et in KOHORT.items():
    u, c = np.unique(et, return_inverse=True)
    seviyeler[ad] = list(u)
    S = np.zeros((n, len(u)))
    for j in range(n):
        S[j] = np.bincount(c, weights=D[j], minlength=len(u))
    cnt = np.bincount(c, minlength=len(u)).astype(float)
    SUM[ad] = (S, cnt)


def kohort_ort(a):
    """a katsayi vektoru -> {eksen: kohort ortalama r_span dizisi}"""
    out = {}
    for ad, (S, cnt) in SUM.items():
        out[ad] = -(a @ S) / cnt
    return out


ana_ort = kohort_ort(a_ana)

print("\n== DUYARLILIK: k degisince kohort ortalama r_span ==")
duy = {"kesik": {}, "ridge": {}}
ref = np.concatenate([ana_ort[a] for a in KOHORT])
for k in K_LIST:
    ak = katsayi_kesik(k)
    ok = kohort_ort(ak)
    v = np.concatenate([ok[a] for a in KOHORT])
    kzk = float(ak @ L)
    korr = float(np.corrcoef(v, ref)[0, 1])
    isaret = float((np.sign(v) == np.sign(ref)).mean())
    duy["kesik"][str(k)] = dict(
        kazanc=kzk, korr_ana=korr, isaret_uyum=isaret, kohort={a: ok[a].tolist() for a in KOHORT}
    )
    print(
        "k=%2d kazanc=%.6f  korr(ana)=%+.3f  isaret_uyumu=%.3f  soguk=%s"
        % (k, kzk, korr, isaret, " ".join("%+.5f" % x for x in ok["soguk"]))
    )

print("\n== DUYARLILIK: ridge lambda ==")
for lam in LAM_LIST:
    al = katsayi_ridge(lam)
    ol = kohort_ort(al)
    v = np.concatenate([ol[a] for a in KOHORT])
    kzl = float(al @ L)
    korr = float(np.corrcoef(v, ref)[0, 1])
    isaret = float((np.sign(v) == np.sign(ref)).mean())
    duy["ridge"]["%g" % lam] = dict(
        kazanc=kzl, korr_ana=korr, isaret_uyum=isaret, kohort={a: ol[a].tolist() for a in KOHORT}
    )
    print(
        "lam=%7.0e kazanc=%.6f  korr(ana)=%+.3f  isaret_uyumu=%.3f  soguk=%s"
        % (lam, kzl, korr, isaret, " ".join("%+.5f" % x for x in ol["soguk"]))
    )

# ---------------------------------------------------------------- Monte Carlo
print("\n== MONTE CARLO: skor yuvarlamasi +-5e-6 (k=%d, 400 cekilis) ==" % K_ANA)
rng = np.random.default_rng(20260829)
NMC = 400
yig = {a: [] for a in KOHORT}
kazlar = []
for _ in range(NMC):
    dS = rng.uniform(-5e-6, 5e-6, size=n)
    dS0 = rng.uniform(-5e-6, 5e-6)
    S_ = np.sqrt(mj) + dS
    M0_ = (SKOR[TABAN] + dS0) ** 2
    L_ = (M0_ + Qd - S_**2) / 2.0
    a_ = katsayi_kesik(K_ANA, L_)
    kazlar.append(float(a_ @ L_))
    ok = kohort_ort(a_)
    for a in KOHORT:
        yig[a].append(ok[a])
mc = {}
for a in KOHORT:
    A = np.array(yig[a])
    mu, sd = A.mean(0), A.std(0)
    isaret_kar = (np.sign(A) == np.sign(ana_ort[a])[None, :]).mean(0)
    mc[a] = dict(
        seviye=seviyeler[a],
        ana=ana_ort[a].tolist(),
        mc_ort=mu.tolist(),
        mc_sd=sd.tolist(),
        isaret_kararlilik=isaret_kar.tolist(),
        t=(np.abs(ana_ort[a]) / np.maximum(sd, 1e-15)).tolist(),
    )
    print("\n  eksen %s" % a)
    for i, s in enumerate(seviyeler[a]):
        print(
            "    %-20s ana %+9.5f  mc %+9.5f +- %.5f  |t|=%6.1f  isaret_kar=%.3f"
            % (
                s,
                ana_ort[a][i],
                mu[i],
                sd[i],
                abs(ana_ort[a][i]) / max(sd[i], 1e-15),
                isaret_kar[i],
            )
        )
print("\n  kazanc MC: %.6f +- %.6f (ana %.6f)" % (np.mean(kazlar), np.std(kazlar), kaz))

# ---------------------------------------------------------------- kaydet
np.save(BURA / "r1_rspan.npy", r)
np.save(BURA / "r1_a.npy", a_ana)
json.dump(
    dict(
        taban=TABAN,
        m0=M0,
        N=N,
        k_ana=K_ANA,
        n_yon=n,
        adlar=adlar,
        ozdegerler=w.tolist(),
        Q=Qd.tolist(),
        L=L.tolist(),
        a=a_ana.tolist(),
        norm2_rspan=kaz,
        gorulen_pay=kaz / M0,
        mse_yildiz=M0 - kaz,
        s_yildiz=float(np.sqrt(M0 - kaz)),
        rspan_ozet=dict(
            ort=float(r.mean()),
            sd=float(r.std()),
            min=float(r.min()),
            maks=float(r.max()),
            q=[
                float(np.quantile(r, q))
                for q in (0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999)
            ],
        ),
        kirilim=KIR,
        duyarlilik=duy,
        monte_carlo=mc,
        mc_kazanc=dict(ort=float(np.mean(kazlar)), sd=float(np.std(kazlar))),
    ),
    open(BURA / "r1_artik.json", "w", encoding="utf-8"),
    indent=1,
    ensure_ascii=False,
)
print("\nYAZILDI r1_artik.json, r1_rspan.npy, r1_a.npy")
