"""y1_rho: aday artik-modeli yonlerinin BLOK-DISI rho'su.
rho tanimi (brifing): w=agirlik(d) (kohorta agirlikli, ort=1),
  norm = sqrt(sum(w*u_ham^2)/sum(w)); u = u_ham/norm; rho = sum(w*r*u)/sum(w)
SE: TRAFO-kumeli (cluster-robust).
Ayrica ayni yonun {1}, {1,p,soguk} ve TAKVIM-YAPAYLIK altuzayina dik bilesenleri.
"""
import json, os, sys, time
import numpy as np

GEC = os.path.dirname(os.path.abspath(__file__))
AM = os.path.join(os.path.dirname(GEC), "am")
sys.path.insert(0, AM)
import b_ana as B

BLOKLAR = ("yaz25", "guz25", "kis26")
_K = {}


def kume(bad):
    if bad not in _K:
        z = np.load(os.path.join(GEC, f"kume_{bad}.npz"))
        _K[bad] = {k: z[k] for k in z.files}
    return _K[bad]


def rho_kumeli(r, g, w, kod):
    """brifing tanimi + trafo-kumeli SE."""
    g = np.asarray(g, np.float64)
    n2 = np.sum(w * g * g) / np.sum(w)
    if not np.isfinite(n2) or n2 <= 1e-24:
        return dict(rho=0.0, se=0.0, norm=0.0)
    u = g / np.sqrt(n2)
    z = r * u
    sw = np.sum(w)
    rho = float(np.sum(w * z) / sw)
    a = np.bincount(kod, weights=w * (z - rho))
    se = float(np.sqrt(np.sum(a * a)) / sw)
    return dict(rho=rho, se=se, norm=float(np.sqrt(n2)))


def dikle(g, Z, w):
    """agirlikli en kucuk kareler ile Z altuzayindan arindir."""
    A = Z * w[:, None]
    G = Z.T @ A
    b = A.T @ g
    c = np.linalg.solve(G + 1e-8 * np.eye(Z.shape[1]) * np.trace(G) / Z.shape[1], b)
    return g - Z @ c


def bazlar(bad, meta, km):
    n = len(meta["r"])
    bir = np.ones(n)
    p = meta["p"]
    sog = meta["soguk"].astype(np.float64)
    Z1 = bir[:, None]
    Z3 = np.column_stack([bir, p, sog])
    aylar = sorted(set(km["ay"].tolist()))
    D = [bir, p, sog]
    for a in aylar[1:]:
        D.append((km["ay"] == a).astype(np.float64))
    gun = km["gun"].astype(np.float64)
    D.append(gun / gun.max())
    D.append(((km["hg"] >= 5).astype(np.float64)))
    Ztak = np.column_stack(D)
    return Z1, Z3, Ztak


TAM = list(range(len(B.OZN)))
TAKVIMSIZ = [i for i in TAM if B.OZN[i] not in set(B.TAKVIM)]
KIMLIK = {"tanim", "tanim_num", "tanim_on2", "tanim_on3", "tanim_on4",
          "tanim_on5", "tanim_uzunluk"}
TANIMSIZ = [i for i in TAM if B.OZN[i] not in KIMLIK]
TANIMSIZ_TAKVIMSIZ = [i for i in TANIMSIZ if B.OZN[i] not in set(B.TAKVIM)]

KONF = {
    "ridge_merkezli_tamozn": dict(model="ridge", hedef_tip="merkezli", kol=TAM, par=None),
    "lgbm_ham_KIMLIKSIZ_takvimsiz": dict(model="lgbm", hedef_tip="ham",
                                         kol=TANIMSIZ_TAKVIMSIZ, par=None),
    "lgbm_merkezli_takvimsiz": dict(model="lgbm", hedef_tip="merkezli",
                                    kol=TAKVIMSIZ, par=None),
}

sec = sys.argv[1:] if len(sys.argv) > 1 else list(KONF)
OUT = []
for ad in sec:
    K = KONF[ad]
    for hb in BLOKLAR:
        t0 = time.time()
        eb = [(b, np.ones(len(B.yukle(b)[1]["r"]), bool)) for b in BLOKLAR if b != hb]
        g, mh, hm, ni, _ = B.kur(eb, (hb, np.ones(len(B.yukle(hb)[1]["r"]), bool)),
                                 K["kol"], K["hedef_tip"], K["model"], par=K["par"])
        meta = {k: v for k, v in mh.items()}
        km = kume(hb)
        r = meta["r"]; w = meta["w"] / meta["w"].mean(); kod = km["kume"]
        Z1, Z3, Ztak = bazlar(hb, meta, km)
        rec = dict(kurulum=ad, blok=hb, agac=int(ni), sn=round(time.time() - t0, 1),
                   g_rms=float(np.sqrt(np.average(g * g, weights=w))),
                   g_ort=float(np.average(g, weights=w)))
        for etiket, Z in (("ham", None), ("dik1", Z1), ("dik3", Z3), ("diktakvim", Ztak)):
            gg = g if Z is None else dikle(g, Z, w)
            rec[etiket] = rho_kumeli(r, gg, w, kod)
        OUT.append(rec)
        print(f"{ad:30s} -> {hb:6s} ham={rec['ham']['rho']:+.5f}+-{rec['ham']['se']:.5f} "
              f"dik1={rec['dik1']['rho']:+.5f}+-{rec['dik1']['se']:.5f} "
              f"dik3={rec['dik3']['rho']:+.5f}+-{rec['dik3']['se']:.5f} "
              f"diktak={rec['diktakvim']['rho']:+.5f}+-{rec['diktakvim']['se']:.5f} "
              f"agac={ni} {time.time()-t0:.0f}s", flush=True)
        with open(os.path.join(GEC, "y1_rho.json"), "w", encoding="utf-8") as fh:
            json.dump(OUT, fh, ensure_ascii=False, indent=1)
print("TAMAM -> y1_rho.json")
