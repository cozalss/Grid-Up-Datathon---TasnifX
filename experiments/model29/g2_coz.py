"""g2 -- olculmus gonderimlerin AFIN span'inda ulasilabilir en iyi MSE.

Cebir (agirliklar toplami 1 olan afin kombinasyon icin Y=||y||^2/N dusuyor):
    A   = X X^T / N          (hesaplanabilir)
    a   = diag(A)
    m_i = LB_i^2             (olculdu)
    F(w) = w'A w - w'a + w'm ,   1'w = 1      <-- TAM MSE (ayni satir kumesinde)

Hata kaynaklari:
  (1) skor yuvarlamasi: dF = w'dm,  sd(dm_i) = 2 s_i * 1e-5/sqrt(12)
  (2) public alt-kume: olculen m PUBLIC uzerinde, A ise TAM kume uzerinde.
      MSE_pub(w) = F(w) - E(w),  E(w) = mean_full(g) - mean_pub(g),
      g_row = (sum_i w_i p_i)^2 - sum_i w_i p_i^2
      => sd(E) = std(g) * sqrt((1-f)/(f N))     [analitik, MC gerekmez]

KAGGLE'A HICBIR SEY GONDERILMEZ.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from g1_yukle import YENI, yukle  # noqa: E402

BURA = Path(__file__).resolve().parent
DELTA_S = 1e-5 / np.sqrt(12.0)
F_PUBLIC = (0.20, 0.30, 0.50)

ADLAR, DOSYALAR, X, SKOR = yukle()
K, N = X.shape
A = (X @ X.T) / N
aa = np.diag(A).copy()
M = SKOR**2
YENI_AD = {d.replace("tuketim_", "").replace(".csv", "") for d in YENI}
IDX = {a: i for i, a in enumerate(ADLAR)}


def F(w: np.ndarray) -> float:
    return float(w @ A @ w - w @ aa + w @ M)


def grad(w: np.ndarray) -> np.ndarray:
    return 2 * A @ w - aa + M


# ---------------------------------------------------------------- yardimcilar
def alt(idx: list[int]):
    """Alt kume icin (A,a,m) dondur."""
    ix = np.array(idx)
    return A[np.ix_(ix, ix)], aa[ix], M[ix]


def afin_coz(idx: list[int], lam: float = 0.0, taban: int = 0):
    """Verilen dosya alt kumesinde afin minimum. lam: ridge (v uzerinde)."""
    As, as_, ms = alt(idx)
    k = len(idx)
    d = np.arange(k) != taban
    # v parametrizasyonu: w = e_taban + sum v_i (e_i - e_taban)
    Z = np.zeros((k, k - 1))
    Z[taban, :] = -1.0
    Z[np.where(d)[0], np.arange(k - 1)] = 1.0
    G = Z.T @ As @ Z
    e = np.zeros(k)
    e[taban] = 1.0
    g0 = 2 * As @ e - as_ + ms
    rhs = -0.5 * (Z.T @ g0)
    v = np.linalg.solve(G + lam * np.eye(k - 1), rhs)
    w = e + Z @ v
    wf = np.zeros(K)
    wf[np.array(idx)] = w
    return wf, G


def l1_coz(idx: list[int], tau: float, negsiz: bool = False):
    """min F s.t. 1'w=1, |w|_1<=tau  (u-v ayrimi ile QP)."""
    As, as_, ms = alt(idx)
    k = len(idx)
    P = np.hstack([np.eye(k), -np.eye(k)])

    def f(z):
        w = P @ z
        return float(w @ As @ w - w @ as_ + w @ ms)

    def fp(z):
        w = P @ z
        return P.T @ (2 * As @ w - as_ + ms)

    cons = [
        {
            "type": "eq",
            "fun": lambda z: P @ z @ np.ones(k) - 1.0,
            "jac": lambda z: P.T @ np.ones(k),
        },
        {"type": "ineq", "fun": lambda z: tau - z.sum(), "jac": lambda z: -np.ones(2 * k)},
    ]
    z0 = np.zeros(2 * k)
    j0 = int(np.argmin(ms))
    z0[j0] = 1.0
    bnd = [(0, None)] * k + [(0, 0)] * k if negsiz else [(0, None)] * (2 * k)
    r = minimize(
        f,
        z0,
        jac=fp,
        bounds=bnd,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 800, "ftol": 1e-14},
    )
    w = P @ r.x
    wf = np.zeros(K)
    wf[np.array(idx)] = w
    return wf, r


# ------------------------------------------------------------- belirsizlikler
def sd_yuvarlama(w: np.ndarray) -> float:
    sd_m = 2 * SKOR * DELTA_S
    return float(np.sqrt(np.sum((w * sd_m) ** 2)))


def sd_public(w: np.ndarray, f: float) -> float:
    u = w @ X
    g = u * u - (w @ (X * X))
    return float(g.std(ddof=1) * np.sqrt((1 - f) / (f * N)))


def rapor_w(w: np.ndarray) -> dict:
    mse = F(w)
    sr = sd_yuvarlama(w)
    sp = {f"f={f}": sd_public(w, f) for f in F_PUBLIC}
    tot = {k: float(np.hypot(sr, v)) for k, v in sp.items()}
    return {
        "mse": mse,
        "rmsle": float(np.sqrt(max(mse, 0.0))),
        "l1": float(np.abs(w).sum()),
        "linf": float(np.abs(w).max()),
        "sd_yuvarlama": sr,
        "sd_public": sp,
        "sd_toplam": tot,
        "w": {ADLAR[i]: float(w[i]) for i in np.argsort(-np.abs(w)) if abs(w[i]) > 1e-6},
    }


# ------------------------------------------------------------------ analizler
def fark_gram(idx: list[int], taban: int = 0):
    """d_i = p_i - p_taban icin G ve b (afin uzayin metrigi)."""
    ix = np.array(idx)
    k = len(ix)
    d = [i for i in range(k) if i != taban]
    As, as_, ms = alt(idx)
    G = np.empty((k - 1, k - 1))
    for r, i in enumerate(d):
        for c, j in enumerate(d):
            G[r, c] = As[i, j] - As[i, taban] - As[taban, j] + As[taban, taban]
    m0 = ms[taban]
    b = (m0 + np.diag(G) - ms[np.array(d)]) / 2.0
    return G, b, m0, [idx[i] for i in d]


def tutarlilik(idx: list[int]) -> dict:
    """Yakin-null yonlerde Cauchy-Schwarz / null tutarliligi -> public oran tahmini."""
    G, b, m0, dad = fark_gram(idx)
    lamv, U = np.linalg.eigh(G)
    sira = np.argsort(lamv)[::-1]
    lamv, U = lamv[sira], U[:, sira]
    kayit = []
    sk = SKOR[np.array(idx)]
    for j in range(len(lamv)):
        v = U[:, j]
        # w-uzayinda sum=0 yonu
        u = np.zeros(K)
        u[np.array(dad)] = v
        u[idx[0]] = -v.sum()
        gozlem = abs(float(u @ (M - aa)))  # null ise 0 olmali
        sd_r = float(np.sqrt(np.sum((u[np.array(idx)] * 2 * sk * DELTA_S) ** 2)))
        h = u @ (X * X)
        sd_p = {f: float(h.std(ddof=1) * np.sqrt((1 - f) / (f * N))) for f in F_PUBLIC}
        cs = float(np.sqrt(m0 * max(lamv[j], 0)) * 2)  # |b.v| ust siniri x2 olcek
        kayit.append(
            {
                "j": j,
                "ozdeger": float(lamv[j]),
                "yon_normu": float(np.sqrt(max(lamv[j], 0))),
                "ihlal": gozlem,
                "sd_yuvarlama": sd_r,
                "sd_public": {str(k_): v_ for k_, v_ in sd_p.items()},
                "cs_siniri": cs,
            }
        )
    return {"ozdegerler": lamv.tolist(), "yonler": kayit}


def loo(idx: list[int]) -> list[dict]:
    """Her dosyayi disarida birak, digerlerinin AFIN span'indan skorunu TAHMIN et."""
    out = []
    for pos, j in enumerate(idx):
        kalan = [i for i in idx if i != j]
        kx = np.array(kalan)
        Xk = X[kx]
        # min ||p_j - sum c_i p_i||^2  s.t. sum c = 1
        Ak = A[np.ix_(kx, kx)]
        bk = (X[kx] @ X[j]) / N
        kk = len(kalan)
        KKT = np.zeros((kk + 1, kk + 1))
        KKT[:kk, :kk] = 2 * Ak
        KKT[:kk, kk] = 1.0
        KKT[kk, :kk] = 1.0
        rhs = np.concatenate([2 * bk, [1.0]])
        sol = np.linalg.lstsq(KKT, rhs, rcond=None)[0]
        c = sol[:kk]
        q = c @ Xk
        r2 = float(np.mean((X[j] - q) ** 2))
        wf = np.zeros(K)
        wf[kx] = c
        tahmin = F(wf) + r2
        cs_bias = 2 * np.sqrt(r2) * np.sqrt(M[idx[0]])  # |2<r,p0-y>|/N ust siniri
        out.append(
            {
                "dosya": ADLAR[j],
                "gercek_mse": float(M[j]),
                "tahmin_mse": float(tahmin),
                "hata_mse": float(tahmin - M[j]),
                "gercek_rmsle": float(SKOR[j]),
                "tahmin_rmsle": float(np.sqrt(max(tahmin, 0))),
                "hata_rmsle": float(np.sqrt(max(tahmin, 0)) - SKOR[j]),
                "artik_norm2": r2,
                "artik_orani": float(np.sqrt(r2 / M[j])),
                "cs_bias_siniri": float(cs_bias),
                "c_l1": float(np.abs(c).sum()),
            }
        )
    return out


def zaman_sinavi(hedefler: list[str], gecmis: list[str]) -> list[dict]:
    """Kronolojik ileri sinav: sadece GECMIS dosyalarla hedefin skorunu tahmin et."""
    gi = [IDX[a] for a in gecmis]
    out = []
    for h in hedefler:
        j = IDX[h]
        gx = np.array(gi)
        Ak = A[np.ix_(gx, gx)]
        bk = (X[gx] @ X[j]) / N
        kk = len(gi)
        KKT = np.zeros((kk + 1, kk + 1))
        KKT[:kk, :kk] = 2 * Ak
        KKT[:kk, kk] = 1.0
        KKT[kk, :kk] = 1.0
        sol = np.linalg.lstsq(KKT, np.concatenate([2 * bk, [1.0]]), rcond=None)[0]
        c = sol[:kk]
        q = c @ X[gx]
        r2 = float(np.mean((X[j] - q) ** 2))
        wf = np.zeros(K)
        wf[gx] = c
        tah = F(wf) + r2
        out.append(
            {
                "hedef": h,
                "gercek_rmsle": float(SKOR[j]),
                "tahmin_rmsle": float(np.sqrt(max(tah, 0))),
                "hata_rmsle": float(np.sqrt(max(tah, 0)) - SKOR[j]),
                "hata_mse": float(tah - M[j]),
                "artik_orani": float(np.sqrt(r2 / M[j])),
                "c_l1": float(np.abs(c).sum()),
            }
        )
    return out


def kademeli(maks: int = 12) -> list[dict]:
    """Ileriye dogru acgozlu secim: en iyi k dosyayla ulasilabilir MSE."""
    secili = [int(np.argmin(M))]
    out = [
        {
            "k": 1,
            "dosyalar": [ADLAR[secili[0]]],
            "mse": float(M[secili[0]]),
            "rmsle": float(SKOR[secili[0]]),
            "l1": 1.0,
        }
    ]
    kalan = [i for i in range(K) if i not in secili]
    while len(secili) < maks and kalan:
        en, eni = None, None
        for i in kalan:
            try:
                w, _ = afin_coz(sorted(secili + [i]))
            except np.linalg.LinAlgError:
                continue
            f = F(w)
            if en is None or f < en:
                en, eni, enw = f, i, w
        secili = sorted(secili + [eni])
        kalan.remove(eni)
        out.append(
            {
                "k": len(secili),
                "dosyalar": [ADLAR[i] for i in secili],
                "eklenen": ADLAR[eni],
                "mse": float(en),
                "rmsle": float(np.sqrt(max(en, 0))),
                "l1": float(np.abs(enw).sum()),
                "sd_yuvarlama": sd_yuvarlama(enw),
                "sd_public_f02": sd_public(enw, 0.2),
            }
        )
    return out


def mc_yuvarlama(cozucu, tekrar: int = 600, tohum: int = 7) -> dict:
    """Skorlari +-5e-6 salla, YENIDEN COZ. Hem ongorulen hem GERCEKLESEN MSE dagilimi."""
    global M
    M0 = M.copy()
    rng = np.random.default_rng(tohum)
    ongoru, gercek, l1 = [], [], []
    for _ in range(tekrar):
        s = SKOR + rng.uniform(-5e-6, 5e-6, K)
        M = s**2
        try:
            w = cozucu()
        except Exception:
            M = M0.copy()
            continue
        ongoru.append(float(w @ A @ w - w @ aa + w @ M))
        M = M0.copy()
        gercek.append(F(w))
        l1.append(float(np.abs(w).sum()))
    M = M0.copy()
    o, g = np.array(ongoru), np.array(gercek)
    return {
        "tekrar": len(o),
        "ongoru_ort": float(o.mean()),
        "ongoru_sd": float(o.std(ddof=1)),
        "ongoru_p5_p95": [float(np.percentile(o, 5)), float(np.percentile(o, 95))],
        "gerceklesen_ort": float(g.mean()),
        "gerceklesen_sd": float(g.std(ddof=1)),
        "gerceklesen_p5_p95": [float(np.percentile(g, 5)), float(np.percentile(g, 95))],
        "sapma_ort": float((g - o).mean()),
        "sapma_maks": float(np.abs(g - o).max()),
        "rmsle_ongoru_sd": float(np.sqrt(np.maximum(o, 0)).std(ddof=1)),
        "l1_ort": float(np.mean(l1)),
        "l1_maks": float(np.max(l1)),
    }


def main() -> None:
    R: dict = {
        "n": int(N),
        "k": int(K),
        "dosyalar": ADLAR,
        "skorlar": {a: float(s) for a, s in zip(ADLAR, SKOR)},
    }
    print("=" * 96)
    print("g2 -- AFIN SPAN COZUMU   K=%d dosya  N=%d satir" % (K, N))
    print("=" * 96)

    tum = list(range(K))
    G, b, m0, dad = fark_gram(tum)
    lamv = np.linalg.eigvalsh(G)[::-1]
    tol = lamv[0] * len(lamv) * np.finfo(float).eps
    rank = int((lamv > tol).sum())
    R["fark_gram"] = {
        "ozdegerler": lamv.tolist(),
        "rank": rank,
        "kosul_sayisi": float(lamv[0] / lamv[rank - 1]),
        "kosul_tam": float(lamv[0] / max(lamv[-1], 1e-300)),
    }
    print("")
    print("Fark-Gram (24x24): rank=%d/24  kosul(rank ici)=%.3e" % (rank, lamv[0] / lamv[rank - 1]))
    print("ozdegerler: " + "  ".join("%.2e" % x for x in lamv))

    Cn = np.corrcoef(X)
    ikili = []
    for i in range(K):
        for j in range(i + 1, K):
            dij = float(np.mean((X[i] - X[j]) ** 2))
            ikili.append((ADLAR[i], ADLAR[j], float(np.sqrt(dij)), float(Cn[i, j])))
    ikili.sort(key=lambda t: t[2])
    R["en_yakin_ciftler"] = [
        {"a": a, "b": bb, "rms_fark": c, "korelasyon": d} for a, bb, c, d in ikili[:12]
    ]
    kum = np.cumsum(lamv) / lamv.sum()
    essiz = {str(p): int(np.searchsorted(kum, p) + 1) for p in (0.99, 0.999, 0.9999, 0.999999)}
    R["essiz_yon_sayisi"] = essiz
    print("")
    print("fark-enerjisinin %%99'u %s yonde, %%99.99'u %s yonde" % (essiz["0.99"], essiz["0.9999"]))
    print("en yakin 5 cift: " + str([(a, bb, round(c, 4)) for a, bb, c, _ in ikili[:5]]))

    T = tutarlilik(tum)
    R["tutarlilik"] = T
    print("")
    print("--- NULL-YON TUTARLILIK (ihlal = u.(m-a); tam-kume + hatasiz olsa 0 olmali) ---")
    print("%3s %11s %11s %10s %12s %9s" % ("j", "ozdeger", "ihlal", "sd_yuv", "sd_pub0.2", "z_yuv"))
    for r in T["yonler"][-8:]:
        print(
            "%3d %11.3e %11.3e %10.2e %12.2e %9.1f"
            % (
                r["j"],
                r["ozdeger"],
                r["ihlal"],
                r["sd_yuvarlama"],
                r["sd_public"]["0.2"],
                r["ihlal"] / max(r["sd_yuvarlama"], 1e-30),
            )
        )

    print("")
    print("--- COZUMLER ---")
    coz_tablo = {}
    w_un, _ = afin_coz(tum)
    coz_tablo["kisitsiz"] = rapor_w(w_un)
    for lam in (1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        w, _ = afin_coz(tum, lam=lam)
        coz_tablo["ridge_%g" % lam] = rapor_w(w)
    for tau in (1.0, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0):
        w, r = l1_coz(tum, tau)
        d = rapor_w(w)
        d["basarili"] = bool(r.success)
        coz_tablo["l1_%g" % tau] = d
    w_sx, r_sx = l1_coz(tum, 1.0, negsiz=True)
    d = rapor_w(w_sx)
    d["basarili"] = bool(r_sx.success)
    coz_tablo["simpleks"] = d
    R["cozumler"] = coz_tablo
    print(
        "%14s %11s %9s %9s %9s %9s %9s"
        % ("cozum", "MSE", "RMSLE", "|w|_1", "sd_yuv", "sd_pub.2", "sd_pub.5")
    )
    for k_, v in coz_tablo.items():
        print(
            "%14s %11.6f %9.5f %9.3f %9.2e %9.2e %9.2e"
            % (
                k_,
                v["mse"],
                v["rmsle"],
                v["l1"],
                v["sd_yuvarlama"],
                v["sd_public"]["f=0.2"],
                v["sd_public"]["f=0.5"],
            )
        )

    print("")
    print("--- MONTE CARLO YUVARLAMA DUYARLILIGI (+-5e-6) ---")
    mc = {}
    mc["kisitsiz"] = mc_yuvarlama(lambda: afin_coz(tum)[0])
    for lam in (1e-6, 1e-4, 1e-2):
        mc["ridge_%g" % lam] = mc_yuvarlama(lambda lam=lam: afin_coz(tum, lam=lam)[0])
    for tau in (1.5, 2.0, 3.0):
        mc["l1_%g" % tau] = mc_yuvarlama(lambda tau=tau: l1_coz(tum, tau)[0], tekrar=200)
    mc["simpleks"] = mc_yuvarlama(lambda: l1_coz(tum, 1.0, negsiz=True)[0], tekrar=200)
    R["mc_yuvarlama"] = mc
    print(
        "%14s %12s %11s %12s %11s %11s %9s"
        % ("cozum", "ongoru_ort", "ongoru_sd", "gercek_ort", "gercek_sd", "sapma_ort", "|w|1")
    )
    for k_, v in mc.items():
        print(
            "%14s %12.6f %11.3e %12.6f %11.3e %11.3e %9.3f"
            % (
                k_,
                v["ongoru_ort"],
                v["ongoru_sd"],
                v["gerceklesen_ort"],
                v["gerceklesen_sd"],
                v["sapma_ort"],
                v["l1_ort"],
            )
        )

    print("")
    print("--- LEAVE-ONE-OUT CAPRAZ DOGRULAMA ---")
    L = loo(tum)
    R["loo"] = L
    print(
        "%20s %9s %9s %10s %8s %10s %8s"
        % ("dosya", "gercek", "tahmin", "hata", "artik%", "cs_bias", "|c|_1")
    )
    for r in sorted(L, key=lambda t: -abs(t["hata_rmsle"])):
        print(
            "%20s %9.5f %9.5f %+10.5f %8.3f %10.2e %8.2f"
            % (
                r["dosya"],
                r["gercek_rmsle"],
                r["tahmin_rmsle"],
                r["hata_rmsle"],
                r["artik_orani"] * 100,
                r["cs_bias_siniri"],
                r["c_l1"],
            )
        )
    he = np.array([r["hata_rmsle"] for r in L])
    hm = np.array([r["hata_mse"] for r in L])
    R["loo_ozet"] = {
        "mae_rmsle": float(np.abs(he).mean()),
        "rms_rmsle": float(np.sqrt((he**2).mean())),
        "maks_rmsle": float(np.abs(he).max()),
        "medyan_rmsle": float(np.median(np.abs(he))),
        "mae_mse": float(np.abs(hm).mean()),
        "rms_mse": float(np.sqrt((hm**2).mean())),
        "maks_mse": float(np.abs(hm).max()),
    }
    print("")
    print(
        "LOO ozet: MAE(RMSLE)=%.5f  RMS=%.5f  medyan=%.5f  maks=%.5f  |  MAE(MSE)=%.5f  RMS(MSE)=%.5f"
        % (
            np.abs(he).mean(),
            np.sqrt((he**2).mean()),
            np.median(np.abs(he)),
            np.abs(he).max(),
            np.abs(hm).mean(),
            np.sqrt((hm**2).mean()),
        )
    )

    print("")
    print("--- KRONOLOJIK ILERI SINAV (yalniz eski dosyalarla yeni skoru tahmin) ---")
    yeniler = [
        "v101_hepsi",
        "v102_kappa_optimum",
        "v109_birlesik",
        "m4_hava_capali",
        "m6_ikiyon",
        "p51_sicak05",
    ]
    eski = [a for a in ADLAR if a not in set(yeniler)]
    Z = zaman_sinavi(yeniler, eski)
    R["zaman_sinavi"] = {"gecmis": eski, "sonuc": Z}
    print("%20s %9s %9s %10s %8s %8s" % ("hedef", "gercek", "tahmin", "hata", "artik%", "|c|_1"))
    for r in Z:
        print(
            "%20s %9.5f %9.5f %+10.5f %8.3f %8.2f"
            % (
                r["hedef"],
                r["gercek_rmsle"],
                r["tahmin_rmsle"],
                r["hata_rmsle"],
                r["artik_orani"] * 100,
                r["c_l1"],
            )
        )

    print("")
    print("--- KADEMELI (acgozlu ileri secim) ---")
    KD = kademeli(12)
    R["kademeli"] = KD
    for r in KD:
        print(
            "  k=%2d  MSE=%.6f  RMSLE=%.5f  |w|1=%7.3f  +%s"
            % (r["k"], r["mse"], r["rmsle"], r["l1"], r.get("eklenen", "-"))
        )

    eski_idx = [IDX[a] for a in eski]
    w_e, _ = afin_coz(eski_idx)
    R["yeni_yonsuz"] = {"kisitsiz": rapor_w(w_e)}
    for tau in (1.5, 2.0, 3.0):
        w, _ = l1_coz(eski_idx, tau)
        R["yeni_yonsuz"]["l1_%g" % tau] = rapor_w(w)
    print("")
    print("--- YENI YONLER (v101/v102/v109/m4/m6/p51) OLMADAN ---")
    v = R["yeni_yonsuz"]["kisitsiz"]
    print("  kisitsiz: MSE=%.6f RMSLE=%.5f |w|1=%.2f" % (v["mse"], v["rmsle"], v["l1"]))
    for tau in (1.5, 2.0, 3.0):
        v = R["yeni_yonsuz"]["l1_%g" % tau]
        print("  l1<=%s: MSE=%.6f RMSLE=%.5f" % (tau, v["mse"], v["rmsle"]))

    (BURA / "g1_gram.json").write_text(
        json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    print("")
    print("yazildi: " + str(BURA / "g1_gram.json"))


if __name__ == "__main__":
    main()
