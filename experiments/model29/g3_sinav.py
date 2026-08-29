"""g3 -- KARAR SINAVI.

1) Ince L1 tau taramasi: ongorulen MSE <-> |w|_1 <-> belirsizlik egrisi.
2) PUBLIC-ALT-KUME ONIYIMSERLIK SIMULASYONU (asil hata mekanizmasi).
   Analist A'yi TAM kumede, m'yi PUBLIC alt kumede olcuyor. Hatasi
   E(w) = w'(A-A_pub)w - w'(a-a_pub).  Ayni gurultu surecini rolleri
   degistirerek TAM olarak taklit edebiliriz: rasgele f oranli maske ile
   A_maske kur, onunla COZ, sonra TAM A ile degerlendir. Optimize edici
   gurultuyu somuruyor mu (winner's curse) -> dogrudan olculur.
3) Null yonlerden public oran f tahmini.
4) Nihai aday + guven araligi.

KAGGLE'A HICBIR SEY GONDERILMEZ.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

import g2_coz as C  # noqa: E402

BURA = Path(__file__).resolve().parent
X, A, aa, M, SKOR, ADLAR = C.X, C.A, C.aa, C.M, C.SKOR, C.ADLAR
K, N = C.K, C.N
TUM = list(range(K))
HEDEF_MSE = 1.00041**2
BIZ_MSE = 1.00284**2


def l1_coz_genel(As, as_, ms, tau, negsiz=False):
    """min w'As w - w'as + w'ms  s.t. 1'w=1, |w|_1<=tau."""
    from scipy.optimize import minimize

    k = len(ms)
    P = np.hstack([np.eye(k), -np.eye(k)])
    AsP = As @ P

    def f(z):
        w = P @ z
        return float(w @ As @ w - w @ as_ + w @ ms)

    def fp(z):
        w = P @ z
        return P.T @ (2 * As @ w - as_ + ms)

    cons = [
        {
            "type": "eq",
            "fun": lambda z: float(np.sum(P @ z) - 1.0),
            "jac": lambda z: P.T @ np.ones(k),
        },
        {"type": "ineq", "fun": lambda z: tau - z.sum(), "jac": lambda z: -np.ones(2 * k)},
    ]
    z0 = np.zeros(2 * k)
    z0[int(np.argmin(ms))] = 1.0
    bnd = ([(0, None)] * k + [(0, 0)] * k) if negsiz else [(0, None)] * (2 * k)
    r = minimize(
        f,
        z0,
        jac=fp,
        bounds=bnd,
        constraints=cons,
        method="SLSQP",
        options={"maxiter": 600, "ftol": 1e-15},
    )
    return P @ r.x, r


def main() -> None:
    R = {}
    print("=" * 96)
    print("g3 -- KARAR SINAVI")
    print("=" * 96)
    print("BIZ  = %.6f MSE (RMSLE %.5f)" % (BIZ_MSE, np.sqrt(BIZ_MSE)))
    print(
        "HEDEF= %.6f MSE (RMSLE %.5f)   gereken dMSE = %+.6f"
        % (HEDEF_MSE, np.sqrt(HEDEF_MSE), HEDEF_MSE - BIZ_MSE)
    )

    # ---------- 1) ince tau taramasi ----------
    print("")
    print("--- 1) L1 TARAMASI: ongoru <-> |w|_1 <-> belirsizlik ---")
    print(
        "%6s %11s %9s %10s %10s %10s %10s"
        % ("tau", "ongoru_MSE", "RMSLE", "sd_yuv", "sd_pub.5", "sd_pub.3", "sd_pub.2")
    )
    tau_tab = []
    for tau in (1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0):
        w, r = l1_coz_genel(A, aa, M, tau)
        d = C.rapor_w(w)
        d["tau"] = tau
        d["basarili"] = bool(r.success)
        tau_tab.append(d)
        print(
            "%6.1f %11.6f %9.5f %10.2e %10.2e %10.2e %10.2e"
            % (
                tau,
                d["mse"],
                d["rmsle"],
                d["sd_yuvarlama"],
                d["sd_public"]["f=0.5"],
                d["sd_public"]["f=0.3"],
                d["sd_public"]["f=0.2"],
            )
        )
    R["tau_taramasi"] = tau_tab
    ulasan = [d for d in tau_tab if d["mse"] <= HEDEF_MSE]
    R["hedefe_ulasan_en_kucuk_tau"] = ulasan[0]["tau"] if ulasan else None
    print(
        "ongoru HEDEF'e (%.6f) ulasan en kucuk tau: %s"
        % (HEDEF_MSE, ulasan[0]["tau"] if ulasan else "YOK (tau=50'ye kadar)")
    )

    # ---------- 3) public oran f tahmini ----------
    print("")
    print("--- 3) NULL YONLERDEN PUBLIC ORAN f TAHMINI ---")
    G, b, m0, dad = C.fark_gram(TUM)
    lamv, U = np.linalg.eigh(G)
    sira = np.argsort(lamv)[::-1]
    lamv, U = lamv[sira], U[:, sira]
    esik = lamv[0] * 1e-9
    nulls = [j for j in range(len(lamv)) if lamv[j] < esik]
    kayit = []
    for j in nulls:
        v = U[:, j]
        u = np.zeros(K)
        u[np.array(dad)] = v
        u[0] = -v.sum()
        o = float(u @ (M - aa))
        sd_r = float(np.sqrt(np.sum((u * 2 * SKOR * C.DELTA_S) ** 2)))
        h = u @ (X * X)
        sh = float(h.std(ddof=1))
        artik = o**2 - sd_r**2
        # artik = sh^2 (1-f)/(f N)  ->  f
        if artik > 0:
            q = artik / (sh**2 * N)
            f_hat = 1.0 / (1.0 + q)
        else:
            f_hat = 1.0
        kayit.append(
            {
                "j": j,
                "ozdeger": float(lamv[j]),
                "ihlal": abs(o),
                "sd_yuvarlama": sd_r,
                "std_h": sh,
                "f_tahmin": float(f_hat),
            }
        )
        print(
            "  j=%2d lam=%9.2e |ihlal|=%9.3e sd_yuv=%8.2e -> f_tahmin=%.3f"
            % (j, lamv[j], abs(o), sd_r, f_hat)
        )
    R["public_f"] = {
        "null_yonler": kayit,
        "f_medyan": float(np.median([k_["f_tahmin"] for k_ in kayit])) if kayit else None,
    }
    f_med = R["public_f"]["f_medyan"]
    print("  -> f (medyan tahmin) = %s" % (("%.3f" % f_med) if f_med else "-"))
    print("  NOT: 4 yonle tahmin kaba. Asagida f=0.2/0.3/0.5 ayri ayri denenir.")

    # ---------- 2) oniyimserlik simulasyonu ----------
    print("")
    print("--- 2) ONIYIMSERLIK SIMULASYONU (alt-kume Gram ile coz, TAM Gram ile olc) ---")
    print("Her tekrar: f oranli rasgele maske -> A_maske,a_maske + yuvarlanmis m ile COZ")
    print("            -> w_sim; ONGORU = maskeli hedef, GERCEK = tam A ile F(w_sim).")
    sim = {}
    rng = np.random.default_rng(2029)
    TAUS = (1.5, 2.0, 3.0, 5.0, 8.0, 15.0)
    for f in (0.2, 0.3, 0.5):
        nrep = 60
        kayitlar = {t: {"ong": [], "ger": []} for t in TAUS}
        t0 = time.time()
        for rep in range(nrep):
            msk = rng.random(N) < f
            Xs = X[:, msk]
            ns = Xs.shape[1]
            As = (Xs @ Xs.T) / ns
            as_ = np.diag(As).copy()
            ms = (SKOR + rng.uniform(-5e-6, 5e-6, K)) ** 2
            for t in TAUS:
                w, _ = l1_coz_genel(As, as_, ms, t)
                kayitlar[t]["ong"].append(float(w @ As @ w - w @ as_ + w @ ms))
                kayitlar[t]["ger"].append(C.F(w))
        sure = time.time() - t0
        sim["f=%g" % f] = {}
        print("")
        print("  f=%g  (%d tekrar, %.0fs)" % (f, nrep, sure))
        print(
            "  %6s %12s %12s %12s %12s %12s"
            % ("tau", "ongoru_ort", "gercek_ort", "ONIYIMSERLIK", "gercek_sd", "gercek_p95")
        )
        for t in TAUS:
            o = np.array(kayitlar[t]["ong"])
            g = np.array(kayitlar[t]["ger"])
            d = {
                "ongoru_ort": float(o.mean()),
                "ongoru_sd": float(o.std(ddof=1)),
                "gercek_ort": float(g.mean()),
                "gercek_sd": float(g.std(ddof=1)),
                "oniyimserlik_ort": float((g - o).mean()),
                "oniyimserlik_sd": float((g - o).std(ddof=1)),
                "gercek_p95": float(np.percentile(g, 95)),
                "gercek_p05": float(np.percentile(g, 5)),
            }
            sim["f=%g" % f]["tau=%g" % t] = d
            print(
                "  %6.1f %12.6f %12.6f %+12.6f %12.3e %12.6f"
                % (
                    t,
                    d["ongoru_ort"],
                    d["gercek_ort"],
                    d["oniyimserlik_ort"],
                    d["gercek_sd"],
                    d["gercek_p95"],
                )
            )
    R["oniyimserlik"] = sim

    # ---------- 4) nihai aday ----------
    print("")
    print("--- 4) NIHAI ADAYLAR: duzeltilmis beklenti ---")
    print("beklenen_gercek = ongoru + oniyimserlik(f);  ceza = ilgili f'teki ortalama sapma")
    print(
        "%6s %11s %11s %11s %11s %11s"
        % ("tau", "ongoru", "f=0.5 bek.", "f=0.3 bek.", "f=0.2 bek.", "RMSLE(f=.3)")
    )
    nihai = []
    for d in tau_tab:
        t = d["tau"]
        if "tau=%g" % t not in sim["f=0.3"]:
            continue
        sat = {"tau": t, "ongoru_mse": d["mse"], "l1": d["l1"]}
        for f in (0.5, 0.3, 0.2):
            ceza = sim["f=%g" % f]["tau=%g" % t]["oniyimserlik_ort"]
            sd = sim["f=%g" % f]["tau=%g" % t]["gercek_sd"]
            sat["f=%g" % f] = {
                "ceza": ceza,
                "beklenen_mse": d["mse"] + ceza,
                "sd": sd,
                "beklenen_rmsle": float(np.sqrt(max(d["mse"] + ceza, 0))),
            }
        nihai.append(sat)
        print(
            "%6.1f %11.6f %11.6f %11.6f %11.6f %11.5f"
            % (
                t,
                d["mse"],
                sat["f=0.5"]["beklenen_mse"],
                sat["f=0.3"]["beklenen_mse"],
                sat["f=0.2"]["beklenen_mse"],
                sat["f=0.3"]["beklenen_rmsle"],
            )
        )
    R["nihai"] = nihai
    R["hedef_mse"] = HEDEF_MSE
    R["biz_mse"] = BIZ_MSE

    # en iyi aday agirliklari
    en = min(nihai, key=lambda s: s["f=0.3"]["beklenen_mse"])
    wbest, _ = l1_coz_genel(A, aa, M, en["tau"])
    R["en_iyi_aday"] = {"tau": en["tau"], "rapor": C.rapor_w(wbest), "beklenen": en}
    print("")
    print("EN IYI ADAY (f=0.3 duzeltmesine gore): tau=%g" % en["tau"])
    for k_, v in C.rapor_w(wbest)["w"].items():
        print("   %-22s %+9.5f" % (k_, v))

    (BURA / "g3_sinav.json").write_text(
        json.dumps(R, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    # g1_gram.json'a ekle
    p = BURA / "g1_gram.json"
    G1 = json.loads(p.read_text(encoding="utf-8"))
    G1["karar_sinavi"] = R
    p.write_text(json.dumps(G1, indent=1, ensure_ascii=False), encoding="utf-8")
    print("")
    print("yazildi: %s ve %s" % (BURA / "g3_sinav.json", p))


if __name__ == "__main__":
    main()
