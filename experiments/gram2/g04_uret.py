"""G4 -- v103 (yeni Gram optimumu) + muhafazakar varyantlar + gonderilmemis aday taramasi.

HICBIR GONDERIM YAPILMAZ. Yalnizca submissions/ altina dosya yazar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from g02_coz import DELTA_S  # noqa: E402
from g03_sinav import coz_kesik, kur  # noqa: E402

KOK = Path(__file__).resolve().parents[2]
GON = KOK / "submissions"
CIK = Path(__file__).resolve().parent


def mse_of(H, w):
    return H["m0"] + float(w @ H["G"] @ w) - 2 * float(w @ H["b"])


def yaz(H, w, dosya, etiket):
    x = H["X"][H["i0"]] + w @ H["D"]
    kirp = int((x < 0).sum())
    x = np.clip(x, 0.0, None)
    tuketim = np.expm1(x)
    yol = GON / dosya
    pd.DataFrame({"id": H["ids"], "tuketim": tuketim}).to_csv(yol, index=False)
    # kirpma sonrasi gercek d ve Q
    d = x - H["X"][H["i0"]]
    Q0 = float(d @ d / H["n"])
    i102 = H["yon"].index("v102")
    d2 = d - H["D"][i102]
    Q2 = float(d2 @ d2 / H["n"])
    return dict(
        dosya=dosya,
        etiket=etiket,
        kirpilan=kirp,
        Q_v83=Q0,
        Q_v102=Q2,
        w_l1=float(np.abs(w).sum()),
        w_max=float(np.abs(w).max()),
    )


def main() -> None:
    H = kur("v83")
    yon, G, b, m0 = H["yon"], H["G"], H["b"], H["m0"]
    i101, i102 = yon.index("v101"), yon.index("v102")
    n = H["n"]
    print("=" * 96)
    print("6) YENI OPTIMUM VE MUHAFAZAKAR VARYANTLAR")
    print("=" * 96)

    # dogrulama: v102'nin kendisi
    w102 = np.zeros(len(yon))
    w102[i102] = 1.0
    w102b = np.zeros(len(yon))
    w102b[i101] = 0.459022
    print(
        f"tutarlilik: MSE(v102 dosyasi)={mse_of(H, w102):.6f} (olculen 1.011091), "
        f"MSE(v83+0.459022*d101)={mse_of(H, w102b):.6f}"
    )
    print(f"           v102 skoru       = {np.sqrt(mse_of(H, w102)):.6f}  (Kaggle 1.00553)")

    kayit = {"m0": m0, "v102_mse": 1.00553**2, "varyantlar": []}

    # --- ana optimum: r=17 (tum gercek ozyonler) ---
    w17, kz17, r17, lam, U, beta = coz_kesik(G, b, r=17)
    # --- muhafazakar: rank kesmeleri ---
    adaylar = []
    for r in [3, 6, 8, 12, 17]:
        w, kz, rr, *_ = coz_kesik(G, b, r=r)
        adaylar.append((f"r{r}", w, f"rank-{r} kesik Gram optimumu"))
    # --- muhafazakar: v102'ye dogru buzme ---
    for kap in [0.6, 0.8]:
        w = w102 + kap * (w17 - w102)
        adaylar.append((f"k{kap:g}_v102", w, f"v102 + {kap}*(v103-v102)"))
    # --- referans: v83'e dogru buzme (istenen bicim) ---
    for kap in [0.6, 0.8]:
        w = kap * w17
        adaylar.append((f"k{kap:g}_v83", w, f"v83 + {kap}*(v103-v83)"))
    # --- ridge ---
    for al in [1e-5, 1e-4]:
        w, kz, rr, *_ = coz_kesik(G, b, r=17, ridge=al)
        adaylar.append((f"ridge{al:g}", w, f"ridge alpha={al:g}"))

    print("\naday        MSE        RMSLE      d(RMSLE) vs 1.00553   |w|1    Q(v83)   Q(v102)")
    for ad, w, aciklama in adaylar:
        mm = mse_of(H, w)
        s = np.sqrt(mm)
        Q0 = float(w @ G @ w)
        Q2 = Q0 - 2 * float(w @ G[:, i102]) + float(G[i102, i102])
        print(
            f"{ad:12s} {mm:.6f}  {s:.6f}   {s - 1.00553:+.6f}          "
            f"{np.abs(w).sum():6.3f}  {Q0:.5f}  {Q2:.5f}"
        )
        kayit["varyantlar"].append(
            dict(
                ad=ad,
                aciklama=aciklama,
                mse=mm,
                rmsle=s,
                w_l1=float(np.abs(w).sum()),
                Q_v83=Q0,
                Q_v102=Q2,
            )
        )

    # ---- gurultu bandi (Monte Carlo: b'nin yuvarlama belirsizligi) ----
    print("\nBELIRSIZLIK BANDI (5-ondalik yuvarlamadan, 4000 Monte Carlo)")
    rng = np.random.default_rng(7)
    sd_s = DELTA_S
    s_all = H["skorlar"]
    i0 = H["i0"]
    idx = [H["adlar"].index(a) for a in yon]
    gercek = {}
    for etiket, r in [("r6", 6), ("r8", 8), ("r12", 12), ("r17", 17)]:
        sonuc = []
        for _ in range(4000):
            e0 = rng.uniform(-5e-6, 5e-6)
            ei = rng.uniform(-5e-6, 5e-6, size=len(idx))
            m0p = (s_all[i0] + e0) ** 2
            mp = (s_all[idx] + ei) ** 2
            bp = (m0p + np.diag(G) - mp) / 2.0
            wp, _kz, _rr, *_ = coz_kesik(G, bp, r=r)
            # gercek b ile degerlendir (w'nin gurultuye duyarliligi)
            sonuc.append(m0 + float(wp @ G @ wp) - 2 * float(wp @ b))
        sonuc = np.array(sonuc)
        s5, s95 = np.percentile(np.sqrt(sonuc), [5, 95])
        gercek[etiket] = (float(s5), float(s95))
        print(
            f"  {etiket}: RMSLE %5-%95 = [{s5:.6f}, {s95:.6f}]  (medyan {np.median(np.sqrt(sonuc)):.6f})"
        )
    kayit["mc_band"] = gercek

    # ---- dosyalari yaz ----
    print("\nDOSYA URETIMI")
    uret = []
    w_r17 = w17
    w_r8 = adaylar[2][1]
    w_k08 = w102 + 0.8 * (w17 - w102)
    for w, dosya, et in [
        (w_r17, "tuketim_v103_gram2.csv", "rank-17 Gram optimumu (ana)"),
        (w_r8, "tuketim_v104_gram2_r8.csv", "rank-8 muhafazakar Gram"),
        (w_k08, "tuketim_v105_gram2_k08.csv", "v102 + 0.8*(v103-v102)"),
    ]:
        bilgi = yaz(H, w, dosya, et)
        bilgi["beklenen_mse"] = mse_of(H, w)
        bilgi["beklenen_rmsle"] = float(np.sqrt(bilgi["beklenen_mse"]))
        uret.append(bilgi)
        print(
            f"  {dosya:34s} kirpilan={bilgi['kirpilan']:6d}  |w|1={bilgi['w_l1']:.3f}  "
            f"Q(v83)={bilgi['Q_v83']:.5f}  Q(v102)={bilgi['Q_v102']:.5f}  "
            f"beklenen={bilgi['beklenen_rmsle']:.6f}"
        )
    kayit["uretilen"] = uret

    # ---- agirliklar ----
    print("\nv103 agirliklari (v83 tabanina gore, d_i katsayilari):")
    for a, wv in sorted(zip(yon, w_r17), key=lambda t: -abs(t[1])):
        if abs(wv) > 1e-4:
            print(f"   {a:>5}: {wv:+.5f}")
    kayit["w_v103"] = {a: float(v) for a, v in zip(yon, w_r17)}
    kayit["w_v104_r8"] = {a: float(v) for a, v in zip(yon, w_r8)}

    # ---- konveks govde ----
    kats = np.array(w_r17)
    taban_kat = 1.0 - kats.sum()
    print(
        f"\nkonveks govde: taban(v83) katsayisi={taban_kat:+.5f}, "
        f"toplam |kat|={np.abs(kats).sum() + abs(taban_kat):.4f}  "
        f"(1.0 = govde ici; negatif katsayi sayisi={int((kats < -1e-4).sum())})"
    )
    kayit["konveks"] = dict(
        taban_kat=float(taban_kat),
        l1_toplam=float(np.abs(kats).sum() + abs(taban_kat)),
        negatif_sayisi=int((kats < -1e-4).sum()),
    )

    (CIK / "g04_rapor.json").write_text(json.dumps(kayit, indent=2), encoding="utf-8")
    print("\nyazildi: g04_rapor.json")


if __name__ == "__main__":
    main()
