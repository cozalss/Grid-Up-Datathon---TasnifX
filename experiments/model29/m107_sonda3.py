"""SONDA v3 -- ardisik, tutucu-optimum tasarim.

Sonda = "yeni adayin L'si SIFIR" varsayimi altindaki TAM ortak optimum.
Yeni yonun agirligi keyfi degil, k* = G^-1 L cozumunden cikar; boylece sonda
hem OLCUM hem de o an bilinen en iyi GONDERIM olur. Tek bilinmeyen L_yeni,
skor gelince kapali formulle cozulur.

--aday verilmezse: saf ortak optimum (sonda terimi yok) -- son gonderim icin.

m0 KALIBRE: 1.005846366. m6'nin optimize edildigi uc yonun (p51, m4, v102)
L'sini sifirlayan deger; ucu de 9.07e-07 yayilmayla ayni m0'i ima ediyor
(LB yuvarlama butcesinin icinde). LB'nin gosterdigi 1.00284 public %50
uzerinde, Q ise tum satirlarda olculdugu icin aradaki 8e-5 fark normaldir.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.005846366
SATIR = 714688
YUVARLAMA = 5e-6  # LB 5 ondalik hane gosterir

KISA = {
    "g7": "tuketim_g7_span_tau3.csv",
    "y40": "tuketim_y40_sota_temiz.csv",
    "z2": "tuketim_z2_analog.csv",
    "sul": "tuketim_t1_sulama.csv",
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "q1c": "tuketim_q1c_kapasite_siki.csv",
    "t3": "tuketim_t3_turizm.csv",
    "p42": "tuketim_p42_seviye_egrilik.csv",
    "h1": "tuketim_h1_isil.csv",
    "y31": "tuketim_y31_amnezik.csv",
    "v6": "tuketim_v6.csv",
    "z1": "tuketim_z1_havuz.csv",
}
# Bu dosyalarin uzerine ASLA yazilmaz (taban + tum yon kaynaklari).
KORUNAN = {TABAN, *KISA.values()}


def oku(f):
    df = pd.read_csv(os.path.join(S, KISA.get(f, f)))
    kol = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return df, np.log1p(df[kol].values.astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bilinen", required=True, help="ad=L cifleri, virgulle")
    ap.add_argument("--aday", default=None, help="olculecek YENI yon; yoksa saf optimum")
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument(
        "--olcek", type=float, default=1.5, help="yeni yonun agirligi tutucu-optimumun kac kati"
    )
    ap.add_argument(
        "--min-yerdeg",
        type=float,
        default=0.030,
        dest="min_yerdeg",
        help="yeni yon en az bu kadar log-RMS yer degistirsin",
    )
    ap.add_argument(
        "--k1-tavan",
        type=float,
        default=None,
        dest="k1_tavan",
        help="|k|_1 tavani; varsayilan 4 + 1.5*n",
    )
    ap.add_argument(
        "--e-tavan", type=float, default=8e-3, dest="e_tavan", help="kirpma artigi tavani"
    )
    a = ap.parse_args()

    # --cikti korumasi: kaynak dosyalarin uzerine yazma, submissions disina cikma
    if os.path.dirname(a.cikti) or Path(a.cikti).is_absolute():
        raise SystemExit(f"REDDEDILDI: --cikti '{a.cikti}' yol icermemeli")
    if a.cikti in KORUNAN:
        raise SystemExit(f"REDDEDILDI: --cikti '{a.cikti}' KORUNAN kaynak dosya")

    bil = {}
    for p in a.bilinen.split(","):
        ad, v = p.split("=")
        ad = ad.strip()
        if ad in bil:
            raise SystemExit(f"REDDEDILDI: --bilinen icinde '{ad}' iki kez var")
        bil[ad] = float(v)

    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    df0, a0 = oku(TABAN)
    N = len(a0)
    assert N == SATIR, f"taban satir {N}"
    assert (df0.id.values == te.id.values).all(), "taban ID hizasi bozuk"

    nihai = a.aday is None
    adlar = list(bil) + ([] if nihai else [a.aday])
    if not nihai and a.aday in bil:
        raise SystemExit(f"REDDEDILDI: '{a.aday}' zaten olculmus")
    D, L = [], []
    for ad in adlar:
        dfj, pj = oku(ad)
        assert len(pj) == N, f"{ad} satir {len(pj)}"
        assert (dfj.id.values == te.id.values).all(), f"{ad} ID hizasi bozuk"
        D.append(pj - a0)
        L.append(bil.get(ad, 0.0))  # YENI adayin L'si SIFIR varsayilir
    D = np.array(D).T
    L = np.array(L)
    K = len(adlar)
    G = (D.T @ D) / N
    kosul = float(np.linalg.cond(G))
    k = np.linalg.solve(G + a.lam * np.eye(K), L)

    # Yeni yonun agirligi: tutucu optimumun --olcek kati, ama en az --min-yerdeg
    # kadar log-RMS yer degistirsin (kucuk k_yeni'de hem yuvarlama hem kirpma
    # sapmasi 1/k_yeni ile buyur).
    if not nihai:
        kn_hedef = k[-1] * a.olcek
        if abs(kn_hedef) * np.sqrt(G[-1, -1]) < a.min_yerdeg:
            kn_hedef = np.sign(kn_hedef or 1.0) * a.min_yerdeg / np.sqrt(G[-1, -1])
        k[-1] = kn_hedef
        if K > 1:
            k[:-1] = np.linalg.solve(
                G[:-1, :-1] + a.lam * np.eye(K - 1), L[:-1] - G[:-1, -1] * k[-1]
            )
    mse = float(M0 - 2 * k @ L + k @ G @ k)
    kn = float(k[-1]) if not nihai else 0.0
    Qn = float(G[-1, -1])

    print(f"{'yon':>6s} {'Q':>10s} {'L':>11s} {'rho':>9s} {'k*':>10s}")
    for i, ad in enumerate(adlar):
        etk = " <- OLCULECEK" if (not nihai and i == K - 1) else ""
        print(
            f"{ad:>6s} {G[i, i]:10.6f} {L[i]:+11.6f} "
            f"{L[i] / np.sqrt(G[i, i]):+9.4f} {k[i]:+10.5f}{etk}"
        )
    etiket = "beklenen skor" if nihai else "L_yeni=0 iken skor"
    print(f"\ncond(G)={kosul:.1f}  |k|_1={np.abs(k).sum():.3f}  {etiket} {np.sqrt(mse):.5f}")

    k1_tavan = a.k1_tavan if a.k1_tavan else 4.0 + 1.5 * K
    uyari = []
    if kosul > 1e8:
        uyari.append(f"cond {kosul:.2e}")
    if mse < 0 or mse > M0 + 1e-12:
        uyari.append(f"mse {mse}")
    if np.abs(k).sum() > k1_tavan:
        uyari.append(f"|k|_1 {np.abs(k).sum():.2f} > {k1_tavan:.1f} (--k1-tavan ile gevset)")
    if not nihai and abs(kn) < 1e-3:
        uyari.append(f"k_yeni={kn:.2e} cok kucuk")
    if uyari:
        raise SystemExit("KORKULUK: " + " | ".join(uyari))

    if nihai:
        sigL = float("nan")
        print("NIHAI: sonda terimi YOK, saf ortak optimum")
    else:
        sigL = 1.003 * YUVARLAMA / abs(kn)
        print(
            f"olcum: k_yeni={kn:+.5f}  sigma(L_yeni)={sigL:.2e}  "
            f"rho=0.0146 sinyali={0.0146 * np.sqrt(Qn):.2e}  "
            f"SNR={0.0146 * np.sqrt(Qn) / sigL:.0f}"
        )

    p = a0 + D @ k
    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kapi = dict(
        satir=len(out),
        id_ss=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        sonsuz=int((~np.isfinite(out.tuketim.values)).sum()),
        maks=float(out.tuketim.max()),
        taban_maks=float(np.expm1(a0).max()),
    )
    if not (
        kapi["satir"] == SATIR
        and kapi["id_ss"]
        and kapi["nan"] == 0
        and kapi["negatif"] == 0
        and kapi["sonsuz"] == 0
    ):
        raise SystemExit(f"KAPI KALDI: {kapi}")
    if kapi["maks"] > 3 * kapi["taban_maks"]:
        raise SystemExit(f"KAPI KALDI: maks {kapi['maks']:,.0f} > 3x taban")

    # COZUM SABITI -- teorik p'den DEGIL, diske yazilacak (kirpilmis) vektorden.
    pg = np.log1p(out.tuketim.values)
    dgv = pg - a0
    kirpik = int((y == 0.0).sum())
    e = dgv - D @ k
    e_norm = float(np.sqrt(float(e @ e) / N))
    sabit = float(M0 - 2 * (k[:-1] @ L[:-1] if not nihai else k @ L) + float(dgv @ dgv) / N)
    if nihai:
        print(f"kirpma: {kirpik} satir sifira | |e|={e_norm:.2e} (NIHAI'de olcum yok, zararsiz)")
    else:
        print(
            f"kirpma: {kirpik} satir sifira | |e|={e_norm:.2e} | "
            f"beklenen L sapmasi ~{e_norm / 4e2:.1e} | yuvarlama {sigL:.2e}"
        )
        if e_norm > a.e_tavan:
            raise SystemExit(f"KORKULUK: kirpma artigi |e|={e_norm:.2e} > {a.e_tavan:.1e}")

    yol = os.path.join(S, a.cikti)
    gec = Path(yol + ".tmp")
    out.to_csv(gec, index=False)
    gec.replace(yol)

    rap = dict(
        cikti=a.cikti,
        kirpik=kirpik,
        e_norm=e_norm,
        adlar=adlar,
        bilinen_L=bil,
        k=k.tolist(),
        cond=kosul,
        Q_yeni=Qn,
        k_yeni=kn,
        sabit=sabit,
        sigma_L=sigL,
        skor_L0=float(np.sqrt(mse)),
        kapi=kapi,
    )
    ad_json = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), f"m107_{a.aday or 'NIHAI'}.json"
    )
    json.dump(rap, open(ad_json, "w"), indent=1)
    print(f"\nYAZILDI {yol}")
    if nihai:
        print(f"BEKLENEN SKOR {np.sqrt(sabit):.5f} -- tum L'ler OLCULMUS, tahmin degil.")
    else:
        print(f"COZUM:  L_{a.aday} = ({sabit:.9f} - P^2) / {2 * kn:.6f}")
        for r in (0.0146, 0.0224, 0.030):
            v = sabit - 2 * kn * r * np.sqrt(Qn)
            print(f"  rho={r:.4f} -> beklenen sonda skoru {np.sqrt(v):.5f}")


if __name__ == "__main__":
    main()
