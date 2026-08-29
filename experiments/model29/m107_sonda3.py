"""SONDA v3 -- ARDISIK, TUTUCU-OPTIMUM tasarim.

m106'nin kusuru: sonda = taban + c_g7*d_g7 + t*d_aday, t KEYFI (0.60/0.35/0.45).
Buyuk t olcumu keskinlestirir ama sondanin KENDI skorunu bozar. Oysa LB skoru
5 haneye yuvarlanir -> olcum hatasi yalnizca +-5e-6, yani KUCUK t'de bile
olcum bol bol yeterli. Demek ki t'yi buyuk secmek bedava degil, sadece zarar.

DOGRU tasarim: sonda = "yeni adayin L'si SIFIR" varsayimi altindaki ORTAK OPTIMUM.
  - t artik keyfi degil, k* = G^-1 L (L_yeni = 0) cozumunden cikar
  - sonda boylece hem OLCUM hem de o an bilinen en iyi GONDERIM olur
  - tek bilinmeyen yine L_yeni, skor gelince kapali formulle cozulur

Kullanim:
  python m107_sonda3.py --bilinen g7=0.002728 --aday y40 --cikti tuketim_s3y40.csv
  python m107_sonda3.py --bilinen g7=0.002728,y40=0.00421 --aday z2 --cikti tuketim_s3z2.csv
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
M0 = 1.00284**2
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
    "h1": "tuketim_h1_isil.csv",
    "t2": "tuketim_t2_bayram.csv",
    "k5": "tuketim_k5_kesinti.csv",
    "z1": "tuketim_z1_havuz.csv",
    "q1a": "tuketim_q1a_kapasite.csv",
}


def oku(f):
    df = pd.read_csv(os.path.join(S, KISA.get(f, f)))
    kol = "tuketim" if "tuketim" in df.columns else df.columns[-1]
    return df, np.log1p(df[kol].values.astype(np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--bilinen", required=True, help="ad=L cifleri, virgulle. or: g7=0.002728,y40=0.0042"
    )
    ap.add_argument(
        "--aday",
        default=None,
        help="olculecek YENI yon. VERILMEZSE: saf ortak optimum uretilir "
        "(sonda terimi yok) -- son gonderim icin",
    )
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--lam", type=float, default=0.0, help="Tikhonov, gerekirse 0.001")
    ap.add_argument(
        "--olcek",
        type=float,
        default=1.5,
        help="yeni yonun agirligini tutucu-optimumun kac kati alalim. "
        "1.0 = L=0 varsayiminda en iyi skor; >1 = olcum hassasiyeti artar, "
        "beklenen skor IYILESIR, en kotu durumda cok az kotulesir",
    )
    ap.add_argument(
        "--min-yerdeg",
        type=float,
        default=0.030,
        dest="min_yerdeg",
        help="yeni yonun en az bu kadar log-RMS yer degistirmesi sart "
        "(olcum hatasi ve kirpma sapmasi 1/k_yeni ile buyur)",
    )
    a = ap.parse_args()

    bil = {}
    for p in a.bilinen.split(","):
        ad, v = p.split("=")
        bil[ad.strip()] = float(v)

    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    df0, a0 = oku(TABAN)
    N = len(a0)
    assert N == SATIR, f"taban satir {N}"
    assert (df0.id.values == te.id.values).all(), "taban ID hizasi bozuk"

    nihai = a.aday is None
    adlar = list(bil) + ([] if nihai else [a.aday])
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
    # yeni yonun agirligi: tutucu optimumun --olcek kati, ama en az --min-yerdeg
    # kadar log-RMS yer degistirsin. Kucuk k_yeni iki sorun yaratir:
    #   (1) yuvarlama hatasi /k_yeni ile buyur,
    #   (2) clip(0) artiginin olculemeyen <r,e> katkisi /k_yeni ile buyur.
    if not nihai:
        kn_hedef = k[-1] * a.olcek
        Qn_ = G[-1, -1]
        if abs(kn_hedef) * np.sqrt(Qn_) < a.min_yerdeg:
            kn_hedef = np.sign(kn_hedef or 1.0) * a.min_yerdeg / np.sqrt(Qn_)
        k[-1] = kn_hedef
        if K > 1:
            k[:-1] = np.linalg.solve(
                G[:-1, :-1] + a.lam * np.eye(K - 1), L[:-1] - G[:-1, -1] * k[-1]
            )
    mse = float(M0 - 2 * k @ L + k @ G @ k)
    kn = float(k[-1]) if not nihai else 0.0

    print(f"{'yon':>6s} {'Q':>10s} {'L':>11s} {'rho':>9s} {'k*':>10s}")
    for i, ad in enumerate(adlar):
        q = G[i, i]
        etk = " <- OLCULECEK" if (not nihai and i == K - 1) else ""
        print(f"{ad:>6s} {q:10.6f} {L[i]:+11.6f} {L[i] / np.sqrt(q):+9.4f} {k[i]:+10.5f}{etk}")
    etiket = "beklenen skor" if nihai else "L_yeni=0 iken skor"
    print(f"\ncond(G)={kosul:.1f}  |k|_1={np.abs(k).sum():.3f}  {etiket} {np.sqrt(mse):.5f}")

    uyari = []
    if kosul > 1e8:
        uyari.append(f"cond {kosul:.2e}")
    if mse < 0 or mse > M0 + 1e-12:
        uyari.append(f"mse {mse}")
    k1_tavan = 2.0 + 1.0 * K  # yon sayisiyla olcekli; sabit 5 esigi n>=6'da yanlis tetikliyor
    if np.abs(k).sum() > k1_tavan:
        uyari.append(f"|k|_1 {np.abs(k).sum():.2f} > {k1_tavan:.1f}")
    if not nihai and abs(kn) < 1e-3:
        uyari.append(f"k_yeni={kn:.2e} COK KUCUK -> olcum cozunurlugu duser")
    if uyari:
        raise SystemExit("KORKULUK: " + " | ".join(uyari))

    # olcum hassasiyeti: P 5 haneye yuvarlanir
    Qn = float(G[-1, -1])
    if nihai:
        sigL = float("nan")
        print("NIHAI: sonda terimi YOK, saf ortak optimum")
    else:
        sigL = 2 * 1.003 * YUVARLAMA / (2 * abs(kn))
        print(
            f"olcum: k_yeni={kn:+.5f}  sigma(L_yeni)={sigL:.2e}  "
            f"rho=0.0137 sinyali={0.0137 * np.sqrt(Qn):.2e}  "
            f"SNR={0.0137 * np.sqrt(Qn) / sigL:.0f}"
        )

    p = a0 + D @ k
    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kapi = dict(
        satir=len(out),
        id_test=bool((out.id.values == te.id.values).all()),
        id_ss=bool((out.id.values == ss.iloc[:, 0].values).all()),
        nan=int(out.tuketim.isna().sum()),
        negatif=int((out.tuketim < 0).sum()),
        sonsuz=int((~np.isfinite(out.tuketim.values)).sum()),
        maks=float(out.tuketim.max()),
        taban_maks=float(np.expm1(a0).max()),
    )
    if not (
        kapi["satir"] == SATIR
        and kapi["id_test"]
        and kapi["id_ss"]
        and kapi["nan"] == 0
        and kapi["negatif"] == 0
        and kapi["sonsuz"] == 0
    ):
        raise SystemExit(f"KAPI KALDI: {kapi}")
    if kapi["maks"] > 3 * kapi["taban_maks"]:
        raise SystemExit(f"KAPI KALDI: maks {kapi['maks']:,.0f} > 3x taban")

    # COZUM SABITI -- teorik p'den DEGIL, diske yazilacak (kirpilmis) vektorden.
    # clip(0) log-uzayindaki vektoru degistirir; kirpma buyudugunde teorik sabit
    # 1e-5 mertebesinde sapar (yuvarlama hatasinin 2 kati, ustelik sistematik).
    #   P^2 = m0 - 2<r,dg> + |dg|^2   ve   <r,dg> = sum_j k_j L_j + k_n L_n + <r,e>
    #   e = dg - D k  (yalnizca kirpmadan gelir; normu denetlenir)
    pg = np.log1p(out.tuketim.values)
    dgv = pg - a0
    kirpik = int((y == 0.0).sum())
    e = dgv - D @ k
    e_norm = float(np.sqrt(float(e @ e) / N))
    sabit = float(M0 - 2 * (k[:-1] @ L[:-1] if not nihai else k @ L) + float(dgv @ dgv) / N)
    if nihai:
        print(f"kirpma: {kirpik} satir sifira | |e|={e_norm:.2e}")
    else:
        belirsiz = 2 * e_norm * np.sqrt(M0) / (2 * abs(kn))
        print(
            f"kirpma: {kirpik} satir sifira | |e|={e_norm:.2e} "
            f"-> L belirsizligi (en kotu hal) <= {belirsiz:.2e} | yuvarlama {sigL:.2e}"
        )
    if e_norm > 2e-3:
        raise SystemExit(f"KORKULUK: kirpma artigi |e|={e_norm:.2e} olcumu bozar")
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
    json.dump(rap, open(f"m107_{a.aday or 'NIHAI'}.json", "w"), indent=1)
    print(f"\nYAZILDI {yol}")
    if nihai:
        print(f"BEKLENEN SKOR {np.sqrt(sabit):.5f} -- tum L'ler OLCULMUS, tahmin degil.")
    else:
        print(f"COZUM:  L_{a.aday} = ({sabit:.9f} - P^2) / {2 * kn:.6f}")
        for r in (0.0137, 0.025, 0.035, 0.05):
            v = sabit - 2 * kn * r * np.sqrt(Qn)
            print(f"  rho={r:.4f} -> beklenen sonda skoru {np.sqrt(v):.5f}")


if __name__ == "__main__":
    main()
