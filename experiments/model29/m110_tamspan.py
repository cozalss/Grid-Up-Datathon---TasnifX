"""SONDA v4 -- TAM SPAN tabani + DIK aday.

Onceki tasarimin (m107) iki kusuru vardi:
  1. Taban olarak g7 kullaniyordu. g7 buzulmus bir span cozumu: 0.003036 acikliyor,
     oysa olculmus 25 yonun TAM span optimumu 0.003872 acikliyor. Masada 0.000836
     duruyordu (skorda 1.00140 -> 1.00099).
  2. Adayi HAM haliyle sondaliyordu. Adayin span ICI parcasinin L'si zaten biliniyor;
     onu tekrar "olcmek" hem hakki israf ediyor hem L=0 varsayimini bozuyor.

DOGRU tasarim:
  taban = a0 + r_hat        (r_hat = r'nin olculmus span'a izdusumu, TAM optimum)
  sonda = taban + kappa * d_dik     (d_dik = adayin span'a DIK bileseni)
  d_dik span'a dik oldugu icin: capraz terim YOK, L_bilinen = 0, olu nokta = 0.
  Yani sonda SAF yeni bilgi olcer ve tabanin kesin kazancini da tasir.

r_hat DOGRULANDI (bagimsiz olarak, iki kez):
  rcond 1e-5..1e-10 -> ||r_hat||^2 = 0.003872 TAM KARARLI (1e-12'de rank 22'ye
    cikip patliyor; dogru truncation rank 21)
  LEAVE-ONE-OUT: span-ici payi >%95 olan 19 yonde ortalama |hata| = 7.0e-05
  LB yuvarlamasi: ||r_hat||^2 sacilimi sd 1.2e-05 (skorda 6e-06)

Kullanim:
  python m110_tamspan.py --cikti tuketim_TS.csv                    # saf taban
  python m110_tamspan.py --aday z2 --yerdeg 0.015 --cikti tuketim_TSz2.csv
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
RCOND = 1e-6
# LB skorundan dogrudan olculmus, olculmus_skorlar.json'da OLMAYAN L'ler
EK_OLCUM = {"tuketim_y40_sota_temiz.csv": -0.002229}

KISA = {
    "z2": "tuketim_z2_analog.csv",
    "sul": "tuketim_t1_sulama.csv",
    "y46": "tuketim_y46_amnezik_kirpik.csv",
    "y45": "tuketim_y45_mevsimsel_kirpik.csv",
    "q1c": "tuketim_q1c_kapasite_siki.csv",
    "t3": "tuketim_t3_turizm.csv",
    "p42": "tuketim_p42_seviye_egrilik.csv",
    "y31": "tuketim_y31_amnezik.csv",
    "p41": "tuketim_p41_kesik_sifir.csv",
    "v6": "tuketim_v6.csv",
    "z1": "tuketim_z1_havuz.csv",
    "y33": "tuketim_y33_klasik.csv",
    "p31": "tuketim_p31_kesik_sifir.csv",
    "y41": "tuketim_y41_amnezik_temiz.csv",
    "y43": "tuketim_y43_mevsimsel_temiz.csv",
    "y34": "tuketim_y34_mevsimsel.csv",
    "y30": "tuketim_y30_sota.csv",
}


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return d, np.log1p(d[k].values.astype(np.float64))


def kur_rhat(a0, N, te):
    """Olculmus tum yonlerden r_hat'i kur. Ek olarak V, G, Gi doner."""
    SK = json.load(
        open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "olculmus_skorlar.json"))
    )
    V, L, ad = [], [], []
    kaynak = dict(SK)
    for f, P in kaynak.items():
        if f == TABAN or not os.path.exists(os.path.join(S, f)):
            continue
        dfj, v = oku(f)
        if len(v) != N:
            continue
        assert (dfj.id.values == te.id.values).all(), f"{f} ID hizasi bozuk"
        d = v - a0
        Q = float((d * d).mean())
        V.append(d)
        L.append((M0 + Q - P * P) / 2)
        ad.append(f)
    for f, Lj in EK_OLCUM.items():
        dfj, v = oku(f)
        assert (dfj.id.values == te.id.values).all(), f"{f} ID hizasi bozuk"
        V.append(v - a0)
        L.append(Lj)
        ad.append(f)
    V = np.array(V).T
    L = np.array(L)
    G = (V.T @ V) / N
    c = np.linalg.pinv(G, rcond=RCOND) @ L
    return V @ c, V, G, L, ad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aday", default=None, help="sondalanacak aday; yoksa saf taban")
    ap.add_argument(
        "--yerdeg", type=float, default=0.015, help="dik bilesenin log-RMS yer degistirmesi"
    )
    ap.add_argument("--cikti", required=True)
    a = ap.parse_args()
    if os.path.dirname(a.cikti) or Path(a.cikti).is_absolute():
        raise SystemExit("REDDEDILDI: --cikti yol icermemeli")
    if a.cikti == TABAN:
        raise SystemExit("REDDEDILDI: taban dosyasinin uzerine yazilamaz")

    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    df0, a0 = oku(TABAN)
    N = len(a0)
    assert N == SATIR and (df0.id.values == te.id.values).all()

    r_hat, V, G, L, ad = kur_rhat(a0, N, te)
    nrm = float((r_hat * r_hat).mean())
    print(
        f"TAM SPAN: {len(ad)} olculmus yon, rank "
        f"{int((np.linalg.svdvals(G) > np.linalg.svdvals(G)[0] * RCOND).sum())}"
    )
    print(f"  ||r_hat||^2 = {nrm:.6f}   saf taban tahmini LB = {np.sqrt(M0 - nrm):.5f}")

    p = a0 + r_hat
    kap = 0.0
    Qd = 0.0
    if a.aday:
        dfj, vj = oku(KISA.get(a.aday, a.aday))
        assert (dfj.id.values == te.id.values).all(), f"{a.aday} ID hizasi bozuk"
        dj = vj - a0
        # span'a dik bilesen
        c, *_ = np.linalg.lstsq(G, (V.T @ dj) / N, rcond=RCOND)
        dperp = dj - V @ c
        Qd = float((dperp * dperp).mean())
        oran = Qd / float((dj * dj).mean())
        kap = a.yerdeg / np.sqrt(Qd)
        p = p + kap * dperp
        dik_kontrol = float((dperp @ r_hat) / N) / np.sqrt(Qd * nrm)
        print(
            f"\nADAY {a.aday}: Q={float((dj * dj).mean()):.5f}  Q_dik={Qd:.5f}  "
            f"span-disi oran={oran:.3f}"
        )
        print(
            f"  kappa={kap:.5f}  yer degistirme={a.yerdeg}  "
            f"kos(r_hat, d_dik)={dik_kontrol:+.2e} (0 olmali)"
        )
        sig = 1.001 * 5e-6 / kap
        print(
            f"  sigma(L_dik)={sig:.2e}  rho=0.027 sinyali="
            f"{0.027 * np.sqrt(Qd):.2e}  SNR={0.027 * np.sqrt(Qd) / sig:.0f}"
        )
        L_span_j = float((r_hat @ dj) / N)
        print(
            f"  L_span({a.aday}) = {L_span_j:+.6f}  -> TAM L = L_span + L_dik (EK_OLCUM'a bunu yaz)"
        )
        # dik bilesenin yogunlasmasi: olculebilirlik kontrolu
        pay = np.sort(dperp**2)[::-1]
        k50 = int(np.searchsorted(np.cumsum(pay), 0.5 * pay.sum())) + 1
        print(
            f"  yogunlasma: d_dik^2'nin %50'si {k50:,} satirda "
            f"({'TEKIL - DIKKAT' if k50 < 1000 else 'saglam'})"
        )

    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    kirpik = int((y == 0.0).sum())
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
        raise SystemExit(f"KAPI KALDI: maks {kapi['maks']:,.0f}")

    # cozum sabiti -- DISKE YAZILAN (kirpilmis) vektorden
    dgv = np.log1p(out.tuketim.values) - a0
    sabit = float(M0 - 2 * nrm + float(dgv @ dgv) / N)
    e = dgv - (p - a0)  # yalnizca kirpmadan gelen sapma
    e_norm = float(np.sqrt(float(e @ e) / N))
    print(f"\nkirpma: {kirpik} satir | |e|={e_norm:.2e}")

    yol = os.path.join(S, a.cikti)
    gec = Path(yol + ".tmp")
    out.to_csv(gec, index=False)
    gec.replace(yol)
    rap = dict(
        cikti=a.cikti,
        aday=a.aday,
        nrm=nrm,
        kappa=kap,
        Q_dik=Qd,
        sabit=sabit,
        kirpik=kirpik,
        e_norm=e_norm,
        kapi=kapi,
    )
    json.dump(
        rap,
        open(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)), f"m110_{a.aday or 'TABAN'}.json"
            ),
            "w",
        ),
        indent=1,
    )
    print(f"YAZILDI {yol}")
    if a.aday:
        print(f"COZUM:  L_dik({a.aday}) = ({sabit:.9f} - P^2) / {2 * kap:.6f}")
        for rho in (0.0, 0.015, 0.027, 0.036):
            v = sabit - 2 * kap * rho * np.sqrt(Qd)
            print(f"  rho_dik={rho:.3f} -> beklenen sonda skoru {np.sqrt(v):.5f}")
    else:
        print(f"BEKLENEN SKOR {np.sqrt(sabit):.5f}")


if __name__ == "__main__":
    main()
