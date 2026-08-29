"""Yeni adaylarin tam olcumu: Q, kurtoz, en kotu %1 payi, TAM kosinus matrisi, kapi.

Kosinusler m6 tabanina gore log-uzayi FARK vektorleri arasinda hesaplanir.
Ayrica olculmus span {m4-v102, m6-v102} disina dusen "yeni" pay raporlanir --
span icinde kalan bilesen ikinci kez sayilmis olur.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

BURA = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BURA)
import z1_ortak as Z  # noqa: E402

S = Z.S
M0 = 1.005688  # m6'nin LB MSE'si (1.00284^2)
R_KAL = 0.0641  # m4'ten kalibre yon kalitesi L/sqrt(Q)

MEVCUT = [
    ("m4", "tuketim_m4_hava_capali.csv"),
    ("y40_sota", "tuketim_y40_sota_temiz.csv"),
    ("y42_kapasite", "tuketim_y42_kapasite_temiz.csv"),
    ("y45_mevsimsel", "tuketim_y45_mevsimsel_kirpik.csv"),
    ("y46_amnezik", "tuketim_y46_amnezik_kirpik.csv"),
]
YENI = [
    ("z1_havuz", "tuketim_z1_havuz.csv"),
    ("z2_analog", "tuketim_z2_analog.csv"),
    ("z2_kantil", "tuketim_z2_kantil.csv"),
    ("z3_ikiasama", "tuketim_z3_ikiasama.csv"),
    ("z3_sinir", "tuketim_z3_sinir.csv"),
]


def logoku(ad):
    return np.log1p(pd.read_csv(os.path.join(S, ad)).tuketim.values)


def main():
    tr, te = Z.yukle()
    msk = Z.maskeler(tr, te)
    A6 = Z.taban()
    V102 = logoku("tuketim_v102_kappa_optimum.csv")
    D4 = logoku("tuketim_m4_hava_capali.csv") - V102
    D6 = A6 - V102

    def nrm(v):
        return v / np.sqrt((v**2).mean())

    E1 = nrm(D4)
    E2 = nrm(D6 - (D6 * E1).mean() * E1)

    ss = pd.read_csv(os.path.join(Z.KOK, "data/raw/sample_submission.csv"))
    r6 = dict(
        log_ort=float(A6.mean()),
        log_std=float(A6.std()),
        alt1kwh=float((np.expm1(A6) < 1).mean()),
        maks=float(np.expm1(A6).max()),
    )
    print(f"REF m6: {r6}")

    ad, vek, rapor = [], [], {}
    for nm, dosya in MEVCUT + YENI:
        yol = os.path.join(S, dosya)
        if not os.path.exists(yol):
            print(f"  YOK {dosya} -- atlandi")
            continue
        d = pd.read_csv(yol)
        b = np.log1p(d.tuketim.values)
        f = b - A6
        o = Z.olcut(f)
        Q = o["Q"]
        dik = f - (f * E1).mean() * E1 - (f * E2).mean() * E2
        yeni_pay = float((dik**2).mean() / Q) if Q > 0 else 0.0
        rej = {
            k: float((f[msk[k]] ** 2).sum() / (f**2).sum()) for k in ("soguk", "kuyruk", "cekirdek")
        }
        kapi = dict(
            satir=int(len(d)),
            id_birebir=bool(len(d) == len(ss) and (d.id.values == ss.iloc[:, 0].values).all()),
            nan=int(d.tuketim.isna().sum()),
            negatif=int((d.tuketim < 0).sum()),
            baslik=list(d.columns) == ["id", "tuketim"],
        )
        rapor[nm] = dict(
            dosya=dosya,
            **o,
            Q_rejim_payi=rej,
            yeni_pay=yeni_pay,
            kos_m4ekseni=float((f * E1).mean() / np.sqrt(Q)),
            kos_m6dik=float((f * E2).mean() / np.sqrt(Q)),
            beklenen_kazanc_m4kalitesi=R_KAL**2 * yeni_pay,
            basabas_skor=float(np.sqrt(M0 + Q)),
            dagilim=dict(
                log_ort=float(b.mean()),
                log_std=float(b.std()),
                maks=float(d.tuketim.max()),
                alt1kwh=float((d.tuketim.values < 1).mean()),
            ),
            kapi=kapi,
        )
        ad.append(nm)
        vek.append(nrm(f))
        print(
            f"{nm:14s} Q={Q:.5f} kurtoz={o['kurtoz']:5.1f} %1pay={100 * o['en_kotu_yuzde1_pay']:5.1f} "
            f"yeni%={100 * yeni_pay:5.1f} kos_m4={rapor[nm]['kos_m4ekseni']:+.3f} "
            f"logort={b.mean():.4f} std={b.std():.4f} <1kWh={rapor[nm]['dagilim']['alt1kwh']:.4f} "
            f"kapi={'OK' if all([kapi['id_birebir'], not kapi['nan'], not kapi['negatif'], kapi['baslik']]) else 'HATA'}"
        )

    Kmat = np.array([[float((a * b).mean()) for b in vek] for a in vek])

    # --- COK DEGISKENLI BAGIMSIZLIK: aday, DIGER TUM yonlerin span'ina goreli
    # artik payi. Ikili kosinus yanlitir; asil onemli olan bu.
    mevcut_ad = [n for n, _ in MEVCUT if n in ad]
    for i, nm in enumerate(ad):
        digerleri = [E1, E2] + [vek[j] for j, n in enumerate(ad) if n != nm and n in mevcut_ad]
        B = np.array(digerleri)
        G = B @ B.T / len(vek[i])
        c = B @ vek[i] / len(vek[i])
        w = np.linalg.solve(G + 1e-10 * np.eye(len(G)), c)
        art = float(1.0 - w @ c)
        rapor[nm]["bagimsiz_pay_mevcutlara_gore"] = max(0.0, art)
        rapor[nm]["beklenen_kazanc_bagimsiz"] = R_KAL**2 * max(0.0, art)
    print(chr(10) + "BAGIMSIZ PAY (mevcut yonlerin TAMAMI birlikte cikarildiktan sonra kalan)")
    for nm in ad:
        print(
            f"  {nm:14s} bagimsiz%={100 * rapor[nm]['bagimsiz_pay_mevcutlara_gore']:5.1f}"
            f"  m4-kalitesinde beklenen dMSE={-rapor[nm]['beklenen_kazanc_bagimsiz']:+.5f}"
        )

    print("\nKOSINUS MATRISI")
    print("               " + " ".join(f"{n[:9]:>9s}" for n in ad))
    for i, n in enumerate(ad):
        print(f"{n:14s} " + " ".join(f"{Kmat[i, j]:+9.3f}" for j in range(len(ad))))

    out = dict(
        taban=dict(dosya="tuketim_m6_ikiyon.csv", LB=1.00284, m0=M0, **r6),
        kalibrasyon=dict(m4_r=R_KAL),
        adaylar=rapor,
        kosinus=dict(adlar=ad, matris=Kmat.tolist()),
    )
    json.dump(out, open(os.path.join(BURA, "z1_yeni.json"), "w"), indent=1)
    print(f"\nyazildi {os.path.join(BURA, 'z1_yeni.json')}")


if __name__ == "__main__":
    main()
