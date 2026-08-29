"""T3 -- uc yeni adayin tam olcumu ve mevcut havuza gore DIKLIK hukmu.

Olcut (kural 36: CV/geri-test LB'yi ONGORMUYOR, aktarim f = -0,42):
  1) mevcut adaylarla |kosinus| <= 0,20
  2) kurtoz <= 10
  3) Q >= 0,01
Ayrica cok degiskenli BAGIMSIZ PAY: aday, mevcut yonlerin TAMAMININ span'i
disinda kalan enerji orani (ikili kosinus yanlitir).
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
M0 = 1.005688
R_KAL = 0.0641

MEVCUT = [
    ("m4_hava", "tuketim_m4_hava_capali.csv"),
    ("y40_sota", "tuketim_y40_sota_temiz.csv"),
    ("q1c_kapasite", "tuketim_q1c_kapasite_siki.csv"),
    ("y45_mevsimsel", "tuketim_y45_mevsimsel_kirpik.csv"),
    ("y46_amnezik", "tuketim_y46_amnezik_kirpik.csv"),
    ("z2_analog", "tuketim_z2_analog.csv"),
    ("g7_span_tau3", "tuketim_g7_span_tau3.csv"),
]
YENI = [
    ("t1_sulama", "tuketim_t1_sulama.csv"),
    ("t2_bayram", "tuketim_t2_bayram.csv"),
    ("t3_turizm", "tuketim_t3_turizm.csv"),
    ("t3_turizm_tatil", "tuketim_t3_turizm_tatil.csv"),
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
        rej = {
            k: float((f[msk[k]] ** 2).sum() / (f**2).sum()) for k in ("soguk", "kuyruk", "cekirdek")
        }
        kapi = dict(
            satir=int(len(d)),
            id_birebir=bool(len(d) == len(ss) and (d.id.values == ss.iloc[:, 0].values).all()),
            nan=int(d.tuketim.isna().sum()),
            negatif=int((d.tuketim < 0).sum()),
            sonsuz=int(np.isinf(d.tuketim.values).sum()),
            baslik=list(d.columns) == ["id", "tuketim"],
        )
        rapor[nm] = dict(
            dosya=dosya,
            **o,
            Q_rejim_payi=rej,
            yeni_pay=float((dik**2).mean() / Q) if Q > 0 else 0.0,
            kos_m4ekseni=float((f * E1).mean() / np.sqrt(Q)),
            basabas_skor=float(np.sqrt(M0 + Q)),
            kapi=kapi,
        )
        ad.append(nm)
        vek.append(nrm(f))
        print(
            f"{nm:16s} Q={Q:.5f} kurtoz={o['kurtoz']:5.1f} %1pay={100 * o['en_kotu_yuzde1_pay']:5.1f} "
            f"soguk%={100 * rej['soguk']:4.1f} kos_m4={rapor[nm]['kos_m4ekseni']:+.3f} "
            f"kapi={'OK' if all([kapi['id_birebir'], not kapi['nan'], not kapi['negatif'], not kapi['sonsuz'], kapi['baslik']]) else 'HATA'}"
        )

    K = np.array([[float((a * b).mean()) for b in vek] for a in vek])
    mevcut_ad = [n for n, _ in MEVCUT if n in ad]
    for i, nm in enumerate(ad):
        digerleri = [E1, E2] + [vek[j] for j, n in enumerate(ad) if n != nm and n in mevcut_ad]
        B = np.array(digerleri)
        G = B @ B.T / len(vek[i])
        c = B @ vek[i] / len(vek[i])
        w = np.linalg.solve(G + 1e-10 * np.eye(len(G)), c)
        art = max(0.0, float(1.0 - w @ c))
        rapor[nm]["bagimsiz_pay_mevcutlara_gore"] = art
        rapor[nm]["beklenen_kazanc_bagimsiz"] = R_KAL**2 * art

    print("\nBAGIMSIZ PAY (E1,E2 + TUM mevcut adaylar cikarildiktan sonra kalan)")
    for nm in ad:
        print(
            f"  {nm:16s} bagimsiz%={100 * rapor[nm]['bagimsiz_pay_mevcutlara_gore']:5.1f}"
            f"  m4-kalitesinde beklenen dMSE={-rapor[nm]['beklenen_kazanc_bagimsiz']:+.5f}"
        )

    print("\nKOSINUS MATRISI")
    print("                 " + " ".join(f"{n[:9]:>9s}" for n in ad))
    for i, n in enumerate(ad):
        print(f"{n:16s} " + " ".join(f"{K[i, j]:+9.3f}" for j in range(len(ad))))

    yeni_ad = [n for n, _ in YENI if n in ad]
    hukum = {}
    for nm in yeni_ad:
        i = ad.index(nm)
        kos = {n: float(K[i, ad.index(n)]) for n in mevcut_ad}
        enb = max(kos, key=lambda k: abs(kos[k]))
        r = rapor[nm]
        hukum[nm] = dict(
            en_buyuk_kosinus_adi=enb,
            en_buyuk_kosinus=kos[enb],
            kosinusler=kos,
            yeni_adaylarla={n: float(K[i, ad.index(n)]) for n in yeni_ad if n != nm},
            kosinus_gecti=bool(abs(kos[enb]) <= 0.20),
            kurtoz_gecti=bool(r["kurtoz"] <= 10),
            Q_gecti=bool(r["Q"] >= 0.01),
        )
        h = hukum[nm]
        h["hukum"] = (
            "DIK"
            if h["kosinus_gecti"] and h["Q_gecti"] and h["kurtoz_gecti"]
            else ("DIK ama kurtoz yuksek" if h["kosinus_gecti"] and h["Q_gecti"] else "DUSTU")
        )
        print(
            f"\n{nm}: en yakin mevcut = {enb} (kos {kos[enb]:+.3f})  "
            f"kos_gecti={h['kosinus_gecti']} kurtoz_gecti={h['kurtoz_gecti']} Q_gecti={h['Q_gecti']}"
            f"  -> {h['hukum']}"
        )

    out = dict(
        taban=dict(dosya="tuketim_m6_ikiyon.csv", LB=1.00284, m0=M0),
        olcut=dict(kosinus_esigi=0.20, kurtoz_esigi=10, Q_esigi=0.01),
        adaylar=rapor,
        kosinus=dict(adlar=ad, matris=K.tolist()),
        hukum=hukum,
    )
    for ek, yol in [
        ("t1_sulama", "t1_sulama.json"),
        ("t2_bayram", "t2_bayram.json"),
        ("t3_turizm", "t2_model.json"),
        ("kesif", "t1_kesif.json"),
    ]:
        p = os.path.join(BURA, yol)
        if os.path.exists(p):
            out.setdefault("ayrinti", {})[ek] = json.load(open(p, encoding="utf-8"))
    json.dump(out, open(os.path.join(BURA, "t1_turizm.json"), "w"), indent=1)
    print(f"\nyazildi {os.path.join(BURA, 't1_turizm.json')}")


if __name__ == "__main__":
    main()
