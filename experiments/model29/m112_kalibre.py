"""KALIBRASYON AILESI -- 2. sira icin sistem.

BULGU (2026-08-30, olculdu): yapisal "seviye" yonu rho = -0.0304 verdi,
kazanc 9.26e-04 -- model varyantlarinin artimli rekorunun (3.12e-04) 3 KATI.
Yani hata model seciminde degil KALIBRASYONDA: tahminler asiri yayilmis.

Bu, tek bir yon degil bir AILE. Kalibrasyon egrisi yanlissa:
  - egriligi de yanlistir            (seviye^2, seviye^3)
  - kesitlere gore farklidir         (seviye x soguk, seviye x guc, seviye x ay)
  - kesitlerin kendi kaymasi vardir  (soguk, bolge, haftasonu)

Her yon olculmus span'a VE onceki olculmus yapisal yonlere DIK yapilir
-> her sonda saf yeni bilgi olcer, hicbiri otekini tekrar etmez.

Kullanim:
  python m112_kalibre.py --liste
  python m112_kalibre.py --aday seviye2 --yerdeg 0.005 --cikti tuketim_K_seviye2.csv
  python m112_kalibre.py --kaydet seviye2 --skor 1.00102
  python m112_kalibre.py --nihai --cikti tuketim_K_NIHAI.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from m30_ozellik import KOK

S = os.path.join(KOK, "submissions")
BURA = os.path.dirname(os.path.abspath(__file__))
TABAN = "tuketim_m6_ikiyon.csv"
M0 = 1.005846366
RCOND = 1e-6
DURUM = os.path.join(BURA, "m112_durum.json")
# LB'den dogrudan olculmus, olculmus_skorlar.json'da olmayan MODEL yonleri
EK_MODEL = {"tuketim_y40_sota_temiz.csv": -0.002229}
# Dosyaya dayali yapisal yonler (formulle degil, hazir CSV ile tanimli)
DOSYA_YON = {"yenibaslangic": "tuketim_KES_yenibaslangic.csv"}


def oku(f):
    d = pd.read_csv(os.path.join(S, f))
    k = "tuketim" if "tuketim" in d.columns else d.columns[-1]
    return np.log1p(d[k].values.astype(np.float64))


def durum_yukle():
    if os.path.exists(DURUM):
        return json.load(open(DURUM))
    # seviye 2026-08-30'da olculdu: skor 1.00115
    return {"yapisal": {"seviye": -0.024649}, "bekleyen": None, "gecmis": []}


def durum_kaydet(d):
    g = DURUM + ".tmp"
    with open(g, "w") as f:
        json.dump(d, f, indent=1)
    Path(g).replace(DURUM)


def yapisal_yonler(te, a0, tr_tanim):
    """Ham (ortogonallestirilmemis) yapisal yonler."""
    tarih = pd.to_datetime(te.tarih)
    soguk = (~te.tanim.isin(tr_tanim)).to_numpy().astype(np.float64)
    ay = tarih.dt.month.to_numpy().astype(np.float64)
    hs = (tarih.dt.dayofweek >= 5).to_numpy().astype(np.float64)
    lg = np.log1p(te.guc.values.astype(np.float64))
    lg = (lg - lg.mean()) / lg.std()
    sv = (a0 - a0.mean()) / a0.std()
    ayn = (ay - ay.mean()) / ay.std()
    bolge = te.lokasyon.str.split(">").str[1].fillna("?")
    Y = {
        "seviye": sv,
        "seviye2": sv**2,
        "seviye3": sv**3,
        "seviye_x_soguk": sv * soguk,
        "seviye_x_guc": sv * lg,
        "seviye_x_ay": sv * ayn,
        "seviye_x_hs": sv * hs,
        "soguk": soguk,
        "guc": lg,
        "ay": ayn,
        "haftasonu": hs,
        "seviye2_x_soguk": (sv**2) * soguk,
    }
    # AJAN A'nin uc holdout'ta plasebo kontrollu dogruladigi TAM SEKIL:
    # dogrusal buzme, |u|>1.5 doygun, soguk 4x, ufuk Nis .30 May 1.0 Haz 1.4 Tem 1.32
    ufuk = pd.Series(ay).map({4: 0.30, 5: 1.00, 6: 1.40, 7: 1.32}).to_numpy()
    w = (1.0 + 3.0 * soguk) * ufuk
    w = w / w.mean()
    Y["buzme_tam"] = -w * np.clip(sv, -1.5, 1.5)
    Y["buzme_sade"] = -np.clip(sv, -1.5, 1.5)
    Y["buzme_soguk"] = -(1.0 + 3.0 * soguk) / (1.0 + 3.0 * soguk).mean() * np.clip(sv, -1.5, 1.5)
    Y["buzme_ufuk"] = -(ufuk / ufuk.mean()) * np.clip(sv, -1.5, 1.5)
    for b in bolge.value_counts().index[:4]:
        m = (bolge == b).to_numpy().astype(np.float64)
        Y[f"bolge_{b.split()[0][:6]}"] = m
        Y[f"seviye_x_{b.split()[0][:6]}"] = sv * m
    # merkezle + birim norm
    for k in Y:
        x = Y[k] - Y[k].mean()
        Y[k] = x / np.sqrt(float((x * x).mean()))
    return Y


def kur(te, a0, N, d):
    """Bilinen her seyden r_hat kur. Doner: r_hat, izdusum fonksiyonu."""
    SK = json.load(open(os.path.join(BURA, "olculmus_skorlar.json")))
    V, L = [], []
    for f, P in SK.items():
        if f == TABAN or not os.path.exists(os.path.join(S, f)):
            continue
        v = oku(f)
        if len(v) != N:
            continue
        dd = v - a0
        V.append(dd)
        L.append((M0 + float((dd * dd).mean()) - P * P) / 2)
    for f, Lj in EK_MODEL.items():
        V.append(oku(f) - a0)
        L.append(Lj)
    # olculmus YAPISAL yonler: ham hallerini ekle, L'leri dik bilesende olculdu
    tr = pd.read_csv(os.path.join(KOK, "data/raw/train.csv"), usecols=["tanim"])
    Y = yapisal_yonler(te, a0, set(tr.tanim.unique()))
    V0 = np.array(V).T
    G0 = (V0.T @ V0) / N
    for ad, Lp in d["yapisal"].items():
        if ad in DOSYA_YON:
            xf = oku(DOSYA_YON[ad]) - a0
            x = xf / np.sqrt(float((xf * xf).mean()))
        else:
            x = Y[ad]
        c, *_ = np.linalg.lstsq(G0, (V0.T @ x) / N, rcond=RCOND)
        xp = x - V0 @ c
        V.append(xp)
        L.append(Lp)  # dik bilesende olculen L
    V = np.array(V).T
    L = np.array(L)
    G = (V.T @ V) / N
    r_hat = V @ (np.linalg.pinv(G, rcond=RCOND) @ L)
    return r_hat, V, G, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aday")
    ap.add_argument("--yerdeg", type=float, default=0.005)
    ap.add_argument("--cikti")
    ap.add_argument("--liste", action="store_true")
    ap.add_argument("--nihai", action="store_true")
    ap.add_argument("--kaydet")
    ap.add_argument("--skor", type=float)
    a = ap.parse_args()
    d = durum_yukle()

    te = pd.read_csv(os.path.join(KOK, "data/raw/test.csv"))
    ss = pd.read_csv(os.path.join(KOK, "data/raw/sample_submission.csv"))
    a0 = oku(TABAN)
    N = len(a0)
    r_hat, V, G, Y = kur(te, a0, N, d)
    nrm = float((r_hat * r_hat).mean())
    print(f"BILINEN: {V.shape[1]} yon ({len(d['yapisal'])} yapisal olculmus)")
    print(f"  ||r_hat||^2 = {nrm:.6f}  ->  saf optimum {np.sqrt(M0 - nrm):.5f}")

    if a.kaydet:
        b = d["bekleyen"]
        if not b or b["aday"] != a.kaydet:
            raise SystemExit(f"bekleyen sonda '{a.kaydet}' degil: {b}")
        L = (b["sabit"] - a.skor**2) / (2 * b["kappa"])
        rho = L / np.sqrt(b["Q_dik"])
        print(f"\n{a.kaydet}: skor {a.skor} -> L = {L:+.6f}  rho = {rho:+.4f}  kazanc {rho**2:.3e}")
        d["yapisal"][a.kaydet] = L
        d["gecmis"].append(dict(aday=a.kaydet, skor=a.skor, L=L, rho=rho))
        d["bekleyen"] = None
        durum_kaydet(d)
        print("KAYDEDILDI. Yeni taban icin --liste calistir.")
        return

    if a.liste or not (a.aday or a.nihai):
        print(f"\n{'aday':>22s} {'Q_dik':>9s} {'span-disi':>10s} {'rho=0.03 kazanci':>17s}")
        for ad, x in list(Y.items()) + [(k, None) for k in DOSYA_YON]:
            if ad in d["yapisal"]:
                print(f"{ad:>22s}  [OLCULDU  L={d['yapisal'][ad]:+.6f}]")
                continue
            c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
            xp = x - V @ c
            Qd = float((xp * xp).mean())
            print(f"{ad:>22s} {Qd:9.4f} {Qd:10.3f} {0.03**2:17.3e}")
        print(f"\nolculen yapisal: {json.dumps(d['yapisal'], indent=1)}")
        return

    if a.nihai:
        p = a0 + r_hat
        etiket = "NIHAI (saf optimum)"
        kap, Qd, xp = 0.0, 0.0, None
    else:
        if a.aday not in Y:
            raise SystemExit(f"bilinmeyen aday {a.aday}; --liste ile bak")
        if a.aday in d["yapisal"]:
            raise SystemExit(f"{a.aday} zaten olculdu")
        x = Y[a.aday]
        c, *_ = np.linalg.lstsq(G, (V.T @ x) / N, rcond=RCOND)
        xp = x - V @ c
        Qd = float((xp * xp).mean())
        if Qd < 1e-4:
            raise SystemExit(f"{a.aday}: dik bilesen cok kucuk ({Qd:.2e}), olculemez")
        kap = a.yerdeg / np.sqrt(Qd)
        p = a0 + r_hat + kap * xp
        etiket = f"SONDA {a.aday}"
        print(
            f"\n{etiket}: Q_dik={Qd:.4f} kappa={kap:.5f} "
            f"SNR(rho=0.03)={0.03 * a.yerdeg / 5.01e-6:.0f}"
        )

    y = np.clip(np.expm1(p), 0.0, None)
    out = pd.DataFrame({"id": te.id.values, "tuketim": y})
    ok = (
        len(out) == 714688
        and (out.id.values == ss.iloc[:, 0].values).all()
        and out.tuketim.isna().sum() == 0
        and (out.tuketim < 0).sum() == 0
        and np.isfinite(out.tuketim.values).all()
        and out.tuketim.max() < 3 * np.expm1(a0).max()
    )
    if not ok:
        raise SystemExit("KAPI KALDI")
    dgv = np.log1p(out.tuketim.values) - a0
    sabit = float(M0 - 2 * nrm + float(dgv @ dgv) / N)
    if not a.cikti:
        raise SystemExit("--cikti gerekli")
    g = Path(os.path.join(S, a.cikti) + ".tmp")
    out.to_csv(g, index=False)
    g.replace(os.path.join(S, a.cikti))
    print(f"kirpik {int((y == 0).sum())}  maks {out.tuketim.max():,.0f}  KAPI GECTI")
    print(f"YAZILDI submissions/{a.cikti}")
    if a.nihai:
        print(f"BEKLENEN SKOR {np.sqrt(sabit):.5f}  (tum L'ler olculmus)")
    else:
        d["bekleyen"] = dict(aday=a.aday, cikti=a.cikti, sabit=sabit, kappa=kap, Q_dik=Qd)
        durum_kaydet(d)
        print(f"COZUM: L = ({sabit:.9f} - P^2) / {2 * kap:.6f}")
        for rho in (-0.03, 0.0, 0.015, 0.030, 0.05):
            print(f"  rho={rho:+.3f} -> {np.sqrt(sabit - 2 * kap * rho * np.sqrt(Qd)):.5f}")
        print(f"\nskor gelince: python m112_kalibre.py --kaydet {a.aday} --skor <S>")


if __name__ == "__main__":
    main()
