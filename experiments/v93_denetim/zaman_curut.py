"""v93 denetimi -- adim 8: ZAMANSAL bolme hipotezlerini null yonlerle CURUT + kappa kalkani."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent


def main() -> None:
    p0 = C.yukle(C.TABAN)
    n = p0.size
    dig = [e for e in C.OLCULENLER if e != C.TABAN]
    D = np.empty((len(dig), n))
    for i, e in enumerate(dig):
        D[i] = C.yukle(e) - p0
    G = (D @ D.T) / n
    m0 = C.ENV[C.TABAN]["skor"] ** 2
    m = np.array([C.ENV[e]["skor"] ** 2 for e in dig])
    b = (m0 + np.diag(G) - m) / 2.0
    U, s, Vt = np.linalg.svd(G, hermitian=True)
    d93 = C.yukle("v93") - p0
    Qf = float(d93 @ d93) / n
    c, _, _ = C.coz(G, (D @ d93) / n, rank=16)

    ids = np.load(C.ONB / "_ids.npy", allow_pickle=True)
    ay = np.array([str(x).split("_")[1][:7] for x in ids])
    aylar = ["2026-04", "2026-05", "2026-06", "2026-07"]

    hip = {"TAM KUME (bolme yok / rastgele)": np.ones(n, bool)}
    for k in range(1, 4):
        hip[f"public = {'+'.join(a[5:] for a in aylar[:k])}"] = np.isin(ay, aylar[:k])
        hip[f"public = {'+'.join(a[5:] for a in aylar[k:])}"] = np.isin(ay, aylar[k:])
    for a in aylar:
        hip[f"public = yalniz {a[5:]}"] = ay == a

    print("=== NULL YONU IMZASI ILE ZAMANSAL HIPOTEZ CURUTMESI ===")
    print("Null yonu v icin b_hesap.v = [b^dilim.v (~0)] + v.(diagG^tam - diagG^dilim)/2")
    print("Yani her hipotez, gozlemlenmis sayiya KESIN bir tahmin veriyor.\n")
    for j, sd_round in ((16, 5.646e-06), (17, 4.850e-06)):
        v = Vt[j]
        goz = float(b @ v)
        print(f"--- yon {j}: GOZLEMLENEN b.v = {goz:+.4e}   (yuvarlama sd={sd_round:.2e}) ---")
        print(f"{'hipotez':>34} {'ongoru':>12} {'fark':>12} {'sigma':>8}")
        for ad, msk in hip.items():
            npub = int(msk.sum())
            Gp = np.einsum("ij,ij->i", D[:, msk], D[:, msk]) / npub
            ong = float(v @ ((np.diag(G) - Gp) / 2))
            z = abs(goz - ong) / sd_round
            bayrak = "  <<< CURUK" if z > 3 else ("  supheli" if z > 2 else "")
            print(f"{ad:>34} {ong:>+12.3e} {goz - ong:>+12.3e} {z:>8.2f}{bayrak}")
        print()

    print("\n=== KAPPA KALKANI: v93_k = v83 + k*(v93-v83) ===")
    print("Beklenen public MSE degisimi = k^2*Q - 2k*(c.b)")
    cb = float(c @ b)
    senar = {ad: msk for ad, msk in hip.items() if ad != "TAM KUME (bolme yok / rastgele)"}
    basliklar = ["TAM"] + [a.replace("public = ", "") for a in senar]
    print(f"{'kappa':>6} " + " ".join(f"{h:>12}" for h in basliklar))
    tablo = []
    for k in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
        satir = [np.sqrt(m0 + k * k * Qf - 2 * k * cb)]
        for ad, msk in senar.items():
            npub = int(msk.sum())
            Dp = D[:, msk]
            Gp = np.einsum("ij,ij->i", Dp, Dp) / npub
            dp = d93[msk]
            Qp = float(dp @ dp) / npub
            # dilimde gercek MSE = m0_dilim + k^2 Qp - 2k*c.b^dilim
            # c.b^dilim = c.b_hesap - c.(diagG^tam - diagG^dilim)/2
            cb_dilim = cb - float(c @ (np.diag(G) - Gp)) / 2
            satir.append(np.sqrt(max(m0 + k * k * Qp - 2 * k * cb_dilim, 0)))
        tablo.append({"kappa": k, "degerler": [float(x) for x in satir]})
        print(f"{k:>6.1f} " + " ".join(f"{x:>12.5f}" for x in satir))
    print("\n(m0 dilim bazinda degismez varsayildi -- taban skoru kaymasi ortak, sirayi etkilemez)")
    print("2. sira = 1.00938 ; v83 mevcut = 1.01318")

    (BURA / "zaman_curut.json").write_text(
        json.dumps({"basliklar": basliklar, "kappa_tablo": tablo}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
