"""v93 denetimi -- adim 4: b vektorunun matematiksel TUTARLILIGI + kirpma + zaman ekseni."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np

BURA = Path(__file__).resolve().parent


def main() -> None:
    digerleri, D, G, b, m0, n = C.gram_kur(C.OLCULENLER)
    U, s, Vt = np.linalg.svd(G, hermitian=True)

    print("=== 1) CAUCHY-SCHWARZ TUTARLILIK SINAMASI ===")
    print("Eger butun skorlar AYNI tam kume uzerinde HATASIZ olculmusse,")
    print("her v icin |b.v| <= ||t-p0||/sqrt(n) * ||sum v_i d_i||/sqrt(n) OLMAK ZORUNDA.")
    # ||t-p0||^2/n = m0 (cunku MSE_0 = ||p0-t||^2/n)
    norm_t_p0 = np.sqrt(m0)
    ihlal = []
    for j in range(18):
        v = Vt[j]
        kombi_norm = np.sqrt(s[j])  # ||sum v_i d_i||/sqrt(n) = sqrt(v'Gv) = sqrt(s_j)
        sol = abs(float(b @ v))
        sag = norm_t_p0 * kombi_norm
        oran = sol / sag if sag > 0 else np.inf
        if j >= 13:
            print(
                f"  yon {j:2d} (sv={s[j]:.3e}): |b.v|={sol:.4e}  sinir={sag:.4e}  oran={oran:.3f}"
                + ("   <<< IHLAL" if oran > 1 else "")
            )
        if oran > 1:
            ihlal.append(
                {"yon": j, "sv": float(s[j]), "b_v": sol, "sinir": sag, "oran": float(oran)}
            )
    print(f"\nIHLAL sayisi: {len(ihlal)}")
    if ihlal:
        w = ihlal[0]
        print(
            f"En keskin ihlal yon {w['yon']}: olculen {w['b_v']:.3e}, "
            f"matematiksel ust sinir {w['sinir']:.3e}  ->  ASIRI {w['oran']:.2f}x"
        )
        print(
            f"  Bu, b'de en az {w['b_v'] - w['sinir']:.2e} buyuklugunde ACIKLANAMAZ gurultu demektir."
        )

    print("\n=== 2) b GURULTU BUTCESI ===")
    # 5-ondalik yuvarlama: skor s -> m=s^2, dm = 2 s ds, |ds| <= 5e-6
    ds = 5e-6
    dm = 2 * 1.02 * ds
    print(f"  5-ondalik yuvarlama: |ds|<=5e-6 -> |dm|<={dm:.3e} -> |eps_i|<={dm:.3e} (m0 dahil)")
    print(
        f"  Olculen aciklanamaz gurultu (yon 16): {ihlal[0]['b_v'] - ihlal[0]['sinir']:.3e}"
        if ihlal
        else ""
    )

    print("\n=== 3) v93 KIRPMA ANALIZI ===")
    p0 = C.yukle(C.TABAN)
    v93 = C.yukle("v93")
    w16, _, _ = C.coz(G, b, rank=16)
    d_teori = D.T @ w16  # kirpmasiz teorik yon
    d93 = v93 - p0
    kirpik = v93 == 0.0
    print(f"  v93'te log1p==0 (yani tahmin==0) satir sayisi: {kirpik.sum()}")
    tasma = (p0 + d_teori) < 0
    print(f"  Teorik log1p'in NEGATIF oldugu satir sayisi   : {tasma.sum()}")
    fark = d93 - d_teori
    print(
        f"  ||d93 - d_teori||^2/n = {float(fark @ fark) / n:.4e}   "
        f"(||d_teori||^2/n = {float(d_teori @ d_teori) / n:.8f})"
    )
    print(f"  ||d93||^2/n           = {float(d93 @ d93) / n:.8f}")
    # kirpmanin dMSE'ye etkisi: sadece etkilenen satirlarda
    etki = np.abs(fark) > 1e-12
    print(f"  d93 != d_teori olan satir sayisi: {etki.sum()}")
    if etki.sum():
        print(f"  bu satirlarda ||fark||^2/n = {float(fark[etki] @ fark[etki]) / n:.4e}")
        print(f"  en buyuk mutlak fark = {np.abs(fark).max():.6f} log birimi")

    print("\n=== 4) ZAMAN EKSENI ===")
    ids = np.load(C.ONB / "_ids.npy", allow_pickle=True)
    tarih = np.array([str(x).split("_")[1] for x in ids])
    ay = np.array([t[:7] for t in tarih])
    aylar = sorted(set(ay.tolist()))
    print(f"  test aylari: {aylar}")
    for a in aylar:
        msk = ay == a
        print(
            f"    {a}: {msk.sum():>7} satir ({msk.mean():.2%})  "
            f"||d93||^2/n_ay = {float(d93[msk] @ d93[msk]) / msk.sum():.6f}"
        )
    ilk = np.isin(ay, aylar[:2])
    son = np.isin(ay, aylar[2:]) if len(aylar) > 2 else ~ilk
    for ad, msk in [("Nis+May", ilk), ("Haz+Tem", son)]:
        Qa = float(d93[msk] @ d93[msk]) / msk.sum()
        print(f"  {ad}: pay={msk.mean():.3f}  Q_alt={Qa:.8f}")

    (BURA / "tutarlilik.json").write_text(
        json.dumps(
            {
                "ihlaller": ihlal,
                "kirpik_satir": int(kirpik.sum()),
                "teorik_negatif": int(tasma.sum()),
                "kirpma_fark_norm2": float(fark @ fark) / n,
                "aylar": aylar,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
