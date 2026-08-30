"""n11 TANI -- blok-disi rho'nun NEGATIF cikmasi gercek mi, kusur mu?

Kontroller:
 1) Her yontemin UYDURMA blogundaki rho'su POZITIF mi? (kurulus geregi
    oyle olmali; degilse hesapta kusur var.)
 2) m148'in GERCEK kurulusu (kapilar+isaretler yaz25'ten) yaz25'te ne
    veriyor? Beklenen ~ +0.13..0.15 (gorevin bildirdigi doyma degeri).
 3) Tek tek eksenlerin blok korelasyonlari bloklar arasi nasil iliskili?
    Isaret uyumu / korelasyon.
 4) kis26'da kurulabilen 97 eksen icin ayni sey.
"""

import json
import os

import n11_analiz as A
import numpy as np

print("havuz:", A.P)
for ad, b in A.BL.items():
    print(
        f"{ad}: gecerli={int(b.gecerli.sum())} plasebo={int((b.gecerli & b.plasebo).sum())} "
        f"kapi={int(b.kapi.sum())} m0b={b.m0:.4f}"
    )

print("\n--- 1/2) ic-blok vs dis-blok rho ---")
for fit, ev in (("guz25", "yaz25"), ("yaz25", "guz25"), ("yaz25", "kis26"), ("kis26", "yaz25")):
    bf, be = A.BL[fit], A.BL[ev]
    sec, T = A.m148_sirasi(bf, 40)
    if len(sec) < 2:
        print(f"{fit}->{ev}: secim bos")
        continue
    ort = [i for i in sec if be.gecerli[i]]
    for kip in ("a_m148", "b_rho_cv"):
        c = A.c_den_k(T, A.agirlik(kip, sec, bf))
        # olcum blogunda kurulamayan eksenleri sifirla
        c2 = c.copy()
        c2[~be.gecerli] = 0.0
        print(
            f"  {fit}->{ev} K={len(sec)} ({len(ort)} olcum blogunda gecerli) {kip:>9s}: "
            f"ic={A.rho(c, bf.G, bf.g, bf.m0):+.4f}  dis={A.rho(c2, be.G, be.g, be.m0):+.4f}"
        )

print("\n--- 3) eksen bazinda bloklar arasi isaret uyumu ---")
for a, b in (("yaz25", "guz25"), ("yaz25", "kis26"), ("guz25", "kis26")):
    ba, bb = A.BL[a], A.BL[b]
    m = ba.gecerli & bb.gecerli & ba.plasebo & bb.plasebo
    ra = ba.g[m] / np.sqrt(ba.m0 * np.maximum(np.diag(ba.G)[m], 1e-12))
    rb = bb.g[m] / np.sqrt(bb.m0 * np.maximum(np.diag(bb.G)[m], 1e-12))
    uy = float((np.sign(ra) == np.sign(rb)).mean())
    kr = float(np.corrcoef(ra, rb)[0, 1])
    print(
        f"  {a} vs {b}: n={int(m.sum())} isaret_uyumu={uy:.3f} kor={kr:+.3f} "
        f"|rho| ort {np.abs(ra).mean():.4f} / {np.abs(rb).mean():.4f}"
    )

print("\n--- 4) m148'in KENDI kurulusu, KENDI blogunda (yaz25) ---")
bf = A.BL["yaz25"]
for K in (10, 20, 25, 40, 60, 100, 136):
    sec, T = A.m148_sirasi(bf, K)
    if len(sec) < 2:
        continue
    c = A.c_den_k(T, A.agirlik("a_m148", sec, bf))
    ong = float(np.sqrt((A.agirlik("a_m148", sec, bf) ** 2).sum()))
    print(
        f"  K={K:3d} secilen={len(sec):3d} ongorulen||BETA||={ong:.4f} yaz25_rho={A.rho(c, bf.G, bf.g, bf.m0):+.4f}"
    )

print("\n--- 5) ZAMAN icinde bolme: yaz25'i ilk/son yariya bol ---")
d = (
    np.load(os.path.join(A.ARA, "n11_zaman_yaz25.npz"))
    if os.path.exists(os.path.join(A.ARA, "n11_zaman_yaz25.npz"))
    else None
)
print("  (n11_zaman.py calistirilmadi)" if d is None else "  var")
json.dump({}, open(os.devnull, "w"))
