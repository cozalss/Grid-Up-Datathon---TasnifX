"""n11 ISARET TANISI.

m148 her eksene  KATS = isaret(rho_cv_yaz25) * 1.95 * |rho_s|  veriyor:
BUYUKLUK liderlik tablosundan (rho_s), ISARET yaz25 blogundan.

Bu betik iki soruyu ayirir:
 A) rho_cv'nin ISARETI, TEST uzayinda olculen rho_s'in isaretiyle uyusuyor
    mu? (rho_s, 28 liderlik olcumunden kurulan r_hat ile test uzayinda
    hesaplanir -- TEK dogrudan test kaniti.)
 B) Blok isaretleri bloklar arasinda ve BLOK ICINDE ZAMAN boyunca kararli
    mi? (yaz25 = Nisan..Temmuz; ilk yari Nis-May, ikinci yari Haz-Tem.
    Test Nisan..Temmuz 2026, yani bir yil sonraki ayni mevsim.)
"""

import json
import os

import n11_analiz as A
import numpy as np
import pandas as pd
from n11_eksen_secimi import AO, ARA, DN, HEDEF_SOGUK, SIC_AILE, kolonlar, st, yap_kur

RHO_S = A.RHO_S
print("--- A) rho_s (TEST) ile blok korelasyonlarinin ISARET uyumu ---")
for ad, b in A.BL.items():
    m = b.gecerli & b.plasebo
    rb = b.g / np.sqrt(b.m0 * np.maximum(np.diag(b.G), 1e-12))
    uy = float((np.sign(rb[m]) == np.sign(RHO_S[m])).mean())
    kr = float(np.corrcoef(rb[m], RHO_S[m])[0, 1])
    # m148'in fiilen kullandigi kume: kapidan gecenler
    mk = b.kapi
    uyk = float((np.sign(rb[mk]) == np.sign(RHO_S[mk])).mean()) if mk.sum() else float("nan")
    print(
        f"  {ad}: n={int(m.sum())} isaret_uyumu(rho_s)={uy:.3f} kor={kr:+.3f} | "
        f"kapidan gecen n={int(mk.sum())} uyum={uyk:.3f}"
    )

# ------------------------------------------------------- B) zaman bolmesi
print("\n--- B) yaz25 ICINDE zaman bolmesi (Nis-May vs Haz-Tem) ---")
YOL = os.path.join(ARA, "n11_yaz25_zaman.npz")
if not os.path.exists(YOL):
    import pyarrow.parquet as pq

    with open(os.path.join(ARA, "n11_havuz.json"), encoding="utf-8") as fh:
        havuz = json.load(fh)
    adlar = [h["eksen"] for h in havuz]
    ih = set()
    for a in adlar:
        kolonlar(a, ih)
    tumk = set(pq.read_schema(os.path.join(DN, "test.parquet")).names)
    ekstra = ["soguk_mu", "ufuk_gun", "tarih", "tanim", "tuketim", "_blok"]
    e = pd.read_parquet(
        os.path.join(DN, "egitim.parquet"), columns=sorted((ih & tumk) | set(ekstra))
    )
    tp_ref = pd.read_parquet(os.path.join(DN, "test.parquet"), columns=sorted(ih & tumk))
    blok = "yaz25"
    blk = e[e._blok == blok]
    sic, sog = blk[blk.soguk_mu == 0], blk[blk.soguk_mu == 1]
    P = [
        np.load(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy")).astype(np.float64)
        for t, aa in SIC_AILE
        if os.path.exists(os.path.join(AO, f"{blok}_{t}_{aa}_uretim.npy"))
    ]
    z = np.load(os.path.join(DN, f"soguk_tahmin_{blok}.npz"))
    idx = np.concatenate([sic.index.values, sog.index.values])
    pb = np.concatenate([np.mean(P, axis=0), np.mean([z[q] for q in z.files], axis=0)])
    bf = e.loc[idx].copy()
    rb = np.log1p(bf.tuketim.values.astype(np.float64)) - pb
    sgm = bf.soguk_mu.values.astype(np.float64)
    ww = np.where(sgm == 1, HEDEF_SOGUK / sgm.mean(), (1 - HEDEF_SOGUK) / (1 - sgm.mean()))
    ww = ww / ww.mean()
    nb = len(rb)
    carpB = {
        "x_sv": st(pb),
        "x_soguk": sgm,
        "x_ufuk": st(bf.ufuk_gun.to_numpy()),
        "x_ay": st(pd.to_datetime(bf.tarih).dt.month.to_numpy().astype(np.float64)),
    }
    kur = yap_kur(bf, carpB, tp_ref)
    ay = pd.to_datetime(bf.tarih).dt.month.to_numpy()
    yari = [(ay <= 5), (ay >= 6)]
    G2 = np.zeros((2, len(adlar), len(adlar)))
    g2 = np.zeros((2, len(adlar)))
    m2 = np.zeros(2)
    X = np.zeros((len(adlar), nb), dtype=np.float32)
    for i, a in enumerate(adlar):
        v = kur(a)
        if v is not None:
            X[i] = v
    for j, msk in enumerate(yari):
        Xs = X[:, msk]
        wj = ww[msk]
        G2[j] = (Xs * wj.astype(np.float32)) @ Xs.T / msk.sum()
        g2[j] = Xs.astype(np.float64) @ (wj * rb[msk]) / msk.sum()
        m2[j] = float((wj * rb[msk] * rb[msk]).mean())
        print(f"  yari {j}: {int(msk.sum()):,} satir m0={m2[j]:.4f}")
    np.savez(YOL, G2=G2, g2=g2, m2=m2)
d = np.load(YOL)
G2, g2, m2 = d["G2"], d["g2"], d["m2"]
r0 = g2[0] / np.sqrt(m2[0] * np.maximum(np.diag(G2[0]), 1e-12))
r1 = g2[1] / np.sqrt(m2[1] * np.maximum(np.diag(G2[1]), 1e-12))
m = A.BL["yaz25"].kapi
print(
    f"  eksen bazinda (kapidan gecen n={int(m.sum())}): isaret uyumu="
    f"{(np.sign(r0[m]) == np.sign(r1[m])).mean():.3f} kor={np.corrcoef(r0[m], r1[m])[0, 1]:+.3f}"
)

print("\n  BILESIK: yariyi A'da kur, B'de olc (ve tersi)")
bf_ = A.BL["yaz25"]
for K in (10, 20, 25, 40, 60, 100, 136):
    sec, T = A.m148_sirasi(bf_, K)
    if len(sec) < 2:
        continue
    sat = []
    for f, ev in ((0, 1), (1, 0)):
        Gf, gf, mf = G2[f], g2[f], m2[f]
        Ge, ge, me = G2[ev], g2[ev], m2[ev]
        s = np.array(sec)
        rcv = A.CARPAN * gf[s] / np.sqrt(mf * np.maximum(np.diag(Gf)[s], 1e-12))
        isr = np.sign(rcv)
        isr[isr == 0] = 1
        ka = isr * A.TAVAN * np.abs(RHO_S[s])
        ca = A.c_den_k(T, ka)
        cb = A.c_den_k(T, rcv)
        # isaret rho_s'ten alinirsa
        isr2 = np.sign(RHO_S[s])
        isr2[isr2 == 0] = 1
        cs = A.c_den_k(T, isr2 * A.TAVAN * np.abs(RHO_S[s]))
        sat.append(
            (
                A.rho(ca, Ge, ge, me),
                A.rho(cb, Ge, ge, me),
                A.rho(cs, Ge, ge, me),
                A.rho(ca, Gf, gf, mf),
            )
        )
    a = np.array(sat)
    print(
        f"  K={K:3d} n={len(sec):3d}  ic={a[:, 3].mean():+.4f}  "
        f"dis a_m148={a[:, 0].mean():+.4f}  dis b_rho_cv={a[:, 1].mean():+.4f}  "
        f"dis isaret_rho_s={a[:, 2].mean():+.4f}"
    )
