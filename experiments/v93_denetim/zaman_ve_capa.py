"""v93 denetimi -- adim 7: zamansal bolmenin TAM hatasi + bagimsiz capa (egitim etiketleri)."""

from __future__ import annotations

import json
from pathlib import Path

import coz as C
import numpy as np
import pandas as pd

BURA = Path(__file__).resolve().parent
KOK = Path(__file__).resolve().parents[2]


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
    d93 = C.yukle("v93") - p0
    Qf = float(d93 @ d93) / n
    c, _, _ = C.coz(G, (D @ d93) / n, rank=16)
    w, _, _ = C.coz(G, b, rank=16)
    print(f"|c|_2 = {np.linalg.norm(c):.4f}  |c|_1 = {np.abs(c).sum():.4f}")
    print(f"|w|_2 = {np.linalg.norm(w):.4f}  |w|_1 = {np.abs(w).sum():.4f}")
    afin = 1.0 - w.sum()
    print(
        f"afin taban katsayisi (v83) = {afin:.4f}   |afin agirlik|_1 = "
        f"{abs(afin) + np.abs(w).sum():.4f}   (ureten ajanin '14,31'i)"
    )

    ids = np.load(C.ONB / "_ids.npy", allow_pickle=True)
    ay = np.array([str(x).split("_")[1][:7] for x in ids])

    print("\n=== ZAMANSAL BOLMENIN TAM (rastgele degil, KESIN) HATASI ===")
    print("Hata = (Q^tam - Q^dilim) - c.(diagG^tam - diagG^dilim)   [MSE birimi]")
    senaryolar = {
        "Nis+May (public %42)": np.isin(ay, ["2026-04", "2026-05"]),
        "Nisan (public %16)": ay == "2026-04",
        "Nis+May+Haz (public %70)": np.isin(ay, ["2026-04", "2026-05", "2026-06"]),
        "Haz+Tem (public %58)": np.isin(ay, ["2026-06", "2026-07"]),
        "Temmuz (public %30)": ay == "2026-07",
    }
    sonuc = {}
    for ad, msk in senaryolar.items():
        npub = int(msk.sum())
        Dp = D[:, msk]
        Gp = np.einsum("ij,ij->i", Dp, Dp) / npub
        dp = d93[msk]
        Qp = float(dp @ dp) / npub
        hata = (Qf - Qp) - float(c @ (np.diag(G) - Gp))
        # o dilimdeki taban MSE de degisir; ama LB'nin gosterecegi
        # skor = sqrt(m0_dilim + Q_dilim - 2 c.b_dilim). Tahminimiz
        # sqrt(m0_dilim + Qf - 2 c.b_calc). Fark tam olarak 'hata'.
        tah = 1.008334
        gercek_yak = np.sqrt(max(tah**2 - hata, 0))
        sonuc[ad] = {
            "n": npub,
            "Q_dilim": Qp,
            "hata_mse": hata,
            "kayma_rmsle": float(gercek_yak - tah),
        }
        print(
            f"  {ad:26s} n={npub:>6}  Q={Qp:.6f}  hata={hata:+.6f} MSE  "
            f"-> LB {tah:.5f} yerine ~{gercek_yak:.5f}  ({gercek_yak - tah:+.5f})"
        )

    print("\n=== BAGIMSIZ CAPA: ||t||^2/n sinirlari vs egitim etiketleri ===")
    P0 = float(p0 @ p0) / n
    print(f"  ||log1p(v83)||^2/n = {P0:.6f}  -> ||p0||/sqrt(n) = {np.sqrt(P0):.6f}")
    print(f"  ||t-p0||^2/n = m0 = {m0:.6f}  -> ||t-p0||/sqrt(n) = {np.sqrt(m0):.6f}")
    alt = np.sqrt(P0) - np.sqrt(m0)
    ust = np.sqrt(P0) + np.sqrt(m0)
    print(f"  ucgen esitsizligi: ||log1p(t)||/sqrt(n) in [{max(alt, 0):.6f}, {ust:.6f}]")
    print(f"  yani E[log1p(t)^2] in [{max(alt, 0) ** 2:.6f}, {ust**2:.6f}]")

    tr = pd.read_csv(KOK / "data/raw/train.csv")
    print(f"  egitim kolonlari: {list(tr.columns)[:8]}  satir={len(tr)}")
    hedef_kol = [k for k in tr.columns if k.lower() in ("tuketim", "hedef", "target")]
    kol = hedef_kol[0] if hedef_kol else tr.columns[-1]
    y = tr[kol].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    ly = np.log1p(np.clip(y, 0, None))
    print(
        f"  egitim '{kol}': n={len(ly)}  E[log1p(y)^2] = {float((ly**2).mean()):.6f}  "
        f"rms = {float(np.sqrt((ly**2).mean())):.6f}"
    )
    # tarih kolonu varsa son 4 ay
    tk = [k for k in tr.columns if "tarih" in k.lower() or "date" in k.lower()]
    if tk:
        d = pd.to_datetime(tr[tk[0]], errors="coerce")
        son = d >= (d.max() - pd.Timedelta(days=122))
        ly2 = np.log1p(np.clip(tr.loc[son, kol].to_numpy(dtype=float), 0, None))
        ly2 = ly2[np.isfinite(ly2)]
        print(
            f"  egitimin son 4 ayi ({d.max().date()} geriye): n={len(ly2)}  "
            f"rms log1p = {float(np.sqrt((ly2**2).mean())):.6f}"
        )
    icerde = max(alt, 0) <= np.sqrt((ly**2).mean()) <= ust
    print(f"  -> egitim rms {np.sqrt((ly**2).mean()):.4f} sinirlarin ICINDE mi? {icerde}")

    (BURA / "zaman_capa.json").write_text(
        json.dumps(
            {
                "zamansal": sonuc,
                "P0": P0,
                "m0": m0,
                "t_norm_alt": float(max(alt, 0)),
                "t_norm_ust": float(ust),
                "egitim_rms_log1p": float(np.sqrt((ly**2).mean())),
                "c_l2": float(np.linalg.norm(c)),
                "w_l1": float(np.abs(w).sum()),
                "afin_l1": float(abs(afin) + np.abs(w).sum()),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
