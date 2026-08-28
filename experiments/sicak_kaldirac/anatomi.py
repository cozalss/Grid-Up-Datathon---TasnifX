"""SICAK KOHORT HATA ANATOMISI -- uc blokta, kis26 agirlikli okunacak."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, SICAK_PAY, bloklari_kur, mse, taban_r  # noqa: E402

pd.set_option("display.width", 220)


def main() -> int:
    bl = bloklari_kur()
    taban = {k: taban_r(b) for k, b in bl.items()}

    print("=" * 100)
    print("0) TABAN -- ham harman ve uretim zinciri")
    print("=" * 100)
    print(
        f"{'blok':7}{'n':>9}{'trafo':>7}{'hamMSE':>9}{'hamRMSLE':>10}"
        f"{'tabanMSE':>10}{'tabanRMSLE':>12}{'kuresel_delta':>15}"
    )
    for k in BLOKLAR:
        b = bl[k]
        rh = np.mean(b.tohum_harman, axis=0) - b.lgc
        mh = mse(b, rh)
        mt = mse(b, taban[k])
        d = float((b.lgy - b.lgc - rh).mean())
        print(
            f"{k:7}{b.n:>9,}{b.cerceve['tanim'].nunique():>7,}{mh:>9.4f}"
            f"{np.sqrt(mh):>10.5f}{mt:>10.4f}{np.sqrt(mt):>12.5f}{d:>+15.5f}"
        )

    print()
    print("=" * 100)
    print("1) HATA AYRISIMI -- MSE = yanlilik^2 + trafo-arasi + trafo-ici")
    print("=" * 100)
    for k in BLOKLAR:
        b = bl[k]
        e = b.lgy - (taban[k] + b.lgc)
        d = pd.DataFrame({"t": b.cerceve["tanim"].to_numpy(), "e": e})
        tort = d.groupby("t")["e"].transform("mean")
        v_arasi = float(((tort - e.mean()) ** 2).mean())
        v_ici = float(((e - tort) ** 2).mean())
        m = float((e * e).mean())
        print(
            f"{k:7} MSE {m:7.4f} = yanlilik^2 {e.mean() ** 2:7.5f} "
            f"({100 * e.mean() ** 2 / m:4.1f}%) + trafo-arasi {v_arasi:6.4f} "
            f"({100 * v_arasi / m:4.1f}%) + trafo-ici {v_ici:6.4f} ({100 * v_ici / m:4.1f}%)"
        )

    print()
    print("=" * 100)
    print("2) YOGUNLASMA -- en kotu trafolar / satirlar hatanin yuzde kacini tasiyor")
    print("=" * 100)
    for k in BLOKLAR:
        b = bl[k]
        e2 = (b.lgy - (taban[k] + b.lgc)) ** 2
        g = pd.DataFrame({"t": b.cerceve["tanim"].to_numpy(), "e2": e2}).groupby("t")["e2"].sum()
        g = g.sort_values(ascending=False)
        n = len(g)
        tot = g.sum()
        p = [
            f"%{100 * g.iloc[: max(1, int(n * q))].sum() / tot:5.1f}"
            for q in (0.01, 0.05, 0.10, 0.25)
        ]
        s = np.sort(e2)[::-1]
        sp = [f"%{100 * s[: max(1, int(len(s) * q))].sum() / s.sum():5.1f}" for q in (0.01, 0.10)]
        print(
            f"{k:7} trafo {n:>5,}  TRAFO en kotu %1 {p[0]} %5 {p[1]} %10 {p[2]} %25 {p[3]}"
            f"  | SATIR en kotu %1 {sp[0]} %10 {sp[1]}"
        )

    print()
    print("=" * 100)
    print("3) HEDEF DEGERI KIRILIMI -- y=0 / ara deger / normal")
    print("=" * 100)
    for k in BLOKLAR:
        b = bl[k]
        e = b.lgy - (taban[k] + b.lgc)
        e2 = e * e
        gr = np.where(
            b.y <= 0, "y=0", np.where(b.y < 10, "0<y<10", np.where(b.y < 100, "10..100", "y>=100"))
        )
        d = pd.DataFrame({"g": gr, "e": e, "e2": e2})
        t = d.groupby("g").agg(n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"))
        t["pay%"] = 100 * d.groupby("g")["e2"].sum() / d["e2"].sum()
        t["satir%"] = 100 * t["n"] / len(d)
        print(f"-- {k} --")
        print(t.round(4).to_string())

    print()
    print("=" * 100)
    print("4) KIRILIMLAR")
    print("=" * 100)
    for k in BLOKLAR:
        b = bl[k]
        c = b.cerceve
        e = b.lgy - (taban[k] + b.lgc)
        d = pd.DataFrame(
            {
                "kova": c["kova"].to_numpy(),
                "il": c["il"].to_numpy(),
                "ilce": c["ilce"].to_numpy(),
                "ay": c["ay"].to_numpy(),
                "hg": c["hg"].to_numpy(),
                "kuyruk": c["kuyruk"].to_numpy(),
                "gecmis": c["gecmis_gun"].to_numpy(),
                "e": e,
                "e2": e * e,
                "sifir": (b.y == 0).astype(float),
            }
        )
        # seviye desili: trafonun gecmis log ortalamasi
        d["seviye"] = pd.qcut(c["t_log_ort"].to_numpy(), 10, labels=False, duplicates="drop")
        for ad, anah in (
            ("kVA kovasi", "kova"),
            ("il", "il"),
            ("ay", "ay"),
            ("haftagunu", "hg"),
            ("seviye desili (t_log_ort)", "seviye"),
        ):
            g = d.groupby(anah, observed=True).agg(
                n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"), sifir=("sifir", "mean")
            )
            g["pay%"] = 100 * d.groupby(anah, observed=True)["e2"].sum() / d["e2"].sum()
            print(f"\n-- {k} : {ad} --")
            print(g.round(4).to_string())
        kes = pd.cut(
            d["gecmis"],
            [-1, 6, 30, 90, 180, 9999],
            labels=["<=6", "7-30", "31-90", "91-180", "180+"],
        )
        g = d.groupby(kes, observed=True).agg(
            n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"), sifir=("sifir", "mean")
        )
        g["pay%"] = 100 * d.groupby(kes, observed=True)["e2"].sum() / d["e2"].sum()
        print(f"\n-- {k} : gecmis uzunlugu --")
        print(g.round(4).to_string())

    print()
    print("=" * 100)
    print("5) OLCEK -- gercek r'nin taban r'sine OLS egimi (>1 fazla buzulmus)")
    print("=" * 100)
    for k in BLOKLAR:
        b = bl[k]
        rg = b.lgy - b.lgc
        for et, rr in (("ham", np.mean(b.tohum_harman, axis=0) - b.lgc), ("taban", taban[k])):
            x = rr - rr.mean()
            egim = float((x * (rg - rg.mean())).sum() / (x * x).sum())
            kor = float(np.corrcoef(rr, rg)[0, 1])
            print(
                f"{k:7} {et:6} egim {egim:6.3f} kor {kor:5.3f} "
                f"std_tah {rr.std():6.4f} std_ger {rg.std():6.4f}"
            )

    print()
    print(f"NOT: sicak dMSE -> test genel dMSE carpani {SICAK_PAY:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
