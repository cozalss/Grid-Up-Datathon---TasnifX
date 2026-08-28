"""SOGUK KOHORT HATA ANATOMISI -- yaz25 odakli, uc blokta karsilastirmali."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ortak import BLOKLAR, SOGUK_PAY, mse, taban_r, tum_bloklar  # noqa: E402

pd.set_option("display.width", 200)


def kova(guc: np.ndarray) -> np.ndarray:
    kenar = [0, 60, 110, 175, 260, 410, 650, 1050, 1e9]
    et = ["<=50", "100", "160", "250", "400", "630", "1000", ">1000"]
    return np.asarray(et)[np.digitize(guc, kenar) - 1]


def main() -> int:
    bloklar = tum_bloklar()

    print("=" * 78)
    print("1) TABAN -- uretim son islemi (cat, beta=0,60, delta=0,1046)")
    print("=" * 78)
    print(
        f"{'blok':6} {'n':>7} {'trafo':>6} {'MSE':>9} {'RMSLE':>8} "
        f"{'yanlilik':>9} {'std(hata)':>10} {'std(r_tah)':>11} {'std(r_ger)':>11}"
    )
    for ad in BLOKLAR:
        b = bloklar[ad]
        r = taban_r(b)
        e = b.lgy - (r + b.lgc)
        rg = b.lgy - b.lgc
        print(
            f"{ad:6} {b.n:>7,} {len(set(b.tanim)):>6,} {mse(b, r):>9.4f} "
            f"{np.sqrt(mse(b, r)):>8.4f} {e.mean():>+9.4f} {e.std():>10.4f} "
            f"{r.std():>11.4f} {rg.std():>11.4f}"
        )

    print()
    print("=" * 78)
    print("2) HATA AYRISIMI -- MSE = yanlilik^2 + var; var = trafo-arasi + trafo-ici")
    print("=" * 78)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r = taban_r(b)
        e = b.lgy - (r + b.lgc)
        d = pd.DataFrame({"t": b.tanim, "e": e})
        tort = d.groupby("t")["e"].transform("mean")
        v_arasi = float(((tort - e.mean()) ** 2).mean())
        v_ici = float(((e - tort) ** 2).mean())
        m = float((e * e).mean())
        print(
            f"{ad:6} MSE {m:7.4f} = yanlilik^2 {e.mean() ** 2:6.4f} "
            f"({100 * e.mean() ** 2 / m:4.1f}%) + trafo-arasi {v_arasi:6.4f} "
            f"({100 * v_arasi / m:4.1f}%) + trafo-ici {v_ici:6.4f} ({100 * v_ici / m:4.1f}%)"
        )

    print()
    print("=" * 78)
    print("3) YOGUNLASMA -- en kotu trafolar hatanin yuzde kacini tasiyor")
    print("=" * 78)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r = taban_r(b)
        e2 = (b.lgy - (r + b.lgc)) ** 2
        d = pd.DataFrame({"t": b.tanim, "e2": e2})
        g = d.groupby("t")["e2"].sum().sort_values(ascending=False)
        tot = g.sum()
        n = len(g)
        pay = [
            f"%{100 * g.iloc[: max(1, int(n * q))].sum() / tot:5.1f}"
            for q in (0.01, 0.05, 0.10, 0.25)
        ]
        print(f"{ad:6} trafo {n:>5,}  en kotu %1 {pay[0]}  %5 {pay[1]}  %10 {pay[2]}  %25 {pay[3]}")

    print()
    print("=" * 78)
    print("4) YAZ25 KIRILIMLARI (test'in mevsim ikizi)")
    print("=" * 78)
    for ad in BLOKLAR:
        b = bloklar[ad]
        r = taban_r(b)
        e = b.lgy - (r + b.lgc)
        d = pd.DataFrame(
            {
                "kova": kova(b.guc),
                "il": b.il,
                "ilce": b.ilce,
                "yas": b.yas,
                "ay": pd.to_datetime(b.tarih).month,
                "hg": pd.to_datetime(b.tarih).dayofweek,
                "e": e,
                "e2": e * e,
                "sifir": (b.y == 0).astype(float),
            }
        )
        print(f"\n-- {ad} : kVA kovasi --")
        g = d.groupby("kova").agg(
            n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"), sifir=("sifir", "mean")
        )
        g["pay%"] = 100 * d.groupby("kova")["e2"].sum() / d["e2"].sum()
        print(g.sort_values("pay%", ascending=False).round(4).to_string())

        print(f"-- {ad} : il --")
        g = d.groupby("il").agg(n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"))
        g["pay%"] = 100 * d.groupby("il")["e2"].sum() / d["e2"].sum()
        print(g.round(4).to_string())

        print(f"-- {ad} : panele giris yasi (gun) --")
        kes = pd.cut(
            d["yas"],
            [-1, 0, 6, 13, 29, 59, 9999],
            labels=["0", "1-6", "7-13", "14-29", "30-59", "60+"],
        )
        g = d.groupby(kes, observed=True).agg(
            n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"), sifir=("sifir", "mean")
        )
        print(g.round(4).to_string())

        print(f"-- {ad} : ay --")
        g = d.groupby("ay").agg(n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"))
        print(g.round(4).to_string())

        print(f"-- {ad} : haftagunu --")
        g = d.groupby("hg").agg(n=("e", "size"), yanlilik=("e", "mean"), mse=("e2", "mean"))
        print(g.round(4).T.to_string())

    print()
    print("=" * 78)
    print("5) OLCEK HATASI -- gercek r'nin tahmin r'sine OLS egimi (1'den buyukse")
    print("   tahmin FAZLA BUZULMUS, kucukse fazla yayilmis)")
    print("=" * 78)
    for ad in BLOKLAR:
        b = bloklar[ad]
        for etiket, rr in (("ham(cat)", b.ham["cat"] - b.lgc), ("taban", taban_r(b))):
            rg = b.lgy - b.lgc
            x = rr - rr.mean()
            egim = float((x * (rg - rg.mean())).sum() / (x * x).sum())
            kor = float(np.corrcoef(rr, rg)[0, 1])
            print(
                f"{ad:6} {etiket:9} egim {egim:6.3f}  korelasyon {kor:5.3f}  "
                f"std_tah {rr.std():5.3f} std_ger {rg.std():5.3f}"
            )

    print()
    print(f"NOT: genel (test) MSE'ye cevrim = SOGUK dMSE x {SOGUK_PAY:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
