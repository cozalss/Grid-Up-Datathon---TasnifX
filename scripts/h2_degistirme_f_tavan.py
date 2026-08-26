"""H2-F: TAVAN -- bu eslestirme kurali EN IYI ihtimalle ne kazandirir?

E adiminda uc blokta da sinyal sifir cikti (|t|<1, isaret tutarsiz). Burada
"sifir" ne kadar sikica olculdu onu ciziyoruz: havuzlanmis korelasyonun %95
ust sinirini alip, coverage ve gercek soguk-taraf hata seviyesiyle carparak
IYIMSER bir tavan uretiyoruz. Tavan bile hedefin (-0,019333) yaninda kaliyorsa
hipotez kapanir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

KOK = "C:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"
P_SOGUK = 0.22159
MSE_TOPLAM = 1.032073
SOGUK_MSE_PAYI = 0.63
HEDEF_DMSE = -0.019333

# E ciktisindan (mekanizmaya en uygun alt kume: olum dogumdan <=60 gun once)
YAKIN60 = {"yaz25": (161, 0.0066), "guz25": (382, 0.0029), "kis26": (269, 0.0073)}
TUM = {"yaz25": (246, -0.0615), "guz25": (764, -0.0193), "kis26": (687, 0.0235)}


def havuzla(d: dict, ad: str) -> None:
    # Fisher z ile havuzla
    zs, ws = [], []
    for blok, (n, r) in d.items():
        zs.append(np.arctanh(r))
        ws.append(n - 3)
    z = float(np.average(zs, weights=ws))
    se = float(1 / np.sqrt(sum(ws)))
    r_hav = np.tanh(z)
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    n_top = sum(n for n, _ in d.values())
    print(f"{ad}: n={n_top}  r_havuz={r_hav:+.4f}  %95 GA [{lo:+.4f}, {hi:+.4f}]  t={z / se:+.2f}")
    return hi


def main() -> None:
    print("=== HAVUZLANMIS KORELASYON (uc blok, Fisher z) ===")
    hi_yakin = havuzla(YAKIN60, "yakin60 (mekanizmaya uygun)")
    hi_tum = havuzla(TUM, "tum eslesmeler        ")

    # Soguk tarafin GERCEK per-satir MSE'si (sampiyon modeli)
    soguk_mse = MSE_TOPLAM * SOGUK_MSE_PAYI / P_SOGUK
    print(
        f"\nsampiyon soguk taraf per-satir MSE = {soguk_mse:.4f}  (RMSLE {np.sqrt(soguk_mse):.4f})"
    )

    # Kapsama: testte kac soguk trafonun TEKIL olu adayi var
    soguk = pd.read_parquet(f"{KOK}/data/interim/h2_soguk_ozet.parquet")
    olu = pd.read_parquet(f"{KOK}/data/interim/h2_olu_havuz.parquet")
    olu_say = olu["anahtar"].value_counts()
    soguk["aday"] = olu_say.reindex(soguk["anahtar"]).fillna(0).astype(int).to_numpy()
    kaps_tekil = float((soguk["aday"] == 1).mean())
    kaps_herhangi = float((soguk["aday"] >= 1).mean())
    print(f"test soguk kapsama: tekil aday {kaps_tekil:.3f}  en az bir aday {kaps_herhangi:.3f}")

    print("\n=== TAVAN (IYIMSER: %95 ust sinir, tam kapsama varsayimi) ===")
    for ad, hi in (("yakin60", hi_yakin), ("tum", hi_tum)):
        for kaps_ad, kaps in (("tekil", kaps_tekil), ("herhangi", kaps_herhangi)):
            # aciklanan varyans payi r^2, yalnizca SEVIYE bileseninde
            d_soguk = -(hi**2) * soguk_mse * kaps
            d_test = P_SOGUK * d_soguk
            print(
                f"  {ad:8s} kapsama={kaps_ad:8s} r_ust={hi:+.4f}  dMSE_test = {d_test:+.6f}"
                f"   hedefin %{100 * abs(d_test / HEDEF_DMSE):.1f}'i"
            )

    print("\nNOT: bu tavan seviye hatasinin TAMAMININ aciklanabilir oldugunu")
    print("varsayiyor (model hicbir sey bilmiyormus gibi). Gercek olculen deger")
    print("uc blokta da SIFIR ve isaret tutarsiz.")


if __name__ == "__main__":
    main()
