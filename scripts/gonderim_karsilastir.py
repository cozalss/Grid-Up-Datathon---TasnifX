"""IKI GONDERIM DOSYASINI KARSILASTIRIR -- fark nerede, ne kadar, riziko ne.

NEDEN BU BETIK
--------------
Gonderim hakki gunde 3. Her hak, bir SORUYA cevap almak icin harcanir ve
sorunun cevabini okuyabilmek icin iki dosyanin TAM OLARAK nerede ayristigini
bilmek gerekir. "Iki dosya farkli" yetmez:

  - Fark yalnizca 8.748 satirdaysa  -> o satirlarin mudahalesi izole olculur.
  - Fark 162.016 satirdaysa         -> tabanlar arasindaki fark olculur.

2026-08-23'te bu betik bir VARSAYIMI kanita cevirdi: v20 ile v23'un tam-sifir
kumeleri BIREBIR ayni cikti (8.748 / 8.748 / kesisim 8.748). Bu, hedge'in iki
taban arasinda tasinabilir oldugu anlamina gelir ve gonderim planinin
"v26 - hedge = v20'nin ortuk skoru" cikarimini dogrular.

    python scripts/gonderim_karsilastir.py submissions/A.csv submissions/B.csv
    python scripts/gonderim_karsilastir.py A.csv B.csv --taban-lb 1.0312

RIZIKO HESABI (--taban-lb)
--------------------------
Sifir tahminlerini pozitif bir tabanla degistirmek, trafo GERCEKTEN oluyse
zarardir. En kotu hal kapali formulle hesaplanir:

    RMSLE_yeni = sqrt(RMSLE_taban^2 + sum(log1p(yeni_deger)^2) / N)

Bu bir tahmin degil, ust sinir: tum degisen satirlarin gercek degeri 0 kabul
edilir. Ölculen dirilis oranlari (%3-41) bu sinirin gerceklesmeyecegini soyler
ama sinirin kendisi kesindir.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

#: Bit duzeyinde "ayni" sayilma esigi. log1p/expm1 gidis-donusu ~1e-16
#: artik birakir; bunu "degisiklik" saymak her satiri farkli gosterirdi.
AYNI_TOLERANS = 1e-12


def yukle(yol: Path) -> pd.DataFrame:
    d = pd.read_csv(yol, encoding="utf-8")
    if list(d.columns) != ["id", "tuketim"]:
        raise SystemExit(f"{yol}: beklenen kolonlar ['id','tuketim'], gelen {list(d.columns)}")
    if d["tuketim"].isna().any():
        raise SystemExit(f"{yol}: NaN iceriyor")
    if (d["tuketim"] < 0).any():
        raise SystemExit(f"{yol}: negatif deger iceriyor")
    return d


def main() -> int:
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("bir")
    a.add_argument("iki")
    a.add_argument(
        "--taban-lb",
        type=float,
        default=None,
        help="En kotu hal riziko hesabi yapilir (ilk dosya taban kabul edilir)",
    )
    ar = a.parse_args()

    p1, p2 = Path(ar.bir), Path(ar.iki)
    d1, d2 = yukle(p1), yukle(p2)

    print(f"A = {p1.name}   {len(d1):,} satir")
    print(f"B = {p2.name}   {len(d2):,} satir")
    if len(d1) != len(d2):
        raise SystemExit("satir sayilari farkli -- karsilastirilamaz")
    if not (d1["id"].to_numpy() == d2["id"].to_numpy()).all():
        raise SystemExit("id sirasi farkli -- karsilastirilamaz")
    print("id sirasi: AYNI")

    v1, v2 = d1["tuketim"].to_numpy(), d2["tuketim"].to_numpy()
    z1, z2 = v1 == 0, v2 == 0
    kesisim = int((z1 & z2).sum())
    print("\nTAM-SIFIR KUMELERI")
    print(f"  A          {int(z1.sum()):>8,}")
    print(f"  B          {int(z2.sum()):>8,}")
    print(f"  kesisim    {kesisim:>8,}")
    print(f"  yalniz A   {int((z1 & ~z2).sum()):>8,}")
    print(f"  yalniz B   {int((z2 & ~z1).sum()):>8,}")
    if z1.any() and z1.sum() == z2.sum() == kesisim:
        print("  -> kumeler BIREBIR AYNI: bu satirlardaki mudahale iki dosya arasinda tasinabilir")

    degisen = ~np.isclose(v1, v2, rtol=AYNI_TOLERANS, atol=AYNI_TOLERANS)
    n = len(v1)
    print("\nFARK")
    print(f"  degisen satir  {int(degisen.sum()):>8,}  (%{100 * degisen.mean():.2f})")
    print(f"  ayni satir     {int((~degisen).sum()):>8,}")
    if degisen.any():
        f = np.log1p(v1) - np.log1p(v2)
        print(f"  ortalama |log farki|  {np.abs(f).mean():.5f}")
        print(f"  RMS log farki         {np.sqrt((f**2).mean()):.5f}")
        # mudahale yalnizca A'nin sifirlarinda mi
        disari = int((degisen & ~z1).sum())
        print(
            f"  A'nin sifir olmayan satirlarinda degisim: {disari:,}"
            + ("   -> mudahale IZOLE" if disari == 0 else "   -> mudahale izole DEGIL")
        )

    if ar.taban_lb is not None:
        hedef = z1 & degisen
        if not hedef.any():
            print("\nRIZIKO: A'nin sifirlarinda degisim yok -- hesap uygulanamaz")
            return 0
        ek = float((np.log1p(v2[hedef]) ** 2).sum())
        yeni = float(np.sqrt(ar.taban_lb**2 + ek / n))
        print(f"\nEN KOTU HAL  (degisen {int(hedef.sum()):,} satirin HEPSI gercekten olu ise)")
        print(f"  ek kare toplami   {ek:.2f}")
        print(f"  {ar.taban_lb:.5f}  ->  {yeni:.5f}   (+{yeni - ar.taban_lb:.5f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
