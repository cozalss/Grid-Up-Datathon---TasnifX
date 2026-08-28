"""TOPLAM YENI DIK ENERJI MUHASEBESI.

Iki ayri soru var, karistirilmamali:

  (a) URETILEN yon envanteri ne kadar YENI enerji tasiyor?
      Sekillendirme oncesi HAM yonler (log1p(aday) - log1p(v102)), span'dan ve
      mevcut 9 dik boyuttan arindirilip acgozlu ortogonallestirilir.

  (b) Bu enerjinin ne kadari KAZANCA doner?
      ``kazanc = kappa^2 * Q``. ``kappa`` LB probuyla cozulecek; simdilik tek
      ampirik tutamak CV'de olculen ``kappa``dir. Gorev tanimindaki f=0.4115
      mevcut envanterin LB'de gerceklesme oraniydi -- bu adaylar icin GECERLI
      OLDUGU VARSAYILAMAZ, o yuzden tablo birden cok f'te veriliyor.

Kaggle'a hicbir sey gondermez.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

import c_olc
import d_varyant
import ortak

ACIK = 0.0202  # 1.lik icin v102'den sonra kalan dMSE acigi


def main() -> None:
    g = ortak.geo()
    tab = c_olc.uretim_tabani()
    mcv = d_varyant._meta_cv()
    E, _ = g.envanter()

    adaylar = sorted(p.stem for p in ortak.ONB.glob("*.npz"))
    print("=" * 112)
    print("(a) HAM ADAY YONLERI -- span + mevcut 9 dik envanter cikarilmis YENI enerji")
    print("=" * 112)
    print(
        f"{'aday':22}{'Q':>10}{'q_perp':>10}{'q_YENI':>10}{'kum':>10}"
        f"{'kappa_CV':>10}{'CV kazanc':>11}  maks kos"
    )
    taban = list(E)
    top = 0.0
    kayit = []
    for ad in adaylar:
        cv, tp = ortak.yukle_aday(ad)
        u = np.log1p(np.clip(tp, 0.0, None)) - g.v102
        r = g.olc(ad, u)
        v, _ = g.perp(u)
        for e in taban:
            v = v - (float(v @ e) / g.n) * e
        qy = float(v @ v) / g.n
        if qy > 5e-6:
            taban.append(v / np.sqrt(qy))
            top += qy
        L = Q = 0.0
        n = 0
        for b in ortak.BLOKLAR:
            d = np.log1p(np.clip(cv[b], 0.0, None)) - tab[b]["taban"]
            rr = tab[b]["lgy"] - tab[b]["taban"]
            L += float(rr @ d)
            Q += float(d @ d)
            n += len(d)
        kap = (L / Q) if Q > 0 else 0.0
        kayit.append({"ad": ad, "q_yeni": qy, "q_perp": r["q_perp"], "kappa_cv": kap})
        print(
            f"{ad:22}{r['Q']:>10.5f}{r['q_perp']:>10.5f}{qy:>10.5f}{top:>10.5f}"
            f"{kap:>+10.3f}{kap * kap * r['q_perp']:>11.5f}"
            f"  {r['maks_kos_ad']} {r['maks_kos']:+.2f}"
        )
    print(f"\nTOPLAM YENI DIK ENERJI (8 aday, {len(taban) - len(E)} yeni boyut) = {top:.6f}")
    print("Mevcut envanter (9 boyut, gorev tanimi)                          = 0.045253")
    print(
        f"Oran                                                             = {top / 0.0452529:.2f}x"
    )

    print("\n" + "=" * 112)
    print("(b) ACIK KAPATMA -- kazanc = f^2 * Q_YENI,  acik = 0.0202")
    print("=" * 112)
    print(f"{'f (gerceklesme orani)':28}{'kazanc':>12}{'acigin %':>12}  not")
    notlar = {
        0.05: "adaylarin CV kappa'sinin tipik buyuklugu",
        0.10: "",
        0.20: "",
        0.30: "",
        0.4115: "GOREV TANIMI: mevcut envanterin LB'de olculen orani",
        1.00: "kuramsal tavan (yon gercekle TAM hizali)",
    }
    for f in sorted(notlar):
        kaz = f * f * top
        print(f"f = {f:<24.4f}{kaz:>12.6f}{100 * kaz / ACIK:>11.1f}%  {notlar[f]}")

    json.dump(
        {
            "adaylar": kayit,
            "toplam_yeni_enerji": top,
            "boyut": len(taban) - len(E),
            "acik": ACIK,
            "kapatma": {str(f): f * f * top / ACIK for f in notlar},
        },
        open(ortak.CIK / "h_ozet.json", "w"),
        indent=2,
        default=float,
    )


if __name__ == "__main__":
    main()
