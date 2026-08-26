"""CURUTUCU B -- bulgunun KENDI ana iddiasini KURAL 1 ile sinar.

Bulgu diyor ki "model olu satirlarda tek sabitten iyi". O ustunluk de trafo
bazinda ayristirilmali: birkac trafodan geliyorsa iddia da kirilgan.

Ayrica:
  * ufka gore kirilim (test ufku 1-122; kural: mevsimsel ikiz yaz25 dahil)
  * dirilme oranlari (mekanizma)
  * v55'te TAM SIFIR tahmin satiri kaldi mi (olu_hedge'in kapsami)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

from curut_olu_a import BLOKLAR, KLER, kuyruk_kovasi, rmse, veri_yukle  # noqa: E402


def main() -> int:
    t0 = time.time()
    veri = veri_yukle()

    print("=" * 100)
    print("B1) 'MODEL > TEK SABIT' USTUNLUGU TRAFO BAZINDA -- KURAL 1 simetrik uygulama")
    print(
        "    (sabit = in-sample orakul ortalama; modelin ustunlugune EN COK katan K trafo atiliyor)"
    )
    print("=" * 100)
    print(f"  {'blok':7}{'trafo':>8}" + "".join(f"{'K=' + str(k):>11}" for k in KLER))
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        ly, lt, tn = v["ly"][m], v["log_t"][m], v["tanim"][m]
        c = ly.mean()
        e_mod = (lt - ly) ** 2
        e_sab = (c - ly) ** 2
        # sabit LEHINE fark: negatifse model daha iyi
        d = e_mod - e_sab
        katki = pd.DataFrame({"tanim": tn, "d": d}).groupby("tanim")["d"].sum().sort_values()
        satir = f"  {b:7}{katki.size:8,}"
        for K in KLER:
            at = set(katki.index[:K])
            tut = ~pd.Series(tn).isin(at).to_numpy()
            # pozitif = tek sabit daha iyi olurdu (model kaybediyor)
            satir += f"{rmse(e_sab[tut]) - rmse(e_mod[tut]):+11.5f}"
        print(satir)
    print("  (deger POZITIF = model o alt kumede tek sabitten IYI)")

    print()
    print("=" * 100)
    print("B2) UFKA GORE: olu satirlarda model vs tek sabit (in-sample orakul sabit)")
    print("=" * 100)
    dilim = [(1, 31), (31, 61), (61, 91), (91, 123)]
    print(
        f"  {'blok':7}{'dilim':>10}{'satir':>9}{'dirilme %':>11}{'model MSE':>11}"
        f"{'sabit MSE':>11}{'fark':>10}"
    )
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        ly, lt, uf = v["ly"][m], v["log_t"][m], v["ufuk"][m]
        for a, z in dilim:
            s = (uf >= a) & (uf < z)
            if s.sum() < 50:
                continue
            c = ly[s].mean()
            em = ((lt[s] - ly[s]) ** 2).mean()
            es = ((c - ly[s]) ** 2).mean()
            print(
                f"  {b:7}{f'{a}-{z - 1}':>10}{int(s.sum()):9,}{100 * (ly[s] > 0).mean():11.2f}"
                f"{em:11.5f}{es:11.5f}{es - em:+10.5f}"
            )

    print()
    print("=" * 100)
    print("B3) KUYRUK KOVASINA GORE dirilme ve model/sabit")
    print("=" * 100)
    etiket = ["1-14", "15-29", "30-59", "60-89", "90+"]
    print(
        f"  {'blok':7}{'kova':>8}{'satir':>9}{'trafo':>7}{'dirilme %':>11}"
        f"{'model ort log1p':>17}{'gercek ort log1p':>18}{'model MSE':>11}{'sabit MSE':>11}"
    )
    for b in BLOKLAR:
        v = veri[b]
        m = v["olu"]
        ly, lt, tn = v["ly"][m], v["log_t"][m], v["tanim"][m]
        kv = kuyruk_kovasi(v["kuyruk"][m])
        for k in range(5):
            s = kv == k
            if s.sum() == 0:
                continue
            c = ly[s].mean()
            print(
                f"  {b:7}{etiket[k]:>8}{int(s.sum()):9,}{len(np.unique(tn[s])):7,}"
                f"{100 * (ly[s] > 0).mean():11.2f}{lt[s].mean():17.4f}{ly[s].mean():18.4f}"
                f"{((lt[s] - ly[s]) ** 2).mean():11.5f}{((c - ly[s]) ** 2).mean():11.5f}"
            )

    print()
    print("=" * 100)
    print("B4) GONDERIMLERDE TAM SIFIR TAHMIN SATIRI (olu_hedge kapsami)")
    print("=" * 100)
    for ad in ("tuketim_v55_gunolcek.csv", "tuketim_v50_nihai30.csv"):
        p = KOK / "submissions" / ad
        if not p.exists():
            continue
        s = pd.read_csv(p, encoding="utf-8")
        sifir = (s["tuketim"] <= 0).sum()
        print(
            f"  {ad:32} satir {len(s):,}  tam sifir {sifir:,}  min {s['tuketim'].min():.6f}"
            f"  <0.01 olan {(s['tuketim'] < 0.01).sum():,}"
        )

    print(f"\nTAMAM  {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
