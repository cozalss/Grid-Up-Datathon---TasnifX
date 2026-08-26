"""H25 -- b_soguk'un YAS-ESLESTIRILMIS kestirimi (h24'un tamamlanmasi).

NEDEN
-----
h24 test yapisinin analogunu buldu (kis26 soguk, 2026 Sub-Mar: yalniz-gecmis
ufuk VE mevsimsel ikiz egitimde) ve b=+0,2431 olctu.

Takim oturumu hakli bir cekince koydu: kis26 icinde AY, UFUK ve TRAFO YASI
birbirine karisik. Ayrica ucuncu bir okuma daha var -- soguk trafolar blok
boyunca dogdugu icin ay ilerledikce YAS de artiyor.

UFUK OKUMASI ELENDI: surukleme ufukla BIRIKIR, yani b ufuk arttikca
YUKSELMELI. Olculen tam tersi (Ara ufuk 1-31 -> +0,4465 ; Mar ufuk 91-121 ->
+0,2197, DUSUYOR). Ters isaret ongoruyor, elenir.

YAS OKUMASI ELENEMEZ -- ve test'in yas profili analogunkinden FARKLI:
    kis26 Sub medyan yas 46 | Mar 67
    TEST            medyan yas 40  (q10 7, q90 75)
O yuzden b, TEST'in yas dagilimiyla yeniden agirliklandirilmalidir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
HARMAN = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}
KOVA = [(0, 7), (7, 30), (30, 60), (60, 10**9)]
P_SOGUK = 0.22159


def main() -> int:
    z = np.load(KOK / "data/interim/deney/soguk_tahmin_kis26.npz")
    m = pd.read_parquet(KOK / "data/interim/kis26_soguk_meta.parquet").reset_index(drop=True)
    pay = sum(HARMAN.values())
    toh = sorted({k.split("_")[0] for k in z.files})
    tah = {
        t: sum(HARMAN[a] * z[f"{t}_{a}"].astype("float64") for a in HARMAN) / pay
        for t in toh
        if all(f"{t}_{a}" in z.files for a in HARMAN)
    }
    lgy = np.log1p(np.clip(m["y"].to_numpy(dtype="float64"), 0, None))
    yas = (m["tarih"] - m.groupby("tanim")["tarih"].transform("min")).dt.days.to_numpy()
    ay = m["tarih"].dt.to_period("M").astype(str).to_numpy()
    ikiz = np.isin(ay, ["2026-02", "2026-03"])

    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tr = set(
        pd.read_csv(KOK / "data/raw/train.csv", usecols=["tanim"], dtype={"tanim": str})[
            "tanim"
        ].unique()
    )
    tc = te[~te["tanim"].isin(tr)].copy()
    tc["yas"] = (tc["tarih"] - tc.groupby("tanim")["tarih"].transform("min")).dt.days

    print("=" * 84)
    print("YAS-ESLESTIRILMIS b  (analog: kis26 soguk Sub-Mar, agirlik: TEST yas dagilimi)")
    print("=" * 84)
    print(
        f"\n  {'yas kovasi':<12} {'TEST payi':>10} {'analog n':>9} {'b':>9} {'SH':>8} {'katki':>10}"
    )
    tot = w_kul = 0.0
    eksik = []
    for lo, hi in KOVA:
        ad = f"{lo}-{'+' if hi > 10**8 else hi}"
        w = float(((tc["yas"] >= lo) & (tc["yas"] < hi)).mean())
        s = ikiz & (yas >= lo) & (yas < hi)
        if int(s.sum()) < 300:
            print(f"  {ad:<12} {w:>10.4f} {int(s.sum()):>9,}   analogda yetersiz")
            eksik.append((ad, w))
            continue
        per = [float((lgy[s] - v[s]).mean()) for v in tah.values()]
        b = float(np.mean(per))
        sh = float(np.std(per, ddof=1) / np.sqrt(len(per)))
        tot += w * b
        w_kul += w
        print(f"  {ad:<12} {w:>10.4f} {int(s.sum()):>9,} {b:>+9.4f} {sh:>8.4f} {w * b:>10.5f}")

    b_yas = tot / w_kul if w_kul else float("nan")
    print(f"\n  kapsanan TEST payi {w_kul:.4f}" + (f"  (kapsanmayan: {eksik})" if eksik else ""))
    print(f"  >>> YAS-ESLESTIRILMIS b = {b_yas:+.4f}")
    print("      duz analog (Sub-Mar tumu) = +0,2431")
    print(f"      fark {b_yas - 0.2431:+.4f}")

    print("\n" + "=" * 84)
    print("SECILEN delta = 0,22 -- HALA DOGRU MU?")
    print("=" * 84)
    print(f"  {'E[b]':>8} {'delta=0,16':>12} {'delta=0,22':>12} {'delta=0,25':>12}")
    for eb in (0.145, 0.1764, b_yas, 0.2431, 0.30):
        s = f"  {eb:>8.4f}"
        for dl in (0.16, 0.22, 0.25):
            s += f" {-(2 * P_SOGUK * dl * eb - P_SOGUK * dl**2):>+12.5f}"
        print(s)
    print("\n  (NEGATIF = kazanc)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
