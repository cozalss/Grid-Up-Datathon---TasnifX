"""YAS BANDI YONU -- her blokta OPTIMUM b, ve prob beklenen degeri.

kuyruk_toplu.py: yanlilik yas bandina gore BLOK ICINDE cok anlamli
(t = 40..67) ama SEKLI bloklar arasi tasinmiyor. Kriter geregi bu
|rho| ~ 0 degil, ISARET TUTARSIZ -- yani PROB ADAYI.

Bu betik her mask icin blok basina optimum b'yi cikarir; prob beklenen
kazanci  E[Q * b^2]  ile hesaplanir (kappa* = b, tavan = -Q*b^2).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))

from ortak import BLOKLAR, SICAK_PAY, bloklari_kur, kuresel_delta, mse, taban_r  # noqa: E402

CIK = Path(__file__).resolve().parent
# testteki Q_dik degerleri (kuyruk_prob_tasarim.py'den, TAM TEST kumesinde)
Q_TEST = {
    "kuyruk <=6g": 0.0417987,
    "genc 7-30g": 0.0193792,
    "genc 7-90g": 0.0741251,
    "genc 7-180g": None,  # asagida hesaplanacak degil; bilgi amacli
}


def opt_b(b, r0, m: np.ndarray) -> tuple[float, float]:
    en, enm = 0.0, None
    for d in np.arange(-0.60, 0.601, 0.02):
        rr = r0 + d * m
        v = mse(b, rr + kuresel_delta(b, rr))
        if enm is None or v < enm:
            en, enm = float(d), v
    taban_m = mse(b, r0 + kuresel_delta(b, r0))
    return en, enm - taban_m


def main() -> int:
    bloklar = bloklari_kur()
    taban = {k: taban_r(bloklar[k]) for k in BLOKLAR}

    maskeler = {
        "kuyruk <=6g": lambda g: g <= 6,
        "genc 7-30g": lambda g: (g > 6) & (g <= 30),
        "genc 7-90g": lambda g: (g > 6) & (g <= 90),
        "genc 7-180g": lambda g: (g > 6) & (g <= 180),
        "yasli >180g": lambda g: g > 180,
    }

    print("=" * 104)
    print("BLOK BASINA OPTIMUM b (seviye-notr, kirpmali).  kappa* = b, tavan = -Q*b^2")
    print("=" * 104)
    print(
        f"{'maske':16}{'yaz25 b':>12}{'guz25 b':>12}{'kis26 b':>12}"
        f"{'|b| ort':>10}{'b RMS':>10}{'isaret':>10}"
    )
    print("-" * 104)
    sonuc = []
    for ad, f in maskeler.items():
        bs = {}
        ns = {}
        for k in BLOKLAR:
            bl = bloklar[k]
            m = f(bl.cerceve["gecmis_gun"].to_numpy()).astype("float64")
            ns[k] = int(m.sum())
            bs[k] = opt_b(bl, taban[k], m)[0] if m.sum() > 0 else float("nan")
        v = np.array([bs[k] for k in BLOKLAR], dtype="float64")
        vv = v[~np.isnan(v)]
        isaret = "AYNI" if (vv > 0).all() or (vv < 0).all() else "TERS"
        print(
            f"{ad:16}{bs['yaz25']:>+12.3f}{bs['guz25']:>+12.3f}{bs['kis26']:>+12.3f}"
            f"{np.abs(vv).mean():>10.3f}{np.sqrt((vv**2).mean()):>10.3f}{isaret:>10}"
        )
        sonuc.append(
            {
                "maske": ad,
                **{f"b_{k}": bs[k] for k in BLOKLAR},
                **{f"n_{k}": ns[k] for k in BLOKLAR},
                "b_rms": float(np.sqrt((vv**2).mean())),
                "isaret": isaret,
            }
        )

    print("\n" + "=" * 104)
    print("PROB BEKLENEN DEGERI  (test Q_dik x b^2);  guz25 ve kis26 kestirimleri kullanildi")
    print("(yaz25 kuyruk bandinda YALNIZ 4 trafo gorur -- o kestirim disarida)")
    print("=" * 104)
    for s in sonuc:
        Q = Q_TEST.get(s["maske"])
        if Q is None:
            continue
        bb = [s["b_guz25"], s["b_kis26"]]
        b2 = float(np.mean([x * x for x in bb]))
        print(
            f"  {s['maske']:14} Q_dik(test) {Q:.6f}   b: "
            f"{bb[0]:+.3f} / {bb[1]:+.3f}   E[b^2] {b2:.5f}   "
            f"beklenen prob kazanci {-Q * b2:+.6f} test MSE"
        )
    print(
        f"\nNOT: sicak dMSE -> test dMSE carpani {SICAK_PAY:.5f} "
        f"(yukaridaki Q'lar ZATEN tam test kumesinde)"
    )
    (CIK / "yas_bandi_b.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
