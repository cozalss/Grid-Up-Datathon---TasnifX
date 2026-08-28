"""PANEL SINIR -- IHTIYATLI GENLIK TARAMASI (kalici kural 12).

"Foldlar YON ve ISARET icin guvenilir, GENLIK icin DEGIL." Blok optimumlari
d_giris -0,20..-0,70, d_cikis -0,80..-1,10. Bu betik OPTIMUMUN ALTINDA
kalan genlikleri hem SICAK hem SOGUK tarafta olcer ve toplami cikarir.

Asiri kaymanin maliyeti kadratiktir: dMSE(d) = p*(d - b)^2 - p*b^2, yani
d = b/2 secmek kazancin %75'ini alir ama b'nin isareti yanlissa bile
zarari b^2*p/4 ile sinirlar.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))
import ortak as S  # noqa: E402

sys.path.insert(0, str(KOK / "experiments" / "soguk_kaldirac"))

CIK = Path(__file__).resolve().parent
GUN = pd.Timedelta(days=1)
TRAIN_BAS = pd.Timestamp("2025-01-01")
TRAIN_SON = pd.Timestamp("2026-03-31")
BLOKLAR = ("yaz25", "guz25", "kis26")
ADAYLAR = [
    (-0.20, -0.30),
    (-0.25, -0.35),
    (-0.30, -0.40),
    (-0.35, -0.45),
    (-0.40, -0.50),
    (-0.45, -0.55),
    (-0.50, -0.60),
]


def mse_k(lgy, lgc, r):
    e = lgy - np.maximum(r + lgc, 0.0)
    return float((e * e).mean())


def kdelta(lgy, lgc, r):
    en, enm = 0.0, mse_k(lgy, lgc, r)
    adim = 0.08
    for _ in range(5):
        for d in np.arange(en - 4 * adim, en + 4.001 * adim, adim):
            m = mse_k(lgy, lgc, r + float(d))
            if m < enm:
                en, enm = float(d), m
        adim /= 4.0
    return en


def train_bayrak():
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    tr = tr.sort_values(["tanim", "tarih"], kind="mergesort")
    onc = tr.groupby("tanim", observed=True)["tarih"].shift(1)
    son = tr.groupby("tanim", observed=True)["tarih"].shift(-1)
    tr["giris"] = (onc.isna() | ((tr["tarih"] - onc) > GUN)) & (tr["tarih"] != TRAIN_BAS)
    tr["cikis"] = (son.isna() | ((son - tr["tarih"]) > GUN)) & (tr["tarih"] != TRAIN_SON)
    return tr[["tanim", "tarih", "giris", "cikis"]]


def esle(tanim, tarih, tb):
    sol = pd.DataFrame({"tanim": np.asarray(tanim, dtype=object), "tarih": pd.to_datetime(tarih)})
    sol["_i"] = np.arange(len(sol))
    j = sol.merge(tb, on=["tanim", "tarih"], how="left").sort_values("_i")
    return (
        j["giris"].fillna(False).to_numpy().astype("float64"),
        j["cikis"].fillna(False).to_numpy().astype("float64"),
    )


def main() -> int:
    tb = train_bayrak()

    # ---------- SICAK ----------
    sb = S.bloklari_kur()
    st = {k: S.taban_r(sb[k]) for k in BLOKLAR}
    sm = {
        k: esle(sb[k].cerceve["tanim"].to_numpy(), sb[k].cerceve["tarih"].to_numpy(), tb)
        for k in BLOKLAR
    }

    # ---------- SOGUK ----------
    import importlib

    spec = importlib.util.spec_from_file_location(
        "soguk_ortak", KOK / "experiments" / "soguk_kaldirac" / "ortak.py"
    )
    C = importlib.util.module_from_spec(spec)
    sys.modules["soguk_ortak"] = C  # dataclass __module__ cozumu icin sart
    spec.loader.exec_module(C)
    cb = C.tum_bloklar()
    ct = {k: C.taban_r(cb[k]) for k in BLOKLAR}
    cm = {k: esle(cb[k].tanim, cb[k].tarih, tb) for k in BLOKLAR}
    for k in BLOKLAR:
        print(
            f"soguk {k}: giris {int(cm[k][0].sum()):,}  cikis {int(cm[k][1].sum()):,}"
            f"  / {cb[k].n:,}"
        )

    print("\n" + "=" * 108)
    print("IHTIYATLI GENLIKLER -- SICAK + SOGUK, seviye-notr, uretim kirpmasiyla")
    print("=" * 108)
    print(
        f"{'d_giris/d_cikis':18}{'SICAK test dMSE':>18}{'SOGUK test dMSE':>18}"
        f"{'TOPLAM':>12}   sicak 3/3  soguk 3/3"
    )
    print("-" * 108)
    kayit = []
    for dg, dc in ADAYLAR:
        # sicak
        s_blok, tn, td = {}, 0.0, 0.0
        for k in BLOKLAR:
            b = sb[k]
            r0 = st[k]
            m0 = mse_k(b.lgy, b.lgc, r0 + kdelta(b.lgy, b.lgc, r0))
            rr = r0 + dg * sm[k][0] + dc * sm[k][1]
            d = mse_k(b.lgy, b.lgc, rr + kdelta(b.lgy, b.lgc, rr)) - m0
            s_blok[k] = d
            tn += b.n
            td += d * b.n
        s_test = td / tn * S.SICAK_PAY
        s_ayni = all(v < 0 for v in s_blok.values()) or all(v > 0 for v in s_blok.values())
        # soguk
        c_blok, tn2, td2 = {}, 0.0, 0.0
        for k in BLOKLAR:
            b = cb[k]
            r0 = ct[k]
            m0 = mse_k(b.lgy, b.lgc, r0 + kdelta(b.lgy, b.lgc, r0))
            rr = r0 + dg * cm[k][0] + dc * cm[k][1]
            d = mse_k(b.lgy, b.lgc, rr + kdelta(b.lgy, b.lgc, rr)) - m0
            c_blok[k] = d
            tn2 += b.n
            td2 += d * b.n
        c_test = td2 / tn2 * C.SOGUK_PAY
        c_ayni = all(v < 0 for v in c_blok.values()) or all(v > 0 for v in c_blok.values())
        print(
            f"{f'{dg:+.2f} / {dc:+.2f}':18}{s_test:>+18.6f}{c_test:>+18.6f}"
            f"{s_test + c_test:>+12.6f}   {'EVET' if s_ayni else 'hayir':>9}"
            f"  {'EVET' if c_ayni else 'hayir':>9}"
        )
        kayit.append(
            {
                "d_giris": dg,
                "d_cikis": dc,
                "sicak_test": s_test,
                "soguk_test": c_test,
                "toplam": s_test + c_test,
                "sicak_bloklar": s_blok,
                "soguk_bloklar": c_blok,
                "sicak_3_3": s_ayni,
                "soguk_3_3": c_ayni,
            }
        )

    print("\nBLOK AYRINTISI (test MSE cinsinden)")
    for r in kayit:
        print(
            f"  {r['d_giris']:+.2f}/{r['d_cikis']:+.2f}  SICAK "
            + " ".join(f"{k} {v * S.SICAK_PAY:+.5f}" for k, v in r["sicak_bloklar"].items())
            + "   SOGUK "
            + " ".join(f"{k} {v * C.SOGUK_PAY:+.5f}" for k, v in r["soguk_bloklar"].items())
        )

    (CIK / "panel_sinir_ihtiyat.json").write_text(
        json.dumps(kayit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
