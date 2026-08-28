"""PANEL SINIR -- PARTI-AYRIK NIHAI OLCUM, TEST NUFUSUNA OLCEKLENMIS.

panel_sinir_parti.py: 100+'lik partilerde dusus VAR ama daha kucuk ve
bloklar arasi daha oynak (guz25 100+ girisinde n=14 ile +0,51). Testteki
giris satirlarinin 2.748/3.860'i 100+ partide (2026-05-11). Bu yuzden
duzeltme PARTI-AYRIK verilir:

    giris, parti <100   -> tam genlik
    giris, parti 100+   -> YARIM genlik
    cikis               -> tam genlik  (testte 100+ cikis YOK)
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))
import ortak as S  # noqa: E402

CIK = Path(__file__).resolve().parent
GUN = pd.Timedelta(days=1)
TRAIN_BAS = pd.Timestamp("2025-01-01")
TRAIN_SON = pd.Timestamp("2026-03-31")
TEST_BAS = pd.Timestamp("2026-04-01")
TEST_SON = pd.Timestamp("2026-07-31")
BLOKLAR = ("yaz25", "guz25", "kis26")
N_TEST = 714_688
PARTI_ESIK = 100
# (d_giris_kucuk, d_giris_buyuk, d_cikis)
ADAYLAR = [
    (-0.30, -0.15, -0.40),
    (-0.40, -0.20, -0.50),
    (-0.50, -0.25, -0.60),
    (-0.40, -0.00, -0.50),
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


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    sicak_set = set(tr["tanim"].unique())

    t2 = tr.sort_values(["tanim", "tarih"], kind="mergesort").copy()
    o1 = t2.groupby("tanim", observed=True)["tarih"].shift(1)
    s1 = t2.groupby("tanim", observed=True)["tarih"].shift(-1)
    t2["giris"] = (o1.isna() | ((t2["tarih"] - o1) > GUN)) & (t2["tarih"] != TRAIN_BAS)
    t2["cikis"] = (s1.isna() | ((s1 - t2["tarih"]) > GUN)) & (t2["tarih"] != TRAIN_SON)
    gp = t2.loc[t2["giris"]].groupby("tarih").size()
    t2["g_buyuk"] = t2["giris"] & (t2["tarih"].map(gp).fillna(0) >= PARTI_ESIK)
    t2["g_kucuk"] = t2["giris"] & ~t2["g_buyuk"]
    tb = t2[["tanim", "tarih", "g_kucuk", "g_buyuk", "cikis"]]

    ort = pd.concat([tr, te], ignore_index=True).sort_values(["tanim", "tarih"], kind="mergesort")
    o2 = ort.groupby("tanim", observed=True)["tarih"].shift(1)
    s2 = ort.groupby("tanim", observed=True)["tarih"].shift(-1)
    ort["giris"] = o2.isna() | ((ort["tarih"] - o2) > GUN)
    ort["cikis"] = s2.isna() | ((s2 - ort["tarih"]) > GUN)
    tt = ort[ort["tarih"] >= TEST_BAS].copy()
    tt["cikis"] &= tt["tarih"] != TEST_SON
    gpt = tt.loc[tt["giris"]].groupby("tarih").size()
    tt["g_buyuk"] = tt["giris"] & (tt["tarih"].map(gpt).fillna(0) >= PARTI_ESIK)
    tt["g_kucuk"] = tt["giris"] & ~tt["g_buyuk"]
    tt["soguk"] = ~tt["tanim"].isin(sicak_set)
    N = {}
    for rej, mm in (("sicak", ~tt["soguk"]), ("soguk", tt["soguk"])):
        for tur in ("g_kucuk", "g_buyuk", "cikis"):
            N[(rej, tur)] = int((tt[tur] & mm).sum())
    print("TEST sinir sayimi:")
    for k, v in N.items():
        print(f"  {k[0]} {k[1]}: {v:,}")

    def esle(tanim, tarih):
        sol = pd.DataFrame(
            {"tanim": np.asarray(tanim, dtype=object), "tarih": pd.to_datetime(tarih)}
        )
        sol["_i"] = np.arange(len(sol))
        j = sol.merge(tb, on=["tanim", "tarih"], how="left").sort_values("_i")
        return {
            t: j[t].fillna(False).to_numpy().astype("float64")
            for t in ("g_kucuk", "g_buyuk", "cikis")
        }

    sb = S.bloklari_kur()
    st = {k: S.taban_r(sb[k]) for k in BLOKLAR}
    sm = {
        k: esle(sb[k].cerceve["tanim"].to_numpy(), sb[k].cerceve["tarih"].to_numpy())
        for k in BLOKLAR
    }
    spec = importlib.util.spec_from_file_location(
        "soguk_ortak", KOK / "experiments" / "soguk_kaldirac" / "ortak.py"
    )
    C = importlib.util.module_from_spec(spec)
    sys.modules["soguk_ortak"] = C
    spec.loader.exec_module(C)
    cb = C.tum_bloklar()
    ct = {k: C.taban_r(cb[k]) for k in BLOKLAR}
    cm = {k: esle(cb[k].tanim, cb[k].tarih) for k in BLOKLAR}

    print("\n" + "=" * 104)
    print("TEST OLCEKLI dMSE (satir basi kazanc x test sinir sayisi / 714.688)")
    print("=" * 104)
    print(
        f"{'gk / gb / c':24}{'yaz25':>13}{'guz25':>13}{'kis26':>13}{'ORT':>13}{'EN KOTU':>13}   3/3"
    )
    print("-" * 104)
    cikti = []
    for dgk, dgb, dc in ADAYLAR:
        blok = {}
        for bl in BLOKLAR:
            toplam = 0.0
            for rej, B, T, M in (
                ("sicak", sb[bl], st[bl], sm[bl]),
                ("soguk", cb[bl], ct[bl], cm[bl]),
            ):
                m0 = mse_k(B.lgy, B.lgc, T + kdelta(B.lgy, B.lgc, T))
                for tur, d in (("g_kucuk", dgk), ("g_buyuk", dgb), ("cikis", dc)):
                    m = M[tur]
                    if m.sum() == 0 or d == 0.0 or N[(rej, tur)] == 0:
                        continue
                    rr = T + d * m
                    kz = mse_k(B.lgy, B.lgc, rr + kdelta(B.lgy, B.lgc, rr)) - m0
                    toplam += kz * B.n / m.sum() * N[(rej, tur)] / N_TEST
            blok[bl] = toplam
        v = list(blok.values())
        cikti.append(
            {
                "d_giris_kucuk": dgk,
                "d_giris_buyuk": dgb,
                "d_cikis": dc,
                **blok,
                "ort": float(np.mean(v)),
                "en_kotu": float(max(v)),
                "uc_ayni": all(x < 0 for x in v),
            }
        )
        print(
            f"{f'{dgk:+.2f} / {dgb:+.2f} / {dc:+.2f}':24}"
            f"{blok['yaz25']:>+13.6f}{blok['guz25']:>+13.6f}{blok['kis26']:>+13.6f}"
            f"{np.mean(v):>+13.6f}{max(v):>+13.6f}   "
            f"{'EVET' if all(x < 0 for x in v) else 'hayir'}"
        )

    (CIK / "panel_sinir_nihai2.json").write_text(
        json.dumps(
            {"test_sayim": {f"{k[0]}_{k[1]}": v for k, v in N.items()}, "sonuc": cikti},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
