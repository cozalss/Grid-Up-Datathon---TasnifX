"""PANEL SINIR -- NIHAI, TEST NUFUSUNA OLCEKLENMIS KAZANC.

panel_sinir_ihtiyat.py'nin sayilari BLOK paylarinda. Blok ile test'in sinir
paylari FARKLI, dolayisiyla dogrudan tasinamaz:

    sinir payi        SICAK giris  SICAK cikis  SOGUK giris  SOGUK cikis
    yaz25                0,00193      0,00232      0,04134      0,01260
    guz25                0,00167      0,00329      0,03149      0,00434
    kis26                0,00229      0,00262      0,02271      0,00943
    TEST (koprulu)       0,00303      0,00139      0,01375      0,00136

Dogru olcekleme SATIR BASI yapilir:

    test dMSE = (satir_basi_giris * n_test_giris
                 + satir_basi_cikis * n_test_cikis) / 714688

Boylece "yaz25 soguk kenarinda cok satir var" avantaji teste sizmaz.
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
ADAYLAR = [(-0.20, -0.30), (-0.30, -0.40), (-0.35, -0.45), (-0.40, -0.50), (-0.50, -0.60)]


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
    tb = t2[["tanim", "tarih", "giris", "cikis"]]

    ort = pd.concat([tr, te], ignore_index=True).sort_values(["tanim", "tarih"], kind="mergesort")
    o2 = ort.groupby("tanim", observed=True)["tarih"].shift(1)
    s2 = ort.groupby("tanim", observed=True)["tarih"].shift(-1)
    ort["giris"] = o2.isna() | ((ort["tarih"] - o2) > GUN)
    ort["cikis"] = s2.isna() | ((s2 - ort["tarih"]) > GUN)
    tt = ort[ort["tarih"] >= TEST_BAS].copy()
    tt["cikis"] &= tt["tarih"] != TEST_SON
    tt["soguk"] = ~tt["tanim"].isin(sicak_set)
    N = {
        ("sicak", "giris"): int((tt["giris"] & ~tt["soguk"]).sum()),
        ("sicak", "cikis"): int((tt["cikis"] & ~tt["soguk"]).sum()),
        ("soguk", "giris"): int((tt["giris"] & tt["soguk"]).sum()),
        ("soguk", "cikis"): int((tt["cikis"] & tt["soguk"]).sum()),
    }
    print("TEST sinir sayimi (train'e koprulu, 2026-07-31 cikis SAYILMIYOR):")
    for k, v in N.items():
        print(f"  {k[0]} {k[1]}: {v:,}")

    def esle(tanim, tarih):
        sol = pd.DataFrame(
            {"tanim": np.asarray(tanim, dtype=object), "tarih": pd.to_datetime(tarih)}
        )
        sol["_i"] = np.arange(len(sol))
        j = sol.merge(tb, on=["tanim", "tarih"], how="left").sort_values("_i")
        return (
            j["giris"].fillna(False).to_numpy().astype("float64"),
            j["cikis"].fillna(False).to_numpy().astype("float64"),
        )

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

    print("\n" + "=" * 110)
    print("SATIR BASI e^2 DUSUSU -> TEST OLCEGI")
    print("=" * 110)
    sonuc = []
    for dg, dc in ADAYLAR:
        blok_test = {}
        for blok in BLOKLAR:
            toplam = 0.0
            ayr = {}
            for rej, B, T, M in (
                ("sicak", sb[blok], st[blok], sm[blok]),
                ("soguk", cb[blok], ct[blok], cm[blok]),
            ):
                r0 = T
                m0 = mse_k(B.lgy, B.lgc, r0 + kdelta(B.lgy, B.lgc, r0))
                for tur, idx, d in (("giris", 0, dg), ("cikis", 1, dc)):
                    m = M[idx]
                    if m.sum() == 0:
                        ayr[(rej, tur)] = 0.0
                        continue
                    rr = r0 + d * m
                    kz = mse_k(B.lgy, B.lgc, rr + kdelta(B.lgy, B.lgc, rr)) - m0
                    per = kz * B.n / m.sum()  # sinir satiri basi e^2 dususu
                    katki = per * N[(rej, tur)] / N_TEST
                    ayr[(rej, tur)] = katki
                    toplam += katki
            blok_test[blok] = toplam
            if (dg, dc) == (-0.35, -0.45):
                print(
                    f"  d={dg:+.2f}/{dc:+.2f}  {blok:6} "
                    + "  ".join(f"{r}-{t} {v:+.6f}" for (r, t), v in ayr.items())
                    + f"   TOPLAM {toplam:+.6f}"
                )
        ayni = all(v < 0 for v in blok_test.values())
        sonuc.append(
            {
                "d_giris": dg,
                "d_cikis": dc,
                **blok_test,
                "ort": float(np.mean(list(blok_test.values()))),
                "en_kotu": float(max(blok_test.values())),
                "uc_ayni": ayni,
            }
        )

    print("\n" + "=" * 110)
    print("TEST OLCEKLI dMSE (her blogun kendi kestirimi, test paylarina tasinmis)")
    print("=" * 110)
    print(
        f"{'d_giris/d_cikis':18}{'yaz25':>13}{'guz25':>13}{'kis26':>13}"
        f"{'ORT':>13}{'EN KOTU':>13}   3/3"
    )
    print("-" * 100)
    for s in sonuc:
        print(
            f"{f'{s[chr(100) + chr(95) + chr(103) + chr(105) + chr(114) + chr(105) + chr(115)]:+.2f} / {s[chr(100) + chr(95) + chr(99) + chr(105) + chr(107) + chr(105) + chr(115)]:+.2f}':18}"
            f"{s['yaz25']:>+13.6f}{s['guz25']:>+13.6f}{s['kis26']:>+13.6f}"
            f"{s['ort']:>+13.6f}{s['en_kotu']:>+13.6f}   "
            f"{'EVET' if s['uc_ayni'] else 'hayir'}"
        )
    (CIK / "panel_sinir_nihai.json").write_text(
        json.dumps(
            {"test_sayim": {f"{k[0]}_{k[1]}": v for k, v in N.items()}, "sonuc": sonuc},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
