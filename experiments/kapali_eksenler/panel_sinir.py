"""PANEL SINIR GUNLERI -- bir trafonun panele GIRDIGI ve CIKTIGI gunler.

docs/43 §6 madde 2: "panel sinir gunu (giris/cikis) 2.387 satir -0,00067"
-- "yuksek guven" damgasi var ama URETIME HIC GIRMEDI. Baska bir ajan
683 satirlik artigin v83'te OLMADIGINI olctu (ort delta -0,4838).

VE testte 2026-05-11'de 2.222 trafo TEK GUNDE panele giriyor (05-03'te
141 daha). Yani bu eksen testte kucuk degil.

SINIR TANIMI (train'den, gercek varlik deseninden):
    giris = onceki takvim gunu panelde YOK  (ilk gun ya da bosluktan donus)
    cikis = sonraki takvim gunu panelde YOK (son gun ya da bosluga giris)

Bu, blok kesmesinin yarattigi YAPAY sinirdan farklidir: blok baslangici
trafonun gercek girisi degilse sinir sayilmaz (train butununden bakilir).

Olcut: SICAK satirlarda log1p uzayinda MSE, URETIM KIRPMASI ile
(sicak_kaldirac/ortak.py). Ayrica SOGUK taraf ayrica raporlanir.

Kullanim:  uv run python experiments/kapali_eksenler/panel_sinir.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(KOK / "experiments" / "sicak_kaldirac"))

from ortak import (  # noqa: E402
    BLOKLAR,
    SICAK_PAY,
    bloklari_kur,
    kuresel_delta,
    mse,
    taban_r,
)

CIK = Path(__file__).resolve().parent
pd.set_option("display.width", 240)
GUN = pd.Timedelta(days=1)


def sinir_bayraklari(df: pd.DataFrame) -> pd.DataFrame:
    """``tanim``/``tarih`` cercevesine giris/cikis sinir bayraklari ekler."""
    d = df[["tanim", "tarih"]].copy()
    d["tarih"] = pd.to_datetime(d["tarih"])
    d = d.sort_values(["tanim", "tarih"], kind="mergesort")
    onc = d.groupby("tanim", observed=True)["tarih"].shift(1)
    son = d.groupby("tanim", observed=True)["tarih"].shift(-1)
    d["giris"] = onc.isna() | ((d["tarih"] - onc) > GUN)
    d["cikis"] = son.isna() | ((son - d["tarih"]) > GUN)
    d["ilk_gun"] = d.groupby("tanim", observed=True)["tarih"].transform("min")
    d["son_gun"] = d.groupby("tanim", observed=True)["tarih"].transform("max")
    d["dogum"] = d["tarih"] == d["ilk_gun"]
    d["olum"] = d["tarih"] == d["son_gun"]
    return d


def train_sinir() -> pd.DataFrame:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        encoding="utf-8",
    )
    return sinir_bayraklari(tr)


def test_sinir() -> pd.DataFrame:
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        encoding="utf-8",
    )
    s = sinir_bayraklari(te)
    s["id"] = te.loc[s.index, "id"].to_numpy()
    return s


def esle(b, ts: pd.DataFrame) -> pd.DataFrame:
    """Blok cercevesine (sicak) train sinir bayraklarini eslestirir."""
    sol = pd.DataFrame(
        {
            "tanim": b.cerceve["tanim"].to_numpy(),
            "tarih": pd.to_datetime(b.cerceve["tarih"].to_numpy()),
        }
    )
    sol["_i"] = np.arange(len(sol))
    j = sol.merge(ts, on=["tanim", "tarih"], how="left", validate="one_to_one")
    j = j.sort_values("_i")
    for c in ("giris", "cikis", "dogum", "olum"):
        j[c] = j[c].fillna(False).astype(bool)
    return j


def main() -> int:
    ts = train_sinir()
    print("=" * 100)
    print("0) TRAIN'DE SINIR NUFUSU")
    print("=" * 100)
    print(
        f"  toplam satir {len(ts):,}   giris {int(ts['giris'].sum()):,} "
        f"({ts['giris'].mean():.4f})   cikis {int(ts['cikis'].sum()):,} "
        f"({ts['cikis'].mean():.4f})   ikisi {int((ts['giris'] & ts['cikis']).sum()):,}"
    )
    print(
        f"  bunlarin dogum/olum olani: dogum {int(ts['dogum'].sum()):,}  "
        f"olum {int(ts['olum'].sum()):,}  "
        f"-> BOSLUK sinirlari: giris {int((ts['giris'] & ~ts['dogum']).sum()):,}  "
        f"cikis {int((ts['cikis'] & ~ts['olum']).sum()):,}"
    )

    tes = test_sinir()
    print("\n" + "=" * 100)
    print("1) TESTTE SINIR NUFUSU -- eksen testte kac satir?")
    print("=" * 100)
    print(
        f"  toplam satir {len(tes):,}   giris {int(tes['giris'].sum()):,} "
        f"({tes['giris'].mean():.4f})   cikis {int(tes['cikis'].sum()):,} "
        f"({tes['cikis'].mean():.4f})"
    )
    ge = tes.loc[tes["giris"], "tarih"].value_counts().sort_values(ascending=False).head(8)
    print("\n  en kalabalik GIRIS gunleri (test):")
    for t, n in ge.items():
        print(f"    {pd.Timestamp(t).date()}  {n:,} trafo")
    ce = tes.loc[tes["cikis"], "tarih"].value_counts().sort_values(ascending=False).head(5)
    print("  en kalabalik CIKIS gunleri (test):")
    for t, n in ce.items():
        print(f"    {pd.Timestamp(t).date()}  {n:,} satir")

    bloklar = bloklari_kur()
    taban = {k: taban_r(bloklar[k]) for k in BLOKLAR}

    print("\n" + "=" * 100)
    print("2) SICAK TARAFTA SINIR GUNU YANLILIGI (uretim tabani, kirpmali olcut)")
    print("   e = log1p(y) - max(r + log1p(guc), 0);  yanlilik>0 => MODEL DUSUK TAHMIN")
    print("=" * 100)
    bayrak: dict[str, pd.DataFrame] = {}
    for ad in BLOKLAR:
        b = bloklar[ad]
        j = esle(b, ts)
        bayrak[ad] = j
        e = b.lgy - np.maximum(taban[ad] + b.lgc, 0.0)
        etiket = np.where(
            j["giris"].to_numpy() & j["cikis"].to_numpy(),
            "giris+cikis",
            np.where(
                j["giris"].to_numpy(), "GIRIS", np.where(j["cikis"].to_numpy(), "CIKIS", "ic")
            ),
        )
        d = pd.DataFrame({"k": etiket, "e": e, "e2": e * e, "y0": (b.y <= 0).astype(float)})
        g = d.groupby("k").agg(
            n=("e", "size"),
            yanlilik=("e", "mean"),
            std=("e", "std"),
            mse=("e2", "mean"),
            y0=("y0", "mean"),
        )
        g["t"] = g["yanlilik"] / (g["std"] / np.sqrt(g["n"]))
        print(f"\n-- {ad} (sicak, taban MSE {mse(b, taban[ad]):.5f}) --")
        print(g.round(4).to_string())

        # dogum/olum ayrisimi
        sub = pd.DataFrame(
            {
                "k": np.where(
                    j["dogum"].to_numpy(),
                    "dogum",
                    np.where(
                        j["olum"].to_numpy(),
                        "olum",
                        np.where(
                            j["giris"].to_numpy(),
                            "bosluk-donus",
                            np.where(j["cikis"].to_numpy(), "bosluk-giris", "ic"),
                        ),
                    ),
                ),
                "e": e,
                "e2": e * e,
            }
        )
        gs = sub.groupby("k").agg(
            n=("e", "size"), yanlilik=("e", "mean"), std=("e", "std"), mse=("e2", "mean")
        )
        gs["t"] = gs["yanlilik"] / (gs["std"] / np.sqrt(gs["n"]))
        print(gs.round(4).to_string())

    print("\n" + "=" * 100)
    print("3) ADAYLAR -- uc blokta, seviye-notr, uretim kirpmasiyla")
    print("=" * 100)

    def yap(maske_ad: str, d: float):
        def f(b, r0, j):
            m = j[maske_ad].to_numpy(dtype="float64")
            return r0 + d * m

        return f

    def yap_ikili(d_g: float, d_c: float):
        def f(b, r0, j):
            return (
                r0
                + d_g * j["giris"].to_numpy(dtype="float64")
                + d_c * j["cikis"].to_numpy(dtype="float64")
            )

        return f

    adaylar = [
        ("P1  giris -0,10", yap("giris", -0.10)),
        ("P2  giris -0,25", yap("giris", -0.25)),
        ("P3  giris +0,10", yap("giris", +0.10)),
        ("P4  giris +0,25", yap("giris", +0.25)),
        ("P5  cikis -0,25", yap("cikis", -0.25)),
        ("P6  cikis -0,50", yap("cikis", -0.50)),
        ("P7  cikis +0,25", yap("cikis", +0.25)),
        ("P8  dogum -0,25", yap("dogum", -0.25)),
        ("P9  olum  -0,50", yap("olum", -0.50)),
        ("P10 giris-0,15 & cikis-0,35", yap_ikili(-0.15, -0.35)),
    ]

    satirlar = []
    for ad_a, fn in adaylar:
        s: dict = {"aday": ad_a}
        tn = td = 0.0
        for k in BLOKLAR:
            b = bloklar[k]
            r0 = taban[k]
            r1 = fn(b, r0, bayrak[k])
            r1 = r1 + kuresel_delta(b, r1)
            dd = mse(b, r1) - mse(b, r0)
            s[k] = dd
            tn += b.n
            td += dd * b.n
        s["GENEL"] = td / tn
        s["testMSE"] = s["GENEL"] * SICAK_PAY
        s["uc_blok_ayni"] = all(s[k] < 0 for k in BLOKLAR) or all(s[k] > 0 for k in BLOKLAR)
        satirlar.append(s)

    print(
        f"\n{'aday':30}{'yaz25':>11}{'guz25':>11}{'kis26':>11}{'GENEL':>11}{'testdMSE':>11}  karar"
    )
    print("-" * 98)
    for s in satirlar:
        if s["testMSE"] >= 0:
            karar = "RED(zararli)"
        elif not s["uc_blok_ayni"]:
            karar = "ters isaret -> PROB?"
        elif s["testMSE"] <= -0.002:
            karar = "KABUL"
        else:
            karar = "red(kucuk)"
        print(
            f"{s['aday'][:30]:30}{s['yaz25']:>+11.5f}{s['guz25']:>+11.5f}"
            f"{s['kis26']:>+11.5f}{s['GENEL']:>+11.5f}{s['testMSE']:>+11.5f}  {karar}"
        )

    print("\n" + "=" * 100)
    print("4) HER BLOKTA OPTIMUM delta (kirpmali, seviye-notr) -- tavan ne?")
    print("=" * 100)
    for maske_ad in ("giris", "cikis", "dogum", "olum"):
        cikti = []
        for k in BLOKLAR:
            b = bloklar[k]
            r0 = taban[k]
            m = bayrak[k][maske_ad].to_numpy(dtype="float64")
            if m.sum() == 0:
                cikti.append(f"{k} YOK")
                continue
            en, enm = 0.0, None
            for d in np.arange(-1.0, 1.001, 0.02):
                rr = r0 + d * m
                v = mse(b, rr + kuresel_delta(b, rr))
                if enm is None or v < enm:
                    en, enm = float(d), v
            taban_m = mse(b, r0 + kuresel_delta(b, r0))
            cikti.append(f"{k}: d*={en:+.2f} kazanc {enm - taban_m:+.6f} (n={int(m.sum()):,})")
        print(f"  {maske_ad:8} " + " | ".join(cikti))

    with (CIK / "panel_sinir.jsonl").open("w", encoding="utf-8") as f:
        for s in satirlar:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nNOT: sicak dMSE -> test dMSE carpani {SICAK_PAY:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
