"""KUYRUK/GENC BAND PROB TASARIMI -- test tarafinda nufus, Q ve diklik.

BULGU (kuyruk_adaylar.py): kuyruk penceresini genisletmek kis26'da BUYUK
kazanc (-0,0075..-0,0122 sicak MSE), guz25'te BUYUK kayip (+0,0028..+0,0064).
|etki| iki blokta da BUYUK, isaret TERS. Bu tam olarak LB probunun cozecegi
durumdur: dMSE(k) = k^2 Q - 2 k L,  k* = L/Q,  tavan -L^2/Q.

Bu betik prob vektorunu TEST TARAFINDA kurar:
  * gecmis_gun = test penceresi baslangici - trafonun train'deki ILK kaydi
  * v_GENC = 1[6 < gecmis_gun <= 90  ve  trafo train'de VAR (sicak)]
    -> 1[kuyruk] ile TANIM GEREGI AYRIK (kuyruk gecmis_gun<=6)
  * dikey bilesen: v_GENC'in 1[sicak-cekirdek] uzerindeki izdusumu cikarilir

Kullanim:  uv run python experiments/kapali_eksenler/kuyruk_prob_tasarim.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[2]
CIK = Path(__file__).resolve().parent
TRAIN_SON = pd.Timestamp("2026-03-31")


def main() -> int:
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
        encoding="utf-8",
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    n_kayit = tr.groupby("tanim").size()
    test_bas = te["tarih"].min()
    print(f"test penceresi: {test_bas.date()} .. {te['tarih'].max().date()}  ({len(te):,} satir)")
    print(f"train sonu    : {tr['tarih'].max().date()}")

    ig = te["tanim"].map(ilk)
    sicak = ig.notna().to_numpy()
    gecmis = (test_bas - ig).dt.days.to_numpy(dtype="float64")

    print(
        f"\nSICAK (train'de tanim var) {int(sicak.sum()):,} satir "
        f"({sicak.mean():.5f})   SOGUK {int((~sicak).sum()):,}"
    )

    kenar = [-1e9, 6, 30, 90, 180, 400, 1e9]
    et = ["<=6g", "7-30g", "31-90g", "91-180g", "181-400g", ">400g"]
    kes = pd.cut(gecmis, kenar, labels=et)
    d = pd.DataFrame({"k": kes, "t": te["tanim"]})
    g = d.groupby("k", observed=True).agg(satir=("t", "size"), trafo=("t", "nunique"))
    g["satir%"] = 100 * g["satir"] / len(te)
    print("\nTESTTE gecmis_gun dagilimi (yalniz sicak satirlar sayiliyor):")
    print(g.to_string())

    kuyruk = sicak & (gecmis <= 6)
    genc = sicak & (gecmis > 6) & (gecmis <= 90)
    genc30 = sicak & (gecmis > 6) & (gecmis <= 30)
    cekirdek = sicak & ~kuyruk

    print(f"\n1[kuyruk]          {int(kuyruk.sum()):,} satir  Q={kuyruk.mean():.7f}")
    print(f"1[sicak-cekirdek]  {int(cekirdek.sum()):,} satir  Q={cekirdek.mean():.7f}")
    print(f"1[GENC 7-90g]      {int(genc.sum()):,} satir  Q={genc.mean():.7f}")
    print(f"1[GENC 7-30g]      {int(genc30.sum()):,} satir  Q={genc30.mean():.7f}")
    print(f"  kuyruk & genc kesisim: {int((kuyruk & genc).sum())}  (0 olmali)")

    sonuc: dict = {}
    for ad, v in (("GENC_7_90", genc.astype("float64")), ("GENC_7_30", genc30.astype("float64"))):
        c = cekirdek.astype("float64")
        k = kuyruk.astype("float64")
        # 1[sicak-cekirdek] ve 1[kuyruk] YONLERINDEN arindir (ikisi zaten dik)
        v_dik = v - (v @ c) / (c @ c) * c - (v @ k) / (k @ k) * k
        Q_ham = float((v * v).mean())
        Q_dik = float((v_dik * v_dik).mean())
        print(
            f"\n{ad}:  Q_ham {Q_ham:.7f}   Q_dik {Q_dik:.7f}   kaybolan pay {1 - Q_dik / Q_ham:.4f}"
        )
        print(
            f"   dik(v, cekirdek) = {float(v_dik @ c):.3e}   "
            f"dik(v, kuyruk) = {float(v_dik @ k):.3e}"
        )
        sonuc[ad] = {"n": int(v.sum()), "Q_ham": Q_ham, "Q_dik": Q_dik}

    # kayit sayisi -- test kuyruk trafolarinin train gecmisi ne kadar kisa?
    kt = sorted(te.loc[kuyruk, "tanim"].unique())
    kk = n_kayit.reindex(kt)
    print(
        f"\nkuyruk trafolari: {len(kt):,}  train kayit sayisi "
        f"min {kk.min()} medyan {int(kk.median())} max {kk.max()}"
    )
    gt = sorted(te.loc[genc, "tanim"].unique())
    gk = n_kayit.reindex(gt)
    print(
        f"genc (7-90g)    : {len(gt):,}  train kayit sayisi "
        f"min {gk.min()} medyan {int(gk.median())} max {gk.max()}"
    )

    print("\n" + "=" * 92)
    print("PROB ARITMETIGI  dMSE(k) = k^2 Q - 2 k L ;  k* = L/Q ;  tavan = -L^2/Q")
    print("=" * 92)
    for ad, s in sonuc.items():
        Q = s["Q_dik"]
        print(f"\n{ad}  (Q_dik = {Q:.7f}, n = {s['n']:,})")
        print(f"  {'varsayilan b':>14}{'tavan dMSE':>14}{'k*':>10}")
        for b in (0.05, 0.10, 0.17, 0.29):
            L = b * Q
            print(f"  {b:>14.3f}{-L * L / Q:>+14.6f}{L / Q:>10.3f}")
    (CIK / "kuyruk_prob_tasarim.json").write_text(
        json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
