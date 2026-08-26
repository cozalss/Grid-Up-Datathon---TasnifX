"""H18 -- OZET PENCERESININ UCUNDA DOGANLAR: gizli ucuncu rejim mi?

IZ
--
H17'de reddettigim artefakt kendi basina bir soru doguruyor:
2026-03-26/27'de train trafo sayisi 3.875 -> 4.424'e firliyor (+~550) ve
03-28'de bir kismi kayboluyor. **TRAIN 2026-03-31'DE BITIYOR**, yani bu
artefakt test modelinin ozet penceresinin TAM UCUNDA.

Bu trafolar model icin SICAK sayilir (train'de tanim'lari var) ama gecmisleri
yalnizca 2-6 GUN. Model "gecmisi var" diye davranir, oysa pratikte kordur.
Sicak/soguk ikili ayriminin yakalayamadigi UCUNCU BIR REJIM olabilir.

SORULAR
-------
1. O trafolarin kaci TEST'te var, kac satir ediyorlar?
2. IKIZDE (yaz25, kesme 2025-03-31) benzer grup: kesmenin son 6 gununde
   dogmus, cok kisa gecmisli trafolar. Modelin onlardaki yanliligi,
   SUREN sicaklardan farkli mi?
3. Farkliysa duz kayma ile duzeltilebilir mi, dMSE ne?

ESIK: -0,002'yi gecmezse gonderim dosyalarina DOKUNULMAZ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
TRAIN_SON = pd.Timestamp("2026-03-31")
YAZ25_KESME = pd.Timestamp("2025-03-31")
P_SICAK = 0.77841


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    son = tr.groupby("tanim")["tarih"].max()

    print("=" * 90)
    print("1. TRAIN KUYRUGU -- gunluk trafo sayisi")
    print("=" * 90)
    gunluk = tr.groupby("tarih")["tanim"].nunique()
    print(gunluk.tail(10).to_string())

    print("\n" + "=" * 90)
    print("2. KUYRUK DOGUMLULARI -- son 6 gunde dogmus trafolar")
    print("=" * 90)
    kuyruk = ilk[ilk >= TRAIN_SON - pd.Timedelta(days=5)]
    print(f"\n  2026-03-26..03-31 arasinda ILK KAYDI olan trafo: {len(kuyruk):,}")
    print(f"  dogum gunu dagilimi: {kuyruk.value_counts().sort_index().to_dict()}")
    gecmis = (TRAIN_SON - kuyruk).dt.days + 1
    print(
        f"  gecmis uzunlugu (gun): min {gecmis.min()} medyan "
        f"{int(gecmis.median())} max {gecmis.max()}"
    )
    # gercekten kac gun KAYDI var
    n_kayit = tr[tr["tanim"].isin(set(kuyruk.index))].groupby("tanim").size()
    print(
        f"  GERCEK kayit sayisi:  min {n_kayit.min()} medyan "
        f"{int(n_kayit.median())} max {n_kayit.max()}"
    )

    te_set = set(te["tanim"].unique())
    testte = [t for t in kuyruk.index if t in te_set]
    te_k = te[te["tanim"].isin(set(testte))]
    print(
        f"\n  >>> TEST'te olanlar: {len(testte):,} trafo, {len(te_k):,} satir "
        f"(test'in %{100 * len(te_k) / len(te):.2f}'si)"
    )
    if len(te_k):
        gn = te_k.groupby("tanim").size()
        print(f"      test gun sayisi: min {gn.min()} medyan {int(gn.median())} max {gn.max()}")

    if len(te_k) == 0:
        print("\n  TEST'te yoklar -> eksen konusuz. HUKUM: CURUDU.")
        return 0

    # ---------------- 3. IKIZDE OLC ----------------
    print("\n" + "=" * 90)
    print("3. IKIZ OLCUMU -- yaz25 (kesme 2025-03-31), kisa gecmisli SICAKLAR")
    print("=" * 90)
    z = np.load(KOK / "data/interim/eksen5/kos_lgbm_yaz25.npz", allow_pickle=True)
    tan = z["tanim"].astype(str)
    tarih = pd.to_datetime(z["tarih"], unit="D", origin="unix")
    gercek = z["gercek"].astype("float64")
    lgguc = z["lg"].astype("float64")
    r_true = np.log1p(np.clip(gercek, 0, None)) - lgguc

    kollar = [k for k in z.files if k.startswith(("A_", "Aplus_", "B_"))]
    # hangi kol hedefe en yakin? (tahminin r uzayinda oldugunu dogrula)
    en_iyi, en_kor = None, -2.0
    for k in kollar:
        kor = float(np.corrcoef(z[k].astype("float64"), r_true)[0, 1])
        if kor > en_kor:
            en_kor, en_iyi = kor, k
    print(f"\n  tahmin kolonlari: {kollar}")
    print(f"  hedefe en yakin: {en_iyi} (kor {en_kor:+.4f}) -> r uzayinda")

    aile = sorted({k.split("_")[0] for k in kollar})
    kullan = "Aplus" if "Aplus" in aile else aile[0]
    tohumlar = sorted({k.split("_")[1] for k in kollar if k.startswith(kullan + "_")})
    tahmin = np.mean([z[f"{kullan}_{s}"].astype("float64") for s in tohumlar], axis=0)
    print(f"  kullanilan aile: {kullan}, {len(tohumlar)} tohum")

    ilk_s = pd.Series(tan).map(ilk)
    # kesmeden ONCE dogmus olanlar = yaz25'te SICAK
    sicak = (ilk_s <= YAZ25_KESME).to_numpy()
    gecmis_gun = (YAZ25_KESME - ilk_s).dt.days.to_numpy() + 1

    print(f"\n  yaz25 blogu: {len(tan):,} satir, sicak {int(sicak.sum()):,}")
    print(f"\n  {'gecmis uzunlugu':<24} {'satir':>9} {'trafo':>7} {'yanlilik b':>12} {'RMSE':>8}")
    gruplar = [
        ("KUYRUK  <=6 gun", 1, 7),
        ("kisa    7-30 gun", 7, 31),
        ("orta    31-90 gun", 31, 91),
        ("SUREN   >90 gun", 91, 10**9),
    ]
    ozet = {}
    for ad, lo, hi in gruplar:
        m = sicak & (gecmis_gun >= lo) & (gecmis_gun < hi)
        if m.sum() == 0:
            print(f"  {ad:<24} {'YOK':>9}")
            continue
        d = r_true[m] - tahmin[m]
        ozet[ad] = (float(d.mean()), int(m.sum()))
        print(
            f"  {ad:<24} {int(m.sum()):>9,} {len(set(tan[m])):>7,} "
            f"{d.mean():>+12.4f} {np.sqrt((d**2).mean()):>8.4f}"
        )

    if "KUYRUK  <=6 gun" in ozet and "SUREN   >90 gun" in ozet:
        bk, nk = ozet["KUYRUK  <=6 gun"]
        bs, ns = ozet["SUREN   >90 gun"]
        fark = bk - bs
        print(f"\n  >>> KUYRUK vs SUREN yanlilik farki = {fark:+.4f}")
        # test'teki paya gore dMSE
        pay = len(te_k) / len(te)
        dmse = -pay * fark**2
        print(f"  >>> test payi {pay:.4f}  ->  duz kayma ile ulasilabilir dMSE {dmse:+.6f}")
        print(f"      (esik -0,002; {'GECIYOR' if dmse < -0.002 else 'GECMIYOR'})")

        # tohum bazinda saglamlik
        per = []
        for s in tohumlar:
            t1 = z[f"{kullan}_{s}"].astype("float64")
            m1 = sicak & (gecmis_gun <= 6)
            m2 = sicak & (gecmis_gun > 90)
            per.append(float((r_true[m1] - t1[m1]).mean() - (r_true[m2] - t1[m2]).mean()))
        v = np.array(per)
        print(
            f"  tohum bazinda fark: {[round(x, 4) for x in per]}  "
            f"ort {v.mean():+.4f} SH {v.std(ddof=1) / np.sqrt(len(v)):.4f}"
        )
    print("\n" + "=" * 90)
    print("HUKUM")
    print("=" * 90)
    print("  dMSE < -0,002 ve tohumlarda tutarli ise S3'e girebilir.")
    print("  Aksi halde deftere yazilir ve 27 Agustos kuyruguna konur.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
