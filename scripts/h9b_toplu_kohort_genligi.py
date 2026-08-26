"""H9b -- TOPLU KATILIM kohortunun gun ekseni genligi, MEVSIM KONTROLLU.

NEDEN H9 YETMEDI
----------------
H9, yaz25'in dogumlarini parti buyuklugune gore ayirdi:

    tekil/kucuk <20   314 trafo  13.922 satir  sigma 0,4378
    orta 20-99        136 trafo   5.219 satir  sigma 0,2143
    TOPLU >=100       168 trafo     666 satir  sigma 0,0559

TOPLU sinifi CAZIP gorunuyor (8 kat kucuk genlik!) ama SAYI GECERSIZ:
yaz25'teki tek >=100 partisi 2025-07-28 (177 trafo) ve pencere 07-31'de
bitiyor -> trafo basina yalnizca ~4 gun. Dort gunluk pencerede olculen
"gun ekseni genligi" anlamsizdir.

TEST'in soguk tarafi ise %68 oraninda TEK bir toplu katilimdan geliyor
(2026-05-11, 1.326 trafo) ve onlarin test ufku ~82 gun. Yani soru gercek ve
yaz25 penceresi onu cevaplayamiyor.

BU BETIK
--------
Train'de TAKIP PENCERESI YETERLI olan toplu kohortu bulur (2025-11-25, ~166
trafo, 2026-03-31'e kadar ~127 gun) ve genligini AYNI PENCEREDE olculen iki
referansla karsilastirir -- boylece MEVSIM kontrol edilir:

    TOPLU kohort        vs   ayni pencerede TEKIL/KUCUK dogumlar
                        vs   ayni pencerede YERLESIK trafolar

Cikan ORANLAR mevsimden arindirilmistir ve yaz25 mevsimine tasinabilir:

    sigma_TOPLU(yaz) ~ sigma_YERLESIK(yaz) x [sigma_TOPLU / sigma_YERLESIK](kis)

Sonra test soguk karisimina (%68 toplu + %32 tekil) agirliklandirilip
c yeniden turetilir.

KURAL: referans HER ZAMAN gercek etiketlerden ve HEDEF NUFUSUN IKIZINDEN.
Test etiketi KULLANILMAZ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
MIN_GUN_TAKIP = 45  # kohortun genligi icin gereken en az takip gunu


def gun_etkisi(tanim, gun, r) -> pd.Series:
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


def olc(alt: pd.DataFrame, etiket: str, ilk_gun_at: int = 7) -> float:
    if alt.empty:
        print(f"    {etiket:<34} YOK")
        return float("nan")
    a = alt[alt["yas"] >= ilk_gun_at] if "yas" in alt else alt
    if len(a) < 400 or a["tarih"].nunique() < 30:
        print(f"    {etiket:<34} yetersiz ({len(a):,} satir, {a['tarih'].nunique()} gun)")
        return float("nan")
    b = gun_etkisi(a["tanim"].to_numpy(), a["tarih"].to_numpy(), a["r"].to_numpy())
    print(
        f"    {etiket:<34} {a.tanim.nunique():>5,} trafo {len(a):>8,} satir "
        f"{a['tarih'].nunique():>4} gun   sigma = {b.std():.4f}"
    )
    return float(b.std())


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tr = tr[tr["tuketim"] > 0].copy()
    tr["r"] = np.log1p(tr["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        tr["guc"].to_numpy(dtype="float64")
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    son = tr.groupby("tanim")["tarih"].max()
    tr["ilk"] = tr["tanim"].map(ilk)
    tr["yas"] = (tr["tarih"] - tr["ilk"]).dt.days

    # ---------- toplu kohortlari bul ----------
    parti = ilk.groupby(ilk).size()
    buyuk = parti[parti >= 100].sort_index()
    print("=" * 96)
    print("1. TRAIN'DEKI TOPLU KATILIM KOHORTLARI (>=100 trafo ayni gun)")
    print("=" * 96)
    print(f"\n  {'dogum gunu':<14} {'trafo':>7} {'takip gunu':>11}  kullanilabilir?")
    kullanilabilir = []
    for gun, n in buyuk.items():
        takip = (tr["tarih"].max() - gun).days
        ok = takip >= MIN_GUN_TAKIP
        if ok:
            kullanilabilir.append(gun)
        print(f"  {str(gun.date()):<14} {n:>7,} {takip:>11}  {'EVET' if ok else 'hayir'}")

    if not kullanilabilir:
        print("\n  Takip penceresi yeterli toplu kohort YOK -> hukum verilemez.")
        return 0

    # ---------- her kullanilabilir kohort icin mevsim-kontrollu karsilastirma ----------
    print("\n" + "=" * 96)
    print("2. MEVSIM KONTROLLU KARSILASTIRMA -- her kohort kendi penceresinde")
    print("=" * 96)
    oranlar = []
    for gun in kullanilabilir:
        bas = gun
        bit = min(gun + pd.Timedelta(days=127), tr["tarih"].max())
        pen = tr[(tr["tarih"] >= bas) & (tr["tarih"] <= bit)].copy()
        ngun = pen["tarih"].nunique()
        print(f"\n  --- kohort {gun.date()}   pencere {bas.date()}..{bit.date()} ({ngun} gun)")

        kohort = set(ilk[ilk == gun].index)
        # ayni pencerede dogan TEKIL/KUCUK trafolar
        pd_ilk = ilk[(ilk >= bas) & (ilk <= bit)]
        p2 = pd_ilk.groupby(pd_ilk).size()
        tekil = set(pd_ilk[pd_ilk.map(p2) < 20].index)
        # YERLESIK: pencerenin >=%90'inda var
        say = pen.groupby("tanim")["tarih"].nunique()
        yerlesik = set(say[say >= 0.9 * ngun].index)

        s_top = olc(pen[pen["tanim"].isin(kohort)], "TOPLU kohort")
        s_tek = olc(pen[pen["tanim"].isin(tekil)], "ayni pencerede TEKIL/KUCUK dogum")
        s_yer = olc(pen[pen["tanim"].isin(yerlesik)], "YERLESIK (mevsim referansi)", 0)
        if not np.isnan(s_top) and not np.isnan(s_yer) and s_yer > 0:
            o_top = s_top / s_yer
            o_tek = s_tek / s_yer if not np.isnan(s_tek) else float("nan")
            print(f"    {'ORAN TOPLU / YERLESIK':<34} {o_top:.3f}")
            print(f"    {'ORAN TEKIL / YERLESIK':<34} {o_tek:.3f}")
            oranlar.append({"kohort": str(gun.date()), "toplu": o_top, "tekil": o_tek})

    if not oranlar:
        print("\n  Oran cikarilamadi -> hukum verilemez.")
        return 0
    d = pd.DataFrame(oranlar)
    print(
        f"\n  ORTALAMA  TOPLU/YERLESIK {d.toplu.mean():.3f}   TEKIL/YERLESIK {d.tekil.mean():.3f}"
    )

    # ---------- 3. yaz25 mevsimine tasi ve c'yi yeniden turet ----------
    print("\n" + "=" * 96)
    print("3. yaz25 MEVSIMINE TASI ve c'yi YENIDEN TURET")
    print("=" * 96)
    S_YERLESIK_YAZ = 0.2710  # h8k'de dogrulandi
    S_MODEL = 0.1527  # v67 test soguk, T3, satir-agirlikli
    KOR = 0.8971
    KALIBRE = 0.80  # yaz25'te capa/etiketli orani (h8i)

    s_top_yaz = S_YERLESIK_YAZ * d.toplu.mean()
    s_tek_yaz = S_YERLESIK_YAZ * d.tekil.mean()
    print(f"\n  sigma_YERLESIK(yaz25)          {S_YERLESIK_YAZ:.4f}   (olculdu)")
    print(
        f"  sigma_TOPLU(yaz25)  tahmini    {s_top_yaz:.4f}   "
        f"= {S_YERLESIK_YAZ:.4f} x {d.toplu.mean():.3f}"
    )
    print(
        f"  sigma_TEKIL(yaz25)  tahmini    {s_tek_yaz:.4f}   "
        f"= {S_YERLESIK_YAZ:.4f} x {d.tekil.mean():.3f}"
    )

    # test karisimi
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tc = te[~te["tanim"].isin(set(tr["tanim"].unique()))]
    ilk_te = tc.groupby("tanim")["tarih"].min()
    p_te = ilk_te.groupby(ilk_te).size()
    b_te = ilk_te.map(p_te)
    tc = tc.assign(toplu=tc["tanim"].map(b_te) >= 100)
    w_top = float(tc["toplu"].mean())
    print(f"\n  TEST soguk karisimi: TOPLU {w_top:.4f}  TEKIL/orta {1 - w_top:.4f}")

    s_hedef = w_top * s_top_yaz + (1 - w_top) * s_tek_yaz
    print(
        f"  sigma_hedef (karisim) = {w_top:.3f}x{s_top_yaz:.4f} + "
        f"{1 - w_top:.3f}x{s_tek_yaz:.4f} = {s_hedef:.4f}"
    )

    c_ham = KOR * s_hedef / S_MODEL
    c_kal = c_ham * KALIBRE
    print(
        f"\n  c_capa  = kor x sigma_hedef / sigma_model = "
        f"{KOR:.4f} x {s_hedef:.4f} / {S_MODEL:.4f} = {c_ham:.3f}"
    )
    print(
        f"  yaz25 kalibrasyonu (x{KALIBRE:.2f} degil, capa ZATEN dusuk tahmin "
        f"ediyordu -> BOLUNUR): {c_ham / KALIBRE:.3f}"
    )
    print("\n  H8'in mevcut secimi c = 2,20")
    print(
        f"  H9b'nin nufus-eslesmis capasi c = {c_ham:.3f} (ham) / {c_ham / KALIBRE:.3f} (kalibreli)"
    )

    print("\n" + "=" * 96)
    print("HUKUM")
    print("=" * 96)
    if c_ham < 2.20:
        print(f"  Nufus eslesmesi c'yi ASAGI cekiyor ({c_ham:.3f} < 2,20).")
        print(f"  TOPLU kohort genligi yerlesikten {d.toplu.mean():.2f} kat.")
    else:
        print(f"  Nufus eslesmesi c'yi ASAGI CEKMIYOR ({c_ham:.3f} >= 2,20)")
        print("  -> 2,20 zaten muhafazakar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
