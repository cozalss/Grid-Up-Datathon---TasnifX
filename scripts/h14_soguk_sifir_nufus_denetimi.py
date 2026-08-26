"""H14 -- "%46 DUVARI" DOGRU NUFUSTA MI OLCULDU?

DENETLENEN IDDIA (docs/41 §2-3, projenin en merkezi sayisi)
-----------------------------------------------------------
    "Agirlikli kutlenin %6'si olan y=0 satirlari soguk MSE'nin %72,5'ini
     tasiyor. Soguk da toplam MSE'nin %63'u -> sifir satirlari tum yarisma
     hatasinin ~%46'si."
    "Toplam MSE'nin ~%46'si mevcut oznitelik kumesiyle INDIRGENEMEZ.
     '1'in alti' hedefinin onundeki asil duvar budur."

Bu cumle stratejiyi belirledi. Ama OLCULDUGU YER: **kis26 soguk**,
61.918 satir, 1.223 trafo. Ayni taban (RMSLE 1,98505) hurdle (§3) ve
kalibrasyon (§4) hukumlerinde de kullanildi -- uc eksen, tek blok.

SUPHE (KUSUR B -- YANLIS NUFUS)
-------------------------------
TEST soguk satirlarinin **%82,5'i >=100'luk TOPLU KATILIM'dan** geliyor
(tik 3'te olculdu). Ve ``son_islem_olay.py`` belgeliyor:

    "PARTI BUYUKLUGU BELIRLEYICI: ayni gun 100'den fazla trafo dogduysa
     dusus neredeyse YOK (-0,11). Bu bir enerjilendirme dalgasi degil,
     veri setine TOPLU KATILIM (geriye dolgu) -- olculen gun tamdir."

Geriye dolgu edilen bir trafo ZATEN CALISIYOR. Gercekten yeni
enerjilendirilmis bir trafo ise ilk aylarinda musteri baglanana kadar
SIFIR uretebilir. Ikisinin sifir orani ayni olmak zorunda DEGIL.

``kis26`` sogugunun toplu-katilim payi HIC OLCULMEDI.

BU BETIK
--------
1. kis26 soguk ve TEST soguk nufuslarinin parti kompozisyonunu olcer.
2. Train'de, ETIKETLI, kohort tipine gore SIFIR ORANINI olcer
   (mevsim kontrollu: her kohort kendi penceresinde, yerlesik referansiyla).
3. Test-benzeri karisim icin beklenen sifir oranini yeniden agirliklandirir.
4. "%46" sayisini test karisimi icin yeniden turetir.

Cikan sayi kis26'nınkinden BELIRGIN kucukse duvar baska yerdedir ve
hurdle/kalibrasyon hukumleri YANLIS NUFUSTA verilmistir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
KIS26_KESME = pd.Timestamp("2025-11-30")
KIS26_BAS, KIS26_BIT = pd.Timestamp("2025-12-01"), pd.Timestamp("2026-03-31")
ILK_GUN_AT = 7


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk = tr.groupby("tanim")["tarih"].min()
    parti = ilk.groupby(ilk).size()
    trafo_parti = ilk.map(parti)

    print("=" * 94)
    print("1. NUFUS KOMPOZISYONU -- kis26 soguk vs TEST soguk")
    print("=" * 94)

    # --- kis26 soguk: kesmeden SONRA dogmus trafolar
    k_soguk = ilk[(ilk > KIS26_KESME) & (ilk <= KIS26_BIT)].index
    kb = trafo_parti.reindex(k_soguk)
    kis_satir = tr[
        tr["tanim"].isin(set(k_soguk)) & (tr["tarih"] >= KIS26_BAS) & (tr["tarih"] <= KIS26_BIT)
    ]
    print(
        f"\nkis26 SOGUK: {len(k_soguk):,} trafo, {len(kis_satir):,} satir "
        f"(docs/41: 1.223 trafo / 61.918 satir)"
    )
    for ad, m in (
        ("TOPLU >=100", kb >= 100),
        ("orta 20-99", (kb >= 20) & (kb < 100)),
        ("tekil/kucuk <20", kb < 20),
    ):
        t = set(kb[m].index)
        s = kis_satir[kis_satir["tanim"].isin(t)]
        print(
            f"  {ad:<18} {len(t):>5,} trafo  {len(s):>8,} satir  "
            f"SATIR PAYI {len(s) / max(len(kis_satir), 1):>7.4f}"
        )

    # --- TEST soguk
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
    print(f"\nTEST SOGUK:  {tc.tanim.nunique():,} trafo, {len(tc):,} satir")
    agirlik = {}
    for ad, lo, hi in (
        ("TOPLU >=100", 100, 10**9),
        ("orta 20-99", 20, 100),
        ("tekil/kucuk <20", 0, 20),
    ):
        t = set(b_te[(b_te >= lo) & (b_te < hi)].index)
        s = tc[tc["tanim"].isin(t)]
        w = len(s) / len(tc)
        agirlik[ad] = w
        print(f"  {ad:<18} {len(t):>5,} trafo  {len(s):>8,} satir  SATIR PAYI {w:>7.4f}")

    # ---------------------------------------------------------------
    print("\n" + "=" * 94)
    print("2. SIFIR ORANI -- kohort tipine gore, MEVSIM KONTROLLU (train, ETIKETLI)")
    print("=" * 94)
    tr["ilk"] = tr["tanim"].map(ilk)
    tr["yas"] = (tr["tarih"] - tr["ilk"]).dt.days
    tr["sifir"] = tr["tuketim"] == 0

    buyuk = parti[parti >= 100].sort_index()
    satirlar = []
    for gun in buyuk.index:
        bit = min(gun + pd.Timedelta(days=127), tr["tarih"].max())
        if (bit - gun).days < 45:
            continue
        pen = tr[(tr["tarih"] >= gun) & (tr["tarih"] <= bit)]
        kohort = set(ilk[ilk == gun].index)
        pd_ilk = ilk[(ilk >= gun) & (ilk <= bit)]
        p2 = pd_ilk.groupby(pd_ilk).size()
        tekil = set(pd_ilk[pd_ilk.map(p2) < 20].index)
        ngun = pen["tarih"].nunique()
        say = pen.groupby("tanim")["tarih"].nunique()
        yerlesik = set(say[say >= 0.9 * ngun].index)

        def oran(t: set, yasli: bool) -> tuple[float, int]:
            a = pen[pen["tanim"].isin(t)]
            if yasli:
                a = a[a["yas"] >= ILK_GUN_AT]
            return (float(a["sifir"].mean()) if len(a) else float("nan")), len(a)

        o_top, n_top = oran(kohort, True)
        o_tek, n_tek = oran(tekil, True)
        o_yer, n_yer = oran(yerlesik, False)
        print(f"\n  kohort {gun.date()}  pencere {gun.date()}..{bit.date()}")
        print(f"    TOPLU     sifir orani {o_top:>7.4f}  ({n_top:>7,} satir)")
        print(f"    TEKIL     sifir orani {o_tek:>7.4f}  ({n_tek:>7,} satir)")
        print(f"    YERLESIK  sifir orani {o_yer:>7.4f}  ({n_yer:>7,} satir)")
        if o_yer and o_yer > 0:
            print(
                f"    ORAN TOPLU/YERLESIK {o_top / o_yer:>6.2f}   "
                f"TEKIL/YERLESIK {o_tek / o_yer:>6.2f}"
            )
        satirlar.append(
            {"kohort": str(gun.date()), "toplu": o_top, "tekil": o_tek, "yerlesik": o_yer}
        )

    d = pd.DataFrame(satirlar)
    print(
        f"\n  ORTALAMA  TOPLU {d.toplu.mean():.4f}   TEKIL {d.tekil.mean():.4f}   "
        f"YERLESIK {d.yerlesik.mean():.4f}"
    )
    print(f"  TEKIL / TOPLU = {d.tekil.mean() / max(d.toplu.mean(), 1e-9):.2f} kat")

    # ---------------------------------------------------------------
    print("\n" + "=" * 94)
    print("3. kis26 SOGUGUN GERCEK SIFIR ORANI ve TEST-BENZERI KARISIM")
    print("=" * 94)
    kis_sifir = float(kis_satir["tuketim"].eq(0).mean())
    print(
        f"\n  kis26 soguk GERCEK sifir orani  {kis_sifir:.4f}  "
        f"({int(kis_satir['tuketim'].eq(0).sum()):,} / {len(kis_satir):,})"
    )
    print(f"  docs/41 §2: 3.250 / 61.918 = {3250 / 61918:.4f}")

    # kis26 icindeki kohort tipine gore sifir orani (dogrudan)
    print("\n  kis26 soguk icinde, kohort tipine gore:")
    for ad, lo, hi in (
        ("TOPLU >=100", 100, 10**9),
        ("orta 20-99", 20, 100),
        ("tekil/kucuk <20", 0, 20),
    ):
        t = set(kb[(kb >= lo) & (kb < hi)].index)
        s = kis_satir[kis_satir["tanim"].isin(t)]
        if len(s):
            print(
                f"    {ad:<18} sifir orani {s['tuketim'].eq(0).mean():>7.4f}  ({len(s):>7,} satir)"
            )
        else:
            print(f"    {ad:<18} YOK")

    # test-benzeri karisim
    tip_oran = {}
    for ad, lo, hi in (
        ("TOPLU >=100", 100, 10**9),
        ("orta 20-99", 20, 100),
        ("tekil/kucuk <20", 0, 20),
    ):
        t = set(kb[(kb >= lo) & (kb < hi)].index)
        s = kis_satir[kis_satir["tanim"].isin(t)]
        tip_oran[ad] = float(s["tuketim"].eq(0).mean()) if len(s) else float("nan")

    gecerli = {k: v for k, v in tip_oran.items() if not np.isnan(v)}
    if gecerli:
        pay = sum(agirlik[k] * v for k, v in gecerli.items())
        w = sum(agirlik[k] for k in gecerli)
        if w > 0:
            beklenen = pay / w
            print("\n  TEST karisimiyla yeniden agirliklandirilmis sifir orani:")
            print(
                f"    {' + '.join(f'{agirlik[k]:.3f}x{v:.4f}' for k, v in gecerli.items())}"
                f" / {w:.3f} = {beklenen:.4f}"
            )
            print(f"    kis26'nin kendi karisimi     {kis_sifir:.4f}")
            print(f"    ORAN test/kis26              {beklenen / max(kis_sifir, 1e-9):.3f}")
            print(
                f"\n  '%46 duvari' test karisiminda kabaca "
                f"{46 * beklenen / max(kis_sifir, 1e-9):.1f}% olur."
            )
    print("\n" + "=" * 94)
    print("HUKUM")
    print("=" * 94)
    print("  Oran 1'e yakinsa duvar gercek ve hukumler saglam -> DENETIM TEMIZ.")
    print("  Oran belirgin <1 ise duvar test'te DAHA KUCUK ve hurdle/kalibrasyon")
    print("  hukumleri YANLIS NUFUSTA verilmistir -> eksen yeniden acilir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
