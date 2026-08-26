"""H9 -- NUFUS ESLESMESI: yaz25'in dogmuslari TEST'in sogugunun ikizi mi?

SORU
----
H8'in capasi sigma_gercek = 0,4255'e dayaniyor ve bu, 2025 Nis-Tem'de DOGMUS
3.074 trafodan olculdu. Ama TEST'in soguk nufusu ayni sey mi?

Test soguk satirlarinin ~%68'i TEK bir toplu katilimdan geliyor (2026-05-11,
1.326 yeni trafo). yaz25'in dogmuslari ise buyuk olcude TEKIL/kucuk parti
olabilir. Ve ``son_islem_olay.py`` zaten belgeliyor:

    "PARTI BUYUKLUGU BELIRLEYICI: ayni gun 100'den fazla trafo dogduysa dusus
     neredeyse YOK (-0,11). Bu bir enerjilendirme dalgasi degil, veri setine
     TOPLU KATILIM (geriye dolgu) -- olculen gun tamdir."

Eger toplu katilim "zaten calisan trafolarin veri setine alinmasi" ise, o
trafolarin gun ekseni genligi YERLESIK profiline (0,2710) daha yakin olur ve
0,4255 test icin FAZLA YUKSEK kalir -> c 2,20'nin ALTINDA olmali.

Tersi de mumkun: toplu katilim gercekten yeni tesis dalgasiysa genlik yuksek
kalir ve 2,20 zaten muhafazakardir.

YONTEM
------
1. yaz25 dogumlarini PARTI BUYUKLUGUNE gore ayir (dogum gununde ayni gun
   dogan trafo sayisi): tekil/kucuk <20, orta 20-99, TOPLU >=100.
2. Her grubun GERCEK gun ekseni sigma'sini AYRI olc (ayni protokol).
3. Test soguk karisimini ayni siniflarla cikar (agirlik = SATIR payi).
4. Karisim-agirlikli sigma_hedef'i hesapla ve c'yi yeniden turet.
5. Kural 1: her grup icin kirpma da raporlanir (satir/trafo sayilariyla).

KURAL: referans HER ZAMAN gercek etiketlerden ve HEDEF NUFUSUN IKIZINDEN.
Test etiketi KULLANILMAZ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
SAMPIYON = "tuketim_v67_c1335_olay.csv"
CAPA_BASI, CAPA_SONU = "2025-04-01", "2025-07-31"
MIN_YAS, MIN_GUN = 7, 60


def gun_etkisi(tanim, gun, r) -> pd.Series:
    """son_islem_gunolcek.py ile birebir ayni fonksiyon."""
    x = pd.DataFrame({"t": tanim, "g": gun, "r": r})
    x["c"] = x["r"] - x.groupby("t")["r"].transform("mean")
    b = x.groupby("g")["c"].mean()
    return b - b.mean()


def sinif(n: int) -> str:
    if n >= 100:
        return "TOPLU >=100"
    if n >= 20:
        return "orta 20-99"
    return "tekil/kucuk <20"


def main() -> int:
    tr = pd.read_csv(
        KOK / "data/raw/train.csv",
        usecols=["tanim", "guc", "tarih", "tuketim"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    ilk_tum = tr.groupby("tanim")["tarih"].min()

    # ---------- 1. yaz25 DOGUMLARI, parti buyuklugune gore ----------
    print("=" * 96)
    print("1. yaz25 DOGUMLARI -- parti buyuklugune gore GERCEK gun ekseni genligi")
    print("=" * 96)
    dogum = ilk_tum[(ilk_tum >= pd.Timestamp(CAPA_BASI)) & (ilk_tum <= pd.Timestamp(CAPA_SONU))]
    parti = dogum.groupby(dogum).size()  # dogum gunu -> o gun dogan trafo sayisi
    trafo_parti = dogum.map(parti)  # trafo -> kendi partisinin buyuklugu
    trafo_sinif = trafo_parti.map(sinif)

    g = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)].copy()
    g["sinif"] = g["tanim"].map(trafo_sinif)
    g["rg"] = np.log1p(g["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        g["guc"].to_numpy(dtype="float64")
    )
    g["ilk"] = g["tanim"].map(ilk_tum)
    g["yas"] = (g["tarih"] - g["ilk"]).dt.days

    print(f"\n  yaz25 dogum sayisi {len(dogum):,}  |  farkli dogum gunu {parti.size}")
    print(f"  en buyuk partiler: {parti.nlargest(5).to_dict()}")

    print(
        f"\n  {'sinif':<18} {'trafo':>7} {'satir':>9} {'sigma_HAM':>10} "
        f"{'sigma_T3':>10} {'T3 trafo':>9} {'T3 satir':>9}"
    )
    sigma = {}
    for s in ("tekil/kucuk <20", "orta 20-99", "TOPLU >=100"):
        alt = g[g["sinif"] == s]
        if alt.empty:
            print(f"  {s:<18} {'YOK':>7}")
            continue
        b = gun_etkisi(alt["tanim"].to_numpy(), alt["tarih"].to_numpy(), alt["rg"].to_numpy())
        # T3 temiz: ilk 7 gun atilmis + >=60 gunluk trafolar
        say = alt.groupby("tanim")["tanim"].transform("size").to_numpy()
        t3 = (alt["yas"].to_numpy() >= MIN_YAS) & (say >= MIN_GUN)
        a3 = alt[t3]
        b3 = (
            gun_etkisi(a3["tanim"].to_numpy(), a3["tarih"].to_numpy(), a3["rg"].to_numpy())
            if len(a3) > 500
            else pd.Series([np.nan])
        )
        sigma[s] = {
            "ham": float(b.std()),
            "t3": float(b3.std()),
            "trafo": alt.tanim.nunique(),
            "satir": len(alt),
        }
        print(
            f"  {s:<18} {alt.tanim.nunique():>7,} {len(alt):>9,} "
            f"{b.std():>10.4f} {b3.std():>10.4f} {a3.tanim.nunique():>9,} {len(a3):>9,}"
        )

    # referans: yerlesik nufus
    x = pd.DataFrame({"t": g["tanim"].to_numpy(), "gg": g["tarih"].to_numpy()})
    tumu = tr[(tr["tarih"] >= CAPA_BASI) & (tr["tarih"] <= CAPA_SONU) & (tr["tuketim"] > 0)]
    xx = pd.DataFrame({"t": tumu["tanim"].to_numpy(), "gg": tumu["tarih"].to_numpy()})
    say2 = xx.groupby("t")["gg"].nunique()
    yerlesik = set(say2[say2 >= 0.9 * xx["gg"].nunique()].index)
    sy = tumu[tumu["tanim"].isin(yerlesik)]
    ry = np.log1p(sy["tuketim"].to_numpy(dtype="float64")) - np.log1p(
        sy["guc"].to_numpy(dtype="float64")
    )
    b_y = gun_etkisi(sy["tanim"].to_numpy(), sy["tarih"].to_numpy(), ry)
    print(f"\n  {'YERLESIK (referans)':<18} {len(yerlesik):>7,} {len(sy):>9,} {b_y.std():>10.4f}")

    # ---------- 2. TEST soguk karisimi ----------
    print("\n" + "=" * 96)
    print("2. TEST SOGUK KARISIMI -- ayni siniflarla")
    print("=" * 96)
    te = pd.read_csv(
        KOK / "data/raw/test.csv",
        usecols=["id", "tanim", "tarih"],
        dtype={"tanim": str},
        parse_dates=["tarih"],
    )
    tc = te[~te["tanim"].isin(set(tr["tanim"].unique()))].reset_index(drop=True)
    ilk_te = tc.groupby("tanim")["tarih"].min()
    parti_te = ilk_te.groupby(ilk_te).size()
    sinif_te = ilk_te.map(parti_te).map(sinif)  # trafo -> parti -> sinif
    tc["sinif"] = tc["tanim"].map(sinif_te)

    print(f"\n  test soguk trafo {tc.tanim.nunique():,}  satir {len(tc):,}")
    print(f"  en buyuk partiler: {parti_te.nlargest(4).to_dict()}")
    print(f"\n  {'sinif':<18} {'trafo':>7} {'satir':>9} {'SATIR PAYI':>11}")
    agirlik = {}
    for s in ("tekil/kucuk <20", "orta 20-99", "TOPLU >=100"):
        alt = tc[tc["sinif"] == s]
        w = len(alt) / len(tc)
        agirlik[s] = w
        print(f"  {s:<18} {alt.tanim.nunique():>7,} {len(alt):>9,} {w:>11.4f}")

    # ---------- 3. KARISIM-AGIRLIKLI HEDEF ve YENI c ----------
    print("\n" + "=" * 96)
    print("3. KARISIM-AGIRLIKLI sigma_hedef ve YENIDEN TURETILEN c")
    print("=" * 96)
    for anahtar, ad in (("ham", "HAM protokol"), ("t3", "T3 temiz protokol")):
        pay = 0.0
        toplam_w = 0.0
        parcalar = []
        for s, w in agirlik.items():
            if s in sigma and not np.isnan(sigma[s][anahtar]):
                pay += w * sigma[s][anahtar]
                toplam_w += w
                parcalar.append(f"{w:.3f}x{sigma[s][anahtar]:.4f}")
        if toplam_w == 0:
            continue
        s_hedef = pay / toplam_w
        print(f"\n  {ad}:  sigma_hedef = ({' + '.join(parcalar)}) / {toplam_w:.3f} = {s_hedef:.4f}")
        print(
            f"    H8'in kullandigi (sinif ayirmadan) = {0.4255 if anahtar == 'ham' else 0.3829:.4f}"
        )
        eski = 0.4255 if anahtar == "ham" else 0.3829
        print(
            f"    oran {s_hedef / eski:.3f}  ->  c yeniden: 2.20 x {s_hedef / eski:.3f}"
            f" = {2.20 * s_hedef / eski:.3f}"
        )

    print("\n" + "=" * 96)
    print("HUKUM")
    print("=" * 96)
    print("  Cikan c 2,20'nin ALTINDAysa H8'i o yone cek (v71/v72 yeniden uretilir).")
    print("  USTUNDEyse 2,20 zaten muhafazakar demektir ve oyle kalir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
