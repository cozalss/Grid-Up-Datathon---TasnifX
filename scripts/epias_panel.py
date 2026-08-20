"""EPIAS plansiz kesinti gecmisini 96 ILCELIK panele cevirir ve dogrular.

NEDEN BU BETIK
--------------
2026-08-18 denetimi (P1-6): tum prova 47 GDZ ilcesinde, 2021-22 aynasinda
yapilmisti. ADM'nin 49 ilcesi, kurumsal yazimlari ve 2023-26 rejimleri hic
denenmemisti. EPIAS Seffaflik Platformu bu bosluu kapatir: 405.819 kayit,
96 ilcenin TAMAMI, 2022-01-01 .. 2026-08-17, ariza sebebi ve etkilenen abone
sayisiyla birlikte.

KAPSAMA BOSLUGU -- EN ONEMLI TUZAK
----------------------------------
EPIAS arsivi DELIK DESIKTIR (olculdu): 1690 gunun 406'sinda HIC kayit yok
(2024Q3 tamamen, 2024Q4'te tek kayit, 2025Q1 bos). Bu gunler "o gun hic
kesinti olmadi" DEGIL, "o gun yayimlanmamis"tir. Ikisini ayirt etmeden panel
kurmak, 406 gunu sahte sifir yapar ve:
  * sifir oranini sisirir (iki asamali model kararini bozar),
  * lag/rolling pencerelerine sahte sifir enjekte eder,
  * modele "kesinti azaliyor" diye YANLIS bir trend ogretir.
Bu yuzden panel ``kapsanan_gun`` bayragiyla uretilir ve varsayilan olarak
YALNIZCA kapsanan gunler dondurulur.

KULLANIM
    python scripts/epias_panel.py                 # rapor + parquet
    python scripts/epias_panel.py --tum-gunler    # bosluklari da yaz (bayrakli)
Cikti: data/external/epias/panel_ilce_gun.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.io_utils import publish_dataframe  # noqa: E402
from gridup.turkish import join_key, split_il_ilce, strip_qualifier  # noqa: E402

KAYNAK = KOK / "data" / "external" / "epias" / "kesinti_plansiz.parquet"
REFERANS = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
CIKTI = KOK / "data" / "external" / "epias" / "panel_ilce_gun.parquet"

#: Panel kolonlari. ``kapsanan_gun`` 0 ise o gun EPIAS'ta HIC kayit yoktur --
#: hedef 0 degil BILINMIYORDUR.
CIKTI_KOLONLARI = [
    "il_key",
    "ilce_key",
    "gun",
    "kesinti_adet",
    "kesinti_dk",
    "etkilenen_abone",
    "kapsanan_gun",
]


#: 2012 BUYUKSEHIR YASASI KURTARMASI (olculdu, 2026-08-18).
#: EPIAS bazi kayitlarda eski "merkez" adini kullaniyor. Yalnizca TEK bir
#: ilceye karsilik gelen adlar eslenir; belirsiz olanlar UYDURULMAZ.
#:   aydin/"aydin merkez" + "aydin" -> efeler  (Aydin'in TEK merkez ilcesi;
#:       EPIAS'ta hic "efeler" kaydi yok, 5.891 kayit boyle kurtarilir)
MERKEZ_KURTARMA: dict[tuple[str, str], str] = {
    ("aydin", "aydin merkez"): "efeler",
    ("aydin", "aydin"): "efeler",
}

#: BELIRSIZ merkez adlari: 2012'de IKI ilceye bolunduler ve EPIAS her ikisini
#: de ayrica yayimliyor. "denizli" kaydinin Merkezefendi'ye mi Pamukkale'ye mi
#: ait oldugu BILINMEZ; nufusa gore bolmek uydurma olur. Bu kayitlar panele
#: ALINMAZ ve raporda gosterilir (sessiz yanlis yerine acik eksik).
BELIRSIZ_MERKEZLER: dict[tuple[str, str], tuple[str, ...]] = {
    ("denizli", "denizli"): ("merkezefendi", "pamukkale"),
    ("manisa", "manisa"): ("sehzadeler", "yunusemre"),
}


def _ilce_anahtari(deger: object) -> str:
    """Kurumsal ilce yazimini panel anahtarina cevirir.

    Iki sinif birden ele alinir: niteleyici ("Koprubasi / Manisa" -> sol) ve
    bilesik ("izmir-karabaglar" -> sag). Ikisi ayni fonksiyonla cozulemez;
    once bilesik ayrilir, sonra niteleyici atilir (2026-08-18, P1-12).
    """
    metin = str(deger)
    _, ilce = split_il_ilce(metin)
    return join_key(strip_qualifier(ilce))


def panel_kur(ham: pd.DataFrame, referans: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Olay kaydini ilce x gun paneline cevirir; kapsama bayragiyla.

    Returns:
        ``(panel, rapor)``. Rapor eslesme, bosluk ve sifir oranlarini tasir.
    """
    ham = ham.copy()
    ham["gun"] = (
        pd.to_datetime(ham["date"], format="mixed", utc=True)
        .dt.tz_convert("Europe/Istanbul")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    basla = pd.to_datetime(ham["startTime"], format="mixed", utc=True)
    bitir = pd.to_datetime(ham["endTime"], format="mixed", utc=True)
    ham["dk"] = (bitir - basla).dt.total_seconds() / 60.0
    ham = ham[ham["dk"] >= 0]
    ham["ilce_key"] = ham["district"].map(_ilce_anahtari)
    ham["il_key"] = ham["province"].map(lambda x: join_key(str(x)))

    # Eski merkez adlarini kurtar; belirsizleri AYIR (uydurma bolme yok).
    kurtarilan = 0
    for (il, eski), yeni in MERKEZ_KURTARMA.items():
        maske = (ham["il_key"] == il) & (ham["ilce_key"] == eski)
        kurtarilan += int(maske.sum())
        ham.loc[maske, "ilce_key"] = yeni
    belirsiz_maske = pd.Series(False, index=ham.index)
    for il, eski in BELIRSIZ_MERKEZLER:
        belirsiz_maske |= (ham["il_key"] == il) & (ham["ilce_key"] == eski)
    belirsiz_sayi = int(belirsiz_maske.sum())
    ham = ham[~belirsiz_maske]

    ref_ciftler = set(zip(referans["il_key"], referans["ilce_key"], strict=True))
    ham_ciftler = set(zip(ham["il_key"], ham["ilce_key"], strict=True))
    eslesen = ham_ciftler & ref_ciftler
    ham = ham[[c in eslesen for c in zip(ham["il_key"], ham["ilce_key"], strict=True)]]

    olculen = (
        ham.groupby(["il_key", "ilce_key", "gun"])
        .agg(
            kesinti_adet=("dk", "size"),
            kesinti_dk=("dk", "sum"),
            etkilenen_abone=("effectedSubscribers", "sum"),
        )
        .reset_index()
    )

    # TAM IZGARA: 96 ilce x her gun. Kapsanan gun = EPIAS'ta o gun EN AZ BIR
    # kayit olan gun (herhangi bir ilcede). Kapsanmayan gunde hedef BILINMIYOR.
    gunler = pd.date_range(olculen["gun"].min(), olculen["gun"].max(), freq="D")
    kapsanan = set(ham["gun"].unique())
    izgara = (
        referans[["il_key", "ilce_key"]]
        .drop_duplicates()
        .merge(pd.DataFrame({"gun": gunler}), how="cross")
    )
    panel = izgara.merge(olculen, on=["il_key", "ilce_key", "gun"], how="left")
    for kolon in ("kesinti_adet", "kesinti_dk", "etkilenen_abone"):
        panel[kolon] = panel[kolon].fillna(0.0)
    panel["kapsanan_gun"] = panel["gun"].isin(kapsanan).astype("int8")
    panel = panel[CIKTI_KOLONLARI].sort_values(["il_key", "ilce_key", "gun"]).reset_index(drop=True)

    # ILCE KAPSAMASI: EPIAS'ta hic kaydi olmayan ilce "hep sifir" DEGIL,
    # "yayimlanmamis"tir. Gun bayragi bunu yakalamaz; ayrica raporlanir.
    kayitli_ilceler = set(zip(ham["il_key"], ham["ilce_key"], strict=True))
    kapsanmayan_ilce = sorted(ilce for il, ilce in ref_ciftler if (il, ilce) not in kayitli_ilceler)

    kapsanan_panel = panel[panel["kapsanan_gun"] == 1]
    rapor: dict[str, object] = {
        "ham_kayit": int(len(ham)),
        "referansla_eslesen_ilce": len({c[1] for c in eslesen}),
        "eslesmeyen_ilce": sorted({c[1] for c in ham_ciftler - ref_ciftler})[:10],
        "merkez_kurtarilan_kayit": kurtarilan,
        "belirsiz_merkez_kayit": belirsiz_sayi,
        "epias_kaydi_olmayan_ilce": kapsanmayan_ilce,
        "gun_araligi": (str(gunler.min().date()), str(gunler.max().date())),
        "toplam_gun": int(len(gunler)),
        "kapsanan_gun": int(len(kapsanan)),
        "bos_gun": int(len(gunler) - len(kapsanan)),
        "panel_satir": int(len(panel)),
        "kapsanan_satir": int(len(kapsanan_panel)),
        "sifir_orani_kapsanan": float((kapsanan_panel["kesinti_adet"] == 0).mean()),
        "sifir_orani_hepsi": float((panel["kesinti_adet"] == 0).mean()),
        "gdz_ilce": int(referans[referans["sirket"] == "GDZ"]["ilce_key"].nunique())
        if "sirket" in referans.columns
        else None,
    }
    return panel, rapor


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tum-gunler",
        action="store_true",
        help="Kapsanmayan gunleri de yaz (kapsanan_gun=0 bayragiyla)",
    )
    parser.add_argument("--out", default=str(CIKTI))
    args = parser.parse_args()

    if not KAYNAK.exists():
        print(f"HATA: {KAYNAK} yok. Once scripts/fetch_epias_outages.py calistir.")
        return 1

    ham = pd.read_parquet(KAYNAK)
    referans = pd.read_parquet(REFERANS)
    panel, rapor = panel_kur(ham, referans)

    print(f"Ham kayit          : {rapor['ham_kayit']:,}")
    print(f"Eslesen ilce       : {rapor['referansla_eslesen_ilce']}/96")
    if rapor["eslesmeyen_ilce"]:
        print(f"  Eslesmeyen (ilk 10): {rapor['eslesmeyen_ilce']}")
    print(
        f"Merkez kurtarma    : {rapor['merkez_kurtarilan_kayit']:,} kayit eski adiyla "
        f"geliyordu (or. 'Aydin Merkez' -> efeler)"
    )
    print(
        f"Belirsiz merkez    : {rapor['belirsiz_merkez_kayit']:,} kayit DISARIDA "
        "(Denizli/Manisa merkezi 2012'de IKIYE bolundu; hangisine ait belli degil)"
    )
    if rapor["epias_kaydi_olmayan_ilce"]:
        print(f"EPIAS'ta hic kaydi olmayan ilce: {rapor['epias_kaydi_olmayan_ilce']}")
    print(f"Gun araligi        : {rapor['gun_araligi'][0]} .. {rapor['gun_araligi'][1]}")
    print(
        f"Kapsama            : {rapor['kapsanan_gun']}/{rapor['toplam_gun']} gun "
        f"({rapor['bos_gun']} gun EPIAS'ta YOK -- sifir DEGIL, bilinmiyor)"
    )
    print(
        f"Sifir orani        : kapsanan gunlerde %{rapor['sifir_orani_kapsanan'] * 100:.1f}; "
        f"tum izgarada %{rapor['sifir_orani_hepsi'] * 100:.1f} (sahte sifirlarla sisirilmis)"
    )

    yazilacak = panel if args.tum_gunler else panel[panel["kapsanan_gun"] == 1]
    publish_dataframe(
        yazilacak.reset_index(drop=True),
        Path(args.out),
        required_columns=CIKTI_KOLONLARI,
        min_rows=1000,
        source="epias://transparency/electricity-service-quality/unplanned-outage",
    )
    print(f"\nYazildi: {args.out}  ({len(yazilacak):,} satir)")
    if not args.tum_gunler:
        print("  (yalnizca kapsanan gunler; bosluklar icin --tum-gunler)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
