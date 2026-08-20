"""KAPSAM DESENI KAPISI -- feature'lar EGITIMDE ve TESTTE ayni dolulukta mi?

NEDEN BU BETIK
--------------
``scripts/veri_sagligi.py`` KAYNAK dosyalarini denetler: kapsam, ilce sayisi,
NaN orani, fizik. Ama bir kaynak kendi kapisini gecip yine de modeli
bozabilir, cunku asil soru kaynagin degil FEATURE'in sorusudur:

    Bu kolon, tahmin anininda egitimdekiyle AYNI dolulukta mi?

Bir kolon egitimde %100, test blogunda %0 doluysa model ona guvenmeyi ogrenir
ve tam teslim aninda o bilgi yok olur. Ters yon de zararlidir: egitimde %0,
testte %100 olan bir kolonu model hic ogrenemez ama ona split acar.

Bu iki desen, 2026-08-20 denetiminde GERCEKTEN bulundu (ikisi de sessizdi):

  * ``hava_saatlik``, ``hava_kalitesi``, ``nem_toprak`` son 30 gunde
    %70-77 doluydu, tum panelde %99.6. Sebep tek bir yanlis sabitti
    (``ARCHIVE_LAG_DAYS``), hicbir kaynak kapisi bagirmadi.
  * ``turizm_aylik`` panelin 2020-2023 bolumunun %0'inda, 2024-2026'nin
    %100'unde doluydu -- ilce tablosunun dar kapsamina hapsoldugu icin.

Kaynak duzeyinde ikisi de "temiz" gorunuyordu. Desen yalnizca EGITIM/TEST
ayriminda gorunur; bu betik tam olarak o ayrimi kurar.

NASIL BOLUNUYOR
---------------
Yarismalarin ezici cogunlugu ILERI BLOK ayirir: test, panelin SON N gunudur.
Burada da oyle yapiliyor -- rastgele bolme bu deseni gizler, cunku rastgele
bir test kumesi egitimle ayni tarih dagilimini tasir.

KULLANIM
--------
::

    python scripts/kapsam_deseni.py
    python scripts/kapsam_deseni.py --test-gun 60 --esik 0.05

Cikis kodu: 0 = tum kolonlar hizali, 1 = en az bir HATA.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gridup.features.external import attach_external  # noqa: E402

REFERANS = ROOT / "data" / "reference" / "ilceler_gdz_adm.parquet"

#: Panelin baslangici -- kaynaklarin kapsamasi beklenen aralikla ayni.
PANEL_BAS = pd.Timestamp("2020-01-01")

#: Varsayilan test blogu uzunlugu (gun). Bir aylik ileri blok, tipik bir
#: yarisma test penceresinin buyuklugu.
VARSAYILAN_TEST_GUN = 30

#: Egitim ve test doluluk oranlari arasinda izin verilen en buyuk fark.
#: %10 dar degil: gercek bir kaynak gecikmesi birkac puan oynatir, ama on
#: puanlik bir fark artik "bazen eksik" degil "sistematik olarak yok"tur.
VARSAYILAN_ESIK = 0.10

#: Ufuk: feature'lar bu kadar gun onceden bilinebilir olmali. Kapsam desenini
#: olcerken de ayni ufku kullaniyoruz ki kaydirmanin actigi bosluklar
#: gercekcen gorunsun.
UFUK = 1


def panel_kur(son: pd.Timestamp) -> pd.DataFrame:
    """96 ilce x [PANEL_BAS, son] tam panelini kurar."""
    if not REFERANS.is_file():
        raise FileNotFoundError(f"Ilce referansi yok: {REFERANS}")
    ilceler = pd.read_parquet(REFERANS)
    gunler = pd.date_range(PANEL_BAS, son, freq="D")
    return pd.MultiIndex.from_product(
        [ilceler["ilce_key"], gunler], names=["ilce_key", "tarih"]
    ).to_frame(index=False)


def panel_sonu() -> pd.Timestamp:
    """Panelin son gunu: GOZLENEN hava tablosunun son gercek gozlemi.

    Bugunu kullanmak yaniltici olurdu -- arsiv birkac gun geriden gelir ve
    o bosluk her kolonu ayni sekilde bos gosterip deseni gizlerdi.
    """
    yol = ROOT / "data" / "external" / "hava_gunluk.parquet"
    kolonlar = ["tarih", "hava_tahmin"]
    frame = pd.read_parquet(yol, columns=kolonlar)
    gercek = frame[~frame["hava_tahmin"].astype(bool)]
    return pd.Timestamp(pd.to_datetime(gercek["tarih"]).max()).normalize()


def desen_olc(
    frame: pd.DataFrame, kolonlar: list[str], *, test_gun: int, zaman: str = "tarih"
) -> pd.DataFrame:
    """Her kolon icin egitim/test doluluk oranlarini olcer."""
    sinir = frame[zaman].max() - pd.Timedelta(days=test_gun - 1)
    test = frame[zaman] >= sinir
    egitim = ~test

    return pd.DataFrame(
        {
            "kolon": kolonlar,
            "egitim_dolu": [float(frame.loc[egitim, k].notna().mean()) for k in kolonlar],
            "test_dolu": [float(frame.loc[test, k].notna().mean()) for k in kolonlar],
            "tekil_deger": [int(frame[k].nunique(dropna=True)) for k in kolonlar],
        }
    )


def rapor(olcum: pd.DataFrame, aile_haritasi: dict[str, str], esik: float) -> tuple[int, int]:
    """Bulgulari yazdirir; (hata, uyari) sayisini doner."""
    olcum = olcum.copy()
    olcum["aile"] = olcum["kolon"].map(aile_haritasi).fillna("?")
    olcum["fark"] = olcum["test_dolu"] - olcum["egitim_dolu"]

    # HATA: testte kayboluyor. Model ona guvenmeyi ogrenir, sonra kaybeder.
    kaybolan = olcum[olcum["fark"] < -esik].sort_values("fark")
    # UYARI: yalnizca son donemde var. Model onu ogrenemez ama split acar.
    yeni = olcum[olcum["fark"] > esik].sort_values("fark", ascending=False)
    # HATA: hicbir yerde bilgi tasimiyor.
    olu = olcum[olcum["tekil_deger"] <= 1]

    if len(kaybolan):
        print(f"\nHATA -- TESTTE KAYBOLAN {len(kaybolan)} kolon (egitimde dolu, testte bos):")
        for satir in kaybolan.itertuples():
            print(
                f"  {satir.aile:16s} {satir.kolon:38s} "
                f"egitim %{100 * satir.egitim_dolu:5.1f} -> test %{100 * satir.test_dolu:5.1f}"
            )
        print(
            "  Bu, EKSIK feature'dan daha kotudur: model kolona guvenmeyi ogrenir,\n"
            "  sonra tam teslim aninda o bilgi yoktur. Kaynak cekicisini panelin\n"
            "  sonuna kadar tekrar calistir."
        )

    if len(olu):
        print(f"\nHATA -- BILGI TASIMAYAN {len(olu)} kolon (tek deger ya da tamamen bos):")
        for satir in olu.itertuples():
            print(f"  {satir.aile:16s} {satir.kolon:38s} tekil deger: {satir.tekil_deger}")

    if len(yeni):
        print(f"\nUYARI -- YALNIZCA SON DONEMDE VAR {len(yeni)} kolon:")
        for satir in yeni.itertuples():
            print(
                f"  {satir.aile:16s} {satir.kolon:38s} "
                f"egitim %{100 * satir.egitim_dolu:5.1f} -> test %{100 * satir.test_dolu:5.1f}"
            )
        print(
            "  Model bu kolonlari egitimde neredeyse hic gormez ama testte onlara\n"
            "  split acabilir. Kapsami genis bir kaynakla degistirmek tercih edilir."
        )

    return len(kaybolan) + len(olu), len(yeni)


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--test-gun", type=int, default=VARSAYILAN_TEST_GUN)
    ayristirici.add_argument("--esik", type=float, default=VARSAYILAN_ESIK)
    ayristirici.add_argument("--kati", action="store_true", help="Uyarilari da hata say")
    args = ayristirici.parse_args()

    son = panel_sonu()
    panel = panel_kur(son)
    print(f"KAPSAM DESENI KAPISI  ({PANEL_BAS.date()} .. {son.date()})")
    print("=" * 74)
    print(
        f"  panel {len(panel):,} satir · test blogu son {args.test_gun} gun · "
        f"esik %{100 * args.esik:.0f}"
    )

    with warnings.catch_warnings():
        # Dusuk eslesme uyarilari burada BEKLENIYOR; olculen sey zaten o.
        warnings.simplefilter("ignore")
        sonuc = attach_external(
            panel, key_column="ilce_key", time_column="tarih", horizon=UFUK, root=ROOT
        )

    aile_haritasi = {k: aile for aile, kolonlar in sonuc.families.items() for k in kolonlar}
    kolonlar = [k for k in sonuc.frame.columns if k not in ("ilce_key", "tarih")]
    print(f"  {len(kolonlar)} feature kolonu · {len(sonuc.families)} aile")

    olcum = desen_olc(sonuc.frame, kolonlar, test_gun=args.test_gun)
    hata, uyari = rapor(olcum, aile_haritasi, args.esik)

    print("\n" + "=" * 74)
    print(f"{len(kolonlar)} kolon · {hata} hata · {uyari} uyari")
    if hata:
        print("\nKAPSAM DESENI BOZUK. Her HATA satiri kapatilmali.")
        return 1
    if uyari and args.kati:
        print("\n--kati modunda uyarilar da hata sayilir.")
        return 1
    print("\nTum feature'lar egitim ve test blogunda ayni dolulukta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
