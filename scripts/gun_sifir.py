"""GUN SIFIR: veri gelmeden once "her sey hazir mi?" tek komutluk KANITI.

NEDEN AYRI BIR BETIK (``ekip_kontrol.py`` varken)
-------------------------------------------------
``ekip_kontrol.py`` KURULUM doktorudur: python surumu, paketler, import,
dosya varligi, konsol kodlamasi. "Bu makinede calisir mi?" sorusunu yanitlar.

Bu betik baska bir soruyu yanitlar: **"yarin veri geldiginde hat calisacak
mi?"** Ikisi ayni sey degil. 2026-08-21 provasinda kurulum 7/7 PASS'ti ve
ayni anda uc ayri gun-1 arizasi acikta duruyordu -- ucu de sessizdi, ucu de
yalnizca gercek bicimde bir dosya uzerinde kosunca ortaya cikti.

Kosulan kapilar, ucuzdan pahaliya:

  1. UYGUNLUK   -- yarisma hedefinin gecmisi modele girdi olabiliyor mu?
                   (eleme riski; gridup.uygunluk)
  2. VERI SAGLIGI -- 20 kaynak: kapsam, butunluk, fizik (veri_sagligi.py)
  3. TAKVIM     -- dis veri yarisma BITISINE kadar uzaniyor mu?
  4. DUSMANCA PROVA -- gercek verinin hasim bicimli kopyasinda uctan uca
                   day_one.py (dusmanca_prova.py). En pahali kapi, en cok
                   sey soyleyen kapi.

Hepsi yesilse cikis kodu 0 ve ekrana VERI GUNU SIRASI basilir.

    python scripts/gun_sifir.py
    python scripts/gun_sifir.py --hizli     # dusmanca provayi atla
    python scripts/gun_sifir.py --sadece-plan
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.uygunluk import (  # noqa: E402
    kaynak_ihlallerini_tara,
    model_girdisi_yasak_yollar,
)

#: Yarismanin son gonderim gunu. Dis veri EN AZ buraya kadar uzanmali;
#: uzanmiyorsa test blogunun son gunlerinde bazi feature'lar NaN olur ve bu,
#: CV'de GORUNMEYEN bir bozulmadir (model egitimde o kolonlara guvenmeyi
#: ogrenir, teslim aninda yoklardir).
YARISMA_BITISI = pd.Timestamp("2026-09-01")

#: Kopru TEK SEFERLIK degildir -- yarisma boyunca tazelenmelidir.
#:
#: OLCULDU (2026-08-21, Open-Meteo forecast API'sine dogrudan sorularak):
#:
#:     cape, wind_speed_10m, wind_gusts_10m, pressure_msl  ->  384 saat (16 gun)
#:     soil_moisture_0_to_1cm                              ->  184 saat (~7,7 gun)
#:
#: ``kopru_saatlik.py`` bilerek EN ZAYIF kaynaga kirpar (bkz. ``ileri_ufuk``):
#: bazi ailelerin dolu bazilarinin bos oldugu bir aralik, hepsinin eksik
#: oldugu bir araliktan daha tehlikelidir -- model egitimde o kolonlara
#: guvenmeyi ogrenir, teslim aninda yoklardir.
#:
#: Pratik sonuc: toprak nemi ufku HER ZAMAN "bugun + ~7 gun" oldugu icin
#: koprü, kapsanmasi istenen son gunden en fazla 7 gun once kosulmalidir.
#: 2026-09-01'i kapsamak icin en gec 2026-08-25 civari tazelenmeli.
KOPRU_TAZELEME_NOTU = (
    "NOT: toprak nemi ufku ~7 gun (olculdu) -- koprü yarisma icinde "
    "TEKRAR kosulmali; 09-01 icin en gec 08-25 civari."
)

#: Takvim kapisinin baktigi zaman serili tablolar: (yol, tarih kolonu).
ZAMANLI_TABLOLAR: tuple[tuple[str, str], ...] = (
    ("data/external/hava_gunluk.parquet", "tarih"),
    ("data/external/hava_saatlik_turev.parquet", "tarih"),
    ("data/external/hava_kalitesi_gunluk.parquet", "tarih"),
    ("data/external/konvektif_gunluk.parquet", "tarih"),
    ("data/external/nem_toprak_gunluk.parquet", "tarih"),
    ("data/external/gunes_gunluk.parquet", "tarih"),
)

#: Acilis yayininda SORULACAK sorular. Cevaplari stratejiyi degistirir, bu
#: yuzden tahmin edilmez -- sorulur.
ACILIS_SORULARI = (
    "Resmi metrik nedir? (2024 GDZ emsali MAE; dogrulanmali)",
    "Harici veri kullanimi serbest mi? (2023 GDZ ACIKCA tesvik etti)",
    "EPIAS/kamuya acik KESINTI GECMISI kullanilabilir mi? "
    "(GDZ'22 Case-1 bunu ACIKCA yasakladi -- cevap 'hayir' ise elimizdeki "
    "405 bin kayit yalnizca prova zeminidir, modele giremez)",
    "Ticari olmayan lisansli model agirliklari (TabPFN-2.5) serbest mi?",
    "Gunluk gonderim hakki kac? Final kac gonderim secilir?",
    "Notebook degerlendirmesinde tam olarak neye bakiliyor? (rubrik)",
)

#: Veri geldiginde kosulacak komut sirasi. Sira ONEMLIDIR: her adim bir
#: oncekinin dogruladigi seye yaslanir.
VERI_GUNU_SIRASI = (
    (
        "Ham dosyalari data/raw/ altina ac",
        "(Kaggle'dan indir; ic ice zip varsa duz ac -- day_one alt dizinleri de tarar)",
    ),
    (
        "Bicimi OKUMADAN once tespit et",
        "python -c \"import sys;sys.path.insert(0,'src');"
        "from gridup import sniff_dialect;print(sniff_dialect('data/raw/train.csv'))\"",
    ),
    (
        "ILK GONDERIM -- tek komut, ~1 dakika",
        "python scripts/day_one.py --data data/raw --metric <RESMI_METRIK>",
    ),
    (
        "Gonderimi Kaggle'a yukle, LB skorunu deftere yaz",
        "(format dogru mu, once bunu gor -- iyi skor sonra)",
    ),
    (
        "LB ile CV arasindaki farki OLC",
        "python scripts/benchmark_gercek.py    # harman + toplayici karsilastirmasi",
    ),
    (
        "Hangi aile gercekten faydali? (cok tohumlu hukum)",
        "python scripts/ablation_gercek.py --tohum 5",
    ),
    (
        "Hiperparametre -- ancak yukaridakiler oturduktan SONRA",
        "python scripts/tune_gercek.py --model catboost",
    ),
)


@dataclass(frozen=True)
class KapiSonucu:
    """Tek bir kapinin sonucu."""

    ad: str
    gecti: bool
    detay: str

    def __str__(self) -> str:
        isaret = "GECTI" if self.gecti else "HATA "
        return (
            f"  [{isaret}] {self.ad}\n           {self.detay}"
            if self.detay
            else (f"  [{isaret}] {self.ad}")
        )


def _betik_kos(ad: str, komut: list[str], *, basari_kodu: int = 0) -> KapiSonucu:
    """Alt betigi kosar; ciktinin SON anlamli satirini detay olarak tasir."""
    sonuc = subprocess.run(  # noqa: S603 -- sabit komut listesi, kullanici girdisi yok
        komut,
        cwd=KOK,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,  # cikis kodunu KapiSonucu tasir; istisna atmak raporu keserdi
    )
    ciktilar = [s for s in (sonuc.stdout or "").splitlines() if s.strip()]
    detay = ciktilar[-1].strip() if ciktilar else (sonuc.stderr or "").strip()[:200]
    return KapiSonucu(ad=ad, gecti=sonuc.returncode == basari_kodu, detay=detay)


def uygunluk_kapisi() -> KapiSonucu:
    """Yarisma hedefinin gecmisi modele girdi olabiliyor mu?"""
    yasak = model_girdisi_yasak_yollar()
    ihlaller = kaynak_ihlallerini_tara()
    if not yasak:
        return KapiSonucu(
            ad="uygunluk",
            gecti=False,
            detay="Manifestte hic 'model_girdisi: false' isareti yok -- kapi KAPALI DEGIL.",
        )
    if ihlaller:
        satirlar = "; ".join(f"{d}:{s}" for d, s, _ in ihlaller[:3])
        return KapiSonucu(ad="uygunluk", gecti=False, detay=f"IHLAL: {satirlar}")
    return KapiSonucu(
        ad="uygunluk",
        gecti=True,
        detay=f"{len(yasak)} artifact modele kapali, modelleme kutuphanesinde 0 referans.",
    )


def takvim_kapisi(bitis: pd.Timestamp = YARISMA_BITISI) -> KapiSonucu:
    """Dis veri yarismanin son gunune kadar uzaniyor mu?"""
    eksikler: list[str] = []
    bulunan = 0
    for gorece, tarih_kolonu in ZAMANLI_TABLOLAR:
        yol = KOK / gorece
        if not yol.exists():
            eksikler.append(f"{Path(gorece).name}=YOK")
            continue
        son = pd.to_datetime(pd.read_parquet(yol, columns=[tarih_kolonu])[tarih_kolonu]).max()
        bulunan += 1
        if son < bitis:
            eksikler.append(f"{Path(gorece).name}={son.date()} ({(bitis - son).days} gun eksik)")

    if eksikler:
        return KapiSonucu(
            ad="takvim",
            gecti=False,
            detay=(
                f"Yarisma {bitis.date()}'de bitiyor ama su tablolar yetismiyor: "
                f"{', '.join(eksikler)}.\n"
                "           Duzeltme: python scripts/fetch_weather_bridge.py "
                "&& python scripts/kopru_saatlik.py\n"
                f"           {KOPRU_TAZELEME_NOTU}"
            ),
        )
    return KapiSonucu(
        ad="takvim", gecti=True, detay=f"{bulunan} zamanli tablo {bitis.date()} sonrasina uzaniyor."
    )


def plani_yazdir() -> None:
    print("\n" + "=" * 78)
    print("VERI GUNU SIRASI")
    print("=" * 78)
    for numara, (baslik, komut) in enumerate(VERI_GUNU_SIRASI, start=1):
        print(f"\n{numara}. {baslik}")
        print(f"   {komut}")

    print("\n" + "=" * 78)
    print("ACILIS YAYININDA SOR")
    print("=" * 78)
    for numara, soru in enumerate(ACILIS_SORULARI, start=1):
        print(f"{numara}. {soru}")

    print("\n" + "=" * 78)
    print("ATIF HUCRESI (notebook'a AYNEN kopyala -- lisans zorunlulugu)")
    print("=" * 78)
    print(
        "Weather data by Open-Meteo.com (CC BY 4.0)\n"
        "ESA WorldCover 10m v200 (CC BY 4.0)\n"
        "Map data from OpenStreetMap contributors (ODbL 1.0)\n"
        "TUIK turizm istatistikleri | AFAD deprem katalogu | NASA FIRMS yangin tespitleri"
    )


def main() -> int:
    ayristirici = argparse.ArgumentParser(description=__doc__)
    ayristirici.add_argument("--hizli", action="store_true", help="Dusmanca provayi atla")
    ayristirici.add_argument("--sadece-plan", action="store_true", help="Kapilari atla, plani yaz")
    args = ayristirici.parse_args()

    if args.sadece_plan:
        plani_yazdir()
        return 0

    print("=" * 78)
    print(f"GUN SIFIR HAZIRLIK KANITI   (yarisma bitisi {YARISMA_BITISI.date()})")
    print("=" * 78)

    kapilar = [uygunluk_kapisi(), takvim_kapisi()]
    for kapi in kapilar:
        print(kapi)

    kapilar.append(_betik_kos("veri sagligi", [sys.executable, "scripts/veri_sagligi.py"]))
    print(kapilar[-1])

    if not args.hizli:
        print("  [ ...  ] dusmanca prova kosuyor (birkac dakika)")
        kapilar.append(_betik_kos("dusmanca prova", [sys.executable, "scripts/dusmanca_prova.py"]))
        print(kapilar[-1])
    else:
        print("  [ATLA ] dusmanca prova (--hizli)")

    basarisiz = [k for k in kapilar if not k.gecti]
    print("\n" + "=" * 78)
    if basarisiz:
        print(f"HAZIR DEGIL -- {len(basarisiz)} kapi kirmizi:")
        for kapi in basarisiz:
            print(f"  - {kapi.ad}: {kapi.detay}")
        print("=" * 78)
        return 1

    print(f"HAZIR -- {len(kapilar)} kapinin hepsi yesil.")
    print("=" * 78)
    plani_yazdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
