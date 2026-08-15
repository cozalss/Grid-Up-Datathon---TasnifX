"""KTB yillik konaklama bultenlerinden il-ilce geceleme tablosu uretir.

NEDEN BU BETIK
--------------
docs/10 bolum 5: Mugla'da yaz nufusu yerlesigin 2-5 kati (ADM'nin resmi
aciklamasi). Ilce bazli geceleme sayisi, "yaz nufus carpani" feature'inin
tek resmi kaynagidir -- kesinti/yuk tahmini icin ilcenin GERCEK insan
yukunu verir, ADNKS kislik nufusu degil.

KAYNAK
------
yigm.ktb.gov.tr > Konaklama Istatistikleri > Yillik Bultenler. Yillik bulten
xlsx'lerinin "Il Ilce" sayfasi: isletme + basit belgeli tesislere gelis ve
geceleme, ilce kirilimida (yabanci/yerli/toplam). 2023-2025 yillari xlsx
olarak yayimlanmis durumda (daha eski yillar rar/pdf -- kapsam disi).

Ham dosyalar data/external/ham/ altina indirilir ve SAKLANIR: KTB sunucusu
hizli ardisik istekte baglantiyi kesiyor (olculdu: ikinci istek
RemoteDisconnected) -- betik dosya zaten varsa indirmez, cevrimdisi calisir.

Kullanim::

    python scripts/fetch_turizm.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.turkish import join_key, strip_qualifier  # noqa: E402

#: Yil -> yillik bulten Eklenti adresi (yigm.ktb.gov.tr, dogrulandi 2026-08).
BULTENLER: dict[int, str] = {
    2023: "https://yigm.ktb.gov.tr/Eklenti/122189,konaklama-yillik-bulten-2023-v2xlsx.xlsx?0",
    2024: (
        "https://yigm.ktb.gov.tr/Eklenti/131446,"
        "konaklama-yillik-bulten-2024---yillik-bultenxlsx.xlsx?0"
    ),
    2025: "https://yigm.ktb.gov.tr/Eklenti/145670,konaklama-yillik-b-lten-2025-ver2xlsx.xlsx?0",
}

#: GDZ + ADM hizmet bolgesi illeri (join_key bicimi).
HEDEF_ILLER = frozenset({"izmir", "manisa", "aydin", "denizli", "mugla"})

SAYFA_ADI = "İl İlçe"
HAM_DIZIN = Path("data/external/ham")
CIKTI_YOLU = Path("data/external/turizm_geceleme.parquet")

#: KTB sunucusu ardisik hizli istekte baglantiyi kesiyor; istekler arasi
#: uzun bekleme sart (olculdu: 6 sn guvenli).
REQUEST_PAUSE_S = 6.0
TIMEOUT_S = 120
RETRIES = 4


def indir(yil: int, url: str) -> Path:
    """Bulteni ham dizine indirir; dosya zaten varsa dokunmaz."""
    hedef = HAM_DIZIN / f"ktb_konaklama_yillik_{yil}.xlsx"
    if hedef.exists():
        print(f"  {yil}: ham dosya mevcut, indirilmedi ({hedef})")
        return hedef

    son_hata: Exception | None = None
    for deneme in range(1, RETRIES + 1):
        try:
            yanit = requests.get(
                url, timeout=TIMEOUT_S,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            yanit.raise_for_status()
            break
        except requests.RequestException as hata:
            son_hata = hata
            if deneme < RETRIES:
                time.sleep(REQUEST_PAUSE_S + 2**deneme)
    else:
        raise RuntimeError(f"{yil} bulteni {RETRIES} denemede inmedi. Son hata: {son_hata}")

    if len(yanit.content) < 10_000:
        raise RuntimeError(f"{yil} bulteni supheli kucuk ({len(yanit.content)} bayt) -- "
                           "muhtemelen hata sayfasi indi.")
    HAM_DIZIN.mkdir(parents=True, exist_ok=True)
    hedef.write_bytes(yanit.content)
    print(f"  {yil}: indirildi ({len(yanit.content):,} bayt)")
    time.sleep(REQUEST_PAUSE_S)
    return hedef


def _kolon_bul(baslik_satiri: pd.Series, aranan: str) -> int:
    """Baslik satirinda join_key esiyle kolon arar; yoksa ValueError."""
    for indeks, deger in baslik_satiri.items():
        if aranan in join_key(str(deger)):
            return int(indeks)
    raise ValueError(f"Baslikta '{aranan}' bulunamadi: {baslik_satiri.tolist()}")


def il_ilce_tablosu(yol: Path, yil: int) -> pd.DataFrame:
    """"Il Ilce" sayfasini ortak semaya cevirir.

    Sayfa yapisi (2023-2025'te ayni, her yil icin YENIDEN dogrulanir):
      satir 0: belge turu basligi, satir 1: grup basliklari (ILLER, ILCELER,
      TESISE GELIS SAYISI, GECELEME, ...), satir 2: YABANCI/YERLI/TOPLAM,
      satir 3+: veri. Il adi yalnizca ilk ilce satirinda yazili (birlesik
      hucre) -- ffill ile tasinir.
    """
    ham = pd.read_excel(yol, sheet_name=SAYFA_ADI, header=None)

    gelis_kolon = _kolon_bul(ham.iloc[1], "tesise gelis")
    geceleme_kolon = _kolon_bul(ham.iloc[1], "geceleme")
    # Grup basi YABANCI'dir; TOPLAM iki kolon sagdadir (satir 2 dogrular).
    for grup_bas in (gelis_kolon, geceleme_kolon):
        if "toplam" not in join_key(str(ham.iloc[2, grup_bas + 2])):
            raise ValueError(f"{yil}: kolon {grup_bas + 2} TOPLAM degil -- yapi degismis.")

    veri = ham.iloc[3:, :].copy()
    veri[0] = veri[0].ffill()

    tablo = pd.DataFrame(
        {
            "yil": yil,
            "il": veri[0].astype(str).str.strip(),
            "ilce": veri[1].astype(str).str.strip(),
            "tesise_gelis": pd.to_numeric(veri[gelis_kolon + 2], errors="coerce"),
            "geceleme": pd.to_numeric(veri[geceleme_kolon + 2], errors="coerce"),
        }
    )
    tablo = tablo[tablo["ilce"].notna() & (tablo["ilce"] != "nan")]
    tablo["il_key"] = tablo["il"].map(join_key)
    tablo["ilce_key"] = tablo["ilce"].map(lambda ad: join_key(strip_qualifier(ad)))
    # Il ara toplam satirlari ("Toplam") ilce DEGILDIR; join'e sizarsa ayni
    # anahtara birden cok il duser (olculdu: 5 il x 3 yil = 15 satir).
    tablo = tablo[~tablo["ilce_key"].str.contains("toplam")]

    hedef = tablo[tablo["il_key"].isin(HEDEF_ILLER)].copy()
    if hedef.empty:
        raise ValueError(f"{yil}: hedef illerden hic satir cikmadi -- il kolonu bozuk olabilir.")
    return hedef.dropna(subset=["geceleme"]).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(CIKTI_YOLU), help="Cikti parquet yolu")
    args = parser.parse_args()

    parcalar: list[pd.DataFrame] = []
    for yil, url in sorted(BULTENLER.items()):
        yol = indir(yil, url)
        tablo = il_ilce_tablosu(yol, yil)
        print(f"  {yil}: {len(tablo)} ilce satiri "
              f"(geceleme toplami {tablo['geceleme'].sum():,.0f})")
        parcalar.append(tablo)

    birlesik = pd.concat(parcalar, ignore_index=True)
    kolonlar = ["yil", "il", "ilce", "il_key", "ilce_key", "geceleme", "tesise_gelis"]
    birlesik = birlesik[kolonlar].sort_values(["yil", "il", "ilce"]).reset_index(drop=True)

    cikti = Path(args.out)
    cikti.parent.mkdir(parents=True, exist_ok=True)
    birlesik.to_parquet(cikti, index=False)

    print(f"Yazildi: {cikti}")
    print(f"  {len(birlesik)} satir, yillar {sorted(birlesik['yil'].unique())}")
    print("  Kaynak: KTB Yatirim ve Isletmeler Gn.Md. yillik konaklama bultenleri.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
