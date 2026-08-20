"""Veri manifestindeki OLCULEN alanlari yeniden hesaplar (sha256, min_rows).

NEDEN BU BETIK
--------------
``data/sources.yml`` iki tur alan tasir:

  * KARAR alanlari -- lisans, yeniden dagitim, dogrulama dayanagi, kaynak
    aciklamasi. Bunlar insan karariyla yazilir ve bu betik onlara DOKUNMAZ.
  * OLCUM alanlari -- ``sha256`` ve ``schema.min_rows``. Bunlar dosyanin
    kendisinden hesaplanir.

Veri her tazelendiginde olcum alanlari degisir. Elle guncellemek iki yonde de
hataya acik: ya unutulur (``verify_sources`` "SHA256 uyusmuyor" der ve bir
sure sonra gurultu sayilip gormezden gelinir), ya da yanlis kopyalanir.

NE YAPMAZ
---------
Yeni artefakt EKLEMEZ. Manifestte olmayan bir dosya icin satir uydurmak,
tedarik zinciri kaydini anlamsizlastirirdi -- yeni bir kaynagin lisansi ve
dagitim hakki insan tarafindan incelenmelidir. Manifestte olmayan dosyalar
uyari olarak listelenir.

``required_columns`` da DEGISTIRMEZ: bir kolonun zorunlu olmasi bir
sozlesmedir, gozlem degil. Kolon eksikse betik bunu HATA olarak bildirir.

KULLANIM
--------
::

    python scripts/manifest_tazele.py --dry-run   # once neyin degisecegini gor
    python scripts/manifest_tazele.py

Cikis kodu: 0 = tamam, 1 = en az bir dosya eksik ya da sozlesme ihlali var.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "sources.yml"

#: ``min_rows`` gercek satir sayisinin bu orani kadar yazilir. Tam sayiyi
#: yazmak kapiyi asiri kirilgan yapar: kaynagin bir sonraki tazelemede bir
#: satir eksik dondurmesi (or. gec gelen bir olay kaydi) kapiyi kirmizi
#: yapardi. %98, "tablo ciddi olcude kuculdu"yu yakalar, dogal oynamayi degil.
MIN_ROWS_PAYI = 0.98

#: ``min_rows`` YALNIZCA YUKARI guncellenir. Asagi cekmek, kapiyi kapatmak
#: degil KALDIRMAKTIR: tablo bir gun yariya inse betik esigi de yariya
#: indirir ve ``verify_sources`` hicbir sey soylemez. Tablo kuculduyse bunu
#: bir insan degerlendirmelidir -- betik yalnizca bildirir.


def dosya_hash(yol: Path) -> str:
    """Dosyanin SHA256'si. Buyuk parquet'ler icin parcali okunur."""
    ozet = hashlib.sha256()
    with yol.open("rb") as akis:
        for parca in iter(lambda: akis.read(1024 * 1024), b""):
            ozet.update(parca)
    return ozet.hexdigest()


def satir_sayisi(yol: Path) -> int | None:
    """Parquet/CSV satir sayisi; okunamayan bicimde ``None``.

    Parquet icin dosya METADATASI okunur, veri degil: 230 bin satirlik bir
    tabloyu satir saymak icin belleğe almak gereksiz. (``read_parquet`` ile
    ``columns=[]`` gecmek de calismaz -- pandas 3.0'da bos kolon listesi
    SIFIR SATIRLIK bir frame dondurur, yani her tablo 0 gorunur.)
    """
    try:
        if yol.suffix == ".parquet":
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(yol).metadata.num_rows)
        if yol.suffix == ".csv":
            return int(len(pd.read_csv(yol, sep=None, engine="python")))
    except (OSError, ValueError, ImportError):
        return None
    return None


def zorunlu_kolonlar_var_mi(yol: Path, kolonlar: list[str]) -> list[str]:
    """Eksik zorunlu kolonlari doner. Okunamayan dosya icin bos liste."""
    if yol.suffix != ".parquet" or not kolonlar:
        return []
    try:
        import pyarrow.parquet as pq

        mevcut = set(pq.ParquetFile(yol).schema_arrow.names)
    except (OSError, ValueError, ImportError):
        return []
    return [k for k in kolonlar if k not in mevcut]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Yazmadan degisiklikleri listele")
    args = ap.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    artefaktlar = manifest["artifacts"]

    print(f"MANIFEST TAZELEME  ({len(artefaktlar)} artefakt)")
    print("=" * 74)

    degisen = 0
    hata = 0
    for artefakt in artefaktlar:
        yol = ROOT / artefakt["path"]
        ad = artefakt["path"].split("/")[-1]
        if not yol.is_file():
            print(f"  [HATA ] {ad}: dosya yok ({yol})")
            hata += 1
            continue

        notlar: list[str] = []

        yeni_hash = dosya_hash(yol)
        if yeni_hash != artefakt.get("sha256"):
            notlar.append(f"sha256 {str(artefakt.get('sha256'))[:12]} -> {yeni_hash[:12]}")
            artefakt["sha256"] = yeni_hash

        sema = artefakt.get("schema", {})
        satir = satir_sayisi(yol)
        if satir is not None and "min_rows" in sema:
            hedef = int(satir * MIN_ROWS_PAYI)
            if hedef > sema["min_rows"]:
                notlar.append(f"min_rows {sema['min_rows']:,} -> {hedef:,} (gercek {satir:,})")
                sema["min_rows"] = hedef
            elif satir < sema["min_rows"]:
                # Esigi ASAGI CEKMIYORUZ (bkz. MIN_ROWS_PAYI notu): tablonun
                # kucumesi tam olarak kapinin yakalamasi gereken seydir.
                print(
                    f"  [HATA ] {ad}: {satir:,} satir, manifest en az "
                    f"{sema['min_rows']:,} bekliyor. Tablo KUCULMUS. "
                    "Esik otomatik dusurulmez -- once kuculmenin nedenini bul."
                )
                hata += 1
                continue

        eksik = zorunlu_kolonlar_var_mi(yol, sema.get("required_columns", []))
        if eksik:
            # Sozlesme ihlali OTOMATIK duzeltilmez: zorunlu kolonu manifestten
            # silmek, kapiyi kapatmak degil kaldirmaktir.
            print(f"  [HATA ] {ad}: zorunlu kolonlar EKSIK: {eksik}")
            hata += 1
            continue

        if notlar:
            degisen += 1
            print(f"  [YENI ] {ad}")
            for not_ in notlar:
                print(f"           {not_}")
        else:
            print(f"  [AYNI ] {ad}")

    print("=" * 74)
    print(f"{degisen} artefakt guncellendi · {hata} hata")

    if hata:
        print("\nHatalar duzeltilmeden manifest YAZILMADI.")
        return 1
    if args.dry_run:
        print("\n--dry-run: dosya yazilmadi.")
        return 0
    if degisen:
        MANIFEST.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nYazildi: {MANIFEST}")
    else:
        print("\nDegisiklik yok.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
