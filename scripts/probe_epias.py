"""EPIAS baglantisini dogrular ve sema kesfi yapar.

AMAC: Yarisma verisi gelmeden once (a) API'nin calistigini, (b) EPIAS'in
il/ilce adlarini NASIL YAZDIGINI, (c) plansiz kesinti kaydinin semasini
ogrenmek.

(b) ozellikle degerli: Turkce join'in nerede kirilacagini 21 Agustos'ta
degil bugun ogrenmis oluruz.

Calistirma::

    # once .env dosyasini olustur (.env.example sablonuna bak)
    python scripts/probe_epias.py
    python scripts/probe_epias.py --period 2025-06 --save

Sifre HICBIR ciktida gorunmez.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup.epias import (  # noqa: E402
    EpiasAuthError,
    EpiasClient,
    EpiasRequestError,
    load_env_file,
)
from gridup.turkish import codepoints, has_combining_dot, join_key  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "data" / "external" / "epias"


def banner(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}")


def show(frame: pd.DataFrame, *, name: str, rows: int = 8) -> None:
    print(f"  {len(frame):,} satir x {frame.shape[1]} kolon")
    print(f"  Kolonlar: {list(frame.columns)}")
    if not frame.empty:
        print(f"\n  Ilk {min(rows, len(frame))} satir:")
        with pd.option_context("display.max_columns", 30, "display.width", 200):
            print(frame.head(rows).to_string(index=False))
    print(f"  [{name}]")


def inspect_turkish_columns(frame: pd.DataFrame, label: str) -> None:
    """Metin kolonlarindaki Turkce yazim bicimini raporlar."""
    text_columns = [
        column
        for column in frame.columns
        if frame[column].dtype == object or str(frame[column].dtype) == "str"
    ]
    if not text_columns:
        return

    print(f"\n  --- {label}: Turkce yazim kontrolu ---")
    for column in text_columns[:6]:
        values = frame[column].dropna().astype(str)
        if values.empty:
            continue
        sample = values.iloc[0]
        dotted = any(has_combining_dot(value) for value in values.head(500))
        flag = "  ! BIRLESIK NOKTA VAR" if dotted else ""
        print(f"    {column:<28} ornek={sample!r:<28} join_key={join_key(sample)!r}{flag}")
        if dotted:
            print(f"      kod noktalari: {codepoints(sample)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--period", default="2025-06", help="Kesinti donemi (YYYY-AA)")
    parser.add_argument("--save", action="store_true", help="Ciktilari parquet olarak kaydet")
    args = parser.parse_args()

    loaded = load_env_file(ROOT / ".env")
    if not loaded:
        print(".env dosyasi bulunamadi.")
        print(f"  {ROOT / '.env.example'} dosyasini '.env' adiyla kopyala ve doldur.")
        return 1

    try:
        client = EpiasClient.from_env()
    except EpiasAuthError as error:
        print(f"HATA: {error}")
        return 1

    banner("1. KIMLIK DOGRULAMA")
    try:
        ticket = client.tgt
    except EpiasAuthError as error:
        print(f"BASARISIZ:\n{error}")
        return 1
    print(f"  TGT alindi ({len(ticket)} karakter). Kullanici: {client.username}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    collected: dict[str, pd.DataFrame] = {}

    banner("2. DAGITIM SIRKETLERI")
    try:
        companies = client.distribution_companies()
        show(companies, name="dagitim_sirketleri")
        inspect_turkish_columns(companies, "dagitim_sirketleri")
        collected["dagitim_sirketleri"] = companies

        # GDZ ve ADM'yi bul
        text = companies.astype(str).agg(" ".join, axis=1)
        for needle in ("GDZ", "ADM"):
            hits = companies[text.str.upper().str.contains(needle, na=False)]
            if not hits.empty:
                print(f"\n  {needle} bulundu:")
                print(hits.to_string(index=False))
    except EpiasRequestError as error:
        print(f"  BASARISIZ: {error}")

    banner("3. DAGITIM BOLGELERI (RESMI IL YAZIMI)")
    try:
        regions = client.distribution_regions()
        show(regions, name="dagitim_bolgeleri", rows=12)
        inspect_turkish_columns(regions, "dagitim_bolgeleri")
        collected["dagitim_bolgeleri"] = regions
    except EpiasRequestError as error:
        print(f"  BASARISIZ: {error}")

    banner(f"4. PLANSIZ KESINTI -- {args.period}")
    try:
        outages = client.unplanned_outages(period=args.period)
        show(outages, name="plansiz_kesinti", rows=6)
        inspect_turkish_columns(outages, "plansiz_kesinti")
        collected["plansiz_kesinti"] = outages
    except EpiasRequestError as error:
        print(f"  BASARISIZ: {error}")
        print("\n  NOT: Endpoint adi veya govde bicimi degismis olabilir.")
        print("  Yardim Masasi > Web Servisler dokumantasyonuna bak.")

    if args.save and collected:
        banner("5. KAYDEDILIYOR")
        for name, frame in collected.items():
            path = OUTPUT_DIR / f"{name}.parquet"
            frame.to_parquet(path, index=False)
            print(f"  {path}  ({len(frame):,} satir)")

    banner("OZET")
    print(f"  Basarili sorgu: {len(collected)}/3")
    if "dagitim_bolgeleri" in collected or "plansiz_kesinti" in collected:
        print("  -> Il/ilce yazimini yukaridaki 'Turkce yazim kontrolu' bolumunden oku.")
        print("     Yarisma verisi geldiginde join_key() ile bunlara baglayacagiz.")
    return 0 if collected else 1


if __name__ == "__main__":
    raise SystemExit(main())
