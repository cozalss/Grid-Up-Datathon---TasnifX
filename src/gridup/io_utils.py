"""Dosya okuma: kodlama, ayirici ve ondalik isareti otomatik tespiti.

NEDEN BU MODUL VAR
------------------
Turk kurumlarindan cikan veri dosyalari (EPDK, TUIK, e-Devlet, SAP/Excel
ihraclari) su ozelliklere sahiptir:

  * Kodlama ``cp1254`` veya ``ISO-8859-9`` -- UTF-8 DEGIL
  * Alan ayirici ``;`` -- cunku ``,`` ondalik isaretidir
  * Ondalik isareti ``,`` ve binlik ayirici ``.``  (``1.234.567,89`` TEK sayidir)
  * Excel'den cikmissa basta BOM (``utf-8-sig``)

``pd.read_csv(path)`` bunlarin hicbirini varsaymaz ve iki sekilde basarisiz olur:

  1. Gurultulu: ``UnicodeDecodeError``  -- iyi, hemen gorursun
  2. SESSIZ: ``1.234.567,89`` uc ayri alana bolunur, ya da cp1254 dosyasi
     latin-1 gibi okunur ve birkac karakter yanlis cikar -- kotusu budur

Ikinci durumda pipeline sonuna kadar calisir ve skorun neden dusuk oldugunu
anlamazsin. Bu modul tespiti ACIKCA yapar ve ne buldugunu raporlar.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .turkish import normalize_columns

__all__ = [
    "DialectGuess",
    "sniff_dialect",
    "read_table",
    "read_any",
    "to_parquet_cache",
]

# Sira ONEMLI: utf-8 cp1254 girdide GURULTULU basarisiz olur (iyi).
# cp1254 ise neredeyse her bayt dizisini kabul eder ve sessizce mojibake uretir.
# Bu yuzden once utf-8 denenir. utf-8-sig once gelir cunku Excel BOM ekler.
_ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp1254", "iso-8859-9", "latin-1")

_DELIMITER_CANDIDATES = (";", ",", "\t", "|")

# Ondalik isaretini bu kadar satira bakarak karar veriyoruz. Buyuk dosyada
# tamamini okumak anlamsiz; ilk birkac KB kararı vermeye yeter.
_SNIFF_BYTES = 64 * 1024


@dataclass(frozen=True)
class DialectGuess:
    """Bir metin dosyasi hakkinda tespit edilen bicim bilgisi."""

    encoding: str
    delimiter: str
    decimal: str
    thousands: str | None
    sample_line: str

    def as_read_csv_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "encoding": self.encoding,
            "sep": self.delimiter,
            "decimal": self.decimal,
        }
        if self.thousands:
            kwargs["thousands"] = self.thousands
        return kwargs

    def __str__(self) -> str:
        delim = {"\t": "\\t"}.get(self.delimiter, self.delimiter)
        return (
            f"kodlama={self.encoding} ayirici='{delim}' "
            f"ondalik='{self.decimal}' binlik={self.thousands or 'yok'}"
        )


def _decode_head(path: Path) -> tuple[str, str]:
    """Dosyanin basini cozer. ``(metin, kodlama)`` dondurur.

    Hicbir aday kodlama calismazsa acik hata firlatir -- sessizce latin-1'e
    dusmek mojibake'i gizler.
    """
    raw = path.open("rb").read(_SNIFF_BYTES)

    for encoding in _ENCODING_CANDIDATES:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"{path}: aday kodlamalarin hicbiri temiz cozmedi {_ENCODING_CANDIDATES}. "
        "Dosya ikili (binary) olabilir veya bilinmeyen bir kodlama kullaniyor."
    )


def _guess_delimiter(head: str) -> str:
    """Ayiriciyi tespit eder.

    ``csv.Sniffer`` once denenir; Turkce dosyalarda ``,`` ondalik isareti oldugu
    icin sik yanilir, o yuzden satir basina tutarlilik kontroluyle dogrulanir.
    """
    lines = [line for line in head.splitlines() if line.strip()][:20]
    if not lines:
        return ","

    best_delimiter, best_score = ",", -1.0
    for candidate in _DELIMITER_CANDIDATES:
        counts = [line.count(candidate) for line in lines]
        if not counts or max(counts) == 0:
            continue
        # Iyi bir ayirici her satirda AYNI sayida gorunur.
        consistent = sum(1 for count in counts if count == counts[0])
        score = consistent / len(counts) * counts[0]
        if score > best_score:
            best_delimiter, best_score = candidate, score

    try:
        sniffed = csv.Sniffer().sniff(head[:8192], delimiters="".join(_DELIMITER_CANDIDATES))
        if best_score <= 0:
            return sniffed.delimiter
    except csv.Error:
        pass

    return best_delimiter


def _guess_decimal(head: str, delimiter: str) -> tuple[str, str | None]:
    """Ondalik ve binlik isaretlerini tespit eder.

    Kural: ayirici ``;`` ise ondalik neredeyse kesin ``,``dir (Turkce Excel).
    Ayirici ``,`` ise ondalik ``.`` olmak zorundadir -- aksi halde dosya
    ayristirilamaz.
    """
    if delimiter == ",":
        return ".", None

    body = head[head.find("\n") + 1 :]  # basligi atla
    tokens = body.split(delimiter)
    comma_decimal_hits = sum(1 for token in tokens if _looks_like_tr_number(token))
    dot_decimal_hits = sum(1 for token in tokens if _looks_like_en_number(token))

    # ';' ayirici gorulduyse ondalik VARSAYILAN olarak ',' kabul edilir.
    # Neden: ';' ayirici zaten ',' ondalik oldugu ICIN secilir (Turkce Excel).
    # Heuristik sayim yanilabilir -- '212.345' gibi nokta-grupli bir trafo ID'si
    # "Ingilizce ondalik" sayilir ve gercek ondalikli alanlar ornekte seyrekse
    # sayaci yanlis tarafa kaydirir. Sonuc sessizdir: decimal='.' ile okunan
    # '1.234,56' hucreleri sayiya cevrilemez, kolon object dtype'da kalir ve
    # sayisal bir feature kategorik muamelesi gorur.
    # Bu yuzden ','den ancak GUCLU kanit varsa vazgeciyoruz.
    if dot_decimal_hits > comma_decimal_hits * 2 and dot_decimal_hits >= 3:
        return ".", None

    has_thousands = any("." in token and "," in token for token in tokens)
    return ",", "." if has_thousands else None


def _looks_like_tr_number(token: str) -> bool:
    """``12,5`` veya ``1.234,56`` bicimi mi?"""
    stripped = token.strip().strip('"')
    if "," not in stripped:
        return False
    left, _, right = stripped.rpartition(",")
    return right.isdigit() and left.replace(".", "").replace("-", "").isdigit()


def _looks_like_en_number(token: str) -> bool:
    """``12.5`` bicimi mi?"""
    stripped = token.strip().strip('"')
    if "." not in stripped or "," in stripped:
        return False
    left, _, right = stripped.rpartition(".")
    return right.isdigit() and left.replace("-", "").isdigit()


def sniff_dialect(path: str | Path) -> DialectGuess:
    """Bir metin dosyasinin kodlama/ayirici/ondalik bicimini tespit eder.

    Okumadan ONCE calistir ve ciktisini kaydet -- veri setinin gercek bicimi,
    juri sunumunda "veriyi anladik" demenin ucuz bir kanitidir.
    """
    path = Path(path)
    head, encoding = _decode_head(path)
    delimiter = _guess_delimiter(head)
    decimal, thousands = _guess_decimal(head, delimiter)
    first_line = next((line for line in head.splitlines() if line.strip()), "")

    return DialectGuess(
        encoding=encoding,
        delimiter=delimiter,
        decimal=decimal,
        thousands=thousands,
        sample_line=first_line[:300],
    )


def read_table(
    path: str | Path,
    *,
    normalize_column_names: bool = True,
    verbose: bool = True,
    **overrides: Any,
) -> pd.DataFrame:
    """Bicimi otomatik tespit ederek CSV/TSV okur.

    Args:
        path: Dosya yolu.
        normalize_column_names: Kolon adlarini ASCII-guvenli snake_case'e cevirir.
            Ham adlar ``frame.attrs["original_columns"]`` icinde saklanir.
        verbose: Tespit edilen bicimi yazdirir.
        **overrides: ``pd.read_csv``a dogrudan gecer; tespiti EZER.

    Returns:
        Yeni DataFrame.
    """
    path = Path(path)
    guess = sniff_dialect(path)
    kwargs = {**guess.as_read_csv_kwargs(), **overrides}

    if verbose:
        print(f"[read_table] {path.name}: {guess}")

    frame = pd.read_csv(path, **kwargs)

    if normalize_column_names:
        mapping = normalize_columns(frame.columns)
        frame = frame.rename(columns=mapping)
        frame.attrs["original_columns"] = mapping

    frame.attrs["source_path"] = str(path)
    frame.attrs["dialect"] = str(guess)
    return frame


def read_any(path: str | Path, **kwargs: Any) -> pd.DataFrame:
    """Uzantiya gore CSV / Parquet / Excel / JSON okur.

    Parquet varsa onu tercih et: 5-20 kat hizli ve dtype'lari korur; boylece
    her notebook calistirmasinda kodlama tespitini tekrarlamazsin.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".feather", ".ftr"}:
        return pd.read_feather(path, **kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)
    if suffix in {".csv", ".tsv", ".txt", ".dat", ""}:
        return read_table(path, **kwargs)

    raise ValueError(f"{path}: desteklenmeyen uzanti '{suffix}'")


def to_parquet_cache(frame: pd.DataFrame, cache_path: str | Path) -> Path:
    """DataFrame'i parquet olarak onbellege yazar ve yolu dondurur.

    Kalip: ham CSV'yi bir kez oku, parquet'e yaz, sonraki her calistirmada
    parquet'ten oku. 12 gunluk bir yarismada bu, yuzlerce dakika kazandirir.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return cache_path
