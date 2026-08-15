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
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .turkish import normalize_columns

__all__ = [
    "DialectGuess",
    "sniff_dialect",
    "sniff_dialect_shared",
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
    """Bir metin dosyasi hakkinda tespit edilen bicim bilgisi.

    ``warnings`` alani, tespitin BELIRSIZ kaldigi durumlari tasir. Bos degilse
    train ve test icin farkli karar verilmis olabilir -- sessiz birakilmaz,
    ``sniff_dialect`` bunlari ekrana da basar.
    """

    encoding: str
    delimiter: str
    decimal: str
    thousands: str | None
    sample_line: str
    warnings: tuple[str, ...] = ()

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
    # ``with`` ZORUNLU. Onceki surum ``path.open("rb").read(...)`` yaziyordu
    # ve dosya tanitiCisi acik kaliyordu (ResourceWarning). Bu, Windows'ta
    # sadece kaynak sizintisi degildir: acik tanitici dosyayi KILITLER ve
    # ayni yola sonradan yazmak PermissionError verir. Yarisma gunu bir
    # dosyayi okuyup ustune yazmak (temizlenmis surumu kaydetmek) son derece
    # olagan bir istir -- ve tam orada patlardi.
    with path.open("rb") as handle:
        raw = handle.read(_SNIFF_BYTES)

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


def _alan_tokenleri(head: str, delimiter: str) -> list[str]:
    """Basligi atlar ve govdeyi SATIR SATIR alanlara boler.

    NEDEN SATIR SATIR: onceki surum ``body.split(delimiter)`` yaziyordu, yani
    satir sonlarini hic gormuyordu. Bir satirin son alani ile sonraki satirin
    ilk alani tek token'a yapisiyordu (``'9,00\\n212.001'``); o token hem ``.``
    hem ``,`` icerdigi icin binlik tespiti dosyadaki KOLON SIRASINA baglaniyordu.
    OLCULDU: ayni sekilde yazilmis train/test ciftinde binlik train'de '.',
    test'te 'yok' cikiyor ve ayni kolon train'de int64, test'te str oluyordu.
    """
    lines = [line for line in head.splitlines() if line.strip()][1:]
    return [alan for line in lines for alan in line.split(delimiter)]


def _guess_decimal(head: str, delimiter: str) -> tuple[str, str | None, tuple[str, ...]]:
    """Ondalik ve binlik isaretlerini tespit eder.

    Kural: ayirici ``;`` ise ondalik neredeyse kesin ``,``dir (Turkce Excel).
    Ayirici ``,`` ise ondalik ``.`` olmak zorundadir -- aksi halde dosya
    ayristirilamaz.

    Returns:
        ``(ondalik, binlik, uyarilar)``.
    """
    if delimiter == ",":
        return ".", None, ()

    tokens = _alan_tokenleri(head, delimiter)
    virgul_ondalik = sum(1 for token in tokens if _looks_like_tr_number(token))
    nokta_ondalik = sum(1 for token in tokens if _looks_like_en_number(token))
    cok_grup = any(_cok_gruplu(token) for token in tokens)

    # BINLIK ICIN **POZITIF KANIT** SART.
    #
    # Onceki surum ';' ayirici gorunce ondaligi varsayilan olarak ',' yapiyor,
    # ve TAM 3 haneli her kesri ("9.801") binlik gruplama kaniti sayiyordu.
    # Bu, en yaygin hedef bicimini SESSIZCE 1000 kat buyutuyordu:
    #
    #   OLCULDU -- 'tuketim_mwh' kolonu, ';' ayirici, cp1254:
    #     girdi  9.801        -> okunan np.int64(9801)   dtype=int64
    #     toplam 143.055      -> okunan 143055
    #   Uyari da cikmiyordu (uyari kosulu tam da bu durumda susuyordu).
    #
    # Hedef MWh oldugunda uc ondalik en olagan haldir; yani bu, gercek
    # yarisma verisinde neredeyse kesin tetiklenecek bir bozulmaydi.
    #
    # Nokta ancak SU DURUMDA binliktir:
    #   * dosyada bir yerde virgul-ondalik var ("1,5" veya "1.234,56"), YA DA
    #   * cok gruplu bir sayi var ("1.234.567") -- tek basina belirsiz degil.
    # Tek basina "9.801" KANIT DEGILDIR; nokta ondalik kabul edilir.
    binlik_kaniti = virgul_ondalik > 0 or cok_grup
    if not binlik_kaniti:
        return ".", None, ()

    grupli = any(_binlik_kaniti(token) for token in tokens)
    binlik = "." if grupli else None
    return ",", binlik, _belirsizlik_uyarisi(grupli, nokta_ondalik, ",")


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


def _cok_gruplu(token: str) -> bool:
    """``1.234.567`` gibi BIRDEN FAZLA nokta grubu var mi?

    Tek grup (``9.801``) ondalik da olabilir binlik de; belirsizdir. Iki veya
    daha fazla grup ise ondalik olarak okunamaz -- kesin binlik kanitidir.
    """
    stripped = token.strip().strip('"')
    govde = stripped.rpartition(",")[0] if "," in stripped else stripped
    return govde.lstrip("-").count(".") >= 2 and _nokta_grupli(govde)


def _kesin_en_ondalik(token: str) -> bool:
    """Ingilizce ondalik oldugu BELIRSIZ OLMAYAN bir bicim mi?

    ``212.345`` Ingilizce ondalik gibi gorunur ama Turkce dosyada bu neredeyse
    her zaman nokta-grupli bir trafo/abone kodudur: kesir kismi TAM 3 hane
    oldugu icin binlik gruplamadan ayirt EDILEMEZ. Bu yuzden 3 haneli kesirler
    kanit sayilmaz. OLCULDU: yalnizca ``212.xxx`` kodlari iceren test dosyasi
    bu kural olmadan 60 "Ingilizce ondalik" kaniti uretiyor ve ondalik isaretini
    ','den '.'ya cevirip ayni veriyi okuyan train'den ayrisiyordu.
    """
    if not _looks_like_en_number(token):
        return False
    kesir = token.strip().strip('"').rpartition(".")[2]
    return len(kesir) != 3


def _nokta_grupli(tamsayi: str) -> bool:
    """``1.234`` / ``212.345.678`` gibi TAM 3 haneli gruplara bolunmus mu?"""
    parcalar = tamsayi.lstrip("-").split(".")
    if len(parcalar) < 2 or not parcalar[0].isdigit() or len(parcalar[0]) > 3:
        return False
    return all(parca.isdigit() and len(parca) == 3 for parca in parcalar[1:])


def _binlik_kaniti(token: str) -> bool:
    """Token, ``.`` ile gruplanmis bir tamsayi kismi tasiyor mu?

    Onceki kural ``"." in token and "," in token`` idi ve YALNIZCA ondalikli
    grupli sayilari (``1.234,56``) goruyordu. Ondaliksiz ``212.345`` bir
    dosyada bulunup digerinde bulunmadiginda binlik karari degisiyordu.
    Burada once kesir kismi atilir, geriye kalan tamsayi kismina bakilir --
    boylece ``1.234,56`` ile ``212.345`` AYNI karari uretir.
    """
    stripped = token.strip().strip('"')
    govde = stripped.rpartition(",")[0] if "," in stripped else stripped
    return _nokta_grupli(govde)


def _belirsizlik_uyarisi(grupli: bool, dot_hits: int, secilen: str) -> tuple[str, ...]:
    """Nokta hem binlik hem ondalik gibi kullanilmissa uyari uretir.

    Iki kanit ayni dosyada bulunuyorsa hangi kolonun hangi kurala tabi oldugunu
    dosyaya bakarak bilemeyiz. Sessizce birini secmek yerine soyleriz: train ve
    test farkli kanit dagilimina sahipse ayni kolon iki dosyada iki farkli
    dtype'a duser ve model egitimi son adimda patlar.
    """
    if not (grupli and dot_hits):
        return ()
    mesaj = (
        f"'.' isareti hem binlik gruplama ('1.234') hem ondalik ('12.5') gibi "
        f"kullanilmis ({dot_hits} belirsiz-olmayan ondalik token). Ondalik "
        f"'{secilen}' secildi -- train ve test icin AYNI karari aldigini dogrula, "
        "aksi halde ayni kolon iki dosyada farkli dtype olur."
    )
    return (mesaj,)


def sniff_dialect(path: str | Path) -> DialectGuess:
    """Bir metin dosyasinin kodlama/ayirici/ondalik bicimini tespit eder.

    Okumadan ONCE calistir ve ciktisini kaydet -- veri setinin gercek bicimi,
    juri sunumunda "veriyi anladik" demenin ucuz bir kanitidir.
    """
    path = Path(path)
    head, encoding = _decode_head(path)
    delimiter = _guess_delimiter(head)
    decimal, thousands, uyarilar = _guess_decimal(head, delimiter)
    first_line = next((line for line in head.splitlines() if line.strip()), "")

    # Belirsizlik SESSIZ kalmaz: bu uyari, train/test'in farkli ayristirilma
    # riskinin tek gorunur isaretidir.
    for uyari in uyarilar:
        print(f"[sniff_dialect] {path.name}: UYARI: {uyari}")

    return DialectGuess(
        encoding=encoding,
        delimiter=delimiter,
        decimal=decimal,
        thousands=thousands,
        sample_line=first_line[:300],
        warnings=uyarilar,
    )


def sniff_dialect_shared(paths: Sequence[str | Path]) -> DialectGuess:
    """Birden fazla dosya icin TEK bir bicim karari uretir.

    NEDEN GEREKLI (olculdu)
    -----------------------
    ``sniff_dialect`` her dosyaya BAGIMSIZ bakar. Train ve test ayni kolonu
    tasiyor ama kanit dagilimi farkliysa iki dosya iki farkli karara varir:

        train.csv  ('enlem' 4 haneli kesir + hedefte '1,5')
          -> ondalik=','  -> enlem dtype = str
        test.csv   (yalnizca 'enlem')
          -> ondalik='.'  -> enlem dtype = float64

    Ayni kolon iki dosyada iki farkli dtype olur. Hicbir hata cikmaz; model
    egitimi ya son adimda patlar ya da kategorik/sayisal karisimi sessizce
    yanlis ogrenir.

    Cozum: dosyalarin BASLIKLARINI havuzlayip tek karar vermek. Kodlama ve
    ayirici ilk dosyadan alinir (bunlar dosya bazinda dogru tespit edilir);
    ONDALIK karari havuzlanmis metinden verilir -- cunku yanlis giden budur.
    """
    yollar = [Path(p) for p in paths]
    if not yollar:
        raise ValueError("En az bir dosya gerekli.")

    ilk_head, encoding = _decode_head(yollar[0])
    delimiter = _guess_delimiter(ilk_head)

    havuz = [ilk_head]
    for yol in yollar[1:]:
        try:
            head, _ = _decode_head(yol)
        except (OSError, ValueError):
            continue
        havuz.append(head)

    decimal, thousands, uyarilar = _guess_decimal("\n".join(havuz), delimiter)
    ilk_satir = next((line for line in ilk_head.splitlines() if line.strip()), "")

    for uyari in uyarilar:
        print(f"[sniff_dialect_shared] {len(yollar)} dosya: UYARI: {uyari}")

    return DialectGuess(
        encoding=encoding,
        delimiter=delimiter,
        decimal=decimal,
        thousands=thousands,
        sample_line=ilk_satir[:300],
        warnings=uyarilar,
    )


def read_table(
    path: str | Path,
    *,
    normalize_column_names: bool = True,
    verbose: bool = True,
    dialect: DialectGuess | None = None,
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
    # ``dialect`` verilirse ONU kullan: train/test/sample icin tek karar
    # (bkz. sniff_dialect_shared). Verilmezse dosyaya tek basina bak.
    guess = dialect if dialect is not None else sniff_dialect(path)
    kwargs = {**guess.as_read_csv_kwargs(), **overrides}

    if verbose:
        print(f"[read_table] {path.name}: {guess}")

    frame = pd.read_csv(path, **kwargs)
    frame = _adlari_normalize_et(
        frame, path, normalize_column_names=normalize_column_names, verbose=verbose
    )
    frame.attrs["dialect"] = str(guess)
    return frame


def _adlari_normalize_et(
    frame: pd.DataFrame,
    path: Path,
    *,
    normalize_column_names: bool,
    verbose: bool,
) -> pd.DataFrame:
    """Kolon adlarini normalize eder ve ham eslemeyi ``attrs``e yazar.

    Bu adim TUM formatlar icin ortaktir. Onceki surumde yalnizca CSV dalinda
    calisiyordu ve ayni veri iki formatta FARKLI kolon adlari veriyordu:
    OLCULDU -- csv ['id', 'dagitilan_enerji_mwh'] / parquet ve json
    ['ID', 'Dagitilan Enerji (MWh)']. Parquet onbellekli train + CSV ornek
    submission karisiminda hedef kolon adi eslesmiyor ve pipeline duruyordu.
    """
    if normalize_column_names:
        mapping = normalize_columns(frame.columns)
        degisen = {ham: yeni for ham, yeni in mapping.items() if ham != yeni}
        if degisen and verbose:
            ornek = list(degisen.items())[:3]
            print(
                f"[normalize] {path.name}: {len(degisen)} kolon adi degistirildi "
                f"(ornek: {ornek}). Ham adlar attrs['original_columns'] icinde."
            )
        frame = frame.rename(columns=mapping)
        frame.attrs["original_columns"] = mapping

    frame.attrs["source_path"] = str(path)
    return frame


def _oku_ikili(path: Path, suffix: str, kwargs: dict[str, Any]) -> pd.DataFrame:
    """CSV DISI formatlari okur; kolon adlarina dokunmaz."""
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path, **kwargs)
    if suffix in {".feather", ".ftr"}:
        return pd.read_feather(path, **kwargs)
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return pd.read_excel(path, **kwargs)
    if suffix == ".json":
        return pd.read_json(path, **kwargs)

    raise ValueError(f"{path}: desteklenmeyen uzanti '{suffix}'")


def read_any(
    path: str | Path,
    *,
    normalize_column_names: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    """Uzantiya gore CSV / Parquet / Excel / JSON okur.

    Parquet varsa onu tercih et: 5-20 kat hizli ve dtype'lari korur; boylece
    her notebook calistirmasinda kodlama tespitini tekrarlamazsin.

    Kolon adi normalizasyonu FORMATTAN BAGIMSIZDIR: ayni veri CSV, parquet ve
    JSON olarak okundugunda AYNI kolon adlarini verir ve ham adlar her zaman
    ``frame.attrs["original_columns"]`` icinde saklanir.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv", ".txt", ".dat", ""}:
        return read_table(
            path,
            normalize_column_names=normalize_column_names,
            verbose=verbose,
            **kwargs,
        )

    frame = _oku_ikili(path, suffix, kwargs)
    return _adlari_normalize_et(
        frame, path, normalize_column_names=normalize_column_names, verbose=verbose
    )


def to_parquet_cache(frame: pd.DataFrame, cache_path: str | Path) -> Path:
    """DataFrame'i parquet olarak onbellege yazar ve yolu dondurur.

    Kalip: ham CSV'yi bir kez oku, parquet'e yaz, sonraki her calistirmada
    parquet'ten oku. 12 gunluk bir yarismada bu, yuzlerce dakika kazandirir.
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return cache_path
