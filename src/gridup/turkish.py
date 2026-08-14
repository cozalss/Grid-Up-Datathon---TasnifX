"""Turkce metin tehlikeleri: buyuk/kucuk harf, join anahtari, siralama.

NEDEN BU MODUL VAR
------------------
Python'un ``.lower()`` metodu locale'den BAGIMSIZDIR ve Turkce esleme yapmaz.
Iki ayri hataya yol acar:

1. Yanlis harf:  ``'ISIK'.lower() -> 'isik'``  (Turkce dogrusu 'isik' degil 'isik'tir;
   ``I`` harfinin Turkce kucugu ``i`` degil ``i``dir.)

2. Birlesik isaret:  ``'I'.lower()`` tek kod noktasi uretmez, IKI uretir:
   ``U+0069 U+0307``. Yani ``'I'.lower() == 'i'`` -> **False**.
   ``casefold()`` ve NFC normalizasyonu bunu duzeltmez.

Ikinci hata bu yarismada somut bir risktir: GDZ (Izmir, Manisa) ve ADM (Aydin,
Denizli, Mugla) bolgesi il/ilce adlari ``I`` ile doludur. Harici veriyi (hava
durumu, nufus) il adiyla join ederken merge SESSIZCE 0 satir dondurur --
istisna yok, uyari yok.

TESHIS
------
Bir merge beklenenden az satir donduruyorsa, once U+0307 ara::

    from gridup.turkish import codepoints
    codepoints("Izmir")   # U+0069'den sonra U+0307 varsa hata budur
"""

from __future__ import annotations

import unicodedata as _ud
from collections.abc import Iterable

__all__ = [
    "tr_lower",
    "tr_upper",
    "join_key",
    "tr_sort_key",
    "tr_sorted",
    "codepoints",
    "has_combining_dot",
    "normalize_columns",
    "diagnose_join",
]

# ``.lower()`` cagrilmadan ONCE i-ciftini eslememiz gerekir; birlesik noktayi
# ureten sey zaten ``.lower()``in kendisidir.
_UPPER_TO_LOWER = str.maketrans({"İ": "i", "I": "ı"})
_LOWER_TO_UPPER = str.maketrans({"ı": "I", "i": "İ"})

# Aksan giderme: YALNIZCA join anahtari icin. Ekranda gosterilecek metin icin ASLA.
_FOLD = str.maketrans("çğıöşü", "cgiosu")

# Turk alfabesi sirasi. Dikkat: 'c' harfinden sonra 'c', ve 'i' harfi 'i'DEN ONCE gelir.
_TR_ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
_RANK = {char: index for index, char in enumerate(_TR_ALPHABET)}
_UNKNOWN_RANK = len(_TR_ALPHABET) + 1

COMBINING_DOT_ABOVE = "̇"


def tr_lower(text: str) -> str:
    """Turkce kurallarina uygun kucuk harf.

    >>> tr_lower("İZMİR")
    'izmir'
    >>> tr_lower("IŞIK")
    'ışık'
    >>> tr_lower("İ") == "i"
    True
    """
    return text.translate(_UPPER_TO_LOWER).lower()


def tr_upper(text: str) -> str:
    """Turkce kurallarina uygun buyuk harf.

    >>> tr_upper("izmir")
    'İZMİR'
    >>> tr_upper("ışık")
    'IŞIK'
    """
    return text.translate(_LOWER_TO_UPPER).upper()


def join_key(text: str) -> str:
    """Iki veri kaynagini eslestirmek icin aksansiz, kucuk harfli anahtar.

    Kaynaklar aksan konusunda anlasmaz: biri ``Mugla`` yazar, digeri ``Mugla``.
    Bu anahtar ikisini de ayni degere indirger.

    UYARI: Bu deger EKRANDA GOSTERILMEZ. Yalnizca eslestirme icin ayri bir
    kolonda saklanir.

    >>> join_key("MUĞLA") == join_key("Muğla") == join_key("Mugla")
    True
    >>> join_key("  İzmir ")
    'izmir'
    """
    folded = _ud.normalize("NFC", tr_lower(text.strip())).translate(_FOLD)
    return " ".join(folded.split())


def tr_sort_key(text: str) -> list[int]:
    """``sorted(key=...)`` icin Turk alfabesi siralamasi.

    Cikplak ``sorted()`` kod noktasina gore siralar ve Turkce'ye ozgu her harfi
    ``z``den SONRA atar.
    """
    return [_RANK.get(char, _UNKNOWN_RANK) for char in tr_lower(text)]


def tr_sorted(values: Iterable[str], *, reverse: bool = False) -> list[str]:
    """Turk alfabesine gore sirali liste dondurur (yeni liste; girdi degismez)."""
    return sorted(values, key=tr_sort_key, reverse=reverse)


def codepoints(text: str) -> list[str]:
    """Her karakterin kod noktasini dondurur. Sessiz join hatalarinin teshisi icin.

    >>> codepoints("İ".lower())
    ['U+0069', 'U+0307']
    """
    return [f"U+{ord(char):04X}" for char in text]


def has_combining_dot(text: str) -> bool:
    """Metinde U+0307 (birlesik ustnokta) var mi? Varsa yanlis ``.lower()`` kullanilmis."""
    return COMBINING_DOT_ABOVE in text


def normalize_columns(columns: Iterable[str]) -> dict[str, str]:
    """Ham kolon adlarini -> guvenli snake_case adlara esleyen sozluk uretir.

    Turkce kolon adlari (``ARIZA_SÜRESİ``, ``İL``, ``TÜKETİM (kWh)``) hem
    ``df.ARIZA_SURESI`` erisimini bozar hem de ``.lower()`` ile birlesik nokta
    uretir. Bu fonksiyon deterministik, ASCII-guvenli adlar verir.

    Cakisma olursa ad korunur ve ``_2``, ``_3`` eklenir -- sessizce kolon
    kaybetmemek icin.

    >>> normalize_columns(["İL", "Kesinti Süresi (dk)", "ARIZA_TİPİ"])
    {'İL': 'il', 'Kesinti Süresi (dk)': 'kesinti_suresi_dk', 'ARIZA_TİPİ': 'ariza_tipi'}
    """
    mapping: dict[str, str] = {}
    seen: dict[str, int] = {}

    for raw in columns:
        base = join_key(str(raw))
        safe = "".join(char if char.isalnum() else "_" for char in base)
        safe = "_".join(part for part in safe.split("_") if part)
        if not safe:
            safe = "kolon"
        if safe[0].isdigit():
            safe = f"k_{safe}"

        count = seen.get(safe, 0) + 1
        seen[safe] = count
        candidate = safe if count == 1 else f"{safe}_{count}"

        # Sonek eklemek YENI bir cakisma yaratabilir: ham kolonlar arasinda
        # zaten 'a_b_2' varsa ve 'A/B' + 'A B' cakismasi 'a_b_2' uretiyorsa
        # iki kolon ayni ada duser. pandas bunu HATA VERMEZ -- frame['a_b_2']
        # artik Series degil, iki kolonlu bir DataFrame doner ve downstream
        # kod ya patlar ya sessizce yanlis calisir.
        while candidate in mapping.values():
            count += 1
            seen[safe] = count
            candidate = f"{safe}_{count}"

        mapping[raw] = candidate

    if len(set(mapping.values())) != len(mapping):  # pragma: no cover - savunma
        raise ValueError("Kolon adi normalizasyonu cakisma uretti; ham adlari kontrol et.")

    return mapping


def diagnose_join(
    left_keys: Iterable[str],
    right_keys: Iterable[str],
    *,
    max_examples: int = 10,
) -> dict[str, object]:
    """Iki anahtar kumesinin neden eslesmedigini raporlar.

    Ham eslesmeyi, ``join_key`` ile normalize edilmis eslesmeyi ve birlesik
    nokta iceren anahtarlari karsilastirir. Merge'den ONCE calistir.

    Donen sozluk: ``raw_matched``, ``normalized_matched``, ``recovered``
    (normalizasyonun kurtardigi satir sayisi), ``left_only``, ``right_only``,
    ``combining_dot_keys``.
    """
    left = list(left_keys)
    right = list(right_keys)
    left_set, right_set = set(left), set(right)

    left_norm = {join_key(key) for key in left}
    right_norm = {join_key(key) for key in right}

    raw_matched = len(left_set & right_set)
    normalized_matched = len(left_norm & right_norm)

    dotted = [key for key in left + right if has_combining_dot(key)]

    return {
        "left_unique": len(left_set),
        "right_unique": len(right_set),
        "raw_matched": raw_matched,
        "normalized_matched": normalized_matched,
        "recovered": normalized_matched - raw_matched,
        "left_only": tr_sorted(left_norm - right_norm)[:max_examples],
        "right_only": tr_sorted(right_norm - left_norm)[:max_examples],
        "combining_dot_keys": dotted[:max_examples],
    }
