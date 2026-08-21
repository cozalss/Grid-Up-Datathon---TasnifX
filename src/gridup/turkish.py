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
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd

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
    "strip_qualifier",
    "split_il_ilce",
    "AnahtarKurtarma",
    "hizala_ilce_anahtarlari",
    "GrupSecimi",
    "grup_adayini_sec",
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


def strip_qualifier(name: str) -> str:
    """``"Koprubasi / Manisa"`` -> ``"Koprubasi"``. Niteleyici ekini atar.

    Ayni ilce adi birden fazla ilde bulunabildigi icin kaynak sistemler ilce
    adini il ile niteler. Referans tablolari yalin adi tutar; ikisi
    eslesmez ve join sessizce satir kaybeder (olculdu: 284 satir).

    Ayirici olarak ``/``, ``-`` ve ``(`` kabul edilir. Bos sonuc uretmez --
    ayirici basta ise ad oldugu gibi doner.
    """
    metin = str(name)
    for ayirici in ("/", "(", " - "):
        if ayirici in metin:
            bas = metin.split(ayirici, maxsplit=1)[0].strip()
            if bas:
                metin = bas
    return metin


#: "il-ilce" / "il|ilce" / "IL / ILCE" gibi BILESIK anahtarlarda kullanilan
#: ayiricilar. 2024 GDZ'de test id'si "2025-07-01-izmir-aliaga" bicimindeydi
#: ve ilce kolonu "izmir-aliaga" olarak geliyordu (docs/01:35).
_BILESIK_AYRACLAR = ("|", "/", "-", "_")


def split_il_ilce(name: str) -> tuple[str | None, str]:
    """Bilesik "il + ilce" dizgesini ``(il, ilce)`` olarak ayirir.

    ``strip_qualifier`` SOL parcayi alir ("Koprubasi / Manisa" -> "Koprubasi")
    cunku orada niteleyici SAGDADIR. Bilesik anahtarlarda ise ilce SAGDADIR
    ("izmir-karabaglar" -> "karabaglar"). Ikisini ayni fonksiyonda cozmek
    imkansizdir; bu yuzden ayri kapi (2026-08-18 denetimi, P1-12: bilesik
    dizgeler referans tablosuyla %0 esliyordu).

    Ayirici yoksa ``(None, ad)`` doner -- yani "bu zaten yalin bir ilce adi".

    Returns:
        ``(il, ilce)``; ikisi de HAM (normalize edilmemis) metindir.
    """
    metin = str(name).strip()
    for ayirici in _BILESIK_AYRACLAR:
        if ayirici in metin:
            sol, _, sag = metin.partition(ayirici)
            sol, sag = sol.strip(), sag.strip()
            if sol and sag:
                return sol, sag
    return None, metin


def _bilesikten_ilce(normalized: str) -> str:
    """Normalize bilesik anahtardan ilce parcasini alir."""
    _, ilce = split_il_ilce(normalized)
    return join_key(ilce)


def _niteleyiciyi_at(normalized: str) -> str:
    """Normalize edilmis anahtardan niteleyici ekini atar."""
    return join_key(strip_qualifier(normalized))


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

    # NITELEYICI EKI KURTARMA -- gercek GDZ verisinde olculdu.
    #
    # Ayni ilce adi Turkiye'de birden fazla ilde bulunabilir (Koprubasi hem
    # Manisa'da hem Trabzon'da). Kaynak sistemler bunu "Kopr��ba�i / Manisa"
    # diye niteler; referans tablolari ise yalin "Kopr��ba�i" tutar.
    #
    # OLCULDU (68.257 satirlik gercek GDZ kesinti kaydi, 47 ilce):
    #   normalize eslesme      : 46/47
    #   eslesmeyen             : 'koprubasi / manisa'
    #   o ilcedeki kayit sayisi: 284  -> hava/komsu join'inde SESSIZCE duserdi
    #
    # Kurtarmayi RAPORLARIZ ama join_key'i degistirmeyiz: kesme islemi
    # ("/" oncesini al) her veri setinde dogru olmayabilir, karari kullanici
    # verir. Rapor ona hangi donusumu uygulayacagini soyler.
    kalan_sol = left_norm - right_norm
    nitelikli = {
        anahtar: _niteleyiciyi_at(anahtar)
        for anahtar in kalan_sol
        if _niteleyiciyi_at(anahtar) in right_norm
    }

    # BILESIK ANAHTAR KURTARMA (P1-12): "izmir-karabaglar" gibi anahtarlarda
    # ilce SAGDADIR; niteleyici kurtarmasi (sol parca) burada ise yaramaz.
    kalan_sonrasi = kalan_sol - set(nitelikli)
    bilesik = {
        anahtar: _bilesikten_ilce(anahtar)
        for anahtar in kalan_sonrasi
        if _bilesikten_ilce(anahtar) != join_key(anahtar)
        and _bilesikten_ilce(anahtar) in right_norm
    }

    return {
        "left_unique": len(left_set),
        "right_unique": len(right_set),
        "raw_matched": raw_matched,
        "normalized_matched": normalized_matched,
        "recovered": normalized_matched - raw_matched,
        "left_only": tr_sorted(kalan_sol - set(nitelikli) - set(bilesik))[:max_examples],
        "right_only": tr_sorted(right_norm - left_norm)[:max_examples],
        "combining_dot_keys": dotted[:max_examples],
        # "koprubasi / manisa" -> "koprubasi" gibi, ek atilinca eslesenler.
        "qualifier_recoverable": dict(list(nitelikli.items())[:max_examples]),
        # "izmir-karabaglar" -> "karabaglar" gibi, bilesik anahtar ayrilinca eslesenler.
        "composite_recoverable": dict(list(bilesik.items())[:max_examples]),
    }


#: ``hizala_ilce_anahtarlari`` kurtarma sirasi. Ilk EŞLEŞEN kazanir.
#: Sira tesadufi degil: en az varsayim yapan once denenir.
_KURTARMA_SIRASI = ("dogrudan", "niteleyici", "bilesik", "takma_ad")


@dataclass(frozen=True)
class AnahtarKurtarma:
    """Tek bir panel anahtarinin referansa nasil baglandiginin kaydi.

    Attributes:
        ham: Panelde gorunen ad, hic dokunulmamis hali.
        anahtar: Referansa baglanan nihai anahtar. Kurtarilamadiysa
            ``join_key(ham)`` -- yani en iyi tahmin, ama UYDURULMAMIS.
        yontem: ``dogrudan`` | ``niteleyici`` | ``bilesik`` | ``takma_ad`` |
            ``BULUNAMADI``. Rapor satirinda okunur olmasi icin buyuk harfli
            olan yalnizca basarisizlik durumudur -- goz once ona gitsin.
    """

    ham: str
    anahtar: str
    yontem: str

    def __str__(self) -> str:  # pragma: no cover -- yalnizca log/rapor icin
        if self.yontem == "dogrudan":
            return f"{self.ham!r} -> {self.anahtar!r}"
        return f"{self.ham!r} -> {self.anahtar!r}  [{self.yontem}]"


def hizala_ilce_anahtarlari(
    ham_anahtarlar: Iterable[str],
    *,
    referans: Iterable[str],
    takma_adlar: Mapping[str, str] | None = None,
) -> dict[str, AnahtarKurtarma]:
    """Panel ilce adlarini REFERANS anahtar kumesine hizalar.

    NEDEN AYRI BIR KAPI (2026-08-21, dusmanca prova olctu)
    ------------------------------------------------------
    ``join_key`` yalnizca buyuk/kucuk harf ve aksan katlar. Gercek veride uc
    tur sapma daha var ve her biri farkli bir yolla cozulur:

        'BOZKURT / DENIZLI'  -> niteleyici eki  -> ``strip_qualifier``
        'izmir-cigli'        -> bilesik anahtar -> ``split_il_ilce``
        'AYDIN MERKEZ'       -> tarihsel ad     -> ``takma_adlar``

    Bunlar tek bir fonksiyonda cozulemez cunku niteleyici SOLDAKI parcayi,
    bilesik anahtar SAGDAKI parcayi tutar. Bu yuzden sirayla denenir.

    KRITIK GUVENCE: her kurtarma adayi ancak ``referans`` kumesine
    DUSUYORSA kabul edilir. Kosulsuz kesme, adinda tire veya parantez
    bulunan mesru bir ilceyi sessizce baska bir ilceye baglardi. Referans
    dogrulamasi bunu imkansiz kilar; kurtarilamayan ad DEGISTIRILMEZ,
    ``BULUNAMADI`` olarak isaretlenir ve cagiran taraf onu raporlar.

    NICIN SESSIZ KISMI ESLESME TEHLIKELI: ``attach_external`` %0 eslesmede
    hata verir, %50 altinda uyarir. 96 ilcenin 91'i eslestiginde (%94,8)
    ikisi de tetiklenmez -- ama eslesmeyen 5 ilcenin BUTUN dis kolonlari
    NaN olur ve model bunu "bilgi yok" degil "bu ilcede orman yok" diye
    okur. Arada kalan bu kor bandi kapatmak bu fonksiyonun tek isidir.

    Args:
        ham_anahtarlar: Panelde gorunen ilce adlari. Tekrarlar sadelestirilir.
        referans: Hedef anahtar kumesi -- tipik olarak dis tablolarin
            ``ilce_key`` kolonu.
        takma_adlar: ``normalize edilmis ad -> referans anahtari`` esleme.
            2012 buyuksehir yasasi yeniden adlandirmalari icin
            (``{"aydin merkez": "efeler"}``).

    Returns:
        ``ham ad -> AnahtarKurtarma``. Anahtarlar HAM adlardir, boylece
        cagiran taraf dogrudan ``Series.map`` ile uygulayabilir.

    Raises:
        ValueError: ``referans`` bos ise. Bos referansla her ad
            "BULUNAMADI" cikar ve bu, sessiz bir toplu arizadir.
    """
    hedef = {str(k) for k in referans}
    if not hedef:
        raise ValueError("referans kumesi bos -- hizalama anlamsiz olurdu")
    alias = {str(k): str(v) for k, v in (takma_adlar or {}).items()}

    sonuc: dict[str, AnahtarKurtarma] = {}
    for ham in ham_anahtarlar:
        metin = str(ham)
        if metin in sonuc:
            continue
        temel = join_key(metin)
        adaylar = {
            "dogrudan": temel,
            "niteleyici": _niteleyiciyi_at(metin),
            "bilesik": _bilesikten_ilce(metin),
            "takma_ad": alias.get(temel, ""),
        }
        for yontem in _KURTARMA_SIRASI:
            aday = adaylar[yontem]
            if aday and aday in hedef:
                sonuc[metin] = AnahtarKurtarma(ham=metin, anahtar=aday, yontem=yontem)
                break
        else:
            sonuc[metin] = AnahtarKurtarma(ham=metin, anahtar=temel, yontem="BULUNAMADI")
    return sonuc


@dataclass(frozen=True)
class GrupSecimi:
    """Grup kolonu adaylarindan hangisinin secildiginin OLCULMUS kaydi.

    Attributes:
        kolon: Secilen kolon adi; hicbir aday yoksa ``None``.
        eslesme_orani: Secilen kolonun benzersiz degerlerinin referans
            anahtar kumesine hizalanabilen orani (0..1).
        adaylar: ``aday -> oran`` tam tablo; rapor icin.
    """

    kolon: str | None
    eslesme_orani: float
    adaylar: dict[str, float]

    def __str__(self) -> str:
        if self.kolon is None:
            return "grup adayi yok"
        tablo = ", ".join(f"{ad}=%{100 * oran:.0f}" for ad, oran in self.adaylar.items())
        return f"grup kolonu {self.kolon!r} secildi (referans eslesmesi: {tablo})"


def grup_adayini_sec(
    frame: pd.DataFrame,  # noqa: F821 -- pandas yalnizca tip icin, calisma aninda degil
    *,
    adaylar: Iterable[str],
    referans: Iterable[str],
) -> GrupSecimi:
    """Grup kolonu adaylari arasindan REFERANSA en cok eslesenini secer.

    NEDEN OLCUM, TERCIH DEGIL (2026-08-21, dusmanca prova olctu)
    ------------------------------------------------------------
    ``suggest_scheme`` birden fazla grup adayi bulunca ilkini aliyordu ve
    gercek veride bu ``il`` oldu -- 5 degerli. Oysa butun dis tablolar
    ``ilce`` anahtarli, 96 degerli. Yanlis aday secilince ``attach_external``
    ya %0 eslesir ya hic cagrilmaz; ikisi de 219 dis kolonun kaybi demektir.

    Dogru aday tahmin edilmez, OLCULUR: her adayin benzersiz degerleri
    ``hizala_ilce_anahtarlari`` ile referansa vurulur ve en yuksek eslesme
    orani kazanir. Berabere kalirsa listedeki ilk aday korunur, yani
    sezicinin kendi sirasina saygi duyulur.

    Hicbir aday eslesmiyorsa (oran 0) bu muhtemelen ilce paneli DEGILDIR --
    baska bir varlik tipi olabilir. O durumda uydurma bir secim yapilmaz;
    ilk aday dondurulur ve oran 0 raporlanir ki cagiran taraf gorsun.

    Args:
        frame: Aday kolonlari iceren tablo.
        adaylar: Sirali aday kolon adlari (sezicinin verdigi sira).
        referans: Hedef anahtar kumesi -- dis tablolarin ``ilce_key``i.

    Returns:
        Olculmus secim kaydi.
    """
    sirali = [ad for ad in adaylar if ad in frame.columns]
    if not sirali:
        return GrupSecimi(kolon=None, eslesme_orani=0.0, adaylar={})

    hedef = {str(k) for k in referans}
    oranlar: dict[str, float] = {}
    for ad in sirali:
        degerler = frame[ad].dropna().astype(str).unique()
        if len(degerler) == 0:
            oranlar[ad] = 0.0
            continue
        kurtarmalar = hizala_ilce_anahtarlari(degerler, referans=hedef)
        tutan = sum(1 for k in kurtarmalar.values() if k.yontem != "BULUNAMADI")
        oranlar[ad] = tutan / len(degerler)

    en_iyi = max(sirali, key=lambda ad: oranlar[ad])
    if oranlar[en_iyi] == 0.0:
        en_iyi = sirali[0]
    return GrupSecimi(kolon=en_iyi, eslesme_orani=oranlar[en_iyi], adaylar=oranlar)
