"""Capraz dogrulama semasi secimi ve sizinti (leakage) tespiti.

BU MODUL YARISMANIN KAZANILDIGI YERDIR.

Kaggle'da siralamayi belirleyen sey genellikle model degil, DOGRULAMA SEMASIDIR.
Yanlis sema iki sekilde oldurur:

  * **Iyimser CV**: lokal skorun yuksek, leaderboard'da cakiliyorsun. Neredeyse
    her zaman sizinti vardir -- zaman sizintisi (gelecegi gorerek gecmisi tahmin
    etmek) veya grup sizintisi (ayni trafo/abone hem train hem valid'de).

  * **Gurultulu CV**: fold'lar arasi sapma o kadar yuksek ki hangi degisikligin
    ise yaradigini goremezsin ve public leaderboard'a gore karar vermeye
    baslarsin -- ki bu, private leaderboard'da coke (shakeup) gitmenin tarifidir.

KARAR AGACI
-----------
    Veride zaman var mi?
      EVET -> Test train'den SONRAKI bir donem mi?
                EVET -> TimeSeriesSplit / ileri zincirleme (forward chaining)
                HAYIR -> zaman bloklu GroupKFold
      HAYIR -> Tekrarlayan varlik var mi (trafo, abone, fider)?
                 EVET -> GroupKFold (+ dengesizse StratifiedGroupKFold)
                 HAYIR -> siniflandirma? StratifiedKFold : KFold

``suggest_scheme()`` bu agaci veriye bakarak calistirir.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from .compat import categorical_columns
from .turkish import join_key

__all__ = [
    "SchemeSuggestion",
    "suggest_scheme",
    "build_splitter",
    "parse_time_series",
    "purged_time_series_split",
    "adversarial_validation",
    "leakage_report",
    "check_train_test_overlap",
    "assert_folds_align",
]

TaskType = Literal["regression", "binary", "multiclass"]

#: Ayristirilamayan tarihlerin sessizce tolere edilecegi ust oran. Uzerinde
#: hata firlatilir -- cunku NaT'lar np.argsort ile dizinin SONUNA gidip SON
#: fold'un dogrulama setine yigilir ve kimse fark etmez.
MAX_UNPARSED_TIME_RATIO = 0.02

_YIL_ONCE = re.compile(r"^\s*\d{4}\s*[-/.]")
_GUN_AY_YIL = re.compile(r"^\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{4})")


@dataclass(frozen=True)
class SchemeSuggestion:
    """Onerilen dogrulama semasi ve gerekcesi."""

    scheme: str
    reason: str
    group_column: str | None
    time_column: str | None
    stratify: bool
    warnings: tuple[str, ...] = ()

    def __str__(self) -> str:
        lines = [f"Onerilen sema: {self.scheme}", f"Gerekce: {self.reason}"]
        if self.group_column:
            lines.append(f"Grup kolonu: {self.group_column}")
        if self.time_column:
            lines.append(f"Zaman kolonu: {self.time_column}")
        for warning in self.warnings:
            lines.append(f"UYARI: {warning}")
        return "\n".join(lines)


def assert_folds_align(n_rows: int, folds: Sequence[tuple[np.ndarray, np.ndarray]]) -> None:
    """Fold indekslerinin verilen frame ile hizali oldugunu dogrular.

    NEDEN GEREKLI: ``cross_validate`` ve ``oof_target_encode`` POZISYONEL
    (``.iloc``) indeksleme kullanir ve fold'larin ayni siradaki, ayni uzunluktaki
    bir frame icin uretildigini VARSAYAR.

    Tipik kaza: fold'lar ham ``train`` uzerinden uretilir, sonra feature
    asamasinda bir ``merge`` (or. hava durumu birlestirmesi), ``dropna()`` veya
    ``sort_values()`` satir sayisini/sirasini degistirir. Indeksler sinirlar
    icinde kaldigi surece **hicbir hata firlamaz** -- CV sessizce YANLIS
    satirlari train/valid olarak esler ve skorlar anlamsizlasir.

    Bu fonksiyon o sessiz hatayi gurultulu bir hataya cevirir.

    Raises:
        ValueError: Fold bossa, indeks sinir disindaysa veya bir fold'un train
            ve valid kumeleri kesisiyorsa.
    """
    if not folds:
        raise ValueError(
            "Fold listesi bos. validation.build_splitter veya purged_time_series_split ile uret."
        )

    for index, (train_idx, valid_idx) in enumerate(folds, start=1):
        for name, indices in (("train", train_idx), ("valid", valid_idx)):
            array = np.asarray(indices)
            if array.size == 0:
                raise ValueError(f"Fold {index}: '{name}' kumesi bos.")
            if array.min() < 0 or array.max() >= n_rows:
                raise ValueError(
                    f"Fold {index}: '{name}' indeksi [{array.min()}, {array.max()}] "
                    f"araliginda ama frame yalnizca {n_rows} satir. "
                    "Fold'lar baska bir frame icin uretilmis olabilir -- "
                    "merge/dropna/sort satir sayisini degistirdi mi?"
                )

        overlap = np.intersect1d(train_idx, valid_idx, assume_unique=False)
        if overlap.size:
            raise ValueError(
                f"Fold {index}: {overlap.size} satir hem train hem valid'de. "
                "Bu dogrudan sizintidir -- CV skoru anlamsiz olur."
            )


def _gun_once_mu(values: pd.Series) -> str:
    """Gun-once mi ay-once mi? Veriden KANITLAR; tahmin etmez.

    Doner: ``"gun"`` | ``"ay"`` | ``"belirsiz"`` | ``"bilinmiyor"``.

    ``"belirsiz"`` ile ``"bilinmiyor"`` ayrimi KRITIKTIR:
      * ``"bilinmiyor"`` -- gg/aa/yyyy kalibi hic goruleMEDI (ornegin ISO
        olmayan serbest metin). Duz cozume dusmek zararsizdir.
      * ``"belirsiz"``  -- kalip GORULDU ama iki bilesen de daima <=12. Bu,
        tuzagin EN TEHLIKELI hali: her kayit sorunsuz cozulur, tek bir NaT
        cikmaz ve tarihler SESSIZCE ay-once okunur. NaT oranina bakan bir
        koruma burada hic devreye girmez.

    NEDEN TAHMIN ETMIYORUZ (bu makinede olculdu, pandas 3.0.3)
    ----------------------------------------------------------
    ``dayfirst=True`` korlemesine denenemez -- pandas onu **ISO tarihlere de
    uygular**:

        pd.to_datetime("2024-01-02", format="mixed")                -> 2024-01-02
        pd.to_datetime("2024-01-02", format="mixed", dayfirst=True) -> 2024-02-01

    Yani "once normal dene, olmazsa dayfirst dene" mantigi ISO veriyi SESSIZCE
    bozar. Bunun yerine ilk iki bileseni okuyup kanit ariyoruz: bir kayitta ilk
    bilesen 12'den buyukse gun-once KANITLANMIS olur.
    """
    metin = values.dropna().astype(str)
    if metin.empty:
        return "bilinmiyor"

    # KANITI TOPLA, ILK ESLESMEDE DONME.
    #
    # Onceki surum ilk ISO satirini gorur gormez "ay" donuyordu. Karisik bir
    # kolonda (bazi satir ISO, bazisi TR) bu, toplanmis gun-once kanitini
    # cope atiyor ve AYNI kolona iki farkli kural uygulaniyordu:
    #
    #   ham          : ['2024-03-01', '15.03.2024', '05.03.2024']
    #   dogru okuma  : ['2024-03-01', '2024-03-15', '2024-03-05']
    #   eski sonuc   : ay=[3, 3, 5]  gun=[1, 15, 3]
    #                  -> '15.03.2024' gun-once, '05.03.2024' ay-once okundu
    #                  -> UYARI YOK, ISTISNA YOK -- tamamen sessiz
    #
    # Ustelik bu bir GERILEMEYDI: daha eski surum ayni girdide gurultulu
    # "2 gecersiz tarih (%66.67)" uyarisi veriyordu. Sessiz yanlis deger,
    # gorunur yanlis degerden kotudur.
    iso_gorulen = False
    ilk_bilesenler: list[int] = []
    ikinci_bilesenler: list[int] = []
    for deger in metin:
        if _YIL_ONCE.match(deger):
            iso_gorulen = True
            continue
        eslesme = _GUN_AY_YIL.match(deger)
        if eslesme is None:
            continue
        ilk_bilesenler.append(int(eslesme.group(1)))
        ikinci_bilesenler.append(int(eslesme.group(2)))

    if not ilk_bilesenler:
        # Saf ISO ise gun-once sorusu gecersizdir; hic kalip yoksa bilinmiyor.
        return "ay" if iso_gorulen else "bilinmiyor"
    if iso_gorulen:
        # Ayni kolonda iki bicim birden. Hangi satirin hangi kurala tabi
        # oldugunu veriden bilemeyiz -- tahmin etmiyoruz.
        return "belirsiz"

    ilk_asan = any(deger > 12 for deger in ilk_bilesenler)
    ikinci_asan = any(deger > 12 for deger in ikinci_bilesenler)
    if ilk_asan and not ikinci_asan:
        return "gun"
    if ikinci_asan and not ilk_asan:
        return "ay"
    # Ikisi de asiyorsa iki bicim ayni kolonda karisik; hicbiri asmiyorsa
    # kanit yok. Iki durumda da tahmin etmiyoruz.
    return "belirsiz"


def parse_time_series(times: pd.Series, *, strict: bool = True) -> pd.Series:
    """Zaman kolonunu kronolojik siralanabilir hale getirir.

    Args:
        times: Ham zaman kolonu (datetime, ISO metin veya TR ``gg.aa.yyyy``).
        strict: ``True`` ise belirsiz bicimde ve asiri NaT'ta **hata firlatir**.
            ``False`` yalnizca tespit (``_detect_time_columns``) icindir.

    Raises:
        ValueError: Bicim belirsizse veya ayristirilamayan oran
            ``MAX_UNPARSED_TIME_RATIO``'yu asarsa.

    NEDEN BU FONKSIYON VAR (olculdu)
    --------------------------------
    Turkce veri gununde ``TARIH`` kolonu ``gg.aa.yyyy`` metni olarak gelirse
    ``pd.to_datetime(errors="coerce")`` 366 gunun **354'unu NaT** yapar --
    yalnizca gun<=12 olanlar (ay olarak okunabilenler) hayatta kalir.
    ``parsed.isna().all()`` False oldugu icin eski koruma DEVREYE GIRMEZ,
    ``np.argsort`` NaT'lari sona atar ve olculen sonuc soydur:

        fold: train_son=2024-12-06  valid_ilk=2024-01-09
              -> 73 train satiri GERCEK takvimde valid'in GELECEGINDE

    Yani ambargo hicbir sey yapmaz ve CV sessizce anlamsizlasir.
    """
    if pd.api.types.is_datetime64_any_dtype(times):
        parsed = pd.to_datetime(times)
    else:
        sira = _gun_once_mu(times)
        if sira == "belirsiz" and strict:
            # Buraya NaT oranina bakarak gelinmez: belirsiz kolonun HEPSI
            # sorunsuz cozulur, yalnizca YANLIS cozulur. Tek koruma, tahmin
            # etmeyi reddetmektir.
            ornek = list(times.dropna().astype(str).head(3))
            raise ValueError(
                "Zaman kolonunun bicimi BELIRSIZ -- gun-once mu ay-once mi "
                f"oldugu veriden kanitlanamadi (ornek: {ornek}).\n"
                "Her iki okuma da hatasiz calisir ama farkli takvim uretir, "
                "yani yanlis secim SESSIZ kalir. Kolonu once kendin cevir:\n"
                "    df['TARIH'] = pd.to_datetime(df['TARIH'], dayfirst=True)"
            )
        parsed = pd.to_datetime(times, errors="coerce", format="mixed", dayfirst=(sira == "gun"))

    if parsed.isna().all():
        raise ValueError("Zaman kolonu ayristirilamadi -- kronolojik bolme yapilamaz.")

    oran = float(parsed.isna().mean())
    if strict and oran > MAX_UNPARSED_TIME_RATIO:
        raise ValueError(
            f"Zaman kolonunun %{oran * 100:.1f}'i ayristirilamadi "
            f"(sinir %{MAX_UNPARSED_TIME_RATIO * 100:.0f}).\n"
            "Bunlar sessizce SON fold'un dogrulama setine yigilirdi. "
            "Kolonu duzelt veya bu satirlari kendin ele al."
        )
    return parsed


def _detect_time_columns(frame: pd.DataFrame) -> list[str]:
    """Datetime kolonlarini bulur; metin olarak saklanmis tarihleri de dener."""
    found = [
        column for column in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[column])
    ]
    if found:
        return found

    # Kolon adi eslestirmesinde ``join_key`` kullaniyoruz, duz ``.lower()``
    # DEGIL. Duz ``.lower()`` bu projede IKI YONDE birden kiriliyor
    # (bu makinede olculdu):
    #
    #   "TARİH".lower()  -> "tari̇h"  (U+0069 + U+0307 birlesik nokta)
    #                       -> "tarih" ipucu ESLESMEZ, kolon KACAR
    #   "TARIH".lower()  -> "tarih"   -> eslesir
    #
    # tr_lower da tek basina yetmez, ters yonde kirilir:
    #   tr_lower("TARIH") -> "tarıh"  -> "tarih" ipucu ESLESMEZ
    #
    # join_key ikisini de "tarih"e indirger (tr_lower + diyakritik katlama).
    # Bu, veri gununun ILK 30 DAKIKASINDA calisan bir yoldur: zaman kolonu
    # bulunamazsa suggest_scheme rastgele KFold onerir ve gelecek sizar.
    name_hints = ("tarih", "date", "time", "zaman", "gun", "saat", "timestamp", "ts")
    for column in frame.columns:
        normalized = join_key(str(column))
        if not any(hint in normalized for hint in name_hints):
            continue
        sample = frame[column].dropna().head(200)
        if sample.empty:
            continue
        # parse_time_series ile: duz to_datetime, TR "gg.aa.yyyy" kolonunda
        # 366 gunun 354'unu NaT yapar -> notna()=0.03 -> kolon HIC BULUNAMAZ
        # -> suggest_scheme KFold onerir -> gelecek sizar. (olculdu)
        try:
            parsed = parse_time_series(sample, strict=False)
        except ValueError:
            continue
        if parsed.notna().mean() > 0.9:
            found.append(column)
    return found


def _detect_group_columns(
    frame: pd.DataFrame, *, min_repeat: float = 2.0, max_cardinality_ratio: float = 0.5
) -> list[str]:
    """Sizintiya yol acabilecek tekrarlayan varlik kolonlarini bulur.

    Bir kolon "grup adayidir" eger: satir sayisi / benzersiz deger sayisi >= 2
    (yani her varlik ortalama en az 2 kez goruluyor) ve kardinalite cok yuksek
    degilse.
    """
    row_count = len(frame)
    if row_count == 0:
        return []

    candidates = []
    for column in frame.columns:
        series = frame[column]

        # Float kolonlari KORU SUZCE atlamayiz. Gercek veride bir tamsayi ID
        # kolonu (trafo_id) tek bir eksik deger yuzunden pandas tarafindan
        # float64'e yukseltilir. Onu atlarsak GroupKFold onerilmez, duz KFold
        # onerilir ve ayni trafo hem train hem valid'e duser -- bu modulun
        # onlemek icin var oldugu grup sizintisinin ta kendisi.
        # Cozum: tamsayiya esit float'lari (ID gorunumlu) aday kabul et.
        if pd.api.types.is_float_dtype(series):
            finite = series.dropna()
            if finite.empty or not np.all(np.equal(np.mod(finite.to_numpy(), 1), 0)):
                continue

        unique = series.nunique(dropna=True)
        if unique <= 1:
            continue
        repeat_factor = row_count / unique
        # ``<=`` ONEMLI, ``<`` DEGIL. Her varlik TAM 2 kez goruluyorsa
        # unique/row_count TAM 0.5'tir ve ``<`` onu ELER: repeat_factor kosulu
        # (2.0 >= 2.0) gecerken kardinalite kosulu sinirda takilir, grup kolonu
        # bulunamaz, KFold onerilir ve ayni varlik hem train hem valid'e duser.
        # OLCULDU: varlik basina 2 satir -> KFold/None, 3 satir -> GroupKFold/ilce.
        if repeat_factor >= min_repeat and unique / row_count <= max_cardinality_ratio:
            candidates.append((column, repeat_factor))

    candidates.sort(key=lambda pair: pair[1], reverse=True)
    return [column for column, _ in candidates]


def suggest_scheme(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
    test: pd.DataFrame | None = None,
    task_type: TaskType | None = None,
    known_group: str | None = None,
    known_time: str | None = None,
) -> SchemeSuggestion:
    """Veriye bakarak dogrulama semasi onerir.

    Bu bir OTOMATIK KARAR DEGIL, bir baslangic noktasidir. Ciktiyi oku ve
    domain bilginle dogrula -- ozellikle grup kolonu secimini.

    Args:
        target: Hedef kolon. **Grup adaylarindan cikarilir.**
        test: Test frame'i. Verilirse, test'te BULUNMAYAN kolonlar grup
            adayligindan cikarilir -- bkz. asagisi.

    NEDEN ``test`` VERMELISIN
    -------------------------
    Grup kolonu, CV'nin test bolunmesini taklit etmesi icin kullanilir; bu
    yuzden **tahmin aninda var olmak zorundadir.** Test'te olmayan bir kolona
    gore gruplamak, gerceklestirilemeyen bir semayi dogruluyormus gibi
    yapmaktir.

    Bu, prova verisinde OLCULMUS gercek bir tuzaktir: ``target`` verilmesine
    ragmen onceki surum grup adayi olarak ``ariza_var_mi`` ve ``ariza_tipi``
    doneriyordu -- ikisi de HEDEFTEN turer ve test'te yoktur. Hedefe gore
    gruplanan bir CV anlamsizdir.
    """
    warnings: list[str] = []

    time_columns = [known_time] if known_time else _detect_time_columns(frame)

    if known_group:
        group_columns = [known_group]
    else:
        group_columns = _detect_group_columns(frame)
        # Hedef ve hedeften turemis kolonlar grup olamaz: CV'yi etikete gore
        # bolmek, dogrulamayi anlamsiz kilar.
        dislanan: list[str] = []
        if target and target in group_columns:
            group_columns = [c for c in group_columns if c != target]
            dislanan.append(f"{target} (hedef)")
        if test is not None:
            yok = [c for c in group_columns if c not in test.columns]
            if yok:
                group_columns = [c for c in group_columns if c in test.columns]
                dislanan.extend(f"{c} (test'te yok)" for c in yok)
        if dislanan:
            warnings.append(
                "Grup adayligindan cikarilanlar: "
                + ", ".join(dislanan)
                + ". Grup kolonu tahmin aninda VAR OLMALIDIR."
            )

    time_column = time_columns[0] if time_columns else None
    group_column = group_columns[0] if group_columns else None

    if len(group_columns) > 1:
        warnings.append(
            f"Birden fazla grup adayi: {group_columns[:5]}. "
            "Yanlis olani secmek sizintiya yol acar -- domain bilgisiyle dogrula."
        )

    stratify = False
    if target and target in frame.columns:
        if task_type in {"binary", "multiclass"}:
            stratify = True
            counts = frame[target].value_counts(normalize=True)
            if len(counts) > 0 and counts.min() < 0.01:
                warnings.append(
                    f"Ciddi sinif dengesizligi (en nadir sinif %{counts.min() * 100:.2f}). "
                    "StratifiedKFold zorunlu; ayrica esik optimizasyonu dusun."
                )
        elif task_type is None:
            unique_targets = frame[target].nunique()
            if unique_targets <= 20:
                warnings.append(
                    f"Hedefte {unique_targets} benzersiz deger var -- siniflandirma olabilir. "
                    "task_type'i acikca belirt."
                )

    if time_column is not None:
        scheme = "TimeSeriesSplit"
        reason = (
            f"'{time_column}' zaman kolonu bulundu. Test kumesi train'den sonraki bir "
            "donemse rastgele KFold GELECEGI SIZDIRIR ve CV'yi yapay olarak yukseltir."
        )
        if group_column:
            warnings.append(
                f"Hem zaman ('{time_column}') hem grup ('{group_column}') var. "
                "En guvenlisi: zamana gore bol, sonra fold sinirinda gruplari temizle "
                "(purged_time_series_split)."
            )
        return SchemeSuggestion(
            scheme, reason, group_column, time_column, stratify, tuple(warnings)
        )

    if group_column is not None:
        scheme = "StratifiedGroupKFold" if stratify else "GroupKFold"
        reason = (
            f"'{group_column}' kolonu tekrarlaniyor (ortalama "
            f"{len(frame) / max(frame[group_column].nunique(), 1):.1f} satir/varlik). "
            "Ayni varligin hem train hem valid'de olmasi modelin ezberlemesine izin verir."
        )
        return SchemeSuggestion(scheme, reason, group_column, None, stratify, tuple(warnings))

    scheme = "StratifiedKFold" if stratify else "KFold"
    reason = "Zaman veya tekrarlayan varlik tespit edilmedi; standart K-kat uygun."
    return SchemeSuggestion(scheme, reason, None, None, stratify, tuple(warnings))


def build_splitter(scheme: str, *, n_splits: int = 5, seed: int = 42, **kwargs: Any) -> Any:
    """Sema adindan sklearn bolucusu uretir."""
    builders = {
        "KFold": lambda: KFold(n_splits=n_splits, shuffle=True, random_state=seed),
        "StratifiedKFold": lambda: StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ),
        "GroupKFold": lambda: GroupKFold(n_splits=n_splits),
        "StratifiedGroupKFold": lambda: StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        ),
        "TimeSeriesSplit": lambda: TimeSeriesSplit(n_splits=n_splits, **kwargs),
    }
    if scheme not in builders:
        raise ValueError(f"Bilinmeyen sema '{scheme}'. Secenekler: {sorted(builders)}")
    return builders[scheme]()


def _equal_count_windows(n_rows: int, n_splits: int) -> list[tuple[int, int]]:
    """Satir sayisi esit dogrulama pencereleri (klasik davranis)."""
    edges = np.linspace(0, n_rows, n_splits + 2, dtype=int)[1:]
    return [(int(edges[i]), int(edges[i + 1])) for i in range(n_splits)]


def _fixed_span_windows(
    sorted_times: np.ndarray,
    n_splits: int,
    test_span: pd.Timedelta,
    skipped: list[str],
) -> list[tuple[int, int]]:
    """Veri sonuna capalanmis, esit ZAMAN uzunlugunda dogrulama pencereleri.

    Fold'lar kronolojik sirada dondurulur (en eski once). Boylece
    ``folds[-1]`` her zaman gercek test blogunun hemen oncesindeki donemdir --
    yarismada en cok guvendigin fold odur.
    """
    last_moment = sorted_times[-1]
    span = test_span.to_timedelta64()

    windows: list[tuple[int, int]] = []
    for fold in range(n_splits):
        # Pencere (lower, upper] yarı-acik araligidir. side="right" ust siniri
        # ICERIR -- yoksa ilk fold veri setinin son anini kaybeder. Ardisik
        # pencereler bu sayede ne cakisir ne de bosluk birakir.
        upper = last_moment - fold * span
        lower = upper - span
        start = int(np.searchsorted(sorted_times, lower, side="right"))
        end = int(np.searchsorted(sorted_times, upper, side="right"))
        if start >= end:
            skipped.append(f"fold {fold + 1}: {test_span} uzunlugunda pencerede hic satir yok")
            continue
        windows.append((start, end))

    if len(windows) < n_splits:
        skipped.append(
            f"veri araligi {n_splits} x {test_span} icin yetersiz "
            f"({pd.Timestamp(sorted_times[-1]) - pd.Timestamp(sorted_times[0])} mevcut)"
        )
    return list(reversed(windows))


def purged_time_series_split(
    times: pd.Series,
    *,
    embargo: pd.Timedelta,
    n_splits: int = 5,
    test_span: pd.Timedelta | None = None,
    verbose: bool = True,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold sinirinda 'ambargo' bosluğu birakan zaman serisi bolmesi.

    Standart ``TimeSeriesSplit`` train'in son ani ile valid'in ilk anini
    bitisik birakir. Feature'lar kayan pencere (rolling) iceriyorsa -- ki
    elektrik yuku/ariza problemlerinde neredeyse her zaman icerir -- son train
    satirlarinin feature'lari ilk valid satirlariyla ayni ham veriyi kullanir.
    Bu, CV'yi iyimser gosteren ince bir sizintidir.

    Args:
        times: Zaman damgalari (frame ile ayni sirada, ayni uzunlukta).
        embargo: Train ile valid arasinda birakilacak bosluk. **ZORUNLU.**
            Kural: **en uzun kayan pencerenden BUYUK olmali.** 7/14/30 gunluk
            pencereler kullaniyorsan ``pd.Timedelta(days=30)`` uygun bir baslangictir.
            Ambargo istemiyorsan ``pd.Timedelta(0)`` yaz -- bilincli bir karar olsun.
        n_splits: Fold sayisi.
        test_span: Verilirse her dogrulama penceresi **tam olarak bu kadar zaman**
            kaplar ve pencereler veri sonundan geriye dogru dizilir. Verilmezse
            satir sayisi esitlenir (eski davranis). Asagiya bak.
        verbose: Uretilen fold sayisi istenenden azsa uyarir.

    Returns:
        ``(train_idx, valid_idx)`` konumsal indeks ciftlerinin listesi.

    ``test_span`` NEDEN ONEMLI (2023 GDZ birincisinden)
    ---------------------------------------------------
    2023 GDZ Elektrik Datathon birincisi ``TimeSeriesSplit(n_splits=3,
    test_size=744)`` kullandi. 744 saat = 31 gun = **test blogunun tam boyu**.
    Bu tesaduf degil: CV, tahmin edilecek ufku birebir taklit etmelidir.

    Satir sayisina gore esit bolme, PANEL veride yanlis pencere uretir. 96
    ilcelik gunluk bir panelde bir "ay" 96 x 30 = 2880 satirdir; satir sayisiyla
    bolersen fold uzunlugu veri yogunluguna gore kayar ve bazi fold'lar iki ay,
    bazilari on gun olur. Skorlar o zaman birbiriyle karsilastirilamaz.

    ``test_span`` verildiginde pencereler **veri sonuna capalanir** ve geriye
    dogru dizilir -- yani son fold, gercek test blogunun hemen oncesindeki
    donemdir. Yarismada en cok onemsedigin fold budur.

    ``embargo`` NEDEN ZORUNLU
    -------------------------
    Onceki surumde varsayilan ``total_span / (n_splits + 1) * 0.01`` idi. Bu,
    3 yillik gunluk bir veri setinde (n_splits=5) **~2 gun** ambargo demektir --
    yani 7, 14 veya 30 gunluk kayan pencerelerin neredeyse tamami fold sinirini
    asar ve tam olarak onlemeye calistigi sizintiya izin verir.

    Sessizce yetersiz bir varsayilan uretmektense, degeri bilincli olarak
    secmeni istiyoruz.
    """
    if len(times) == 0:
        raise ValueError("Bos zaman serisi ile bolme yapilamaz.")
    if test_span is not None and test_span <= pd.Timedelta(0):
        raise ValueError(f"test_span pozitif olmali, {test_span} verildi.")
    # Negatif ambargo train'i valid'in ICINE tasirir -- yani tam olarak bu
    # fonksiyonun onlemek icin var oldugu sizintinin kendisi, ustelik sessizce.
    # OLCULDU: embargo=-40 gun -> train'in son ani valid'in ilk anindan 39 gun
    # SONRA. Hicbir uyari cikmiyordu.
    if embargo < pd.Timedelta(0):
        raise ValueError(
            f"embargo negatif olamaz ({embargo}). Negatif ambargo train'i "
            "valid'in icine tasirir ve dogrudan sizinti uretir. "
            "Ambargo istemiyorsan pd.Timedelta(0) yaz."
        )

    # Metin olarak saklanmis tarih SOZLUKSEL siralanir ("2024-1-10" < "2024-1-2")
    # ve fold'lar kronolojik olmaz. Cevirmeyi garanti altina aliyoruz.
    parsed = parse_time_series(pd.Series(times).reset_index(drop=True))

    values = parsed.to_numpy()
    # NaT'lar np.argsort ile dizinin SONUNA gider ve fold penceresi onlari
    # gercek tarihmis gibi kullanir -- olculdu: 5 gecersiz tarihin 5'i de SON
    # fold'un valid setine yigildi, tek bir uyari cikmadan. parse_time_series
    # %2 ustunde zaten hata firlatiyor; buradaki kalinti azinligi fold'lardan
    # tamamen DISLIYORUZ (sessizce dogrulama setine koymuyoruz).
    gecerli = np.flatnonzero(~pd.isna(values))
    if gecerli.size == 0:
        raise ValueError("Zaman kolonu ayristirilamadi -- kronolojik bolme yapilamaz.")
    order = gecerli[np.argsort(values[gecerli], kind="stable")]
    sorted_times = values[order]

    atilan = len(values) - gecerli.size
    if atilan and verbose:
        print(
            f"[purged_time_series_split] {atilan} satirin tarihi ayristirilamadi; "
            "hicbir fold'a KONULMADI (aksi halde son fold'un valid setine yigilirdi)."
        )
    # Panel veride (ayni gunde birden cok satir) satir-sayisi esitleyen
    # pencereler ZAMAN uzunlugu esit olmayan fold'lar uretir ve skorlar
    # birbiriyle karsilastirilamaz hale gelir. 2023 birincisi test_size=744
    # (=31 gun) kullandi; biz de test blogunun boyunu vermeliyiz.
    if test_span is None and verbose and pd.Series(sorted_times).duplicated().any():
        print(
            "[purged_time_series_split] PANEL veri (tekrarlayan zaman damgasi) + "
            "test_span=None: fold'lar satir sayisina gore bolunuyor, ZAMAN "
            "uzunluklari esit degil. Test blogunun boyunu ver: "
            "test_span=pd.Timedelta(days=<ufuk>)."
        )

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    skipped: list[str] = []

    if test_span is None:
        windows = _equal_count_windows(len(order), n_splits)
    else:
        windows = _fixed_span_windows(sorted_times, n_splits, test_span, skipped)

    for fold, (valid_start, valid_end) in enumerate(windows):
        if valid_start >= valid_end:
            skipped.append(f"fold {fold + 1}: dogrulama penceresi bos")
            continue

        boundary = sorted_times[valid_start]
        train_mask = sorted_times < (boundary - embargo)

        train_idx = order[train_mask]
        valid_idx = order[valid_start:valid_end]

        if len(train_idx) == 0:
            skipped.append(f"fold {fold + 1}: ambargo ({embargo}) train tarafini tamamen bosaltti")
            continue
        folds.append((train_idx, valid_idx))

    # Dusen fold'lari SESSIZ birakmayiz: "5 istedim, 3 aldim" farki, skorlarin
    # neden beklenenden gurultulu oldugunu acikladigi halde gorunmez kalir.
    if verbose and len(folds) != n_splits:
        print(f"[purged_time_series_split] {n_splits} fold istendi, {len(folds)} uretildi.")
        for reason in skipped:
            print(f"  atlandi -- {reason}")

    if not folds:
        raise ValueError(
            f"Hicbir fold uretilemedi. Ambargo ({embargo}) veri araligina gore "
            "cok buyuk olabilir; kuculterek veya n_splits'i azaltarak dene."
        )

    return folds


def adversarial_validation(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    feature_columns: Sequence[str] | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """Train ve test ayni dagilimdan mi geliyor? Bir siniflandirici ile olcer.

    YONTEM: train'e 0, test'e 1 etiketi ver ve ayirt etmeye calis.
      * AUC ~ 0.5  -> dagilimlar ayni. Rastgele CV guvenli.
      * AUC > 0.8  -> ciddi kayma var. ``top_features`` sana HANGI kolonun
                      ayristirdigini soyler; genellikle bir tarih, ID veya
                      zamanla artan sayac kolonudur.
      * AUC ~ 1.0  -> bir kolon train/test'i mukemmel ayiriyor. O kolonu
                      feature olarak KULLANMA; ama test'e en cok benzeyen train
                      orneklerini secmek icin kullan.

    Bu, public leaderboard'a hic submission yapmadan test dagilimini ogrenmenin
    en ucuz yoludur.

    Returns:
        ``auc``, ``top_features``, ``sample_weights`` (test'e benzerlik olasiligi),
        ``verdict`` iceren sozluk.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError("adversarial_validation icin lightgbm gerekli.") from exc

    shared = [column for column in train.columns if column in test.columns]
    columns = list(feature_columns) if feature_columns else shared
    # Istenen ama ortak olmayan kolonlari SESSIZ dusurmeyiz: kullanici o kolonu
    # bilerek istedi ve "incelendi" saniyor. Test'te olmamasi zaten baslibasina
    # bir bulgudur.
    dusen = [column for column in columns if column not in shared]
    columns = [column for column in columns if column in shared]
    if not columns:
        raise ValueError("Train ve test arasinda ortak feature kolonu yok.")
    notlar: list[str] = []
    if dusen:
        notlar.append(
            f"Istenen {len(dusen)} kolon train+test'te ortak degil, kullanilmadi: {dusen[:10]}"
        )

    combined = pd.concat(
        [train[columns].assign(_is_test=0), test[columns].assign(_is_test=1)],
        ignore_index=True,
    )
    labels = combined.pop("_is_test").to_numpy()

    # Ham datetime kolonu LightGBM'e verilemez ve numpy DTypePromotionError ile
    # coker (olculdu). Oysa train/test'i en cok ayiran kolon TAM OLARAK odur --
    # atlamak bu fonksiyonun isini yapmamasi demektir. Epoch'a ceviriyoruz.
    zaman_kolonlari = [
        column
        for column in combined.columns
        if pd.api.types.is_datetime64_any_dtype(combined[column])
    ]
    for column in zaman_kolonlari:
        combined[column] = combined[column].astype("int64")
    if zaman_kolonlari:
        notlar.append(f"Datetime kolonlari epoch'a cevrildi: {zaman_kolonlari[:10]}")

    # Surumden bagimsiz kategorik tespiti: pandas 3.0'da metin 'str' dtype'indadir
    # ve is_object_dtype onu GORMEZ -- bkz. compat.is_categorical_like.
    for column in categorical_columns(combined):
        combined[column] = combined[column].astype("category")

    oof = np.zeros(len(combined))
    importances = np.zeros(len(columns))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, valid_idx in splitter.split(combined, labels):
        model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.1,
            num_leaves=31,
            random_state=seed,
            verbose=-1,
            # 'split' (varsayilan) kac kez BOLUNDUGUNU sayar, ne kadar
            # AYIRDIGINI degil. Olculdu: gercek ayirici kolonun split onemi
            # 896.7, gurultununki 413.0 (2.2 kat); gain'de 4008.7'ye 8.1
            # (495 kat). Yani split, sucluyu gurultunun icinde kaybediyor.
            importance_type="gain",
        )
        model.fit(combined.iloc[train_idx], labels[train_idx])
        # lightgbm predict_proba tip imzasi birlesim (ndarray | sparse | list);
        # asarray ile ndarray'e sabitlenir, davranis degismez.
        oof[valid_idx] = np.asarray(model.predict_proba(combined.iloc[valid_idx]))[:, 1]
        importances += model.feature_importances_ / n_splits

    auc = float(roc_auc_score(labels, oof))
    ranked = sorted(zip(columns, importances, strict=True), key=lambda pair: pair[1], reverse=True)

    # Agirliklarin KAC satiri gercekten tasidigini olc (Kish etkin ornek
    # buyuklugu). AUC 1.0'a yakinken model train satirlarinin neredeyse
    # hepsine ~0 verir; olculdu: 400 train satiri, agirlik toplami 2.004,
    # ESS 2.0 -- yani "bunlarla agirliklandir" tavsiyesi modeli 2 satirla
    # egitmek demek olur.
    weights = oof[: len(train)]
    kare_toplam = float(np.square(weights).sum())
    ess = float(weights.sum() ** 2 / kare_toplam) if kare_toplam > 0 else 0.0
    ess_orani = ess / max(len(train), 1)

    if auc < 0.6:
        verdict = "Dagilimlar benzer. Rastgele CV guvenli."
    elif auc < 0.8:
        verdict = (
            "Orta duzey kayma. Ilk siradaki feature'lari incele; ambargo/zaman bazli CV dusun."
        )
    elif ess_orani < 0.05:
        verdict = (
            "CIDDI kayma. Ayristiran feature'lari modelden cikar veya zamana gore bol. "
            f"sample_weights KULLANMA: etkin ornek buyuklugu {ess:.0f}/{len(train)} "
            f"(%{ess_orani * 100:.1f}) -- agirliklandirma modeli neredeyse bos bir "
            "egitim setiyle birakir."
        )
    else:
        verdict = (
            "CIDDI kayma. Ayristiran feature'lari modelden cikar veya zamana gore bol. "
            "sample_weights ile test'e benzeyen train orneklerini agirliklandir."
        )

    return {
        "auc": auc,
        "top_features": ranked[:15],
        "sample_weights": weights,
        "sample_weight_ess": ess,
        "sample_weight_ess_ratio": ess_orani,
        "notes": tuple(notlar),
        "verdict": verdict,
    }


def _kategorik_hedef_turevi(
    column_series: pd.Series,
    target_series: pd.Series,
    column: str,
    threshold: float,
) -> str | None:
    """Kategorik bir kolon hedefi belirliyor mu? Eta-kare ile olcer.

    Eta-kare = 1 - (grup ici varyans / toplam varyans). 1.0'a yakinsa kolonu
    bildiginde hedefi de biliyorsun demektir.

    Neredeyse benzersiz kolonlar (ID) TRIVIAL olarak eta-kare 1.0 verir --
    onlari eliyoruz; zaten 4. kontrol ID'leri ayrica yakaliyor.
    """
    birlikte = pd.DataFrame({"k": column_series, "h": target_series}).dropna()
    if len(birlikte) < 30:
        return None
    benzersiz = birlikte["k"].nunique()
    if benzersiz < 2 or benzersiz > len(birlikte) / 2:
        return None

    toplam_varyans = float(birlikte["h"].var(ddof=0))
    if not np.isfinite(toplam_varyans) or toplam_varyans <= 0:
        return None
    grup_ici = float(
        birlikte.groupby("k", observed=True)["h"].transform(lambda s: s - s.mean()).var(ddof=0)
    )
    eta_kare = 1.0 - grup_ici / toplam_varyans
    if eta_kare < threshold:
        return None
    return (
        f"'{column}' kategorik ama hedefin varyansinin %{eta_kare * 100:.1f}'ini "
        f"acikliyor ({benzersiz} seviye) -- hedeften turemis olabilir. "
        "Sayisal olmadigi icin korelasyon kontrolunden kaciyordu."
    )


def _zaman_ortusmesi_kontrolu(
    train: pd.DataFrame,
    test: pd.DataFrame,
    time_column: str | None,
    findings: dict[str, Any],
) -> None:
    """Train'in sonu test'in basindan sonra mi? Kolonu gerekirse kendisi bulur.

    ``time_column`` verilmediginde bu kontrol eskiden SESSIZCE atlaniyordu ve
    README quickstart tam bu sekilde cagiriyordu. OLCULDU: test train'in
    ortasinda basladigi halde time_column'suz cagri "0 kritik" dedi, ayni veri
    time_column ile "1 kritik" verdi. Raporun en agir bulgusu, kullanici bir
    argumani atladigi icin kayboluyordu.
    """
    if time_column is None:
        adaylar = [c for c in _detect_time_columns(train) if c in test.columns]
        if not adaylar:
            findings["info"].append(
                "Zaman kolonu bulunamadi -- donem ortusmesi kontrolu YAPILAMADI. "
                "Veride tarih varsa time_column= ile ver."
            )
            return
        time_column = adaylar[0]
        findings["info"].append(
            f"time_column verilmedi; '{time_column}' otomatik secildi "
            f"(adaylar: {adaylar[:5]}). Yanlissa acikca belirt."
        )

    if time_column not in test.columns:
        return

    train_times = parse_time_series(train[time_column], strict=False)
    test_times = parse_time_series(test[time_column], strict=False)
    if not (train_times.notna().any() and test_times.notna().any()):
        return

    train_max, test_min = train_times.max(), test_times.min()
    if train_max > test_min:
        findings["critical"].append(
            f"Zaman ortusmesi: train {train_max} tarihine kadar uzaniyor ama "
            f"test {test_min} tarihinde basliyor. Rastgele CV GELECEGI SIZDIRIR."
        )
    else:
        findings["info"].append(
            f"Temiz zaman ayrimi: train <= {train_max}, test >= {test_min}. "
            "Ileri zincirleme CV kullan."
        )


def leakage_report(
    train: pd.DataFrame,
    target: str,
    *,
    test: pd.DataFrame | None = None,
    time_column: str | None = None,
    correlation_threshold: float = 0.95,
) -> dict[str, Any]:
    """Sizinti belirtilerini tarar. Modeli egitmeden ONCE calistir.

    Kontroller:
      1. Hedefle neredeyse mukemmel korelasyonlu kolonlar (dogrudan sizinti)
      2. Test'te BULUNMAYAN train kolonlari (tahmin aninda erisilemez)
      3. Zaman kolonlari: train'in sonu test'in basindan sonra mi (donem ortusmesi)
      4. Tek basina hedefi belirleyen sabit/ID benzeri kolonlar
      5. Hedefle ayni benzersiz-deger imzasina sahip kolonlar
    """
    findings: dict[str, Any] = {"critical": [], "warning": [], "info": []}

    if target not in train.columns:
        raise ValueError(f"Hedef kolon '{target}' train icinde yok.")

    target_series = train[target]
    numeric_target = pd.api.types.is_numeric_dtype(target_series)

    # 1. Hedefle asiri korelasyon
    if not numeric_target:
        # Bu kontrol atlandiginda SESSIZ KALMAYIZ: "0 kritik" ozeti, en guclu
        # dedektorun hic calismadigini gizler ve kullanici "sizinti yok" saniyor.
        findings["info"].append(
            f"Hedef '{target}' sayisal degil ({target_series.dtype}) -- korelasyon "
            "tabanli sizinti kontrolu ATLANDI. Bu, raporun en guclu kontroludur. "
            "Hedefi sayisal kodlayip (or. pd.factorize) tekrar calistir."
        )
    else:
        for column in train.columns:
            if column == target:
                continue
            if pd.api.types.is_numeric_dtype(train[column]):
                valid = train[[column, target]].dropna()
                if len(valid) < 30:
                    continue
                correlation = float(valid[column].corr(valid[target]))
                if abs(correlation) >= correlation_threshold:
                    findings["critical"].append(
                        f"'{column}' hedefle {correlation:.4f} korelasyonlu -- "
                        "muhtemelen hedefin turevi veya gelecek bilgisi."
                    )
                    continue
                # Pearson yalnizca DOGRUSAL iliskiyi gorur. Hedefin tam
                # tersinir monoton bir donusumu (log1p, sqrt, rank) sizintinin
                # ta kendisidir ama Pearson'u esigin altinda kalir.
                # OLCULDU: log1p(hedef) -> Pearson 0.8841, Spearman 1.0000.
                sira = float(valid[column].corr(valid[target], method="spearman"))
                if abs(sira) >= correlation_threshold:
                    findings["critical"].append(
                        f"'{column}' hedefle Spearman {sira:.4f} (Pearson "
                        f"{correlation:.4f}) -- hedefin MONOTON donusumu olabilir; "
                        "dogrusal olmadigi icin korelasyon kontrolunden kacti."
                    )
            else:
                # Metin/kategorik kolonlar eskiden TAMAMEN atlaniyordu: hedefi
                # kovalara bolen bir metin kolonu 'tertemiz' raporlaniyordu.
                # Grup ici varyansi olcuyoruz (eta-kare): 1.0'a yakinsa kolon
                # hedefi neredeyse belirliyor demektir.
                bulgu = _kategorik_hedef_turevi(
                    train[column], target_series, column, correlation_threshold
                )
                if bulgu:
                    findings["critical"].append(bulgu)

    # 2. Test'te olmayan kolonlar
    if test is not None:
        missing_in_test = [
            column for column in train.columns if column != target and column not in test.columns
        ]
        if missing_in_test:
            findings["warning"].append(
                f"Test'te bulunmayan {len(missing_in_test)} train kolonu: "
                f"{missing_in_test[:10]} -- tahmin aninda erisilemez, feature yapma."
            )

    # 3. Zaman ortusmesi
    #
    # ``time_column`` verilmediginde bu kontrol eskiden SESSIZCE atlaniyordu ve
    # README quickstart tam bu sekilde cagiriyordu. OLCULDU: test train'in
    # ortasinda basladigi halde time_column'suz cagri "0 kritik" dedi, ayni
    # veri time_column ile "1 kritik" verdi. Raporun en agir bulgusu, kullanici
    # bir argumani atladigi icin kayboluyordu. Artik kendimiz ariyoruz.
    if test is not None:
        _zaman_ortusmesi_kontrolu(train, test, time_column, findings)

    # 4. ID benzeri kolonlar
    for column in train.columns:
        if column == target:
            continue
        unique_ratio = train[column].nunique(dropna=False) / max(len(train), 1)
        if unique_ratio > 0.99:
            findings["warning"].append(
                f"'{column}' neredeyse benzersiz (%{unique_ratio * 100:.1f}) -- ID olabilir. "
                "Sirali ID'ler zaman sizintisi tasir."
            )

    # 5. Sabit kolonlar (sizinti degil ama gurultu)
    constant = [column for column in train.columns if train[column].nunique(dropna=False) <= 1]
    if constant:
        findings["info"].append(f"Sabit kolonlar (bilgi tasimaz, cikar): {constant}")

    findings["summary"] = (
        f"{len(findings['critical'])} kritik, "
        f"{len(findings['warning'])} uyari, {len(findings['info'])} bilgi"
    )
    return findings


def check_train_test_overlap(
    train: pd.DataFrame, test: pd.DataFrame, key_columns: Sequence[str]
) -> dict[str, Any]:
    """Train ve test arasinda ayni anahtar degerleri var mi?

    Ortusen anahtarlar iki anlama gelir: ya veri sizintisi vardir ya da GroupKFold
    zorunludur. Her ikisi de bilmen gereken seydir.
    """
    columns = [
        column for column in key_columns if column in train.columns and column in test.columns
    ]
    if not columns:
        return {"overlap": 0, "note": "Ortak anahtar kolonu yok."}

    train_keys = set(map(tuple, train[columns].astype(str).to_numpy()))
    test_keys = set(map(tuple, test[columns].astype(str).to_numpy()))
    shared = train_keys & test_keys

    return {
        "key_columns": columns,
        "train_unique": len(train_keys),
        "test_unique": len(test_keys),
        "overlap": len(shared),
        "overlap_ratio": len(shared) / max(len(test_keys), 1),
        "note": (
            "Ortusme var -> GroupKFold kullan, aksi halde model ezberler."
            if shared
            else "Ortusme yok -> gruplar dogal olarak ayrik."
        ),
    }
