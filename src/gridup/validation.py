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

__all__ = [
    "SchemeSuggestion",
    "suggest_scheme",
    "build_splitter",
    "purged_time_series_split",
    "adversarial_validation",
    "leakage_report",
    "check_train_test_overlap",
    "assert_folds_align",
]

TaskType = Literal["regression", "binary", "multiclass"]


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
            "Fold listesi bos. validation.build_splitter veya "
            "purged_time_series_split ile uret."
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


def _detect_time_columns(frame: pd.DataFrame) -> list[str]:
    """Datetime kolonlarini bulur; metin olarak saklanmis tarihleri de dener."""
    found = [
        column
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
    ]
    if found:
        return found

    name_hints = ("tarih", "date", "time", "zaman", "gun", "saat", "timestamp", "ts")
    for column in frame.columns:
        if not any(hint in column.lower() for hint in name_hints):
            continue
        sample = frame[column].dropna().head(200)
        if sample.empty:
            continue
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
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
        if repeat_factor >= min_repeat and unique / row_count < max_cardinality_ratio:
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
                "Grup adayligindan cikarilanlar: " + ", ".join(dislanan) +
                ". Grup kolonu tahmin aninda VAR OLMALIDIR."
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


def build_splitter(
    scheme: str, *, n_splits: int = 5, seed: int = 42, **kwargs: Any
) -> Any:
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
            skipped.append(
                f"fold {fold + 1}: {test_span} uzunlugunda pencerede hic satir yok"
            )
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

    # Metin olarak saklanmis tarih SOZLUKSEL siralanir ("2024-1-10" < "2024-1-2")
    # ve fold'lar kronolojik olmaz. Cevirmeyi garanti altina aliyoruz.
    parsed = pd.to_datetime(times, errors="coerce")
    if parsed.isna().all():
        raise ValueError("Zaman kolonu ayristirilamadi -- kronolojik bolme yapilamaz.")

    values = parsed.to_numpy()
    order = np.argsort(values, kind="stable")
    sorted_times = values[order]

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
            skipped.append(
                f"fold {fold + 1}: ambargo ({embargo}) train tarafini tamamen bosaltti"
            )
            continue
        folds.append((train_idx, valid_idx))

    # Dusen fold'lari SESSIZ birakmayiz: "5 istedim, 3 aldim" farki, skorlarin
    # neden beklenenden gurultulu oldugunu acikladigi halde gorunmez kalir.
    if verbose and len(folds) != n_splits:
        print(
            f"[purged_time_series_split] {n_splits} fold istendi, {len(folds)} uretildi."
        )
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
    columns = [column for column in columns if column in shared]
    if not columns:
        raise ValueError("Train ve test arasinda ortak feature kolonu yok.")

    combined = pd.concat(
        [train[columns].assign(_is_test=0), test[columns].assign(_is_test=1)],
        ignore_index=True,
    )
    labels = combined.pop("_is_test").to_numpy()

    # Surumden bagimsiz kategorik tespiti: pandas 3.0'da metin 'str' dtype'indadir
    # ve is_object_dtype onu GORMEZ -- bkz. compat.is_categorical_like.
    for column in categorical_columns(combined):
        combined[column] = combined[column].astype("category")

    oof = np.zeros(len(combined))
    importances = np.zeros(len(columns))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for train_idx, valid_idx in splitter.split(combined, labels):
        model = lgb.LGBMClassifier(
            n_estimators=200, learning_rate=0.1, num_leaves=31,
            random_state=seed, verbose=-1,
        )
        model.fit(combined.iloc[train_idx], labels[train_idx])
        oof[valid_idx] = model.predict_proba(combined.iloc[valid_idx])[:, 1]
        importances += model.feature_importances_ / n_splits

    auc = float(roc_auc_score(labels, oof))
    ranked = sorted(
        zip(columns, importances, strict=True), key=lambda pair: pair[1], reverse=True
    )

    if auc < 0.6:
        verdict = "Dagilimlar benzer. Rastgele CV guvenli."
    elif auc < 0.8:
        verdict = (
            "Orta duzey kayma. Ilk siradaki feature'lari incele; "
            "ambargo/zaman bazli CV dusun."
        )
    else:
        verdict = (
            "CIDDI kayma. Ayristiran feature'lari modelden cikar veya zamana gore bol. "
            "sample_weights ile test'e benzeyen train orneklerini agirliklandir."
        )

    return {
        "auc": auc,
        "top_features": ranked[:15],
        "sample_weights": oof[: len(train)],
        "verdict": verdict,
    }


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
            if column == target or not pd.api.types.is_numeric_dtype(train[column]):
                continue
            valid = train[[column, target]].dropna()
            if len(valid) < 30:
                continue
            correlation = float(valid[column].corr(valid[target]))
            if abs(correlation) >= correlation_threshold:
                findings["critical"].append(
                    f"'{column}' hedefle {correlation:.4f} korelasyonlu -- "
                    "muhtemelen hedefin turevi veya gelecek bilgisi."
                )

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
    if time_column and test is not None and time_column in test.columns:
        train_times = pd.to_datetime(train[time_column], errors="coerce")
        test_times = pd.to_datetime(test[time_column], errors="coerce")
        if train_times.notna().any() and test_times.notna().any():
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
        column
        for column in key_columns
        if column in train.columns and column in test.columns
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
