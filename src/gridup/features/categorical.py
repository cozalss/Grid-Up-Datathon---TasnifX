"""Kategorik kodlama: frekans, sayim ve fold-disi (out-of-fold) hedef kodlama.

HEDEF KODLAMA TABULAR YARISMALARIN EN GUCLU VE EN TEHLIKELI TEKNIGIDIR.

Fikir: her kategoriyi, o kategorideki hedefin ortalamasiyla degistir. Yuksek
kardinaliteli kolonlarda (trafo_id, ilce, abone_grubu) one-hot'un yapamadigi
seyi yapar.

TEHLIKE: Naif uygulama -- tum train uzerinde ortalama alip ayni train'e
uygulamak -- her satirin KENDI hedefini feature'ina karistirir. Sonuc: CV
skorun ucar, leaderboard skorun cakilir. Bu, Kaggle'da en sik yapilan hatadir.

COZUM: Kodlamayi FOLD ICINDE hesapla. Bir satirin kodlamasi, o satirin
bulunmadigi fold'lardan gelmelidir. Bu modul bunu zorunlu kilar --
``oof_target_encode`` fold indeksi olmadan CALISMAZ.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ..compat import safe_str
from ..validation import assert_folds_align

__all__ = [
    "FrequencyEncoder",
    "TargetEncodingResult",
    "add_frequency_encoding",
    "add_count_encoding",
    "oof_target_encode",
    "add_combination_features",
    "reduce_rare_categories",
]


class FrequencyEncoder:
    """Train frekanslarini bir kez ogrenip sonraki frame'lere uygular.

    ``fit`` yalnizca egitim dagilimini gorur. Boylece train ve test ayri ayri
    kodlandiginda ayni kategorinin iki farkli sayisal anlama gelmesi engellenir.
    """

    def __init__(self, columns: Sequence[str], *, suffix: str = "_frekans") -> None:
        self.columns = tuple(columns)
        self.suffix = suffix
        self._frequencies: dict[str, pd.Series] | None = None

    def fit(self, frame: pd.DataFrame) -> FrequencyEncoder:
        """Frekans haritalarini ``frame`` (train) uzerinden ogren."""
        frequencies: dict[str, pd.Series] = {}
        for column in self.columns:
            if column not in frame.columns:
                raise KeyError(f"Kolon '{column}' frame icinde yok.")
            mapping = frame[column].value_counts(normalize=True, dropna=False)
            # Eksik deger bir kategori degildir. Paydada kalir ama transform'da
            # daima NaN kalmasi icin haritadan cikarilir.
            frequencies[column] = mapping[mapping.index.notna()].copy()
        self._frequencies = frequencies
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Fit edilmis train haritalarini yeni bir frame'e uygula."""
        if self._frequencies is None:
            raise RuntimeError("FrequencyEncoder.transform oncesinde fit cagrilmalidir.")

        new_columns = {}
        for column in self.columns:
            if column not in frame.columns:
                raise KeyError(f"Kolon '{column}' frame icinde yok.")
            mapped = frame[column].map(self._frequencies[column]).astype("float32")
            new_columns[f"{column}{self.suffix}"] = mapped.mask(
                mapped.isna() & frame[column].notna(), 0.0
            )
        return frame.assign(**new_columns)

    def fit_transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Train haritasini ogren ve ayni frame'e uygula."""
        return self.fit(frame).transform(frame)


class TargetEncodingResult(tuple):
    """Iki elemanli eski sonucu, OOF kapsam maskesiyle zenginlestirir.

    Tuple alt sinifi oldugu icin ``encoded_train, encoded_test = result`` ve
    ``result[0]`` gibi mevcut kullanimlar aynen calisir.
    """

    covered: np.ndarray

    def __new__(
        cls,
        encoded_train: pd.DataFrame,
        encoded_test: pd.DataFrame | None,
        covered: np.ndarray,
    ) -> TargetEncodingResult:
        result = super().__new__(cls, (encoded_train, encoded_test))
        coverage = np.asarray(covered, dtype=bool).copy()
        coverage.setflags(write=False)
        result.covered = coverage
        return result

    @property
    def encoded_train(self) -> pd.DataFrame:
        return self[0]

    @property
    def encoded_test(self) -> pd.DataFrame | None:
        return self[1]


def add_frequency_encoding(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    reference: pd.DataFrame | None = None,
    suffix: str = "_frekans",
) -> pd.DataFrame:
    """Her kategoriyi goreli frekansiyla kodlar. YENI frame dondurur.

    Hedefi KULLANMAZ, dolayisiyla sizinti riski yoktur ve fold disina cikmasi
    gerekmez. Cogu zaman hedef kodlamaya yakin performans verir -- once bunu dene.

    Args:
        reference: Frekanslarin hesaplanacagi referans (genellikle
            ``pd.concat([train, test])``). ``None`` ise ``frame`` kullanilir.
            Train+test uzerinden hesaplamak MESRUDUR: test'in hedefini degil,
            yalnizca dagilimini kullanir.
    """
    source = reference if reference is not None else frame
    return FrequencyEncoder(columns, suffix=suffix).fit(source).transform(frame)


def add_count_encoding(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    reference: pd.DataFrame | None = None,
    suffix: str = "_sayim",
) -> pd.DataFrame:
    """Her kategoriyi mutlak gorulme sayisiyla kodlar. YENI frame dondurur.

    Frekans kodlamasindan farki: mutlak buyukluk bilgisini korur. Bir trafonun
    veri setinde 10.000 kez gorunmesi ile 3 kez gorunmesi farkli seylerdir --
    ikincisi muhtemelen yeni kurulmus veya arizali bir kayittir.
    """
    source = reference if reference is not None else frame
    new_columns = {}

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        counts = source[column].value_counts(dropna=False)
        mapped = frame[column].map(counts).astype("float32")
        # Bkz. add_frequency_encoding: eksik deger 0 DEGIL, NaN kalir.
        new_columns[f"{column}{suffix}"] = mapped.mask(mapped.isna() & frame[column].notna(), 0.0)

    return frame.assign(**new_columns)


def _smoothed_means(
    categories: pd.Series,
    target: pd.Series,
    *,
    prior: float,
    smoothing: float,
) -> pd.Series:
    """Bayesci yumusatma ile kategori ortalamalari.

    Formul::

        kodlama = (n * kategori_ortalamasi + m * genel_ortalama) / (n + m)

    ``n`` az oldugunda kodlama genel ortalamaya yaklasir -- yani 3 ornekli bir
    kategori, 3.000 ornekli bir kategori kadar guvenilir sayilmaz. Yumusatma
    olmadan nadir kategoriler saf gurultu tasir ve model onlari ezberler.
    """
    stats = target.groupby(categories, observed=True).agg(["sum", "count"])
    return (stats["sum"] + prior * smoothing) / (stats["count"] + smoothing)


def oof_target_encode(
    train: pd.DataFrame,
    target: pd.Series,
    columns: Sequence[str],
    folds: Iterable[tuple[np.ndarray, np.ndarray]],
    *,
    test: pd.DataFrame | None = None,
    smoothing: float = 20.0,
    noise_level: float = 0.0,
    seed: int = 42,
    suffix: str = "_hedef_kod",
    uncovered_policy: str = "error",
) -> TargetEncodingResult:
    """Sizintisiz (fold-disi) hedef kodlama. YENI frame'ler dondurur.

    Her train satirinin kodlamasi, o satirin AIT OLMADIGI fold'lardan hesaplanir.
    Test kodlamasi tum train'den hesaplanir (test hedefi zaten bilinmiyor,
    sizinti yok).

    Args:
        train: Egitim feature'lari.
        target: Hedef degerler (``train`` ile ayni uzunlukta, ayni sirada).
        columns: Kodlanacak kategorik kolonlar.
        folds: ``(train_idx, valid_idx)`` ciftleri -- CV bolucusunun ciktisi.
            ZORUNLU: sizintisiz kodlamanin tek yolu budur.
        smoothing: Bayesci yumusatma agirligi. Yuksek deger = daha muhafazakar.
            Kategori basina ortalama ornek sayisiyla ayni mertebede sec.
        noise_level: Kodlamaya eklenecek gaussian gurultunun standart sapmasi
            (hedefin std'sine oranla). Asiri uyumu kirar; 0.01-0.05 tipiktir.
        seed: Gurultu icin tohum.
        uncovered_policy: Hicbir valid fold'una girmeyen train satirlari icin
            politika. Varsayilan ``"error"`` kismi OOF kapsamini reddeder.
            ``"nan"`` bu satirlari NaN birakir ve ``result.covered`` ile maskeyi
            tasir; modelleme katmani yalnizca kapsanan satirlari kullanmalidir.

    Returns:
        Iki elemanli tuple-uyumlu ``TargetEncodingResult``. ``test``
        verilmemisse ikinci eleman ``None``; ``covered`` OOF kapsam maskesidir.

    Raises:
        ValueError: ``folds`` bossa, politika gecersizse veya varsayilan
            politikada OOF kapsami kismiysa.
    """
    fold_list = list(folds)
    if not fold_list:
        raise ValueError(
            "oof_target_encode fold gerektirir. Fold'suz hedef kodlama HER SATIRIN "
            "KENDI HEDEFINI sizdirir ve CV'yi anlamsiz kilar."
        )

    if len(train) != len(target):
        raise ValueError(f"train ({len(train)}) ve target ({len(target)}) uzunluklari farkli.")

    # Fold'lar gercekten bu frame icin mi uretildi? Kontrol etmezsek yanlis
    # satirlar kodlanir ve hicbir hata firlamaz.
    assert_folds_align(len(train), fold_list)

    if uncovered_policy not in {"error", "nan"}:
        raise ValueError(
            f"uncovered_policy yalnizca 'error' veya 'nan' olabilir; verilen: {uncovered_policy!r}"
        )

    covered = np.zeros(len(train), dtype=bool)
    for _, apply_idx in fold_list:
        covered[apply_idx] = True
    if uncovered_policy == "error" and not covered.all():
        missing = int((~covered).sum())
        raise ValueError(
            "OOF hedef kodlama kapsami kismi: "
            f"{missing}/{len(train)} train satiri hicbir valid fold'unda yok. "
            "Bu satirlari egitime sessizce katmak temporal sizintidir. "
            "Kapsanan satirlari result.covered ile secmek icin "
            "uncovered_policy='nan' kullanin."
        )

    target_values = pd.Series(np.asarray(target, dtype=float), index=range(len(target)))
    # Test kodlamasi icin tum train'den hesaplanan prior. Test hedefi zaten
    # bilinmedigi icin burada sizinti yok.
    global_prior = float(target_values.mean())
    rng = np.random.default_rng(seed)

    train_encoded: dict[str, np.ndarray] = {}
    test_encoded: dict[str, np.ndarray] = {}

    for column in columns:
        if column not in train.columns:
            raise KeyError(f"Kolon '{column}' train icinde yok.")

        categories = train[column].reset_index(drop=True)
        oof = np.full(len(train), np.nan, dtype="float64")
        fold_priors = np.full(len(train), np.nan, dtype="float64")

        for fit_idx, apply_idx in fold_list:
            # Prior da FOLD ICINDEN hesaplanir. Global prior kullanmak, satirin
            # kendi hedefinden gelen 1/N buyuklugunde bir katkiyi kendi
            # kodlamasina sizdirir -- kucuk ama fold izolasyonunu tam kilmayan
            # bir ihlal. Kucuk veri setlerinde (12 gunluk datathon boyutlari)
            # ve yuksek smoothing'de etkisi olcülebilir hale gelir.
            fold_prior = float(target_values.iloc[fit_idx].mean())
            means = _smoothed_means(
                categories.iloc[fit_idx],
                target_values.iloc[fit_idx],
                prior=fold_prior,
                smoothing=smoothing,
            )
            oof[apply_idx] = categories.iloc[apply_idx].map(means).to_numpy(dtype="float64")
            fold_priors[apply_idx] = fold_prior

        # KAPSANAN ama fit fold'unda gorulmemis kategori -> o fold'un ortalamasi.
        # Kapsanmayan satir ayri bir durumdur ve ``nan`` politikasinda NaN kalir.
        unseen_in_fold = covered & np.isnan(oof)
        oof[unseen_in_fold] = fold_priors[unseen_in_fold]

        if noise_level > 0:
            # Tek satirlik hedefte std() (ddof=1) NaN doner ve TUM kolonu
            # NaN yapar -- yukaridaki NaN doldurma guvencesini de asar.
            spread = float(target_values.std())
            scale = spread * noise_level if np.isfinite(spread) else 0.0
            if scale > 0:
                # ravel(): toplama sonucu 'n boyutlu' tiplenir; oof 1 boyutlu
                # kalmali (eski numpy stub'lari bu catismayi yakaliyor).
                oof = (oof + rng.normal(0.0, scale, size=len(oof))).ravel()

        train_encoded[f"{column}{suffix}"] = oof.astype("float32")

        if test is not None:
            if column not in test.columns:
                raise KeyError(f"Kolon '{column}' test icinde yok.")
            full_means = _smoothed_means(
                categories, target_values, prior=global_prior, smoothing=smoothing
            )
            mapped = test[column].map(full_means).to_numpy(dtype="float64")
            test_encoded[f"{column}{suffix}"] = np.where(
                np.isnan(mapped), global_prior, mapped
            ).astype("float32")

    encoded_train = train.assign(**train_encoded)
    encoded_test = test.assign(**test_encoded) if test is not None else None
    return TargetEncodingResult(encoded_train, encoded_test, covered)


def add_combination_features(
    frame: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    *,
    separator: str = "__",
) -> pd.DataFrame:
    """Iki kategorik kolonu birlestirip yeni kategori uretir. YENI frame dondurur.

    NEDEN: ``ilce`` ve ``ariza_tipi`` ayri ayri zayif olabilir ama
    ``ilce__ariza_tipi`` bir etkilesim yakalayabilir -- or. kirsal bir ilcede
    agac temasi arizasi sik, sanayi ilcesinde asiri yuk arizasi sik.

    Kardinalite patlamasina dikkat: 50 ilce x 20 ariza tipi = 1000 kategori.
    ``reduce_rare_categories`` ile birlikte kullan.
    """
    new_columns = {}
    for left, right in pairs:
        for column in (left, right):
            if column not in frame.columns:
                raise KeyError(f"Kolon '{column}' frame icinde yok.")
        name = f"{left}{separator}{right}"
        # safe_str: pandas 2.x'te .astype(str) NaN'i "None" stringine cevirir ve
        # "None__bornova" gibi UYDURMA kategoriler uretir. Taraflardan biri
        # eksikse sonuc da eksik olmali -- surumden bagimsiz olarak.
        left_text = safe_str(frame[left])
        right_text = safe_str(frame[right])
        combined = left_text + separator + right_text
        either_missing = frame[left].isna() | frame[right].isna()
        new_columns[name] = combined.mask(either_missing).astype("category")
    return frame.assign(**new_columns)


def reduce_rare_categories(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    min_count: int = 10,
    other_label: str = "_DIGER",
    reference: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Nadir kategorileri tek bir kova altinda toplar. YENI frame dondurur.

    NEDEN: Bir kategori 3 kez gorunuyorsa, model onun icin ogrendigi sey
    gurultudur ve test'te muhtemelen hic gorunmeyecektir. Bunlari birlestirmek
    hem varyansi hem bellek kullanimini dusurur.

    ``reference`` verilirse esik ORADA hesaplanir -- train ve test'in ayni
    kategorileri birlestirmesi icin bu gereklidir.
    """
    source = reference if reference is not None else frame
    new_columns = {}

    for column in columns:
        if column not in frame.columns:
            raise KeyError(f"Kolon '{column}' frame icinde yok.")
        counts = source[column].value_counts(dropna=False)
        keep = set(counts[counts >= min_count].index)
        as_object = frame[column].astype(object)
        new_columns[column] = as_object.where(as_object.isin(keep), other_label).astype("category")

    return frame.assign(**new_columns)
