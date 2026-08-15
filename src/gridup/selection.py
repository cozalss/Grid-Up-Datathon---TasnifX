"""Feature secimi: SHAP geri eleme ve null importance.

NEDEN BU MODUL VAR
------------------
Bu pipeline kolayca 400+ feature uretir (takvim x lag x rolling x grup x hava x
komsu). Hepsini modele vermek uc sorun yaratir:

  * **Varyans.** Gurultulu feature'lar fold'lar arasi sapmayi buyutur; kucuk
    iyilesmeler gorunmez hale gelir.
  * **Asiri uyum.** Ozellikle yuksek kardinaliteli kodlamalar.
  * **Aciklanabilirlik.** 400 feature'lik bir modeli juriye anlatamazsin.

UYARI -- DOGRULANMAMIS ATIF KALDIRILDI
Onceki surumde burada "2024 GDZ Datathon birincisi 490 -> 97 feature indirdi"
yaziyordu. Bu iddia **DOGRULANAMADI ve muhtemelen YANLISTIR**:
  * Kaggle'da GDZ'nin 2024 yarismasi YOKTUR
  * Gercek olan 2023 birincisinin notebook'u okundu: ``stage_one_exclude = []``
    -- yani HIC feature secimi yapmamis, 490->97 diye bir sey olmamis
Feature secimi yine de mesru bir tekniktir; ama bu modulun gerekcesi bir
yarisma anekdotu DEGIL, yukaridaki uc somut sebeptir.
Feature sayisi-skor egrisi sunumda guclu bir slayt olur.

IKI YONTEM, FARKLI SORULAR
--------------------------
``shap_backward_selection``
    "Modelin kararina en az katkida bulunan feature hangisi?" Katkiyi SHAP ile
    olcer, en dusukleri atar, CV'yi yeniden calistirir. Pahali ama en guvenilir.

``null_importance_filter``
    "Bu feature'in onemi SANS ESERI olabilir mi?" Hedefi karistirip modeli
    tekrar egitir; gercek onem, karistirilmis dagilimin ustunde degilse
    feature gurultudur. Ucuz, ilk eleme icin ideal.

Sira: once ``null_importance_filter`` ile bariz gurultuyu at (dakikalar),
sonra ``shap_backward_selection`` ile ince ayar yap (saatler).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .models import ModelKind, cross_validate, fit_without_validation, starter_params

__all__ = [
    "SelectionStep",
    "SelectionResult",
    "mean_absolute_shap",
    "fold_shap_importance",
    "null_importance_filter",
    "shap_backward_selection",
]


@dataclass(frozen=True)
class SelectionStep:
    """Geri elemenin tek adimi."""

    n_features: int
    score: float
    dropped: tuple[str, ...]
    elapsed_seconds: float


@dataclass
class SelectionResult:
    """Feature secimi sonucu."""

    best_features: list[str]
    best_score: float
    history: list[SelectionStep] = field(default_factory=list)
    greater_is_better: bool = False

    def curve(self) -> pd.DataFrame:
        """Feature sayisi - skor egrisi. Juri sunumunda dogrudan slayt olur."""
        return pd.DataFrame(
            [
                {"feature_sayisi": step.n_features, "skor": step.score}
                for step in self.history
            ]
        ).sort_values("feature_sayisi")

    @property
    def selection_optimism(self) -> float:
        """``best_score`` ile adim skorlarinin ortalamasi arasindaki fark.

        ``best_score`` AYNI fold'larda kosulan N korele denemenin en iyisidir;
        yani bagimsiz bir kumede beklenenden IYIMSERDIR. Bu sayi o iyimserligin
        kaba bir olcusudur -- sifir degilse, kazancin bir kismi secimden gelir.
        """
        if len(self.history) < 2:
            return 0.0
        ortalama = float(np.mean([step.score for step in self.history]))
        return abs(self.best_score - ortalama)

    def summary(self) -> str:
        if not self.history:
            return "Adim yok."
        first, last = self.history[0], self.history[-1]
        direction = "yukseldi" if self.best_score > first.score else "dusdu"
        lines = [
            f"Baslangic: {first.n_features} feature, skor {first.score:.6f}",
            f"Bitis:     {last.n_features} feature, skor {last.score:.6f}",
            f"En iyi:    {len(self.best_features)} feature, skor {self.best_score:.6f}"
            f"  (skor {direction})",
            f"Toplam {len(self.history)} adim.",
        ]
        # SECIM YANLILIGI ACIKCA SOYLENIR.
        #
        # best_score, AYNI fold'lar uzerinde kosulan N denemenin minimumudur.
        # Bunu "modelin skoru" diye raporlamak, 200 noktali bir izgarada esik
        # secip ayni veride skor bildirmekle ayni hatadir. OLCULDU: 6 adimda
        # best 0.620982, adim ortalamasi 0.627206 -> ~%1 iyimserlik.
        if len(self.history) >= 2:
            lines.append(
                f"DIKKAT: bu skor {len(self.history)} korele denemenin EN IYISIDIR "
                f"(adim ortalamasindan {self.selection_optimism:.6f} uzakta). "
                "Secim yanliligi tasir -- kazanci bagimsiz bir kumede dogrula."
            )
        return "\n".join(lines)


def mean_absolute_shap(
    model: Any, features: pd.DataFrame, *, sample_size: int = 5000, seed: int = 42
) -> pd.Series:
    """Feature basina ortalama mutlak SHAP degeri.

    Args:
        sample_size: SHAP hesabi O(satir) maliyetlidir. 5.000 satir, siralama
            icin fazlasiyla yeterli -- 500.000 satirda hesaplamak 100 kat
            pahali ve siralamayi degistirmez.

    Returns:
        Feature adi -> ortalama |SHAP|, buyukten kucuge sirali.

    Raises:
        ImportError: ``shap`` kurulu degilse.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Feature secimi icin 'shap' gerekli: pip install shap") from exc

    sample = features
    if len(features) > sample_size:
        sample = features.sample(n=sample_size, random_state=seed)

    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(sample)

    # Cok sinifli modeller liste veya 3B dizi dondurur; sinifar arasi ortalama al.
    if isinstance(values, list):
        values = np.mean([np.abs(block) for block in values], axis=0)
    else:
        values = np.abs(values)
        if values.ndim == 3:
            values = values.mean(axis=2)

    return (
        pd.Series(values.mean(axis=0), index=sample.columns)
        .sort_values(ascending=False)
    )


def fold_shap_importance(
    models: Sequence[Any],
    features: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    sample_per_fold: int = 2000,
    seed: int = 42,
) -> pd.DataFrame:
    """Her fold'un modelini KENDI VALIDATION satirlarinda degerlendirir.

    NEDEN BU, TEK MODELDEN FARKLI VE NEDEN ONEMLI
    ---------------------------------------------
    SHAP'i bir fold modelinin **egitim** satirlarinda hesaplamak, o modelin
    ezberledigi orgu uzerinde olcum yapmaktir. Asiri uyan bir feature -- ornegin
    yuksek kardinaliteli bir kimlik kodlamasi -- egitim satirlarinda cok yuksek
    SHAP alir cunku model onlari gercekten ezberlemistir. Validation satirlarinda
    ise katkisi cokerdi.

    Sonuc: geri elemede tam da atilmasi gereken feature'lar en yuksek onemi alir
    ve **elemeden kurtulur**. Eleme, gurultuyu temizlemek yerine gurultuyu korur.

    Bu, ``shap_backward_selection``in onceki surumundeki gercek bir hataydi:
    ``models[0]`` ile TUM egitim cercevesinde hesap yapiliyordu.

    Args:
        models: ``CVResult.models`` -- fold sirasiyla.
        features: Modellerin egitildigi tam feature cercevesi.
        folds: ``models`` ile AYNI sirada ``(train_idx, valid_idx)`` ciftleri.
        sample_per_fold: Fold basina SHAP ornek satiri.

    Returns:
        ``feature``, ``mean_abs_shap``, ``rank_std`` kolonlu DataFrame,
        onemine gore BUYUKTEN KUCUGE sirali.

        ``rank_std`` bedava bir kararlilik olcusudur: bir feature'in siralamasi
        fold'lar arasinda cok oynuyorsa, ortalama SHAP'i yuksek olsa bile
        guvenilmezdir -- muhtemelen belirli bir donemin artefaktidir.

    Raises:
        ValueError: ``models`` ve ``folds`` uzunluklari farkliysa.
    """
    if len(models) != len(folds):
        raise ValueError(
            f"models ({len(models)}) ve folds ({len(folds)}) uzunluklari farkli. "
            "Fold sirasi korunmali -- aksi halde her model yanlis satirlarda olculur."
        )

    per_fold: list[pd.Series] = []
    for model, (_, valid_idx) in zip(models, folds, strict=True):
        validation = features.iloc[valid_idx]
        if validation.empty:
            continue
        per_fold.append(
            mean_absolute_shap(model, validation, sample_size=sample_per_fold, seed=seed)
        )

    if not per_fold:
        raise ValueError("Hicbir fold icin validation satiri bulunamadi.")

    matrix = pd.concat(per_fold, axis=1)
    ranks = matrix.rank(ascending=False, axis=0)

    return (
        pd.DataFrame(
            {
                "feature": matrix.index,
                "mean_abs_shap": matrix.mean(axis=1).to_numpy(),
                "rank_std": ranks.std(axis=1).fillna(0.0).to_numpy(),
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


#: Null-importance icin agac sayisi. Bilerek DUSUK: bu adim bir SIRALAMA
#: adimidir, en iyi modeli kurmak degil. Cok agac split sayisini doyurur ve
#: gercek sinyali gurultuden ayirt edilemez kilar (olculdu: 5000 agacta
#: gercek sinyalin 2/3'u kaybediliyor).
NULL_IMPORTANCE_TREES = 300

#: Agac sayisi anahtari modele gore farklidir ve kutuphaneler takma adlarin
#: AYNI ANDA verilmesini reddeder.
_TREE_COUNT_KEY: dict[str, str] = {
    "lightgbm": "n_estimators",
    "xgboost": "n_estimators",
    "catboost": "iterations",
}


def null_importance_filter(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    *,
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    params: dict[str, Any] | None = None,
    n_runs: int = 5,
    percentile: float = 75.0,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Onemi SANS ESERI aciklanabilecek feature'lari isaretler.

    YONTEM: Hedefi ``n_runs`` kez karistir ve her seferinde modeli egit. Bir
    feature karistirilmis hedefte de yuksek onem aliyorsa, o onem feature'in
    yapisindan gelir (yuksek kardinalite, cok benzersiz deger) -- gercek
    sinyalden degil.

    Karar kurali: gercek onem, null dagiliminin ``percentile``'inden BUYUK
    olmali. 75 muhafazakar bir esiktir; 50 daha agresif eler.

    Args:
        n_runs: Kac karistirma. 5 iyi bir denge; artirmak esigi stabilize eder.

    Returns:
        ``keep`` (tutulacak feature listesi), ``drop``, ``scores`` (DataFrame:
        gercek onem, null percentile, oran).
    """
    y = np.asarray(target).ravel()
    model_params = dict(params) if params else starter_params(kind, task_type)
    # ACIK ATAMA -- setdefault DEGIL.
    #
    # Onceki surum ``setdefault("n_estimators", 300)`` yaziyordu ve bu OLU
    # KODDU: ``starter_params`` zaten 5000 doner, dolayisiyla 300 hicbir
    # zaman uygulanmazdi. Niyet edilen 300 yerine 5000 agac egitiliyordu.
    #
    # OLCULDU (3 gercek sinyal + 40 saf gurultu, N=4000):
    #   5000 agac : 23.6 sn, tutulan gercek 1/3   <- sinyalin 2/3'u KAYIP
    #    300 agac :  1.1 sn, tutulan gercek 3/3
    # 21 KAT hizli VE dogru. Cok agacli modelde split sayisi doyuyor ve
    # gercek sinyal ile gurultu ayirt edilemez hale geliyor.
    #
    # Kullanici acikca params verirse ona dokunmayiz.
    #
    # ANAHTAR MODELE GORE DEGISIR. Onceki surum her modele "n_estimators"
    # yaziyordu; CatBoost'un baslangic parametrelerinde ise "iterations" var
    # ve CatBoost ikisini AYNI ANDA kabul etmez:
    #   CatBoostError: only one of the parameters iterations, n_estimators,
    #                  num_boost_round, num_trees should be initialized
    # Yani catboost yolu HER ZAMAN cokuyordu (olculdu) -- ne setdefault'lu
    # eski surumde ne acik atamali yeni surumde calisiyordu.
    if not params:
        model_params[_TREE_COUNT_KEY.get(kind, "n_estimators")] = NULL_IMPORTANCE_TREES

    from .models import _extract_importance, _prepare_categoricals

    prepared, _, categorical = _prepare_categoricals(train, None, kind)
    columns = list(prepared.columns)

    def _importance(target_values: np.ndarray, run_seed: int) -> np.ndarray:
        run_params = dict(model_params)
        run_params["random_state"] = run_seed
        if kind == "catboost":
            run_params["random_seed"] = run_seed
            run_params.pop("random_state", None)
        model = fit_without_validation(kind, run_params, prepared, target_values, categorical)
        return _extract_importance(model, columns)

    if verbose:
        print(f"Gercek onem hesaplaniyor ({len(columns)} feature)...")
    actual = _importance(y, seed)

    rng = np.random.default_rng(seed)
    null_runs = []
    for run in range(n_runs):
        shuffled = rng.permutation(y)
        null_runs.append(_importance(shuffled, seed + run + 1))
        if verbose:
            print(f"  null kosusu {run + 1}/{n_runs}")

    null_matrix = np.vstack(null_runs)
    null_threshold = np.percentile(null_matrix, percentile, axis=0)

    # Sifira bolme: null esigi 0 ise gercek onem >0 olmasi yeterlidir.
    with np.errstate(divide="ignore", invalid="ignore"):
        no_null_signal = np.where(actual > 0, np.inf, 0.0)
        ratio = np.where(null_threshold > 0, actual / null_threshold, no_null_signal)

    scores = pd.DataFrame(
        {
            "feature": columns,
            "gercek_onem": actual,
            "null_esik": null_threshold,
            "oran": ratio,
        }
    ).sort_values("oran", ascending=False)

    keep = scores.loc[scores["oran"] > 1.0, "feature"].tolist()
    drop = scores.loc[scores["oran"] <= 1.0, "feature"].tolist()

    if verbose:
        print(f"\nTutulan: {len(keep)}   Atilan: {len(drop)}")
        if drop:
            print(f"  Atilanlardan ornekler: {drop[:8]}")

    return {"keep": keep, "drop": drop, "scores": scores}


def shap_backward_selection(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    kind: ModelKind = "lightgbm",
    task_type: str = "regression",
    metric: str = "rmse",
    params: dict[str, Any] | None = None,
    drop_per_step: int = 25,
    min_features: int = 20,
    max_steps: int = 20,
    patience: int = 3,
    shap_sample: int = 2000,
    seed: int = 42,
    progress: Callable[[str], None] | None = print,
) -> SelectionResult:
    """SHAP tabanli geri eleme. Her adimda en zayif feature'lari atar.

    NOT: onceki surum bu proseduru "2024 GDZ birincisinin birebir yontemi"
    diye tanitiyordu -- DOGRULANMADI, bkz. modul docstring'i.

    Args:
        drop_per_step: Her adimda kac feature atilacak.
        min_features: Bu sayinin altina inme.
        max_steps: Ust sinir -- 12 gunluk yarismada zaman butcesi kontrolu.
        patience: Kac adim ust uste iyilesme olmazsa dur.
        shap_sample: SHAP hesabi icin ornek satir sayisi.

    Returns:
        ``SelectionResult``. ``best_features`` EN IYI CV skorunu veren kumedir --
        son adim degil. Egri genellikle once iyilesip sonra bozulur.

    UYARI -- SURE: Her adim TAM bir CV kosusu demektir. 20 adim x 5 fold =
    100 model egitimi. Once ``null_importance_filter`` ile kabaca ele, sonra
    bunu az sayida adimla calistir.
    """
    from .metrics import get_metric

    _, greater_is_better, _ = get_metric(metric)
    y = np.asarray(target).ravel()
    model_params = dict(params) if params else starter_params(kind, task_type)

    features = list(train.columns)
    history: list[SelectionStep] = []
    best_features, best_score = list(features), None
    stale = 0

    for step in range(max_steps):
        started = time.perf_counter()
        subset = train[features]

        result = cross_validate(
            subset, y, folds, kind=kind, task_type=task_type, metric=metric,
            params=model_params, early_stopping_rounds=100, verbose=False,
        )
        elapsed = time.perf_counter() - started

        improved = (
            best_score is None
            or (result.overall_score > best_score if greater_is_better
                else result.overall_score < best_score)
        )
        if improved:
            best_score = result.overall_score
            best_features = list(features)
            stale = 0
        else:
            stale += 1

        if progress:
            marker = " *" if improved else ""
            progress(
                f"adim {step + 1:>2}: {len(features):>4} feature  "
                f"{metric}={result.overall_score:.6f}  ({elapsed:.0f} sn){marker}"
            )

        if stale >= patience:
            # SON ADIM DA KAYDEDILIR.
            #
            # Onceki surum burada history'ye yazmadan ``break`` ediyordu, yani
            # BEDELI ODENMIS bir CV kosusunun sonucu kayboluyordu -- ustelik
            # tam da durma kararini gerekcelendiren adim. OLCULDU: 3 CV kosusu
            # yapildi, history'de 2 adim vardi ve summary() "Toplam 2 adim"
            # diyordu. Juriye sunulan eleme egrisinin son noktasi eksikti.
            history.append(
                SelectionStep(len(features), result.overall_score, (), elapsed)
            )
            if progress:
                progress(f"{patience} adimdir iyilesme yok -- duruluyor.")
            break

        if len(features) - drop_per_step < min_features:
            if progress:
                progress(f"Alt sinira ({min_features}) ulasildi -- duruluyor.")
            history.append(
                SelectionStep(len(features), result.overall_score, (), elapsed)
            )
            break

        # SHAP'i her fold'un KENDI VALIDATION satirlarinda hesapla.
        # Tek modelin tum egitim cercevesinde hesaplamak, asiri uyan
        # feature'lara sisirilmis onem verir ve tam da elenmeleri gereken
        # feature'lari elemeden kurtarir. Ayrintili gerekce: fold_shap_importance.
        importance = fold_shap_importance(
            result.models, subset, folds, sample_per_fold=shap_sample, seed=seed
        )
        weakest = importance.tail(drop_per_step)["feature"].tolist()

        history.append(
            SelectionStep(len(features), result.overall_score, tuple(weakest), elapsed)
        )
        features = [column for column in features if column not in weakest]

    return SelectionResult(
        best_features=best_features,
        best_score=float(best_score) if best_score is not None else float("nan"),
        history=history,
        greater_is_better=greater_is_better,
    )
