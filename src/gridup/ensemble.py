"""Model harmanlama: tepe tirmanma (hill climbing) ve sira ortalamasi.

Kaggle'da ilk 10 ile ilk 100 arasindaki fark genellikle tek bir model degil,
BIRDEN FAZLA CESITLI modelin harmanidir. Ama harmanlama agirliklari elle
secilmez -- OOF tahminleri uzerinde OGRENILIR.

TEMEL KURAL: Cesitlilik (diversity) tek tek performanstan onemlidir.
Korelasyonu 0.99 olan iki mukemmel model, korelasyonu 0.85 olan iki iyi
modelden daha kotu harmanlanir.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metrics import get_metric

__all__ = [
    "hill_climb_weights",
    "rank_average",
    "correlation_matrix",
    "greedy_forward_selection",
    "stack_oof",
    "prune_by_correlation",
]


def correlation_matrix(predictions: dict[str, np.ndarray]) -> pd.DataFrame:
    """Model tahminleri arasindaki korelasyon matrisi.

    OKUMA KILAVUZU:
      > 0.99  -> modeller aslinda ayni; harmanlamak kazanc getirmez
      0.90-0.98 -> saglikli cesitlilik, harmanlama ise yarar
      < 0.85  -> cok farkli; biri belirgin kotuyse harmanlamak zarar verebilir
    """
    frame = pd.DataFrame(predictions)
    return frame.corr()


def hill_climb_weights(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "rmse",
    n_iterations: int = 200,
    step: float = 0.01,
    verbose: bool = True,
) -> dict[str, float]:
    """OOF tahminleri uzerinde harmanlama agirliklarini ogrenir.

    YONTEM (tepe tirmanma): Bos bir harmandan basla; her adimda hangi modelden
    ``step`` kadar EKLEMEK metrigi en cok iyilestiriyorsa onu ekle. Iyilesme
    kalmayinca dur.

    Neden dogrusal regresyon degil: tepe tirmanma agirliklari dogal olarak
    negatif olmayan tutar ve toplamı 1'e normalize eder; asiri uyum riski
    cok daha dusuktur. Kaggle'da yillardir standart yontem budur.

    KRITIK: Agirliklar OOF uzerinde ogrenilir, egitim tahminleri uzerinde
    DEGIL. Aksi halde harman da asiri uyum yapar.

    Returns:
        ``{model_adi: agirlik}`` -- toplami 1.0.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names = list(oof_predictions)
    matrix = np.column_stack([oof_predictions[name] for name in names])
    weights = np.zeros(len(names))

    def score_of(weight_vector: np.ndarray) -> float:
        total = weight_vector.sum()
        if total == 0:
            return -np.inf if greater_is_better else np.inf
        blended = matrix @ (weight_vector / total)
        return float(metric_fn(y_true, blended))

    # Ilk adim: en iyi tekil modelden basla.
    single_scores = [float(metric_fn(y_true, matrix[:, index])) for index in range(len(names))]
    best_index = int(np.argmax(single_scores) if greater_is_better else np.argmin(single_scores))
    weights[best_index] = 1.0
    best_score = single_scores[best_index]

    if verbose:
        print(f"Baslangic: {names[best_index]}  {metric}={best_score:.6f}")

    for iteration in range(n_iterations):
        candidate_scores = []
        for index in range(len(names)):
            trial = weights.copy()
            trial[index] += step
            candidate_scores.append(score_of(trial))

        candidate_index = int(
            np.argmax(candidate_scores) if greater_is_better else np.argmin(candidate_scores)
        )
        candidate_score = candidate_scores[candidate_index]

        improved = (
            candidate_score > best_score if greater_is_better else candidate_score < best_score
        )
        if not improved:
            if verbose:
                print(f"{iteration} adimda yakinsadi.")
            break

        weights[candidate_index] += step
        best_score = candidate_score

    normalized = weights / weights.sum()
    # strict=True: uzunluklar yapisal olarak esit, ama esit DEGILSE zip sessizce
    # kisa olana kirpar ve bir modeli harmandan gorunmez sekilde dusurur.
    result = {name: float(weight) for name, weight in zip(names, normalized, strict=True)}

    if verbose:
        print(f"Harman {metric}={best_score:.6f}")
        for name, weight in sorted(result.items(), key=lambda pair: pair[1], reverse=True):
            if weight > 0.001:
                print(f"  {name:<28} {weight:.4f}")

    return result


def greedy_forward_selection(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "rmse",
    max_models: int = 30,
    with_replacement: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """Tekrarli acgozlu secim (Caruana yontemi) ile harman kurar.

    Tepe tirmanmanin akrabasi: her turda harmana EKLENECEK en iyi modeli sec
    (ayni model birden fazla kez secilebilir -- bu ona daha yuksek agirlik verir).

    Az sayida modelle (3-6) tepe tirmanma yeterlidir; 10+ modelde bu yontem
    daha kararli sonuc verir.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names = list(oof_predictions)
    selected: list[str] = []
    running_sum = np.zeros_like(y_true, dtype="float64")
    best_score = -np.inf if greater_is_better else np.inf

    for step in range(max_models):
        candidates = names if with_replacement else [n for n in names if n not in selected]
        if not candidates:
            break

        scores = []
        for name in candidates:
            trial = (running_sum + oof_predictions[name]) / (len(selected) + 1)
            scores.append(float(metric_fn(y_true, trial)))

        index = int(np.argmax(scores) if greater_is_better else np.argmin(scores))
        improved = scores[index] > best_score if greater_is_better else scores[index] < best_score
        if not improved:
            if verbose:
                print(f"{step} modelde yakinsadi.")
            break

        selected.append(candidates[index])
        running_sum = running_sum + oof_predictions[candidates[index]]
        best_score = scores[index]

    if not selected:
        raise ValueError("Hicbir model secilmedi -- tahminleri kontrol et.")

    counts = {name: selected.count(name) / len(selected) for name in set(selected)}

    if verbose:
        print(f"Harman {metric}={best_score:.6f}  ({len(selected)} adimda)")
        for name, weight in sorted(counts.items(), key=lambda pair: pair[1], reverse=True):
            print(f"  {name:<28} {weight:.4f}")

    return counts


def prune_by_correlation(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    metric: str = "rmse",
    max_correlation: float = 0.99,
    max_members: int = 8,
) -> list[str]:
    """Birbirine cok benzeyen modelleri eler, cesitliligi korur.

    YONTEM: En iyi modelden basla. Sirayla ekle; ama eklenecek model zaten
    secilmis bir modelle ``max_correlation`` uzerinde korelasyona sahipse ATLA.

    NEDEN: Korelasyonu 0,995 olan iki model harmanlandiginda kazanc yaklasik
    sifirdir -- ama her ikisi de tahmin suresini, model boyutunu ve
    aciklanabilirlik maliyetini iki katina cikarir. Juri notebook'u okuyacagi
    bu yarismada bu maliyet gercektir.

    Args:
        max_correlation: Bu esigin ustunde korelasyon varsa ikincisi elenir.
        max_members: Harmanda en fazla kac model kalsin.

    Returns:
        Tutulacak model adlari, en iyiden baslayarak.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    scores = {
        name: float(metric_fn(y_true, prediction))
        for name, prediction in oof_predictions.items()
    }
    order = sorted(scores, key=lambda name: scores[name], reverse=greater_is_better)

    frame = pd.DataFrame(oof_predictions)
    correlations = frame.corr()

    kept: list[str] = []
    for name in order:
        if len(kept) >= max_members:
            break
        if any(abs(correlations.loc[name, other]) > max_correlation for other in kept):
            continue
        kept.append(name)

    return kept


#: Meta-model bu kadar satirdan az ile egitilirse degenere kabul edilir.
MIN_META_TRAIN_ROWS = 20


def _resolve_base_coverage(
    oof_predictions: dict[str, np.ndarray],
    base_covered: np.ndarray | None,
    *,
    verbose: bool,
) -> np.ndarray:
    """Temel modellerin GERCEK tahmin urettigi satirlarin maskesi.

    ``base_covered`` verilmezse tespit ederiz: **tum uyelerde ayni anda tam
    sifir** olan satirlar dolgudur. Bu kalip tesadufen olusmaz -- gercek
    tahminlerin hepsinin ayni satirda bit-birebir 0.0 olmasi pratikte
    imkansizdir.

    Tespit YETERLI DEGILDIR, sadece son savunmadir: dogrusu
    ``CVResult.oof_covered`` maskelerinin kesisimini VERMEKTIR.
    """
    diziler = [np.asarray(v, dtype="float64") for v in oof_predictions.values()]
    n = len(diziler[0])

    if base_covered is not None:
        maske = np.asarray(base_covered, dtype=bool).ravel()
        if len(maske) != n:
            raise ValueError(
                f"base_covered uzunlugu ({len(maske)}) OOF uzunluguyla ({n}) uyusmuyor."
            )
        return maske

    dolgu = np.ones(n, dtype=bool)
    for dizi in diziler:
        dolgu &= dizi == 0.0
    maske = ~dolgu

    oran = float(dolgu.mean())
    if oran > 0 and verbose:
        print(
            f"  NOT: {int(dolgu.sum()):,} satirda ({oran:.1%}) TUM uyeler tam sifir -- "
            "temel OOF kapsami disi sayildi ve meta-egitimden cikarildi.\n"
            "  Daha guvenlisi: base_covered=... ile CVResult.oof_covered maskelerinin "
            "kesisimini ver."
        )
    return maske


def stack_oof(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    test_predictions: dict[str, np.ndarray] | None = None,
    base_covered: np.ndarray | None = None,
    meta: str = "ridge",
    metric: str = "rmse",
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, object]:
    """Ikinci seviye model (stacking) ile harmanlar.

    Hill climbing agirliklari OGRENIR ama hepsi POZITIF ve toplami 1'dir.
    Stacking bu kisiti kaldirir: bir model sistematik olarak fazla tahmin
    ediyorsa meta-model ona NEGATIF agirlik verip duzeltebilir.

    Bedeli aciklanabilirliktir. Bu yarismada notebook juri tarafindan
    okunuyor; stacking kazanci kucukse hill climbing tercih edilmelidir.
    ``vs_hill_climbing`` alani bu karari vermen icin ikisini de raporluyor.

    KRITIK -- SIZINTI: Meta-model, birinci seviye OOF tahminleri uzerinde
    AYNI FOLD'LARLA capraz dogrulanir. Tum OOF uzerinde tek seferde egitmek,
    meta-modelin birinci seviyenin hatalarina asiri uymasina yol acar.

    Args:
        meta: ``ridge`` (dogrusal, aciklanabilir) veya ``lgbm`` (dogrusal
            olmayan, daha guclu ama daha opak).

    Returns:
        ``oof`` (meta-modelin OOF tahmini), ``test``, ``score``,
        ``vs_hill_climbing``, ``coefficients`` iceren sozluk.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    names = list(oof_predictions)
    features = pd.DataFrame(oof_predictions)
    y = np.asarray(y_true, dtype="float64")

    # TEMEL MODELLERIN KAPSAMI -- bu blogun varlik sebebi
    # ---------------------------------------------------
    # Birinci seviye OOF dizilerinde, hicbir fold'un valid tarafinda olmayan
    # satirlar SIFIRDIR (gercek tahmin degil, dolgu). Meta-modeli o satirlarla
    # egitmek, sifir feature'lari GERCEK hedefle eslestirmek demektir.
    #
    # TimeSeriesSplit'te bu felakete donusur: ilk fold'un egitim kumesi en
    # ESKI bloktur ve o blok hicbir temel fold'un valid tarafinda degildir --
    # yani meta-egitim verisinin %100'u sifirdir.
    #
    # OLCULDU (3 uye, TimeSeriesSplit(4), N=3000):
    #   fold-1 meta katsayilari : [0. 0. 0.]   (hepsi TAM sifir)
    #   fold-1 test tahmin std  : 5.55e-17     (TAM SABIT model)
    #   test RMSE (maskesiz)    : 1.0791
    #   test RMSE (maskeli)     : 0.5352       <- IKI KATI fark
    # Sabit model test harmanina girdigi icin harman 3/4 oraninda BUZUSUYOR.
    base_mask = _resolve_base_coverage(oof_predictions, base_covered, verbose=verbose)

    meta_oof = np.zeros(len(y), dtype="float64")
    covered = np.zeros(len(y), dtype=bool)
    coefficient_rows: list[np.ndarray] = []
    fold_models: list[object] = []
    atlanan: list[int] = []

    for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
        egitim = train_idx[base_mask[train_idx]]
        if len(egitim) < MIN_META_TRAIN_ROWS:
            # Bu fold'un meta-modeli DEGENERE olurdu. Modeli listeye
            # EKLEMIYORUZ: sabit bir model test harmanini buzusturur.
            atlanan.append(fold_index)
            continue
        model = _build_meta(meta, seed)
        model.fit(features.iloc[egitim], y[egitim])
        uygula = valid_idx[base_mask[valid_idx]]
        if len(uygula) == 0:
            atlanan.append(fold_index)
            continue
        meta_oof[uygula] = model.predict(features.iloc[uygula])
        covered[uygula] = True
        fold_models.append(model)
        if hasattr(model, "coef_"):
            coefficient_rows.append(np.asarray(model.coef_, dtype="float64").ravel())

    if atlanan:
        warnings.warn(
            f"Stacking: {len(atlanan)} fold ATLANDI (fold {atlanan}) -- temel "
            f"modellerin OOF kapsami disinda kaldiklari icin meta-modelleri "
            "degenere olurdu. Kalan fold'larla devam ediliyor.\n"
            "Bu normaldir: TimeSeriesSplit'te ilk donem hicbir fold'un valid "
            "tarafinda olmaz.",
            UserWarning,
            stacklevel=2,
        )
    if not fold_models:
        raise ValueError(
            "Stacking icin kullanilabilir fold kalmadi: temel modellerin OOF "
            "kapsami cok dar. Daha az fold veya daha kucuk test_span dene."
        )

    score = float(metric_fn(y[covered], meta_oof[covered])) if covered.any() else float("nan")

    # Karsilastirma: hill climbing ne veriyordu?
    weights = hill_climb_weights(
        {name: oof_predictions[name][covered] for name in names},
        y[covered], metric=metric, verbose=False,
    )
    hill_blend = sum(
        weights[name] * oof_predictions[name][covered] for name in names
    )
    hill_score = float(metric_fn(y[covered], hill_blend))

    stacking_wins = score > hill_score if greater_is_better else score < hill_score

    test_blend = None
    if test_predictions:
        test_features = pd.DataFrame({name: test_predictions[name] for name in names})
        # Fold modellerinin ortalamasi -- tek model yerine, varyansi dusurur.
        test_blend = np.mean(
            [model.predict(test_features) for model in fold_models], axis=0
        )

    coefficients = (
        dict(zip(names, np.mean(coefficient_rows, axis=0), strict=True))
        if coefficient_rows else {}
    )

    if verbose:
        print(f"Stacking ({meta}) {metric}: {score:.6f}")
        print(f"Hill climbing  {metric}: {hill_score:.6f}")
        print(
            "  -> " + (
                "stacking kazandi" if stacking_wins
                else "hill climbing kazandi (daha aciklanabilir, onu tercih et)"
            )
        )
        if coefficients:
            print("  Meta-model katsayilari:")
            for name, value in sorted(coefficients.items(), key=lambda p: -abs(p[1])):
                marker = "  (negatif -- duzeltici)" if value < 0 else ""
                print(f"    {name:<28} {value:+.4f}{marker}")

    return {
        "oof": meta_oof,
        "test": test_blend,
        "score": score,
        "coverage": float(covered.mean()),
        "coefficients": coefficients,
        "vs_hill_climbing": {
            "stacking": score,
            "hill_climbing": hill_score,
            "stacking_wins": bool(stacking_wins),
            "hill_weights": weights,
        },
    }


def _build_meta(meta: str, seed: int) -> object:
    """Meta-model orneği kurar."""
    if meta == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=1.0, random_state=seed)

    if meta == "lgbm":
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=200, learning_rate=0.05, num_leaves=7,
            min_child_samples=50, verbose=-1, random_state=seed,
        )

    raise ValueError(f"Bilinmeyen meta model '{meta}'. 'ridge' veya 'lgbm'.")


def rank_average(
    predictions: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Sira bazli agirlikli ortalama.

    NE ZAMAN: metrik siralamaya duyarliysa (AUC, MAP, NDCG) ve modellerin
    cikti olcekleri farkliysa. Bir model 0-1 olasilik, digeri -3..+7 skor
    uretiyorsa deger ortalamasi ikincisine haksiz agirlik verir; sira
    ortalamasi bu sorunu tamamen ortadan kaldirir.
    """
    if not predictions:
        raise ValueError("Bos tahmin listesi.")

    ranked = [pd.Series(prediction).rank(pct=True).to_numpy() for prediction in predictions]

    if weights is None:
        weights = [1.0 / len(ranked)] * len(ranked)
    if len(weights) != len(ranked):
        raise ValueError("Agirlik sayisi tahmin sayisiyla uyusmuyor.")

    total = sum(weights)
    return sum(
        (weight / total) * column for weight, column in zip(weights, ranked, strict=True)
    )
