"""Model harmanlama: tepe tirmanma (hill climbing) ve sira ortalamasi.

Kaggle'da ilk 10 ile ilk 100 arasindaki fark genellikle tek bir model degil,
BIRDEN FAZLA CESITLI modelin harmanidir. Ama harmanlama agirliklari elle
secilmez -- OOF tahminleri uzerinde OGRENILIR.

TEMEL KURAL: Cesitlilik (diversity) tek tek performanstan onemlidir.
Korelasyonu 0.99 olan iki mukemmel model, korelasyonu 0.85 olan iki iyi
modelden daha kotu harmanlanir.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from .metrics import get_metric

__all__ = ["hill_climb_weights", "rank_average", "correlation_matrix", "greedy_forward_selection"]


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
