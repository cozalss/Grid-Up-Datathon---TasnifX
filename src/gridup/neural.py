"""Varlik gomulu (entity embedding) sinir agi -- harmana CESITLILIK icin.

NEDEN VAR, VE NEDEN BIRINCIL MODEL DEGIL
----------------------------------------
Tablo ve panel verisinde gradyan artirmali agaclar hala hukmediyor. 2023 GDZ
Elektrik Datathon'unda **birinci yalnizca CatBoost kullandi** (1.488); Prophet
tabani 4.270, RandomForest 6.794 aldi ve ilk on herkese acik notebook'un
hicbirinde sinir agi yok. Bu modulu "GBDT'yi yenecek" diye yazmiyoruz.

Yazma sebebimiz tek ve olculebilir: **harman cesitliligi.**

Bir sinir agi, agaclardan YAPISAL OLARAK farkli hatalar yapar:

  * Agac, uzayi eksen-hizali dikdortgenlere boler; MLP surekli ve duzgun bir
    yuzey ogrenir. Yumusak egilimlerde (sicaklik -> yuk) MLP daha iyi
    ekstrapole eder, kesikli siçramalarda agac kazanir.
  * Agac her ilceyi ayri bir kategori olarak gorur; **gomulu** ilceleri
    OGRENILMIS bir uzaya yerlestirir ve birbirine benzeyen ilceler bilgi
    paylasir. 96 ilcenin bazilari az veriye sahipse bu gercek bir kazanctir.

Tek basina 2 puan kotu olan bir model bile, %15 agirlikla harmana girdiginde
skoru DUSUREBILIR -- cunku harmanin kazanci korelasyonun DUSUK olmasindan
gelir, uyelerin tek tek iyi olmasindan degil.

KULLANIM
--------
``cross_validate`` ile ayni imzayi ve ayni ``CVResult``i dondurur, yani
mevcut harmanlama makinesine dogrudan takilir::

    agac = cross_validate(X, y, folds, kind="lightgbm", metric="mape")
    sinir = neural_cross_validate(X, y, folds, cat_columns=["ilce"], metric="mape")

    agirliklar = hill_climb_weights(
        {"lgbm": agac.oof_predictions, "nn": sinir.oof_predictions}, y, metric="mape"
    )

SIZINTI DISIPLINI (bu modulun en onemli kismi)
----------------------------------------------
Sinir agi on-isleme gerektirir ve **on-isleme sizintinin en sik girdigi
yerdir.** Uc sey yalnizca FOLD'UN EGITIM TARAFINDAN ogrenilir:

  1. Sayisal kolonlarin ortalama/standart sapmasi
  2. Kategorik kolonlarin sozlugu (valid'de gorulmemis kategori -> UNK)
  3. Hedefin ortalama/standart sapmasi

Tum veri uzerinde ``StandardScaler().fit(X)`` cagirmak sessiz bir sizintidir
ve CV'yi iyimser gosterir. Burada bu imkansiz: on-isleyici her fold'da
sifirdan kurulur.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .compat import is_categorical_like
from .metrics import get_metric
from .models import CVResult

__all__ = [
    "NeuralConfig",
    "neural_cross_validate",
]

#: Gorulmemis kategori icin ayrilmis indeks. Sozluk 0'i UNK'ye verir,
#: gercek kategoriler 1'den baslar.
UNKNOWN_INDEX = 0


@dataclass
class NeuralConfig:
    """Sinir agi hiperparametreleri.

    Varsayilanlar yarisma olceginde (10k-500k satir) CPU'da dakikalar
    icinde egitilecek sekilde secildi. Agresif buyutmek genelde GBDT'yi
    yakalamaya yetmez ama harman katkisini da artirmaz -- cesitlilik
    modelin BUYUKLUGUNDEN degil, FARKLILIGINDAN gelir.
    """

    hidden_sizes: tuple[int, ...] = (256, 128)
    dropout: float = 0.15
    #: Gomulu boyutu: min(max_embedding, (kategori_sayisi + 1) // 2)
    max_embedding: int = 32
    learning_rate: float = 3e-3
    weight_decay: float = 1e-5
    batch_size: int = 1024
    max_epochs: int = 120
    patience: int = 12
    seed: int = 42
    device: str | None = None  # None -> varsa cuda, yoksa cpu


def _require_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - kurulum yolu
        raise ImportError(
            "Sinir agi icin torch gerekli: pip install torch\n"
            "Kaggle'da torch KURULUDUR; yerelde yoksa GBDT harmani zaten "
            "calisir -- bu model opsiyonel bir cesitlilik uyesidir."
        ) from exc
    return torch


def _split_columns(
    frame: pd.DataFrame, cat_columns: Sequence[str] | None
) -> tuple[list[str], list[str]]:
    """Kategorik ve sayisal kolonlari ayirir.

    ``cat_columns`` verilmezse ``compat.is_categorical_like`` ile otomatik
    tespit edilir -- pandas 3.0'da duz metin kolonlari ``str`` dtype'tir ve
    ``is_object_dtype`` onlari KACIRIR; ortak yardimciyi kullaniyoruz.
    """
    if cat_columns is None:
        categorical = [c for c in frame.columns if is_categorical_like(frame[c])]
    else:
        missing = sorted(set(cat_columns) - set(frame.columns))
        if missing:
            raise KeyError(f"Kategorik kolon(lar) frame'de yok: {missing}")
        categorical = list(cat_columns)
    numeric = [c for c in frame.columns if c not in set(categorical)]
    return categorical, numeric


class _FoldPreprocessor:
    """Tek bir fold'un EGITIM tarafindan ogrenilen on-isleyici.

    Butun istatistikler ``fit`` cagrisinda ve YALNIZCA egitim satirlarindan
    hesaplanir. ``transform`` bunlari degistirmez -- dolayisiyla dogrulama
    tarafina bilgi sizmasi yapisal olarak imkansizdir.
    """

    def __init__(self, categorical: Sequence[str], numeric: Sequence[str]) -> None:
        self.categorical = list(categorical)
        self.numeric = list(numeric)
        self.vocabularies: dict[str, dict[Any, int]] = {}
        self.cardinalities: list[int] = []
        self.means: np.ndarray = np.zeros(0)
        self.stds: np.ndarray = np.ones(0)
        self.target_mean: float = 0.0
        self.target_std: float = 1.0

    def fit(self, frame: pd.DataFrame, target: np.ndarray) -> _FoldPreprocessor:
        for column in self.categorical:
            # Sozluk EGITIM tarafindan; 0 UNK'ye ayrilmis, gercekler 1'den.
            values = pd.unique(frame[column].astype("object").dropna())
            self.vocabularies[column] = {v: i + 1 for i, v in enumerate(values)}
            self.cardinalities.append(len(values) + 1)

        if self.numeric:
            block = frame[self.numeric].to_numpy(dtype="float64")
            self.means = np.nanmean(block, axis=0)
            stds = np.nanstd(block, axis=0)
            # Sabit kolonda std=0 -> bolme patlar. 1'e cekiyoruz: kolon
            # merkezlendikten sonra hep 0 olur, yani model onu gormezden gelir.
            self.stds = np.where(stds > 1e-12, stds, 1.0)

        self.target_mean = float(np.mean(target))
        target_std = float(np.std(target))
        self.target_std = target_std if target_std > 1e-12 else 1.0
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """``(kategorik_indeksler, olceklenmis_sayisallar)`` dondurur."""
        if self.categorical:
            codes = np.stack(
                [
                    frame[column]
                    .astype("object")
                    .map(self.vocabularies[column])
                    .fillna(UNKNOWN_INDEX)
                    .to_numpy(dtype="int64")
                    for column in self.categorical
                ],
                axis=1,
            )
        else:
            codes = np.zeros((len(frame), 0), dtype="int64")

        if self.numeric:
            block = frame[self.numeric].to_numpy(dtype="float64")
            scaled = (block - self.means) / self.stds
            # NaN'i 0 yapiyoruz: olceklendikten SONRA 0 = "ortalama deger".
            # Bu, eksikligi ortalamayla doldurmakla aynidir ve sizinti icermez
            # cunku ortalama egitim tarafindan geliyor.
            scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            scaled = np.zeros((len(frame), 0), dtype="float64")

        return codes, scaled.astype("float32")

    def scale_target(self, target: np.ndarray) -> np.ndarray:
        return ((target - self.target_mean) / self.target_std).astype("float32")

    def unscale_target(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.target_std + self.target_mean


def _build_network(config: NeuralConfig, cardinalities: Sequence[int], n_numeric: int):
    """Gomulu + MLP agini kurar."""
    torch = _require_torch()
    nn = torch.nn

    class TabularNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.ModuleList()
            embedding_width = 0
            for cardinality in cardinalities:
                size = min(config.max_embedding, max(2, (cardinality + 1) // 2))
                self.embeddings.append(nn.Embedding(cardinality, size))
                embedding_width += size

            layers: list[nn.Module] = []
            width = embedding_width + n_numeric
            for hidden in config.hidden_sizes:
                layers += [
                    nn.Linear(width, hidden),
                    nn.BatchNorm1d(hidden),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
                width = hidden
            layers.append(nn.Linear(width, 1))
            self.mlp = nn.Sequential(*layers)
            self.input_width = embedding_width + n_numeric

        def forward(self, codes, numeric):
            parts = [emb(codes[:, i]) for i, emb in enumerate(self.embeddings)]
            if numeric.shape[1] > 0:
                parts.append(numeric)
            return self.mlp(torch.cat(parts, dim=1)).squeeze(-1)

    return TabularNet()


def _train_one_fold(
    network,
    train_tensors,
    valid_tensors,
    config: NeuralConfig,
    torch,
) -> tuple[Any, int]:
    """Bir fold egitir, en iyi durumu geri yukler.

    Erken durdurma dogrulama kaybina bakar. **UYARI:** raporlanan fold
    skoru bu yuzden bir miktar iyimserdir -- ayni blok hem durdurma karari
    hem de skorlama icin kullanilir. GBDT tarafinda da ayni durum vardir;
    modeller arasi KARSILASTIRMA adil kalir, ama mutlak skor gercek bir
    holdout'tan biraz daha iyi gorunur.
    """
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    loss_fn = torch.nn.MSELoss()
    codes_tr, numeric_tr, y_tr = train_tensors
    codes_va, numeric_va, y_va = valid_tensors

    n_rows = len(y_tr)
    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    generator = torch.Generator().manual_seed(config.seed)

    for epoch in range(config.max_epochs):
        network.train()
        order = torch.randperm(n_rows, generator=generator)
        for start in range(0, n_rows, config.batch_size):
            batch = order[start : start + config.batch_size]
            # BatchNorm tek satirlik batch'te patlar; son parcayi atla.
            if len(batch) < 2:
                continue
            optimizer.zero_grad()
            prediction = network(codes_tr[batch], numeric_tr[batch])
            loss = loss_fn(prediction, y_tr[batch])
            loss.backward()
            optimizer.step()

        network.eval()
        with torch.no_grad():
            valid_loss = float(loss_fn(network(codes_va, numeric_va), y_va))

        if valid_loss < best_loss - 1e-6:
            best_loss, best_epoch, bad_epochs = valid_loss, epoch + 1, 0
            best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    if best_state is not None:
        network.load_state_dict(best_state)
    return network, best_epoch


def _to_tensors(
    preprocessor: _FoldPreprocessor,
    frame: pd.DataFrame,
    values: np.ndarray | None,
    torch,
    device,
):
    """Frame'i ``(kategorik, sayisal, hedef)`` tensor uclusune cevirir.

    Modul duzeyinde durur, fold dongusunun ICINDE tanimlanmaz: dongude
    tanimlanan bir kapanis, dongu degiskenini GEC baglar ve ileride biri
    cagriyi dongu disina tasidiginda sessizce YANLIS fold'un on-isleyicisini
    kullanir. Burada on-isleyici acikca parametre olarak geciyor.
    """
    codes, scaled = preprocessor.transform(frame)
    tensors = (
        torch.as_tensor(codes, device=device),
        torch.as_tensor(scaled, device=device),
    )
    if values is None:
        return (*tensors, None)
    return (*tensors, torch.as_tensor(preprocessor.scale_target(values), device=device))


def neural_cross_validate(
    train: pd.DataFrame,
    target: np.ndarray | pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    cat_columns: Sequence[str] | None = None,
    metric: str = "rmse",
    test: pd.DataFrame | None = None,
    config: NeuralConfig | None = None,
    verbose: bool = True,
) -> CVResult:
    """Varlik gomulu sinir agi ile capraz dogrulama.

    ``models.cross_validate`` ile **ayni ``CVResult``i** dondurur; boylece
    ``hill_climb_weights``, ``stack_oof`` ve ``ablation_ensemble`` bu modeli
    diger uyelerden ayirt etmez.

    Args:
        train: Feature frame'i (hedef ICERMEZ).
        target: Hedef degerler.
        folds: ``(train_idx, valid_idx)`` ciftleri -- GBDT ile **AYNI**
            fold'lari ver, yoksa OOF tahminleri harmanlanamaz.
        cat_columns: Kategorik kolonlar. ``None`` ise otomatik tespit.
        metric: Resmi metrik adi.
        test: Verilirse test tahminleri fold ortalamasi olarak uretilir.
        config: Hiperparametreler.
        verbose: Fold skorlarini yazdirir.

    Returns:
        ``CVResult``. ``feature_importance`` **kaba bir vekildir** (ilk
        katman agirliklarinin normu) ve GBDT importance'i ile ayni olcekte
        DEGILDIR -- siralama icin kullan, karsilastirma icin kullanma.

    Raises:
        ImportError: torch kurulu degilse.
        ValueError: ``folds`` bossa.
    """
    torch = _require_torch()
    if not folds:
        raise ValueError("En az bir fold gerekli.")

    config = config or NeuralConfig()
    device = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(config.seed)

    y = np.asarray(target, dtype="float64").ravel()
    categorical, numeric = _split_columns(train, cat_columns)
    metric_fn, _, _ = get_metric(metric)

    oof = np.zeros(len(y), dtype="float64")
    covered = np.zeros(len(y), dtype=bool)
    test_predictions = np.zeros(len(test)) if test is not None else None
    fold_scores: list[float] = []
    importance_total = np.zeros(len(categorical) + len(numeric))
    started = time.time()

    if verbose:
        print(
            f"[sinir agi] {device} | {len(categorical)} kategorik, "
            f"{len(numeric)} sayisal feature"
        )

    for fold_index, (train_idx, valid_idx) in enumerate(folds, start=1):
        preprocessor = _FoldPreprocessor(categorical, numeric).fit(
            train.iloc[train_idx], y[train_idx]
        )

        train_tensors = _to_tensors(
            preprocessor, train.iloc[train_idx], y[train_idx], torch, device
        )
        valid_tensors = _to_tensors(
            preprocessor, train.iloc[valid_idx], y[valid_idx], torch, device
        )

        network = _build_network(config, preprocessor.cardinalities, len(numeric)).to(device)
        network, best_epoch = _train_one_fold(
            network, train_tensors, valid_tensors, config, torch
        )

        network.eval()
        with torch.no_grad():
            scaled_prediction = network(valid_tensors[0], valid_tensors[1]).cpu().numpy()
        prediction = preprocessor.unscale_target(scaled_prediction)

        oof[valid_idx] = prediction
        covered[valid_idx] = True
        score = float(metric_fn(y[valid_idx], prediction))
        fold_scores.append(score)

        if test is not None and test_predictions is not None:
            test_codes, test_scaled = preprocessor.transform(test)
            with torch.no_grad():
                raw = network(
                    torch.as_tensor(test_codes, device=device),
                    torch.as_tensor(test_scaled, device=device),
                ).cpu().numpy()
            test_predictions += preprocessor.unscale_target(raw) / len(folds)

        importance_total += _first_layer_importance(network, preprocessor, torch)

        if verbose:
            print(f"  fold {fold_index}/{len(folds)}  {metric}={score:.6f}  ({best_epoch} epok)")

    overall = (
        float(metric_fn(y[covered], oof[covered])) if covered.any() else float("nan")
    )
    coverage = float(covered.mean())
    if verbose and coverage < 0.999:
        print(
            f"  NOT: OOF kapsami %{coverage * 100:.1f} "
            "(TimeSeriesSplit ilk donemi hic dogrulamaz -- beklenen davranis)."
        )

    importance = (
        pd.DataFrame(
            {
                "feature": categorical + numeric,
                "importance": importance_total / len(folds),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return CVResult(
        oof_predictions=oof,
        oof_covered=covered,
        test_predictions=test_predictions,
        fold_scores=fold_scores,
        overall_score=overall,
        feature_importance=importance,
        models=[],
        elapsed_seconds=time.time() - started,
        metric_name=metric,
        model_kind="neural",
    )


def _first_layer_importance(network, preprocessor: _FoldPreprocessor, torch) -> np.ndarray:
    """Ilk katman agirliklarinin feature basina L2 normu -- KABA vekil.

    GBDT'nin split-tabanli importance'i ile ayni sey DEGILDIR ve onunla
    karsilastirilamaz. Yalnizca "bu model hangi girdilere daha cok agirlik
    verdi" sorusuna kaba bir cevaptir.
    """
    with torch.no_grad():
        first_linear = next(m for m in network.mlp if isinstance(m, torch.nn.Linear))
        weights = first_linear.weight.detach().abs().mean(dim=0).cpu().numpy()

        scores: list[float] = []
        offset = 0
        for embedding in network.embeddings:
            width = embedding.embedding_dim
            scores.append(float(weights[offset : offset + width].sum()))
            offset += width
        scores.extend(float(w) for w in weights[offset:])

    expected = len(preprocessor.categorical) + len(preprocessor.numeric)
    return np.asarray(scores[:expected], dtype="float64")
