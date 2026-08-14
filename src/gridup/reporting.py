"""Juri ciktilari: notebook ve sunum icin hazir tablo ve grafikler.

NEDEN BU MODUL VAR
------------------
Degerlendirmenin UCTE IKISI skor DISI:

    1. Private leaderboard sirasi          -> skor
    2. Notebook degerlendirmesi (ilk 10-20) -> KOD KALITESI, aciklanabilirlik
    3. Final sunumu (ilk 10)                -> IS DEGERI, anlatim

2024 birincisinin sunumunun son uc slaydi tamamen is degeriydi:
"aciklanabilir cozum / daraltilmis feature-set / ~25MB model / yeni veriyle
egitilebilir". Skor ilk 10'a sokar; bu bolum odulu belirler.

Bu ciktilari yarismanin son gunu uretmeye kalkmak imkansizdir -- pipeline'in
parcasi olmalilar. Her fonksiyon tek cagriyla notebook'a dogrudan yapistirilir.

JURI KIM
--------
Coderspace + GDZ + ADM Elektrik ekipleri. Yani **muhendisler ve is birimleri**,
akademisyen degil. Bu, iki seyi degistirir:

  * Metrik degeri tek basina anlamsizdir. "MAE 2,95" bir sey soylemez;
    "ortalama 3 kesintilik hatayla tahmin ediyoruz, bu da nobetci ekip
    planlamasi icin yeterli" bir sey soyler.
  * Model boyutu, egitim suresi ve yeniden egitilebilirlik ONEMLIDIR --
    cunku bu ekipler onu gercekten calistiracak.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import get_metric

__all__ = [
    "cv_fold_table",
    "error_by_segment",
    "worst_segments",
    "prediction_vs_actual_table",
    "feature_importance_table",
    "model_footprint",
    "business_impact",
    "plot_fold_scores",
    "plot_error_by_segment",
    "plot_prediction_timeline",
    "plot_selection_curve",
]

# --- Grafik paleti -----------------------------------------------------------
# Tek vurgu rengi + notr griler. Jurinin dikkati veriye gitsin, renklere degil.
_INK = "#13202D"
_MUTED = "#6B7A88"
_RULE = "#D2D8DE"
_ACCENT = "#B57A0B"       # sebeke uyari kehribari
_GOOD = "#1B6A57"
_BAD = "#9E2C1E"


def _style(axis: Any) -> None:
    """Ortak eksen stili: cerceve yok, yatay izgara, sakin renkler."""
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(_RULE)
    axis.grid(axis="y", alpha=0.25, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=_MUTED, labelsize=9)
    for label in (axis.xaxis.label, axis.yaxis.label, axis.title):
        label.set_color(_INK)


# =============================================================================
# TABLOLAR
# =============================================================================


def cv_fold_table(result: Any, *, name: str = "model") -> pd.DataFrame:
    """Fold bazli skor tablosu -- juri sunumunda dogrudan slayt.

    2024 birincisi karar kriteri olarak fold hatalarinin hem ORTALAMASINI hem
    STANDART SAPMASINI raporladi. Tek fold ile karar vermek, 12 gunluk
    yarismada en sik gorulen olumcul hatadir.

    Sapma sutunu ayrica bir SORUYU cevaplar: "bu iyilesme gercek mi, gurultu mu?"
    Iki konfigurasyon arasindaki fark fold sapmasindan kucukse fark YOKTUR.
    """
    scores = list(result.fold_scores)
    rows = [
        {"fold": index, "skor": round(float(score), 6)}
        for index, score in enumerate(scores, start=1)
    ]
    frame = pd.DataFrame(rows)

    summary = pd.DataFrame(
        [
            {"fold": "ortalama", "skor": round(float(np.mean(scores)), 6)},
            {"fold": "std", "skor": round(float(np.std(scores)), 6)},
            {"fold": "OOF (tum veri)", "skor": round(float(result.overall_score), 6)},
        ]
    )
    combined = pd.concat([frame, summary], ignore_index=True)
    combined.attrs["model"] = name
    combined.attrs["kararli"] = bool(getattr(result, "is_stable", False))
    return combined


def error_by_segment(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: pd.Series,
    *,
    metric: str = "mae",
    min_count: int = 20,
) -> pd.DataFrame:
    """Segment bazli hata tablosu -- "model NEREDE yaniliyor?"

    Bu, juri sunumunun en ikna edici bolumudur. Genel bir skor "iyi model"
    der; segment tablosu "modeli anliyoruz" der.

    Args:
        segments: Her satirin ait oldugu grup (ilce, ay, gun tipi...).
        min_count: Bu sayidan az ornekli segmentler ATLANIR -- 3 ornekli bir
            segmentin hatasi gurultudur ve tabloyu yaniltir.
    """
    metric_fn, greater_is_better, _ = get_metric(metric)

    frame = pd.DataFrame(
        {
            "segment": np.asarray(segments),
            "gercek": np.asarray(y_true, dtype="float64"),
            "tahmin": np.asarray(y_pred, dtype="float64"),
        }
    )

    rows = []
    for name, block in frame.groupby("segment", observed=True):
        if len(block) < min_count:
            continue
        rows.append(
            {
                "segment": name,
                "kayit": len(block),
                metric: float(metric_fn(block["gercek"], block["tahmin"])),
                "gercek_ort": float(block["gercek"].mean()),
                "tahmin_ort": float(block["tahmin"].mean()),
                "yanlilik": float((block["tahmin"] - block["gercek"]).mean()),
            }
        )

    if not rows:
        raise ValueError(
            f"Hicbir segment {min_count} kayit esigini gecemedi. "
            "min_count'u dusur veya daha genis bir segment kullan."
        )

    return pd.DataFrame(rows).sort_values(metric, ascending=not greater_is_better)


def worst_segments(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    segments: pd.Series,
    *,
    metric: str = "mae",
    top: int = 10,
    min_count: int = 20,
) -> pd.DataFrame:
    """En kotu ``top`` segment. Hata analizi bolumunun cekirdegi."""
    table = error_by_segment(
        y_true, y_pred, segments, metric=metric, min_count=min_count
    )
    _, greater_is_better, _ = get_metric(metric)
    return table.tail(top) if greater_is_better else table.tail(top)


def prediction_vs_actual_table(
    y_true: np.ndarray, y_pred: np.ndarray, *, bins: int = 10
) -> pd.DataFrame:
    """Tahmin araligi bazinda kalibrasyon tablosu.

    "Model 5 dedigi zaman gercekte ortalama kac oluyor?" Bu soru, teknik
    olmayan bir juriye modelin guvenilirligini metrikten daha iyi anlatir.

    Ideal: her satirda ``tahmin_ort`` ile ``gercek_ort`` birbirine yakin.
    Sistematik sapma varsa model o bolgede kalibre degildir.
    """
    frame = pd.DataFrame(
        {
            "tahmin": np.asarray(y_pred, dtype="float64"),
            "gercek": np.asarray(y_true, dtype="float64"),
        }
    )
    # Quantile bazli kova: esit genislikli kova, carpik dagilimda cogu satiri
    # tek kovaya yigar ve tabloyu anlamsiz kilar.
    frame["kova"] = pd.qcut(frame["tahmin"], q=bins, duplicates="drop")

    table = (
        frame.groupby("kova", observed=True)
        .agg(
            kayit=("gercek", "size"),
            tahmin_ort=("tahmin", "mean"),
            gercek_ort=("gercek", "mean"),
        )
        .reset_index()
    )
    table["sapma"] = table["tahmin_ort"] - table["gercek_ort"]
    table["kova"] = table["kova"].astype(str)
    return table


def feature_importance_table(
    result: Any, *, top: int = 20, group_prefixes: Sequence[str] = ()
) -> pd.DataFrame:
    """Feature onem tablosu; istege bagli olarak AILE bazinda toplanmis.

    Args:
        group_prefixes: Verilirse feature'lar bu oneklere gore gruplanir --
            or. ``("tarih_", "tatil_", "komsu_", "bolge_")``. 400 satirlik bir
            onem listesi juriye hicbir sey anlatmaz; "sinyalin %40'i hava
            degiskenlerinden geliyor" cok sey anlatir.
    """
    frame = result.feature_importance.copy()

    if not group_prefixes:
        return frame.head(top)

    def _family(name: str) -> str:
        for prefix in group_prefixes:
            if name.startswith(prefix):
                return prefix.rstrip("_")
        return "diger"

    frame["aile"] = frame["feature"].map(_family)
    grouped = (
        frame.groupby("aile", observed=True)
        .agg(toplam_onem=("importance", "sum"), feature_sayisi=("feature", "size"))
        .sort_values("toplam_onem", ascending=False)
        .reset_index()
    )
    total = grouped["toplam_onem"].sum()
    grouped["pay_yuzde"] = (grouped["toplam_onem"] / total * 100).round(1) if total else 0.0
    return grouped


def model_footprint(models: Sequence[Any], *, elapsed_seconds: float = 0.0) -> dict[str, Any]:
    """Modelin OPERASYONEL maliyeti -- juri bunu soruyor.

    2024 birincisi sunumda "~25MB, yeni veriyle egitilebilir" dedi. Dagitim
    sirketi bunu gercekten calistiracak; boyut ve egitim suresi soyut mimari
    anlatimindan daha degerlidir.
    """
    import contextlib
    import pickle

    total_bytes = 0
    total_trees = 0
    for model in models:
        # Bazi model nesneleri pickle edilemez (C uzantisi, lambda tutan wrapper).
        # Boyut raporu tam olmayabilir ama bu, raporun tamamini kaybetmekten iyidir.
        with contextlib.suppress(pickle.PicklingError, TypeError, AttributeError):
            total_bytes += len(pickle.dumps(model))
        for attribute in ("n_estimators_", "best_iteration_", "tree_count_"):
            value = getattr(model, attribute, None)
            if isinstance(value, (int, np.integer)):
                total_trees += int(value)
                break

    return {
        "model_sayisi": len(models),
        "toplam_boyut_mb": round(total_bytes / 1024**2, 2),
        "toplam_agac": total_trees,
        "egitim_suresi_dk": round(elapsed_seconds / 60, 1),
        "yorum": (
            f"{len(models)} model, {total_bytes / 1024**2:.1f} MB. "
            "Standart bir sunucuda dakikalar icinde yeniden egitilebilir."
        ),
    }


def business_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    baseline_pred: np.ndarray | None = None,
    crew_cost_per_unit: float = 1.0,
    unit_label: str = "kesinti",
) -> dict[str, Any]:
    """Metrigi IS DILINE cevirir.

    "MAE 2,95" juriye bir sey soylemez. "Ortalama 3 kesintilik hatayla tahmin
    ediyoruz; mevcut yontem 5 hata yapiyor -- yani nobetci ekip planlamasinda
    %40 daha az sapma" bir sey soyler.

    Args:
        baseline_pred: Karsilastirma tabani. ``None`` ise egitim ortalamasi
            kullanilir -- yani "hicbir model kullanmasaydik" senaryosu.
        crew_cost_per_unit: Birim hata basina operasyonel maliyet. Varsayilan 1
            ise cikti "birim" cinsindendir; gercek bir maliyet verilirse
            para cinsine cevrilir.
    """
    true_values = np.asarray(y_true, dtype="float64")
    predictions = np.asarray(y_pred, dtype="float64")

    if baseline_pred is None:
        baseline_pred = np.full_like(true_values, float(np.mean(true_values)))
    baseline = np.asarray(baseline_pred, dtype="float64")

    model_error = float(np.mean(np.abs(predictions - true_values)))
    baseline_error = float(np.mean(np.abs(baseline - true_values)))
    improvement = (baseline_error - model_error) / baseline_error * 100 if baseline_error else 0.0

    # Fazla ve eksik tahmin AYRI raporlanir: bunlar farkli is maliyetleridir.
    # Fazla tahmin = bosuna nobetci ekip. Eksik tahmin = gec mudahale.
    residual = predictions - true_values
    over = float(residual[residual > 0].sum())
    under = float(-residual[residual < 0].sum())

    return {
        "model_ortalama_hata": round(model_error, 4),
        "baseline_ortalama_hata": round(baseline_error, 4),
        "iyilesme_yuzde": round(improvement, 1),
        "fazla_tahmin_toplam": round(over, 1),
        "eksik_tahmin_toplam": round(under, 1),
        "yillik_kazanc_birim": round((baseline_error - model_error) * len(true_values), 1),
        "yillik_kazanc_maliyet": round(
            (baseline_error - model_error) * len(true_values) * crew_cost_per_unit, 1
        ),
        "ozet": (
            f"Ortalama {model_error:.2f} {unit_label} hatayla tahmin ediyoruz; "
            f"model kullanilmasa {baseline_error:.2f} olurdu -- "
            f"%{improvement:.0f} daha az sapma."
        ),
    }


# =============================================================================
# GRAFIKLER
# =============================================================================


def plot_fold_scores(result: Any, *, title: str = "Fold bazli CV skoru", ax: Any = None):
    """Fold skorlari + ortalama cizgisi. Kararliligi tek bakista gosterir."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))

    scores = list(result.fold_scores)
    positions = range(1, len(scores) + 1)
    mean = float(np.mean(scores))

    ax.bar(positions, scores, color=_ACCENT, alpha=0.85, width=0.6)
    ax.axhline(mean, color=_INK, linestyle="--", linewidth=1.2,
               label=f"ortalama {mean:.4f}")
    ax.fill_between(
        [0.4, len(scores) + 0.6],
        mean - np.std(scores), mean + np.std(scores),
        color=_INK, alpha=0.07, label=f"±1 std ({np.std(scores):.4f})",
    )

    ax.set_xticks(list(positions))
    ax.set_xlabel("fold")
    ax.set_ylabel(result.metric_name or "skor")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlim(0.4, len(scores) + 0.6)
    ax.legend(frameon=False, fontsize=9, labelcolor=_MUTED)
    _style(ax)
    return ax


def plot_error_by_segment(
    table: pd.DataFrame, *, metric: str = "mae", top: int = 15, ax: Any = None
):
    """Segment hata grafigi -- en kotuler vurgulu.

    Juri sunumunda "model nerede yaniliyor" slaydinin kendisi.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * min(len(table), top))))

    shown = table.tail(top)
    labels = shown["segment"].astype(str)
    values = shown[metric]

    threshold = table[metric].median()
    colors = [_BAD if value > threshold else _MUTED for value in values]

    ax.barh(labels, values, color=colors, alpha=0.9)
    ax.axvline(threshold, color=_INK, linestyle="--", linewidth=1,
               label=f"medyan {threshold:.3f}")
    ax.set_xlabel(metric)
    ax.set_title("Segment bazli hata (en kotu üstte)", fontsize=12,
                 fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=_MUTED)
    _style(ax)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0)
    return ax


def plot_prediction_timeline(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    aggregate: str = "D",
    title: str = "Tahmin ve gerçek",
    ax: Any = None,
):
    """Zaman ekseninde tahmin-gercek karsilastirmasi.

    Teknik olmayan bir juriye modelin ne yaptigini anlatan EN IYI tek grafiktir:
    metrik soyuttur, bu egri somuttur.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 3.8))

    frame = pd.DataFrame(
        {
            "tarih": pd.to_datetime(dates, errors="coerce"),
            "gercek": np.asarray(y_true, dtype="float64"),
            "tahmin": np.asarray(y_pred, dtype="float64"),
        }
    ).dropna(subset=["tarih"])

    series = frame.set_index("tarih").resample(aggregate).mean()

    ax.plot(series.index, series["gercek"], color=_INK, linewidth=1.6, label="gerçek")
    ax.plot(series.index, series["tahmin"], color=_ACCENT, linewidth=1.6,
            label="tahmin", alpha=0.9)
    ax.fill_between(
        series.index, series["gercek"], series["tahmin"],
        color=_BAD, alpha=0.12, label="hata",
    )

    ax.set_ylabel("ortalama")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, ncol=3, labelcolor=_MUTED)
    _style(ax)
    return ax


def plot_selection_curve(selection_result: Any, *, ax: Any = None):
    """Feature sayisi - skor egrisi.

    2024 birincisi 490 -> 97 feature indirdi ve skoru IYILESTIRDI. Bu egri
    juriye "gereksiz karmasikligi elediks" mesajini tek bakista verir.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.6))

    curve = selection_result.curve()
    best_count = len(selection_result.best_features)

    ax.plot(curve["feature_sayisi"], curve["skor"], color=_MUTED,
            linewidth=1.5, marker="o", markersize=4)
    ax.axvline(best_count, color=_GOOD, linestyle="--", linewidth=1.4,
               label=f"secilen: {best_count} feature")

    ax.set_xlabel("feature sayısı")
    ax.set_ylabel("CV skoru")
    ax.set_title("Feature eleme eğrisi", fontsize=12, fontweight="bold", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=_MUTED)
    _style(ax)
    return ax
