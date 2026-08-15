"""Grid Up Datathon - yarisma pipeline'i.

Coderspace x GDZ Elektrik x ADM Elektrik, 21 Agustos - 1 Eylul 2026.

VERI GELDIGINDE ILK 30 DAKIKA
-----------------------------
::

    from gridup import profile, read_any, suggest_scheme, leakage_report

    train = read_any("data/raw/train.csv")     # kodlama/ayirici otomatik
    test  = read_any("data/raw/test.csv")

    print(profile(train, test, target="HEDEF").report())        # 1. ne var elimizde
    print(suggest_scheme(train, target="HEDEF", test=test))     # 2. hangi CV semasi
    print(leakage_report(train, "HEDEF", test=test))            # 3. sizinti var mi

``suggest_scheme``a ``test`` vermeyi ATLAMA: test'te bulunmayan kolonlar grup
adayi olmaktan cikar. Aksi halde hedeften turemis bir kolona gore gruplama
onerilebilir ve CV anlamsiz olur (prova verisinde olculdu).

Bu uc cikti, sonraki 12 gunun her kararini belirler. Once bunlari oku.

TASARIM SOZLESMESI
------------------
* Feature fonksiyonlari girdi DataFrame'ini ASLA degistirmez, yeni frame dondurur
* Hedef kullanan her kodlama fold-disi (out-of-fold) calisir -- sizinti imkansiz
* Kayan pencereler mevcut satiri DISLAR (shift(1)) -- hedef sizintisi imkansiz
* Her sabit ``config.py`` icinde yasar -- notebook'ta sihirli sayi yok
"""

from __future__ import annotations

__version__ = "0.1.0"

from .ablation import FeatureGroup, ablation_ensemble, leave_one_group_out
from .compat import environment_report, reduce_memory
from .config import CONFIG, PATHS, CompetitionConfig, Paths, set_global_seed
from .io_utils import read_any, read_table, sniff_dialect, to_parquet_cache
from .metrics import (
    get_metric,
    optimize_threshold,
    postprocess_predictions,
    rmse,
    rmsle,
)
from .models import COUNT_OBJECTIVES, CVResult, cross_validate, starter_params
from .neural import NeuralConfig, neural_cross_validate
from .panel import build_panel, panel_coverage
from .profiling import profile, quick_look
from .refit import (
    estimate_full_data_rounds,
    extract_best_iterations,
    fold_train_fraction,
    multi_seed_refit,
)
from .submission import validate_submission, write_submission
from .tuning import TuningResult, tune_with_optuna
from .turkish import diagnose_join, join_key, tr_lower, tr_sorted, tr_upper
from .two_stage import (
    conditional_quantile_from_hurdle,
    fit_conditional_quantile_ladder,
    fit_quantile_ladder,
    fit_two_stage,
    mae_optimal_quantile,
    tune_threshold,
    zero_baseline_score,
)
from .validation import (
    adversarial_validation,
    build_splitter,
    leakage_report,
    purged_time_series_split,
    suggest_scheme,
)
from .zoo import ZooEntry, make_model_zoo, sweep_count_objectives

__all__ = [
    # konfig
    "CONFIG", "PATHS", "CompetitionConfig", "Paths", "set_global_seed",
    # okuma
    "read_any", "read_table", "sniff_dialect", "to_parquet_cache",
    # kesif
    "profile", "quick_look", "environment_report", "reduce_memory",
    # turkce
    "tr_lower", "tr_upper", "join_key", "tr_sorted", "diagnose_join",
    # dogrulama
    "suggest_scheme", "build_splitter", "leakage_report",
    "adversarial_validation", "purged_time_series_split",
    # panel
    "build_panel", "panel_coverage",
    # metrik
    "rmse", "rmsle", "get_metric", "optimize_threshold", "postprocess_predictions",
    # model
    "cross_validate", "CVResult", "starter_params", "COUNT_OBJECTIVES",
    # tam veri yeniden egitim
    "multi_seed_refit", "estimate_full_data_rounds", "extract_best_iterations",
    "fold_train_fraction",
    # iki asamali (sifir-siskin)
    "fit_two_stage", "tune_threshold", "zero_baseline_score",
    "mae_optimal_quantile", "fit_quantile_ladder", "conditional_quantile_from_hurdle",
    "fit_conditional_quantile_ladder",
    # model zoo ve arama
    "make_model_zoo", "sweep_count_objectives", "ZooEntry",
    "tune_with_optuna", "TuningResult",
    # sinir agi (harman cesitliligi)
    "neural_cross_validate", "NeuralConfig",
    # feature grubu ablasyonu ve dayaniklilik harmani
    "FeatureGroup", "ablation_ensemble", "leave_one_group_out",
    # submission
    "write_submission", "validate_submission",
]
