"""Grid Up Datathon - yarisma pipeline'i.

Coderspace x GDZ Elektrik x ADM Elektrik, 21 Agustos - 1 Eylul 2026.

VERI GELDIGINDE ILK 30 DAKIKA
-----------------------------
::

    from gridup import profile, read_any, suggest_scheme, leakage_report

    train = read_any("data/raw/train.csv")     # kodlama/ayirici otomatik
    test  = read_any("data/raw/test.csv")

    print(profile(train, test, target="HEDEF").report())   # 1. ne var elimizde
    print(suggest_scheme(train, target="HEDEF"))           # 2. hangi CV semasi
    print(leakage_report(train, "HEDEF", test=test))       # 3. sizinti var mi

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
from .panel import build_panel, panel_coverage
from .profiling import profile, quick_look
from .submission import validate_submission, write_submission
from .turkish import diagnose_join, join_key, tr_lower, tr_sorted, tr_upper
from .validation import (
    adversarial_validation,
    build_splitter,
    leakage_report,
    purged_time_series_split,
    suggest_scheme,
)

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
    # submission
    "write_submission", "validate_submission",
]
