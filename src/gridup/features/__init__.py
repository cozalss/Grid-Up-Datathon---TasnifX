"""Feature muhendisligi modulleri.

Tum fonksiyonlar AYNI SOZLESMEYE uyar:
  * girdi DataFrame'i asla degistirmez
  * yeni bir DataFrame dondurur
  * uretilen kolon adlari deterministik ve Turkce

Bu sozlesme sayesinde fonksiyonlar zincirlenebilir::

    features = (
        raw
        .pipe(add_calendar_features, time_column="tarih")
        .pipe(add_turkish_holiday_features, time_column="tarih")
        .pipe(add_lag_features, "tuketim", [1, 7], time_column="tarih",
              group_columns=["trafo_id"])
    )
"""

from .aggregate import add_group_statistics, add_ratio_features, add_target_free_aggregates
from .categorical import (
    add_combination_features,
    add_count_encoding,
    add_frequency_encoding,
    oof_target_encode,
    reduce_rare_categories,
)
from .temporal import (
    add_calendar_features,
    add_cyclical_features,
    add_expanding_features,
    add_lag_features,
    add_rolling_features,
    add_turkish_holiday_features,
    shared_origin,
)

__all__ = [
    "add_calendar_features",
    "add_cyclical_features",
    "add_turkish_holiday_features",
    "add_lag_features",
    "add_rolling_features",
    "add_expanding_features",
    "shared_origin",
    "add_frequency_encoding",
    "add_count_encoding",
    "oof_target_encode",
    "add_combination_features",
    "reduce_rare_categories",
    "add_group_statistics",
    "add_ratio_features",
    "add_target_free_aggregates",
]
