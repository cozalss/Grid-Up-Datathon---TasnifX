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
from .outage_reason import add_reason_features, classify_reason, reason_family_report
from .school import add_school_calendar_features, school_calendar
from .spatial import (
    add_neighbour_feature_mean,
    add_neighbour_target_lag,
    haversine_matrix,
    nearest_neighbours,
)
from .temporal import (
    add_calendar_features,
    add_cyclical_features,
    add_expanding_features,
    add_lag_features,
    add_mass_event_features,
    add_previous_month_features,
    add_rolling_features,
    add_turkish_holiday_features,
    add_upcoming_holiday_features,
    shared_origin,
)
from .weather import (
    add_consecutive_extreme_days,
    add_physical_derivatives,
    add_precip_anomaly,
    add_regional_aggregates,
    add_weather_accumulators,
    aggregate_hourly_to_daily,
    circular_mean,
)

__all__ = [
    "add_calendar_features",
    "add_cyclical_features",
    "add_turkish_holiday_features",
    "add_lag_features",
    "add_rolling_features",
    "add_expanding_features",
    "add_mass_event_features",
    "add_previous_month_features",
    "add_upcoming_holiday_features",
    "shared_origin",
    "add_frequency_encoding",
    "add_count_encoding",
    "oof_target_encode",
    "add_combination_features",
    "reduce_rare_categories",
    "add_group_statistics",
    "add_ratio_features",
    "add_target_free_aggregates",
    # ariza sebebi
    "classify_reason",
    "add_reason_features",
    "reason_family_report",
    # okul takvimi
    "school_calendar",
    "add_school_calendar_features",
    # mekansal
    "nearest_neighbours",
    "add_neighbour_target_lag",
    "add_neighbour_feature_mean",
    "haversine_matrix",
    # hava
    "aggregate_hourly_to_daily",
    "add_regional_aggregates",
    "add_physical_derivatives",
    "add_weather_accumulators",
    "add_consecutive_extreme_days",
    "add_precip_anomaly",
    "circular_mean",
]
