"""Hava ve mekansal feature testleri.

Bir kismi GERCEK indirilmis Open-Meteo verisi uzerinde calisir (data/external).
Veri yoksa o testler atlanir -- CI'da veri olmayabilir ama yerelde varsa
gercek veriyle dogrulanmis olur.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridup.features.spatial import (
    add_neighbour_feature_mean,
    add_neighbour_target_lag,
    haversine_matrix,
    nearest_neighbours,
)
from gridup.features.weather import (
    NEUTRAL_TEMPERATURE_C,
    add_physical_derivatives,
    add_regional_aggregates,
    add_weather_accumulators,
    aggregate_hourly_to_daily,
    circular_mean,
)

WEATHER_PATH = Path(__file__).resolve().parents[1] / "data" / "external" / "hava_gunluk.parquet"


@pytest.fixture(scope="module")
def real_weather():
    if not WEATHER_PATH.exists():
        pytest.skip("Gercek hava verisi yok: scripts/fetch_weather.py calistir")
    return pd.read_parquet(WEATHER_PATH)


class TestCircularMean:
    def test_wraps_around_zero(self):
        """350 ve 10 derecenin ortalamasi 180 DEGIL 0'dir."""
        assert circular_mean(np.array([350.0, 10.0])) == pytest.approx(0.0, abs=1e-6)

    def test_arithmetic_mean_would_be_wrong(self):
        values = np.array([350.0, 10.0])
        assert np.mean(values) == pytest.approx(180.0)
        assert abs(circular_mean(values) - 180.0) > 100

    def test_simple_case_matches_arithmetic(self):
        assert circular_mean(np.array([80.0, 100.0])) == pytest.approx(90.0, abs=1e-6)

    def test_all_nan_returns_nan(self):
        assert np.isnan(circular_mean(np.array([np.nan, np.nan])))


class TestHourlyAggregation:
    @pytest.fixture
    def hourly(self):
        times = pd.date_range("2024-01-01", periods=48, freq="h")
        return pd.DataFrame(
            {
                "zaman": times,
                "konum_key": "izmir",
                "sicaklik": np.concatenate([np.arange(24.0), np.arange(24.0) + 10]),
                "ruzgar_yonu": np.full(48, 45.0),
            }
        )

    def test_produces_one_row_per_day(self, hourly):
        daily = aggregate_hourly_to_daily(
            hourly, time_column="zaman", group_columns=["konum_key"],
            value_columns=["sicaklik"],
        )
        assert len(daily) == 2

    def test_keeps_quantiles_not_just_mean(self, hourly):
        daily = aggregate_hourly_to_daily(
            hourly, time_column="zaman", group_columns=["konum_key"],
            value_columns=["sicaklik"], quantiles=(0.1, 0.9),
        )
        assert "sicaklik_q10" in daily.columns
        assert "sicaklik_q90" in daily.columns
        assert "sicaklik_max" in daily.columns
        # Tepe deger ortalamadan belirgin yuksek olmali -- ortalama bunu siler.
        assert (daily["sicaklik_max"] > daily["sicaklik_mean"]).all()

    def test_direction_columns_become_sin_cos(self, hourly):
        daily = aggregate_hourly_to_daily(
            hourly, time_column="zaman", group_columns=["konum_key"],
            value_columns=["sicaklik"], direction_columns=["ruzgar_yonu"],
        )
        assert "ruzgar_yonu_sin" in daily.columns
        assert "ruzgar_yonu_cos" in daily.columns
        assert daily["ruzgar_yonu_sin"].iloc[0] == pytest.approx(np.sin(np.deg2rad(45)), abs=1e-5)


class TestRegionalAggregates:
    def test_regional_max_exceeds_local_when_neighbour_is_stormy(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.to_datetime(["2024-01-01"] * 3),
                "konum_key": ["izmir", "mugla", "aydin"],
                "ruzgar": [10.0, 90.0, 12.0],
            }
        )

        result = add_regional_aggregates(
            frame, time_column="tarih", value_columns=["ruzgar"], quantiles=(0.5,)
        )

        izmir = result[result["konum_key"] == "izmir"].iloc[0]
        assert izmir["bolge_ruzgar_max"] == 90.0
        # Yerel deger bolgenin cok altinda -> negatif sapma
        assert izmir["ruzgar_bolge_fark"] < 0

    def test_missing_column_raises(self):
        frame = pd.DataFrame({"tarih": pd.to_datetime(["2024-01-01"]), "a": [1.0]})
        with pytest.raises(KeyError):
            add_regional_aggregates(frame, time_column="tarih", value_columns=["yok"])


class TestPhysicalDerivatives:
    @pytest.fixture
    def daily(self):
        dates = pd.date_range("2024-06-01", periods=20, freq="D")
        return pd.DataFrame(
            {
                "tarih": dates,
                "konum_key": "izmir",
                "sicaklik_ort": np.full(20, 28.0),
                "sicaklik_max": np.full(20, 34.0),
                "sicaklik_min": np.full(20, 24.0),   # tropik gece (>22)
                "yagis_toplam": np.concatenate([np.zeros(15), [25.0], np.zeros(4)]),
                "firtina_max": np.full(20, 40.0),
            }
        )

    def test_cooling_degree_days_when_hot(self, daily):
        result = add_physical_derivatives(
            daily, group_columns=["konum_key"], time_column="tarih"
        )
        expected = 28.0 - NEUTRAL_TEMPERATURE_C
        assert result["sogutma_derece_gun"].iloc[0] == pytest.approx(expected)
        assert result["isitma_derece_gun"].iloc[0] == 0.0

    def test_consecutive_tropical_nights_accumulate(self, daily):
        result = add_physical_derivatives(
            daily, group_columns=["konum_key"], time_column="tarih"
        )
        assert result["ardisik_sicak_gece"].iloc[0] == 1
        assert result["ardisik_sicak_gece"].iloc[19] == 20

    def test_drought_counter_resets_on_rain(self, daily):
        result = add_physical_derivatives(
            daily, group_columns=["konum_key"], time_column="tarih"
        )
        assert result["kuraklik_gunu"].iloc[14] == 15   # 15 gun yagissiz
        assert result["kuraklik_gunu"].iloc[15] == 0    # yagmur gunu

    def test_first_rain_after_drought_is_flagged_once(self, daily):
        result = add_physical_derivatives(
            daily, group_columns=["konum_key"], time_column="tarih", drought_days=10
        )
        flags = result["kuraklik_sonrasi_ilk_yagmur"]
        assert flags.sum() == 1
        assert flags.iloc[15] == 1

    def test_wet_wind_index_exists_when_both_present(self, daily):
        result = add_physical_derivatives(
            daily, group_columns=["konum_key"], time_column="tarih"
        )
        assert "islak_ruzgar" in result.columns

    def test_input_not_mutated(self, daily):
        before = daily.copy()
        add_physical_derivatives(daily, group_columns=["konum_key"], time_column="tarih")
        pd.testing.assert_frame_equal(daily, before)


class TestWeatherAccumulators:
    def test_backward_window_uses_past_only(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=5, freq="D"),
                "konum_key": "izmir",
                "ruzgar": [10.0, 50.0, 20.0, 15.0, 12.0],
            }
        )
        result = add_weather_accumulators(
            frame, group_columns=["konum_key"], time_column="tarih",
            value_columns=["ruzgar"], windows=(3,), horizon=0,
        )
        # horizon=0 -> bugun dahil; 2. satirin 3'luk max'i {10, 50, 20} = 50
        assert result["ruzgar_geri3_max"].iloc[2] == pytest.approx(50.0)

    def test_lead_window_looks_forward(self):
        frame = pd.DataFrame(
            {
                "tarih": pd.date_range("2024-01-01", periods=5, freq="D"),
                "konum_key": "izmir",
                "ruzgar": [10.0, 12.0, 15.0, 80.0, 11.0],
            }
        )
        result = add_weather_accumulators(
            frame, group_columns=["konum_key"], time_column="tarih",
            value_columns=["ruzgar"], windows=(), lead_windows=(3,), horizon=0,
        )
        # 1. satirdan itibaren 3 gunluk ileri max: {12, 15, 80} = 80
        assert result["ruzgar_ileri3_max"].iloc[1] == pytest.approx(80.0)


class TestSpatial:
    @pytest.fixture
    def coordinates(self):
        return pd.DataFrame(
            {
                "konum_key": ["izmir", "manisa", "aydin", "mugla"],
                "lat": [38.4237, 38.6191, 37.8560, 37.2153],
                "lon": [27.1428, 27.4289, 27.8416, 28.3636],
            }
        )

    def test_haversine_beats_euclidean_at_turkish_latitudes(self, coordinates):
        distances = haversine_matrix(
            coordinates["lat"].to_numpy(), coordinates["lon"].to_numpy()
        )
        # Izmir-Manisa gercekte ~33 km
        assert 25 < distances[0, 1] < 45
        assert distances[0, 0] == 0.0
        # Simetrik olmali
        np.testing.assert_allclose(distances, distances.T, atol=1e-9)

    def test_nearest_neighbours_excludes_self(self, coordinates):
        neighbours = nearest_neighbours(coordinates, key_column="konum_key", k=2)
        assert not (neighbours["konum_key"] == neighbours["komsu"]).any()

    def test_manisa_is_izmirs_closest(self, coordinates):
        neighbours = nearest_neighbours(coordinates, key_column="konum_key", k=1)
        izmir = neighbours[neighbours["konum_key"] == "izmir"].iloc[0]
        assert izmir["komsu"] == "manisa"

    def test_distance_cap_drops_far_neighbours(self, coordinates):
        neighbours = nearest_neighbours(
            coordinates, key_column="konum_key", k=3, max_distance_km=40
        )
        assert (neighbours["mesafe_km"] <= 40).all()

    def test_duplicate_key_raises(self):
        frame = pd.DataFrame(
            {"konum_key": ["a", "a"], "lat": [1.0, 2.0], "lon": [1.0, 2.0]}
        )
        with pytest.raises(ValueError, match="tekrarlayan"):
            nearest_neighbours(frame, key_column="konum_key")

    def test_neighbour_target_lag_requires_positive_horizon(self, coordinates):
        neighbours = nearest_neighbours(coordinates, key_column="konum_key", k=1)
        frame = pd.DataFrame(
            {
                "konum_key": ["izmir"],
                "tarih": pd.to_datetime(["2024-01-01"]),
                "kesinti": [1.0],
            }
        )
        with pytest.raises(ValueError, match="sizinti"):
            add_neighbour_target_lag(
                frame, neighbours, key_column="konum_key", time_column="tarih",
                target_column="kesinti", horizon=0,
            )

    def test_neighbour_target_lag_uses_neighbours_past(self, coordinates):
        neighbours = nearest_neighbours(coordinates, key_column="konum_key", k=1)
        dates = pd.date_range("2024-01-01", periods=4, freq="D")
        frame = pd.concat(
            [
                pd.DataFrame({"konum_key": key, "tarih": dates, "kesinti": values})
                for key, values in [
                    ("izmir", [0.0, 0.0, 0.0, 0.0]),
                    ("manisa", [7.0, 8.0, 9.0, 10.0]),
                ]
            ],
            ignore_index=True,
        )

        result = add_neighbour_target_lag(
            frame, neighbours, key_column="konum_key", time_column="tarih",
            target_column="kesinti", horizon=1, statistics=("max",),
        )

        column = "komsu_kesinti_ufuk1_max"
        izmir = result[result["konum_key"] == "izmir"].sort_values("tarih")
        # Izmir'in komsusu Manisa; 2. gunde Manisa'nin 1 gun oncesi = 7.0
        assert izmir[column].iloc[1] == pytest.approx(7.0)
        assert pd.isna(izmir[column].iloc[0])

    def test_neighbour_feature_mean_needs_no_shift(self, coordinates):
        neighbours = nearest_neighbours(coordinates, key_column="konum_key", k=1)
        frame = pd.DataFrame(
            {
                "konum_key": ["izmir", "manisa"],
                "tarih": pd.to_datetime(["2024-01-01"] * 2),
                "ruzgar": [10.0, 90.0],
            }
        )

        result = add_neighbour_feature_mean(
            frame, neighbours, key_column="konum_key", time_column="tarih",
            value_columns=["ruzgar"], statistics=("max",),
        )

        izmir = result[result["konum_key"] == "izmir"].iloc[0]
        assert izmir["komsu_ruzgar_max"] == pytest.approx(90.0)


class TestOnRealWeatherData:
    """Gercek Open-Meteo verisi uzerinde -- sentetik degil."""

    def test_dataset_covers_all_five_provinces(self, real_weather):
        provinces = {"İzmir", "Manisa", "Aydın", "Denizli", "Muğla"}
        assert provinces <= set(real_weather["konum"].unique())

    def test_no_missing_values(self, real_weather):
        assert int(real_weather.isna().sum().sum()) == 0

    def test_regional_aggregates_work_on_real_data(self, real_weather):
        subset = real_weather[real_weather["tarih"] >= "2025-01-01"]
        result = add_regional_aggregates(
            subset, time_column="tarih",
            value_columns=["ruzgar_max", "sicaklik_ort"], quantiles=(0.9,),
        )
        assert "bolge_ruzgar_max_q90" in result.columns
        assert len(result) == len(subset)
        assert result["bolge_ruzgar_max_max"].notna().all()

    def test_physical_derivatives_work_on_real_data(self, real_weather):
        subset = real_weather[real_weather["tarih"] >= "2025-01-01"].copy()
        result = add_physical_derivatives(
            subset, group_columns=["konum_key"], time_column="tarih"
        )
        # Ege yazi: tropik gece ve kuraklik serisi gercekten olusmali
        assert result["ardisik_sicak_gece"].max() > 5
        assert result["kuraklik_gunu"].max() > 10
        assert result["sogutma_derece_gun"].max() > 5

    def test_summer_is_hotter_than_winter(self, real_weather):
        """Veri sagligi: mevsimsellik gercekten var mi?"""
        by_month = real_weather.groupby(real_weather["tarih"].dt.month)["sicaklik_ort"].mean()
        assert by_month.loc[7] > by_month.loc[1] + 10
