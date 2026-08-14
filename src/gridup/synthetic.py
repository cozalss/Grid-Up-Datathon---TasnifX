"""Sentetik elektrik dagitim sebekesi veri seti ureteci.

NEDEN BU MODUL VAR
------------------
Gercek veri 21 Agustos'ta geliyor. Ama pipeline'in o gun CALISIYOR olmasi
gerekiyor -- veri geldigi gun hata ayiklamak, 12 gunluk yarismanin en degerli
saatlerini yakar.

Bu modul, gercek verinin muhtemel SEKLINI taklit eden bir veri seti uretir:
Turkce kolon adlari, ``İ``/``I`` iceren il-ilce adlari, zaman ekseni, tekrarlayan
varlik (trafo), kategorik kolonlar, mevsimsel sinyal, hava durumu etkisi ve
train/test zaman ayrimi.

Boylece pipeline'in her parcasi -- kodlama tespiti, TR join anahtarlari, sizinti
kontrolu, CV semasi secimi, feature uretimi, egitim, submission -- gercek veriden
ONCE uctan uca test edilir.

BU VERI GERCEK DEGILDIR. Yalnizca pipeline'i dogrulamak icindir. Uretilen
sayilar gercek GDZ/ADM degerlerini temsil etmez.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SyntheticSpec", "make_distribution_dataset", "GDZ_ADM_DISTRICTS"]

# GDZ (Izmir, Manisa) ve ADM (Aydin, Denizli, Mugla) bolgesinden ornek ilceler.
# Bilerek ``İ``, ``I``, ``ı`` iceren adlar secildi -- Turkce join tuzaklarini
# test etmek icin.
GDZ_ADM_DISTRICTS: dict[str, tuple[str, ...]] = {
    "İzmir": ("Konak", "Karşıyaka", "Bornova", "Buca", "Çeşme", "Ödemiş", "Bergama"),
    "Manisa": ("Şehzadeler", "Yunusemre", "Akhisar", "Salihli", "Turgutlu"),
    "Aydın": ("Efeler", "Kuşadası", "Didim", "Nazilli", "Söke"),
    "Denizli": ("Pamukkale", "Merkezefendi", "Çivril", "Sarayköy"),
    "Muğla": ("Bodrum", "Fethiye", "Marmaris", "Milas", "Datça"),
}

_FAULT_TYPES = (
    "Kablo Arızası", "Trafo Arızası", "Ağaç Teması", "Aşırı Yük", "Yıldırım", "3. Şahıs",
)
_CUSTOMER_GROUPS = ("Mesken", "Ticarethane", "Sanayi", "Tarımsal Sulama", "Aydınlatma")
_VOLTAGE_LEVELS = ("OG", "AG")

# Turizm yogun ilceler: yaz aylarinda nufus ve yuk patlar (Ege'ye ozgu sinyal).
_TOURISM_HEAVY = {"Çeşme", "Kuşadası", "Didim", "Bodrum", "Fethiye", "Marmaris", "Datça"}
# Tarimsal sulama yogun ilceler: yaz sulama sezonunda pompa yuku artar.
_AGRICULTURE_HEAVY = {"Akhisar", "Salihli", "Turgutlu", "Nazilli", "Söke", "Çivril", "Sarayköy"}


@dataclass(frozen=True)
class SyntheticSpec:
    """Sentetik veri seti parametreleri."""

    n_transformers: int = 300
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"
    freq: str = "D"
    seed: int = 42
    missing_rate: float = 0.03
    test_months: int = 3
    task: str = "regression"  # "regression" (kesinti suresi) | "binary" (ariza var mi)


def _build_transformer_registry(spec: SyntheticSpec, rng: np.random.Generator) -> pd.DataFrame:
    """Trafo envanteri: her trafonun sabit ozellikleri."""
    provinces = list(GDZ_ADM_DISTRICTS)
    rows = []

    for index in range(spec.n_transformers):
        province = provinces[rng.integers(len(provinces))]
        districts = GDZ_ADM_DISTRICTS[province]
        district = districts[rng.integers(len(districts))]

        rows.append(
            {
                "TRAFO_ID": f"TR{index:05d}",
                "İL": province,
                "İLÇE": district,
                "GERİLİM_SEVİYESİ": _VOLTAGE_LEVELS[rng.integers(len(_VOLTAGE_LEVELS))],
                "ABONE_GRUBU": _CUSTOMER_GROUPS[rng.integers(len(_CUSTOMER_GROUPS))],
                "KURULU_GÜÇ_KVA": float(rng.choice([50, 100, 160, 250, 400, 630, 1000])),
                "ABONE_SAYISI": int(rng.integers(15, 900)),
                "TESİS_YILI": int(rng.integers(1985, 2024)),
                "FİDER_NO": f"F{rng.integers(1, 60):03d}",
            }
        )

    return pd.DataFrame(rows)


def _seasonal_load(dates: pd.DatetimeIndex, district: str, rng: np.random.Generator) -> np.ndarray:
    """Ilce profiline gore mevsimsel yuk egrisi uretir.

    Uc bilesen: yillik mevsimsellik (yaz kliması + kis isitmasi), haftalik
    dongu (hafta sonu dususu) ve ilceye ozgu yaz carpani.
    """
    day_of_year = dates.dayofyear.to_numpy()
    day_of_week = dates.dayofweek.to_numpy()

    # Cift tepeli yillik egri: yaz (kliması) ve kis (isitma) zirveleri.
    summer_peak = np.exp(-(((day_of_year - 200) / 45) ** 2))
    winter_peak = 0.7 * np.exp(-(((day_of_year - 15) / 40) ** 2))
    winter_peak += 0.7 * np.exp(-(((day_of_year - 350) / 40) ** 2))

    seasonal = 1.0 + 0.6 * summer_peak + 0.4 * winter_peak

    if district in _TOURISM_HEAVY:
        seasonal = seasonal + 1.4 * summer_peak  # turizm nufus patlamasi
    if district in _AGRICULTURE_HEAVY:
        # Sulama sezonu: Mayis-Eylul arasi pompa yuku.
        irrigation = ((day_of_year >= 120) & (day_of_year <= 260)).astype(float)
        seasonal = seasonal + 0.9 * irrigation

    weekly = np.where(day_of_week >= 5, 0.85, 1.0)
    noise = rng.normal(1.0, 0.08, size=len(dates))

    return seasonal * weekly * noise


def _synthetic_weather(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Ege iklimine benzer gunluk hava verisi."""
    day_of_year = dates.dayofyear.to_numpy()

    temperature = (
        17.0
        + 12.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365)
        + rng.normal(0, 3.0, size=len(dates))
    )
    # Yagis kis aylarinda yogun (Akdeniz iklimi).
    rain_intensity = 0.5 + 0.5 * np.cos(2 * np.pi * day_of_year / 365)
    precipitation = rng.gamma(shape=0.6, scale=6.0, size=len(dates)) * rain_intensity
    wind = np.abs(rng.gamma(shape=2.0, scale=4.0, size=len(dates)))
    # Yildirim yaz sonu firtinalarinda; ariza ile guclu iliskisi olacak.
    storm = (wind > 22) & (precipitation > 12)

    return pd.DataFrame(
        {
            "SICAKLIK_C": temperature.round(1),
            "YAĞIŞ_MM": precipitation.round(1),
            "RÜZGAR_KMH": wind.round(1),
            "FIRTINA_VAR": storm.astype(int),
        },
        index=dates,
    )


def make_distribution_dataset(
    spec: SyntheticSpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Sentetik dagitim sebekesi veri seti uretir.

    Returns:
        ``(train, test, cozum)`` -- ``cozum`` test'in gercek hedef degerlerini
        icerir; yerel olarak "private leaderboard" simule etmeye yarar.

    Uretilen veri seti bilerek su zorluklari icerir:
      * Turkce kolon ve deger adlari (``İL``, ``İLÇE``, ``Ağaç Teması``)
      * Zaman ekseni + zaman bazli train/test ayrimi (rastgele CV sizdirir)
      * Tekrarlayan varlik (``TRAFO_ID``) -> grup sizintisi riski
      * Eksik degerler
      * Mevsimsellik, hava durumu etkisi, ilce profili etkisi
      * Saga carpik hedef dagilimi (kesinti suresi)
    """
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)

    registry = _build_transformer_registry(spec, rng)
    dates = pd.date_range(spec.start_date, spec.end_date, freq=spec.freq)
    weather = _synthetic_weather(dates, rng)

    frames = []
    for _, transformer in registry.iterrows():
        district = transformer["İLÇE"]
        load_curve = _seasonal_load(dates, district, rng)

        base_load = transformer["ABONE_SAYISI"] * 0.9 + transformer["KURULU_GÜÇ_KVA"] * 0.35
        consumption = base_load * load_curve

        # Yaslanan ekipman daha cok arizalanir -- gercek sebekelerde boyledir.
        age = 2026 - transformer["TESİS_YILI"]
        age_factor = 1.0 + age / 45.0

        # Ariza olasiligi: yuk + yas + firtina birlesimi.
        load_stress = consumption / (transformer["KURULU_GÜÇ_KVA"] * 2.2 + 1)
        storm_effect = weather["FIRTINA_VAR"].to_numpy() * 2.4
        wind_effect = weather["RÜZGAR_KMH"].to_numpy() / 40.0

        logit = -4.2 + 0.9 * load_stress + 0.6 * age_factor + storm_effect + wind_effect
        fault_probability = 1.0 / (1.0 + np.exp(-logit))
        has_fault = rng.random(len(dates)) < fault_probability

        # Kesinti suresi: saga carpik (log-normal), arizasiz gunlerde 0.
        duration = np.where(
            has_fault,
            rng.lognormal(mean=3.6, sigma=0.85, size=len(dates)) * (1 + 0.35 * storm_effect),
            0.0,
        )

        fault_type = np.where(
            has_fault,
            rng.choice(_FAULT_TYPES, size=len(dates)),
            None,
        )

        frames.append(
            pd.DataFrame(
                {
                    "TARİH": dates,
                    "TRAFO_ID": transformer["TRAFO_ID"],
                    "İL": transformer["İL"],
                    "İLÇE": district,
                    "FİDER_NO": transformer["FİDER_NO"],
                    "GERİLİM_SEVİYESİ": transformer["GERİLİM_SEVİYESİ"],
                    "ABONE_GRUBU": transformer["ABONE_GRUBU"],
                    "KURULU_GÜÇ_KVA": transformer["KURULU_GÜÇ_KVA"],
                    "ABONE_SAYISI": transformer["ABONE_SAYISI"],
                    "TESİS_YILI": transformer["TESİS_YILI"],
                    "TÜKETİM_KWH": consumption.round(2),
                    "SICAKLIK_C": weather["SICAKLIK_C"].to_numpy(),
                    "YAĞIŞ_MM": weather["YAĞIŞ_MM"].to_numpy(),
                    "RÜZGAR_KMH": weather["RÜZGAR_KMH"].to_numpy(),
                    "ARIZA_TİPİ": fault_type,
                    "KESİNTİ_SÜRESİ_DK": duration.round(1),
                    "ARIZA_VAR_MI": has_fault.astype(int),
                }
            )
        )

    full = pd.concat(frames, ignore_index=True)

    # Gercekci eksik degerler: olcum cihazlari veri kaybeder.
    for column in ("TÜKETİM_KWH", "SICAKLIK_C", "RÜZGAR_KMH"):
        mask = rng.random(len(full)) < spec.missing_rate
        full.loc[mask, column] = np.nan

    full = full.sort_values(["TARİH", "TRAFO_ID"]).reset_index(drop=True)
    full.insert(0, "ID", np.arange(len(full)))

    # Zamana gore train/test ayrimi -- gercek yarismalar boyle boler.
    cutoff = full["TARİH"].max() - pd.DateOffset(months=spec.test_months)
    train = full[full["TARİH"] <= cutoff].reset_index(drop=True)
    test_full = full[full["TARİH"] > cutoff].reset_index(drop=True)

    target_column = "KESİNTİ_SÜRESİ_DK" if spec.task == "regression" else "ARIZA_VAR_MI"
    leak_columns = ["KESİNTİ_SÜRESİ_DK", "ARIZA_VAR_MI", "ARIZA_TİPİ"]

    test = test_full.drop(columns=leak_columns)
    solution = test_full[["ID", target_column]].copy()

    return train, test, solution
