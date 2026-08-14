"""Notebook sablonlarini uretir.

Neden elle .ipynb yazmak yerine bir uretec: JSON'u elle yazmak hataya acik ve
diff'i okunamaz. Bu betik notebook'lari kaynak koddan uretir, boylece sablonlar
sürüm kontrolünde okunabilir kalir.

Calistirma::

    python scripts/build_notebooks.py
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parents[1] / "notebooks"

_KERNEL = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip().splitlines(True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(True),
    }


def write_notebook(cells: list[dict], path: Path) -> None:
    """Notebook'u yazar.

    Hucre ``id``leri DETERMINISTIK atanir (uuid degil): boylece notebook'u
    yeniden uretmek sahte bir diff uretmez ve git gecmisi okunabilir kalir.
    nbformat 4.5+ ``id`` alanini zorunlu kilma yolunda.
    """
    stamped = [{**cell, "id": f"{path.stem}-{index:02d}"} for index, cell in enumerate(cells)]
    notebook = {"cells": stamped, "metadata": _KERNEL, "nbformat": 4, "nbformat_minor": 5}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Yazildi: {path}  ({len(stamped)} hucre)")


# ============================================================================
# 01 - KESIF (EDA)
# ============================================================================

EDA_CELLS = [
    markdown("""
# Grid Up Datathon — 01 · Veri Keşfi

**Takım:** _(takım adını yaz)_ · **Tarih:** _(gün)_

Bu notebook, veri setinin ilk saatinde çalıştırılır. Amacı üç soruyu cevaplamak:

1. **Elimizde ne var?** — kolonlar, tipler, eksikler, boyut
2. **Hangi doğrulama şeması doğru?** — zaman var mı, tekrarlayan varlık var mı
3. **Sızıntı var mı?** — modeli eğitmeden önce bilmemiz gereken tek şey

> Bu üç çıktı, sonraki 12 günün her kararını belirler.
"""),
    code("""
import sys
from pathlib import Path

# Kaggle'da: /kaggle/input/<yarisma>/  · yerelde: data/raw/
IS_KAGGLE = Path("/kaggle/input").exists()
if IS_KAGGLE:
    # DIKKAT: gridup Kaggle imajinda KURULU DEGILDIR. Onceki surumde sys.path
    # yalnizca YERELDE ayarlaniyordu; Kaggle'da 'import gridup' ModuleNotFound
    # veriyordu. Once offline paket dataset'indeki wheel'i kur, o yoksa ham
    # kaynagi sys.path'e ekle.
    import glob
    import subprocess

    _whl = glob.glob("/kaggle/input/*/gridup-*.whl")
    if _whl:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", _whl[0], "-q"],
            check=False,
        )
    else:
        for _src in glob.glob("/kaggle/input/*/src"):
            sys.path.insert(0, _src)
else:
    sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridup import environment_report, profile, read_any, set_global_seed
from gridup.compat import categorical_columns
from gridup.profiling import quick_look
from gridup.turkish import codepoints, has_combining_dot, join_key
from gridup.validation import leakage_report, suggest_scheme

set_global_seed(42)

# Ortamı yazdır — jüri tekrarlanabilirliğe bakıyor, bu ucuz bir puan.
for key, value in environment_report().items():
    print(f"{key:<26} {value}")
"""),
    markdown("""
## 1 · Veriyi oku

`read_any` kodlamayı, ayırıcıyı ve ondalık işaretini **otomatik tespit eder**.
Türk kurumlarından gelen dosyalar `cp1254` + `;` + ondalık `,` olabilir; düz
`pd.read_csv` bunları sessizce bozar.
"""),
    code("""
DATA_DIR = Path("/kaggle/input/GRID-UP-YARISMA-SLUG") if IS_KAGGLE else Path("../data/raw")

train = read_any(DATA_DIR / "train.csv")
test  = read_any(DATA_DIR / "test.csv")

try:
    sample_submission = read_any(DATA_DIR / "sample_submission.csv")
    print("sample_submission kolonları:", list(sample_submission.columns))
except FileNotFoundError:
    sample_submission = None
    print("sample_submission bulunamadı — dosya adlarını kontrol et:")
    print(sorted(p.name for p in DATA_DIR.glob("*")))

print(f"\\ntrain {train.shape}   test {test.shape}")
train.head()
"""),
    markdown("""
## 2 · Otomatik profil

Tek çağrı; elle 2 saat sürecek keşfin yerini alır. Özellikle şunları işaretler:
çarpıklık, sıfır yığılması, ID-benzeri kolonlar, yüksek kardinalite, şema farkı
(= sızıntı adayları) ve **birleşik nokta (U+0307)** taşıyan Türkçe kolonlar.
"""),
    code("""
# TODO: hedef kolon adını veri geldiğinde doldur
TARGET = "HEDEF_KOLON"

dataset_profile = profile(train, test, target=TARGET)
print(dataset_profile.report())
"""),
    code("""
# Kolon bazlı kompakt tablo — hızlı gözden geçirme için
quick_look(train)
"""),
    markdown("""
## 3 · Doğrulama şeması — **yarışmanın kazanıldığı karar**

Yanlış şema iki şekilde öldürür:
- **İyimser CV:** lokal skor yüksek, leaderboard'da çakılıyorsun → sızıntı var
- **Gürültülü CV:** hangi değişikliğin işe yaradığını göremiyorsun → public LB'ye
  göre karar vermeye başlıyorsun → private LB'de çöküyorsun (shakeup)
"""),
    code("""
suggestion = suggest_scheme(train, target=TARGET)
print(suggestion)
"""),
    markdown("""
## 4 · Sızıntı taraması

Modeli eğitmeden **önce** çalıştır. `critical` bulgular varsa dur ve çöz.
"""),
    code("""
TIME_COLUMN = None   # TODO: varsa zaman kolonu adı

findings = leakage_report(train, TARGET, test=test, time_column=TIME_COLUMN)
print(findings["summary"], "\\n")

for severity in ("critical", "warning", "info"):
    for message in findings[severity]:
        print(f"[{severity.upper()}] {message}")
"""),
    markdown("""
## 5 · Hedef dağılımı

Hedefin şekli metrik seçimini ve dönüşüm kararını belirler:
- **Çarpıklık > 2** → `log1p` dönüşümü dene
- **Sıfır yığılması > %40** → iki aşamalı model düşün (önce sıfır mı, sonra miktar)
- **Sınıf dengesizliği** → eşik optimizasyonu şart, 0.5 varsayılanı yanlış
"""),
    code("""
target_values = train[TARGET].dropna()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(target_values, bins=60, color="#4C6EF5", edgecolor="white", linewidth=0.4)
axes[0].set_title("Ham dağılım")
axes[0].set_xlabel(TARGET)

positive = target_values[target_values > 0]
axes[1].hist(np.log1p(positive), bins=60, color="#12B886", edgecolor="white", linewidth=0.4)
axes[1].set_title("log1p (yalnızca pozitifler)")

axes[2].boxplot(target_values, vert=True, widths=0.5)
axes[2].set_title("Kutu grafiği — aykırı değerler")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

plt.tight_layout()
plt.show()

print(f"çarpıklık = {target_values.skew():.3f}")
print(f"sıfır oranı = {(target_values == 0).mean():.3%}")
print(target_values.describe())
"""),
    markdown("""
## 6 · Eksik veri haritası

Eksikliğin **rastgele olup olmadığı** önemlidir. Bir kolon yalnızca belirli bir
dönemde veya belirli bir varlıkta eksikse, bu bir sinyaldir — doldurmadan önce
`_eksikti` bayrağı ekle.
"""),
    code("""
missing = (train.isna().mean() * 100).sort_values(ascending=False)
missing = missing[missing > 0]

if len(missing):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.32 * len(missing))))
    ax.barh(missing.index[::-1], missing.values[::-1], color="#FA5252")
    ax.set_xlabel("eksik %")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.show()
else:
    print("Eksik değer yok.")
"""),
    markdown("""
## 7 · Türkçe metin sağlığı

`İ` harfinin `.lower()` sonucu **iki kod noktasıdır** (U+0069 U+0307), dolayısıyla
`'İ'.lower() != 'i'`. İl/ilçe adıyla yapılan bir join **sessizce 0 satır** döner.
Harici veri (hava, nüfus) eklemeden önce bunu kontrol et.
"""),
    code("""
text_columns = categorical_columns(train)

for column in text_columns[:10]:
    sample = train[column].dropna().astype(str).head(300)
    if any(has_combining_dot(v) for v in sample):
        print(f"! {column}: BİRLEŞİK NOKTA var — yanlış .lower() kullanılmış")
    else:
        print(f"  {column}: temiz ({train[column].nunique()} benzersiz)")

# Kanıt: naif yaklaşım başarısız, join_key başarılı
print("\\n'İ'.lower() =", codepoints("İ".lower()), "->", "İ".lower() == "i")
print("join_key('İZMİR') == join_key('Izmir') ->", join_key("İZMİR") == join_key("Izmir"))
"""),
    markdown("""
## 8 · Zaman ekseni (varsa)

Train ve test'in zaman aralıkları ayrık mı? Ayrıksa **rastgele KFold geleceği
sızdırır** ve CV'yi yapay olarak yükseltir.
"""),
    code("""
if TIME_COLUMN:
    train_times = pd.to_datetime(train[TIME_COLUMN])
    test_times = pd.to_datetime(test[TIME_COLUMN])

    print(f"train: {train_times.min()} → {train_times.max()}")
    print(f"test:  {test_times.min()} → {test_times.max()}")
    print(f"boşluk: {test_times.min() - train_times.max()}")

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.hist(train_times, bins=80, alpha=0.75, label="train", color="#4C6EF5")
    ax.hist(test_times, bins=40, alpha=0.75, label="test", color="#FA5252")
    ax.legend()
    ax.set_title("Zaman dağılımı")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
"""),
    markdown("""
## 9 · Bulgular

> **Bu hücreyi doldur.** Jüri notebook'u okuyacak; burası "veriyi anladık"
> demenin yeri.

| # | Bulgu | Sonuç / aksiyon |
|---|-------|-----------------|
| 1 | | |
| 2 | | |
| 3 | | |

**Seçilen CV şeması:** …
**Tespit edilen sızıntı riskleri:** …
**İlk feature hipotezleri:** …
"""),
]

# ============================================================================
# 02 - BASELINE
# ============================================================================

BASELINE_CELLS = [
    markdown("""
# Grid Up Datathon — 02 · Baseline

Amaç: **en hızlı geçerli submission**. Optimize etmeden önce çalışan bir uçtan
uca hattın olsun. İlk gün hedefi tek bir sayı: leaderboard'da bir skor.

Sıra: fold'lar → feature → eğit → doğrula → yaz.
"""),
    code("""
import sys
from pathlib import Path

IS_KAGGLE = Path("/kaggle/input").exists()
if IS_KAGGLE:
    # DIKKAT: gridup Kaggle imajinda KURULU DEGILDIR. Onceki surumde sys.path
    # yalnizca YERELDE ayarlaniyordu; Kaggle'da 'import gridup' ModuleNotFound
    # veriyordu. Once offline paket dataset'indeki wheel'i kur, o yoksa ham
    # kaynagi sys.path'e ekle.
    import glob
    import subprocess

    _whl = glob.glob("/kaggle/input/*/gridup-*.whl")
    if _whl:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", _whl[0], "-q"],
            check=False,
        )
    else:
        for _src in glob.glob("/kaggle/input/*/src"):
            sys.path.insert(0, _src)
else:
    sys.path.insert(0, str(Path.cwd().parent / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from gridup import (
    conditional_quantile_from_hurdle, cross_validate, fit_quantile_ladder,
    fit_two_stage, make_model_zoo, read_any, set_global_seed,
    sweep_count_objectives, tune_with_optuna, write_submission, zero_baseline_score,
)
from gridup.compat import categorical_columns
from gridup.ensemble import hill_climb_weights, prune_by_correlation, stack_oof
from gridup.experiment import ExperimentLog, ExperimentRecord
from gridup.features import (
    add_calendar_features, add_frequency_encoding, add_lag_features,
    add_neighbour_target_lag, add_physical_derivatives, add_regional_aggregates,
    nearest_neighbours, shared_origin,
)
from gridup.metrics import inverse_log_transform, log_transform_target
from gridup.models import starter_params
from gridup.refit import estimate_full_data_rounds, extract_best_iterations, multi_seed_refit
from gridup.selection import null_importance_filter, shap_backward_selection
from gridup.validation import adversarial_validation, build_splitter, purged_time_series_split

set_global_seed(42)

DATA_DIR = Path("/kaggle/input/GRID-UP-YARISMA-SLUG") if IS_KAGGLE else Path("../data/raw")
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("../submissions")

TARGET = "HEDEF_KOLON"     # TODO
ID_COLUMN = "id"           # TODO
TIME_COLUMN = None         # TODO
GROUP_COLUMN = None        # TODO
METRIC = "rmse"            # TODO — yarışmanın resmi metriği (2024'te MAE idi!)
TASK = "regression"        # regression | binary | multiclass
LOG_TARGET = False         # metrik RMSLE ise veya hedef çok çarpıksa True

# TAHMİN UFKU — en pahalı sessiz hatanın kaynağı.
# Test ileriideki bir BLOK ise (ör. bir sonraki ay), o bloğun son gününü
# tahmin ederken elindeki en taze veri blok uzunluğu kadar eskidir.
# shift(1) ile hesaplanan lag'ler CV'de harika görünür, private LB'de çöker.
# Veri geldiğinde: HORIZON = (test.tarih.max() - test.tarih.min()).days + 1
HORIZON = 1                # TODO
"""),
    code("""
train = read_any(DATA_DIR / "train.csv")
test  = read_any(DATA_DIR / "test.csv")
print(train.shape, test.shape)
"""),
    markdown("""
## 1 · Fold'lar — feature üretmeden ÖNCE

Sıra önemli: hedef kodlama fold'lara ihtiyaç duyar. Fold'ları önce sabitle ki
tüm deneyler **aynı bölmeler** üzerinde karşılaştırılabilir olsun.
"""),
    code("""
if TIME_COLUMN:
    train[TIME_COLUMN] = pd.to_datetime(train[TIME_COLUMN])
    test[TIME_COLUMN] = pd.to_datetime(test[TIME_COLUMN])
    HORIZON = int((test[TIME_COLUMN].max() - test[TIME_COLUMN].min()).days) + 1
    print(f"Tahmin ufku (test blok uzunluğu): {HORIZON} gün")
    # Ambargo: en uzun kayan pencerenden BÜYÜK olmalı (zorunlu parametre)
    folds = purged_time_series_split(train[TIME_COLUMN], n_splits=5,
                                     embargo=pd.Timedelta(days=30))
elif GROUP_COLUMN:
    splitter = build_splitter("GroupKFold", n_splits=5)
    folds = list(splitter.split(train, groups=train[GROUP_COLUMN]))
else:
    scheme = "StratifiedKFold" if TASK != "regression" else "KFold"
    splitter = build_splitter(scheme, n_splits=5, seed=42)
    folds = list(splitter.split(train, train[TARGET] if TASK != "regression" else None))

for i, (tr, va) in enumerate(folds, 1):
    print(f"fold {i}: train={len(tr):>8,}  valid={len(va):>8,}")
"""),
    markdown("""
## 2 · Feature'lar

**Kural:** train ve test'e *aynı* fonksiyon uygulanır. Ayrı kod yolları,
eğitim/servis uyumsuzluğunun bir numaralı kaynağıdır.
"""),
    code("""
# ORTAK zaman başlangıcı: train ve test için ayrı ayrı hesaplanırsa test'in
# gün sayacı yeniden 0'dan başlar ve model test'i train'in geçmişi sanır.
# Bu hata lokal CV'de GÖRÜNMEZ — sadece leaderboard çöker.
ORIGIN = shared_origin(train, test, time_column=TIME_COLUMN) if TIME_COLUMN else None

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Train ve test'e aynı dönüşümleri uygular. Girdiyi değiştirmez.\"\"\"
    out = frame.copy()
    if TIME_COLUMN:
        out = add_calendar_features(out, TIME_COLUMN, include_year=False, origin=ORIGIN)
    # categorical_columns: pandas 2.x ve 3.x'te de doğru çalışır.
    # Düz `dtype == object` kontrolü pandas 3.0'da metin kolonlarını KAÇIRIR.
    categorical = categorical_columns(out)
    if categorical:
        out = add_frequency_encoding(out, categorical[:12])
    return out

train_features = build_features(train)
test_features = build_features(test)

drop = {TARGET, ID_COLUMN, TIME_COLUMN} - {None}
FEATURES = [c for c in train_features.columns
            if c not in drop and c in test_features.columns]
print(f"{len(FEATURES)} feature")
"""),
    markdown("## 3 · Eğit"),
    code("""
y = train_features[TARGET].to_numpy()
if LOG_TARGET:
    y = log_transform_target(y)

params = starter_params("lightgbm", TASK)

result = cross_validate(
    train_features[FEATURES], y, folds,
    kind="lightgbm", task_type=TASK, metric=METRIC,
    params=params, test=test_features[FEATURES],
)

print(result.summary())
"""),
    markdown("""
## 4 · Submission

`write_submission` yazmadan önce doğrular: NaN, sonsuz, eksik ID, sabit tahmin,
negatif değer. Kaggle'ın "Submission Scoring Error" mesajı sana hiçbir şey söylemez.
"""),
    code("""
predictions = result.test_predictions
if LOG_TARGET:
    predictions = inverse_log_transform(predictions)

path = write_submission(
    test_features[ID_COLUMN].to_numpy(),
    predictions,
    OUT_DIR / "baseline_lgbm.csv",
    id_column=ID_COLUMN,
    target_column=TARGET,
)
"""),
    markdown("""
## 5 · Deney defterine yaz

Submission gönderdikten **sonra** leaderboard skorunu geri yaz:

```python
log.record_lb("baseline_lgbm", 12.3456)
print(log.cv_lb_correlation())
```

CV–LB korelasyonu bu yarışmanın en önemli tek sayısıdır. r > 0.8 ise CV'ne
güven; r < 0.5 ise CV şemanı düzeltmeden devam etme.
"""),
    code("""
log = ExperimentLog(OUT_DIR.parent / "experiments" / "deneyler.jsonl")

log.add(ExperimentRecord(
    name="baseline_lgbm",
    cv_score=result.overall_score,
    metric=METRIC,
    model_kind="lightgbm",
    n_features=len(FEATURES),
    fold_scores=result.fold_scores,
    notes="baseline: takvim + frekans kodlama",
    submission_path=str(path),
))

log.leaderboard()
"""),
    markdown("""
## Sonraki adımlar

Sıra önemli — her adımdan sonra CV'yi ölçüp deftere yazın.

**1 · Kayma kontrolü**
```python
sonuc = adversarial_validation(train[FEATURES], test[FEATURES])
print(sonuc["auc"], sonuc["verdict"])   # AUC > 0.8 ise ayrıştıran feature'ı çıkar
```

**2 · Ufuk-farkındalıklı lag/rolling** — en güçlü aile
```python
out = add_lag_features(out, TARGET, [1, 7, 28], time_column=TIME_COLUMN,
                       group_columns=[GROUP_COLUMN], horizon=HORIZON)
```

**3 · Hazır harici veri** (indirilmiş, `data/` altında)
```python
hava = pd.read_parquet("../data/external/hava_gunluk.parquet")
ilceler = pd.read_parquet("../data/reference/ilceler_gdz_adm.parquet")
komsu = nearest_neighbours(ilceler, key_column="ilce_key",
                           latitude_column="lat", longitude_column="lon", k=3)
out = add_neighbour_target_lag(out, komsu, key_column="ilce_key",
                               time_column=TIME_COLUMN, target_column=TARGET,
                               horizon=HORIZON)
```
Havada **ortalama değil `max` ve quantile** kullanın — hasarı rüzgârın ortalaması
değil tepesi yapar: `add_regional_aggregates`, `add_physical_derivatives`.

**4 · Sayım hedefiyse objective süpürmesi**
```python
zoo = sweep_count_objectives(train[FEATURES], y, folds, metric=METRIC)
print(zoo.leaderboard())   # poisson / tweedie / mae / l2 aynı fold'larda
```

**5 · Sıfır oranı > %40 ise iki aşamalı model**
```python
print(zero_baseline_score(y, metric="mae"))   # önce bunu geçtiğini gör
sonuc = fit_two_stage(train[FEATURES], y, folds, metric=METRIC)
```
Metrik MAE ise `q* = 1 − 0.5/p` çözücüsünü kullanın — `expected` ve
`thresholded` modlarının **ikisi de** MAE altında suboptimaldir:
```python
merdiven = fit_quantile_ladder(train[FEATURES], y, folds)
tahmin = conditional_quantile_from_hurdle(sonuc.oof_probability,
                                          {q: r.oof_predictions for q, r in merdiven.items()})
```

**6 · Hiperparametre araması** — objective'i de arama uzayına koyun
```python
tuned = tune_with_optuna(train[FEATURES], y, folds, metric=METRIC,
                         timeout=3600, search_objective=True)
print(tuned.objective_comparison())
```

**7 · Model zoo + harman**
```python
zoo = make_model_zoo(train[FEATURES], y, folds, metric=METRIC, test=test[FEATURES])
secilen = prune_by_correlation(zoo.oof_matrix, y, max_members=5)
agirliklar = hill_climb_weights({k: zoo.oof_matrix[k] for k in secilen}, y, metric=METRIC)
stack = stack_oof(zoo.oof_matrix, y, folds, test_predictions=zoo.test_matrix)
```
`stack_oof` hem stacking hem hill climbing skorunu raporlar. Fark küçükse
**hill climbing'i tercih edin** — jüri notebook'u okuyacak, açıklanabilirlik değerli.

**8 · Feature eleme**
```python
temiz = null_importance_filter(train[FEATURES], y)          # dakikalar
secim = shap_backward_selection(train[temiz["keep"]], y, folds)  # saatler
```

**9 · Son gün: çok tohumlu tam veri refit**
```python
tur = estimate_full_data_rounds(extract_best_iterations(result.models), n_folds=len(folds))
final = multi_seed_refit(train[FINAL], y, test[FINAL], params=tuned.best_params,
                         n_estimators=tur, seeds=range(15))
```
"""),
    markdown("""
## Jüri çıktıları

Bunları **son gün üretmeye kalkmayın** — pipeline'ın parçası olmalı.
Değerlendirmenin üçte ikisi notebook + sunum.
"""),
    code("""
from gridup.reporting import (
    business_impact, cv_fold_table, error_by_segment,
    feature_importance_table, model_footprint,
    plot_error_by_segment, plot_fold_scores, plot_prediction_timeline,
)

# 1 · Fold tablosu — kararlılığı gösterir
display(cv_fold_table(result))
plot_fold_scores(result); plt.show()

# 2 · Model NEREDE yanılıyor — sunumun en ikna edici bölümü
segment_hatasi = error_by_segment(y_true, y_pred, test[GROUP_COLUMN], metric=METRIC)
plot_error_by_segment(segment_hatasi, metric=METRIC); plt.show()

# 3 · Sinyal nereden geliyor — 400 satırlık önem listesi yerine aile dağılımı
display(feature_importance_table(result, group_prefixes=(
    "tarih_", "tatil_", "komsu_", "bolge_", "sebep_")))

# 4 · Operasyonel maliyet — jüri bunu soruyor, modeli gerçekten çalıştıracak
print(model_footprint(result.models, elapsed_seconds=result.elapsed_seconds))

# 5 · İş dili — "MAE 2.95" değil, "ortalama 3 kesinti hatayla tahmin ediyoruz"
print(business_impact(y_true, y_pred, unit_label="kesinti")["ozet"])
"""),
    markdown("""
## Sunum için not

Jüri koltuğunda **mühendisler ve iş birimleri** var, akademisyen değil.
2024 birincisinin sunumunun son üç slaydı tamamen iş değeriydi: açıklanabilir
çözüm, daraltılmış feature seti, ~25 MB model, yeni veriyle eğitilebilirlik.

Skor ilk 10'a sokar; bu bölüm ödülü belirler.
"""),
]


def main() -> int:
    write_notebook(EDA_CELLS, NOTEBOOK_DIR / "01_kesif.ipynb")
    write_notebook(BASELINE_CELLS, NOTEBOOK_DIR / "02_baseline.ipynb")

    # Üretilen dosyalar geçerli JSON mu?
    for path in NOTEBOOK_DIR.glob("*.ipynb"):
        json.loads(path.read_text(encoding="utf-8"))
    print("Tüm notebook'lar geçerli JSON.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
