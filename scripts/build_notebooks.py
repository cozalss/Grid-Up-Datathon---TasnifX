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
if not IS_KAGGLE:
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
if not IS_KAGGLE:
    sys.path.insert(0, str(Path.cwd().parent / "src"))

import pandas as pd

from gridup import cross_validate, read_any, set_global_seed, write_submission
from gridup.compat import categorical_columns
from gridup.experiment import ExperimentLog, ExperimentRecord
from gridup.features import add_calendar_features, add_frequency_encoding
from gridup.metrics import inverse_log_transform, log_transform_target
from gridup.models import starter_params
from gridup.validation import build_splitter, purged_time_series_split

set_global_seed(42)

DATA_DIR = Path("/kaggle/input/GRID-UP-YARISMA-SLUG") if IS_KAGGLE else Path("../data/raw")
OUT_DIR = Path("/kaggle/working") if IS_KAGGLE else Path("../submissions")

TARGET = "HEDEF_KOLON"     # TODO
ID_COLUMN = "id"           # TODO
TIME_COLUMN = None         # TODO
GROUP_COLUMN = None        # TODO
METRIC = "rmse"            # TODO — yarışmanın resmi metriği
TASK = "regression"        # regression | binary | multiclass
LOG_TARGET = False         # metrik RMSLE ise veya hedef çok çarpıksa True
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
    # Ambargo: en uzun kayan pencerenden BÜYÜK olmalı
    folds = list(purged_time_series_split(train[TIME_COLUMN], n_splits=5,
                                          embargo=pd.Timedelta(days=30)))
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
def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Train ve test'e aynı dönüşümleri uygular. Girdiyi değiştirmez.\"\"\"
    out = frame.copy()
    if TIME_COLUMN:
        out = add_calendar_features(out, TIME_COLUMN, include_year=False)
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

1. **Adversarial validation** — `validation.adversarial_validation(train, test)`
   ile train/test kayması var mı ölç
2. **Hedef kodlama** — `features.oof_target_encode` (yüksek kardinaliteli kolonlar)
3. **Lag / rolling** — zaman varsa en güçlü feature ailesi
4. **Harici veri** — `scripts/fetch_weather.py` ile hava durumu
5. **CatBoost + XGBoost** — çeşitlilik için, sonra `ensemble.hill_climb_weights`
6. **Eşik optimizasyonu** — sınıflandırmaysa `metrics.optimize_threshold`
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
