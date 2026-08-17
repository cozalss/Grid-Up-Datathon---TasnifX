"""Notebook sablonlarini uretir.

Neden elle .ipynb yazmak yerine bir uretec: JSON'u elle yazmak hataya acik ve
diff'i okunamaz. Bu betik notebook'lari kaynak koddan uretir, boylece sablonlar
sürüm kontrolünde okunabilir kalir.

Anlati kalibi: her markdown hucresi bir SEBEP anlatir ve her sayi olculmus bir
kosunun ciktisidir -- kaynak betik sayinin yaninda yazar. Kod hucreleri yarisma
verisi icin GENELDIR (yollar/kolon adlari parametrik); prova sayilari yalnizca
markdown'da "gercek GDZ verisinde olculdu" diye gecer.

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


# Kaggle onyukleme blogu iki notebook'ta da birebir ayni olmali -- tek kaynak.
_KAGGLE_BOOTSTRAP = """
import sys
from pathlib import Path

# Kaggle'da: /kaggle/input/<yarisma>/  · yerelde: data/raw/
IS_KAGGLE = Path("/kaggle/input").exists()
if IS_KAGGLE:
    # DIKKAT: gridup Kaggle imajinda KURULU DEGILDIR. Onceki surumde sys.path
    # yalnizca YERELDE ayarlaniyordu; Kaggle'da 'import gridup' ModuleNotFound
    # veriyordu. Once offline paket dataset'indeki wheel'i kur, o yoksa ham
    # kaynagi sys.path'e ekle. (Path.glob, glob.glob degil: juri notebook'u
    # ruff'tan geciyor -- PTH207.)
    import subprocess

    _whl = sorted(Path("/kaggle/input").glob("*/gridup-*.whl"))
    if _whl:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             str(_whl[0]), "-q"],
            check=False,
        )
    else:
        for _src in Path("/kaggle/input").glob("*/src"):
            sys.path.insert(0, str(_src))
else:
    sys.path.insert(0, str(Path.cwd().parent / "src"))
"""


# ============================================================================
# 01 - KESIF (EDA)
# ============================================================================

EDA_CELLS = [
    markdown("""
# Grid Up Datathon — 01 · Keşif

**Takım:** _(takım adını yaz)_ · **Tarih:** _(gün)_

## Problem neden zor?

Elektrik kesintisi tahmininin üç yapısal zorluğunu tahmin etmedik, **ölçtük**:
yarışmadan önce hattın tamamını 68.257 gerçek GDZ kesinti kaydında prova ettik
(İzmir + Manisa, 47 ilçe, 2021-05 → 2022-08, saat damgalı olay kaydı). Bu
notebook'taki her sayı o provanın çıktısıdır; yanında hangi betikle ölçüldüğü yazar.

1. **Sıfır-şişkin hedef.** Gerçek ilçe × gün panelinde günlerin %35.0'ında hiç
   kesinti yok (`scripts/benchmark_gercek.py`). Ortalamayı optimize eden model hem
   sıfırları hem kesintileri ıskalar; metrik, kayıp ve model buna göre seçilmeli.
2. **Gürültü.** Aynı modelin fold skorları 150.8 → 461.4 dk arasında salınıyor
   (`scripts/real_data_rehearsal.py`). Tek fold'a — veya tek skora — bakan her
   karar yanıltıcıdır.
3. **Mekânsal yapı.** Fırtına ilçe sınırı tanımaz; komşu ilçeler aynı gün arızalanır.
   Coğrafya sinyal kaynağıdır ama aynı zamanda sızıntı riskidir: komşu ilçenin
   geçmişi de ancak tahmin ufku kadar kaydırılarak kullanılabilir.

Bu notebook üç karar üretir: **hedef + panel tanımı**, **doğrulama şeması** ve
**sızıntı duvarı**. Sonraki 12 günün her deneyi bu üç karara yaslanır.
"""),
    code(
        _KAGGLE_BOOTSTRAP
        + """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridup import (
    build_panel, environment_report, panel_coverage, profile, read_any, set_global_seed,
)
from gridup.compat import categorical_columns
from gridup.profiling import quick_look
from gridup.turkish import codepoints, has_combining_dot, join_key, strip_qualifier
from gridup.validation import leakage_report, purged_time_series_split, suggest_scheme

set_global_seed(42)

# Ortami yazdir -- juri tekrarlanabilirlige bakiyor, bu ucuz bir puan.
for key, value in environment_report().items():
    print(f"{key:<26} {value}")
"""
    ),
    markdown("""
## 1 · Veriyi oku — neden `read_any`

Türk kurum dosyaları `cp1254` kodlama, `;` ayırıcı ve ondalık `,` ile gelebilir;
düz `pd.read_csv` bunları sessizce bozar. `read_any` kodlamayı ve ayırıcıyı
**kanıtlayarak** seçer. Gerçek GDZ dosyasında ölçülen: kodlama `utf-8-sig` çıktı
ve `İL → il` dahil 8 kolon adı normalize edildi (`scripts/real_data_rehearsal.py`).
Yani bu bir varsayım değil, ilk gerçek dosyada karşılaştığımız davranış.
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

Tek çağrı, elle iki saat sürecek keşfin yerine geçer: çarpıklık, sıfır yığılması,
ID-benzeri kolonlar, train/test şema farkı (= sızıntı adayı) ve birleşik nokta
(U+0307) taşıyan Türkçe kolonlar işaretlenir. Amaç grafik biriktirmek değil,
**karar listesi** çıkarmaktır.
"""),
    code("""
# TODO: hedef kolon adini veri geldiginde doldur
TARGET = "HEDEF_KOLON"

dataset_profile = profile(train, test, target=TARGET)
print(dataset_profile.report())
"""),
    code("""
# Kolon bazli kompakt tablo -- hizli gozden gecirme icin
quick_look(train)
"""),
    markdown("""
## 3 · Hedef dağılımı

Hedefin şekli üç kararı belirler: metrik, dönüşüm, model ailesi. Gerçek GDZ
verisinde hedefi `kesinti_dk = endtime − starttime` olarak kurduk; medyan 104 dk
ama maksimum 17.359 dk = 12.1 gün (`scripts/real_data_rehearsal.py`). Ağır sağ
kuyruk + sıfır yığılması birlikte görülür ve MAE-ailesi kayıpları ile iki aşamalı
(hurdle) adayları öne çıkarır — hangisinin kazandığı 02 numaralı notebook'ta
**ölçülmüş** olarak duruyor.

- Çarpıklık > 2 → `log1p` dönüşümünü dene
- Sıfır yığılması büyükse → iki aşamalı model adayı; kararı sezgi değil ölçüm versin
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
## 4 · Eksik veri haritası

Eksikliğin **rastgele olup olmadığı** önemlidir. Bir kolon yalnızca belirli bir
dönemde veya belirli bir ilçede eksikse bu bir sinyaldir — doldurmadan önce
`_eksikti` bayrağı ekle. Doldurma bayrağının kendisi ise asla feature olmaz
(aşağıda, panel bölümünde neden).
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
## 5 · Türkçe tuzaklar — sessiz satır kaybı

İki tuzak da provada **gerçekten başımıza geldi**; ikisi de hata fırlatmaz,
sadece satır kaybettirir:

- `'İ'.lower()` iki kod noktası üretir (U+0069 U+0307), dolayısıyla
  `'İ'.lower() != 'i'`. İl/ilçe adıyla yapılan join sessizce 0 satır döner.
- Gerçek veride ilçe adı **nitelenmiş** geldi: `Köprübaşı / Manisa` (Köprübaşı
  hem Manisa'da hem Trabzon'da var). Referans tablomuz yalın `Köprübaşı` tutuyor;
  47 ilçenin 46'sı normalize eşleşti, bu tek ilçenin **284 kaydı** sessizce
  düşecekti. `strip_qualifier` + `join_key` sonrası eşleşme 68.257/68.257 = %100.0
  (`scripts/real_data_rehearsal.py`).

Ders: her join'den sonra satır sayısı ve eşleşme oranı **doğrulanır**; harici veri
(hava, nüfus) eklemeden önce bu hücre koşulur.
"""),
    code("""
text_columns = categorical_columns(train)

for column in text_columns[:10]:
    sample = train[column].dropna().astype(str).head(300)
    if any(has_combining_dot(v) for v in sample):
        print(f"! {column}: BİRLEŞİK NOKTA var — yanlış .lower() kullanılmış")
    else:
        print(f"  {column}: temiz ({train[column].nunique()} benzersiz)")

# Kanit 1: naif yaklasim basarisiz, join_key basarili
print("\\n'İ'.lower() =", codepoints("İ".lower()), "->", "İ".lower() == "i")
print("join_key('İZMİR') == join_key('Izmir') ->", join_key("İZMİR") == join_key("Izmir"))

# Kanit 2: niteleyici eki ayri bir adimdir -- join_key kesmez, strip_qualifier keser
print("strip_qualifier('Köprübaşı / Manisa') ->", strip_qualifier("Köprübaşı / Manisa"))
"""),
    markdown("""
## 6 · Panel kararı — olay kaydından ilçe × gün ızgarasına

Kesinti verisi **olay kaydıdır**: her satır bir arıza, saat damgalı (`14:23` gibi),
kesintisiz gün için satır yok. Model ise düzenli bir ilçe × gün paneli ister.
Bu dönüşümün iki ölçülmüş tuzağı var:

1. **Saat damgası ızgaraya oturtulmazsa hedef sessizce buharlaşır.** Günlük ızgara
   gece yarılarından oluşur; `14:23` damgalı kayıt merge'de hiçbir güne eşleşmez.
   Kontrollü ölçümde hedef kütlesinin **%90.4'ü** yok oldu; günde ~3 olaylı veride
   kayıp %99.8 (`src/gridup/panel.py`, `tests/test_panel_uydurma.py`). Hata yok,
   uyarı yok — model hep sıfır öğrenir. `build_panel` damgayı ızgaraya oturtur ve
   hedef kütlesini doğrular.
2. **Dolgu değeri hedefin türüne bağlıdır.** "Kayıt yok", sayım/süre hedefinde
   gerçek 0'dır; ölçüm hedefinde (tüketim, gerilim) bilinmeyendir → NaN. Gerçek GDZ
   panelinde 7.710 satır (%34.8) sıfır dolgu aldı ve `_dolduruldu` bayrağı **asla
   feature olmaz** — modelin "bu satır dolgu" bilgisini öğrenmesi sızıntıdır.

Gerçek veride sonuç: 68.257 kayıt → 47 ilçe × 472 gün = 22.184 satır, doluluk
%65.2, hedef kütlesi %100.00 korundu (`scripts/real_data_rehearsal.py`).
"""),
    code("""
# Olay kaydi -> panel donusumu. Yarisma verisi HAZIR panelse bu adim atlanir.
ENTITY_COLUMN = "ILCE_KOLONU"      # TODO: varlik anahtari (join_key'den gecmis olmali)
RAW_TIME_COLUMN = "TARIH_KOLONU"   # TODO: ham zaman damgasi kolonu

# Once OLC: doluluk %100'u asiyorsa ayni gunde birden cok olay var demektir.
# (Eski olcum bunu maskeliyordu ve %304.8 doluluk raporlamisti -- duzeltildi.)
kapsam = panel_coverage(train, entity_columns=[ENTITY_COLUMN], time_column=RAW_TIME_COLUMN)
print(f"beklenen {kapsam['expected_rows']:,.0f}  gercek {kapsam['actual_rows']:,.0f}"
      f"  doluluk %{kapsam['coverage'] * 100:.1f}")

panel = build_panel(
    train, entity_columns=[ENTITY_COLUMN], time_column=RAW_TIME_COLUMN,
    value_columns=[TARGET],
)
# Hedef kutlesi korunmali -- korunmadiysa izgaraya oturtma bozuk demektir.
# Gercek GDZ verisinde %100.00 olculdu; sapma varsa devam etmeden DUR.
korunan = panel[TARGET].sum() / train[TARGET].sum()
print(f"panel {panel.shape}   hedef kutlesi %{korunan * 100:.2f} (100 olmali)")
"""),
    markdown("""
## 7 · Doğrulama şeması — yarışmanın kazanıldığı karar

Yanlış şema iki yönden öldürür: **iyimser CV** (sızıntı → leaderboard'da çöküş)
veya **gürültülü CV** (hangi değişikliğin işe yaradığı görünmez → public LB'ye
göre karar → shakeup). Zaman + panel verisinde seçimimiz
`purged_time_series_split` ve gerekçeleri ölçülü:

- **`test_span` = tahmin ufku.** 2023 GDZ Datathon birincisi
  `TimeSeriesSplit(n_splits=3, test_size=744)` kullandı; 744 saat = 31 gün =
  test bloğunun tam boyu (`src/gridup/validation.py`). CV, tahmin edilecek ufku
  birebir taklit etmelidir. Panelde satır sayısına göre eşit bölme, zaman
  uzunlukları eşit olmayan fold'lar üretir ve skorlar karşılaştırılamaz olur.
- **`embargo` bilinçli seçilir ve ufuktan küçük olmaz.** Kayan pencereli feature'lar
  fold sınırını aşarsa son train satırları ilk valid satırlarıyla aynı ham veriyi
  görür — sessiz bir iyimserlik. Kütüphane bu yüzden embargo'yu zorunlu parametre
  yapar; sessiz küçük varsayılan (~2 gün) tam da önlemeye çalıştığı sızıntıya izin
  veriyordu.
- Provadaki kurulum: embargo 31 gün, 4 fold × 31 gün; her fold'un valid'i
  47 ilçe × 31 gün = 1.457 satır (`scripts/real_data_rehearsal.py`).
"""),
    code("""
TIME_COLUMN = None   # TODO: zaman kolonu (panel kurduysan panelin gun kolonu)

suggestion = suggest_scheme(train, target=TARGET)
print(suggestion)

if TIME_COLUMN:
    train_times = pd.to_datetime(train[TIME_COLUMN])
    test_times = pd.to_datetime(test[TIME_COLUMN])
    print(f"train: {train_times.min()} -> {train_times.max()}")
    print(f"test:  {test_times.min()} -> {test_times.max()}")
    print(f"bosluk: {test_times.min() - train_times.max()}")

    # Tahmin ufku = test blogunun boyu. CV bunu birebir taklit etmeli
    # (2023 birincisinin test_size=744 saat = 31 gun secmesinin sebebi).
    HORIZON = int((test_times.max() - test_times.min()).days) + 1
    folds = purged_time_series_split(
        train_times, n_splits=4,
        embargo=pd.Timedelta(days=max(HORIZON, 30)),
        test_span=pd.Timedelta(days=HORIZON),
    )
    for i, (tr, va) in enumerate(folds, 1):
        print(f"fold {i}: train={len(tr):>8,}  valid={len(va):>6,}")
"""),
    markdown("""
## 8 · Sızıntı duvarı

Kesinti verisinde en tehlikeli kolonlar **aynı günün bilgisini** taşıyanlardır:
arıza sebebi, etkilenen abone sayısı, o günkü yük. Tahmin anında bunlar bilinmez.
Provada bunu bilerek ölçtük: aynı-gün kolonları feature bırakıldığında gain
tablosunun tepesine oturuyorlar — `effectedsubscribers` tek başına en iyi meşru
feature'ın (`sicaklik_max`) yaklaşık **24 katı** gain topladı; ID kolonu bile ~8
katıyla ikinci sıradaydı (`scripts/real_data_rehearsal.py`). Böyle bir model CV'de
parlar, gerçek tahmin gününde o kolonlar olmadığı için çöker.

Duvarın üç kuralı:

1. Aynı günün bilgisi feature olamaz; yalnızca ufuk kadar kaydırılmış geçmiş
   agregatları (lag/rolling) meşrudur.
2. Hedeften türetilen hiçbir şey aynı satırın feature'ı olamaz.
3. `leakage_report` model eğitilmeden **önce** koşulur; `critical` bulgu varsa
   çözülmeden devam edilmez.
"""),
    code("""
findings = leakage_report(train, TARGET, test=test, time_column=TIME_COLUMN)
print(findings["summary"], "\\n")

for severity in ("critical", "warning", "info"):
    for message in findings[severity]:
        print(f"[{severity.upper()}] {message}")
"""),
    markdown("""
## Çıktı: üç karar

> Bu tablo veri gününde doldurulur — jüri "veriyi anladık" iddiasının kanıtını
> burada görür. Prova satırı, doldurulmuş bir örnek olarak bırakıldı.

| Karar | Prova (gerçek GDZ, ölçüldü) | Yarışma verisi (doldur) |
|---|---|---|
| Hedef + panel | `kesinti_dk`; 47 ilçe × 472 gün; kütle %100.00 | … |
| CV şeması | purged, embargo 31 g, 4 fold × 31 g `test_span` | … |
| Sızıntı duvarı | sebep / abone / yük → yalnız ufuk-kaydırmalı lag | … |
"""),
]

# ============================================================================
# 02 - BASELINE
# ============================================================================

BASELINE_CELLS = [
    markdown("""
# Grid Up Datathon — 02 · Baseline

Amaç: **en hızlı geçerli submission** — ama rastgele değil, ölçülmüş bir planla.
Yarışmadan önce, 68.257 gerçek GDZ kesinti kaydında (47 ilçe, 2021-05 → 2022-08)
hangi feature ailesinin katkı verdiğini ve model reçetelerinin OOF davranışını
**aynı purged fold'larda** ölçtük (`scripts/ablation_gercek.py`,
`scripts/benchmark_gercek.py`). Veri gününde deney değil **icra** yapacağız:
plan sabittir, sayılar yarışma verisinde yeniden ölçülür.

Sıra: fold'lar → feature'lar (ölçülen öncelikle) → model (ölçülen reçete) →
harman → submission.
"""),
    code(
        _KAGGLE_BOOTSTRAP
        + """
import matplotlib.pyplot as plt
import pandas as pd

from gridup import (
    conditional_quantile_from_hurdle, cross_validate, fit_conditional_quantile_ladder,
    fit_two_stage, make_model_zoo, read_any, set_global_seed,
    sweep_count_objectives, tune_with_optuna, write_submission, zero_baseline_score,
)
from gridup.compat import categorical_columns
from gridup.ensemble import hill_climb_weights, prune_by_correlation, stack_oof
from gridup.experiment import DataArtifact, ExperimentProvenance, ExperimentRecord
from gridup.features import (
    add_calendar_features, add_lag_features,
    add_neighbour_target_lag, add_physical_derivatives, add_regional_aggregates,
    nearest_neighbours, shared_origin,
)
from gridup.models import starter_params
from gridup.pipeline import FoldPlan
from gridup.recipe import CVRecipe, FeatureRecipe, ModelRecipe, PipelineRecipe
from gridup.refit import (estimate_full_data_rounds, extract_best_iterations,
                          fold_train_fraction, multi_seed_refit)
from gridup.selection import null_importance_filter, shap_backward_selection
from gridup.stores import SQLiteExperimentStore
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
# Test ilerideki bir BLOK ise (ör. bir sonraki ay), o bloğun son gününü
# tahmin ederken elindeki en taze veri blok uzunluğu kadar eskidir.
# shift(1) ile hesaplanan lag'ler CV'de harika görünür, private LB'de çöker.
# Veri geldiğinde: HORIZON = (test.tarih.max() - test.tarih.min()).days + 1
HORIZON = 1                # TODO
"""
    ),
    code("""
train = read_any(DATA_DIR / "train.csv")
test  = read_any(DATA_DIR / "test.csv")
print(train.shape, test.shape)
"""),
    markdown("""
## 1 · Fold'lar — feature üretmeden önce

Hedef kodlama ve lag'ler fold'lara ihtiyaç duyar; fold'lar önce sabitlenir ki
bugünün ve yarının **bütün** deneyleri aynı bölmeler üzerinde karşılaştırılabilsin.
Şemanın gerekçesi 01 numaralı notebook'ta; özeti: `test_span` = tahmin ufku
(2023 birincisinin `test_size=744` paraleli), `embargo` bilinçli ve ufuktan küçük
değil.
"""),
    code("""
if TIME_COLUMN:
    train[TIME_COLUMN] = pd.to_datetime(train[TIME_COLUMN])
    test[TIME_COLUMN] = pd.to_datetime(test[TIME_COLUMN])
    HORIZON = int((test[TIME_COLUMN].max() - test[TIME_COLUMN].min()).days) + 1
    print(f"Tahmin ufku (test blok uzunlugu): {HORIZON} gun")
    # test_span = ufuk: fold'lar zaman uzunlugu esit pencereler olsun.
    # embargo >= ufuk: kayan pencereler fold sinirini asmasin.
    folds = purged_time_series_split(
        train[TIME_COLUMN], n_splits=4,
        embargo=pd.Timedelta(days=max(HORIZON, 30)),
        test_span=pd.Timedelta(days=HORIZON),
    )
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
## 2 · Feature aileleri — önceliği tahmin değil ölçüm belirledi

Feature önemi (gain) aile önceliğini **söyleyemez**: korele kolonlar tek tek
"önemli" görünür ama biri silinince diğeri işi devralır. Bu yüzden ölçü
leave-one-group-out'tur: aile tümüyle silinir, aynı purged fold'larla MAE yeniden
ölçülür. Gerçek GDZ verisinde, 76 feature'lı tam model MAE 313.64 / hep-sıfır
366.97 iken (`scripts/ablation_gercek.py` → `experiments/ablasyon_gercek.json`):

| Aile | Δ MAE (silinince kayıp) | Kolon | Veri günü kararı |
|---|---|---|---|
| lag / rolling | **+22.34** | 11 | İlk kurulacak — kalanların toplamından büyük |
| hava | +2.47 | 24 | İkinci; ortalama değil `max`/quantile agregatları |
| komşu lag | +0.18 | 3 | Marjinal; ucuz, ufuk şartıyla kalır |
| frekans | 0.00 | 1 | Tam ızgara panelde sabit 1/47 → sıfır bilgi |
| takvim | −0.19 | 15 | Gürültü bandında |
| tatil | −4.66 | 15 | Silinince MAE düşüyor → ilk elenecek aday |
| güneş | −5.03 | 7 | İlk elenecek aday |

Tam modelin en yüksek gain'li kolonu da aynı hikâyeyi anlatıyor: 93 günlük,
ufuk-kaydırmalı kayan ortalama (`kesinti_dk_ufuk31_kayan93_mean`). Geçmiş kesinti
davranışı en güçlü sinyaldir — ama ancak ufuk kadar kaydırılmışsa meşrudur.

Ablasyonun fold_std'si 94.31, skorun ~%30'u: küçük deltalar kesin hüküm değildir.
Sıralamanın ucu ise (lag ≫ diğerleri) gürültünün çok üstünde.
"""),
    code("""
# ORTAK zaman baslangici: train ve test icin ayri ayri hesaplanirsa test'in
# gun sayaci yeniden 0'dan baslar ve model test'i train'in gecmisi sanir.
# Bu hata lokal CV'de GORUNMEZ -- sadece leaderboard coker.
ORIGIN = shared_origin(train, test, time_column=TIME_COLUMN) if TIME_COLUMN else None

def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    \"\"\"Train ve test'e ayni donusumleri uygular. Girdiyi degistirmez.\"\"\"
    out = frame.copy()
    if TIME_COLUMN:
        out = add_calendar_features(out, TIME_COLUMN, include_year=False, origin=ORIGIN)
    # Dagilim/frekans kodlamasi temporal fold icinde fit edilmedikce kapali.
    # Tum train veya test uzerinde ayri fit, gelecekteki kategori dagilimini
    # validasyon ozelliklerine tasir.
    return out

train_features = build_features(train)
test_features = build_features(test)

# LAG AILESI -- ablasyonda tek basina en buyuk katki (+22.34 MAE).
# Train + test BIRLIKTE kurulur: test satirlarinin lag'i train'in son
# gunlerinden gelir. horizon=HORIZON kaydirma sayesinde hicbir satir kendi
# gununun (veya daha yakinin) bilgisini goremez -- sizinti duvari korunur.
if TIME_COLUMN and GROUP_COLUMN:
    n_train = len(train_features)
    butun = pd.concat([train_features, test_features], ignore_index=True, sort=False)
    butun = add_lag_features(
        butun, TARGET, shifts=[HORIZON, 2 * HORIZON, 3 * HORIZON],
        time_column=TIME_COLUMN, group_columns=[GROUP_COLUMN], horizon=HORIZON,
    )
    train_features = butun.iloc[:n_train].reset_index(drop=True)
    test_features = butun.iloc[n_train:].reset_index(drop=True)

drop = {TARGET, ID_COLUMN, TIME_COLUMN} - {None}
FEATURES = [c for c in train_features.columns
            if c not in drop and c in test_features.columns]
print(f"{len(FEATURES)} feature")
"""),
    markdown("""
## 3 · Model reçetesi — dokuz aday, aynı fold'lar, aynı bütçe

2023 GDZ birincisi CatBoost'u MAE kaybıyla, 2024 birincisi (Pikachow) LightGBM'i
Optuna ile kullandı; ama o seçimler o yılların verisinde yapıldı. Dokuz reçeteyi
gerçek GDZ verisinde aynı fold, aynı feature seti ve aynı ağaç bütçesiyle
yarıştırdık (`scripts/benchmark_gercek.py` → `experiments/benchmark_gercek.json`):

| Reçete | MAE (dk) | Not |
|---|---|---|
| **catboost_mae** | **304.89** | Görünür OOF lideri; henüz bilimsel kazanan değil |
| lgb_mae | 310.14 | En iyi LightGBM — kayıp = metrik |
| iki_asama_medyan | 316.95 | Koşullu merdiven + q\\*=1−0.5/p kuralı |
| iki_asama (eşikli) | 317.04 | Eşik 0.680; medyan kuralının farkı 0.1 dk'ya indi |
| lgb_sqrt | 324.03 | Rohlik reçetesi: sqrt(y)+L2, geri-kare |
| lgb_tweedie | 327.83 | Sıfır-şişkin hedefe uygun, hızlı aday — harmanda değerli |
| iki_asama_medyan_kalibre | 328.15 | İzotonik kalibrasyon BOZDU (Brier 0.207→0.241) |
| xgb | 402.83 | Baseline'ın altında |
| lgb_l2 | 403.78 | Metrik MAE iken L2 kaybı ~94 dk kaybettiriyor |
| hep-sıfır | 366.97 | Alt çizgi; bunu geçemeyen model rafa kalkar |

Bu tablo **3. dalga** feature setiyle ölçüldü: önceki 49 kolona Hawkes-esinli
üstel bozunum (3g/14g yarı ömür) ve bölgesel toplu-olay payı eklendi (56 kolon,
ikisi de ufuk=31 kaydırmalı) — sayıların önceki dalgadan (en iyi tekil 312.74)
oynamasının nedeni bu. Örnek ağırlığı (`recency_activity_weights`) tek başına
kazandırmıştı ama Hawkes ile ÇATIŞTI (aynı yenilik sinyali iki kanaldan →
lgb_mae 310.14→335.30); ölçüm sonucu kanonik koşu ağırlıksız.

Dört ölçülmüş ders:

- **Kayıp fonksiyonu model seçiminden önce gelir.** Aynı LightGBM, kayıp
  `l2 → mae` değişince ~94 dk kazanıyor. Yarışma metriği neyse kayıp odur.
- **MAE'nin optimali medyandır ama kazancı feature setine bağlıdır.** Karışımın
  medyanı: p≤0.5 ise tam 0, değilse koşullu dağılımın q\\*=1−0.5/p kantili
  (`conditional_quantile_from_hurdle`). Önceki dalgada eşikli hurdle'ı 4.5 dk
  geçmişti (317.23→312.74); Hawkes feature'ları eklenince fark 0.1 dk'ya indi —
  kural bedava, ama sinyali feature'lar zaten taşıyorsa kazanç erir.
- **Kalibrasyonu varsaymadık, ölçtük.** "Eşik 0.680, sınıflandırıcı kalibresiz
  olduğu için 0.5'ten sapmış olabilir" hipotezini izotonik kalibrasyonla test
  ettik: Brier kötüleşti, MAE kötüleşti. Eşik sapması verinin gerçeği.
- **Bu tablonun ilk sürümü sızıntılıydı ve bunu çekişmeli denetim yakaladı.**
  Ham kaydın `id` kolonu panel dolgusunun birebir kopyası çıktı (y==0 ile uyum
  0.9975) ve tüm skorları ~60 dk iyimser gösterdi. Yukarıdaki sayılar, `id`
  dahil ham olay kolonlarının tamamı sızıntı duvarının arkasına alındıktan
  sonraki dürüst ölçümdür.
"""),
    code("""
y = train_features[TARGET].to_numpy()

# Kayip = yarisma metrigi. Gercek GDZ olcumu: ayni LightGBM'de l2 -> mae
# gecisi ~94 dk kazandirdi (experiments/benchmark_gercek.json).
params = (starter_params("lightgbm", TASK, objective="mae") if METRIC == "mae"
          else starter_params("lightgbm", TASK))

result = cross_validate(
    train_features[FEATURES], y, folds,
    kind="lightgbm", task_type=TASK, metric=METRIC,
    params=params, test=test_features[FEATURES],
    target_transform=("log1p" if LOG_TARGET else None),
    early_stopping_metric=METRIC,
)

print(result.summary())
"""),
    markdown("""
## 4 · Harman — ve kapsam maskesi neden şart

Gerçek GDZ ölçümünde hill-climb harmanı aynı OOF üzerinde tekil modellerden
daha düşük göründü. Bu yalnızca `apparent_oof_best` sonucudur: ağırlıklar aynı
OOF'ta seçildiği için bağımsız, eşleştirilmiş en az 6 outer anchor olmadan
bilimsel kazanan ilan edilmez (`scripts/benchmark_gercek.py`). Bir ölçülmüş ders:
"en iyi 3 üyeyi
harmanla" kısayolu bir önceki dalgada 311.83 verdi, çünkü en iyi üçü birbirinin
kopyası çıktı — harmanı üye kalitesi değil **hata çeşitliliği** taşır; hill-climb
tüm adayları görünce işe yaramayana zaten 0 ağırlık veriyor (lgb_tweedie tekil
6. sıradayken harmanda ağırlık alan iki üyeden biri olması bunun kanıtı).
Ridge stacking ise rekabet dışı: purged şemada ilk dönem hiçbir fold'un valid
tarafına düşmediği için meta-modelin eğitim kapsamı daralıyor. Tercihimiz hill
climbing — ağırlıklar jüriye tek satırda açıklanabilir.

**Kapsam maskesi:** purged bölme ilk dönemi hiçbir valid'e koymaz; o satırların
OOF değeri tahmin değil **dolgudur** (0.0). Maskesiz harman kurmak skoru ölçülü
biçimde şişirir: rmse 2.213 → 2.755, **%24.5 sapma** (`src/gridup/ensemble.py`,
`tests/test_harman_kapsami.py`). Bu yüzden harman/stack her zaman `covered`
maskesi üzerinde kurulur.
"""),
    code("""
zoo = make_model_zoo(train_features[FEATURES], y, folds, metric=METRIC,
                     test=test_features[FEATURES])

# KAPSAM MASKESI SART: purged ilk donemi hicbir fold'un valid tarafina koymaz;
# o satirlarda OOF degeri dolgudur ve harman skorunu %24.5'e kadar sisirir
# (olculdu, ensemble.py). covered_oof_matrix maskeyi otomatik uygular.
kapsanan, oof = zoo.covered_oof_matrix()
y_kapsanan = y[kapsanan]
print(f"OOF kapsami: %{zoo.coverage * 100:.1f}")

secilen = prune_by_correlation(oof, y_kapsanan, max_members=5)
agirliklar = hill_climb_weights({k: oof[k] for k in secilen}, y_kapsanan, metric=METRIC)
print("harman agirliklari:", agirliklar)

# Stacking'i ancak fold kapsami genisse dene -- gercek GDZ'de purged semada
# 645.48 ile rekabet disiydi (benchmark_gercek.json):
# stack = stack_oof(zoo.oof_matrix, y, folds, test_predictions=zoo.test_matrix,
#                   base_covered=zoo.oof_covered)
"""),
    markdown("""
## 5 · Submission

`write_submission` yazmadan önce doğrular: NaN, sonsuz, eksik ID, sabit tahmin,
negatif değer. Kaggle'ın "Submission Scoring Error" mesajı hiçbir şey söylemez;
hatayı gönderMEDEN yakalamak bir submission hakkı kurtarır.
"""),
    code("""
predictions = result.test_predictions

path = write_submission(
    test_features[ID_COLUMN].to_numpy(),
    predictions,
    OUT_DIR / "baseline_lgbm.csv",
    id_column=ID_COLUMN,
    target_column=TARGET,
)
"""),
    markdown("""
## 6 · Deney defteri

Submission gönderdikten **sonra** leaderboard skorunu geri yaz:

```python
store.record_lb(record.run_id, 12.3456)
```

CV–LB korelasyonu bu yarışmanın en önemli tek sayısıdır. r > 0.8 ise CV'ne
güven; r < 0.5 ise CV şemanı düzeltmeden devam etme.
"""),
    code("""
run_recipe = PipelineRecipe(
    seed=42,
    # embargo_days ACIKCA yazilir: fold'lar yukarida
    # embargo=pd.Timedelta(days=max(HORIZON, 30)) ile uretildi. Bu alan
    # bos birakilirsa provenance kaydi "0 gun ambargo" der ve juriye giden
    # notebook, gercekte kosandan BASKA bir semayi belgeler.
    cv=CVRecipe(
        n_splits=len(folds),
        splitter="purged_time_series",
        embargo_days=max(HORIZON, 30),
    ),
    features=FeatureRecipe(
        horizon=HORIZON,
        target_shifts=(HORIZON, 2 * HORIZON, 3 * HORIZON),
    ),
    model=ModelRecipe(
        kind="lightgbm", metric=METRIC, early_stopping_metric=METRIC,
    ),
)
fold_plan = FoldPlan.from_folds(folds, n_rows=len(train_features))
provenance = ExperimentProvenance.capture(
    recipe_fingerprint=run_recipe.fingerprint,
    data_artifacts=[
        DataArtifact.from_path(DATA_DIR / "train.csv"),
        DataArtifact.from_path(DATA_DIR / "test.csv"),
        DataArtifact.from_path(path),
    ],
    feature_names=FEATURES,
    fold_fingerprint=fold_plan.fingerprint,
)
store = SQLiteExperimentStore(OUT_DIR.parent / "experiments" / "experiments.db")
record = store.add(ExperimentRecord(
    name="baseline_lgbm",
    cv_score=result.overall_score,
    metric=METRIC,
    model_kind="lightgbm",
    n_features=len(FEATURES),
    fold_scores=result.fold_scores,
    params=params,
    features=list(FEATURES),
    notes="baseline: takvim + frekans + ufuk-kaydirmali lag",
    submission_path=str(path),
    provenance=provenance,
))

print("run_id:", record.run_id)
pd.DataFrame(store.load()).tail()
"""),
    markdown("""
## 7 · Dürüst sınırlar — bu sayıların söylemediği şeyler

- **CV gürültülü.** Provada tek modelin fold skorları 150.8 → 461.4 dk salındı
  (std 122.35, `scripts/real_data_rehearsal.py`); ablasyonda fold_std 94.31,
  benchmark'ta 22.4–113.1. İki reçete arasındaki 2–3 dakikalık fark hüküm
  değildir; kararlar aile düzeyindeki büyük farklara yaslanır.
- **Seçim yanlılıkları aynı yönde birikir.** Erken durdurma, skorun ölçüldüğü
  fold'da ağaç sayısı seçer (ölçülen sapma %0.16); Optuna araması ~%0.3; SHAP
  geri eleme +0.0137 — üçü de iyimser yönde (`src/gridup` içindeki ölçümler).
  Nihai model kararı ayrılmış bir holdout veya LB doğrulaması ister.
- **İki sızıntıyı kendi denetimimiz yakaladı.** İlk prova aynı günün
  `effectedsubscribers` kolonunu feature almıştı; benchmark'ın ilk sürümünde ise
  ham kaydın `id` kolonu panel dolgusunun birebir kopyası çıktı (y==0 ile uyum
  0.9975) ve tüm skorları ~60 dk iyimser gösterdi. İkisi de çekişmeli denetimle
  bulundu, düzeltildi ve bu sayfadaki sayılar düzeltilmiş ölçümlerdir
  (prova 334.29, güncel harman 302.64). Sızıntı "bizde olmaz" denen şey değil,
  sistematik aranan şeydir.
- **Bu ölçümlerin kapsamı 2021–22 verisidir.** Ablasyon ve benchmark sayıları
  2021-05→2022-08 GDZ kaydında, `kesinti_dk` hedefi ve 47 ilçeyle ölçüldü.
  2026 yarışması muhtemelen farklı hedef ve 96 ilçeyle gelecek — "tatil zarar
  veriyor" gibi sonuçlar oraya taşınmaz, **1. günde yeni veride yeniden
  ölçülür** (`scripts/ablation_gercek.py` hazır, ~10 dk).
- **Public LB bir fold değildir.** LB rastgele bölmeyse zaman-temelli CV ile
  uyuşmayabilir; CV–LB korelasyonu düşükken LB'ye göre model seçmek shakeup'ta
  kaybettirir. Önce şema, sonra karar.
"""),
    markdown("""
## 8 · Veri günü planı

Sıra ölçümden geliyor (`experiments/benchmark_gercek.json` → `gun1_recetesi`):

1. **Saat 0–2 · Keşif:** 01 notebook'u — panel, fold'lar, sızıntı duvarı.
   İlk iki saatin sonunda üç karar da verilmiş olmalı.
2. **İlk submission adayı:** `catboost_mae` — 2023 birincisinin reçetesi,
   3. dalga feature'larıyla görünür OOF lideri (MAE 304.89; hep-sıfır 366.97),
   fakat bağımsız outer kanıt olmadan bilimsel kazanan değildir.
   Optimize etmeden önce LB'de bir sayı:
   ```python
   print(zero_baseline_score(y, metric="mae"))   # once bunu gectigini gor
   ```
   İki aşamalı + medyan kuralını aynı fold'larda ölçün (eşikli 317.04, medyan
   316.95): `q* = 1 − 0.5/p` kantil çözücüsü bedava ama kazancı feature setine
   bağlı — önceki dalgada eşikliyi 4.5 dk geçen fark, Hawkes feature'ları
   sinyali taşıyınca 0.1 dk'ya indi. Merdiven **koşullu** olmalı — marjinal
   `fit_quantile_ladder` burada ölçülmüş şekilde yanlış sonuç verir:
   ```python
   sonuc = fit_two_stage(train_features[FEATURES], y, folds, metric=METRIC)
   merdiven = fit_conditional_quantile_ladder(train_features[FEATURES], y, folds)
   tahmin = conditional_quantile_from_hurdle(sonuc.oof_probability, merdiven)
   ```
   Kalibrasyonu deneme — gerçek GDZ'de izotonik kalibrasyon Brier'i de MAE'yi de
   kötüleştirdi (`calibrate_positive_probability` ölçüp söyler).
3. **Feature'lar ablasyon sırasıyla:** önce lag (+22.34), sonra Hawkes bozunumu
   + toplu-olay payı (`add_event_decay_features` 3g/14g +
   `add_mass_event_features`, ikisi de ufuk şart; birlikte lgb_mae
   323.13→310.14), sonra hava (+2.47; `add_regional_aggregates` +
   `add_physical_derivatives`, ortalama değil `max`/quantile), komşu lag ucuzsa
   (`nearest_neighbours` + `add_neighbour_target_lag`, ufuk şart).
   `recency_activity_weights`'i Hawkes'la BİRLİKTE kullanmayın — aynı yenilik
   sinyali iki kanaldan verilince kaybettirdi (310.14→335.30, ölçüldü); önce
   feature'sız ölçün. Tatil/güneş en sona — gerçek veride negatif ölçüldüler.
   Her adımda kayma kontrolü:
   ```python
   sonuc = adversarial_validation(train_features[FEATURES], test_features[FEATURES])
   print(sonuc["auc"], sonuc["verdict"])   # AUC > 0.8 ise ayristiran feature'i cikar
   ```
4. **Aynı fold'larda** `lgb_mae`, `lgb_tweedie` ve `lgb_sqrt` eklenir
   (`sweep_count_objectives` ile) → TÜM üyeler üzerinde kapsam maskeli,
   kararlılık-cezalı hill-climb harmanı (`stability_penalty=0.5`; gerçek
   GDZ'de 302.64 — catboost_mae 0.75 + lgb_tweedie 0.25; "en iyi 3" kısayolu
   önceki dalgada 311.83 verdi — çeşitlilik kaliteden değerli).
5. **Hiperparametre araması** ancak harman oturduktan sonra — objective de arama
   uzayına girer: `tune_with_optuna(..., search_objective=True)`.
6. **Son gün:** çok tohumlu tam veri refit + jüri çıktıları:
   ```python
   tur = estimate_full_data_rounds(
       extract_best_iterations(result.models), n_folds=len(folds),
       mean_train_fraction=fold_train_fraction(folds, len(train)))
   final = multi_seed_refit(train_features[FEATURES], y, test_features[FEATURES],
                            params=params, n_estimators=tur, seeds=range(15))
   ```
   Feature elemesi gerekirse: `null_importance_filter` (dakikalar) →
   `shap_backward_selection` (saatler).

Bu sayılar gerçek GDZ provasında ölçüldü; yarışma verisinde **yeniden ölçülür**.
Plan sabit, sayılar değişebilir — değişirse karar da değişir.
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

# Juri ciktilari OOF uzerinden hesaplanir: y_true = kapsanan satirlarin gercek
# degeri, y_pred = ayni satirlarin fold-disi tahmini. Kapsanmayan satirlar
# (purged semada ilk donem) DOLGU tasir, skora girmez.
kapsanan, y_pred = result.covered_predictions()
y_true = y[kapsanan]
grup_dilimi = train_features.loc[kapsanan, GROUP_COLUMN]

# 1 · Fold tablosu -- kararliligi gosterir
display(cv_fold_table(result))
plot_fold_scores(result); plt.show()

# 2 · Model NEREDE yaniliyor -- sunumun en ikna edici bolumu
segment_hatasi = error_by_segment(y_true, y_pred, grup_dilimi, metric=METRIC)
plot_error_by_segment(segment_hatasi, metric=METRIC); plt.show()

# 3 · Sinyal nereden geliyor -- 400 satirlik onem listesi yerine aile dagilimi
display(feature_importance_table(result, group_prefixes=(
    "tarih_", "tatil_", "komsu_", "bolge_", "sebep_")))

# 4 · Operasyonel maliyet -- juri bunu soruyor, modeli gercekten calistiracak
print(model_footprint(result.models, elapsed_seconds=result.elapsed_seconds))

# 5 · Is dili -- "MAE 2.95" degil, "ortalama 3 kesinti hatayla tahmin ediyoruz"
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
