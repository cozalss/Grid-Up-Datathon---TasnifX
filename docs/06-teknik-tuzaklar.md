# Teknik Tuzaklar

Bu dosyadaki her madde **bu makinede ölçülmüştür**, varsayılmamıştır. Tarih: 14 Ağustos 2026.
Ortam: Windows 11 Pro 26200, Python 3.11.9, Türkçe locale (cp1254).

---

## 1 · Türkçe `İ` — sessiz join katili

Python'un `.lower()` metodu **locale'den bağımsızdır** ve Türkçe eşleme yapmaz.

```python
'İ'.lower()          # 'i̇'  = U+0069 U+0307  (2 kod noktası!)
'İ'.lower() == 'i'   # False
'ISTANBUL'.lower()   # 'istanbul'  — Türkçe doğrusu 'ıstanbul'
```

`casefold()` ve NFC normalizasyonu bunu **düzeltmez**.

### Neden bu yarışmada önemli

GDZ (İzmir, Manisa) ve ADM (Aydın, Denizli, Muğla) bölgesi il/ilçe adları `İ` doludur.
Harici veriyi (hava durumu, nüfus) il adıyla join ederken:

```python
a = {"İSTANBUL": 34, "İZMİR": 35, "DİYARBAKIR": 21}
b = ["istanbul", "izmir", "diyarbakir"]
naive = {k.lower(): v for k, v in a.items()}
# eşleşen: 0 / 3 — istisna yok, uyarı yok
```

**Sıfır eşleşme, sessizce.** Bu davranış `smoke_test.py` adım 2'de kanıtlanmıştır.

### Çözüm

```python
from gridup.turkish import join_key, tr_lower, diagnose_join

join_key("MUĞLA") == join_key("Mugla")   # True
diagnose_join(sol_anahtarlar, sag_anahtarlar)   # merge'den ÖNCE çalıştır
```

**Kural:** Bir merge beklenenden az satır döndürüyorsa, ilk bakacağın şey U+0307'dir.

---

## 2 · pandas 3.0 — kaldırılan API'ler

Bu makinede ölçülen (pandas 3.0.3, numpy 2.4.6):

| API | Durum | Yerine |
|---|---|---|
| `DataFrame.applymap` | **KALDIRILDI** | `.map` |
| `DataFrame.append` | **KALDIRILDI** | `pd.concat([...])` |
| `np.NaN` | **KALDIRILDI** | `np.nan` |
| `np.float_` | **KALDIRILDI** | `np.float64` |
| Copy-on-Write | **her zaman açık** | kapatılamaz |

**Copy-on-Write'ın sonucu:** zincirli atama artık **sessizce hiçbir şey yapmaz**:

```python
df[df.a > 1]['b'] = 0        # SESSİZCE ETKİSİZ
df.loc[df.a > 1, 'b'] = 0    # doğru
```

2023 öncesi her Kaggle notebook'u ve öğreticisi ilk kalıbı kullanır.

### Kaggle uyumsuzluğu — asıl risk

Kaggle imajı genellikle **daha eski** bir pandas (2.x) taşır. Yerelde çalışan kod
Kaggle'da çökebilir veya tersi. Kod her iki sürümde de çalışmalı.

Kaggle'dan kopyalanan bir hücreyi yapıştırmadan önce:

```python
from gridup.compat import assert_no_removed_api
print(assert_no_removed_api(hucre_kaynagi))   # boş liste = temiz
```

---

## 3 · pandas 3.0 — metin artık `str` dtype

Bu makinede ölçülen dtype yüklemleri:

| Seri | dtype | `is_object_dtype` | `is_string_dtype` |
|---|---|---|---|
| `pd.Series(["a"])` | `str` | **False** | True |
| `.astype("category")` | `category` | False | True |
| `dtype=object` | `object` | True | **False** |

pandas 3.0'da düz metin kolonları artık `object` **değil** `str` dtype'ındadır ve
`is_object_dtype()` onları **görmez**. pandas 2.x'te tam tersi.

**Bu hata bu oturumda gerçekten yaşandı:** `adversarial_validation` metin kolonlarını
kategoriye çevirmeyi atladı ve LightGBM 10 dakika sonra
`ValueError: pandas dtypes must be int, float or bool` fırlattı.

### Çözüm

```python
from gridup.compat import is_categorical_like, categorical_columns
```

Her iki sürümde de doğru çalışır; bool ve datetime'ı bilerek dışlar.

---

## 4 · Türk CSV dosyaları

Türkçe locale ondalık ayırıcı olarak `,` kullanır, dolayısıyla Excel CSV'yi `;` ile yazar:

```
İL;TÜKETİM;ORAN
İzmir;1.234.567,89;12,5
```

`1.234.567,89` **tek sayıdır** — `.` binlik ayırıcıdır.

```python
pd.read_csv(p)                                          # BOZULUR
pd.read_csv(p, sep=";", decimal=",", thousands=".",     # doğru
            encoding="cp1254")
```

**Kodlama:** TÜİK / e-Devlet / SAP ihraçları genellikle `cp1254` veya `ISO-8859-9`.
Bu ikisi yalnızca birkaç konumda farklıdır — yanlış olanla okunan dosya **makul görünen**
metin üretir, birkaç karakteri yanlış olur. Çökmekten kötüdür.

**Tespit sırası önemli:** önce `utf-8` dene (cp1254 girdide gürültülü başarısız olur),
sonra `cp1254`. Ters sırada cp1254 neredeyse her baytı kabul eder ve sessizce mojibake üretir.
`utf-8-sig` en başta olmalı — Excel BOM ekler.

`gridup.io_utils.read_any()` bunların hepsini otomatik yapar ve ne bulduğunu **yazdırır**.

---

## 5 · Windows / PowerShell

| Refleks | Bu makinede | Yerine |
|---|---|---|
| `python3` | Microsoft Store kısayolu, **çalışmaz** | `python` |
| `.venv/bin/activate` | **yok** | `.venv\Scripts\Activate.ps1` |
| `cmd1 && cmd2` | PS 5.1'de **parser hatası** | `cmd1; if ($?) { cmd2 }` |
| `a ?? b`, `x ? y : z` | PS 5.1'de **parser hatası** | `if/else` |
| `open(p)` (encoding'siz) | **cp1254** okur/yazar | `encoding="utf-8"` |
| Türkçe karakter yazdırma | `UnicodeEncodeError` | `$env:PYTHONIOENCODING='utf-8'` |

`gh` CLI bu makinede **kurulu değil**. GitHub işlemleri için REST API + `curl` kullan.

**Uzun yol sınırı aktif** (`LongPathsEnabled = 0`): projeyi sürücü köküne yakın tut.
`.venv/Lib/site-packages/...` altındaki iç içe bağımlılık ağaçları 260 karakteri aşabilir.

---

## 6 · Kaggle ortamı

**İnternet kapalı olabilir.** Harici veri akışı:

1. Yerelde indir → `data/external/hava_gunluk.parquet`
2. Kaggle'a **Dataset** olarak yükle
3. Notebook'a input olarak ekle ve oku

Notebook içinde canlı API çağırmak, yarışmanın son günü sessizce çökmenin en hızlı yoludur.

**Bellek:** Kaggle notebook'ları 16–30 GB RAM ile sınırlı. Büyük veri setinde
`gridup.compat.reduce_memory()` çalıştır — düşük kardinaliteli metin kolonlarını
`category`ye çevirir, bu hem bellek kazandırır hem LightGBM'in yerel kategorik desteğini açar.

**Yol:** `/kaggle/input/<yarisma-slug>/` okuma, `/kaggle/working/` yazma.
`gridup.config.Paths.for_kaggle(slug)` bunu halleder.

---

## 7 · Determinizm

`set_global_seed(42)` Python, numpy, PYTHONHASHSEED ve varsa torch'u sabitler.

**Ama:** LightGBM ve XGBoost `num_threads > 1` ile **bit düzeyinde determinizmi garanti
etmez**. Tam tekrarlanabilirlik gerekiyorsa LightGBM'de `deterministic=True` + tek iş
parçacığı kullan — yavaşlama pahasına.

Jüri tekrarlanabilirliğe bakıyor. Notebook'un ilk hücresinde `environment_report()`
çıktısını bırakmak ucuz bir puandır.
