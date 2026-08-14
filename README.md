# Grid Up Datathon

**Coderspace × GDZ Elektrik × ADM Elektrik** · 21 Ağustos – 1 Eylül 2026 · Kaggle In-Class

Bu repo, yarışma verisi açıklanmadan **önce** hazırlanmış bir yarışma pipeline'ıdır.
Amaç: 21 Ağustos'ta veri geldiğinde hata ayıklamakla değil, **model geliştirmekle**
başlamak.

---

## Durum

| | |
|---|---|
| Testler | 89 test, tamamı geçiyor (`pytest`) |
| Uçtan uca kanıt | `scripts/smoke_test.py` — sentetik veri üzerinde 14 adım, ~60 sn |
| Sentetik holdout | RMSLE **1.198** vs medyan baseline **1.653** → **%27.6** kazanç |
| Harici veri | Open-Meteo hava durumu çekicisi gerçek veriyle doğrulandı |
| Ortam | Python 3.11.9 · pandas 3.0.3 · LightGBM 4.6 · XGBoost 3.2 · CatBoost 1.2.10 |

---

## Kurulum

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Elle:

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
python scripts\smoke_test.py
```

> **Windows notu:** `python3` bu makinede Microsoft Store kısayoluna gider ve çalışmaz —
> her zaman `python` yaz. Sanal ortam `Scripts/` altındadır, `bin/` değil.

---

## Veri geldiğinde — ilk 30 dakika

```python
from gridup import profile, read_any, suggest_scheme, leakage_report

train = read_any("data/raw/train.csv")     # kodlama/ayırıcı/ondalık otomatik
test  = read_any("data/raw/test.csv")

print(profile(train, test, target="HEDEF").report())   # 1 · elimizde ne var
print(suggest_scheme(train, target="HEDEF"))           # 2 · hangi CV şeması
print(leakage_report(train, "HEDEF", test=test))       # 3 · sızıntı var mı
```

Bu üç çıktı, sonraki 12 günün her kararını belirler. Detaylı akış:
[docs/05-ilk-24-saat.md](docs/05-ilk-24-saat.md)

Sonra `src/gridup/config.py` içindeki `CompetitionConfig` alanlarını doldur —
pipeline'ın geri kalanı dokunulmadan çalışır.

---

## Yapı

```
src/gridup/
  turkish.py        İ/I tuzağı, join anahtarı, TR sıralama, kolon normalizasyonu
  compat.py         pandas 3.0 / numpy 2.x uyumluluk katmanı, bellek düşürme
  io_utils.py       kodlama + ayırıcı + ondalık otomatik tespiti (cp1254, ';', '1.234,56')
  profiling.py      otomatik EDA raporu — çarpıklık, sıfır yığılması, ID kolonları, şema farkı
  validation.py     CV şeması seçimi, ambargolu zaman bölmesi, adversarial validation, sızıntı taraması
  features/
    temporal.py     takvim, döngüsel kodlama, TR tatil, lag, kayan/genişleyen pencere
    categorical.py  frekans, sayım, fold-dışı hedef kodlama, nadir kategori birleştirme
    aggregate.py    grup istatistikleri, sapma/oran/z-skor, oran feature'ları
  metrics.py        RMSE/RMSLE/MAE/MAPE/SMAPE/AUC/F1 + eşik optimizasyonu + log dönüşümü
  models.py         LightGBM/XGBoost/CatBoost tek arayüz, OOF + test tahmini + feature önemi
  ensemble.py       tepe tırmanma ağırlıkları, açgözlü seçim, sıra ortalaması, korelasyon
  submission.py     yazmadan önce doğrulama (NaN, ∞, eksik ID, sabit tahmin, negatif)
  experiment.py     JSONL deney defteri + CV↔LB korelasyon takibi
  synthetic.py      sentetik dağıtım şebekesi verisi (pipeline'ı veriden önce kanıtlar)

notebooks/          01_kesif.ipynb · 02_baseline.ipynb
scripts/            smoke_test.py · fetch_weather.py · build_notebooks.py
docs/               yarışma brifingi, strateji, runbook
tests/              89 test — sızıntı korumaları, TR metin, uçtan uca
```

---

## Tasarım sözleşmesi

Bu dört kural pipeline'ın tamamında geçerlidir ve testlerle korunmaktadır:

1. **Feature fonksiyonları girdiyi değiştirmez** — her zaman yeni DataFrame döner.
2. **Hedef kullanan her kodlama fold-dışıdır.** `oof_target_encode` fold verilmezse
   çalışmaz, hata fırlatır — sızıntılı kodlama üretmek imkânsızdır.
3. **Kayan pencereler mevcut satırı dışlar** (`shift(1)`). pandas varsayılanı dahil eder;
   bu, hedef türevli bir kolonda doğrudan sızıntıdır.
4. **Sabitler `config.py` içinde yaşar.** Notebook'ta sihirli sayı yok.

---

## Bilinen tuzaklar

**Türkçe `İ`** — `'İ'.lower()` tek karakter değil **iki kod noktası** üretir
(U+0069 U+0307), dolayısıyla `'İ'.lower() != 'i'`. İl/ilçe adıyla yapılan bir join
istisna fırlatmadan **0 satır** döner. Her zaman `gridup.turkish.join_key()` kullan.

**pandas 3.0** — `applymap`, `append`, `np.NaN`, `np.float_` **kaldırıldı**.
Copy-on-Write her zaman açık, zincirli atama sessizce etkisiz. Düz metin kolonları
artık `object` değil `str` dtype — `is_object_dtype()` onları görmez.
`gridup.compat.is_categorical_like()` her iki pandas sürümünde de doğru çalışır.

**Türk CSV'leri** — `;` ayırıcı, `,` ondalık, `.` binlik, `cp1254` kodlama.
`1.234.567,89` **tek sayıdır**. `read_any()` bunu otomatik halleder.

**Kaggle'da internet kapalı olabilir** — harici veriyi yerelde indirip Kaggle Dataset
olarak yükle. Notebook içinde canlı API çağırma.

---

## Dokümanlar

| Doküman | İçerik |
|---|---|
| [00-yarisma-brief.md](docs/00-yarisma-brief.md) | Takvim, kurallar, diskalifiye riskleri, takım kurma |
| [05-ilk-24-saat.md](docs/05-ilk-24-saat.md) | Veri geldiğinde saat saat ne yapılacak |
| [06-teknik-tuzaklar.md](docs/06-teknik-tuzaklar.md) | Ortam, sürüm, kodlama, platform tuzakları |

---

## Lisans / atıf

Harici veri kaynakları kendi lisanslarına tabidir. Open-Meteo verisi CC-BY-4.0 — notebook'ta
atıf ver.
