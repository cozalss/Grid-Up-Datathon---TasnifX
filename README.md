# Grid Up Datathon

**Coderspace × GDZ Elektrik × ADM Elektrik** · 21 Ağustos – 1 Eylül 2026 · Kaggle In-Class

Bu repo, yarışma verisi açıklanmadan **önce** hazırlanmış bir yarışma pipeline'ıdır.
Amaç: 21 Ağustos'ta veri geldiğinde hata ayıklamakla değil, **model geliştirmekle**
başlamak.

---

## Durum

| | |
|---|---|
| Testler | **426 test**, tamamı geçiyor (`pytest`) · ruff temiz |
| Uçtan uca kanıt | `scripts/smoke_test.py` — sentetik veri üzerinde 14 adım, ~60 sn |
| Sentetik holdout | RMSLE **1.200** vs medyan baseline **1.653** → **%27,4** kazanç |
| Bağımsız denetim | 3 + 7 mercekli çekişmeli denetim — bulgular kapatıldı, çürütülenler atıldı |
| Araştırma | 13 agent'lık derin araştırma → [docs/01-strateji-brifingi.md](docs/01-strateji-brifingi.md) |
| Önceki yarışma | 2023 GDZ Datathon birincisinin çözümü + 558 satırlık forum dökümü incelendi |
| Ölçek provası | `scripts/scale_rehearsal.py` — 100k ve 500k satırda süre/bellek ölçüldü |
| Harici veri | Open-Meteo hava durumu çekicisi gerçek veriyle doğrulandı |
| Yerel ortam | Python 3.11.9 · pandas 3.0.3 · numpy 2.4.6 · sklearn 1.8.0 |
| **Kaggle ortamı** | Python 3.12 · pandas 3.0.4 · **numpy 2.0.2** · sklearn 1.9.0 — numpy Kaggle'da **daha eski** |

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
print(suggest_scheme(train, target="HEDEF", test=test))# 2 · hangi CV şeması
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
  panel.py          olay kayıtlarından tam panel (eksik "olay olmadı" günleri sıfırla doldurur)
  profiling.py      otomatik EDA raporu — çarpıklık, sıfır yığılması, ID kolonları, şema farkı
  validation.py     CV şeması seçimi, ambargolu zaman bölmesi, adversarial validation, sızıntı taraması
  features/
    temporal.py     takvim, döngüsel kodlama, TR tatil, ufuk-farkındalıklı lag ve kayan pencere
    categorical.py  frekans, sayım, fold-dışı hedef kodlama, nadir kategori birleştirme
    aggregate.py    grup istatistikleri, sapma/oran/z-skor, oran feature'ları
    solar.py        güneş geometrisi + açık-hava ışınımı (pvlib) — sızıntısız, deterministik
  metrics.py        RMSE/RMSLE/MAE/MAPE/SMAPE/AUC/F1 + eşik optimizasyonu + log dönüşümü
  models.py         LightGBM/XGBoost/CatBoost tek arayüz, OOF + test tahmini + feature önemi
  ensemble.py       tepe tırmanma ağırlıkları, açgözlü seçim, sıra ortalaması, korelasyon
  submission.py     yazmadan önce doğrulama (NaN, ∞, eksik ID, sabit tahmin, negatif)
  experiment.py     JSONL deney defteri + CV↔LB korelasyon takibi
  synthetic.py      sentetik dağıtım şebekesi verisi (pipeline'ı veriden önce kanıtlar)
  ablation.py       feature grubu ablasyonu + dayanıklılık harmanı (2023 birincisinin mimarisi)
  neural.py         varlık gömülü sinir ağı — harmana çeşitlilik üyesi

notebooks/          01_kesif.ipynb · 02_baseline.ipynb
scripts/            smoke_test.py · full_pipeline.py · day_one.py · scale_rehearsal.py
                    build_kaggle_package.py · fetch_weather.py · build_notebooks.py
docs/               yarışma brifingi, strateji, runbook
tests/              426 test — sızıntı korumaları, TR metin, tasarım sözleşmeleri, uçtan uca
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

## Araştırmadan gelen üç kritik kural

**1 · Lag ufkunu tahmin ufkuna sabitleyin.** 2024 birincisinin en yüksek önemli
feature'ları `shift_29_rolling_3_sum` idi. Test bloğu bir aylıksa, ayın 28'ini tahmin
ederken 1 günlük lag **yoktur**. `shift(1)` ile hesaplanan rolling CV'de harika görünür,
private LB'de çöker.

```python
HORIZON = (test[TIME].max() - test[TIME].min()).days + 1
out = add_lag_features(out, TARGET, [1, 7, 28], time_column=TIME,
                       group_columns=[GROUP], horizon=HORIZON)
```

**2 · Panel yapısını sıfırla doldurun.** "O gün kesinti olmadı" satırları veri setinde
bulunmayabilir. Doldurulmazsa lag/rolling kayar ve model sıfır tahmin etmeyi öğrenemez.

```python
panel = build_panel(events, entity_columns=["il", "ilce"], time_column="tarih")
```

**3 · Havada ortalama değil max ve quantile.** Hasarı rüzgârın ortalaması değil tepesi
yapar. 2024 birincisinin importance listesinin tepesi `wind_speed_10m_max` ve `..._q01`
gibi quantile türevleriyle doluydu.

Ayrıntı: [docs/01-strateji-brifingi.md](docs/01-strateji-brifingi.md)

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
