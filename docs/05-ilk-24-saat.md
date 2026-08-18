# İlk 24 Saat — Runbook

Veri 21 Ağustos'ta, açılış buluşmasından sonra geliyor. Bu doküman o günün saat saat
planıdır. **Amaç: ilk 6 saatte leaderboard'da bir skor.**

Neden bu kadar acele: erken bir baseline, sonraki her deneyin karşılaştırılacağı zemini
kurar. Zemin yoksa 12 gün boyunca "iyileşiyor muyuz?" sorusuna cevap veremezsin.

---

## Saat 0 — Açılış buluşması (14:00–15:00)

YouTube sohbetinden [00-yarisma-brief.md](00-yarisma-brief.md) sonundaki 10 soruyu sor.
En kritik üçü:

1. **Resmi metrik nedir?** → tüm strateji buna bağlı
2. **Harici veri serbest mi?** → hava durumu en güçlü sinyal olabilir
3. **Kaggle takım kurma son tarihi?** → kaçırırsan takım kurulamaz

Aynı anda: Kaggle linki gelir gelmez **herkes yarışmaya katılsın ve kuralları kabul etsin**,
sonra takımı birleştirin. Bunu ertelemeyin.

---

## Saat 1 — Veriyi tanı (⏱ 45 dk)

```powershell
cd c:\Users\cemmo\Documents\Datahon
$env:PYTHONIOENCODING='utf-8'
```

Veri dosyalarını `data/raw/` altına koy, sonra `notebooks/01_kesif.ipynb` aç:

```python
train = read_any(DATA_DIR / "train.csv")
test  = read_any(DATA_DIR / "test.csv")

print(profile(train, test, target=TARGET).report())
```

**Bu rapordan çıkarman gerekenler — hepsini bir yere yaz:**

| Soru | Nerede görürsün |
|---|---|
| Hedef regresyon mu sınıflandırma mı? | `target_summary.gorev_tahmini` |
| Hedef çarpık mı? (→ log1p) | `carpiklik > 2` |
| Sıfır yığılması var mı? (→ iki aşamalı model) | `sifir_orani > 0.4` |
| Sınıf dengesizliği? (→ eşik optimizasyonu) | `sinif_dagilimi` |
| Zaman kolonu var mı? | `time_columns` |
| Şema farkı = sızıntı adayları | `schema_diff.train_only` |
| Yüksek kardinaliteli kolonlar (→ hedef kodlama) | işaretlenmiş kolonlar |
| ID-benzeri kolonlar (→ feature yapma) | işaretlenmiş kolonlar |

---

## Saat 2 — CV şemasını **sabitle** (⏱ 30 dk)

Bu, yarışmanın en önemli tek kararıdır. Yanlışsa geri kalan 11 gün boşa gider.

```python
print(suggest_scheme(train, target=TARGET, task_type=TASK))
print(leakage_report(train, TARGET, test=test, time_column=TIME_COLUMN))
```

**Karar ağacı:**

```
Test, train'den SONRAKİ bir dönem mi?
  EVET → purged_time_series_split (ambargo = en uzun rolling penceresinden büyük)
  HAYIR ↓
Tekrarlayan varlık var mı (trafo/abone/fider)?
  EVET → GroupKFold (dengesizse StratifiedGroupKFold)
  HAYIR ↓
Sınıflandırma mı?
  EVET → StratifiedKFold    HAYIR → KFold
```

**`critical` sızıntı bulgusu varsa DUR.** Çözmeden ilerleme — o sızıntıyla eğitilen her
model yanlış yönde optimize edilir.

Fold'ları bir kez üret ve **sabitle**. Tüm deneyler aynı bölmeler üzerinde karşılaştırılmalı;
yoksa iki deneyin skorlarını karşılaştıramazsın.

---

## Saat 3 — İlk submission (⏱ 60 dk)

`notebooks/02_baseline.ipynb`. Hedef: **çalışan bir uçtan uca hat**, iyi skor değil.

```python
result = cross_validate(train[FEATURES], y, folds, kind="lightgbm",
                        metric=METRIC, test=test[FEATURES])
write_submission(test[ID], result.test_predictions, "submissions/baseline.csv",
                 sample=sample_submission)
```

**Kaggle'a gönder.** Skoru gördüğün an iki şeyi biliyorsun:
1. Submission formatı doğru (bu tek başına bir gündür, ilk gün halletmek altın değerinde)
2. CV ile LB arasındaki fark ne kadar

```python
log.record_lb("baseline", <LB_SKORU>)
```

---

## Saat 4 — Adversarial validation (⏱ 30 dk)

Train ve test aynı dağılımdan mı geliyor? Bu, **hiç submission harcamadan** test
dağılımını öğrenmenin yoludur.

```python
sonuc = adversarial_validation(train[FEATURES], test[FEATURES])
print(sonuc["auc"], sonuc["verdict"])
```

| AUC | Anlamı | Aksiyon |
|---|---|---|
| ~0.5 | dağılımlar aynı | rastgele CV güvenli |
| 0.6–0.8 | orta kayma | `top_features` incele, zaman bazlı CV düşün |
| > 0.8 | ciddi kayma | ayrıştıran feature'ları çıkar veya `sample_weights` kullan |
| ~1.0 | bir kolon mükemmel ayırıyor | o kolonu feature yapma |

---

## Saat 5–6 — Feature'lar (⏱ 90 dk)

Sırayla, her adımdan sonra CV'yi ölç ve deftere yaz. **Aynı anda birden fazla şey değiştirme** —
hangisinin işe yaradığını bilemezsin.

1. **Takvim** (zaman varsa) — `add_calendar_features`, `add_turkish_holiday_features`
   TR dini bayramları hicri takvimle **kayar**; `ay + gün` kolonları onları yakalayamaz.
2. **Grup istatistikleri** — `add_group_statistics(frame, ["ilce"], ["tuketim"])`
   Satırın kendi grubuna göre sapması, ham değerden genellikle daha güçlüdür.
3. **Frekans/sayım kodlama** — `add_frequency_encoding` (hedef kullanmaz, sızıntısız)
4. **Lag / rolling** (zaman + varlık varsa) — en güçlü aile
5. **Fold-dışı hedef kodlama** — `oof_target_encode` (yüksek kardinaliteli kolonlar)

---

## Saat 7+ — Harici veri

```powershell
python scripts\fetch_weather.py --start 2020-01-01 --end 2026-09-01 --all-districts
```

**Join'i mutlaka `join_key` ile yap:**

```python
# Hava parquet'i ILCE bazli: anahtar ilce_key. (konum_key "il-ilce" bilesigidir,
# il_key ile eslesmez -- olculdu: %0 eslesme.)
from gridup.turkish import join_key, strip_qualifier
train["ilce_key"] = train["ilce"].map(lambda ad: join_key(strip_qualifier(ad)))
train["tarih"] = pd.to_datetime(train["tarih"]).dt.normalize()
merged = train.merge(hava, on=["ilce_key", "tarih"], how="left")

# Kaç satır eşleşti? SESSİZ SIFIR EŞLEŞMEYE KARŞI KONTROL:
print(merged["sicaklik_ort"].notna().mean())   # ~1.0 olmalı
```

Bu oran düşükse `diagnose_join()` çalıştır.

---

## Gün sonu kontrol listesi

- [ ] Leaderboard'da bir skor var
- [ ] CV şeması sabitlendi ve gerekçesi yazıldı
- [ ] `critical` sızıntı bulgusu kalmadı
- [ ] Adversarial validation AUC'si biliniyor
- [ ] En az 3 deney deftere yazıldı, LB skorları geri işlendi
- [ ] Takım Kaggle'da birleşti
- [ ] Notebook okunabilir durumda (jüri bunu okuyacak)

---

## 12 günlük kaba dağılım

| Gün | Odak |
|---|---|
| 1 | Keşif, CV şeması, ilk submission |
| 2–3 | Feature mühendisliği, harici veri |
| 4–5 | Model çeşitliliği (LGBM + CatBoost + XGB), hata analizi |
| 6–7 | Hiperparametre araması (Optuna), eşik optimizasyonu |
| 8–9 | Harmanlama, CV↔LB korelasyon kontrolü |
| 10–11 | Hata analizi → son feature turu, notebook temizliği |
| 12 | **Final submission seçimi**, notebook + writeup kilitleme |

**Son gün kuralı:** yeni fikir denemeyi bırak. Son gün yapılan değişikliklerin çoğu
public leaderboard'a uyum sağlar ve private'da zarar verir.

**Final submission seçimi:** iki farklı aile seç — en iyi CV'li tek model ve en iyi harman.
İkisi de aynı aileden olursa risk çeşitlendirilmemiş olur.

```python
print(log.cv_lb_correlation())   # r > 0.8 ise CV'ne güven
```
