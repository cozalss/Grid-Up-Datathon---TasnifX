# Veri Günü Kontrol Listesi

21 Ağustos, saat 15:00. Kaggle linki geldi. Bu dosya o andan itibaren
izlenecek adımları içerir.

---

## Hazır olanlar (bugün itibarıyla)

| Varlık | Yer | Durum |
|---|---|---|
| Hava durumu, 2020–2026 | `data/external/hava_gunluk.parquet` | **96 ilçe**, 231.648 satır, sıfır eksik |
| Güneş fiziği, 2020–2026 | `data/external/gunes_gunluk.parquet` | 96 ilçe × 2435 gün, sıfır eksik |
| 96 ilçe + koordinat + nüfus | `data/reference/ilceler_gdz_adm.parquet` | Komşuluk grafiği doğrulandı |
| Arıza sebebi taksonomisi | `gridup.features.outage_reason` | 919 metin → 22 aile, %0,88 sınıflanamayan |
| Uçtan uca betik | `scripts/day_one.py` | Sentetik veride 6 sn'de submission |
| Kaggle offline paketi | `kaggle.com/datasets/cemzal/gridup-offline-paket` | Yüklendi (özel), dosyalar doğrulandı |
| Test paketi | `tests/` | 575 test |

---

## 2023 yarışmasından doğrulanan kurallar

Bunlar tahmin değil; 2023 GDZ Elektrik Datathon'unun forum kayıtlarından ve
Kaggle API'sinden **doğrulandı**. 2026 için garanti değil ama en olası
varsayımlar bunlar — açılış yayınında teyit ettirin.

| Kural | 2023'teki hali | Sonucu |
|---|---|---|
| Public/Private ayrımı | **Rastgele %50/%50 satır**, zamansal değil | Public LB **güvenilir sinyal**; shakeup riski düşük |
| Günlük submission | **3** (GDZ'nin üç yarışmasında da) | Toplam ~36 deneme |
| Metrik | MAPE | Hedefte sıfır varsa `mape()` uyaracak |
| Harici veri | Serbest | Hava + güneş verimiz hazır |
| Test bloğu | İleri zamanlı pencere (1 ay) | `test_span=` ile birebir taklit edilir |

> Public LB rastgele bölmeyse **overfit etmek kolaydır**: 36 denemenin
> hepsini LB'ye bakarak seçerseniz private'ta düşersiniz. Seçimi **CV'ye**
> yaptırın, LB'yi yalnızca doğrulama olarak kullanın.

---

## Zaman bütçesi (ölçüldü, `scripts/scale_rehearsal.py`)

100 bin ve 500 bin satırda ölçüldü (`scripts/scale_rehearsal.py`):

| İşlem | 100k | 500k |
|---|---|---|
| LightGBM tam CV (3 fold, 500 ağaç) | 4,8 sn | 16 sn |
| CatBoost tam CV (500 iter) | 37,6 sn | 118 sn — **7 katı** |
| Sinir ağı CV (60 epok) | 10,5 sn | 66 sn |
| Model zoo (3 model × 3 fold) | 86,7 sn | **328 sn** ← en yavaş |
| Optuna tek deneme | 22,1 sn | 49,6 sn |
| SHAP önem | 0,8 sn | 0,9 sn |

**Bağlayan kısıt hesaplama değil, sizin dikkatiniz.** 60 saatlik yarışmada
makine binlerce koşu çıkarabilir ama insan ~140 hipotez kurup yorumlayabilir.
Sonuç: Optuna'yı gece boyu çalıştırın, gündüzü hipotez seçmeye ayırın.

Araştırmayı **LightGBM ile** yapın; CatBoost'u yalnızca final harmana katın.

---

## Saat 0 — Açılış yayını (14:00)

Sorulacak on soru: [00-yarisma-brief.md](00-yarisma-brief.md) sonu.
**En kritik üçü:** resmî metrik · harici veri serbest mi · ilk 10 mu ilk 20 mi.

Aynı anda: Kaggle linki gelir gelmez **herkes yarışmaya katılsın**, kuralları
kabul etsin. **Takım birleşmeden kimse submission yapmasın.**

---

## Saat 1 — İlk komut

Veriyi `data/raw/` altına koyun, sonra:

```powershell
cd c:\Users\cemmo\Documents\Datahon
python scripts\day_one.py --data data\raw
```

Betik `sample_submission.csv`'den hedef ve ID kolonunu **kendisi çıkarır**.
Zaman ve grup kolonunu bildiğinizde ekleyin:

```powershell
python scripts\day_one.py --data data\raw --time TARIH --group ILCE --metric mae
```

**Çıktıdan not alınacaklar** — bunlar sonraki 12 günün kararlarını belirler:

| Soru | Nerede görünür |
|---|---|
| Hedef sayım mı sürekli mi? | `gorev_tahmini` |
| Sıfır oranı? | `sifir_orani` — %40 üstüyse iki aşamalı modeli düşün |
| Çarpıklık? | `carpiklik` — 2 üstüyse log1p |
| Tahmin ufku kaç gün? | 5/7 aşamasında yazdırılır |
| Kritik sızıntı var mı? | 3/7 aşaması |
| Panel doluluk oranı? | 4/7 aşaması |

---

## Saat 2 — Şema doğrulaması

Veri geldiğinde **ilk kontrol edilecek dört şey:**

**1 · Hedef "manevra" kayıtlarını sayıyor mu?**
EPİAŞ verisinde en sık kayıt `-SCADA - MANEVRA` idi — bu bir arıza değil,
operatörün şebeke manevrası. `Borçtan Kesme` de öyle. Yarışma hedefi bunları
içeriyorsa modelleme tamamen değişir.

```python
from gridup.features import reason_family_report
print(reason_family_report(train["ARIZA_SEBEBI"]))
```

**2 · İlçe adları tablomuzla eşleşiyor mu?**

```python
from gridup.turkish import diagnose_join, join_key
import pandas as pd
ref = pd.read_parquet("data/reference/ilceler_gdz_adm.parquet")
print(diagnose_join(train["ILCE"].unique(), ref["ilce"].unique()))
```

Eşleşmeyen varsa **2012 büyükşehir yasası** şüphelisi: Efeler, Şehzadeler,
Yunusemre, Seydikemer gibi ilçeler 2012'de kuruldu; eski kayıtlar "Merkez"
diyor olabilir.

**3 · Hava verisi join oluyor mu?**

```python
hava = pd.read_parquet("data/external/hava_gunluk.parquet")
train["il_key"] = train["IL"].map(join_key)
merged = train.merge(hava, left_on=["il_key","TARIH"],
                     right_on=["konum_key","tarih"], how="left")
print(merged["sicaklik_ort"].notna().mean())   # ~1.0 olmalı
```

**4 · Aykırı değerler**
EPİAŞ verisinde maksimum kesinti süresi **39.084 dakika (27 gün)** çıktı.
Etkilenen abone medyanı 97 ama maksimum 92.449. Aynı desenleri burada arayın —
ASHRAE dersi: ilk 5 takımın *hepsi* feature çokluğu yerine aykırı değer
tespitine öncelik verdi.

---

## Saat 3 — İlk submission

`day_one.py` zaten bir tane üretti. Kaggle'a gönderin. Amaç skor değil,
**format doğrulaması** — bu tek başına bir gün kazandırır.

Sonra LB skorunu deftere yazın:

```python
from gridup.experiment import ExperimentLog
log = ExperimentLog("experiments/deneyler.jsonl")
log.record_lb("gun1_baseline", <LB_SKORU>)
```

---

## Saat 4+ — Sırayla ekleyin

Her adımdan sonra CV'yi ölçün ve deftere yazın. **Aynı anda birden fazla şey
değiştirmeyin** — hangisinin işe yaradığını bilemezsiniz.

1. **Adversarial validation** — `adversarial_validation(train, test)`.
   AUC > 0,8 ise ciddi kayma var, ayrıştıran feature'ları inceleyin.

2. **Ufuk-farkındalıklı lag/rolling** — en güçlü aile:
   ```python
   out = add_lag_features(out, TARGET, [1,7,28], time_column=TIME,
                          group_columns=[GROUP], horizon=HORIZON)
   ```
   `horizon` **zorunlu**. Test bloğu bir aylıksa 1 günlük lag yoktur.

3. **Hava** — ortalama değil `max` ve quantile. Bölge geneli agregat da ekleyin:
   `add_regional_aggregates`, `add_physical_derivatives`.

4. **Komşu ilçe** — `nearest_neighbours` + `add_neighbour_target_lag(horizon=H)`.

5. **Model çeşitliliği** — LightGBM + CatBoost + XGBoost, aynı fold'lar.
   Sayım hedefiyse `COUNT_OBJECTIVES` ile poisson/tweedie/mae süpürün.

6. **Feature seçimi** — önce `null_importance_filter` (dakikalar), sonra
   `shap_backward_selection` (saatler).

7. **Harmanlama** — `hill_climb_weights` OOF üzerinde.

8. **Son gün** — `multi_seed_refit` + `postprocess_predictions`.

---

## Kırmızı bayraklar

| Gördüğünüz | Anlamı |
|---|---|
| Join sonrası satır sayısı düştü | Türkçe `İ` tuzağı — `diagnose_join` çalıştırın |
| CV mükemmel, LB berbat | Sızıntı. Önce ufuk, sonra origin, sonra fold hizalaması |
| `fold_std` skorun %10'undan büyük | CV gürültülü — küçük iyileşmelere güvenmeyin |
| Model "hep sıfır" baseline'ını geçemiyor | Sorun modelde değil yaklaşımda |
| Feature önemleri hep 0 | Önem çıkarımı bozuk, feature'lar değil |
| CV–LB korelasyonu r < 0,5 | **CV şemanız yanlış.** Düzeltmeden devam etmeyin |

---

## Son gün kuralı

Yeni fikir denemeyi bırakın. Son gün yapılan değişikliklerin çoğu public
leaderboard'a uyum sağlar ve private'da zarar verir.

**Final submission: iki FARKLI aile seçin** — en iyi CV'li tek model ve en iyi
harman. İkisi de aynı aileden olursa risk çeşitlendirilmemiş olur.

```python
print(log.cv_lb_correlation())   # r > 0.8 ise CV'nize güvenin
```
