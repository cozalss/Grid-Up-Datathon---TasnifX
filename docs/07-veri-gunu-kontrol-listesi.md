# Veri Günü Kontrol Listesi

21 Ağustos, saat 15:00. Kaggle linki geldi. Bu dosya o andan itibaren
izlenecek adımları içerir.

---

## Hazır olanlar (bugün itibarıyla)

| Varlık | Yer | Durum |
|---|---|---|
| Hava durumu, 2020–2026-09-02 | `data/external/hava_gunluk.parquet` | **96 ilçe**, 233.952 satır, sıfır eksik; ERA5 arşivi 2026-08-09'a kadar + Open-Meteo forecast köprüsü (`hava_tahmin=1`, 2304 satır, dikiş 0,70 °C) — `scripts/fetch_weather_bridge.py` |
| Güneş fiziği, 2020–2026 | `data/external/gunes_gunluk.parquet` | 96 ilçe × 2435 gün, sıfır eksik |
| 96 ilçe + koordinat + nüfus | `data/reference/ilceler_gdz_adm.parquet` | Komşuluk grafiği doğrulandı |
| Arıza sebebi taksonomisi | `gridup.features.outage_reason` | 919 metin → 22 aile, %0,88 sınıflanamayan |
| Uçtan uca betik | `scripts/day_one.py` | Sentetik veride 6 sn'de submission |
| Kaggle offline paketi | `kaggle.com/datasets/cemzal/gridup-offline-paket` | Yüklendi (özel), dosyalar doğrulandı |
| Okul takvimi 2021–2026 (MEB, doğrulanmış) | `gridup.features.school` | 6 ders yılı, ara/yarıyıl/yaz |
| Saatlik hava türevleri | `data/external/hava_saatlik_turev.parquet` | 96 ilçe × 2.414 gün: basınç min/ort, ≥15/20 m/s rüzgâr saatleri, hamle saatleri, yön değişimi |
| Deprem katalogu (AFAD) | `data/external/depremler.parquet` | 373 deprem, M4,0–6,6, Ege kutusu 2020–2026 |
| Yangın tespitleri (NASA FIRMS) | `data/external/yanginlar.parquet` | 30.575 uydu sıcak-noktası, 5 il, 2020–2024 (poligon değil tespit; FRP yoğunluk vekili) |
| Turizm gecelemeleri (KTB) | `data/external/turizm_geceleme.parquet` | İlçe bazlı geceleme/geliş, 2023–2025, 231 satır; Alsancak→Konak katlanmış, her satır 96 ilçe referansında — Muğla yaz-nüfus vekili |
| Turizm aylık il serisi (KTB) | `data/external/turizm_aylik_il.parquet` | 81 il × ay, 2019-01…2026-06 (7290 satır): geliş/geceleme (yabancı-yerli-toplam) + doluluk. **Çapraz doğrulandı:** 12 ay toplamı = yıllık bülten (%0,00, 81 il), sonraki bültenlerde revizyon yok. `kapsam_rejimi` 1/2/3 = **ölçülen** kapasite kırılmaları 2022-09 ve 2025-07 (başlık 2022-11'de değişti, veri 2 ay önce); yıllar arası kıyasta `doluluk` kullan. `features.tourism`: lag ≥2 ay zorunlu (vars. 12); ilçe-ay tahmini = yıllık ilçe payı × il aylık profil, `districts=` ile tesissiz ilçeler 0. **+ Belediye belgeli seri** 2019-01…2022-10 (46 bülten) → `*_belediye` ve `*_tum_belgeli` (bakanlık+belediye; 2022-09 dikişini kapatır, 2022-11'de ters düşüş — kusursuz sürekli seviye serisi yok, bkz. docs/15) |
| EPDK bölge sınıfı | `gridup.features.demografi.epdk_bolge_sinifi` | Resmi kentsel/kentaltı/kırsal eşikleri |
| Gerçek veri ölçümleri | `experiments/ablasyon_gercek.json` · `benchmark_gercek.json` | 68.257 gerçek GDZ kaydında |
| Ekip kurulum doktoru | `scripts/ekip_kontrol.py` | Tek komutla 7 kontrol |
| Test paketi | `tests/` | 1206 test |

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

Veriyi `data/raw/` altına koyun (**yalnızca** yarışma dosyaları; alt dizinde
başka CSV bırakmayın — betik sığ dosyayı tercih eder ama gölgeyi uyarır), sonra:

```powershell
cd c:\Users\cemmo\Documents\Datahon
python scripts\fetch_weather_bridge.py            # hava arşivini bugün+16'ya köprüle
python scripts\day_one.py --data data\raw --metric mae --task regression
```

`--task regression`: sayım hedefi (0..8 gibi) profilde "multiclass" görünür;
metrik regresyon metriğiyse betik zaten regresyona çözer, bayrak bunu açık
kılar. Betik ayrıca metrikten objective türetir (MAE → L1), bileşik
`unique_id`'yi test kolonlarından sentezler ve **HARİCİ VERİ KAPSAMI**
satırlarını basar — test bloğunun sonu bir serinin son gününü aşıyorsa
"UYARI: N gün AÇIK" görürsünüz; o aileyi ya yenileyin ya ufuk-kaydırmalı kullanın.

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
# NOT: read_any kolon adlarini normalize eder ("ARIZA SEBEBİ" -> "ariza_sebebi").
from gridup.features import reason_family_report
print(reason_family_report(train["ariza_sebebi"]))
```

**2 · İlçe adları tablomuzla eşleşiyor mu?**

```python
from gridup.turkish import diagnose_join, join_key
import pandas as pd
ref = pd.read_parquet("data/reference/ilceler_gdz_adm.parquet")
print(diagnose_join(train["ilce"].unique(), ref["ilce"].unique()))
```

Eşleşmeyen varsa **2012 büyükşehir yasası** şüphelisi: Efeler, Şehzadeler,
Yunusemre, Seydikemer gibi ilçeler 2012'de kuruldu; eski kayıtlar "Merkez"
diyor olabilir.

**3 · Hava verisi join oluyor mu?**

```python
# Hava parquet'i ILCE bazlidir: anahtar ilce_key = join_key(strip_qualifier(ilce)).
# konum_key "il-ilce" bilesik dizgedir; il_key ile eslesmez (olculdu: %0).
from gridup.turkish import strip_qualifier
hava = pd.read_parquet("data/external/hava_gunluk.parquet")
train["ilce_key"] = train["ilce"].map(lambda ad: join_key(strip_qualifier(ad)))
train["tarih"] = pd.to_datetime(train["tarih"]).dt.normalize()
merged = train.merge(hava, on=["ilce_key", "tarih"], how="left")
print(merged["sicaklik_ort"].notna().mean())   # ~1.0 olmalı; degilse diagnose_join
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
from gridup.stores import SQLiteExperimentStore
store = SQLiteExperimentStore("experiments/experiments.db")
store.record_lb("<day_one-run-id>", <LB_SKORU>)
```

---

## Saat 4+ — Sırayla ekleyin (sıra ÖLÇÜLDÜ, tahmin değil)

Sıra, 68.257 gerçek GDZ kaydında leave-one-group-out ablasyonla ölçüldü
(`scripts/ablation_gercek.py` → `experiments/ablasyon_gercek.json`).

> **Kapsam sınırı:** bu sayılar 2021–22 verisi, `kesinti_dk` hedefi ve 47
> ilçe içindir. 2026 hedefi/kapsamı farklıysa sıralamayı taşımayın —
> **1. günde ablasyonu yeni veride tekrar koşun** (betik hazır, ~10 dk;
> yol haritası: `ablation_gercek.py` içindeki panel kurulumunu yeni kolon
> adlarına uyarlamak yeter). Her
adımdan sonra CV'yi ölçün ve deftere yazın. **Aynı anda birden fazla şey
değiştirmeyin** — hangisinin işe yaradığını bilemezsiniz.

1. **Adversarial validation** — `adversarial_validation(train, test)`.
   AUC > 0,8 ise ciddi kayma var, ayrıştıran feature'ları inceleyin.

2. **Ufuk-farkındalıklı lag/rolling** — ölçülen katkı **+22,3 dk**, diğer tüm
   ailelerin toplamından büyük. İlk kurulacak aile budur:
   ```python
   out = add_lag_features(out, TARGET, shifts=[31, 62, 93], time_column=TIME,
                          group_columns=[GROUP], horizon=HORIZON)
   ```
   `horizon` **zorunlu**. Test bloğu bir aylıksa 1 günlük lag yoktur.

3. **Hava** — ölçülen katkı +2,5 dk. Ortalama değil `max` ve quantile; bölge
   geneli agregat da ekleyin: `add_regional_aggregates`, `add_physical_derivatives`.

4. **Komşu ilçe** — ölçülen katkı +0,2 dk (marjinal ama ucuz):
   `nearest_neighbours` + `add_neighbour_target_lag(horizon=H)`.

   > **UYARI — tatil ve güneş aileleri gerçek veride NEGATİF ölçüldü**
   > (−4,7 ve −5,0 dk: çıkarılınca skor İYİLEŞTİ). Fold gürültüsü büyük
   > olduğu için kesin hüküm değil; ama bu ikisini ancak CV kazancı
   > ölçerek ekleyin, asla varsayılan olarak değil. Okul takvimi ailesi de
   > (`add_school_calendar_features`) aynı kurala tabi: hazır, ama ölçmeden girmez.

5. **Model çeşitliliği** — sıra gerçek veride ölçüldü, 3. dalga feature
   setiyle (Hawkes bozunumu + toplu-olay payı dahil;
   `experiments/benchmark_gercek.json`): önce **catboost_mae** (MAE 304,9 —
   2023 birincisinin reçetesi bu feature setiyle kazanan tekil), sonra
   `lgb_mae` (310,1), sonra **iki aşamalı + medyan kuralı**
   (`fit_conditional_quantile_ladder` + `conditional_quantile_from_hurdle`,
   316,9 — eşikli 317,0'a farkı 0,1'e indi: sinyali Hawkes feature'ları
   taşıyınca kuralın kazancı eridi), sonra `lgb_sqrt` (324,0) ve
   `lgb_tweedie` (327,8 — tekil zayıf ama harmanda ağırlık alıyor). Sayım
   hedefiyse `COUNT_OBJECTIVES` ile süpürün. Kalibrasyonu varsaymayın:
   `calibrate_positive_probability` ölçer — gerçek veride İYİLEŞTİRMEDİ
   (Brier 0,207→0,241), eşik 0,680 verinin gerçeğiydi. Örnek ağırlığını
   (`recency_activity_weights`) Hawkes'la birlikte KULLANMAYIN: aynı yenilik
   sinyali iki kanaldan verilince kaybetti (lgb_mae 310,1→335,3, ölçüldü).

6. **Feature seçimi** — önce `null_importance_filter` (dakikalar), sonra
   `shap_backward_selection` (saatler; 2024 birincisi Pikachow da aynısını
   yaptı: SHAP ile 490→97 feature).

7. **Harmanlama** — `hill_climb_weights` OOF üzerinde, **kapsam maskesiyle**,
   **TÜM üyelerle** ve **kararlılık cezasıyla** (`stability_penalty=0.5` +
   fold dilimleri; gerçek veride 302,6 — catboost_mae 0,75 + lgb_tweedie
   0,25. "En iyi 3" kısayolu önceki dalgada 311,8'e geriletti — harmanı üye
   kalitesi değil hata çeşitliliği taşır). Stacking'e zaman ayırmayın: aynı
   ölçümde baseline'ın bile altında kaldı.

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
