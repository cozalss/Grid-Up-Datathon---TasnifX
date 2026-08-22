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
| Test paketi | `tests/` | 1455 test |

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

3. **Harici veri — tek komut, ölçülmüş sıra.** `attach_external(panel,
   key_column=…, time_column=…, horizon=H)` on iki aileyi tek çağrıyla bağlar
   (hava, hava kalitesi, konvektif/CAPE, nem-toprak, güneş, yangın, deprem,
   turizm yıllık+aylık, İZSU, EPİAŞ). Eksik kaynak sessiz NaN değil raporlanan
   atlamadır; **%0 eşleşme hata verir**. `families=[…]` ile aile seçilir.

   **LOGO ablasyonu, gerçek GDZ verisi, 2026-08-18** (delta = aile silinince
   MAE'nin kötüleşmesi; tam model 312,46, sıfır-baseline 366,97; tohum
   gürültüsü ±1,2 dk → |delta| < 2 gürültü bandındadır):

   | Aile | delta | Yorum |
   |---|---:|---|
   | lag | **+15,88** | Tartışmasız birinci; ilk kurulacak aile |
   | konvektif (CAPE) | **+4,12** | Yıldırım vekili — ikinci en değerli |
   | epias (ulusal tüketim) | **+3,13** | "Dekorasyon" sanılıyordu, değil |
   | izsu (su profili) | **+2,69** | 2 kolon, yalnızca İzmir, yine de kazandırıyor |
   | hava_saatlik (basınç/rüzgâr saati) | **+2,65** | Cephe geçişi sinyali |
   | komsu | +2,13 | Ucuz, gürültü bandının hemen üstünde |
   | deprem | +1,07 | Gürültü bandında |
   | hava_kalitesi | +0,70 | Gürültü bandında |
   | takvim | +0,45 | Gürültü bandında |
   | turizm (yıllık/aylık) | 0,00 | Bu veride etkisiz (2021-22 penceresi) |
   | nem_toprak | −0,15 | Etkisiz |
   | yangın | −0,34 | Etkisiz |
   | **hava (günlük)** | **−0,58** | Tek başına ölçülünce katkı YOK |
   | **güneş** | **−1,16** | Çıkarılınca skor iyileşti |
   | **tatil** | **−3,19** | En zararlı aile — varsayılan olarak ekleme |

   > **Sürpriz ve ders:** "hava en değerli harici veridir" beklentisi bu
   > veride tutmadı — günlük hava tek başına nötr, değeri **saatlik türevleri**
   > (basınç düşüşü, eşik üstü rüzgâr saati) ve **CAPE** taşıyor. Tatil ve
   > güneş aileleri negatif ölçüldü; ölçmeden eklemeyin.

   **İKİNCİ ÖLÇÜM — 96 ilçe, sayım hedefi, 2022-26 (EPİAŞ paneli):**
   `scripts/ablation_epias.py` aynı aileleri **ADM dahil 96 ilçede** ve
   **kesinti SAYISI** hedefinde ölçtü (123.264 satır, 1284 kapsanan gün,
   tam MAE 2,906 · sıfır-baseline 6,039 · fold std 0,19):

   | Aile | delta | Aile | delta |
   |---|---:|---|---:|
   | **lag** | **+0,954** | ilce_gecmisi | +0,021 |
   | gunes | +0,094 | nem_toprak | +0,016 |
   | hava | +0,078 | hava_saatlik | +0,012 |
   | turizm_aylik | +0,065 | tatil | +0,008 |
   | takvim | +0,063 | **konvektif** | **−0,020** |
   | yangin | +0,051 | **epias** | **−0,146** |

   > **İKİ ÖLÇÜM ÇELİŞİYOR — ve bu, bilmemiz gereken şey.** 47 ilçe/dakika
   > panelinde CAPE (+4,12) ve EPİAŞ (+3,13) en değerli iki harici aileydi;
   > 96 ilçe/sayım panelinde ikisi de **negatif**. Buna karşılık `lag` ailesi
   > orada MAE'nin %5'ini, burada **%33'ünü** taşıyor. Yani harici verinin
   > katkısı hedef tipine ve panele göre değişiyor; hiçbiri "kesin faydalı"
   > değil. **Veri günü kuralı:** gerçek hedef geldiğinde `ablation_epias.py`
   > kalıbıyla YENİDEN ölç, bu tabloları kopyalama. Tek istikrarlı sonuç:
   > **önce lag ailesi**.

   İki teknik ders (ikisi de ölçüldü):
   - **Kapsama boşluğu sahte sıfır üretir.** EPİAŞ arşivinde 1690 günün
     406'sı boş; bunları 0 saymak sıfır oranını %54,5 → %65,4 şişiriyor.
     `epias_panel.py` `kapsanan_gun` bayrağı üretir, skorlama yalnızca
     kapsanan satırlarda yapılır.
   - **Boşluklu panelde satır-kaydırma yanlış lag verir.** Panel sürekli
     ızgaraya tamamlanıp (boşluk günlerinin hedefi NaN) feature üretilince
     ve sonra kapsanan satırlara inilince tam MAE 2,980 → **2,906** düzeldi.

4. **Komşu ilçe** — `nearest_neighbours` + `add_neighbour_target_lag(horizon=H)`
   (delta +2,13; ucuz).

5. **Model çeşitliliği** — sıra gerçek veride ölçüldü
   (`experiments/benchmark_gercek.json`, 2026-08-18 koşusu; ilçe kimliği +
   genişleyen ilçe istatistikleri feature setine girdikten sonra):

   | Model | MAE | Not |
   |---|---:|---|
   | iki_asama_medyan | **301,81** | Hurdle + koşullu medyan kuralı |
   | catboost_mae | **302,73** | 2023 birincisinin reçetesi |
   | lgb_mae | 306,77 | |
   | iki_asama | 310,60 | Eşikli hurdle |
   | iki_asama_medyan_kalibre | 314,10 | Kalibrasyon İYİLEŞTİRMEDİ |
   | lgb_tweedie | 316,93 | Tekil zayıf, harmanda çeşitlilik |
   | lgb_sqrt | 320,14 | Fit-uzayı erken durdurmayla (önce 393 = artefakttı) |
   | xgb | 368,13 | |
   | lgb_l2 | 369,15 | **Sıfır-baseline'ın (366,97) altında** — MAE'de L2 eğitme |

   **İlçe kimliği + genişleyen ilçe istatistikleri** (`add_expanding_features`,
   `horizon=UFUK`) her modeli iyileştirdi: catboost 304,30→302,73,
   lgb_mae 310,58→306,77, iki_asama_medyan 314,50→301,81. Sayım hedefiyse
   `COUNT_OBJECTIVES` ile süpürün. Kalibrasyonu varsaymayın:
   `calibrate_positive_probability` ölçer — gerçek veride İYİLEŞTİRMEDİ
   (Brier 0,193→0,217). Örnek ağırlığını (`recency_activity_weights`)
   Hawkes'la birlikte KULLANMAYIN: aynı yenilik sinyali iki kanaldan verilince
   kaybetti (lgb_mae 310,1→335,3, ölçüldü).

6. **Feature seçimi** — önce `null_importance_filter` (dakikalar), sonra
   `shap_backward_selection` (saatler; 2024 birincisi Pikachow da aynısını
   yaptı: SHAP ile 490→97 feature).

7. **Harmanlama — YUVALANMIŞ kontrol olmadan güvenme.** Örnek-içi
   hill-climb harmanı 298,59 diyor (catboost_mae + lgb_tweedie +
   iki_asama_medyan + iki_asama). Ama ağırlıklar **geçmiş fold'larda öğrenilip
   sonraki fold'da** skorlanınca: harman **359,00**, aynı satırlarda tek başına
   catboost_mae **349,71** → **harman GEÇMİYOR**. Örnek-içi skor, üye sayısı
   kadar serbestlik dereceli bir optimizasyonun kendi verisinde ölçülmesidir.
   `benchmark_gercek.py` artık bu kontrolü otomatik yapar (`harman.yuvalanmis`).

   **Tohum gürültüsü ölçüldü:** catboost_mae 5 tohumda 301,21–304,80
   (yayılım 1,24, aralık 3,59). 5-tohum ortalaması 302,22 — tekil ortalamadan
   **0,90 MAE kazanç**, üstelik yapısal yanlılık olmadan. Yani: **harman
   yerine tohum ortalaması**. Stacking'e zaman ayırmayın (1078,82).

   **DIŞ ÇAPA KANITI (bağımsız, 8 çapa — `scripts/outer_anchor_kosusu.py`).**
   Yuvalanmış kontrol tek bir bölünmedir; kazanan kapısı en az ALTI bağımsız,
   eşleştirilmiş çapa ister. Kosu üretildi (2021-12-18 → 2022-08-22, 31 günlük
   pencere, her çapada ağırlıklar YALNIZCA o çapanın iç CV'sinde tırmanıldı):

   | Aday | Ortalama MAE | En iyi–en kötü çapa | Çapa galibiyeti |
   |------|-------------:|--------------------:|----------------:|
   | catboost_mae | **386,95** | 206,40 – 480,38 | 3/8 |
   | harman       | 388,15 | 229,27 – 478,83 | 1/8 |
   | lgb_tweedie  | 404,59 | 260,78 – 506,19 | 3/8 |
   | lgb_mae      | 405,91 | 248,94 – 524,60 | 1/8 |

   Karar: **KAZANAN: YOK.** catboost_mae ortalamada önde ama çapa galibiyetini
   lgb_tweedie ile PAYLAŞIYOR (3–3) ve harmanla arası 1,20 MAE — çapalar arası
   yayılım 274 MAE. Yani sıralama istatistiksel olarak kararsız; "OOF'ta birinci"
   demek gerçek üstünlük demek değil. Uygulamada anlamı: **tek modele değil,
   catboost_mae 5-tohum ortalamasına oynayın**; harmanı yalnızca çeşitlilik
   sigortası olarak, ikinci gönderim slotunda tutun.
   Yarışma günü yeniden koşun: `python scripts/outer_anchor_kosusu.py` sonra
   `python scripts/benchmark_gercek.py --outer experiments/outer_anchors.json`.

8. **Son gün** — `multi_seed_refit(..., sample_weight=…, target_transform=…)`
   + `postprocess_predictions`. Refit artık CV'deki ağırlık/dönüşümü aynen
   taşır (denetim öncesi taşımıyordu: benchmark'ın ölçtüğü konfigürasyon
   gönderime aktarılamıyordu).

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
