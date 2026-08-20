# Dünya standardı araştırması — sistemimiz nerede eksik?

**Tarih:** 2026-08-20 · **Soru:** "Kazandıracak, dünyanın en iyisi sistem" için ne eksik?
**Yöntem:** (a) literatür taraması — kesinti tahmini SOTA, (b) yarışma meta'sı —
2025-26 Kaggle tabular kazanan pratikleri, (c) kendi repomuzun envanteri.

> **Takvim düzeltmesi:** Yarışma **21 Ağustos 15:00 → 1 Eylül 23:59, 12 gün**
> (docs/00). Tek günlük sprint değil. Aşağıdaki işlerin hepsi bu pencereye sığar;
> "zaman yok" gerekçesi geçersiz.

---

## Özet — üç gerçek boşluk

Repomuzun **kütüphanesi** güçlü (ablasyon, kalibrasyon, sızıntı kapıları, dış çapa
hakemi, iki-aşamalı model, sinir ağı, stacking, hill-climb hepsi var). Boşluk
kodda değil, **üç başka yerde**:

| # | Boşluk | Kanıt | Beklenen etki |
|---|--------|-------|---------------|
| **A** | **Maruziyet verisi yok** — altyapı ve bitki örtüsü | literatür ablasyonu 2,6× | **Büyük** |
| **B** | **Harmanda tek aile** (hepsi ağaç) + MAE'de yanlış toplayıcı | GM playbook, M5 | **Orta-büyük** |
| **C** | **Feature hacmi 61** ("binlerce" pratiğine karşı) | GM playbook #3 | Orta |

---

## A · Maruziyet verisi — en büyük belgelenmiş kol

### A1. Altyapı maruziyeti (ŞU AN: sıfır feature)

*Deep Learning-Based Weather-Related Power Outage Prediction with Socio-Economic
and Power Infrastructure Data* (arXiv 2404.03115) ablasyonu — **aynı model, aynı
hedef**:

| Girdi kümesi | MAE |
|---|---|
| Yalnız hava | 0,03547 |
| + istasyon mesafeleri | 0,01916 |
| + sosyo-ekonomik **+ altyapı sayımları** | **0,01346** |

Hava-yalnız → tam küme **2,6 kat** hata düşüşü. Kullandıkları altyapı
feature'ları: idari birim başına **direk, hat, kule, trafo, şalt, izolatör,
kesici, terminal, jeneratör** sayıları — hepsi **OpenStreetMap**'ten.

> **Uyarı — sayı birebir taşınmaz.** O çalışmanın hedefi "nüfusa bölünmüş
> saatlik etkilenen abone oranı"; bizimki ilçe-gün kesinti dakikası/sayısı.
> 2,6× bir *büyüklük vaadi* değil, **yön kanıtı**: maruziyet feature'ları
> hava feature'larından daha çok bilgi taşıyor. Gerçek kazanç bizim LOGO
> ablasyonumuzla ölçülecek.

**Bizde ne var:** `data/reference/ilceler_gdz_adm.parquet` → yalnızca
`nufus`, `alan_km2`, `lat`, `lon`. **Hiçbir altyapı sayımı yok.**

**Nasıl kapatılır (~1 gün):** OSM `power=*` etiketleri, Türkiye extract'ı
üzerinden ilçe poligonuna sayım. Araç: `earth-osm` (pip) veya doğrudan
Overpass QL. Lisans **ODbL** — yeniden dağıtıma izinli, **atıf zorunlu**
(`sources.yml`'e `redistribution: allowed` + notebook'a atıf hücresi).
Üretilecek feature'lar: `direk_yogunlugu = direk / alan_km2`,
`hat_km_per_km2`, `trafo_basina_nufus`, `ilce_hat_uzunlugu`.

### A2. Bitki örtüsü (ŞU AN: sıfır feature)

Ağaç teması dağıtım şebekesinde kesintinin **en büyük tek sebep sınıfı**.
Vegetation Risk Model literatürü (19 değişkenli Random Forest, AUC-ROC **0,832**)
en güçlü belirleyicileri sayıyor: **canopy height, canopy cover, LAI, hatta
yakınlık, tür kompozisyonu, eğim/bakı**.

**Nasıl kapatılır (~yarım gün):** **ESA WorldCover 10m v200** — ücretsiz,
**CC-BY-4.0**, AWS açık S3 (`s3://esa-worldcover/`), 11 sınıf. İlçe poligonu
başına sınıf oranları: `agac_orani`, `calilik_orani`, `tarim_orani`,
`yerlesim_orani`, `su_orani`. **Statik ilçe feature'ı** — zaman boyutu yok,
dolayısıyla **sızıntı riski sıfır**, ufuk/ambargo hesabına hiç girmiyor.

> **Neden bu ikisi bizim için özellikle güçlü:** ilçeler arası fark
> (kırsal Bozdoğan vs kentsel Karabağlar) şu an modele yalnızca `nufus` ve
> `ilce_kimlik` üzerinden gidiyor. Ağaç örtüsü + direk yoğunluğu,
> "bu ilçede rüzgâr neden kesinti yapar, öbüründe yapmaz" sorusunun
> **fiziksel** cevabı. Rüzgâr × ağaç örtüsü etkileşimi literatürün
> merkezindeki terim — bizde etkileşimin bir yarısı hiç yok.

### A3. Maruziyet normalizasyonu (docs/01'de yazılı, UYGULANMAMIŞ)

docs/01 satır 82, Enefit notu: *"Hedefi kapasiteye bölerek normalize;
`target / exposure` MAE'de doğrudan kazanç."* Bu not orada duruyor ama
**uygulanmadı**. A1/A2 geldiğinde maruziyet paydası da gelir
(nüfus, direk sayısı, hat-km).

---

## B · Harman neden çöktü — iki ayrı tasarım hatası

Yuvalanmış kontrolde harmanı reddettik (359,00 vs tek başına 349,71) ve dış
çapa da onu doğrulamadı (388,15 vs 386,95). **Ama iki hata vardı:**

### B1. Çeşitlilik yok — dört üyenin dördü de ağaç tabanlı

Üyeler: `catboost_mae`, `lgb_mae`, `lgb_tweedie`, `iki_asama_medyan`.
Hepsi gradient boosting. Harmanın matematiği **hata dekorelasyonuna** dayanır;
aynı aileden dört model aynı hataları yapar, harman kazanamaz.

NVIDIA *Kaggle Grandmasters' Playbook* (2025) #2 ve #4, kazanan harmanları
şöyle tarif ediyor: *"gradient-boosted trees, neural nets, and Support Vector
Regression"* (Rainfall, 2.'lik) · *"XGBoost, CatBoost, neural nets, and linear
models"* (Calorie, 1.'lik). **Karışık aile.**

`src/gridup/neural.py` bizde **var** ve çalışıyor — ama benchmark üyesi değil,
hatta coverage'dan muaf tutulmuş. Lineer model (Ridge / QuantileRegressor) hiç yok.

### B2. MAE metriğinde **ortalama** ile harmanladık — medyan olmalıydı

Bu bir kavram hatası: **MAE'yi minimize eden tahmin medyandır, ortalama
değil.** Harmanı ağırlıklı *ortalama* ile kurmak, metrikle çelişen bir
toplayıcı seçmektir. M5 ve MAE-metrikli yarışmalarda üst sıradaki takımların
ortak pratiği **üye tahminlerinin medyanı**.

`src/gridup/ensemble.py` envanteri: `hill_climb_weights`, `power_mean_blend`,
`rank_average`, `stack_oof` — **medyan harman YOK.**

> **Sonuç:** "Harman geçmiyor" hükmümüz, doğru testin yanlış adaya
> uygulanmasıydı. Reddedilen şey *ağırlıklı ortalama harman*. Çeşitli aileli
> **medyan** harman hiç denenmedi. Yuvalanmış kontrol + dış çapa kapıları
> zaten kurulu — yeni adayı aynı kapıdan geçiririz. Geçmezse yine reddederiz;
> ama bu kez doğru şeyi reddetmiş oluruz.

### B3. Tohum sayısı 5 → 25/100

Ölçtük: 5 tohum ortalaması tekil ortalamaya karşı **+0,90 MAE**. Playbook #7:
*"ensembling XGBoost models across 100 different seeds clearly outperformed
single-seed training"* (Fertilizers). Bizde altyapı hazır
(`multi_seed_refit`) — sadece sayıyı büyütmek gerekiyor. **En ucuz kazanç.**

---

## C · Feature hacmi — iddia DÜZELTİLDİ (2026-08-21)

> **Bu bölümün ilk hâli yanlıştı.** "Bizde 61 feature var, havuz çok küçük"
> demiştim. 61, **`benchmark_gercek.py`'nin çekirdek reçetesinin** sayısıdır —
> o betik harici aileleri hiç bağlamaz, model karşılaştırması için bilerek dar
> tutulmuştur. Gerçek havuz ölçüldü (96 ilçelik EPİAŞ paneli, 123.264 satır,
> `attach_external` tüm ailelerle):
>
> **14 aile → 219 harici kolon**, 2,9 saniyede, 143 MB.
>
> Yani havuz "çok küçük" değil. İki betiğin feature seti farklı ve ben birinin
> sayısını diğerinin iddiası sanmışım.

Playbook #3 (*"Thousands of new features made the difference"*) yine de geçerli
bir yön gösteriyor, ama gerekçe değişiyor: eksik olan **ham kolon sayısı değil,
sistematik ETKİLEŞİM üretimi**. 219 kolonun neredeyse tamamı tekil ölçüm;
çarpım/oran/pencere kombinasyonları yok:

- ilçe × hava etkileşimleri (rüzgâr × ağaç örtüsü, yağış × eğim)
- çoklu pencere rolling (3/7/14/28/56 gün; mean/max/std/quantile)
- komşu ilçe agregatları (`features/spatial.py` var, kullanımı sınırlı)
- hedef geçmişinin dağılım istatistikleri (son 28 günün 0-oranı, tepe günü,
  son kesintiden bu yana geçen gün)

Yaklaşım: **üret → `null_importance_filter` → `shap_backward_selection`**
(zaten kurulu; 2024 birincisi Pikachow 490→97 yapmıştı). Havuzu büyütüp
budamak, küçük havuzu budamamaktan iyidir.

---

## D · Yeni aile fırsatı — ve LİSANS TUZAĞI

**TabPFN-2.5** (Prior Labs, Kasım 2025): tabular foundation model.
TabArena'da *tek forward pass'ta*, **4 saat tune edilmiş AutoGluon 1.4'ü
geçiyor**; ≤100k satır / ≤2000 feature aralığında "en iyi varsayılan model".

Bizim veri: **~22k satır × 61 feature** — tam kullanım aralığında.
Ve GBDT'den **tamamen farklı bir tümevarım yanlılığı** → B1'in aradığı
çeşitlilik üyesi.

> ⚠️ **DUR — kural sorusu.** TabPFN-2.5 **model ağırlıkları ticari olmayan
> lisansta** (`tabpfn-2.5-license-v1.1`). Bu, **ödüllü ve sponsorlu** bir
> yarışmada net bir risk. Açılış buluşmasında sorulacaklar listesine
> eklenmeli. Cevap gelene kadar **kullanılmamalı**; jüri kapısında (Kapı 2)
> lisans sorunu çıkması, skor kazancından çok daha pahalıya patlar.
>
> Lisans sorunu olmayan çeşitlilik üyeleri: `neural.py` (bizim),
> Ridge / QuantileRegressor (sklearn, BSD), RandomForest / ExtraTrees.

---

## E · Doğrulanan şeyler (değiştirmiyoruz)

- **CV yapısı veri yapısına uymalı** — playbook'un ana vurgusu; bizde purged
  time-series + embargo + ufuk geometrisi kurulu. Bu tarafımız iyi.
- **GBDT zaman serisinde hâlâ pratik en iyi** (TabPFN büyük veride bellekte
  tıkanıyor, zaman serisinde XGBoost öne çıkıyor). Omurgayı değiştirmiyoruz.
- **Çok tohumlu refit + %100 veriyle yeniden eğitim** (playbook #7) —
  `refit_full` 2024 birincisinde 3,02→2,95 taşımıştı; bizde kurulu.
- **Train-test dağılım kayması kontrolü** (playbook #1) —
  `adversarial_validation` bizde var (`validation.py:754`).
  Veri günü **ilk saatte** koşulmalı.

---

## Öncelik sırası (etki ÷ maliyet)

| Sıra | İş | Durum | Neden burada |
|---|---|---|---|
| 1 | **ESA WorldCover ilçe sınıf oranları** | ✅ **bitti** | Statik, sızıntısız, lisansı temiz, literatürün merkezi |
| 2 | **OSM altyapı sayımları + yoğunluklar** | ✅ **bitti** | Literatürdeki en büyük tek ablasyon kazancı |
| 3 | **Medyan harman** | ✅ **bitti** | Kavram hatasının düzeltilmesi; kapılar zaten kurulu |
| 4 | **Tohum eğrisi + `--tohum N`** | ✅ **bitti** | Sayıyı kanıta bağladı, tahmine değil |
| 3b | Çeşitli aile (NN, Ridge) harmana | açık | Medyan toplayıcının asıl kazandığı yer |
| 5 | **Feature fabrikası + SHAP budama** | açık | Havuzu büyüt, sonra buda |
| 6 | **Maruziyet normalizasyonu** (`hedef / maruziyet`) | açık | 1-2 geldiği için artık anlamlı |
| 7 | Optuna ince ayar (P1-15) | veri bekliyor | Betik hazır |
| — | TabPFN-2.5 | **HAYIR** | Lisans cevabı gelmeden kullanılmaz |

### 1-2 · Ne üretildi (2026-08-20)

| Çıktı | Satır | İçerik |
|---|---|---|
| `data/external/arazi_ortusu_ilce.parquet` | 96/96 | 11 WorldCover sınıf oranı + `bitki_ortusu_orani`, `agac_yerlesim_orani` |
| `data/external/osm_altyapi_ilce.parquet` | 96 | direk/kule/trafo/şalt sayıları, iletim & dağıtım hat-km, yoğunluklar |

Ölçülen ağaç örtüsü ilçeler arasında **%3,1 – %82,9** aralığında (ortalama %42,3);
yerleşim %0 – %63. En ağaçlı Kavaklıdere/Ula/Köyceğiz, en kentsel
Karabağlar/Konak/Bayraklı — yani sinyal gerçek ve ilçeleri gerçekten ayırıyor.

Her ikisi de `attach_external`'a **statik ilçe ailesi** olarak bağlandı
(`arazi_ortusu`, `osm_altyapi`). Statik tablolar için ayrı bir birleştirme
yolu yazıldı; nedeni şu incelik:

> Zamanlı ailelerde yanlış bir join'i er geç ufuk/ambargo kapısı yakalar.
> **Statik tabloda öyle bir kapı yoktur** — zaman ekseni olmadığı için sızıntı
> denetimi hiç devreye girmez. Bu yüzden tekillik (`validate="many_to_one"`),
> satır sayısı korunumu ve %0 eşleşmede `ValueError` elle kuruldu; 8 test bunu
> sabitliyor.

**Geometri uyarısı:** ilçe poligonumuz yok (referans tabloda merkez + alan var).
Her ilçe **alanına eşit bir daireyle** temsil edildi. Yani sayılar "ilçe
sınırlarının tam içi" değil, "ilçe merkezinin çevresinde, ilçe büyüklüğünde bir
daire". Uzun/kıyı ilçelerinde sapar. İki aile de **aynı** daireyi kullanıyor ki
ağaç oranı ile direk yoğunluğu aynı alanı ölçsün. Yarışma günü ilçe poligonu
verilirse yükseltilmeli.

**Son sözü ablasyon söyler.** İkisi de LOGO ablasyonundan geçmeden gönderime
girmez — literatürdeki 2,6× bizim hedefimize birebir taşınmaz, yön kanıtıdır.

### ⚠️ OSM bulgusu — literatürün varsayımı Türkiye'de TUTMUYOR

Çekim öncesi iki ilçe provası (Konak = kentsel, Bozdoğan = kırsal) şunu gösterdi:

| Tip | Konak | Bozdoğan |
|---|---:|---:|
| `power=pole` (**dağıtım direği**) | **0** | **0** |
| `power=minor_line` (**dağıtım hattı**) | **0 km** | **0 km** |
| `power=tower` (iletim direği) | 16 | 77 |
| `power=line` (iletim hattı) | 3,0 km | 39,7 km |

Yani **Türkiye OSM'inde dağıtım şebekesi haritalanmamış.** arXiv 2404.03115'in
kullandığı feature'ın (direk sayısı) tam karşılığı bizde **yok**; elimizde
yalnızca **iletim** altyapısı var.

Bu, ailenin değersiz olduğu anlamına gelmez — iletim yoğunluğu, bir ilçenin
şebeke içindeki ağırlığı için makul bir vekildir. Ama beklenti düşürülmeli:
literatürdeki kazancın kaynağı dağıtım seviyesi maruziyetiydi ve o kısım eksik.

Betik bu yüzden **tüm ilçelerde aynı değeri taşıyan kolonları düşürür ve
raporlar**. Sabit bir kolon sıfır bilgi taşır; modele girerse yalnızca feature
sayısını şişirir ve "bu aile 9 kolon getirdi" gibi yanıltıcı bir izlenim yaratır.
Düşülenler sessizce yok olmaz: burada "sıfır" bir **ölçüm sonucudur** — o altyapı
tipi Türkiye OSM'inde yok demektir.

**Bu, varsaymayıp ölçmenin karşılığıdır.** "Literatür diyor ki direk sayısı en
güçlü feature" deyip 9 kolon eklemiş olsaydık, bunların çoğu sabit sıfır olarak
modele girecek, ablasyonda "OSM ailesi hiçbir şey katmıyor" çıkacak ve nedenini
bilmeyecektik.

**Kritik kural:** 1-6'nın hepsi mevcut kapılardan geçmek zorunda —
LOGO ablasyonu + yuvalanmış kontrol + dış çapa. Ölçülmeyen hiçbir şey
gönderime girmez. Bu repoda "iyi fikir" değil, "ölçülmüş kazanç" para eder;
harman iddiasını iki kez böyle çürüttük.

---

## Açılış buluşmasında sorulacaklar (docs/00 listesine ek)

1. **Dış veri serbest mi?** (OSM, ESA WorldCover, EPİAŞ — hepsi buna bağlı)
2. **Ticari olmayan lisanslı model ağırlığı kullanılabilir mi?** (TabPFN-2.5)
3. **Resmî metrik nedir?** (MAE ise medyan toplayıcı + tam sayı yuvarlama
   doğrudan skor kazandırır)
4. **EPİAŞ kesinti geçmişi** test dönemini kapsıyor — kullanımı serbest mi?

---

## Kaynaklar

- arXiv 2404.03115 — *Deep Learning-Based Weather-Related Power Outage Prediction
  with Socio-Economic and Power Infrastructure Data*
- ScienceDirect S0951832025007793 — GNN + LiDAR ağaç riski ile kesinti tahmini
- ScienceDirect S0378779623003759 — ağaç kaynaklı kesintilerde yerel çevre değişkenleri
- Wiley 2050-7038.12154 — veri-güdümlü bitki örtüsü kaynaklı kesinti tahmini
- NVIDIA Technical Blog — *The Kaggle Grandmasters' Playbook: 7 Battle-Tested
  Modeling Techniques for Tabular Data*
- arXiv 2511.08667 — *TabPFN-2.5* teknik raporu / TabArena sonuçları
- ESA WorldCover v200 — esa-worldcover.org (CC-BY-4.0)
- OpenStreetMap `power=*` — wiki.openstreetmap.org/wiki/Power_networks (ODbL)
- M5 Forecasting tartışmaları — MAE'de medyan toplayıcı pratiği
