# Grid Up Datathon — TasnifX

**Coderspace × GDZ Elektrik × ADM Elektrik** · 21 Ağustos – 1 Eylül 2026 · Kaggle In-Class

Bu depo, veri gelmeden **önce** hazır olmak için kuruldu. İçinde çalışan bir pipeline,
6 yıllık doğrulanmış harici veri ve 2023 yarışmasının kazanan çözümünden çıkarılmış
dersler var.

> **Ekip arkadaşım, buraya ilk kez bakıyorsan:** doğrudan [5 dakikada başla](#5-dakikada-başla)
> bölümüne git. Sonra [veri günü oyun planını](#veri-günü-oyun-planı) oku. Gerisi referans.

---

## Durum (2026-08-28)

| | |
|---|---|
| **Liderlik Tablosu** | **3. Sıra (TasnifX)**: `1.00284` · **1.** Grid Grinders `0.99064` · **2.** Atakan Aldemir `1.00041` · **4.** Ahmet B. ALTUNOK `1.00480` · **5.** Şaban Özdoğan `1.00510` — 2. sıra için gereken `−0,0024` (2026-08-29) |
| **28 Ağustos** | `v101` → `1.01614` (ıskaladı, `L`yi çözdü) · `v102 = v83 + 0.459·(v101−v83)` → **`1.00553`**, ön kayıtla 5 hanede birebir, 9.→2. sıra · `v109` → `1.01818` (CV türevli yönler `κ≈0`). Kota 3/3 kullanıldı |
| **29 Ağustos SONUCU** | 3 hak: `m4` **1.04300** (öngörü 1.00 — YANILDI, aktarım `f=−0,42`) · `p51` prob **1.00946** · `m6` iki-yön optimum **1.00284** (öngörü 1.00292). `1.00553 → 1.00284`, **5.→3. sıra**. Bulgu: geri-test SICAK dedi, LB SOĞUK dedi (`κ_soğuk 0,337` vs `κ_sıcak 0,122`). Ayrıntı: [`docs/55`](docs/55-29-agustos-bilancosu.md) |
| **1. sıra** | **ULAŞILAMIYOR** — gereken `Q·δ*² = 0.028257`, ölçülen en iyi `0.018500`. Nüfus büyütmek kapatmıyor (δ, Q'dan hızlı çürüyor). Varılabilecek yer `0.996–0.999` ile sağlam 2. sıra: [`docs/52`](docs/52-olu-trafo-curudu-ve-prob-kampanyasi-2026-08-28.md) §14.6 |
| **⚠ GÖNDERME** | `tuketim_v89_genis_taban.csv`, `v88`, `v87`, `sota_v1` — dördü de **diri** 93 trafoyu sıfırlıyor, beklenen skor `~1.13`. Ölü trafo tezi çürüdü: [`docs/52`](docs/52-olu-trafo-curudu-ve-prob-kampanyasi-2026-08-28.md) §1 |
| **Boru Hattı & Modeller** | **YENİ (2026-08-28):** `experiments/model29/m71_nihai_hava.py` — ileri-pencere doğrudan tahmin + hava/nem/turizm (LightGBM Huber+L1, `v102`'nin LB-kalibre seviyesine rejim çapası). `kis26` penceresinde üretim hattının en iyi CV yığınını **%6,4 dövüyor** (1,1063 → 1,0359). Eski hat: `scripts/sota_tuketim_pipeline.py` |
| **⚠ Doğrulama (CV)** | **KURAL 36: ileri-pencere geri-testi LB'yi ÖNGÖRMÜYOR** — ölçüldü, `f = −0,42`. Geri-testte %7 daha iyi olan model LB'de %2,9 daha kötü çıktı, üstelik kazancın yönünü de ters gösterdi. Geri-test yalnız **yön üretmek** için kullanılır; büyüklük ve işaret LB'de ölçülür |
| **⚠ Ölçüm Uyarısı** | `tanim_num` kimlik-ezberi: soğuk satırların çözülebilirliği `yaz25` %97.2 · `guz25` %97.7 · `kis26` %0 · **TEST %0**. Soğuk hükümler **yalnız `kis26`** ile verilir |
| **Gönderim Yetkisi** | **GERİ ALINDI** (2026-08-27). Kullanıcının açık onayı olmadan hiçbir dosya gönderilmez |
| **Kapı Denetimi** | `scripts/kapi_denetim.py` GECTI (714.688 satır, 0 NaN, 0 negatif, birebir ID sıralaması) |

---

## 5 dakikada başla

```powershell
git clone https://github.com/cozalss/Grid-Up-Datathon---TasnifX.git
cd Grid-Up-Datathon---TasnifX

python -m pip install --require-hashes -r requirements/uv-bootstrap.txt
uv sync --locked --extra full --extra dev

uv run python scripts\ekip_kontrol.py  # kurulum doktoru: 7 kontrol, ~3 sn
uv run python -m pytest -q             # tüm testler geçmeli
uv run python scripts\smoke_test.py    # uçtan uca kanıt, ~42 sn
```

`ekip_kontrol` geçemediğin her maddede düzeltme komutunu kendisi söyler —
takıldıysan önce onu çalıştır, sonra sor.

Çalıştıysa hazırsın. `smoke_test` sentetik veri üretip pipeline'ın her adımını geçirir
ve sonunda geçerli bir submission dosyası yazar.

`uv sync --locked`, yerel makineyi CI ile aynı hash-doğrulamalı 170 paketlik grafa
bağlar ve `.venv` ortamını otomatik oluşturur.

> **Windows notu:** `python3` değil `python` yaz — bu makinede `python3` bozuk bir
> Store kısayoluna gidiyor.

---

## Veri günü oyun planı

**21 Ağustos 14:00** — açılış yayını. Sorulacak on soru:
[`docs/00-yarisma-brief.md`](docs/00-yarisma-brief.md) sonunda.
En kritik üçü: **resmî metrik** · **harici veri serbest mi** · **ilk 10 mu ilk 20 mi**.

### İlk 30 dakika — üç komut

```python
from gridup import profile, read_any, suggest_scheme, leakage_report

train = read_any("data/raw/train.csv")            # kodlama/ayırıcı/ondalık otomatik
test  = read_any("data/raw/test.csv")

print(profile(train, test, target="HEDEF").report())        # 1 · elimizde ne var
print(suggest_scheme(train, target="HEDEF", test=test))     # 2 · hangi CV şeması
print(leakage_report(train, "HEDEF", test=test))            # 3 · sızıntı var mı
```

Bu üç çıktı sonraki 12 günün her kararını belirler.

> `suggest_scheme`'e `test=` vermeyi **atlama**. Test'te bulunmayan kolonlar grup
> adaylığından çıkar — yoksa hedeften türemiş bir kolona göre gruplama önerilebilir
> ve CV anlamsız olur. (Prova verisinde ölçüldü.)

### İlk 2 saat — tek komutla submission

```powershell
python scripts\day_one.py --data data\raw --metric mae
python scripts\day_one.py --data data\raw --time TARIH --group ILCE --metric mae
```

Betik `sample_submission.csv`'den hedef ve ID kolonunu kendisi çıkarır.
**Amaç skor değil, format doğrulaması** — bu tek başına bir gün kazandırır.

Detaylı saat saat akış: [`docs/07-veri-gunu-kontrol-listesi.md`](docs/07-veri-gunu-kontrol-listesi.md)

---

## 2023 yarışmasından doğrulanan kurallar

Bunlar tahmin değil — 2023 GDZ Elektrik Datathon'unun forum kayıtlarından ve Kaggle
API'sinden **doğrulandı**. 2026 için garanti değil; açılış yayınında teyit ettirin.

| Kural | 2023'teki hali | Bizim için sonucu |
|---|---|---|
| Public/Private ayrımı | **Rastgele %50/%50 satır** (zamansal değil) | Public LB güvenilir sinyal; shakeup riski düşük |
| Günlük submission | **3** (GDZ'nin üç yarışmasında da) | Toplam ~36 deneme |
| Metrik | MAPE | Hedefte sıfır varsa `mape()` uyarır |
| Harici veri | Serbest | Hava + güneş + EPİAŞ verimiz hazır |
| Test bloğu | İleri zamanlı 1 aylık pencere | `test_span=` ile birebir taklit edilir |
| Kazanan skor | Public 1.546 → **Private 1.488** | Prophet baseline 4.270 |

> **Tuzak:** Public LB rastgele bölmeyse overfit etmek *kolaydır*. 36 denemenin
> hepsini LB'ye bakarak seçerseniz private'ta düşersiniz.
> **Seçimi CV'ye yaptırın**, LB'yi yalnızca doğrulama olarak kullanın.

---

## Zaman bütçesi (bu makinede ölçüldü)

Ölçek provası **100k ve 500k satır** üzerinde koşuldu (2 tekrar, en kısası):

| İşlem | 100k | 500k |
|---|---|---|
| LightGBM tam CV (3 fold, 500 ağaç) | 4,8 sn | 16 sn |
| CatBoost tam CV (500 iter) | 37,6 sn | 118 sn — LightGBM'in **7 katı** |
| Sinir ağı CV (60 epok) | 10,5 sn | 66 sn |
| **Model zoo** (3 model × 3 fold) | 86,7 sn | **328 sn** ← en yavaş, +762 MB |
| Optuna tek deneme | 22,1 sn | 49,6 sn |
| SHAP önem | 0,8 sn | 0,9 sn — örneklem sabit, ölçekle büyümüyor |

**Bağlayan kısıt hesaplama değil, dikkat.** İki ölçekte de aynı sonuç: 60 saatlik
yarışmada makine binlerce koşu çıkarır ama insan ~144 hipotez kurup yorumlayabilir.

→ Optuna'yı **gece boyu** çalıştırın — 500k'da bile 8 saatte ~580 deneme sığar.
→ Araştırmayı **LightGBM ile** yapın; CatBoost'u yalnızca final harmana katın.
→ Model zoo 500k'da 5,5 dakika — günde en fazla birkaç kez, her feature setinde değil.

Kendi makinenizde ölçmek için: `python scripts/scale_rehearsal.py`

---

## Dört tasarım sözleşmesi

Bunlar dokümantasyon değil — **testlerle zorlanıyor**. İhlal edilirse test kırılır.

**1 · Feature fonksiyonları girdiyi asla değiştirmez.**
22 fonksiyonun tamamı otomatik keşfediliyor ve üç kontrolden geçiyor: girdiyi
değiştirmiyor, yeni frame döndürüyor, satır sayısını **ve sırasını** koruyor.
Yeni fonksiyon eklenirse test kırılır — kapsam kendiliğinden büyür.

**2 · Hedefe dokunan her fonksiyon ya fold alır ya gerekçesi yazılıdır.**
33 fonksiyon taranıyor: 12'si `folds` alıyor, 21'inin neden güvenli olduğu kayıtlı.

**3 · Tahmin ufku zorunludur.**
`add_lag_features`, `add_rolling_features`, `add_expanding_features` — üçünde de
`horizon` **varsayılansız**. Test bloğu bir aylıksa `horizon=1` yirmi dokuz günlük
sızıntıdır ve CV mükemmel görünür.

**4 · Aynı girdi aynı submission'ı üretir.**
LightGBM, XGBoost, CatBoost ve sinir ağı — dördü de **ayrı süreçlerde bit düzeyinde
aynı** çıktı veriyor, GPU dahil.

---

## Depo haritası

```
src/gridup/
  ── çekirdek ──────────────────────────────────────────────────────────
  config.py         sabitler, global tohum
  compat.py         pandas 3 / numpy 2 uyumu, bellek indirgeme
  io_utils.py       kodlama+ayırıcı+ondalık otomatik tespit (Türkçe CSV)
  turkish.py        İ/ı tuzağı, join anahtarı, Türkçe sıralama
  panel.py          varlık × zaman paneli, eksik gün doldurma
  profiling.py      veri profili, işaretlenmiş kolonlar
  experiment.py     JSONL deney defteri + günlük submission bütçesi

  ── doğrulama ─────────────────────────────────────────────────────────
  validation.py     ambargolu CV, şema önerisi, sızıntı raporu, adversarial
  metrics.py        RMSE/RMSLE/MAE/MAPE/SMAPE/AUC/F1 + eşik optimizasyonu
  submission.py     yazmadan önce doğrulama (NaN, ∞, eksik ID, sabit tahmin)

  ── model ─────────────────────────────────────────────────────────────
  models.py         LightGBM/XGBoost/CatBoost tek arayüz, OOF + kapsam maskesi
  refit.py          çok tohumlu tam-veri refit
  two_stage.py      sıfır-şişkin hedef için hurdle model + kuantil merdiveni
  zoo.py            model zoo, sayım objective süpürmesi
  tuning.py         Optuna araması (objective de arama uzayında)
  selection.py      SHAP geri eleme, null importance
  ensemble.py       tepe tırmanma, stacking, korelasyon budama
  ablation.py       feature grubu ablasyonu + dayanıklılık harmanı
  neural.py         varlık gömülü sinir ağı — harmana çeşitlilik

  ── feature ───────────────────────────────────────────────────────────
  features/temporal.py   takvim, TR tatil, Ramazan ayı, ufuk-farkındalıklı lag
  features/weather.py    saatlik→günlük, bölgesel agregat, fiziksel türevler
  features/solar.py      güneş geometrisi + açık-hava ışınımı (sızıntısız)
  features/spatial.py    haversine, komşu ilçe sinyali
  features/aggregate.py  grup istatistikleri, sapma/oran
  features/categorical.py fold-dışı hedef kodlama, frekans, nadir birleştirme
  features/outage_reason.py 919 serbest metin → 22 arıza ailesi
  features/school.py     MEB okul takvimi 2021–2026 (6 ders yılı doğrulanmış)

  ── veri ──────────────────────────────────────────────────────────────
  epias.py          EPİAŞ Şeffaflık istemcisi (tüketim, üretim, kesinti)
  synthetic.py      sentetik dağıtım şebekesi verisi
  reporting.py      jüri çıktıları: fold tablosu, segment hatası, iş değeri

scripts/
  smoke_test.py            uçtan uca kanıt (~42 sn)
  full_pipeline.py         20 entegrasyon kontrolü
  day_one.py               ham dosya → submission, tek komut
  scale_rehearsal.py       süre/bellek ölçümü, 12 günlük bütçe
  build_kaggle_package.py  internetsiz Kaggle paketi
  build_notebooks.py       notebook şablonları
  fetch_weather.py         96 ilçe hava (Open-Meteo)
  fetch_epias_load.py      Türkiye tüketim + üretim
  fetch_districts.py       ilçe koordinatları

docs/
  00-yarisma-brief.md            yarışma kuralları + açılışta sorulacaklar
  01-strateji-brifingi.md        derin araştırma bulguları
  05-ilk-24-saat.md              ilk gün runbook
  06-teknik-tuzaklar.md          bilinen tuzaklar
  07-veri-gunu-kontrol-listesi.md saat saat veri günü planı

tests/    1455 test — sızıntı, Türkçe, sözleşmeler, determinizm, özellik tabanlı
```

---

## Hazır harici veri

Gerçek GDZ provası hariç hepsi `data/` altında ve **gitignore'da** (aşağıdaki komutlarla üretilir).

| Veri | Boyut | Nasıl üretilir |
|---|---|---|
| Hava, 96 ilçe, 2020–2026-09 | 233.952 × 23 | `python scripts/fetch_weather.py --all-districts` + `fetch_weather_bridge.py` (forecast köprüsü, `hava_tahmin` bayrağı) |
| Güneş fiziği, 96 ilçe | 233.760 × 9 | `features/solar.py` (pvlib, deterministik) |
| Türkiye saatlik tüketim | 58.044 × 2 | `python scripts/fetch_epias_load.py` |
| Türkiye saatlik üretim | 58.044 × 18 | aynı betik — **yakıt kırılımı** |
| 96 ilçe + koordinat + nüfus | 96 × 10 | `python scripts/fetch_districts.py` |
| MEB okul takvimi 2021–2026 | kod içinde | `gridup.features.school` — indirme gerekmez |
| **Gerçek GDZ provası** (68.257 kayıt) | 11 MB | **repoda geliyor** — `data/prior/ayna/` (kaynak: Kaggle `tmlalper/manisa-izmir-plansiz-elektrik-kesintileri`, halka açık) |

Gerçek veride ölçülen her şey `experiments/ablasyon_gercek.json` ve
`experiments/benchmark_gercek.json` içinde: feature ailesi öncelik sırası
(lag +22,3 baskın; tatil/güneş negatif) ve model OOF sıralaması bulunur. Harman
aynı OOF üzerinde seçildiği için bilimsel "şampiyon" ilan edilmez; bağımsız,
eşleştirilmiş outer kanıt yoksa `kazanan=null` kalır. Veri günü
planı bu ölçümlere yaslanır — `docs/07`. **Kapsam:** sayılar 2021–22 verisi ve
`kesinti_dk` hedefi içindir; 2026 verisi gelince ablasyon 1. günde yeniden
koşulur (betik hazır, ~10 dk).

EPİAŞ için `.env` gerekir — `.env.example`'ı kopyalayıp kendi bilgilerinizi yazın.
**`.env` asla commit edilmez.**

> **Sızıntı uyarısı:** EPİAŞ serileri **geçmiş gözlemdir**, tahmin değil.
> Feature'a çevirirken `add_lag_features(horizon=TAHMİN_UFKU)` kullanın —
> elle `shift` yazmayın. Kural: **lag ≥ tahmin ufku**.

---

## Ekip çalışması

**Deney defteri.** Her koşuyu kaydedin; 8. günde "en iyi skorumu hangi feature
setiyle almıştım?" sorusuna cevap veremezseniz o skoru bir daha üretemezsiniz.

```python
from gridup.stores import SQLiteExperimentStore

# day_one.py; veri hashleri, reçete/fold parmak izi, parametreler ve feature
# listesini atomik olarak kaydeder ve ekrana run_id basar.
store = SQLiteExperimentStore("experiments/experiments.db")
store.record_lb("<day_one-run-id>", 1.5234)  # submission SONRASI
```

**İş bölümü önerisi.** Değerlendirmenin üçte ikisi notebook ve sunumda:

| Rol | Odak |
|---|---|
| Feature | `features/` — hipotez üret, CV'de ölç, deftere yaz |
| Model | `models`/`zoo`/`tuning`/`ensemble` — Optuna geceye, harman sona |
| Notebook & sunum | `reporting.py` çıktıları, jüri anlatısı, tekrar üretilebilirlik |

**Kural.** Aynı anda birden fazla şey değiştirmeyin — hangisinin işe yaradığını
bilemezsiniz. Her adımdan sonra CV'yi ölçün ve deftere yazın.

---

## Kırmızı bayraklar

| Gördüğünüz | Anlamı |
|---|---|
| Join sonrası satır sayısı düştü | Türkçe `İ` tuzağı — `diagnose_join` çalıştırın |
| CV mükemmel, LB berbat | Sızıntı. Önce **ufuk**, sonra origin, sonra fold hizalaması |
| `fold_std` skorun %10'undan büyük | CV gürültülü — küçük iyileşmelere güvenmeyin |
| Model "hep sıfır" baseline'ını geçemiyor | Sorun modelde değil yaklaşımda |
| Feature önemleri hep 0 | Önem çıkarımı bozuk, feature'lar değil |
| CV–LB korelasyonu r < 0,5 | **CV şemanız yanlış.** Düzeltmeden devam etmeyin |
| Eşik optimizasyonu F1 > 0,99 verdi | Muhtemelen eğitim tahminlerinde optimize ettiniz |

---

## Kaggle'da internet kapalıysa

```powershell
python scripts\build_kaggle_package.py --wheels --upload
```

Notebook'un ilk hücresine `kaggle_paket/notebook_bootstrap.py` içeriğini yapıştırın.
Wheel + harici veri + eksik paketler oradan yüklenir.

`--upload` yolu lisans, immutable kaynak, şema, artifact ve wheel hash kapılarının
tamamını çalıştırır. Doğrudan Kaggle CLI ile yükleme desteklenmez.

> **Kaynak değiştiyse paketi yeniden üretin.** Yüklü paket bayatlarsa
> internetsiz notebook düzeltilmiş sandığınız kodu **çalıştırmaz**.

---

## Son gün kuralı

Yeni fikir denemeyi bırakın. Son gün yapılan değişikliklerin çoğu public leaderboard'a
uyum sağlar ve private'da zarar verir.

**Final submission: iki FARKLI aile seçin** — en iyi CV'li tek model ve en iyi harman.
Aynı ailenin iki varyantını seçmek çeşitlilik sağlamaz.

---

## Dürüst sınırlar

Bu depo hazırlıktır, garanti değil. Bilinen açıklar:

- **Sistem hiç gerçek yarışma verisi görmedi.** Tüm prova sentetik veriyle yapıldı.
- **Metrik, hedef şekli ve panel yapısı henüz bilinmiyor.** 2023 varsayımları
  yol gösterir ama 2026 farklı olabilir.
- **`ADMINISTRATIVE_LEAVE` tablosu** birincil kaynaktan doğrulandı ama gelecek
  yıllar için tahmin içerir.

Bulunan ve kapatılan hataların kaydı git geçmişindedir — her commit ölçümle birlikte
neyin neden değiştiğini anlatır.
