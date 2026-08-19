# GDZ 2024 birincisi (Pikachow) + 2024–2026 Kaggle metası

> **Neden bu belge:** "2023'e bakıyoruz ama meta değişmedi mi?" sorusuna
> sistematik cevap. Beş paralel araştırma hattı (15 Ağustos 2026) + 2024
> birincisinin 29 slaytlık final sunumunun satır satır analizi. Her iddia
> kaynaklı; "bizde var mı" sütunu koda karşı doğrulandı, aktarılanlar
> gerçek veride **ölçüldü**.

---

## 1 · En büyük keşif: GDZ Elektrik Datathon 2024 yapılmış ve birebir emsal

Önceki bir denetim "Kaggle'da GDZ 2024 yoktur" diyip doğru atıfları silmişti —
Kaggle sayfası login'siz 404 verdiği için. **Yarışma gerçek** (24 Nisan –
8 Mayıs 2024, 192 takım, 4.575 submission) ve görevi 2026'nın provası gibi:

| | GDZ 2024 | 2026 beklentimiz |
|---|---|---|
| Hedef | İlçe × gün **bildirimsiz kesinti sayısı** | Benzer (kesinti sayısı/süresi) |
| Metrik | **MAE** | Muhtemelen MAE ailesi |
| Bölge | İzmir + Manisa (GDZ) | + Aydın/Denizli/Muğla (ADM) |
| Test | Şubat 2024 (tam ay) | 12 günlük yarışma, benzer blok |

Birinci: **Pikachow — Anıl Öztürk + Ahmet Tarık Karakaş** (final sunumu:
anilozturk.net/wp-content/uploads/2024/05/GDZ24-Datathon-Sunum-Pikachow.pdf).
Aynı Anıl Öztürk 2025'te CIBMTR'de (3.325 takım) AutoGluon'la 2. oldu —
muhtemel rakip profili "playground grandmaster" seviyesi.

### Pikachow mimarisi → bizde karşılığı

| Pikachow (kaynak: slayt) | Bizde | Durum |
|---|---|---|
| LightGBM + Optuna TPE, objective arama uzayında (s.23) | `tune_with_optuna(search_objective=True)` | ✅ vardı |
| SHAP geri eleme 490→97 feature (s.22) | `shap_backward_selection` | ✅ vardı |
| 25 seed × full-data → mean → **round** → **clip** (s.24) | `multi_seed_refit` + `postprocess_predictions` | ✅ vardı |
| Hava quantile'ları, bölge-geneli ("allstates") agregat (s.26) | `aggregate_hourly_to_daily`, `add_regional_aggregates` | ✅ vardı |
| Rolling(3,7,14,29,58) + expanding, shift-29 (s.18) | `add_rolling_features`, `add_expanding_features` | ✅ vardı |
| 7 doğrulama ayı time-series split, fold ort+std (s.21) | `purged_time_series_split` + raporlama | ✅ vardı (bizimki embargo'lu, daha katı) |
| **Geçen ay istatistikleri + last_month_same_day** (s.18, s.26 — importance tepesi) | `add_previous_month_features` | 🆕 **bu turda eklendi** |
| **İleriki 3-7-15 gün içinde bayram** (s.18) | `add_upcoming_holiday_features` | 🆕 **bu turda eklendi** |
| İlçe ID ana feature (s.26'da açık ara 1.) | kategorik `ilce_key` zaten modelde | ✅ |
| Yerel CV 2,77 > LB 1,71 — CV kötümserdi (s.25) | "CV'ye güven" ilkemizin saha kanıtı | 📌 not |

Sunumun yerel kopyası: `data/prior/av/gdz24-pikachow-sunum.pdf` (gitignore'da;
telif nedeniyle repoya konmadı).

---

## 2 · 2024–2026 meta taraması — beş hat, özet hükümler

**GBDT üçlüsü hâlâ taht'ta.** 2024'te 37 kazanan çözümde LightGBM/CatBoost/XGB;
2025 aynı. Çekirdeğimiz doğru. (mlcontests.com 2024+2025 raporları)

**TabM** (ICLR 2025, MLP-ansambl): CIBMTR 1.'sinde ve birçok üst sırada; tek
başına 25/3300. Bizim entity-embedding NN'den mimari olarak farklı → harmana
aday; pip wheel yeter, ağırlık dosyası yok. **Gün-1 sonrası opsiyon** — çekirdek
değil. **AutoGluon 1.x**: Playground'da 2026'ya dek birincilikler; bizim el
yapımı stack'e "ikinci görüş" olabilir ama offline paketi ağır. **TabPFN v2**:
50k satır sınırda, sıfır-şişkin kanıtı yok — düşük öncelik. **GRANDE/RealMLP**:
yarışma kanıtı sıfır — atla.

**Zaman serisi kazananlarının ortak paydası** (Rohlik ×2, Sticker Sales, VN1,
Jane Street): (1) LB'ye değil çoklu-pencere doğrulamaya güven; (2) çeşitlilikli
ansambl, çoğu düz ortalama; (3) takvim/tatil mühendisliği hiperparametreden
önce; (4) hedef dönüşümü ölçülür — **sqrt+L2 iki bağımsız madalyada MAE'yi ve
Tweedie'yi yendi**, log1p bir çözümde kazandırıp diğerinde kaybettirdi;
(5) basit baseline bekçisi. INFORMS 2025 kesinti yarışması 1.'si: **two-stage
hurdle + GBDT** — bizim ana mimarimizin aynısı.

**Küçük yarışma taktikleri:** adversarial validation ilk 24 saat; sabit-değer
LB probing submission yakar, yapma; kural sayfasında external-data maddesini
gün-1 oku (GDZ 2024'te belirsizdi); 3-5 seed ortalaması standart; takım
birleşme son tarihi **24 Ağustos 23:59** — erken birleş.

---

## 3 · Bu turda uygulanan ve ÖLÇÜLEN değişiklikler

**Kanonik sayılar `experiments/benchmark_gercek.json` dosyasındadır** (son koşu
2026-08-18, 177 sn, 61 feature). Bu bölüm o koşunun özetidir; çelişki görürsen
JSON kazanır.

| Değişiklik | Ölçüm | Hüküm |
|---|---|---|
| **İlçe kimliği + genişleyen ilçe istatistikleri** (`add_expanding_features`, ufuk=31) | catboost 304,30→**302,73**; lgb_mae 310,58→306,77; iki_asama_medyan 314,50→**301,81** | ✅ repodaki en geniş etkili tek feature değişikliği |
| **MAE-optimal medyan kuralı** (koşullu merdiven + q\*=1−0.5/p) | **301,81** — en iyi tekil | ✅ hurdle + medyan kuralı |
| **sqrt hedef dönüşümü** | 393,00 → **320,14** | ⚠️ 393 bir ARTEFAKTTI: koruma erken durdurmayı kapatıp 2000 sabit ağaç koşturuyordu. Artık `early_stopping_space="fit"`. Yine de kazanmıyor |
| **İzotonik kalibrasyon** | Brier 0,193→0,217 | ❌ kazandırmadı |
| **Tüm-üye hill-climb harmanı** | örnek-içi 298,59 ama **yuvalanmış 359,00** vs tek başına catboost **349,71** | ❌ yuvalanmış kontrolde GEÇMİYOR — gün-1'de kurma |
| **5 tohumlu CatBoost ortalaması** | tekil ort. 303,12 → **302,22** (yayılım 1,24; aralık 3,59) | ✅ yanlılıksız, ucuz; harmanın yerine bu |
| **12 harici aile ablasyonu** (`attach_external`) | lag +15,9 · CAPE +4,1 · EPİAŞ +3,1 · İZSU +2,7 · saatlik hava +2,7 · **tatil −3,2** · **güneş −1,2** · **günlük hava −0,6** | ✅ öncelik listesi artık ölçüm (docs/07 §4-3) |

> **Kapsam uyarısı (değişmedi):** bu sayılar 2021-22 verisi, `kesinti_dk`
> hedefi, 47 ilçe. 1. günde yeni veride yeniden ölçülür. Tohum gürültüsü
> ±1,24 dk ölçüldüğü için 2 dk'nın altındaki farklar hüküm taşımaz.

## 4 · Bilerek YAPILMAYANLAR

- **TabM/AutoGluon entegrasyonu** — kanıt güçlü ama 6 gün kala çekirdek
  değiştirmek risk; gün-1 sonrası "harman üyesi ekle" opsiyonu olarak duruyor.
- **Croston/ADIDA istatistiksel üye** — literatür "kısa/seyrek seride hâlâ
  rekabetçi" diyor; hill-climb zaten işe yaramayana 0 ağırlık verir. Gün-1
  sonrası ucuz deney.
- **Pseudo-labeling / per-group çarpan kalibrasyonu** — duruma bağlı,
  yarışma verisi görülmeden kurulmaz.
