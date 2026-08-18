# Sistem denetimi — 18 Ağustos 2026 (veri gününe 3 gün)

Altı bağımsız denetçi (sızıntı/doğrulama, veri günü hazırlığı, modelleme,
kod/test/CI, belge/yönetişim, harici veri bağlantısı) repoyu **salt-okunur**
taradı; her bulgu dosya:satır ve ölçümle doğrulandı, iddialar tekrar
çalıştırılarak sınandı. Bu belge altı raporun tekrarları birleştirilmiş,
önceliklendirilmiş sentezidir.

## Puan kartı

| Boyut | Puan | Ağırlık | Tek cümle |
|---|---:|---:|---|
| Sızıntı & doğrulama disiplini | **8** | %20 | Ufuk duvarı 8 feature ailesinde bozulma testinden geçti (0/28 sızıntı); tek açık: (grup, gün) tekilliği temporal.py'de denetlenmiyor |
| Veri günü hazırlığı (day_one) | **6** | %25 | Türkçe CSV, submission kapısı, ekip_kontrol sağlam; ama 2024 şemasıyla beslenince day_one **iki kez çöktü**, `_prova` klasörü gerçek veriyi gölgeliyor |
| Modelleme & deney kanıtı | **6** | %20 | Bit-bit tekrar üretilebilir; ama "harman 302,6" örneklem-içi (nested: CatBoost tek başına daha iyi), day_one metrikten bağımsız L2 eğitiyor, ilçe kimliği feature setinde yok (−5 MAE ölçüldü) |
| Kod / test / CI | **7** | %10 | Sözleşme testleri gerçek (mutasyon yok, 0 geniş except); 64 KB kodlama tespit hatası, jüri notebook'u üreticiden sapmış |
| Belge & yönetişim | **6** | %10 | Gizli anahtar hijyeni ve manifest kapısı gerçek; jüriye giden sayılar bayat, AFAD/KTB "kurum izni" kanıtsız, ayna CSV lisanssız |
| Harici veri bağlantısı | **5** | %15 | Anahtarlar %100 eşleşiyor, sızıntı temiz; ama 11 kaynağın **8'i yalnızca kütüphane fonksiyonu**, hiçbir pipeline çağırmıyor; hava 2026-08-09'da bitiyor, köprü yok |
| **Genel** | **6,4 / 10** | | **Güçlü kütüphane, kırılgan Gün-1 yolu.** Üç günde 8'e çıkarılabilir; aşağıdaki P0 listesi bunun için. |

Denetçilerin ortak yargısı: leaderboard'u bu modelleme katmanı değil,
**veri günü akışının kırılmadan çalışması ve harici verinin gerçekten
bağlanması** belirleyecek. Top-20 makul; top-10 için P0+P1 şart.

## Çapraz doğrulanan güçlü yanlar (kanıtlı)

- **Ufuk duvarı gerçek.** Lag/rolling/expanding/Hawkes/kitle-olay/önceki-ay/
  son-olay/komşu-lag: son 30 günün hedefi bozulunca 0/28 kolon değişti;
  D gününe tek spike, D+ufuk öncesi hiçbir satırda görünmedi. Testler kolon
  adı değil davranış ölçüyor (`test_derin_kazi.py:463`, `test_harici_sizinti.py:48`).
- **Sözleşme uygulanıyor, belgelenmiyor:** `test_sozlesme.py` her public
  feature fonksiyonunu otomatik keşfedip mutasyonsuzluk/satır sırası kontrol
  ediyor; `src/`de 0 `except Exception`, 405/418 fonksiyon tam tipli.
- **Türkçe CSV uçtan uca:** cp1254 + `;` + ondalık virgül + `TARİH;İLÇE` +
  `dd.mm.yyyy` doğru okunuyor, submission orijinal başlıkla yazılıyor;
  `write_submission` satır/ID/sıra/NaN'da fail-closed.
- **Tekrar üretilebilirlik:** benchmark tekil modeller son basamağa kadar
  aynı (310,58 / 304,30 / 338,77 / 393,00); provenance hash'leri; 10/10
  manifest hash'i diskle eşleşiyor; `.env` hiç commit'lenmemiş, tarih taraması 0.
- **Anahtarlar temiz:** her ilçe-düzeyi kaynak 96 referansla %100 eşleşiyor,
  0 duplicate, hava/güneş 0 eksik.

## Bulgular — birleştirilmiş, önceliklendirilmiş

Şiddet = "21 Ağustos'ta ne olur". Birden fazla denetçinin bağımsız bulduğu
maddeler ★ ile işaretli.

### P0 — Gün-1 akışını kırar (bugün-yarın kapat)

| # | Bulgu | Kanıt | Düzeltme |
|---|---|---|---|
| P0-1 ★ | `data/raw/_prova/` gerçek `train.csv`'yi **gölgeliyor**: `find_files` `rglob` + ilk aday | `day_one.py:171,184`; makinede `_prova` mevcut, uyarı aynı dosya adını yazıyor | Klasörü sil/taşı; `glob` (özyinelemesiz); uyarıda `relative_to` yaz |
| P0-2 | Düşük kardinaliteli sayım hedefi → profil "multiclass" → LightGBM çöker | `profiling.py:308-330`, `day_one.py:96`; 2024 şemasında EXIT=1 | Metrik regresyonsa (mae/rmse/…) `regression`'a çöz; docs/07+README komutuna `--task regression` |
| P0-3 | day_one **metrikten bağımsız L2** eğitiyor — benchmark'ta L2 (400) sıfır tabanının (367) altında | `day_one.py:513` `starter_params` objective yok; `metrics.py:10` "MAE→L1" | metrik→objective haritası (mae→`mae`/CatBoost `MAE`) |
| P0-4 | Bileşik `unique_id` (2024 emsali) test'te yok → CV bittikten sonra KeyError | `day_one.py:340,552`; docs/01:35 emsali kaydediyor | CV'den ÖNCE kontrol; sample deseninden id türet ya da net mesajla dur |
| P0-5 | Test sırası ≠ sample sırası → hata; `align_to_sample=True` hiçbir çağrıda yok | `submission.py:308`; grep 0 | `day_one.py:551`'e `align_to_sample=True` |
| P0-6 ★ | (grup, gün) tekilliği `temporal.py`/`pipeline.py`'de denetlenmiyor → olay-düzeyi veride ufuk duvarı **sessizce** yok olur (kanıt: 3 olay/gün, ufuk 7 → satırların %94'ü <7 gün) | `temporal.py:892,938,1014,1395,1482`; `pipeline.py:196-221`; guard yalnızca `spatial.py:310` | `_tek_satir_dogrula`'yı temporal/pipeline'a taşı; notebook 02 ham train yerine 01'in panelini kullansın + tekillik assert |
| P0-7 ★ | Jüri notebook'u 02 üreticiden **sapmış** (`CVRecipe` ambargo alanı yok → provenance "0 gün ambargo" der); Kaggle'da `strict_provenance` git olmadığından **submission yazıldıktan sonra** çöküyor; `sample_submission` okunmuyor | `02_baseline.ipynb:418` vs `build_notebooks.py:718-726`; `:745` store; `:694` sample yok | `build_notebooks.py` çalıştır + commit; `strict_provenance=not IS_KAGGLE`; sample= geç; byte-karşılaştırma testi |
| P0-8 ★ | Kaggle offline paketi **bayat**: wheel'de `national/point_events/tourism.py` yok, 6 modül farklı, `turizm_aylik_il`/`izsu`/`epias` yok, eski `yanginlar` | `kaggle_paket/` 17 Ağu 20:53; `VERI_DOSYALARI` manifestten kopuk | `VERI_DOSYALARI`'nı `sources.yml`'den türet; `--wheels --upload`; internetsiz Kaggle'da bir prova koşusu |
| P0-9 ★ | docs/05 ve docs/07 hava-join örnekleri `il_key`↔`konum_key` → **%0 eşleşme**; büyük harf kolon adları `read_any` sonrası KeyError; docs/07:190 lag örneği eski `lags` (61/92/123 kaydırıyor) | docs/07:141-145,190; docs/05:150-155 | `left_on=["ilce_key","tarih"]`, küçük harf; `shifts=[31,62,93]` |
| P0-10 | Hava arşivi **2026-08-09'da bitiyor**, tahmin köprüsü yok, NaN politikası yok → test bloğu Ağustos ortasını aşarsa 17 hava kolonu yalnızca testte NaN | `hava_gunluk` max; `fetch_weather.py:169-197` `today−6` tavanı | Open-Meteo forecast köprüsü (`past_days=92&forecast_days=16`) + `hava_kaynak` bayrağı; day_one ilk satırı "test max tarih vs hava max tarih"; kapsanmıyorsa ufuk-kaydırmalı hava |

### P1 — Skoru ve savunulabilirliği belirler (19-20 Ağustos)

| # | Bulgu | Kanıt | Düzeltme |
|---|---|---|---|
| P1-1 ★ | "Harman 302,6/303,75" **örneklem-içi**; nested (3 fold ağırlık, 4. fold skor) 305,49 vs CatBoost tek 304,30; tohum gürültüsü ~4 MAE | `benchmark_gercek.py:690-780`; ölçüm | `harman_ve_stack`'e nested kontrol; kazanamıyorsa 5-tohum catboost_mae gönder |
| P1-2 | Feature setinde **ilçe kimliği yok**; `ilce_key` kategorik + ufuk-güvenli expanding ilçe istatistikleri **−5,0 MAE** (304,3→299,3), en büyük tekil kazanç | ölçüm, aynı foldlar | `ozellik_kur`'a ekle; gerçek veride abone/nüfus statikleri + `init_score` |
| P1-3 | HORIZON = test süresi; train→test **boşluğunu yok sayıyor** (10 gün boşlukta CV lag 20 gün, test 30 gün); embargo `max(h,30)` yanlış gerekçeli, CV'yi 31 gün bayatlatıyor | `day_one.py:428,437`, `full_pipeline.py:192,287`, `validation.py:552,587` | `HORIZON=(test.max−train.max).days`; `embargo=boşluk` (bitişikse 0); docstring düzelt |
| P1-4 | sqrt dönüşüm verdiği (393) **artefakt**: guard erken durmayı kapatıp 2000 sabit ağaç koşturuyor; ES ile 326,5; docs/08 315,5 diyor | `models.py:111-127`; ölçüm | fit-uzayı ES metriği izni; benchmark yeniden; docs/08 güncelle |
| P1-5 ★ | Harici verinin 8'i **kütüphane-only** (`hava_saatlik_turev` parquet'ini hiçbir kod okumuyor; KTB/İZSU/EPİAŞ/yangın/deprem/okul yalnızca testte); orkestratör yok | bağlantı tablosu (grep) | `attach_external(panel, key, time, horizon)` tek orkestratör; her kaynak `ablation_gercek.py`'de aile olarak ölçülsün |
| P1-6 | EPİAŞ plansız kesinti geçmişi (96 ilçe, 2022→) hiç çekilmemiş: 78 satır/1 gün prova; betik+kimlik hazır | `epias/kesinti_plansiz.parquet`; `fetch_epias_outages.py` | Şimdi çek (~30 dk); 96 ilçede ablasyon; kural sorusu: bu veri serbest mi? |
| P1-7 | Nihai gönderim yolu benchmark konfigürasyonlarını üretemiyor: `multi_seed_refit` ağırlık/dönüşüm/iki-aşama geçirmiyor; fold-ortalaması test tahmini son ayı az kullanıyor | `refit.py:297-308`; `models.py:1325` | `sample_weight`/`target_transform` passthrough veya son-fold ağırlıklı test |
| P1-8 | Rastgele/iç içe test bölünmesi için kod yolu yok (455 gün ufuk → "hiç fold yok") | `day_one.py:371,451` | iç içe ise GroupKFold/KFold'a düş, nedenini yaz |
| P1-9 | `build_panel` `value_columns=None` → tüm sayısal kolonları topluyor (nüfus/kVA olay sayısıyla çarpılır, testte ham) | `panel.py:232-254`; day_one | `value_columns=[target]` |
| P1-10 ★ | UTF-8 dosyada 64 KB sınırı çok baytlı karakteri bölerse **sessizce cp1254** okunur → tüm ilçe adları bozuk, join %0 (11 MB gerçek dosyada ~%2,8 olasılık) | `io_utils.py:69,435-441`; yeniden üretildi | `UnicodeDecodeError.start >= len−3` ise kırp/incremental decoder; >64 KB test |
| P1-11 | Harici join'ler eşleşme oranını hiç teşhis etmiyor (%0 eşleşme = sessiz NaN) | `national.py:227`, `tourism.py:117`; `diagnose_join` yalnızca 2 script'te | merge sonrası %0 → ValueError, <%50 → warn (3 fonksiyon) |
| P1-12 | "il-ilçe" bileşik dizgeler (`izmir-karabağlar`) referansla eşleşmiyor; `strip_qualifier` sol tarafı alıyor | `turkish.py:190`; ölçüm | `split_il_ilce()` yardımcısı; `diagnose_join`'e ekle |
| P1-13 | Bilimsel kazanan kapısı **hiç ateşlenemiyor** (6 dış çapa ister, hiçbir betik üretmiyor); fold skorları JSON'da yok; `fold_std≈90` mevsim seviyesi, gürültü değil | `benchmark_gercek.py:137,862`; `evaluation.py:128` | fold skorlarını sakla; eşleştirilmiş farklar; `n_splits=8` aylık çapa |
| P1-14 | Benchmark hedefi dakika; 2024 hedefi **sayım**dı. Sayımda sıralama aynı (catboost_mae 2,363 en iyi) ama commit'li kanıt yok | `benchmark_gercek.py:113-118`; docs/14 | `--hedef adet` anahtarı, iki JSON commit |
| P1-15 | Hiperparametre gerçek veride hiç ayarlanmadı; 100 deneme ≈ 40 dk | tüm üyeler `starter_params` | 40 dk Optuna; yalnızca eşleştirilmiş kazanç > tohum gürültüsüyse al |

### P2 — Jüri (Kapı 2-3) güvenilirliği (20 Ağustos)

| # | Bulgu | Kanıt | Düzeltme |
|---|---|---|---|
| P2-1 ★ | AFAD/KTB "**kurum izni alındı**" manifestte olgu gibi; repoda/tarihte **hiçbir kanıt** (tarih, sayı, dosya) yok; `redistribution: allowed` olduğu için Kaggle'a **yüklenmiş** | `sources.yml:169-176,227-234,256-263`; docs/12:126-133 | Kanıtı ekle (`data/licenses/…` + `evidence_ref`) **ya da** `basis: unverified`, `local-use-only`, paketten çıkar |
| P2-2 | Git'le dağıtılan tek veri (ayna CSV, 11,3 MB) manifestte **yok**, lisans kaydı yok | `git ls-files data`; LICENSE "bkz sources.yml" | Kaggle lisansını kaydet ya da untrack + indirme komutu |
| P2-3 ★ | Jüriye giden sayılar bayat: docs/07 §5/§7 ve notebook tablosu bir önceki JSON'u (304,9/310,1/324,0/302,6, eşik 0,680), docs/08 iki önceki (312,7/315,5/0,606); JSON'da 393,00 / 303,75 / 0,72 | `benchmark_gercek.json` vs docs/07:209-231, `build_notebooks.py:588-615`, docs/08:84-88 | JSON'dan üret (runtime tablo); `test_belge_tutarliligi`'ye MAE tutarlılığı ekle |
| P2-4 | README:332 "**Sistem hiç gerçek veri görmedi**" — aynı README 68.257 gerçek kayıt diyor | README:240,332 | "gerçek *yarışma* verisi görmedi; 2021-22 GDZ aynasında prova" |
| P2-5 | Atıf yok: Open-Meteo CC-BY **zorunlu**, NASA metni manifestte hazır — notebook'ta 0 | grep | manifest `attribution` alanlarından tek markdown hücresi |
| P2-6 | Final seçimi **1 mi 2 mi?** docs/14:88 "TEK" [D] vs README/docs/01/05/07/09 "Final-2 kuralı" | 5 yer | Açılış günü soru listesine; beş yeri uzlaştır |
| P2-7 | Sunum iskeleti yok; docs/13 §7 zinciri (kesinti→planlı bakım→SAIDI/SAIFI→kalite faktörü→gelir tavanı) hiçbir yerde kullanılmıyor; `plot_prediction_timeline` import edilip çağrılmıyor; `business_impact` `crew_cost=1.0` | grep | `docs/17-sunum-iskeleti.md` (10 slayt) + `03_sonuclar` hücre bloğu |
| P2-8 | Bayat sayılar (doğrulandı): docs/00 "330 test/20 konum", README sözleşme sayıları 22/33 (gerçek 35/34), full_pipeline 21 vs 23 kontrol, "231.648×20" (×22), yangın "30.575…2024" (42.925…2026-08-17), docs/12 "8 artefakt/6 uyarı" (10/8), "60 saatlik yarışma" (12 gün), README modül haritası 16 script/8 modül eksik | tablo, belge raporu M1/M7 | Tek geçişte düzelt; `test_belge_tutarliligi` kapsamını genişlet |
| P2-9 | Runbook hataları: docs/05:135 `add_group_statistics` target zorunlu → hata; docs/05:144 `--districts` (eski 15 merkez) vs `--all-districts`; docs/01:85 `refit_full=True` API değil | docs | Düzelt |
| P2-10 | `neural.py` hiçbir yerde ölçülmüyor (CI omit, yerelde torch yok, `neural.yml` manuel); kapsam kapısı omit sayesinde geçiyor | `pyproject.toml:81`; `bdd84f3` | CPU-torch job (paths tetikli) ya da dürüst not; 21 Ağustos için ölü ağırlık |
| P2-11 | 103 `print` uyarısı vs 9 `warnings.warn`; 4 ölü public export; `turkish.py:8-12` docstring İ/ı kaybetmiş; `requirements.txt` yetim ve pyproject'le çelişiyor; "Kaggle pandas 2.x" anlatısı vs ölçülen 3.0.4 | kod raporu M1/M2/M4/M5 | Mekanik süpürme (yarım gün) |
| P2-12 | Sessiz FIRMS anahtar sızıntısı: 4xx/5xx'te URL'li hata mesajı MAP_KEY'i yazdırır | `fetch_yangin_api.py:119,131,142` | İstisna metninden anahtarı sil |

### Kabul edilen / eylem yok

- Aynı gün gözlenen hava feature olarak (ufuk 0) — kural/erişilebilirlik kararı; veri günü sorusu #6.
- Hawkes/kitle-olay kazancı iki bağımsız ailede aynı yönde (323→310, 322→305) — repodaki en güçlü feature kanıtı.
- Isotonic kalibrasyon sonraki fold etiketiyle (kaybeden varyant, ihmal edilebilir).
- Rüzgâr eşik saatleri dejenere (`ruzgar_20ms_saat` hep 0; ERA5 yumuşatılmış) — eşikleri 8/10/12 m/s'ye çek (P1-5 ile birlikte).

## Üç günlük plan

**18 Ağustos (bugün) — Gün-1 akışı kırılmasın:** P0-1…P0-6, P0-9 (yarım gün); day_one'ı elle yapılmış 2024-format fikstürle (`tarih, ilce="izmir-aliağa", bildirimsiz_sum`, `unique_id`) koştur ve `test_gun_bir_betigi`'ne ekle; P1-10/11/12 (yarım gün).

**19 Ağustos — Skor kolu:** P1-1, P1-2, P1-3, P1-4 (yarım gün, benchmark yeniden); P1-5 + P1-6 (yarım gün: `attach_external` + EPİAŞ çekimi + 96-ilçe ablasyon); P0-10 hava köprüsü.

**20 Ağustos — Jüri ve paket:** P0-7, P0-8 (notebook + wheel + Kaggle internetsiz prova); P2-1, P2-2, P2-3, P2-4, P2-5 (belge/lisans dürüstlüğü); P2-7 sunum iskeleti; P1-7 nihai gönderim yolu.

**21 Ağustos sabahı:** hava/saatlik/FIRMS/EPİAŞ yeniden çek; day_one ilk satırı kapsam kontrolü; açılış sorularına P2-6 (1 mi 2 mi) ve "harici veri serbest mi" ekle.

## Yöntem notu

Denetçiler paralel oturumun devam eden işini (`veri_sagligi.py`) kapsam
dışı tuttu. Denetim sırasında `.venv` başka bir oturumca yeniden kuruldu;
bazı ölçümler sistem Python 3.11 ile tekrarlandı ve venv sonuçlarıyla
birebir eşleşti. Ham raporlar (≈2.500 satır) oturum kaydında; bu belge
tekrarları birleştirir, hiçbir bulgu eklemez.
