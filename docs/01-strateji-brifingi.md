# Strateji Brifingi

13 agent'lık derin araştırmanın (web + GitHub + Kaggle + geçmiş çözümler) adversarial
doğrulamadan geçmiş sentezi. Kaynak etiketleri: `[D]` = birincil kaynaktan doğrulandı,
`[?]` = doğrulanamadı, dikkatli kullan.

---

## 0 · Önce bir çelişki: ilk 10 mu, ilk 20 mi?

| Kaynak | Diyor ki |
|---|---|
| Size gelen e-posta (17:28, "Güncel Program") | "Private Leaderboard'da **İlk 20**'ye girenlerin Notebookları" |
| coderspace.io yarışma sayfası (TR + EN) `[D]` | "veri etabında **ilk 10 sırada** yer alan ekipler" |

**Bunu çözmeyin — daha katı olana göre planlayın.** İlk 10'u hedefleyin. E-posta daha
güncel olabilir ama üzerine strateji kurulacak sayı, yanılma payı bırakmayan olmalı.
Açılış buluşmasında (21 Ağustos 14:00) **bunu ilk soru olarak sorun.**

2024'te 192 takım vardı `[?]`. 2026'da 5 il + 75.000 TL ödülle katılım daha yüksek olacak.

---

## 1 · En yüksek etkili 10 karar

| # | Karar | Neden |
|---|---|---|
| **1** | **24 Ağustos 23:59'a kadar Kaggle'da takım BİRLEŞMİŞ olmalı** `[D]` — ve **birleşmeden önce kimse submission yapmasın** | Kaggle'ın merge kuralı: birleşen takımın toplam submission'ı, merge anındaki tek-takım limitini aşamaz. 4 kişi 3 gün 5'er submission yaparsa 60 > 15 → **merge reddedilir** → takım kurulamaz. Sıfır maliyetle önlenebilir tek felaket. |
| **2** | **2024'ün metriği MAE'ydi, RMSE değil** `[D]` — birincilik takımının kendi sunum slaytından | MAE ⇒ optimal tahmin koşullu **medyan**. `objective="mae"` (L2 değil), ve tam sayı hedefte **yuvarlama** doğrudan skor kazandırır. Metrik 21 Ağustos'ta netleşecek; ama MAE ihtimaline hazır olun. |
| **3** | **Lag ufkunu tahmin ufkuna sabitleyin** `[D]` | 2024 birincisinin en yüksek önemli feature'ları `shift_29_rolling_3_sum` ve `shift_29_expanding_sum`. Test bloğu bir aylıksa, ayın 28'ini tahmin ederken **1 günlük lag yoktur**. `shift(1)` ile hesaplanan rolling CV'de harika görünür, private LB'de çöker. → Kodda `horizon` parametresi olarak uygulandı. |
| **4** | **Hava durumunda ortalama değil MAX ve QUANTILE** `[D]` | Birincinin importance listesinin tepesi: `wind_dir_10m_..._q01`, `effective_cloud_cover_..._q08`, `wind_speed_10m_max`, `t_apparent_..._q01`. Hasarı rüzgârın ortalaması değil **tepesi** yapar. Ayrıca hem ilçe bazında hem **bölge geneli** (`allstates`) ayrı setler üretilmiş. |
| **5** | **Validasyon: 5–7 aylık expanding-window, karar kriteri mean VE std** `[D]` | Birinci 7 doğrulama ayı kullandı ve fold hatalarının **ortalamasını ve standart sapmasını** birlikte raporladı. Kritik gözlem: local 2.77 → LB 1.71; ikinci konfigürasyon local **2.80 (daha kötü)** → LB **1.69 (daha iyi)**. Yani local ≈ LB × 1.63 ve **sıralama korelasyonu mükemmel değil.** Bunu bilmeyen takım panikleyip LB'ye overfit eder. |
| **6** | **Panel yapısını sıfırla doldurun** `[D]` | "O gün kesinti olmadı" satırları veri setinde **eksik gelebilir**. Doldurulmazsa lag/rolling kayar ve model sıfır tahmin etmeyi öğrenemez. → `build_panel()` olarak eklendi. |
| **7** | **Komşu ilçe (spatial lag) feature'ı** `[D]` | Fırtına ve sıcak hava dalgası ilçe sınırı tanımaz; komşu ilçedeki dünkü kesinti bugünkü için güçlü sinyaldir. Birinci çözümde açıkça kullanılmış — ucuz ve yüksek getirili. |
| **8** | **Türkçe join'i 1. günde çözün** `[D]` | `'İ'.lower()` iki kod noktası döner; ilçe adıyla hava/koordinat/nüfus join'i **sessizce sıfır satır** verir. Ayrıca 2024'te `ilce` kolonu `şehir` + `ilçe` olarak **ikiye bölünüyordu** — yani birleşik string taşıyordu. |
| **9** | **Ortam asimetrisi: numpy Kaggle'da DAHA ESKİ** `[D]` | Kaggle v170-CPU: Python 3.12, pandas 3.0.4, **numpy 2.0.2**, sklearn 1.9.0, optuna 4.9.0. Yerel: Python 3.11.9, pandas 3.0.3, **numpy 2.4.6**, sklearn 1.8.0. `np.astype()` serbest fonksiyonu numpy 2.1'de eklendi → **Kaggle'da YOK**. Notebook jüriye gideceği için yerelde çalışıp Kaggle'da patlayan kod eleme sebebi. |
| **10** | **Notebook'u son güne bırakmayın** `[D]` | Birincinin sunumunun son 3 slaytı tamamen **iş değeri**ydi: "ensemble ve açıklanabilir çözüm / regularized pipeline / daraltılmış feature-set / küçük model kümesi (~25MB) / yeni veriyle eğitilebilir". "Deployment maliyeti 25MB" gibi somut bir cümle, jüri (dağıtım şirketi iş birimleri) için soyut mimari anlatımından değerlidir. |

---

## 2 · Hizmet bölgesi

| | GDZ | ADM | Toplam |
|---|---|---|---|
| İller | İzmir, Manisa | Aydın, Denizli, Muğla | **5 il** |
| İlçe `[D]` | 47 | 49 | **96** |
| Mahalle `[D]` | 2.383 | — | — |
| Tüketici `[D]` | 3,85 mn | ~2,4 mn `[?]` | ~6,3 mn |
| OG fideri `[D]` | 1.970 | 3.406 | ~5.376 |

> **Çözülmemiş çelişki:** GDZ yüzölçümü Coderspace'te 13.123 km², gdzelektrik.com.tr'de
> ~26.000 km². İkisi de birinci elden kurumsal kaynak. Yüzölçümünü feature yapacaksanız
> TÜİK il alanlarından kendiniz hesaplayın.

---

## 3 · En olası problem tipleri

GDZ geçmişinde **üç farklı** hedef tipi görüldü `[?]`: çağrı sayısı (RMSE), MWh (MAPE),
kesinti sayısı (MAE). Bu yüzden `config.py` tek anahtarla üç senaryoya da dönebilir.

**2024 formatı** `[D]`: hedef `bildirimsiz_sum` (plansız kesinti sayısı), panel =
tarih × ilçe. `bildirimli_sum` **test setinde de veriliyordu** — yani ikinci hedef değil,
**bilinen-gelecek kovaryat**. Bakım takvimi ileriden bilindiği için planlı kesintinin
**ileri (lead)** rolling'i bedava sinyaldir ve rakiplerin çoğu bunu kullanmaz.

| Problem | Hedef tipi | Muhtemel metrik | İlk 3 feature |
|---|---|---|---|
| Plansız kesinti sayısı | sayım, sıfır-şişkin | MAE | ufuk-lag'li hedef geçmişi · rüzgâr max/quantile · tatil |
| Yük/tüketim tahmini | sürekli | MAPE / RMSE | derece-gün · takvim · geçen yıl aynı gün |
| Arıza/bakım önceliği | sınıflandırma | AUC / F1 | ekipman yaşı · yük stresi · geçmiş arıza |

---

## 4 · Kanıtlanmış prior art

| Yarışma | Kazanan yaklaşım | Buraya taşınabilir ders |
|---|---|---|
| **GDZ Datathon 2024** `[D]` | 490 → 97 feature SHAP backward selection; 25 seed full-data refit; mean blend + round + clip | Feature eleme **eğrisi** jüri sunumunda güçlü slayt; multi-seed neredeyse bedava kazanç |
| ASHRAE GEPIII `[?]` | İlk 5'in **hepsi** feature çokluğu yerine manuel outlier tespitine öncelik verdi | "Kayıt yok" ile "kesinti yok" karışması en olası veri tuzağı |
| M5 `[?]` | Tweedie NLL objective; hiyerarşik 220 model | Sıfır-şişkin sayım hedefinde `tweedie`/`poisson` dene |
| Enefit `[?]` | Hedefi kapasiteye bölerek normalize; Polars ile feature factory | `target / exposure` normalizasyonu MAE'de doğrudan kazanç |
| Rossmann `[?]` | "Day counters": olaya kaç gün kaldı / geçti | Planlı kesintiden N gün sonra plansız artıyor mu — doğrudan test edilebilir |

**`refit_full=True` etkisi** `[D]`: 2024 birincisinde skoru 3,02 → 2,95 taşıdı. Tek satır, ~%2.

---

## 5 · Kaggle operasyon el kitabı

### Takım kurma
1. Herkes **kendi hesabından** yarışmaya girer, kuralları kabul eder
2. Team sekmesi → **Send Merge Request** → karşı taraf **kabul eder** (çift onay)
3. Takım adı = başvuru formundaki isimle **birebir aynı**
4. **Son tarih: 24 Ağustos 23:59** `[D]`
5. **Birleşmeden önce submission YAPMAYIN**

### Validasyon karar ağacı
```
Test train'den SONRAKİ blok mu?
  EVET → purged_time_series_split(embargo ≥ en uzun rolling penceresi)
         + horizon = test blok uzunluğu
  HAYIR ↓
Tekrarlayan varlık var mı (trafo/ilçe/abone)?
  EVET → GroupKFold           HAYIR → Stratified/KFold
```

### Shakeup önleme
- **CV-LB korelasyonunu ölçün** (`ExperimentLog.cv_lb_correlation()`). r > 0.8 → CV'ye güven.
  r < 0.5 → CV şemanız yanlış, düzeltmeden devam etmeyin.
- 2024'te local ≈ LB × 1.63 ve sıralama korelasyonu mükemmel değildi `[D]` — **mutlak farkı
  değil, sıralamayı** izleyin.
- **Final submission: iki FARKLI aile seçin** — en iyi CV'li tek model + en iyi harman.
  İkisi de aynı aileden olursa risk çeşitlendirilmemiş olur.

### Diskalifiye riskleri
Birden fazla takıma katılım · takım penceresi dışında birleşme · etik dışı davranış ·
sponsor çalışanı/1. derece akraba

---

## 6 · Harici veri kataloğu

| Kaynak | Erişim | Key? | Çözünürlük | Join anahtarı |
|---|---|---|---|---|
| **Open-Meteo Archive** `[D]` | REST, `archive-api.open-meteo.com` | **Hayır** | saatlik, 1940→ | lat/lon → ilçe |
| holidays (TR) `[D]` | `pip install holidays` | Hayır | günlük | tarih |
| TÜİK ilçe nüfusu `[?]` | CSV indirme | Hayır | yıllık | il+ilçe |
| EPİAŞ Şeffaflık `[?]` | REST | Kayıt | saatlik | tarih |
| TR ilçe koordinatları `[?]` | açık GeoJSON repoları | Hayır | nokta | il+ilçe |

**Kaggle'da internet kapalı olabilir** `[D]` → yerelde indir, **Kaggle Dataset olarak yükle**,
notebook'ta `/kaggle/input/...` üzerinden oku. Canlı API çağıran notebook, API değişince
veya internet kapalıyken kırılır — ve o notebook jüriye gidiyor.

> `holidays` kütüphanesinin TR implementasyonu 1936–2032 arası İslami tarihleri önceden
> hesaplanmış sözlükte tutuyor ve **arife günleri için HALF_DAY kategorisi** sunuyor `[?]`.
> Rakiplerin çoğu bunu bilmiyor.

---

## 7 · Regülasyon uyarısı

**1 Ocak 2026'da "bildirimli" tanımı genişledi** `[?]`: gece 00–06 arası 30 dk altı manevra
kesintileri artık bildirimli sayılıyor. Test seti bu tarihten sonraya düşüyorsa
bildirimli/bildirimsiz oranı **yapısal olarak kaymıştır**; bayraklanmazsa model geçmişin
oranını ezberler.

---

## 8 · 12 günlük plan (21 Ağustos – 1 Eylül)

| Gün | Odak | Çıktı |
|---|---|---|
| 1 | Açılış, veri, profil, CV şeması, **ilk submission** | LB'de bir skor |
| 2–3 | Panel + takvim + hedef geçmişi (ufuk-lag'li) + hava | 100+ feature |
| 4–5 | Komşu ilçe, exposure normalizasyon, model zoo (LGB/XGB/Cat) | OOF matrisi |
| 6–7 | Optuna (objective'i de arama uzayına koyun), hata analizi | tuned params |
| 8–9 | SHAP backward selection, hill climbing harman | daraltılmış set + harman |
| 10–11 | Multi-seed full-data refit, notebook temizliği | final model |
| 12 | **Final submission seçimi**, writeup kilitleme | teslim |

**Son gün kuralı:** yeni fikir denemeyi bırakın. Son gün yapılan değişikliklerin çoğu
public LB'ye uyum sağlar ve private'da zarar verir.

---

## 9 · Açık sorular (21 Ağustos'ta sorun)

1. İlk 10 mu ilk 20 mi? (yukarıdaki çelişki)
2. Resmî metrik?
3. Harici veri serbest mi?
4. Günlük submission limiti ve final submission sayısı?
5. Public/private bölünmesi hangi kritere göre?
6. Notebook Kaggle'da çalışmak zorunda mı, internet açık mı?
7. Notebook değerlendirme kriterleri ve ağırlıklandırma?
8. Final sunumu kaç dakika?

---

## 10 · Araştırmanın güvenilirlik notu

13 agent'lık workflow'un **8'i tamamlandı, 4'ü bağlantı hatasıyla düştü**
(coderspace-ekosistem araştırması ve 3 doğrulama ajanı). Bu yüzden:

- domain ve tooling raporları **adversarial doğrulamadan geçti** → `[D]` etiketleri güvenilir
- prior-art, kaggle-craft, external-data raporları **doğrulanmadı** → `[?]` etiketli
  maddeleri kendi kaynağınızdan teyit edin

Sentez ajanı ayrıca bağımsız birincil doğrulama yaptı: Coderspace TR+EN sayfaları, 2024
birincisi Pikachow'un 29 slaytlık sunum PDF'i, 2024 üçüncüsünün notebook'u, Kaggle
docker-python v170-CPU sürüm notları. `[D]` etiketleri bu doğrulamalara dayanıyor.
