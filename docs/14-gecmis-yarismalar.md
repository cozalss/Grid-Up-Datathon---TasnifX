# Geçmiş Yarışmalar — GDZ/ADM ne sordu, nasıl sordu

`docs/08` 2024 birincisinin **çözümünü** analiz ediyor. Bu belge farklı bir soruyu
cevaplıyor: **sponsor her seferinde NE SORDU ve soru nasıl evrildi?**

Etiketler: `[D]` doğrulandı · `[?]` tek kaynak · `[—]` bilgi bulunamadı.
Araştırma: 17 Ağustos 2026.

---

## 1 · Dört edisyon, tam tablo

| | **2022** (iki vaka) | **2023** | **2024** | **2026 Grid Up** |
|---|---|---|---|---|
| Tarih `[D]` | 20 Ağu – 15 Eyl | 7 – 23 Nis | 24 Nis – 8 May | 21 Ağu – 1 Eyl |
| **Süre** `[D]` | **26 gün** | **17 gün** | **15 gün** | **12 gün** |
| Soru | **V1:** plansız kesinti tahmini · **V2:** kesintiden doğan **çağrı sayısı** | Dağıtılan **enerji** tahmini | İlçe × gün **kesinti sayısı** | **?** |
| Hedef | V2: `Cagri_Count` `[D]` | `Dağıtılan Enerji (MWh)` `[D]` | `bildirimsiz_sum` `[D]` | ? |
| **Granülerlik** | V2: **olay kaydı** `[D]` | şebeke merkezi × gün `[?]` | **ilçe × gün** `[D]` | ? |
| **Metrik** `[D]` | V1: **F1** · V2: **RMSE** | **MAPE** | **MAE** | ? |
| Kapsam | GDZ (İzmir+Manisa) | GDZ | GDZ | **GDZ + ADM, 5 il** |
| Katılım | `[—]` | 234 takım `[D]` | 192 takım · 4.575 gönderim `[D]` | ? |
| Ödül `[D]` | 30/20/10 + 5 mansiyon | `[—]` | 50/40/30 | **75/50/25** |
| Günlük submission `[D]` | 3 | 3 | `[?]` | ? |
| Jüri sunumu `[D]` | ilk 10 | ilk 30 → ilk 10 | ilk 10 | ilk 10 (veya 20) |

---

## 2 · Örüntüler — asıl değer burada

### 2.1 · Metrik HİÇ tekrarlamamış `[D]`

**F1 → RMSE → MAPE → MAE.** Dört yarışma, dört farklı metrik. Emsalden 2026'nın
metriğini tahmin etmek bu yüzden zayıf bir bahis.

**Sonuç:** `metrics.py`'nin altı metriği birden desteklemesi bir fazlalık değil,
bu örüntünün doğrudan gereği. Metrik 21 Ağustos'ta öğrenilecek ve `config.py` tek
anahtarla dönecek — doğru tasarım.

### 2.2 · Süre her yıl kısalıyor `[D]`

**26 → 17 → 15 → 12 gün.** Dört yılda süre yarıdan aza indi, ödül ise iki buçuk
katına çıktı (60k → 150k TL).

**Sonuç:** Hazırlık avantajının değeri her edisyonda **artıyor**. 26 günde
altyapıyı yarışma sırasında kurabilirdiniz; 12 günde kuramazsınız. Bu repo tam
olarak bu eğilime karşı bir bahis.

### 2.3 · Granülerlik sabit değil — en riskli varsayım burada `[D]`

- **2022 V2:** tek tek **kesinti olay kayıtları** (her satır bir arıza; kolonlar
  arasında sebep, süre, kentsel/kırsal hat sayıları, kümülatif kesinti saatleri)
- **2023:** zaman serisi (şebeke merkezi × gün)
- **2024:** **ilçe × gün paneli**

Yani sponsor aynı veriyi **üç farklı şekilde** sunmuş. `build_panel()`'in var olma
sebebi tam olarak bu: olay kaydı gelirse panele çevirmek gerekir; panel gelirse
çevirmeye gerek yoktur.

> **Uyarı:** 2024 formatını (ilçe × gün) 2026 için garanti saymayın. 2022'de olay
> kaydıydı ve 2026'da ADM'nin de katılması veri şemasını yeniden şekillendirebilir.

### 2.4 · Sınıflandırma daha önce sorulmuş `[D]`

2022 V1 **F1** ile ölçüldü — yani sayım/regresyon değil, **sınıflandırma**. Önceki
tahminimde "sınıflandırma ~%10" demiştim; bu bulgu o olasılığı yükseltiyor. `metrics.py`
içindeki eşik optimizasyonu ve `two_stage.py` bu senaryoyu zaten karşılıyor.

### 2.5 · Aynı anda İKİ vaka sorulabilir `[D]`

2022'de tek yarışmada iki ayrı problem vardı (kesinti + çağrı). 2026'da ADM ve GDZ
birlikte olduğu için **iki şirket = iki alt problem** kurgusu mümkün. Açılış
yayınında sorulacak.

### 2.6 · Çağrı merkezi bir kez hedef olmuş `[D]`

2022 V2'nin hedefi `Cagri_Count`'tu. Bu, `docs/13`'teki bulguyla örtüşüyor: çağrı
merkezi memnuniyeti EPDK'nın izlediği kalite göstergelerinden biri ve ADM 2016'da
çağrı merkezi kapasitesini dört katına çıkarmış. Yani **çağrı yükü tahmini şirket
için canlı bir problem** — 2026'da tekrar gelebilir.

---

## 3 · Sabit kalan operasyonel kurallar `[D]`

Bunlar dört edisyonda da değişmedi ve 2026 için en güvenilir varsayımlar:

- **Günde 3 submission** (GDZ markalı üç yarışmanın üçünde de)
- **Final için TEK submission seçimi** — Kaggle'ın standart 2 seçim hakkı değil
- **Public/private %50/%50** (taranan 11 Coderspace yarışmasının 10'unda)
- **Test seti ileri zamanlı blok**
- **Veri %100 CSV**, ayırıcı virgül, başlıklar ASCII/İngilizce ama **değerlerde
  Türkçe karakter var**
- **İki aşamalı değerlendirme:** Kaggle sıralaması + jüri önünde model sunumu
- **Takım kurma penceresi dar** (2024'te yarışma başladıktan sonra 3 gün; 2026'da
  **24 Ağustos**)

---

## 4 · 2023'ün özel dersi: yardımcı dosya `[D]`

2023'te `med.csv` verildi — **Major Event Day**, yani kesinti süresinin kabul
edilebilir limiti aştığı günler. Sayfa açıkça şöyle diyordu:

> *"Kesinti olduysa o gün tahmin edilen enerjinin sapmasının yüksek olmasını
> bekleriz. Çünkü kesinti nedeniyle dağıtılamamış."*

Yani sponsor, metriğin (MAPE) patlayacağı günleri **önceden işaretlemişti**.
2024'te aynı rolü `bildirimli_sum` (planlı kesinti) oynadı ve **test setinde de
veriliyordu** — bilinen-gelecek kovaryat.

**Gün-1 refleksi:** `set(test.columns) - {ID}` boş mu? Boş değilse orada bedava
sinyal var ve rakiplerin çoğu kullanmıyor.

---

## 5 · 2025'te edisyon yapılmamış görünüyor `[?]`

2022, 2023, 2024 doğrulandı; 2025 için hiçbir kayıt bulunamadı. Grid Up 2026,
markanın yeniden ve **daha büyük** (iki şirket, 5 il, en yüksek ödül) dönüşü
gibi duruyor — ve `docs/13`'teki ADMS/OMS programıyla aynı yıla denk gelmesi
tesadüf değil.

---

## 6 · Kaynaklar

- 2022 etkinlik sayfası: [coderspace.io/en/events/gdz-elektrik-datathon](https://coderspace.io/en/events/gdz-elektrik-datathon/)
- 2022 V2 çözümü (hedef ve kolonlar): [github.com/arukemre/Gediz-Elektrik-POWER-OUTAGE](https://github.com/arukemre/Gediz-Elektrik-POWER-OUTAGE)
- 2023 etkinlik sayfası: [coderspace.io/en/events/gdz-elektrik-datathon-2023](https://coderspace.io/en/events/gdz-elektrik-datathon-2023/)
- 2023 Kaggle meta + sayfa metinleri: `data/prior/gdz_pages_clean.txt` (depoda)
- 2024 etkinlik sayfası: [coderspace.io/etkinlikler/gdz-elektrik-datathon-2024](https://coderspace.io/etkinlikler/gdz-elektrik-datathon-2024/)
- 2024 sonuç ve katılım: [Coderspace başarı hikayesi](https://coderspace.io/basari-hikayeleri/gdz-elektrik/)
- 11 yarışmalık metrik/kural taraması: `data/prior/istihbarat/sonuc_5.json` (depoda)
- 2026: [Grid Up etkinlik sayfası](https://coderspace.io/en/events/grid-up-datathon/)
