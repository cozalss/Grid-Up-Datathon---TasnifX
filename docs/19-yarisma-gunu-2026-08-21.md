# Yarışma Günü — 21 Ağustos 2026

> **UYARI — 14:00 açılış yayınından SONRA yazıldı.** Bu belgenin 2. ve 3.
> bölümleri (bölge bulgusu, düşmanca prova) yarışmanın **kesinti tahmini**
> olduğu varsayımıyla yazılmıştı. Görev **tüketim tahmini** çıktı. Ölçümler
> doğru, ama bir kısmının konusu artık ilgisiz. Geçerli olan ve olmayan
> aşağıda ayrıldı.

---

## 0. GÖREV — açılış yayınında açıklandı (14:00)

**Kesinti değil TÜKETİM tahmini. İlçe değil TRAFO bazında.**

```
Eğitim   Ocak 2025 – Mart 2026      15 ay
Test     Nisan 2026 – Temmuz 2026    4 ay  (=122 gün ufuk)
Seviye   Trafo bazında · profil tarihi kırılımında
Hedef    Aktif tüketim (kWh, günlük)
```

Test dönemi **geçmişte** (bugün 21 Ağustos 2026) → hava tahmini gerekmiyor,
hepsi arşiv. Hava verimiz 2020-01-01 → 2026-09-05, test dönemini tamamen
kapsıyor.

### Veri alanları (slayttan, birebir)

| Alan | Açıklama |
|---|---|
| **Tanım Numarası** | Her trafoya ait tekil tanımlayıcı numara |
| **Güç** | İlgili trafonun **kurulu gücü** |
| **Profil Tarihi** | Tüketimin ölçüldüğü **tarih ve saat** |
| **Aktif Tüketim Günlük (kWh)** | Belirtilen profil tarihindeki aktif enerji tüketimi |
| **Lokasyon** | Trafonun bağlı olduğu **işletme/bölge** bilgisi |

> ⚠️ **EN KRİTİK BELİRSİZLİK: `Lokasyon` ilçe DEĞİL.** GDZ'nin iç işletme
> birimi. Bizim 232 dış kolonun tamamı `ilce_key` anahtarlı — doğrudan join
> **yok**. Veri açılınca ilk bakılacak şey budur.
>
> | Lokasyon ne çıkarsa | Ne yapılır |
> |---|---|
> | İl adı (`İZMİR`) | İlçe verisini `il_key` ile toplulaştır — 10 satır |
> | İşletme adı (`İZMİR KUZEY`) | Elle işletme→ilçe grubu haritası — 1-2 saat |
> | Opak kod | Yalnızca zaman bazlı aileler bağlanır; mekân bazlılar düşer |
>
> `data/reference/ilceler_gdz_adm.parquet` — 96 ilçe, **lat/lon**, nüfus,
> alan, şirket (GDZ/ADM). Koordinat verilirse eşleme kolay.

> ⚠️ İkinci belirsizlik: "Profil Tarihi = tarih **ve saat**" ama hedef
> "Aktif Tüketim **Günlük**". Trafo başına günde 1 satır mı 24 satır mı?
> Panel tanımını değiştirir.

### Dış veri AÇIKÇA serbest

> "Güvenilir ve doğrulanabilir dış veri kaynakları kullanılabilir"

Slaytta sayılanlar: sıcaklık, nem, yağış, rüzgâr, güneşlenme · resmî
tatiller, hafta sonları, özel günler · gün/hafta/ay, mevsimsellik, lokasyon,
trafo gücü. **Hepsi elimizde.** Liste kapalı değil ("kullanılabilecek");
turizm / arazi örtüsü / OSM listede yok ama yasak da değil.

### Jüri rubriği (slayt: "Veriyi Nasıl Okumalıyız?")

Notebook bu beşini **ölçerek** cevaplamalı — 2. elemenin anahtarı:

1. Her trafonun tüketim davranışı aynı mı?
2. Tüketim mevsimsel değişiyor mu?
3. Hafta içi / hafta sonu farkı var mı?
4. Trafo gücü ve lokasyon davranışı etkiliyor mu?
5. Hava koşulları ne kadar etkili?

### Aile envanteri — yeni göreve göre

| Durum | Aileler |
|---|---|
| **Doğrudan değerli** | `hava`, `hava_saatlik`, `nem_toprak`, `gunes` (organizatör saydı) |
| **Ayrıştırıcı** | `turizm_*` (Nis–Tem = Ege sezonu, listede YOK), `epias` (ulusal tüketim) |
| **Muhtemel** | `arazi_ortusu` (kentleşme→yoğunluk), `izsu` (İzmir, sezonluk nüfus) |
| **ÖLDÜ** | `konvektif`, `yangin`, `deprem`, `hava_kalitesi`, `osm_altyapi` — hepsi kesinti fiziğiydi. `maruziyet` rüzgâr×ağaç kolu da öyle. |

`maruziyet`in **kentsel kolu yaşıyor**: `sogutma × yerlesim`,
`sicak_sureklilik` — klima yükü mekanizması tüketim için birebir doğru.

**`Güç` doğal normalizasyondur:** `tüketim / güç` = kullanım oranı. Büyük
trafonun çok tüketmesi bilgi değil; kapasitesine göre ne tükettiği bilgidir.

### Bilinen kısıtlar

- **CV ince olacak.** 15 ay eğitim + 4 aylık doğrulama bloğu = en fazla 3
  fold, ilkinin eğitim tarafı boşalıyor (ölçüldü). Kararlar LB'ye yaslanacak.
- **Temmuz 2026 turizm verisi yok.** TÜİK ~2 ay gecikmeli; 4 ayın 3'ü kapsanıyor.
- **EPİAŞ uygunluk kapısı artık gereksiz** — kesinti verisi hedefin geçmişi
  değil. Kaldırmak manifestte tek bayrak (`model_girdisi`).
- `gun_sifir` takvim + veri sağlığı kapıları **anlamsızlaştı** (test dönemi
  geçmişte, hava tahmini gerekmiyor).

### İlk komut

```bash
python scripts/day_one.py --data data/raw --metric <RESMİ_METRİK>
```

---

## 1. KURAL RİSKİ — EPİAŞ kesinti geçmişi modele giremez

Coderspace'in **GDZ'22 Case-1** yarışması (bizimkiyle aynı problem: günlük
kesinti tahmini) kural sayfasında aynen şunu yazıyor:

> ⚠️ Arıza sonuç verileri internette halka açık olarak erişilebilmektedir.
> Notebook değerlendirme sürecinde, **bu verilerin kullanılmadığı ve modelde
> girdi olarak yer almadığı** konusu detaylı olarak incelenecektir.

Elimizdeki `data/external/epias/kesinti_plansiz.parquet` tam olarak o veridir:
405.819 gerçek GDZ+ADM plansız kesinti kaydı, 2022-01-01 → 2026-08-17.

Aynı ailenin **GDZ 2023** yarışması ise tam tersini söylüyor:

> Ekiplerin **kullandıkları dış veriler**, üretilen feature'lar, alternatif
> çözüm yöntemlerinin kullanılıp kullanılmadığı gibi kriterler
> değerlendirilecektir.

**Ayrım net:** kural dış veriyi değil, **hedefin geçmişini** yasaklıyor —
çünkü o, sızıntının ta kendisi.

| Veri | Durum |
|---|---|
| Hava, arazi örtüsü (WorldCover), OSM altyapı, turizm, deprem, yangın | Serbest ve **ödüllendiriliyor** |
| EPİAŞ tüketim/üretim (yük verisi) | Serbest — kesinti sonucu değil |
| **EPİAŞ kesinti geçmişi** | **Model girdisi olamaz** — prova zemini olarak serbest |

### Kod düzeyinde zorlanıyor

Manifestte prosa bir uyarı zaten vardı. Bir JSON dosyasındaki cümle hiçbir
kodu durdurmaz. Artık üç katman:

1. **Manifest** — 3 artifact `model_girdisi: false` + kural alıntısı
2. **Statik** — `gridup.uygunluk` `src/gridup/` altında referans arar
3. **Çalışma anı** — `features/external.py` ithal anında aile→yol haritasını denetler

`scripts/` bilerek dışarıda: provanın bu veriyi okuması **gereklidir**; yasak
olan modele girdi olmasıdır.

### Açılış yayınında sorulacak

1. Resmî metrik nedir? (2024 GDZ emsali MAE)
2. Harici veri serbest mi?
3. **Kamuya açık kesinti geçmişi kullanılabilir mi?**
4. Ticari olmayan lisanslı model ağırlıkları (TabPFN-2.5) serbest mi?
5. Günlük gönderim hakkı kaç, final kaç gönderim?
6. Notebook değerlendirme rubriği nedir?

---

## 2. ÖLÇÜLMÜŞ BÖLGE BULGUSU — literatür Ege'de tersine dönüyor

Kesinti literatürü (ABD/İngiltere) tek bir mekanizma anlatır: rüzgâr hattın
kendisini nadiren koparır, **ağacı** devirir. Dolayısıyla ağaç örtüsü bir
çarpandır.

**162.240 satırlık gerçek EPİAŞ paneli × hava verisiyle ölçtük:**

```
yerlesim_orani   -> kesinti   rho = +0,155      <- EN GÜÇLÜ TEK SİNYAL
agac_orani       -> kesinti   rho = -0,058      <- NEGATİF
ruzgar × agac    -> kesinti   rho = -0,031      (yaz)
sicak × yerlesim -> kesinti   rho = +0,136      (yaz)
```

Ege'de kesinti **sayısını** belirleyen şey fırtına değil, **şebekenin
büyüklüğü**. Sebep tablosu bunu doğruluyor:

| Sebep | Kayıt |
|---|---|
| İç Tesisat | 78.621 |
| OG Fider Açması | 72.269 |
| AG Pano Kol Sigorta Atışı | 28.727 |
| AG Sigorta Atışı | 23.838 |
| AG Havai Branşman Arızası | 20.526 |

Ekipman arızası, fırtına hasarı değil.

### İki rejim var ve mevsime göre ayrışıyor

```
ay   ort.kesinti   rho(rüzgâr, kesinti)
01      2,36            +0,264     <- fırtına rejimi
02      2,69            +0,095
07      3,82            +0,081     <- EN YÜKSEK kesinti, EN DÜŞÜK rüzgâr
08      3,64            +0,048
11      1,88            +0,182

kış (11-3): +0,155        yaz (6-8): +0,050
```

**Yaz kesintileri termal/yük kaynaklı, kış kesintileri fırtına kaynaklı.**
Yarışma dönemi (ağustos sonu – eylül) yaz rejiminin içinde.

Bu yüzden `gridup.features.maruziyet` iki kollu:

- **Fırtına kolu** (literatür): `rüzgâr × ağaç`, `hamle_saat × ağaç × yaprak_mevsimi`,
  `rüzgâr × ağaç × öncül_yağış_7g`, `… × toprak_nem`
- **Kentsel kol** (ölçüm): `soğutma × yerleşim`, `sıcak_süreklilik`,
  `sıcak_süreklilik × yerleşim`, `şebeke × soğutma`

> `sıcak_süreklilik` = üst üste kaçıncı sıcak gün. Trafo ilk sıcak günde
> değil, soğumaya fırsat bulamadığı üçüncü-dördüncü gününde arızalanır.

### Skor iddiası yok

Düşmanca provada MAE 3,251 (etkileşimsiz) → 3,277 (12 etkileşimle). Fark
**0,026**; ölçülmüş tohum gürültüsü **1,078**. Bu fark gürültünün içinde ve
"faydalı" ya da "zararlı" **denemez**. Ölçülen şey korelasyonlardır
(162.240 satır), MAE kazancı değil. Karar, gerçek yarışma verisinde çok
tohumlu ablasyona bırakılır.

---

## 3. DÜŞMANCA PROVANIN YAKALADIĞI ÜÇ KÖR NOKTA

Gerçek kesinti verisinden hasım biçimli bir yarışma dosyası üretip
`day_one.py`'ı üstünde koşturduk (cp1254 + `;` + ondalık virgül + Türkçe
başlıklar + BÜYÜK HARF ilçe adları + bileşik ID + sabit/boş kolonlar).
Üç arıza çıktı ve **üçü de sessizdi**.

| # | Arıza | Ölçülen sonuç |
|---|---|---|
| 1 | Sezilen grup kolonu `args.group_column`'a **geri yazılmıyordu** | 27 feature (beklenen 260); 15 dış ailenin hiçbiri bağlı değil |
| 2 | Nitelikli ilçe adları (`BOZKURT / DENIZLI`, `AYDIN MERKEZ`) eşleşmiyordu | 96 ilçenin 91'i eşleşti (%94,8) → ne hata ne uyarı; 5 ilçenin 232 dış kolonu NaN |
| 3 | Metin hedef 7/7'de çöküyordu | Profil + CV + feature + 5 tohumlu yeniden eğitim koştuktan **sonra**; gönderimsiz geçen bir saat |

`attach_external` %0'da durur, %50 altında uyarır. Gerçek veri tam **aradaki
kör banda** düştü. Şimdi `hizala_ilce_anahtarlari` var ve her kurtarma
**referansa koşullu** — aday referansta yoksa ad değiştirilmez, `BULUNAMADI`
diye raporlanır.

Bonus: üreteç ilk denemede test dosyasına aynı günün sonuç kolonlarını
koymuştu ve **sızıntı kapısı koşuyu haklı olarak durdurdu**
(Spearman 0,9935 / 0,9775). Kapının çalıştığının kanıtı.

### Dört senaryo — biçim kadar ŞEKİL de değişebilir

Yukarıdaki üç arıza tek bir *şekilde* bulundu: günlük sayım + MAE. Ama
yarışma verisinin şekli de değişebilir ve her şekil ayrı bir kör nokta
saklıyor. Dördünü de koşturduk:

| Senaryo | Neden olası | Sonuç |
|---|---|---|
| `sayim` | 2024 GDZ emsali (MAE) | ✅ geçerli submission |
| `ikili` | **GDZ'22 Case-1 emsali (F1)** | ❌ **hat komple çöküyordu** → düzeltildi |
| `soguk_ilce` | Test'te train'de olmayan ilçeler | ✅ geçerli submission |
| `ic_ice` | Test günleri train'in arasında | ✅ sızıntı kapısı durdurdu (doğru) |

**`ikili` senaryosu ilk koşuda hattı komple çökertti:**

```
ValueError: early_stopping_metric='f1', lightgbm icin desteklenmiyor
```

Hata mesajı doğruydu — F1 bir *eşiğe* bağlıdır, tur başına değerlendirilemez.
Ama doğru cevap durmak değil: olasılık temelli bir vekille (logloss) erken
durdurup **eşiği sonradan, fold-dışı tahminlerde optimize etmek**.

> AUC değil logloss seçildi. İkisi de eşikten bağımsız ama AUC yalnızca
> *sıralamayı* ölçer; iyi sıralanmış ama kötü kalibre olasılıklar AUC'yi
> yüksek gösterip tam da eşik aramasını bozar.

İkinci yarısı daha sinsiydi: `optimize_threshold` depoda **zaten vardı**,
iyi belgelenmişti — ve hiçbir yerden çağrılmıyordu. Yani F1 senaryosunda
0,5 eşiğiyle gönderim yapılacaktı. Bu panelde günlerin %65'i sıfır;
dengesiz veride 0,5 neredeyse hiçbir zaman optimum değildir.

Üçüncüsü ilk çalışan koşuda çıktı:

```
Esik optimizasyonu: esik=0.010  f1=0.8996   (0,5'te de 0.8996)
```

Berabere ama "her şeye evet de" eşiği seçilmiş — `np.argmax` ilk maksimumu
döndürüyor, ızgara küçükten büyüğe gidiyor. OOF'ta yalnızca *berabere* kalan
uç bir eşik, kuyruktaki birkaç örneğe uyuyor demektir. Artık beraberlikte
**0,5'e en yakın** eşik kazanıyor: kanıt yokken varsayılana yaslanmak doğrudur.

`ic_ice`'nin durdurulması kusur değil kanıttır — serpiştirilmiş bölme gerçek
bir sızıntı tehlikesidir ve varsayılan olarak geçilmemelidir. Açık bayrakla
devam edilince `forecast_geometry` iç içeliği tespit ediyor, zaman-ileri
şemayı **uygulamıyor** ve GroupKFold'a düşüyor.

`gun_sifir.py` artık dördünü de kapı olarak koşuyor.

---

## 4. KOŞU SIRASI

```
python scripts/gun_sifir.py          # hazır mıyız -- 4 kapı + plan
```

Veri geldiğinde:

| # | Adım | Komut |
|---|---|---|
| 1 | Ham dosyaları aç | `data/raw/` altına |
| 2 | Biçimi okumadan tespit et | `sniff_dialect('data/raw/train.csv')` |
| 3 | **İlk gönderim** (~1 dk) | `python scripts/day_one.py --data data/raw --metric <METRİK>` |
| 4 | Kaggle'a yükle, LB skorunu deftere yaz | format doğru mu — iyi skor sonra |
| 5 | LB–CV farkını ölç | `python scripts/benchmark_gercek.py` |
| 6 | Hangi aile faydalı? | `python scripts/ablation_gercek.py --tohum 5` |
| 7 | Hiperparametre | `python scripts/tune_gercek.py --model catboost` |

### Operasyonel not — hava köprüsü tazelenmeli

Open-Meteo forecast API'sine doğrudan sorularak **ölçüldü** (2026-08-21):

```
cape, wind_speed_10m, wind_gusts_10m, pressure_msl   384 saat (16 gün)
soil_moisture_0_to_1cm                               184 saat (~7,7 gün)
```

`kopru_saatlik.py` bilerek **en zayıf kaynağa kırpar** — bazı ailelerin dolu
bazılarının boş olduğu aralık, hepsinin eksik olduğu aralıktan tehlikelidir.
Sonuç: köprü, kapsanması istenen son günden en fazla 7 gün önce koşulmalı.

> **2026-09-01'i kapsamak için en geç 2026-08-25 civarı `kopru_saatlik.py`
> tekrar koşulmalı.** `gun_sifir.py` takvim kapısı bunu raporluyor.

### Atıf hücresi (notebook'a aynen)

```
Weather data by Open-Meteo.com (CC BY 4.0)
ESA WorldCover 10m v200 (CC BY 4.0)
Map data from OpenStreetMap contributors (ODbL 1.0)
TÜİK turizm istatistikleri | AFAD deprem kataloğu | NASA FIRMS yangın tespitleri
```

---

## 5. DOĞRULANMIŞ YARIŞMA BİLGİSİ

| | |
|---|---|
| Platform | Kaggle **In-Class** (sayfa davetli) |
| Takvim | 21 Ağu 15:00 → 1 Eyl 23:59 · sunumlar 7-10 Eylül |
| Takım kurma son gün | **24 Ağustos 23:59** (Kaggle üzerinde) |
| Ödül | 75.000 / 50.000 / 25.000 TL |
| Emsal metrik | GDZ 2024 (günlük ilçe kesinti sayısı) → MAE |
| Gönderim | 2023'te günde 3, final 1-2 |
| Jüri | Private LB → ilk 30 notebook → 10 takım |

Kaynak: [coderspace.io/etkinlikler/grid-up-datathon](https://coderspace.io/etkinlikler/grid-up-datathon/).
2026 için metrik, submission formatı, LB bölünmesi ve dış veri politikası
**yayımlanmamıştır** — hepsi davetli Kaggle sayfasının Overview → Evaluation
ve Rules sekmelerinde.
