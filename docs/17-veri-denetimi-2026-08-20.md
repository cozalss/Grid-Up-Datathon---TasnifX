# Harici veri denetimi — 20 Ağustos 2026

Bu belge, yarışmadan bir gün önce yapılan harici veri turunu kaydeder: dışarıdan
gelen bir eleştirinin madde madde **ölçülerek** doğrulanması, doğrulananların
kapatılması, yanlış çıkanların çürütülmesi ve bu tur sırasında bulunan yeni
hataların kalıcı kapılara bağlanması.

Belgenin amacı "ne yaptık" listesi değil, **hangi iddianın hangi ölçüme
dayandığını** bırakmaktır. Ölçülmemiş hiçbir madde burada "kapandı" diye
yazılmaz.

---

## 1. Dış eleştirinin madde madde karşılığı

| # | İddia | Sonuç | Dayanak |
|---|---|---|---|
| 1 | Tabloların zaman ucu tutmuyor | **Doğru** | hava 08-09, saatlik 08-10, nem 08-12, hava kalitesi 08-12, konvektif 08-13 |
| 2 | Yangın ilçeye bağlı değil, her koşuda hesaplanıyor | **Kısmen yanlış** | Yarıçaplar tanımlı; ön-üretim maliyeti **1,17 sn** (232.608 satır) — kazanç yok |
| 3 | Deprem katalogu neredeyse boş | **Doğru, hatta fazlası** | M≥4'te 373 olay / 217 gün; "132 dışarıda" değil **240 dışarıda** |
| 4 | Turizm ve İZSU kapsamı dar | **Doğru ama iki farklı sınıf** | Aşağıda ayrıştırıldı |
| 5 | Rüzgâr yönü ve quantile yok | **Doğru** | Tabloda yalnızca `yon_std` ve `yon_degisim` vardı |
| — | "Bonus: CAPE/yıldırım eksik" | **Bayat** | CAPE 2026-08-20 sabahı bağlanmıştı: 185.284 satır, 96/96 |

### Madde 4 neden iki ayrı sınıf

Eleştiri İZSU (30/96) ile turizmi (83/96) aynı kefeye koyuyor. Ölçüm ikisinin
**taban tabana zıt** olduğunu gösterdi:

```
izsu           2020: %31,2   2021: %31,2   ...   2026: %31,2
turizm_aylik   2020: % 0,0   2021: % 0,0   ...   2024-2026: %100,0
```

İZSU'nun boşluğu **zamanla sabittir**: 30 ilçenin bir özniteliğidir, 66'sının
yoktur ve bu eğitimde de testte de aynıdır. Model bunu "bu ilçe sınıfı" diye
okur — güvenlidir.

Turizmin boşluğu **zamanla değişir**: eğitimin ilk dört yılı tamamen boş, test
bloğu tamamen dolu. Bu, deponun her yerde kaçındığı desendir. Sebebi de
ölçüldü: `turizm_aylik` ailesi il aylık serisini **2023-2025'lik ilçe
tablosuyla** çarpıyor ve onun dar kapsamına hapsoluyor.

---

## 2. Kapatılanlar

### 2.1 Rüzgâr yönü ve gün içi dağılım

Kanıt: 2024 Enerji Datathonu birincisinin (Pikachow) en yüksek önemli tek
değişkeni `wind_dir_10m q01` idi — yani yönün kendisinin günlük dağılımı.
Bizim tabloda yönden yalnızca *türetilmiş* iki ölçü vardı, yönün kendisi yoktu.

Eklenen kolonlar:

| Kolon | Ne ölçer |
|---|---|
| `yon_sin`, `yon_cos` | Baskın yönün birim vektörü — 359° ile 1° komşu olur |
| `hamle_yon_sin/cos` | **En şiddetli hamle saatindeki** yön — hasarı yapan rüzgârın açısı |
| `ruzgar_q25..q90` | Gün içi dağılım: aynı maksimumun farklı yayılmaları |
| `hamle_q90` | Hamle dağılımı |
| `basinc_std` | Gün içi basınç oynaklığı |
| `basinc_dusus_3s` | 3 saatlik en sert basınç düşüşü — klasik fırtına öncüsü |
| `yon_q01`, `yon_q99` | Kazananın ölçülmüş değişkeni (aşağıdaki uyarıyla) |

`yon_q01`/`yon_q99` **dairesel olarak tutarlı değildir**: 350° ile 10° sayısal
olarak uzak görünür, oysa aralarında 20° vardır. Yine de taşınıyorlar çünkü
doğrudan kanıt var. Yorumları "gün içinde görülen en düşük pusula değeri"
değil, "yön rejiminin kaba imzası" olmalı. Yönün fiziksel olarak doğru hâli
`yon_sin`/`yon_cos`'tur.

### 2.2 Arşiv gecikmesi — tırtıklı ucun asıl sebebi

`ARCHIVE_LAG_DAYS = 6` sabiti **ölçülmemişti**; "birkaç gün geriden gelir"
varsayımıydı. Ölçüldü (2026-08-20, iki konum):

```
bugün+0 (2026-08-20) -> HTTP 200, son dolu gün 2026-08-20, 0 boş
bugün+1 (2026-08-21) -> HTTP 400
bugün+2, +3, +7      -> HTTP 400
```

Sınır **bugündür**. Sabit 6 → 1'e indirildi (1 gün pay: koşu gece yarısına
denk gelirse ya da sunucunun günü geride kalırsa 400 yememek için).

Bu tek sabit, bütün çekicileri aynı anda altı gün geriye kırpıyordu — ve tam
da hepsini *aynı anda* kırptığı için hizalama kontrolünden görünmez geçerdi.

### 2.3 Deprem eşiği ve enerji ağırlığı

M≥4,0 → **M≥3,0**: 373 olay / 217 gün → **3.037 olay / 1.076 gün**.

Eşiği düşürmek ancak ağırlık düzeltilirse güvenlidir. Richter **logaritmiktir**:
M5 bir deprem M3'ün ~1000 katı enerji bırakır ama büyüklük değeri olarak
yalnızca 1,67 katıdır. `buyukluk` kolonunu toplayan bir yoğunluk feature'ı
otuz küçük sarsıntıyı bir büyük depremden önemli gösterirdi.

Eklenen `enerji = 10^(1,5·(M−4))` kolonu bunu düzeltir. Kanıt:

> **Toplam sismik enerjinin %97,2'si hâlâ M≥4,0 olaylarından geliyor.**

Yani eşiği düşürmek sinyali sulandırmadı; yalnızca "sismik olarak hareketli
gün" kavramını ölçülebilir kıldı. Bu oran sağlık kapısında **%90 alt sınırla**
kilitlendi.

### 2.4 İl düzeyi aylık turizm — yeni aile

`turizm_il_aylik`, ilçe tablosuna **hiç dokunmadan** panelin tamamını kapsar:
2020-2026 arası **her yıl %100, 96/96 ilçe**.

Taşınan tek ölçü **doluluk oranıdır**, ham `geceleme` değil. Gerekçe ölçüldü:
KTB'nin kapsam rejimi iki kez değişti —

```
rejim 1: 2019-2022/08    (işletme belgeli)
rejim 2: 2022/09-2025/06 (işletme + işletme_basit)
rejim 3: 2025/07-2026    (işletme_basit)
```

Rejim sınırında seviye **tanımsal** olarak zıplar. Ege 5 ilinde, rejim
değişmeyen ayları kontrol grubu alarak ölçüldü:

| Ölçü | Kırılma |
|---|---|
| ham `geceleme` | **1,31×** |
| `yil_payi` | **1,31×** (geçiş yılında yıl toplamı iki rejimi karıştırır) |
| **`doluluk`** | **0,92×** — pratikte kırılma yok |

1,31×'lik sıçrama **2025/07'de**, yani bir yarışmada test bloğunun oturduğu
yerde başlar. Model onu "turizm patladı" diye okurdu. Doluluk oran olduğu için
pay ve payda birlikte genişler, kırılma sadeleşir.

### 2.5 Yangın yarıçapı 10 km

25 ve 50 km vardı; 10 km eklendi. Üçü **ayrı mekanizma** ölçer, aynı şeyin üç
ölçeği değil: 10 km doğrudan hasar (alev/ısı hatta ulaşır), 25 km duman
kaynaklı izolatör atlaması, 50 km bölgesel yangın havası vekili.

---

## 3. Bu tur bulunan YENİ hatalar

Aşağıdakilerin hiçbiri dış eleştiride yoktu; hepsi denetim sırasında çıktı.

### 3.1 Sessiz bayatlık — üç çekicide aynı hata

```python
if ckpt.is_file() and not args.fresh:
    continue          # aralığı kapsıyor mu? SORULMUYOR
```

`fetch_hava_kalitesi`, `fetch_konvektif`, `fetch_nem_toprak` kontrol noktasını
yalnızca **var mı** diye soruyordu. Sonucu tam anlamıyla sessizdir: `--end`
ileri taşınır, betik 96 ilçenin hepsi için "kontrol noktasından" yazar, **exit
0** döner ve tablo eski tarihte kalır. Hava kalitesi tablosu tam olarak böyle
08-12'de takılı kalmıştı.

Ortak `checkpoint_covers` yazıldı, üç çağrı yeri düzeltildi ve
`test_kontrol_noktasi_kapsam_ile_atlanir` ile kilitlendi. Kapının gerçekten
kapı olduğu, hata geçici olarak geri konularak doğrulandı.

### 3.2 Ölü eşik — `ruzgar_20ms_saat` hiç tetiklenmiyordu

Eşikler genel fırtına literatüründen alınmış, Ege'de ne sıklıkta göründüğü
**hiç ölçülmemişti**. 40 ilçenin 2.326.080 saatinde ölçüldü:

```
SÜREKLİ RÜZGÂR (10 m)        HAMLE
  >= 8 m/s : %1,3052           >=15 m/s : %2,74
  >=10 m/s : %0,3290           >=20 m/s : %0,26
  >=12 m/s : %0,0727           >=25 m/s : %0,02
  >=15 m/s : %0,0056           max: 35,0 m/s
  >=20 m/s : %0,0000  <-- HİÇ
  max görülen: 18,5 m/s
```

`ruzgar_20ms_saat` 2,3 milyon saatte **bir kez bile** tetiklenmedi; her satırda
0 yazıyordu. `ruzgar_15ms_saat` de fiilen ölü (ilçe başına 6,5 yılda ~3 saat).

Sebep fiziksel: ERA5 ~25 km ızgarada uzamsal **ortalamadır**, nokta
ölçümlerdeki uçları söndürür. Aynı sönümleme hamle parametrizasyonunda yoktur.

Yeni eşikler ölçülen dağılımdan seçildi: rüzgâr 8/10/12 m/s, hamle 15/20/25 m/s.

### 3.3 Köken bilgisi feature olarak veriliyordu

`hava_tahmin`, satırın arşivden mi tahmin API'sinden mi geldiğini söyler.
Feature kümesine giriyordu. Eğitim satırlarının **tamamı** arşivdir — yani
kolon eğitimde sabittir ve model onu hiç öğrenemez; geleceğe tahmin
üretilirken tamamı 1 olur. Taşıdığı tek şey "bu satır test bloğunda"
bilgisidir. Tabloda kaldı (denetim onu kullanıyor), panelden çıkarıldı.

### 3.4 Geri çekilme merdiveni yanlış pencereye ayarlıydı

Open-Meteo üç ayrı limit penceresi işletir. Merdiven (65/130/300 sn) yalnızca
**dakikalık** limit için doğruydu; üç deneme toplam ~8 dakika eder. 21:41'de
**saatlik** limit doldu ve sunucu "bir sonraki saat" dedi — 19 dakika. Eski
merdivenle kalan 23 ilçenin her biri sekiz dakika boşuna deneyip düşecekti.

Yarışma günü bu somut bir kayıptır: "yeniden dene" demek yeterliyken ekip
çekimin başarısız olduğunu görür. `rate_limit_beklemesi` artık 429 gövdesindeki
pencereyi okuyup saat sınırına kadar bekliyor.

### 3.5 Ham saatlik veri saklanmıyordu

Kontrol noktası yalnızca **günlük agregatı** tutuyordu; gerekçe "ham veri
yüzlerce MB olur" idi. Ölçüldü: float32'ye düşürülmüş dört kolon ilçe başına
~1 MB (toplam ~75 MB). Buna karşılık bedeli ağırdı — tabloya tek bir türev
kolon eklemek 96 ilçenin tamamını baştan indirmeyi gerektiriyordu.

Artık ham veri duruyor ve yeni bir kolon `--yeniden-topla` ile **ağa hiç
dokunmadan** üretiliyor. Bu turda 3.2'deki eşik kalibrasyonu tam olarak bu
sayede bedava oldu.

### 3.6 `fetch_weather.py` hiç çalışmıyordu

`ilceler_gdz_adm.parquet` için `.metadata.json` yan dosyası yoktu. Çekici
ilçe listesini `validate_published_dataframe` ile okur; yan dosya olmadan
**ilk satırda** `ValueError` ile düşer. Yani hava verisini tazelemek imkânsızdı.

**1.200'den fazla test yeşilken bunu hiçbiri görmedi** — çünkü hiçbiri
çekicinin açılış yolunu denemiyordu. Yarışma günü bunun bedeli somuttur:
veriyi tazelemek isteyen ekip sebebi belirsiz bir hatayla karşılaşır.

Yan dosya, veri **değiştirilmeden** yeniden yayımlanarak üretildi (içerik
birebir aynı: 96 satır, 96 tekil ilçe) ve
`test_referans_tablosu_yayin_dogrulamasindan_geciyor` ile kilitlendi.

### 3.7 Sağlık kapısının kendisinde açılan delik

Köprü bayrağını (`tahmin`) sağlık kapısına tanıtırken, tarih ve bayrak
kolonlarını **tek okumada** istedim. Bayrak kolonu henüz olmayan tablolarda
bu `KeyError` verdi ve `son_gercek_gozlem` `None` döndü — yani üç panel
tablosu hizalama kontrolünden **sessizce düştü**. Çıktıdaki dökümde
görünmedikleri fark edilmeseydi kapı "geçti" diyecekti.

Bayrak artık ayrı okunuyor ve yokluğu hata değil: köprü kurulmamış bir
tabloda tüm satırlar gerçektir.

> Not: Bu, kapı yazmanın kendi risk sınıfıdır. Bir kapının çıktısına
> bakarken sorulacak soru "geçti mi" değil, **"neyi denetledi"**dir.

---

## 3b. Kaynaklar arası tutarlılık — ölçülen

Saatlik ham veriden hesaplanan günlük maksimum rüzgâr, günlük API'nin
verdiğiyle karşılaştırıldı (12 ilçe × 2.423 gün):

| Ölçüm | Sonuç | Anlamı |
|---|---|---|
| Korelasyon | **0,999** | İki tablonun **gün sınırı aynı** (ikisi de Europe/Istanbul) |
| Oran | **tam 3,6000** | Günlük tablo **km/sa**, saatlik türev **m/s** |

Saat dilimi kaymasının ne kadar sinsi olduğu da ölçüldü: aynı korelasyon
üç saat kaydırılmış veride **0,99124** çıkıyor. Yani veri, bozukken bile
neredeyse kusursuz görünür. Bu yüzden koruma veri düzeyinde değil **kod
düzeyinde** kuruldu: `test_zaman_serisi_istekleri_yerel_gun_siniri_kullanir`,
Open-Meteo'dan zaman serisi çeken her betiğin `timezone=Europe/Istanbul`
geçmesini zorunlu kılar.

Birim farkı bir hata değildir (her tablo kendi içinde tutarlı), ama eşik
karşılaştırırken hatırlanmalıdır. Birimin sessizce değişmesi ise ayrı bir
kapıyla korunuyor: değişse eşik kolonları ölür ve `kapsam_deseni.py` ölü
kolonu reddeder.

---

## 3c. Geleceğe köprü — en büyük açık kapatıldı

`fetch_weather_bridge.py` günlük hava tablosunu bugün+16'ya köprülüyordu ve
gerekçesi açıktı: *"test bloğu arşivin bittiği tarihi aşarsa 17 hava kolonu
yalnızca testte NaN olur."*

Ama o köprü **tek bir tabloyu** kurtarıyordu. Aynı tehlike, saatlikten
türetilen üç tablonun tamamında açıktaydı: `hava_saatlik_turev`,
`konvektif_gunluk`, `nem_toprak_gunluk`. Panelin geleceğe bakan her satırında
günlük hava ailesi dolu, diğerleri **tamamen boş** kalacaktı.

`scripts/kopru_saatlik.py` bunu kapatır. Ölçülen iki gerçek onu ucuz kılıyor:

1. Forecast API, üç tablonun ihtiyaç duyduğu **on üç değişkenin tamamını**
   aynı adla ve aynı birimle tek yanıtta veriyor (NaN ≤ %1,4). İlçe başına
   üç değil **bir** istek.
2. Forecast API **arşivden ayrı kotada** işliyor — arşiv 429 verirken
   forecast 200 döndü. Yani arşiv tıkandığında bile köprü kurulabilir.

Kota haritası da ölçüldü:

```
archive              ┐ AYNI kota
historical-forecast  ┘
forecast             — ayrı
air-quality          — ayrı
```

Günlük kolonlar arşiv çekicilerinin **kendi** `gunluge_indir` /
`aggregate_daily` fonksiyonlarıyla üretilir; kopyalanmış bir toplama mantığı
eşikler değiştiğinde sessizce ayrışırdı — bu depoda bugün tam olarak o hata
bulundu (madde 3.4).

---

## 4. Kalıcı kapılar

Bu turda bulunan hataların hiçbiri bir daha sessizce geri gelemez:

| Kapı | Neyi imkânsız kılar |
|---|---|
| `veri_sagligi.py` → panel hizalaması | Panel tabloları birbirinden 7 günden fazla ayrışamaz |
| `veri_sagligi.py` → panel tazeliği | Hepsi *birlikte* bayatlayamaz (hizalı ama ölü) |
| `veri_sagligi.py` → sismik enerji | Enerji ağırlığı bozulursa M3 eşiği sinyali sulandırır |
| **`kapsam_deseni.py`** (yeni) | Feature eğitimde dolu / testte boş olamaz; bilgi taşımayan kolon kalamaz |
| `test_kontrol_noktasi_kapsam_ile_atlanir` | Kontrol noktası varlığa göre atlanamaz |
| `test_il_aylik_turizm_ham_seviye_tasimaz` | Rejim kırılmalı ham seviye geri konamaz |
| `test_enerji_agirligi_logaritmik_olcegi_duzeltiyor` | Enerji kolonu düşürülemez |
| `test_saatlik_limit_saat_basina_kadar_bekliyor` | Merdiven yanlış pencereye geri dönemez |

`kapsam_deseni.py` özellikle önemli: `veri_sagligi.py` **kaynak** dosyalarını
denetler, bu ise **feature**'ları denetler. İkisi farklı soru sorar ve bu turda
bulunan iki hatayı (3.2 ve 3.3) yalnızca ikincisi gördü.

---

## 4b. Panel ufku — hangi güne kadar *her şey* dolu

Köprüler kurulunca yeni bir soru ortaya çıktı: panel ileriye ne kadar uzamalı?

İki ölçüm yanıtı belirledi:

**1. Panel ancak en zayıf kaynağı kadar uzayabilir.** Hava kalitesi API'si
ileriye yalnızca **7 gün** verir; hava/toprak/konvektif 16 gün. Havayı +16'ya
uzatıp hava kalitesini +7'de bırakmak, 8-16. günlerde tam olarak kaçındığımız
asimetriyi yeniden kurardı. Bu yüzden ufuk, köprülerin **tavanlarının en
küçüğü** olarak seçilir ve sınırı koyan kaynak raporlanır.

**2. Bazı kaynaklar tahmin edilemez.** Open-Meteo yarının sıcaklığını verir;
EPİAŞ yarının **tüketimini veremez**, çünkü o henüz gerçekleşmemiştir. Yani
paneli havayla ileriye taşımak, o günlerde EPİAŞ ailesini zorunlu olarak boş
bırakır. Bu bir veri kusuru değil, fiziktir.

Bu ayrım `kapsam_deseni.py`'yi yeniden şekillendirdi: kapı artık paneli
**en dar kaynağın** ulaştığı güne kadar kurar ve sınırlayanı yazar. Böylece
ölçtüğü şey "panel ne kadar uzayabilir" değil, **"her şeyin dolu olduğu en
uzak nokta nerede"** olur.

Ölçülen son durum:

```
panel ufku 2026-08-15 · sınırlayan kaynak: epias_tuketim
209 feature kolonu · 0 hata · 14 uyarı
```

---

## 4d. Tazeleme sırası — tek komut

Veri tazelemenin bir **sırası** vardır ve sırayı bozmak paneli sessizce bozar:

```
arşiv çekicileri  ->  köprüler  ->  kapılar
```

Çekiciler tabloyu kontrol noktalarından **yeniden üretir**; bu, önceki köprü
koşusunun eklediği tahmin satırlarını siler. Yani "nem tablosunu tazeleyeyim"
demek, farkında olmadan panelin ileri ucunu delmektir — ve o delik yalnızca
`veri_sagligi.py` çalıştırılırsa görünür.

**Yarışma günü tam olarak yapılacak hata budur:** veri tazelenir, kimse
köprüyü yeniden kurmaz, panelin son günlerinde bazı aileler boş kalır ve CV
bunu **görmez** (çünkü CV de aynı boş veriyle koşar).

`scripts/veri_tazele.py` sırayı garanti eder ve sonunda kapıları koşar.
İki mod var: tam tazeleme (saatler, arşiv kotası) ve `--yalniz-kopru`
(dakikalar, ayrı kotalar).

### Boru hattını çalıştırmak yeni bir hata buldu

İlk koşuda çalışıp **ikinci koşuda tamamen bozulan** bir hata vardı:
`past_days`, tablonun *max* tarihinden hesaplanıyordu. Köprü bir kez
kurulduktan sonra tablo geleceğe uzanıyor (2026-08-26) ve
`bugün − tablo_ucu` **negatif** çıkıyordu → `past_days=-3` → her ilçe için
HTTP 400.

Yarışma günü yapılacak şey tam olarak ikinci koşudur. Referans artık
**tahmin olmayan** son gün; dikiş kontrolü de arşiv satırlarıyla yapılıyor
(önceki tahmini yeni tahminle kıyaslamak, aynı modelin çıktısını kendisiyle
kıyaslamak olurdu ve farkı yapay olarak sıfırlardı).

İkisi de `test_arsiv_ucu_tahmin_satirlarini_saymiyor` ve
`test_dikis_kontrolu_tahmin_satirlarini_kendisiyle_kiyaslamaz` ile kilitlendi.

---

## 4c. Son durum — ölçülen

| Kapı | Sonuç |
|---|---|
| `veri_sagligi.py` | 17 kaynak · **0 hata** · 3 uyarı (tahmin türevli kuyruklar) |
| `kapsam_deseni.py` | 209 kolon · **0 hata** · 14 uyarı |
| `verify_sources.py` | 15 artefakt · **0 hata** · 15 uyarı (yeniden üretilebilirlik) |
| `veri_tazele.py --yalniz-kopru` | 6 adım · **hepsi geçti** (exit 0) |
| `scan_secrets.py` | **0 bulgu** |
| `pytest` | **1273 geçti** · 61 atlandı · **0 hata** |

Panel tablolarının tamamı **2026-08-26**'da bitiyor, delik yok. Köprü dikiş
farkları: basınç 0,325 hPa · CAPE 0,000 · PM10 0,000 · nem 4,743 (toleranslar
2,0 / 150 / 10 / 8).

---

## 5. Açık kalanlar

Dürüstlük gereği kapatılmayan maddeler:

- **`turizm_aylik` ve `turizm_yillik` aileleri hâlâ tehlikeli deseni taşıyor**
  (eğitim %30,7 → test %85,4). Kaldırılmadılar çünkü ilçe çözünürlüğü
  taşıyorlar ve hangisinin daha iyi olduğu ancak yarışma verisiyle yapılacak
  bir ablasyonla belirlenir. `kapsam_deseni.py` bunları **uyarı** olarak
  raporlar; susturulmadılar.
- **`konvektif` ailesi 2021-05'te başlıyor** (eğitim %79,6 → test %100).
  Kaynağın kendi sınırıdır, `kapsam_basi` alanıyla **beyan edilmiştir** —
  gizlenmemiştir.
- **İZSU 30/96'da kalıyor.** Yukarıda gösterildiği gibi bu güvenli sınıftır;
  genişletmek İZSU'nun yayımlamadığı veriyi gerektirir.
- **EPİAŞ tazelenemedi.** Ulusal tüketim/üretim 2026-08-15'te duruyor.
  Yenileme denendi, kimlik doğrulama başarısız oldu: birincil uçta HTTP 503,
  yedek uçta HTTP 401 (iki kez, aynı yanıtlar). Servis kesintisi mi kimlik
  bilgisi sorunu mu ayırt edilemedi — bu, panelin ufkunu 08-15'te sınırlayan
  tek kaynak. Mevcut veri sağlam ve kapılar yeşil; ama EPİAŞ dönerse
  `python scripts/fetch_epias_load.py --start 2020-01-01 --end <bugün-1>`
  ufku dört gün genişletir.
- **Üç tablonun son 13-17 günü tahmin türevli** (`hava_gunluk` 17,
  `nem_toprak` 14, `konvektif` 13). Panelde delik yok; değerler ERA5 yeniden
  analizinden değil tahmin modelinden geliyor. Sağlık kapısı bunu **uyarı**
  olarak sayıyor, susturmuyor. Arşivle değiştirmek için ilgili çekiciyi
  `--end` ile yeniden çalıştırıp köprüyü yenilemek gerekir; Open-Meteo saatlik
  kotası nedeniyle bu birkaç saat sürer.
- **Deprem olaylarının 240'ı beş ilin dışında.** `point_events` **mesafeyle**
  çalışır, il adıyla değil — Balıkesir'deki bir deprem Manisa ilçesinin 50 km
  yarıçapına düşebilir. Yani "il dışında" otomatik olarak "işe yaramaz"
  değildir; ama bu ölçülmedi.
