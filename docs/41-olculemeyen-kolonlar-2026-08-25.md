# Ölçülemeyen kolonlar ve soğuk tarafın tabanı — 25 Ağustos 2026

**Bu dosya [40-olcut-tamiri](40-olcut-tamiri-2026-08-24.md)'nin devamıdır.** Orası
ölçütün nasıl tamir edildiğini anlatır; burası o ölçütün **gönderilen modelin
kendisine** çevrildiğinde ne bulduğunu.

---

## 0. Bir cümlede

Gece dört eksen kapandı (sıcak kapasite, soğuk hurdle, kalibrasyon, takvim) ve
**gönderilen modelin içinde dört ölçülemeyen kolon** bulundu. Yapılandırma
değişmedi; ölçülemeyen soru LB'ye sorulmak üzere bir gönderim hakkına bağlandı.

---

## 1. Sıcak kapasite ÇÜRÜDÜ — dört trafodan geliyormuş

docs/40 §7b bunu "açık olan TEK büyük soru" olarak bırakmıştı: `iterations`
250→500 ve `depth` 6→7, kis26'da +0,0142 (3/3 tohum), yaz25/guz25'te ise
kayıp. Havuzlanmış t sıfır. Karar LB'ye bırakılmıştı.

### Önce kapsama argümanı hükmü kis26'ya verdi

`t_gy_*` doluluğu ölçüldü:

```
TEST sıcak     556.319 satır    t_gy DOLU %52,6
kis26 doğrulama 382.158 satır   t_gy DOLU %58,0   <- teste çok yakın
yaz25 doğrulama 254.296 satır   t_gy DOLU  %0,0   <- KAPSAMASIZ
guz25 doğrulama 280.327 satır   t_gy DOLU  %0,0   <- KAPSAMASIZ
```

Önem ağırlıklandırma dilinde bu **sıfır kapsama tabakasıdır**:
`w = p_test/p_dogrulama` tanımsızdır. Testin sıcak kütlesinin yarısı o iki
blokta hiç yok. Ve etki tabakaya bağımlı ölçüldü:

```
kap500d7   t_gy DOLU (n=221.700)   +0,02884   t=+4,42   3/3
kap500d7   t_gy BOŞ  (n=160.458)   -0,00755   t=-4,39   0/3
```

Yani "bloklar zıt" görüntüsü çözülmüş gibiydi: kapasite ancak gerçek sinyal
varken işe yarıyor, yoksa gürültüye uyduruyor. kis26'nın karışımı (%58/%42)
testinkine (%52,6/%47,4) neredeyse birebir eşit. Hüküm: **AL.**

### Sonra kalıcı kural 1 hükmü yıktı

Trafo bazında ayrıştırma (docs/40 §7c'de bu gece yazılan kural):

```
aday        trafo   toplam d(MSE)   EN BÜYÜK   ilk5
kap500      3.417      +8366,1        %27,6    %87,8
kap500d7    3.417      +9368,9        %32,4   %100,6
```

Kırpılmış hüküm — **en büyük K trafo atılınca**:

```
K      kalan   fark        SH        t      tohum   genele
0      3.417   +0,01422   0,00325   +4,38    3/3   -0,00762
1      3.416   +0,00994   0,00181   +5,48    3/3   -0,00533
5      3.412   +0,00016   0,00086   +0,19    1/3   -0,00009
25     3.392   -0,00556   0,00138   -4,03    0/3   +0,00298
50     3.367   -0,00888   0,00087  -10,23    0/3   +0,00476
```

**Beş trafo çıkınca kazanç sıfır; yirmi beş çıkınca hüküm anlamlı biçimde
tersine dönüyor.** En büyük dördü: 71630035 (630 kVA), 70342188 (400 kVA),
70441025 (630 kVA), 700573511 (100 kVA, medyan tüketim 2 kWh). Bayatlık
ağırlığı bu satırlara 16,2 kata kadar çıkıyor.

### Ve "mekanizma" da yanlıştı

kis26'nın **eğitim** parçasında `t_gy` doluluğu **%0,0**. Model o kolonu hiç
kullanamıyor. Öyleyse "t_gy dolu" ayrımı bir mekanizma değil, sadece *hangi
trafolar / hangi tarihler* olduğunu gösteren bir vekil.

> **HÜKÜM: sıcak kapasite REDDEDİLDİ.** Yapılandırma 250/depth6'da kalıyor.
> Bu, bir gönderim hakkını ve muhtemelen skoru kurtardı.

---

## 2. Soğuk hatanın anatomisi — kayıp nerede

kis26 soğuk, 61.918 satır, 1.223 trafo, kVA ağırlıklı (ESS %69,9):

```
kova            n      ağır.pay     MSE   MSE payı   ort tahmin   ort gerçek
y = 0        3.250       %6,04    47,340    %72,5        625,3          0,0
0<y<=10        205       %0,24     6,274     %0,4         53,5          6,4
10<y<=100    2.922       %3,58     3,155     %2,9        184,6         56,4
100<y<=1e3  22.149      %27,13     0,684     %4,7        345,6        477,3
1e3<y<=5e3  29.404      %49,10     1,078    %13,4        756,5       2114,2
5e3<y<=2e4   3.972      %13,85     1,712     %6,0       1902,9       6342,9
```

**Ağırlıklı kütlenin %6'sı olan `y = 0` satırları soğuk MSE'nin %72,5'ini
taşıyor.** Soğuk da toplam MSE'nin %63'ü → sıfır satırları **tüm yarışma
hatasının ~%46'sı**.

Yoğunlaşma aşırı: MSE'nin %50'si 1.041 satırdan (%1,68), ve **1.223 trafonun
11'inden**. Bu, soğuk tarafta her kazancın doğal olarak yoğunlaşacağı anlamına
gelir — kalıcı kural 1'in neden zorunlu olduğunun sayısal karşılığı.

---

## 3. Hurdle ÇÜRÜDÜ — sıfırlık öğrenilemiyor

Cebir tam: `log1p(0) = 0` olduğu için

```
E[log1p(y)|x] = p(x)·0 + (1-p(x))·E[log1p(y) | y>0, x]
```

Eşik yok, sert sıfırlama yok (kareli kayıpta eşik her zaman daha kötüdür).
Kuruldu: sıfır sınıflayıcısı + pozitiflerde regresör, ikisi de maskelenmiş
eğitim setinde.

```
SIFIR SINIFLAYICI   AUC 0,5728
  p in [0,00, 0,05)  n=49.185   gerçek sıfır oranı  %4,5
  p in [0,05, 0,20)  n= 8.062   gerçek sıfır oranı  %8,7
  p in [0,20, 0,50)  n= 4.058   gerçek sıfır oranı  %7,0
  p in [0,50, 0,80)  n=   523   gerçek sıfır oranı  %8,0
  p in [0,80, 1,01)  n=    90   gerçek sıfır oranı  %0,0   <- TERS
```

Üst kovada gerçek sıfır oranı **sıfır**. Trafo düzeyine çıkarınca daha da
kötü: AUC **0,5648**, ve en yüksek olasılıklı 10 trafonun **hiçbiri** ölü değil.

Sonuç: taban 1,98505 → hurdle 2,03266 (**t = −18,14**, 0/3). Kırpınca daha da
kötüleşiyor.

**Neden öğrenilemiyor:** ölü trafonun statik imzası zayıf. Ham veride
ilçe/kimlik-öneki düzeyinde desen VAR (önek 719 → %21,6 ölü, önek 707 → %0;
URLA %16,7 vs KARABURUN %0) ama üretim zaten `tanim_on2..on5` ve `ilce_key`
kolonlarını taşıyor — yani model bu sinyale sahip ve tek tek trafo düzeyinde
kestiği yer 0,57. kVA'nın ayırt ediciliği yok (AUC 0,5104).

> Bu bir **negatif ama değerli** sonuç: toplam MSE'nin ~%46'sı mevcut öznitelik
> kümesiyle **indirgenemez**. "1'in altı" hedefinin önündeki asıl duvar budur.

---

## 4. Kalibrasyon doymuş — orakül bile 0,002 veriyor

Üretim son işlemi `r' = ort(r) + 0,60·(r − ort(r))`; yani **eğimi sabitlenmiş
bir afin harita**, hiçbir veriden uydurulmamış. Doğru soru: uydurulursa ve
**görülmemiş trafolara** aktarılırsa ne olur? (Trafo bazında 5 kat çapraz
uydurma — bir trafonun satırları asla hem uydurmada hem değerlendirmede olmaz.)

```
yöntem                    RMSLE    üretime göre    genele
ÜRETİM beta=0,60        1,98505      +0,00000    +0,00000
orakül afin (hile)      1,97942      -0,00563    -0,00211   <- ÜST SINIR
afin global             1,98810      +0,00306    +0,00114   kötü
yalnız eğim             1,99195      +0,00691    +0,00258   kötü
kVA kovası afin         2,05065      +0,06560    +0,02454   çok kötü
izotonik                1,99784      +0,01280    +0,00479   kötü
```

Kendi etiketine uyduran bir orakül bile genele **0,002** veriyor; çapraz
uydurulan her varyant üretimden **kötü**. Beta taraması dibi 0,50'de ve kazanç
0,00016 (genele 0,00006).

> **HÜKÜM: kalibrasyon/büzme ekseni kapandı.** beta=0,60 kalıyor.

---

## 5. Takvim/tatil ekseni de kapandı

Kurban Bayramı 2026 (27-30 Mayıs) test penceresinin içinde ve `YALIN_CIKARILAN`
`tk_`, `tatil`, `ramazan` ailelerinin tamamını (25 kolon) atıyor. Etki ölçüldü —
trafo bazında merkezlenmiş, 15 günlük hareketli ortalamayla mevsimden arındırılmış
gün sapması:

```
Kurban 2025      +0,0013   (+0,02 std)
Ramazan Bayramı 2025  -0,0734   (-1,40 std)
1 Mayıs          +0,0650   (+1,24 std)
Yılbaşı          +0,0768   (+1,46 std)
günlük std        0,0525
hafta günü aralığı ~0,06   (Paz -0,041 … Per +0,018)
```

Günlük etkinin std'si **0,0525 log birimi**. Soğuk RMSLE 1,98'in yanında
görünmez; karesi 0,0028, soğuk MSE 3,32'nin binde biri. **Kapandı.**

---

## 6. BULGU: gönderilen modelde dört ÖLÇÜLEMEYEN kolon

Kalıcı kural 2 bu gece iki *adayı* çürüttü ama hiç *gönderilen* modele
uygulanmamıştı. 105 üretim kolonunun tamamı denetlendi.

### 6.1 Hiçbir blokta ölçülemeyenler

Bir kolon ancak bir blokta **hem eğitimde hem doğrulamada** doluysa
ölçülebilir. Dördü değil:

| kolon | üretim eğitimi | test | yaz25 e/d | guz25 e/d | kis26 e/d |
|---|---|---|---|---|---|
| `t_ay_sapma` | %5,5 | %46,8 | %7,9 / %0,0 | %9,8 / %0,0 | %0,9 / %15,8 |
| `t_gy_log_ort` | %21,2 | %52,6 | %30,5 / %0,0 | %32,7 / %0,0 | %0,0 / %58,0 |
| `t_gy_sifir_orani` | %21,2 | %52,6 | %30,5 / %0,0 | %32,7 / %0,0 | %0,0 / %58,0 |
| `t_gy_gun` | %21,2 | %52,6 | %30,5 / %0,0 | %32,7 / %0,0 | %0,0 / %58,0 |

Sebep yapısal ve kaçınılmaz: `t_gy` ("geçen yılın aynı dönemi") ancak 2026
satırlarında dolabilir, çünkü veri 2025-01-01'de başlıyor.

- yaz25/guz25 **doğrulamaları** 2025'tedir → dolamaz,
- kis26'nın **eğitimi** tamamen 2025'tir → öğrenemez.

Üretim tam ortada: **%21,2 ile öğrenip %52,6 ile tahmin ediyor** — rig'in asla
üretmediği tek bileşim.

### 6.2 Üstüne değer desteği kopuk

```
kolon               test değerlerinin destek dışı payı
ozet_pencere_gun          %100,0   (tasarım gereği: rig kimliği)
t_gy_gun                   %74,8   <- eğitim maks 90 gün, test 122
yas                        %41,8
t_gun_sayisi               %37,5
ulusal_yil_once             %6,1
```

`t_gy_gun` eğitimde en fazla 90 (2026 Ocak-Mart), testte 122'ye çıkıyor. Test
satırlarının **dörtte üçü** eğitimde hiç görülmemiş bir aralıkta.

Bu, bu gece `gp_ilce_ay`ı çürüten desenin **aynısı** — farkı aday değil,
**gönderilen modelin içinde** olması.

### 6.3 Ne yapıldı

Doğrulama bu soruyu tanımı gereği cevaplayamaz. Tek hakem LB'dir. Bu yüzden:

- `scripts/tuketim_model.py` içine `GRIDUP_CIKAR` çevre değişkeniyle açılan bir
  **ablasyon kancası** eklendi. Varsayılan KAPALI; üretim yapılandırması
  değişmedi (`105 kolon`).
- Ablasyon koşusu doğrulandı: `ABLASYON: 105 -> 101 kolon`.
- Dördü de `t_` önekli, yani soğuk maskesi (1,00) hepsini zaten NaN yapıyor →
  bu **saf sıcak-taraf sorusudur**. Soğuk satırlar `rejim_birlestir.py` ile
  v50'den alınacak, böylece soğuk taraf 30 tohumda kalır ve tohum cezası
  yalnızca sıcak payı (~%37) kadar olur.

---

## 6b. Soğuk uzmanın `yas` kayması — ölçüldü, 3 tohumda geçti, 6 tohumda ÇÖKTÜ

Soğuk uzman `maske=1,00` ile **yapay** soğutulmuş satırlarda eğitilir. Maske
geçmişi siler ama `yas`ı silmez: 400 günlük bir trafo maskelenince "geçmişi
olmayan 400 günlük trafo" olur. Gerçek soğuk trafo ise **yenidir**.

```
eğitim (maskeli) satırların  %73,0'ü  yas > 121
DOĞAL soğuk yas maksimumu:   yaz25 115 | guz25 121 | kis26 120 | TEST 121
dört kümede de yas > 121 payı TAM SIFIR
```

Üstüne `ozet_pencere_gun` bir **rig kimliği** değişkeni: kis26 eğitiminde
{90, 212}, doğrulamasında {334}, testte {455} — tamamen ayrık.

Bu, §6'daki dört kolondan **farklı olarak ÖLÇÜLEBİLİR**: doğrulamanın soğuk
satırları da doğaldır, yani rig üretimdeki kaymanın aynısını üretir.

### Üç tohum "AL" dedi

```
aday        fark       SH        t      tohum   en büyük trafo   ilk5
-yas      +0,00974  0,00402   +2,42     3/3        %67,7        %109,4
-pencere  -0,00300  0,00170   -1,77     1/3       -%448,7      -%1445,3
-ikisi    +0,01122  0,00287   +3,91     3/3        %40,0         %72,1
```

`-ikisi` kırpmaya da dayanıyordu: K=5'te t=+3,87 (3/3), K=10'da t=+2,80 (3/3).
Bu geceki çürüyen adaylardan niteliksel olarak farklı görünüyordu.

### Altı tohum ÇÖKERTTI

```
-ikisi   fark +0,00838  SH 0,00290  t +2,89  5/6
         EN BÜYÜK TRAFO %67,0   ilk5 %119,1   (3 tohumda %40,0 / %72,1 idi)

K      fark        t      tohum
0    +0,00838   +2,89     5/6
5    +0,00259   +1,14     4/6      <- 3 tohumda +3,87 idi
10   +0,00020   +0,09     4/6
25   -0,00393   -1,70     2/6
50   -0,00896   -4,38     0/6
```

Tohum sayısı ikiye katlanınca yoğunlaşma **%40 → %67**'ye çıktı ve kırpılmış
hüküm çöktü. Üç tohumluk sonuç şanslı bir çekilişti.

> **HÜKÜM: REDDET.** Ayrıca kalıcı kural 1'e bir ek: **soğuk tarafta üç tohum
> yetmez.** 1.223 trafoluk bir katta kVA ağırlıklı `d(MSE)` o kadar ağır
> kuyrukludur ki üç tohumun eşlenik SH'si yoğunlaşmayı gizleyebiliyor.
> `ozet_pencere_gun` ayrıca **işe yarıyor** (atmak zararlı, t=-1,77 ve
> kırpınca -8,36) — destek dışı olması onu değersiz yapmıyor.

---

## 6c. BULUNAN: son işlem, ZIT muamele gerektiren iki ekseni aynı katla eziyor

Üretim son işlemi tüm ofseti tek bir genel ortalamaya büzer:
`r' = ort(r) + 0,60·(r − ort(r))`. James-Stein savı yalnızca **aşırı yayılmış**
bir eksen için geçerlidir. İki eksen ayrı ayrı soruldu (kis26 soğuk, etiketli):

| eksen | model std | gerçek std | korelasyon | OLS eğimi | hüküm |
|---|---|---|---|---|---|
| **TRAFO** | 0,4027 | 1,7849 | +0,218 | **+0,795** | aşırı yayılmış → büzme DOĞRU |
| **GÜN** | 0,0470 | 0,1075 | **+0,865** | **+1,828** | az yayılmış → büzme ZARARLI |

Gün ekseninde korelasyon 0,865: model rampanın **yönünü biliyor, genliğini
bilmiyor** — ve büzme onu daha da düzleştiriyor. Trafo ekseninde ise büzme
haklı (LB üç kez doğruladı, −0,0295).

### Testte bu eksen çok daha güçlü

```
gün ekseni std (tahmin edilen ofset)   kis26 0,0470   TEST 0,2738   -> 5,8 kat
```

Kuadratik kayıpta `c` katsayısının maliyeti: kis26'da `c=0,60` yerine `c=1,00`
gün ekseni MSE'sini 0,00711 → 0,00503 düşürüyor. Varyans oranıyla ölçeklenince
testte beklenen etki **soğuk MSE'de ≈ −0,07**, yani genele kabaca −0,003…−0,007.

### `scripts/son_islem_gunsade.py`

v44'te çürüyen **beş** değişiklikten yalnızca birincisi: gün ekseni koruması.
Hücre etiketi yok, kapı yok, pencere yok, kis26'dan uydurulan parametre yok.

Seyrek gün sorunu (1 Nisan'da 1 satır, 31 Temmuz'da 1.962) uydurma bir eşikle
değil ampirik-Bayes ile çözüldü:

```
hedef_d = (n_d·ort_gun_d + M·ort_genel) / (n_d + M)
M       = σ²_gün_içi / σ²_günler_arası     ETİKETSİZ ölçülür
```

Ölçülen: σ²_arası 0,07495, σ²_içi 0,23076 → **M = 3,1**. EB ağırlığı medyan
%99,8, minimum %24,5 (n=1 olan gün üretim davranışına düşüyor).

**Doğrulama (`tuketim_v54_gunsade.csv`):**

```
satır/id sample_submission ile birebir      TAMAM
SICAK satırlar DEĞİŞMEDİ (556.319 satır)    azami fark 0,00e+00
tam sıfır kümesi                            4.517 = 4.517
gün ekseni std                              0,27377 -> 0,27355  (üretim 0,16426 ederdi)
soğuk RMS log farkı                         0,10929
```

**Bağımsız teyit** — dönüşümü kurarken kullanılmayan 2025 aynı-ay referansı:

| ay | v50 | v54 | 2025 gerçek |
|---|---|---|---|
| 04 | +0,368 | +0,291 | +0,041 |
| 05 | +0,294 | +0,163 | +0,006 |
| 06 | +0,462 | +0,443 | +0,471 |
| 07 | +0,654 | +0,762 | +0,952 |

Dört ayın üçü referansa **yaklaşıyor**, en büyük iki açık (Mayıs ve Temmuz)
belirgin kapanıyor.

> Üretim varsayılanı DEĞİŞMEDİ. Hüküm LB'ye bırakıldı.


---

## 7. Gönderim planı (25 Ağustos, kota 03:00'te açılıyor)

| hak | dosya | ne için | beklenti |
|---|---|---|---|
| 1 | `tuketim_v50_nihai30.csv` | tabanı kilitle | ~1,01710 |
| 2 | ablasyon (101 kolon, soğuk=v50) | **ölçülemeyen soruyu sor** | okuma: tohum cezası düşülerek |
| 3 | koşula bağlı — aşağı bak | | |

**Okuma kuralı.** Probe 5 tohum, v50 30 tohum. σ = 0,15671 ile beklenen tohum
cezası tam sıcak tarafta ~+0,0007. Yani:

- probe ≈ 1,0178 → dört kolon **nötr**
- probe belirgin altı → kolonlar **zarar veriyor**, kalan 6 günde 30 tohumla
  yeniden üretilir
- probe belirgin üstü → kolonlar **işe yarıyor**, konu kapanır

**Üçüncü hak adaptif:** probe zarar gösterirse `t_gy_*` ile `t_ay_sapma`yı
ayıran ikinci bir ablasyon; göstermezse tohum uzatması (v50 + 15-20 tohum,
kesin ama küçük: −0,00014).

Public LB takımın **en iyi** skorunu gösterdiği için kötü bir gönderim
sıralamayı düşürmez — yalnızca bir hak harcar. Bu, ölçülemeyen soruyu LB'ye
sormanın maliyetini bir hakla sınırlar.

---

## 8. Dürüst muhasebe

1,00'in altı MSLE cinsinden −0,0355 demek. §2 ve §3 birlikte gösteriyor ki
toplam hatanın ~%46'sı, mevcut öznitelik kümesiyle **öğrenilemeyen** soğuk
sıfırlarda. §4 kalibrasyonun, §5 takvimin, §1 kapasitenin kapandığını gösteriyor.

Bu geceye kadar ölçülen eksen sayısı yirmiyi geçti ve **hepsi** "üretim zaten
doğru" dedi. Geriye tek meşru kanal kaldı: **rig'in göremediği yerler.** §6 o
kanalın ilk somut hedefini buldu.
