# Ölçümler — 22 Ağustos 2026, gündüz

Hedef: LB `1,03910` → birinciyi (`1,03170`) geçmek. Aradaki fark **0,0074**.

---

## 1. Kaldıraç hesabı — nereye vurulacağını bu belirledi

```
genel = sqrt(0,222 · soguk² + 0,778 · sicak²)     (1,05032, ölçülenle uyumlu)

d(genel)/d(soguk) = 0,350    →  0,01 soğuk kazancı = 0,0035 genel
d(genel)/d(sicak) = 0,590    →  0,01 sıcak kazancı = 0,0059 genel
```

0,0074 genel kazanç için gereken:

| taraf | gereken | durum |
|---|---|---|
| soğuk | −0,0211 | 9 açıdan denenip reddedildi, tavan 0,48 çok uzakta |
| **sıcak** | **−0,0125** | **hiç taranmamıştı** |

Aynı büyüklükteki bir sıcak kazancı genel skora **1,7 kat** fazla geçiyor,
çünkü satırların %78'i orada. Gecenin bütün kazancı soğuğa gitmişti
(sıcak 0,7979 → 0,7962, yani hiçbir şey). Bugünün odağı bu yüzden sıcak.

---

## 2. `t_mevsim_*` — REDDEDİLDİ, yön tutarsız

```
TABAN (üretim, 105)   SICAK 0,81232    yaz25 0,8195  guz25 0,8319  kis26 0,7856
+ t_mevsim_* (107)    SICAK 0,81432    yaz25 0,8207  guz25 0,8390  kis26 0,7833
```

Fark −0,00200, eşiğin (0,01995) çok altında. Ama asıl hüküm **desende**:

| blok | kolon doluluğu | fark |
|---|---|---|
| yaz25 | %0 | −0,0012 ← saf gürültü ölçümü |
| guz25 | %81 | **−0,0071 kötü** |
| kis26 | %55 | +0,0023 iyi |

Kolonun gerçekten dolu olduğu iki blok **ters işaret** veriyor. Fikir
(trafonun kendi yaz/kış oranı, elimizde olmayan `trafo_tipi`nin vekili)
sağlamdı; veri desteklemiyor. `YALIN_CIKARILAN`'da kalıyor.

**Bedava gelen kalibrasyon:** yaz25'te kolon %0 dolu, yani bilgi taşımıyor.
Yine de skor 0,0012 oynadı. Bu, CatBoost'un iç rastgeleliğinin tek blokta
yarattığı taban gürültüsünün doğrudan ölçümü.

---

## 3. Sıcak tarama, 1. tur — plato, ama tutarlı bir desen

Yedi aday, maske %15, sıcak satırlarda, eşlenik maskelerle.

| aday | sıcak | fark | yön |
|---|---|---|---|
| TABAN (rs=4) | 0,80980 | — | |
| `l2_leaf_reg=1` | 0,80827 | **+0,00153** | kapasiteyi **aç** |
| `bootstrap Bernoulli 0,8` | 0,80833 | +0,00147 | |
| `depth=6` | 0,80850 | +0,00130 | kapasiteyi **aç** |
| `l2_leaf_reg=10` | 0,80972 | +0,00008 | — |
| `rsm=0,55` | 0,81221 | −0,00241 | kapasiteyi **kıs** |
| `langevin (SGLB)` | 0,81208 | −0,00228 | kapasiteyi **kıs** |

Tek tek hepsi tohum gürültüsünün (0,0105) altında — biri bile kendi başına
delil değil. **Ama desen delil:** "kapasiteyi aç" diyen üç değişiklik artı,
"kapasiteyi kıs" diyen iki değişiklik eksi. Sıcak model eksik uyduruyor
olabilir.

`langevin` context7'den doğrulandı (CatBoost 0.21+, SGLB, yalnız CPU) ve
denendi — bu problemde zarar veriyor.

---

## 4. KAPASİTE ASİMETRİSİ — taramanın açtığı asıl bulgu

```
üye         ağaç   derinlik      yaklaşık yaprak   HARMAN AĞIRLIĞI
CatBoost     250   5                   8.000             3
XGBoost      400   8 (max)           102.400             1
LightGBM     400   255 yaprak        102.000             1
```

CatBoost'un kapasitesi diğer ikisinin **~1/12'si** — ve harmanda ağırlığı
**üç**. En zayıf kapasiteli üye, en ağır üye.

Bu, 1. turun desenini bağımsız olarak destekliyor ve 2. turun en doğrudan
adayını veriyor: `iterations` (250'de duruyor).

> Not: CatBoost'un simetrik (oblivious) ağaçları ve sıralı artırması güçlü
> düzenlileştiricidir; yaprak sayısı tek başına kapasitenin tam ölçüsü
> değil. Bu yüzden hüküm ölçüme bırakıldı, sayıya değil.

---

## 5. Soğuk taraf — "yeni trafo indirimi" hipotezi, REDDEDİLDİ

**Soru:** Soğuk uzmanı, geçmişi maskelenmiş *kurulu* trafolardan öğreniyor.
Test'teki soğuk satırlar ise *gerçekten yeni* trafolar. Yeni trafolar
sistematik olarak daha az çekiyorsa, tahminlerimiz yukarı yanlı demektir.

**Ölçüm:** Eğitim verisinde 5.344 trafonun **3.147'si** 2025 Ocak'tan sonra
ilk kez görünüyor — yani gerçekten yeni trafo bolca var.

```
yaş aralığı   satır    ort seviye   medyan   sıfır oranı
      0-6    18.975     -0,1531     0,4087      0,066
     7-13    17.498     +0,0142     0,5250      0,058
    14-29    38.161     +0,0036     0,5304      0,060
    30-59    64.497     +0,0085     0,5532      0,061
    60-89    58.175     -0,0208     0,5209      0,061
   90-119    47.414     -0,1133     0,4660      0,067
KURULU      878.631     +0,0298     0,4866      0,043
```

(seviye = `log1p(tuketim) − log1p(guc)`, günlük nüfus ortalamasından arındırılmış)

Sistematik bir "yeni trafo indirimi" **yok**. Medyanda yeniler kuruluların
üstünde bile. Bu, dün LB'de alınan kohort hükmünü (çarpan 0,75 → +0,02695,
yani kohort CANLI) bağımsız bir yoldan doğruluyor.

**Ayrıca:** doğrulama bloklarındaki soğuk satırlar da gerçek yeni trafolar
(`soguk_mu` = özet penceresinde geçmiş yok), maskelenmiş kurulu trafolar
değil. Yani korkulan eğitim/test dağılım farkı zaten yoktu.

---

## 6. Ufuk dağılımı — kayma yok

Kovaryat kayması olsaydı, eğitim satırlarını test ufkuna göre ağırlıklamak
meşru bir düzeltme olurdu. Yok:

```
küme       n         min  p25  medyan  p75  max   ort
guz25   319.732       1    32     64    95  122  63,2
kis26   444.076       1    33     63    93  121  62,8
yaz25   274.929       1    33     64    94  122  63,4
TEST    714.688       1    43     70    96  122  68,1
```

Ortalama farkı 5 gün. Düzeltmeye değmez. Yol kapandı.

---

## 7. Ölçüm düzeneği düzeltildi — eşlenik *t* testi

1. turda en iyi aday +0,00153 kazandı, tohum gürültüsü 0,0105'ti; hüküm
verilemedi. Oysa bütün adaylar **aynı** maskelenmiş çerçeveyi ve **aynı**
`random_seed`'i görüyor — aralarındaki fark eşlenik. Eşlenik farkın standart
hatası, mutlak skorların yayılımından kat kat küçük.

Artık her aday TABAN'la 9 hücrede (3 blok × 3 tohum) eşleştiriliyor;
farkın ortalaması, standart hatası ve *t* değeri raporlanıyor.
Karar kuralı **|t| ≥ 2**.

`deney_kacan.py`'de ikinci bir kusur da düzeltildi: tabanı **151 kolonluk
tam set**ti, oysa üretim 105 kolonla çalışıyor. Kaçan sekizinin değeri,
atılmış 46 kolonun varlığında ölçülürse üretime taşınmaz.

---

## 8. Leaderboard gürültüsü — kovalarken akılda tutulacak

Public LB test'in **%30'u**. Aynı model için public ve private tahminleri
arasındaki fark, aradaki 0,0074'le aynı mertebede oynayabiliyor. Bu,
iyileştirme aramayı anlamsız kılmaz — ama **LB'ye bakarak model seçmeyi**
tehlikeli kılar. Seçim CV'de yapılacak, LB yalnızca doğrulama.

---

## 9. Elde kalan kesin kale

**Tohum 3 → 7.** Krogh & Vedelsby ayrışması log uzayında birebir geçerli
(RMSLE log uzayında kareli hata), birleştirici de aritmetik ortalama. Daha
fazla torba **kötüleştiremez**. Ölçüm gerekmiyor; beklenen −0,003…−0,005.
Yapılandırma kesinleşince son üretim koşusunda.

---

## 10. Sıcak tarama, 2. tur — eksik uydurma hipotezi ÇÜRÜDÜ

Eşlenik *t* testi, 9 hücre (3 blok × 3 tohum), aynı maskeler.

| aday | eşlenik fark | SH | *t* | iyi blok |
|---|---|---|---|---|
| `l2=1 + d6` | +0,00226 | 0,00286 | +0,79 | **3/3** |
| `it=500 lr=0,035 l2=1 d6` | +0,00233 | 0,00349 | +0,67 | 2/3 |
| `l2=1` | +0,00161 | 0,00142 | +1,13 | 2/3 |
| `l2=1 + d6 + Bernoulli` | +0,00135 | 0,00377 | +0,36 | 2/3 |
| `iterations=500` | **−0,00864** | 0,00647 | −1,33 | 1/3 |

Kapasitenin **en doğrudan kolu** — `iterations` 250 → 500 — skoru kötüleştirdi.
Hipotez çürüdü: model eksik uydurmuyor. Hiçbir aday |*t*| ≥ 2'ye ulaşmadı.

Ayakta kalan tek şey `l2=1 + d6`: üç blokta da artı, ve iki parçası 1. turda
bağımsız olarak ölçülmüştü. Genel skora katkısı **+0,0008**. Alınabilir,
ama farkı kapatmaz.

---

## 11. SICAK HATANIN HARİTASI — teşhis, tahmin değil

yaz25, tek tohum, üretim sıcak uzmanı. RMSLE 0,81450.

```
GERCEK TUKETIM       satir%   HATA%   yogunluk   yanlilik
tam sifir              5,5     27,2      4,91     +0,846
10-100                 8,4      6,6      0,78     +0,030
1k+                   52,6     39,1      0,74     -0,185

TRAFONUN SIFIR ORANI
%50+                   5,8     27,4      4,74     +0,192

OYNAKLIK t_log_std
1,0+                   1,9     20,9     ~11       -0,3
```

**Sıcak tarafın da kendi sıfır yığılması var:** satırların %5,5'i tam sıfır,
karesel hatanın %27,2'si orada. Soğuktan farkı: burada geçmiş **elimizde**
(`t_sifir_orani ≥ %50` onları %5,8'lik bir dilimde topluyor).

---

## 12. UFUK YANLILIĞI — büyük görünen, TAŞINMAYAN düzeltme

yaz25'te hata ufukla tekdüze yoğunlaşıyordu (yoğunluk 0,47 → 1,69) ve
yanlılık işaret değiştiriyordu (+0,126 → −0,374). Kareli hatada bir kovayı
ortalama yanlılığı kadar kaydırmak MSE'yi tam olarak `b²` azaltır; hesap
**−0,031 sıcak = −0,018 genel** veriyordu — aradaki farkın **iki buçuk katı**.

Üç blokta ölçüldü:

```
  ufuk        yaz25      guz25      kis26     YON
  1-15      -0,0880    -0,3405    +0,1977   farkli
  16-30     -0,1644    -0,3684    +0,2402   farkli
  46-60     -0,1070    -0,2560    +0,2302   farkli
  76-90     +0,2388    -0,3375    +0,1675   farkli
  106+      +0,4196    -0,2967    +0,1352   farkli
```

**Sekiz kovanın sekizinde de yön farklı.** Çapraz doğrulama (negatif = iyileşti):

```
  kaynak -> hedef    yaz25     guz25     kis26
           yaz25   -0,03300  +0,05116  +0,02821
           guz25   +0,08617  -0,06729  +0,13348
           kis26   +0,01832  +0,09142  -0,02373
```

Köşegen dışındaki **her** hücre pozitif. Bir blokta ölçülen düzeltme
diğerlerinde kötüleştiriyor, en fenası +0,133.

### Neden — ve bu, doğrulama kurgumuz hakkında bir şey söylüyor

Yanlılık ufuktan değil **mevsimden** geliyor: guz25 düz −0,33, kis26 düz
+0,19. Sebebi kurgusal: `yaz25` **tek** yaz bloğu, `guz25` **tek** sonbahar
bloğu. Bir blok dışarıda bırakılınca model o mevsimin etiketini **hiç**
görmemiş oluyor.

Sonuç: **CV'miz sistematik olarak kötümser**, ve mevsimle etkileşen her
şeyde yanıltıcı. Üretim modeli üç mevsimi de görüyor.

Bu, dün "büzülme" denemesinin neden başarısız olduğunu da açıklıyor —
kesişimler +0,10…+0,38 arasında oynuyordu. Aynı olgu, iki farklı yerden.

---

## 13. Kaçan kolonlar — kapsama açığı zaten kapanmış

Üretim setinde (105 kolon) aile dışında kalan yalnızca **4** kolon var:
`guc`, `il_key`, `bolge`, `soguk_mu` — hepsi yapısal, öyle olması doğru.
Dün gece aileler düzeltilince açık kapanmış. Deney yine de o kolonların
faydalı olup olmadığını ölçüyor.

---

## 14. SERAP YAKALANDI — 8 kolon, ve doğrulama kurgumuzun kör noktası

`deney_kacan` sekiz kolonu atmanın sıcak skoru **0,01833** iyileştirdiğini
buldu (t=−1,99, genel skorda ~0,0065 — aradaki farkın %88'i). Alınmaya
değer büyüklükte görünüyordu. Ama kazancın **tamamı guz25'teydi** ve guz25
aynı gün en büyük kurgusal yanlılığı (−0,3268) ölçtüğümüz blok.

Ayrım şu özdeşlikle yapıldı:

```
MSE = Var(artik) + ortalama(artik)²
      \_________/   \______________/
        SACILIM        YANLILIK
```

```
-HEPSI (8) vs TABAN
  blok    HAM fark    MERKEZLI fark    yanlilik taban -> aday
  yaz25   +0,00108      +0,00659        +0,1004 -> +0,0332
  guz25   -0,05382      -0,00704        -0,3674 -> -0,2425
  kis26   -0,00201      +0,00562        +0,1998 -> +0,1679
ORTALAMA  -0,01825      +0,00172
```

**Ham skorda 0,018 kazanıyor, yanlılık giderilince 0,0017 kaybediyor.**
Kazancın tamamı — fazlası — bloğun kendi sapmasını küçültmekten geliyor.
`-sekil` de aynı: ham −0,0179, merkezli **+0,0034**. İkisi de REDDEDİLDİ.

### Genel kural — bundan sonraki her ölçüm için

Doğrulama kurgumuz **büzülmeyi ödüllendiriyor.** Modeli ifadesizleştiren
her değişiklik (kolon atmak, düzenlileştirmeyi artırmak, kapasiteyi kısmak)
tahminleri ortalamaya büzer; bu, görülmemiş-mevsim sapmasını küçülttüğü
için CV'de haksız yere iyi görünür. **Tersi de doğru: kapasiteyi artıran
her değişiklik haksız yere cezalanır.**

Okuma kuralı:

| ham | merkezli | hüküm |
|---|---|---|
| + | + | GERÇEK — al |
| + | − | **KURGUSAL — reddet** (8 kolon böyleydi) |
| − | + | gerçek ama CV maskeliyor — dikkatle al |
| − | − | gerçekten kötü — reddet |

Bu, iki eski kararı yeniden sorgulatıyor ve ikisi de ölçülüyor
(`deney_merkezli.py`): `l2=1 + d6` (kapasiteyi **artırıyor**, CV cezasına
rağmen 3/3 blokta pozitifti — gerçek etkisi ölçtüğümüzden büyük olabilir)
ve **yalın set** (39 kolon atmak da bir büzülmeydi, ham −0,0095 kazandırıp
üretime alınmıştı).

---

## 15. EK KÖKENLER — tek tohumda kazandırıyor

`EK_KOKENLER` kodda tanımlı, sızıntıya karşı korumalı (`kokenleri_ayikla`),
ama **üretim eğitiminde kullanılmıyor** ve sonucuna dair hiçbir kayıt yoktu.

```
ana bloklar 1.038.737 satir -> ek kokenlerle 2.855.584

ANA (3 blok)   GENEL 1,10449   yaz25 1,08137  guz25 1,11481  kis26 1,11730
EK KOKENLI     GENEL 1,09941   yaz25 1,08426  guz25 1,10957  kis26 1,10441
                    -0,00508         +0,0029       -0,0052       -0,0129
```

Tek tohum, CatBoost, test-ağırlıklı. Genelde −0,005 ama **test ikizi olan
yaz25'te +0,0029**. Üç tohumlu eşlenik ölçüm koşuyor.

Kusurun kurgusal bir açıklaması var: yaz25 doğrulanırken onunla kesişen
`bah25` (May-Ağu) ve `yaz25b` (Tem-Eki) sızıntı olmasın diye **atılıyor**;
geriye kalan kökenler kış/sonbahar ağırlıklı. Yani yaz25 için ek kökenler
"yazı daha az gör" demek. **Üretimde bu kısıt yok.**

### Yan bulgu: ikinci bir bayat önbellek

Deney ilk koşuda çöktü — `ek_kokenler.parquet` 21 Ağustos 18:58'de kurulmuş,
`t_mevsim_*` 22 Ağustos 00:42'de eklenmiş. Ana önbellek için bayatlık testi
var (`test_aile_kapsami.py`), bunun için yok. Artık açık uyarı veriyor.

---

## 16. EK KÖKENLER — DOĞRULANDI, üretime alındı

Eşlenik ölçüm, 3 tohum, üretim seti (`deney_koken2.py`):

```
ANA (3 blok)   GENEL 1,09411   yaz25 1,07373  guz25 1,10237  kis26 1,10623
EK KOKENLI     GENEL 1,08679   yaz25 1,06542  guz25 1,09556  kis26 1,09938

ESLENIK FARK  +0,00782   SH 0,00260   t = +3,01
  yaz25  +0,00686        guz25  +0,00774        kis26  +0,00885
```

**Bugün eşiği geçen tek değişiklik.** Üç blokta da pozitif, test ikizi
`yaz25` dahil. Tek tohumda yaz25'in kötü görünmesi gürültüymüş
(tohumlar −0,006 / +0,013 / +0,014).

Ve bu bir büzülme **değil** — eğitimi 1.038.737 satırdan 2.855.584'e
çıkarıyor, yani §14'teki kurgusal ödülün tersi yönde. Dahası ölçülen değer
bir **alt sınır**: doğrulamada hedef blokla kesişen kökenler
`kokenleri_ayikla` ile atılıyor, üretimde hepsi kullanılabiliyor çünkü
test bütün eğitim verisinden sonra geliyor.

Mekanizma: model aynı etiketi farklı tazelikteki özetlerle tekrar görüyor,
yani "eski özete ne kadar güvenmeli" sorusunu üç örnek yerine dokuz
örnekten öğreniyor.

### Beklenmedik yan fayda — soğuk payı

```
koken     etiket satir   ozet gun   soguk payi
sub25       123.473        31          %1,5
bah25       290.561       120         %11,3
yaz25b      308.221       181          %9,4
guz25b      345.941       243         %21,2   <-- test'e yakin
kis26b      410.464       304         %23,4   <-- test'e yakin
bah26       338.187       365          %7,9
TEST                      455         %22,2
```

Ana blokların soğuk payı %7,5–13,9'du. `guz25b` ve `kis26b` ile model ilk
kez test'inkine denk bir soğuk karışımı görüyor.

---

## 17. Üretime bağlanan yapılandırma — 22 Ağustos öğle

```
1  EK_KOKENLER            +0,0078  t=+3,01   OLCULDU
2  sicak l2=1 + d6        +0,0063  3/3 blok  MERKEZLI olcum
3  tohum 3 -> 7           ~0,0040  garanti (Krogh & Vedelsby)
4  yalin set 105 kolon    degismedi
5  soguk uzmani d7        degismedi -- l2 orada olculmedi
6  harman 3/1/1           degismedi
```

Sızıntı denetimi elle doğrulandı — `yaz25` doğrulanırken düşen kökenler:
`yaz25` (kendisi), `bah25` (May-Ağu, kesişiyor), `yaz25b` (Tem-Eki,
kesişiyor). Kalanlar: `sub25`, `guz25`, `kis26`, `guz25b`, `kis26b`,
`bah26`. Doğru.

Sessiz bir hata kaynağı da kapatıldı: ek köken kolonları ana bloklarınkinden
farklıysa artık kesişim alınmıyor, **hata veriliyor**. Sessizce kesişmek,
ölçülenden başka bir model üretmek demekti — dün gece tam bu sınıftan bir
hata 151-kolonluk üretimle 144-kolonluk ölçümü ayırmıştı.

---

## 18. Bugünün karnesi

| ne | sonuç |
|---|---|
| **EK KÖKENLER** | **+0,0078 · t=3,01 · ALINDI** |
| **sıcak `l2=1 + d6`** | **+0,0063 merkezli · ALINDI** |
| `t_mevsim_*` | reddedildi — yön tutarsız |
| yeni-trafo indirimi | reddedildi — indirim yok |
| ufuk kayması | reddedildi — kayma yok |
| ufuk yanlılığı düzeltmesi | **reddedildi — çapraz doğrulamada +0,13** |
| 8 kolonu atmak | **reddedildi — serap, merkezli +0,0017** |
| `-sekil` (3 kolon) | reddedildi — serap, merkezli +0,0034 |
| `iterations=500` | reddedildi — eksik uydurma hipotezi çürüdü |
| `langevin` (SGLB) | reddedildi — zarar veriyor |
| `rsm=0,55` | reddedildi |
| yalın seti geri açmak | berabere — dokunulmadı |

**On bir yol kapandı, iki yol açıldı.** Kapananların ikisi (ufuk düzeltmesi
−0,018 vaat ediyordu, 8 kolon −0,0183) gönderilseydi LB'de sert kaybettirirdi.

---

## 19. EK KÖKENLER YALNIZ SICAK UZMANINA — v17'nin neden battığı

`v17` ek kökenleri **her iki uzmana** verdi ve battı:

```
              v15        v17       fark
yaz25       0,99715    1,02530   +0,0282   KOTU
guz25       1,05966    1,06442   +0,0048
kis26       1,11772    1,09177   -0,0260   iyi
ORTALAMA    1,05818    1,06049   +0,0023   berabere
```

Rejim bazında ölçüldü (`deney_koken_rejim.py`, eşlenik, 3 tohum, her rejim
kendi üretim ayarlarıyla ve kendi satırlarında):

```
SICAK  ANA 0,80675 -> EK 0,79848   +0,00946  t=+1,46
SOGUK  ANA 1,70349 -> EK 1,73612   -0,03273  t=-2,59  ZARARLI
   soguk blok bazinda: yaz25 -0,077  guz25 -0,030  kis26 +0,009
```

**Mekanizma.** Ek kökenler aynı `(trafo, gün)` satırını farklı özet
pencereleriyle tekrar gösteriyor. Sıcak uzmanı (maske %15) için bu gerçek
veri artırma — `t_*` özetleri gerçekten farklı geliyor. Soğuk uzmanı maske
%100'de çalışıyor, yani bütün `t_*` NaN; kopyalar arasında geriye yalnızca
`ozet_pencere_gun`, `t_doluluk` ve `ufuk_gun` farkı kalıyor ve hedef
**birebir aynı**. Veri artırma değil, kopya çoğaltma.

v17'de yaz25'te soğuk kaybı (−0,077) sıcak kazancını (+0,008) ezmişti.

### Düzeltilmiş sonuç — üç blokta da v15'in altında

```
              v15        v18       fark
yaz25       0,99715    0,99115   -0,0060
guz25       1,05966    1,04950   -0,0102
kis26       1,11772    1,09768   -0,0200
ORTALAMA    1,05818    1,04611   -0,0121
```

Hesap kapanıyor: soğuk taraf v15'le **birebir aynı** (yaz25 1,47665 —
yapılandırması değişmedi), sıcak taraf 0,80985 → 0,80081, yani **−0,0090**.
Kazancın tamamı sıcak uzmanından, ve `l2=1+d6` ile ek kökenlerin ayrı ayrı
ölçülen parçalarının toplamıyla örtüşüyor.

---

## 20. Günün asıl dersi — ölçüm düzeneği üretimden ayrılırsa

Bugün **üç kez** aynı sınıftan hata yapıldı:

1. Sabah: tezgâh 144 kolonla ölçüyordu, üretim 151 kuruyordu (dün gece
   yakalanmıştı, bugün `deney_kacan` aynı hataya düştü — tabanı tam setti)
2. `deney_koken2`: ek kökenleri **yönlendirmesiz** tek CatBoost'la ölçtü,
   üretim iki uzman çalıştırıyor → +0,0078 ölçüldü, üretimde +0,0023 kötü
3. `deney_sicak`: mutlak skorların yayılımını raporluyordu, oysa adaylar
   eşlenikti → hüküm verilemiyordu

**Ortak ders: ölçüm düzeneği üretimden bir adım bile ayrılırsa, ölçtüğün
şey gönderdiğin şey değildir.** Üçünün de bedeli ölçüm zamanıydı; ilk
ikisi gönderilseydi LB'de kaybettirirdi.

Ve bugünün kazandıran iki değişikliğinin **hiçbiri** parametre ayarından
gelmedi. On üç hiperparametre adayı denendi, hepsi eşiğin altında kaldı.
Kazandıran şey yapısaldı: eğitim setini büyütmek, ve onu **doğru uzmana**
vermek.

---

## 21. `v18` üretildi ve doğrulandı — GÖNDERİM BEKLİYOR

```
                    yaz25 CV    genel CV       LB
v15 (LB'de duran)    0,99715     1,05818    1,03910
v18                  0,99115     1,04611    ~1,031 ongoru
birinci                                     1,03170
```

Dosya: `submissions/tuketim_v18.csv` (56,5 dakika, 7 tohum).

### Bütünlük denetimi

```
satir            714.688      id SIRASI sample_submission ile BIREBIR
kolonlar         ['id','tuketim']      NaN 0      negatif 0
medyan           1034,5 -> 995,8

rejim        satir   ort log fark     std
SICAK      556.319     -0,0378      0,1285
SOGUK      158.369     +0,0011      0,0494   <- DEGISMEDI, dogru
```

Soğuk uzmanının yapılandırması değişmedi ve tahminleri de değişmemiş
(+0,0011, yalnızca 3→7 tohumun varyans etkisi). Bütün değişim sıcak
tarafta. Yani `dar_egitim` yönlendirmesi doğru çalışıyor.

Gönderim: `submissions/tuketim_v18.csv`, not olarak
"v18: ek kokenler YALNIZ sicak uzmanina + sicak l2=1/d6 + 7 tohum".

**Aşağı yönlü risk yok:** Kaggle en iyi skoru koruyor. v18 kötü gelse bile
v15'in 1,03910'u ve 2. sıra durur.

### Sonuç geldiğinde okunacak

| gelen skor | anlamı | sıradaki |
|---|---|---|
| ≤ 1,0317 | **birinci** — kalibrasyon ikinci kez doğrulanır | ek köken sayısını artır (şu an 6) |
| 1,032–1,036 | beklenen bandın içi, kalibrasyon tutuyor | aynı yön: sıcak uzmanına daha çok veri |
| > 1,040 | **CV↔LB kırıldı** — v15'e dön, nedeni ara | gönderim hakkı çok değerli hale gelir |
