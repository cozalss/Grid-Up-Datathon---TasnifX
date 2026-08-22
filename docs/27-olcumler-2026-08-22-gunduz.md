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
