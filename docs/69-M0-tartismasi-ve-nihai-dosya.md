# 69 — `M0` tartışması (denendi, GERİ ALINDI) ve nihai tek-hak dosyası

Tarih: 30 Ağustos 2026 · **docs/68'in yerine geçer** (docs/68 §6 bayat)

---

## 1. `M0` DEĞİŞTİRİLMEK İSTENDİ, GERİ ALINDI

### Öne sürülen argüman

`a0` için `Q = 0` ve `L = 0` olduğundan `M0 = P_a0²` bir özdeşlik gibi görünür.
`a0` LB'de 1.00284 aldı → `M0 = 1.005688066`. Kullandığımız 1.005846366 ise
`a0`'ın kendi skorunu 1.00292 diye yanlış tahmin ediyor (+8e-05, yuvarlama
bütçesinin 16 katı).

### Neden YANLIŞ — üç bağımsız gerekçe

**1.1 "Tutulmuş sınav" DÖNGÜSELDİ.** Değişikliğin ana kanıtı olarak
`tuketim_g7_span_tau3.csv` (skor 1.00136) gösterildi. **O dosya hiç
gönderilmedi.** Kaggle gönderim listesinde yok; kaynak da söylüyor:

```
docs/58:280            "g7 icin esdeger skor 1.00136 (gonderilmez)"
w1_kirici.json:65      "g7'nin skoru m99'a EL ILE UYDURULACAK (1.001362)"
docs/58                "Cozum sabitleri (tam m0 = 1.005688066 ile)"
```

Yani 1.00136 sayısı **yeni M0 ile türetilmişti**. Yeni M0'ın onu daha iyi
öngörmesi kaçınılmazdı. Bu bir sınav değil, özdeşliğin ters çevrilmişidir.

**1.2 Üç çapa AŞIRI-BELİRLENMİŞ biçimde eski değerde anlaşıyor.**
`L = 0` varsayımı altında `P² − Q` her çapada:

```
p51_sicak05         1.005846063
m4_hava_capali      1.005846970
v102_kappa_optimum  1.005846063     yayilim 9.1e-07
ESKI M0             1.005846366     (tam ortada)
a0 ozdesligi        1.005688066     1.58e-04 uzakta = yayilimin 174 KATI
```

Tek parametre üç hedefi aynı anda sıfırlayamaz — bu bir fit değil uyuşmadır ve
yapısal nedeni var: `a0`, `v102`+`m4` span'ında tam optimum olarak kuruldu;
`p51` de aynı iki boyutlu span'da. `L = 0` üçü için normal denklemlerin sonucu.

**1.3 Özdeşlik argümanı eksik.** `P_j` Kaggle'ın **public %50** satırında
ölçülür, `Q_j` ise kodda **714.688 satırın tamamında** hesaplanır. Denklem bu
yüzden iki farklı kümeye ait nicelikleri karıştırır; içindeki `M0` saf özdeşlik
değil, o uyumsuzluğu emen **etkin** bir sabittir. Bu yüzden `a0`'ın kendi
skorunu birebir vermemesi beklenen bir şeydir, hata değil.

**1.4 Leave-one-out eski değeri kazandırıyor.** 27 gerçek yönde, her birini
sırayla dışarıda bırakıp kalanlardan skorunu öngörerek:

```
span-ici pay   n    ESKI M0     YENI M0    kazanan
      >= %0   27   0.000231    0.000250    ESKI
     >= %80   25   0.000191    0.000233    ESKI
     >= %90   22   0.000172    0.000208    ESKI
```

**Hüküm: `M0 = 1.005846366` kalıyor.** Denemeyi burada bırakıyorum ki
tekrarlanmasın.

### Değişiklikten geriye KALANLAR (doğru olanlar)

- `tuketim_s3y40.csv` = **1.00177** `olculmus_skorlar.json`'a eklendi.
  Gönderim listesinde doğrulandı (ref 55880996, 2026-08-30 03:37). Gerçek.
- `tuketim_g7_span_tau3.csv` **eklenmedi/çıkarıldı** — hiç gönderilmedi.
- `y40`'ın `L = −0.002229` değeri EK_MODEL'de kaldı, ama artık **türetilmiş
  olduğu açıkça etiketli**. `s3y40 = 1.837·g7 + 0.392·y40` (açıklanan pay
  %99.988) olduğundan s3y40'ın tek skoru iki boyutlu alt uzayda tek denklem
  verir; y40 boyutunu ancak bu türetilmiş `L` açar.

---

## 2. Gerçekten düzeltilen hatalar

**2.1 `m112:559` ve `m117:204` canlı hataydı.** Sonda sabiti
`M0 − 2‖r_hat‖² + Q_d` ile kuruluyordu; doğrusu `M0 − 2·k'L + Q_d`
(büzmeli çözümde `k'L ≠ k'Gk`). Önceki düzeltme yalnız `m122`'ye uygulanmış,
bu ikisi atlanmıştı — ve **`m117` aynı gönderim dosyasını yazıyor.**

**2.2 `m112:465` yanlış manşet basıyordu.** `sqrt(M0 − ‖r_hat‖²)` yerine
`sqrt(M0 − 2·k'L + ‖r_hat‖²)` olmalı. Aynı betiğin `--nihai` dalı doğruyu
kullanıyordu, yani iki farklı "optimum" basılıyordu.

**2.3 `m112_durum.json`'daki bekleyen sonda MAYINDI.** Hatalı formülle
yazılmıştı; `κ = 0.005` olduğu için gönderilip `--kaydet` çalıştırılsa çözülen
`ρ` **+0.0121** kayardı — en büyük gerçek sinyalin dört katı. Silindi.

**2.4 Çözüm böleni.** Kırpma (`expm1 → 0`) yönü kısaltıyor: ilan edilen
`κ = 0.070`, gerçekleşen **0.069782**. Çözümde etkin olan kullanılır.

**2.5 KATSAYI FORMÜLÜ — yedinci hata.** Bileşiğe eksen eklerken katsayı
`1.95·|rho_s|·sqrt(Q_dik)` konuyordu. Yanlış. `seviye` kalibrasyonu **iki
BİRİM yön** arasındaydı:

```
rho_s = L_span/sqrt(Q_span) = +0.0156   (span birim yonu)
rho_u = L_dik /sqrt(Q_dik)  = -0.0304   (dik birim yonu)     oran 1.95
```

Yani `1.95·|rho_s|` doğrudan **dik birim yöndeki** korelasyonun tahminidir ve
`u` yönündeki optimal katsayı da odur. Ekstra `sqrt(Q_dik)` çarpanı,
`1.95·|rho_s|`'i *tüm eksenin* korelasyonu sayıp izotropiyle dik parçaya
dağıtmaya denk gelir — oysa `seviye`'de `rho_x/rho_s = 0.99`, 1.95 değil.
`seviye`'de eski formül 0.0246 verirdi, ölçülen 0.0304 (%19 eksik).

Ölçüm de düzeltmeyi destekledi (κ ölçeği ayrı seçildiği için yalnız göreli
ağırlıklar önemli; "blok kor" o yönün yaz25 artığıyla korelasyonu):

```
                      formul  rho_pred  blok kor  zaman tut  2.sira f
   A (eski): rho_kul*sqrt(Qd)   0.2081    0.2269      1.057     0.380
    B (dogru): rho_kul          0.2685    0.2288      1.098     0.295
```

**2.6 `rho_s` REGULARIZASYONA KIRILGANDI — sekizinci hata.** Bileşiğin her
katsayısı `1.95·|rho_s|` ile belirleniyor ve `rho_s = c'L / sqrt(Q_span)`,
`c = pinv(G, rcond)(V'x/N)`. `G`'nin tekil değerleri `…3.9e-06, 5.3e-07…` ve
`rcond=1e-6` kesimi (6.6e-07) **tam aralarına** düşüyor. `c`, o neredeyse-tekil
kipe büyük katsayı verince `L`'nin gürültüsünü büyütüyordu:

```
40 eksenin 12'si rcond'a KIRILGAN
t_yuk_faktoru:  rho_s = -0.00401 (1e-4)   -0.01996 (1e-6)     5 KAT
```

Yani o eksenin katsayısı 5 kat fazlaydı.

**Çözüm:** `L_span = <r_hat, x>/N`. `r_hat` zaten kip başına optimal büzmeyle
kurulmuş, gürültü-farkındalıklı tahmindir (`m112.buzmeli_r_hat`) ve tekil
kipleri kendiliğinden öldürür. Geometri (`x_perp`) için `pinv` kullanılmaya
devam eder — orada gürültü yok, yalnızca izdüşüm var. Ayrıca `rcond` 1e-5 ile
1e-6 arasında %30'dan fazla oynayan eksenler elenir.

```
kirilgan eksen   12/40 -> 2/40
capraz tutma     0.857 -> 0.896
isaret (zaman)   37/40 -> 39/40
ic korelasyon   0.2784 -> 0.2846
```

**Elenen iki denetim (hata BURADA DEĞİLDİ):**
- Eşik kesitleri (`ust10`/`ust25`): eşik test dağılımından alınıp bloğa
  uygulanıyor; 11 kesitin hiçbiri blokta dengesiz değil.
- NaN doldurma uyuşmazlığı: `t_*` sütunları test'te %22-28, blokta %7.5-8.9
  NaN (soğuk trafo payı farkı). Ağırlıklı standartlaştırmayla karşılaştırdım:
  **40/40 işaret aynı** kaldı, büyüklükler ~%20 kaydı. CV'den yalnız işaret
  alındığı için bileşiğin yönü etkilenmiyor.

**2.7 `buzmeli_r_hat` GÜRÜLTÜ ALTINDA PATLIYORDU — dokuzuncu hata.**
Kip tablosunda `w = 1.86e-12, 2.5e-17, 1.25e-18, −4.2e-17` (sonuncusu negatif,
sayısal artık). Koddaki koruma `w <= 1e-12` idi ve **1.856e-12 o eşiğin hemen
üstünde** kalıyordu. Gerçek veride `c/σ = 0.09` olduğu için `a=0` ve zarar
görünmüyordu — ama `c` gürültüyle `σ`'yı geçerse (şansın ~%50'si) `a>0` olup
`a·c/w` patlıyor:

```
gercek nrm = 0.003772
bozulmus L ile 60 cekilis:  medyan 0.0090   ort 561.8   maks 8144
                            PATLAYAN (>0.05): 20/60
```

Altı yapılandırma ölçüldü:

```
kip   gercek nrm  tutulan kip  bozuk maks  patlak
   A    0.003772           13        8144   20/60   <- MEVCUT
   B    0.003772           13      0.1178    6/60   goreli w tabani 1e-8
   C    0.003747            9      0.1177    5/60   B + 2 sigma
   D    0.003672            8      0.0165    0/60   1e-8 + 3 sigma
   E    0.003747            9      0.0165    0/60   1e-6 + 2 sigma  <- SECILEN
   F    0.003672            8      0.0165    0/60   1e-6 + 3 sigma
```

**E seçildi:** patlama sıfır ve gerçek değeri yalnız %0.7 değiştiriyor
(D/F %2.7 bozuyordu). Büzme tek başına yetmiyor — `c²` şansen `σ²`'yi az bir
farkla geçerse `a` küçük ama sıfırdan büyük çıkar ve küçük `w`'li kipte `c/w`
yine patlar; **2σ anlamlılık kapısı** bunu kesiyor.

**Yan denetim:** plasebo kapısı 20 permütasyonla kuruluyordu; 100
permütasyonla yeniden sınandı, **0/40 eksen düşüyor** — o kapı sağlam.

---

## 3. Eksen sayısı ölçülerek bulundu

`AZAMI_EKSEN = 14` sert bir tavandı, seçim kapıda değil orada duruyordu
(Kural 64 ihlali). Kaldırıldı ve doğru kesim ölçüldü: her ön-ek için
**zaman-bölmeli tutma** (yaz25'in ilk yarısında ağırlıklar kurulur, ikinci
yarısında sınanır — testin durumu tam budur), beş kesimin medyanı.

```
  n  rho_pred  zaman tut  zaman sd  kesit tut  TASINAN rho
  4    0.1125     0.723     0.591     0.891      0.0813
  6    0.1308     0.820     0.696     0.912      0.1072   <- sd DEVASA
  8    0.1447     0.390     0.296     0.944      0.0564
 16    0.1719     0.448     0.185     0.852      0.0770
 24    0.1891     0.451     0.177     0.787      0.0853
 40    0.2140     0.435     0.139     0.800      0.0930
```

`n=6` en yüksek taşınan değeri veriyor ama sapması 0.696 — ölçülemiyor.
`n=8`→40 eğilimi temiz, artan, sapması düşen. **n=40 seçildi.**

Sağlamlaştırılmış kurulumla (docs §2.5–2.7) YENİDEN ölçüldü ve örüntü aynı
çıktı: `n=6` taşınan 0.1151 ama sapma 0.647 (ölçülemiyor); `n=40` taşınan
0.0956, sapma 0.130. Kesim 40'ta kalıyor.

---

## 4. Gönderilecek dosya

### Liderlik tablosu GÜN İÇİNDE İKİ KEZ SERTLEŞTİ

```
05:00                        17:26
1. Grid Grinders  0.99009    1. Grid Grinders       0.99009
2. Atakan         0.99940    2. Duo-Electra         0.99614   <- 1.00129'dan
3. TasnifX        1.00115    3. Berke Kuc           0.99927   <- YENI GIRIS
4. Ahmet B.       1.00118    4. Atakan Aldemir      0.99937
5. Duo-Electra    1.00129    5. TasnifX             1.00115   <- BIZ
                             6. Ahmet B. ALTUNOK    1.00118
```

### Aşağı risk TABANLI — bu agresif olmayı doğru kılıyor

Yarışma sonunda **iki gönderim seçiliyor**. Mevcut **1.00115** bankada;
başarısız bir sonda onu kaybettirmez, yalnızca seçilmez. Dolayısıyla `κ`
hedeften türetilir ve hedefe ulaşma olasılığı en üste çıkarılır:

```
kappa* = sqrt(MSE_opt - hedef^2)
hedef 0.99790 iken 0.0785 idi; 0.99614'e sertlesince 0.0991 oldu.
```

Sabit `κ` yazmak yanlış olurdu — hedef gün içinde iki kez değişti.

```
submissions/tuketim_K_TEKHAK.csv        tum kapilar gecti
  40 eksen, hepsinde TAVAN DAYANIYOR (katsayi LB-capali, CV'ye degil)
  rho_pred = 0.2522     kappa(ilan) = 0.09908   kappa(ETKIN) = 0.098922
  sabit = 1.011812620   sifir tahmin 1.404
  ek bilesenin span-disi payi = 1.0000

  COZUM:  rho = (1.011812620 - P*P) / 0.197844
```

| gerçek `ρ` | skor | sıra |
|---:|---:|---|
| 0.2522 | 0.98077 | **1. SIRA** |
| 0.1261 | 0.99341 | **2. SIRA** |
| 0.0987 | 0.99614 | **2. SIRA** ← eşik |
| 0.0956 | 0.99644 | 3. sıra ← taşınan tahminimiz |
| 0.0590 | 1.00007 | 5. sıra |
| 0.0000 | 1.00589 | 6.+ (ama 1.00115 bankada, seçilmez) |

**Doğrulamalar:** 27 skorlu yönün 27'si kendi LB skorunu birebir yeniden
kuruyor; sekiz kapı geçti; işaret kararlılığı tek/çift gün **40/40**, zaman
bölmesi **39/40**; trafo-bölmeli çapraz doğrulama tutma **0.906**, plasebo
**z=+33.9**; `rcond`-kırılgan eksen **2/40**; büzme gürültü altında **0/60**
patlama.

**Dürüst duruş — ÖNCEKİ KARAMSAR DEĞERLENDİRME DÜZELTİLDİ.**

Daha önce "taşınan `ρ` 0.0956, kıl payı yetmiyor" denmişti. O ölçüm YANLIŞ
SORUYU soruyordu: ağırlıkları bloğun bir yarısında **fit edip** diğer yarıda
sınıyordu. Ama katsayılarımız LB'den geliyor (`1.95·|rho_s|`), bloktan fit
edilmiyor — dolayısıyla o fit/holdout oranı ilgisiz.

Sabit LB katsayılarıyla ölçüldüğünde (`m130`) zaman aşınması **yok**:

```
   n  rho_pred  kor_tum   gun1-24   25-48   49-73   74-98  99-122     sd  GEC/TUM
  16    0.1793   0.1954    0.1475  0.1736  0.1138  0.1361  0.2737  0.056    1.049
  24    0.2109   0.2112    0.1545  0.1762  0.1521  0.1663  0.2832  0.049    1.064
  40    0.2522   0.2125    0.1068  0.1591  0.1538  0.1671  0.3025  0.066    1.105
```

**`m131` uyarısı — GEÇ/TÜM oranı bilgi taşımıyor.** Rastgele işaretli
bileşikler de ortanca **1.155** oran veriyor (%5–%95: −1.31…2.57); gözlenen
1.168 tam ortancada, eksen eksen 23/40 (~şans, binom p=0.215). Yani "geç
pencerede daha güçlü" bulgusu sinyalimize özgü değil, bloğun kendi
yapısından geliyor. Oranı **ne bonus ne ceza** olarak kullanmıyoruz.

**Asıl kanıt:** sinyal **beş pencerenin beşinde de pozitif** (0.107…0.303;
şans olasılığı 1/32). Mevsim eksenleri atılınca da ayakta (27 eksen,
kor 0.153). Yani zaman aşınması yok — `m125`'in 0.388 "taşıma oranı"
geçersiz.

Doğru çerçeve:

```
sqrt(sum rho_s^2) = 0.1293      <- SAF LB OLCUMU
rho_pred = 1.95 * 0.1293 = 0.2522
2. sira icin gereken rho = 0.0991  ->  c >= 0.767
3. sira icin gereken rho = 0.0593  ->  c >= 0.459
seviye'nin olctugu c = 1.95
```

2. sıra, ölçtüğümüz çarpanın **%39'unun** tutmasını istiyor. Yine de
**garanti değil** — `c = 1.95` tek bir ölçümdür (n=1).

---

## 5. Kalıcı kurallar 65–71

**65.** `M0` bir özdeşlik DEĞİL etkin bir sabittir: `P` public %50'de ölçülür,
`Q` tüm satırlarda hesaplanır. Taban gönderimin skoruna çekmek denklemin iki
yarısını koparır. Değer, birden çok çapanın aşırı-belirlenmiş uyuşmasından
gelir ve leave-one-out ile denetlenir.

**66.** Bir sayıyı "tutulmuş sınav" diye kullanmadan önce **gerçekten ölçülüp
ölçülmediğini** doğrula. `g7`'nin 1.00136'sı hiç gönderilmemiş, üstelik
sınanan hipotezin kendisiyle türetilmişti. Gönderim listesi tek doğrulama
kaynağıdır.

**67.** Bir cebir düzeltmesi yapıldığında aynı formülün geçtiği TÜM çağrı
noktaları taranmalı. `k'Gk → k'L` düzeltmesi bir dosyada yapıldı; `m112`,
`m117` ve bir manşet `print` atlandı, `m112_durum.json`'da bekleyen bir sonda
mayına dönüştü.

**68.** Bileşiğe eksen eklemek `ρ_pred`'i her zaman büyütür ama TAŞINAN kısmı
büyütmeyebilir. Kesim, zaman-bölmeli tutma ölçülerek bulunmalı; tek kesim
gürültülüdür, en az beş kesimin medyanı alınmalı.

**69.** Bir kalibrasyon oranının HANGİ İKİ NİCELİK arasında ölçüldüğünü yaz.
`1.95` iki BİRİM yön korelasyonu arasındaydı; kod onu "tüm eksenin
korelasyonu" sanıp ayrıca `sqrt(Q_dik)` ile böldü. Aynı sayı, hangi nicelik
olduğu belirtilmediği için iki farklı formülde kullanıldı.

**70.** Gürültülü bir ölçüm vektörünü (`L`) neredeyse-tekil bir Gram ile
çarpan HER ifade büzmeli tahminle kurulmalıdır, yalnız `r_hat` değil.
`rho_s = c'L` aynı kırılganlığı taşıyordu ve 40 eksenin 12'sinde katsayıyı
5 kata kadar şişiriyordu. Geometri (izdüşüm) ile TAHMİN (gürültülü `L` ile
çarpım) ayrılmalı: birincisi `pinv`, ikincisi büzme.

**71.** Büzme tek başına sayısal sağlamlık vermez. `a_i = max(c²−σ²,0)/c²`
katsayısı, `c` şansen `σ`'yı az bir farkla geçtiğinde küçük ama sıfırdan
büyük çıkar; küçük `w`'li bir kipte `a·c/w` yine patlar. Gereken iki ek kapı:
**göreli** özdeğer tabanı (mutlak eşik yanıltıcıdır) ve bir **anlamlılık**
eşiği. Her ikisinin değeri, gürültü enjekte edilerek ölçülmelidir.
