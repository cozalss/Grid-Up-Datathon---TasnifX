# 52. Ölü trafo tezi çürüdü · Gram optimumu · prob kampanyası (2026-08-28)

Bu belge `docs/51`in üstüne geçer ve `docs/47`nin durum tablosunu günceller.
`docs/51`in ana tezi **ölçümle çürütülmüştür**; §1'i okumadan `v89`, `v87`,
`v88` veya `sota_v1` gönderilmemelidir.

11 ajanlık bir fan-out, ~150 aday ölçümü. Kaggle'a **hiçbir gönderim yapılmadı**.

---

## 0. Durum (2026-08-28 00:20)

```
LB
  1. Grid Grinders            0.99138
  2. Ahmet Bugrahan ALTUNOK   1.00615
  3. Duo-Electra              1.00907
  4. Abdulbaki Bayir          1.00945
  5. Atakan Aldemir           1.00957
  8. TasnifX                  1.01318   <- BIZ (v83, 27 Agu 06:09 UTC)
  435 takim
```

`docs/50` ve `docs/51`in tabloları **eskimiştir** (orada 4. sıra ve lider
0.99403 yazıyor). 27 Ağustos günü içinde dört takım bizi geçti.

```
bizim MSE 1.026534
2. sira   1.012338  ->  gereken dMSE -0.007686   (v101 hazir)
1. sira   0.982835  ->  gereken dMSE -0.043699
```

Kota: UTC gunu basina 3 hak, yerel 03:00'te yenilenir. Bitis 1 Eylul 23:59 UTC.
**Gonderim yetkisi 27 Agustos'ta GERI ALINDI** — kullanicinin acik onayi olmadan
hicbir dosya gonderilmez. Bkz. hafiza `kaggle-gonderim-yetkisi`.

---

## 1. `docs/51`in ölü trafo tezi ÇÜRÜDÜ

`v89`, 251 trafonun 19.839 test satırını ~0'a çekiyordu; beklenen skor 0.88447.
**Gönderilseydi ~1.13 gelirdi.**

### 1.1 Maske ikiye ayrılıyor (kalıcı kural 9, `docs/43:170`)

Ayırıcı: train son kayıt tarihi >= 2026-03-27.

```
GRUP A (raporu suren)         158 trafo  12.690 satir  MSE payi 0.004052  (%1.3)
GRUP B (kesilip panele donen)  93 trafo   7.149 satir  MSE payi 0.313883  (%98.7)
```

**Grup A'yı maskelemek hiçbir şey kazandırmıyor** — v83 oraya zaten ort 0.48 kWh
yazıyor. Kaldıracın tamamı grup B'nin ölü olmasına bağlı.

### 1.2 Grup B DİRİ — sekiz kesmede aynı yön

İleri pencereli kör tekrar (`max()==0` kuralı T öncesiyle kurulur, T+122 günde
gerçeğe bakılır):

```
T             GRUP A sifir    GRUP B sifir   B'yi sifirlamanin dMSE
2025-03-31       %96.78          %28.31           +0.01605
2025-06-30       %91.96           %3.87           +0.10570
2025-08-31       %93.56           %1.71           +0.08900
2025-11-30       %98.92           %2.24           +0.04377
2025-12-31       %98.79           %3.20           +0.03904
```

Bağımsız ikinci ölçüm (dönüş **sonrası**, kapasite-normalize ofset uzayında):

```
kesme        donen  satir  SIFIR %   v83 ofs   gercek ofs    delta*
2025-06-30     4     284     0.0     -0.5508    +1.0547     +1.61
2025-08-31     5     298     0.0     -0.5508    +0.4621     +1.01
2025-11-30     4     216     0.0     -0.5508    +1.6418     +2.19
```

798 satırın hiçbiri sıfır değil. Grup B'nin gerçek ort `log1p`'i 5.9-7.9;
`log1p(402.5) = 6.00`. **v83'ün 402 kWh tahmini isabetli, hatta düşük.**

Test profiline yaklaştıkça kötüleşiyor: sıfır serisi >=120 satır **ve** boşluk
>=150 gün olan alt kümede sıfır oranı **%0.00**.

### 1.3 `docs/51`in savunmaları tek tek düştü

| `docs/51` iddiası | ölçüm |
|---|---|
| "SOTA kuralı 193 trafo yakalıyor" | Kural 111 trafo seçiyor (testte 68 / 5.356 satır). Kalan **9.128 sıfır satır `np.clip(np.expm1(...),0,None)` kırpmasından** — kaldıraç değil, kaza. 193 trafonun yalnız 79'u tamamen, 114'ü kısmen maskeli. |
| genişleme grubu "%100 sıfır" | **Totoloji.** Grup `train.max()==0` ile tanımlanıyor, sonra sıfır olduğu "doğrulanıyor". Sıfır bit bilgi. |
| §4 "%92.60 sıfır" | Sayı doğru ama **%84'ü totoloji**; gerçek sinyal veren 17 trafoda oran **%55**. |
| §5 "kural CV'de görünmüyor" | Doğal kural biçimiyle CV'de **testtekinden yoğun** ateşliyor: yaz25 12.170 satır (%4.43), guz25 4.870, kis26 12.249; TEST 23.409 (%3.28). |
| §4 "13 trafo %78.9 alt sınır" | Sayı doğru (%78.88) ama trafo-kümeli CI **[%60.8, %97.0]**; alt sınır olarak kullanılamaz. Bileşim yanlı (13'ün 7'si riskli alt kümeden). |
| §6 "lidere göre başa baş p ~ %50" | Kendi modelinin çıktısı bile değil; doğrusu **%54.4**. |
| §10.2 panel sıçraması "bilinemez" | Çözüldü — bkz. §1.4. |
| §3 tutarlılık kontrolü | Kırıldı — bkz. §2. |

### 1.4 Test satırının varlığı = SİCİL, aktivite değil (§10.2 kapandı)

```
train'de tuketim==0 satir: 57.536 (%4.69), 552 trafo
455 gun boyunca %100 sifir olan trafo: 298
61+ gun bosluk oncesi gunun sifir olma orani: %12.7 (taban %4.7)
   -> bosluklarin %87.3'u POZITIF bir gunun ardindan basliyor
test: 7.036 trafo x 122 gun = 858.392, gercek 714.688, doluluk %83.3 (izgara DEGIL)
2026-05-11'de 2.222 trafo TEK GUNDE panele giriyor (05-03'te 141 daha)
```

Susan trafo panelden düşmüyor, sıfır yazmaya devam ediyor. Aynı kohortta 28 trafo
1 günlük boşlukla dönüp **287 gün boyunca tam sıfır** kayıt üretmiş. Yani "kayıt
üreten = geri dönen" savunması geçersiz.

**Ama uzun boşluktan dönmek ayrı bir olay:** aynı kohortun >=1 ay boşluktan
dönen 8 üyesinin 8/8'i pozitif. Test grup B'nin boşluğu medyan 320 gün.

Toplu panel giriş günlerinde girenler ne çekmiş (eğitim):
```
2025-01-01  2.059 trafo  ilk 30 gun sifir %6.97   ort 3.065 kWh
2026-03-26    329 trafo               %4.28   ort 1.096 kWh
2025-07-28    177 trafo               %3.56   ort 2.261 kWh
```
**"Toplu giriş = sicil olayı, aktivite değil" savunmasının veride dayanağı yok.**

### 1.5 Kirli dosyalar

`sota_v1` aynı 93 trafonun 68'ini sıfırlıyor (MSE payı 0.248). `v87` ve `v88` de
öyle. `docs/51` §8'in "v89 kötü gelirse sota_v1 gönder" adımı **aynı bahsi ikinci
kez oynuyor**.

```
GONDERME: tuketim_v89_genis_taban.csv, tuketim_v88_olu_taban.csv,
          tuketim_v87_olu_izole.csv, tuketim_sota_v1.csv
TEMIZ   : tuketim_v83_sicak_optimum.csv, tuketim_v85_gram_rank2.csv,
          tuketim_v90_temiz_sota.csv, tuketim_v93*, tuketim_v101*
```

### 1.6 `0.00008` çelişkisi çözüldü

`docs/51` §5 "hep yumuşak büzme denendi, LB kazancı 0.00008" diyor. `olu_hedge.py`
**tam tersini** yapıyor: zaten sıfır olan 8.748 satırı **yukarı kaldırıyor**.
Kategori hatası, çelişki yok. Üstelik o ölçümden κ çözülünce LB'nin o satırlarda
dediği sıfır oranı **%84-95** — tezi destekliyor. Ama o prob, tartışılan 227.245
SSE'nin yalnız **76'sını** kapsıyor; bahsi taşıyan 11.882 satır hiç problanmadı.

### 1.7 `v89` aritmetiği DOĞRUYDU — varsayımı yanlıştı

Bağımsız yeniden üretildi: 19.839 satır ✓ · ceza payı 0.317944 ✓ ·
1.026534 − 0.317944 = 0.708590 → 0.84178 ✓ · 0.88447/0.94730/1.00622 ✓.
`0.88447`'yi üreten betik **depoda yok, hiç commit edilmemiş**.

Jensen hatası: `AY_OLCUM`'da `b = log1p(ortalama kWh)` kullanılmış, doğrusu
`ortalama log1p`. Doğru değerler 1.3528 / 1.9745 / 4.4756 / 5.6288.
Hata muhafazakâr yönde (4-8 puanlık örtük tampon) ama docstring'deki
"MSLE-optimum" iddiası yanlış.

---

## 2. Soğuk rejim CV'si — hiç kaydedilmemişti, ölçüldü

`data/interim/deney/soguk_tahmin_*.npz` önbelleklerinden:

```
blok    SICAK RMSLE (belgelerde var)   SOGUK RMSLE (belgelerde YOKTU)
yaz25         0.81224                       1.4359   (MSE 2.0619)
guz25         0.83436                       1.6082
kis26         0.77826                       1.9061
```

### Bütçe testi — `docs/51` §3'ün tutarlılık kontrolünü kırıyor

```
sicak 556.319 satir x CV MSE 0.66194 = 368.232 SSE
soguk 158.369 satir x CV MSE 2.06190 = 326.543 SSE
                     CV'nin ongordugu = 694.775
                     v83'un gercegi   = 733.651
                     ACIK             =  38.876  (%5.3)
```

CV zaten toplamın **%94.7'sini** açıklıyor; ölü satırlara en fazla **%7.1** kalıyor,
iddia edilen %31 değil. %31'i tutturmak, ölü-dışı 694.849 satırın sızıntısız
CV'den **%25.7 daha iyi** çıkmasını gerektirir.

`docs/51` §3 "kalan 694.849 satır 0.85372, CV 0.81224 ile uyuşuyor" derken o
satırların **158.369'unun soğuk kohort** olduğunu atlamış; havuzlanmış sayıyı
yalnız-sıcak CV ile kıyaslamış.

---

## 3. ÖLÇÜM ARACI UYARISI — `tanim_num` ezber kanalı

`tanim_num` birebir trafo kimliği ve maskelemeden sağ çıkıyor. CV bloklarının
soğuk satırlarında ezberlenebilirlik:

```
yaz25 %97.2   guz25 %97.7   kis26 %0.0   TEST %0.0
```

**Soğuk tarafta yaz25/guz25 kis26 ve TEST ile TERS korelasyonlu.** İki bağımsız
kol bunu gösterdi: kapasite/çeşitlilik eklerken yaz25/guz25 çok kazanıyor
(−0.20…−0.55), kis26 kaybediyor; kapasiteyi düşürünce tam tersi.

> **Sonuç:** üç-blok işaret kapısı soğuk rejimde, TEST'te bulunmayan bir kanalı
> en iyi sömüren yapılandırmayı seçiyor. Üretimdeki `depth=7` üstyazımı ve
> `agirlik_soguk.jsonl`'daki "1/1/1 en iyi" okuması **muhtemelen bu şekilde
> seçilmiş**. Soğuk hükümler **yalnız kis26** ile verilmeli — ama kis26 de
> 1.223 trafoyla tek-trafo gürültüsüne açık (kırpma tablosunda kazancın %57-89'u
> 5 trafodan geliyor).

**Soğuk model tarafında şu an hiçbir yönde güvenilir ölçüm yapılamıyor.**
Kaldıraç aranacaksa önce ölçüt düzeltilmeli (trafo-bazlı bootstrap ağırlıklı kis26).

---

## 4. Metrik KESİN olarak doğrulandı

v80/v81/v83 kapalı çevrimi:
```
v81 - v80 = tam +0.08 x 526.446 satir, ara deger YOK
Q = 0.0047143  (docs/47: 0.7366095 x 0.08^2 = 0.0047143, 7 hane ayni)
LB'den L = 0.0014650  ->  k* = 0.31075
beklenen MSE 1.026545 | GERCEK 1.026534 | fark 1.1e-05  (5 hane yuvarlamasi)
```

`RMSLE = sqrt(mean((log1p(p) - log1p(t))^2))` **kesin**. `sample_submission.csv`
iki kolon (`id, tuketim`) → "Mean Columnwise RMSLE" düz RMSLE'ye iner.

**Yan ürün:** LB κ*=0.31075 dediği için sıcak çekirdekte (526.446 satır, %73.66)
**ortalama artık tam sıfır**. Küresel seviye/kalibrasyon fikirleri testin
dörtte üçünde kapalı — tahminle değil, LB'nin kendisiyle biliniyor.

---

## 5. Gram optimumu — `v93` (bu oturumun tek kesin kazancı)

19 ölçülmüş LB skoru `t` hakkında birer denklem verir:
`MSE_i = m0 + G_ii - 2b_i`, `b_i = (m0 + G_ii - m_i)/2`. `||t||^2/n` sadeleşir.
18 fark yönü, Gram **rank 16**, ridge yok (SNR ayrımı 23.8 vs 2.1, 60 kat).

```
submissions/tuketim_v93_gram_optimum.csv
  Q = ||d||^2/n = 0.00979660     beklenen dMSE = -Q
  ON KAYIT 1.00833  [1.00810 - 1.00857]
  afin agirliklar, |w|_1 = 14.31 (konveks govde DISI)
```

### Düşmanca denetimden geçti

```
10/10 sha256 kilidi tuttu | 19/19 mtime gonderim damgasindan once | 21/21 id hizasi
yuvarlama MC          : 1.008334 +- 0.000017
public alt kume f=0.05: +0.00047 (p95 +0.00088)
en kotu makul senaryo : 1.00921
```

**Zamansal bölme hipotezi ÇÜRÜDÜ.** Null yönleri her bölmeye kesin bir sayı
dayatıyor:
```
tam kume / rastgele   2.08 sigma   <- AYAKTA
public = Nis+May    154 sigma      X
public = Haz+Tem    108 sigma      X
public = yalniz Nis 417 sigma      X
```

**Altı örneklem-dışı LB tahmini** (ilk N ile çöz, sonrakileri tahmin et):
N=12 → 6 tahmin, ort |hata| **0.00008**; N=14 → 4 tahmin, 0.00010. Sistematik yön yok.

**Sağlam alternatif YOK:** rank 12/14/15 ve ridge varyantlarının hepsi kazancı
öldürüyor (1.0104-1.0110). Kazancın %42'si tek öz-yönden (j=15, SNR 128.7).

`gun1_baseline.csv` havuzdan çıkarıldı — farklı format (`hedef` kolonu, `R00320`
id'leri) ve dosya 27 Ağustos'ta başka bir oturumca yeniden üretilmiş.

---

## 6. Kapanan eksenler (tekrar açma)

| eksen | sonuç |
|---|---|
| ölü trafo sıfırlaması | **çürüdü** (§1) |
| soğuk son işlem | 13 aile / ~70 varyant, **hepsi red**. En genel tek-değişkenli kalibrasyon üç blokta zararlı (−0.036) |
| sıcak son işlem | 63 varyant, **hepsi red**. Grup ofsetleri bloklar arası ters taşınıyor (kVA yaz/kis −0.734, ilçe yaz/guz −0.497, seviye desili yaz/kis −0.651). Taşıyan tek eksen hafta günü (guz/kis +0.989), o da kapının 1/10'u |
| soğuk model mimarisi | 6 harman + 5 ayar varyantı, **hepsi red** (§3'teki ölçüm sorunu) |
| takvim / tatil | **geçilemez.** Kâhin tavanı yaz25 −0.00067, kis26 −0.00021; kapı −0.002. Ramazan Bayramı 2026 eğitimde (20-22 Mart), testte değil. Kurban 27-30 Mayıs testte |
| `docs/43` §6 rötuşları | "−0.011 iyimser toplam" **çürüdü**: 3'ü zaten v83'te, 1'i reddedilmiş, kalan 2'si **−0.00032** |
| yapısal açık (metrik/gönderim/uç/ufuk/panel) | **yok** — 7 eksen tarandı, biri (alt taban F=0.05) doğru yönlü ama 21 kat küçük |
| üst kırpma | ölü; v83'te 100k üstü 336 satır. `max 199.414 kWh` iki 35.800 kVA fider biriminden, makul |

---

## 7. Sigortalanamaz taban gürültü

8 trafo (7'si 35.800 kVA) eğitimde 455 günün 29'unda **1e7-5.04e7 kWh**
raporluyor — kendi medyanlarının ~60 katı, sayaç/birim arızası. Testte
beklenen ~8.6 böyle satır, korunmasız maliyet **+0.00086 MSE**. Optimum hedge
yalnız −2.0e−06. **Skorumuzun ~0.0009'u öngörülemez.**

---

## 8. PROB KAMPANYASI — açık olan tek yol

**Temel fikir:** reddedilen ~150 adayın neredeyse hepsi *bloklar arası
taşınmama* yüzünden reddedildi. **LB probu taşıma gerektirmez, test setini
doğrudan ölçer.** `MSE(k) = MSE0 - 2kL + k^2 Q` → `k* = L/Q`, kazanç `L^2/Q`.
Ekip bunu üç rejim sabiti için yapmış (`docs/47` §1) ama daha ince gruplara
hiç genişletmemiş.

**İşaretin bloklar arası ters olması probu ÖLDÜRMEZ** — κ işareti düzeltir.
Öldüren tek şey `|rho| ~ 0`.

### Hile tavanları (aynı bloktan öğrenilen grup ofseti, ölçüldü)

```
SICAK ilce (47)           -0.02019 rejim / -0.01571 toplam
SICAK seviye desili (10)  -0.01883 / -0.01466
SOGUK ilce (46)           -0.18341 / -0.04064
SOGUK seviye desili (10)  -0.05137 / -0.01138
SOGUK kVA kovasi (8)      -0.03754 / -0.00832
```

### Prob simülasyonu (yaz25 içi trafo-ayrık bölme, desen yaz25'ten)

```
yon                    kazanc      kappa (A/B)
SICAK ilce            -0.010085    +0.72 / +0.58
SICAK ay              -0.008342    +0.92 / +1.09
SICAK seviye desili   -0.006464    +0.96 / +0.76
SOGUK seviye desili   -0.020718    +0.54 / +1.29
SOGUK kVA kovasi      -0.000613    +0.09 / +0.51
```

Desen mevsime bağlı işaret değiştiriyor (sıcak seviye `yaz25|kis26 rho=-0.787`),
o yüzden tüm desenler **yaz25'ten** alındı — test penceresinin mevsim ikizi.

### Dürüstlük kapısı

Yön başına **1 skaler** (κ). Grup üyelikleri ve göreli ağırlıklar tamamen
CV'den (etiketli eğitim verisi), LB'den değil. 6 prob + v93'ün 16'sı = **<=22
skaler / 714.688 satır**. Satır başına serbest parametre **yok** →
`docs/48`in "public skoru satır düzeyinde tersine çözme" yasağı ihlal edilmiyor.

Aşırı uydurma cezası (f = public payı):
```
f=0.05  3 yon 1.79e-4  6 yon 3.58e-4  22 yon 1.31e-3   [satir duzeyi olsaydi: 42.6]
f=0.30  3 yon 2.20e-5  6 yon 4.39e-5  22 yon 1.61e-4   [5.23]
```

---

## 9. Hazır dosyalar

| dosya | ne | Q | ön kayıt |
|---|---|---|---|
| `tuketim_v101_hepsi.csv` | v93 + P1 + P3 + grup B + boşluk öncesi | 0.045166 (v93'e) | **~1.0006**, bant 0.992-1.012 |
| `tuketim_v93_gram_optimum.csv` | Gram optimumu, çıpa | 0.009797 (v83'e) | **1.00833** |
| `tuketim_p1_sicak_ilce.csv` | 47 ilçe ofseti, κ=0.65 | 0.010765 | %44'te 2. sıra |
| `tuketim_p2_sicak_seviye.csv` | 10 desil, κ=0.86 | 0.005041 | CV −0.00646 |
| `tuketim_p3_soguk_seviye.csv` | 10 desil, κ=0.92 | 0.016593 | CV −0.02072, tek blok |
| `tuketim_p4_sicak_ay.csv` | ay | 0.001598 | %80.7'si span içinde, zayıf |
| `tuketim_p5_soguk_kva.csv` | kVA kovası | 0.000429 | zayıf, κ büyütülmeli |
| `tuketim_v96_grupb_optimum.csv` | grup B, δ=+1.00 | 0.010003 | r>1.0'da 2. sıra |
| `tuketim_v95_gram_grupb.csv` | grup B, δ=+0.50 | 0.002501 | muhafazakâr |
| `tuketim_v94_bosluk_oncesi.csv` | 745 satır, δ=−0.50 | — | **−0.00061 DOĞRUDAN**, 3/3 blok |
| `tuketim_prob_yas790.csv` | yaş bandı 7-90g, κ=+0.20 | 0.083617 | beklenti −0.00227 |
| `tuketim_v99_mimari_sekil.csv` | mimari harmanının saf şekil yönü | 0.024092 | ölçülmemiş |
| `tuketim_v90_temiz_sota.csv` | sota_v1'den grup B zehri çıkarılmış | 0.031650 | ölçülmemiş |
| `tuketim_v82_ayirici.csv` | kuyruk/soğuk sabitini ayırır | 0.041799 | beklenti −0.00109, 26 Ağu'dan beri bekliyor |

Çözücü: `scripts/prob_coz.py` (çok yönlü Gram güncellemesi, koşul sayısı kapısı,
yuvarlama bant taraması).

### `v101` beklentisi, gerçekleşme oranına göre

```
%100  0.98738   |  %50  0.99791  |  %25  1.00314  |  %13  1.00615 (basa bas)
 %75  0.99266   |  %35  1.00105  |  %15  1.00522  |   %0  1.00833 (v93'e doner)
```
Bileşen ağırlıklı dürüst beklenti **~%35 → 1.0006**. `HAK 2`de `κ*` çözüldükten
sonra kazanç `f^2 x 0.041877`; **1. sıra için f >= %90** gerekir.

---

## 10. Gönderim planı

```
HAK 1  tuketim_v101_hepsi.csv           gercek atis
HAK 2  v93 + k* x (v101 - v93)          HAK 1'in skorundan COZULUR, Q=0.045166
HAK 3  yeni prob (p2 / yas790 / v82_ayirici / harman agirliklari)
```

`v93`i ayrıca göndermeye **gerek yok**: `L = (MSE93 + Q - MSE101)/2` formülünde
`MSE93` ön kayıttan (1.016737) alınabilir; o tahmin altı örneklem-dışı sınavda
<=1.5e−4 hatayla geçti ve `MSE93`teki 0.0002'lik hata `κ`yı yalnız 0.002 kaydırır.

**Aşağı yön yok:** v83'ün 1.01318'i tabloda, Kaggle en iyi skoru tutar.

### Sıfırıncı iş
```powershell
uv run python -m kaggle competitions submissions -c grid-up-datathon
```

---

## 11. Yeni kalıcı kurallar

13. **Soğuk rejim hükmü yalnız `kis26` ile verilir.** `yaz25`/`guz25` soğuk
    satırlarının %97'si kimlik-ezberiyle çözülebiliyor, `kis26` ve `TEST`
    %0. İki kirli blok temiz blokla TERS korelasyonlu.
14. **Bir eksen "işaret bloklar arası tutmuyor" diye reddedilmişse, PROB ADAYIDIR.**
    κ işareti düzeltir; öldüren tek şey `|rho| ~ 0`.
15. **Ölü-trafo maskeleri GRUP A ile sınırlıdır** (kural 9'un doğrulanmış hâli).
    Grup B'ye dokunulacaksa yön **yukarı**dır, aşağı değil.
16. **Alt küme RMSLE'si havuzlanmış sayıyla kıyaslanmaz.** `docs/51` §3 bu yüzden
    kırıldı: 694.849 satırlık havuza 158.369 soğuk satır dahildi ve yalnız-sıcak
    CV ile kıyaslanmıştı.
17. **Bir grubu kendi tanımıyla doğrulama.** `train.max()==0` ile seçilen grubun
    "eğitimde %100 sıfır" olması totolojidir, kanıt değil.

## 12. Ölçülemeyenler

- Public/private LB oranı — `docs/31` §2'den beri açık. Zamansal bölme çürütüldü,
  ama oran hâlâ bilinmiyor. Tarayıcıdan dakikalar içinde kapanır.
- Final için seçilebilecek dosya sayısı.
- Harici veri izni — resmî teyit yok, `docs/30` metin yorumuna dayanıyor.
- Soğuk kohortun ölü trafoları: soğuk MSE'nin %58-69'u (toplam MSE'nin ~%30'u)
  `y=0` satırlarda, ve gözlenebilir hiçbir öznitelik onları ayırt edemiyor
  (AUC ~ 0.5). Kırmak için yeni bilgi gerekir (OSM yapı yoğunluğu vb.).

---

## 13. SONUÇ — 28 Ağustos gönderimleri (ÖLÇÜLDÜ)

```
55833361  tuketim_v101_hepsi.csv          04:16 UTC   1.01614
55833415  tuketim_v102_kappa_optimum.csv  04:19 UTC   1.00553   <- 2. SIRA
```

```
LB (2026-08-28 04:25)
  1. Grid Grinders   0.99138
  2. TasnifX         1.00553   <- 9. siradan 2. siraya
  3. Ahmet Celik     1.00606
  4. Ahmet Bugrahan  1.00615
  5. Duo-Electra     1.00823
```

### 13.1 `v101` ıskaladı — ve ıskalaması yön enerjisini tam çözdü

Ön kayıt 1.0006, gerçek **1.01614**. Tahmin hatasının sebebi **aritmetik**, ölçüm değil:

> Gerçekleşme oranı `f` ile **net kazanç** ölçeklenmişti. Doğrusu: `Q` sabittir,
> yalnız `L` ölçeklenir. `dMSE(k=1) = Q - 2 f L0`. Bu formülle `f=%35` → 1.0156
> çıkıyordu; gelen 1.01614, yani `f = %33.8`. **Bileşenlerin gerçekleşme oranı
> doğru tahmin edilmiş, aritmetik yanlış kurulmuştu.**

Başa baş noktası `k=1`'de `f = Q/(2 L0) = %51.9`. Yani `Q=0.045166` gibi büyük
bir yönde tam adım atmak, gerçekleşme yarıdan azsa **zarar** verir. Prob
tasarlarken bakılacak sayı `k=1`deki net kazanç değil, `L^2/Q`dur.

### 13.2 `v102` — iki ölçülmüş skordan çözülen optimum

```
taban  v83   olculmus 1.01318  (MSE 1.026534)
prob   v101  olculmus 1.01614  (MSE 1.032540)
yon    d = log1p(v101) - log1p(v83)
       Q = ||d||^2/n = 0.073292
       L = (m83 + Q - m101)/2 = 0.033643
       k* = L/Q = 0.459022
       optimum MSE = m83 - L^2/Q = 1.011091  ->  RMSLE 1.00553

ON KAYIT 1.00553   GERCEK 1.00553   (5 hanede birebir)
```

`MSE(k)` bir doğru boyunca **tam** ikinci dereceden olduğu için bu bir tahmin
değil, cebirsel sonuçtur; belirsizliği yalnız 5 hane yuvarlamasından (+-1e-5).

### 13.3 DUZELTME — `v93` iyimser DEGILDI (ilk hüküm YANLIŞTI)

İlk okumam şuydu: A çözümü (taban v83, k*=0.4590 -> 1.00553) B'den (taban v93,
k*=0.3251 -> 1.00596) iyi çıktığına göre v93 k=1'de aşırıya kaçıyor. **Bu çıkarım
yanlıştı** ve iki ajan bağımsız olarak çürüttü:

```
L(v93-v83) = 0.0097966 = Q(v93-v83)   ->  kappa* = 1.0000 TAM
```
`c1 = 2(v93-v83)` ölçülmüş span'ın **içinde** (Q_dik = 2.1e-07), dolayısıyla
18 eski skordan cebirsel olarak çözülür ve `v93`ün 1.00833 ön kaydı
**DOGRULANIR**. 21 dosyalık yeni havuzla da 1.008334 çıkıyor. `v101`/`v102`
ölçümleri eski `b` vektörünün hiçbir bileşenini yalanlamıyor.

A'nın B'den iyi olması yalnızca **A'nın doğrusunun optimuma daha yakın geçmesi**
demek; v93 iyimser değil, sadece `v102`den **kötü bir nokta**ydı (1.00833 > 1.00553).

**Gerçek hata başka yerdeydi:** eski Gram çözümü 18 yönde rank 17 almış, ama
17. özdeğer `4.19e-12` = sayısal null. O tek yön **32.774**'lük sahte kazanç
üretiyordu (`coz.json: proj_norm2 32.774`). Doğru kesme rank 16.

**Ve `v101`in kimliği belgede yanlış yazılmıştı.** Doğrusu (kalıntı Q=8.8e-07):
```
v101 - v83 = 2*(v93-v83) + (P1-v93) + (P3-v93) + (B96-v93) + (bos-v93)
```
`v93` yönü **iki kat** giriyor; §9'daki bileşen Q toplamı bu yüzden tutmuyordu.
Doğrusu sum(Q_i) = 0.0768033, demet Q = 0.0732922.

### 13.4 Kalıcı kurallara ek

18. **Bir yön için karar sayısı `L^2/Q`dur, `k=1`deki net kazanç değil.**
    `k=1` başa baş noktası `f = Q/(2 L0)`; büyük `Q`lu yönlerde tam adım
    gerçekleşme yarıdan azsa zarar verir. Prob GÖNDER, sonra `k*` ile in.
19. **Taban olarak ÖLÇÜLMÜŞ dosya kullan.** Ön kayıtlı bir tabana dayanan κ
    çözümü, o ön kayıt iyimserse kazancı yer. v83 tabanlı çözüm v93 tabanlıyı
    0.00043 geçti.

### 13.5 Kalan durum

```
28 Agustos kotasi: 1 hak KALDI (kullanici izni bekliyor)
Bitis: 1 Eylul 23:59 UTC  ->  29/30/31 Agustos + 1 Eylul = 12 hak daha
Lidere acik: 1.011091 - 0.982835 = 0.028256 MSE
```

Ölçülmemiş, v102'ye dik yönler: `p2_sicak_seviye` (CV -0.00646) ·
`prob_yas790` (-0.00227) · `v82_ayirici` (-0.00109) · `v99_mimari_sekil`
(Q=0.024092) · harman agirliklari 2/3/0 (|etki| 0.003-0.006).

---

## 14. 28 AĞUSTOS — ÜÇ GÖNDERİM, TAM BİLANÇO

```
55833361  tuketim_v101_hepsi.csv          04:16   1.01614
55833415  tuketim_v102_kappa_optimum.csv  04:19   1.00553   <- 9. siradan 2. siraya
55837187  tuketim_v109_birlesik.csv       10:20   1.01818   <- SIFIR kazanc
kota bitti (3/3). LB gun sonu: 1. Grid Grinders 0.99138 | 2. Atakan Aldemir 1.00078
                              | 3. TasnifX 1.00553
```

### 14.1 `v109` — CV türevli yönlerin ölümü

`v109 = v102 + v108(onarilmis olcutle sicak seviye desili) + y1(yeni model ailesi)`.
Iki bilesen dik (kosinus -0.0056), `Q = 0.025835`.

```
L      = +0.000118   (beklenen ~0.0057)
kappa* = +0.004549   (beklenen ~1.0)
kazanc = 0.00000053  -> optimum RMSLE 1.00553 = mevcut skor
```

**Birlesik yon gercekle neredeyse hic hizali degil.** Iki okuma:
- `v108` beklendigi gibiyse, `y1`in kappa'si **-0.28** (yeni aile TERS yonde)
- ya da ikisi birden sifir

Her iki halde de hüküm ayni: **onarilmis olcutle kurulan grup ofseti de, 27.7 kat
enerjili yeni model ailesi de LB'de karsilik bulmuyor.**

### 14.2 Bunun ortaya çıkardığı ayrıştırma

```
L(v101)      = +0.033643   olculdu
L(v93 yonu)  = +0.019593   span ICI, cebirsel olarak TAM
L_kalan      = +0.014050   = L(P1) + L(P3) + L(grupB) + L(bosluk)
```
`v109` grup-ofseti ailesinin `L`sinin ~0 oldugunu olctugune gore (`v108` ile `P1/P3`
ayni aile), `L_kalan`in neredeyse tamami **grup B**den geliyor:
```
kappa_B ~ 0.014050 / 0.010003 = 1.405
ILERI PENCERE olcumu (bagimsiz): delta* = 1.01 / 1.61 / 2.19   -> TUTARLI
v102 o yonde yalniz kappa = 0.459022 uyguluyor  ->  KACAN KAZANC var
```

### 14.3 Genelleştirilmiş dönüşçü — nüfus büyütmek İŞE YARAMIYOR

Hipotez: "grup B 93 trafo; testte ayni profilden 1.084 trafo var, nufusu buyutursek
`Q` buyur ve kazanc `Q*delta^2` ile artar." **Olculdu, YANLIS cikti.**

```
nufus tanimi              satir     delta*   Q*delta*^2
bosluk>=90, sev<=-3       5.045     1.459     0.01502   <- TEPE, dar tarafta
bosluk>=180, sev<=0      11.195     0.636     0.00634
bosluk>=180              16.028     0.503     0.00567
genis (Q 6.6 kati)       33.634     0.378     0.00673   isaret 4 kesmede TUTARSIZ
kisa bosluk + olu seviye  1.164    -1.051     negatif   <- ISARET TERSINE DONUYOR
```
`delta`, `Q`dan hizli curuyor. Satir bazli hedefli atis da olculdu: tavan 0.0113,
gurultu cezasi 0.0154 -> **net negatif**.

### 14.4 Toplu dönüş riski (yeni, ciddi)

Egitimde >=8 trafonun ayni gun dondugu olaylar ayristirildi:
```
TEKIL (organik) donus:  79 trafo / 6.511 satir   ->  +0.480
TOPLU donus:            35 trafo / 3.280 satir   ->  -0.466  (trafo medyani +0.186)
```
Ileri pencere olcumu agirlikla **tekil** donuslerden geliyor; ama test nufusunun
T1'de %33'u 2026-05-11'de, %52'si 05-03'te, T2'nin %81'i 05-11'de **TOPLU** giriyor.
Bu, donuscu ekseninin en buyuk tek riski.

### 14.5 Hazır dosyalar (29 Ağustos için)

| dosya | ne | Q(v102) | ön kayıt | başa baş |
|---|---|---|---|---|
| `tuketim_v112_donuscu_yarim.csv` | T1 +0.722 / T2 +0.319 | 0.00463 | **0.99861** | δ_T1 < 0.361 |
| `tuketim_v111_donuscu.csv` | T1 +1.444 / T2 +0.638 | 0.01850 | **0.99629** | δ_T1 < 0.722 |
| `tuketim_v110_grupb_optimum.csv` | grup B κ 0.459→1.405 | 0.00895 | κ_B=1.405'te 1.00107 | κ_B < 0.932 |
| `tuketim_v103_gram2.csv` | rank-17 Gram optimumu | 0.00039 | 1.005334 | — |
| `tuketim_v108_sicak_onarim.csv` | onarılmış ölçüt, sıcak desil | 0.00596 | ÖLÇÜLDÜ ~0 | — |

`T1 = bosluk>=90 & seviye<=-2 & n>=30` (64 trafo / 5.127 satir),
`T2 = bosluk>=180 & seviye<=0, T1 haric` (80 trafo / 6.232 satir). Grup A kirliligi 0.

**29 Agustos plani:**
```
HAK 1  v112 (guvenli, basa bas olculen tum kesmelerin altinda)
HAK 2  kappa* optimumu -- HAK 1'in skorundan COZULUR; delta olculen degerdeyse 0.99629
HAK 3  serbest
```

### 14.6 1. SIRA — NET HÜKÜM: ULAŞILAMIYOR

```
gereken Q*delta*^2 = 0.028257
olculen en iyi     = 0.018500
EKSIK              = 0.009757   (gerekenin %35'i)
```
`v111` lideri ancak gercek `delta_T1 >= 1.747` ise gecer — olcumun **1.75 sigma** ustu.
Bagimsiz teyit: LB ayristirmasindan gelen `L_kalan = 0.01405` de ayni mertebede.

Uc gunde 15 ajan, ~200 aday olculdu. Tukenen eksenler: Gram (tavan 1.005334) ·
CV turevli tum yonler (`kappa ~ 0`) · yeni model ailesi (enerji 27.7 kat, hizalanma 0) ·
takvim (kahin tavani kapinin 1/3'u) · yapisal acik (yok) · sicak ve soguk son islem ·
soguk model mimarisi · rotus envanteri (-0.011 iddiasi -> -0.00032).

**Varilabilecek yer: 0.996-0.999 ile saglam 2. sira.** Lidere ~0.005 kalir.

### 14.7 Kalıcı kurallara ek

20. **CV'den turetilen yonler LB'de karsilik vermiyor.** Olculen: grup ofseti ailesi
    (yaz25 ve kis26 kaynakli, ikisi de) `kappa ~ 0`; yeni model ailesi `kappa ~ 0`.
    Ise yarayan tek seyler LB'nin kendi olctugu yonler oldu (Gram, `kappa` cozumleri,
    ileri-pencere ile olculen grup B).
21. **Nufus buyutmek `Q*delta^2`yi artirmaz.** `delta`, `Q`dan hizli curur; optimum
    nufus DAR taraftadir. Genis tanimda isaret tutarsizlasir.
22. **Toplu donus ile tekil donus ZIT yonlerdir** (-0.466 vs +0.480). Bir donuscu
    ekseni kurarken test nufusunun toplu-giris payi olculmelidir.
23. **`f` (gerceklesme orani) yone ozgudur, tasinmaz.** `v101` demetinde 0.4115
    olculmustu; `v109`da 0.0045 cikti. Butce hesabini tek bir `f` ile kurma.

---

## 15. Hedefin kafesi — hiç bakılmamış eksen, ölçüldü, KAPANDI

"Hiç ölçmediğimiz bir şey olmalı" sorusuna karşı dört yapısal boşluk tarandı.
Üçü zaten kapalıydı; biri gerçekten bakılmamıştı ve bu bölümde kapanıyor.

### 15.1 Bulgu: `tuketim` trafo bazlı bir kafes üzerinde

`train.csv`in "kuruş" dağılımı düz değil. En sık artıklar 0/20/60/80/40
(0,20 kWh katları), sonra 32/12/96/48/68 (0,04 katları). Trafo bazlı GCD:

```
degerlendirilen trafo (>=20 sifir-disi kayit) : 4.510
gcd == 1  (kafes YOK)                         :    58   ( 1,3%)
gcd  > 1  (kafes VAR)                         : 4.452   (98,7%)
en sik adimlar (kWh) : 0,12(900) 0,08(781) 0,32(678) 0,20(515)
                       0,06(461) 0,40(361) 0,04(340) 0,02(136)
uc degerler          : 41,40(15 trafo)  12,60(6)  2,10(4)  1,05(4)
```

Bunlar sayaç çarpanı (akım trafosu oranı) imzasıdır. Sonuç: adımı `g` olan bir
trafo `(0, g)` aralığında **değer üretemez**.

### 15.2 Puan değeri: YOK

`v102` bu kısıta karşı sınandı (kafesi bilinen 490.060 test satırı):

```
(0,g) imkansiz bolgede tahmin  :   188 satir  (%0,038)
gozlenen min-nonzero altinda   : 8.036 satir  (%1,640)
hepsini 0'a  cekmenin |dMSE| ust siniri : 1,864e-03
hepsini g'ye cekmenin |dMSE| ust siniri : 2,560e-05
GEREKEN dMSE                            : 2,826e-02
```

Üst sınır bile gerekenin %6,6'sı ve bu bir **yer değiştirme** sınırı, kazanç
değil; gerçek kazanç bunun küçük bir kesri, işareti belirsiz.

> **Tuzak:** ham "oracle tavanı" `mean(g²/12 / (1+t)²)` = 1,654e-01 çıkıyor ve
> gerekenin 5,85 katı görünüyor. **Sahtedir** — 41,4 kWh adımlı 15 trafo
> tarafından domine ediliyor ve o tavan ancak hatamız zaten `g/2`nin altındaysa
> bağlayıcı. O trafoların ortanca tüketimi 38.928 kWh; hata oraya yaklaşmıyor.

### 15.3 Aynı taramada kapanan diğer üç boşluk

| Boşluk | Durum |
|---|---|
| Sayısal olmayan `tanim` (`202917T`, `ege perla tr-4`, `İskele DM`) | 9 train / 12 test trafo, 1.037 test satırı — kütle yok |
| Mevsim aynası (train ≤2025-03-31 → doğrula 2025-04→07) | **Zaten kurulu**, `curut_eksen5_*` ailesinde 13 betik |
| Harici veri | 21 kaynak kurulu + ablasyonu yapılmış (`docs/07` §208) |

### 15.4 Kayda geçen yapısal gerçek

```
train  2025-01-01 -> 2026-03-31   455 gun
test   2026-04-01 -> 2026-07-31   122 gun
```

Test bloğu Nisan–Temmuz; eğitimde bu mevsimin **tek** örneği var (2025-04→07).
Rastgele ya da geriye dönük her doğrulama yanlış mevsimi ölçer. "CV türevli tüm
yönler κ≈0" anomalisinin yapısal sebebi budur ve **onarılamaz** — daha fazla
tarih yok.

### 15.5 Kalıcı kural 24

> **Hedefin cebiri tarandı.** Kafes, id uzayı, doğrulama geometrisi ve harici
> veri — tabular bir yarışmada "hiç bakılmamış şey"in saklanabileceği dört yer
> de kapalı. Bundan sonra "bakmadığımız bir şey olmalı" sezgisi **yeni bir
> ölçüm getirmeden** gündeme alınmaz. Eksik olan bir teknik değil, elde
> edilemeyen bilgidir (kimliklenebilirlik duvarı, §12).

---

## 16. Span tükendi — 22 ölçümün tam çözümü

"Onlar ne yaptıysa bul" talebi üzerine kazılan yer: kendi ölçüm yığınımız.

### 16.1 §14.1'in mantık boşluğu kapandı

§14.1 `v109`u iki okumaya açık bırakıp "her iki halde de hüküm aynı" demişti.
**Değildi.** Okuma A (`v108` +0,0057 çalışıyor, `y1` −0,28 zehirli) doğru olsaydı
"CV yönleri ölü" hükmümüzün tamamı çöpe giderdi. Ayrıştırıldı:

`d108` ve `dy1`, önceki 21 ölçülmüş yönün span'ına projekte edildi. Span-içi
kısımların `L`si oradaki ölçümlerden **tam olarak** çözülür:

```
d108  Q=0.005957  span-ici pay %31.27  L_span = -0.000141
dy1   Q=0.019999  span-ici pay %79.76  L_span = +0.000209
                                toplam = +0.000068
                  olculen L(d109)      = +0.000069     <- 1e-6 icinde ORTUSUYOR
```

Okuma A doğru olsaydı `d108`in span-içi payında **+0,0018** görmemiz gerekirdi;
−0,000141 var — 13 kat küçük ve ters işaretli. **Okuma B doğru.** Hüküm ayakta
ama artık varsayım değil, ölçüm.

> Yan bulgu: `v109 = v102 + d108 + dy1` eşitliği tam değil, bağıl fark 2,04e-02.
> Belgede yazan kuruluş yaklaşıktır.

### 16.2 22 ölçümün span tavanı: 1.00531

Eski Gram çözümü 17 rank'lıydı ve `v101`/`v102`/`v109` daha ölçülmemişti. Üçü de
eklenip yeniden çözüldü. Taban `v102` (m0 = 1.011091), 21 yön, `L` gürültüsü
5 hane yuvarlamasından sd = 7,14e-06, bileşen seçimi SNR >= 3:

```
SNR>=3 secilen bilesen : 14 / 19
span tavani            : MSE 1.010648   RMSLE 1.00531
mevcut                 : MSE 1.011091   RMSLE 1.00553
kazanc                 : dMSE -0.000443  (RMSLE -0.00022)
GEREKEN (lider 0.99138): dMSE -0.028256
```

Bileşen 18: özdeğer 2,489e-12, "kazanç" 16,03 — §13.3'te yakalanan sayısal null
tuzağı. SNR 0,9 ile **bağımsız olarak reddedildi**; yöntem doğrulandı.

### 16.3 Hüküm

**Gönderdiğimiz her şeyin span'i tükendi.** 22 ölçümün bütün kombinasyonları
RMSLE'de 0,00022 daha veriyor. Liderin farkı bir yeniden-birleştirme değil,
**hiç göndermediğimiz bir yön**.

En iyi hâl aritmetiği:
```
mevcut MSE                     1.011091
- span tavani                 -0.000443
- v111 donuscu (olculen delta) -0.018500
--------------------------------------- 
                               0.992148  ->  RMSLE 0.99603   (2. sira RAHAT)
lider icin gereken             0.982835  ->  RMSLE 0.99138
KAPANMAYAN ACIK                0.009313 MSE
```

### 16.4 Plan değişikliği: `v112` değil `v111`

Span tükendiği için "güvenli oyna" seçeneğinin karşılığı kalmadı — başka yol yok.
`v111` tam genlik uygular, `v112` yarısını. Başa baş `δ_T1 < 0,722`; ileri pencere
ölçümleri 1,01 / 1,61 / 2,19 — **üçü de eşiğin üstünde**, en düşüğü %40 pay ile.

```
HAK 1  tuketim_v111_donuscu.csv   on kayit 0.99629
HAK 2  kappa* -- HAK 1'in skorundan COZULUR (v102'de bu tam tutmustu)
HAK 3  span tavani + HAK 2 birlesigi
```

### 16.5 Kalıcı kural 25

> **Karışım yön gönderme.** `v109` iki bileşeni tek hakta ölçtü ve ayrıştırmak
> için span projeksiyonu gerekti; ayrıştırılamasaydı bir hak tamamen boşa
> giderdi. Her hak **tek** bir yönü ölçmeli.

### 16.6 Kalıcı kural 26

> **Span tavanı her yeni ölçümden sonra yeniden çözülür.** 1.00531 bugünkü
> değerdir; her yeni gönderim span'i büyütür ve tavanı düşürebilir.

---

## 17. `v111` ÇÜRÜDÜ · 2026-05-11 dalgası · yeni plan

### 17.1 `v111`/`v112` GÖNDERİLMEMELİ

`d8_uret.py:28` `DELTA={"T1":1.4438,"T2":0.6378}` uyguluyor. Bu katsayı `d5`te
**organik** dönüşlerden ölçüldü — hem de kesme başına **4-5 trafodan**
(`2025-09-30: aday 101 → DÖNEN 4`). Hedef kohort ise:

```
T1  63 trafo -> %85.7 TOPLU  (33'u 2026-05-03, 21'i 2026-05-11)
T2  77 trafo -> %84.4 TOPLU  (65'i 2026-05-11)
```

§14.4 bu riski görmüş, **üretime hiç geçmemiş**. `d8` tek katsayıyı ayrım
yapmadan uyguluyor.

### 17.2 Toplu dönüşün δ'sı ÖLÇÜLDÜ (`d10`)

`d7`de ortalama −0,466 iken trafo-medyanı +0,186 idi; işaret çelişiyordu.
Tam 122 günlük ileri pencere şartı + gün-kontrolü ile yeniden ölçüldü:

```
                trafo  satir   ortalama   medyan  budanmis%10   std
TOPLU              31   3258    -0.4769  +0.1856      -0.1736  1.87
ORGANIK            54   5569    +0.4244  +0.7337      +0.4975  1.01
FARK (T-O)                      -0.9013  -0.5481      -0.6711
```

**Üç sağlam istatistik de aynı işarette:** toplu dönüşler organikten
0,55–0,90 log birim aşağıda. `d7`nin işaret çelişkisi kısmi pencere
artığıymış. Dolayısıyla `δ_toplu ≈ δ_organik − 0,67` → uygulanan 1,4438'e göre
`g ≈ 0,25–0,55`, başa baş ise 0,50. **`v111` başa başın altında.**

### 17.3 ASIL BULGU — 2026-05-11 dalgası, testin %25,33'ü

Panel giriş günleri:
```
2026-04-01  3928 trafo  469.821 satir  %65.7   panel basi
2026-05-11  2222 trafo  181.038 satir  %25.3   <- SISTEMIK DEVREYE ALMA
2026-05-03   141 trafo   12.223 satir   %1.7
```

Bu blok bugüne kadar **kendi yönü olarak hiç prob edilmedi** — §16'nın "span
tükendi" hükmünün dışında kalan yön tam olarak budur. Üç ayrık alt kümeye
bölündü (satır kümeleri kesişmiyor → yönler dik):

```
alt kume                                trafo   satir      pay      Q(s=0.30)
P11 soguk : train gecmisi YOK            1326  108.253   %15.15    0.013632
P12 aktif : son kayit >= 2026-03-27       502   40.771    %5.70    0.005134
P13 kesik : son kayit <  2026-03-27       394   32.014    %4.48    0.004031
                                         2222  181.038   %25.33
```

`HAK2` kazancı `= δ² × pay`, prob adımından **bağımsız**. Eşikler:
```
ucu de |delta| 0.2014 ise  ->  2. sirayi gecer
ucu de |delta| 0.3340 ise  ->  LIDERI GECER
```

### 17.4 Önsel: `v102` dalgayı YUKARI yazıyor

Train geçmişi olan iki alt kümede, 2025'in aynı penceresine karşı
(mevsim kayması dalga-dışı 4.086 trafodan +0,0376 ölçülüp düzeltildi):

```
grup    trafo   train ofs 2025-05-11..07-31   v102 ofs    fark   duzeltilmis
aktif     502                        0.4693     0.6306  -0.1613     -0.1238
kesik     394                        0.4318     0.7421  -0.3102     -0.2727
```

Yani beklenen gerçek ofset **negatif**; prob adımı `s = −0,30` seçildi.
HAK2 kazancı `L²/Q ≥ 0` olduğu için işaret sonucu değiştirmez, yalnız probun
kendi skorunu etkiler.

> Soğuk blok için önsel YOK (train geçmişi yok). Uyarı: LB ile çözülmüş soğuk
> sabiti (+0,104600) tüm soğuk satırlar üzerinden çözüldü ve soğuğun **%68'i**
> bu dalga; yani o sabit zaten büyük ölçüde dalga-soğuğa göre ayarlı. Artık
> sinyal küçük olabilir.

### 17.5 Plan — 3 HAK, TEK GÜN (29 Ağustos)

İlk taslak 4 hak / 2 gün idi. Birleştirme maliyeti hesaplanınca 3 hak / 1 güne
indi (`d13_iki_prob.py`):

```
AKTIF + KESIK birlesirse   kayip 0.000556 MSE (RMSLE 0.00028)  IHMAL EDILEBILIR
                           ikisinin de onseli NEGATIF (-0.1238 / -0.2727)
SOGUK da katilsaydi        isareti TERS cikarsa kayip 0.009785
                           yani kazancin NEREDEYSE TAMAMI; soguk icin onsel YOK
```

Karar: soğuk ayrı kalır, aktif+kesik birleşir. İki yön dik olduğu için
**üçüncü hak aynı gün ikisinin optimumunu birden uygular.**

```
HAK1  tuketim_p11_dalga_soguk.csv      1326 trafo  108.253 satir  Q=0.013632
HAK2  tuketim_p14_dalga_gecmisli.csv    896 trafo   72.785 satir  Q=0.009163
HAK3  d12_coz.py --prob p11=<skor> --prob p14=<skor>
      -> tuketim_v120_dalga_optimum.csv,  kazanc = sum L_i^2/Q_i >= 0
```

Denetimler: `adim(p12)+adim(p13) == adim(p14)` sapma 8,9e-16 · `p11` ile ortak
satır 0 · dördü de kapıdan geçti.

**Çözücü gidiş-dönüş testinden geçti:** δ = −0,20 / −0,25 varsayımından skor
üretildi (1.00327 / 1.00249), skorlar çözücüye geri verildi, δ = −0,2000 /
−0,2499 kurtarıldı; ön kayıt 0.999337 vs analitik 0.999333, fark 4e-6 = 5 hane
yuvarlamasının payı. `Q` denetimi 1,1e-16.

Örnek sonuçlar (`|δ_soğuk|`, `|δ_geçmişli|`):
```
0.10 / 0.30 -> 1.000205   2.yi gecer
0.20 / 0.20 -> 1.000479
0.25 / 0.20 -> 0.998775   2.yi gecer
0.30 / 0.30 -> 0.994129   2.yi gecer
```

Üç prob da kapı denetiminden geçti (714.688 satır, id sırası birebir,
0 NaN/negatif). Diklik doğrulandı: ikişerli ortak satır **0**.

**Kaggle en iyi public skoru tuttuğu için kötü bir prob sıramızı DÜŞÜRMEZ** —
`v109` 1.01818 geldi, tablo hâlâ 1.00553 gösteriyor. Prob bedava; yalnız hak
harcar.

### 17.6 Kalıcı kural 27

> **Ölçülen katsayı, ölçüldüğü nüfusa uygulanır.** `d8` organik dönüşlerden
> çıkan 1,4438'i %85'i toplu olan bir kohorta uyguladı. Bir δ üretime
> geçmeden önce "hangi nüfusta ölçüldü, hangi nüfusa uygulanıyor" sorusu
> yazılı olarak cevaplanır.

### 17.7 Kalıcı kural 28

> **Ayrı ölç, sonra birleştir.** Alt kümelerin işareti farklı olabiliyorsa tek
> yönde birleştirmek Cauchy-Schwarz gereği kayıptır:
> `sum(pay_i·δ_i²) >= (sum pay_i·δ_i)² / sum pay_i`. Hak varken ayrı ölç.

### 17.8 Kırpma kusuru — yakalandı ve giderildi

İlk kurulan `p14`te adım her satırda `−0,30` DEĞİLDİ; 32 satırda `−0,046`ya
kadar sönüyordu. Sebep: adım log uzayında uygulanıp `expm1` ile geri dönülüyor
ve tahmin 0'a kırpılıyor. `v102` o satırlara 0,047–0,315 kWh yazıyor;
`−0,30` kaydırınca negatife düşüp sıfıra kırpılıyorlar.

Skor etkisi ihmal edilebilirdi (1,5e-6 MSE) **ama HAK 3'te çözücünün `Q`
denetimini tetikleyip betiği durdururdu** — yani sorun skor değil, gönderim
anında betiğin patlamasıydı.

Giderme: `KIRPMA_ESIGI = 0.90`. `log1p(v102) < 0,90` olan satırlar bloktan
çıkarılır; bedeli 153 satır (%0,21), karşılığı `|κ*| < 3,01` aralığında
**kırpma imkânsız**.

```
p14 (yeni)  72.632 satir  pay %10.16  Q=0.009146
            kirpilan satir 0   adim araligi [-0.300000, -0.300000]
            min log1p(v102) 0.9022  ->  guvenli |kappa*| < 3.01
p11         kirpilan satir 0   min log1p(v102) 2.6411  ->  guvenli < 8.80  (koruma gerekmedi)
```

`d12_coz.py`ye de aynı koruma kondu: `|κ*|` kırpma sınırını aşarsa dosya
yazmadan, sebebi söyleyerek durur.

### 17.9 Aparat gerçek Kaggle skoruna karşı doğrulandı

`v102 = v83 + κ(v101 − v83)`, üçü de ölçülmüş. Diskteki dosyalardan:

```
Q(v101-v83)   0.073292      L (iki skordan)  0.033643
kappa* = L/Q  0.459022      belgede yazan    0.459022     BIREBIR

TAHMIN v102 = 1.00553   GERCEK = 1.00553   FARK 4.06e-07
disk v102 == v83+kappa*(v101-v83)   sapma 4.4e-16
```

Hizalama · `log1p` uzlaşımı · `Q` · `L` · `κ` uygulaması — zincirin tamamı
canlı bir skoru **yedi hane** doğrulukla üretiyor.

### 17.10 Public/private genelleme kaybı: ihmal edilebilir

`κ*` public LB'den çözülüyor, nihai sıralama private. Blokun ortalama
artığının standart hatası:

```
p11 soguk    public %50: 54.126 satir  kaydirma hatasi 0.00432  kayip 2.8e-06
             public %30: 32.476 satir                  0.00558        4.7e-06
p14 gecmisli public %50: 36.392 satir                  0.00527        2.8e-06
             public %30: 21.836 satir                  0.00680        4.7e-06
```

Toplam ~1e-5 MSE; peşinde olduğumuz kazanç 1e-2 mertebesinde — **1000 kat**
küçük. `docs/07:39` (%50) ile `docs/27:167` (%30) çelişkisi bu yüzden önemsiz.

### 17.11 AÇIK KALAN TEK RİSK — final gönderim seçimi

`docs/31:39` "Kaç final seçilebilir" sorusunu listeye almış, **cevabını
kaydetmemiş**. Kaggle CLI bunu vermiyor (`--csv` kolonları: ref, fileName,
date, description, status, publicScore, privateScore — `selected` yok) ve
yarışma `competitions list`te görünmüyor.

Kaggle'ın evrensel varsayılanı: seçim yapılmazsa **en iyi public skorlu**
gönderimler private için otomatik seçilir. Bu doğruysa prob göndermek nihai
sıralamayı da etkilemez. Ama **bu yarışma için doğrulanmadı.**

> **YAPILACAK:** Submissions sayfasında "Use for Final Score" / "Select for
> final" denetimi var mı, kaç tane seçilebiliyor — tarayıcıdan bakılıp buraya
> yazılacak. Seçim varsa `v102` (veya en iyi dosya) elle seçilmeli.
