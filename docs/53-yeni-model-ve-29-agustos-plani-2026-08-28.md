# 53 — Yeni model (ileri-pencere doğrudan tahmin) ve 29 Ağustos planı

**Tarih:** 2026-08-28 gecesi · **Kaggle'a hiçbir şey gönderilmedi** (kota 3/3 doluydu).

`docs/52` §21'in P41/P42 planı **iptal**. Yerine ölçülmüş bir model geliyor.

---

## 1. Neden plan değişti

`docs/52` §21, iki *kör* yöne (ölçülmemiş, düşük kapsamlı) bahis oynuyordu ve
2. sıra için `|δ| ≥ 0,0497` gerektiriyordu — garanti yok, hata payı `±0,016`.

Bu oturumda **ölçülmüş** bir alternatif kuruldu: mevcut hattı geri-testte döven
yeni bir model. Yeni planın eşiği yok; **başabaş noktası `1,06482`**, yani yeni
model bundan iyi skor aldığı sürece harman kazandırıyor (mevcut skorumuz `1,00553`).

---

## 2. Kapanan eksen: soğuk × ölü (hata kütlesinin ~%48'i)

Geri-testte (kesim 2025-10-31) ölçüldü:

```
dilim            satir   %satir   RMSLE   %KUTLE
SOGUK x OLU      7.720    1,9%   6,2292   48,6%
SOGUK x DIRI    88.180   21,5%   1,1615   19,3%
SICAK x OLU      9.397    2,3%   0,4356    0,3%
SICAK x DIRI   305.167   74,3%   0,8005   31,8%
```

Geçmişi olmayan ve hedef pencerede baştan sona **tam sıfır** tüketen trafolar
soğuk trafoların %7,5'i, ama toplam hatanın yarısı. Bilinseler RMSLE 1,2249 → 0,878.

**Sıkı kurulumda ölçüldü ve KAPANDI.** Kesimler tam 4 ay ayrık (hedef-satır
kesişimi kanıtlanmış 0), permütasyon boş dağılımına karşı:

```
ozellik ailesi          AUC 10-31   AUC 11-30
VARLIK DESENI              0,551      0,514
DALGA/KOHORT               0,477      0,581
MEKAN                      0,632      0,543
GUC                        0,498      0,522
KIMLIK/ID                  0,516      0,577
HEPSI                      0,642      0,538
karistirilmis etiket    0,528+-0,079  0,510+-0,043
```

Hiçbiri boş dağılımdan ayrışmıyor (|z| < 1,7). Bağımsız permütasyon testinde
gerçek etiketle AUC 0,348 vs karıştırılmışta 0,522±0,081 → **z = −2,14**.

Ekonomik eşik de kapalı: diri trafoyu sıfırlamanın bedeli 48,1-49,3, ölüyü
kaçırmanınki 26,1-27,7 → başabaş olasılık **0,64**; hiçbir trafonun `p_ölü`'sü
0,5'i bile geçmiyor. Büzme de boş: güç-grubu tabanının test-optimal çarpanı
`k* = 1,0013–1,0286`, yani taban zaten ölçek-optimal.

**`docs/52`'nin hükmü doğrulandı. Bu eksene bir daha hak harcanmayacak.**

---

## 3. Açılan eksen: ileri-pencere doğrudan tahmin

### 3.1 Mimari farkı

Mevcut hat bir **panel modeli**: özet penceresinden özellik çıkarıp blok etiketlerini
tahmin ediyor. Yeni model **doğrudan tahmin**: kesim tarihinde geçmişten özellik
çıkarıp `kesim+1 .. kesim+4ay` penceresini tahmin etmeyi öğreniyor — testin
gerçek problemi bu. 10 aylık kesim (2025-03-31 … 2025-12-31) eğitim örneği veriyor.

`experiments/model29/m30_ozellik.py` (özellikler), `m33_durust.py` (tezgâh),
`m46_nihai.py` (üretim).

### 3.2 İKİ SIZINTI bulundu ve kapatıldı

1. **Pencere çakışması.** Eğitim kesimlerinin hedef pencereleri doğrulama
   penceresini kesiyordu → aynı `(trafo, gün)` satırı hem eğitimde hem doğrulamada.
   İlk ölçüm 0,8889 dedi; çakışma kapatılınca **0,9621**. Sızıntı 0,073 değerindeymiş.
   Çözüm: eğitim kesiminin hedef penceresi doğrulama kesiminde kesiliyor (`tavan=`).
2. **ID ezberi.** (1) kapatılınca kendiliğinden kapanıyor: doğrulamada soğuk olan
   trafonun tanım gereği kesimden önce satırı yok. Ayrıca ölçüldü — `idnum`'ı
   çıkarmak **kötüleştiriyor** (0,9621 → 0,9746), yani gerçek sinyal de taşıyor.

### 3.3 Ölçüm — aynı pencere, aynı soğuk payı (%13,9), birebir karşılaştırılabilir

`kis26` = 2025-12-01 … 2026-03-31 (geri-testte kesim 2025-11-30):

| | sıcak | soğuk | test-karışımı |
|---|---|---|---|
| `v83` (gönderilen zincirin atası) | 0,77826 | 1,90610 | 1,1140 |
| üretim hattının en iyi CV yığını | 0,76150 | 1,86720 | 1,1063 |
| yeni, L2 kaybı | 0,7176 | 1,8825 | 1,0891 |
| yeni, Huber(α=2, λ=20) | 0,6875 | 1,8261 | 1,0499\* |
| **yeni, Huber+L1 harmanı** | — | — | **1,0416\*** |

`*` seviye yanlılığı giderilmiş. **−%5,8.** Kazancın tamamı sıcak taraftan.

### 3.4 Ölçülen üç gerçek

1. **Sağlam kayıp L2'yi döver.** İki doğrulama kesiminde de: L2 ort 1,1730 →
   Huber 1,1539. Hedefin log uzayındaki kuyrukları eğitimde gürültü, genellemiyor.
   L1 tek kesimde daha iyi (1,0486) ama diğerinde düşüyor (1,2722) → **tek başına alma**.
   `Huber+L1` ortalaması İKİ kesimde de kazanıyor (1,0416 / 1,2283) → **bu alındı**.
2. **CatBoost(MAE) kötü** (1,1930 ort) — üçlü harmanı da aşağı çekiyor. Alınmadı.
3. **Global seviye yanlılığı ÖNGÖRÜLEMEZ.** Geri-testte işaret dönüyor:
   kesim 2025-11-30'da `−0,1535`, 2025-09-30'da `+0,1735`. Değeri ~0,012 RMSLE
   ama yönü bilinemiyor.

### 3.5 Seviye çapası — `v102`'nin LB bilgisini ücretsiz devralma

`docs/52` §9: `v102`'nin sıcak çekirdekte **ortalama artığı TAM SIFIR** (LB κ\*=0,31075
ile ölçüldü) ve "global seviye" yönü elekte %99,07 kapsamla boş.
**Yani `v102`'nin seviyesi LB-optimum.**

Yeni modelin log seviyesi `v83`'ün üç rejiminde (soğuk / kuyruk / sıcak-çekirdek)
`v102`'ninkine kaydırıldı:

```
soguk    -0,3799      kuyruk   -0,1114      cekirdek -0,0833
```

Bu, öngörülemez seviye gürültüsünü siliyor ve LB'de ölçülmüş tek bilgiyi koruyor.

### 3.6 Mevsimsellik denetimi — model öğrenmiş

Sıcak trafoların son-7-gün seviyesine göre ay kayması:

```
ay   YENI MODEL    v102     2025 GERCEK
 4     -0,0171   -0,0810      +0,0073
 5     +0,0563   -0,0625      -0,0229
 6     +0,2848   +0,2045      +0,2948
 7     +0,4439   +0,4610      +0,6161
```

Şekil doğru. Temmuz her iki modelde de düşük kalıyor (0,44/0,46 vs 2025'te 0,62) —
2026 için ayrı bir yön olabilir, ölçülmedi, **bu planın parçası değil**.

---

## 4. 29 AĞUSTOS PLANI

Dosya: `submissions/tuketim_m3_hl1_capali.csv` — kapı denetiminden geçti
(714.688 satır, ID birebir, 0 NaN, 0 negatif, maks 139.941).

```
Q = ||yeni - v102||^2 / N = 0,122746
m0 = 1,00553^2 = 1,011091
BASABAS: yeni model 1,06482'ten iyi skor alirsa harman KAZANDIRIR
```

| HAK | ne | amaç |
|---|---|---|
| **1** | `tuketim_m3_hl1_capali.csv` gönder | yeni modelin skoru `S` ÖLÇÜLÜR |
| **2** | `python experiments/model29/m50_harman_coz.py tuketim_m3_hl1_capali.csv <S>` → çıkan dosyayı gönder | optimum `κ*` harmanı |
| **3** | yedek: ikinci model ailesi (üçüncü yön) ya da HAK2'nin doğrulaması | — |

`S` ölçülünce optimum:

```
      S      kappa*   optimum RMSLE
  0,95000   +0,9423     0,94979
  0,97000   +0,7859     0,96710
  0,99000   +0,6262     0,98130      <- 1. SIRA (lider 0,99138)
  1,00553   +0,5000     0,99015      <- 2. SIRA (esik 1,00078)
  1,02000   +0,3806     0,99665      <- 2. SIRA
  1,04000   +0,2128     1,00276      <- iyilesme, 3. sira
  1,06482    0,0000     1,00553      <- basabas
```

**`P41`/`P42` planına üstünlüğü:** o plan iki *ölçülmemiş* yönde `|δ| ≥ 0,0497`
gerektiriyordu ve hata payı `±0,016` idi. Bu plan geri-testte ölçülmüş bir modele
dayanıyor ve `1,06482`'ye kadar her sonuçta kazandırıyor.

**Dürüst kayıt:** geri-test kazancının LB'ye ne oranda taşınacağı ÖLÇÜLMEDİ.
`docs/52` §14.1'de yeni bir model ailesi bir kez LB'de `κ ≈ 0,0045` ölçtü.
Fark: o yön `Q = 0,0258` idi, bu `Q = 0,1227` (4,8 kat) ve bu model geri-testte
üretim hattını aynı pencerede %5,8 dövüyor. **Garanti değil; ama maliyeti sıfır**
(Kaggle en iyi public skoru tutar).

---

## 5. Yeni kalıcı kurallar

**32.** *İleri-pencere geri-testinde eğitim kesimlerinin hedef pencereleri
doğrulama penceresini KESMEMELİ.* Kesiyorsa aynı `(trafo, gün)` satırı iki
yerde birden olur; bu depoda 0,073 RMSLE değerinde sahte kazanç üretti.

**33.** *Bu veride sağlam kayıp (Huber/L1) L2'yi döver — RMSLE ile ölçülse bile.*
Log uzayındaki kuyruklar eğitimde genellemeyen gürültü. Ama tek kesimde seçme:
L1 bir kesimde kazanıp diğerinde kaybetti; iki kesimde de kazanan harman alındı.

**34.** *Global seviyeyi geri-testten öğrenme — işareti kesimden kesime dönüyor.*
Seviye yalnızca LB'de ölçülür; `v102`'nin rejim seviyeleri LB-kalibre olduğu için
yeni modeller ona çapalanır.

---

## 6. Sıcak taraf — ölçülmüş ama bu sürüme girmeyen bulgular

Ayrı bir kolda ölçüldü (`experiments/model29/m10_sicak_seviye.json`), GBM zaten
çoğunu kendi buluyor ama elle kurulacak bir sürüm için kayda geçiyor:

- **Pencere:** son-7-gün ortalaması son-28'i döver (4 kesim ort 0,8303 vs 0,8584);
  sıralama ufuktan bağımsız. Medyan her pencerede ortalamadan kötü.
- **Mevsimsellik en büyük kalem:** Mart sonuna göre trafo kayması
  Nis +0,007 / May −0,025 / Haz +0,302 / **Tem +0,625**. Trafo-bazlı mevsimsellik
  gürültü (yıl-üzeri korelasyon 0,146) → global/ilçe profili kullan.
- **Trend gürültü:** son-90-gün eğimi, optimum shrinkage λ ≈ 0. Kullanma.
- **Shrinkage gereksiz:** geçmişi ≥4 satır olan her kovada optimum ağırlık w=1,0;
  4 günlük kendi geçmişi bile grup ortalamasını yener.
- **Hata kütlesi aşırı yoğun:** en kötü %1 satır kütlenin %67'si; 2.809 trafodan
  en kötüsü 15'i %43'ü. Neredeyse hepsi tek kalıp: geçmişi tam sıfır olan trafo
  hedefte uyanıyor. Bu grupta **mükemmel sabit bile 2,14 RMSLE** — indirgenemez.

---

## 7. GECE EKI — hava verisi (28 Ağustos 18:00–21:00)

LB listesi yenilendi, **4. sıraya düştük**: `1. Grid Grinders 0.99064 · 2. Atakan Aldemir
1.00041 · 3. Şaban Özdoğan 1.00543 · 4. TasnifX 1.00553 · 5. Ahmet Çelik 1.00559`.
3.–5. sıra 0,00016 içinde; gerçek eşikler 2. için `−0,0051`, 1. için `−0,0149`.

### 7.1 Grup A denetimi — kapalı
Geçmişi baştan sona sıfır olan sıcak trafolar (test'te 25.566 satır). Geri-testte
modelin yanlılığı yalnız `+0,33` (2025-11-30: gerçek 0,2668 / model 0,5997) ve
**sabitle değiştirmek KAYBETTIRIYOR** (−0,014 … −0,024 her sabitte). Model bu
kohortu zaten doğru ele alıyor. Eksen kapalı.

### 7.2 Model çeşitliliği
İki kesimde de kazananlar: yakınlık ağırlığı `1,15^i` (−0,0053 / −0,0006),
havuz+rejim-uzmanı ortalaması (−0,0029 / −0,0050), `lr=0,02` (−0,0013 / −0,0015).

### 7.3 HAVA VE DIŞ VERİ — büyük kazanç
`ilce_key` eşleme **%100** (train ve test), test döneminin **47 ilçe × 122 gün**
tamamı kapsanıyor, NaN yok. Aile bazında (huber tek başına, taban 1,0637 / 1,2440):

```
aile                                 11-30      09-30    hukum
A  sicaklik CDD/HDD + 3/7/14g ort   +0,0183   +0,0290   ALINDI
C  nem / toprak / yagis             +0,0109   +0,0261   ALINDI
G  trafo x sicaklik duyarliligi     +0,0066   +0,0210   ALINDI
E  turizm / su                      +0,0018   +0,0098   ALINDI
B  gunes / gun uzunlugu             +0,0001   +0,0156   (A ile ortusuyor)
D  statik ilce (arazi/OSM)          -0,0007   -0,0034   ELENDI
```

`huber+l1` harmanında, test-karışımı (çapalı):

| konfig | 2025-11-30 | 2025-09-30 | ort |
|---|---|---|---|
| havasız düz (`m3`) | 1,0415 | 1,2285 | 1,1350 |
| havasız + yakınlık | 1,0364 | 1,2281 | 1,1323 |
| **havalı düz (`m4`)** | **1,0359** | **1,1995** | **1,1177** |
| havalı + yakınlık | 1,0323 | 1,2016 | 1,1169 |

Hava İKİ kesimde de kesin kazanıyor. **Yakınlık ağırlığı hava girince KARIŞIK**
(−0,0036 / +0,0021) → kural 33 gereği alınmadı. Üretim: **havalı düz**.

Üretim hattının `kis26` kaydına göre: **1,1063 → 1,0359 = −%6,4.**

### 7.4 Neden hava bu kadar önemli — 2026 yazı DAHA SERİN
```
ay   2025 sic  2026 sic    fark   CDD22 orani (2026/2025)
 5     19,79     17,92    -1,87        0,17
 6     26,52     24,77    -1,74        0,65
 7     29,85     27,59    -2,26        0,72
```
Test penceresi klima sezonu ve 2026'nın soğutma yükü 2025'in %65–72'si.
Bunun modele yansıması ölçüldü — son-7-gün seviyesine göre ay kayması:
```
ay     v102       m3    m4 HAVA   2025 GERCEK
 4   -0,0810  -0,0915   -0,0324     +0,0073
 5   -0,0625  -0,0314   -0,0162     -0,0229
 6   +0,2045  +0,1781   +0,1328     +0,2948
 7   +0,4610  +0,3339   +0,3161     +0,6161
```
`m4`'ün Haziran/Temmuz'da daha ılımlı kalması **kusur değil**, gerçekleşmiş havanın
sonucu. Ayrıca `m4`'ün seviyesi `v102`'ninkine doğal olarak çok daha yakın çıktı
(çekirdek çapası `−0,0833` yerine yalnız `+0,0167`) — bağımsız bir tutarlılık işareti.

### 7.5 GÜNCEL PLAN — `m4` birincil aday

Dosya: `submissions/tuketim_m4_hava_capali.csv` (714.688 satır, ID birebir,
0 NaN, 0 negatif, maks 142.376, 133 öznitelik).

```
Q(m4, v102) = 0,121581      BASABAS 1,06427
      S      kappa*   optimum RMSLE
  0,95000   +0,9466     0,94982
  0,97000   +0,7887     0,96720
  0,99000   +0,6274     0,98144      <- 1. SIRA
  1,00553   +0,5000     0,99030      <- 2. SIRA
  1,02000   +0,3795     0,99679      <- 2. SIRA
  1,04000   +0,2100     1,00286
```

| HAK | ne |
|---|---|
| **1** | `tuketim_m4_hava_capali.csv` → `S` ölçülür |
| **2** | `python experiments/model29/m50_harman_coz.py tuketim_m4_hava_capali.csv <S>` |
| **3** | yedek üçüncü yön: `tuketim_m3_hl1_capali.csv` (`Q(m4,m3)=0,0193`, korelasyon 0,997) |

### 7.6 KAPATILAN BOSLUK — hava ozellikleri YAZ penceresinde de dogrulandi

§7.3'un iki dogrulama kesimi (11-30 -> Ara-Mar, 09-30 -> Eki-Oca) **soğutma
derece-günü sıfır** olan pencereler. Yani oradaki hava kazancı neredeyse tamamen
ISITMA'dan geliyordu; test penceresi (Nis-Tem) ise SOĞUTMA hakimiyetinde.
Bu, ölçülmemiş bir varsayımdı. Ölçüldü:

```
kesim         hedef      ort CDD22   egitim satir   HAVASIZ   HAVALI   KAZANC
2025-05-31    Haz-Eyl       5,487        198.572    1,1727    1,1617   +0,0110
2025-06-30    Tem-Eki       4,326        402.500    1,1251    1,1002   +0,0249
2025-11-30    Ara-Mar       0,000      1.978.830    1,0416    1,0359   +0,0057
```

**Hava kazancı soğutma pencerelerinde 2-4 kat DAHA BÜYÜK**, üstelik oralarda
eğitim verisi 5-10 kat daha az. Endişe ters yöndeydi ve ölçüm `m4`'un lehine çıktı.

**Kalıcı kural 35.** *Bir özellik ailesinin kazancını, o ailenin mekanizmasının
AKTİF olduğu bir pencerede doğrula.* Hava özelliklerini CDD=0 olan kış
pencerelerinde ölçüp yaz testine genellemek, ölçüm değil varsayımdı.
