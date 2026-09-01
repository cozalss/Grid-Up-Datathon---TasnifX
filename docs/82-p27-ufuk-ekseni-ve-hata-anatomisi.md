# 82 — p27: UFUK EKSENİ, SEVİYE KAYMASI VE HATA ANATOMİSİ

Tarih: 1 Eylül 2026. Ölçüm çıktısı: `experiments/model29/p_kalici/p27_ufuk_anatomi.json`
Betikler: `p_kalici/p27_*.py`. Tezgâh: **ÜRETİM harmanı** (`p27_ortak.py`) —
sıcak cat3/xgb1/lgbm1/sinir_agi1.4, soğuk cat-tekil, son işlem β=0.60.
Bütün bileşikler test kohortuna (soğuk %22,16) ağırlıklı.

---

## 0. TEK CÜMLE

**Ufuk, SAÇILIMI büyütür (indirgenemez); YANLILIĞI ise blokların eğitiminde
bulunmayan takvim ayları yaratır — ve testte o eksiklik YOK.** Bu yüzden ufuk /
seviye ekseni gerçek bir fırsat değil, CV tezgâhının bir yapaylığıdır.

Önceki agent'ın "ufuk ekseninde BÜYÜK SİNYAL" notu **doğru ölçümdü, yanlış
yorumdu.** Sinyal gerçek ama testte karşılığı yok.

---

## 1. UFUK EĞRİSİ — ölçüm

Ham RMSLE ufukla her blokta güçlü büyüyor (sıcak):

| blok | 1-15 | 46-60 | 106-122 |
|---|---|---|---|
| yaz25 | 0,433 | 0,704 | 1,095 |
| guz25 | 0,585 | 0,741 | 1,005 |
| kis26 | 0,613 | 0,743 | 0,847 |

Ama **yanlılık/saçılım ayrışımı üç blokta ÜÇ FARKLI şey söylüyor** (sıcak,
ortalama log artık):

| blok | 1-15 | 46-60 | 106-122 | desen |
|---|---|---|---|---|
| yaz25 | **+0,018** | +0,015 | **+0,444** | tekdüze ARTAN |
| guz25 | −0,270 | −0,280 | −0,354 | DÜZ (sabit −0,30 ofset) |
| kis26 | **+0,196** | +0,179 | **+0,092** | tekdüze AZALAN |

Ortak bir "ufuk yasası" **yok**. Yanlılığın MSE payı yaz25'te %16,4'e çıkarken
kis26'da %1,2'ye düşüyor.

---

## 2. TEŞHİS — sebep ufuk değil, TAKVİM KAPSAMI

`tuketim_model.py: kokenleri_ayikla` hedef blokla kesişen **her** kökeni atıyor.
Sonuç: her doğrulama bloğu kendi hedef aylarını eğitimde göremiyor.

| blok | hedef ayları | eğitimde görülmüş mü |
|---|---|---|
| yaz25 | 4,5,6,7 | hiçbiri |
| guz25 | 8,9,10,11 | hiçbiri |
| kis26 | 12,1,2,3 | **2 ve 3 GÖRÜLÜYOR** (sub25 kökeni) |
| **TEST** | **4,5,6,7** | **DÖRDÜ DE GÖRÜLÜYOR** (yaz25 + bah25 kökenleri) |

### kis26 doğal deneyi (aynı blok, aynı model) — belirleyici

| altküme | ort. ufuk | sıcak yanlılık | sıcak RMSE |
|---|---|---|---|
| 12,01 GÖRÜLMEYEN | **32,2 gün** | **+0,205** | 0,686 |
| 02,03 GÖRÜLEN | **92,4 gün** | **+0,107** | 0,801 |

Ufuk üç kat daha uzun olmasına rağmen yanlılık **yarıya iniyor**. Saçılım ise
(RMSE) ufukla büyümeye devam ediyor. İki eksen böylece ayrışıyor:
**kapsam → yanlılık, ufuk → saçılım.**

yaz25 aynı mekanizmayı doğruluyor: Nisan (Mart'a komşu, −0,02) → Temmuz
(görülmemiş yazın en derini, **+0,40**). Model yazı hiç görmediği için yaz
tepesini üretemiyor.

**Üretim modeli 12 ayın hepsini etiket olarak görüyor. Bu yanlılık kaynağı
testte yok.**

---

## 3. KÂHİN SEVİYE TAVANI — büyük ama ULAŞILAMAZ

Kova başına artık ortalamasını sıfırlamak (rejim ayrı):

| düzeltme | CV bileşik | LB ölçekli | LB kazanç |
|---|---|---|---|
| yok (taban) | 1,0669 | 1,0019 | — |
| global sabit | 1,0454 | 0,9817 | +0,0195 |
| ufuk 8 kova | 1,0342 | 0,9712 | +0,0300 |
| takvim ayı | 1,0355 | 0,9724 | +0,0287 |
| **gün (122)** | **1,0323** | **0,9694** | **+0,0317** |
| gün × haftanın günü | 1,0323 | 0,9694 | +0,0317 (ek yapı YOK) |

Tavan birinciliği geçmeye yeterdi. **Ama kazancın çoğu blok-düzeyi SABİT
ofsette** (guz25'te +0,0399 / +0,0418 = %95). O sabit de bloktan bloğa işaret
değiştiriyor (yaz25 +0,15, guz25 −0,30, kis26 +0,17).

### Dürüst blok-dışı tahminciler: **27/27 NEGATİF**

| tahminci | sonuç |
|---|---|
| ufuk kovası yanlılığı, blok→blok (9 eşleme) | **9/9 negatif** (−0,005 … −0,087) |
| global sabit kayma, diğer iki bloktan | **3/3 negatif** (−0,010 … −0,044) |
| ulusal seri regresyonu (5 model × 3 blok) | **15/15 negatif** (−0,005 … −0,125) |

**p11'in 3/3 kaybının açıklaması budur:** blok yanlılıkları işaret olarak zıt,
başka bloğun eğrisini ödünç almak hatayı kabaca ikiye katlıyor.

### Ulusal çapa — denendi, kapalı

Testin **gerçek günlük ulusal tüketimi** öznitelik olarak VERİLMİŞ (122 gün,
NaN yok). Umut verici görünüyordu. Ama gün-düzeyi yanlılığın ulusal seriyle
korelasyonu bloklar arası tutarsız (kor(sıcaklık): yaz25 **+0,83**, kis26
**−0,58**). Blok-dışı sınav 15/15 negatif.

Not: gönderim dosyasının seviye katmanı zaten **LB'den** çözülmüş (m111 κ,
span a0+r_hat). CV seviyeyi kestiremediğine göre bu doğru yaklaşımdır.

---

## 4. SIFIR CEBİ — belgelerdeki teşhis EKSİKMİŞ, ama sonuç yine kapalı

### Yeni olgu
Blok sıfırlarının **%68-84'ü, 4 ay boyunca HİÇ üretmeyen trafolardan** geliyor
(186 / 139 / 178 trafo). Yani asıl problem satır değil **trafo** düzeyinde.

### Belgelerdeki "AUC 0,58-0,61" yanıltıcı
O rakam **yalnız soğuk** altkümesi için geçerli. Blok-dışı gerçek AUC'ler:

| blok | trafo düzeyi AUC | satır AUC | satır AUC (soğuk) |
|---|---|---|---|
| yaz25 | 0,972 | 0,988 | 0,700 |
| guz25 | 0,922 | 0,966 | 0,541 |
| kis26 | 0,956 | 0,961 | 0,464 |

### Yine de hiçbir uygulama kazanmıyor — ve sebebi kesin
Optimal yumuşak büzme `(1−q)^γ`, sert eşik, kapılı büzme — **16 varyantın hepsi
0/3 veya 1/3**. Sebep:

**Üretim modeli bu trafolara zaten ortalama log 2,2-3,9 tahmin veriyor**
(canlı trafolarda 6,5-6,8) — yani ~100 kat düşük. Doğru yakalananları
sıfırlamanın kâhin değeri sadece **+0,0005 … +0,0052**.

Kâhin +0,11…+0,25'in **tamamı sınıflandırıcının KAÇIRDIĞI trafolarda** — ve
onlar aynı özniteliklerden görünmüyor (guz25'te yanlış pozitiflerin gerçek
log tüketimi ortalama 4,5-4,9, yani canlı ve büyük; onları sıfırlamak yıkıcı).

**Ulaşılabilir artık kazanç yok.** Kullanıcının saydığı üç fikir de kapanıyor:
trafo düzeyi "hiç çalışmıyor" tespiti = ölçüldü, kazanç yok; kVA×ilçe sıfır
oranı önsel = zaten `t_sifir_orani`/`g_*` içinde; komşu eşzamanlılığı = test'te
komşunun aynı günü de bilinmiyor, öznitelik olarak mevcut değil.

---

## 5. İZO-EĞRİ

Ölçek: CV bileşik 1,0661 → LB 1,00115 (katsayı 0,93907).
LB ölçeğinde sıcak **0,7346**, soğuk **1,6209**.

| hedef | yalnız sıcak | yalnız soğuk | orantılı |
|---|---|---|---|
| **1. sıra 0,98038** | 0,6977 (**−%5,02**) | 1,5626 (**−%3,60**) | −%2,07 |
| **3. sıra 0,99556** | 0,7248 (−%1,34) | 1,6053 (−%0,96) | −%0,56 |

### Çıkarım: birinci sıfırları ÇÖZMEMİŞ
Kâhin sıfır dedektörü bileşiği **0,7223**'e indirirdi. Birinci 0,98038'de.
Yani birinci ne sıfırları ne seviyeyi çözmüş — sadece biraz daha iyi bir taban
modeli var. **Kaçırdığımız tek büyük yapısal numara yok.** Bu, bütün
eksenlerin kapalı çıkmasıyla tutarlı.

---

## 6. HATA CEPLERİ (kohort ağırlıklı MSE payı, blok ortalaması)

| cep | MSE payı | o cep mükemmel olsa CV |
|---|---|---|
| soğuk TÜM | %54-63 | 0,66-0,71 |
| sıfır TÜM | %41-53 | 0,74-0,80 |
| — sıfır/soğuk | %31-41 | 0,83-0,87 |
| — sıfır/sıcak | %11-12 | 1,01-1,02 |
| büyük y≥100 | %37-47 | 0,75-0,86 |
| soğuk pozitif | %19-24 | 0,91-0,97 |
| küçük y<10 | %4-5 | 1,02-1,06 |

**Yoğunlaşma aşırı:** trafoların üst %1'i MSE'nin **%38-49'unu**, üst %5'i
**%66-71'ini** taşıyor. (Bu ekseni p09 kapatmıştı: bloklar arası trafo sapma
korelasyonu işaret değiştiriyor.)

---

## 7. YAN BULGU — CV tezgâhı üretimi SİSTEMATİK OLARAK KÖTÜMSER

docs/80 §3 tezgâhın üretim harmanını ölçmediğini göstermişti. Buna **ikinci,
bağımsız bir tezgâh–üretim farkı** ekleniyor:

Her doğrulama bloğu kendi takvim aylarından yoksun bırakılıyor; üretim modeli
ise yoksun değil. Ölçülen yanlılık bütçesi (blok başına +0,016…+0,042 bileşik)
üretimde **büyük ölçüde yok**. Bu, "CV 1,0661 → LB 1,00115" ölçek farkının
(0,939) bir bölümünü açıklıyor.

**Pratik sonuç:** CV tezgâhında mevsimsel dışdeğerlemeyi iyileştiren her şey
LB'de karşılıksız kalır. Aday seçerken bu yanlılığa dikkat.

*(Bu bir mekanizma açıklaması; kis26 doğal deneyi güçlü kanıt ama tek blokluk
bir doğal deney. Spekülatif olan kısım: ölçek farkının NE KADARININ bundan
geldiği — onu ölçmedim.)*

---

## 8. SIRALI FİKİR LİSTESİ

### Kapalı (bu turda ölçüldü — TEKRAR AÇMA)
| eksen | kanıt |
|---|---|
| ufuk kovası seviye kalibrasyonu | 9/9 negatif blok-dışı |
| global/kohort sabit kayma | 3/3 negatif |
| ulusal seri günlük çapa | 15/15 negatif |
| trafo düzeyi ölü tespiti (yumuşak/sert) | 16 varyant, 0-1/3 |
| satır düzeyi sıfır büzme | 0/3 |

### Açık, sıralı (24 saatte ölçülebilirliğe göre)

**1. docs/80 §9 sıcak taraf taraması — TEK GERÇEK ADAY.**
Yarım kalmış: cat τ=480 (yaz25 −0,0098 ham, **üretim ağırlığıyla −0,0046**),
lgbm huber (−0,0055 üretim), xgb huber α=2 (−0,0012 üretim). Üçü bağımsızsa
toplam ~−0,011 CV; taşıma 0,5 ile **~−0,005 LB** → 3. sıra sınırı.
*Ölçülebilirlik: 3 blok × 3 tohum ızgara, mevcut tezgâhla birkaç saat.*
*Bu benim bulgum değil, devir belgesinden geliyor; en yüksek beklenen değer bu.*

**2. `sinir_agi`'nın ölçülmemiş olması (docs/80 §8.2).**
Sıcak harmanın **%21,9'u** ve hiçbir ölçümde yok. Boru hattındaki en büyük
ölçülmemiş nesne. *24 saatte tek blok tek tohum ölçülebilir (~20 dk/fit) ama
karar verecek ızgara sığmaz. Ağır eğitim yasağı kapsamında.*

**3. Sıcak önbellek köken sorunu (docs/80 §8.1).**
Bir fikir değil bir **geçerlilik açığı**: sıcak tezgâh üretimi birebir
üretmiyor (maxabs 0,325). Sıcak taraftan çıkacak her hüküm bu şüpheyle
maluldür. 1. maddeye başlamadan önce bunun büyüklüğü bilinmeli.

### Yeni büyük fikir bulunamadı — dürüst hüküm
Bana verilen iki eksen de (ufuk/seviye, hata anatomisi) kapandı. İzo-eğri
birincinin de sıfırları/seviyeyi çözmediğini gösteriyor. Elde kalan tek
gerçekçi yol, **docs/80'in hazır adayı + §9'un yarım kalan sıcak taraması**.
