# Yarışma kuralları — API'den doğrulandı (22 Ağustos 2026, gece)

[00-yarisma-brief](00-yarisma-brief.md) §"Bilinmeyenler" 10 açık soru listeliyordu.
Bu belge, Kaggle API'sinden **ölçülerek** kapatılanları ve hâlâ açık kalanları
ayırır. Kaynak: `GET /api/v1/competitions/list?group=entered` (kimlikli).

---

## 1. Kapanan sorular — hepsi API çıktısı

| alan | değer | brief sorusu |
|---|---|---|
| `evaluationMetric` | **Mean Columnwise RMSLE** (`mcrmsle`) | 2 |
| `maxDailySubmissions` | **3** | 4 |
| `maxTeamSize` | **4** | — |
| `deadline` | **2026-09-01T23:59:00Z** | — |
| `mergerDeadline` | 2026-09-01T23:59:00Z | 7 |
| `isKernelsSubmissionsOnly` | **False** — CSV gönderimi serbest | 10 |
| `submissionsDisabled` | False | — |
| `teamCount` | 263 | — |
| `userRank` | **4** | — |
| `category` | Community | — |

**Metrik notu:** MCRMSLE = her hedef kolonun RMSLE'sinin ortalaması.
`sample_submission.csv` **tek** hedef kolon içeriyor (`tuketim`), dolayısıyla
MCRMSLE burada **düz RMSLE'ye eşit**. Yaptığımız her ölçüm geçerli.

---

## 2. Kapanmayan iki soru — giriş duvarı

`overview/evaluation` sayfası hem `WebFetch` hem oturumsuz tarayıcıda **HTTP 404**
veriyor: Community yarışması, sayfa yalnızca katılmış ve **giriş yapmış** hesaba
görünüyor. API bu iki alanı yayımlamıyor.

| soru | nerede yazıyor | neden önemli |
|---|---|---|
| **Public/private yüzdesi** | Overview → Evaluation | gördüğümüz her skor public; sıralama private'la |
| **Kaç final seçilebilir** | Submissions ekranı, "Select for final" | 2 ise birbirine benzemeyen iki dosya seçilmeli |

İkisi de tarayıcıdan 30 saniyede okunur. Okununca bu tablo güncellenecek.

---

## 3. Üç düzeltme — mevcut belgelerdeki hatalar

### (a) Sıralama: 3. değil **4.**

[29-durum](29-durum-2026-08-22-gece.md) §1 dört takım listeliyor. Aynı gece 18:55'te
bir takım daha geçmiş. API `userRank = 4` bunu bağımsız olarak doğruluyor.

```
1. Saliha Rana Uzun      1,03170
2. Uğur Çelik            1,03257   <- 22 Agustos 18:55, belgede YOK
3. Sadettin Şamil Verdil 1,03330
4. TasnifX (BIZ)         1,03370
5. Seyit kaan Gunes      1,03433
6. Kanzi                 1,03770
```

Beş takım **0,0026** içinde. Alan hızlı hareket ediyor.

### (b) Bitiş saati UTC — elimizde 3 saat daha var

`deadline` **UTC**. Türkiye saatiyle **2 Eylül 02:59**, 1 Eylül 23:59 değil.

Aynı ofset gönderim kotasını da açıklıyor: kota 00:00 UTC'de sıfırlanıyor, bu da
TR saatiyle **03:00** demek — [29-durum](29-durum-2026-08-22-gece.md) §1'deki
gözlemle birebir uyuşuyor. İki bağımsız gözlem aynı ofseti veriyor.

Pratik sonuç: 1 Eylül 03:00'te açılan 3 hak, **2 Eylül 02:59'a kadar** kullanılabilir.

### (c) Takım kurma son tarihi — iki farklı tarih var

```
coderspace.io etkinlik sayfasi : 24 Agustos 2026, 23:59
Kaggle mergerDeadline          :  1 Eylul  2026, 23:59 UTC
```

Bunlar çelişmiyor, **farklı şeyler**: Kaggle'ın teknik sınırı 1 Eylül, düzenleyicinin
idari kuralı 24 Ağustos. Bağlayıcı olan **dar olanı** — 24 Ağustos, yani **2 gün**.

> **AÇIK İŞ:** Takım Kaggle'da fiilen birleşmiş mi? Leaderboard'da `TasnifX` tek satır
> gösteriyor ama bu, birleşmiş takımla tek kişilik takımı ayırt etmiyor. Birleşme
> gerekiyorsa 24 Ağustos'tan önce yapılmalı; sonrası diskalifiye sebebi olarak
> listelenmiş ([00-yarisma-brief](00-yarisma-brief.md) §Diskalifiye riskleri).

---

## 4. Private leaderboard — sıralamayı bu belirliyor

`kaggle competitions submissions -v` çıktısında `privateScore` sütunu **her satırda
boş**. Yani gördüğümüz 1,03370 **public** skor; private yarışma bitince açılacak.

Bu yeni bir bilgi değil, brief zaten yazmış:

> **1–4 Eylül** — *Private leaderboard ilk 20'nin* notebook'ları toplanır

Yani üç kapının birincisi private LB. Public LB yalnızca bir göstergedir.

**Buna karşı konumumuz iyi:** model seçimi **CV ile** yapıldı; LB yalnızca
kalibrasyon çapası olarak, iki noktada kullanıldı. Public LB'ye uydurma yapılmadı.

Gerçek aşırı-uydurma riski LB'de değil **CV'de**: [29-durum](29-durum-2026-08-22-gece.md)
§9 kendi uyarısını koymuş — *"40+ yapılandırma aynı üç blokta seçildi."*

---

## 5. Gizli bölünme yarınki ölçümleri bozar mı — hayır

Endişe: public testin bir dilimiyse, 0,0025'lik farklar gürültüde kaybolur mu?
Aday dosyalar arasındaki fark ölçüldü:

```
v25 vs v23  farkli satir :   8.748  (%1,22)   -> hedge sorusu (B)
v25 vs v26  farkli satir : 162.016 (%22,7)    -> taban sorusu (A)
```

Public %30 olsa bile hedge ~2.600, taban ~48.000 satırda ölçülür. Rastgele bölünme
oranı koruduğu için **beklenen etki değişmez**, yalnızca varyans artar — ve her iki
karşılaştırma da beklenen etkinin kat kat üstünde sinyal taşıyor.

Ek kanıt: kalibrasyon iki gönderimde **0,0006** sapmayla tuttu. Public örnekleme
gürültüsü büyük olsaydı bu tutarlılık ortaya çıkmazdı.

---

## 6. Gönderim planı değişmiyor

Bu belgedeki hiçbir bulgu [29-durum](29-durum-2026-08-22-gece.md) §2'deki üç dosyayı
değiştirmiyor. Metrik beklendiği gibi, limit beklendiği gibi, ölçümler gürültünün
üstünde. Değişen tek şey: **rakip bir sıra daha yakın** ve final seçimi için
öğrenilmesi gereken bir soru daha var.
