# 79 — DURUM 31 Ağustos 16:15 (oturum duraklatıldı)

Kullanıcı PC'yi kapatıyor. Birkaç saat sonra devam. Bu belge, kaldığımız
yerden devam etmek için gereken HER ŞEYİ içerir.

## 1. Yarışma durumu

Bitiş: 1 Eylül 23:59 UTC (= 2 Eylül 02:59 TSS).
Haklar: 3 adet, 1 Eylül 00:00 UTC (= 03:00 TSS) açılıyor.
Bugünkü kota TÜKENDİ (D1 1.00177, D2 1.00159, Y1 1.00297 — hepsi sonda).

Liderlik tablosu (31 Ağu ~15:45 TSS, kaggle CLI ile çekildi):

| # | takım | skor |
|---|---|---|
| 1 | Grid Grinders | 0.98110 |
| 2 | Duo-Electra | 0.99536 |
| 3 | Abdülbaki Bayır | **0.99556** |
| 4 | Berke Kuç | 0.99648 (bugün 0.99927'den düştü) |
| 5 | shining stars | 0.99870 |
| 12 | **TasnifX** | **1.00115** |

Hedef: ilk 3. Gereken kazanç **0.00559**. Hedef HAREKET EDİYOR.

## 2. BU OTURUMUN ANA BULGUSU — küçük CV kazançları LB'ye TAŞINMIYOR

Taşıma oranı n=1'den n=11'e çıkarıldı ve zayıflatma (errors-in-variables)
itirazı ölçülüp ÇÜRÜTÜLDÜ (λ=0.930, düzeltme sadece %7).

Brüt eğim katmanlaması (27 dosya, ortak taban m6_ikiyon):

| CV kazancının büyüklüğü | n | eğim | R² |
|---|---|---|---|
| **abs(dCV) >= 0.02** | 8 | +2.28 | **0.986** |
| abs(dCV) < 0.02 | 19 | −0.47 | **0.036** |

**Bu, projenin bütün tarihini açıklıyor.** LB'yi 1.169 -> 1.001'e getiren
BÜYÜK değişikliklerdi (R²=0.986). Son iki haftadır uğraşılan 0.002-0.008'lik
ince kazançlar ölçülebilir biçimde HİÇ taşınmıyor (R²=0.036).

Bahsimiz 0.0082 -> taşınmayan bandın içinde. **P(tutar) = %3.5.**

Ek cebir düzeltmesi: `CARPAN=0.798` bir KORELASYON oranıdır, kazanç oranı
değil. CV'den seçilmiş ölçekte kazanç oranı `2C-1 = 0.596` ve bu da gereken
0.6817'nin ALTINDA. Projenin kendi n=1 çapası bile eşiği karşılamıyor.

Vekil yöntemi uyarısı: yeniden kurulan CV vekili, gerçek blok CV'sinin
bilindiği çiftlerde 4-12 sigma SİSTEMATİK sapıyor ve işaret bile ters
(v27->v30 gerçek −0.0095, vekil +0.0174). Vekil tabanlı sayılara az güven.

**STRATEJİK SONUÇ: küçük dikkatli bahisler ölçülmüş biçimde işe yaramıyor.
Tek şans, 0.02'yi AŞAN tek bir büyük CV kazancı.**

## 3. p06 (soğuk harman ağırlığı) — KIRMIZI TAKIM KIRDI

İddia: cat/xgb/lgbm eşit -> (0.05/0.35/0.60), +0.00752.
Düzeltilmiş: **+0.0031** (en iyimser) ile **+0.0013** arası. P(kazanç<=0) %29-39.

Kırılma sebepleri:
1. **Kohort uyuşmazlığı (ana kusur).** Test soğuklarının %82.3'ü (75,90] gün
   penceresi kovasında; yaz25'in payı %11.8. Pencere yapısında teste TV
   uzaklığı: yaz25 0.705 (EN KÖTÜ), guz25 0.519, kis26 0.621.
   Test dağılımına ağırlıklandırınca: yaz25 +0.0232 -> +0.0112,
   guz25 +0.0111 -> **−0.0012 (işaret döndü)**, kis26 −0.062 -> −0.043.
2. **Seçim gürültüsü.** 231 noktalı ızgara, guz25 yüzeyi DÜZ (en iyi 20 nokta
   0.001 aralığında). Önyüklemede (0.05/0.35/0.60) yalnız %1.5 tekrar seçiliyor.
3. **Sahte blok-dışı doğrulama.** p06'nın kendi sızıntısız bloğu [0.75,0,0.25]
   veriyor ve yaz25'te tabandan 0.067 KÖTÜ; p06 bunu bırakıp yalnız guz25'e
   dönmüş — yani dış bloklardan HEMFİKİR OLANI seçmiş.

KIRILAMAYANLAR: aritmetik doğru (MSE uzayında ağırlıklandırma), delta dosyası
bit-birebir yeniden üretildi, soğuk maske bağımsız kurulunca birebir aynı
(158.369 satır / 2024 trafo), kırpma etkisi sıfır, aşırı uyum YOK (kazanç ince
ayardan değil YÖNden geliyor: cat'ı düşür).

## 4. p08 (ölü trafo) — KABUL, ama küçük

Kural: son 30 günde max<=0 VE geçmiş sıfır oranı >=%99 VE kesintisiz sıfır
serisi >=30 gün -> tahmini SIFIRLAMA, **x0.25 (veya x0.50) ile küçült**.

| blok | satır | kesinlik | ort gerçek | dMSE |
|---|---|---|---|---|
| yaz25 | 11.770 | %96.76 | 65.4 | −0.00174 |
| guz25 | 3.360 | %95.60 | 79.2 | −0.00100 |
| kis26 | 5.502 | %97.69 | 19.6 | −0.00127 |

Üç blokta da kazanıyor, blok-dışı seçim üç katmanda da aynı kurala iniyor.
Test'te 135 trafo / 15.533 satır (%2.17), HEPSİ SICAK. LB karşılığı −0.0005..−0.0009.
YUMUŞATMA ŞART: tam sıfırlamada P(kazanç)=0.72; x0.25'te 0.956; x0.50'de 0.996.
NOT: 0.0008 < 0.02, yani taşınmayan bandın içinde.

## 5. KAPANAN YOLLAR (tekrar açma)

- p09 trafo bazlı KALICI sapma: bloklar arası korelasyon işaret değiştiriyor
  (guz->kis +0.356, kis->yaz −0.048, yaz->guz +0.082; tavan 0.99). 9 dürüst
  blok-dışı seçimin hiçbiri anlamlı pozitif değil.
- p09b mevsim eşleşmeli 12-ay transferi: ÖLÇÜLEMEZ. Artık verisi tam 12 ay
  (2025-04..2026-03); azami gecikme 11 ay. Her mevsimden TEK örnek var.
  Ayrıca mekanizma zaten modelde: `t_gy_log_ort` (365 gün öncesi, kapsam
  yaz25 YOK / guz25 YOK / kis26 kısmi / **TEST TAM**), `t_mevsim_genlik`
  (yaz−kış farkı), `t_ay_sapma`. Elle ofset = ÇİFT SAYMA.
  YAPI GERÇEK ama mevsimlik: Gördes +0.96, Ödemiş +0.67, Tire +0.57 vs
  Foça −0.30, Bergama −0.22, Urla −0.20 (sulama coğrafyası). Manisa +0.28
  vs İzmir +0.06. Küçük trafolar (guc Q1) +0.30.
- p11 fikir5 ufuk/ofset kalibrasyonu: blok-dışı 3/3 KAYIP.
- p11 fikir4 komşu havuzu harmanı: çapa tek başına 1.70+ (model 1.41), 3/3 KAYIP.
- p11 fikir3 yalnız-statik soğuk model: KONUSUZ — 33 `t_*` kolonunun HEPSİ
  soğukta zaten tam NaN. Soğuk uzman ZATEN yalnız-statik.
- p11 fikir2 soğuk kohorta tek skaler kayma: ham 3/3 pozitif ama TEST
  dağılımına ağırlıklandırınca (75,90] kovasında −0.0047. RET.
- p11 aile ağırlığının yeniden seçimi: üç ölçütte de blok-dışı KAYBETTİRİYOR.
- Daha önce kapananlar: p01 kalibrasyon (19/20 kayıp), p02 sıfırdan taban
  (0.946 vs 0.867), p04 alan bilgisi (tavan +0.0012), p05 iki aşamalı ayrışma
  (7/7 negatif), p06'nın 5 fikri.

## 6. AÇIK KALAN TEK YOL — huber soğuk kayıp fonksiyonu

Soğuk lgbm, kayıp fonksiyonu taraması (`p_kalici/p11_b_lgbm.json`):

| aday | yaz25 | guz25 |
|---|---|---|
| TABAN (üretim) | 1.42095 | 1.62332 |
| **huber a=1.0** | **1.36304** | 1.58673 |
| huber a=2.0 | — | **1.57750** |
| huber a=0.5 | — | 1.67111 |
| huber a=0.2 | — | 1.72978 |
| fair c=1 | — | 1.61639 |
| l1 | 1.41739 | — |
| yaprak 63 | 1.47561 | — |
| yaprak31 lr.03 a800 | 1.49136 | — |
| min_child 200 | 1.42555 | — |
| min_child400 y127 | 1.44024 | — |

İKİ BLOK DA AYNI YÖNÜ SÖYLÜYOR (p06 bunu hiç başaramadı).
Hiperparametre değişiklikleri HEPSİ kaybettiriyor -> kazanç ince ayardan
değil KAYIP FONKSİYONUNDAN geliyor, aşırı uyum değil.
Mekanizma: soğuk MSE'nin %55.5'i tüketim=0 satırlarından; L2 bu aykırı
değerlerin peşinden koşuyor, huber kırpıyor.

**SON DURUM (agent durdurulduğunda, ÇÖZÜLMEMİŞ):**
> "Kritik gerilim: yaz25'te AĞIRLIKLI ölçüm NEGATİF ama PG-hakim kovada
> POZİTİF. Ayrıştırıyorum."

Yani huber henüz kapıdan GEÇMEDİ. Devam edince İLK İŞ bu gerilimi çözmek.

Eğer geçerse aritmetik: soğuk 1.43592 -> 1.363 ise bileşik CV kazancı
**0.0234**, yani GÜVENİLİR TAŞINMA BANDININ (>=0.02) içinde. Bu, elimizdeki
tek "bandın doğru tarafında" bahis.

## 7. HAZIR DOSYALAR

`experiments/model29/p_kalici/aday_csv/` (12 aday CSV + 4 delta npy).
Hepsi doğrulandı: 714688 satır, id sırası test.csv ile birebir, NaN yok,
negatif yok, sonlu. **submissions/ altına YAZILMADI.**

| dosya | eski (şişkin) beklenti | GERÇEK beklenti |
|---|---|---|
| p10_span_hafif_olu25 | 0.99281 | ~1.0008 (P(3.sıra) %3.5) |
| p10_ypseviye_hafif_olu50 | 0.99308 | ~1.0008 |

**BU DOSYALAR ARTIK GÖNDERİLMEZ.** Taşıma ölçümü onları geçersiz kıldı.

Yedek (ölçülmüş, elimizde duruyor): `submissions/tuketim_YP_seviye.csv` = 1.00115.

## 8. DEVAM EDİNCE YAPILACAKLAR (sırayla)

1. **Liderlik tablosunu yeniden çek** (`kaggle competitions leaderboard
   grid-up-datathon --show`). Hedef hareket ediyor.
2. **Huber gerilimini çöz.** Agent'ı yeniden başlat (veya kendin ölç):
   yaz25'te test dağılımına ağırlıklı ölçüm neden negatif, PG-hakim kovada
   neden pozitif? Hangisi teste daha yakın? Ağırlıklandırma hücrelerinde
   etkin trafo sayısı kaç (yaz25 ess=34.3, PG(75,90]=30 trafo -- çok zayıf)?
   guz25 (PG 150 trafo) ve kis26 (PG 168 trafo) ne diyor? Karar oradan gelmeli.
3. Huber GEÇERSE: üretim hattında soğuk uzmanı huber ile yeniden eğit, test
   tahmini üret, tek BÜYÜK bahis olarak hazırla (~1 saat).
4. Huber GEÇMEZSE: SICAK tarafa bir agent sal. Satırların %78'i orada,
   ölçüm gücü soğuğun ~20 katı (soğukta testin baskın kovasında yaz25'te
   yalnız 30 trafo var -- yapısal sınır). Ama unutma: 0.02 bandını aşmayan
   hiçbir şey taşınmıyor.
5. Gönderim SIRASI: her hak bir ölçümdür. İlk hak taşıma oranını gerçek
   veriyle ölçer: `oran = (1.00115 - gelen) / CV_kazanci`.

## 9. DEĞİŞMEYEN KURALLAR

- **ONAY OLMADAN HİÇBİR GÖNDERİM YAPILMAZ.** Kullanıcı her seferinde ayrı
  onay verir. Gönderimden sonra MUTLAKA liste okunur (zaman aşımına uğrayan
  betik "gönderilmedi" demek değil).
- Son seçim (2 dosya) TARAYICIDAN yapılır, API'den yapılamaz. Önce
  "You selected X of N" satırı okunur.
- Paralel oturumlar aynı çalışma ağacını paylaşıyor: commit'te yalnız kendi
  dosyalarını stage'le.
- Python yazarken Write aracı (bash heredoc ters bölüleri bozuyor).
- subprocess'te text=True yanında encoding="utf-8".
- str.replace() öncesi `assert hedef in s`.
- Ara dosyalar scratchpad'e (Bash /tmp ile Windows Python uyuşmuyor).

---

# EK — 31 Ağustos 20:40 GÜNCELLEMESİ: §2'DEKİ 0.02 KURALI ÇÜRÜTÜLDÜ

**§2'yi ve ona dayanan tüm "ELENDİ" hükümlerini GEÇERSİZ SAYIN.**

## Neden çürüdü

1. **p12e'nin zayıflatma çürütmesi yanlış varyansla yapılmış.** λ=0.930, vekilin
   TOHUM gürültüsüyle (7.60e-06) hesaplanmıştı. Ama vekilin hatası tohum değil
   SİSTEMATİK ve doğrudan ölçüldü: **3.97e-04** — çiftlerde gözlenen x
   saçılımından (1.09e-04) BÜYÜK. Doğru λ = **−2.65**, yani güvenilirlik SIFIR.
   `b_cift = −0.161` (GA95 [−0.50, +0.17]) **hiçbir bilgi taşımıyor.**
2. **Band sınırı ölçüm gürültü tabanıyla birebir çakışıyor.** σ_çift = 0.01992;
   band sınırı 0.02 = **1.00 × σ_çift**. "Taşınmayan band" ile "ölçemediğimiz
   yer" aynı yer.
3. **Simülasyon:** kural HİÇ yokken bile, ölçülen vekil gürültüsüyle, gözlenen
   ayrışma (R² 0.986 vs 0.036) **%15.8** olasılıkla çıkıyor. Gürültü sıfırlanınca
   **%0**. Ayrışmayı üreten mekanizma gürültüdür. (Ama simülasyon ayırt edici
   DEĞİL: LR = 1.63, kural-gerçek lehine, ihmal edilebilir.)
4. **En temiz kanıt — vekile bulaşmamış GERÇEK blok CV'si:**

| çift | gerçek dCV | gerçek dLB | oran | band içi |
|---|---|---|---|---|
| v27→v30 | −0.00952 | −0.00723 | 0.759 | EVET |
| v30→v46 | −0.00692 | −0.00191 | 0.276 | EVET |
| v27→v46 | −0.01644 | −0.00914 | 0.556 | EVET |

   Üçü de bandın İÇİNDE, işaret uyumu 3/3, kesmesiz eğim **b = 0.568**
   (alternatif eşleştirmede 0.967). Aynı çiftlerde **vekil 3/3 YANLIŞ işaret**
   verdi. n=2 bağımsız — kanıt değil, güçlü işaret.

## YENİ KARAR KURALI (0.02 kapısının yerine)

0.02'yi gönderim kapısı olarak KULLANMA — o bir ölçüm sınırı, olgu değil. Kapı:
  (a) kazanç vekilden değil GERÇEK blok CV'sinden mi ölçüldü,
  (b) blok-dışı seçim hedef bloktan bilgi kullanmadan mı yapıldı,
  (c) bloklar arası işaret tutarlı mı,
  (d) tohum sayısı arttıkça kazanç ayakta kalıyor mu.
Taşıma oranı için muhafazakâr nokta tahmin **0.5**.

## HUBER α=0.5 — YENİDEN DEĞERLENDİRME

CV kazancı (test bileşimi): yaz25 −0.00736, guz25 +0.02067, kis26 +0.02458
→ ort **+0.01263**, bloklar arası se 0.01006, P(kazanç>0) = 0.83.
Gereken taşıma oranı: 0.00559/0.01263 = **0.443**.

| oran | beklenen LB | P(3. sırayı geçer) |
|---|---|---|
| 0.25 | 0.99799 | 0.22 |
| **0.50** | **0.99484** | **0.55** |
| 0.568 (gerçek çift A) | 0.99398 | 0.60 |
| 0.967 (gerçek çift B) | 0.98894 | 0.72 |

Tam Monte Carlo: medyan LB **0.99519**, GA80 [0.9825, 1.0050],
**P(3. sıradan iyi) = 0.52**, P(2. sıradan iyi) = 0.51.

Kapı sonucu: (a) GEÇER, (b) GEÇER, (c) **2/3** (yaz25 ters), (d) **ZAYIF**
(eklenen her tohum kazancı aşağı çekti: +0.0156 → +0.0126).

## AŞAĞI YÖNLÜ RİSK ASLINDA YOK

Monte Carlo "P(mevcuttan kötü) = 0.19" diyor. **Ama bu bir kayıp değil:**
yarışma sonunda TARAYICIDAN 2 dosya seçiyoruz ve ölçülmüş
`submissions/tuketim_YP_seviye.csv` = 1.00115 elimizde duruyor. Skor kötü
gelirse o dosyayı seçmeyiz. Maliyet yalnızca bir gönderim hakkı.

Yani karar basit: **%52 ilk-3 olasılığı, fiilen sıfır aşağı yönlü risk.**

## DEVAM EDEN İŞLER (20:40)

- `p14_test.py` (P14_ADAY=huber_a05) KOŞUYOR → `p14_soguk_huber_test_log.npy`
  ve `p14_soguk_huber_yalniz_test_log.npy` üretecek. Gönderim adayı bu.
- Sıcak taraf taraması (p15) sürüyor. cat ELENDİ (yaz25 +, guz25 −).
  lgbm yaz25'te üç alfada da kazanıyor (0.84463 → 0.80975 / 0.81162 / 0.81928),
  guz25 karşılığı bekleniyor. Yakınlık ağırlıklandırması ilk nokta:
  cat yaz25 tau120 → 0.828848 vs taban 0.833132 (−0.0043).
- Sıcak kazanç çıkarsa soğuk deltayla TOPLANIR (farklı satır grupları).

Kaynak: `experiments/model29/p_kalici/p17_band.json`, betik `p17_band.py`.

---

# EK 2 — 21:00: p14 TEST CIKTISI GUVENILMEZ (denetim coktu)

`p14_test.py` calistirildi (P14_ADAY=huber_a05) ve dort dosya uretti, AMA
betigin kendi DENETIMI COKTU:

```
denetim_lgbm_uretim:  maxabs = 5.850299   kor = 0.8614
```

Betigin belgesi "tutmazsa DUR" diyor; kodda daha once **maxabs 4.8e-07** ile
dogrulandigi not dusulmus. Simdi yedi mertebe sapma var. Ve kontrol `raise`
etmiyor, yalnizca log yaziyor -- dosya SESSIZCE uretildi.

Anlami: `delta = (lgbm_huber - lgbm_uretim)/3` ve `lgbm_uretim` olarak
`p06_test_soguk_aile.npy` kullaniliyor. Denetim tam olarak o dizinin dogru
referans olup olmadigini sinuyordu ve KALDI. p06 zaten kirilmis bir calisma;
dizilerinin hangi yapilandirmayla uretildigi supheli.

**KARAR: p14_soguk_huber_test_log.npy ve kardesleri KULLANILMAYACAK.**
Test tahmini `p18_yeniden_egit.py` ile, birebir dogrulama sarti altinda
yeniden uretilecek.

**NOT:** p14 agent'i ayni dosyayi reddetti ama YANLIS gerekcyle (curutulmus
0.02 bandi). Denetimin coktugunu fark etmedi. `p14_ozet.json` icine
`00_GECERSIZ_HUKUM_UYARISI` ve `00_p14_TEST_CIKTISI_GUVENILMEZ` anahtarlari
eklendi.

# EK 3 — 21:00: SICAK TARAMA ARA SONUCLARI (tek tohum, ELEME asamasi)

Her aile FARKLI alfa istiyor -- tek global alfa ikisini birden kacirirdi:

| aile | TABAN yaz25 | en iyi | kazanc | not |
|---|---|---|---|---|
| lgbm | 0.84463 | a=0.5 -> **0.80975** | **-0.0349** | uc alfada da kazaniyor |
| xgb  | 0.83630 | a=2.0 -> **0.82876** | -0.0075 | a=0.2 felaket (1.41) |
| cat  | 0.83313 | a=1.0 -> 0.82217 | -0.0110 | **guz25'te TERS (0.8392 vs 0.8361)** |

cat guz25: huber_a05 0.8705, l1 0.8826 -- hepsi kotu. cat kis26 l1 0.7807 vs
TABAN 0.7696, yine kotu. **cat huber ELENDI.**

YAKINLIK AGIRLIKLANDIRMASI (yeni fikir, GDZ raporundaki sebeke buyume
verilerinden): `w = exp(-(kesim - tarih).days / TAU)`

| cat yaz25 | RMSLE | kazanc |
|---|---|---|
| TABAN (tau=sonsuz) | 0.833132 | -- |
| tau=120 | 0.828848 | -0.0043 |
| **tau=480** | **0.823415** | **-0.0098** |

Yumusak agirliklandirma kazandiriyor, sert olan degil. Optimum 480 ile
sonsuz arasinda olabilir; izgara {240,480,960,1920} ve UC AILE icin de
kosulacak.

**UYARI: hepsi TEK TOHUM, TEK/IKI BLOK.** cat tam boyle basladi ve guz25'te
dondu. Asama 2 (3 tohum x 3 blok) hukmu verecek.

Kaba aritmetik (lgbm+xgb iyilesir, cat degismez): sicak ~+0.009,
soguk huber +0.0126 -> toplam ~0.022 CV. Tasima 0.5 ile LB 1.00115 -> 0.99035.
