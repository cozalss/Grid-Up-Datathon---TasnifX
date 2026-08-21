# Ölçümler — 21 Ağustos 2026 gecesi

[22-durum](22-durum-2026-08-21-aksam.md) belgesinin devamı. Orada
kaydedilmiş sonuçlar geçerli; bu belge o günün akşamında yapılan yeni
ölçümleri taşır. Gönderim hakkı 03:00'te (00:00 UTC) açılıyor, bu yüzden
tüm akşam saf ölçüm penceresi oldu.

---

## 1. En önemli ölçüm: model naif tabana ne katıyor

Hiç model olmadan, "bu trafonun son 30 gündeki log ortalamasını tahmin et"
(soğuk satırlarda kapasite × sabit):

| blok | naif `t_log_son30` | sıcak | soğuk |
|---|---|---|---|
| yaz25 | 1,1406 | 0,8632 | 1,8038 |
| guz25 | 1,1732 | 0,9190 | 1,8014 |
| kis26 | 1,0558 | 0,6924 | 1,8292 |
| **ortalama** | **1,1232** | **0,8249** | **1,811** |

Bizim en iyi ölçülmüş modelimiz **1,10665**. Fark **0,017** — karar
eşiğimiz **0,01995**. Yani 144 öznitelik, üç GBDT ailesi, kapasite ofseti,
soğuk maskeleme ve harman birlikte, sıfır parametreli bir tahminden
**ayırt edilemez** durumda.

Bu, "ne ekleyelim" sorusunu "neden mevcut yapı bu kadar az katıyor"
sorusuna çevirir.

## 2. Sıcak rejim: iki öznitelik modeli geçiyor

`t_log_son30` (yakın geçmiş) ile `t_log_ort` (uzun ortalama) arasında en
iyi karışım ağırlığı, ufka göre düzgün biçimde kayıyor:

| ufuk (gün) | en iyi w (w=1 → tamamen son30) | son30 tek | karışım |
|---|---|---|---|
| 0–14 | 0,956 | 0,4998 | 0,4992 |
| 15–29 | 0,847 | 0,5960 | 0,5896 |
| 30–44 | 0,813 | 0,6690 | 0,6604 |
| 45–59 | 0,748 | 0,7503 | 0,7361 |
| 60–74 | 0,590 | 0,8173 | 0,7822 |
| 75–89 | 0,518 | 0,9367 | 0,8944 |
| 90–104 | 0,495 | 0,9789 | 0,9344 |
| 105–122 | 0,503 | 1,0303 | 0,9893 |

Havuzlanmış: son30 tek **0,8062**, ufka göre değişen karışım **0,7792**.
Modelin sıcak skoru 0,80–0,84 bandında. **İki öznitelik ve sekiz parametre,
144 öznitelikli topluluğu geçiyor.**

Model `ufuk_gun`, `t_log_son30` ve `t_log_ort`'un üçüne de sahip. Yani bu
etkileşimi öğrenebilecekken öğrenmiyor. En olası neden soğuk maskeleme:
eğitimde trafoların %22'sinin geçmişini siliyoruz, bu modele geçmişe
güvenmemeyi öğretiyor.

## 3. Soğuk rejim: tavan çok uzakta

| blok | küresel sabit | ilçe ort. | **kâhin: trafonun kendi seviyesi** |
|---|---|---|---|
| yaz25 | 1,8038 | 1,7405 | **0,7196** |
| guz25 | 1,8014 | 1,7838 | **0,3909** |
| kis26 | 1,8292 | 1,8233 | **0,3362** |

Soğuk trafonun kendi seviyesini bilseydik hata 1,81 → **~0,48**.
Elimizdeki en iyi bilgi (ilçe) bu 1,33'lük açığın **0,03**'ünü kapatıyor.

Segment taraması (çapraz-blok, soğuk satırlar):

```
kova x ilce    1,7811      guc kovasi   1,8052
ilce           1,7825      il           1,8059
kova x uzunluk 1,7880      ID onek2     1,8113
uzunluk        1,7987      kuresel      1,8115
```

**Sonuç:** problem tavana yakın DEĞİL. Tavan uzakta, biz ulaşamıyoruz.

## 4. ÖLÇÜLDÜ VE REDDEDİLDİ — bu geceki üç yol

### 4.1 ID sayısal komşuluğu

Belge "ID öneki işe yaramıyor" diyordu; önek eşleşmesi yanlış metrik.
Sayısal komşuluk gerçekten sinyal taşıyor — sıcak trafolar arasında:

| ID mesafesi | ort abs(Δseviye) | rastgele | oran |
|---|---|---|---|
| en yakın (≤50) | 1,4855 | 1,7633 | **0,842** |
| 2. | 1,5153 | 1,7569 | 0,862 |
| 3. | 1,5701 | 1,7786 | 0,883 |
| 5. | 1,6098 | 1,7890 | 0,900 |
| 10. | 1,6438 | 1,7826 | 0,922 |
| 25. | 2,0634 | 1,7593 | 1,173 |

Mesafeyle düzgün bozuluyor — artefakt değil. İlçe etkisi de değil (aynı
ilçede rastgele 1,7041, aynı ilçede komşu 1,5016).

**Ama kullanılamıyor, iki ayrı nedenle:**

1. `√(1−ρ) = 0,842` → **ρ ≈ 0,29**. Kareli hata altında tek komşu
   kullanmak varyansı ikiye katlar; kazanmak için ρ > 0,5 gerekir.
   Ölçüldü: K=1 ham komşu 2,7310, küresel sabit 1,8389.
2. Küçültmeli segment ortalaması da çalışmıyor, çünkü sinyal soğuk
   trafolara **transfer olmuyor**. ID boşluğuna göre segment ortalamasının
   soğuk trafo seviyesini açıklama gücü:

   ```
   bosluk>10    R2 0,000-0,008      bosluk>1000   R2 0,000-0,004
   bosluk>50    R2 0,000-0,003      bosluk>10000  R2 0,000-0,007
   [kiyas] ilce R2 0,000-0,003
   ```

**Neden:** soğuk trafolar toplu katılımla kendi ID bloklarında beliriyor
(2.024'ün 1.666'sı Mayıs 2026'da), yerleşik sıcak trafoların olduğu ID
bölgelerinde değil. Komşuluk sinyali var ama komşu yok.

### 4.2 `lokasyon` alanında daha ince kırılım

Kapandı: **47 benzersiz değer**, en fazla 3 parça
(`İZMİR>METROPOL>KARABAĞLAR` = il > bölge > ilçe). Üçü de zaten öznitelik.
Mahalle/fider düzeyi **yok**.

### 4.3 Aykırı değer temizliği ASHRAE ölçeğinde değil

ASHRAE'de (aynı metrik) aykırı satır atmak en büyük tek kaldıraçtı
("789.682 satır attım ve 1,04'ü kırdım"). Bizde ölçüldü:

```
oran>24 kWh/kVA/gun :  1.511 satir (%0,145)  -> log-varyansin %0,5'i
oran>50             :    335 satir (%0,032)  -> %0,2
oran>200            :     45 satir (%0,004)  -> %0,1
```

Fiziksel üst sınır ~24 kWh/kVA/gün (7/24 tam yük). 37 trafoda aşılıyor.
Yapı: **5 trafoda günlerin %90+'ı aşırı** (tutarlı birim hatası — model
geçmişten öğrenir, dokunulmamalı), **21 trafoda %5'ten az** (düzensiz
sıçrama — gürültü).

Beklenen kazanç eşiğin altında. Yine de denenecek: bedava.

## 5. Yapısal kör nokta: `t_ay_sapma`

| kolon (sıcak satırlar) | yaz25 | guz25 | kis26 | **TEST** |
|---|---|---|---|---|
| `t_ay_sapma` | **0,0%** | **0,0%** | 15,8% | **46,8%** |
| `t_hg_sapma` | 100% | 99,4% | 98,4% | 95,6% |

Aylık profil kolonu eğitimde neredeyse tamamen NaN, test'te sıcak
satırların yarısında dolu. Sebep giderilemez: bir bloğun hedef ayları,
profil kaynağından çıkarılan bloğun kendisidir; 15 aylık veriyle önceki
yıl yok.

**Sonuç: CV bu kolonun değerini ne olumlu ne olumsuz ölçebiliyor.**
Ölçülebilir tek alt küme kis26'nın dolu satırları — orada sınanmalı.
Aksi halde bu bir kumar, ölçüm değil.

## 6. Araştırma — kaynaklı bulgular

Üç bağımsız araştırma koşusundan, listemizde OLMAYAN ve sayısı olan:

| bulgu | kaynak | sayı |
|---|---|---|
| Tohum torbalama (XGBoost) | Allstate 2., Noskov | −%0,91 CV / −%0,67 LB |
| Tohum torbalama (N=100) | Deotte, PS S5E6 | +0,004 MAP@3 (%1,06) |
| Tohum torbalama | ASHRAE 2. | **"pek yardımcı olmadı"** |
| Segment bazlı hedef ayrımı | Bike Sharing, casual/registered | **−0,0241** (özellik değişimiyle karışık) |
| Farklı normalizasyonlu 4. üye | ASHRAE 1. | +0,002 (çeşitlilik için) |
| Ağaç tabanlı 2. seviye istifleme | ASHRAE 9./13./20. | **üçü de aşırı uydurdu** |
| Doğrusal birleştirici (Ridge) | ASHRAE 9., Mercari 1. | çalışan tek istifleme |
| Sihirli çarpan, özel LB kazancı | ASHRAE 13. | 0,004–0,005 — **bizim gürültü tabanımızın altında** |

Son satır önemli: altın madalyası bu numaraya bağlı görünen ekipte bile
gerçek (private) etki 0,005'ti; bizim gürültü tabanımız 0,00998. Daha önce
verdiğimiz ret kararını bağımsız olarak doğruluyor.

### CatBoost — hiç dokunulmamış üç parametre

Araştırma, üretim çağrısının yalnızca altı anahtar verdiğini ve gerisinin
kütüphane varsayılanı olduğunu saptadı:

* `l2_leaf_reg=3.0` **CatBoost'un kendi varsayılanı** — "ayarlandı"
  denmesine rağmen hiç oynatılmamış.
* `bootstrap_type` sessizce `MVS` (varsayılan); `Bernoulli`/`Bayesian`
  hiçbir arama uzayında yer almamış.
* `random_strength` hiç dokunulmamış.

`nan_mode` için hüküm: **`Min`de bırak.** CatBoost, eksik değerleri
diğerlerinden ayıran bir bölmenin her zaman aday olacağını garanti ediyor;
Min/Max bu garantiyi değiştirmiyor. Ayrıca 19 `t_*` kolonu birlikte NaN
oluyor ve `soguk_mu` bayrağı aynı bilgiyi zaten açıkça taşıyor.

## 7. Sıradaki ölçümler

`scripts/deney_ileri.py` bu gece yazıldı. Dört deney:

```
--deney soguk_oran   maske orani x CatBoost + rejim yonlendirmesi + buzulme
--deney torba        tohum torbalama egrisi + torbalanmis harman + rejim agirligi
--deney yalin        yalin oznitelik setleri + set harmani
--deney agirlik      soguk ornek agirligi (soguk_agirliklari olu koddu)
--sure               tek fit suresi
```

Ölçülen fit süreleri: **CatBoost 35 sn, XGBoost 45 sn, LightGBM 23 sn**,
maskeleme 0,2 sn. Yani 40–60 fitlik bir deney bütçesi rahat.

Sonuçlar `experiments/ileri_sonuclar.jsonl`.

---

## 8. SIFIR YIĞILMASI — hatanın haritasındaki en yoğun nokta

Soğuk satırların dağılımı incelendiğinde:

```
tam sifir      satir %5,16   ->  SOGUK karesel hatanin %52,3'u
<1 kWh         satir %5,29   ->  %53,5
<100 kWh       satir %12,33  ->  %67,2
>p99 (buyuk)   satir %1,00   ->  %2,1
```

Soğuk rejim toplam hatanın %59'u olduğuna göre, **tüm satırların ~%1,15'i
toplam karesel hatanın ~%31'ini** taşıyor.

Ve yapı satır bazında dağınık değil, **trafo bazında ikili**:

```
1.636 soguk trafo (%91,9)   hic sifir yok
   84 soguk trafo (% 4,7)   %80+ sifir      <- olu trafolar
   61 trafo (% 3,4)         arada
```

Mükemmel sınıflandırma soğuk RMSLE'yi 1,84 → ~1,27, genel skoru **~0,94**
yapardı. Yarısı bile 1,04.

### Ama tahmin edilemiyor — ölçüldü

Tek öznitelikli ayırıcılık (AUC), 1.781 soğuk trafo, 84 ölü:

```
guc 0,513   satir sayisi 0,545   tanim_num 0,542   yas 0,559
pencere 0,559   uzunluk 0,550    p_* kolonlari 0,512-0,559
```

Hepsi rastgeleden ayırt edilemez. Kırıntılar: 8 hanelilerde %5,2 ölü,
9 hanelilerde %1,6 (ama n=4); Urla %16,7, Bayındır %13,5 (küçük örneklem).

**ID komşuluğundan ölülük** de çalışmıyor — üç bloğun ikisinde etki sıfır
ya da ters:

| blok | komşu ölüyse P(ölü) | komşu canlıysa | AUC |
|---|---|---|---|
| yaz25 | 0,000 | 0,032 | 0,446 |
| guz25 | 0,000 | 0,033 | 0,483 |
| kis26 | 0,273 | 0,048 | 0,604 |

## 9. Yeniden numaralandırma — yok

İki ID bloğu var (8 haneli 810.516 satır, 700-önekli 9 haneli 412.893).
Trafolar yeniden numaralandırılmış olsaydı "soğuk" trafoların bir kısmının
eski numarayla geçmişi olurdu. **On bir dönüşümde sıfır eşleşme:**

```
ham, bas 1/2/3 hane at, son 1/2 hane at, '700' oneki at,
'7'/'70'/'700' ekle, sifir doldur 9   ->  hepsi 0 eslesme
```

Not: soğuk trafoların %34,1'inin eğitim setinde ID mesafesi ≤1 olan bir
komşusu var, %49,9'unun ≤2. Komşuluk **fiziksel olarak** var; taşıdığı
bilgi yok.

## 10. Rejim yönlendirmesi — KANITLANDI (CatBoost)

63 CatBoost fit, 3 tohum, 3 blok:

| maske | sıcak | soğuk |
|---|---|---|
| 0,00 | 0,8136 | 1,8215 |
| **0,15** | **0,8128** | 1,7851 |
| 0,2216 *(mevcut)* | 0,8219 | 1,7792 |
| 0,35 | 0,8239 | 1,7852 |
| 0,50 | 0,8181 | 1,7733 |
| 0,70 | 0,8830 | 1,7688 |
| **1,00** | 1,6299 | **1,7595** |

İki eğri ters yönde ve **tekdüze**. Tek oran ikisini birden en iyi yapamaz.

```
mevcut uretim (maske %22, tek tohum)   1,11618
ayni + tohum torbalama                 1,10805
YONLENDIRME (sicak %15 / soguk %100)   1,09608
```

Üç blokta da aynı yönde. Kazanç 0,012; ham eşik 0,020 ama o eşik
tek-tohum gürültüsü için, burada tahminler torbalanmış.

**Bağımsız dayanak:** DropoutNet'in kendi oran taraması (NeurIPS 2017,
Şekil 2) soğuk başlangıç için tekdüze artıyor — 0,378 (oran 0) → 0,659
(oran 0,9), iç optimum yok. NeurIPS hakemi tam bu soruyu sormuş
("neden maskeli tek model yerine ayrı bir soğuk model?") ve yazarlar
cevap vermemiş. Ayrıca Saar-Tsechansky & Provost (JMLR 2007, 15 veri
kümesi): kolonu **silmek** NaN bırakmaktan belirgin iyi (%3,76 vs %8,73
doğruluk kaybı) — soğuk uzmanı için `t_*` kolonları atılmalı.

## 11. Büzülme — ilk deneme başarısız, nedeni bulundu

Çapraz-blok eğim kestirimi −0,033 verdi. Eğimler:

```
yaz25  soguk +0,6469  kesim +0,2772     sicak +1,0365  kesim -0,0771
guz25  soguk +0,7203  kesim +0,3764     sicak +1,0628  kesim +0,0885
kis26  soguk +0,7584  kesim +0,1015     sicak +0,9671  kesim -0,1778
```

Soğuk eğimler tutarlı biçimde **1'in altında** (0,65–0,76), yani fazla
varyans gerçek ve büzülme doğru yön. Hata kesim terimini de taşımaktı;
kesim bloktan bloğa +0,10 ile +0,38 arasında değişiyor. Düzeltme: yalnızca
eğimi taşı, merkez olarak bloğun **kendi tahmin ortalamasını** kullan
(etiket kullanmaz, sızıntı yok).

Teori (araştırmadan, kapalı form): `kazanç = (μ_ŷ − μ_y)² + σ_ŷ²(1−λ*)²`
ve model sabit tahminciyi **ancak λ* > 0,5 ise** geçer. Bizim λ* = 0,65–0,76,
yani model sabitten iyi ama büzülmesi gerekiyor.

## 12. YÖNLENDİRME HARMANDA DA TUTUYOR — karar verildi

54 fit (3 blok × 2 maske × 3 aile × 3 tohum), tohum torbalanmış.

Rejim bazında en iyi karışım gerçekten farklı:

| ağırlık | sıcak satırlar | soğuk satırlar |
|---|---|---|
| cat tek | 0,8128 | 1,7595 |
| 1/1/1 | 0,8003 | 1,7495 |
| 2/1/1 | **0,7976** | 1,7417 |
| 3/1/1 | 0,7979 | **1,7404** |
| 4/1/1 | 0,7989 | 1,7409 |
| 6/1/1 | 0,8010 | 1,7430 |

Ama 2/1/1 ile 4/1/1 arası fark 0,001 — gürültü. Bu yüzden 64 hücrelik
ızgaradan en iyi hücreyi seçmek **aşırı uydurma** olur:

```
en iyi ızgara hücresi  sicak[2/1/1] soguk[3/1/1]   1,08133
SECILEN                sicak[3/1/1] soguk[3/1/1]   1,08143
```

Fark 0,0001. 3/1/1 ikisinde de optimumun gürültü mesafesinde ve zaten
kullandığımız ağırlık — ek serbestlik derecesi getirmiyor.

### Blok kırılımı — kazanç tekdüze DEĞİL

| blok | soğuk (maske %15) | soğuk (uzman, %100) | fark |
|---|---|---|---|
| **yaz25** | 1,7254 | **1,6228** | **−0,1026** |
| guz25 | 1,7737 | **1,7151** | −0,0586 |
| kis26 | 1,8747 | 1,8833 | +0,0086 |

İki blokta çok kazandırıyor, birinde ihmal edilebilir kaybettiriyor. En çok
kazandırdığı blok **yaz25** — test döneminin mevsimsel ikizi (aynı aylar,
aynı ufuk uzunluğu). Bu, kararı güçlendiriyor.

Olası açıklama: bloğun özet penceresi uzunluğu (yaz25 90 gün, guz25 212,
kis26 334, TEST 455). Kısa pencerede sıcak trafoların geçmişi de ince, o
yüzden maske %15 modeli soğuğa kötü genelliyor. TEST'in penceresi en uzun,
yani bu eksende kis26'ya benziyor — kayıt altına alınmalı, ama +0,0086
soğuk = genel skorda +0,003, ve diğer iki blokta kazanç 10-30 katı.

### Sıcak tarafta beklenmedik kırılım

```
sicak skoru:  yaz25 0,8130   guz25 0,8263   kis26 0,7544
```

kis26 (kısa ufuklu blok) naif tabanın **altında**. Sorun uzun ufukta, ve
bu, ufka göre ağırlıklandırılmış naif taban hipotezini destekliyor.

## 13. Büzülme — düzeltildi, çalışıyor, ama eşiğin çok altında

```
beta=1,00 (buzulme yok)   1,08143
beta=0,95                 1,08082
beta=0,90                 1,08043
beta=0,85                 1,08026   <- en iyi
beta=0,80                 1,08031
beta=0,70                 1,08107
```

Eğri düzgün ve minimumu var — yani mekanizma gerçek. Ama kazanç **0,0012**,
gürültü tabanının (0,00998) sekizde biri. Üstelik β'yı doğrulama eğrisinden
seçmek serbestlik derecesi ekler. **Reddedildi**, kayıt altında.

## 14. Üretime bağlanan yapılandırma

```python
REJIM_MASKELERI = {"sicak": 0.15, "soguk": 1.00}
AILE_AGIRLIKLARI = {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0}   # degismedi
```

`REJIM_MASKELERI = None` eski tek-model davranışına döndürür.
