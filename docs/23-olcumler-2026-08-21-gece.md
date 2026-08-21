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
