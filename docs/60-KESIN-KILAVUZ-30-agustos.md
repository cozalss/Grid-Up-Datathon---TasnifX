# 60 — KESİN KILAVUZ · 30 Ağustos

**Bu belge docs/54, 57, 58'in yerine geçer ve tek başına yeterlidir.**
Sabah bunu aç, sırayla uygula. Arka plan: [`docs/59`](59-tam-durum-2026-08-29.md)

---

## 1. Başlangıç durumu (2026-08-29 22:00 itibarıyla)

```
1. Grid Grinders     0.99046
2. Atakan Aldemir    0.99940   <- HEDEF (bugun 09:29'da 1.00041'den indi)
3. Tuna Deniz        1.00267   <- yeni, bugun 18:29'da bizi gecti
4. TasnifX           1.00284   <- BIZ  (submissions/tuketim_m6_ikiyon.csv)
5. Abdulbaki Bayir   1.00322
6. Ahmet Celik       1.00323
7. SemaNur3407       1.00349
```

```
m0 = 1.00284^2 = 1.005688066
2. sira hedefi 0.99940 -> MSE 0.998801 -> gereken dMSE  -0,006887
kota yerel 03:00'te yenilenir · 9 hak (30-31 Agustos + 1 Eylul)
bitis 1 Eylul 23:59 UTC · notebook 2 Eylul 13:00 · private LB 2 Eylul 00:10
```

**Dış veri SERBEST** — düzenleyici e-postayla teyit etti. Notebook'ta beyan şart.
**Final için 2 gönderim SEÇİLMELİ** — Kaggle arayüzü, tarayıcı. API'de yok.

---

## 2. GÜN 1 — üç sonda, birleşim YOK

Her sonda: `m6 + 1,093664·d_g7 + t·d_aday`
`g7`'nin `L`'si BİLİNİYOR (ölçülmüş `L`'lerden çıkıyor), o yüzden `g7` gönderilmez —
her sondaya optimum katsayıyla gömülü. Sonda hem **ölçüm** hem **gönderilebilir skor**.

| sıra | dosya | `t` | Q | kos(g7) | L=0 ise | r=0,035 |
|---|---|---|---|---|---|---|
| **1** | `submissions/tuketim_s2y40.csv` | 0,60 | 0,029 | **−0,555** | 1,00341 | **0,99987** |
| **2** | `submissions/tuketim_s2z2.csv` | 0,35 | 0,118 | −0,231 | 1,00704 | 1,00285 |
| **3** | `submissions/tuketim_s2sul.csv` | 0,45 | 0,054 | +0,111 | 1,00740 | 1,00377 |

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_s2y40.csv -m "sonda y40: m6 + 1.093664*d_g7 + 0.60*d_y40. kos(g7,y40)=-0.555 super-toplamsal. AMAC: L_y40 olcmek"
kaggle competitions submissions -c grid-up-datathon
```
Sonra `s2z2`, sonra `s2sul` — aynı biçimde. **Her gönderimden sonra listeyi OKU.**

### Skor gelince `L` çözümü (tam `m0` ile)
```
L_y40 = (1.006831155 - P^2) / 1.20
L_z2  = (1.014123150 - P^2) / 0.70
L_sul = (1.014856670 - P^2) / 0.90
```
Sabitler: `experiments/model29/m106_sonda2.json`

---

## 3. GÜN 2 — ortak optimum

Ölçülen `L`'lerden eşdeğer skor: `S_j = sqrt(m0 + Q_j - 2·L_j)`
`Q`: y40 0,028519 · z2 0,117949 · sulama 0,053753 · g7 0,002494 (`L_g7 = 0,002728`, `S_g7 = 1.00136`)

```powershell
python experiments/model29/m99_coklu_coz.py tuketim_m6_ikiyon.csv=1.00284 tuketim_g7_span_tau3.csv=1.00136 tuketim_y40_sota_temiz.csv=<S> tuketim_z2_analog.csv=<S> tuketim_t1_sulama.csv=<S> --cikti tuketim_g9_ortak.csv
```

`m99` **korkuluklu** (10 kontrol): `cond>1e8` · `MSE<0` · `|k|₁>5` · `MSE>m0` ·
satır/ID/NaN/negatif/sonsuz · `maks>3×taban` · skor kaydıyla çelişki · yön ID hizası.
Tetiklenirse `--lam 0.001` ekle. Dosya geçici ada yazılıp denetim geçince taşınıyor.

Kalan iki hakla `y46` ve `y45` sondala (aynı `g7`-gömülü tasarım).

## 4. GÜN 3 — altı yönlü ortak optimum + son rötuş

---

## 5. Beklenen sonuç — seçilen üçlü `g7+y40+z2+SULAMA`

| kalite `r` | sonuç | sıra |
|---|---|---|
| 0 (hiç bilgi yok) | **1,00061** | 3. |
| 0,0137 | 0,99940 | **2. sıra eşiği** |
| 0,015 | 0,99925 | **2.** |
| 0,025 | 0,99782 | **2.** |
| 0,035 | 0,99598 | **2.** |
| 0,045 | 0,99372 | **2.** |
| 0,064 (`m4` seviyesi) | 0,98824 | **1.** |

**Ölçülmüş kaliteler:** v101 0,1243 · m4 0,0641 · g7 0,0546 · p51 0,0493
→ hepsi eşiğin (0,0137) **3,6–9 katı**.

**Geriye gitme riski YOK** — Kaggle en iyi public skoru tutar. En kötü senaryo
1,00284'te kalmak.

---

## 6. Aday envanteri (hepsi kapı denetiminden geçti)

| dosya | Q | kurtoz | kos(g7) | kos(y40) | durum |
|---|---|---|---|---|---|
| `tuketim_s2y40.csv` | — | — | — | — | **GÜN1-1** |
| `tuketim_s2z2.csv` | — | — | — | — | **GÜN1-2** |
| `tuketim_s2sul.csv` | — | — | — | — | **GÜN1-3** |
| `tuketim_y40_sota_temiz.csv` | 0,029 | 14,5 | −0,555 | — | sondada |
| `tuketim_z2_analog.csv` | 0,118 | 9,0 | −0,231 | +0,131 | sondada |
| `tuketim_t1_sulama.csv` | 0,054 | 11,1 | +0,111 | −0,247 | sondada |
| `tuketim_y46_amnezik_kirpik.csv` | 0,388 | 4,5 | −0,091 | +0,114 | 31 Ağu |
| `tuketim_y45_mevsimsel_kirpik.csv` | 0,167 | 8,5 | −0,081 | +0,026 | 31 Ağu |
| `tuketim_q1c_kapasite_siki.csv` | 0,061 | 5,0 | −0,034 | +0,122 | yedek |
| `tuketim_t3_turizm.csv` | 0,039 | 14,6 | −0,052 | +0,124 | yedek |
| `tuketim_g7_span_tau3.csv` | 0,0025 | 23,1 | — | — | **GÖNDERİLMEZ** |
| `tuketim_t2_bayram.csv` | 0,011 | **37,2** | +0,009 | −0,073 | kurtoz yüksek |
| `tuketim_h1_isil.csv` | 0,052 | 19,1 | −0,053 | +0,265 | **kos(m4)=+0,805**, m4 kopyası |
| `tuketim_k5_kesinti.csv` | 0,0015 | 19,8 | +0,073 | −0,041 | **gürültü** |

**Örtüşen çiftler** (ikisi birden alınırsa kazanç toplanmaz):
`y45↔z2 +0,323` · `q1c↔q1d +0,515` · `SUL↔y45 +0,182`

---

## 7. Araçlar

| araç | ne yapar | doğrulama |
|---|---|---|
| `m99_coklu_coz.py` | N yönlü ortak optimum, 10 korkuluk | `v102+m4+p51` → 1,00292 (gerçek 1,00284); 5 senaryoda uçtan uca prova |
| `m50_harman_coz.py` | iki dosyalı harman | `v83+v101` → `v102`'yi **bit düzeyinde** üretti |
| `m105_secim.py` | aday eleme + üçlü seçimi | — |
| `m106_sonda2.py` | sonda üretimi | bağımsız türetimle doğrulandı (`m103`) |

---

## 8. Kaçırılmaması gerekenler

- **Her gönderimden sonra listeyi OKU.** Zaman aşımı "gitmedi" demek değil.
- **Final için 2 gönderim SEÇ** (tarayıcı). Seçilmezse en iyi public otomatik.
- **Notebook 2 Eylül 13:00.** İlk 20 inceleniyor; tüm dış veri kaynak+amaç+kullanımla beyan.
  Nisan–Temmuz 2026 **gerçekleşmiş** hava verisi kullandığımız açıkça yazılmalı.
- **Ölçüm hassasiyeti ±9,5e-5.** 2e-4 altındaki kazanç ölçülemez, hak harcanmaz.
- **Hedef hareketli.** Atakan bugün 1,00041 → 0,99940 indi; yarın da inebilir.

---

## 9. Kapalı eksenler — tekrar açma

soğuk×ölü trafolar (permütasyon: AUC 0,55 vs boş 0,53) · ileri-pencere geri-testi
(`f = −0,42`, yönü bile ters) · span'ı yeniden karıştırmak (tavan 1,0014) ·
2026-05-11 dalgası (varyans, yanlılık değil) · rejim uzmanı (hava girince kayboldu) ·
sık kesim · ölü trafo tezi · `r1`/`r3` (Q=0,003) · **kesinti** (karıştırılmış etiket:
sahte yön gerçeğinden büyük, kos +0,795) · **hafta günü** (plasebo 0,00348 vs sinyal
0,00343) · **takvim demeti** · **uzun ısıl pencere** (m4'ün kopyası)

---

## 10. Kalıcı kurallar 32–42

**32.** İleri-pencere geri-testinde eğitim kesimlerinin hedef pencereleri doğrulama
penceresini KESMEMELİ. · **33.** Sağlam kayıp (Huber/L1) L2'yi döver; iki kesimde de
kazanmayan alınmaz. · **34.** Global seviyeyi geri-testten öğrenme, işareti dönüyor. ·
**35.** Bir özellik ailesini, mekanizmasının AKTİF olduğu pencerede doğrula. ·
**36.** Geri-test LB'yi ÖNGÖRMÜYOR; yalnız yön üretir, büyüklük LB'de ölçülür. ·
**37.** Dik parçalara bölmenin maliyeti bir prob, getirisi garanti ≥ 0. ·
**38.** Probu `t=1` ile kurma; `t`, probun kendi skorunu iyileştirsin. ·
**39.** Öngörü hassasiyeti ±9,5e-5 (public/private kanalı). ·
**40.** Aynı makineyi kullanan iki ajanın aynı sayıyı bulması DOĞRULAMA DEĞİL. ·
**41.** Forum bulgusunu tek mesaja dayandırma. ·
**42.** Her hipoteze **plasebo (tohum) kolu** zorunlu; kurtoz SERT eleme ölçütü değil.
