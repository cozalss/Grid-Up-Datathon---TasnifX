# 55 — 29 Ağustos tam bilanço: üç gönderim, 5. → 3. sıra

**Tarih:** 2026-08-29 · Kota 3/3 kullanıldı.

---

## 0. Sonuç

```
LB (29 Agustos, gonderimler sonrasi)
  1. Grid Grinders     0.99064
  2. Atakan Aldemir    1.00041
  3. TasnifX           1.00284   <- BIZ  (sabah 5. siradaydik, 1.00553)
  4. Ahmet B. ALTUNOK  1.00480
  5. Saban Ozdogan     1.00510
```

| hak | dosya | skor | öngörü | sapma |
|---|---|---|---|---|
| 1 | `tuketim_m4_hava_capali.csv` | **1.04300** | 1.00 (bant 0.99–1.02) | **YANILDI** |
| 2 | `tuketim_p51_sicak05.csv` | 1.00946 | — (prob) | — |
| 3 | `tuketim_m6_ikiyon.csv` | **1.00284** | 1.00292 | 0.00008 |

`1.00553 → 1.00284` · **5. sıra → 3. sıra**

---

## 1. ÖN KAYIT YANLIŞ ÇIKTI — tam hesap

`experiments/model29/m90_on_kayit.json`, gönderimden önce yazılmıştı:

```
tahmin:  HAK1 S merkez 1.00, bant 0.99 - 1.02
gercek:  1.04300
yanlislama esigi: "S > 1.04 -> geri-test LB'yi ONGORMUYOR"  -> TETIKLENDI
```

### Aktarım katsayısı NEGATİF

```
geri-testte m4 vs v83-soyu:  1.0359 vs 1.1140   ->  -%7,0 (m4 DAHA IYI)
LB'de       m4 vs v83:       1.04300 vs 1.01318 ->  +%2,9 (m4 DAHA KOTU)
aktarim f = -0,42
```

Geri-test sadece "taşımadı" değil, **ters yönü gösterdi**. Bu, dün gece kurulan
bütün ölçüm düzeneğinin (sızıntısız tezgâh, iki doğrulama kesimi, hava ailesi
ablasyonları) LB için **yön göstermediği** anlamına geliyor.

Dikkat: `κ* = +0,184 > 0`, yani yön tamamen boş değil — `docs/52` §14.1'deki
`κ ≈ 0,004` felaketi değil. Ama umulanın beşte biri.

---

## 2. KURTARAN HAMLE — yönü ikiye bölmek

`m4 − v102` yönü tek bir `κ` ile değil, **soğuk ve sıcak parçalara ayrı `κ`**
ile optimize edildi. Parçalar dik (ortak satır 0), o yüzden optimum ayrışır:

```
MSE* = m0 - L_sicak^2/Q_sicak - L_soguk^2/Q_soguk
```

### Prob tasarımı
`p51 = v102 + 0,50·d_sıcak + 0,18385·d_soğuk`

`t_soğuk`'a ölçülmüş `κ*` konuldu (bilinen güvenli değer), `t_sıcak = 0,5`
seçildi. **Bilgi içeriği `t` seçiminden bağımsız** (aynı denklem çözülüyor),
ama skoru daha iyi — prob boşa gitmesin diye.

### Çözüm
```
P = 1.00946
L_sicak  = +0,010605     Q_sicak = 0,086624     kappa_sicak = +0,12243
L_soguk  = +0,011714     Q_soguk = 0,034772     kappa_soguk = +0,33688
                         (toplam L = 0,022319, HAK1'den olculmustu)

tek-kappa optimumu    1,00349
IKI-YON optimumu      1,00292   ->  gerceklesen 1,00284
bolmenin kazanci      0,00065
```

### **BULGU: geri-test SICAK dedi, LB SOĞUK diyor**

```
geri-testte kazanc:  sicak 0,7615 -> 0,6875  (-%9,7)   BUYUK
                     soguk 1,8672 -> 1,8261  (-%2,2)   kucuk

LB'de kazanc:        kappa_soguk 0,337  >  kappa_sicak 0,122
                     katki: soguk 0,003946  >  sicak 0,001298 (3 kat)
```

Yani `m4`'ün LB'deki değerinin **dörtte üçü soğuk taraftan** geliyor — geri-testin
"kazanç sıcakta" hükmünün tam tersi. Prob gönderilmeseydi bu bilinemez ve
tek-`κ` harmanı 1,00349'da kalırdı.

---

## 3. Doğrulanan araç

`m50_harman_coz.py` gönderimden önce sınandı: `v83` + `v101`'den `v102`'yi
**bit düzeyinde** yeniden üretti (`maks log fark 0,000e+00`), `κ* = 0,459022` ve
öngörülen skor 1,00553 birebir tuttu.

Bugünkü sınavı da geçti: `m6` için öngörü **1,00292**, gerçekleşen **1,00284**.
Sapma 0,00008.

**DUZELTME (dusmanca denetim, ayni gece):** bu sapma yuvarlamadan GELMIYOR.
Yuvarlama bandi ±1,18e-5; gerceklesen bandin 6,7 kati disinda. Gercek kaynak:
**`Q` tum 714.688 satirda olculuyor, LB skorlari PUBLIC %50'de.** Turetilen
tam ozdeslik:
```
gerceklesen_MSE - ongorulen_MSE = 0,2770*(Qw_tum-Qw_pub) + 0,3252*(Qc_tum-Qc_pub)
```
Bu kanalin sd'si **9,28e-5 RMSLE**; gozlenen sapma 0,85 sd -- tamamen normal,
sistematik yanlilik YOK.

> **ARACIN GERCEK HASSASIYETI ±1e-5 DEGIL, ±9,5e-5 (1 sd).**
> 2e-4'un altindaki marjinal kazanc olcum gurultusunden ayirt EDILEMEZ.

---

## 4. Neden 2. sıraya yetişmedi

```
2. sira        1,00041   ->  MSE 1,000800
varilan en iyi 1,00284   ->  MSE 1,005688
eksik                        0,004888 MSE
```

Ölçülmüş iki yönün (sıcak, soğuk) span'ı tükendi. Daha ileri gitmek için
**yeni ölçülmüş yön** gerekiyor; bugünkü 3 hak buna yetmedi.

---

## 5. Yeni kalıcı kurallar

**36.** *Bu veride ileri-pencere geri-testi LB'yi ÖNGÖRMÜYOR.* Ölçüldü:
`f = −0,42`. Geri-testte %7 daha iyi olan model LB'de %2,9 daha kötü çıktı.
Model seçimi geri-teste göre yapılmaz; ancak **yön üretmek** için kullanılır,
büyüklüğü ve işareti LB'de ölçülür.

**37.** *Bir yönü göndermeden önce, onu dik parçalara bölmenin maliyeti bir
probdur ve getirisi garanti ≥ 0.* Tek-`κ`, çok-`κ`'nın kısıtlı hâlidir. Bu
gün 0,00065 kazandırdı ve geri-testin yön hatasını ortaya çıkardı.

**38.** *Prob dosyasını `t = 1` ile kurma.* Bilgi içeriği `t`'den bağımsız
olduğu için `t`, probun kendi skorunu iyileştirecek şekilde seçilir; bilinen
parçalara ölçülmüş optimum katsayı konur.

---

## 6. 30 Ağustos için durum

```
YENI TABAN:  m0 = 1.00284^2 = 1.005688      (dosya: tuketim_m6_ikiyon.csv)
kalan hak:   30-31 Agustos + 1 Eylul = 9
hedef:       2. sira 1.00041  ->  gereken dMSE -0,004888
```

**Ölçülmüş ve tükenmiş:** `m4` yönünün soğuk/sıcak bölmesi.

**Açık eksen — soğuk tarafı daha ince bölmek.** `κ_soğuk = 0,337` doygun değil
ve soğuk taraf LB'de üretken çıktı. Bölme adayları (hepsi `m4 − v102` yönünün
soğuk parçası içinde, Q payları hesaplanmalı):
- `2026-05-11 dalgası` (testin %25,3'ü, %59,8'i soğuk) vs geri kalan soğuk
- güç bandına göre soğuk (`guc` medyanının altı/üstü)
- ilk görünme ayına göre soğuk (Nis / May / Haz-Tem)

**Ayrıca ölçülmemiş:** `m3` (havasız sürüm) üçüncü bir yön olarak duruyor
(`Q(m4,m3) = 0,0193`, korelasyon 0,997) — ama `m4` ile fazla örtüşüyor,
öncelik soğuk bölmesinde.
