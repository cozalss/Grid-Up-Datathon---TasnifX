# GENIS SPAN YENIDEN KURULUMU — 31 Agustos 2026

> ## ⚠ EN GUNCEL KARAR (02:10) — asagidaki "136 eksen" ANLATISI ASILDI
>
> `n11` blok-disi olcumu, eksen eklemenin GERCEKLESEN `rho`'yu
> **DUSURDUGUNU** gosterdi. Sabah 136 eksene genisletmek YANLISTI.
>
> ```
> K=17   rho 0.1591  (+%29.8)
> K=25   rho 0.1642  (+%33.9)   <- TEPE, secildi
> K=40   rho 0.1451  (+%18.4)
> K=136  rho 0.1226  (taban)
> ```
>
> Kazanc %95 AO [+%12, +%56], P(iyi) = 1.00. Ongorulen `||BETA||` ise
> `0.2141 -> 0.4788` BUYUYOR: eksen eklemek **tahmini sisirip gerceklesen
> `rho`'yu dusuruyor**.
>
> `m148` artik `K_AZAMI` (varsayilan **25**) ile kirpiyor.
>
> **Guncel beklenti (`n14_birlesik_beklenti.json`):**
>
> | yapilandirma | medyan skor | P(1.) | P(2.) | P(ilk uc) |
> |---|---|---|---|---|
> | K=136, blok kazanci YOK | 0.99523 | %20.9 | %23.2 | %91.1 |
> | **K=25, blok kazanci YOK** | **0.99070** | **%40.9** | **%46.4** | **%97.5** |
> | K=25, blok kazanci VAR (OLCULMEDI) | 0.98450 | %59.1 | %65.6 | %99.1 |
>
> Asagidaki bolumler tarihsel kayittir; genis span calismasi bosa gitmedi
> — `H_carpim40` ailesini acmak, K egrisini olcebilmemizi sagladi ve
> kesim kararinin dayanagini olusturdu.


Bu belge, 31 Agustos gecesi yapilan **buyuk yeniden kurulumu** kaydeder.
docs/72'nin yerini alir; oradaki plan artik GECERSIZDIR.

---

## 1. NEDEN YENIDEN KURDUK

Iki bagimsiz sebep ayni anda ortaya cikti.

### (a) Esik beklenenden COK daha asagi gidiyor

Liderlik tablosu 31 Agustos 00:52 (yerel):

| # | takim | skor |
|---|---|---|
| 1 | Grid Grinders | 0.99009 |
| 2 | Abdulbaki Bayir | 0.99556 |
| 3 | Duo-Electra | 0.99614 |
| 4 | Berke Kus | 0.99927 |
| 5 | Atakan Aldemir | 0.99937 |
| 6 | Ahmet Celik | 1.00047 |
| 7 | Saban Ozdogan | 1.00049 |
| **8** | **TasnifX** | **1.00115** |

Esik zaman serisine dogrusal + ussel-sonumlu iki model uyduruldu
(`n02_esik_tahmini.json`). **Bitis (1 Eylul 23:59 UTC) tahmini:**

| hedef | merkez | %80 aralik |
|---|---|---|
| 2. sira | **0.9897** | [0.9870, 0.9908] |
| 1. sira | **0.9872** | [0.9776, 0.9903] |

Yani 2. sira icin gereken `toplam rho^2` **0.0109 degil 0.0225**.
Eski plan (docs/72) bu esige karsi **P(2. sira) = %15.6** veriyordu.
Yetersiz.

### (b) Kullanilmayan buyuk bir kaldirac vardi

`m144_yeni_aileler.json` icinde **kapidan gecmis 329 eksen** duruyordu.
`m148_demet_plani.py` bunlarin yalnizca **10'unu** kullaniyordu. Gerekce
30 Agustos'ta soyleydi: "tasiyici betige yarisma bitmeden 200 satirlik
ureteç GIRMEZ."

Bu gerekce YANLISTI. En verimli aile `H_carpim40` (285 eksen) su
adlandirmayi kullaniyor:

```
M[asiri_sicak:x_soguk]x[cdd22_ort14:ust10]
```

Yani **iki parca da `kur()`'un ZATEN bildigi adlar**. Ureteç gerekmiyor,
sekiz satirlik bir ozyineleme yetiyor.

---

## 2. NE DEGISTI

### 2.1 Genis span

`kur()` artik `M[a]x[b]` adlarini ozyinelemeyle insa ediyor ve aday listesi
m144'un kapidan gecen 329 ekseninin TAMAMI.

**Kapilar yeniden uygulanir** — m144'un kararina guvenilmez. Dongu her aday
icin `Qs >= 0.02`, `|rho_s| >= 0.015`, rcond kararliligi (1e-5 vs 1e-6, %30),
`Q_dik >= 0.25`, plasebo `|z| >= 3` ve tavan (`|rho_cv| >= 1.95|rho_s|`)
kapilarindan BASTAN gecirir.

```
369 aday -> 136 eksen kabul
bilesigin ongorulen rho: 0.2774 -> 0.4832   (%74 artis)
```

Kesim `Q_dik` kapisindan geliyor ve **kendiliginden** oluyor: her yeni aday
o ana kadar kabul edilmis TUM eksenlere diklestiriliyor, artigi %25'in
altina dusen giremiyor. Sert bir eksen tavani YOK.

### 2.2 Demet kurulusu: hipotez yerine AILE BLOKLARI

**Eski kurulus kusurluydu.** `BETA = toplam KATS[i]*U[i]` ve `GD_1` tam
olarak `BETA/||BETA||` idi — yani tahminimizin TAMAMI 1. sondadaydi.
Kalan uc sonda ongorulen `rho`'su **tam olarak sifir** olan saf kumarlardi
(`m148_demet.json`: `1.1e-16, 4.2e-17, 2.5e-16`).

Yeni kurulus BETA'yi dik bloklara boler:

```
                              blok  eksen  ||BETA_b||
                   H_carpim40/hava     78      0.3758
                   m121_taban/hava     26      0.2028
                   H_carpim40/yapi     18      0.1691
                   m121_taban/yapi     14      0.1500
                            TOPLAM    136      0.4832
```

**Neden baskin:** ortonormal bloklarda

```
toplam_k rho_k^2  >=  rho_BETA^2
```

her zaman saglanir; esitlik ANCAK bloklar arasi goreli agirliklandirmamiz
tam isabetliyse olur. Bolme KAYIPSIZ ve tipik olarak KAZANCLI. Bedeli
yalnizca blok basina bir olcum hatasi (asagiya bakiniz — ihmal edilebilir).

Artik **dort sondanin dordu de gercek sinyal tasiyor**, eskiden biri
tasiyordu.

### 2.3 Kappa yeniden optimize edildi

Eski `0.0517` tekduze degeri DAR SPAN doneminde (ongorulen toplam 0.2774)
secilmisti. Genis spanda blok basina ongorulen `rho` neredeyse iki kat.
`n06_kappa.py` blok basina optimize eder:

| blok | ongorulen | medyan gercek rho | kappa* |
|---|---|---|---|
| 1 | 0.3758 | 0.1098 | **0.130** |
| 2 | 0.2028 | 0.0593 | **0.070** |
| 3 | 0.1691 | 0.0494 | **0.060** |
| 4 | 0.1500 | 0.0438 | **0.055** |

Olcum kaybi `1.11e-05 -> 6.45e-06 rho^2` (gereken 0.0225'in %0.05'inden
%0.03'une). Asil kazanc olcum degil, **her sonda dosyasinin kendi yedek
degeri**: blok 1 icin `+0.0174` (eskiden `+0.0087`).

Zincir buyutme carpani (kirmizi takim K1'in itirazi) en fazla **2.0** —
reddedilen deger 11.8 idi, guvenli bolgede.

---

## 3. BEKLENEN SONUC

`skor^2 = TABAN_MSE - toplam rho_k^2`, `TABAN_MSE = 1.00202690`.
`gerceklesen rho = |c| * ongorulen / 1.95`, `|c|` log-normal
(medyan 0.57, %90 GA [0.17, 1.26]).

| \|c\| | toplam rho^2 | NIHAI SKOR |
|---|---|---|
| 0.00 | 0.00000 | 1.00101 |
| 0.30 | 0.00553 | 0.99825 |
| 0.57 | 0.01995 | **0.99100** |
| 0.70 | 0.03008 | **0.98587** |
| 0.81 | 0.04028 | **0.98069** |
| 1.00 | 0.06139 | 0.96986 |

**Bitisteki tahmini esige karsi olasiliklar (MODEL A ile):**

| | eski (dar span) | yeni (genis span) |
|---|---|---|
| medyan skor | 0.99773 | 0.99101 |
| P(1. sira) | %12.0 | %39.6 |
| P(2. sira) | %15.6 | %46.0 |
| P(ilk uc) | %69.9 | %92.4 |

### ⚠ BU SAYILAR HENUZ DOGRULANMADI — iki rakip model var

Yukaridaki tablo **Model A**'ya dayanir: `gerceklesen rho = |c| * ||BETA|| / 1.95`,
yani eksen eklemek dogrudan `rho`'yu buyutur.

**Model B (doyum)** ise buna karsi cikiyor. Yarim kalan bir blok-disi olcum
(`n01_K_asiri_uyum.json`) su egriyi verdi:

| K (eksen) | ongorulen rho | **gerceklesen rho** | oran |
|---|---|---|---|
| 10 | 0.137 | 0.126 | 0.92 |
| 25 | 0.201 | **0.149** | 0.74 |
| 50 | 0.278 | 0.123 | 0.44 |
| 63 | 0.300 | 0.124 | 0.41 |

Yani gerceklesen korelasyon `K ~ 25`'te **doyuyor**; daha fazla eksen
yalnizca TAHMINI sisiriyor. Bu dogruysa 136 eksene genisleme `rho`'yu
artirmaz.

**IKI MODEL MEDYANDA ANLASIYOR** (`n13_iki_model.py`):

| model | medyan rho | medyan skor | P(1.) | P(2.) | P(ilk 3) |
|---|---|---|---|---|---|
| A (\|c\| carpani) | 0.141 | 0.99101 | %39.6 | %46.0 | %92.4 |
| B (doyum) | 0.130 | 0.99249 | %20.0 | %23.4 | **%99.8** |

Anlasmazlik **ust kuyrukta**: doyum dogruysa 1. sira olasiligi coker ama
ilk uc neredeyse kesinlesir. **Her iki modelde de en kotu durum ~1.00101
(saf span), yani bugunku 1.00115 yedegimizden IYI — plan asagi yonlu
korumalidir.**

Karar `n09_K_karari.json` ile verilecek. Sonuca gore `K_AZAMI` ortam
degiskeniyle eksen sayisi tek komutla kirpilir.

### GUNCELLEME 01:45 — `|c|` OLCULDU, tablo degisti

`n10` `|c|`'yi **LB'nin kendi 29 olcumu uzerinde** birak-birini-disarida
ile olctu (vekil blok kullanmadan, span cebirinin tam ic carpimlariyla):

```
|c| = 0.43   %90 GA [0.18, 0.80]     (eskiden 0.57 [0.17, 1.26])
```

Merkez daha DUSUK, aralik daha DAR. Iki yan bulgu:

- **`sigma_L` dogrudan olculdu.** `G`'nin uc TAM SIFIR kipi var; `Vu = 0`
  oldugu icin `u'L = 0` olmak ZORUNDA ve gozlenen sapma saf olcum
  hatasidir: **2.94e-06**, LB yuvarlamasinin 1.02 kati. m112'nin varsaydigi
  `2.27e-04` (77 kat buyuk) **veriyle reddedildi**. Ayni kipler
  `|ΔM0| <= 4e-06` siniri da koyuyor; `M0 = 1.005846366` bu testi geciyor.
- **`1.95` carpani pratikte dislandi:** `P(|c| >= 1.95) = 0.0004`.
  m148 bundan **zarar gormez** (`rho`'yu LB'de olcer, katsayiyi olcumden
  koyar). Ama m117–m125 ailesinden hazir bir dosya gonderilseydi
  katsayilari `rho_s^2 (2·1.95·0.43 − 1.95^2) = −2.13 rho_s^2`, yani
  **negatif kazanc** verirdi.

**Olculen `|c|` ile guncel tablo:**

| durum | medyan skor | P(1.) | P(2.) | P(ilk uc) |
|---|---|---|---|---|
| blok kazanci HARIC | 0.99344 | %20.5 | %23.4 | **%95.4** |
| blok kazanci dahil (OLCULMEDI) | 0.98924 | %43.7 | %52.7 | %98.1 |

Iki model (`|c|` ve doyum) olculen `|c|` ile **birbirine yakinsadi** —
ikisi de P(2.) ~ %23 veriyor.

**Okunusu:** 3. sira (basari esigi) **%95+ ile hemen hemen kesin**.
2. sira tamamen **blok bolmesi kazancinin** buyuklugune bagli, ve o kazanc
henuz olculmedi. `n09` tam onu olcuyor.

`kappa` da olculen degerlerle yeniden turetildi:
**`[0.052, 0.050, 0.045, 0.040]`** (zincir buyutmesi en fazla 2.09).

**Doyum dogrulanirsa tasarim sonucu:** darbogaz eksen sayisi degil
AGIRLIKLANDIRMA hatasidir. m148'in blok bolmesi tam da onu onarir
(bloklar arasi agirligi LB secer). O durumda **blok sayisini 4'ten 5'e
cikarmak** cok daha degerli hale gelir -- yedek gonderim hakki harcanarak.
Bu da olculuyor.

---

## 4. KAPANAN SORU: span'e daha cok dosya katilabilir mi?

**Hayir.** `TABAN_MSE = M0 - 2kL + ||r_hat||^2` ve span `V`, LB skoru
BILINEN gonderimlerden kuruluyor. Diskte ~200 CSV var ama Kaggle
gecmisinde **toplam 30 gonderim** yapilmisiz ve bunlarin **29'u zaten
span'de** (`olculmus_skorlar.json` 27 + `m112_durum.json` 2). Gerisi hic
gonderilmedigi icin skoru yok, span'e katilamaz.

Bu kaldirac tuketilmistir.

---

## 5. HENUZ ACIK OLANLAR

| soru | durum |
|---|---|
| `\|c\|` eksen sayisi 50->136 olunca bozuluyor mu? | olculuyor |
| Ikinci nesil carpimlar (136 eksenin ikili carpimlari) `\|\|BETA\|\|`'yi daha da buyutur mu? | taraniyor |
| `Q_dik` esigi 0.25 dogru mu, gevsetmeli mi? | olculuyor |
| Blok bolmesi "aile" mi "oran" (\|rho_cv\|/\|KATS\|) mi olmali? | olculmedi, statuko "aile" |
| kappa 0.130'da KIRPMA sapmasi oz-denetimi geciyor mu? | sinanacak |

---

## 6. DEGISMEYEN KURALLAR

- **ONAY OLMADAN HICBIR GONDERIM YAPILMAZ.** Komut kullaniciya verilir.
- Gonderimden SONRA gonderim listesi OKUNUR (zaman asimina ugrayan betik
  "gonderilmedi" demek degildir).
- Nihai 2 gonderim secimi YALNIZCA tarayicidan yapilir. Once
  "You selected X of N" satiri OKUNUR.
- Yedek secim: `tuketim_YP_seviye.csv` (1.00115).
- Kota 3/gun, 00:00 UTC = 03:00 yerelde sifirlanir.
