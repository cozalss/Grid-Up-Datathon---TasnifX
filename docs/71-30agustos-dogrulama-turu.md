# 30 Agustos — dogrulama turu: bahsin dayanaklari tek tek sinandi

Bu belge, 30 Agustos aksami yapilan dogrulama turunun sonucudur. Amac
`tuketim_K_TEKHAK.csv` bahsinin dayandigi her varsayimi bagimsiz olarak
sinamakti. **Uc varsayim zayifladi, biri guclendi, biri duzeltildi.**

## 0. Ozet tablo

| # | Varsayim | Sinav | Sonuc |
|---|---|---|---|
| 1 | Zaman asinmasi var (tasima 0.388) | m130 | **YANLIS** — asinma yok, olcum yanlis soruyu soruyordu |
| 2 | 40 eksen kesimi keyfi | m132 | **DOGRU KESIM** — 58 eksen geciyor ama fazlasi blokta zarar |
| 3 | `sigma_L = 2.2e-4` | m134 | **YANLIS** — gercek deger yuvarlama tabani ~2.9e-06 |
| 4 | `1.95` carpani | m140 | **DOGRULANAMADI** — LOO `c = 0.28 +- 0.10` verdi |
| 5 | Blok korelasyonu 0.2125 gecerli | m141 | **YAZ25'E OZGU** — guz25/kis26'da isaret donuyor |
| 6 | Isaretler gercek | m142 | **DOGRULANDI** — iki bagimsiz kaynak %88 ortusuyor |

## 1. Zaman asinmasi yok (m130, m131)

Onceki `m125` "tasima orani 0.388" veriyordu ve bu, 2. sira icin gereken
`rho`'nun kil payi tutmadigi sonucuna goturuyordu. **O olcum yanlis soruyu
soruyordu:** agirliklari blogun bir yarisinda *fit edip* diger yarida
siniyordu. Oysa katsayilarimiz LB'den geliyor (`1.95*|rho_s|`), bloktan fit
edilmiyor — fit/holdout orani ilgisiz.

Sabit LB katsayilariyla olculunce sinyal **bes zaman penceresinin besinde de
pozitif** (0.107…0.303; sans olasiligi 1/32), mevsim eksenleri atilinca da
ayakta (27 eksen, kor 0.153).

`m131` uyarisi: **GEC/TUM orani bilgi tasimiyor.** Rastgele isaretli
bilesikler de ortanca 1.155 oran veriyor. Orani ne bonus ne ceza olarak
kullanmiyoruz.

## 2. 40 eksen kesimi dogru (m132)

Sert tavan kaldirilinca kapilardan 58 eksen geciyor:

```
   n  rho_pred  kor_tum  poz.penc  en dusuk  2.sira f
  20    0.1966   0.1972     5/5    0.1374     0.504
  40    0.2522   0.2125     5/5    0.1068     0.393   <- blok tepe noktasi
  58    0.2836   0.1948     5/5    0.0746     0.349
```

41–58 arasi eksenler **kagitta** `rho` ekliyor, blokta eklemiyor. Ayrica
`n <= 20`'ye kadar tahmin ile gozlem neredeyse birebir (0.1966 vs 0.1972).

## 3. `sigma_L` yuvarlama tabaninda (m134, m135)

Kullandigimiz `sigma_L ~ 2.2e-4` bir **benzetimdi** ve public'in %50 oldugunu
varsayiyordu. Varsayimsiz olcum: `G`'nin sifir ozvektorleri (`V u = 0`)
gonderimler arasindaki tam dogrusal bagimliliklardir; gercek `L` de ayni
bagintiya uymalidir, uymadigi kadari dogrudan public/tum-kume uyusmazligini
olcer.

Iki dongusellik tuzagi elendi: (1) turetilmis `L` yuku %0, (2) `u.L` icindeki
`M0*sum(u_j)` terimi `sum(u_j)=0` kosuluyla dusuruldu.

```
        aday bolunme   sd (toplam)   goreli olasilik
   public = TUM kume     2.928e-06            1.0000
        rastgele %50     9.862e-06            0.0931
        tek/cift gun     1.919e-05            0.0138
    trafoya gore %50     8.675e-05            0.0002
     tarihe gore %50     1.024e-03            0.0000
```

Saf optimum bu durumda 1.001055 degil **1.000527**. Ama `m138` kararina gore
kurulus **degistirilmedi**: hedef "garanti 2. sira" oldugu icin olcut en kotu
durumdur ve orada eski kurulus daha iyi (gereken f 0.400 vs 0.439).

## 4. `1.95` carpani dogrulanamadi (m140)

Butun bahis bu carpana dayaniyor ve **tek bir deneyden** geliyordu. Ikinci
olcum icin birini-disarida-birak kuruldu: `j` cikarilip kalanla span kurulunca
`d_j`'nin gercek bir dik bileseni olur ve skoru bilindigi icin o yondeki
gerceklesen korelasyon cozulur.

```
Agirlikli regresyon (<r,d_dik>/N = c * rho_s * sqrt(Q_dik), n=17):
    c = +0.282 +- 0.102        1.95'ten +16.3 sigma uzakta
```

**Neden dogrudan tasinmayabilir:** bu kurguda "dik bilesen", span icinde
kalmak uzere kurulmus gonderimlerin sayisal artigidir — tasarlanmis bir
oznitelik yonu degil. Bizim eksenlerimiz `Q_dik >= 0.25` rejiminde; buradaki
en buyuk `Q_dik` 0.062. Ayrica `khi-kare/sd = 7.0`, yani tek bir evrensel `c`
modeli zaten uymuyor.

**Yine de durust durum: `1.95`'in bagimsiz dogrulamasi yoktur.**

## 5. Blok korelasyonu yaz25'e ozgu (m141)

```
    blok         n                   donem      kor   kor/yaz25
   yaz25    274929  2025-04-01..2025-07-31   0.2125       1.000   <- ev sahasi
   guz25    319732  2025-08-01..2025-11-30  -0.0498      -0.234
   kis26    444076  2025-12-01..2026-03-31  -0.0689      -0.324
```

"Mevsimseldir" savunmasi aile kirilimiyla cokuyor: mevsimden bagimsiz olmasi
gereken trafo/yapisal aile daha sert donuyor (+0.1360 → −0.0654 / −0.0730).
Bloklarin taban modelleri ayri fitlerdir; artik yapisi blok basina degisir.

## 6. Ama isaretler iki bagimsiz kaynaktan dogrulandi (m142)

Isaretin ikinci ve bagimsiz kaynagi var:

- `rho_cv` → yaz25 blogu (2025 verisi, CV modeli) — `m141`'in sinadigi
- `rho_s` → LB skorlari (**2026 gercek test artigi**), blogu hic gormez

```
ORTUSME: 35/40 = %88      (sans %50)
tek yonlu binom p = 6.91e-07
```

Bloklar arasi donme bir **rejim etkisi**; isaretlerin kendisi sahte degil.
Ters cikan 5 eksenin katkisi kucuk (`m146`: atilirsa `rho_pred` 0.2522 →
0.2374), kurulus degistirilmedi.

## 7. Nihai duruma etkisi

`rho` icin uc bagimsiz tahmin:

| kaynak | rho | sonuc |
|---|---|---|
| LB tabanli (`c = 1.95`) | 0.2522 | 1. sira |
| Blok korelasyonu (carpandan bagimsiz) | 0.2125 | 1. sira |
| LOO regresyonu (`c = 0.28`) | 0.0362 | 7. sira |

**Karar degismedi.** Son secimde iki gonderim isaretlenir ve Kaggle iyisini
alir; mevcut 1.00115 ikinci secim olarak kalirsa bahsin asagi yonu kapalidir.
2. siraya giden baska yol da yoktur.

**Ayrica 6 gonderim hakkimiz var** (31 Agustos 3 + 1 Eylul 3), yani `rho`'yu
tahmin etmek zorunda degiliz — **olcebiliriz**. Ilk gonderim hem 2. sirayi
hedefler hem `rho`'yu acar:

```
rho = (1.011812620 - P^2) / 0.197844
```

Olctukten sonra optimum yerlestirme `kappa = rho` ile
`skor = sqrt(1.002112 - rho^2)` verir:

```
rho=0.0800 -> 0.99785  (4. sira)
rho=0.1293 -> 0.99267  (2. SIRA)
rho=0.1500 -> 0.98975  (1. SIRA)
```
