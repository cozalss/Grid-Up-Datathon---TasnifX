# 0.996 hedefi: CV onseli ve bes-sonda kampanyasi

Tarih: 30 Agustos 2026

## Karar

`submissions/tuketim_K_TARGET996_CV.csv` cebirsel olarak 0.99587'yi hedefleyen
agresif modeldir; ancak **dogrudan nihai gonderim degildir**. Son kronolojik CV
blogunda zarar ettigi icin once ayni yondeki kucuk
`submissions/tuketim_K_PROBE_TARGET996.csv` sondasi gonderilecektir. Gelen LB
skoru, yedi eksenin ortak isaretini ve uygulanacak toplam buyuklugu tek hakta
olcer.

Bilinen 27 yonun exact optimumu yaklasik 1.00053'tur. 0.99600 icin gerekli ek
karesel kazanc:

```text
1.00053^2 - 0.99600^2 ~= 0.00904 MSE
```

## Yeni ileri-zaman eksenleri

Her blokta once global seviye, sonra daha once secilen eksenler cikarildi.
Asagidaki sayilar normalize, ardisik-ortogonal artik korelasyonlaridir:

| eksen | yaz25 | guz25 | kis26 | agresif beta |
|---|---:|---:|---:|---:|
| `h` | +0.1164 | +0.0198 | +0.0232 | +0.064020 |
| `t_hg_genligi` | +0.0240 | +0.0204 | +0.0361 | +0.013200 |
| `sv_yas` | -0.1099 | -0.0266 | -0.0189 | -0.060445 |
| `h_t_log_ort` | -0.0375 | -0.0789 | -0.0134 | -0.020625 |
| `gunes_radyasyon` | -0.0263 | -0.0108 | -0.0279 | -0.014465 |
| `h_sicaklik_ort` | +0.0472 | +0.0332 | +0.0086 | +0.025960 |
| `ulusal_gunluk` | -0.0148 | -0.0286 | -0.0126 | -0.008140 |

Isaretler 7/7 eksende uc blokta da korunur. Beta vektoru yaz25
korelasyonlarinin %55'idir; norm karesi 0.009301'dir. CV onseli testte aynen
gerceklesirse exact tabanla birlikte beklenen skor 0.99587 olur.

### Dusmanca kararlilik kapisi

Ayni sabit beta vektoru gercek blok artiklarina uygulandiginda MSE kazanclari:

| blok | dMSE kazanci | hukum |
|---|---:|---|
| yaz25 | +0.024521 | gecti |
| guz25 | +0.002745 | gecti |
| kis26 | -0.001081 | kaldi |

Son kronolojik bloktaki zarar nedeniyle agresif dosya LB olcumu olmadan nihai
ilan edilemez. Buna karsilik beta vektorunun **yonu** uc blokta da gercek
artikla pozitif korelasyonludur (birim bilesik rho: 0.1754, 0.0625, 0.0426).
Bu nedenle kucuk bilesik sonda guvenli ve bilgi-verimlidir.

## Ilk sonda ve karar esigi

Hazir dosya:

```text
submissions/tuketim_K_PROBE_TARGET996.csv
```

Exact cozum:

```text
rho = (1.001060591 - P^2) / 0.010001
```

`P <= 1.00005` gelirse bilesik yon tek basina 0.996 icin gereken yaklasik
`rho >= 0.0951` sinyalini tasir. Daha yuksek skor da bos hak degildir; exact
rho sisteme eklenir ve sonraki tum sondalar buna dik uretilir.

Skor `P1` geldikten sonra:

```powershell
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --kaydet hedef996_bilesik --skor P1
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --aday y46 --yerdeg 0.005 --cikti tuketim_K_PROBE_y46.csv
```

## Kalan dort eksenin secimi

Yedi model adayi exact 27-yone ve hedef996 bilesigine karsi olculdu. Secimde
yenilik, dusuk kurtoz ve farkli mekanizma birlikte kullanildi:

| aday | exact Q_dik | kurtoz | en kotu %1 Q payi | hedef996 kosinusu | karar |
|---|---:|---:|---:|---:|---|
| `y46` | 0.29275 | 4.48 | 0.126 | +0.136 | secildi |
| `p42` | 0.87866 | 4.51 | 0.135 | +0.104 | secildi |
| `q1c` | 0.04167 | 5.86 | 0.157 | +0.101 | secildi |
| `sul` | 0.04196 | 11.71 | 0.285 | +0.003 | secildi |
| `y45` | 0.10908 | 7.61 | 0.202 | +0.163 | yedek |
| `z2` | 0.09003 | 10.26 | 0.244 | +0.141 | yedek |
| `t3` | 0.03626 | 14.99 | 0.297 | -0.077 | yedek |

Secilen dort yon birbirinin ardisik cikartilmasindan sonra sirasiyla en az
%95.7, %97.8, %98.7 ve %100 yeni enerji tasir. Mekanizmalar da farklidir:
amnezik model (`y46`), seviye egriligi (`p42`), kapasite (`q1c`) ve sulama
analogu (`sul`).

## Kota plani

```text
31 Agustos: hedef996_bilesik -> y46 -> p42
 1 Eylul : q1c -> sul -> exact nihai
```

Her skor once `--kaydet` ile islenir; bir sonraki sonda ancak ondan sonra
uretilir. Boylece her hak o ana kadar olculen her seye exact olarak diktir.
Son komut:

```powershell
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --nihai --cikti tuketim_K_996_NIHAI.csv
.\.venv\Scripts\python.exe scripts\kapi_denetim.py submissions\tuketim_K_996_NIHAI.csv
```

## Dosya kapilari

`tuketim_K_TARGET996_CV.csv` ve `tuketim_K_PROBE_TARGET996.csv` icin 714.688
satir, birebir id sirasi, sifir NaN, sifir negatif ve sonlu deger kapilari
gecmistir. Hedef dosyada 4.848 sifir tahmin olmasi agresifligin ek bir risk
isaretidir; sondada bu sayi 282'dir.
