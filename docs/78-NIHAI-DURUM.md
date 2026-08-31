# NIHAI DURUM — 31 Agustos 2026, sabah

Bu belge **gecerli olan tek durum kaydidir**. docs/75 tarihsel kayittir
(icindeki "136 eksen" ve "%46" sayilari ASILDI).

Uygulama adimlari: **docs/76** · Karar agaci: **docs/77**

---

## 1. NEREDEYIZ

Liderlik tablosu (31 Agustos 09:15):

| # | takim | skor |
|---|---|---|
| 1 | Grid Grinders | **0.98110** |
| 2 | Duo-Electra | **0.99536** |
| 3 | Abdulbaki Bayir | 0.99556 |
| 4 | Tuna Deniz | 0.99886 |
| 5 | Berke Kus | 0.99927 |
| ... | | |
| **10** | **TasnifX** | **1.00115** |

**1. sira MENZIL DISI.** Grid Grinders 30 Agu 23:44'te tek gonderimde
0.99009 -> 0.98110 sicradi. Bizim en iyimser senaryomuz 0.97697, yani
teorik olarak mumkun ama yalnizca en ust `|c|` degerinde.

**2. sira esigi 0.99536.** Sicramalar SEYREK DEGIL: son ~10 saatte ilk
12'de dort sicrama (Grid Grinders -0.00899, Tuna Deniz -0.00381,
Duo-Electra -0.00037, Ahmet Bugrahan -0.00143).

---

## 2. YAPILANDIRMA — hepsi olculdu

| ayar | deger | dayanak |
|---|---|---|
| eksen sayisi `K_AZAMI` | **25** | n11/n09 — cokus sigortasi (asagida) |
| blok sayisi `DEMET_HEDEF` | **4** | n09 — B=5 kazanci P=%54, yedek hakki yakar |
| blok bolmesi `BLOK_KIP` | **oran** | {hava,yapi} × {\|rho_cv\|/\|KATS\| yuksek,dusuk} |
| bloklar `\|\|BETA_b\|\|` | [0.1302, 0.1162, 0.0904, 0.0849] | toplam **0.2141** |
| `kappa` | [0.035, ...] | n06, olculen \|c\| ile |
| taban dosya | `tuketim_D1_demet.csv` | md5 `056cd5a2aea181ebd876f5daf00f2abd` |
| yedek secim | `tuketim_YP_seviye.csv` | 1.00115 |

Demetlerin dikligi: en buyuk sapma **1.33e-15**.

---

## 3. OLCULEN NICELIKLER

| nicelik | deger | nasil olculdu |
|---|---|---|
| `TABAN_MSE` | 1.00202690 | `M0 - 2kL + \|\|r_hat\|\|^2` |
| `\|c\|` | **0.434** %90 GA [0.184, 0.798] | n10 — LB'nin 29 olcumu, birak-birini-disarida |
| `sigma_L` | **2.94e-06** | n10 — G'nin uc TAM SIFIR kipinde `u'L=0` olmak ZORUNDA |
| K orani (B=1) | 1.391 | n09 — K=25 / K=136 |
| blok kazanci | 1.271 | n09 — B=4 / B=1, K=25 |
| bitis esigi (2.) | **0.98674** zarf [0.98319, 0.99536] | n17 — sicrama riski dahil |

### Yikilan iki varsayim

- **`1.95` carpani pratikte disland**i: `P(\|c\| >= 1.95) = 0.0004`.
  m148 bundan zarar gormez (`rho`'yu LB'de olcer). Ama m117–m125
  ailesinden hazir bir dosya gonderilseydi katsayilari
  `rho_s^2 (2·1.95·0.43 − 1.95^2) = −2.13 rho_s^2` — **negatif kazanc**.
- **m112'nin `sigma_L = 2.27e-04` varsayimi reddedildi** (77 kat buyuk).

---

## 4. NEDEN K=25 (gerekce DEGISTI)

Once "K=25 gerceklesen rho'yu %34 artiriyor" sanildi. Bu **yalnizca tek
bilesik (B=1) icin** dogru. n09 dort bloklu kurulumu olctu:

| K | B=1 | **B=4 (bizim plan)** | TAVAN |
|---|---|---|---|
| 25 | 0.1427 | **0.1814** | 0.2946 |
| 136 | 0.1026 | **0.1871** | 0.4236 |

B=4'te K=25 ile K=136 arasinda **fark yok** (P=0.67). K=25'in gerekcesi
**cokus sigortasi**: zincir kirilip sabit-agirlik bilesigine dusersek
K=25 bize 0.143, K=136 yalnizca 0.103 verir.

**Asil darbogaz agirliklandirma.** Tavan (0.2946) ile B=4 (0.1814)
arasindaki bosluk gercek, ama n15 bunun **kapatilamadigini** gosterdi:
vekilin gurultusu karar esiginin 6 kati; fit yariminda tavanin %93'unu
yakalayan bolme, olcum yariminda statukonun ALTINA dusuyor. Bu bosluk
**acik kalem** olarak birakildi.

---

## 5. BEKLENEN SONUC — TEK BILINMEYENE INDI

`n18` ve `n19` gosterdi ki onceki "dort bagimsiz yol" aslinda **tek
parametrenin dort degeri**. Cebir sadelesiyor:

```
gerceklesen rho_LB = |c| * rho_s(bilesik)
rho_s(bilesik) = ||BETA|| / 1.95 = 0.2141 / 1.95 = 0.10979
```

`KATS[i] = 1.95*|rho_s_i|` oldugu icin `rho_s(bilesik)` DOGRUDAN bilinir.
Geriye tek bilinmeyen kaliyor: **`|c|`**.

| `\|c\|` | kaynak | `rho` | NIHAI SKOR | siralama |
|---|---|---|---|---|
| 0.184 | n10 %90 GA alt uc | 0.0202 | 1.00081 | kucuk kazanc |
| **0.434** | n10 nokta (n=19, **farkli nesne**) | 0.0477 | 0.99988 | kucuk kazanc |
| 0.798 | n10 %90 GA ust uc | 0.0876 | 0.99717 | 3.-4. |
| 1.320 | `CARPAN 0.798`'in ima ettigi | 0.1449 | 0.99047 | **2. sira** |
| **1.986** | seviye ekseni (n=1, **dogru nesne**) | 0.2181 | 0.97697 | **1. SIRA** |

### Iki capa, iki kusur — hicbiri digerini baskilamiyor

- `|c| = 1.986` — **n=1** ama **DOGRU nesnede** olculdu (`seviye`, bir
  OZNITELIK EKSENI; bizim demet yonlerimizle ayni turden).
- `|c| = 0.434` — **n=19** ama **FARKLI nesnede** (gonderim FARKI yonleri).
  `n10` kendi raporunda "oznitelik eksenlerine tasindigi GOSTERILMEMISTIR"
  diye uyariyor.

Biri kucuk ornek, digeri yanlis nesne. **4.6 kat ayrisiyorlar.**

`n18`'in ek bulgusu: `CARPAN = 0.798`'in icinde `1.95` **gomulu**
(`CARPAN = |c| * T`, seviye icin `1.9864 * 0.4016`). Yani `0.798`, `n10`'un
reddettigi degeri tasiyor. Ayrica `m148`'in TAVAN kapisi da bagimsiz kanit
TASIMIYOR: kapi `<=>` `T_j <= T(seviye)`, yani ayni n=1 kalibrasyonun tekrari.

### DEGISMEYEN — asagi yonlu koruma

**Her senaryoda skor `<= 1.00101 < 1.00115` (yedegimiz).** Gonderim hicbir
durumda bir sey KAYBETTIRMEZ; ya kazandirir ya kazandirmaz.

**D1 tam bu sayiyi olcer:** `|c| = 1.95 * rho_1 / 0.1302`, ve `sigma(|c|)
= 0.066` -- iki capa arasinda 23 sigma var. **Tek gonderim, alti katlik
belirsizligi kapatir.**

---

## 6. KIRMIZI TAKIM — bulunan ve duzeltilenler

| # | bulgu | durum |
|---|---|---|
| K8 | `m161` zincir sinamasi **dongusel**di: gercek skoru m148'in kendi formuluyle uretiyordu, hicbir hata yakalayamazdi | **`m162` yazildi** — skor sentetik gercek hedeften DOGRUDAN |
| K3 | Iki oz-denetimin ikisi de **cebirsel ozdeslik**ti (`artik` = 2.7e-17, her zaman gecer) | Yerine **CSV geri okuma** — gonderilen sey diskteki dosyadir |
| K1 | Kayit/CSV desenkronizasyonu hic kontrol edilmiyordu (0.00153 skor kaybi gozlendi) | Tutarlilik kontrolu, saparsa DURUYOR |
| K2 | `m148_olcumler.json`'daki 0.99976 **gercek LB skoru degildi**, m161'in sentetigiydi | `OLCUM_DOSYA` ile ayrildi |
| K5 | `\|rho\| > 0.20` kapisi **kazandigimiz** senaryoda duruyordu | 0.40 |
| K4 | `kappa_etkin = \|\|ek\|\|` yaziliyordu, cebir `<ek,GD_k>` istiyor | duzeltildi, dik bilesen hata butcesinde |
| K7 | Olu `_YM` maskesi ve yanlis yorum | kaldirildi |

**Temiz cikanlar:** Gram-Schmidt dikligi 4.8e-16, capraz terim cebiri,
sirasiz sonda kapisi, yinelenen eksen yok, atomik CSV yazimi.

### `m162` — BAGIMSIZ DOGRULAMA GECTI

Sentetik gercek artik `r_syn` kuruldu (`<r_syn,r_hat>/N = kL`,
`<r_syn,GD_k>/N = rho_k`, `ort(r_syn^2) = M0`), gercek log hedef
`t = a0 + r_syn`, ve her dosyanin skoru **dogrudan** `sqrt(ort((log1p(CSV) - t)^2))`
ile hesaplandi — m148'in hicbir formulu kullanilmadan.

Sentetik gercek `rho = [0.09, -0.05, 0.04, 0.02]`:

| sonda | gercek | cozulen | hata |
|---|---|---|---|
| 1 | +0.0900 | +0.0901 | +1.4e-04 |
| 2 | -0.0500 | -0.0503 | -3.1e-04 |
| 3 | +0.0400 | +0.0389 | -1.1e-03 |
| 4 | +0.0200 | +0.0204 | +4.1e-04 |

```
NIHAI DOSYANIN GERCEK SKORU  = 0.994702   (dogrudan)
BETIGIN BILDIRDIGI BEKLENTI  = 0.994710   (fark -8.5e-06)
KUSURSUZ OLCUMLE ULASILABILIR = 0.994699  (fark +2.1e-06)
```

Bu; capraz terim cebirini, isaret islemesini, kirpmayi, CSV gidis-donusunu
ve nihai birlestirmeyi BAGIMSIZ bir gercek karsisinda dogrular. Kalan
~1e-3 hata LB'nin 5 ondalikli yuvarlamasindandir, beklenen buyukluk.

---

## 7. YARIN

Adim adim komutlar **docs/76**'da. Ozet:

1. Kotayi ve md5'i dogrula, tarayicida **"You selected X of N"** oku
2. `D1` gonder → skoru `m148_olcumler.json`'a yaz
3. **`|c|` kalibrasyonu**: `C_OLCULEN=<deger> python n06_kappa.py` → `m148`
4. `D2`, `D3`, `D4` ayni dongu
5. `NIHAI=1 python m148_demet_plani.py` → `Z_NIHAI` gonder
6. 6. hak: **uyarlanabilir 5. sonda** (docs/77) ya da yedek — D4'ten sonra karar
7. Tarayicidan 2 secim: `Z_NIHAI` + `YP_seviye`

**ONAY OLMADAN HICBIR GONDERIM YAPILMAZ.**

---

## 8. SON KONTROL — kappa TUM `|c|` ARALIGINDA SAGLAM

`n06` kappa'yi `|c| = 0.434` onseliyle secti. Ama `n18`/`n19` gosterdi ki
`|c|` 0.184 ile 1.986 arasinda olabilir. Kappa o aralikta bozuluyor mu?

| `\|c\|` | `rho_1` | `sigma(rho_1)` | goreli | `sigma(\|c\|)` | D1 skoru |
|---|---|---|---|---|---|
| 0.184 | 0.0123 | 0.00443 | %36.0 | 0.066 | 1.00120 |
| 0.434 | 0.0290 | 0.00443 | %15.3 | 0.066 | 1.00061 |
| 0.798 | 0.0533 | 0.00443 | %8.3 | 0.066 | 0.99976 |
| 1.320 | 0.0881 | 0.00443 | %5.0 | 0.066 | 0.99854 |
| 1.986 | 0.1326 | 0.00443 | %3.3 | 0.066 | 0.99698 |

**Sonuc: DEGISIKLIK GEREKMIYOR.**

- `sigma(|c|) = 0.066` her durumda ayni. Iki capa (0.434 ve 1.986) arasinda
  1.55 fark var, yani **23 sigma** -- D1 ikisini kesin ayirir.
- Toplam olcum kaybi `9.26e-05`, 2. sira icin gereken `0.01129`'un
  **%0.82'si**. Ihmal edilebilir.
- En kotu durumda (|c| = 0.184) D1'in kendi skoru 1.00120, yani yedegimizden
  (1.00115) az kotu. Ama D1 bir SONDADIR, secilecek dosya degil; yedek
  isaretli kaldigi surece bu bir kayip degildir.

Bu, acik kalan son operasyonel soruydu.
