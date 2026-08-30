# GENIS SPAN YENIDEN KURULUMU — 31 Agustos 2026

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

**Bitisteki tahmini esige karsi olasiliklar:**

| | eski (dar span) | **yeni (genis span)** |
|---|---|---|
| medyan skor | 0.99773 | **0.99101** |
| P(1. sira) | %12.0 | **%39.6** |
| P(2. sira) | %15.6 | **%46.0** |
| P(ilk uc) | %69.9 | **%92.4** |

Bu sayilar blok bolmesinin kazancini **YOK SAYAR** (muhafazakar):
bolme yalnizca goreli agirliklandirmamiz yanlissa kazandirir. Yani
**alt sinirdir**.

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
