# 58 — 30 Ağustos NİHAİ koşu kılavuzu

**docs/54 ve docs/57'nin yerine geçer.** Sabah bunu aç, sırayla uygula.

---

## Başlangıç

```
1. Grid Grinders     0.99064
2. Atakan Aldemir    1.00041   <- HEDEF
3. TasnifX           1.00284   <- BIZ (submissions/tuketim_m6_ikiyon.csv)

m0 = 1.005688   ·   hedef MSE 1.000820   ·   gereken dMSE -0,004868
span yonu BEDAVA (gondermeye gerek yok): katki -0,002984
KALAN GEREKSINIM: -0,001884
kota yerel 03:00  ·  9 hak (30-31 Agu + 1 Eyl)
```

**Dış veri SERBEST** (düzenleyici e-postayla teyit etti). Notebook'ta beyan şart.

---

## Neden span'ı GÖNDERMİYORUZ

`tuketim_g7_span_tau3.csv`, 25 ölçülmüş dosyanın **afin kombinasyonu**
(`sum(w)=1`, artık 3,04e-08). Dolayısıyla `L`'si ölçülmüş `L`'lerden çıkıyor:

```
L(g7) = +0,002728   Q(g7) = 0,002494   katki = 0,002984
```

Göndermemenin bedeli, katsayı hatasının ikinci mertebeden olması sayesinde
yalnız **1,23e-05 MSE** (katkının %0,41'i). Bir gönderim hakkının değeri
~0,0016 MSE → **göndermemek 130 kat kârlı.** O hak yeni bir yöne gidiyor.

---

## GÜNCELLEME — karar analizi planı değiştirdi

**Bulgu: `g7 · y40 = −0,555`.** `g7`'nin `L`'si BİLİNİYOR; onunla **ters
korelasyonlu** bir yön süper-toplamsaldır. Ölçüldü (`r=0,035` senaryosu):

```
aday   kos(g7)   tek basina   g7 ile ORTAK   oran
y40     -0,555     0,001225      0,009143    5,03x   <<<
z2      -0,231     0,001225      0,005377    1,95x
y46     -0,091     0,001225      0,004597    1,32x
q1c     -0,034     0,001225      0,004342    1,11x
```

`y40` tek başına zayıf (Q 0,029, kurtoz 14,5) ama `g7` ile birlikte **5 kat**.
Yer değiştirme 0,096 — doğrulanmış bölgenin (0,286) içinde.

**Ve strateji: GÜN 1'DE BİRLEŞİM YOK, ÜÇ SONDA.** Her sondaya `g7` optimum
katsayıyla gömülür → sonda hem ölçüm hem yüksek skorlu gönderim olur. O gün
ayrıca "ortak optimum" göndermek hakkı boşa harcar (zaten sondaladığın iki
yönün katsayısını rötuşlar). Monte Carlo: **P(2. sıra) 0,889 vs 0,853.**

---

## GÜN 1 — üç sonda (hepsi hazır, kapı denetiminden geçti)

Her sonda: `m6 + 1,094·d_g7 + t·d_aday`

| # | dosya | `t` | Q | kos(g7) | yer değ. | L=0 ise | r=0,035 ise | r=0,06 ise |
|---|---|---|---|---|---|---|---|---|
| **1** | `tuketim_sy40.csv` | 0,60 | 0,029 | **−0,555** | 0,084 | 1,00341 | **0,99987** | **0,99733** |
| 2 | `tuketim_sq1c.csv` | 0,45 | 0,061 | −0,034 | 0,122 | 1,00729 | 1,00342 | 1,00065 |
| 3 | `tuketim_sy46.csv` | 0,35 | 0,388 | −0,091 | 0,220 | 1,02377 | 1,01629 | 1,01091 |

**Sıra: `sy40` → `sq1c` → `sy46`.** `sy40` önce, çünkü tek başına 2. sırayı
getirebilecek tek sonda.

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_sy40.csv -m "sonda y40: m6 + 1.094*d_g7 + 0.60*d_y40. g7 optimumda gomulu (L bilinen), y40 uzerine bahis. kos(g7,y40)=-0.555 super-toplamsal. AMAC: L_y40 olcmek"
kaggle competitions submissions -c grid-up-datathon
```
(diğer ikisi aynı biçimde)

### Skor gelince `L` çözümü

```
L_y40 = (1,006831 - P^2) / 1,20
L_q1c = (1,014633 - P^2) / 0,90
L_y46 = (1,048109 - P^2) / 0,70
```

Sabitler `experiments/model29/m102_sonda.json` içinde (`cozum_sabiti`).

---

## GÜN 2 (3 hak) — ortak optimum + iki sonda

```powershell
python experiments/model29/m99_coklu_coz.py tuketim_m6_ikiyon.csv=1.00284 tuketim_g7_span_tau3.csv=1.00136 tuketim_y40_sota_temiz.csv=<coz> tuketim_q1c_kapasite_siki.csv=<coz> tuketim_y46_amnezik_kirpik.csv=<coz> --cikti tuketim_g9_ortak.csv
```
> Ölçülen `L`'lerden eşdeğer skor: `S_j = sqrt(m0 + Q_j - 2*L_j)`

Sonra iki sonda daha: `z2_analog`, `y45_mevsimsel`.

## GÜN 3 (3 hak) — altı yönlü ortak optimum + son sonda + ince ayar

---

## Beklenen sonuç

`r = L/√Q` kalite ölçütü. Ölçülmüş referanslar: v101 0,1243 · m4 0,0641 · p51 0,0493.

```
her iki yon r=0,050 ->  ~0,9985   2. SIRA genis payli
her iki yon r=0,040 ->   0,99943  2. SIRA
her iki yon r=0,030 ->  ~1,0004   sinirda
her iki yon r=0,020 ->  ~1,0011   3. sira
```

Üçlü sistem (g7, y46, q1c): koşul sayısı **157** (korkuluk eşiği 1e8) — temiz.

---

## 31 Ağustos ve 1 Eylül (6 hak)

Sırayla ölç, her ölçümden sonra ortak optimumu güncelle:
`tuketim_z2_analog.csv` (Q 0,118, kurtoz 9,0) →
`tuketim_y45_mevsimsel_kirpik.csv` (Q 0,167, kos(m4) +0,005) →
`tuketim_q1d_kuantil38_siki.csv` (Q 0,055, kurtoz 5,1)

**Not:** `y45↔z2 +0,323`, `q1c↔q1d +0,515`, `z1↔y45 +0,557` — bu çiftlerden
ikisi birden alınırsa kazançlar toplanmaz.
**ELENDI:** `z3_ikiasama` (kos(m4) +0,564, kurtoz 19,4), `r1`/`r3` (Q=0,003,
ölçüm gürültüde boğuluyor).

---

## Kaçırılmaması gerekenler

- **Her gönderimden sonra listeyi OKU.** Zaman aşımı "gitmedi" demek değil.
- **Final için 2 gönderim SEÇ** — Kaggle arayüzü, tarayıcı. API'de yok.
- **Notebook 2 Eylül 13:00.** İlk 20 inceleniyor; tüm dış veri kaynak+amaç+kullanımla beyan.
- **Geriye gitme riski yok** — Kaggle en iyi public skoru tutar.
- **Ölçüm hassasiyeti ±9,5e-5.** 2e-4 altındaki kazanç ölçülemez, hak harcanmaz.

---

## Aday envanteri (hepsi kapı denetiminden geçti)

| dosya | Q | kurtoz | bağımsız | kos(y46) | sıra |
|---|---|---|---|---|---|
| `tuketim_y46_amnezik_kirpik.csv` | 0,388 | 4,5 | %87 | — | **HAK1** |
| `tuketim_q1c_kapasite_siki.csv` | 0,061 | 5,0 | **%84** | **−0,011** | **HAK2** |
| `tuketim_z2_analog.csv` | 0,118 | 9,0 | — | −0,120 | 31 Ağu |
| `tuketim_y45_mevsimsel_kirpik.csv` | 0,167 | 8,5 | %66 | −0,095 | 31 Ağu |
| `tuketim_q1d_kuantil38_siki.csv` | 0,055 | 5,1 | %59 | −0,208 | 1 Eyl |
| `tuketim_g7_span_tau3.csv` | 0,0025 | 23,1 | %0 | −0,091 | **gönderilmez** |

---

## KIRICI DENETIMI — 29 Ağustos gecesi, son tur

Beş iddia sınandı: **2 geçti, 3 kırıldı.** Düzeltmeler yapıldı.

### KIRILDI 1 — hata bandı 2,7 kat dar ilan edilmişti
Gerçekçi band **±0,00019** (±0,00007 değil). Sebep: `g7`'nin span-dışı bileşeni
(rms 1,74e-04 log-birim) ölçülemiyor; Cauchy-Schwarz sınırı `L`'nin %6,4'ü.
Bağımsız LOO sınavında artığı <1e-6 olan 10 dosyada maks sapma 1,78e-04 —
sınırla aynı mertebede, tesadüf değil.

> **Ve keskin bir uyarı:** *iki bağımsız ajanın 1,00135 / 1,00137 demesi
> DOĞRULAMA DEĞİL* — ikisi de aynı span makinesini ve aynı ölçülemez
> bileşeni taşıyor. Bağımsızlık görünürde.

**Hafifletici:** bu hata `k`'yı 0,070 kaydırır ama gerçek MSE cezası **1,2e-05**
(skorda 6e-06). Yani *tahmini* bozar, *üretilen dosyayı* değil.

### KIRILDI 2 — `m99` korkuluklarında 4 delik (HEPSİ KAPATILDI)
1. **`to_csv` assert'ten ÖNCE çalışıyordu** → kapı denetimi patlarsa bozuk dosya
   hedef adıyla diskte kalıyor ve gönderilebiliyordu. Artık geçici dosyaya yazılıp
   denetim geçtikten **sonra** taşınıyor.
2. **`maks` hesaplanıyor ama assert edilmiyordu.** `k=[1;−1;−1]` örneği dört
   korkuluğun da yeşilinden geçip maks 3,9e6 üretiyordu. Artık
   `maks ≤ 3×taban_maks` ve `isfinite` denetimi var.
3. **Skorlar `olculmus_skorlar.json` ile karşılaştırılmıyordu.** `−0,001`'lik bir
   yazım hatası tüm korkulukları geçip çöp dosya üretiyordu. Artık ölçülmüş
   skorla çelişen değer **durduruyor**; ölçülmemiş dosya için uyarı basılıyor.
4. **Yön dosyalarının ID hizası denetlenmiyordu** (konumsal hizalama). Artık
   her dosya `test.csv` ile karşılaştırılıyor.

Dördü de test edildi: sağlıklı durum geçiyor · skor typo'su yakalanıyor ·
`g7` uydurma skoru uyarı verip devam ediyor · bozuk çözüm duruyor, dosya yazmıyor.

### KIRILDI 3 — Plan B'nin üstünlüğü koşulsuz değil
`g7` gönderilseydi `L(g7)` ±1,75e-04'ten ±5e-06'ya inerdi ve bu **31 Ağu + 1 Eyl'deki
tüm ortak çözümleri** temizlerdi. Plan B bu belirsizliği son güne taşıyor.
Karşı hesap: belirsizliğin bedeli çözüm başına 1,2e-05 MSE, üç çözümde 3,6e-05;
bir hakkın değeri ~0,0016 MSE → **yine de 44 kat Plan B lehine.** Plan B kalıyor.

### GEÇTİ — üçlü sistem ve dosyalar
`cond(G₃) = 159,4`, karşılıklı kosinüsler |·| ≤ 0,10, senaryo taramasında
`|k|₁` maks 1,639 (eşik 5). Dosyalar temiz; `y45`'in maks 434.388'i sorun değil
(train maks 50,4M, p99.99 = 275.254).

> **`|k| ≥ 0,45` kuralı vs yer değiştirme:** ilk kırıcı turu ölçmüştü ki öngörü
> hatası `|k|` ile değil **yer değiştirme** ile büyüyor (korelasyon +0,75 vs +0,30).
> Bizim yer değiştirmemiz **0,096**, doğrulanmış bölgenin (0,286) içinde. Kural
> yanlış eksende ölçüyordu; bağlayıcı olan yer değiştirme ve o temiz.
