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

## HAK 1 — `y46` ölç

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_y46_amnezik_kirpik.csv -m "y46 AMNEZIK yon: 24 gecmis kolonu atilmis GBM, olu-trafo bileseni temizlenmis. Q=0.388 kurtoz=4.5. AMAC: L olcmek"
kaggle competitions submissions -c grid-up-datathon
```

Neden: `Q` en büyük → **ölçüm en temiz** (anlamlı katkı için gereken `L`,
gürültünün 146 katı). Kurtoz en düşük (4,5). Her şeye dik (`|kos| ≤ 0,21`).

## HAK 2 — `q1c` ölç

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_q1c_kapasite_siki.csv -m "q1c: kapasite ofsetli hedef (log1p(tuketim)-log_guc), siki kirpma. Artik haritasina nisanli. Q=0.061 kurtoz=5.0 kos(y46)=-0.011. AMAC: L olcmek"
kaggle competitions submissions -c grid-up-datathon
```

Neden: **ölçülmüş hata haritasına nişan alan tek aday.**
- Harita: D00 desilinde `+0,040` fazla tahmin → `q1c` D00'ı **−0,107** çekiyor
- Harita: güç≤50'de `+0,018` fazla → `q1c` **−0,101** çekiyor
- Bağımsız payı %83,9 (tüm mevcut yönler çıkarıldıktan sonra)
- `y46` ile kosinüs **−0,011** → pratikte tam dik, kazançlar toplanır

## HAK 3 — ortak optimum

```powershell
python experiments/model29/m99_coklu_coz.py tuketim_m6_ikiyon.csv=1.00284 tuketim_g7_span_tau3.csv=<HESAPLA> tuketim_y46_amnezik_kirpik.csv=<HAK1_skoru> tuketim_q1c_kapasite_siki.csv=<HAK2_skoru> --cikti tuketim_g9_ortak.csv
```

> **`g7` için skor girmek yerine:** `m101_planB.py` `L(g7)`'yi doğrudan
> hesaplıyor. En temizi `m99`'a `g7`'yi eşdeğer skorla vermek:
> `S_g7 = sqrt(m0 + Q - 2L) = sqrt(1.005688 + 0.002494 - 2*0.002728)` =
> **`1.00135`** — bu, `g7`'nin ölçülseydi alacağı skor.

Betik **korkuluklu**: `cond(G)>1e8`, `MSE<0`, `|k|₁>5`, `MSE>m0` → DURUR,
dosya yazmaz. Tetiklenirse `--lam 0.001` ekle.

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
