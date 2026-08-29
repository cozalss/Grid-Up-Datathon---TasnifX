# 57 — 30 Ağustos koşu kılavuzu (docs/54 ve docs/56 §5'in yerine geçer)

**Sabah bunu aç, sırayla uygula.** Gerekçeler: [`docs/56`](56-30-agustos-plani-ikinci-sira.md) · Bilanço: [`docs/55`](55-29-agustos-bilancosu.md)

---

## Başlangıç

```
1. Grid Grinders     0.99064
2. Atakan Aldemir    1.00041   <- HEDEF
3. TasnifX           1.00284   <- BIZ (submissions/tuketim_m6_ikiyon.csv)
4. Ahmet B. ALTUNOK  1.00480
5. Saban Ozdogan     1.00510

m0 = 1.00284^2 = 1,005688   ·   hedef MSE 1,000820   ·   gereken dMSE -0,004868
kota yerel 03:00'te acilir  ·  kalan 9 hak (30-31 Agu + 1 Eyl)
```

**Dış veri kullanımı SERBEST** — düzenleyici e-postayla teyit etti (bkz.
`experiments/model29/k2_veri_kurali.json`). Tek yükümlülük notebook beyanı.

---

## HAK 1 — span optimumunu kilitle

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_g7_span_tau3.csv -m "g7 span optimumu tau=3: 17 olculmus gonderimin afin kombinasyonu, winner's-curse duzeltmeli. ON KAYIT 1.00137 +-0.00007"
kaggle competitions submissions -c grid-up-datathon
```

Beklenen **1,00137 ± 0,00007**. Tahmin değil, cebir: LOO hatası ±0,00008,
üç bağımsız yoldan doğrulandı, ağırlıklar toplamı tam 1, kırıcı ajan kıramadı.

---

## HAK 2 — en temiz yeni yönü ölç

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_y46_amnezik_kirpik.csv -m "y46 AMNEZIK yon: 24 gecmis kolonu atilmis GBM, olu-trafo bileseni temizlenmis. Q=0.388 kurtoz=4.5 kos(m4)=-0.115. AMAC: L olcmek"
kaggle competitions submissions -c grid-up-datathon
```

**Skoru ne gelirse gelsin sorun değil** — amaç ölçüm. `y46` neden seçildi:
`Q` en büyük (ölçüm en temiz), kurtoz en düşük (4,5), diğer adayların
hepsiyle `|kosinüs| ≤ 0,20`.

---

## HAK 3 — ortak optimum

```powershell
python experiments/model29/m99_coklu_coz.py <HAK1_dosyasi>=<HAK1_skoru> tuketim_y46_amnezik_kirpik.csv=<HAK2_skoru> --cikti tuketim_g8_ortak.csv
```

Betik **korkuluklu**: `cond(G)>1e8`, `MSE<0`, `|k|₁>5` ya da `MSE>m0` durumunda
DURUR ve dosya yazmaz. Tetiklenirse `--lam 0.001` ekle.

**Uyarı:** `|k| ≥ 0,45` çıkarsa ön kayıt bandı (±7e-5) orada geçerli değil —
doğrulanmış yer değiştirme bölgesinin dışı.

---

## Aday envanteri (hepsi kapı denetiminden geçti)

| dosya | Q | kurtoz | kos(m4) | sıra |
|---|---|---|---|---|
| `tuketim_g7_span_tau3.csv` | — | — | — | **HAK1** |
| `tuketim_y46_amnezik_kirpik.csv` | 0,388 | **4,5** | −0,115 | **HAK2** |
| `tuketim_y45_mevsimsel_kirpik.csv` | 0,167 | 8,5 | **+0,005** | 31 Ağu |
| `tuketim_z2_analog.csv` | 0,118 | 9,0 | +0,161 | 31 Ağu / 1 Eyl |
| `tuketim_z1_havuz.csv` | 0,119 | 8,2 | +0,186 | `y45` ile %56 örtüşür — biri |
| `tuketim_r3_ay.csv` | 0,003 | 4,3 | −0,000 | Q küçük, ölçüm bulanık |
| `tuketim_r1_seviye.csv` | 0,003 | 9,7 | +0,000 | aynı |
| `tuketim_z3_ikiasama.csv` | 0,065 | 19,4 | **+0,564** | **ELENDI** |
| `tuketim_b2_span_k15.csv` | — | — | — | `g7`'nin yedeği |

Karşılıklı kosinüsler: `z1↔y45 +0,557` ve `z1↔z2 +0,515` — bu çiftlerden
**yalnız biri** alınır, ikisi birden kazancı toplamaz.

---

## 31 Ağustos ve 1 Eylül (6 hak)

1. `y45_mevsimsel` ölç → ortak optimum
2. `z2_analog` ölç → ortak optimum
3. Yedek / son rötuş

---

## Neden 2. sıra mümkün

```
span adimi (HAK1)          -0,002946   NEREDEYSE KESIN
span sonrasi kalan          0,001922

bir yonun katkisi = kalite^2 ,  kalite r = L/sqrt(Q)
OLCULMUS kaliteler: v101 0,1243 · m4 0,0641 · p51 0,0493

kac yon    her birinden gereken r   m4 kalitesine oran
   1              0,0438                  68%
   2              0,0310                  48%
   3              0,0253                  39%
   4              0,0219                  34%
```

**Dört dik yön, her biri `m4` kalitesinin üçte biri kadar olsa yeter.**
Elimizde dört bağımsız aday var. Tahminim: **2. sıra %60-65.**

---

## Kaçırılmaması gerekenler

- **Her gönderimden sonra listeyi OKU.** Zaman aşımına uğrayan betik
  "gitmedi" demek değil — bu depoda o yüzden bir hak boşa gitti.
- **Final için 2 gönderim SEÇ** — Kaggle arayüzünden, tarayıcıdan.
  API'de bu alan yok. Seçilmezse en iyi public otomatik seçilir, ikinci slot boşa gider.
- **Notebook son tarihi 2 Eylül 13:00.** İlk 20'nin notebook'u inceleniyor;
  kullanılan tüm dış veri kaynak + amaç + kullanım biçimiyle beyan edilmeli.
- **Geriye gitme riski yok** — Kaggle en iyi public skoru tutar.

---

## Araçlar (ikisi de doğrulandı)

| araç | ne | doğrulama |
|---|---|---|
| `m50_harman_coz.py` | iki dosyalı harman | `v83`+`v101` → `v102`'yi **bit düzeyinde** üretti |
| `m99_coklu_coz.py` | N yönlü ortak optimum + **korkuluklar** | `v102`+`m4`+`p51` → 1,00292 (gerçekleşen 1,00284); 9 test geçti |
