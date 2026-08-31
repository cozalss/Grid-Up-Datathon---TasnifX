# 80 — DEVİR BELGESİ (31 Ağustos 22:00)

Kullanıcı Claude hesabı değiştiriyor. **Bu belge tek başına yeterlidir.**
Önce bunu oku, sonra `docs/79-DURUM-31agustos-1615.md` (EK 1-3 dahil).

---

## 1. DURUM ÖZETİ — TEK PARAGRAF

Yarışma 2 Eylül 02:59 TSS'te bitiyor. 3 gönderim hakkı var, 1 Eylül 03:00
TSS'te açılıyor. Şu an 12. sıradayız (LB 1.00115), hedef ilk 3 (3. sıra
0.99556). Bugün ÜRETİM HATTINDA BİR HATA bulundu: soğuk harman 23 Ağustos'ta
yalnız kış26'ya bakılarak cat-tekile indirilmiş ve bu, test dönemine mevsimsel
olarak denk gelen blokta ciddi zarar veriyor. Eski 3/1/1 harmanına dönüş dört
kapıyı da geçti ve **gönderime hazır aday üretildi.** ONAY OLMADAN GÖNDERİM YOK.

---

## 2. GÖNDERİLECEK ADAY

**`experiments/model29/p_kalici/aday_csv/p20_harman_ESKI_3_1_1_V1_seviyesiz.csv`**

Bağımsız doğrulandı (hem agent hem ben):
- 714.688 satır, id sırası `data/raw/test.csv` ile birebir
- NaN yok, negatif yok, hepsi sonlu
- Değişen: tam olarak **158.369 soğuk satır** (%22.16). Sıcak satırlarda
  görülen 2371 fark yalnız kayan nokta gürültüsü (maks **4.4e-16**)
- Log kayma: ort **+0.00000** (seviye oynamıyor, yalnız YAPI), std 0.2050,
  aralık [−1.68, +1.02]

### Beklenti (mevcut 1.00115; 3. sıra 0.99556; 2. sıra 0.99518)

| senaryo | test dMSE | beklenen LB |
|---|---|---|
| kırpmasız, taşıma oranı 1.0 | +0.01921 | **0.99151** ✓ 2.'yi geçer |
| kırpmasız, taşıma oranı 0.5 | +0.01921 | 0.99634 ✗ kıl payı |
| K=25 kırpık, taşıma 0.5 | +0.00540 | 0.99980 ✗ |

En büyük zayıflık **kırpma**: aykırı değer kırpmasıyla kazanç 3-5 kat
küçülüyor. Gerçek değer bu aralıkta.

### Alternatifler (aynı klasörde, hepsi doğrulandı)
- `p20_harman_ESIT_V1_seviyesiz.csv` — daha büyük ham kazanç (+0.1199 yapı)
  ama kırpma merdiveninde K=25'te kış26'da işaret döndürüyor
- `p20_harman_ESKI_3_1_1_V2_seviyeli.csv`, `p20_harman_ESIT_V2_seviyeli.csv`
  — **KULLANMA**, seviye bileşeni çift sayım (aşağıda §4)

### YEDEK (ölçülmüş, dokunulmadı)
`submissions/tuketim_YP_seviye.csv` = **1.00115**. Son seçimde bu her koşulda
elimizde. Bu yüzden aşağı yönlü risk YOK — kötü skor gelirse o dosyayı seçmeyiz.

---

## 3. BUGÜNÜN ANA BULGUSU — ÜRETİM HARMANI

`scripts/tuketim_model.py` `REJIM_AYARLARI` (satır ~926 ve ~990):

```python
"sicak": {"maske": 0.15,
          "cat": {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6},
          "agirlik": {"cat": 3.0, "xgb": 1.0, "lgbm": 1.0, "sinir_agi": 1.4}},
"soguk": {"maske": 1.00, "cat": {"depth": 7}, "ek_koken": False,
          "agirlik": {"cat": 1.0}},     # yorum: "SOGUK HARMAN -> YALNIZ cat"
```

Değişim commit'i **d04243f (2026-08-23)**. Öncesinde soğuk da 3/1/1'di.

**AMA BÜTÜN CV ÖLÇÜMLERİMİZ EŞİT HARMAN ÜZERİNDE YAPILMIŞ** (`p02_duzeltme.blok()`
`soguk_tahmin_*.npz`'nin bütün anahtarlarını eşit ortalıyor). Yani tezgah
üretimi ölçmüyor. Ampirik kanıt (gönderim dosyalarının soğuk satırlarını üç
aile kolonuna regresyon, yöntem v27/v30'da kalibre edildi):

| dosya | b_cat | b_xgb | b_lgbm |
|---|---|---|---|
| tuketim_YP_seviye.csv (1.00115) | **1.115** | −0.099 | −0.016 |
| tuketim_m6_ikiyon.csv | 1.076 | −0.071 | −0.005 |
| v27_v18hedge (d04243f ÖNCESİ) | 0.597 | 0.238 | 0.165 |

23 Ağustos'tan sonraki her dosyada cat payı ~1.0. Kırılma d04243f ile örtüşüyor.

**Tezgah, gönderdiğimizden 0.017 daha iyi bir nesneyi ölçüyor.** Bu, projenin
"küçük CV kazançları LB'ye taşınmıyor" bilmecesinin bir parçası olabilir.

### GEÇERSİZ OLAN ÖLÇÜMLER (bunlara dayanma)
- **p06** soğuk harman ağırlığı — üretimde olmayan harmanı optimize etmiş
- **p11/p14 soğuk lgbm huber** (+0.0126) — üretimde lgbm YOK
- **p17_band.json §5_huber_beklentisi** ve P(3.sıra)=0.52 hesabı
- **p07/p10'un 12 aday CSV'si** (`aday_csv/p10_*.csv`) — delta cat-tekil tabana
  eklenirse harmanı (0.717, 0.017, 0.267)'ye götürür, amaçlanan yere değil
- **p11_b_lgbm.json'da etiket hatası**: "TABAN" sütunu eşit harman değil,
  yalnız-lgbm

---

## 4. SEÇİLEN ADAYIN GEREKÇESİ

### Kazanç YAPI ve SEVİYE diye ayrışıyor

| aday | toplam | = YAPI | + SEVİYE |
|---|---|---|---|
| ESIT | +0.1013 (2/3) | **+0.1199 (3/3)** | −0.0186 (0/3) |
| **ESKI_3_1_1** | +0.0753 (2/3) | **+0.0867 (3/3)** | −0.0114 (0/3) |

kış26 tersliğinin **tamamı seviye bileşeninde**; yapıda kış26 bile pozitif
(+0.0157). Seviye üç blokta da zarar veriyor çünkü gönderim dosyasının seviye
katmanı ZATEN LB'den çözülmüş (m111 κ, span a0+r_hat) — bir de buradan
oynatmak **çift sayım**. Bu yüzden **V1_seviyesiz** doğru aday.

### Neden ESIT değil 3/1/1
**Kırpma merdiveni**: K ∈ {0,5,10,25,50} aykırı değer kırpmasının HEPSİNDE
3/1/1 üç blokta da pozitif. ESIT K=25'te kış26'da işaret döndürüyor (−0.0065).
Tohum tutarlılığı 3/1/1 için **9/9**, ESIT için 8/9.

### Dört kapı (ön-kayıtlı)
| kapı | ESKI_3_1_1 |
|---|---|
| (a) gerçek blok CV'sinden ölçüldü (vekil değil) | ✓ |
| (b) blok-dışı seçim, 0 parametre | ✓ |
| (c) işaret tutarlılığı | **3/3 (yapıda)** |
| (d) tohumla ayakta | **9/9** |

### Önyükleme (trafo kümeli, 500, kohort ağırlıklı, seviyesiz)
| blok | dMSE | GA95 | P(+) |
|---|---|---|---|
| yaz25 | +0.1058 | [+0.032, +0.189] | **1.000** |
| guz25 | +0.1386 | [+0.043, +0.249] | **0.996** |
| kis26 | +0.0157 | [−0.029, +0.059] | 0.756 |

### Ek destek
3/1/1 bir yenilik değil, **LB'de ölçülmüş bir noktaya geri dönüş**: d04243f
öncesi üretim harmanıydı ve o dönemin BİRİNCİLİĞİ (LB 1.01750) onunla alındı.

Aday nasıl üretildi: `p06_test_soguk_aile.npy` (158369×3, cat/xgb/lgbm, tohum
1000-1002) kullanıldı, eğitim GEREKMEDİ. Gönderim zinciri soğuk satırlarda
afin; eğim dosyadan ölçüldü (**s=0.8132**, yöntem v27/v30'da kalibre edildi).
Uygulama: `log1p(yeni) = log1p(eski) + s·(r_3/1/1 − r_cat − ort)`.

---

## 5. KAPANAN YOLLAR (tekrar açma — hepsi ölçüldü)

- **soğuk cat huber**: yaz25 taban 1.5817 → α=4.0 1.5882, α=2.0 1.6062,
  α=1.0 1.6220, α=0.5 1.6533, α=0.2 1.6737. guz25 taban 1.6964 → α=1.0
  1.7914, α=0.5 1.8178. **Tekdüze kötü.** cat'ta dayanıklı kayıp yok.
- **sıcak cat huber**: yaz25 +0.011 ama guz25 TERS (0.8392 vs 0.8361), l1
  her blokta kötü. ELENDİ.
- **soğuk lgbm huber**: kazanç gerçek ama üretimde lgbm yok (§3)
- **p09** trafo bazlı kalıcı sapma: bloklar arası korelasyon işaret
  değiştiriyor; 9 dürüst blok-dışı seçimin hiçbiri anlamlı pozitif değil
- **p09b** mevsim eşleşmeli 12-ay transferi: ÖLÇÜLEMEZ (artık verisi tam 12
  ay, her mevsimden tek örnek); mekanizma zaten modelde (`t_gy_log_ort`,
  `t_mevsim_genlik`, `t_ay_sapma` — kapsam TEST TAM)
- **p11** ufuk/ofset kalibrasyonu (3/3 kayıp), komşu havuzu (3/3 kayıp),
  yalnız-statik soğuk model (konusuz: 33 `t_*` kolonu soğukta zaten tam NaN),
  soğuk skaler kayma (ağırlıklı ölçümde negatif)
- **p01** kalibrasyon (19/20 kayıp), **p02** sıfırdan taban (0.946 vs 0.867),
  **p04** alan bilgisi (tavan +0.0012), **p05** iki aşamalı ayrışma (7/7 negatif)
- **Sıfır sınıflandırıcısı**: soğukta AUC yalnız 0.58-0.61, öğrenilemiyor.
  (Soğuk MSE'nin %55.5'i tüketim=0 satırlarından; kâhin dedektör soğuğu
  1.436 → 0.939 yapardı ama sınıflandırılamıyor.)

## 6. KABUL EDİLEN, KÜÇÜK

**p08 ölü trafo kuralı** — son işlem katmanında, harmandan BAĞIMSIZ, hâlâ
geçerli. Kural: son 30 günde max<=0 VE geçmiş sıfır oranı >=%99 VE kesintisiz
sıfır serisi >=30 gün → tahmini SIFIRLAMA, **×0.25 veya ×0.50 ile küçült**.
Üç blokta da kazanıyor (dMSE −0.0017/−0.0010/−0.0013), test'te 135 trafo /
15.533 satır, hepsi SICAK. LB karşılığı −0.0005..−0.0009.
Delta dosyaları: `aday_csv/p08_olu_delta_log.npy` (×0.25),
`p08_olu_delta_log_c050.npy` (×0.50). **Soğuk harman adayıyla ÇAKIŞMAZ**
(biri sıcak, diğeri soğuk satırlar) — üstüne eklenebilir.

## 7. ÇÜRÜTÜLEN KENDİ KURALIMIZ — 0.02 BANDI

docs/79 §2'deki "|dCV|<0.02 taşınmıyor" kuralı **GEÇERSİZ** (`p17_band.json`):
- p12e λ'yı TOHUM gürültüsüyle (7.60e-06) hesaplamış; vekilin hatası SİSTEMATİK
  ve ölçüldü (3.97e-04). Doğru λ = **−2.65** → güvenilirlik SIFIR
- Band sınırı 0.02 = **1.00 × σ_çift**, yani ölçüm gürültü tabanıyla çakışıyor
- Simülasyon: kural hiç yokken bile örtüşme %15.8 olasılıkla çıkıyor
- **Vekile bulaşmamış GERÇEK blok CV'si** (rekor.jsonl): v27→v30 oran 0.759,
  v30→v46 0.276, v27→v46 0.556 — üçü de bandın İÇİNDE, işaret 3/3, eğim 0.568

**Taşıma oranı için muhafazakâr nokta tahmin: 0.5.** Yeni karar kuralı:
(a) gerçek blok CV'sinden mi ölçüldü, (b) blok-dışı seçim temiz mi,
(c) bloklar arası işaret tutarlı mı, (d) tohum arttıkça ayakta mı.

## 8. AÇIK SORUNLAR

1. **Sıcak önbellek kökeni açıklanamadı.** `p18_yeniden_egit.py` sıcak tarafta
   birebir tutmadı (maxabs 0.325, kor 0.99989). LightGBM'in belirlenimli
   olduğu ayrıca kanıtlandı (aynı hücre iki kez, bit-birebir aynı) — yani
   `aile_onbellek/*_uretim.npy` bugünkü `aile_onbellegi.py` ile üretilmemiş.
   Soğuk tarafta birebir GEÇTİ (maxabs 8.9e-16). Soğuk harman adayını
   etkilemiyor ama sıcak tezgahı şüpheli.
2. **`sinir_agi` harmanın %21.9'u ve hiçbir ölçümde yok** (`aile_onbellegi.py:15-17`
   açıkça yazıyor: "sinir_agi ızgaraya GİREMEZ, tek fit ~20 dakika").
3. **p14_test.py denetimi çöktü** (maxabs 5.85, kor 0.8614) ve kontrol `raise`
   etmiyor, sadece log yazıyor. Ürettiği dosyalar KULLANILMAYACAK.
4. Kış26'nın seviye tersliği gerçek bir blok etkisi, açıklanmadı.

## 9. YARIM KALAN İŞLER

- **Sıcak taraf taraması** (`p15_*`): cat τ ızgarası ({240,480,960,1920},
  üç aile, üç blok, 3 tohum). Tek ölçüm var: cat yaz25 τ=480 → 0.823415 vs
  taban 0.833132 (**−0.0098 ham, üretim ağırlığıyla −0.0046**). Umut verici,
  doğrulanmadı. lgbm huber (yaz25 −0.0349 ham, üretimde −0.0055) ve xgb
  huber α=2.0 (−0.0075 ham, üretimde −0.0012) de doğrulanmayı bekliyor.
- **Soğuk cat** (`p19_*`): huber kapandı; MAE/Quantile/MAPE, τ ve
  hiperparametre ızgarası kalmıştı.
- **p18 düzeneği** (`p18_yeniden_egit.py`): parametrik tam yeniden eğitim.
  Süre boş makinede blok ~1.4 sa + test ~0.85 sa = **~2.3 sa**.
  ÖNEMLİ ANALİTİK SONUÇ: bu hatta harman sabit ağırlıklı aritmetik ortalama
  ve aileler bağımsız eğitiliyor — "aileler birbirine uyum sağlar" mekanizması
  YOK. Blok uzayında DELTA = TAM (~1e-15). Yani yama ile tam yeniden eğitim
  ancak (a) değişiklik ailelerarası ortaksa (TAU), (b) tohum kümesi değişirse,
  (c) TEST tarafında (cebir + son işlem) ayrışır.

## 10. GÖNDERİM PLANI

**Haklar 1 Eylül 03:00 TSS. Yarışma 2 Eylül 02:59 TSS. Pencere 24 saat.**
03:00'te göndermek ZORUNLU DEĞİL — kullanıcı bunu açıkça söyledi.

1. **Hak 1:** `p20_harman_ESKI_3_1_1_V1_seviyesiz.csv`. Skor dakikalar içinde
   gelir ve **taşıma oranını GERÇEK veriyle ölçer**:
   `oran = (1.00115 − gelen) / 0.01921`
2. **Hak 2:** gelen orana göre. Oran yüksekse ESIT varyantı (daha büyük ham
   kazanç); düşükse p08 ölü trafo deltasını üstüne ekle; sıcak taraftan
   doğrulanmış kazanç çıkmışsa onu birleştir.
3. **Hak 3:** ilk ikisinin ölçümüyle en iyi bileşim.

**SON SEÇİM: yarışma bitiminde TARAYICIDAN 2 dosya seçilir** (API'den
yapılamaz). Önce "You selected X of N" satırı okunur. Yedek `YP_seviye`
(1.00115) her koşulda seçeneklerden biri.

## 11. DEĞİŞMEYEN KURALLAR

- **ONAY OLMADAN HİÇBİR GÖNDERİM YOK.** Her hak için ayrı onay.
- Gönderimden sonra MUTLAKA `kaggle competitions submissions grid-up-datathon`
  ile liste okunur — zaman aşımına uğrayan betik "gönderilmedi" demek değil
  (bir hak böyle boşa gitti).
- Liderlik tablosunu her turda çek: `kaggle competitions leaderboard
  grid-up-datathon --show`. Hedef HAREKET EDİYOR (Berke Kuç bugün 0.99927 →
  0.99648, Şaban 1.00049 → 0.99810).
- Paralel oturumlar aynı çalışma ağacını paylaşıyor: commit'te yalnız kendi
  dosyalarını stage'le.
- Python yazarken Write aracı (bash heredoc ters bölüleri bozuyor).
- subprocess'te `text=True` yanında `encoding="utf-8"`.
- `str.replace()` öncesi `assert hedef in s`.
- Ara dosyalar scratchpad'e (Bash /tmp ile Windows Python uyuşmuyor).
- **Ölçüm hükmü vermeden önce**: gerçek blok CV mi, blok-dışı seçim temiz mi,
  3 tohum × 3 blok mu, kohort ağırlıklı mı, ÜRETİM harmanı mı.

## 12. LİDERLİK TABLOSU (31 Ağu 19:38 TSS)

| # | takım | skor |
|---|---|---|
| 1 | Grid Grinders | 0.98110 |
| 2 | Duo-Electra | 0.99518 |
| 3 | Abdülbaki Bayır | **0.99556** ← hedef |
| 4 | Berke Kuç | 0.99648 |
| 5 | Şaban Özdoğan | 0.99810 |
| 12 | **TasnifX** | **1.00115** |

Gereken kazanç: **0.00559**.

## 13. DOSYA HARİTASI

- `docs/79-DURUM-31agustos-1615.md` — önceki durum + EK 1/2/3
- `docs/80-DEVIR-31agustos-2200.md` — **bu belge**
- `experiments/model29/p_kalici/` — bütün ölçüm çıktıları (json) ve betikler
  - `p20_harman.json` ana tablo + hüküm, `p20_onyukleme.json`, `p20_kis26.json`,
    `p20_aday.json`, `p20_yapi.json`
  - `p18_harman_hukmu.json` harman kanıtı, `p18_hazirlik.json` hat haritası
  - `p17_band.json` 0.02 kuralının çürütülmesi
  - `p14_ozet.json` (İÇİNDE `00_GECERSIZ_HUKUM_UYARISI` var — oku)
  - `p19_soguk_cat.json`, `p15_ozet.json`
- `experiments/model29/p_kalici/aday_csv/` — bütün aday CSV'ler ve delta npy'ler
- `scripts/tuketim_model.py` — `REJIM_AYARLARI` (harman tanımı)
- `scripts/son_islem.py` — üretim son işlemi (β=0.60, LB'de üç kez doğrulanmış)
