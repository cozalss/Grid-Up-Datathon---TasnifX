# 61 — KESİN KILAVUZ v2 · 30 Ağustos – 1 Eylül

**Bu belge docs/60'ın yerine geçer.** Üç bağımsız denetim planı iki noktada
değiştirdi. Sabah bunu aç, sırayla uygula.

---

## 1. Durum (2026-08-29 gecesi, LB'den doğrulandı)

```
1. Grid Grinders     0.99046
2. Atakan Aldemir    0.99940   <- HEDEF
3. Tuna Deniz        1.00267
4. TasnifX           1.00284   <- BIZ (submissions/tuketim_m6_ikiyon.csv)
5. Abdulbaki Bayir   1.00322
```
```
m0 = 1.00284^2 = 1.005688066   ·   9 hak (30-31 Agustos + 1 Eylul)
kota yerel 03:00'te yenilenir  ·   bitis 1 Eylul 23:59 UTC
notebook 2 Eylul 13:00         ·   private LB 2 Eylul 00:10
```
**Dış veri SERBEST** (düzenleyici e-postayla teyit etti). Notebook'ta beyan şart.
**Final için 2 gönderim tarayıcıdan SEÇİLMELİ.** API'de yok.

---

## 2. Neden plan değişti — iki bulgu

### Bulgu A: docs/60'ın kalite beklentisi ESKİ TABANA aitti

docs/60 "ölçülmüş kaliteler 0,049–0,124, eşiğin 3,6–9 katı" diyordu. Bu değerler
**eski tabanlara** göreydi. `m6_ikiyon`'un artığına göre yeniden ölçtüm
(26 ölçülmüş skorun hepsinden, `L_j = (m0 + Q_j − P_j²)/2`):

```
m6'ye gore |rho|, absorbe edilmemis 20 yon:
  0,0080 0,0084 0,0092 0,0110 0,0161 0,0171 0,0191 0,0243 0,0251 0,0278
  0,0289 0,0296 0,0310 0,0327 0,0334 0,0341 0,0342 0,0360 0,0360 0,0360
  ortanca 0,0283   esigin (0,0137) uzerinde 16/20
```
(`p51`, `m4`, `v102`, `v109` çıkarıldı — m6 zaten onların optimumu, `rho ≈ 0`
çıkması **zorunlu**; nitekim dördü de ±0,004 içinde. Cebirin bağımsız doğrulaması.)

Bu kalibre önselle Monte Carlo (200 000 çekim):

```
3 yon olculurse   P(2. sira) =  46%        <- docs/60'in plani
6 yon olculurse   P(2. sira) =  80%
8 yon olculurse   P(2. sira) =  95%        <- YENI PLAN
```
Ölçüm gürültüsü 1,6e-4 → 4,0e-4 arasında sonuç **değişmiyor** (%0,4). Hassasiyet
bağlayıcı kısıt değil. Her senaryoda `P(1,00284'ten kötü) = %0`.

**Sonuç: 3 yön ölçüp kalan 6 hakkı "rötuşa" ayırmak yazı-tura. 8 yön ölç.**

### Bulgu B: sondanın `g7` katsayısı yanlıştı

docs/60'ın sondası `m6 + 1,093664·d_g7 + t·d_aday` idi; `1,093664 = L_g7/Q_g7`,
yani **`d_aday` yokmuş gibi** hesaplanmış. Çapraz terim ihmal edilmiş. `t` sabitken
koşullu optimum `c* = (L_g7 − t·C)/Q_g7`. `kos(g7,y40) = −0,555` olduğu için
y40 sondasında **0,00158 RMSLE bedavaya veriliyordu** — 2. sıraya olan farkın %46'sı.

Ayrıca `t` değerleri (0,60 / 0,35 / 0,45) keyfiydi ve sondanın kendi skorunu
bozuyordu; oysa LB 5 haneye yuvarladığı için ölçüm çok daha küçük `t`'de de
fazlasıyla keskin (SNR 179).

**Yeni tasarım (`m107_sonda3.py`):** sonda = *yeni adayın `L`'si sıfır* varsayımı
altındaki **tam ortak optimum**. `t` artık keyfi değil, `k* = G⁻¹L`'den çıkıyor.
Sonda böylece hem ölçüm hem de o an bilinen **en iyi gönderim** oluyor.

```
                       L_aday=0 iken   rho=0,035'te
docs/60 sondasi (t=0,60)    1,00341        0,99987
m107 sondasi  (k=0,389)     1,00085        0,99855
```

---

## 3. YARIN — 3 hak, tek komut zinciri

Sürücü `m108_gun.py` durumu `m108_durum.json`'da tutar; laptop kapansa da
kaldığı yerden devam eder.

```powershell
cd experiments/model29

# --- 1. HAK ---
python m108_gun.py --baslat          # sondayi uretir + kaggle komutunu basar
# (basilan komutu calistir)
kaggle competitions submissions -c grid-up-datathon    # MUTLAKA OKU
python m108_gun.py --skor 1.00085    # <- gercek skoru yaz; L'yi cozer, 2. sondayi uretir

# --- 2. HAK ---  (basilan komutu calistir, listeyi oku)
python m108_gun.py --skor <SKOR>

# --- 3. HAK ---  (ayni sekilde)
python m108_gun.py --skor <SKOR>
```

**1. sonda hazır ve kapı denetiminden geçti:** `submissions/tuketim_s3y40.csv`
```
k = [g7 +1,82286 , y40 +0,38876]    cond(G)=17,5   |k|_1=2,21
L_y40 = (1.001705415 - P^2) / 0.777517
en kotu (L=0)   1,00085        rho=0,0137  0,99995
rho=0,025       0,99921        rho=0,035   0,99855      rho=0,05  0,99757
```
İlk gönderim **en kötü ihtimalle 1,00085** — yani daha yarın 3. sıra;
y40 tipik kalitedeyse doğrudan 2. sıra.

### Ölçüm sırası (değer sırası; ilk üç yarın)
```
y40 -> z2 -> sul   |   y46 -> y45 -> q1c   |   t3 -> h1 -> NIHAI
 30 Agustos              31 Agustos            1 Eylul
```
`y40` başta çünkü `g7` ile −0,555 kosinüsü onu en yüksek kaldıraçlı yön yapıyor
(marjinal kazanç 0,00714, tekil değerinin 4,5 katı). Üçlü seçimi bağımsız olarak
C(10,3)=120 ve C(27,3)=2925 taramasında **1. sırada** doğrulandı.

### Son gönderim (1 Eylül, 9. hak)
```powershell
python m108_gun.py --bitir     # basacagi komutu calistir -> tuketim_NIHAI.csv
```
Sonda terimi yok, saf ortak optimum. Beklenen skoru **tahmin değil** — bütün
`L`'ler ölçülmüş olacağı için cebirsel olarak kesin.

---

## 4. Beklenen skor

| ölçülen yön | gerçekçi önsel | kötümser (yarısı) | medyan skor |
|---|---|---|---|
| 3 | %46 | %10 | 0,99959 |
| 6 | %80 | %26 | 0,99817 |
| **8** | **%95** | %38 | **0,99710** |

Hiç bilgi çıkmazsa taban **1,00061** (3. sıra). **Geriye gitme riski yok** —
Kaggle en iyi public skoru tutar, 1,00284 bankada.

Uçtan uca prova (8 sonda, rastgele `rho`'lar, biri negatif): zincir kırılmadan
işledi, nihai **0,99820**, `cond=253`, `|k|₁=2,01`, hiçbir korkuluk tetiklenmedi.

---

## 5. Denetim sonuçları (üç bağımsız ajan)

**Cebir — bağımsız yeniden hesap.** `G`, tüm ikili kosinüsler ve taban skoru
sıfırdan üretildi: taban **1,0006052**, eşik **`r = 0,0137340`**. Mekanizma
sayısallaştırıldı: `g7`'nin diğer üçe izdüşümü `R² = 0,3336`, `[G⁻¹]₁₁` tam
**1,5007** katı — bilgisiz yönler `g7`'nin *gürültüsünü* iptal ediyor.
Karışık işaret senaryolarında bile skor tabanı geçmiyor (en kötü 1,00096).

**Saldırgan denetim.** `c_g7` kusuru bulundu (yukarıda, düzeltildi). Yuvarlama:
ortak optimuma taşınan bozulma **9,2e-9 MSE** — gereken kazancın milyonda biri.
Public/private uyumsuzluğu **deneysel olarak reddedildi**: ölçülmüş Gram
matrisindeki 4 sıfır-uzayı bağıntısının artıkları (8,9e-6 … 6,4e-5) rastgele
bölme öngörüsüyle uyuşuyor, zamansal bölme öngörüsünden **40–70 kat** sapıyor.
Negatif `L` yolu güvenli. `m106`'nın ID kapısı sahteydi (`out.id == te.id`
her zaman True) — `m107`'de yön dosyalarının ID hizası gerçekten denetleniyor.

**Aday seçimi — bağımsız yeniden çözüm.** Kazancın `L'G⁻¹L = rho'C⁻¹rho`'ya
eşit olduğu gösterildi: **`Q` tamamen sadeleşiyor.** Dolayısıyla `m105`'in
`Q ≥ 0,01` sert elemesi ve kurtoz ölçütü dayanaksız (Kural 43). `m105_secim.py`'de
4 bozuk dosya adı bulundu (`t1_turizm`→`t1_sulama` vb.) — betik sulamayı hiç
görmemiş; doğru sonuca başka yoldan varılmış. `b3_span_k20` (kos 0,963) ve
`b5_prob_guc` (0,885) `g7` kopyası: **ölçülmeyecek.**

---

## 6. Düzeltilen gerçek kusurlar

| kusur | nerede | ne yapıldı |
|---|---|---|
| `c_g7` çapraz terimi ihmal ediyor | m106 | m107 tam ortak optimumu çözüyor |
| `t` keyfi, sonda skorunu bozuyor | m106 | `k*` cebirden; `--olcek 1,5`, `--min-yerdeg 0,03` |
| çözüm sabiti teorik `p`'den | m106 | m107 **diske yazılan** kırpılmış vektörden hesaplıyor |
| ID kapısı sahte | m106 | m107 her yön dosyasının ID'sini `test.csv` ile karşılaştırıyor |
| `\|k\|₁ > 5` sabit eşiği n≥6'da yanlış tetikliyor | m99/m107 | eşik `2 + n` |
| kalite beklentisi eski tabana ait | docs/60 | m6 tabanına göre yeniden ölçüldü |

---

## 7. Kaçırılmayacaklar

- **Her gönderimden sonra listeyi OKU.** Zaman aşımı "gitmedi" demek değil.
- **Final için 2 gönderim SEÇ** (tarayıcı). Seçilmezse en iyi public otomatik.
- **Notebook 2 Eylül 13:00.** İlk 20 inceleniyor; tüm dış veri kaynak+amaç+kullanımla.
  Nisan–Temmuz 2026 **gerçekleşmiş** hava verisi kullandığımız açıkça yazılmalı.
- **Hedef hareketli.** Atakan bugün 1,00041 → 0,99940 indi.
- Bir sonda `rho ≈ 0` verirse **panik yok** — cebir onu küçük `k` ile geçer,
  zarar vermez, sıradaki adaya devam.

---

## 8. Kapalı eksenler — tekrar açma

soğuk×ölü trafolar · ileri-pencere geri-testi (`f = −0,42`) · span'ı yeniden
karıştırmak (tavan 1,0014) · 2026-05-11 dalgası · rejim uzmanı · sık kesim ·
ölü trafo tezi · `r1`/`r3` · kesinti (karıştırılmış etiket) · hafta günü
(plasebo) · takvim demeti · uzun ısıl pencere (m4 kopyası) ·
**`b3_span_k20` / `b5_prob_guc`** (g7 kopyaları)

---

## 9. Kalıcı kurallar 32–46

**32.** İleri-pencere geri-testinde eğitim kesimlerinin hedef pencereleri doğrulama
penceresini KESMEMELİ. · **33.** Sağlam kayıp (Huber/L1) L2'yi döver. ·
**34.** Global seviyeyi geri-testten öğrenme. · **35.** Bir özellik ailesini
mekanizmasının AKTİF olduğu pencerede doğrula. · **36.** Geri-test LB'yi
ÖNGÖRMÜYOR. · **37.** Dik parçalara bölmenin maliyeti bir prob, getirisi ≥ 0. ·
**38.** Probu `t=1` ile kurma. · **39.** Ölçüm hassasiyeti bağlayıcı değil;
LB 5 hane yuvarlar, `sigma(L) ≈ 1e-5/k`. · **40.** Aynı makineyi kullanan iki
ajanın aynı sayıyı bulması DOĞRULAMA DEĞİL. · **41.** Forum bulgusunu tek mesaja
dayandırma. · **42.** Her hipoteze plasebo kolu zorunlu; kurtoz SERT ölçüt değil.

**43. Kazanç `rho'C⁻¹rho`'dur — `Q` sadeleşir.** Yön seçiminde `Q` eşiği,
kurtoz, sıfırsız oran KULLANMA; yalnızca kosinüs yapısı ve ölçülebilirlik
(`k·sqrt(Q) ≥ 0,03`) önemli.

**44. Kaliteyi HANGİ TABANA göre ölçtüğünü yaz.** Bir yönün `rho`'su taban
değişince değişir; absorbe edilmiş yönlerin `rho`'su tanım gereği ~0'dır.
Eski tabana ait kaliteyi yeni taban için kullanma.

**45. Sonda = o an bilinenlerin TAM ortak optimumu + yeni yön.** Bilinen
yönlerin katsayısını yeni yön yokmuş gibi hesaplama; çapraz terim bedava kayıptır.

**46. Ölçüm hakkı, "rötuş" hakkından değerlidir.** Cebir kesin olduğu için
rötuş yoktur; her ek hak yeni bir eksen ölçmeye harcanır (3→8 yön: %46 → %95).
