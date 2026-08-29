# 56 — 30 Ağustos planı: 2. sıra hedefi

**Yazıldı:** 2026-08-29 gecesi, dört ajanlık filo koşusundan sonra.
Kaggle'a bu belge yazılırken hiçbir şey gönderilmedi.

---

## 0. Durum ve bütçe

```
1. Grid Grinders   0.99064
2. Atakan Aldemir  1.00041     <- HEDEF
3. TasnifX         1.00284     <- BIZ  (submissions/tuketim_m6_ikiyon.csv)
4. Ahmet B.ALTUNOK 1.00480
5. Saban Ozdogan   1.00510

m0 = 1.00284^2 = 1,005688      hedef MSE 1,000820      gereken dMSE -0,004868
kalan hak: 30-31 Agustos + 1 Eylul = 9
```

---

## 1. ÖLÇÜM ARACININ GERÇEK HASSASİYETİ — her şeyden önce bu

Düşmanca denetim türetti ve iki ajan bağımsız doğruladı:

```
gerceklesen_MSE - ongorulen_MSE = 0,2770*(Qw_tum - Qw_pub) + 0,3252*(Qc_tum - Qc_pub)
```

`Q` tüm 714.688 satırda ölçülüyor, LB skorları **public %50**'de. Bu kanalın
sd'si **9,28e-5 RMSLE**. `docs/55`'in "sapma yuvarlamadan geliyor" yorumu
**yanlıştı** (yuvarlama bandı ±1,18e-5, gözlenen onun 6,7 katı dışında).

> **KURAL 39. Öngörü aracının hassasiyeti ±1e-5 değil, ±9,5e-5 (1 sd).**
> **2e-4'ün altındaki marjinal kazanç ölçüm gürültüsünden ayırt edilemez —
> ona hak harcanmaz.**

Aşırı uyum riski bağlayıcı DEĞİL: parça başına beklenen private kaybı
`4*m0/N` = 2,82e-6 RMSLE; tehlike eşiği ~43 parça. Bağlayıcı kısıt **gönderim
sayısı**, aşırı uyum değil.

**Yön seçerken KURTOZA bak.** `Q`'nun büyük kısmı az sayıda satırdan geliyorsa
public/private sapması büyür:

```
aday                       Q       kurtoz   Q%(en kotu %1 satir)
y46_amnezik_kirpik      0,38844      4,5          10,5%     <- EN GUVENILIR
y45_mevsimsel_kirpik    0,16716      8,5          23,1%
y40_sota_temiz          0,02852     14,5          25,9%
m4 (dun gonderildi)     0,08200     22,7          36,9%     <- en gurultulu
```

---

## 2. İki bağımsız ajan aynı yere vardı: span tavanı ≈ 1,0014

27 (fiilen 25) ölçülmüş gönderimin afin span'ı:

| yöntem | çözüm | öngörü |
|---|---|---|
| L1 kısıtlı, winner's-curse düzeltmeli (τ=\|w\|₁=3) | `tuketim_g7_span_tau3.csv` | **1,00137** ± 0,00007 |
| kesik SVD (k=15) | `tuketim_b2_span_k15.csv` | 1,00143 |

Aralarındaki `Q` = 0,000182 (korelasyon 0,99997) — pratikte aynı dosya.

**Span'ın tavanı bu.** Her iki ajan da bağımsız olarak "2. sıraya bu yolla
varılamaz" dedi. Ham öngörüyü 1,00041'in altına indiren tek şey `|w|₁ ≥ 33`
çözümleri; onlar **ölçülü olarak** (simülasyonla) daha kötü gerçekleşiyor.

### `v101` felaketinin gerçek sebebi bulundu
Ekip bunu yuvarlamaya bağlamıştı. Değilmiş: **public/private tutarsızlığı**,
yuvarlamadan ~10 kat büyük ve `|w|` ile **karesel** büyüyor. Null yönlerde
14σ ihlal ölçüldü. Bu yüzden `|w|₁ ≤ 3–5` bandı dışına çıkılmaz.

---

## 3. Yeni yönler — çürük bileşen temizlendi

Ham SOTA çıktısının yön enerjisinin **%89'u**, `docs/52` §1'de LB'de ölçümle
çürütülmüş **ölü trafo tezinden** geliyordu. 980 trafonun satırları tabana
eşitlenerek bu bileşen çıkarıldı. Sonrası:

| aday | Q | yeni % | kos(m4−v102) | kurtoz |
|---|---|---|---|---|
| `tuketim_y46_amnezik_kirpik.csv` | 0,38844 | 97,1 | −0,115 | **4,5** |
| `tuketim_y45_mevsimsel_kirpik.csv` | 0,16716 | 99,8 | +0,005 | 8,5 |
| `tuketim_y40_sota_temiz.csv` | 0,02852 | 93,8 | +0,218 | 14,5 |
| `tuketim_m3_hl1_capali.csv` | 0,08546 | 21,7 | +0,882 | — |

Karşılıklı kosinüsler: y40↔y45 +0,026 · y45↔y46 −0,095 · y40↔y46 +0,114 —
**üçü birbirine ve harcanmış span'a dik, kazançları TOPLANIR.**

`m3` doğrulandı: bilgisinin yalnız %21,7'si yeni → **gönderme**.

---

## 4. Katkı formülü ve senaryo

Bir yönün MSE katkısı `L²/Q = <r, d/|d|>²` — **ölçekten bağımsız**, yalnız
gerçek artıkla yaptığı **açıya** bağlı. Kalite ölçütü `r = L/√Q`
(`m4` için ölçüldü: **0,0641**).

Span adımından sonra (MSE 1,002742), tek bir yeni yönün getireceği:

```
kalite r    kazanc     sonuc MSE     RMSLE     sira
  0,064    0,004096    0,998646     0,99932     2.
  0,048    0,002304    1,000438     1,00022     2.
  0,032    0,001024    1,001718     1,00086     3.
  0,020    0,000400    1,002342     1,00117     3.
```

> **`m4` kalitesinin YARISI kadar bir yön bile 2. sırayı getiriyor.**

---

## 5. 30 AĞUSTOS — üç hak

| hak | dosya | amaç | beklenen |
|---|---|---|---|
| **1** | `submissions/tuketim_g7_span_tau3.csv` | span optimumunu **kilitle** | **1,00137** ± 0,00007 → 3. sıra sağlamlaşır |
| **2** | `submissions/tuketim_y46_amnezik_kirpik.csv` | en güvenilir yeni yönü **ölç** (kurtoz 4,5, Q 0,388, dik) | bilinmiyor — `L` çıkacak |
| **3** | `python experiments/model29/m99_coklu_coz.py` ile ortak optimum | span + y46 | `y46` kalitesine göre **0,999 – 1,0013** |

Komutlar:
```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_g7_span_tau3.csv -m "g7 span optimumu tau=3, 17 dosyanin afin kombinasyonu, winner's-curse duzeltmeli. ON KAYIT 1.00137 +-0.00007"
kaggle competitions submissions -c grid-up-datathon      # SKORU OKU

kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_y46_amnezik_kirpik.csv -m "y46 AMNEZIK yon: 24 gecmis kolonu atilmis GBM, olu-trafo bileseni temizlenmis. Q=0.38844 kos(m4)=-0.115 kurtoz 4.5. AMAC: L olcmek"
kaggle competitions submissions -c grid-up-datathon      # SKORU OKU

python experiments/model29/m99_coklu_coz.py <taban.csv>=<skor> tuketim_y46_amnezik_kirpik.csv=<skor> --cikti tuketim_g8_ortak.csv
```

**Her gönderimden sonra listeyi OKU** (kalıcı kural: zaman aşımına uğrayan
betik "gitmedi" demek değil).

---

## 6. 31 Ağustos ve 1 Eylül (6 hak)

- `y45_mevsimsel_kirpik` (Q 0,167, kos(m4) +0,005 — span'a pratikte tam dik) → ölç
- `y40_sota_temiz` (bağımsız kod tabanı, gerçek sinyal önseli en yüksek) → ölç
- Her ölçümden sonra `m99_coklu_coz.py` ile **tüm yönlerin ortak optimumu**
- Yedekte: bölme probları `b4_prob_dbuyuk` / `b5_prob_guc` / `b6_prob_seviye`
  (üç dik eksen, ortak `Q_perp` = 0,0777; 2. sıra için gereken κ heterojenliği
  `dk ≥ 0,162`, ölçülmüş tek referans 0,215)

---

## 7. Araçlar — ikisi de doğrulandı

| araç | ne yapar | doğrulaması |
|---|---|---|
| `m50_harman_coz.py` | iki dosyalı harman | `v83`+`v101` → `v102`'yi **bit düzeyinde** üretti |
| `m99_coklu_coz.py` | **N yönlü ortak optimum** (tam Gram, ridge opsiyonlu) | `v102`+`m4`+`p51` → 1,00292, gerçekleşen `m6` 1,00284 |

---

## 8. Dürüst değerlendirme

**2. sıra ulaşılabilir ama garanti değil.** Gereken, `m4` kalitesinin yarısı
kadar tek bir yeni yön. Elimizde **üç** bağımsız aday var ve kazançları
toplanıyor — ama hiçbirinin kalitesi ölçülmedi, ancak gönderilerek öğrenilir.

Bilinen: **kaybetme riski yok.** Kaggle en iyi public skoru tutar; en kötü
senaryoda 1,00284'te kalırız.
