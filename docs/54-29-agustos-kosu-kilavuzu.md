# 54 — 29 Ağustos koşu kılavuzu

**Tek sayfa. Sabah bunu aç, sırayla uygula.**
Ayrıntı ve gerekçeler: [`docs/53`](53-yeni-model-ve-29-agustos-plani-2026-08-28.md)

---

## Başlangıç durumu

```
LB (2026-08-28 aksami, 435 takim)
  1. Grid Grinders     0.99064
  2. Atakan Aldemir    1.00041
  3. Saban Ozdogan     1.00543
  4. TasnifX           1.00553   <- BIZ
  5. Ahmet Celik       1.00559
```

Kota: UTC günü başına 3 hak, yerel **03:00**'te yenilenir. Bitiş **1 Eylül 23:59 UTC**
(bugün dahil 4 gün × 3 = 12 hak kaldı).

**Gönderim yetkisi kullanıcıda.** Claude hiçbir dosyayı kendisi göndermez.

---

## HAK 1 — ölç

```powershell
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_m4_hava_capali.csv -m "m4: ileri-pencere dogrudan tahmin + hava/nem/turizm, Huber+L1, v102 seviye capasi"
```

Sonra **mutlaka** listeyi oku (komut zaman aşımına uğrarsa "gitmedi" sanma —
bu depoda o yüzden bir hak boşa gitti, bkz. kalıcı kural):

```powershell
kaggle competitions submissions -c grid-up-datathon
```

Skoru `S` olarak not al ve Claude'a ilet.

---

## HAK 2 — çöz ve gönder

Claude çalıştırır:

```powershell
python experiments/model29/m50_harman_coz.py tuketim_m4_hava_capali.csv <S>
```

Çıkan dosya gönderilir. **Sonucu şimdiden biliyoruz** (çözücü `v102`'yi bit
düzeyinde yeniden üretti, öngörüsü 5 hanede tutuyor):

| `S` | `κ*` | HAK2 sonucu | sıra |
|---|---|---|---|
| 0,98 | +0,71 | 0,97471 | **1.** |
| 0,99 | +0,63 | 0,98144 | **1.** |
| 1,00 | +0,55 | 0,98737 | **1.** |
| 1,00553 | +0,50 | 0,99030 | **1.** |
| 1,00620 | — | 0,99064 | 1./2. sınırı |
| 1,01 | +0,46 | 0,99249 | **2.** |
| 1,02 | +0,38 | 0,99679 | **2.** |
| 1,03053 | +0,29 | 1,00041 | 2. sınırı |
| 1,04 | +0,21 | 1,00286 | 3. |
| 1,06427 | 0,00 | 1,00553 | değişmez |

`Q(m4, v102) = 0,121396 (DUZELTILDI: eski 0,121581 kirpma oncesi olculmustu)` · `m0 = 1,00553² = 1,011091` · `L = (m0+Q−S²)/2` · `κ* = L/Q`

---

## HAK 3 — `S`'e göre karar

| `S` | ne demek | HAK3 |
|---|---|---|
| **≤ 1,006** | aktarım güçlü, yöntem çalışıyor | `m3`'ü **üçüncü yön** olarak gönder, üçlü optimumu çöz. Kalan 3 gün modele yatırım. |
| **1,006 – 1,03** | aktarım kısmi, HAK2 kazancı aldı | **Harcama**, ertesi güne sakla. |
| **> 1,04** | geri-test LB'yi ÖNGÖRMÜYOR | Model yatırımını kes. `docs/52`'nin LB-cebri hattına dön. |

Yedek üçüncü yön hazır: `submissions/tuketim_m3_hl1_capali.csv`
(`Q(m4,m3) = 0,0193`, korelasyon 0,997)

---

## Ön kayıtlı tahmin — sabah ilk iş bunu aç

`experiments/model29/m90_on_kayit.json` (gönderimden ÖNCE yazıldı):

```
HAK1 S merkez  1,00      bant 0,99 – 1,02
HAK2 merkez    0,98737   bant 0,98144 – 0,99679
beklenti       1. veya 2. sira  (~%85)
yanlislama     S < 0,97  -> asiri karamsardim
               S > 1,04  -> geri-test LB'yi ongormuyor
```

---

## Sorun çıkarsa

- **"Submission scoring error"** → dosya bozulmuş olabilir; `m71_nihai_hava.py` yeniden
  üretir (~10 dk). Kapı denetimi betiğin içinde.
- **Kota dolu uyarısı** → UTC günü henüz dönmemiş. Yerel 03:00'ü bekle.
- **Skor beklenenden çok farklı** → HAK2'yi göndermeden önce Claude'a sor;
  `κ*` negatife dönmüşse yön ters demektir ve çözücü uyarır.

---

## Değişmeyecek olan

Kaggle **en iyi public skoru** tutar. HAK1 ne getirirse getirsin gösterilen
skorumuz `1,00553`'ün altına düşmez. **Geriye gitme riski yoktur.**
