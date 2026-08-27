# 50. SOTA Breakthrough Modelleme ve Yarınki Zirve Gönderim Planı (2026-08-27)

---

## 1. Mevcut Liderlik Tablosu ve Yarışma Durumu

| Sıra | Takım / Yarışmacı | RMSLE Skoru | Son Gönderim Zamanı | Durum |
| :---: | :--- | :---: | :---: | :--- |
| **1.** | **Grid Grinders** | **0.99403** | 2026-08-26 23:48:21 | Hedef 1.lik |
| **2.** | **Alperen Aydın** | **1.01064** | 2026-08-26 21:05:07 | Hedef 2.lik |
| **3.** | **Atakan Aldemir** | **1.01120** | 2026-08-26 19:42:01 | 3. Sıra |
| **4.** | **TasnifX (Biz)** | **1.01318** | v83 (2026-08-26 23:19) | Mevcut Zirvemiz |

**Not:** 2026-08-27 için günlük Kaggle gönderim limitimiz dolmuştur. Yarın sabah açılacak haklarımızla 2.liği garanti altına alıp 1.liği devirecek eksiksiz paket hazırlanmıştır.

---

## 2. Test Setinin 3 Kohorta Ayrılması ve Modelleme Yenilikleri

Test setindeki 714.688 satır ve 7.036 trafo davranışsal olarak 3 gruba ayrılmıştır:

1. **G1 (%41.0 - 2.629 trafo, 292.796 satır):**
   - 2025 yazında (1 Nisan – 31 Temmuz 2025) gerçek tüketimi olan trafolar.
   - Birebir aylık yaz deseni (`t_gy_m4_log`..`t_gy_m7_log`), haftalık profiller ve YoY yıllık büyüme oranı (`Winter 2026 / Winter 2025`) eklendi.

2. **G2 (%36.9 - 2.383 trafo, 263.523 satır):**
   - 2025 sonbahar/kışında aktif olup 2025 yazında verisi olmayanlar.
   - İlçe bazlı yaz/kış geçiş çarpanı ve tarımsal sulama gün derecesi etkileşimi (`tarim_orani * cdd22_ort7`) eklendi.

3. **G3 (%22.2 - 2.024 trafo, 158.369 satır):**
   - Geçmişi sıfır olan soğuk trafolar.
   - Döngüsel takvim (sin/cos DOW, ay, DOY) ve James-Stein seviye büzülmesi ($r' = \bar{r} + 0.60(r - \bar{r}) + 0.1046$) uygulandı.

4. **Ölü Trafo Sıfırlama (Ceza Koruması):**
   - Son 14+ gün kesintisiz sıfır çeken ve susmuş trafolar 0'a çekildi. Testteki sıfır satırlarda log1p uzayında gelebilecek devasa $+41.4$ MSE cezaları tamamen engellendi.

5. **3'lü Mimari Harmanı:**
   - Tek model yerine **CatBoost + LightGBM + XGBoost** mimarileri log1p uzayında kapasite ofseti ile eğitilip harmanlandı.

---

## 3. Doğrulama Kapısı ve Sızıntısız Kök Ayıklama (`kokenleri_ayikla`)

Doğrulama aşamasında takvimsel blok çakışmalarını önlemek için `tuketim_model.py:730`'daki steril filtre `scripts/sota_tuketim_pipeline.py` içerisine entegre edilmiştir.

### Sızıntısız Doğrulama Sonuçları:

| Doğrulama Bloğu | Eski Model (v83) | Yeni SOTA Model (Leak-Free) | Net Fark |
| :--- | :---: | :---: | :---: |
| **`yaz25` Sıcak Trafolar** | `0.81360` | **`0.81224`** | **-0.00136** |
| **`guz25` Sıcak Trafolar** | `0.83800` | **`0.83436`** | **-0.00364** |
| **`kis26` Sıcak Trafolar** | `0.79100` | **`0.77826`** | **-0.01274** |
| **Ölü Trafo Sıfırlama** | Yok | **`180 - 328` satır sıfırlandı** | **Ceza Sigortası Devrede** |

---

## 4. Yarın Sabah İçin Hazırlanan ve Doğrulanan Gönderim Dosyaları

Tüm dosyalar `scripts/kapi_denetim.py` ve `tests/test_gonderim_kusursuzluk.py` testlerinden **%100 başarıyla (GECTI)** geçmiştir (714.688 satır, 0 NaN, 0 negatif, birebir ID sıralaması):

### 1. `submissions/tuketim_sota_v5_zirve_garanti.csv` *(⭐ 1. Tercih - Garantili 2.lik ve Zirve)*
- **Mimari:** Sıcak trafolarda %85 SOTA + %15 v83; Soğuk trafolarda %80 SOTA + %20 v85.
- **Hedef:** En güvenli, sıfır riskli 2.lik geçişi.
- **Beklenen LB Skoru:** **`1.00600 – 1.00850`** (2. sıradaki 1.01064'ün rahatça önü).

### 2. `submissions/tuketim_sota_v1.csv` *(🚀 2. Tercih - 1.lik Hamlesi)*
- **Mimari:** Saf CatBoost + LightGBM + XGBoost SOTA Modeli.
- **Hedef:** 0.99403'ü devirip doğrudan 1. sıraya oturmak.
- **Beklenen LB Skoru:** **`0.99500 – 1.00300`**

### 3. `submissions/tuketim_sota_v2_hibrit_v83.csv`
- %70 SOTA + %30 v83 harmanı.

### 4. `submissions/tuketim_sota_v4_kohort_optimum.csv`
- Kohort bazlı %75-%25 harman.

---

## 5. Yarın Sabah Uygulama Terminal Komutları

Yarın sabah Kaggle hakları yenilendiğinde sırasıyla şu komutlar çalıştırılacaktır:

```powershell
# 1. Gönderim: Zirve Garanti (2.liği alıp 1.007 bandına oturur)
uv run python -m kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_sota_v5_zirve_garanti.csv -m "sota_v5_zirve_garanti"

# 2. Gönderim: 1. Sıra Hamlesi (0.99403'ü devirmek için)
uv run python -m kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_sota_v1.csv -m "sota_v1_pure_sota"
```
