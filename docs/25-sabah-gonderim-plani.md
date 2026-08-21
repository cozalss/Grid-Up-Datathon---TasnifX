# Sabah gönderim planı — 22 Ağustos 2026

Gönderim hakkı **03:00'te** (00:00 UTC) yenilendi. Üç dosya hazır ve
doğrulandı (satır sayısı, id sırası, NaN/negatif, dağılım).

---

## Komutlar — sırayla çalıştır

```bash
cd c:/Users/cemmo/Documents/Datahon

# 1) ANA MODEL -- yonlendirmeli, CV 1,08143
kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_v13.csv \
  -m "v13: rejim yonlendirmesi (sicak maske %15 / soguk maske %100), 3/1/1 harman, 3 tohum (CV 1,08143)"

# 2) KOHORT PROBU -- v13 ile ARADAKI TEK FARK 2026-05-03 kohortu
kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_v14.csv \
  -m "v14: v13 + 2026-05-03 kohortunun 9.107 soguk satiri log1p x0,75"

# 3) YONLENDIRME KONTROLU -- v13 ile ARADAKI TEK FARK yonlendirme
kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_v12.csv \
  -m "v12: yonlendirmesiz kontrol (maske %22,16 tek model)"

# skorlari oku
kaggle competitions submissions -c grid-up-datathon
```

---

## Sonuçlar nasıl okunacak

### Soru 1 — kohort ölü mü? (`v14` eksi `v13`)

Bu, gecenin en büyük bahsi. 9.107 satır, test'in %1,27'si.

| v14 − v13 | anlamı | ne yapılacak |
|---|---|---|
| **≈ −0,15** | kohort **ÖLÜ** | çarpanı 0,75'ten **0,3–0,4**'e indir, ayrıca 2026-07-01 (19 soğuk) ve 2026-05-13 (36 soğuk) kohortlarını da düzelt |
| ≈ −0,05 | kısmen ölü | çarpanı 0,6 civarına ayarla |
| ≈ 0 | ayırt edilemiyor | çarpanı 0,9'a çek, bir gün daha ölç |
| **≈ +0,02** | kohort **CANLI** | düzeltmeyi tamamen bırak, bir daha dokunma |

Beklenen: eğer ölüyse skor **1,08 → 0,93**; tam düzeltmeyle **0,71**'e kadar.

### Soru 2 — yönlendirme test'te tutuyor mu? (`v13` eksi `v12`)

| v13 − v12 | anlamı |
|---|---|
| ≈ −0,017 veya daha iyi | doğrulama bloklarındaki kazanç test'e **transfer oldu** |
| ≈ 0 | tutmadı ama zarar da yok — bırak |
| pozitif | **geri al**, `REJIM_MASKELERI = None` |

### Soru 3 — CV↔LB kalibrasyonu (`v13`in mutlak skoru)

Tek çapamız: `yaz25` test-ağırlıklı CV 1,1404 → LB 1,16922, fark **+0,029**.
Yeni modelin `yaz25` CV'si **1,04789**, yani öngörü **~1,077**.

- Gelen skor 1,06–1,09 bandındaysa → çapa geçerli, on gün boyunca CV'ye
  güvenerek karar verebiliriz.
- Bandın dışındaysa → CV↔LB ilişkisi kırık, her kararı LB ile doğrulamak
  gerekir ve günde 3 hak çok değerli hale gelir.

---

## Sonra ne yapılacak

1. **Yığın deneyinin sonucunu oku** (`experiments/ileri_sonuclar.jsonl`,
   ad `YIGIN`). Eşiği geçtiyse yarının modeli o; geçmediyse `v13` kalır.
2. Kohort cevabı geldiyse **doğru çarpanla yeni dosya üret**:
   `python scripts/kohort_probu.py --carpan <deger> --cikti tuketim_v15.csv`
3. Takım arkadaşının GDZ kesinti CBS işini sor — trafo koordinatı
   çıkarsa, bu gece ölçülen 1,33'lük soğuk-trafo açığına dokunabilecek
   **tek** şey odur.

---

## Gecenin durumu — tek bakışta

```
LB'de duran (dun sabahki model)     1,16143
gece kanitlanan, gonderilmemis      1,08143   <- v13
kohort probu, dogruysa              ~0,93     <- v14
lider (dun)                         1,04644
```

Ölçülen 16 yapılandırma, eşiği geçen 1 (rejim yönlendirmesi).
Ayrıntılı kayıt: [23-olcumler](23-olcumler-2026-08-21-gece.md).
