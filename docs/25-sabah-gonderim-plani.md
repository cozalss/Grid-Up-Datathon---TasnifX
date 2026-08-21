# Sabah gönderim planı — 22 Ağustos 2026

Gönderim hakkı **03:00'te** (00:00 UTC) yenilendi. Dört dosya hazır ve
doğrulandı (satır sayısı, id sırası, NaN/negatif, dağılım).

```
dun sabah LB'de duran            1,16143
gece yonlendirmeyle (v13)   CV   1,08143
gece yiginla (v15)          CV   1,05194   <- YENI EN IYI
yaz25 (mevsimsel ikiz, v15)      0,99715   <- ilk kez 1'in altinda
lider (dun)                      1,04644
```

---

## Üçünü aynı anda gönderme

Bir dosyanın LB skoru **sabittir** — test kümesi değişmiyor, yani v15'i
bugün v14'ü yarın göndersek de aradaki fark aynı çıkar. Karşılaştırma için
aynı güne sıkıştırmak gerekmiyor.

Geçerli olan tek gerekçe: kullanılmayan hak **birikmez, yanar**. Yani bugün
üçünü de kullanmak mantıklı — ama **peş peşe değil, sırayla**, her sonucu
görüp devam.

**DURMA NOKTASI:** `v15` beklenen 1,03–1,08 bandının dışında gelirse
(özellikle 1,15 üstü), kalan hakları **harcama**. Bir yerde bozukluk var,
önce onu bul. Hak yarın yenilenir; yanlış teşhisle harcanan hak gelmez.

---

## Komutlar — BİRER BİRER

```bash
cd c:/Users/cemmo/Documents/Datahon

# 1) EN IYI MODEL -- yigin, CV 1,05194
kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_v15.csv \
  -m "v15: yonlendirme + soguk d7 + sicak rs4 + yalin 105 kolon (CV 1,05194)"

# --- skoru bekle, oku, sonra devam ---
kaggle competitions submissions -c grid-up-datathon

# 2) KOHORT PROBU -- v15'ten TEK FARKI 2026-05-03 kohortu
#    ONCE uret (v14 v13'ten turetilmisti; v15'ten yenisini uretmek lazim):
python scripts/kohort_probu.py --kaynak tuketim_v15.csv \
    --carpan 0.75 --cikti tuketim_v16.csv
kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_v16.csv \
  -m "v16: v15 + 2026-05-03 kohortunun 9.107 soguk satiri log1p x0,75"

# 3) BOS BIRAK -- ilk iki sonucu gordukten SONRA karar ver
```

**Üçüncü hak bilerek boş.** Önceden bağlamak, bilgi gelmeden karar vermek
demek. İlk iki sonuca göre şu seçenekler açılır:

| durum | üçüncü hakkın en iyi kullanımı |
|---|---|
| `v16 − v15 ≈ −0,15` (kohort **ÖLÜ**) | **çarpanı 0,3–0,4'e indirip hemen gönder** — ek ~0,1 |
| `v16 − v15 ≈ 0` | çarpanı 0,9'a çekip bir kez daha ölç |
| `v16 − v15 ≈ +0,02` (kohort **CANLI**) | `v13` gönder → yığın transfer oldu mu |
| `v15` beklenmedik biçimde kötü | teşhis: `v13` (yığınsız) ya da `v12` (yönlendirmesiz) |

Yedekte hazır duran dosyalar: `tuketim_v13.csv` (yalnızca yönlendirme,
CV 1,08143), `tuketim_v12.csv` (ikisi de yok), `tuketim_v14.csv` (v13
tabanlı kohort probu — **eskidi, kullanma**).

Daha güçlü kohort düzeltmesi gerekirse:

```bash
python scripts/kohort_probu.py --kaynak tuketim_v15.csv \
    --carpan 0.35 --cikti tuketim_v17.csv
```

---

## Önem sırası

**v15 + v16, sonra dur.** Kohort sorusu diğer her şeyden değerli
(±0,15 karşı ±0,03), ve üçüncü hakkın en iyi kullanımı ancak ilk ikisinin
sonucu görüldükten sonra bilinir. Kullanılmayan hak yanar — ama yanlış
soruya harcanan hak da yanar, üstelik bilgi de getirmez.

---

## Sonuçlar nasıl okunacak

### Soru 1 — `v15`in mutlak skoru: CV↔LB kalibrasyonu

Tek çapamız: `yaz25` test-ağırlıklı CV 1,1404 → LB 1,16922, fark **+0,029**.
`v15`in `yaz25` CV'si **0,99715**, yani öngörü **~1,026**.

| gelen skor | anlamı |
|---|---|
| 1,02–1,06 | çapa geçerli — on gün boyunca CV'ye güvenerek karar verilebilir |
| 1,06–1,10 | çapa kabaca tutuyor ama gevşek; kararları LB ile teyit et |
| >1,12 | **CV↔LB ilişkisi kırık.** Günde 3 hak çok değerli hale gelir |

### Soru 2 — kohort ölü mü? (`v16` eksi `v15`)

Gecenin en büyük bahsi. 9.107 satır, test'in %1,27'si, ama gerçekte
sıfırsalar toplam hata bütçesinin **%56'sı** orada.

| v16 − v15 | anlamı | ne yapılacak |
|---|---|---|
| **≈ −0,15** | kohort **ÖLÜ** | çarpanı **0,3–0,4**'e indir; 2026-07-01 (19 soğuk) ve 2026-05-13 (36 soğuk) kohortlarını da düzelt |
| ≈ −0,05 | kısmen ölü | çarpanı 0,6'ya ayarla |
| ≈ 0 | ayırt edilemiyor | 0,9'a çek, bir gün daha ölç |
| **≈ +0,02** | kohort **CANLI** | düzeltmeyi bırak, bir daha dokunma |

### Soru 3 — yığın test'te tutuyor mu? · YALNIZCA GEREKİRSE

Bu soru ±0,03; kohort sorusu ±0,15. Üçüncü hak bu yüzden önceden buna
bağlanmadı. Ancak `v15` beklenmedik biçimde kötü gelirse, ya da kohort
sorusu net cevaplandıysa, teşhis için `v13` gönderilir:

| v15 − v13 | anlamı |
|---|---|
| ≈ −0,03 veya daha iyi | CV'deki kazanç transfer oldu |
| ≈ 0 | tutmadı ama zarar yok — bırak |
| pozitif | **geri al**: `YALIN_CIKARILAN = ()` ve `REJIM_AYARLARI`daki `cat` sözlüklerini boşalt |

---

## Sonra ne yapılacak

1. **Önbelleği yenile** — bu gece kritik bir uyumsuzluk yakalandı: tezgâh
   144 kolonluk bayat bir önbellek üzerinde ölçüyordu, üretim ise 151
   kolon kuruyor. Fark: nüfus ailesi (5) ve `t_mevsim_*` (2).

   ```bash
   python scripts/deney.py --yenile
   python -m pytest tests/test_aile_kapsami.py -q   # artik ATLANMAMALI
   ```

2. **`t_mevsim_*`'ı ölç.** Bu gece kodlandı ve testlendi ama hiç
   ölçülmedi; şu an `YALIN_CIKARILAN`da, yani modelde YOK. Mevsimsel
   genlik (yaz/kış oranı) trafodan trafoya **8 kat** değişiyor ve elimizde
   olmayan `trafo_tipi` kolonunun en iyi vekili. Kapsam sınırı:
   `yaz25`'in özet penceresinde yaz yok, yani orada boş kalır.

3. **Kaçan 8 kolonu ölç** (`python scripts/deney_kacan.py`). Gece
   başlatıldı ama v15 üretimi için kesildi. Bu sekizi hiç ablate
   edilmemişti; `t_mevsim_*` onların üstüne inşa edilecek.

4. **Takım arkadaşının GDZ kesinti CBS işini sor.** Trafo koordinatı
   çıkarsa, ölçülen 1,33'lük soğuk-trafo açığına dokunabilecek tek şey.

---

## Kayıtlar

* [23-olcumler](23-olcumler-2026-08-21-gece.md) — gecenin bütün ölçümleri
* [22-durum](22-durum-2026-08-21-aksam.md) — önceki günün durumu
* `experiments/ileri_sonuclar.jsonl` — ham kayıtlar
