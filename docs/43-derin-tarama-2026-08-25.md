# Derin tarama — 25 Ağustos 2026 (akşam)

**Tetikleyici:** yarışma sahibinin forum cevapları + "veriyi detaylı taradık mı"
sorusu. Sekiz eksenli ajan taraması + bağımsız ölçümler.

---

## 0. Yarışma sahibinin cevapları — ne değiştirdi

| soru | cevap | sonuç |
|---|---|---|
| public/private ayrımı | **YOK**, zaman skalası baz | **LB = final skor.** Shake-up riski sıfır; LB'den öğrenmek tam meşru ve kesin. `docs/42 §3`'teki "public LB alt küme mi" sorusu KAPANDI — D_sicak20 probu gereksiz. |
| tüketim = 0 | devre dışı / tüketim yok | Gerçek sıfır. Ölçüm eksiği değil. |
| yalnız-train trafolar | devreden çıkarılanlar | 332 trafo |
| yalnız-test trafolar | yeni devreye alınanlar | 2.024 trafo |

---

## 1. Panelin yapısı — ilk kez tam çıkarıldı

```
train 1.226.237 satir  2025-01-01..2026-03-31 (455 gun)  5.344 trafo
test    714.688 satir  2026-04-01..2026-07-31 (122 gun)  7.036 trafo
ortak 5.012 | yalniz-train 332 | yalniz-test 2.024
```

**Test ilk-gün dağılımı bir OLAY gösteriyor:**

```
2026-04-01 -> 3.928 trafo
2026-05-11 -> 2.222 trafo   <- 1.326'si YENI, 896'sinin train gecmisi VAR
2026-05-03 ->   141 | 2026-04-30 -> 119 | 2026-05-07 -> 102
```

896 eskinin train kaydı **erken bitmiş** (493'ü 2026-03-27, 137'si 2025-08-20).
Yani 05-11 toplu bir *devreye alma* değil, bir **raporlama/entegrasyon**
olayı. Train'de de aynı desen var (2025-06-17, 07-28, 09-10, 11-25, 2026-03-26).

**Sıfırlar bir DURUM, günlük olay değil:**

```
P(sifir | dun sifir) = 0,9755      P(sifir | dun sifir degil) = 0,00114
4.792/5.344 trafonun HIC sifiri yok;  312 trafonun sifir orani >0,9
```

Mevsimsel sıfır (sulama tipi: kış sıfır/yaz dolu) yalnız **11 trafo** — bu
hipotez ölü. Aylık sıfır oranındaki mevsimsellik (%7,4 → %2,3) kompozisyon.

---

## 2. ÖLÜ TRAFO EKSENİ — bir MAYIN bulundu ve etkisiz hale getirildi

Kuyruğu ≥60 gün sıfır olan 204 test trafosu (21.599 satır) tek bir grup
DEĞİL. Panel bunları kusursuz ikiye bölüyor:

| grup | tanım | trafo | satır | v55 ort log1p | gerçek (ileri pencere) |
|---|---|---|---|---|---|
| **A** | train sonuna kadar rapor etti | 145 | 16.872 | 0,286 | ikiz optimal 0,221 — **v55 DOĞRU** |
| **B** | raporu kesilmiş, teste sonradan girdi | 59 | 4.727 | 5,626 | **22/22 gözlemde DİRİLİYOR**, ort log1p ~6,97 |

> **Bu 204 trafoya log1p 0,20 yazmak dMSE'yi +0,29 büyütür — skor 1,156'ya
> fırlar.** İlk taramanın "hepsini 0,2'ye çek" önerisi bir mayındı.
> Ölü-kuyruk temelli her kural GRUP A ile sınırlı olmalıdır.

Ayırt edici tek kural: trafonun train'deki **son kayıt tarihi** 2026-03-27+ mı
(A) yoksa daha eski mi (B).

---

## 3. Gün ekseni / hava — KAPALI (v55 doğru)

`hava_gunluk.parquet` **2026-08-28'e kadar dolu**: test penceresinin GERÇEK
havası elimizde. Bu yeni bir bilgi kaynağı gibi görünüyordu; değil:

```
kismi kor(v55 gun faktoru, 2026 GERCEK sicaklik | 2025 gun faktoru) = +0,403
kismi kor(v55 gun faktoru, 2025 sicaklik        | 2025 gun faktoru) = +0,132
```

**v55 zaten 2026'nın gerçek havasını kullanıyor.** Hava tabanlı bağımsız gün
faktörü kestirimi ile v55 arasındaki RMS fark 0,0892; hava modelinin kendi
örneklem-dışı RMSE'si 0,0928. Yani fark modelin kendi hatasının altında.
Değiştirmek **dMSE +0,00065 (ZARARLI)**.

### 2026 yazı 2025'ten SERİN — ve v55 bunu uyguluyor

```
             t_ort   CDD22
2025 Nis-Tem 23,02   3,473
2026 Nis-Tem 21,76   2,398      <- Temmuz 30,29 -> 27,82 (2,5 C serin)
2025/2026 Ocak-Mart: 9,99 / 9,76  (neredeyse ayni)
```

Bu, ajanların "seviye açığı +0,08…+0,15" tahminlerinin neden şişkin olduğunu
açıklıyor: hepsi 2026'nın serinliğini hesaba katmıyor.

---

## 4. SEVİYE KAYMASI — gerçek ama ajanların dediğinden KÜÇÜK

İki yönlü sabit etki (`ofs_it = α_i + m_t`, yalnız pozitif satırlar) ile:

```
2026 buyume kuklasi (hava-ayarli)     = +0,0635  (SH 0,0079, t=+8,06)
hava farki (2026-2025, ayni model)    = -0,0738
beklenen 2026 Nis-Tem (2025'e gore)   = -0,0103
v55'in verdigi                        = -0,0394
>>> ACIK = +0,0291        (dogrusal trend surumu: +0,0484)
```

Bağımsız çapa — **ulusal yük serisi test penceresini kapsıyor** (2026-08-20'ye
kadar):

```
                 HAM YoY   HAVA-AYARLI YoY
Ocak-Mart        +0,0294      +0,0274 (t=+2,79)
Nisan-Temmuz     +0,0000      +0,0198 (t=+1,78)
```

Ulusal büyüme yaza **taşınıyor** (0,020 vs 0,027). Yerel panel ulusaldan
~2,2 kat hızlı büyüyor (0,059/0,027) → yerel yaz büyümesi ≈ +0,043.

**Hüküm: b ≈ +0,03 … +0,05.** Ajan 8'in önerdiği δ=0,08 bu bantta ZARARLI
olurdu (b=0,03 ise +0,00125 dMSE).

---

## 5. KAPANAN EKSENLER (bu gece, gerçek etiketle)

Önbellekteki doğrulama tahminleriyle (`sicak_tahmin.npz`, üç blok, trafo
bazında 5 kat çapraz):

| eksen | ölçüm | hüküm |
|---|---|---|
| **son pencere çapası** modele eklemek | yaz25 −0,0044, guz25 −0,0072, kis26 +0,0005 (K=1'de işaret döner) | model güncel seviyeyi **zaten tam soğuruyor** |
| **geçen yıl aynı mevsim** (`t_gy`) ek ağırlık | kis26 c3=−0,031, kazanç −0,00079 | **ZARARLI** — model onu da soğurmuş |
| soğuk trafo seviyesi (kimlik/ilçe) | as-of OOF R² 0,015 | kapalı (üçüncü kez doğrulandı) |
| **ölü-doğan sınıflayıcı** | sızıntısız AUC **0,529** | kapalı |
| kimlik öneki > ilçe iddiası | ağırlıksız grup-ort std artefaktı; LOO'da on4 R² 0,006 / ilçe 0,002 | **ÇÜRÜDÜ** |
| harman uzayı (Jensen) | log uzayı, 1e-15 hassasiyette doğrulandı | kayıp yok |
| medyan harmanı | ortalamadan 1,8-2,2 kat kötü | reddedildi |

### Kendi hatam: sızıntı

İlk kurulumda ölü-doğan sınıflayıcısı **AUC 0,813** verdi. Komşu öznitelikleri
train'in SON 122 gününden hesaplanmıştı ve geç doğan trafolarda kendi hedef
penceresiyle örtüşüyordu. As-of (doğumdan önceki 122 gün) kurulumda **0,529**.
Ekibin `docs/41 §3` hükmü doğruymuş.

---

## 6. ELDE KALAN — uygulanabilir kazançlar

| # | iş | satır | dMSE | güven | betik |
|---|---|---|---|---|---|
| 1 | olay günü düzeltmesi (doğum/dönüş/son gün) | 4.081 | −0,00116 | yüksek | `scripts/son_islem_olay.py` |
| 2 | panel sınır günü (giriş/çıkış) | 2.387 | −0,00067 | yüksek | ajan 4 (1 ile ÖRTÜŞÜR — birleştir) |
| 3 | grup B kaldırması | 4.727 | −0,00393 | orta | ajan 4 |
| 4 | soğuk kVA profili genliği | 158.369 | −0,00481 | orta | ajan 3 |
| 5 | seviye kayması δ≈0,04 | 500.295 | −0,0006…−0,0019 | orta | `scripts/son_islem_seviye.py` |
| 6 | 35 → 50 tohum | tümü | −0,00020 | yüksek | `birlestir_tohum.py` |

Toplam iyimser ≈ **−0,011 MSE → RMSLE ~1,0105**. Grid Grinders 1,00635.
**Rötuşlar tek başına yetmiyor** — bu, `docs/42`'nin hükmünü doğruluyor.

---

## 7. Kalıcı kurallara EKLENENLER

9. **Ölü-kuyruk kuralları GRUP A ile sınırlıdır.** Ayırt edici: train son
   kayıt tarihi ≥ 2026-03-27. Grup B (raporu kesilip panele dönen) diriliyor.
10. **Mevsimsel bir seviye iddiası, test penceresinin HAVASI ile
    karşılaştırılmadan verilmez.** 2026 Nis-Tem 2025'ten 1,3 °C serin; bunu
    atlayan her "büyüme açığı" ölçümü ~0,05 log birim şişer.
11. **Komşu/geçmiş temelli öznitelikler AS-OF hesaplanır** (hedef trafonun
    doğumundan/kesme anından ÖNCE biten pencere). Aksi halde kohort etiketi
    sızar ve AUC 0,53 → 0,81 gibi sahte sonuç verir.
12. **Gün-bazlı olay ölçümünde panel gün etkisi çıkarılır**; yerel referans
    tek başına yetmez (2025-07-28'de panel +0,47, işaret bile ters çıkıyordu).
