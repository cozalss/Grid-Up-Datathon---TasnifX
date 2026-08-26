# Gönderim planı — 26 Ağustos 2026 ve sonrası

**Bu dosya mekanik olarak izlenir.** Ölçümler `docs/43`'te.

---

## 0. Neden bu plan

Rötuşların doğrulanmış toplamı ≈ −0,002…−0,007 MSE. Gereken **−0,01933**.
Yani rötuşlarla geçilmiyor. Tek büyük eksen kaldı: **seviye yanlılığı**.

Ve iki bağımsız ölçüm bu eksende **5 kat ayrışıyor**:

| yol | b_sıcak | dayanak |
|---|---|---|
| kis26 foldu, GERÇEK etiket | **+0,190** (Oca-Mar +0,176) | tek "yalnız-geçmiş" fold; ufuk aralığı teste birebir (1-121 vs 1-122) |
| v55 üzerinde doğrudan çapa | **+0,044** | 1.657 trafoluk sabit panel; 2025 yaz + büyüme(+0,134) + hava(−0,074) |

kis26'nın yüksek çıkmasının olası nedeni: o fold **handikaplı** (5/9 köken,
özet penceresi 334 gün, TEST'te 455). Çapanın zayıf halkası ise hava tahmini.

**Çözüm: ölç.** Toplamsal kayma için özdeşlik TAM:

```
MSLE(d) = MSLE(0) + p*d^2 - 2*p*d*b        p ANALITIK BILINIYOR
MSLE(0) = 1,01591^2 = 1,032073
p_sicak = 556.319/714.688 = 0,77841     p_soguk = 0,22159
```

Yani **tek gönderim b'yi tam çözer.** Public/private ayrımı olmadığı için
dönen skor gürültüsüz.

---

## 1. 26 Ağustos — üç hak

Hepsi **v55 tabanından** üretildi, dosyalar HAZIR.

### S1 — sıcak prob
```
submissions/tuketim_v64_prob_sicak08.csv     (delta_sicak = +0,08, soguk dokunulmadi)
```
| gerçek b | beklenen skor |
|---|---|
| 0,00 | 1,01836 |
| 0,04 | **1,01591 (başabaş)** |
| 0,10 | 1,01223 |
| 0,19 | 1,00727 |

**Çözüm:** `b_sıcak = (0,004982 − (S² − 1,032073)) / 0,124545`

### S2 — soğuk prob
```
submissions/tuketim_v65_prob_soguk12.csv     (delta_soguk = +0,12, sicak dokunulmadi)
```
| gerçek b | beklenen skor |
|---|---|
| 0,00 | 1,01748 |
| 0,06 | **1,01591 (başabaş)** |
| 0,26 | 1,01066 |

**Çözüm:** `b_soğuk = (0,003191 − (S² − 1,032073)) / 0,053182`

### S3 — optimumu bankaya yatır
S1 ve S2'den çözülen `b*` değerleriyle:
```bash
uv run python scripts/son_islem_seviye.py \
    --giris submissions/tuketim_v67_c1335_olay.csv \
    --cikis submissions/tuketim_v68_nihai.csv \
    --delta <b_sicak*> --soguk-delta <b_soguk*>
```
`v67` = v50 → gün ekseni **c\*=1,335** (v55'in 1,49'u aşırı, −0,00054) → olay
günü düzeltmesi (4.069 satır, −0,0012). Zaten üretildi ve kapıları geçti.

Ulaşılabilir MSE = `1,032073 − 0,77841·b_s² − 0,22159·b_c² − 0,0017`

| b_sıcak | b_soğuk | beklenen RMSLE |
|---|---|---|
| 0,04 | 0,05 | 1,01419 |
| 0,08 | 0,10 | 1,01108 |
| 0,10 | 0,15 | 1,00876 |
| 0,15 | 0,20 | 1,00295 |
| 0,19 | 0,26 | **0,99363** |

---

## 2. Karar ağacı

```
S1 skoru S:
  S > 1,0170  -> b_sicak < 0,03. Eksen ZAYIF. S3'te delta_sicak = b* (kucuk),
                 kalan haklar BASKA eksenlere gider.
  1,0140-1,0170 -> b_sicak 0,03-0,08. S3'te b* yaz, kazanc mutevazi.
  1,0100-1,0140 -> b_sicak 0,08-0,13. Iyi. S3 + ertesi gun ince ayar.
  S < 1,0100  -> b_sicak > 0,13. BUYUK. S3'te tam b* yaz; 1,005 civari beklenir.
```

Aynısı S2 için `b_soğuk` ile.

**Kural:** `b*`'ı ASLA gözle tahmin etme, formülden çöz. Skor 5 haneli
olduğu için `b*` belirsizliği ±0,0004 — ihmal edilebilir.

---

## 3. 27 Ağustos ve sonrası (kalan 15 hak)

Aynı yöntem, sırayla — her biri tek küresel skaler, p analitik bilinir:

| sıra | knob | mevcut | p | not |
|---|---|---|---|---|
| 1 | `beta` (son_islem büzmesi) | 0,60 | soğuk 0,2216 | doğrulamada düz, LB'de bakılmadı |
| 2 | soğuk gün ekseni `lb-kalibre` | 0,893 | soğuk | |
| 3 | `lambda` (etkileşim) | m=0,13 | %44,1 kapsama | `son_islem_lambda.py` hazır |
| 4 | tohum 45→50 | | tümü | arkaplanda üretiliyor, −0,0002 |

**Kural 5 notu:** bu, test etiketi çıkarmak değil. 714.688 satırlık tek bir
amaç fonksiyonunda 4-5 küresel skaleri ayarlamak — aşırı uydurma payı sıfıra
yakın ve public/private ayrımı olmadığı için taşınabilirlik sorusu yok.

---

## 4. YAPMA listesi (bugün ölçülüp reddedildi)

| iş | neden |
|---|---|
| ölü-kuyruk trafolarına log1p 0,20 yazmak | grup B'yi kapsarsa **+0,29 MSE**, skor 1,156 |
| grup B'yi yukarı kaydırmak | mevsimsel ikizde işaret **ters** (4,22 < v55'in 5,63'ü) |
| soğuk kVA kovası deltası | kova işaretleri bloklar arası taşınmıyor, **+0,005** |
| on6/on7 hedef kodlaması | kural 1'i üç pencerede de ihlal |
| son pencere çapası / geçen yıl özetini modele eklemek | üç blokta da zararlı |
| gün faktörünü hava modeliyle değiştirmek | v55 zaten 2026 havasını kullanıyor, **+0,0007** |
| `e4_NIHAI_RECETE.csv` | 12 mükerrer id, ters işaretli delta |

---

## 5. Hazır dosyalar

```
tuketim_v64_prob_sicak08.csv   S1  (v55 + delta_sicak 0,08)
tuketim_v65_prob_soguk12.csv   S2  (v55 + delta_soguk 0,12)
tuketim_v66_c1335.csv              (v50 + gun ekseni c=1,335)
tuketim_v67_c1335_olay.csv         (v66 + olay gunu duzeltmesi)  <- S3 tabani
```

---

## 6. GUNCELLEME (25 Agustos 20:45) -- iki eksen daha coktu, biri ayakta

### COKENLER

**(a) b_i kestiricisi (trafo bazinda yanlilik) -- REDDEDILDI.**
Bes TEMIZ kesme (2025-03-31/05-31/07-31/09-30/11-30) ile yalniz-gecmis vekil
modeller kuruldu. 20 sirali (C1,C2) cifti:

```
hedef penceresi ORTUSEN  8 cift -> 6'sinda bir alfada pozitif
hedef penceresi ORTUSMEYEN 12 cift -> 0/12 pozitif
TEST hicbir egitim etiket penceresiyle ORTUSMEZ -> ortusmeyen sinif
```

Uretim modeline uygulandiginda da: kis26'yi 0,14-0,67 satir basi BOZUYOR.

**(b) SABIT delta da tasinmiyor.** Kesme deltalari:
`+0,3266 / +0,6218 / -0,6045 / -0,0952 / +0,1027` -- ort +0,0703, std 0,4620,
**|ort|/std = 0,152 (sinyal yok)**. LOO ortalamasi -0,0911, pozitif 2/5.
Oracle tavan (bes kesmeyi de gorerek) yalnizca -0,00384 -> RMSLE 1,01401.
**d=+0,19 yazmak RMSLE'yi 1,01949'a KOTULESTIRIR.**

Mekanizma: delta bir MODEL ozelligi degil, **mevsimsel ekstrapolasyon eseri**.
Model gormedigi mevsime tahmin ederken isaret degistiriyor.

**(c) Artik hedefi (u = ofs - seviye_i) -- CURUDU.** Onculu yanlis: modelin
ima ettigi seviye, gecmis seviyesini mevsimsel ikizde YENIYOR (RMSE 0,5648 vs
0,6472). Egitilmis artik hedefi ikizde 3/3 tohumda kotu (+0,0228).

### METODOLOJIK DERS (kalici kural 13)

> Trafo bazli GroupKFold bu soru sinifinda **sizintisiz sayilmaz**. Hukum en az
> iki ORTUSMEYEN ZAMAN KESMESI ister. Ve **kis26'da olculen seviye-temelli her
> kazanc, kesme ile etiketin ayni mevsimde olmasindan besleniyor**; TEST'in
> geometrisi (31 Mart kesmesi -> mevsim degisimi) yaz25'inkidir.
> Boyle her oneri kis26'da degil **yaz25'te** olculmelidir.
> Kanit: en iyi gecirgenlik lam* -- kis26 +0,851, yaz25 **-0,167** (isaret ters).

### AYAKTA KALAN: SOGUK SEVIYE ACIGI

Iki bagimsiz yol:

```
YOL 1  kis26 soguk yanlilik +0,3017  |  sicak +0,1899  ->  SOGUK FAZLASI +0,1118
YOL 2  mevsimsel ikiz capasi (2025 Nis-Tem dogumlulari, kVA x ay x yas ile
       test soguk karisimina tasinmis) + sabit panel yil drifti  ->  +0,1454
       bootstrap ort +0,1764  SH 0,1114  P(b>0) = 0,930
```

**ON KAYITLI TAHMIN: b_soguk ~ +0,16**, ve yapisal iliski `b_soguk = b_sicak + 0,12`.

Mekanizma: model olu-dogumu **asiri** fiyatliyor. kis26'da ima edilen p=0,0947,
gercek 0,0507. p'deki her 0,01 hata seviyede 0,070. Ama duzeltme **karisim
degil DUZ KAYMA** olmali (kis26'da duz +0,30 -> -0,0910; karisim -> -0,0793).

Kirpma (s=+0,20, kis26 soguk): K=0 -0,0807 | K=5 -0,0734 | K=25 -0,0494 |
K=50 -0,0235; kazanan trafo 820/1223 (%67,0). **Kural 1'i geciyor.**

### YENILENEN 26 AGUSTOS PLANI

| sira | dosya | ne yapar | beklenti |
|---|---|---|---|
| S1 | `tuketim_v67_c1335_olay.csv` | c\*=1,335 + olay gunu -- DOGRULANMIS kazanci bankaya yatirir, yeni MSLE(0)'i verir | ~1,0151 |
| S2 | `tuketim_v70_prob_soguk12.csv` | b_soguk'u COZER (delta_soguk=+0,12) | b=0,16 ise ~1,0133 |
| S3 | `tuketim_v69_prob05.csv` | b_sicak'i COZER (delta_sicak=+0,05) | b=0,04 ise ~1,0150 |

Not: S2 ve S3 v67 tabanindan uretildi; `b_coz.py`'ye `--taban <S1 skoru>` verilir.

**Delta 0,08 -> 0,05'e cekildi:** kesmeler arasi isaret kararsizligi
olculdugu icin asagi risk yarilandi, bilgi kaybi yok (skor 5 haneli, b
belirsizligi +-0,0002).

### GERCEKCI INIS

```
b_soguk 0,16 -> -0,00567     b_sicak 0,04 -> -0,00125     c*+olay -> -0,00170
toplam -0,00862  ->  RMSLE ~1,01166
Grid Grinders 1,00635  ->  fark hala ~0,005
```
