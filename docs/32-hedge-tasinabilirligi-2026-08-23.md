# Hedge taşınabilirliği — ölçüldü (23 Ağustos 2026, 00:15)

Gönderim planı ([29-durum](29-durum-2026-08-22-gece.md) §2) bir **çıkarıma**
dayanıyordu:

> `v26 − hedge` → v20'nin örtük skoru → kalibrasyon üçüncü kez sınanır

Bu, hedge'in v23 ile v20'ye **aynı şekilde** uygulandığını varsayar. v20 ve v23
farklı modeller; tam-sıfır satırları farklı olabilirdi. Ölçüldü — değilmiş.

Araç: `scripts/gonderim_karsilastir.py` (bu bulgu için yazıldı, tekrarlanabilir).

---

## 1. Tam-sıfır kümeleri BİREBİR aynı

```
python scripts/gonderim_karsilastir.py submissions/tuketim_v20.csv \
                                       submissions/tuketim_v23.csv

TAM-SIFIR KUMELERI
  A (v20)       8,748
  B (v23)       8,748
  kesisim       8,748
  yalniz A          0
  yalniz B          0
```

İki farklı model, **aynı 8.748 satır**. Şaşırtıcı değil: bir satırın tam sıfır
olması, trafonun eğitim sonunda ölü olmasından geliyor — model ayarından değil.

---

## 2. Müdahale de aynı — iki koşu rakamı rakamına eşit

```
v23 -> v25_hedge                    v20 -> v26_v20hedge
  degisen satir     8.748             degisen satir     8.748
  ort |log farki| 0,00420             ort |log farki| 0,00420
  RMS log farki   0,04947             RMS log farki   0,04947
  ek kare toplami 1749,35             ek kare toplami 1749,35
  sifir-disi degisim:  0              sifir-disi degisim:  0
```

Tek bir basamak bile farklı değil. `olu_hedge.py` kova tabanlarını trafonun ölü
gün sayısından okuyor; o sayı eğitim verisinden geliyor, gönderim dosyasından
değil — bu yüzden iki tabana da aynı değerler yazılıyor.

---

## 3. Sonuç: hedge'in etkisi tabanlar arasında taşınabilir

O 8.748 satır v23 ve v20'de **aynı** (ikisi de 0), v25 ve v26'da da **aynı**
(ikisi de aynı taban). Dolayısıyla kare hata toplamlarında:

```
SSE(v25) - SSE(v23)  =  SSE(v26) - SSE(v20)      TAM ESITLIK
```

RMSLE'ye çevirince eşitlik tam değil (`sqrt` doğrusal değil) ama fark ihmal
edilebilir — en kötü hal hesabı iki tabanda **+0,00120** ve **+0,00119** verdi,
yani 0,00001 sapma.

**Pratik sonuç: `v26`, düz `v20`'yi kesin olarak baskılıyor.** İkisi de aynı
bilgiyi veriyor (v20'nin örtük skoru), ama v26 tahtada 1,0287, v20 ise 1,0312
duruyor. [29-durum](29-durum-2026-08-22-gece.md) §2'deki eleme kararı doğruymuş —
ve artık gerekçesi çıkarım değil, ölçüm.

---

## 4. En kötü hal — bağımsız olarak doğrulandı

[30-harici-veri-denetimi](30-harici-veri-denetimi-2026-08-22-gece.md) §11
"hepsi gerçekten ölüyse **+0,00118**" demişti. Kapalı formülle yeniden hesaplandı:

```
RMSLE_yeni = sqrt(RMSLE_taban^2 + 1749,35 / 714.688)

  1,01820  ->  1,01940   (+0,00120)     v25 tabani
  1,03120  ->  1,03239   (+0,00119)     v26 tabani
```

Belgedeki sayı tutuyor. Riziko/kazanç: **+0,0012 karşılığında −0,0023…−0,0033**,
yaklaşık 2,5:1 asimetri.

---

## 5. Fark haritası — hangi gönderim hangi soruyu ölçüyor

```
v25 vs v23  ->   8.748 satir (%1,22)   B sorusu: hedge tutuyor mu
v25 vs v26  -> 162.016 satir (%22,7)   A sorusu: hangi taban dogru
v20 vs v23  -> 162.016 satir (%22,7)   (ayni fark, hedge oncesi)
```

Üç dosya iki soruyu birbirinden bağımsız olarak kapatıyor. Ölçümlerin public
leaderboard dilimine dayanıklılığı: [31-yarisma-kurallari](31-yarisma-kurallari-dogrulama-2026-08-22.md) §5.

---

## 6. AÇIK KALAN — sığınağın gerekçesi kendi verisiyle ölçülmemiş

`v26` "sığınak" olarak konuldu; gerekçesi *"v23'ün soğuk ağırlığı yanlış olabilir"*
ve bunun tek dayanağı `kis26 = 1,11354`. [29-durum](29-durum-2026-08-22-gece.md) §3
bu sayıyı kendi yıldızıyla işaretlemiş:

> `(*izolasyon kosusundan; v23 kosusu yaz25 icin 0,97558 verdi)`

Yani **A belirsizliğine biçilen %50, v23 üstünde hiç ölçülmemiş bir sayıya
dayanıyor.** Bu sığınağı gereksiz yapmaz — belirsizlik gerçek, ve v26'nın
maliyeti düşük — ama o %50 göründüğünden yumuşaktır ve öyle kayda geçmelidir.

Yarınki `v25` vs `v26` sonucu bu soruyu **doğrudan** kapatacak.

---

## 7. Kod notu — sessiz bir boşluk (engelleyici değil)

[`scripts/olu_hedge.py`](../scripts/olu_hedge.py) `taban()` fonksiyonu `olu == 0`
için `0.0` döner; `expm1(0) = 0`, yani böyle bir satır **sessizce sıfırda kalır**
ve betik uyarmaz — doğrulaması yalnızca NaN/negatif bakıyor.

Bu koşuda tetiklenmedi: her iki hedge'li dosyada tam-sıfır satır sayısı **0**,
ölçülerek doğrulandı. Yine de betik, hedge'lemeyi amaçladığı bir satırı atlarsa
bunu söylemelidir. Yarışma sonrası düzeltilecek; gönderim öncesi dokunulmuyor.
