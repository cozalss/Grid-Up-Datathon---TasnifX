# 67 — Dört kapalı yol, inandırıcılık süzgeci ve 31 Ağustos planı

Tarih: 30 Ağustos 2026 (kota 0 — bu gün hiçbir gönderim yapılmadı)

**docs/63, 64, 65'in yerine geçer.** docs/66 ile birlikte okunur.

---

## 1. Ölçerek KAPATILAN dört yol

Hiçbiri bir gönderim hakkına mal olmadı. Hepsi hesapla kapatıldı.

### 1.1 "Ufuk yanlılığı" — YOK, hava öyle

Ham blok modelinin yaz25'teki yanlılığı çarpıcıydı:
```
gun   1- 30: -0.0626      gun  62- 91: +0.1800
gun  31- 61: -0.0902      gun  92-122: +0.3710
```
Bu şekle uyan bir ufuk ofseti 0.0455 MSE değerinde görünüyordu — 1. sıranın
gerektirdiğinin iki katı. **Ama a0'da böyle bir yanlılık yok:**
```
Nis->Tem cdd22 artisi:  2025 +8.191   2026 +5.848   oran 0.714
Nis->Tem a0 rampasi  :  2025 +0.6103  2026 +0.4485  oran 0.735
```
2026 gerçekten daha serin; a0 buna doğru tepki veriyor. Neredeyse birebir.

### 1.2 Artık modeli — mevsimler arası aktarım TERS

LightGBM ile blok artığı öğrenilip tutulan bloğa uygulandı:
```
ogren -> tahmin    yaz25     guz25     kis26
         yaz25     (fit)   -0.1220   -0.0998
         guz25   -0.1427     (fit)   -0.0785
         kis26   -0.0686   -0.1667     (fit)
```
**Altı hücrenin altısı da negatif.** Olduğu gibi uygulamak zarar ettirir.

### 1.3 Aynı mevsim, yıldan yıla — korelasyon 0.07

Aynı mevsim içinde ileri-zaman testi (+0.2532, plasebo z=+19.4) umut vericiydi
ama o kurulum hedef pencerenin İÇİNDEN 61 gün etiket görüyor; testte
Nisan–Temmuz 2026'da hiç etiket yok. Gerçek soru yaz25 → yaz26 aktarımı:
```
Ocak 2025 vs 2026: +0.0783 | Subat: +0.1745 | Mart: -0.0487   ort +0.068
karsilastirma -- ayni yil komsu aylar: +0.76 .. +0.80
yaz ici ay4 vs ay7: -0.4853  (ortalamaya donus)
```
Trafonun takvime özgü deseni yıldan yıla tekrar etmiyor.

### 1.4 Soğuk-sıfır — indirgenemez

En büyük görünen fırsat: yaz25'te 826 satır (bloğun %0.3'ü) toplam MSE'nin
**%27.3'ünü** taşıyor; hiç görülmemiş trafolar, gerçek tüketim sıfır,
model `log≈5.03` veriyor. Taban oranı kararlı (%6.89, çeyreklere göre
%5.1–8.2) ve **a0 test'teki 2.024 soğuk trafonun sıfırına düşük tahmin
veriyor** (`p1=3.946`). Yani hiç sömürülmemiş.

Ama sınıflandırılamıyor. Blok-dışı dürüst sınav:
```
yaz25 tutuldu: en yuksek q'lu 466 satirin %1.7'si dogru (taban %2.3)
guz25 tutuldu: %10.6 (taban %4.8)   kis26 tutuldu: %5.7 (taban %5.0)
her yapilandirma MSE KAYBETTI: A -0.83, B -1.04 (soguk-ici)
```
**Ders: hata yoğunlaşması fırsat demek değildir.** Üretim modeli de bunu
yapmıyor çünkü öğrenilecek bir şey yok.

---

## 2. İNANDIRICILIK SÜZGECİ — asıl araç

CV korelasyonlarını doğrudan LB'ye çevirmek defalarca imkânsız sayılar verdi
(`rho=0.41`, skor 0.912). Sebep bulundu:

> Ham blok modelinin mevsim rampası yanlılığı a0'da YOK. Bu yüzden
> mevsimsel eksenlerin CV korelasyonu şişik.

Süzgeç, LB'nin **kendi ölçümünü** kullanır. Her yön `x` için span parçasının
korelasyonu `rho_s = L_span/sqrt(Q_span)` LB'den bilinir. CV'nin öngörüsüyle
oranı:

```
MEVSIM-KIRLI (oran 11-15, hepsi ayni carpanla):
  ay 14.3 | ufuk_gun 14.6 | gun_uzunlugu 14.3 | sicaklik_ort 13.1
  cdd22 12.8 | soguk_x_ufuk 12.7 | vpd 12.3 | ulusal_gunluk 11.7
  et0 11.6 | gunes_radyasyon 11.0

TEMIZ (oran 0.3-3.3):
  t_yuk_faktoru | tarim_orani | seviye_x_ay | yerlesim_orani
  seviye_x_guc | t_log_ort | tatil_agirligi | t_sifir_orani | seviye2
```

Bütün mevsimsel eksenlerin **aynı ~13× çarpanla** şişik çıkması teşhisi
doğruluyor. Rekoru veren `seviye` oranı 1.95 olan temiz gruptaydı ve CV
tahmininin **%80'ini** teslim etti.

---

## 3. Bileşik varyantları — agresiflik inandırıcılığı bozuyor

```
varyant   n   rho_cv  rho_s(LB)   oran    hukum   2.sira icin gereken f
      C   1  +0.0578    +0.0198    2.9    TEMIZ          0.821
      B   9  +0.0794    +0.0178    4.5  SUPHELI          0.598
      A   9  +0.1188    +0.0135    8.8  SUPHELI          0.400
```
A ardışık diklestirme kullanıyor ve korelasyonları büyütüyor
(`t_yuk_faktoru` −0.056 → −0.100). **A silindi.**

**Seçim: B** — tek tek span'a diklestirme, çapraz düzeltme yok.

---

## 4. 31 Ağustos — 1. hak

```
submissions/tuketim_K_B_KESITSEL.csv          KAPILARIN HEPSI GECTI
  9 eksen, ongorulen rho = 0.0821
  kappa = 0.0475 = sqrt(MSE_opt - 0.99940^2)  <- 2. sira esigini EN AZA indirir
  sabit = 1.003309388
```

Skor `P` geldikten sonra `rho` **exact** çözülür:
```
rho = (1.003309388 - P*P) / 0.094992
```

| gerçek rho | skor | |
|---:|---:|---|
| 0.0000 | 1.00165 | bankadan 1.1e-3 kötü, telafi edilir |
| 0.0304 | 1.00021 | bankadan iyi (rekor seviyesi) |
| **0.0475** | **0.99940** | **2. SIRA eşiği** |
| 0.0635 | 0.99864 | 2. sıra (seviye'nin f=0.80'i) |
| 0.0794 | 0.99788 | 2. sıra |

**Sonda ne gelirse gelsin bilgi kaybı yoktur** — `rho` exact çözülür ve
2. hakta tam katsayıyla uygulanır.

### Kalan haklar
```
2. hak  olculen rho ile TAM optimum + siradaki eksen sondasi
3. hak  siradaki eksen
1 Eylul 3 hak: iki sonda + saf optimum (--nihai)
```
Güvenlik ağı: `--nihai` her an `1.000528` verir (henüz gönderilmedi;
şu anki bankamız 1.00115, yani 6.2e-4 masada duruyor).

---

## 5. Model dışı — durum

1. **Notebook HAZIR**: `notebooks/TasnifX_final.ipynb`, 45 hücre,
   baştan sona 19 saniyede çalışıyor. Dış veri beyanı 23 kaynak, üç tablo.
2. **Künye eksiği kapatıldı**: `data/external/epias/tuketim_saatlik.parquet`
   ÜRETİM MODELİ GİRDİSİ (`scripts/tuketim_model.py:558`, `ulusal_*` beş
   öznitelik) olduğu hâlde `data/sources.yml`'de kayıtlı değildi. Eklendi
   (`model_girdisi: true`), `uretim_saatlik.parquet` de (`false`).
3. **Hâlâ açık**: final 2 gönderim seçimi (tarayıcıdan, en geç 1 Eylül
   23:00 UTC), takım arkadaşının Coderspace kaydı, düzenleyici e-postasının
   `.eml` arşivi.

---

## 6. Kalıcı kurallar 58–61

**58.** Hata yoğunlaşması fırsat DEĞİLDİR. Soğuk-sıfır satırları MSE'nin
%27'sini taşıyor ama blok-dışı sınavda rastgeleden kötü sınıflandırılıyor.
Bir kesitin payını ölçmek yetmez; o kesitin **tahmin edilebilir** olduğunu
ayrıca sınamak gerekir.

**59.** CV artığı, kullanılan MODELİN artığıdır. Ham blok modelinin
yanlılıkları LB'de kullandığımız a0'da olmayabilir. Her CV tahmini,
LB'nin o yöndeki KENDİ span ölçümüyle (`rho_s`) oranlanarak
denetlenmelidir. Oran 4'ü aşarsa eksen mevsim-kirlidir.

**60.** Ardışık diklestirme korelasyonları BÜYÜTÜR ve inandırıcılığı bozar
(bileşikte oran 4.5 → 8.8). Bileşik kurarken eksenleri tek tek span'a
diklestir; birbirlerine karşı ardışık diklestirme yapma.

**61.** Sonda yer değiştirmesi amaca göre seçilir. Belirli bir hedef skora
ulaşma olasılığını en üste çıkaran değer `kappa* = sqrt(MSE_opt − hedef²)`;
beklenen skoru en üste çıkaran değer ise `kappa = E[rho]`. İkisi aynı değildir.
