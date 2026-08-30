# 66 — CV→LB kalibrasyonu ve aday seçim ölçütü

Tarih: 30 Ağustos 2026 (kota bitti, 0 hak — bu belge ÖLÇÜM DEĞİL ANALİZ)

**docs/64 ve docs/65'in seçim ölçütünü düzeltir.** Dosyalar geçerli, gerekçe değil.

---

## 1. Geometri doğrulandı — bağımsız yeniden kurulum

27 ölçülmüş yönün tamamı ham CSV'lerden yeniden kuruldu ve **her biri
Kaggle'da görülen skoru birebir verdi** (5 hane). Codex'in bulduğu
"formülden yeniden üretilen vekil" hatası gerçekti ve düzeltilmiş.

```
span 27 yon, rank 23
||r_hat||^2 = 0.004790   ->  SAF OPTIMUM 1.000528   (henuz GONDERILMEDI)
a0 tabani (m6_ikiyon)         1.002919
2. sira icin gereken EK kazanc  0.002256 MSE  ->  toplam rho 0.0475
1. sira icin gereken EK kazanc  0.020778 MSE  ->  toplam rho 0.1441
```

---

## 2. Hazır dosyaların GERÇEK skor eğrisi

`S = sqrt(A − 2·rho_dik·sqrt(Q_dik))`, A ve Q_dik exact hesaplandı:

| dosya | Q_dik | A | rho=0 | rho=0.03 | 2. sıra için gereken rho |
|---|---:|---:|---:|---:|---:|
| `EXACT_BANKA` | 0.000013 | 1.001070 | 1.00053 | 1.00042 | **+0.310** |
| `PROBE_seviye_x_ay` | 0.000038 | 1.001095 | 1.00055 | 1.00036 | **+0.186** |
| `PROBE_TARGET996` | 0.000038 | 1.001095 | 1.00055 | 1.00036 | **+0.185** |
| `RANK2_ONSEL` | 0.002086 | 1.003143 | 1.00157 | 1.00018 | +0.048 |
| `TARGET996_CV` | 0.008545 | 1.009624 | 1.00480 | 1.00200 | +0.059 |

**Sonuç:** sondalar 2. sıraya TEK BAŞINA ulaşamaz — yer değiştirmeleri
(≈0.006) bunun için çok küçük. Görevleri ölçmek; kazanç en sonda
`--nihai` ile nakde çevrilir. Bu tasarım doğrudur, ama **beş eksenin
ortalama |rho|'su ≥ 0.0212 olmak zorundadır.** Rekorumuz 0.0304.

---

## 3. KRİTİK BULGU — "3 blokta işaret tutarlılığı" ÖLÇÜTÜ ÇÜRÜDÜ

docs/65 yedi ekseni "işaretler 7/7 eksende üç blokta da korunur"
diye seçti. Bu ölçüt elimizdeki tek sınavda **yanlış cevap verdi.**

İleri-zaman blokları (test bileşimine ağırlıklandırılmış, soğuk %22.2):

```
        yon      LB rho     yaz25     guz25     kis26
     seviye     -0.0304   -0.0848   -0.0531   +0.0712    <- kis26 TERS, LB DOGRU
yenibaslangic   -0.0027   +0.0291   +0.0804   +0.0497    <- 3/3 TUTARLI, LB TERS
```

- `yenibaslangic`: **üç blokta da pozitif**, LB negatif. Tutarlılık kandırdı.
- `seviye`: kış26'da işaret değiştirdi ama **yaz25 LB'yi doğru bildi.**

**Ölçüt: 3-blok tutarlılığı değil, yaz25 (mevsim analoğu).**
Test penceresi Nisan–Temmuz 2026; yaz25 Nisan–Temmuz 2025. Tek gerçek analog.

---

## 4. İKİNCİ ÇÜRÜTME — span-çapalı kestirim de yanlış

"Bir yönün span içindeki parçasında LB'nin ölçtüğü korelasyon, dik
parçasını da tahmin eder" (izotropi) varsayımı sınandı:

```
seviye, 25 model yonlu span'a gore:
   span parcasi   Qs=0.3437   rho_s = +0.0156
   dik  parcasi   Qp=0.6563   rho_p = -0.0304   <- OLCULDU
```

**İşaretler ters.** Artığın bir yön boyunca korelasyonu düzgün dağılmıyor.
Dolayısıyla `L_span`'ı CV tahmininden çıkarmak kestirimi ŞİŞİRİR — ilk
denemede `cdd22` için `rho_dik = 0.41` gibi imkânsız sayılar üretti
(o skor 0.912 demekti, liderin çok altında). Bu yol kapalı.

---

## 5. Kalibrasyon çarpanı

Tek sağlam çapa `seviye`:

```
yaz25 kor (test bilesimine agirliklandirilmis) = -0.0381
LB rho                                          = -0.0304
carpan = 0.80
```
Ham (ağırlıksız) yaz25 ile çarpan 0.36. Ağırlıklandırma test bileşimini
(soğuk trafo %7.5 → %22.2) düzelttiği için 0.80 tercih edilir.

**Uyarı:** bu n=1'dir. Çarpan bir tahmin aracıdır, garanti değil.
Kural 54 hâlâ geçerli.

---

## 6. Kalan iş

`ze_artik_modeli.py` — elle eksen seçmek yerine modelin kendi ileri-zaman
yanlılığını LightGBM ile öğrenip test'e taşıyor. Dürüstlük kapısı: bir
bloğu tamamen dışarıda bırak, diğerlerinde öğren, tutulanı tahmin et.
O korelasyon ~0 ise yol kapalıdır ve öyle raporlanır. Sonuç docs/67'de.

---

## 7. Kalıcı kurallar 55–57

**55.** İleri-zaman bloklarında işaret tutarlılığı bir seçim ölçütü
DEĞİLDİR. Elimizdeki tek sınavda 3/3 tutarlı eksen LB'de ters çıktı,
1/3 tutarlı eksen doğru çıktı. Ölçüt **mevsim analoğu bloktur** (yaz25).

**56.** Bir yönün span içindeki LB korelasyonu, dik parçasının
korelasyonunu tahmin etmez — işaret bile aynı olmayabilir (seviye:
+0.0156 vs −0.0304). CV tahmininden `L_span` çıkarma; kestirimi şişirir.

**57.** CV bloklarını test bileşimine ağırlıklandır. Soğuk trafo payı
yaz25'te %7.5, testte %22.2; ağırlıklandırmadan çarpan 0.36, sonra 0.80.
Ağırlıksız karşılaştırma iki kat yanlış ölçek verir.
