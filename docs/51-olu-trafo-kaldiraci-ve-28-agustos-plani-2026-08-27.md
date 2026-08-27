# 51. Ölü trafo kaldıracı — ölçümler ve 28 Ağustos planı (2026-08-27)

Bu belge `docs/50`nin üstüne geçer. `docs/50`deki tahmini skorlar ve "sıfır
risk" nitelemeleri ölçümle değiştirilmiştir.

## 1. 27 Ağustos sonucu

Üç hak kapalı çevrimle harcandı, iki tahmin de birebir tuttu.

| dosya | ön kayıtlı tahmin | gerçek |
|---|---:|---:|
| `tuketim_v80_optimum.csv` | 1.013412 | **1.01341** |
| `tuketim_v81_sicak08.csv` | prob | 1.01429 |
| `tuketim_v83_sicak_optimum.csv` | 1.013185 | **1.01318** |

Sıcak çekirdek ekseni çözüldü; orada kalan pay yok.

Gün sonu tablosu:

```
1  Grid Grinders   0.99403
2  Alperen Aydin   1.01064
3  Atakan Aldemir  1.01120
4  TasnifX         1.01318
```

## 2. SOTA hattındaki sızıntı — bulundu ve kapatıldı

Doğrulamada ek kökenler isim önekiyle (`startswith`) eleniyordu. Kökenler
takvim aralıkları olduğu için bu koruma çalışmıyordu:

```
yaz25   122 gunun 92'si egitimde  (%75)
guz25   122/122                   (%100)
kis26   121/121                   (%100)
```

Doğru koruma eski hatta zaten vardı (`tuketim_model.py::kokenleri_ayikla`,
aralık kesişimi). SOTA hattına aynısı taşındı. Düzeltme sonrası kazanç
**−0.11102 → −0.00136**; iddia edilen iyileşmenin %98'i sızıntıymış.

Fold güvenliğinin geri kalanı doğrulandı: özet penceresi etiketin bir gün
öncesinde bitiyor, `profil_kaynak` etiket penceresini dışlıyor. Başka sızıntı
bulunamadı. Üçlü mimari harmanı (CatBoost + LightGBM + XGBoost) gerçek.

## 3. Asıl kaldıraç: ölü trafo sıfırlaması

`sota_v1` ile `v83` arasındaki farkın ayrışımı:

```
sifirlanan  14.484 satir (%2.0) : %88.6
geri kalan 700.204 satir        : %11.4
```

Üç mimari, takvim, sulama ve kohort ayrımı birlikte farkın %11'i. Bahsin
tamamı sıfırlamada.

### Tutarlılık kontrolü

```
v83 toplam MSE                        1.02653
  maskeli 19.839 satirdaki pay        0.31794  (%31.0)  [HIPOTETIK ust sinir]
  geri kalan 694.849 satir            RMSLE 0.85372
```

0.85372, sızıntısız CV'nin sıcak trafo skoruyla (0.81224) uyuşuyor.

> **Düzeltme 1 — maske sayımı.** İlk analizlerde bu pay 0.91640 (%89.3) ve canlı
> şebeke RMSLE'si 0.3319 çıkmıştı. Sebebi bulundu: maske
> `v89["tuketim"] != v83["tuketim"]` ile, yani **tam eşitsizlikle** hesaplanıyordu.
> CSV yazımındaki `float → metin → float` gidiş dönüşü ~9.100 satırda son biti
> değiştiriyor (bağıl fark ~3e-16); bunlar maske sanılınca sıradan canlı trafolar
> cezaya ekleniyor ve pay 0.318 yerine 0.916 çıkıyor. Maske **bağıl toleransla**
> hesaplanmalıdır:
> `(|v89 − v83| / max(|v83|, 1e-9)) >= 1e-9`.
> `hesapla_ceza_bilancosu.py` ve `derin_trafo_istihbarati.py` düzeltildi.
>
> **Düzeltme 2 — normalizasyon.** Alt küme RMSLE'si `sqrt(SSE_altküme / n_altküme)`
> ile hesaplanır. Tam N ile normalize edilirse 0.8418 çıkar; doğrusu **0.85372**.
> Sonucu değiştirmiyor, ikisi de CV ile tutarlı.
>
> Ayrıca 0.31794 bir **üst sınırdır**; yalnız o satırların tamamı gerçekten 0 ise
> gerçekleşir.

## 4. Kuralın doğruluğu — etiketli veriyle ölçüm

Maskelenen **tam o trafolarla**, Nisan–Temmuz 2025 (testin mevsim ikizi):

```
Nisan   2.039 satir  %96.76 sifir
Mayis   2.091 satir  %96.27
Haziran 1.356 satir  %91.22
Temmuz    703 satir  %72.26   <- sulama reaktivasyonu
TUMU    6.189 satir  %92.60
```

Son 122 günde (Aralık–Mart) aynı trafolar **%97.0** sıfır; kıyas olarak canlı
trafolar %1.2.

### Panel yanlılığı

Ölü trafolar yaz aylarında panelden düşüyor (ölü/canlı yoğunluk oranı
0.85 → 0.25). Düşük sıfır oranları bu yüzden seçilim yanlılığı taşıyor:
kayıt üreten = geri dönen. Testte oran 1.00 — tüm grup mevcut.

Sürekli kapsamlı 13 trafoluk alt küme %78.9 veriyor (ters yönde yanlı).
Dolayısıyla gerçek oran **%79–97** bandında; daha dar bilinemiyor.

**Başa baş noktası %57.**

## 5. Neden bugüne kadar bulunamadı

Kural her blokta kaç satır sıfırlıyor:

```
yaz25 CV:        0 satir
guz25 CV:      180 satir  -> gercekten 0 orani %0.00
kis26 CV:      328 satir  -> gercekten 0 orani %0.00
TEST:       14.484 satir
```

Kural uzun özet penceresi istiyor; CV bloklarının penceresi 90–334 gün,
testinki 455 gün. CV'de testin binde beşi kadar ateşleniyor. Üstelik ölü
trafolar eğitim panelinde satır üretmediği için hedef penceresinde yalnız
geri dönenler görünüyor — veri ters sinyal veriyor.

Kendi geçmişimizde de olgu biliniyordu (`features/trafo.py` içindeki %96,0
ölçümü; v7 ölü-trafo öznitelikleri; v25/v27 hedge). Ama hep **yumuşak büzme**
denendi ve LB'de izole ölçülen kazanç 0.00008 çıktı. Sebebi:

```
log1p x0.90 -> kazancin %19.0'u
log1p x0.75 -> %43.7
log1p x0.60 -> %64.0
SERT SIFIR  -> %100.0
```

MSLE log uzayında cezalandırdığı için 250'yi 150'ye çekmek hatanın ancak
%17'sini siliyor. Yumuşak versiyonu ölçüp fikri atmışız.

## 6. Gönderilecek dosya: v89

`tuketim_v89_genis_taban.csv` = v83 tabanı + 19.839 satırda ay bazlı taban.
694.849 satıra dokunulmadı.

### Maske genişlemesi

SOTA kuralı 193 trafo yakalıyor. Eğitimde 455 gün boyunca hiç tüketmemiş 298
trafo var; 122'si kuralın dışında kalmış, bunların 58'i test setinde mevcut.
Ölçülen davranışları maskedekilerden iyi:

```
Nis-Tem 2025      4.514 satir  %100.00 sifir
Ara-Mar son 122g  3.016 satir  %100.00 sifir
```

**Toplam etkilenen: 251 trafo, 19.839 satır.**

### Ay bazlı taban

Sert sıfır bıçak sırtıdır. MSLE'de belirsizlik altındaki optimum sabit
`v* = (1−p)·b`; p ve b tam bu trafolardan ay ay ölçüldü:

```
Nisan   p=%96.8  ->  0.12 kWh
Mayis   p=%96.3  ->  0.17 kWh
Haziran p=%91.2  ->  0.82 kWh
Temmuz  p=%72.3  ->  6.67 kWh
ek grup p=%97.0  ->  0.23 kWh
```

### Beklenen skor

| sürüm | ölçülen p | −20 puan | −40 puan |
|---|---:|---:|---:|
| v87 sert sıfır | 0.92455 | — | — |
| v88 sabit 1 kWh | 0.92497 | 0.96602 | — |
| **v89** | **0.88447** | **0.94730** | 1.00622 |

Lidere göre başa baş `p ~= %50`. Ölçülen aralık %79–97.

## 7. Coğrafi doğrulama

251 trafonun dağılımı (veriden, doğrulandı): 192 İzmir, 59 Manisa. İlçe
yoğunlaşması Urla 38, Bayındır 20, Menderes 13, **Akhisar 12**, Konak 11,
Aliağa / Bergama / Tire / Ödemiş 10'ar, Salihli 9. Güç grupları: 26 trafo
≤100 kVA, 136 trafo 100–400, 65 trafo 400–1000, 22 trafo 1000–5000 kVA.

Tarımsal ilçelerdeki yoğunlaşma, Temmuz'da ölçülen %72.3'lük reaktivasyonla
tutarlı — v89'un 6.67 kWh Temmuz tabanı bu riske karşı.

> **Not.** `lokasyon` alanı sabit biçimli değildir: İzmir kayıtları üç parçalı
> (`İZMİR>GÜNEY BÖLGE>TORBALI`), Manisa kayıtları iki parçalı
> (`MANİSA>AKHİSAR`). İlçeyi ">" ile bölüp körü körüne üçüncü parçayı almak
> **tüm Manisa ilçelerini düşürür** — 59 trafo, Akhisar ve Salihli dahil.
> Parça sayısına göre dallanmak gerekir.

**Trafo sayıları:** eğitim tarafında aday 315, bunların **64'ü test setinde
yok** ve maskeye giremez. Testte etkili olan **251**. İki sayıyı payda olarak
karıştırmamak gerekir; `derin_315_odak.py` bu yüzden düzeltildi.

> **Düzeltme.** Bu trafoların medyan gücü 400 kVA'dır, ancak tüm test setinin
> medyanı da 400 kVA'dır; kapasite bakımından ayırt edilebilir değiller.
> "Özel müşteri / müstakil abone trafosu" çıkarımının veride dayanağı yoktur.
> Veri setinde yalnız `id, tanim, guc, tarih, lokasyon` bulunur; "yazlık site",
> "terk edilmiş şantiye", "pasif kuyu trafosu" gibi nitelemeler veriden
> türetilemez.

## 8. 28 Ağustos — üç hak

### Sıfırıncı iş

```powershell
uv run python -m kaggle competitions submissions -c grid-up-datathon
```

Zaman aşımına uğrayan bir betik "hiçbir şey olmadı" demek değildir.

### HAK 1

```powershell
uv run python -m kaggle competitions submit -c grid-up-datathon `
  -f submissions/tuketim_v89_genis_taban.csv `
  -m "v89 v83 + genisletilmis olu maske (251 trafo) + ay bazli optimum taban"
```

Skor gelince:

```powershell
uv run python scripts/olu_kappa_coz.py --prob-score V89_SCORE
```

### HAK 2 — karar ağacı

```
v89 < 1.00      -> sota_v1 gonder (mimariyi olc)
1.00 .. 1.013   -> sota_v1 gonder (kismi kazanc, mimari ne katiyor gor)
> 1.013         -> v85 gonder, 2. sirayi bankala (tahmin 1.010497)
```

### HAK 3

κ veya Gram ile **çöz**, tahmin etme.

**Kritik:** sıfırlamayı harmana karıştırma, harmanın üstüne uygula. Log uzayında
`0.85 * log(v83) + 0.15 * 0` sıfır vermez — v7'de 14.484 sıfırın hepsi bu yüzden
kaybolmuştu.

## 9. Dosya envanteri

| dosya | rol | durum |
|---|---|---|
| `tuketim_v89_genis_taban.csv` | HAK 1 | kapı geçti |
| `tuketim_sota_v1.csv` | HAK 2 adayı | kapı geçti |
| `tuketim_v85_gram_rank2.csv` | HAK 2 sigortası, tahmin 1.010497 | kapı geçti |
| `tuketim_v88_olu_taban.csv` | yedek, sabit 1 kWh | kapı geçti |
| `tuketim_v87_olu_izole.csv` | yedek, sert sıfır | kapı geçti |
| `tuketim_v83_sicak_optimum.csv` | çıpa, 1.01318 | gönderildi |

Araçlar: `kapi_denetim.py`, `olu_kappa_coz.py`, `gram_ansambl.py`,
`olustur_v89_genis_taban.py`.

## 10. Dürüst belirsizlik

Güven: **~%80–87** ile 1. sıra. Çözülemeyen iki nokta:

1. Yüksek kapsamlı bir yaz gözlemi yok. Temmuz'daki düşüşün sulamadan mı panel
   boşluğundan mı geldiği veriden ayrılamıyor.
2. Testte ölü/canlı panel yoğunluk oranı 1.00; eğitimde en yüksek 0.89 idi.
   Bu sıçramanın iki okuması var ve hangisinin doğru olduğu bilinmiyor.

Kaldıraç çapraz doğrulamaya görünmez olduğu için güven tamamen CV dışı
ölçümlere dayanıyor. Hikâye tutarlı; kanıt değil.

**Aşağı yön sınırlı:** v83 tabloda duruyor, Kaggle en iyi skoru tutar. v89 kötü
gelirse sıralama düşmez, bir hak öğrenmeye gitmiş olur.
