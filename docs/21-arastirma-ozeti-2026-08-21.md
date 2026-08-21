# Altı paralel araştırmanın özeti (2026-08-21)

Altı bağımsız derin araştırma koşturuldu: RMSLE yarışma teknikleri, trafo yük
tahmini literatürü, soğuk-başlangıç yöntemleri, GBDT/harman state-of-the-art,
RMSLE kuramı + sıfır ele alma, ve soğuk-başlangıç yarışma kanıtları.

Aşağıdakiler **kaynağıyla birlikte** kaydedildi. Ölçülmüş olanla iddia edilmiş
olan ayrı işaretlendi. Bizim verimizde doğrulananlar ayrıca belirtildi.

---

## 1. Kesinleşen: amaç fonksiyonumuz doğru, değiştirilmemeli

RMSLE tanımı gereği `z = log1p(y)` uzayında düz kareli hatadır. Gneiting
(2011, arXiv:0912.0902, Teorem 2.6 — Osband bürüme ilkesi) uyarınca optimal
nokta tahmini `expm1(E[log1p(y)|x])`'tir. Yani `log1p` hedefe + L2 amaç
fonksiyonu **metriğin birebir kendisidir.**

Ölçüldü (200k satır, %56 sıfır, diğer her şey sabit):

| RMSLE | tabana göre | amaç |
|---|---|---|
| 0,98361 | %0,00 | **L2, log1p(y) hedef — TABAN** |
| 1,10645 | +%12,49 | tweedie(1.9), ham y |
| 1,11388 | +%13,24 | gamma, ham y |
| 1,14958 | +%16,87 | tweedie(1.1), ham y |
| 1,16890 | +%18,84 | poisson, ham y |
| 1,08980 | +%10,79 | quantile(0.5) |
| 1,01037 | +%2,72 | sıfır satırlar 2× ağırlıklı |

**Mekanizma:** `gamma`/`tweedie`/`poisson` LightGBM'de **log-bağ (link)**
kullanır, log-hedef değil. Bağ koşullu **ortalamaya** uygulanır; dönüşüm
etikete. Jensen gereği `expm1(E[log1p Y|x]) < E[Y|x]`, yani bu amaçlar RMSLE
optimumunu sistematik olarak aşar.

**M5 kanıtı buraya taşınmaz:** M5'in metriği WRMSSE, yani **ham ölçekte**
kareli hata — orada Tweedie doğru eşleşmedir. 4. sıranın kodu `objective:
tweedie, tweedie_variance_power: 1.1` ve **hiçbir yerde log dönüşümü yok.**

**Yarışma kanıtı:** RMSLE metrikli 8 üst çözümün 8'i de log-dönüşüm + kareli
hata kullanmış. Hiçbiri tweedie/poisson/gamma kullanmamış. (Mercari 1., ASHRAE
1. ve 2., Recruit, Favorita 5., Bimbo, NYC Taxi, Store Sales.)

## 2. Kesinleşen: sihirli çarpan ve Duan düzeltmesi ZARARLI

Optimal tahminde `E[ẑ* − z] = 0` olduğundan, herhangi bir sabit `c` kaydırması
için **kapalı form**: `MSLE(c) = MSLE* + c²`.

Ölçüldü: `c = +0,05` → 0,98477 (kuram 0,98470). Gerçek Jensen düzeltmesi
(`c = σ̂²/2`) uygulanınca **+%7,5** (0,98343 → 1,05717).

Rossmann'ın meşhur 0,985 çarpanı **RMSPE için** türetilmiştir, RMSLE için
değil. Kazananın kendi belgesi: *"0,995 hem doğrulama hem public test için
optimaldi. Ama forumdaki 0,985 matematiksel tahminini uygulamalıydım."*
Yani ampirik ayar public LB ile hemfikirdi ve **ikisi de yanlıştı.**

## 3. Kesinleşen: harman log uzayında yapılmalı

Krogh & Vedelsby ayrışması (NeurIPS 1994) bir **özdeşliktir**, sınır değil:
`(q̄ − y)² = ortalama üye hatası − ortalama ayrışma`. RMSLE `log1p` uzayında
kareli hata olduğundan bu birebir geçerlidir — **ama yalnızca birleştirici
aritmetik ortalama ise.** Wood ve ark. (JMLR 2023): *"farklı bir birleştirici
kullanırsak sonuç artık geçerli değildir."*

Yani doğru form `expm1(mean(log1p(tahmin)))`. Ham uzayda ortalama almanın
böyle bir garantisi yok.

**Koşul:** geometrik ortalamanın aritmetiği geçmesi, log-uzayı artığının
merkezlenmiş olmasına bağlı. Ölçüldü: log uzayında −0,25 yanlılık varken
aritmetik (0,26273) geometriği (0,30747) **geçiyor**. Önce yanlılığı gider.

Bizim hattımız zaten log uzayında ortalıyor — doğru.

## 4. Bizim verimizde ölçüldü: kapasite ofseti

`log1p(y) − log1p(guc)` hedefiyle eğitip tahmine geri eklemek. Log uzayında
satır-başı sabit kaydırma olduğu için L2 optimumu değişmez — metrik açısından
birebir aynı problem, ama ağaçların ölçeği öğrenmesi gerekmez.

Varlık-dışarıda-bırakmalı ölçüm (%28,8 trafo tamamen çıkarıldı, 356.218 soğuk
satır):

| tahminci | soğuk RMSLE |
|---|---|
| global ortalama | 2,0331 |
| ilçe ortalaması | 1,9043 |
| `guc` kovası (20 qcut) | 1,7696 |
| **OFSET, global — TEK parametre** | **1,7746** |
| OFSET × ilçe | 1,7682 |
| **OFSET × ilçe × ay** | **1,7574** |
| OFSET × `guc` kovası | 1,7827 |
| OFSET × ilçe × kova | 1,8169 |

Esneklik (elastisite) ölçüldü: satır düzeyinde **1,0630**, trafo düzeyinde
**1,0727**, kesişim ≈ 0. Yani ofset formu haklı.

**Ofsetten sonra `guc` kovası eklemek ZARARLI** (1,7682 → 1,8169) — ofset
kapasiteyi zaten soğuruyor, kesişim yalnızca seyrek hücre gürültüsü.

## 5. Bizim verimizde ölçüldü: soğuk trafolar daha BÜYÜK

| | medyan `guc` | 10. yüzdelik |
|---|---|---|
| eğitim / sıcak test | 400 kVA | 100 |
| **soğuk test** | **630 kVA** | 160 |

Eğitimden çıkan hiçbir grup ortalaması soğuk nüfus için yansız değil.
Ekstrapolasyon riski yok (soğuk trafoların hiçbiri eğitim `guc` aralığının
dışında değil) ama tabakalama/ağırlıklandırma gerekiyor.

## 6. Uygulanmamış tek teknik: soğuk-başlangıç simülasyonu

DropoutNet (Volkovs ve ark., NeurIPS 2017): *"neural network models can be
explicitly trained for cold start through dropout."* Mekanizma — eğitim
sırasında rastgele seçilen varlıkların geçmişten türemiş öznitelikleri
sıfırlanır, böylece model servis anındaki girdi dağılımını eğitimde görür.
**Tek model hem sıcak hem soğuk rejimi öğrenir; ayrı model gerekmez.**

Bizim eğitim bloklarımızdaki soğuk satırlar doğal olarak oluşuyor ama yanlı
bir altküme: 2025 ortasında devreye alınmış trafolar. Test'teki soğuk dilim
başka bir kesim (bkz. §5).

**Ayrı model yönlendirmesi için hiçbir yarışma kanıtı bulunamadı.** Favorita
2.'nin özel yeni-ürün sezgiseli skoru <0,001 oynatmış; 3.'nün soğuk-satır
hilesi public LB'de 0,501→0,499 yapıp **private'ta çökmüş.**

## 7. Bizim verimizde ölçüldü: CDD taban sıcaklığı yanlış

MGM (Eurostat uyumlu) resmî Türkiye tanımı: `CDD = (Tort − 22)⁺`,
`HDD = (18 − Tort)⁺ · 1[Tort ≤ 15]`. Bizim hava tablomuz **taban 18** kullanıyor.

Tüketim sapmasıyla korelasyon (trafo seviyesi arındırılmış, 1,23M satır):

| değişken | r |
|---|---|
| CDD taban 18 (mevcut) | +0,1804 |
| **CDD taban 22** | **+0,1966** |
| CDD taban 24 | +0,1968 |
| HDD taban 18 | **−0,0038** |
| HDD MGM formu | +0,0030 |
| sıcaklık ortalaması | +0,1001 |

**HDD sıfır korelasyonlu.** Bu bölgede ısıtma elektrikle değil; ısıtma-derece-
günü bu problemde bilgi taşımıyor.

## 8. Bizim verimizde ölçüldü: `tanim` numarası yapı taşıyor

| ID bloğu | trafo | medyan kWh/kVA/gün | test'te soğuk oranı |
|---|---|---|---|
| 8 haneli | 3.626 | **3,12** | %20,8 |
| 9 haneli (`700…`) | 1.413 | **1,95** | **%44,0** |

Aynı anma gücü, **%60 farklı yoğunluk.** Trafo düzeyinde 5-katmanlı CV
(hedef: ortalama günlük log tüketim): `log(kVA)` 0,4184 → `+ilçe` 0,4372 →
**`+ID uzunluğu` 0,4606**.

TEDAŞ *Numaralama İşleri Teknik Şartnamesi* (2009) neden böyle bir yapı
olduğunu açıklıyor: varlık kayıtları `TM / DM / İM / KÖK / TR / TRP` ayırır,
mülkiyeti kaydeder, özel müşteri trafolarına **"ö"** ekler. Ama 8 vs 9 haneli
bloğun anlamını **yayımlanmış hiçbir belge doğrulamıyor** — etki ölçüldü,
mekanizma doğrulanmadı.

## 9. Bizim verimizde ölçüldü: sulama mevsimselliği devasa ve ilçeye özel

Trafo başına aylık endeks (ay ortalaması ÷ kendi yıllık ortalaması), ilçe
medyanı:

| ilçe | Oca | Nis | Tem | Ara | oynama |
|---|---|---|---|---|---|
| **GÖRDES** | 0,55 | 0,67 | **1,86** | 0,47 | **3,4×** |
| **SARIGÖL** | 1,06 | 0,54 | **1,95** | 0,87 | 3,6× |
| ÖDEMİŞ | 1,05 | 0,65 | 1,59 | 1,02 | 2,4× |
| ÇEŞME | — | 0,76 | 1,56 | — | turizm tepeli |
| FOÇA / DİKİLİ | — | — | <1,0 | >1,0 | **kış tepeli** |
| SOMA / NARLIDERE | 1,02 | 0,91 | 1,14 | 1,10 | düz |

En az **üç ayrı ilçe arketipi**: sulama-tepeli, turizm-tepeli, kış-tepeli.
Nisan–Mayıs çukuru (Sarıgöl 0,54) sulama öncesi döneme denk geliyor; saf bir
CDD modeli bunu kaçırır.

Fiziksel mekanizma yayımlanmış: Gediz havzasında yeraltı suyu pompalama
0,4068 kWh/m³, sulama ihtiyacı 2.500 m³/ha (pamuk) – 8.928 m³/ha (sofralık
üzüm), yani mevsim başına ≈1.000–3.600 kWh/ha, Haz–Eyl'de yoğunlaşıyor.

## 10. Yayımlanmış çapa: kWh/kVA/gün önsel değeri

| kaynak | değer |
|---|---|
| TEDAŞ 2024 Sektör Raporu (Türkiye geneli) | ≈3,34 kWh/kVA/gün |
| GDZ 2026–2030 duyurusu (38.199 trafo, 18.355 GWh) | 1.316 kWh/trafo/gün |
| **bizim train setimiz** | medyan **2,76**; p25 1,37; p75 4,86 |
| Ecuador EEASA, 16.696 trafo | yüklenme %31,79 |
| IEEE/GE, 55 ABD şirketi | yük faktörü %26,6 |

Üç bağımsız çapa da uyumlu. GDZ'nin 1.316 kWh/gün rakamı, 122 günlük
tahminimizin toplam seviyesi için bedava bir akıl sağlığı kontrolü.

## 11. Reddedilenler — ve nedenleri

| yol | neden reddedildi |
|---|---|
| Hiyerarşik Bayes / GPBoost / MERF | Soğuk varlıkta rastgele etki önselin ortalamasına çöker; tahmin `E[y\|öznitelik]`'e indirgenir — LightGBM'in zaten yaptığı şey |
| Zaman serisi temel modelleri (Chronos, TimesFM, Moirai) | Bağlam penceresi mimarisi; boş geçmişte çalışmaz. Ölçüldü: L=0'da ortalama sıra 5,9–7,8, LightGBM 2,2 |
| Croston / SBA / TSB | Ham ölçekte ortalama tahmin eder (yanlış fonksiyonel); tek değişkenli; sıfır-tespitte AUC 0,500 |
| Hiyerarşik uzlaştırma (MinT) | Doğrusal bir haritadır; olmayan varlıklar-arası yayılımı **üretemez**. Bizim soğuk hatamızın %81'i varlıklar-arası dağılım |
| Ayrı soğuk model | Hiçbir yarışma kanıtı yok; literatür tersini söylüyor (tek model + maskeleme) |
| Büyük Optuna bütçesi | LightGBM 13 regresör içinde **sonuncu**: 50 denemenin medyan kazancı %0,0 |
| `reg:squaredlogerror` | Hessian negatife düşüyor; ölçüldü %0,8–1,0 daha kötü |
| Tail için yeniden ağırlıklandırma | Global metrikte kütle doğruluğunu tail için takas eder; ölçüldü +%2,7 – +%12 |

## 12. Sırada ne var

Bu belgedeki hiçbir şey bizim CV'mizde henüz kanıtlanmadı. Karar eşiğimiz
ölçüldü: **tohum yayılması 0,00998, eşik 0,01995.** Bu eşiğin altındaki her
fark gürültü.

Ölçülecek adaylar, beklenen değer sırasıyla:

1. Soğuk-başlangıç maskeleme (§6) — %10 / %22 / %35
2. Kapasite ofseti (§4)
3. LightGBM parametreleri — `min_child_samples` 40 → 200–1000, `num_leaves` sınırı
4. Aile-ötesi harman (CatBoost + XGBoost + LightGBM, log uzayında)
5. CDD tabanı 22/24, HDD'yi at (§7)
6. Sıfır-süreci öznitelikleri + geçen yıl aynı pencerede sıfır oranı
7. Segment bazlı log-uzayı ofseti, ampirik-Bayes küçültmeli (n = **trafo**, satır değil)
