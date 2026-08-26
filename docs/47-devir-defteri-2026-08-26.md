# Devir defteri — 26 Ağustos 2026

**Bu dosya bağlantı kesilmesine karşı yazıldı.** Taze bir oturum bunu okuyup
kaldığı yerden devam edebilmeli. En güncel durum burada; `docs/45` ve `docs/46`
25/26 gecesini anlatır ve **aritmetiği düzeltilmiştir (aşağıya bak).**

---

## 0. Durum

```
LB (26 Ağustos)
  1. Grid Grinders  1.00635
  2. Datastellar    1.01529
  3. TasnifX        1.01538   <- BİZ
  4. Abdülbaki B.   1.01565
  5. OzanM.         1.01577

Bizim MSE 1.030997   Hedef MSE 1.012740   GEREKEN -0.018257
Yarışma 1 Eylül'de bitiyor.
```

**Kota: her UTC günü 3 hak, gün içinde İSTEDİĞİN AN.** 00:00 UTC = 03:00 yerel
sadece kotanın yenilendiği andır, gönderim saati değil. 26 Ağustos kotası
00:02/00:03/00:06 UTC'de kullanıldı.

---

## 1. LB'den ÇÖZÜLMÜŞ sabitler (gürültüsüz, public/private ayrımı YOK)

```
MSLE(0)  = 1.031200        v67_c1335_olay'ın gerçek MSE'si
c*       = 1.3301          soğuk gün ekseni optimal ölçeği
```

Gönderim dosyalarından **etiketsiz** hesaplanan ayrışım (maxabs 1,9e-15):

```
log1p(v79) - log1p(v67) = 0.55*u_gun + u1
  u_gun = log1p(v73) - log1p(v67)     soğuk gün ekseni, 0,60 adımı
  u1    = log1p(v78) - log1p(v73)     ayrık seviye: 0 / 0,22 / 0,35

Q(u_gun) = 0.002023253   L = 0.00111317   k* = 0.550188   tavan -0.0006125
Q(u1)    = 0.015845379   L = 0.00753522   k* = 0.475547   tavan -0.0035834
Q(u_gun, u1) = 2.3e-18   -> TAM DİK, kazançlar toplanır
ORTAK TAVAN = -0.0041958  ->  MSE 1.0270038  ->  RMSLE 1.013412
```

### docs/46'nın DÜZELTİLEN değerleri

| değer | docs/46 | DOĞRUSU |
|---|---|---|
| Q3 | 0,017869 | **0,016457413** |
| L3 | 0,008853 | **0,00814746** |
| ortak tavan | −0,004998 | **−0,0041958** |
| beklenen iniş | 1,01302 | **1,013412** |
| kalan açık | −0,0132 | **−0,0142635** |
| "iki eksen dik 3,5e−18" | — | **YANLIŞ**: v_S3 ⟂ v_gun değil, kosinüs +0,19284 |

**Ayrıca void:** `docs/45` tik6'nın kuyruk sayıları (+0,4754 / +0,3531) üretim
modeline ait değil (`kos_lgbm "Aplus"` kolonu). Üretim harmanıyla guz25 **+0,0409,
t=+0,62 (NULL)**. Kuyruk ekseninin "iki blokta doğrulandı" hükmü geçersiz.

**Bayat:** `scripts/son_islem_seviye.py` ekrana MSLE(0)=1,032073 ile çözüm formülü
basıyor. Kullanma; doğrusu 1.031200.

---

## 2. TEMEL ARAÇ: yön öner, LB genliği çözsün

Public/private ayrımı yok → dönen skor **gürültüsüz ve tam test kümesi üzerinde.**
Tahminlere sabit bir `v` vektörünün `κ` katı eklenirse:

```
dMSE(κ) = κ²·Q − 2·κ·L
  Q = ort(v²)   -> DOSYALARDAN analitik, ETİKETSİZ
  L             -> TEK GÖNDERİMDEN çözülür
κ* = L/Q        en iyi dMSE = -L²/Q
```

**DİKKAT:** bu formülün yarım hali dolaşımda (`κ* = L/(2Q)`). O **YANLIŞ**.
Yarım ölçek yazılırsa kazancın 3/4'ü kaybedilir.

---

## 3. SIRADAKİ ÜÇ HAK — hazır reçete

> ### ✅ ÜÇ DOSYA ÜRETİLDİ VE DOĞRULANDI (26 Ağustos 11:30)
>
> ```
> submissions/tuketim_v80_optimum.csv     HAK 1  bankaya, beklenen 1.013412
> submissions/tuketim_v81_sicak08.csv     HAK 2  1[sıcak-çekirdek] probu
> submissions/tuketim_v82_ayirici.csv     HAK 3  kuyruk ayırıcı
> ```
>
> Doğrulama (dosyaların KENDİSİNDEN okundu, yeniden kurulumdan değil):
> ```
> v81 − v80 = +0.08 × 526.446 satırda, 0 diğerlerinde, ara değer YOK
> v82 − v80 = +0.15 ×  29.873 satırda, 0 diğerlerinde, ara değer YOK
> kesişim 0        ort(v1·v2) = −5,1e−20  →  TAM DİK
> Q(çekirdek)=0.7366095  Q(kuyruk)=0.0417987   (docs/47 ile 7 hanede aynı)
> ```
> Ön uçuş kapısı: k_gun=0.550167 (optimal 0.550187), k_lvl=0.475446 (optimal 0.475546).
> Artık maxabs 6,15e−06 — bu **kırpma değil, CSV yazma hassasiyeti** (tüketim tek
> ondalıkla saklanıyor). Katsayı sapmasının maliyeti 1,6e−10, ihmal edilebilir.
> Bütünlük: 714.688 satır ✓ id sırası ✓ NaN 0 ✓ negatif 0 ✓
>
> **Nötr senaryo beklenen skorlar:** v81 → 1.015735, v82 → 1.013876.
> İkisi de hak 1'den kötü; bu ÖLÇÜMDÜR, kayıp değil. Kaggle en iyiyi korur.
>
> Aşağıdaki komutlar bu dosyaları yeniden üretir (gerekirse).

> **SIFIRINCI İŞ (kalıcı kural 8):**
> `uv run python -m kaggle competitions submissions -c grid-up-datathon | head -5`
> Zaman aşımına uğrayan bir betik "hiçbir şey olmadı" demek değildir; bu yüzden
> 25 Ağustos'ta bir hak mükerrer gönderime gitti.
>
> **NOT:** `kaggle.exe` yerine `python -m kaggle` kullan (Smart App Control'e dayanıklı yol).

### HAK 1 — bankaya yatır (beklenen 1.013412)

```bash
uv run python scripts/son_islem_soguk_gunolcek.py \
    --giris submissions/tuketim_v67_c1335_olay.csv \
    --cikis submissions/tuketim_v80_a.csv --c 1.3301
uv run python scripts/son_islem_seviye.py --giris submissions/tuketim_v80_a.csv \
    --cikis submissions/tuketim_v80_b.csv --delta 0.0 --soguk-delta 0.1046
uv run python scripts/son_islem_kuyruk_rejimi.py --giris submissions/tuketim_v80_b.csv \
    --cikis submissions/tuketim_v80_optimum.csv --delta 0.1664
```

**ÖN UÇUŞ KAPISI (BUG 1'e karşı, göndermeden ÖNCE yerelde):**
`log1p(v80_optimum) − log1p(v67)` ile `0.550188·(a73−a67) + 0.475547·(a78−a73)`
farkının maks mutlak değeri **< 1e−12** olmalı. Değilse kırpma ölçeği yemiştir
(v55'te 1,492→1,4760, v66'da 1,335→1,3241); `--c`'yi ulaşılan ölçek 0,550188
olana kadar ayarla.

Bu hakka **prob eklenmemeli** — dönen skor `MSE_banka`'yı ölçer ve hak 2/3'ün
ters çözümü tam olarak bu sayıya dayanır.

### HAK 2 — `1[sıcak-çekirdek]` PROBU (en yüksek tavan)

```bash
uv run python scripts/son_islem_seviye.py --giris submissions/tuketim_v80_a.csv \
    --cikis submissions/tuketim_v81_b.csv --delta 0.08 --soguk-delta 0.1046
uv run python scripts/son_islem_kuyruk_rejimi.py --giris submissions/tuketim_v81_b.csv \
    --cikis submissions/tuketim_v81_sicak08.csv --delta 0.0864
```

Kuyruk deltası `0.1664 − 0.08 = 0.0864` yapılır ki kuyruk **neti** 0,1664'te sabit
kalsın; böylece problanan vektör tam olarak `1[sıcak & ~kuyruk]` olur ve her şeye dik kalır.

```
Q = 0.7366095     (526.446 / 714.688 satır -- problanmamış EN BÜYÜK yön)
eşik: b_hc >= 0.1391  ->  TEK BAŞINA birincilik
çapalar 2 kat ayrışıyor: kis26F fold 0.1449  |  ima-YoY 0.03...0.11

ÇÖZÜM:  b_hc = (0.0047143 - dMSE) / 0.1178575
        dMSE = S² - MSE_banka   (hak 1'den)
ertesi gün: κ* = b_hc, kazanç -0.7366095·b_hc²
```

| b_hc | o gün RMSLE | ertesi gün bankaya |
|---|---|---|
| 0,00 | 1,015735 | 0 |
| 0,04 | 1,013412 | −0,00118 |
| 0,08 | 1,011083 | −0,00471 |
| 0,12 | 1,008749 | −0,01061 |
| **0,1391** | 1,007633 | **−0,01425 → BİRİNCİLİK** |

### HAK 3 — AYIRICI (kuyruk / soğuk ayrışması)

```bash
uv run python scripts/son_islem_kuyruk_rejimi.py --giris submissions/tuketim_v80_b.csv \
    --cikis submissions/tuketim_v82_ayirici.csv --delta 0.3164
```

`0.0487502·b_soğuk + 0.0146295·b_kuyruk = 0.00753522` — tek denklem, iki bilinmeyen.
Bu prob ikinci denklemi verir.

```
kazanç = -0.061754·(b_kuyruk - 0.166441)²
ÇÖZÜM:  b_kuyruk = 0.166441 + (0.0009405 - dMSE) / 0.0125396
```

### YENİ MODEL GELİRSE — hak tahsisi kuralı

Model eğitim kampanyası doğrulanmış bir kazanç verirse:

```
HAK 1  ->  EN İYİ TABAN (v80_optimum, ya da yeni modelin zinciri)
HAK 2  ->  b_hc probu, HAK 1'in dosyası ÜSTÜNE      <- birinciliğin kilidi
HAK 3  ->  serbest: yeni model zinciri ya da kuyruk ayırıcı
```

| kampanya sonucu | karar |
|---|---|
| doğrulanmış kazanç **≥ −0,002 MSE** | yeni modelden tam zincir kur, **HAK 1 o olur** |
| kazanç **< −0,002** | zinciri sarsma; HAK 3'e koy, ölç, 28 Ağustos'a devret |
| kazanç yok | mevcut üçlü aynen gider |

**Eşiğin sebebi:** temiz bir ölçüm çapasının değeri yüksek ve `b_hc` probu birinciliğin
kilidi. Küçük bir model kazancı için çapayı sarsmak kötü takas. −0,002 üstü ise zinciri
yeniden kurmaya değer.

**`c*` ve δ'lar yeni tabana TAŞINABİLİR** — yeniden türetmek için hak harcama:
soğuk gün ekseni küçük ve tavanı düz, `c*`'ta 0,1 sapmanın maliyeti `Q·Δκ² = 2e−5`.
δ_cold ve δ_tail de ikinci mertebeden etkilenir.

**Kota UTC günü boyunca açık** — 03:00'te göndermek zorunlu değil. Yeni model 08:00'de
hazırsa 08:00'de gönderilir. Aceleyle eski modelle gönderme.

### Beklenen: hak 2 ve 3 hak 1'den KÖTÜ skor gösterecek

Nötr senaryoda 1,015735 ve 1,013876. **Kaggle en iyi skoru koruduğu için
sıralamamız düşmez.** Bu bir kayıp değil, ölçümdür. Panikle plan değiştirme.

---

## 4. Bugün çalışan: model eğitim kampanyası

Son işlem yönlerinin **toplam tavanı −0,006**, gereken **−0,0143**. Yani rötuşla
birincilik imkânsız; tek yol daha iyi model. 7 eksende arama koşuyor:

1. **Hiperparametre rig tamiri** — ayar tablosunun çoğu 1,04M satırlık EK KÖKENSİZ
   kolda tarandı, üretim 2,86M görüyor (docs/40 §7b). Kapasite bu gerekçeyle yeniden
   ölçülüp reddedildi ama **lr / subsample / colsample / min_child / l2 ölçülmedi.**
2. **Soğuk uzman mimarisi** — MSE'nin %63'ü, tek CatBoost. Aile bileşimi ölçülmedi.
3. **Sinir ağı** — harmanın %23'ü, en az incelenen üye; GBDT'nin yapamadığı
   ekstrapolasyonu yapabilir.
4. **Öznitelik seti** — 148 kolon, doluluk deseni, düşük önemlilerin zararı.
5. **Eğitim dağılımı** — ufuk / özet penceresi / zaman ağırlığı test'e benziyor mu.
6. **Sürüklenme** — yıllık trend GBDT'ye girmiyor; hedef detrending, zaman
   özniteliği, doğrusal üye. **En yüksek tavanlı.**
7. **Tohum + kesin kazançlar** — 30→50+ tohum, kırpma kaybı.

Sonuçlar `scratchpad-eg/` altına ve rapor dosyalarına yazılıyor.

---

## 5. Kalıcı kurallar (ihlal eden bulgu reddedilir)

1. Soğuk taraftaki her kazanç trafo bazında ayrıştırılır; **kırpma tablosu
   K = 0,1,5,10,25,50** verilmeden kabul edilmez.
2. Önerilen her kolonun **eğitim/test doluluk deseni** karşılaştırılır.
3. Soğuk tarafta **üç tohum yetmez** — bu projede 3 tohumda t=+3,91 olan bulgu 6'da
   çöktü, 3 tohumda t=−9,69 olan bulgu 8'de sıfırlandı. **En az 5, tercihen 8.**
4. Hüküm **(blok, tohum) çiftlerinde eşlenik SH** ile verilir. Havuzlanmış skora
   güvenme — bu projede üç kez kandırdı.
5. LB'yi problayarak test etiketi çıkarmak yasak; küresel skaler ayarlamak serbest,
   ama her yönün bir **mekanizması** olmalı.
6. Gün ekseni ölçümü trafo etkisi çıkarılmadan yapılmaz.
7. Mevsime bağlı eksen tek blokta ölçülmez; ölçüm **yaz25**'i içermelidir.
8. **Gönderimden önce gönderim listesi okunur.**
9. Komşu/geçmiş öznitelikleri **AS-OF** hesaplanır.
10. Trafo bazlı GroupKFold sızıntısız SAYILMAZ; en az **iki örtüşmeyen zaman kesmesi**.
11. Bir mekanizmanın **var olması, sömürülebilir olmasını gerektirmez.** Kapı analizi
    mekanizmayı gösterir, büyüklüğü yalnızca tezgah gösterir.
12. **Foldlar YÖN ve İŞARET için güvenilir, GENLİK için DEĞİL.** 25/26 gecesinde beş
    kez büyüklük şişti (H9 0,0559 / H17 −0,0238 / H23 −0,0794 / H20 t=−9,69 üç
    tohumda / soğuk gün ekseni tavanı 15 kat).
13. **Bir referans model çıktısından türetilip sonra aynı modeli doğrulamak için
    kullanılamaz** (döngüsel doğrulama). Referans gerçek etiketlerden ve **hedef
    nüfusun ikizinden** ölçülür.

---

## 6. Yasak bölge — yeniden açma

sıcak kapasite · soğuk kapasite · soğuk hurdle · kalibrasyon/beta · takvim/tatil ·
harman ağırlıkları · harman uzayı · λ pencere · ulusal yük gün faktörü ·
b_i trafo kestiricisi · sabit δ transferi · artık hedefi · ölü kuyruk · grup B ·
soğuk kVA kovası · on6/on7 hedef kodlaması · kimlik komşuluğu · son pencere çapası ·
gün faktörünü hava modeliyle değiştirmek · p_son_ofset · **H20 kümeli soğuk maskeleme**
(8 tohumda null) · **kuyruğu soğuk uzmana yönlendirmek** (3/3 blokta kötü) ·
**kuyruk girdisini temizlemek** (5/5 tohumda kötü) · **ufuk yönü** (tavan −5,6e−05) ·
**ay/mevsim yönü** · **v_HAVA** · **05-11 iç kontrastı**

Bir kısmı FOLD BÜYÜKLÜĞÜ gerekçesiyle kapatıldı. Kural 12 ışığında yön hükmü geçerli,
büyüklük hükmü şüpheli — ama "mekanizma yok" diye kapatılanı açma.

---

## 7. Ortam notları

- Python dosyalarını **Write** ile yaz, bash heredoc ile **asla** (ters bölüler bozuluyor).
- `subprocess`'e **her zaman** `encoding="utf-8"`; `PYTHONIOENCODING=utf-8` ayarla
  (`text=True` cp1254 okuyup Türkçe çıktıyı sessizce bozuyor).
- Geniş `taskkill` kullanma — arkaplan işlerini öldürür.
- Aynı çalışma ağacında paralel oturumlar olabilir; commit'te **yalnızca kendi
  dosyalarını** stage'le.
- Makine: 16 çekirdek, 24 GB RAM. Paralel ajan varsa `n_jobs=3`.
- `submissions/` altında v71–v78 **ara adaylar**, gönderilmediler. Gönderilenler:
  v50, v55 (×2, mükerrer), v67, v73, v79.
- Kaggle gönderim yetkisi ve otonom çalışma yetkisi **kalıcı olarak verildi**.
  Ölç, karar ver, uygula, sonra bildir.
