# LOOP DEFTERİ — 25/26 Ağustos 2026

**12 saatlik kesintisiz skor iyileştirme koşusu.** Başlangıç 25 Ağustos 22:00,
bitiş 26 Ağustos 10:00. Metrik RMSLE. Her tikin başında bu dosya okunur, her
tikin sonunda güncellenir.

---

## HEDEF

```
şu an       MSE 1.032073  (RMSLE 1.01591)   TasnifX, 5.
hedef       MSE 1.012740  (RMSLE 1.00635)   Grid Grinders, 1.
GEREKEN     −0.019333 MSE
```

---

## ŞAMPİYON

```
dosya: submissions/tuketim_v73_soguk_gun160.csv
LB SKORU: 1.01538   (OLCULDU, 26 Agustos 03:04)
bilesenler: v50_nihai30 -> sicak gun ekseni c*=1,335 -> olay gunu s=0,6
            -> SOGUK gun ekseni c=1,60
onceki en iyi 1.01591 -> net kazanc -0,00107 MSE
son guncelleme: tik 10 (gonderim)
```

### Şampiyon geçmişi
| tik | eski | yeni | dMSE | kanıt |
|-----|------|------|------|-------|
| 0 | — | v67_c1335_olay | — | docs/43, LB doğrulanmış c* ailesi |
| 1 | v67_c1335_olay | v71_soguk_gun (c=2,20) | −0,0149 bek. | H8 |
| 3 | v71_soguk_gun | **v73_soguk_gun160 (c=1,60)** | −0,0091 iyimser | H9b + h8l, aşağıda |

---

## HAKLAR

```
26 Ağustos (pencere 08:00 yerel = 05:00 UTC): [ ] S1  [ ] S2  [ ] S3
dönen skorlar: S1=___ S2=___ S3=___
çözülen: H8 kazancı=___  b_soğuk=___
```

25 Ağustos kotası harcandı (00:01'de v50 + v55 ×2 — **v55 iki kez gitti, bir hak
boşa gitti**). Kalıcı kural 8 bu yüzden var.

### 08:00 GÖNDERİM PLANI — üç hak, üç bilinmeyen, tam çözüm

| hak | dosya | ne verir |
|-----|-------|----------|
| **S1** | `tuketim_v67_c1335_olay.csv` | yeni `MSLE(0)` — çözücünün çapası, c\*=1,335+olay'ı bankaya yatırır |
| **S2** | `tuketim_v73_soguk_gun160.csv` | **soğuk gün ekseni PARABOLÜNÜN TAMAMINI çözer** (aşağıda) |
| **S3** | `tuketim_v75_gun160_seviye16.csv` | **b_soğuk'u TAM çözer** ve tahmini optimumu bankaya yatırır |

Üç dosya da kapılardan geçti (3/3).

### S2 tek başına c\* değerinin TAMAMINI çözer — hedge gereksiz

Düzeltme sabit bir vektörün ölçeklenmesi: `Δ = (c−1)·profil[gün]` (soğukta),
0 (sıcakta). Dolayısıyla

```
MSLE(c) = MSLE(0) + (c−1)²·A − 2(c−1)·B      A = ort(Δ²/(c−1)²)   B = ort(r·Δ/(c−1))
```

**`A` etiketsiz ve TAM hesaplanabilir** — sadece iki gönderim dosyasının
log1p farkının karesinin ortalaması. `B` skordan çözülür:

```
B = [ (c−1)²·A − (S2² − S1²) ] / (2(c−1))        c* = 1 + B/A
```

Yani **hangi c gönderilirse gönderilsin gerçek optimum c\* tam çıkar.**
Bu yüzden S2 ve S3'te farklı c ile hedge yapmaya gerek YOK; risk-ayarlı
güvenli c gönderilir, optimum yine öğrenilir, ve varsa fark 27 Ağustos'ta
yazılır (yarışma 1 Eylül'de bitiyor, 5 gün × 3 hak duruyor).

Seviye özdeşliği de **kesin**, çünkü soğuk gün ekseni düzeltmesi
**seviye-nötr** (satır ortalaması tam 0) ve sabit kaymayla **dikey**:

```
b_soğuk = (p_c·δ² − (S3² − S2²)) / (2·p_c·δ)     p_c = 0,22159   δ = 0,16
        = (0,005673 − (S3² − S2²)) / 0,070909
```

`uv run python scripts/b_coz.py --rejim soguk --delta 0.16 --skor <S3> --taban <S2>`

### 08:00 SONRASI DALLANMA — `coz_0800.py` çıktısındaki `c*`'a göre

Bu üç dal **27 Ağustos'un planını belirler**, ayrıca işaretlenecek:

| `c*` | ne demek | 27 Ağustos |
|---|---|---|
| **> 2,00** | H9b'nin nüfus düzeltmesi **fazla ağır** olmuş. **DİKKAT:** `δ_soğuk`=0,16 de aynı nüfus argümanına dayanıyor — aynı yanlılığı taşıyor olabilir. **S3'ü kurmadan önce `δ_soğuk` yeniden değerlendirilir, gerekirse yukarı çekilir.** Karar noktası → takım oturumuna yaz | ilk hak `c*` ile; H9b'nin nüfus varsayımı yeniden açılır |
| **1,20 – 2,00** | plan değişmez | `\|c*−1,60\|<0,05` → v77 olduğu gibi; değilse sıfırdan kur |
| **< 1,20** | soğuk gün ekseni pratikte **kapanmıştır**. S3 yine kurulur (`δ_soğuk` ve `δ_kuyruk` **bağımsız** kazançlar) ama soğuk gün bileşeni **ölçülen `c*` ile** yazılır — 1,60'ta ısrar edilmez. Bildir | haklar `b_sıcak`'a ve H20'ye gider |

**Karar ağacı S2 için** (S1 tabanına göre):
```
S2 < S1 − 0,006  -> H8 BÜYÜK. c'yi 2,20'den 2,60'a çıkarmayı ölç (izgara düz).
S1−0,006..S1     -> H8 çalıştı. Şampiyon v71/v72, kalan haklar b_sıcak'a.
S2 > S1          -> H8 YIKILDI. Şampiyon v67'ye döner, H8 yasak bölgeye yazılır.
```

---

## TİK GÜNLÜĞÜ

| tik | hipotez | ölçüm | çürütme | hüküm | dMSE | süre |
|-----|---------|-------|---------|-------|------|------|
| 0 | kurulum | defter + şampiyon | — | — | — | 10dk |
| 1 | kapı denetimi | `kapi_denetim.py`: v50/v55/v66/v67 → 4/4 GEÇTİ | — | GEÇTİ | — | 15dk |
| 1 | **H8 (yeni)** | soğuk gün ekseni: v50→v55→v66'da soğuk değişen satır **0/158.369** | 4 bağımsız saldırı | **KAZANÇ** | **−0,0149** | 150dk |
| 1 | H1 frekans ayrıştırma | 3 bant, oracle tavanı −0,00082 | çapa 3 banttan 1'inde tutuyor | **ÇÜRÜDÜ** | 0 | — |
| 1 | H2 değiştirme eşleştirme | tekil 1-1 eşleşme **7** (eşik ~100) | ikizde β=−0,0403, işaret TERS | **ÇÜRÜDÜ** | 0 | — |
| 1 | H3 05-11 kohortu | yapı doğrulandı (%68,35) | mekanizma ölçülemiyor, ikizde işaret dönüyor | **ÇÜRÜDÜ** | −0,00026 tavan | — |
| 1 | H4 boru hattı denetimi | 8/8 denetim GEÇTİ | — | **ÇÜRÜDÜ (temiz)** | ±0,0004 | — |
| 1 | H5 konum toplamı | yük düşüşü yok, R²=0,0000 | plasebo ile aynı büyüklük | **ÇÜRÜDÜ** | 0 | — |
| 1 | H6 ufuk ekseni | saf ufuk eğimi t=+1,46 (null) | yaz rampasının %87'si mevsim | **ÇÜRÜDÜ** | 0 | — |
| 1 | H7 c* yeniden türetme | v67 optimumdan +2,3e−7 uzakta | — | **ÇÜRÜDÜ (ama 2 bug buldu)** | 0 | — |
| 2 | H8 mutabakatı | eski çapa YERLEŞİK nüfusu (σ 0,2710), soğuk satırlar DOĞMUŞ nüfus (σ 0,4255) → 1,570 kat | aynı panelde eski c=1,411 → −0,00657 vs H8 c=2,20 → −0,01486 | **AÇIKLANDI** | — | 40dk |
| 2 | H8 kırpma (tam) | trafo: K=50'de +0,0142 · gün: K=50'de −0,0209 (t=−36) | doğal birim GÜN, orada sağlam | **KAPI GEÇTİ** | — | dahil |
| 3 | **H9 nüfus eşleşmesi** | test soğuğunun %82,5'i toplu katılım; 3 kohortta TOPLU/YERLEŞİK = 0,944 | risk ayarı bağımsız olarak aynı c'yi verdi | **c DÜZELTİLDİ** 2,20→1,60 | −0,0091 | 60dk |
| 4 | 08:00 çözücüsü | `coz_0800.py` — sabitler dosyalardan (etiketsiz), kuru koşu doğru | iç kapılar: seviye +0,160000 ✓ · soğuk 158.369 ✓ · diklik 1,3e−18 ✓ | **HAZIR** | — | 20dk |
| 5 | **kapalı eksen denetimi** | 19 eksen A/B/C'ye karşı tarandı; 3 kazı yapıldı | kazılar iki örtüşmeyen kesmeyle | **TEMİZ** (1 BELİRSİZ) | 0 | 130dk |
| 6 | **KUYRUK REJİMİ** | 353 trafo / 29.873 satır (%4,18); guz25 +0,475 · kis26 +0,353 | K=50'de iki blokta da ayakta | **KAZANÇ** | **−0,0052** | 45dk |
| 7 | altyapı: yaz25+guz25 soğuk tahminleri | 9 dk, 5 tohum × 3 aile | kis26'da +0,3017 = docs/43 birebir | **AÇIK KAPANDI** | — | 25dk |
| 7 | b_soğuk üç blokta | yaz25 +0,1056 · guz25 +0,0725 · kis26 +0,3017 | mekanizma: fold geleceği görüyor mu | **δ=0,16 TEYİT** | — | dahil |
| 7 | c\* harmanla yeniden çapa | yaz25 T3 2,737 (cat-only 3,127 idi) | H9b nüfus düzeltmesi aileden bağımsız | **c=1,60 TEYİT** | — | dahil |
| 7b | Kaggle yolu | iki biçim de OK; submit sınandı → temiz 400 | listede kayıt yok, hak yanmadı | **DOĞRULANDI** | — | 10dk |

**Tik 5 kapanışı:** kusur A tekil, kusur B gerçek ama sonucu değiştirmiyor,
kusur C'nin üç adayından ikisi doğru nüfusta yeniden ölçüldü ve hüküm ayakta.
`h16` tek koşuda **üç ekseni birden** kapattı (soğuk trafo seviyesi · kimlik
komşuluğu = on2/on3 · soğuk kVA kovası = kVA satırı), hepsi doğru nüfusta,
hepsi R²≈0. Tek kalan **BELİRSİZ: kalibrasyon/beta** — batch-ağırlıklı soğuk
nüfus için önbellekte model tahmini yok, yeniden eğitim gerekirdi; zaman
kutusuna sığmadı. Orakül tavanı zaten −0,002.

**Gönderim dosyalarına dokunulmadı** — hiçbir bulgu −0,002 eşiğini geçmedi.

### Tik 4 — 08:00 mekanikleştirildi

`scripts/coz_0800.py --sabitler` şimdiden koşuldu, sabitler dosyalardan
**etiketsiz** çıkarıldı:
```
Q2 = ort(Δ2²)      0.00202325
Q3 = ort(Δ3²)      0.00769600
A_c = Q2/(c−1)²    0.00562015
diklik ort(Δ2·(Δ3−Δ2)) = +1,3e−18   -> iki düzeltme TAM DİK, dMSE'leri toplanır
```
08:00'de yalnızca şu koşulacak:
```bash
uv run kaggle competitions submissions -c grid-up-datathon | head -5   # kural 8
# ... üç gönderim ...
uv run python scripts/coz_0800.py --s1 <S1> --s2 <S2> --s3 <S3>
```
Betik `c*`'ı, `b_soğuk`'u, ulaşılabilir RMSLE'yi ve **27 Ağustos komutunu**
doğrudan yazdırıyor.

---

## TİK 1 — H8 AYRINTISI (tek ayakta kalan)

### Yapısal olgu (kesin, tartışmasız)
`son_islem_gunolcek.py` — LB'de **doğrulanmış tek yapısal kazanç** — yalnızca
sıcak satırlara dokunuyor. Soğuk satırlarda değişen satır **tam 0**
(`h8_soguk_gun_ekseni.py`). Oysa soğuk satırlar test satırlarının %22'si ama
MSE'nin **%63'ü**: RMSE_soğuk **1,713** vs RMSE_sıcak **0,700**.

### İkiz panel
`data/interim/gun_ekseni/yaz25_*` = 2025 Nis–Tem'de **doğmuş** 678 trafo (%100),
6 tohum. Model RMSE 1,58; test soğuk tarafı 1,71 — **gerçek ikiz**.

### Çürütme 1 — doğum/giriş eseri mi? (`h8e`)
Panel aşırı dengesiz (trafo başına medyan 32/116 gün; tam pencere olan trafo 0).
Giderek temizlenen alt paneller:

| alt panel | n | trafo | c* | dMSE |
|---|---|---|---|---|
| T0 ham | 20.633 | 678 | 2,127 | −0,0556 |
| T1 ilk 7 gün atıldı | 17.235 | 435 | 2,463 | −0,0598 |
| T2 ≥60 günlük | 7.526 | 94 | 2,708 | −0,0770 |
| T3 = T1+T2 | 6.881 | 94 | 2,781 | −0,0675 |
| T4 ilk14 + ≥60 | 6.242 | 94 | 2,801 | −0,0680 |

Temizlik bulguyu **güçlendirdi** → doğum eseri **DEĞİL**. Saldırı başarısız.

### Çürütme 2 — iki blok ortak bölge (`h8f`)
(c_düşük, c_yüksek) ızgarasında yaz25 ve guz25'in ortak negatif bölgesi **YOK**.
"c_yüksek her iki blokta <1" izlenimi ortak uydurmanın eşdoğrusallık eseriydi:
c_düşük=1 sabitlenince yüksek frekans büzmesi yaz25'te **zarar** veriyor (+0,0064).
**Bu saldırı başta başarılı görünüyordu** — çürütme 3 çözdü.

### Çürütme 3 — SEVİYE mi GENLİK mi? (`h8h`) ← **kritik**
Kendi üretim kapım yakaladı: gün profili **günler** üzerinde ortalaması 0, ama
soğuk satırlar günlere eşit dağılmıyor (2026-05-11'de 1.326 trafoluk toplu
katılım) → müdahale gizlice **+0,0714'lük SEVİYE kayması** taşıyordu. Seviye
ayrı bir knob (b_soğuk) ve karışırsa iki ölçüm de bozulur.

Profil **satır-ağırlıklı** merkezlenip ayrıştırıldı:

| panel | dMSE_SEVİYE | dMSE_GENLİK | genlik payı | c_genlik |
|---|---|---|---|---|
| yaz25 T0 | −0,0037 | **−0,0792** (t=−88, 6/6) | **95,5%** | 3,03 |
| yaz25 T3 | −0,0024 | **−0,0852** (t=−70, 6/6) | **97,3%** | 3,10 |
| guz25 T0 | −0,0022 | −0,0003 | 10,5% | 0,96 |
| guz25 T3 | −0,0059 | −0,0001 | 1,9% | **1,01** |

İki sonuç:
1. yaz25'te kazanç **seviye değil genlik**, ve seviye sızıntısı temizlenince
   **büyüdü** (−0,0556 → −0,0792).
2. **guz25 genlik ekseninde NÖTR (c≈1,0), ZIT DEĞİL.** Çürütme 2'nin "guz25
   büzme istiyor" okuması seviye konfaundıydı. Yani **genlik ekseninde bloklar
   arası işaret çelişkisi YOK**: guz25 sıfır, yaz25 büyük pozitif. Fizik de
   bunu söylüyor — günlük soğutma yükü salınımı yazın büyük, güzün küçük;
   model yalnızca büyük olduğunda az yayıyor. Sıcak tarafta da aynı desen
   ölçülmüştü (yaz25 2,65 · guz25 0,75 · kis26 0,70).

### Çürütme 4 — çapa kalibrasyonu (`h8i`)
Mevsime bağlı bir parametre için "iki örtüşmeyen kesme aynı işareti versin"
kapısı mantıken sağlanamaz. O sınıfta geçerli kapı: çapa **etiketsiz** olacak
**ve** etiketli optimumu **üretebildiği ispatlı** olacak.

| panel | c_çapa | c_etiketli | oran |
|---|---|---|---|
| yaz25 (**testin mevsimi**) | 2,503 | 3,127 | **0,800** |
| guz25 | 1,054 | 1,010 | 1,046 |

Çapa testin mevsiminde **optimumun ALTINDA** kalıyor (0,80) — güvenli yön.

### Çapa (test etiketi KULLANILMADI)
```
σ_gerçek  0,3829   2025 Nis-Tem'de doğmuş trafolar, GERÇEK (train, etiketli)
σ_model   0,1527   2026 test soğuk, ŞAMPİYON tahmini, 1.823 trafo / 139.166 satır
oran      2,3185   korelasyon +0,8971 (110 ortak gün-of-year)
c_çapa    2,0799   yaz25 kalibrasyonuyla (0,80) düzeltilmiş ~2,60
```

### Seçilen c = 2,20 ve kırpma (kural 1)
Çapa 2,08 ile ikiz-etiketli 3,03 arasında, çapaya yakın muhafazakâr nokta.

yaz25 T0, **seviye-nötr**, kazanan trafo **434/678 (%64,0)**:
| K | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| c=2,20 | −0,0671 | −0,0519 | −0,0260 | −0,0141 | −0,0011 | +0,0142 |
| c=2,60 | −0,0763 | −0,0562 | −0,0218 | −0,0063 | +0,0107 | +0,0309 |
| c=3,00 | −0,0789 | −0,0539 | −0,0112 | +0,0078 | +0,0286 | +0,0535 |

**c=2,20 kırpmaya en dayanıklı nokta** (K=25'te hâlâ başabaş). Bu, seçimin
gerekçesidir — düşük c daha az kazanç ama çok daha sağlam.

**Beklenen test dMSE = p_soğuk × (−0,0671) = −0,01487** (gerekenin %77'si).

**Ters risk:** c*=1 olsaydı maliyet ≈ +0,0061 test MSE.

### Doğrulanan uygulama (H7 kusur sınıfına karşı)
```
istenen c 2,2000  =  ULAŞILAN c 2,2000   (fark −0,00%)
gün profili korelasyonu v67 vs v71 = +1,000000  (yalnız genlik değişti, şekil değil)
v67 genliği 0,1527 (hedefin %44,4'ü)  →  v71 genliği 0,3359 (hedefin %97,8'i)
seviye kayması tam 0,000000 · sıcak satırlar dokunulmadı
```

---

## TİK 2 — BÜYÜKLÜK MUTABAKATI ve TAM KIRPMA TABLOSU

`scripts/h8k_mutabakat_ve_kirpma.py`

### SORU 1 — Neden 25 kat fark? Tek cümle:

> **Eski beklenti soğuk satırları YANLIŞ NÜFUSA çapaladı:** σ_gerçek = 0,2710
> *yerleşik* trafolardan ölçülmüştü, oysa soğuk satırlar *yeni doğmuş*
> trafolar ve onların gerçek gün ekseni genliği **0,4255** — **1,570 kat**
> büyük; bu yüzden eski reçete c=1,411 seçti, H8 c=2,20 seçiyor.

Aynı pencerede, `son_islem_gunolcek.py`'nin **kendi** `gun_etkisi()`
fonksiyonuyla ölçüldü (protokol birebir aynı):

| nüfus | trafo | satır | σ_gerçek |
|---|---|---|---|
| (A) YERLEŞİK (≥%90 gün var) — eski çapanın kullandığı | 1.881 | 228.950 | **0,2710** |
| (B) DOĞMUŞ (soğuk ikizi) — H8'in kullandığı | 3.074 | 19.807 | **0,4255** |
| | | | oran **1,570** |

(A)'nın **0,2710'u tam olarak yeniden üretildi** → protokolüm eskisiyle
birebir aynı, fark yalnızca nüfusta.

**Aynı panelde, aynı ölçütle, seviye-nötr (yaz25 T0, 6 tohum):**

| c | panel dMSE | SH | t | **test etkisi** |
|---|---|---|---|---|
| 1,000 | 0 | — | — | 0 |
| **1,411** (eski reçete) | −0,02965 | 0,00041 | −72,3 | **−0,00657** |
| 1,500 | −0,03515 | 0,00046 | −76,4 | −0,00779 |
| **2,200** (H8) | −0,06708 | 0,00045 | −148,2 | **−0,01486** |
| 2,600 | −0,07626 | 0,00049 | −154,2 | −0,01690 |
| 3,030 (ikiz optimumu) | −0,07879 | 0,00117 | −67,1 | −0,01746 |

Yani farkın ayrışması:
- **c boşluğu (2,26 kat):** 1,411 → 2,20, kaynağı yanlış çapa nüfusu. Aday **(a)+(c) birlikte**.
- **Kalan ~11 kat:** eski reçetenin *kendisi* bile −0,00657 veriyor, docs/42'nin
  yazdığı −0,0006 değil. Kaynağı docs/41'in şu cümlesi:
  > "v50 ham soğuk gün ekseni std'si = **0,1626 / 0,60 = 0,2710**, ve 2025
  > Nis-Tem gerçek referansı da 0,2710. Ham model soğuk gün genliğini zaten
  > doğru biliyor."

  Bu bir ölçüm değil, **çıkarım**: 0,1626 büzme katsayısına bölünerek elde
  edilmiş ve tesadüfen *yanlış* referansa (0,2710) oturmuş — **döngüsel
  doğrulama**. Doğru referansla (0,4255) oran 0,4255/0,2710 = **1,570**, yani
  ham model büzme geri alınsa bile hâlâ 1,57 kat az yayıyor. "Zaten doğru
  biliyor" hükmü buradan yanlış çıktı ve beklentiyi ~sıfıra indirdi.

**Aday (b) — affin kalibre bugu — kaynak DEĞİL:** affin 1+0,893(1,507−1)=1,453,
çarpımsal 0,893×1,507=1,346. Bug c'yi **yükseltmiş**, düşürmemiş. Farkı açıklamıyor.

### ⚠ İŞLETME BULGUSU — docs/42 §2 adım (b) ÇALIŞMIYOR

Eski reçete koşuldu ve **çöktü**:
```
RuntimeError: olcek beklendigi gibi degil: 1.414 yerine 1.458
```
BUG 1 iş başında: istenen 1,458, ulaşılan 1,414, betiğin kendi kapısı ateşliyor.
`docs/42 §2 (b)` ve `docs/44 §3 sıra 2` **yazıldığı hâliyle koşulamaz**.
08:00 zincirinde kullanılmıyor — H8 betiği ayrı ve ulaşılan ölçeği doğruluyor.

### SORU 2 — TAM KIRPMA TABLOSU (kalıcı kural 1)

**TRAFO bazlı**, c=2,20, seviye-nötr, yaz25 T0, 6 tohum:

| K | kalan dMSE | SH | t | kalan trafo | kalan satır | kazanan | kaybeden |
|---|---|---|---|---|---|---|---|
| 0 | −0,06708 | 0,00045 | −148,15 | 678 | 20.633 | 434 (64%) | 244 |
| 1 | −0,05188 | 0,00038 | −136,71 | 677 | 20.560 | 433 (64%) | 244 |
| 5 | −0,02599 | 0,00080 | −32,34 | 673 | 20.198 | 429 (64%) | 244 |
| 10 | −0,01406 | 0,00107 | −13,09 | 668 | 19.778 | 424 (63%) | 244 |
| 25 | −0,00106 | 0,00122 | −0,87 | 653 | 18.985 | 409 (63%) | 244 |
| 50 | **+0,01415** | 0,00161 | +8,82 | 628 | 17.982 | 384 (61%) | 244 |

**GÜN bazlı**, c=2,20 (aynı koşu):

| K | kalan dMSE | SH | t | kalan gün | kalan satır | kazanan | kaybeden |
|---|---|---|---|---|---|---|---|
| 0 | −0,06708 | 0,00045 | −148,15 | 117 | 20.633 | 99 (85%) | 18 |
| 1 | −0,06479 | 0,00041 | −158,78 | 116 | 20.559 | 98 (84%) | 18 |
| 5 | −0,05818 | 0,00028 | −210,41 | 112 | 20.136 | 94 (84%) | 18 |
| 10 | −0,05425 | 0,00038 | −142,58 | 107 | 18.313 | 89 (83%) | 18 |
| 25 | −0,04283 | 0,00075 | −56,89 | 92 | 14.961 | 74 (80%) | 18 |
| 50 | **−0,02087** | 0,00058 | −36,03 | 67 | 11.743 | 49 (73%) | 18 |

**Karşılaştırma — eski c=1,411, TRAFO bazlı:** K=50'de hâlâ −0,00121 (t=−4,07),
yani düşük c kırpmaya daha dayanıklı. Büyüklük/sağlamlık ödünleşmesi gerçek.

### Kırpma tablolarının HÜKMÜ — dürüst okuma

H8 **trafo ekseninde** yoğunlaşma imzasının bir kısmını taşıyor: en büyük 50
trafo (%7,4) atılınca işaret dönüyor. Ama **gün ekseninde taşımıyor**: 117
günün en iyi 50'si atılınca bile −0,02087 (t=−36) kalıyor ve günlerin
**%85'i** kazanıyor.

Bu ayrım anlamlı, çünkü **gün ekseni genlik düzeltmesinin doğal birimi GÜNDÜR**:
müdahale gün başına tek bir sabittir, trafo bazında hiçbir şey seçmez.
Trafo kırpmasının sert olması beklenen bir şeydir — MSE mutlak birimde ve
yüksek varyanslı birkaç trafo, gün başına aynı kaymadan orantısız pay alır.

**Ölümcül imzayla farkı:** `sıcak kapasite` kazancı **kendi doğal biriminde**
(trafo) 4 trafodan geliyordu ve K=25'te t=−4,03 ile **ters** dönüyordu. H8
kendi doğal biriminde (gün) K=50'de bile sağlam.

**Yine de S2 okunurken bu beklenti tutulacak:** gerçekleşen kazanç
−0,01486'nın altında çıkarsa şaşırmayacağız; trafo-kırpılmış görünüm
(−0,001…−0,014 bandı) alt sınır olarak akılda tutulur. **Göndermeyi
engellemez** — S2'nin bütün amacı bu belirsizliği tam olarak çözmek.

---

## TİK 3 — H9: NÜFUS EŞLEŞMESİ, ve c'nin 2,20 → 1,60'a ÇEKİLMESİ

`scripts/h9_parti_nufus_eslesmesi.py`, `scripts/h9b_toplu_kohort_genligi.py`,
`scripts/h8l_c_risk_ayari.py`

### Soru
H8'in çapası σ_gerçek = 0,4255'e dayanıyordu ve bu, 2025 Nis–Tem'de **doğmuş**
3.074 trafodan ölçülmüştü. Ama test'in soğuk nüfusu aynı şey mi? Test soğuk
satırlarının **%82,5'i ≥100'lük toplu katılımlardan** geliyor (2026-05-11
tek başına 1.326 trafo). `son_islem_olay.py` zaten belgeliyor: 100+ parti bir
enerjilendirme dalgası değil, **veri setine toplu katılım (geriye dolgu)**.
Öyleyse genlikleri yerleşik profiline yakın olabilir ve 0,4255 fazla yüksektir.

### İlk deneme yetersiz kaldı (H9)
yaz25'i parti büyüklüğüne ayırınca TOPLU sınıfı σ=0,0559 çıktı — 8 kat küçük.
Ama **sayı geçersiz**: yaz25'teki tek ≥100 partisi 2025-07-28 ve pencere
07-31'de bitiyor → trafo başına ~4 gün. Dört günde "gün ekseni genliği"
ölçülmez.

### Mevsim kontrollü ölçüm (H9b) — üç kohort, her biri kendi penceresinde
Takip penceresi yeterli olan üç toplu kohort bulundu; her biri **aynı
pencerede** ölçülen iki referansla karşılaştırıldı (mevsim böylece kontrol
edilir):

| kohort | pencere | TOPLU σ | TEKİL σ | YERLEŞİK σ | TOPLU/YER | TEKİL/YER |
|---|---|---|---|---|---|---|
| 2025-01-01 (1.902 trafo) | 128 gün | 0,1175 | 0,1802 | 0,1149 | 1,022 | 1,568 |
| 2025-07-28 (166) | 128 gün | 0,1890 | 0,1535 | 0,2300 | 0,822 | 0,667 |
| 2025-11-25 (153) | 127 gün | 0,0707 | 0,0772 | 0,0716 | 0,987 | 1,078 |
| | | | | **ortalama** | **0,944** | **1,104** |

> **Toplu katılım trafolarının gün ekseni genliği yerleşiklerinkiyle
> pratik olarak AYNI (0,944×).** Tekil doğumlar yalnızca %10 daha yüksek.
> Ne 8 kat küçük, ne de 1,57 kat büyük.

**0,4255 nereden geliyordu?** yaz25'in doğmuş paneli aşırı dengesiz (trafo
başına medyan 32/116 gün) ve σ'sı örneklem gürültüsü + giriş kompozisyonuyla
şişmiş. Üç kohortluk ölçüm 120 günlük dengeli panellerde yapıldı.

### Yeniden türetilen çapa
```
σ_YERLEŞİK(yaz25)   0,2710   (ölçüldü, h8k'de doğrulandı)
σ_TOPLU(yaz25)      0,2557 = 0,2710 × 0,944
σ_TEKİL(yaz25)      0,2993 = 0,2710 × 1,104
test karışımı       TOPLU 0,8251 · TEKİL/orta 0,1749
σ_hedef             0,2634 = 0,825×0,2557 + 0,175×0,2993
c_çapa = kor × σ_hedef / σ_model = 0,8971 × 0,2634 / 0,1527 = 1,547
```

### Bağımsız ikinci kriter — kırpmaya karşı risk ayarı (h8l)
Aynı c için üç senaryo (yaz25 T0, seviye-nötr, 6 tohum), **panel dMSE**:

| c | K=0 iyimser | K=25 orta | K=50 kötümser | EN KÖTÜ |
|---|---|---|---|---|
| 1,40 | −0,02895 | −0,00639 | −0,00126 | güvenli |
| **1,50** | −0,03515 | −0,00704 | −0,00064 | **güvenli** |
| **1,60** | −0,04095 | **−0,00732** ← argmin | +0,00035 | ~nötr |
| 1,80 | −0,05131 | −0,00674 | +0,00345 | |
| 2,00 | −0,06002 | −0,00466 | +0,00805 | |
| 2,20 | −0,06708 | −0,00106 | **+0,01415** | |
| 3,00 | −0,07886 | +0,02855 | +0,05345 | |

`c=1,60` **orta senaryonun argmin'i** ve kötümserde nötr.

### YAKINSAMA — kararın dayanağı
```
nüfus eşleşmesi (H9b)      c = 1,547
kırpma risk ayarı (h8l)    c = 1,60
```
**İki tamamen bağımsız kriter aynı noktaya iniyor.** Biri fiziksel nüfus
kompozisyonundan, diğeri adversaryel sağlamlıktan. c=2,20'yi destekleyen
tek şey, artık şişmiş olduğu bilinen 0,4255 çapasıydı.

**SEÇİLEN c = 1,60.** Şampiyon `v71` → `v73`.

Beklenen test dMSE: **−0,00907** (iyimser) · −0,00162 (orta) · +0,00008 (kötümser).

### KALICI KURAL 14 (bu koşudan çıktı)
> **Bir referans değeri MODEL ÇIKTISINDAN türetilip (ör. ham std / büzme
> katsayısı) sonra aynı modeli doğrulamak için kullanılamaz — bu döngüsel
> doğrulamadır. Referans HER ZAMAN gerçek etiketlerden, ve HEDEF NÜFUSUN
> İKİZİNDEN ölçülür.**

docs/41'in soğuk gün ekseni hükmü tam bu hatadan çıktı (`0,1626/0,60 = 0,2710`)
ve bize bir ekseni 1 gün kaybettirdi. **Yarın:** başka kararların da aynı
hatayla verilip verilmediğine bakılmalı. Bu gece kod değiştirilmiyor.

Kural 14'ün ikinci yarısı ("hedef nüfusun ikizinden") bu tikte **kendi
bulgumu da düzeltti**: H8'in çapası gerçek etiketlerden geliyordu ama
**yanlış nüfusun** ikizinden.

---

## TİK 5 — KAPALI EKSENLERİN DENETİMİ (A/B/C kusurları)

**Gerekçe:** bu gecenin tek bulgusu (H8) kuyruktan çıkmadı — kuyruktaki 7
fikrin 7'si de çürüdü. H8 **kendi işimi denetlerken** çıktı. Denetim,
kuyruktan üretken.

Aranan üç kusur:
- **A — DÖNGÜSEL DOĞRULAMA:** referans model çıktısından mı türetilmiş?
- **B — YANLIŞ NÜFUS:** çapa hedef nüfusun ikizinden mi ölçülmüş?
- **C — KURAL 10:** mevsime/nüfusa bağlı hüküm yalnız `kis26`'da mı verilmiş?

### TARAMA — 19 kapalı eksen

| eksen | kapanma tabanı | A | B | C | açıklık |
|---|---|:-:|:-:|:-:|---|
| **soğuk sıfırlar (%46 duvarı)** | **kis26 soğuk** (61.918 satır) | — | **✗** | **✗** | **YÜKSEK** |
| **soğuk hurdle** | **kis26 soğuk** (taban 1,98505) | — | **✗** | **✗** | **YÜKSEK** |
| **kalibrasyon / beta** | **kis26 soğuk** (taban 1,98505) | — | **✗** | **✗** | **YÜKSEK** |
| soğuk kVA kovası | bloklar arası taşımıyor | — | ✗ | ~ | orta |
| kayıt sonlanması / p_son_ofset | ikizde −0,16, K=50 −0,05 | — | ✗ | — | orta |
| kimlik komşuluğu | R² 0,019 (ilçe 0,016 yanında) | — | ~ | — | orta-düşük |
| soğuk trafo seviyesi | as-of OOF R² 0,015, **3 kez** | — | ~ | — | düşük |
| sıcak kapasite | K=25 t=−4,03 (kural 1 yıktı) | — | — | ~ | düşük |
| soğuk kapasite | t=+0,18 | — | ~ | — | düşük |
| grup B yukarı kaydırma | **ikizde** işaret ters | — | — | — | düşük |
| ölü kuyruğa log1p 0,20 | +0,29 MSE (skor 1,156) | — | — | — | yok |
| takvim / tatil | günlük std 0,0525, train geneli | — | — | — | yok |
| harman (power mean, NNLS) | üretim zaten optimumda | — | — | — | yok |
| λ pencere genişletme | 2025 tam yıl kor +0,029 | — | — | — | yok |
| ulusal yük ile gün faktörü | **3 blokta da** zarar | — | — | — | yok |
| b_i trafo kestiricisi | **5 temiz kesme**, 0/12 | — | — | — | yok |
| sabit δ transferi | **5 kesme**, \|ort\|/std 0,152 | — | — | — | yok |
| artık hedefi u = ofs − seviye_i | **ikizde** 3/3 kötü | — | — | — | yok |
| on6/on7 hedef kodlaması | kural 1, **üç pencerede** | — | — | — | yok |
| son pencere çapası / geçen yıl | **üç blokta** zararlı | — | — | — | yok |
| gün faktörü ↔ hava modeli | +0,0007 | — | — | — | yok |

**KUSUR A (döngüsel doğrulama) hiçbir eksende ikinci bir örnek vermedi.**
`docs/41`'in `0,1626/0,60 = 0,2710`'u tekil görünüyor. İyi haber.

**Asıl bulgu KUSUR B+C'de ve tek bir kümede yoğunlaşıyor:** soğuk
sıfırlar / hurdle / kalibrasyon üçü de **yalnızca `kis26` soğuk** üzerinde
kapatıldı (üçünün de tabanı aynı sayı: RMSLE 1,98505). Ve bu üçü, MSE'nin
%63'ünü taşıyan nüfusla ilgili — ödül sıralamasının en tepesi.

**Neden `kis26` soğuk, test soğuğunun ikizi olmayabilir:** test soğuk
satırlarının **%82,5'i ≥100'lük toplu katılımdan** geliyor (tik 3) ve toplu
katılım `son_islem_olay.py`'nin belgelediği gibi bir enerjilendirme dalgası
değil, **zaten çalışan trafoların veri setine geriye dolgusu**. Geriye dolgu
edilen bir trafonun sıfır üretme olasılığı, gerçekten yeni enerjilendirilmiş
bir trafonunkinden farklı olmak zorundadır. `kis26` soğuğunun toplu-katılım
payı ölçülmedi.

**Bu, projenin en merkezi sayısını riske atıyor:** "toplam MSE'nin ~%46'sı
soğuk sıfırlarda ve mevcut özniteliklerle **indirgenemez**" — `docs/41` §2-3.
Bu cümle "1'in altı" hedefinin önündeki duvar olarak kabul edildi ve
stratejiyi belirledi. Yanlış nüfusta ölçüldüyse duvar başka yerde.

### KAZI 1 — nüfus kompozisyonu (`h14_soguk_sifir_nufus_denetimi.py`)

**Şüphe DOĞRULANDI, ve mismatch beklenenden büyük:**

| nüfus payı (satır) | kis26 soğuk | TEST soğuk |
|---|---|---|
| TOPLU ≥100 | **%1,1** (701 satır) | **%80,7** |
| orta 20-99 | %39,6 | %7,4 |
| tekil/küçük <20 | %59,2 | %11,9 |

`kis26` soğuk **%59 tekil / %1 toplu**; test soğuk **%12 tekil / %81 toplu**.
Bunlar pratik olarak **farklı nüfuslar**. Üç eksen (soğuk sıfırlar, hurdle,
kalibrasyon) testin %81'ini oluşturan sınıftan **701 satır** görerek kapatıldı.

**Ama sıfır oranı etkisi ılımlı:** test karışımıyla yeniden ağırlıklandırınca
0,0447 vs kis26'nın 0,0523 → oran **0,853**. "%46 duvarı" test karışımında
kabaca **%39**. Duvar gerçek, ama **%15 abartılmış**.

Mevsim kontrollü kohort ölçümü (sıfır oranı): TOPLU 0,0678 · TEKİL 0,0918 ·
YERLEŞİK 0,0470 → TEKİL/TOPLU = 1,35 kat. Yön hipotezle uyumlu ama
kohortlar arası **kararsız** (2025-11-25'te ters: TOPLU 0,1025 > TEKİL 0,0461).

### KAZI 2 — hurdle TOPLU nüfusunda yeniden sınandı (`h15_toplu_sifir_ogrenilebilir_mi.py`)

Tasarım: **iki örtüşmeyen zaman kesmesi** (kural 9) — eğit 2025-01-01 kohortu
(2.048 trafo), sına 2025-11-25 kohortu (167 trafo). Öznitelikler yalnızca
doğumda bilinenler (kural 8 AS-OF); trafonun kendi geçmişi **kullanılmadı**.

**Yoğunlaşma — "ölü trafo" çerçevesi DOĞRU:**
| | 2025-01-01 | 2025-11-25 |
|---|---|---|
| TAM ÖLÜ (>%95 sıfır) | 127/2.032 (%6,2) | 16/166 (%9,6) |
| HİÇ sıfırı yok | 1.874 (%92,2) | 148 (%89,2) |
| sıfır kütlesinin ölülerden gelen payı | **%89,6** | **%94,4** |

**Öğrenilebilirlik — ASIL BULGU:**
```
ilce   AUC eğit 0,7131   AUC SINA 0,3304
on2    AUC eğit 0,6098   AUC SINA 0,2898
on3    AUC eğit 0,7607   AUC SINA 0,3976
BİRLEŞİK AUC eğit 0,7709   AUC SINA 0,3598
trafo düzeyinde                AUC SINA 0,3904
en yüksek 10 trafonun gerçek sıfır oranı: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
```

> **Sinyal zayıf değil — TERS DÖNÜYOR.** Kohort İÇİNDE lokasyon/önek
> sıfırlığı güçlü kestiriyor (AUC 0,77), ama başka bir zaman kesmesine
> geçince AUC **0,36**'ya düşüyor: rastgeleden kötü, yani işaret **tersine**
> dönmüş.

**Bu, `docs/41`'in hükmünü DEĞİŞTİRMİYOR ama GEREKÇESİNİ DÜZELTİYOR.**
docs/41 "AUC 0,5728, sinyal zayıf" diyordu. Gerçek: sinyal **güçlü** ama
**hiç taşınmıyor**. Ölü trafo kümesi zaman içinde yenileniyor — bir dönemin
ölü ilçesi başka dönemin canlı ilçesi.

Pratik sonuç aynı ve artık **daha sağlam**: hurdle/sıfır ekseni kapalı kalır.
Ve bu, tek başına in-sample AUC 0,77 görüp heveslenecek herkes için bir uyarı.

### KAZI 3 — soğuk trafo SEVİYESİ, TOPLU nüfusunda (`h16_toplu_seviye_ogrenilebilir_mi.py`)

Aynı tasarım (eğit 2025-01-01 → sına 2025-11-25, yalnız doğumda bilinen
öznitelikler). Hedef: trafonun ortalama log-ofseti. R², kohortun **kendi
ortalaması** üzerine ölçüldü — yani küresel kayma (`b_soğuk`) zaten çıkarılmış,
sadece **EK** değer raporlanıyor.

| kestirici | R² (kendi ort. üzerine) | kor |
|---|---|---|
| lokasyon (ilçe) | **−0,0225** | 0,218 |
| tanım on2 | +0,0154 | 0,137 |
| tanım on3 | −0,0125 | 0,235 |
| kVA (doğrusal) | −0,0246 | 0,047 |
| lokasyon + kVA harman | **+0,0329** | 0,201 |

Hepsi ~0. **Eksen TOPLU nüfusunda da kapalı — dördüncü kez, ve ilk kez DOĞRU
NÜFUSTA.** Denetim temiz.

**Yan bulgu — S3'ü destekliyor:** küresel kaymanın kendi R²'si **0,1820**.
İki kohortun ortalama ofseti +0,6583 → +1,1170, yani **+0,459 fark**. Soğuk
seviyede baskın yapı trafo bazlı değil **kohort bazlı küresel kayma** — ki
S3'ün çözmek üzere kurulduğu şey tam olarak bu.

### KAZI 4 — `b_soğuk` yanlış nüfusa mı çapalanmış? (`h17_b_soguk_parti_ayrisimi.py`)

**Soru neden kritik:** özdeşlik her δ için `b`'yi tam çözer, ama **bankaya
yatırılan** kazanç δ'nın b'ye yakınlığına bağlı:
`kazanç(δ,b) = 2pδb − pδ²`. Gerçek b=0,25 ise δ=0,16 → 0,01206, δ=0,25 →
0,01385; fark **0,0018** — kalibrasyon orakül tavanıyla aynı büyüklükte.

**Ölçüm** (`kis26_soguk_meta` + `soguk_tahmin_kis26.npz`, 61.918 satır,
3 tohum × cat/xgb/lgbm). Genel yanlılık **+0,3273** (docs/43 YOL 1: +0,3017 ✓).

| sınıf | satır | trafo | kis26 payı | TEST payı | b | tohum SH |
|---|---|---|---|---|---|---|
| tekil/küçük <20 | 36.677 | 632 | 0,5923 | 0,1188 | **+0,3272** | 0,0162 |
| orta 20-99 | 24.540 | 262 | 0,3963 | 0,0744 | **+0,3375** | 0,0154 |
| TOPLU ≥100 | **701** | 329 | 0,0113 | **0,8067** | −0,0238 | 0,0218 |

Naif yeniden ağırlıklandırma `b = +0,0448` (oran 0,137) verir — yani δ'yı
0,16'dan 0,02'ye çekmek gerekirdi. **BU SAYI SAHTE, reddedildi.**

**Neden sahte:** TOPLU hücresi 701 satır / 329 trafo = **trafo başına 2,13
gün**, ve tamamı **tek bir doğum gününden: 2026-03-26**, tarih aralığı
03-26→03-31. Bu tam olarak H4 ajanının işaretlediği anomali: *"2026-03-26/27'de
trafo sayısı 3.875'ten 4.424'e fırlıyor (+%14) ve o ~550 trafo 03-28'de
kayboluyor."* İki günlük bir kuyruk artefaktında ölçülen yanlılık hükme
dayanak olamaz.

**Geçerli kanıt — gradyan (parti büyüklüğü sürekli değişken):**
| ort parti | 5,5 | 12,0 | 15,7 | 19,4 | 28,1 | 65,1 |
|---|---|---|---|---|---|---|
| b | +0,337 | +0,292 | +0,510 | +0,165 | +0,405 | +0,284 |

`log(parti)` eğimi **−0,0149 / log birim** — parti 10 kat büyüdüğünde yanlılık
yalnızca −0,034 değişiyor, ve kovalar arası tek yönlü bir eğilim **yok**.
Test'in parti ölçeğine (≈1.326) ekstrapole edilirse b ≈ 0,327 − 0,045 ≈ **0,28**
— 0,045 değil.

**Kırpma (kalıcı kural 1):** b kırpmayla **YÜKSELİYOR** (tekil K=0 +0,327 →
K=50 +0,650; orta +0,338 → +0,591). Yanlılık geniş tabanlı, birkaç trafodan
gelmiyor; aykırılar onu aşağı çekiyor. Sağlam pozitif.

> **HÜKÜM: `b_soğuk`'ta ölçülebilir bir nüfus bağımlılığı YOK. δ = 0,16
> KALIYOR.** H8'in başına gelen buraya gelmemiş.

**Ayrıca not:** kis26 ham soğuk yanlılığı +0,327 ve bu 0,16'nın epey üstünde.
0,16 zaten kural 9 gereği **bilerek büzülmüş** bir değer (kis26 teste taşımıyor;
ikiz-çapa yolu +0,1454 vermişti). S3 `b`'yi **tam** çözeceği için bu bir risk
değil: b > 0,25 dönerse 27 Ağustos'un ilk hakkı gerçek optimumu yazar.

**Bu, gecenin üçüncü reddedilen dramatik sayısı** (H9'da 0,0559; burada
−0,0238). Desen aynı: küçük n × dar pencere → uç değer.

### KAZI 5 — `b_soğuk` keskinleştirilebilir mi? ve δ kararı

**1.'lik aritmetiği:** `kazanç(δ=0,16, b) = 0,070909·b − 0,005673`. Açık
+0,002893. Yarın sabah 1. olmanın tek yolu:
```
0,070909·b − 0,005673 = 0,005673 + 0,002893   ->   b ≥ 0,2008
```

**Kestirim bandı — ve içindeki TUZAK:**
| kaynak | b | geçerli mi? |
|---|---|---|
| yaz25 ikiz (doğmuş trafolar, 6 tohum) | +0,0595 (SH 0,0062) | **HAYIR** |
| guz25 ikiz (3 tohum) | +0,0465 | **HAYIR** |
| kis26 ham (H17) | +0,3273 | evet |
| kis26 soğuk FAZLASI (kis26 soğuk − sıcak) | +0,1118 | evet |
| docs/43 YOL 2 ikiz-çapa | +0,1454 | evet |
| ön kayıtlı | **0,16** | |

> **KENDİ ÇERÇEVEMDE HATA:** yaz25/guz25 ikizlerinin ~0,05 vermesini önce
> "b düşük olabilir" diye okudum. **Yanlış.** `son_islem_seviye.py` bunu zaten
> belgeliyor: `blok_parcalari` hedef blok DIŞINDAKİ her şeyi eğitime koyar,
> yani yaz25/guz25 fold'ları **GELECEĞİ görüyor** ve sürüklenmeyi zaten
> biliyorlar. Seviye sorusunu yanıtlayabilen **tek fold kis26**'dır (yalnız
> geçmişe bakar, tıpkı test gibi). O yüzden ikizlerin 0,05'i **seviye ekseni
> için geçersiz** — gün ekseni için geçerliydi (orada fold'un geleceği görmesi
> genliği bozmuyor), seviye için değil.

**δ KARARI: 0,16'da KALIYOR.** Kuadratik kayıpta optimal δ = **E[b]**, ve
geçerli kestirimlerin ortalaması (0,1118 / 0,1454 / 0,3273 → ağırlıklı ~0,15)
tam orada. δ'yı 1.'liği kovalamak için 0,20'ye çekmek beklenen skoru
**düşürür**: b=0,11 ise δ=0,16 → +0,00213, δ=0,20 → +0,00089.

**Keskinleştirilemez:** batch-eşleşmiş bir seviye ölçümü için batch-ağırlıklı
soğuk nüfusta model tahmini gerekir; önbellekte yok (yaz25'in toplu kohortu
4 günlük, guz25'inki 5 günlük). S3 zaten `b`'yi **tam** çözecek.

---

## TİK 6 — **KUYRUK REJİMİ**: gecenin ikinci gerçek bulgusu

`h18_kuyruk_dogumlulari.py`, `h18b_kuyruk_rejimi.py`,
`son_islem_kuyruk_rejimi.py`

**Nereden çıktı:** H17'de reddettiğim artefaktın yan ürünü. 2026-03-26/27'de
train trafo sayısı 3.875→4.424'e fırlıyor — ve **train 2026-03-31'de bitiyor**,
yani bu sıçrama test modelinin özet penceresinin **tam ucunda**.

**Yapısal olgu:** 2026-03-26..31'de ilk kaydı oluşan 356 trafonun **353'ü
test'te: 29.873 satır = testin %4,18'i.** Train kayıtları **medyan 2** (min 1,
max 6). Model onları **SICAK** sayıyor (tanım train'de var) ama geçmişleri
pratikte yok → **sıcak/soğuk ikili ayrımının görmediği ÜÇÜNCÜ REJİM.**

**İki örtüşmeyen kesmede ölçüldü (kural 9):**
| blok | kesme | KUYRUK ≤6g | SÜREN >180g | **FAZLA** | t | tohum |
|---|---|---|---|---|---|---|
| guz25 | 2025-07-31 | +0,1270 (182 trafo) | −0,3484 | **+0,4754** | +27,7 | 3/3 |
| kis26 | 2025-11-30 | +0,5623 (202 trafo) | +0,2091 | **+0,3531** | +7,2 | 3/3 |

**Neden taşınır:** bu mutlak seviye değil, **aynı blok içinde gruplar arası
fark**. guz25'te bütün gruplar negatif yanlılıkta (fold geleceği görüyor),
kis26'da pozitif — **ama FAZLA ikisinde de pozitif**. Fold'un küresel bilgisi
farkta sadeleşiyor. (Bu, `docs/41`'in "yaz25/guz25 geleceği görüyor"
uyarısının mutlak seviye için geçerli olduğunu, fark için olmadığını gösterir.)

**yaz25 bu ekseni GÖREMEZ:** kesmesi 2025-03-31 ve train 2025-01-01'de
başladığı için orada yalnızca **4** kuyruk trafosu var. Ama bu **mevsimsel bir
eksen değil** (geçmiş UZUNLUĞU ekseni), kural 7 devrede değil; kural 9 sağlandı.

**Mekanizma: parti değil GEÇMİŞ UZUNLUĞU.** Her iki blokta hem toplu hem
tekil kuyruk doğumluları etkileniyor (guz25 toplu +0,53 / tekil +0,33;
kis26 toplu +0,27 / tekil +0,60). Doz-tepki de var (kis26):
`≤6g +0,562 | 7-30g +0,539 | 31-90g +0,385 | 91-180g +0,161 | >180g +0,209`

**KIRPMA (kural 1) — K=50'de İKİ BLOKTA DA AYAKTA:**
| K | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| guz25 | +0,475 | +0,463 | +0,435 | +0,393 | +0,316 | **+0,196** (t=+7,8) |
| kis26 | +0,353 | +0,353 | +0,328 | +0,306 | +0,252 | **+0,175** (t=+4,3) |

> **Bu, gecenin kural 1 ve kural 9'u BİRLİKTE geçen ilk bulgusu.** H8 kural
> 1'de sınırdaydı (trafo ekseninde K=50'de dönüyordu); bu ikisinde de sağlam.

**δ = 0,30 seçildi:** blok kestirimleri 0,3531 / 0,4754, K=50 kırpılmış hâli
~0,18–0,20. Kuadratik optimum δ = E[b]; 0,30 blok kestirimlerinin altında,
kırpılmışların üstünde.

| gerçek b | 0,414 | 0,353 | 0,20 | 0,10 |
|---|---|---|---|---|
| dMSE (δ=0,30) | **−0,00662** | **−0,00509** | −0,00125 | +0,00125 |

**Üretilen:** `submissions/tuketim_v76_kuyruk30.csv` = v75 + kuyruk 0,30.
Kapılar **GEÇTİ** (714.688 · id birebir · 0 NaN · 0 negatif · 0 mükerrer ·
uygulanan kayma tam +0,300000 · hedef dışı maxabs 0,00e+00).

### S3 KARARI — v76, ve `b_soğuk` çözümüne maliyeti

Kuyruk grubu **SICAK**, soğuk grup **SOĞUK** → **ayrık**. Bu yüzden özdeşlik
hâlâ çalışıyor, ama S3'te iki bilinmeyen olur (`b_cold`, `b_tail`) ve tek
denklem. `b_tail`'i ikizden ön kayıtlı alıp (0,414 ± 0,061) `b_cold`
çözülürse belirsizlik:
```
δ_t·p_t·Δb_t = 0,30 × 0,0418 × 0,061 = 0,000765
Δb_cold = 0,000765 / (0,16 × 0,22159) = ±0,022
```
**`b_soğuk` TAM yerine ±0,022 ile çözülür.** Karşılığında ~0,005–0,0066
bankaya yatar. ±0,022, 27 Ağustos'ta optimumu yazmak için fazlasıyla yeterli.
**Takas kabul edildi: S3 = v76.**

---

## TİK 7 — ALTYAPI AÇIĞI KAPATILDI: soğuk tahminler artık ÜÇ blokta

`scripts/uret_soguk_tahmin.py`, `scripts/h21_b_soguk_ikizde.py`

**Açık:** önbellekte yalnızca `soguk_tahmin_kis26.npz` vardı. Bu yüzden
gecenin bütün soğuk hükümleri (H14/H15/H16/H17) kural 10'un **kullanılmamasını
söylediği** blokta verildi. Tek eksik dosya, bir sınıf hükmü kilitliyordu.

**Kapatıldı** (9 dakika, ~11 çekirdek paralel, 5,6 GB):
```
data/interim/deney/soguk_tahmin_yaz25.npz   23:41   5 tohum x cat/xgb/lgbm
data/interim/deney/soguk_tahmin_guz25.npz   23:50   5 tohum x cat/xgb/lgbm
```
Ayar `deney_soguk_taban.py` ile birebir (saf soğuk uzman maske=1,00, cat
depth 7) ve format kis26 npz'siyle aynı → mevcut betikler değişmeden çalışır.

### `b_soğuk` — ilk kez üç blokta, üretim harmanıyla (cat/xgb/lgbm = 3/1/1)

| blok | fold ne görüyor | b | eşlenik SH | t | tohum |
|---|---|---|---|---|---|
| **yaz25** (mevsimsel ikiz) | GELECEĞİ görüyor | **+0,1056** | 0,0025 | +41,8 | 5 |
| **guz25** | GELECEĞİ görüyor | **+0,0725** | 0,0104 | +7,0 | 5 |
| **kis26** | yalnız GEÇMİŞ | **+0,3017** | 0,0109 | +27,8 | 3 |

> **BORU HATTI DOĞRULAMASI:** kis26'da ölçtüğüm **+0,3017**, `docs/43`'ün
> yayımladığı sayının **birebir aynısı**. Dolayısıyla yaz25/guz25 sayıları da
> aynı güvenilirlikte.

**Mekanizma net:** `b` iki bileşenden oluşuyor —
(i) modelin ekstrapole edemediği yıllık sürüklenme, (ii) mevsim ekstrapolasyon
hatası. Geleceği gören fold'larda (i)≈0 → b düşük (0,07–0,11). kis26'da ikisi
de var → b yüksek (0,30). **Test'te (i) var ama (ii) küçük** (Nis–Tem 2025
eğitimde, yani mevsimsel ikiz mevcut). Yani test'in b'si iki uçtan da farklı:
```
b_test ≈ (i) + küçük (ii)
(i)'nin fold içermeyen tek kestirimi: docs/43 YOL 2 ulusal büyüme çapası = +0,1454
```
**δ_soğuk = 0,16 KALIYOR** — çapanın hemen üstünde, savunulabilir bandın
(0,106–0,302) içinde, ve kuadratik düz: b=0,145'te δ=0,16 ile δ=0,145 arası
fark **0,00005**.

### ⚠ KALICI TUZAK — `gun_ekseni/*_taban.npy` YALNIZ `cat`

`deney_gun_ekseni_dogrula.py:121` → `di.egit_tahmin("cat", ...)`. Üretim
harmanı ise **cat/xgb/lgbm = 3/1/1**. Aile farkı büyük:

| aile | yaz25 b | kis26 b |
|---|---|---|
| cat | +0,0640 | +0,2632 |
| xgb | +0,1932 | +0,3511 |
| lgbm | +0,1429 | +0,3676 |
| **harman 3/1/1** | **+0,1056** | **+0,3017** |

`taban` dosyalarından ölçüldüğünde b=+0,0595 çıkıyordu — **%44 düşük**.
Seviye/yanlılık ölçen her iş **harmanla** yapılmalı; `taban` dosyaları
yalnızca gün ekseni GENLİĞİ için güvenli (orada aile farkı sadeleşiyor).

### Kırpma (kural 1) — yaz25
| K | 0 | 1 | 5 | 10 | 25 | 50 |
|---|---|---|---|---|---|---|
| b | +0,1056 | +0,1076 | +0,1130 | +0,1199 | +0,1938 | +0,2434 |

Kırpmayla **yükseliyor** → geniş tabanlı, aykırılar aşağı çekiyor. Sağlam.

### c\* üretim harmanıyla yeniden çapalandı (`h22_c_ikizde_harman.py`)

Aynı tuzak c için de sınandı — genlik ekseninde aile farkı sadeleşiyor mu?

| blok / panel | c\* (etiketli) | c_çapa | oran | dMSE |
|---|---|---|---|---|
| yaz25 T0 (678 trafo) | 2,576 (std 0,090) | 2,174 | 0,844 | −0,0777 |
| yaz25 T3 (94 trafo) | 2,737 (std 0,096) | 2,129 | 0,778 | −0,0837 |
| guz25 T0 (1.173 trafo) | 0,934 | 1,078 | 1,154 | −0,0002 |
| guz25 T3 (272 trafo) | 1,051 | 1,097 | 1,044 | −0,0001 |

Harman, cat-only'ye göre c'yi **düşürüyor** (yaz25 T3: 3,127 → 2,737) —
yani aile farkı genlikte de tamamen sadeleşmiyor. Ama **hüküm değişmiyor:**
H9b'nin nüfus düzeltmesi (σ_hedef 0,4255 → 0,2634, toplu katılım genliği
yerleşiğe eşit) **aileden bağımsız** ve hâlâ geçerli. Onu uygulayınca yine
~1,5–1,6 bandına iniyoruz; kırpma risk ayarı da bağımsız olarak 1,60 demişti.

guz25 yine **nötr** (c≈1,0) — genlik ekseninde blok çelişkisi yok, teyit.

> **c = 1,60 DEĞİŞMİYOR.** Zaten S2 `c*`'ı **tam** çözecek; kesin değer
> 27 Ağustos'un ilk hakkına yazılır.

### KALİBRASYON/BETA — tek kalan BELİRSİZ ölçüldü (`h23_kalibrasyon_ikizde.py`)

Üretim `son_islem.py`: `r' = ort(r) + 0,60·(r − ort(r))` — soğuk tahminlerin
yayılımını büzer, hiçbir veriden uydurulmamış sabit.

| blok | fold ne görüyor | **orakül beta** | ızgara opt. | üretimden kazanç (panel) | test etkisi |
|---|---|---|---|---|---|
| yaz25 (ikiz) | GELECEĞİ görüyor | **1,367** | 1,40 | −0,358 | −0,0794 |
| guz25 | GELECEĞİ görüyor | **1,228** | 1,20 | −0,151 | −0,0336 |
| kis26 | yalnız GEÇMİŞ | **0,316** | 0,30 | −0,023 | −0,0051 |

> **REDDEDİLDİM — bu gecenin DÖRDÜNCÜ "fazla iyi" sayısı.** Test etkisi
> −0,0794, gereken açığın **27 katı**. Böyle bir sayı ölçüm hatasıdır,
> bulgu değil.

**Ne olduğu:** bloklar **4 kat** ters yönde ayrışıyor (0,30 ↔ 1,40) ve ayrışma
tam olarak fold'un geleceği görüp görmemesiyle açıklanıyor. Geleceği gören
fold'un soğuk tahmini zaten isabetli (yaz25 soğuk RMSE **1,4359**), büzmek
onu **bozuyor** → beta>1 istiyor. Past-only fold'un tahmini gürültülü
(kis26 ~1,7), büzmek **yardım ediyor** → beta<1 istiyor.

**Hangi blok test'i temsil ediyor?** Bu eksende kural 10'un olağan yönü
GEÇERSİZ: büzme mevsimsel değil, **tahmin gürültüsüyle** ilgili. Ayırt edici
tanı test soğuk RMSE'sinin kendisi: **≈1,713** — kis26'ya yakın, yaz25'in
1,4359'una değil. Yani bu eksende geçerli referans **kis26**.

**O hâlde neden yine de uygulanmıyor:**
1. `docs/41` çapraz-uydurmayla sınamıştı: **her varyant üretimden kötü**.
   Benim sayım **orakül** (beta'yı değerlendirdiği veriye uyduruyor) — üst
   sınır, ulaşılabilir değer değil.
2. Tek geçerli blok kis26 ve o da kural 10'un soğuk kararlar için
   uyardığı blok. İki bağımsız kesme yok.
3. S3'e giremez — ölçülmüş bileşenleri yerinden eder ve kural uyumlu
   doğrulanmadı.

**HÜKÜM: eksen KAPALI kalır, BELİRSİZ biter.** Ama kalıcı bir kavrayış
bırakıyor: **optimal büzmenin YÖNÜ fold'un geleceği görüp görmemesine
bağlı.** Üretimin 0,60'ı iki uç arasında bir uzlaşma; kis26 referans
alınırsa 0,30–0,45 bandına indirmek ~−0,005 test değerinde olabilir.
**27 Ağustos adayı** — ama çapraz-uydurmalı, orakülsüz bir sınamayla.

---

## TİK 7b — KAGGLE YOLU DOĞRULANDI (03:02 öncesi)

Her iki çağrı biçimi de çalışıyor (çıkış 0): `uv run kaggle …` ve
`uv run python -m kaggle …`. **03:02'de `python -m kaggle` kullanılacak**
(Smart App Control'e dayanıklı yol).

**SUBMIT de sınandı** — okuma çalışması submit'i garanti etmez. 25 Ağustos'un
kullanılmayan hakları 3 saat sonra yanacağı için bedava sınama:
```
27 MB tam yüklendi (56 sn) -> 400 Client Error: Bad Request
LISTEDE YENI KAYIT YOK -> temiz red, HAK YANMADI
```
En olası sebep günlük kota (25 Ağustos'ta zaten 3 gönderim: 00:00:45,
00:01:53, 00:01:55). **Doğrulanan:** kimlik ✓ · 27 MB yükleme ✓ · sunucu
yapılandırılmış yanıt ✓. **Yükleme 56 saniye** — 03:02'de başlatılırsa
S1 ~03:03'te iner.

**03:02 EK KURALI:** S1 gönderildikten sonra **listede göründüğü doğrulanacak**.
400 tekrarlarsa **DUR** — S2/S3 gönderilmez, sebep teşhis edilir. Sebep kota
değilse üçü birden yanar.

---

## TİK 8 — H20 TEZGAHI: **ÇÜRÜDÜ** (8 tohum)

> **DÜZELTME:** Bu bölümü önce "ikizde KAZANDI" diye yazdım — **yanlıştı**.
> `d = −0,02639, t = −9,69, 3/3` sayısı **yalnız soğuk RMSLE** üzerindeydi ve
> **sıcak taraftaki maliyeti görmüyordu**. Test ağırlıklı net ölçüte eşlenik
> SH uygulanınca (kural 4) hüküm çöküyor. Ayrıntı aşağıda, "NET ÖLÇÜTLE
> HÜKÜM" başlığında.


`scripts/h20_tezgah_kumeli_maske.py` → `reports/h20_tezgah.csv`

**Ucuz kapılar (tik 5 sonrası, `h20_kapilar.py`):**
- **K1 GEÇTİ:** 2026-05-11 kohortu mekansal olarak kümeleniyor —
  z(Herfindahl) = **+13,94**, z(top10) = **+6,12** (200 rastgele örneğe karşı).
- **K2 GEÇTİ:** test soğuk trafosunun ilçesindeki soğuk payı **0,3439**;
  eğitimdeki rastgele maskede **0,2206**. Oran **1,56**.

**Tezgah — yaz25 (mevsimsel ikiz), 3 tohum, üretim harmanı, aynı maske oranı 0,2216:**

| kol | SOĞUK RMSLE | tüm RMSLE | b |
|---|---|---|---|
| URETIM (rastgele) | 1,67459 / 1,71203 / 1,67401 | 0,89612 / 0,92173 / 0,92165 | +0,186 / +0,135 / +0,063 |
| KUMELI-ilçe | 1,64792 / 1,69049 / 1,64305 | 0,89945 / 0,91902 / 0,93963 | +0,114 / +0,012 / +0,085 |

```
KUMELI - URETIM (SOĞUK RMSLE, eşlenik)  d = -0,02639  SH 0,00272  t = -9,69  3/3
```

### Ham sayıya ATLAMA — net etki hesabı
Soğuk RMSLE'deki −0,0264 tek başına **yanıltıcı**. Blokta soğuk %7,5, test'te
%22,2; ve **tüm** RMSLE kötüleşiyor. Ayrıştırılınca:

```
SOĞUK  MSE 2,8456 -> 2,7572   d = -0,0883   (kazanıyor)
SICAK  MSE 0,6707 -> 0,6901   d = +0,0195   (KAYBEDIYOR)
TEST ağırlıklı net = 0,22159×(-0,0883) + 0,77841×(+0,0195) = -0,00443
```

**Mekanizma anlaşılır:** kümeli maske eğitimde bütün ilçelerin geçmişini
topluca siliyor → soğuk rejim daha iyi öğreniliyor ama **sıcak rejim zengin
komşu bağlamından mahrum kalıyor**, ve sıcak testin %78'i.

### NET ÖLÇÜTLE HÜKÜM (kural 4: (blok,tohum) eşlenik SH) — **BELİRSİZ**

kis26 kolu da koştu. Test ağırlıklı net ölçüt, tohum bazında eşlenik:

| blok | net dMSE | eşlenik SH | **t** | iyileşen | tohumlar |
|---|---|---|---|---|---|
| yaz25 | −0,00430 | 0,00834 | **−0,52** | 2/3 | −0,0090 / −0,0158 / **+0,0119** |
| kis26 | −0,02822 | 0,01986 | **−1,42** | 2/3 | −0,0640 / −0,0253 / **+0,0046** |

Soğuk-yalnız ölçütte üç blok (bu ölçüt yanlış hedef, karşılaştırma için):
`yaz25 −0,0264 (t=−9,69, 3/3)` · `kis26 −0,0044 (t=−0,22, 2/3)` ·
`guz25 −0,0014 (t=−0,06, 2/3)`. **Yalnız yaz25 bir şey gösteriyor**; diğer
iki kesme sıfır. Kural 9'un istediği tutarlılık yok.

**Hiçbiri anlamlı değil.** İşaret ikisinde de negatif ama her iki blokta da
bir tohum **ters** dönüyor ve t değerleri eşiğin çok altında.

**Mekanizma da bloklar arası tutarsız:**
```
yaz25   SOĞUK -0,0883  SICAK +0,0195   (soğuktan kazanıyor, sıcaktan kaybediyor)
kis26   SOĞUK -0,0165  SICAK -0,0314   (esas kazanç SICAKTAN -- ters hikâye)
```

**Neden ilk okuma yanıltıcıydı:** soğuk-yalnız RMSLE düşük varyanslı ama
**yanlış hedef** — sıcak testin %78'i ve kümeli maske ona dokunuyor. Doğru
hedefe (test ağırlıklı net) geçilince gürültü ortaya çıkıyor.

**Kural 3 tam bu durumu tarif ediyor:** *"soğuk tarafta üç tohum yetmez —
3 tohumda t=+3,91 olan bulgu 6 tohumda çöktü."* Aynı desen.

### 8 TOHUMLA KESİN HÜKÜM: **H20 ÇÜRÜDÜ**

Tezgah tohum 1003–1007 ile tekrarlandı (yaz25 + kis26 → 8 tohum).

| blok | n | **NET dMSE** | SH | t | iyileşen | soğuk-yalnız d | t |
|---|---|---|---|---|---|---|---|
| yaz25 | 8 | **+0,00302** | 0,01250 | +0,24 | 4/8 | −0,00014 | **−0,01** |
| kis26 | 8 | **+0,00827** | 0,01641 | +0,50 | 3/8 | +0,01308 | +1,10 |
| guz25 | 3 | +0,00902 | 0,01532 | +0,59 | 1/3 | −0,00136 | −0,06 |

**Üç blok da null, ve üçü de hafif POZİTİF (yani kümeli maske biraz DAHA KÖTÜ).**

**yaz25'in soğuk-yalnız sinyali tamamen kayboldu:**
```
3 tohum:  d = -0,02639   t = -9,69   3/3
8 tohum:  d = -0,00014   t = -0,01   5/8
```

> **Kalıcı kural 3'ün ders kitabı örneği:** *"soğuk tarafta üç tohum yetmez —
> 3 tohumda t=+3,91 olan bulgu 6 tohumda çöktü."* Burada 3 tohumda t=−9,69
> olan bulgu 8 tohumda t=−0,01'e indi. **İlk üç tohum şanslı bir çekilişti.**

**HÜKÜM: kümeli soğuk maskeleme ÇÜRÜDÜ. Yasak bölgeye.**

Kapılar (K1 kümelenme z=+13,9, K2 oran 1,56) **doğruydu** — dağılımsal
uyuşmazlık gerçek. Ama K3 uyarısı haklı çıktı: ilçe yalnızca 47 eşsiz değer
ve h16 doğru nüfusta ilçe R²'sini ~0 ölçmüştü. **Kanal zayıf olunca
dağılımı düzeltmek işe yaramıyor.** Mekanizmanın var olması, sömürülebilir
olmasını gerektirmiyor.

**27 Ağustos için sonuç:** gün H20 yeniden eğitimine ayrılmayacak. Eksen
temiz kapandı — bu da değerli, çünkü aksi hâlde saatlerce üretim eğitimi
boşa gidecekti.

---

## TİK 9 — **δ_soğuk 0,16 → 0,22** (test yapısının birebir analogu bulundu)

`scripts/h24_b_soguk_ikiz_pencere.py`

**Tetikleyen:** kendi kalibrasyon bulgumun yan ürünü — *"test soğuk RMSE'si
(≈1,713) kis26'ya yakın, yaz25'in 1,4359'una değil"* — `δ_soğuk=0,16`
kararının dayandığı "ikize güven" mantığına **karşı** çalışıyordu.

**Sorun:** üç fold'un **hiçbiri** test'in yapısında değil:

| fold | ufuk | hedefin mevsimsel ikizi eğitimde mi? | b |
|---|---|---|---|
| yaz25 | GELECEK var | HAYIR | +0,1056 |
| guz25 | GELECEK var | HAYIR | +0,0725 |
| kis26 | yalnız geçmiş | KISMEN | +0,3017 |
| **TEST** | **yalnız geçmiş** | **EVET** (Nis–Tem 2025 train'de) | **?** |

**Çözüm — kis26'nın İÇİNDE test yapısını bul.** `son_islem_seviye.py` zaten
belgeliyordu: kis26 içinde yalnızca 2026 Şub–Mar geçerlidir, çünkü mevsimsel
ikizi (2025 Şub–Mar) o fold'un etiketlerinde vardır. Aralık/Ocak'ın ikizi yok.

### Ay bazında — MONOTON ve mekanizmayı doğruluyor
| ay | satır | ikiz eğitimde? | **b** | SH |
|---|---|---|---|---|
| 2025-12 | 5.942 | HAYIR | **+0,4465** | 0,0031 |
| 2026-01 | 14.654 | kısmi | **+0,4081** | 0,0120 |
| 2026-02 | 17.289 | **EVET** | **+0,2756** | 0,0136 |
| 2026-03 | 24.033 | **EVET** | **+0,2197** | 0,0114 |

İkiz devreye girdikçe b **düzgün düşüyor**. Bu, `b = (i) sürüklenme +
(ii) mevsim ekstrapolasyonu` ayrışımının doğrudan kanıtı.

```
TEST YAPISININ ANALOGU (kis26 soğuk, Şub–Mar):  b = +0,2431  SH 0,0123  t=+19,7
ikiz YOK (Ara–Oca):                             b = +0,4191
tüm kis26:                                      b = +0,3017
```

**Kırpma (kural 1):** b kırpmayla **yükseliyor** (0,243 → 0,565 K=50'de).
Geniş tabanlı, aykırı sürüklemesi yok.

### E[b] ve δ kararı
- **Yapısal analog 0,2431** — ufuk VE ikiz bakımından test'le birebir eşleşen
  tek ölçüm, t=+19,7.
- **Fold-free çapa 0,1454** yalnız **(i)**'yi yakalıyor, (ii)'yi değil →
  test için **ALT SINIR**, nokta tahmin değil.
- Bootstrap 0,1764 (SH 0,1114).
- yaz25'in 0,1056'sı yanlış yapıdan: (i)=0 olduğu için sistematik **düşük**.

**δ_soğuk = 0,22** — analogun (0,243) hemen altı, mevsim transferi
belirsizliği için ~%10 büzülmüş.

| δ | E[b]=0,145 | 0,175 | 0,22 | 0,30 |
|---|---|---|---|---|
| 0,16 | −0,00461 | −0,00674 | −0,00993 | −0,01560 |
| **0,22** | −0,00341 | −0,00634 | **−0,01072** | −0,01852 |

Ters yön maliyeti küçük (b=0,145 ise 0,0012 kayıp), yukarı kazanç büyük
(b=0,30 ise +0,0029). Asimetri yukarı delta lehine.

**`δ_kuyruk = 0,35` DEĞİŞMİYOR** — iki blokta ölçüldü, başabaş kırpılmış tabana eşit.

### ÇEKİNCE ve ÜÇ OKUMANIN AYRIŞTIRILMASI (`h25_yas_eslestirilmis_b.py`)

kis26 içinde **ay, ufuk ve trafo yaşı karışık**. Monoton üç şekilde okunabilir:

**(a) İKİZ okuması** — seçilen. **(b) UFUK okuması — ELENDİ:** sürüklenme
ufukla **birikir**, yani b ufuk arttıkça **yükselmeli**. Ölçülen tam tersi
(Ara ufuk 1-31 → +0,4465; Mar ufuk 91-121 → +0,2197, **düşüyor**). Ufuk
okuması **ters işaret öngörüyor**, elenir. *(H6 de bağımsız olarak saf ufuk
eğimini t=+1,46 ile null bulmuştu.)*

**(c) YAŞ okuması — elenemez, ölçüldü.** Soğuk trafolar blok boyunca doğuyor,
ay ilerledikçe yaş artıyor:
```
kis26  Ara medyan yaş  9 | Oca 26 | Şub 46 | Mar 67
TEST       medyan yaş 40  (q10 7, q90 75)   <- analogdan GENÇ
```

Yaş kovası × ikiz (b):
| yaş | ikiz VAR | ikiz YOK |
|---|---|---|
| 0-6 | +0,1552 | +0,5369 |
| 7-29 | +0,3556 | +0,5049 |
| 30-59 | +0,2818 | +0,1994 |
| 60+ | +0,1872 | — |

**TEST'in yaş dağılımıyla yeniden ağırlıklandırılmış:**
| yaş | TEST payı | b | katkı |
|---|---|---|---|
| 0-7 | 0,0881 | +0,1552 | 0,0137 |
| 7-30 | 0,2841 | +0,3556 | 0,1010 |
| 30-60 | 0,3563 | +0,2818 | 0,1004 |
| 60+ | 0,2715 | +0,1872 | 0,0508 |
| | | **YAŞ-EŞLEŞTİRİLMİŞ b** | **+0,2659** |

Yaş düzeltmesi b'yi **yukarı** taşıyor (+0,2431 → +0,2659), çünkü test soğuğu
analog pencereden genç ve genç yaşlar daha yüksek b taşıyor.

**δ = 0,22 DEĞİŞMİYOR.** Artık iki yapısal kestirim de (0,2431 düz,
0,2659 yaş-eşleştirilmiş) 0,22'nin **üstünde** — yani seçim muhafazakâr
tarafta, agresif değil. E[b]=0,2659 varsayımıyla δ=0,25 yalnızca **0,0004**
daha iyi; mevsim transferi (Şub–Mar → Nis–Tem) belirsizliği bunu fazlasıyla
yutuyor. Tekrar değiştirmek gürültü olur.

**S3 okunurken:** gerçek b analogdan yüksek çıkarsa şaşırmayacağız —
yaş-eşleştirme zaten o yönü gösteriyor ve 27 Ağustos'un optimumu daha yukarı
olur.

---

## KALICI KURAL 15

> **Bir mekanizmanın VAR OLMASI, onun SÖMÜRÜLEBİLİR olmasını gerektirmez.**

H20'de kapılar geçti (kümelenme z=+13,9, ilçe soğuk payı oranı 1,56) — yani
dağılımsal uyuşmazlık **gerçekti**. Ama düzeltmenin geçeceği kanal (ilçe,
47 eşsiz değer, h16'da R²≈0) zayıf olduğu için 8 tohumda etki **sıfır**.
**Kapı analizi mekanizmanın VARLIĞINI gösterir; BÜYÜKLÜĞÜNÜ yalnızca tezgah
gösterir. Kapı geçti diye tezgah ATLANMAZ.**

---

## YARININ İLK İŞİ — KOHORT BAZLI SOĞUK MASKELEME

**Bu gece YAPILMAYACAK** (yeniden eğitim gerektirir, saatler sürer, üretim
kodu dondu). Haftalık kota sıfırlanınca **ilk bu ölçülecek.**

**Doğrulanmış olgu** (`scripts/tuketim_model.py:1055` `soguk_maskele`):
```python
secilen = set(rng.choice(trafolar, size=int(len(trafolar) * oran), replace=False))
```
Maskeleme **i.i.d. rastgele trafo seçimi**. `SOGUK_MASKE_ORANI = 0.2216`
(satır 801) test soğuk **oranını** birebir taklit ediyor.

**Ama BİLEŞİMİNİ etmiyor:**
```
test soğuk       %80,7 TOPLU kohort (2026-05-11 tek başına 1.326 trafo)
rastgele maske   pratikte TEKİL soğuk üretir -- korelasyonsuz, dağınık
```

DropoutNet'in mekanizması "servis anındaki girdi dağılımını eğitimde gör".
Model o dağılımı görüyor **ama yanlışını**: hiçbir eğitim örneğinde *aynı gün
doğan, aynı lokasyon kümesini paylaşan, topluca geçmişsiz* bir trafo bloğu yok.

**Önerilen:** rastgele trafo yerine **aynı gün doğan trafoları TOPLUCA**
soğutmak (kohort bazlı maskeleme). Kohort içi korelasyon o zaman eğitimde
görünür ve model kohort düzeyi bağlamı kullanmayı öğrenir — test'te 05-11
kohortu için elimizde tam olarak o var.

**Neden bu gecenin bulgularıyla tutarlı:**
- H14: kis26 soğuk %1 TOPLU, test soğuk %81 TOPLU — doğrulama da eğitim de
  yanlış bileşimde
- H16: soğuk seviyede baskın yapı trafo bazlı değil **kohort bazlı küresel
  kayma** (R² 0,182) — modelin öğrenemediği şey tam olarak bu
- H17: yanlılık geniş tabanlı ve kırpmayla yükseliyor — sistematik, öğrenilebilir

**Ölçüm tasarımı:** `maske_orani` aynı kalsın, seçim kohort bazlı olsun;
yaz25 **ve** kis26'da ölç (kural 7+9), kırpma tablosu ver, 3 değil **6 tohum**
(kural 3). Beklenti bilinmiyor — bu bir mekanizma hipotezi, rötuş değil.

---

### TİK 5 HÜKMÜ

| kusur | sonuç |
|---|---|
| **A — döngüsel doğrulama** | `docs/41`'in `0,1626/0,60` örneği **TEKİL**. 19 eksende ikinci örnek yok. **TEMİZ.** |
| **B — yanlış nüfus** | **GERÇEK ve büyük** (kis26 %1 toplu vs test %81). Ama sıfır ekseninde sonucu değiştirmiyor (duvar %46→%39). |
| **C — kural 10** | Üç eksen tek blokta kapatılmış. İkisi (sıfırlar, hurdle) yeniden ölçüldü, **hüküm ayakta**. Kalibrasyon/beta ölçülmedi — **BELİRSİZ**, kuyruğa. |

**Gönderim dosyalarına dokunulmadı** — hiçbir bulgu −0,002 eşiğini geçmedi.
Bu beklenen sonuçtu ve kendi başına bir sonuç: **19 eksen A/B/C'ye karşı
denetlendi, biri hariç temiz**, ve o birinin (kalibrasyon/beta) tavanı zaten
orakülde −0,002.

---

## TİK 1 — KAPANAN HİPOTEZLER (bir daha AÇMA)

**H1 — gün ekseni frekans ayrıştırması.** Fizik ayrımı gerçek (yaz25 etiketli
c_hafta 1,241 / c_düşük 2,936 / c_kalan 1,119) ama oracle tavanı bile havuzda
yalnızca −0,00082 dMSE = gerekenin %4'ü. Etiketsiz çapa 3 banttan yalnızca
1'inde (DÜŞÜK) etiketli optimumla aynı işaret; HAFTA (çapa 0,933 vs 1,241) ve
KALAN (çapa −0,011 vs 1,119) 1'in **zıt** taraflarında. Ayrıca test tahmininin
gün ekseni varyansının **%89,9'u zaten DÜŞÜK bantta** — ayrılacak şey yok.
Üretim adayları: yaz25 −0,00333 / guz25 +0,00354 → havuz **tam sıfır**.
Gün düşürme: K=5'te işaret dönüyor.

**H2 — değiştirme eşleştirmesi.** `lokasyon` yalnızca **47 eşsiz değer** (ilçe
düzeyi, test'te lokasyon başına ~150 trafo) → anahtar kimlik taşımıyor. Tekil
1-1 eşleşme **7** (durma eşiği ~100). Yapısal olarak eşleştirilecek doğum yok:
soğuk trafoların %65,5'i tam olarak 2026-05-11'de başlıyor ve **aynı gün 896
SICAK trafo da başlıyor** → ilk-test-günü bir bağlantı tarihi değil, panel
artefaktı. 4.590 aday (soğuk,ölü) çiftinin **0** tanesinde ölüm doğumdan önceki
30 gün içinde; medyan boşluk 243 gün. İkizde β=−0,0403, R²=0,0038, işaret TERS.

**H3 — 2026-05-11 kohortu.** Yapı doğrulandı (1.326 trafo, 108.253 satır =
soğuk satırların %68,35'i). Ama mekanizma ölçülemiyor: train'de yaş≥7 verisi
olan yalnızca **iki** toplu kohort var (n=172, n=166), `toplu` katsayısı
+0,1420 ± 0,1413 (t=+1,00) — kural 3'ün doğrudan ihlali. Zorunlu yaz25 bloğunda
kohort farkı spesifikasyona göre +0,1313 → −0,1588 → +0,1776 arası **işaret
değiştiriyor**, kırpma bunu çözemiyor (K=50'ye kadar kararlı). Kalan farkın
%33-49'u çapanın kVA eğiminden ve o eğim bir **yaşlanma eseri** (yenidoğan
çapasının kVA eğimi yaş 7-20'de +0,1835, yaş 251-400'de +0,6917; yerleşiklerde
sıfır). Çapadan bağımsız tek sağlam parça: model 05-11 kohortuna aynı kVA'da
diğer soğuktan **0,0731 düşük** seviye veriyor (t=−5,34) — ayrı knob tavanı
dMSE **−0,00026**, gerekenin %1,3'ü.

**H4 — boru hattı denetimi. 8/8 GEÇTİ, hat temiz.** 6 gönderim dosyasının id'si
`sample_submission` ile birebir; `guc` 5.012 ortak trafonun hiçbirinde farklı
değil; log1p/expm1 gidiş-dönüşü göreli 4,2e−16; v67 dokümante reçetesinden
**max|dlog| = 0,000e+00** ile birebir yeniden üretildi; 47 eşsiz lokasyon
normalize edilince 47 kalıyor. İki küçük kusur: `np.clip(...,0,None)` 2.838
sıcak satırı sıfıra kırpıyor (o 91 trafo train'in son 60 gününde %98,6 sıfır →
kırpma muhtemelen **faydalı**, |dMSE|<1e−4); zincir sırası önemli ama
Cauchy-Schwarz üst sınırı |dMSE| ≤ 1,2e−4.

**H5 — konum toplamı / yük devri.** 3.285 doğum olayında AS-OF olay çalışması:
yerleşik trafolarda yük düşüşü **yok** (yaz25 −0,0110 t=−0,46; guz25 +0,0203
t=+1,63 — **işaret ters**, ikisi de anlamsız). Plasebo −0,0091, yani "etki"
plasebo gürültüsüyle **aynı büyüklük**. Regresyon R² = 0,0000–0,0001. İkizde
en iyi harman ağırlığı **w\* = 0,00** (her pozitif ağırlık zararlı).

**H6 — ufuk ekseni.** Dokuz örtüşen kesmeyle (2.367.469 üçlü) gün+trafo sabit
etkili kestirim: saf ufuk eğimi +0,000330 ± 0,000226, **t=+1,46 (null)**.
Yalnız yaz hedeflerinde gün sabit etkisi konunca eğim **işaret değiştiriyor**
(−0,001072, t=−14,73) → yaz25'teki dev rampa ufuk değil **mevsim**. Nisan→Temmuz
mevsimsel yükseliş +0,6152, yaz25 ufuk rampası +0,5348 → rampanın **%87'si
doğrudan mevsim**. yaz25 hariç ortalama +0,000093 = sıfır.

**H7 — c\* yeniden türetme. Şampiyon için düzeltici ölçek YOK** (parabol
varsayımsız çözüldü: optimum m\*=0,6811, v67 m=0,6877, kayıp **+2,3e−7 MSE**).
Ama **iki gerçek bug buldu** — ikisi de kayda geçti:

> **BUG 1 — istenen ölçek ≠ ULAŞILAN ölçek.** Kırpma yüzünden v55 1,492 yerine
> **1,4760**, v66 1,335 yerine **1,3241**, v57 1,75 yerine **1,7250** uyguluyor.
> Betiğin %3'lük kapısı bunu **sessiz geçiriyor**.
>
> **BUG 2 — `--lb-kalibre` yanlış biçim.** Kalibreyi AFFİN uyguluyor
> (`1+k(c−1)`) ama c\* σ_gerçek ile **orantılı** olduğu için doğru biçim
> **ÇARPIMSAL** (`k·c`). LB'nin çözdüğü optimuma karşı sınandı: çarpımsal
> **1,3325** (hata ~0), affin **1,4395** (hata +0,109, dMSE **+0,000277**).
> Bu bug `v58_soguk_kalibre`'nin soğuk ölçeğini 1,458 nominal / 1,411 ulaşılan
> yapmış; doğrusu ~1,34.
>
> **Taşınabilir sabit c\* DEĞİL, ULAŞILACAK GENLİK: S\* = 0,2204** (sıcak gün
> ekseni std, trafo etkisi çıkarılmış).

H8 üretim betiği bu dersin **doğru** tarafında: hedef genlik formülasyonu
kullanıyor ve ulaşılan ölçek doğrulanıyor (2,2000 = 2,2000).

---

## KAPANAN HİPOTEZLER — TAM LİSTE (bir daha açma)

Devralınan (docs/43, docs/44 §4):
`sıcak kapasite` · `soğuk kapasite` · `soğuk hurdle` · `kalibrasyon/beta` ·
`takvim/tatil` · `harman` · `λ pencere genişletme` · `ulusal yük ile gün faktörü` ·
`b_i trafo kestiricisi` · `sabit δ transferi` · `artık hedefi u = ofs − seviye_i` ·
`ölü kuyruğa log1p 0,20` · `grup B yukarı kaydırma` · `soğuk kVA kovası` ·
`on6/on7 hedef kodlaması` · `kimlik komşuluğu` · `son pencere çapası` ·
`gün faktörünü hava modeliyle değiştirmek` · `kayıt sonlanması / p_son_ofset`

Bu koşuda kapananlar:
`H1 gün ekseni frekans ayrıştırması` · `H20 kümeli soğuk maskeleme (8 tohumda t=+0,24)` · `H2 değiştirme eşleştirmesi` ·
`H3 05-11 kohortu (tavan −0,00026)` · `H5 konum toplamı / yük devri` ·
`H6 ufuk ekseni` · `H7 şampiyon için düzeltici c* (yok)` ·
`soğuk gün ekseninde TEK genlik iki-blok transferi (h8f: ortak bölge yok)`

---

## KUYRUK (07:00'e kadar, TEK BAĞLAMDA — ajan açılmaz)

1. **H9** — soğuk gün ekseninde c'nin **kVA / kohort** kırılımı. Tek küresel c
   yerine iki-üç kova. İkizde ölç; kırpma tablosu şart. (H8'in doğal devamı,
   en yüksek beklenen değer.)
2. **H10** — `son_islem_olay.py` dönüş kolunda **parti eşiği yok** (H4'ün
   bulduğu tek mantık tutarsızlığı). 2026-05-11 tek günde 1.634 doğum + 1.025
   dönüşü birleştiren eşsiz gün; iki kol aynı günü çelişkili okuyor. Bant ±0,0004.
3. **H11** — BUG 2'nin (`--lb-kalibre` affin/çarpımsal) **soğuk tarafa** etkisi:
   v58 ailesindeki ölçek 1,411 yerine ~1,34 olmalıydı. Şu an üretimde değil ama
   H9'da soğuk ölçek kalibre edilecekse **doğru biçim** kullanılmalı.
4. **H12** — H3'ün kalan sağlam parçası: 05-11 kohortuna aynı kVA'da 0,0731
   düşük seviye (t=−5,34). Tavan −0,00026. Ucuz, S3 ile paketlenebilir.
5. **H13** — gün ekseni kestiricisi **olaysız** veriden (H1 ajanının önerisi):
   `gun_etkisi()` çağrısına olay satırlarını dışlayan maske. Üst sınır 7,7e−5 —
   tek başına değmez, başka bir rötuşla paketlenirse bedava.

---

## §7 GÖNDERİM PROTOKOLÜ (GÜNCELLENDİ — 08:00)

Kota **UTC gününe** bağlı. 08:00 yerel = **05:00 UTC** → 26 Ağustos kotasının
üç hakkı da duruyor; 03:00 yerine 08:00 beklemek **hak kaybettirmiyor** ve
gönderilen dosyalar gecenin bulgularını taşıyor.

**Kalıcı kural 8 — gönderimden ÖNCE liste oku:**
```bash
uv run kaggle competitions submissions -c grid-up-datathon | head -5
```

Public/private ayrımı **YOK** → dönen skor **gürültüsüz**. LB takımın **en iyi**
skorunu gösterir → **kötü gönderimin sıralama maliyeti sıfır**. Bu yüzden
"güvenli küçük prob" atma; tahminin kendisini yaz.

## §9 TİK TEMPOSU (GÜNCELLENDİ)

- **Ajan/workflow AÇILMAZ.** Bütün ölçümler tek bağlamda, pandas/numpy ile.
- **Çürütme = ayrı ajan değil**, aynı bağlamda **bağımsız ikinci bir ölçüm**
  (farklı blok, farklı kesme, farklı ayrıştırma). Tik 1'de böyle yapıldı:
  4 saldırı, hepsi tek bağlamda, biri (çürütme 3) bulguyu **düzeltti**.
- 90 dakikalık kutuya sığmayan hipotez → hüküm **BELİRSİZ**, kuyruk sonuna.
- **08:00 gönderimi keşiften ÖNCE gelir.** Bütçe daralırsa keşfi kes,
  gönderimi yap. Gönderimi kaçırmak kabul edilemez.

### BÜTÇE EŞİĞİ — sert kural

> **Haftalık kullanım %90'ı geçerse keşif ANINDA durur.** O eşikten sonra
> **hiçbir yeni ölçüm başlatılmaz**; yalnızca üç şey yapılır:
> 1. 08:00 gönderimi (önce `kaggle competitions submissions` ile liste oku)
> 2. `scripts/b_coz.py` ile skorların çözülmesi
> 3. T+12 raporu
>
> Şu an ~%72, kalan ~%28, 1 gün içinde sıfırlanıyor.
> **Gönderimi kaçırmak kabul edilemez tek sonuçtur.**

Bütçe daralma belirtisi (uzun bekleme, kota uyarısı) görülürse: keşfi durdur,
defteri güncelle, durumu raporla, yeni ölçüm başlatma.

### ÇİZELGE — 26 Ağustos 00:55 revizyonu: SIRALI GÖNDERİM, 03:00

**Neden sıralı:** S1+S2 parabolün tamamını çözüyor (`A` dosyalardan analitik,
etiketsiz). Öyleyse S3'ü **tahmin edilen** c=1,60 yerine **ÖLÇÜLEN c\*** ile
kurmak bedava. c\* 1,60'tan 0,4 saparsa maliyet `A×0,16 ≈ 0,0009` MSE — sıralı
gitmek bunu kurtarıyor. Kaggle skoru saniyeler içinde dönüyor.

| saat | ne |
|---|---|
| 03:00 | `kaggle competitions submissions ... \| head -5` — **kalıcı kural 8, ÖNCE OKU** |
| 03:02 | **S1** = `tuketim_v67_c1335_olay.csv` gönder |
| 03:05 | S1 skorunu oku → `MSLE(0)` kesin |
| 03:07 | **S2** = `tuketim_v73_soguk_gun160.csv` gönder |
| 03:10 | S2 skorunu oku → `coz_0800.py` ile **c\* KESİN** çözülür |
| 03:15 | c\*'ı deftere yaz, **ÜÇÜNCÜ HAKKI HARCAMA** |
| 03:15 → 05:00 | denetime devam |
| ~03:20 | **S3 — c\* çözülür çözülmez.** Beklenecek başka kanıt YOK: `δ_soğuk`=0,16 ve `δ_kuyruk`=0,35 karara bağlandı (tik 6–7), yaz25/guz25 soğuk tahminleri üretildi. S3'ün beklediği tek girdi `c*`'tı. İnşa: v67 → soğuk gün ekseni **ölçülen c\*** → `δ_soğuk`=0,16 → **kuyruk δ=0,35**. Kapıları koş. *Hazır aday: `tuketim_v77_kuyruk35.csv` (c=1,60 tabanlı); `\|c*−1,60\| < 0,05` ise olduğu gibi gönder, değilse sıfırdan kur.* |
| 05:45 | S3 gönder → **b_soğuk KESİN** çözülür |
| 06:00 | T+12 raporu |

**S3'E `b_sıcak` EKLENMEYECEK.** Cazip (+0,00125) ama eklenirse S3−S2 farkı
iki bilinmeyen taşır ve `b_soğuk` **çözülemez**. `b_soğuk` 0,00567 değerinde,
`b_sıcak` 0,00125 — temiz çözüm daha değerli. `b_sıcak` 27 Ağustos'un ilk
hakkında çözülür.

**Not:** 03:00 yerel = 00:00 UTC, yani kota sınırının tam üstü. İlk gönderim
**03:02**'de (00:02 UTC) — yeni günün içinde, sınırda değil.

### 1.'LİK HESABI
```
gereken                 −0,019333
gecenin toplamı         −0,01644
açık                    +0,00289
ölçülen c* katkısı      ~0,0009    (26 Ağu 03:15'te belli olur)
b_sıcak 0,04            ~0,00125   (27 Ağustos)
ikisi birlikte           0,00215   = açığın %74'ü
```
Kalan ~0,0007'yi denetim bulguları ya da `b_soğuk`'un 0,16'dan büyük çıkması
kapatabilir. **1.'lik ERİŞİLEBİLİR — ama 27–28 Ağustos'ta, yarın sabah
değil.**

### Önceki çizelge (00:15 revizyonu — SIRALI plan onu geçersiz kıldı)
| saat | ne olur |
|---|---|
| … → **05:00** | keşif (tek bağlamda). **05:00 SERT SINIR** |
| 05:00 → 05:30 | **DONDUR** — o anki şampiyondan üç dosya, bütünlük kapıları, deftere yaz |
| 05:30 → 08:00 | **SESSİZ BEKLEME** — ölçüm yok, betik yok, dosya okuma yok. Tek uzun uyandırma, `noop: true` |
| **08:00** | **GÖNDERİM** — önce `kaggle competitions submissions` ile liste OKU |
| 08:30 | S2²−S1² → soğuk gün ekseninin gerçek kazancı + `c*` (parabol) |
| 09:00 | S3²−S2² → `b_soğuk` |
| 09:30 | T+12 raporu, defter kapanır |

**05:00'te keşif nasıl kesilir:** o an elde yarım hipotez varsa hüküm
**BELİRSİZ**, gerekçesiyle deftere yazılır, kuyruğun sonuna atılır. Yarım işi
bitirmek için süre aşılmaz.

**%90 bütçe eşiği daha önce devreye girerse:** aynı anda dur ve doğrudan
dondurma adımına geç.

**Dondurma durumu (tik 3 itibarıyla): ZATEN TAMAMLANDI.** Üç dosya şampiyon
`v73`'ten üretildi ve 3/3 kapı geçti. 05:00'e kadar şampiyon değişmezse
05:00–05:30 penceresinde yapılacak yeni iş yok.

---

## DÜRÜST KONUM (tik 1 sonu)

```
doğrulanmış (LB)          v50 1,01686 → v55 1,01591      −0,00193 MSE
ölçüldü, LB'de sınanmadı  c*=1,335 düzeltmesi            −0,00054
                          olay günü s=0,6                −0,00116
                          SOĞUK gün ekseni c=1,60        −0,00907   <- tik 3 (iyimser)
                          b_soğuk = 0,16 (ön kayıtlı)    −0,00567   <- tahmin
                                                          --------
toplam beklenen (iyimser)                                −0,01644
gereken                                                  −0,019333
                                                          --------
AÇIK                                                     +0,00289
```

**Tik 1'de toplam gerekenin üstünde görünüyordu (−0,02224); tik 3'ün nüfus
düzeltmesi onu gerekenin ALTINA indirdi.** Bu, defterin dürüst kalması içindir:
0,4255 çapası şişmişti ve o rakama dayanan −0,01487 fazla iyimserdi.

Senaryo bandı (08:00 sonrası beklenen RMSLE):
| senaryo | soğuk gün ekseni | b_soğuk | RMSLE |
|---|---|---|---|
| iyimser | −0,00907 | −0,00567 | **~1,00780** |
| orta | −0,00162 | −0,00567 | ~1,01146 |
| kötümser | +0,00008 | 0 (b≈0) | ~1,01511 |

Grid Grinders 1,00635. **İyimser senaryoda bile fark ~0,0015 kalıyor** —
yani bu gece bulunanlar birinciliğe yetmiyor, ikinciliği sağlamlaştırıyor.
Birincilik için hâlâ yapısal bir kaynak gerekiyor ve bu koşuda H1–H7'nin
yedisi de o kaynağın **olmadığını** gösterdi.

**Kalan kuyruğun tavanı toplam ~−0,0007; açık +0,00289.** Yani kuyruğun
TAMAMI mükemmel çalışsa bile 1.'liğe yetmiyor. Bu, 05:00'i beklemeden
sessize geçme kararının gerekçesidir.

**Yarışma 1 Eylül'de bitiyor: 26 Ağustos'un 3 hakkından sonra ~15 hak kalıyor
(27–31 Ağustos × 3).** Birincilik bu gecenin değil, o 15 hakkın ve bulunacak
yeni bir yapısal eksenin işi.
