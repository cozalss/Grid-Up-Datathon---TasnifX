# Ölçüt tamiri ve bayatlık ekseni — 24 Ağustos 2026 (10 saatlik loop)

**Compact sonrası okunacak ikinci belge.** İlki hâlâ
[39-loop-sonucu](39-loop-sonucu-2026-08-24.md) — üretim yapılandırması ve
kalıcı kurallar orada. Bu belge o günün akşamı yapılan ölçüt tamirini,
kapanan eksenleri ve loop sonrası tek gerçek fırsat kümesini kaydeder.

Başlangıç durumu: LB **1,01750**, birincilik, fark 0,00548. Hedef 1,00 altı.

---

## 1. Asıl iş: ölçütün kendisi yanlış karışımı ölçüyordu

Dün gece LB, `kis26` üzerinde iyileşen bir revizyonun bütününü +0,00414 ile
çürüttü. Teşhis doğru konmuştu — hata modelde değil **ölçütte** — ama ilaç
yalnızca soğuk tarafa yazılmıştı. Aynı hastalık sıcak tarafta duruyordu.

`scripts/olcut.py` doğrulama satırlarını testin ortak dağılımına taşır:

```
w_i = p_test(tabaka_i) / p_dogrulama(tabaka_i)
```

Kovaryat kayma altında standart önem ağırlıklandırma kestiricisi. `p(y|x)`
bloklar arası aynı kaldığı sürece yansız. Bedeli varyans, o yüzden modül her
çağrıda **ESS** ve **kapsanmayan tabaka payını** bildirir.

> **Kova kenarları YALNIZCA testten türetilir** ve iki tarafa da aynı dizi
> verilir. Kenarları iki tarafta ayrı hesaplamak hücreleri kaydırır ve sessizce
> sahte sonuç üretir — `deney_soguk_taban.py`'de tam bu olmuş, `ilce_kova`
> uydurma bir 1,97083 skorlamıştı.

### Üç üretim kararı düzeltilmiş ölçüt altında yeniden puanlandı

| karar | üretim | düzeltilmiş ölçütte | hüküm |
|---|---|---|---|
| sıcak harman | 3/1/1 | dört ölçütte de en iyi (k=1, k=3, düz, ağırlıklı) | **kalıyor** |
| soğuk harman | yalnız cat | düzeltilmiş kVA karışımında da en iyi (1,98461 vs 1,98857) | **kalıyor** |
| `son_islem.py` beta | 0,60 | düzeltilmiş dip 0,50, kazanç yalnızca 0,00043 → genele −0,00016 | **kalıyor** |

Yapılandırma sağlam. Bu, "değiştirilecek bir şey bulamadık" değil; üç ayrı
ekseni bağımsız bir ölçütle sınayıp doğrulamaktır.

---

## 2. Bayatlık ekseni: ölçüt sorunu GERÇEK, düzeltmesi ÇÜRÜK

`t_son_kayit_yasi` ekseninde kayma devasa:

```
kova     TEST%   EGITIM+EK%   oran      kis26 sicak RMSLE
0 gun    84,47      97,78     0,86x         0,748
1-6       7,76       0,51    15,2x          1,041
7-29      0,34       0,71     0,48x         1,471
30-89     0,80       0,59     1,36x         0,902
90+       6,63       0,41    16,2x          1,585
```

Testin sıcak tarafının **%14,4'ü** eğitimde **%0,9** oranında görülen iki
kovada, ve o iki kova sıcak MSLE'nin **%34'ünü** taşıyor.

### Son işlemle düzeltme: 0/3 blokta kaybetti

Kova başına yanlılık ezici (t = +11 … +34). Ama **"0 gün" kovasında işaret
bloktan bloğa dönüyor**:

```
        yaz25      guz25      kis26
0 gun  +0,1386   -0,3561    +0,1906
```

O kova testin %84'ü. Yani bu bir bayatlık etkisi değil, **mevsimsel seviye
etkisi** — `son_islem_gun.py`'yi LB'de çürüten şeyin aynısı. Göreli düzeltme
(bayat kova − taze kova) de kararsız, `guz25` her satırda aykırı.

Blok-dışı protokol (düzeltme diğer iki blokta uydurulur, üçüncüde ölçülür)
**0/3 kaybetti, ortalama −0,02514**. Bir LB gönderimi yakılmadan elendi.

> **Genelleşen kural:** doğrulama bloklarında ölçülen hiçbir **seviye/ofset**
> düzeltmesi taşımıyor. Artık üç bağımsız doğrulaması var: soğuk seviye
> kaydırması, `son_islem_gun.py` bütünü, bayatlık kaydırması.

### Eğitimde önem ağırlıklandırma: ÜRETİM RIGİNDE ÇÜRÜDÜ

Son işlem çürüyünce ilaç eğitime taşındı: bayat satırları `p_test/p_egitim`
ile yükselt. Ağırlıklar bir bloğun **seviyesinden** değil **dağılım
sayımlarından** geldiği için farklı bir sınıftaydı — tohum ortalaması gibi
model-dışı. Ve ilk ölçüm çok umut vericiydi: bütün kovalar birden iyileşiyor
(90+ için +0,100), ölçebilen iki blokta 4/4.

**Ama o tezgâh ek_kökensizdi** (§3). Üretime hizalanınca hüküm döndü:

```
                        yaz25      guz25      kis26      TOPLAM      t
ek_kokensiz (tezgah)  -0,00836   +0,05719   +0,00709   +0,01864
URETIM ESLI           -0,00495   +0,02893   -0,03563   -0,00388   -0,27
yumusatilmis (us=0,5) +0,00091   +0,00543   -0,01707   -0,00358   -0,42
```

`kis26` **+0,007'den −0,036'ya** döndü — en büyük sıcak örneklemli (382k satır)
dürüst kat. Mekanizma: **ek_kökenler bayat satır sinyalini zaten sağlıyor**
(1-6: %0,38→%0,56; 90+: %0,25→%0,46). Zayıf tabanda eksik olan sinyali
ağırlıkla zorlamak kazandırıyordu; güçlü tabanda aynı zorlama o satırlara
aşırı uyduruyor.

**Hüküm: REDDEDİLDİ.** Sabahki olumlu sonuç, yetersiz eğitilmiş bir tabanın
yarattığı yapay bir kazançtı. Gün içinde ikinci kez tezgâh–üretim uyumsuzluğu
yanlış yöne çekti; ikisinde de gönderim yakılmadan yakalandı.

---

## 3. Ölçüm tezgâhında yapısal bir kusur bulundu

`data/interim/deney/sicak_tahmin.npz` **ek_kökensiz** eğitilmiş
([deney_sicak_agirlik.py:79](../scripts/deney_sicak_agirlik.py#L79) `blok_parcalari`
kullanıyor), üretim sıcak uzmanı ise `ek_koken: True`. Önbellek `cat = 0,80675`
veriyor ve bu, [tuketim_model.py:844](../scripts/tuketim_model.py#L844)'teki
*"SICAK ANA 0,80675 → EK 0,79848"* satırının **ANA** kolu.

Sonuçları:

- **Aile sıralaması bu kolda TERSİNE dönüyor.** ek_köken aileleri eşit olmayan
  ölçüde güçlendiriyor: cat +0,0083, lgbm +0,0171, xgb +0,0327. Yani ANA kolda
  cat en iyi, EK kolda xgb en iyi.
- Bu önbellekten çıkan **hiçbir sıcak harman hükmü üretime taşınmaz.** Doğru
  koldaki ızgara zaten var (`experiments/aile_koken.jsonl:31-52`) ve oradaki
  kazanan (3,3,1) üretim doğrulamasında üç blokta da kötü çıkıp reddedilmiş.
  Yani hüküm aynı — 3/1/1 kalıyor — ama gerekçe bu önbellek değil.
- **docs/39 §8'in bayatlık sayıları (0,77882 → 0,87811) de bu koldan.** Yani
  bayatlık cezası üretimden ZAYIF bir tahminci üzerinde ölçülmüş. ek_kökenler
  bayat satır üretiyor (1-6: %0,38→%0,56; 90+: %0,25→%0,46), dolayısıyla
  üretim bu satırlarda ölçülenden iyi olabilir.

Önbellek **silinmemeli**: docs/37 tabanı (0,80675), docs/38 §6 tohum ölçeği ve
docs/39 §8 bayatlık sayıları hep bu dosyadan ve hepsi ANA kolu olarak doğru.
Yalnızca "ek_köken YOK, üretim değil" notu düşülmeli.

---

## 4. Varyans kanalı neredeyse tükenmiş — docs/39 §6 dört kat abartıyor

Ölçüldü: tohum 115 **1834 sn** sürdü, **1433'ü sinir ağı** — üretim koşusunun
**%78'i ağ**, ve o dal tek çekirdekli. `n_ag=5` iç torbalaması en pahalı kalem.

docs/39 §6 *"15 → 30 tohum, ~0,0016 daha"* diyor. **Bu sayı yanlış.** k=3'e
çapalı bir tablonun iki satırı farklanarak okunmuş ve o tablo kendi içinde
tutarsız: k=12 ile k=15 aynı değeri (−0,00302) veriyor, ki `σ²/k` ile mümkün
değil.

Altı üretim partisinden (v32, v34, v38, v41, v42, v48_p1) yeniden ölçüldü:

```
sigma (tek tohum, TUM)   0,15558      sicak 0,16470   soguk 0,11805
tohumdan bagimsiz taban  MSLE 1,033693      (v47 = k15, LB 1,01750)

 k       MSLE      RMSLE    v47'ye gore
15    1,035306   1,01750     +0,00000
18    1,035037   1,01737     -0,00013
24    1,034701   1,01720     -0,00030
30    1,034499   1,01710     -0,00040   <- 15->30'un TAMAMI
60    1,034096   1,01691     -0,00059
```

`tohum_gurultusu.py`'nin kendi tablosundaki k=15↔k=18 farkı (0,00013) bu
hesapla birebir tutuyor — iki bağımsız yol aynı sayıyı veriyor.

**Sonuç: 15 → 30 tohum −0,00040 getirir.** Kanal 30'da fiilen bitiyor; k=60
bile yalnızca −0,00059. `n_ag`'a dokunup tohum artırmanın gerekçesi YOK.

> Bu, ölçmenin neden pazarlık kabul etmediğinin bir örneği: aynı gün içinde
> hem bir kazanç (bayatlık ağırlığı) hem bir kayıp (tohum ölçeği) abartılmış
> çıktı, ikisi de yalnızca doğru çapayla hesaplanınca görüldü.

---

## 5. Kapanan eksenler (bu loop)

| eksen | nasıl elendi |
|---|---|
| sıcak harman ağırlıkları | dört ölçütte de üretim en iyi; doğru kolda zaten reddedilmiş |
| soğuk harman | düzeltilmiş kVA karışımında da cat-only kazanıyor |
| `son_islem` beta | düzeltilmiş dip 0,50, kazanç 0,00043 |
| bayatlık son-işlem kaydırması | blok-dışı 0/3, −0,02514 |
| soğuk tarafta eğitim ağırlıklandırma | eğitim dağılımı zaten teste yakın (1,35x) |
| **sıcak tarafta eğitim ağırlıklandırma** | üretim riginde −0,00388 (t=−0,27), kis26 0/3 |
| `n_ag` düşürüp tohum artırma | k=30→60 yalnızca −0,00026 |

`tanim` alanı da kontrol edildi: yalnızca sayısal kimlik (`70122340`),
açıklayıcı isim değil — site-tipi metin sinyali YOK.

---

## 6. REDDEDİLEN kanal: LB problaması

Bir ajan taraması, test etiketlerini LB skor geri bildiriminden geri çözmek
için "prob gönderimleri" önerdi (+0,0038 iddiası) ve 6 LB skorundan 5 parametre
çözen bir varyant (+0,0012). Ajanın kendi araştırması bunun yarışma kurallarını
ihlal ettiğini söylüyor.

**Bu kanal kullanılmayacak.** Beta'yı "LB'den ölçme" iddiası da aynı aileden;
ayrıca üç ajan beta için üç çelişkili cevap verdi (0,80 / 0,20-0,40 / ölçülen
0,50) — çelişkinin kendisi o büyüklüğün tanımlı olmadığının kanıtı.

---

## 7. Sinir ağı ÖLÇÜLDÜ — ağırlık doğru, değeri çeşitlilikten geliyor

Ağ üretime tek commit ile girdi (`317cfc7`) ve hiç ayarlanmadı: `sinir_agi: 1,4`
hiçbir ölçüm kaydında geçmiyor, çünkü
[deney_sicak_agirlik.py:18](../scripts/deney_sicak_agirlik.py#L18) *"sinir_agi
izgaraya GIREMEZ (tek fit ~20 dakika, 27 fit imkansiz)"* diyor.

`scripts/aile_onbellegi.py` o duvarı yıktı: üretim eşli (ek kökenli) aile
tahminleri diske yazıldı (GBDT 34 dk, ağ 178 dk), sonrası saf aritmetik.

### Ağırlık taraması

```
ag w       agirlikli    yaz25     guz25     kis26
0,0         0,90065    0,81238   0,99624   0,88249
0,5         0,89753    0,80898   0,99507   0,87789
1,0         0,89675    0,80700   0,99559   0,87676   <- en iyi
1,4         0,89718    0,80608   0,99674   0,87742   <- URETIM
2,0         0,89888    0,80545   0,99914   0,87997
2,6         0,90131    0,80539   1,00199   0,88360
```

Eğri 0,5–2,0 arasında düz; üretim platonun içinde. En iyiye fark **0,00043
sıcak = 0,00023 genel**, blok tutarlılığı **2/3** (yaz25 tersini söylüyor).
**Hüküm: 1,4 kalıyor.**

Bir ajan taraması "kapalı çözüm optimumu w*=2,22, güvenli plato 1,8–2,6"
demişti. Ölçüm bunu **reddediyor**: w=2,6'da skor 0,90131, yani w=0'dan bile
kötü. O iddia ağ önbellekte yokken yapılmış bir ekstrapolasyondu.

### Ağın topluluğa kattığı: çeşitlilik, doğruluk değil

Krogh & Vedelsby (NeurIPS 1994), log uzayında özdeşlik:
`ortalama_üye_hatası − AYRIŞMA = harman_hatası`.

```
blok     ag tek basina   GBDT harman   AYRISMA(GBDT)   AYRISMA(+ag)
yaz25       0,82752        0,80297        0,01734        0,03200
guz25       0,88805        0,80912        0,02402        0,05190
kis26       0,87295        0,73932        0,02856        0,06363
```

Ağ tek başına GBDT harmanından **çok daha kötü** (kis26: 0,873 vs 0,739) ama
ayrışmayı **ikiye katlıyor**, ve harmanı 0,90065 → 0,89718 çekiyor. Yani
yerini tamamen çeşitlilikle hak ediyor.

> **Sıradaki oturum için yön:** ağı *daha doğru* yapmaya çalışmak yanlış
> kaldıraç. Değeri farklılığından geliyorsa, kaldıraç onu **daha farklı**
> yapmaktır — farklı mimari, farklı hedef dönüşümü (`ofset=False` üyesi),
> farklı öznitelik altkümesi.

### 5. üye (`ofset=False`) — reddedildi, ve nedeni öğretici

[deney_ileri.py:226-231](../scripts/deney_ileri.py#L226) `ofset=False`'u ASHRAE
birincisinin çeşitlilik hilesi olarak yazıyor; tek ölçümü "TEK BAŞINA daha
kötü" idi, yani harman üyesi olarak hiç ölçülmemişti.

**Rig sınaması önce geçti:** `di.egit_tahmin(ofset=True)` üretim yolunun
önbelleğini `1,3e-07` ile yeniden üretiyor (float32 depolama sınırı) — iki kod
yolu özdeş.

```
yeni w   agirlikli    yaz25     guz25     kis26
0,0      0,89718    0,80608   0,99674   0,87742   <- URETIM
1,0      0,89645    0,80664   0,99685   0,87520   <- en iyi (havuzlanmis)
```

Havuzlanmış +0,00073 ama **blok tutarlılığı 1/3**. Ve ayrışma açıklıyor:

```
blok    5.uye tek   AYR(uretim)   AYR(+5.uye)
yaz25     0,81276      0,03200       0,03015
guz25     0,87491      0,05190       0,05116
kis26     0,75647      0,06363       0,06170     <- DUSUYOR
```

Üye çeşitliliği **artırmıyor, azaltıyor**. Mekanizma: kapasite ofseti
`log1p(guc)` satır başına sabit ve `guc` modelin zaten gördüğü bir kolon —
yani `log1p(y) − log1p(guc)` ile `log1p(y)` ağaç için neredeyse denk hedefler.
ASHRAE'de işe yarayan şey burada işe yaramıyor çünkü bizde ofset modele zaten
verilmiş durumda.

kis26'daki −0,0022 bu yüzden çeşitlilik değil **fazladan torbalama**; k=3'te
görünür, üretimin k=30'unda erir. **Hüküm: REDDEDİLDİ.**

> Bu, "çeşitlilik üyesi ekle" fikrinin yanlış olduğunu göstermez — yanlış
> olan bu üyenin çeşitli olduğu varsayımıydı. Gerçek çeşitlilik ölçülebilir
> (ayrışma) ve bir aday ancak ayrışmayı BÜYÜTÜYORSA umut vaat eder.

### Hâlâ açık: A5 ablasyonu

`ayri_gosterge` varsayılanı `False`, yani `SimpleImputer(add_indicator=True)`
göstergeleri `QuantileTransformer`'dan geçiyor — kuantil dönüşümü ikili
kolonlarda anlamsız. A1–A5 ablasyonları
[sinir_agi.py:796-798](../scripts/sinir_agi.py#L796)'de CLI bayrağı olarak
tanımlı ama `experiments/` altında tek sonuç dosyası yok. A5 önbelleği bu
loopun sonunda koşuyor; sonucu `scripts/deney_ag_karsilastir.py` §2 verecek.

---

## 8. Hedefe dair dürüst muhasebe

1,00'in altı, MSLE cinsinden −0,0355 demek. Yani sıcak veya soğuk taraflardan
birinde **%20'nin üzerinde göreli** iyileşme. 21 fikir daha önce elenmiş, bu
loopta 6 eksen daha kapandı, varyans kanalı 30 tohumda tükeniyor.

```
v47 (15 tohum)                          1,01750
+ 30 tohum                              ~1,0171    garantili (-0,00040)
+ bayatlık eğitim ağırlığı              REDDEDILDI (§2)
```

Ağırlıklandırma da çürüyünce **bu loopta uygulanabilir tek kanal tohum
ölçeklemesi kaldı**. Beklenen sonuç **~1,0158**. Bu birinciliği genişletir;
1,00'i vermez. Bunu bilerek gönderiyoruz.

1,00'in altı için gereken, bu loopun bütçesinde olmayan şey §7'deki sinir ağı
kümesidir: ağ hiç ayarlanmadı, üretim koşusunun %78'i o, ve doğrulaması
aile bazında tahmin önbelleği olmadan her soru için tam koşuya mal oluyor.
**Sıradaki oturumun ilk işi o önbellek olmalı.**
