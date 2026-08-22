# Tavan analizi — 22 Ağustos 2026

Bu belge **neyin mümkün olduğunu** ölçer. Kalan dokuz günde hangi fikrin
denenmeye değer olduğuna karar verirken önce buraya bakılmalı: burada
tavanı düşük çıkan bir fikir, ne kadar zarif olursa olsun, zaman kaybıdır.

Bütün ölçümler `yaz25` bloğu (test döneminin mevsimsel ikizi), 254.296
sıcak satır, 2.213 trafo.

---

## 1. Kalibrasyon — bütün öngörülerin dayanağı

```
v15  yaz25 CV 0,99715 -> LB 1,03910   fark +0,04195
v18  yaz25 CV 0,99115 -> LB 1,03370   fark +0,04255
                                      sapma  0,0006
```

**`LB ≈ yaz25_CV + 0,0423`**, iki bağımsız noktada 0,0006 sapmayla. Aşağıdaki
her LB öngörüsü bu formülle üretildi.

---

## 2. Hatanın anatomisi

```
yaz25   sicak 0,80081   soguk 1,47665   -> test-agirlikli 0,99146
                                           (olculen 0,99115, yuvarlama)
```

### Sıcak taraf tam ikiye ayrılıyor

```
SEVIYE (trafo bazinda ortalama artik)   MSE 0,33336   %49,5   RMSLE 0,5774
SEKIL  (trafo ici sapma)                MSE 0,34006   %50,5   RMSLE 0,5831
```

Seviye hatası std **0,601** (ortalama +0,109) — tipik sapma `exp(0,6) = 1,8` kat.

Ve seviye hatası **eldeki hiçbir öznitelikle korele değil**: en güçlüsü
`t_sifir_orani` r = −0,145; `t_gun_sayisi`, `guc`, `t_log_ort`, `t_log_son30`,
`t_doluluk` hepsi |r| < 0,09. Yani sonradan düzeltilemez.

### Soğuk taraf neredeyse tamamen bilinmeyen seviye

```
soguk seviye hatasi = sqrt(1,4767² − 0,583²) = 1,3567
```

Şekil bileşeni sıcak tarafla aynı (~0,583) varsayılırsa, soğuk hatanın
**%92'si** trafonun bilinmeyen seviyesi. Dokuz açıdan denendi, kapanmadı.

---

## 3. Oracle merdiveni — neyin ne kadar değdiği

Her satır, o bilgi **bedava verilseydi** ulaşılacak sıcak RMSLE.

| oracle | sıcak RMSLE | not |
|---|---|---|
| tek sabit (global) | 2,2267 | taban |
| **ÜRETİM MODELİMİZ** | **0,80081** | |
| trafo × haftagünü | 0,5977 | seviyenin **yalnız 0,012 üstünde** |
| trafo seviyesi | 0,6101 | |
| trafo seviyesi + ilçe-gün şekli | 0,5753 | ilçe şekli 0,035 katıyor |
| trafo × ay | 0,4033 | trafoya özgü aylık — **bilinemez** |
| trafo × gün | 0 | tam oracle |

**Okunuşu:** hafta günü ve ilçe-gün şekli küçük (0,012 ve 0,035). Büyük
olan tek şey trafoya özgü aylık değişim — ve o, tanımı gereği bilinemez.

---

## 4. Seviye tahmincisinin tavanı — asıl sayı

Blok dışı uydurup `yaz25`'te sınayarak (6.131 → 2.213 (blok, trafo) satırı):

| seviye tahmincisi | hata std | → sıcak | → LB |
|---|---|---|---|
| **doğrusal: son30** | **0,5459** | **0,7603** | **1,0087** |
| doğrusal: son30 + ort | 0,5503 | 0,7635 | 1,0106 |
| `t_log_son30` (ham) | 0,5650 | 0,7741 | 1,0171 |
| **MODEL (mevcut, 105 kolon)** | **0,6010** | **0,8008** | **1,0338** |
| LightGBM orta (yaprak 31) | 0,6108 | 0,8082 | 1,0384 |
| LightGBM sığ (yaprak 8) | 0,6125 | 0,8095 | 1,0392 |
| LightGBM `linear_tree` | 0,6770 | 0,8593 | 1,0712 |

**En basit tahminci kazanıyor.** Tek değişkenli doğrusal regresyon bütün
GBDT'leri ve 105 kolonluk üretim modelini geçiyor.

Sebep: hedef esasen `son30 + gürültü`. 6.131 satırlık bir tabloda ağaçlar
gürültüyü ezberliyor; doğrusal model ezberleyemiyor. `linear_tree` en kötüsü
— yaprakta doğrusal uydurmak, az veriyle varyansı patlatıyor.

> Bu ölçüm bir araştırma önerisini (LightGBM `linear_tree`, "ağaçlar doğrusal
> özdeşliği basamaklarla kuruyor" sorununa çözüm olarak) **otuz saniyede**
> eledi. Fikir doğruydu, uygulaması bu veri ölçeğinde çalışmıyor.

---

## 5. Sonuç: 1'in altı ULAŞILAMAZ

```
                                              sicak    yaz25    LB
MEVCUT (v18)                                 0,8008  0,99146  1,0338   <- gercek 1,03370
seviye tahmini MUKEMMEL calissa              0,7603  0,97434  1,0166
   (0,5459'luk tahminci, kusursuz aktarimla)
1'IN ALTI ICIN GEREKEN                       0,7461  0,95770  1,0000
SEVIYE TAM COZULSE (oracle, ulasilamaz)      0,5831  0,86521  0,9075
```

Ulaşılabilir en iyi seviye tahmincisi **kusursuz** çalışsa bile LB ~1,009–1,017.
1'in altı için gereken 0,7461'lik sıcak skor, ölçülen tavanın (0,7603) dışında.

Aradaki farkı kapatacak bilgi kaynağı **yok**, ve bu ölçüldü:

| kaynak | hüküm |
|---|---|
| dış veri (turizm/su/yangın/deprem/hava kalitesi/konvektif/saatlik hava) | toplam varyansın %0,20'si, tahmini kazanç ~0,0001 — eşiğin **150 katı altında** |
| `tanim` anlamsal token'ları | %99,8'i saf sayısal kod; harf içeren %0,15 ve hepsi yer adı |
| trafo koordinatı (GDZ kesinti CBS) | `trafo_kodu` %100 boş, koordinat trafonun değil |
| geçen yıl aynı mevsim (lag-365) | güncel çeyrekten zayıf (r 0,825 vs 0,969), kapsam %33,7 |
| soğuk trafo seviyesi | hatanın %92'si, dokuz açıdan denenip kapanmadı |

**Hedef 1'in altı değil, birinciyi (1,03170) net farkla geçmek olmalı.**
Ulaşılabilir bant **1,009–1,021**, yani 0,01–0,02 fark.

---

## 6. Dış veri denetiminin üç yapısal bulgusu

Bunlar tek tek ölçümlerden daha kalıcı:

1. **Hedef varyansının %87,1'i trafolar arası seviye.** Dış veri ilçe
   çözünürlüklü olduğu için buraya hiç dokunamıyor. Erişebildiği havuz
   toplam varyansın %1,68'i ve üretim bunun **%88,4'ünü zaten alıyor**.

2. **Test ilçeleri = eğitim ilçeleri.** Bu, ilçe-statik her kolonu
   (`arazi_ortusu` ek, `osm_altyapi` ek, `turizm_yillik`, `su_yaz_kis`)
   `ilce_key` karşısında **tanım gereği fazlalık** yapıyor. Genelleme
   argümanı burada geçerli değil.

3. **Pozitif çıkan dört dış veri ailesinin dördü de TAKVİM ölçüyor.**
   `gunes_deklinasyon` (nunique=365, saf yılın günü), `turizm_il_doluluk`
   (il×gün R²=1,0000), `su_ay_endeksi` (ilçe×ay), `yaprak_mevsimi`
   (R²=1,0000). Doğru deney onları eklemek değil, takvimi geri koymayı
   ölçmek.

Ayrıca bir varsayım düzeltildi: panel **2 il / 47 ilçe** (İzmir, Manisa) —
beş il değil. Muğla/Aydın turizmi üzerine kurulacak her beklenti temelsiz.

---

## 7. `v20` — soğuk uzmanına hafta günü

```
              v18        v20      fark
yaz25       0,99115    0,98863   -0,0025
guz25       1,04950    1,05023   +0,0007
kis26       1,09768    1,09700   -0,0007
ORTALAMA    1,04611    1,04529   -0,0008

sicak       0,80081    0,80081   DEGISMEDI  <- dokunulmadi, dogru
soguk       1,47665    1,46901   -0,0076
```

Sıcak taraf birebir aynı — hafta gününün yalnızca soğuk uzmanına verildiğinin
doğrudan kanıtı. LB öngörüsü **0,98863 + 0,0423 = 1,03093**, birincinin
(1,03170) altında.

---

## 8. LB PROBU — CV'den ölçülemeyen kaymayı LB'den çözmek

**Sorun.** Model Nisan 2025 – Mart 2026 ile eğitilip Nisan–Temmuz 2026'yı
tahmin ediyor: zaman ekseninde eğitim aralığının **dışında**, ve GBDT'ler
dışdeğerleme yapamaz. Sistematik bir kayma olabilir.

CV'den ölçülemiyor çünkü blok yanlılıkları mevsimle karışıyor (yaz25 +0,082,
guz25 −0,343, kis26 +0,192 — işaretler tutarsız).

**Çözüm.** LB skorlaması belirlenimci ve RMSLE² kaymada kuadratik:

```
S1² = S0² − 2·δ·E[r] + δ²      ->      E[r] = (S0² + δ² − S1²) / (2δ)
```

Tek prob optimal kaymayı **analitik** olarak veriyor. Eğitim gerekmiyor:
`pred' = expm1(log1p(pred) + δ)`.

Okuma tablosu (S0 = v20'nin skoru, δ = 0,10):

| gelen S1 | çözülen E[r] | optimal LB | kazanç |
|---|---|---|---|
| S0 − 0,009 | ≈ +0,14 | S0 − 0,0095 | büyük |
| S0 − 0,005 | ≈ +0,10 | S0 − 0,0047 | orta |
| S0 − 0,002 | ≈ +0,07 | S0 − 0,0022 | küçük |
| ≈ S0 | ≈ +0,05 | S0 − 0,0012 | ihmal |
| S0 + 0,003 | ≈ −0,02 | S0 − 0,0001 | yok |

**Neden güvenli:** private LB'de E[r] aynı olacak (aynı model, aynı dönem,
n=214k'da örnekleme hatası ihmal edilebilir). Bu, public'e aşırı uydurma
riski taşımayan nadir LB-probu türü. Ayrıca Kaggle en iyi skoru koruduğu
için aşağı yönlü risk yok.

**Beklenti ılımlı tutulmalı:** test'in özet penceresi 2026-03-31'de bitiyor,
yani `t_log_son30` zaten Mart 2026'yı okuyor ve yıllık büyümenin çoğu o
pencereye girmiş. Dışdeğerlenecek kısım yalnızca Nisan–Temmuz arası 1–4 ay.
Ölçülen panel büyümesi (aynı trafo, Oca–Mar 2025 → Oca–Mar 2026) satır
ağırlıklı **+0,149**, medyan **+0,057** — ama bunun çoğu zaten fiyatlanmış.
