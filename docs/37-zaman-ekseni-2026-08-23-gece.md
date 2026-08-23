# Durum — 23 Ağustos 2026 gecesi (soğuk son işlem yeniden kuruldu)

**Compact sonrası okunacak tek belge.** Öncekiler:
[36-birincilik](36-birincilik-2026-08-23.md) (birinciliğe çıkış ve kaybediliş),
[35-ezber-kanali](35-ezber-kanali-2026-08-23.md) (CV neden kırıldı),
[30-harici-veri-denetimi](30-harici-veri-denetimi-2026-08-22-gece.md) (dış veri kapalı).

---

## 1. Sıralama (23 Ağustos 23:00)

```
1. Data4Win           1,02298   (gonderim 18:21)   <- YENI, bir anda geldi
2. Bilalcan Ustabas   1,02522
3. TasnifX (BIZ)      1,02639
4. Ugur Celik         1,02901
5. Saliha Rana Uzun   1,02927
```

Gün içinde birinci **iki kez** değişti. Sabit bir hedef yok; 1 Eylül'e
kadar ~24 gönderim hakkımız var.

---

## 2. GÜNÜN EN ÖNEMLİ BULGUSU: büzme, zaman eksenini de eziyordu

`son_islem.py` ofsetin **tamamını** tek bir sabite (tahminin kendi genel
ortalaması) çekiyordu. Ama ofsetin iki ekseni var:

```
trafo ekseni   asiri yayilmis   BUZULMELI
zaman ekseni   gercek mevsim rampasi   KORUNMALI
```

Beta `kis26` (Ara–Mar) üzerinde ayarlandı ve **orada ay ekseni varyansı
0,00113** — kış ofseti düz olduğu için zaman eksenini ezmek bedava
göründü. Test penceresinin (Nis–Tem) mevsimsel ikizinde aynı varyans
**0,15298**:

> **Test penceresinde zaman ekseni, beta'nın ayarlandığı kattan 136 KAT güçlü.**

Gönderim dosyalarında doğrudan ölçüldü (158.369 soğuk satır):

```
ay   v27 ofset   v30 ofset   2025 ayni ay gercek
04     +0,1219     +0,2611          +0,0408
05     +0,1169     +0,2581          +0,0056
06     +0,4138     +0,4363          +0,4706
07     +0,7642     +0,6465          +0,9517

gun-ortalamalarinin std'si   0,3003 -> 0,1802   (tam olarak x0,60 = beta)
```

v27'nin May→Tem rampası zaten gerçeğin **altındaydı** (0,647 vs 0,946);
büzme onu 0,389'a indirip daha da uzaklaştırdı.

### Düzeltme: `scripts/son_islem_gun.py`

```
r        = log1p(tahmin) - log1p(guc)
gun_ort  = o GUNUN soguk satirlarindaki r ortalamasi     (modelden)
hucre    = ampirik-Bayes ilce x kova ofset ortalamasi    (egitimden)
etki     = hucre - o gunun soguk satirlarindaki hucre ortalamasi
taban    = gun_ort + etki
r'       = taban + beta * (r - taban)
```

Gün ortalaması **tam olarak** korunuyor (etki gün içinde merkezlendiği
için; kodda assert ile 1e-9 eşiğinde kontrol ediliyor).

---

## 3. İkinci düzeltme: hedef artık koşullu

Genel ortalama, bir tahminin çekilebileceği en kaba hedef. Efron–Morris
(1975) James-Stein'i tam bu yönde genelleştirir: hedef ne kadar
bilgiliyse büzmenin yanlılık maliyeti o kadar küçüktür.

Hücre yapısı **üç blokta** tarandı (`deney_taban_ince.py`) — saf grup
ortalamasında ezberlenecek bir şey olmadığı için `yaz25`/`guz25` bu
karşılaştırma için geçerli:

```
kova sayisi 24, ebeveyn secimi (uc blok ortalamasi, dusuk = iyi)
  ebeveyn ilce       M=500    1,75711   <- ilce her hucrede kazandi
  ebeveyn kova       M=500    1,76009
  ebeveyn toplamsal  M=500    1,76058
```

`ilce` tek başına `kova`'dan iyi, o yüzden seyrek hücre oraya düşmeli.

### `kis26` soğuk, 3 tohum, aynı önbelleklenmiş tahminler

```
URETIM (v30: kendi ortalamasi, beta=0,60)        1,83979
GUN korumali, hucresiz,          beta=0,30       1,83438
GUN + ilce x kova M=2000,        beta=0,25       1,83114
  ...ayni kurgu SOGUK HARMAN cat-only ile        1,82250   <- URETIM
(kis26 ust siniri: kis26'ya uydurulmus afin OLS  1,80850)
```

**Sızıntı denetimi.** `kis26`'da soğuk trafoların **%0,0'ı** eğitim
parçasında (yaz25 %94,1, guz25 %94,9 — docs/35'teki ezber kanalı).
Yani `kis26` testle yapısal olarak birebir aynı; ölçüm dürüst.

---

## 4. Soğuk harman: 3/1/1 → yalnız cat

Yeni son işlem altında, aynı önbelleklenmiş tahminler üzerinde:

```
harman     beta=1,00   0,30      0,25      0,20
3/1/1        1,86931  1,83083  1,83041  1,83031
5/1/1        1,85717  1,82767  1,82750  1,82758
YALNIZ cat   1,84106  1,82245  1,82250  1,82274   <- SECILEN
```

Beta ne olursa olsun sıralama aynı. Üretim doğrulaması da doğruladı:
`kis26` soğuk **1,86509 → 1,83606**.

---

## 5. Hazır gönderim dosyaları

```
tuketim_v31_gun.csv    v27 (eski model) + son_islem_gun
                       -> son islemi TEK BASINA olcer (v30'un 1,02639'una karsi)
tuketim_v33_gun.csv    v32 (yeni model: cat-only soguk + sinir agi) + son_islem_gun
                       -> en iyi beklenen
tuketim_v35_gun.csv    v33 ile ayni ama 6 tohum (uretiliyor)
```

Üretim doğrulaması (v32, tek tohum 42):

```
blok    RMSLE    sicak    soguk    TEST-AGIRLIKLI
yaz25  0,88084  0,79985  1,56641      1,02064
guz25  0,96267  0,80509  1,70540      1,07193
kis26  0,97193  0,74263  1,83606      1,08459
ORTALAMA (test-agirlikli) 1,05905     (onceki kosu 1,09551 kis26'da)
```

### Beklenen LB

Kalibrasyon: bir önceki değişiklikte `kis26` soğuk −0,02952 → LB −0,00723,
yani **aktarım katsayısı 0,245**.

```
v33: kis26 soguk 1,83979 -> 1,82250 = -0,01729  ->  LB -0,0042
     + rampa onarimi (kis26'da GORUNMEZ, dogrudan olculdu)  -0,0016
     TOPLAM  ~ -0,0058   ->   1,02639 - 0,0058 = ~1,0206
```

---

## 6. Bugün ölçülüp ELENEN (hiçbiri üretime girmedi)

| fikir | sonuç |
|---|---|
| ofset uzayında seviye kolonları (`t_dofs_*`) | −0,03287 SH 0,01415 **t=−2,32** · guz25'te 0/3 |
| doğal soğuk satırları sıcak eğitimden atma | 0,80675 → 0,81085 **kötü** |
| sıcak harman ağırlıkları (eski "+0,0032" iddiası) | yüzey **düz**: en iyi (5,2,2) +0,00009, tek tohum ters yön → REDDET |
| ufka göre tam afin kalibrasyon | **0/3 blok**, ortalama +0,06463 |
| ufka göre yalnız eğim, gün merkezli | 1/3 blok, +0,00852 |
| günlük seviyeyi düzleştirme (W=7…121) | en iyi 1,82245 vs 1,82250 — **hiç** |
| soğuk seviyeye sabit kayma | yıl-üzeri eşli fark ort +0,187 / **medyan +0,059** — tahminin belirsizliği kazançtan büyük |
| günlük şekli sıcak modelden alma | soğuk↔sıcak gün korelasyonu yalnız **+0,37**, vekil artık 0,1854 vs 0,2003 |

**Orakül sınırı:** `kis26` soğukta gerçek gün ortalamaları bilinseydi
1,82250 → **1,80266**. Ulaşılamıyor — modelin gün seviyesi hatası
0,27 RMS ve bu gürültü öngörülebilir değil. Soğuk son işlem pratik
tavanında.

---

## 7. Kalıcı kurallar (36'dakilere ek)

1. **Soğuk kararları `kis26` ile verilir** — ama bu kural MODELE bakan
   ölçümler içindir. Saf grup ortalamasında ezberlenecek bir şey yoktur,
   o yüzden taban karşılaştırmaları üç blokta da geçerlidir.
2. **`kis26`'da bedava görünen her şey testte bedava değildir.** Kışın düz
   olan bir eksen (zaman) Nisan–Temmuz'da 136 kat güçlü. Bir son işlem
   önerirken "hangi ekseni eziyorum" diye sor.
3. **Blok-dışı taşınmayan hiçbir kalibrasyon alınmaz.** Ufuk kesmesi,
   ufuk eğimi, sıfır sınıflandırıcı çarpanı — üçü de aynı duvara çarptı.
4. **Tohum eklemek risksizdir.** `--tohum-baslangic` ile var olan bir
   gönderime ek tohum üretilip log uzayında birleştirilebilir
   (`birlestir_tohum.py`); yanlılık değişmez, varyans ~1/k düşer.
