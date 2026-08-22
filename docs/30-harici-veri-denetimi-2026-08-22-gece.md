# Harici veri: üçüncü ve son denetim (2026-08-22 gece)

Soru: **skoru iyileştirecek dış veri var mı?**
Yanıt bu kez varsayımla değil, bu gece yapılan dört yeni ölçümle veriliyor.

---

## 1. Dış verinin bağlanabileceği anahtarlar — tamamı sayıldı

Veri kümesinde trafo düzeyinde yalnızca üç alan var: `tanim`, `guc`, `lokasyon`.

| anahtar | çözünürlük | durum |
|---|---|---|
| `lokasyon` | **tam 47 ilçe**, daha ince hiçbir şey yok | test ilçeleri = eğitim ilçeleri, yeni ilçe yok |
| `tarih` | gün | doymuş (bkz. §3) |
| `guc` | kVA | `guc_*` 4 kolon olarak işlenmiş |
| `tanim` | GDZ iç kodu | **hiçbir kamu kaydında karşılığı yok** |

`lokasyon` alanı İzmir'de `İL>BÖLGE>İLÇE`, Manisa'da `İL>İLÇE` — orta alan
tutarsız, ama üçüncü alan her durumda ilçe. **İlçeden ince konum yok.**

Veri kümesi yalnızca **İzmir + Manisa (GDZ)**; ADM (Aydın/Denizli/Muğla) yok.

---

## 2. `tanim` kodu ne taşıyor — ölçüldü

Kod 7–16 hane arası karışık; ilk hanelerde gerçek bir hiyerarşi var
(59 farklı ilk-3, 124 ilk-4, 226 ilk-5) ve önek ile il arasında güçlü bağ
(77/78 yalnız Manisa, 71/72/73 yalnız İzmir). Ama **seviye taşımıyor**:

```
OUT-OF-FOLD seviye tahmini (log, düşük iyi; ham std 2,1946)
  ilçe x kova   [üretim]      1,9867     <- en iyi
  tanim ilk3                  2,1255
  tanim ilk4                  2,1018
  tanim ilk5                  2,1012
  ilçe x kova x tanim-ilk4    2,0749     <- birleşim daha kötü (parçalanma)
```

Zaten `tanim_num`, `tanim_uzunluk`, `tanim_on2..on5` üretimde mevcut.
**Bu kanal kapalı.**

---

## 3. Gün düzeyi kanal DOYMUŞ — asıl bulgu

Trafo sabit etkisi ve ay etkisi çıkarıldıktan sonra günlük artık std = **0,0801**.
Üretimdeki `ulusal_*` (5 kolon) + hava kolonları bu varyansın **%76,8'ini**
açıklıyor. Geriye kalan std 0,0386.

Bu, **her gün düzeyli dış veri kaynağının** (tatil, tarife, okul, grev, maç,
kesinti) toplam olarak erişebileceği havuzun tamamı:

```
kalan gün varyansı        0,0386^2 = 0,00149
toplam MSE                1,0337^2 = 1,0685
tamamı çözülse RMSLE kazancı        0,00072
```

**Gün düzeyi dış verinin tavanı 0,0007.** Tek tek her fikir bunun altında.

---

## 4. İki aday tek tek ölçüldü ve elendi

### 4a. Kesinti (EPİAŞ plansız kesinti, ilçe x gün, test dönemini kapsıyor)

47/47 ilçe kusursuz eşleşti. Kesintinin tüketimi **mekanik** olarak düşürmesi
beklenirdi. Ölçüm tersini söyledi:

```
Pearson(artık, log1p(kesinti_dk))        +0,0404
kesinti_dk kademesi   0      1-60   60-300  300-1k  1k-3k   3k+
ortalama artık      -0,005  +0,010  -0,002  +0,003  +0,013  +0,017
```

İşaret **pozitif**: kesintiler yüksek yük günlerinde (sıcak dalgası şebekeyi
zorlar) ve yoğun ilçelerde oluyor. Karıştırıcı etki, mekanik etkiden büyük.
Açıkladığı varyans artığın %0,17'si. **Elendi.**

### 4b. Tatil takvimi (elimizde var, yalın sette atılmış)

Test dönemi 8 tatil günü içeriyor (%6,6), aralarında **Kurban Bayramı
2026-05-27..30**. Ham etki büyük:

```
eğitimde Kurban 2025-06-06..09 günlük artık: -0,149 -0,139 -0,113 -0,086
ortalama -0,1224  =  günlük artık std'sinin 1,5 katı
```

Ama `ulusal_*` + hava kontrol edildikten sonra:

```
TÜM TATIL   ham -0,0375  ->  kalan +0,0168   t=+1,69
KURBAN      ham -0,1224  ->  kalan -0,0376   t=-1,95   <- eşik altı
```

Sebep açık: ulusal tüketim Kurban'da 968k'dan **631k'ya** düşüyor.
Model bayramı `ulusal_gunluk` üzerinden zaten görüyor.
Kalan etkinin skora katkısı **0,00002**. **Elendi.**

---

## 5. Doğru çekildiği doğrulanan kaldıraçlar

Bunlar denetlendi ve **hatasız** bulundu — boşuna tekrar bakılmasın:

| kaldıraç | durum |
|---|---|
| hava verisi çözünürlüğü | ilçe başına (96 ilçe), il başına değil — azami |
| test dönemi havası | %100 **gözlem** (archive-api), tahmin değil |
| test dönemi ulusal tüketimi | dolu, gerçek; Mayıs 2026 = Mayıs 2025 − %7,4 |
| `ulusal_yil_once` / `yillik_buyume` | test aylarında hesaplanmış |
| arazi örtüsü, OSM altyapı, nüfus | ilçe başına işlenmiş, üretimde |

---

## 6. Sonuç

**Skoru iyileştirecek dış veri yok** — ve bu artık üç bağımsız yoldan ölçüldü:

1. Hedef varyansının %87,1'i trafolar arası; dış veri en ince ilçe düzeyinde.
2. Gün düzeyi kanalın kalan tamamı 0,0007 değerinde; `ulusal_*` + hava
   o havuzun %76,8'ini zaten almış.
3. Trafo düzeyine bağlanacak anahtar (`tanim`) hiçbir kamu kaydında yok.

Kalan gerçek boşluk dış veride değil, **soğuk uzmanında**: test satırlarının
%22,2'si eğitimde hiç görülmemiş 2.024 trafodan geliyor. Ölçülen açık,
ilçe x kova oracle'ına 0,0165 (genel skorda 0,006). Bu bir **modelleme**
sorunu, veri sorunu değil.
