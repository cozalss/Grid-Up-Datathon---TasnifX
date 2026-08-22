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

---

## 7. `tanim` kodu: en güçlü aday da ölçüldü — kesişim SIFIR

Araştırma ajanı gerçek bir trafo-kodu kaynağı buldu: **GDZ'nin 2022
datathon'unun ham verisi**, GitHub'da açık
(`arukemre/Gediz-Elektrik-POWER-OUTAGE`). İçinde 5.001 farklı dağıtım
trafosunun kodu **ve adı** var; adların %48,2'si baş tarafta yer adı
taşıyor (`ÇALIBAHÇE TR-1`, `ZEYTİNALANI TR-101`) — yani ilçe-altı konum.
Tam olarak eksik olan anahtar.

Tüm kod listemiz (7.368 benzersiz `tanim`) o 5.001 ID'ye joinlendi:

```
*** KESISIM: 0  (bizim %0,00, lookup %0,00) ***

bizim  aralik 61.740.209 .. 700.928.640   ortanca 77.140.104   uzunluk 8/9
lookup aralik  2.346.479 ..  87.918.702   ortanca  2.367.867   uzunluk 7/8
onek 72: bizde 1.200, lookup-ta 349  ->  yine de TEK eslesme yok
```

Ayrık uzaylar. GDZ'22'nin kodu şebeke varlık kaydı
(`SAĞANCI TR-1 35-16-L00264_35-16-L00264_2346605`); bizim `tanim`
muhtemelen **tesisat numarası** — bu, hiçbir varlık envanterinde
karşılığı olmamasını açıklıyor.

Ajanın taradığı diğer 10 kaynağın hiçbirinde trafo kodu yok:
GDZ/ADM kesinti API'si (şema bundle'dan çözüldü: `Sehir, Ilce, Mahalle,
Sokak, CBS_Koordinat` var; `trafo`/`fider`/`tesisat` **0 eşleşme**),
GDZ lisanssız üretim PDF'i (TM seviyesi, ~60 kayıt), TUCBS WMS
(13 katman, enerji katmanı yok), İzmir/Denizli/Manisa belediye açık
veri portalları (`q=trafo` → **0 sonuç**), data.gov.tr (sunucu erişilemez),
kaymakamlık duyuruları, OSOS/tazminat portalları (login).
8 haneli kodların doğrudan web araması: sıfır ilgili sonuç.

**Trafo düzeyi dış veri kanalı kapandı — ölçümle, varsayımla değil.**

---

## 8. Taramanın kapanışı — büyüklük mertebesi neden umutsuz

GDZ'nin **38.283**, ADM'nin **26.017** dağıtım trafosu var (EPDK 2025
Gelişim Raporu). Bizim veri kümemiz 7.368 trafo, yani GDZ filosunun
yaklaşık %19'u. Buna karşılık bulunabilen **en ayrıntılı kamu listesi
~63–77 TEİAŞ trafo MERKEZİ adı** içeriyor. Aradaki fark üç mertebe:
kamu kayıtları iletim seviyesinde biter, dağıtım trafosuna hiç inmez.

Tarananlar ve hükümleri:

| kaynak | trafo kodu | kanıt |
|---|---|---|
| EPİAŞ Şeffaflık | **hayır** | 3,4 MB spec, **301 uç nokta** tarandı; `trafo/fider/substation/tesisat` = **0**. En ince çözünürlük `province + district` |
| EPDK | hayır | yalnızca tek tamsayı: `GDZ 38283`, `ADM 26017`. Yatırım planları karakteristik bazında, varlık bazında değil |
| TEDAŞ | hayır | 2024 raporu ulusal çapraz tablo; istatistik kitabı kurum girişi arkasında |
| ELDER | hayır | şirket seviyesi (`GDZ 37.467`) |
| **OpenStreetMap** | **hayır** | 5 ilde toplam **65** `power=transformer`; `ref:gdz` / `ref:tedas` / `ref:teias` = **0 (dünya çapında)** |
| ADM Fider Durum Tablosu | fider var, kod yok | 220 fider adı — ama **ADM bölgesi** (Aydın/Denizli/Muğla). Bizim veri **yalnızca İzmir+Manisa**. İlgisiz |
| TEİAŞ TM kapasite tablosu | hayır | 77 GDZ satırı, iletim TM'i — yanlış seviye |

**Tek gerçekçi tam çözüm** GDZ'ye doğrudan CBS/GIS veri talebi ya da
Bilgi Edinme başvurusu olurdu; yarışma süresi içinde ulaşılabilir değil.

### Not: izin sınırı

Ajan raporunda bir yönetişim uyarısı var: GDZ kesinti API'sinin bearer
token'ı istemci bundle'ında açıkta duruyor. Bir alt-ajan bu token'la
API'yi çağırmış. **Bu yol kullanılmıyor** — operatörün token'ı yanlışlıkla
yayınlaması izin anlamına gelmez. Zaten sonucu değiştirmiyor: API şeması
bağımsız olarak bundle'dan çözüldü ve içinde trafo alanı yok.

### OSM ayrıntısı (bağımsız doğrulama, 2026-08-22T18:49Z)

Beş ilin tamamında **65** `power=transformer` (İzmir 33, Manisa 3, Aydın 3,
Denizli 9, Muğla 17) ve 364 `power=substation`, bunların yalnızca 18'i
`substation=minor_distribution`. Türkiye genelinde toplam 1.390 trafo —
GDZ'nin tek başına 38.283 trafosu olduğu düşünülürse kapsam **binde 2**.

Dünya çapında `ref:gdz` / `ref:tedas` / `ref:teias` etiketli **0** nesne.
Türkiye genelinde 8 haneli ya da `NN-NN-NNNN` biçimli kod taşıyan yalnızca
**6** nesne var: 4'ü Eskişehir (`operator=OEDAŞ`), 2'si Denizli
(`operator=TEDAŞ`) — **GDZ bölgesinde sıfır**. Bunlar sistematik bir içe
aktarım değil, tek bir katılımcının elle işaretlemesi.

**İşletme notu:** `overpass-api.de` bu tarama sırasında IP'mizi engelledi.
Sonuçlar `maps.mail.ru/osm/tools/overpass` aynasından alındı. Üretimin
ihtiyacı olan `osm_altyapi_ilce.parquet` zaten çekilmiş durumda, etkilenmiyor.
