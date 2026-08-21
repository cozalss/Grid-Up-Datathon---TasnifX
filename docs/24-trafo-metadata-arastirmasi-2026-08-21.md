# Trafo koordinatı / abone sayısı / tip araştırması — 21 Ağustos 2026

## Sonuç

Yarışmadaki `tanim` kodlarını doğrudan **trafo koordinatı + bağlı abone sayısı
+ trafo tipi** ile eşleyen açık bir tablo bulunamadı. Bu, verinin var olmadığı
anlamına gelmiyor: TEDAŞ numaralama şartnamesi bu alanların dağıtım şirketi
CBS'sinde tutulmasını zorunlu kılıyor; aynı şartname teknik ve abonelik
bilgilerini gizli bilgi sayıyor.

Uydurma eşleşme yapılmadı. Buna karşılık daha önce gözden kaçan resmî GDZ
kesinti API'si bulundu ve tekrar çekilebilir hale getirildi.

## Yeni bulunan resmî açık kaynak

GDZ planlı kesinti sayfasının kendi web istemcisi şu uçları kullanıyor:

- `/api/test-outages`
- `/api/unplanned-outages`
- `/api/outages-v2`
- `/api/outages-v2-unplanned`

`/api/outages-v2` yanıtında `Kesinti_ID`, il/ilçe/mahalle/sokak,
`CBS_Koordinat`, adres noktaları ve zaman alanları var. 21 Ağustos snapshot'ı:

- 330 kesinti-ilçe satırı
- 290 benzersiz kesinti
- 330/330 dolu CBS koordinatı
- 311/330 dolu adres-nokta listesi
- 0 satırda trafo kodu
- 0 satırda trafo tipi
- arayüz kodunda `Kesintiden_Etkilenen_Abone_Sayisi` desteği var, fakat planlı
  kesinti yanıtı bu alanı fiilen göndermiyor

`CBS_Koordinat` **kesinti noktasıdır**. Trafo kodu olmadığı için yarışmadaki
`tanim` ile eşleşmeden trafo koordinatı olarak kullanılamaz.

Çekici:

```powershell
python scripts/fetch_gdz_kesinti_cbs.py
```

Çıktı: `data/research/gdz_kesinti_cbs_snapshot.parquet`

## Diğer kaynakların hükmü

### TEDAŞ numaralama şartnamesi

Şartname, `TRAFOBİNATİP` katmanında X/Y/Z koordinatı, mevcut ad, yeni kod,
tip (beton, sac, monoblok, bina içi, TRP), özellik (TM/DM/İM/KÖK/trafo),
mülkiyet ve besleyen fideri; trafo katmanında dahili/harici, cins, güç, marka,
seri no ve imal tarihini zorunlu alanlar olarak tanımlıyor. Abone katmanında da
`Beslendiği Trafo` ilişkisi var. Yani istenen üç bilgi kurum CBS'sinde gerçekten
mevcut.

Kaynak: [TEDAŞ Elektrik Dağıtım Şebekesinin Numaralama İşleri Teknik
Şartnamesi](https://sedas.com/Home/TenderFileDownload?fileId=29348), özellikle
sayfa 9–13. Aynı belgenin gizlilik bölümü teknik ve abonelik bilgilerinin üçüncü
kişilere açıklanmamasını öngörüyor.

### Türkiye Ulusal CBS sınıflandırması

2026 CBS Kurulları karar kitabında elektrik-trafo verisi `Hizmete Özel` olarak
sınıflanıyor. Bu da ulusal açık WMS/WFS'den neden trafo envanteri çıkmadığını
açıklıyor.

Kaynak: [Türkiye CBS Kurulları Kararları
Kitabı](https://webdosya.csb.gov.tr/v2/cbs/2026/02/Cbs_Kararlar_Kitabi_2026-0116-BASKI_trimmed-20260205120726.pdf).

### OpenStreetMap

İzmir–Manisa geniş kutusunda `power=transformer` sorgusu 64 nesne verdi;
yalnızca 3 nesnede `ref`, 5 nesnede `name` vardı ve 7.368 yarışma `tanim`
değerinin hiçbiriyle doğrudan eşleşmedi. Kapsama, yarışma trafolarının çok küçük
bir bölümüdür; toplu en yakın-komşu eşleştirmesi güvenilir değildir.

### GDZ yatırım PDF'leri

2016–2025 yatırım listelerindeki trafo/DM/KÖK adları ve mahalleler tarandı.
Belgeler `M-3115`, `700096...` gibi yarışma kodlarını taşımıyor; eski işletme
adlarını (`M-...`, `K-...`, `TR-...`) kullanıyor. 7–9 haneli yarışma kodlarıyla
doğrudan eşleşme sıfır.

Kaynak: [GDZ yatırım
projeleri](https://www.gdzelektrik.com.tr/bilgi-merkezi/yasal-bildirimler/yatirim-projeleri).

## Yarışma açısından doğru karar

1. Kesinti CBS snapshot'ını günlük biriktir; ileride açık bir trafo/ref alanı
   eklenirse geçmiş noktalar hemen değer kazanır.
2. Organizatörden yalnız statik ve anonimleştirilmiş `tanim → koordinat grid'i,
   abone sayısı bandı, tip` eşlemesi istenmeli. Bu üç alan kurum CBS'sinde var;
   teknik dayanak şartnamenin sayfa 9–13'üdür.
3. Mevcut durumda `nüfus / ilçe trafo sayısı` abone vekili kullanılabilir; bunu
   gerçek abone sayısı diye sunmamak gerekir.
4. `guc` değerinden beton/sac/monoblok/TRP tipi kesin çıkarılamaz. Kapasiteye
   göre tip etiketi üretmek veri uydurmak olur.
5. Hedef dönemindeki gerçekleşmiş kesintileri sonradan özellik yapmak yarışma
   kuralı ve zaman sızıntısı açısından kullanılmamalıdır. Buradaki snapshot
   yalnız statik envanter araştırması içindir.
