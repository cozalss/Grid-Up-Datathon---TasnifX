# KTB/YİGM açık istatistik kaynak haritası

Kültür ve Turizm Bakanlığı Yatırım ve İşletmeler Genel Müdürlüğü'nün
(yigm.ktb.gov.tr) turizm istatistiklerinin **doğrulanmış** haritası.
2026-08-17'de ham HTML'den `grep` ile çıkarıldı (LLM özetine değil,
`/Eklenti/` bağlantılarına dayanır). Sunucu hızlı ardışık istekte
bağlantıyı keser: istekler arası **6 sn** bekle, `.xls` için `xlrd` gerekir.

Tüm yollar `https://yigm.ktb.gov.tr/` köküne eklenir.

## Bağlı olanlar (kod çekiyor)

| Seri | Betik | Sayfa | Kapsam |
|---|---|---|---|
| Bakanlık belgeli konaklama, **yıllık il-ilçe** | `scripts/fetch_turizm.py` | TR-208783 | 2023–2025 xlsx |
| Bakanlık belgeli konaklama, **aylık il** | `scripts/fetch_turizm_aylik.py` (`BULTENLER`) | TR-201125 → yıl sayfaları TR-232592…TR-448570 | 2019-01…2026-06 |
| Belediye belgeli konaklama, **aylık il** | `scripts/fetch_turizm_aylik.py` (`BELEDIYE_BULTENLER`) | TR-201128 → TR-232595…TR-311439 | 2019-01…**2022-10** (sonra Bakanlık serisine katıldı) |

## Kapsam kırılmaları — üç kaynaktan doğrulandı

1. **2022-09**: ölçülen örtük yatak sıçraması (Türkiye 1,04M→1,24M).
   Kapasite bültenlerinde "basit konaklama" sütunları **ilk kez Eylül
   2022**'de görünür (TR-275534). Metaveri (TR-201124) 7334 sayılı Kanun'la
   belediye belgeli tesislerin basit belgeye geçtiğini ve "2022 Kasım'dan
   itibaren" Bakanlık serisinde yayımlandığını yazar — veri Eylül'de
   başlamış, başlık Kasım'da değişmiş. **2021-10…2022-08** arasında yeni
   basit belgeler "işletme belgeli" tesis sayımına karışmış (İzmir 225 →
   1007 tesis, Eylül'de 246 + 1292 basit olarak ayrışır) — o dönemin
   işletme sayımına güvenme.
2. **2025-07**: örtük yatak Türkiye 1,46M→1,73M, Muğla 133k→219k, doluluk
   sabit; başlık değişmedi. Turizm amaçlı konut kiralama belgelendirmesiyle
   uyumlu; kapasite bültenleriyle ayrıca doğrulanabilir (aşağıdaki harita).

`turizm_aylik_il.parquet` içinde `kapsam_rejimi` (1/2/3) bu kırılmaları
taşır. `*_tum_belgeli` = bakanlık + belediye: **ölçüldü** — 2022-09 dikişini
kapatır (Türkiye örtük yatak oranı 1,04; bakanlıkta 1,19) ama 2022-11'de
tersine düşer (1,44M → 1,03M): belediye belgeli tesislerin çoğu basit belgeye
geçmemiş, istatistikten çıkmış. Muğla Ağustos yatak: bakanlık 82k→125k
(2022→23, +%52), tüm 154k→125k (−%19). **Kusursuz sürekli seviye serisi
yok**; 2019–2022 gerçek turist yükü için `tum_belgeli` daha yakın (Muğla'da
belediye pansiyonları +%40), mevsim şekli için `doluluk`, seviye kıyası için
`kapsam_rejimi` şart.

## Bağlı olmayan, doğrulanmış adaylar

### A · Bakanlık belgeli tesis KAPASİTESİ (tesis/oda/yatak, aylık, il)

Sayfa TR-201136 → arşiv TR-275534 (yıl sayfaları TR-407327…TR-449119).
2018-12…2026-07; **2019-12 eksik** (2020 sayfasındaki ilk bağlantı 31.01.2019
dosyasının kopyası), **2026-06 sayfada yok**.

- 2022-07'ye kadar dosyalar "GG.AA.YYYY tarihi itibarıyla" anlık görüntü:
  ayın 7–13'ündeki görüntü **bir önceki ayın** durumudur (08.08.2022 vs
  "Ağustos 2022" karşılaştırmasıyla doğrulandı).
- 2022-08'e kadar 7 sütun (işletme + yatırım belgeli × tesis/oda/yatak),
  eski `.xls`, sayfa `Sheet0`; satır sırası bayt sırası (İzmir/Şanlıurfa/
  Şırnak sonda) — **ada göre birleştir**, konuma göre değil.
- 2022-09'dan itibaren 10 sütun (+ basit konaklama), sayfa `Sayfa1`,
  dipnot "Veriler geçicidir".

Değeri: `geceleme / (yatak × gün)` bültenin doluluk oranını yeniden üretir;
kırılma tarihlerini doğrular. Panel için düşük marjinal değer (yavaş stok
değişkeni). Ayrıca belediye belgeli kapasite: TR-275524, 2018-12…2022-08,
4 sütun.

Referans ay → dosya (kısaltılmış; tam liste ajan raporunda, gerekirse
sayfadan yeniden çıkar):

```
(2019,5) 63687,10062019-tarihi-itibariyla-bakanlik-belgeli-tesis-istatsitiklerixls.xls
(2021,6) 83388,09072021-tarihi-itibariyla-bakanlik-belgeli-konaklama-tesisi-istatistiklerixls.xls
(2022,8) 102597,agustos-2022--bakanlik-belgeli-konaklama-tesisi-istatistiklerixls.xls
(2022,9) 111861,eylul2022bakanlikbelgelixlsx.xlsx      <- basit sutunlari ilk kez
(2022,10) 111862,ekim2022bakanlikbelgelixlsx.xlsx
(2022,11) 111863,kasim2022bakanlikbelgelixlsx.xlsx
(2025,7) 136968,2025temmuzbakanlikbelgelixlsx.xlsx
(2026,7) 150386,2026-temmuz-bakanlik-belgelixlsx.xlsx
```

### B · Sınır istatistikleri (yabancı ziyaretçi girişleri, aylık)

Sayfa TR-249702 → TR-249704 (yıl sayfaları TR-249706 (2016) … TR-448060
(2026)); yıllık bültenler TR-249709. 2016-01…2026-06, hepsi eski `.xls`
(3,5–5 MB).

- Aylık dosyada `İl ve Taşıt(Ay)` sayfası: **sınır kapısının bağlı olduğu
  il** × taşıt (deniz/hava/kara/tren) — varış ili DEĞİL. Manisa yok (kapı
  yok), Denizli ~0. Örnek 2026-06: İzmir 169.491, Muğla 504.724, Aydın
  146.062.
- **Yıllık bültende `T5-SınırK.-Ay Göre G.Yabancı`**: kapı × ay. Kapılar
  ilçeye eşlenir: Aydın — Kuşadası, Didim; Denizli — Çardak; İzmir — Çeşme,
  Aliağa, Adnan Menderes (Gaziemir), Dikili, Foça, Seferihisar; Muğla —
  Bodrum, Turgutreis, Dalaman, Göcek, Milas-Bodrum, Datça, Marmaris,
  Fethiye, Yalıkavak, Güllük, Bozburun. 2025: Dalaman 1,81M, A. Menderes
  1,45M, Milas-Bodrum 0,91M, Kuşadası 0,94M.
- Yayın gecikmesi ~t+25 gün. Yalnızca yabancı; yerli hareketi yok.

Değeri: konaklama serisinin yabancı bileşeni için bağımsız çapraz kontrol;
ilçe×gün paneli için havalimanı/liman ilçelerine doğrudan eşlenen aylık
yabancı giriş sinyali. **Gün-1 sonrası aday.** Yıllık xlsx'ler:

```
2016 53123 · 2017 67753 · 2018 63274 · 2019 72101 · 2020 81888
2021 93664 · 2022 111370 · 2023 122231 · 2024 131301 · 2025 145395,2025yilliksinirbultenixlsx.xlsx
```

Aylık 2026: `144878,ocak-2026-haber-bultenixls.xls`, `145624,s-bat-2026-…`,
`146347,mart-2026-…XLS`, `147906,nisan--2026-…XLS`, `148357,mayis-2026-bultenixls.XLS`,
`150184,haziran-2026-bultenixls.xls`.

### C · Belediye belgeli konaklama, yıllık il-ilçe (2000–2022)

TR-211090; 2017–2022 xlsx: `60356`, `74186`, `74187`, `119761`, `119762`,
`119763`. Yıllık ilçe tablomuz 2023'te başladığı için 2019–2022 ilçe
kırılımına ancak bu + Bakanlık yıllık arşiviyle (TR-201126, çoğu .rar)
ulaşılır. Bağlanmadı.

## Metaveri alıntıları (doğrulanmış)

- TR-201124 (Bakanlık, güncelleme 23/12/2024): "…belediye belgeli olarak
  faaliyet gösteren konaklama tesislerine Basit Konaklama Turizm İşletmesi
  Belgesi alma zorunluluğu getirilmiştir. Belgelendirme süreci tamamlanan bu
  tesislere ilişkin bilgiler 2022 yılı Kasım ayından itibaren Bakanlık
  Belgeli Konaklama İstatistikleri olarak yayımlanmaktadır."
- TR-201130 (Belediye, 23/01/2019): "Doluluk oranı: … kullanıma sunulan
  yatak kapasitesinin ay boyunca ne ölçüde kullanıldığını gösteren
  değişkendir." Yanıt vermeyenler il/ilçe/yatak-büyüklüğü gruplarıyla
  **imputasyonla** doldurulur (Bakanlık serisinde de aynı 5 adımlı yöntem).
  Yani seriler ölçüm değil, kısmen tahmin içerir.
