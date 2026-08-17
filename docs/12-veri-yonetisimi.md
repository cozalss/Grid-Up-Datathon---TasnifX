# DATA_PRIVACY ve veri yonetisimi

Bu belge Grid Up calisma alaninda verinin neden alindigini, ne kadarinin
tutuldugunu, nasil dogrulandigini ve bir olayda ne yapilacagini tanimlayan
uygulanabilir sozlesmedir. Yarisma verisinin kendi kullanim kosullari bu
belgeden ustundur.

## Kapsam ve mahremiyet siniri

Sistem yalnizca elektrik dagitim modellemesi icin gereken zaman, il/ilce veya
konum bazinda toplulastirilmis kayitlari kullanir. Mevcut kaynaklarda kisi adi,
kimlik numarasi, telefon, e-posta, acik adres, tekil sayac/abone kimligi veya
kisi duzeyi hareket verisi yoktur. Kesinti verisindeki etkilenen abone sayisi
ve mahalle bilgisi olay duzeyinde toplulastirilmis operasyonel bilgidir; kisi
kaydi degildir.

Yeni bir kaynak kisiye ait ya da yeniden kimliklendirmeye elverisli alan
icerirse otomatik olarak kapsama alinmaz. Indirme durdurulur, dosya karantina
altina alinir ve amac, hukuki dayanak, erisim ile saklama karari yazili olarak
onaylanmadan pipeline'a baglanmaz. Gizli degerler `.env` icinde kalir; log,
metadata, notebook, hata mesaji ve surum kontrolune yazilmaz.

## Amac ve veri minimizasyonu

Izin verilen amaclar sunlardir:

- elektrik yuku/kesinti tahmini icin hava, takvim ve bolgesel sinyal uretmek;
- il/ilce anahtarlarini dogrulamak ve mekansal komsuluk kurmak;
- model performansi, sizinti ve veri kalitesi kontrollerini yeniden uretmek;
- yarisma teslimini ve bilimsel karar kaydini denetlenebilir kilmak.

Bir kolon bu amaclardan birine baglanamiyorsa alinmaz. API yanitlarindan sadece
gerekli alanlar normalize edilir; serbest metin, ham HTTP basliklari, cookie,
token ve gereksiz saglayici alanlari turetilmis tablolara tasinmaz. Model icin
ilce yeterliyse daha ayrintili konum tutulmaz. Kisi duzeyi zenginlestirme ve
farkli amacla yeniden kullanim yasaktir.

## Veri siniflari ve saklama

| Sinif | Ornek | Konum | Saklama kurali |
|---|---|---|---|
| Yarisma girdisi | train/test/sample submission | `data/raw/` | Yarisma kosullari; proje bitisinde erisim gozden gecirilir |
| Mutable harici ham | KTB XLSX | `data/external/ham/` | Son kullanilan surum + kanit metadata; yarisma bitisinden 90 gun sonra silme incelemesi |
| Dogrulanmis turetilmis | hava, kesinti, turizm, ilce parquet/CSV | `data/external/`, `data/reference/` | Yeniden uretilebilir oldugu surece en gec ham veriyle ayni inceleme tarihi |
| Provenance kaniti | SHA-256, kaynak, sema, satir sayisi, lisans kaydi | yan metadata ve `data/sources.yml` | Veri silinse de denetim icin proje arsiv omru boyunca |
| Sirlar | API kimlik bilgileri | yalniz `.env`/ortam | Gereksinim biter bitmez iptal; asla surum kontrolunde degil |

Saklama incelemesinde artik gerekli olmayan veri guvenli bicimde silinir;
metadata kaydi silinen dosyanin hash'ini ve silme tarihini koruyabilir. Yedek
ve paylasilmis kopyalar ayni sureye tabidir.

## Provenance ve yayin sozlesmesi

CSV, parquet ve JSON ana ciktilari hedef dosyaya dogrudan yazilmaz. Yayin akisi
ayni dizindeki gecici dosyaya yazma, `flush`, `fsync` ve atomik `os.replace`
adimlaridir. Yayinlanmadan once gerekli kolonlar ile asgari satir sayisi
dogrulanir. Basarili yayindan sonra yan metadata su kanitlari tasir:

- SHA-256 ve bayt boyutu;
- tam kolon sirasi ve satir sayisi;
- kaynak tanimi ve UTC olusturma zamani;
- metadata sema surumu ve dosya formati.

Mutable bir indirme veya devam-etme cache'i yan metadata yoksa, kaynak
degismisse, hash/boyut uyusmuyorsa ya da yeniden okunan tablo sema/satir
kontrolunden gecmiyorsa kullanilmaz. Eski hedef, yeni veri tum kontrolleri
gecene kadar korunur. Veri dosyasi yayinlanip metadata yazimi kesilirse cache
bir sonraki kosuda fail-closed reddedilir.

Kaynak URL'si, erisim tarihi, yerel hash, lisans/atif ve donusum yolu
`data/sources.yml` ile birlikte tutulur. Bir URL'nin acik olmasi verinin serbest
lisansli oldugu anlamina gelmez. Lisans belirsizse dosya yeniden dagitilmaz;
yalniz izin verilen erisim ve kullanim sinirlarinda yerel olarak islenir.
Notebook ve sunumlarda kaynak atfi korunur.

## Erisim ve paylasim

Ham ve turetilmis veri yalniz proje ekibinin ihtiyac duyan uyelerine acilir.
Repository, notebook ciktilari, hata loglari ve submission paketleri sir veya
gereksiz ham veri icermemelidir. Harici paylasimdan once en az su kontroller
yapilir:

1. kisiye ait veya yeniden kimliklendirmeye elverisli alan yok;
2. kaynak lisansi ve yarisma kurali yeniden dagitima izin veriyor;
3. dosya hash'i kayitli ve sema beklenen sema;
4. paylasim paketi yalnız gerekli dosyalari iceriyor.

## Veri olayi yonetimi

Bozuk hash, beklenmeyen sema, eksik gun/konum, yetkisiz erisim, sir sizintisi
ve lisans ihlali veri olayi sayilir. Olayda:

1. indirme, yayin ve model kosulari durdurulur; supheli dosya kullanilmaz;
2. dosya silinmeden once hash, zaman, kaynak ve etkilenen kosular kaydedilir;
3. token/sir etkilenmisse derhal iptal edilip yenilenir;
4. etkilenmis cache, model, submission ve paylasilmis kopyalar belirlenir;
5. temiz kaynak yeniden indirilir, sema/satir/hash kontrolleri calistirilir;
6. kok neden, kapsam, duzeltme ve tekrar-onleme karari olay kaydina yazilir;
7. lisans veya kisi verisi suphesi varsa yeniden dagitim, yetkili inceleme
   tamamlanana kadar durur.

Bir hata sessizce bos tabloya, eski cache'e veya kismi yayina dusurulmez.
Olay kapatilmadan once ilgili testler, kaynak manifesti ve turetilmis ciktilar
yeniden dogrulanir.

## Sorumluluk ve gozden gecirme

Veriyi ekleyen kisi kaynak, amac, minimum sema, satir esigi, lisans ve saklama
kaydindan sorumludur. Model kosusunu yapan kisi kullanilan dosyalarin hash ve
provenance kaydini korur. Bu belge yeni kaynak eklendiginde, yarisma kosullari
degistiginde, bir veri olayi sonrasinda veya en gec proje kapanisinda yeniden
gozden gecirilir.

## Lisans ve yeniden dagitim incelemesi (17 Agustos 2026)

Sekiz harici artefaktin tamami tek oturumda incelendi. Sonuclar
`data/sources.yml` icine `verification` alani olarak yazildi; `security/
verify_sources.py` bunlari her kosuda dogrular.

| Artefakt | Lisans | Dayanak |
|---|---|---|
| `hava_gunluk.parquet` · `hava_saatlik_turev.parquet` | CC-BY-4.0 | Open-Meteo yayimlanmis acik lisans |
| `yanginlar.parquet` | NASA-EOSDIS-open | NASA FIRMS SSS: yeniden dagitim acikca serbest, atif isteniyor |
| `ilceler_gdz_adm.parquet` · `.csv` | MIT | `ubeydeozdmr/turkiye-api` deposunun SPDX lisansi (GitHub API ile dogrulandi) |
| `gunes_gunluk.parquet` | MIT | Kendi turetilmis eserimiz: pvlib (BSD-3) ile MIT koordinat tablosundan hesaplandi |
| `depremler.parquet` (AFAD) | kurum teyidi | **Yayimlanmis acik lisans YOK.** Yeniden dagitim izni kurumdan dogrudan alindi |
| `turizm_geceleme.parquet` (KTB/YIGM) | kurum teyidi | **Yayimlanmis acik lisans YOK.** Yeniden dagitim izni kurumdan dogrudan alindi |

**Iki statunun farki onemlidir ve karistirilmamalidir.** "Yayimlanmis acik
lisans", herkesin okuyup dogrulayabilecegi bir metindir. "Kurum teyidi", bu
ekibe verilmis belgelenmis bir izindir; SPDX ile tanimlanabilir bir hak devri
degildir ve ucuncu bir tarafa otomatik gecmez. AFAD ve KTB satirlari manifestte
bu ayrimi acikca tasir.

### Neden 6 uyari hala duruyor ve KAPATILMAMALI

`verify_sources.py` calistiginda **0 hata, 6 uyari** doner. Alti uyarinin
tamami tek ve ayni sebeptedir: `source.immutable=false` -- yani ust kaynak
canli bir HTTP servisidir ve bugun ayni sorguyu tekrarlarsak farkli cevap
alabiliriz.

Bu uyari DOGRUDUR ve susturulmamalidir. Kaydettigi sey su: dagittigimiz sey,
degisebilen bir kaynagin hash ile sabitlenmis bir anlik goruntusudur. Etiketi
`true` yapmak bu bilgiyi yok eder ve manifesti yalan soyler hale getirir.
Uyariyi durustce kapatmanin tek yolu ham cevabi arsivleyip `snapshot_ref`i o
arsive baglamaktir. Yarisma penceresinde bu yapilmadi; karar bilincli.

Uyari metni 17 Agustos'ta kesinlestirildi: onceden dort farkli kosul tek bir
"lisans/yeniden dagitim/immutable incelemesi acik" mesajina dusuyordu ve
"lisans hic bilinmiyor" ile "lisans tamam, yalnizca ust kaynak degisebilir"
ayirt edilemiyordu. Kapinin katiligi aynidir; yalnizca hangi kosulun bozuk
oldugunu artik soyluyor.
