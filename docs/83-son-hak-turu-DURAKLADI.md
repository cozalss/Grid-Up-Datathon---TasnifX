# 83 - SON HAK TURU (1 Eylul 16:40 TSS) -- KULLANICI TARAFINDAN DURDURULDU

Devam etmek icin: bu belgeyi + docs/81 + docs/82 oku. Workflow yeniden baslatilabilir
(asagida). GONDERIM YAPILMADI, submissions/ degismedi, commit yok.

## 1. OLCULEN GERCEKLER (bugun, canli API + kendi hesaplarimiz)

- **1 hak kaldi.** `kaggle competitions submission-limits`: today 2, remaining 1,
  max_daily_submissions=3. Bitisten once yeni kota penceresi ACILMIYOR.
- **Deadline 2026-09-01 23:59 UTC = 2 Eylul 02:59 TSS.**
- **Son secim N=2 TEYIT EDILDI** (kullanici tarayicidan okudu). Yani
  `submissions/tuketim_YP_seviye.csv` (1.00115) her kosulda secili kalir ->
  **asagi yonlu risk YOK**; son hak beklenen skoru degil **P(2. sira)**'yi maksimize
  etmeli.
- **LB (1 Eyl 13:26 UTC):** 1. 0.97654 | 2. 0.99413 | 3. 0.99475 | 4. 0.99487 |
  5. 0.99496 | 6. 0.99502 | 7. 0.99556 | **16. biz 1.00115**.
- 2. sira suruklenmesi: 0.99536 (30 Agu) -> 0.99518 (31 Agu) -> 0.99413 (1 Eyl).
  **Bitis tahmini 0.9927-0.9934, merkez 0.99310.**
- 2. olmak = 0.00143'luk bantta sikismis **ALTI takimi** private'ta gecmek.
- **Tasarim hedefi T = 0.9931 -> kappa* = 0.12298 (gereken rho >= 0.1230).**
  Kabul alt siniri T = 0.99413 -> kappa* = 0.11436 (rho >= 0.1144).
- Public/private bolunme gurultusu (gercek CV artiklarindan): sigma ~ 0.0011
  (aralik 0.0008-0.0018). Public 0.9940 -> P(2.) ~ %11; 0.9931 -> ~%39.
  Bugunku 1.00115 hedeften **6.4 sigma** uzakta; gurultu kurtarmaz.
- "public %50" iddiasi KAYNAKSIZ varsayim; dolayli kanit f>=0.2 diyor. Karar buna
  duyarli degil (f kucukse gurultu buyur, lehimize).
- p34 geometrisinin kendi ongoru hatasi ~1e-3 (H1 -0.00098, H2 +0.00110) --
  bolunme gurultusunun 3-6 KATI. Hakim belirsizlik MODEL HATASI.

## 2. p34_son_hak.csv (kuyruk kapagi k=2) CURUTULDU -- GONDERME

1. **Kanit tabani %44 sisik.** p34 "H1'de kapak ekseninin span-disi rho'su +0.0155
   olculdu" diyor. H1 birebir yeniden uretildi (fark 0.0): H1'e kapak DISINDA bilerek
   `0.019344 * u_Y1` da eklenmis. Span-disi ic carpimin **%33.8'i o terimden**.
   **Kapagin kendi LB rho'su = +0.010739.** (Kodun yorumundaki "+0.0164" hicbir
   dosyada yok.)
2. **CV->LB tasima orani 0.185.** CV havuz k=8 -> +0.0581; LB -> +0.0107.
3. **MEVSIMSEL IKIZ (en onemli metodoloji bulgusu):**
   test penceresi 2026-04-01..07-31, **yaz25 blogu 2025-04-01..07-31 -- birebir ayni
   takvim araligi (122 gun).** Kapak ailesinin rho'su tam o blokta SIFIR
   (k=8: +0.0021 +-0.0042; k=2 dik bilesen: +0.0212). Umut baglanan +0.07..+0.12
   degerleri test'e mevsimsel olarak EN UZAK iki bloktan (guz25/kis26) geliyor.
4. Aile tavani ~0.02-0.03. Tasima orani 1 varsayilsa bile en iyimser blok 0.99600
   (~7. sira). **P(2. sira) pratikte 0.**
5. Soguk taraf kapali: soguk satirlarin kesim oncesi gecmisi **tam sifir** ->
   trafo-bazli kapak TANIMSIZ. Kapasite-ofset kapagi soguk tarafta havuzda
   -0.055..-0.117 (ZARARLI). Kapasite-ofset uzayi sicak tarafta da olu.

### Bundan cikan SECIM OLCUTU (bundan sonraki her yon icin)
**Bir yonun rho'sunu HAVUZDA degil `yaz25` blogunda olc.** Havuz, test'e alakasiz
iki mevsimin sinyalini karistirip yaniltiyor. yaz25 mevsimsel ikizdir.

## 3. YARIM KALAN OLCUMLER (workflow durduruldugunda kosuyordu)

- `zemin:sifir-cebi-denetimi` (EN ONEMLI): docs/82 §4 ile §6 birbiriyle celisiyor.
  §6 "sifir cebi MSE'nin %41-53'u", §4 "yakalananlari sifirlamanin kahin degeri
  yalnizca +0.0005..+0.0052". Ikisi ayni anda dogru olamaz. Ayrica "16 varyant 0/3"
  hukmu **CV bilesiginde KAZANC** olcumu; kazanc = -2*kappa*rho + kappa^2 ve tam
  sifirlama cok buyuk kappa demektir -> pozitif rho bile negatif kazanc verir.
  Olculmesi gereken sey kazanc degil **birim yon basina rho**. Yarim kaldi.
- `zemin:taban-cebiri`: LOO geri-uyum sinavi + kappa* tablosu. Yarim kaldi.
  (p34 degerleri: MSE_taban durust 1.0013719 -> skor 1.00069; cebirsel 1.0011824.)

## 4. DEVAM KOMUTU

Workflow betigi:
`C:\Users\Cem\.claude\projects\C--Users-Cem-Desktop-Datahon-Laptop-Grid-Up-Datathon---TasnifX\d8509f77-6f9b-4e1d-b980-62e299ed4fc5\workflows\scripts\son-hak-2sira-wf_0dc32cba-3c1.js`
Run ID: `wf_0dc32cba-3c1`

Yeniden baslatilirsa BITEN iki ajan (yarisma mekanigi, kuyruk kapagi) onbellekten
doner; yarim kalan ikisi bastan kosar. Betige eklenmesi gereken: yukaridaki
**yaz25 mevsimsel ikiz olcutu** ve **p34 kapak adayinin curutuldugu**.

## 5. KURAL
ONAY OLMADAN GONDERIM YOK. `kaggle competitions submit` yalnizca kullanici calistirir.
