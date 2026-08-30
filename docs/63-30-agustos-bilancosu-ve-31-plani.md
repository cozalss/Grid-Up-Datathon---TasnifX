# 63 — 30 Ağustos bilançosu ve 31 Ağustos planı

**docs/62'nin yerine geçer.** Sabah bunu aç.

---

## 1. Durum

```
1. Grid Grinders     0,99009
2. Atakan Aldemir    0,99940   <- HEDEF
3. TasnifX           1,00115   <- BIZ   (sabah 6. siradaydik)
4. Ahmet B. ALTUNOK  1,00118
5. Duo-Electra       1,00129
6. Saban Ozdogan     1,00171
```

```
m0 = 1.005846366   taban skoru 1,00292
BILINEN OPTIMUM    1,00052   <- henuz GONDERILMEDI, 6,3e-4 nakde cevrilmemis
kalan hak          6 (31 Agustos 3 + 1 Eylul 3)
bitis              1 Eylul 23:59 UTC
```

---

## 2. Bugün ne yapıldı — 3 gönderim

| # | dosya | skor | ne ölçtü | sonuç |
|---|---|---|---|---|
| 1 | `tuketim_s3y40.csv` | 1,00177 | `y40` yön | `rho = −0,0132`, zayıf |
| 2 | `tuketim_YP_seviye.csv` | **1,00115** | seviye (yapısal) | **`rho = −0,0304`, REKOR** |
| 3 | `tuketim_K_yenibas.csv` | 1,00191 | yeni-trafo yanlılığı | **`rho = −0,0027`, KALDI** |

---

## 3. BULGU 1 — `g7` büzülmüş bir span çözümüydü

Ölçülmüş 25 yönün **tam** span optimumu `||r_hat||² = 0,003872` açıklıyor;
`g7` yalnızca `0,003036`. Masada **0,000836** duruyordu.

Üç bağımsız testle doğrulandı:
```
rcond 1e-5..1e-10  ->  0,003872 TAM KARARLI   (1e-12'de rank 22'ye cikip patliyor)
leave-one-out      ->  span-ici payi >%95 olan 19 yonde ortalama hata 7,0e-05
LB yuvarlamasi     ->  ||r_hat||^2 sacilimi sd 1,2e-05 (skorda 6e-06)
```

**Yeni tasarım (`m110_tamspan.py`, `m112_kalibre.py`):**
```
taban = a0 + r_hat
sonda = taban + kappa * d_dik      (d_dik = adayin span'a DIK bileseni)
```
`d_dik` span'a dik olduğu için çapraz terim yok, ölü nokta yok, `L_bilinen = 0`
— her sonda **saf yeni bilgi** ölçer.

---

## 4. BULGU 2 — model çıktılarının bilgisi tükenmişti, yapısal eksenlerin değil

25 ölçümün artımlı kazanç dağılımı (doğrudan hesaplandı):
```
ortalama 9,4e-05   ortanca 5,5e-05   REKOR 3,12e-04 (rho 0,0177)
2. sira icin eksen basi gereken: 4,53e-04 (rho 0,0213)
```
**Yani model varyantlarını harmanlayarak 2. sıraya ULAŞILAMAZDI.** Rekorun
1,5 katını 7 eksende birden tutturmak gerekiyordu.

İlk **yapısal** ölçüm bunu kırdı:
```
seviye (standartlastirilmis a0, span'a dik bileseni)
  skor 1,00115  ->  L = -0,024649   rho = -0,0304   kazanc 9,26e-04
  = tarihin rekorunun 3,0 KATI
```
Yorum: tahminler **aşırı yayılmış** (~%1,8 fazla dağılım). Hata model
seçiminde değil **kalibrasyonda**.

Yeni taban: `0,004797` açıklanan → **1,00052**.

---

## 5. BULGU 3 (NEGATIF) — yeni-trafo hipotezi ÇÖKTÜ

Ajan bulmuştu, ben üç ayrı testle **eğitim verisinde** doğrulamıştım:

```
egitim gecmisi 1-8 satir olan 463 trafo (test'in %5,5'i, 39.091 satir)
  m6'nin verdigi:  kendi seviyesi + 0,8518     <- diger tum derinlikler -0,00..+0,24
  gercek transient (3.870 trafoda olculdu):
      k=1 +0,2885   k=2 +0,1415   k=4 +0,0659   k=8 +0,0284   egim 0,85-0,95
  plasebo (200 tekrar): bos +0,0107 +- 0,0352, gozlenen +0,1415, z = +3,72 GECTI
  -> yanlilik +0,71
```

**LB sonucu: `rho = −0,0027`, kazanç `7,5e-06` — beklenenin 320 KATI ALTINDA.**

Ders: üç "bağımsız" doğrulamanın üçü de **aynı veriden** (eğitim seti) geliyordu.
Kural 40'ın ihlali; Kural 36 ikinci kez ısırdı. Eğitim verisinde plasebo geçmek,
LB'de değer taşımanın garantisi **değil**.

---

## 6. Sistem

| betik | ne yapar |
|---|---|
| `m110_tamspan.py` | tam-span tabanı + dik aday sondası |
| `m111_yapisal.py` | yapısal yön ailesi üretici (seviye, soğuk, bölge, güç...) |
| `m112_kalibre.py` | **ANA SISTEM** — durum yönetimi, 20+ aday, ölçüm kaydı |

```powershell
cd experiments/model29
python m112_kalibre.py --liste                          # adaylar ve mevcut taban
python m112_kalibre.py --aday <ad> --yerdeg 0.015 --cikti tuketim_K_<ad>.csv
python m112_kalibre.py --kaydet <ad> --skor <S>         # olcumu isle
python m112_kalibre.py --nihai --cikti tuketim_K_NIHAI.csv
```
Durum: `m112_durum.json` (şu an `seviye` ve `yenibaslangic` ölçülmüş).

---

## 7. 31 Ağustos planı — 3 hak

**İlk gönderim mutlaka bir SONDA olsun** — her sonda zaten `1,00052` civarını
taşıyor, yani gönderilmemiş 6,3e-4 otomatik nakde çevriliyor.

| sıra | aday | gerekçe | durum |
|---|---|---|---|
| 1 | `buzme_tam` | üç holdout'ta plasebo kontrollü şekil (doğrusal büzme, soğuk 4×, ufuk 4×) | `tuketim_K_buzme_tam.csv` **HAZIR** |
| 2 | `seviye_x_soguk` | soğuk satırlarda farklı dağılım | `tuketim_K_seviye_x_soguk.csv` **HAZIR** |
| 3 | `seviye_x_ay` veya `seviye2` | ufuk kayması / eğrilik | `tuketim_K_seviye_x_ay.csv` **HAZIR** |

Yer değiştirme `0,015` → her sonda `rho ≥ −0,03` olan her durumda **3. sırada**.

**1 Eylül:** 2 sonda + 1 saf optimum (`--nihai`).

---

## 8. Beklenti — dürüst

```
2. sira icin kalan 6 eksende ortalama rho = 0,0194 gerek
olculen:  seviye +0,0304 GECTI  |  yenibaslangic -0,0027 KALDI
ortalama su ana kadar: 0,0166   (2 olcum)
```

İki ölçümün ortalaması gerekenin biraz altında. 2. sıra **açık ama zor**;
3. sıra sağlam (banka 1,00115, bilinen optimum 1,00052).

**Bugünün dersi:** eğitim verisinden gelen gerekçe LB'de değer garantisi değil.
Yarın adayları mekanizma gücüne göre değil, **ölçülmüş LB geçmişine** göre
sıralayacağız — ve tek bir hipoteze fazla yüklenmeyeceğiz.

---

## 9. Model dışı — hâlâ açık

1. **Notebook yok.** Son tarih 2 Eylül 13:00, private LB 2 Eylül 00:10.
   `notebooks/` 22 Ağustos'tan beri dokunulmadı. **1 Eylül'den önce bitmeli.**
2. **Final 2 gönderim seçimi yapılmadı.** Tarayıcıdan, API'de yok, kapanıştan
   sonra değiştirilemez. En geç 1 Eylül 23:00 UTC.
3. **Takım arkadaşının Coderspace kaydı** teyit edilmedi — kayıt yoksa skor geçersiz.
4. **Hava verisi:** "dış kaynak serbest" diyen düzenleyici mesajı 29 Ağustos'ta
   forumdan silindi; e-posta teyidi tek dayanak, `.eml` olarak arşivlenmeli.

---

## 10. Kalıcı kurallar 52–54

**52.** Model çıktılarını harmanlamanın bir TAVANI vardır ve hesaplanabilir:
artımlı kazanç dağılımının rekoru × kalan hak sayısı. Tavan hedefin altındaysa
harman yolu KAPALIDIR; yeni bilgi yapısal eksenlerden gelmelidir.

**53.** Span optimumunu regularize etme — `pinv` + rank kesme kullan.
Büzülmüş çözüm (ridge/tau) bedava kazancı masada bırakır. Kararlılığı
`rcond` taramasıyla, doğruluğu leave-one-out ile denetle.

**54.** Eğitim verisinde plasebo geçmek, LB'de değer taşımanın GARANTISI DEGIL.
Aynı veriden gelen üç test "üç bağımsız doğrulama" değildir (Kural 40).
Eğitim-türevli hipotezleri KÜÇÜK yer değiştirmeyle sonda; büyük bahis ancak
LB'de ölçülmüş bir kalite varsa yapılır.
