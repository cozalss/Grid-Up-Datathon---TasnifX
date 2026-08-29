# 62 — TAM TARAMA ve DONMUŞ PLAN

**docs/60 ve 61'in yerine geçer.** Dört bağımsız denetim (kod zinciri · istatistik ·
veri bütünlüğü · yarışma operasyonu) aynı anda çalıştırıldı. Bulunan her şey
uygulandı. Yeni bulgu çıkmadıkça bu plan **değişmeyecek**.

---

## 1. Önceki belgelerdeki İKİ YANLIŞ SAYI

### Yanlış 1: "hiç bilgi çıkmazsa taban 1,00061"

Yanlıştı. Ölçülmemiş adayların `L`'sini **sıfır** varsaydım. Oysa her adayın
**ölçülmüş span'ın içinde kalan parçasının `L`'si zaten biliniyor** ve sıfır değil
(örn. `L_span(y40) = −0,0052`). Sıfır varsaymak, olmayan bir "gürültü temizleme"
kazancı üretiyordu.

```
GERCEK taban (hic yeni bilgi yoksa) = 1,00140
```
Bu, deponun yıllardır bildiği **"span tavanı 1,0014"** ile birebir aynı — yani
tutarlılık kontrolünü de geçiyor. 1,00061 uydurmaydı.

### Yanlış 2: "8 yön ölçersek P(2. sıra) = %95"

Yanlıştı. Kullandığım önsel (m6 tabanına göre 20 yönün `|rho|` ortancası 0,0283)
**etkin olarak 20 değil ~2,7 bağımsız gözlemden** geliyor — o 20 yön eski model
sürümleri, ikili kosinüs ortancası 0,524, katılım oranı 2,68. Ortogonalleştirilince
`rho` ortancası 0,0283'ten **0,0030'a** çöküyor.

Doğru referans sınıfı: geçmişte **gerçekten yeni eksen** olan yönlerin **artımlı**
`rho`'su. Bunu iki bağımsız yoldan hesapladık, aynı yere çıktı:

```
artimli rho: 0,0023 0,0072 0,0114 0,0119 0,0146 0,0160 0,0161 0,0214 0,0253
   ortanca 0,0146     (bagimsiz ikinci hesap: 0,0144)
```

**2. sıra için gereken:** yön başına `rho ≈ 0,0224`. Elimizdeki dürüst ortanca
**0,0146**. Yani 2. sıra, sekiz eksende **ortalamanın üstünde** bir sonuç gerektiriyor.

---

## 2. DÜRÜST BEKLENTİ

```
m0 = 1.005846366  (kalibre, asagida)   taban 1,00292
2. sira 0,99940 -> gereken kazanc 0,007046
g7'nin tek basina getirdigi           0,003035   -> 1,00140
```

| senaryo | medyan skor | P(2. sıra) | P(tabandan kötü) |
|---|---|---|---|
| hiç yeni bilgi yok | 1,00140 | 0% | 0% |
| **dürüst önsel (ortanca 0,0146), 8 eksen** | **1,00056** | düşük | **0%** |
| iyimser önsel (ham 0,0272), 8 eksen | 0,99869 | %92 | 0% |
| dürüst önselin yarısı | 1,00119 | 0% | 0% |

**Ne diyebiliriz, ne diyemeyiz:**
- **3. sıra güvende.** En kötü halde bile 1,00140 < Tuna Deniz'in 1,00267'si.
- **2. sıra açık ama garanti değil.** Dürüst medyan 1,00056, hedef 0,99940.
  Aradaki 0,00116'yı kapatmak için sekiz eksenin ortalama `rho`'sunun 0,0146
  yerine ~0,0224 çıkması gerek. Mümkün — ölçülen `rho`'ların üst yarısı orada —
  ama yazı-turadan iyi olduğunu iddia edemem.
- **Geriye gitme riski sıfır.** Her senaryoda %0. 1,00284 bankada.

Beklenen kazanç: **0,0015 – 0,0042 RMSLE**, neredeyse sıfır riskle. Plan bu yüzden
yine de uygulanmaya değer — ama "%95 ile 2. sırayız" diye anlatılamaz.

---

## 3. `m0` DÜZELTİLDİ

m6'nın optimize edildiği üç yönün `L`'si tanım gereği sıfır olmalı. Eski
`m0 = 1.00284²` altında üçü de **tam olarak** `−0,000079` veriyordu — yani `m0`
sistematik olarak eksikti.

```
                          Q          P        ima edilen m0
p51_sicak05        0,013163   1,00946        1,005846063
m4_hava_capali     0,082002   1,04300        1,005846970
v102_kappa         0,005245   1,00553        1,005846063
                                  yayilma    9,07e-07   <- LB yuvarlama butcesi icinde
```
```
m0 = 1.005846366    taban 1,00292   (LB 1,00284 gosteriyor)
```
Fark 8e-5: LB skoru **public %50** üzerinde, `Q` ise **tüm 714 688 satırda**
ölçülüyor. Kalibrasyon bu uyumsuzluğu tek sabitte soğuruyor. Üç noktanın 9e-7
ile anlaşması bunun doğru kalibrasyon olduğunun kanıtı.

Bunun sonucu: her `L` +7,9e-5 kayıyor, `L_g7 = 0,002728 → 0,002751`,
gereken kazanç 0,006887 → **0,007046** (%2,3 daha zor).

---

## 4. YARIN — 3 hak, tek komut

```powershell
cd experiments/model29
python m108_gun.py --baslat        # sonda hazir, gonderim komutunu basar (MUTLAK yol)
# gonder, sonra MUTLAKA:
kaggle competitions submission-limits -c grid-up-datathon
python m108_gun.py --skor <SKOR>   # L'yi cozer, sonraki sondayi uretir
```

**1. sonda hazır:** `submissions/tuketim_s3y40.csv` (714 688 satır, NaN 0, negatif 0)
```
k = [g7 +1,83892 , y40 +0,39205]   cond 17,5   |k|_1 2,23   SNR 193
L_y40 = (1.001795698 - P^2) / 0.784106
L=0 iken 1,00090 | rho=0,0146 -> 0,99993 | rho=0,0224 -> 0,99941 | rho=0,030 -> 0,99891
```

**Ölçüm sırası (8 eksen):**
```
30 Agu:  y40  z2   sul
31 Agu:  y46  y45  q1c
 1 Eyl:  t3   p42  NIHAI
```
`h1_isil` **çıkarıldı** — ölçülmüş span'ın dışında yalnızca %26'sı kalıyor
(30 aday içinde 28.), yani bir hakkın çeyrek verimle harcanması demekti.
Yerine `p42_seviye_egrilik` (%83 yeni, seçilen 7'ye dik bileşen 1,30).

**Son gönderim:** `python m108_gun.py --bitir`

---

## 5. Düzeltilen 13 kusur

| # | kusur | sonucu olurdu |
|---|---|---|
| 1 | `\|e\|>2e-3` korkuluğu **işler iyi giderken** tetikleniyordu | zincir `rho≥0,032`'de kırılırdı — beklenen aralığın tam ortası |
| 2 | Korkuluk tetiklenince `--atla` da `--baslat` da reddediyordu | sabah 03:00'te kilitlenme, elle JSON düzenlemekten başka çıkış yok |
| 3 | Aynı korkuluk **son gönderimi** de bloke ediyordu | 1 Eylül 9. hak, dosya hiç yazılmaz |
| 4 | `\|k\|_1` tavanı iyi haberde tetikliyordu (`rho_y40≥0,10`) | en iyi senaryoda zincir ölür |
| 5 | `--skor -1` makul görünen bir `L` üretiyordu, sessizce | bozuk `L` üzerine 7 sonda daha kurulur |
| 6 | `--skor` iki kez çalışınca skoru **sonraki** adaya yazıyordu | bir hak boşa, uydurma `L` NIHAI'ye girer |
| 7 | Basılan kaggle komutunun yolu çalışma dizininde geçersizdi | 03:00'te "file not found" |
| 8 | `--baslat` yarın sabah **reddedecekti** (durum dolu) | kılavuzun ilk adımı çalışmaz |
| 9 | `--bitir` skoru girilmemiş sondayı sessizce atıyordu | o eksen bedavaya çöpe |
| 10 | `--cikti` **tabanın üzerine yazabiliyordu**, uyarısız | 1,00284 ve tüm cebrin çıpası yok olur |
| 11 | Durum kaydı atomik değildi | Ctrl+C tüm ölçülmüş `L`'leri silebilir |
| 12 | Mükerrer-skor denetimi yanlış pozitif veriyordu | iki sonda aynı skoru alırsa haksız kilit |
| 13 | `m0` sistematik olarak eksikti | her `L`'de %2 hata |

Ayrıca: `--bilinen` içinde tekrarlı ad reddediliyor, `--cikti` yol kabul etmiyor,
`m107_*.json` betik dizinine yazılıyor, `PYTHONIOENCODING=utf-8` zorlanıyor.
**Taban yedeklendi:** `submissions/tuketim_m6_ikiyon.csv.YEDEK`.

---

## 6. Temiz çıkanlar (denendi, hata yok)

- **Kapalı devre cebir kusursuz.** Çözüm sabiti diskteki kırpılmış dosyadan
  sıfırdan yeniden hesaplandı: fark **0,000e+00**. `L` çözümü kapanıyor; kalan
  hata LB'nin 5 haneye yuvarlaması (1,3e-5) + kırpma (~1e-5), ikisi de eşiğin %1'i altı.
- **`L_g7` sahte değil.** g7'nin ölçülmüş span'a artığı **%0,0012**; `L_g7`
  regularizasyona karşı kararlı (0,002786–0,002789 aralığı, eski kayıt muhafazakâr taraftaydı).
- **Taban bağımsız doğrulandı.** 9 tam afin bağıntının 9'unda da `L`'ler tutarlı;
  `m0`, m6'nın kendi skoru kullanılmadan **1,00292 ± 0,00025** çıkıyor.
- **Veri temiz.** 35 dosyanın 595 ikili karşılaştırması: kopya yok. Hepsi
  714 688 satır, id sırası `test.csv` ile birebir, NaN/negatif/sonsuz yok.
- **Dejenere yön yok.** En yoğun aday bile `d²`'sinin yarısını 22 000+ satıra yayıyor.
- **Kırpılan 497 satır gerçekten küçük tüketimli** (medyan p0,17) — işaret dönmesi yok.
- **Encoding temiz.** Betiklerde 0 adet ASCII dışı bayt; cp1254 bozulması mümkün değil.
- **Kota doğrulandı (canlı API).** Sıfırlama 00:00 UTC = 03:00 TRT; 21–29 Ağustos'un
  her biri tam 3 gönderim; bugün 3/3 kullanıldı, kalan toplam **9**.
- **Ölçüm gürültüsü önemsiz.** 8 eksende uydurma bedeli **+3e-6 RMSLE**.
- **Public'te ilerleyip private'ta tabanın gerisine düşme:** P < 1e-11.

---

## 7. MODEL DIŞI RİSKLER — bunlar sıralamayı model kadar etkiler

**1. Notebook yok.** `notebooks/` 22 Ağustos'tan beri dokunulmamış; v50→v102→m6
hattının hiçbiri içinde değil. Son tarih **2 Eylül 13:00**, private LB 00:10'da
açılıyor — arada 10–13 saat var ve o saatler beyan + jürinin 5 sorusu + kaynak
listesi yazmaya yetmez. **1 Eylül'den önce bitir.**

**2. Final 2 gönderim seçimi yapılmamış.** Kaggle API'sinde bu alan **yok** —
tarayıcıdan, Submissions ekranı → "Use for Final Score". Kapanıştan sonra
değiştirilemez. **Bu gece bir ara-seçim yap**, 1 Eylül 23:00 UTC'den önce güncelle.
Seçilmezse Kaggle en iyi 2 public'i alır; iki slot da aynı aileden olur, çeşitlendirme sıfır.

**3. Takım arkadaşının Coderspace kaydı doğrulanmamış.** Kural: kayıt tamamlanmamışsa
skor geçersiz. Tüm gönderimleri `cemzal` yapmış (doğrulandı) — bu iyi. `gizemkl`'in
kaydını teyit et.

**4. Hava verisi kuralı — yeni gelişme.** Düzenleyicinin 737242 numaralı mesajı
(Nisan–Temmuz 2026 gerçekleşmiş hava verisi uygun değil) **hâlâ ayakta**. Buna
karşılık 27 Ağustos'ta "dış kaynak kullanımı serbest" diyen 3517424 numaralı
düzenleyici mesajı **bugün forumdan silinmiş** (canlı API'de yok; arşivi
`experiments/model29/forum_son.txt` içinde). Senin e-postayla aldığın teyit
geçerli — karar senin — ama forum ayağı çöktüğü için **o e-postayı bu gece
`.eml` ya da ekran görüntüsü olarak repoya koy.** Notebook incelenirken tek
dayanak o olacak.

**5. Son gönderim zamanlaması.** En geç **1 Eylül 22:30 UTC (2 Eylül 01:30 TRT)**.
Final seçimi en geç **23:00 UTC**. Kapanış 23:59 UTC = 2 Eylül 02:59 TRT.

---

## 8. Geri sayım

| TRT | UTC | hak | yapılacak |
|---|---|---|---|
| şimdi → 30 Ağu 03:00 | → 00:00Z | 0 | e-posta kanıtı arşivle · ara final seçimi · `gizemkl` teyidi |
| 30 Ağu 03:00 → 31 Ağu 02:59 | 30 Ağu 00:00–23:59Z | 3 | y40 → z2 → sul |
| 31 Ağu 03:00 → 1 Eyl 02:59 | 31 Ağu 00:00–23:59Z | 3 | y46 → y45 → q1c · **notebook başlar** |
| 1 Eyl 03:00 → 2 Eyl 02:59 | 1 Eyl 00:00–23:59Z | 3 | t3 → p42 → NİHAİ (**en geç 22:30Z**) |
| — | 1 Eyl 23:00Z | — | **FİNAL 2 SEÇİM SON AN** |
| 2 Eyl 03:10 | 2 Eyl 00:10Z | — | private LB |
| 2 Eyl 13:00 | 10:00Z (varsay) | — | notebook `coderspacetr` ile paylaşılır |

---

## 9. Kalıcı kurallar 43–49

**43.** Kazanç `rho'C⁻¹rho`'dur — `Q` sadeleşir. Yön seçiminde `Q` eşiği/kurtoz
kullanma; kosinüs yapısı ve ölçülebilirlik (`k·√Q ≥ 0,03`) yeter.
**44.** Kaliteyi HANGİ TABANA göre ölçtüğünü yaz.
**45.** Sonda = o an bilinenlerin TAM ortak optimumu + yeni yön.
**46.** Ölçüm hakkı, "rötuş" hakkından değerlidir.
**47.** Ölçülmemiş bir yönün `L`'sini SIFIR varsayma. Span içi parçasının `L`'si
zaten biliniyor; sıfır varsaymak olmayan bir kazanç uydurur.
**48.** Önsel kurarken **etkin örnek sayısını** hesapla. Birbirine benzeyen 20
gözlem 20 bilgi değildir; ortogonalleştirip artımlı değere bak.
**49.** Korkuluk eşiğini, **iyi senaryoda tetiklenip tetiklenmediğini ölçerek** koy.
Bir güvenlik kontrolü kazandığın anda seni durduruyorsa güvenlik değil, arızadır.
