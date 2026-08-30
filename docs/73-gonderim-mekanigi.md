# GÖNDERİM MEKANİĞİ VE TAKVİM — kesinleştirme

Yazım: 30 Ağustos 2026, 21:20 yerel (18:20 UTC).
Her iddianın yanında **kaynağı** var: `API` (bugün çalıştırılan komut çıktısı),
`ÇIKARIM` (ölçülen veriden mantıkla türetildi), `DOKÜMAN` (Kaggle yayını),
`BELGE` (bu depodaki önceki doğrulama kaydı).

> Bu belge hazırlanırken **hiçbir gönderim yapılmadı**. Yalnızca okuma
> komutları çalıştırıldı.

---

## 1. KOTA — ne zaman, kaç hak

### Ölçülen durum (bugün)

```
$ python -m kaggle competitions submission-limits grid-up-datathon
Submissions today: 3
Lifetime submissions: 30
Remaining today: 0
```
`API`, 30 Ağustos 18:06 UTC.

### Günlük sınır = 3

```
max_daily_submissions = 3
```
`API` — `competitions_list(group='entered')` alanı. Ayrıca `ÇIKARIM`: 30
gönderimlik geçmişte **hiçbir takvim gününde 3'ten fazla** gönderim yok
(24, 25, 26, 27, 28, 29, 30 Ağustos'un her biri tam 3).

### Sıfırlanma anı = 00:00 UTC = **03:00 Türkiye saati**

Üç bağımsız kanıt:

1. `ÇIKARIM` — 25 Ağustos'un üç hakkı **00:00:45, 00:01:53, 00:01:55**
   damgalarıyla kullanıldı. Gün sınırından 45 saniye sonra gönderim kabul
   edildi; demek ki sınır tam o damganın gece yarısında.
2. `ÇIKARIM` — API damgaları **UTC**'dir. Kanıt: 24 Ağustos'un hakları
   `04:19–04:30` damgalıydı, 25 Ağustos'unkiler `00:00–00:02`. Damgalar yerel
   saat (UTC+3) olsaydı bunların hepsi aynı UTC gününe (24 Ağustos) düşer,
   6 gönderim ederdi — 3 sınırıyla çelişir. Damgalar UTC'dir, sıfırlanma da
   00:00 UTC'dir.
3. `BELGE` — `docs/40-olcut-tamiri-2026-08-24.md` satır 32: "Bugünün üç hakkı
   04:19-04:30 UTC'de kullanıldı. Sıfırlanma **00:00 UTC = yerel 03:00**."
   Aynı sonuç `docs/31` §3(b)'de bağımsız olarak kayıtlı.

### Türkiye saatiyle takvim

| pencere (yerel, UTC+3) | hak |
|---|---|
| şimdi → 31 Ağu 02:59 | **0** — bugünün kotası bitti |
| **31 Ağu 03:00 → 1 Eyl 02:59** | **3** |
| **1 Eyl 03:00 → 2 Eyl 02:59** | **3** (son pencere, yarışma bitişiyle biter) |

**Toplam kalan: 6.** `docs/72`'deki sayı doğrulandı.

> Kullanılmayan hak **yanar**, devretmez (`ÇIKARIM` — 25 Ağustos'ta kullanılmayan
> haklar ertesi güne eklenmedi; `docs/45` satır ~915).

---

## 2. SON TARİH

```
deadline             = 2026-09-01 23:59:00   (UTC)
merger_deadline      = 2026-09-01 23:59:00   (UTC)
submissions_disabled = False
category             = Community
team_count = 466      user_rank = 7
```
`API` — bugün çekildi; 22 Ağustos'ta okunan değerle **birebir aynı**
(`docs/31` §1), yani son tarih değiştirilmedi.

**Türkiye saatiyle bitiş: 2 Eylül 2026, 02:59.**

Bu, kota takvimiyle şunu doğuruyor: 1 Eylül 03:00 (yerel) açılan 3 hak
**tam olarak yarışma bitişine kadar** kullanılabilir. Bitişten sonra ayrı bir
kota penceresi yok.

### Son gönderim / son seçim anı

- **Son gönderim:** 2 Eylül 02:59 yerel (`API`, `deadline`).
- **Son seçim değişikliği:** §3'e bakın. Güvenli sınır olarak
  **1 Eylül 22:00 UTC = 2 Eylül 01:00 yerel** alınır, bitişe dakikalar kala
  değil.

---

## 3. SEÇİM MEKANİĞİ

### Kaggle'ın resmi kuralı — üç madde

**Kaynak:** [Kaggle Competitions Documentation](https://www.kaggle.com/docs/competitions)
(bugün web araması ile doğrulandı; sayfanın kendisi giriş gerektirdiği için
doğrudan alıntı arama sonucundan alındı).

**(a) Seçim yapılmazsa ne olur — otomatik seçim, PUBLIC'e göre**

> "If you do not select submission(s) to be scored before the competition
> closes, the platform will automatically select those which performed the
> highest on the public leaderboard, unless otherwise communicated in the
> competition."

Yani seçim yapılmazsa Kaggle **public skoru en yüksek** gönderimleri seçer.
Bu bizim için **tehlikelidir**: sonda gönderimlerimiz (D1–D4) tasarım gereği
public'te kötü skor verebilir, ama asıl dosyamız `Z_NIHAI` en sonda gelir.
Otomatik seçim yanlış dosyaları işaretleyebilir.

> **KURAL: seçimi ELLE yap. Otomatiğe bırakma.**

**(b) Seçilenlerden hangisi sayılır — PRIVATE'ta iyi olan**

> "your final score and placement at the end of the competition will be
> whichever selected submission performed best on the private leaderboard."

> "In most Kaggle competitions, you select 2 of 2 submissions to be evaluated
> for your final leaderboard score, with the evaluated submission with the
> best Private Score used for your final score."

Bu, `docs/72`'deki **yedek seçim** stratejisinin temelidir: `YP_seviye`
(1.00115) ikinci seçim olarak işaretlendiği sürece, asıl bahsimiz private'ta
kötü çıksa bile **1.00115'in altına düşmeyiz**. Bahsin aşağı yönü kapalıdır.

**(c) Kaç gönderim seçilir**

Genel kural 2'dir, ancak sayı yarışmaya göre değişebilir
("unless otherwise communicated in the competition"). Bu yarışma için
**kesin sayı tarayıcıdaki seçim ekranından teyit edilmelidir** — API bu
bilgiyi vermiyor (aşağıya bakınız).

> **DOĞRULANAMADI:** Bu yarışmanın seçim sayısının 2 olduğu, yarışma
> sayfasından teyit edilmedi (giriş gerektiriyor). Tarayıcıya girildiğinde
> ilk iş bunu kontrol etmek olmalı. 1 ise, yedek strateji çalışmaz ve
> karar değişir — o durumda **`Z_NIHAI` yerine hangi dosyanın seçileceği
> yeniden değerlendirilmelidir.**

**(d) Community / in-class fark var mı**

Bu yarışma bir Community yarışmasıdır. Kaggle dokümantasyonu seçim
semantiği için Community yarışmalarına ayrı bir kural tanımlamıyor; yukarıdaki
üç madde geçerli kabul edilir. Yine de (c)'deki teyit tarayıcıdan yapılmalıdır.

### API'den seçim yapılabilir mi — **HAYIR**

`API` (yerel `kaggle` 2.2.4 paketinin kaynağı, bugün tarandı):

- `kagglesdk` competition servisinin **tüm** RPC listesi okundu:
  `list_submissions`, `create_submission`, `get_submission`,
  `download_submission`, `get_submission_limits`, `start_submission_upload`,
  `get_leaderboard`, … — **seçim/final ile ilgili tek bir uç nokta yok.**
- `ApiSubmission` nesnesinin alanları: `date, description, error_description,
  file_name, private_score, public_score, ref, status, submitted_by,
  submitted_by_ref, team_name, total_bytes, url`. **`selected` / `is_final`
  gibi bir alan yok** — API seçimi yazamadığı gibi **okuyamıyor** da.

**Sonuç: seçim yalnızca tarayıcıdan yapılır ve yalnızca tarayıcıdan
doğrulanabilir.** Bu, `docs/72` §2 Adım 4'teki notu doğrular.

### Yalnız GÖNDERİLMİŞ dosyalar seçilebilir — kritik sonuç

Seçim ekranı gönderim geçmişinden seçtirir. Bu şu demek:

> `tuketim_Z_NIHAI.csv` **Kaggle'a gönderilmediyse seçilemez.**
> Diskte üretilmiş olması hiçbir şey ifade etmez.

Zincir hesabı (§4.1) bu kısıtın etrafında kuruldu.

---

## 4. BAŞARISIZLIK SENARYOLARI

### 4.1 Zincir kısıtı — en büyük risk (bu belgede bulundu)

`ÇIKARIM` — `experiments/model29/m148_demet_plani.py` satır 421–425 okundu:
her sondanın **tabanı, önceki sondaların ölçülmüş `rho_j`'lerini içeriyor**
(`taban = a0 + r_hat + Σ_{ölçülen} rho_j·G_j`). Yani sondalar **paralel
gönderilemez, sıralıdır**: D2'yi üretmek için D1'in LB skoru gerekir.

```
gereken gönderim: D1, D2, D3, D4, Z_NIHAI  = 5
elde olan       : 3 (31 Ağu) + 3 (1 Eyl)   = 6
```

Slack yalnızca **1 gönderim**, üstelik bölünme zorunlu:
**31 Ağustos'ta D1–D2–D3 bitmezse, 1 Eylül'de 4 gönderim gerekir ama 3 var.**
Bu yüzden §5'teki liste üç sondayı da 31 Ağustos sabahına sığdırır.

**Kaçış planı (hak biterse):** elde kalan son **sonda** dosyası (`D_k`)
nihai olarak seçilmemelidir — içinde ölçülmemiş `+kappa·G_k` sapması var ve
beklenen skoru tabandan **kötüdür** (`rho_k = 0` ise skor `sqrt(sabit)`,
D1 için 1.00235 — mevcut 1.00115'ten kötü). Bu nedenle **son hak her zaman
`Z_NIHAI`'ye ayrılır**. Zincir gerekirse kısaltılır: betik eksik sonda
görünce uyarı verip devam ediyor, yani yalnız D1–D2–D3 ölçülüp `Z_NIHAI`
erken üretilebilir.

### 4.2 `SubmissionStatus.ERROR` dönerse

`API` — `SubmissionStatus` yalnız üç değer alır: `PENDING`, `COMPLETE`, `ERROR`.
Sebebi `ApiSubmission.error_description` taşır.

Yapılacak:
```
python -m kaggle competitions submissions -c grid-up-datathon -v
```
`status` sütunu `ERROR` ise `error_description`'ı oku. **DOĞRULANAMADI:**
hatalı bir gönderimin günlük kotadan düşülüp düşülmediği ölçülmedi (30
gönderimin hiçbiri ERROR olmadı). Bu yüzden ERROR'ı **kota kaybı say** ve
sonraki gönderimden önce `submission-limits` ile `Remaining today`'i oku.

Bu yarışmada dosya formatı defalarca doğrulandı; `m148_demet_plani.py`
yazmadan önce altı kapıyı geçiriyor (satır 433–444: 714 688 satır, id sırası
`sample_submission` ile birebir, 0 NaN, 0 negatif, hepsi sonlu, üst sınır).

### 4.3 Skor çok geç gelirse

`ÇIKARIM` — geçmişteki 30 gönderimin hepsi `COMPLETE`; liderlik tablosundaki
TasnifX damgası `2026-08-30 05:07:53.966`, gönderim damgası `05:07:53.967`.
Puanlama **saniyeler içinde** dönüyor.

- Skor 10 dakika içinde gelmezse durumu döngüyle **okumaya devam et**,
  **tekrar gönderme**.
- Zincir gecikirse §4.1'deki kısaltma devreye girer.

### 4.4 Dosya büyükse / komut zaman aşımına uğrarsa

`ÖLÇÜLDÜ` — gönderim dosyaları **~28,4 MB** (`tuketim_D1_demet.csv`
28 394 529 bayt, 714 689 satır = 714 688 + başlık). `BELGE` `docs/45`:
27 MB yükleme **56 saniyede** tamamlandı.

> **MUTLAK KURAL** (`kaggle-gonderim-once-liste-oku` hafızası; 25 Ağustos'ta
> bir hak tam bu yüzden yandı): zaman aşımına uğrayan bir komut
> "gönderilmedi" demek **değildir**. Araç süreci öldürür, sunucudaki yan etki
> kalır.
>
> Her gönderimden **önce ve sonra**:
> ```
> python -m kaggle competitions submissions -c grid-up-datathon -v | head -3
> ```
> ve en üst satırın **damgasına** bak. Yeni kayıt varsa gönderim
> **olmuştur** — tekrarlama.

Kotanın üstüne gönderim denenirse sunucu `400 Client Error: Bad Request`
verir ve **listeye kayıt düşmez, hak yanmaz** (`BELGE` `docs/45`, 25 Ağustos
ölçümü) — "temiz red", güvenli taraf.

### 4.5 İnternet / API kesilirse — tarayıcı yolu

`https://www.kaggle.com/competitions/grid-up-datathon/submissions` →
**Submit Prediction** → dosyayı sürükle → açıklamayı yaz → Submit.
28 MB tarayıcıdan da yüklenir. Seçim zaten yalnız burada yapılıyor (§3),
yani tarayıcı oturumunun **çalışır olması her hâlükârda şart** — gönderim
gününden önce giriş yapılıp doğrulanmalı.

---

## 5. KONTROL LİSTESİ — saatli

Tüm saatler **Türkiye saati (UTC+3)**. `PY = ./.venv/Scripts/python.exe`

### T-0: 31 Ağustos 02:50 — hazırlık (kota açılmadan)

| # | komut | çıktıda ne aranacak | geçiş koşulu |
|---|---|---|---|
| 1 | `$PY -m kaggle competitions submission-limits grid-up-datathon` | `Remaining today` | `0` normal — 03:00'ı bekle |
| 2 | `$PY -m kaggle competitions submissions -c grid-up-datathon -v \| head -3` | en üst satırın damgası | `2026-08-30 05:07` olmalı; başka bir şey varsa **DUR** |
| 3 | `ls -l submissions/tuketim_D1_demet.csv` | 28 394 529 bayt | tutmuyorsa dosyayı yeniden üret |
| 4 | tarayıcıda Kaggle'a gir, submissions sayfasını aç | oturum açık mı | §3 ve §4.5 için şart |

### T+1: 31 Ağustos 03:00 — SONDA 1

```
$PY -m kaggle competitions submissions -c grid-up-datathon -v | head -3     # ÖNCE OKU
$PY -m kaggle competitions submit -c grid-up-datathon \
    -f submissions/tuketim_D1_demet.csv -m "D1 H1 yonu sondasi"
$PY -m kaggle competitions submissions -c grid-up-datathon -v | head -3     # SONRA OKU
```
- **Aranan:** ikinci okumada `tuketim_D1_demet.csv` satırı yeni damgayla.
- Komut zaman aşımına uğrarsa **tekrar gönderme**, yalnız listeyi oku (§4.4).
- `status` `PENDING` ise 30 saniyede bir listeyi tekrar oku.
- `COMPLETE` olunca `publicScore` = **P1**.

### T+2: skor gelince (~03:05) — SONDA 2 üret

`experiments/model29/m148_olcumler.json` dosyasını **elle** yaz (P1 yerine
LB'den okunan sayı, örn. `0.99967`):

```json
{"1": 0.99967}
```

sonra:

```
$PY experiments/model29/m148_demet_plani.py
```
- **Aranan:** `OLCULEN rho_k` bloğu, `SIRADAKI: sonda 2`,
  `URETILDI: submissions/tuketim_D2_demet.csv`, yeni `COZUM:` satırı.
- `KAPI KALDI` yazıyorsa **DUR** — dosya yazılmadı, sebebi teşhis et.
- Sonra T+1 adımını `tuketim_D2_demet.csv` ile tekrarla → **P2**.

### T+3: ~03:15 — SONDA 3

`m148_olcumler.json`'a `"2": P2` ekle, betiği çalıştır,
`tuketim_D3_demet.csv`'yi gönder → **P3**.

> **31 Ağustos'un üç hakkı burada biter.** `Remaining today = 0` ile doğrula.

### T+4: 31 Ağustos gündüz — KARAR

`"3": P3` yazılır, betik çalıştırılır. Çıkan `toplam rho^2` ve
"su anki nihai skor" satırı `docs/72` §3 beklenti tablosuyla karşılaştırılır.
**1 Eylül için iki seçenek:**

- **A (3 ölçüm yeterliyse):** D4 atlanır. 1 Eylül'de `Z_NIHAI` gönderilir,
  2 hak yedekte kalır. **Düşük risk.**
- **B (D4 gerekliyse):** 1 Eylül 03:00'te D4, skoru gelince `Z_NIHAI`,
  1 hak yedek. Zincirin bir halkası kalır.

Karar 31 Ağustos akşamına kadar verilir ve buraya yazılır.

### T+5: 1 Eylül 03:00 — son pencere

Seçilen senaryoya göre yürüt. **Değişmez kural:** son pencerenin
**en az bir hakkı `Z_NIHAI` için ayrılır** ve 1 Eylül 12:00'den (yerel)
sonraya bırakılmaz.

### T+6: 1 Eylül, en geç 22:00 yerel — SON SEÇİM (tarayıcı)

1. `https://www.kaggle.com/competitions/grid-up-datathon/submissions` aç.
2. **İki** gönderim işaretle:
   - `tuketim_Z_NIHAI.csv` (ya da o ana kadarki en iyi birleşik)
   - `tuketim_YP_seviye.csv` — public 1.00115, **YEDEK**
3. Sayfayı **yenile** ve işaretlerin durduğunu gözle doğrula.
   API bunu okuyamaz (§3); tek doğrulama gözdür — ekran görüntüsü al.

> Seçim son ana bırakılmaz: bitiş 2 Eylül 02:59 yerel, hedef **2 Eylül 01:00**,
> yani **1,5+ saat** pay.

### T+7: kapanış kontrolü

```
$PY -m kaggle competitions submissions -c grid-up-datathon -v
$PY -m kaggle competitions leaderboard grid-up-datathon -s | head -12
```
Tüm gönderimlerin `COMPLETE` olduğunu ve LB'deki TasnifX skorunun beklenenle
uyuştuğunu doğrula.

---

## 6. DOĞRULANAMAYANLAR

Kaynak bulunamadı; **tahmin yürütülmedi**:

- Public/private test bölünmesinin **yüzdesi**. (`BELGE` `docs/31` §2:
  Community yarışması olduğu için Overview sayfası oturumsuz 404 veriyor;
  API bu alanı yayımlamıyor.) Tarayıcıdan 30 saniyede okunabilir.
- `SubmissionStatus.ERROR` dönen bir gönderimin **günlük kotadan düşülüp
  düşülmediği**. Bu depoda hiç ERROR yaşanmadı. Güvenli varsayım: düşülür.
- Kaggle seçim ekranının bitişten **kaç dakika önce** kapandığı.
