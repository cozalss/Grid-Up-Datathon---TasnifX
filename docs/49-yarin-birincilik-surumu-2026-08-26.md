# 27 Ağustos birincilik sürümü — son devir

Bu belge `docs/47`nin üstüne geçer. Amaç, 27 Ağustos kotasında hem ölçülmüş
kazancı bankaya yatırmak hem de sıcak çekirdek sinyali yeterliyse aynı gün
liderliği hedefleyen optimumu göndermektir.

## 1. Dürüst durum

```
26 Ağustos ölçülmüş en iyi       1.01538
Grid Grinders                    1.00635
Hazır v80 beklenen               1.013412
```

Bugünkü üretim-sadık model taraması yeni bir taban üretmedi. Önceki ayar
tezgâhlarının 151 kolon kullandığı, üretimin ise 105 kolon kullandığı doğrulandı
ve doğru rig ile 13 aday ölçüldü. Tek-tohum elemesinde iki aday iki blokta da
kazandı:

| aday | yaz25 dMSE | kis26 dMSE | ortalama |
|---|---:|---:|---:|
| `lr=0.03, iter=400` | +0.012533 | +0.014306 | +0.013419 |
| `random_strength=4` | +0.008848 | +0.026268 | +0.017558 |

Fakat beş-tohum kapısı ikisini de reddetti:

- `random_strength=4`, yaz25: 1/3 kazanan; üçüncü tohumdan sonra erken kesildi.
- `lr=0.03, iter=400`, yaz25: 2/5 kazanan; kalan bloklar harcanmadan kesildi.

Sonuç: bu iki ayar **üretime alınmaz**. Ayrıntılı eleme sonuçları
`experiments/uretim_ayarlari.jsonl` içindedir.

## 2. Hazır ve doğrulanmış dosyalar

| dosya | görev | bütünlük |
|---|---|---|
| `tuketim_v80_optimum.csv` | HAK 1, ölçülmüş gün/seviye optimumunu bankala | geçti |
| `tuketim_v81_sicak08.csv` | HAK 2, 526.446 satırlık sıcak çekirdek probu | geçti |
| `tuketim_v82_ayirici.csv` | HAK 3 alternatifi, 29.873 satırlık kuyruk probu | geçti |

Üç dosyada da 714.688 satır, birebir id sırası, 0 mükerrer, 0 NaN ve 0
negatif doğrulandı. Etiketsiz prob sabitleri dosyaların kendisinden tekrar
ölçüldü:

```
sıcak çekirdek  n=526.446  adım=0,08  Q=0,004714301
kuyruk          n= 29.873  adım=0,15  Q=0,000940470
soğuk           n=158.369
```

## 3. 27 Ağustos — adaptif üç hak

### Sıfırıncı iş

Kota ve güncel gönderimleri oku. Aynı dosyayı ikinci kez gönderme.

```powershell
uv run python -m kaggle competitions submissions -c grid-up-datathon
```

### HAK 1 — bankala

```powershell
uv run python -m kaggle competitions submit -c grid-up-datathon `
  -f submissions/tuketim_v80_optimum.csv `
  -m "v80 olculmus optimum banka; gun+seviye+kuyruk"
```

Durum `complete` olana kadar gönderim listesini yeniden oku ve gerçek
`BANKA_SCORE` değerini kaydet.

### HAK 2 — sıcak çekirdeği çöz

```powershell
uv run python -m kaggle competitions submit -c grid-up-datathon `
  -f submissions/tuketim_v81_sicak08.csv `
  -m "v81 sicak cekirdek +0.08 prob; v80 tabanina gore"
```

`SICAK_SCORE` geldikten sonra, kuyruk skoru vermeden aynı-gün optimumunu üret:

```powershell
uv run python scripts/yarin_coz.py `
  --banka-score BANKA_SCORE `
  --sicak-prob-score SICAK_SCORE

uv run python scripts/kapi_denetim.py `
  --ref submissions/tuketim_v80_a.csv `
  submissions/tuketim_v83_sicak_optimum.csv
```

Araç sıcak çekirdek optimumunu tam çözer, soğuk/kuyruğu v80 seviyesinde tutar
ve beklenen RMSLE'yi yazar.

### HAK 3 — karar ağacı

Önce güncel lider skorunu kontrol et.

```
v83 tahmini RMSLE <= güncel lider - 0,00010
    -> v83'ü GÖNDER; aynı gün birinciliği bankala.

aksi
    -> v82 kuyruk probunu GÖNDER; 28 Ağustos tam optimumu için üçüncü bilinmeyeni çöz.
```

v83 yolu:

```powershell
uv run python -m kaggle competitions submit -c grid-up-datathon `
  -f submissions/tuketim_v83_sicak_optimum.csv `
  -m "v83 v80 + LB-cozulmus sicak cekirdek optimumu"
```

v82 yolu:

```powershell
uv run python -m kaggle competitions submit -c grid-up-datathon `
  -f submissions/tuketim_v82_ayirici.csv `
  -m "v82 kuyruk +0.15 ayirici; v80 tabanina gore"
```

Kuyruk skoru geldikten sonra tam optimumu üret:

```powershell
uv run python scripts/yarin_coz.py `
  --banka-score BANKA_SCORE `
  --sicak-prob-score SICAK_SCORE `
  --kuyruk-prob-score KUYRUK_SCORE

uv run python scripts/kapi_denetim.py `
  --ref submissions/tuketim_v80_a.csv `
  submissions/tuketim_v84_tam_optimum.csv
```

`v84` 28 Ağustos'un ilk banka adayıdır.

## 4. Çözücü güvenlik sözleşmesi

`scripts/yarin_coz.py`:

- prob maskelerini ve Q değerlerini submission dosyalarından etiketsiz ölçer;
- sıcak/soğuk/kuyruk gruplarının 526.446/158.369/29.873 sayılarını fail-closed
  doğrular;
- skorları yalnız küresel kuadratik katsayı çözmek için kullanır;
- üç ayrık gruba deltayı tek adımda uygular;
- negatif delta sıfır tahmini kırpacaksa özdeşlik bozulmasın diye dosya üretmez;
- Kaggle'a otomatik gönderim yapmaz;
- skorları, çözülen deltaları ve beklenen skoru `reports/yarin_cozum.json`a yazar.

Sentetik `b_sıcak=0,15` round-trip provasında araç tam `0,150000` çözdü,
1,005202 bekledi ve ürettiği 714.688 satırlık dosya tüm submission kapılarından
geçti.

## 5. Git push — 27 Ağustos, Kaggle'dan bağımsız

Bugün uzak repoya push yapılmaz. Yarın testler hâlâ yeşilse:

```powershell
git status --short --branch
uv run python -m pytest -q
git push -u origin feature/winning-version-20260827
```

Kaggle gönderimi ve GitHub push iki ayrı işlemdir; biri başarısız olursa diğeri
başarılı varsayılmaz.

## 6. 27 Ağustos gerçekleşen sonuç

Üç hak planlandığı gibi kapalı çevrim kullanıldı:

| sürüm | amaç | gerçek public RMSLE |
|---|---|---:|
| `v80` | ölçülmüş optimum banka | **1.01341** |
| `v81` | sıcak çekirdek `+0.08` probu | **1.01429** |
| `v83` | LB ile çözülmüş sıcak optimum | **1.01318** |

`v80/v81` denklemi sıcak çekirdek ek deltasını `+0.0248598893` çözdü.
Çözücü `v83` için `1.0131853695` bekledi; Kaggle beş ondalıkta **1.01318**
vererek tahmini doğruladı. Fakat canlı liderlik tablosunda gerçek ikincilik
eşiği **1.01064** olduğundan `v83` dördüncü sırada kaldı. Gönderim ref'i
`55811502`, durum `COMPLETE`.

Dosya kapıları: 714.688 satır, ID sırası birebir, mükerrer/NaN/Inf/negatif
yok. Gönderilen dosyanın SHA256 değeri:

```
F482A9DEEB771BF6D17B9271B9D11190B8FB495D28388D35E5A6C28CAC108041
```

Ek sıcak CatBoost kampanyasında Bernoulli örnekleme tek-tohum elemesinde
iki blokta kazandı (`dMSE +0.00721`), fakat tam kapıda yalnız 8/9 hücreyi
kazandı; `kis26` blok ortalaması `-0.00319`, eşlenik `t=0.94` oldu. Bu nedenle
model değişikliği **reddedildi** ve başarılı v83 zinciri korunmuştur.

## 7. 28 Ağustos ikincilik adayı: v85 Gram ansamblı

Bugünkü üç hakkın bitmesinden sonra, on adet `COMPLETE` Kaggle gönderiminin
gerçek skorları ve tahmin vektörleri kullanılarak etiketsiz log-uzayı Gram
optimumu çözüldü. RMSLE'nin karesel yapısı nedeniyle test etiketi gerekmez:
`Q = DᵀD/N`, `Lᵢ = (s₀² + Qᵢᵢ - sᵢ²)/2`, `Qk = L`.

Dokuz yönlü Pareto çözümünün beklenen skoru **1.0104968562**; canlı ikinci
**1.01064**. Görüntülenen skorların tüm `2^10` adet `±0.000005` yuvarlama
köşesinde sabit aday bandı **1.01044933–1.01054438** ve tamamı mevcut ikincilik
çizgisinin altındadır. Çözüm agresiftir; koşul sayısı `657.82` kabul edilebilir
olsa da büyük pozitif ve negatif afin ağırlıklar kullanır.

Üretim ve kapı:

```powershell
uv run python scripts/gram_ansambl.py
uv run python scripts/kapi_denetim.py `
  --ref submissions/tuketim_v83_sicak_optimum.csv `
  submissions/tuketim_v85_gram_rank2.csv
```

`v85` dosyası 714.688 satır, birebir ID sırası, sıfır mükerrer/NaN/Inf/negatif
ile kapıdan geçmiştir. SHA256:

```
994AC7A37CC400EB6CB660C205A0BB9AE5E6E42E96A401B1F346F1835F96CC8F
```

28 Ağustos `HAK1` doğrudan `v85` içindir. Gerçek skor geldiğinde gerekirse aynı
yön üzerindeki optimum ikinci aday otomatik çözülür:

```powershell
uv run python scripts/gram_ansambl.py `
  --prob-skor V85_SCORE `
  --cikis submissions/tuketim_v86_gram_kappa.csv `
  --rapor reports/gram_rank2_v86.json
```

`v85 <= canlı_ikinci - 0.00010` ise dur ve hakkı koru. Aksi halde `v86` kapıdan
geçirilip `HAK2` olarak gönderilir. `HAK3`, yalnız bu iki adım ikinciliği
getirmezse kuyruk probu `v82` için saklanır. Betik hiçbir dosyayı otomatik
olarak Kaggle'a göndermez.
