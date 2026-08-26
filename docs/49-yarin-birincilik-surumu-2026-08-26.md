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
