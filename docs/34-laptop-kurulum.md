# İkinci makineye taşıma (laptop) — 2026-08-23

> ### ⚠ GÜNCEL YOL: `scripts/tasima_tam.py`
>
> Bu belge 23 Ağustos'ta yazıldı ve **konuşma geçmişini kapsamıyordu**.
> Artık tek komut her şeyi topluyor ve SHA-256 ile doğruluyor:
>
> ```powershell
> python scripts/tasima_tam.py --hedef E:/DATAHON_TASIMA    # kaynak makine
> python scripts/tasima_tam.py --dogrula E:/DATAHON_TASIMA  # hedef makine
> ```
>
> Paket dört parça taşır (~1,70 GB): `data/` tamamı (899 MB, içinde
> yeniden çekilmesi kotaya takılan `data/external`), seçili gönderimler
> (217 MB), **Claude konuşma geçmişi + memory** (512 MB, yeniden
> üretilemez) ve gizli dosyalar (`.env`, `kaggle.json`).
> Paketin içine adım adım `KURULUM.md` yazılır.
>
> Aşağısı, senaryo anlatımı için duruyor.

Hedef makine: Lenovo Ideapad Slim 3 15IRH10, i7-13620H (6P+4E / 16 iş
parçacığı), **24 GB RAM**, 1 TB SSD.

## Önce: klonlamak YETMEZ

```
git clone getirir        16 MB   (287 dosya: kod, belge, test)
gelmeyen data/          842 MB
gelmeyen submissions/   649 MB
```

`.gitignore` tüm `*.csv` ve `*.parquet` dosyalarını dışarıda bırakıyor.
Klonlanmış depoda kod var, işleyecek veri yok.

## Ne yapmak istediğinize göre iki senaryo

### A. Sadece GÖNDERİM yapacaksanız — ~100 MB, 10 dakika

Eğitim yok, sadece hazır dosyaları Kaggle'a yollamak.

```powershell
git clone <depo> Datahon ; cd Datahon
pip install kaggle
# kaggle.json'i %USERPROFILE%\.kaggle\ altina kopyala (git'te YOK, sir)
# submissions/ icinden yalnizca gonderilecek CSV'leri kopyala (28 MB/dosya)
kaggle competitions submit -c grid-up-datathon -f submissions/tuketim_v25_hedge.csv -m "..."
```

### B. EĞİTİM de yapacaksanız — ~1 GB kopyalama, ~1 saat kurulum

**Yol 1 — ÖNERİLEN: `data/` klasörünü olduğu gibi kopyalayın** (USB/bulut,
842 MB). Dış veri çekme betikleri bu gece kırılgan çıktı (Overpass IP'mizi
engelledi, EPİAŞ 503 verdi) — sıfırdan çekmeye güvenmeyin.

**Yol 2 — sıfırdan:** `kaggle competitions download -c grid-up-datathon`
ile ham veriyi indirin, sonra `scripts/fetch_*.py` ile dış veriyi çekin.
Saatler sürer ve kaynakların ayakta olmasına bağlıdır.

Kurulum:

```powershell
git clone <depo> Datahon ; cd Datahon
py -3.12 -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -r requirements\constraints-py312.txt
# data/ klasorunu buraya kopyala
python scripts\tuketim_model.py --sadece-dogrulama    # ~12 dk, kurulum testi
```

Önbellek yoksa `deney.cerceveleri_kur()` ham veriden **kendisi kurar**;
`data/raw/` ve `data/external/` yeterlidir, `data/interim/` yeniden üretilir.

## RAM: sığıyor ama dar

Ölçüldü (bu depoda, gerçek çerçevelerle):

```
uretim cercevesi   4.148.269 satir x 157 kolon   4,65 GB
105 uretim kolonu                                3,48 GB
TAHMINI TEPE RAM                           11,6 - 18,6 GB
laptop RAM                                       24    GB
```

24 GB'in ~4'ünü Windows alır, kalan ~20 GB tepe değere yakın. Bu yüzden:

- Eğitim sırasında tarayıcı ve diğer uygulamaları **kapatın**
- Pagefile'i SSD'de **32 GB**'a çıkarın (tepe aşılırsa çökmez, yavaşlar)
- **Şarjda** çalıştırın; pilde CPU yarı güce iner, eğitim iki katına çıkar

## Hız beklentisi

Mevcut makine Ryzen 5 7600 (6 çekirdek / 12 iş parçacığı, 47 GB).
Laptop 16 iş parçacığı ama dizüstü ısı sınırıyla; sürekli yükte kabaca
**%10–30 daha yavaş**. 55 dakikalık tam eğitim orada **60–80 dakika**.

## Kontrol listesi

```
[ ] Windows kurulu mu (ilan FreeDOS diyor -- kurulmadiysa once o)
[ ] Python 3.12
[ ] git clone
[ ] data/ klasoru kopyalandi (842 MB)
[ ] .kaggle/kaggle.json kopyalandi
[ ] pip install -r requirements/constraints-py312.txt
[ ] python scripts/tuketim_model.py --sadece-dogrulama  -> hatasiz bitiyor
```

Son satır geçerse makine hazırdır.
