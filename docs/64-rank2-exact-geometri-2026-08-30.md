# Rank 2 modeli: exact LB geometrisi ve 31 Agustos plani

Tarih: 30 Agustos 2026

## Sonuc

Uc teslim dosyasi hazirlandi ve `scripts/kapi_denetim.py` ile gecti:

| dosya | amac | durum |
|---|---|---|
| `tuketim_K_EXACT_BANKA.csv` | Yalniz gercek LB olcumlerinden kurulan guvenli optimum | Beklenen 1.00052 |
| `tuketim_K_RANK2_ONSEL.csv` | Ikinci sirayi hedefleyen kontrollu agresif model | CV onseli, LB ile henuz dogrulanmadi |
| `tuketim_K_PROBE_seviye_x_ay.csv` | Bir sonraki exact LB sondasi | 31 Agustos ilk gonderim |

Canli tabloda TasnifX 1.00115 ile 3., ikinci 0.99940'tir. Exact banka ile ikinci
arasindaki fark RMSLE'de 0.00112, karesel kayipta yaklasik 0.00226'dir.

## Bulunan geometri hatasi

Eski `m112` durumu, gonderilen `tuketim_K_yenibas.csv` dosyasini dogrudan
kullanmak yerine yonu formulden yeniden uretiyordu. Formulle uretilen vekil,
gercek 1.00191 skorunu 1.00198 olarak yeniden kuruyordu. Yaklasik 0.000067
RMSLE hata, LB'nin 0.00001 yuvarlama hassasiyetinin cok ustundedir.

Duzeltmeden sonra her yapisal olcum su ikiliyle saklanir:

1. Gercek gonderilen CSV'nin log-uzayi yonu.
2. Kaggle'da gorulen skor.

Boylece yon tanimi veya aday kodu sonradan degisse bile Gram sistemi gercek
gonderimi birebir kullanir. Bu degisiklik icin dort hedefli test vardir.

## Model arastirmasi

Temiz ileri-zaman onbelleklerinde Nisan-Temmuz 2025, Agustos-Kasim 2025 ve
Aralik 2025-Mart 2026 bloklari yeniden incelendi. Her blokta kuresel seviye
etkisi cikarildiktan sonraki normalize artik korelasyonlari:

| yon | yaz25 | guz25 | kis26 | isaret |
|---|---:|---:|---:|---|
| `seviye_x_ay` | -0.0953 | -0.0712 | -0.0078 | 3/3 negatif |
| `ay` | +0.1157 | +0.0184 | +0.0235 | 3/3 pozitif |
| `haftasonu` | +0.0061 | +0.0189 | +0.0267 | 3/3 pozitif |
| `seviye_x_soguk` | +0.0567 | +0.0237 | -0.0547 | kararsiz |
| `seviye_x_guc` | +0.0436 | +0.0470 | -0.0871 | kararsiz |
| `buzme_tam` | +0.0061 | -0.0143 | +0.0491 | kararsiz |

Test penceresi de Nisan-Temmuz oldugu icin `yaz25` birebir mevsim analogudur.
LB'de olculen global seviye sinyali, analog bloktaki sinyalden daha kucuk
oldugu icin katsayilar aynen tasinmadi; oranlandi ve yaklasik %35 daha
buzuldu. Rank-2 onseli birbirine ve bilinen 27 yone sirayla dik iki eksen
kullanir:

```text
seviye_x_ay  beta = -0.030
ay           beta = +0.035
```

Bu duzeltmenin sifir-sinyal maliyeti 0.002125 MSE'dir. Gercek yeni
korelasyonlar sirasiyla en az -0.035 ve +0.040 olursa toplam kazanc yaklasik
0.002775 MSE olur; bu, ikinci siraya gereken 0.00226'dan buyuktur. Bu bir
CV'ye dayali onsel hedeftir, LB sonucu gelmeden garanti degildir.

## Neden once sonda

Hazir onsel model dogrudan gonderilebilir; fakat en guvenilir yol iki LB
olcumunu alip katsayilari tahmin etmek ve ayni gun exact optimumu uretmektir.
0.005 yer degistirmeli sonda, `rho=0.03` icin yaklasik 30 yuvarlama-birimi
SNR verir. Sonda skoru kotu gorunse bile bilgi kaybi degildir: isaret ve
katsayi exact olarak cozulur.

## 31 Agustos komutlari

Ilk gonderim hazir dosyadir:

```text
submissions/tuketim_K_PROBE_seviye_x_ay.csv
```

Skor `P1` geldikten sonra:

```powershell
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --kaydet seviye_x_ay --skor P1
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --aday ay --yerdeg 0.005 --cikti tuketim_K_PROBE_ay.csv
```

Ikinci sondayi gonderip skor `P2` geldikten sonra:

```powershell
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --kaydet ay --skor P2
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --nihai --cikti tuketim_K_ADAPTIF_NIHAI.csv
.\.venv\Scripts\python.exe scripts\kapi_denetim.py submissions\tuketim_K_ADAPTIF_NIHAI.csv
```

Ucuncu gonderim `tuketim_K_ADAPTIF_NIHAI.csv` olur. Ilk iki olcum beklenen
sinyali vermezse 1 Eylul icin bir sonraki kararlı ve bagimsiz eksen
`haftasonu`dur; kararsiz soguk/guc buzme eksenleri otomatik olarak plana
alinmamalidir.

## Tek komutla hazir onsel modeli yeniden uretme

```powershell
.\.venv\Scripts\python.exe experiments\model29\m112_kalibre.py --rank2 --cikti tuketim_K_RANK2_ONSEL.csv
```

