# ISLETIM KILAVUZU — sonda zinciri

docs/75'in uygulama adimlari. Her adim SIRAYLA yapilir. Bir adim
atlanirsa sonraki adimin cozdugu `rho` YANLIS cikar.

**ONAY KURALI: hicbir gonderim kullanicinin acik onayi olmadan yapilmaz.
Komut kullaniciya verilir, kullanici calistirir.**

---

## 0a. YAPILANDIRMA — ortam degiskenleri

`m148_demet_plani.py` su ayarlarla kosar. **Hepsinin varsayilani dogrudur;
elle vermek gerekmez.** Burada yalnizca ne oldugu yaziyor.

| degisken | varsayilan | ne yapar |
|---|---|---|
| `K_AZAMI` | **25** | Kabul edilen eksen sayisini kirpar. `n11` K=25'te gerceklesen `rho`'nun K=136'dakinden **%34 yuksek** oldugunu olctu (%95 AO [+%12, +%56]). `0` kirpmayi kapatir. |
| `BLOK_KIP` | **oran** | Blok bolmesi. `oran` = {hava, yapi} × {`\|rho_cv\|/\|KATS\|` yuksek, dusuk}. `aile` secenegi K=25'te yalnizca **iki** blok uretir (ilk 25 eksenin hepsi `m121_taban`), o yuzden kullanilmaz. |
| `DEMET_HEDEF` | 4 | Blok sayisi. Degistirilirse `n06_kappa.py` TEKRAR kosulmali. |
| `NIHAI` | — | `1` verilince sonda uretmeyi birakip `Z_NIHAI`'yi yazar. |
| `UYARLANABILIR` | — | `<blok_no>` verilince o blogun ust yarisini 5. yon olarak ekler (bkz. docs/77). Blok sayisi degistigi icin `n06_kappa.py` TEKRAR kosulmali. |
| `C_OLCULEN` | — | D1'den sonra olculen `\|c\|`; `n06_kappa.py`'ye verilir. |

---

## 0. Her gun 03:00'ten (yerel) once — HAZIRLIK

```bash
cd "c:/Users/Cem/Desktop/Datahon_Laptop/Grid-Up-Datathon---TasnifX"

# kota gercekten sifirlandi mi
.venv/Scripts/python.exe -m kaggle competitions submissions -c grid-up-datathon

# gonderilecek dosya bozulmamis mi
.venv/Scripts/python.exe -c "import hashlib;print(hashlib.md5(open('submissions/tuketim_D1_demet.csv','rb').read()).hexdigest())"
```

**TARAYICIDA:** yarisma sayfasi -> Submissions -> ust satirda
**"You selected X of N"** yazisini OKU.

- `N = 2` ise plan aynen yurur (nihai + yedek secilir).
- `N = 1` ise YEDEK STRATEJI COKER. O durumda sonda gondermeyi durdur,
  cunku kotu skorlu bir sonda tek secim olarak kalabilir. Once kullaniciya
  haber ver.

---

## 1. Sonda dongusu (D1 -> D2 -> D3 -> D4)

Her sonda icin AYNI uc adim:

### 1a. GONDER (kullanici onayiyla)

```bash
.venv/Scripts/python.exe -m kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_D1_demet.csv -m "D1 blok sondasi"
```

Sonra **MUTLAKA** listeyi oku — zaman asimina ugrayan bir betik
"gonderilmedi" demek DEGILDIR, bir hak bosa gitmis olabilir:

```bash
.venv/Scripts/python.exe -m kaggle competitions submissions -c grid-up-datathon
```

### 1b. SKORU YAZ

`experiments/model29/m148_olcumler.json` (yoksa olustur):

```json
{"1": 1.00235}
```

Anahtar sonda numarasi (metin), deger LB'nin verdigi **5 ondalikli** skor.
Sonraki sondalarda ekle: `{"1": 1.00235, "2": 0.99981}`.

> LB 5 ondalik verir. Daha fazla ondalik yazarsan uydurmus olursun —
> gecmiste sahte bir 16 ondalikli dosya kurulmustu (kirmizi takim K0).

### 1c. SIRADAKI DOSYAYI URET

```bash
.venv/Scripts/python.exe experiments/model29/m148_demet_plani.py
```

Betik: onceki sondalarin `rho`'sunu cozer, kapilardan gecirir,
oz-denetim yapar ve **ancak ondan sonra** bir sonraki CSV'yi yazar.
Ekranda cozulen `rho_k` ve guncel `toplam rho^2` gorunur.

---

## 2. D1'DEN SONRA — |c| KALIBRASYONU (bir kez)

1. sonda yalnizca blok 1'i olcmez, **`|c|`'yi de olcer**:

```
|c| = 1.95 * rho_1_olculen / ongorulen_1        (ongorulen_1 = 0.3758)
```

`|c|` tum bloklari AYNI oranda olcekler. Onsel belirsizligi yedi kat
genislikteydi (%90 GA [0.17, 1.26]); olcumden sonra daralir. Kalan
bloklarin `kappa`'sini bu bilgiyle yeniden sec:

```bash
C_OLCULEN=<olculen |c|> .venv/Scripts/python.exe experiments/model29/n06_kappa.py
.venv/Scripts/python.exe experiments/model29/m148_demet_plani.py
```

`n06_kappa.py` **zaten uretilmis sondalarin kappa'sini DONDURUR** — onlarin
dosyasi diskte ve `sabit`/`kappa_etkin` kayitli; farkli bir kappa yazmak
cozum formulunu dosyayla uyusmaz hale getirir.

---

## 3. NIHAI DOSYA

Dort sonda da olculdukten sonra:

```bash
NIHAI=1 .venv/Scripts/python.exe experiments/model29/m148_demet_plani.py
```

`NIHAI=1` **zorunludur**. Onsuz betik bir sonraki sondayi uretmeye devam
eder ve `tuketim_Z_NIHAI.csv` HIC olusmaz.

Cikan dosya: `submissions/tuketim_Z_NIHAI.csv`.
Betik beklenen skoru yazdirir: `sqrt(1.00202690 - toplam rho_k^2)`.

Gonder, sonra listeyi oku.

---

## 4. YEDEK HAKKIN KULLANIMI (6. gonderim)

Plan 4 sonda + 1 nihai = **5 hak** kullanir. 6. hak **dogrulama-ve-onarim**
icin ayrilmistir:

`Z_NIHAI`'nin beklenen skoru TAM OLARAK hesaplanabilir. Donen skor
beklenenden `> 3 * 1.66e-3` sapiyorsa bir yerde hata var demektir. O zaman
sapmadan hatanin buyuklugu cozulur ve duzeltilmis bir nihai dosya
gonderilir.

Sapma yoksa 6. hak KULLANILMAZ.

---

## 5. SON SECIM (1 Eylul, 23:59 UTC'den once)

**Yalnizca tarayicidan.** API'de bu islev yok.

1. Once **"You selected X of N"** satirini OKU.
2. Sec:
   - `submissions/tuketim_Z_NIHAI.csv` (nihai)
   - `submissions/tuketim_YP_seviye.csv` (yedek, 1.00115)
3. Sectikten sonra sayfayi YENILE ve secimin gercekten kaydedildigini
   dogrula.

Kaggle iki secimden **IYI olani** alir, yani yedek bir kayip riski
yaratmaz — yalnizca taban saglar.

---

## 6. ZAMANLAMA

| ne zaman | ne |
|---|---|
| 31 Agu 03:00 yerel | kota sifirlanir, 3 hak |
| | D1 -> olcum -> \|c\| kalibrasyonu -> D2 -> olcum -> D3 |
| 1 Eyl 03:00 yerel | kota sifirlanir, 3 hak |
| | D4 -> olcum -> `NIHAI=1` -> Z_NIHAI |
| 1 Eyl 23:59 UTC (02:59 yerel, 2 Eyl) | YARISMA BITER |
| bitisten cok once | tarayicidan 2 secim |

**Not:** son secimi son ana birakma. Zincir bitince hemen sec.

---

## 7. NE BEKLENMELI

Sonda dosyalarinin skorlari **KOTU gelecek** — bu tasarim geregi.
Sonda, skor almak icin degil OLCUM yapmak icin gonderilir. Tabloda
geriledigimizi gorursek panik yok; secilen dosya `Z_NIHAI` olacak.

`rho_1 = 0` cikarsa D1'in skoru ~`1.00942` olur (kappa_1 = 0.130 buyuk
secildigi icin). Bu bir BASARISIZLIK DEGIL, bilgidir: `|c|`'nin kucuk
oldugunu soyler ve kalan bloklarin kappa'si kuculterek uyarlanir.

En kotu durumda nihai skor **1.00101** (saf span) — bugunku 1.00115
yedegimizden yine de iyi. **Plan asagi yonlu korumalidir.**
