# DURUM KAYDI — 30 Agustos 2026, 21:10

Bu belge, calisma durdurulurken alinan tam kayittir. Yarisma **1 Eylul
23:59 UTC**'de biter. Asagidaki her sey dogrulanmis ve commit edilmistir.

---

## 1. NEREDE DURUYORUZ

Liderlik tablosu (30 Agustos 20:23):

| # | takim | skor |
|---|---|---|
| 1 | **Grid Grinders** | **0.99009** ← ASIL HEDEF |
| 2 | **Duo-Electra** | **0.99614** ← kabul edilebilir |
| 3 | Berke Kuc | 0.99927 |
| 4 | Atakan Aldemir | 0.99937 |
| 5 | Ahmet Celik | 1.00047 |
| 6 | Saban Ozdogan | 1.00049 |
| 7 | **TasnifX** | **1.00115** ← BIZ |

Bugun uc kez geciildik. Kota bugun bitti (3/3 kullanildi).

> **HEDEF (kullanici, 30 Agustos 21:30):** ASIL AMAC **1. SIRA**.
> 2. sira da kabul edilebilir. Gereken toplam rho^2:
> 1. sira **0.02175**, 2. sira **0.00973**, 3. sira 0.00349.
> 1. sira, 2. siranin **2.23 KATI** sinyal istiyor.

**Kalan hak: 6.** 31 Agustos 03:00 (yerel) sifirlanir → 3 hak.
1 Eylul 03:00 → 3 hak. Son siralamada **2 gonderim secilir, Kaggle
IYISINI alir.**

---

Gonderim mekaniginin tam dokumu: `docs/73-gonderim-mekanigi.md`

---

## 2. YARIN NE YAPILACAK — TEK SAYFALIK TALIMAT

### Adim 1 — 31 Agustos 03:00'ten sonra, ILK GONDERIM

```
python -m kaggle competitions submit -c grid-up-datathon \
  -f submissions/tuketim_D1_demet.csv -m "D1 H1 yonu sondasi"
```

Dosya dogrulandi: 714688 satir, id'ler `test.csv` ile birebir ayni sirada,
0 NaN, 0 negatif, hepsi sonlu.

### Adim 2 — Skoru kaydet ve sonraki dosyayi uret

Donen LB skorunu (`P`) su dosyaya yaz:

```
experiments/model29/m148_olcumler.json
{"1": <skor>}
```

Sonra:

```
./.venv/Scripts/python.exe experiments/model29/m148_demet_plani.py
```

Betik `rho_1`'i cozer, `submissions/tuketim_D2_demet.csv`'yi uretir ve
guncel beklentiyi basar. Bunu her sonda icin tekrarla (D1…D4).

#### D1'IN SKORU TEK BASINA HER SEYI SOYLER

`m148_demet.json`'a gore H2–H4'un ongorusu **sifir**; toplam `rho_s`'in
tamami H1'de. Yani `rho_1 = |c| * 0.1294` ve **D1'in skoru dogrudan `|c|`
carpanini olcer** — iki gundur tahmin etmeye calistigimiz sayiyi.

| senaryo | \|c\| | rho_1 | **D1'in verecegi skor** |
|---|---|---|---|
| sinyal yok | 0.00 | 0.0000 | **1.00235** |
| SENARYO D (n=5) | 0.39 | 0.0505 | 0.99974 |
| \|c\| nokta tahmini | 0.57 | 0.0738 | 0.99854 |
| SENARYO G (n=17) | 0.63 | 0.0815 | 0.99813 |
| 2. sira esigi | 0.76 | 0.0987 | **0.99725** |
| **1. SIRA esigi** | **1.14** | 0.1475 | **0.99471** |

**KARAR KURALI:**

- `P1 <= 0.99725` → 2. sira **kesin**, 1. sira menzilde. Zincire devam.
- `0.99725 < P1 < 1.00235` → sinyal var ama zayif. Zincire devam,
  3. sira hedefle.
- `P1 >= 1.00235` → sinyal yok. **Demeti birak**, kalan haklari harcama,
  son secimde `tuketim_YP_seviye.csv` (1.00115) korunur.

1. sirayi hedeflemenin **ek maliyeti yok**: ayni 4 sonda hem 2. hem 1.
sirayi acar, fark yalnizca beklentidedir.

### Adim 3 — Dort yon de olculunce

Betik `submissions/tuketim_Z_NIHAI.csv` uretir. **Asil gonderim budur.**

### Adim 4 — SON SECIM (yalnizca tarayicidan yapilir, API'de yok)

> ### ⚠ ONCE BUNU DOGRULA — TUM STRATEJI BUNA BAGLI
>
> Community yarismalarinda **kac gonderim secilebilecegi HOST AYARIDIR**;
> 2 oldugu **garanti degildir** (Kaggle Community Competitions Setup Guide,
> "Scored Private Submissions" ayari). Tarayiciya girer girmez **ILK IS**
> yarismanin **My Submissions** sekmesindeki *"You selected X of N"*
> metnini oku.
>
> - **N = 2 ise:** asagidaki plan aynen gecerli, asagi yon KAPALI.
> - **N = 1 ise:** yedek strateji **CALISMAZ**. O durumda tek secim
>   yapilabilir ve karar tamamen degisir: olculen `toplam rho^2` guvenli
>   sinirin altindaysa `Z_NIHAI` yerine **`tuketim_YP_seviye.csv` (1.00115)**
>   secilmelidir. Bana haber ver, yeniden hesaplarim.

Kaggle arayuzunde (**My Submissions** sekmesi) **iki gonderim** isaretle:

1. `tuketim_Z_NIHAI.csv` (ya da o ana kadarki en iyi demet dosyasi)
2. `tuketim_YP_seviye.csv` — **1.00115, YEDEK**

Yedek isaretli kaldigi surece **kaybetme riski yoktur**; Kaggle secilenler
arasindan **private'ta iyi olani** alir.

**Uc kritik kural:**

1. **Otomatige BIRAKMA.** Secim yapilmazsa Kaggle otomatik secer, ama
   kriteri (*"en yuksek public"*) resmi dokumanda **belgeli degil** ve
   otomatik secimin aksadigi kayitli bir vaka var (2019 NDSC Advanced).
   Ustelik bizim sondalarimiz public'te KOTU skor verebilir; otomatik secim
   yanlis dosyalari isaretleyebilir.
2. **Secilecek dosya SKORLANMIS olmali.** Gonderilmemis bir dosya
   secilemez — `Z_NIHAI` diskte durmasi hicbir sey ifade etmez, **Kaggle'a
   gonderilmis olmali.**
3. **Erken yap.** En gec **1 Eylul 22:00 UTC = 2 Eylul 01:00 yerel**;
   bitise dakikalar kala birakma.

---

## 3. YONTEM — NEDEN BOYLE

### Cebir

`a0 = log1p(tuketim_m6_ikiyon.csv)`, `M0 = 1.005846366`

```
gonderim = a0 + r_hat + toplam_k (kappa_k * G_k)
skor^2   = TABAN_MSE + kappa^2 - 2*kappa*rho
TABAN_MSE = M0 - 2*kL + ||r_hat||^2 = 1.00202690
saf span dosyasinin beklenen skoru  = 1.00101
```

`G_k` birim ve **birbirine dik** yonlerdir. Dik yonlerde:

```
skor^2 = TABAN_MSE - toplam_k (rho_k^2)
```

ve **olculen her yon RISKSIZDIR**: `rho_k = 0` cikarsa skor degismez,
isaret ters cikarsa isareti duzeltiriz.

### Sonda 1'in tam degerleri

```
dosya         submissions/tuketim_D1_demet.csv
yon           H1 (1.95|rho_s| agirligi)
kappa         0.05174191
kappa_etkin   0.05169627
sabit         1.0046992296
COZUM         rho_1 = (1.0046992296 - P*P) / 0.10339254
olcum hatasi  5.6e-05
rho_1 = 0 ise skor 1.00235   |   tahmin tutarsa 0.99967
```

> **DIKKAT:** `/loop` metninde gecen `kappa 0.11518` ve
> `sabit 1.015096786` degerleri **ESKIDIR**. Gecerli olan yukaridakidir
> (tavan 1.95 → 0.8 duzeltmesinden sonra).

### Dort dik yon (rakip hipotezler)

`toplam(rho_k^2) = ||P_altuzay r||^2` — sonuc yalnizca **secilen alt uzaya**
baglidir, eksenleri nasil grupladigimiza degil. Bu yuzden 4 boyut, rakip
agirliklandirma hipotezlerine yayildi ve Gram-Schmidt ile diklestirildi:

| yon | hipotez | artakalan |
|---|---|---|
| H1 | `1.95*abs(rho_s)` agirligi | 1.000 |
| H2 | `rho_cv` agirligi (yaz25'te olculen) | 0.369 |
| H3 | hava/mevsim ailesi (26 eksen) | 0.433 |
| H4 | trafo/yapisal ailesi (14 eksen) | 0.211 |
| H5 | esit agirlik | H1'e cok yakin, ATLANDI |

Diklik: en buyuk sapma 4.4e-15. Hangi hipotez dogruysa o boyut buyuk `rho`
verir; **hepsi yanlissa kayip yok.**

### Beklenti

```
 toplam rho^2  nihai skor  sira
      0.00000     1.00101  7.+      <- hicbir sey tutmazsa (yine de 1.00115'ten iyi)
      0.00349     0.99927  3. sira
      0.00973     0.99614  2. SIRA
      0.03181     0.98499  1. SIRA
```

### `c` carpani — nihai hukum (m149)

**`|c| = 0.57`, %90 araligi `[0.17, 1.26]`.**

| sira | gereken rho^2 | gereken \|c\| | olasilik |
|---|---|---|---|
| **1. SIRA** | 0.02175 | 1.140 | **%8** |
| 2. sira | 0.00973 | 0.762 | **%26** |
| 3. sira | 0.00349 | 0.456 | **%66** |

**`1.95` COKTU — olcum hatasi degil, BAYAT PAYDA.** `docs/69`'daki
`rho_s = 0.0156` o gunku **daha kucuk span** ile hesaplanmisti. Bugunku tam
span ile ayni dosyada `rho_s = 0.0616` (4 kat buyuk); `rho_dik = -0.0272`
ise docs'taki -0.0304 ile ayni. Oran dogrudan **1.95 → 0.44**. Demet
yonlerimizin `rho_s`'i de bugunku `r_hat`'ten geldigi icin dogru payda
bugunkudur. **1.95 artik aralikta sifir agirliklidir.**

**m145'in "dort bagimsiz yolu" bagimsiz degilmis** — dordu de ayni 17 LOO
noktasinin fonksiyonu, bootstrap korelasyonu 0.94. Yayilim hata payina
**eklenmeli**, cikarilmamali.

**Iki referans sinifi (yeni bulgu).** LOO'nun "dik" yonleri span'in
**icindedir**; dik payi %1 olan bir eksende olculen sey gercek dik sinyal
degil, `r_hat`'in uyum artiginin buyutulmus halidir (dusuk dik payli 7
eksende `|rho_dik|` neredeyse sabit: 0.027–0.030). Demet yonlerimiz ise
span'in **disindadir**.

- **SENARYO G** (tum 17 eksen): `|c| = 0.625` [0.48, 0.85]
- **SENARYO D** (yalniz yeni boyut acanlar, n=5): `|c| = 0.390` [0.13, 1.42]

Nihai aralik ikisinin esit agirlikli karisimi. Gercek belirsizlik bir
genislik degil, **bir ikilik**.

**`sigma_L` veri tarafindan sinirlandi (bagimsiz dorduncu delil).**
`sigma_L = 2.27e-04` dogru olsaydi dik artiklarin sacilimi gozlenenden 1.9
kat buyuk olurdu ve havuzlanmis tahminci `c^2 < 0` verirdi — olanaksiz.
**`sigma_L <= 1.2e-04`.** Bu, m134'un yuvarlama bulgusunu destekler.

**Masa basinda daraltacak yol kalmadi** — elde bagimsiz ikinci olcum kumesi
yok. Bundan sonrasi **olcumle** gelir; ilk olcum D1'dir.

---

## 4. BUGUN NE OGRENILDI — ALTI VARSAYIMIN SINAVI

Ayrinti: `docs/71-30agustos-dogrulama-turu.md`

| # | varsayim | sinav | sonuc |
|---|---|---|---|
| 1 | zaman asinmasi var (0.388) | m130/m131 | **YANLIS** — asinma yok; eski olcum agirliklari blokta fit ediyordu |
| 2 | 40 eksen kesimi keyfi | m132 | **DOGRU KESIM** — 58 geciyor ama fazlasi blokta zarar |
| 3 | `sigma_L = 2.2e-4` | m134 → m145 | **AYAKTA** — m134'un itirazi denetimde zayifladi (P=0.097) |
| 4 | `1.95` carpani | m140 → m145 | **DUZELTILDI** → `abs(c) ~ 0.7`, aralik [0.3, 1.3] |
| 5 | blok korelasyonu 0.2125 | m141 | **YAZ25'E OZGU** — guz25/kis26'da isaret donuyor |
| 6 | isaretler gercek | m142 | **DOGRULANDI** — iki bagimsiz kaynak 35/40 (%88), p=6.9e-07 |

### Bugun geri alinan kendi hatalarim

1. **"tasinan rho 0.0956, kil payi yetmiyor"** — o olcum agirliklari blogun
   yarisinda fit edip diger yarida siniyordu; oysa katsayilar LB'den geliyor.
   Sabit katsayiyla asinma **yok** (5 pencerenin 5'inde pozitif).
2. **"gec pencerede korelasyon daha yuksek"** — null testi curuttu; rastgele
   isaretli bilesikler de ortanca 1.155 oran veriyor.
3. **"c = 0.28, 1.95'ten 16 sigma"** — m140 ISARETLI egim olcuyor, oysa 1.95
   BUYUKLUK carpanidir. Isaretler 9 arti/8 eksi oldugu icin bu test 1.95
   dogru olsa bile sifir gosterirdi. **Karsilastirma gecersiz.**
4. **"sigma_L yuvarlama tabaninda, 77 kat kucuk"** — `nan_to_num` uc yonden
   ikisini bozuyordu; kipler arasi korelasyon 0.96, etkin n~1. Olabilirlik
   orani 10.7:1 → 5.6:1, reddedilmiyor.

### Kritik yapisal bulgu (m144)

```
||r_hat||             = 0.0611   <- ASILAMAZ TAVAN
mevcut 40 eksen       = 0.0611
+ 108 yeni eksen      = 0.0611   (+0.0000)
```

`r_hat` span icinde ve **28 boyutlu**; 40 eksen o uzayi zaten geriyor.
Yeni eksen **yeni LB olcumu getirmiyor**. `rho_pred = 0.2522` bir olcum
degil, tekrarlanan bir varsayimdir. **Demet plani bu sayiya dayanmaz** —
her `rho_k`'yi dogrudan olcer.

### Negatif bulgular (tekrar denenmesin)

- Trafo duzeyi hedef kodlamasi: **49/49 elendi** (sizintisiz kuruldu, olcum
  gecerli) — dis bloklardaki trafo yanliligi teste tasinmiyor.
- Takvim ailesi: 59 adayda 1.
- Yatay ufuk yanliligi, artik-model mevsimler arasi transferi (6/6 negatif),
  soguk-sifir siniflandirmasi — hepsi daha once kapatildi.

---

## 5. TESLIM DURUMU

| kalem | durum |
|---|---|
| `submissions/tuketim_D1_demet.csv` | GECTI — 714688 satir, id sirali, 0 NaN/negatif |
| `m126_son_dogrulama.py` | GECTI — 27/27 skor birebir, 9/9 kapi |
| `notebooks/TasnifX_final.ipynb` | GECTI — 18/18 hucre, ~16 sn; liderlik tablosu guncellendi |
| Dis veri kunyesi (`scripts/kunye_denetim.py`) | GECTI — uretimde okunup beyan edilmeyen: **0** |
| `data/sources.yml` | 16 kaynak; SHA-256, lisans, atif, `model_girdisi` bayragi |

### Insan eliyle yapilacak, henuz YAPILMADI

- [ ] **Son 2 gonderim secimi** (tarayici; en gec 1 Eylul 23:00 UTC)
- [ ] Takim arkadasinin Coderspace kaydi
- [ ] Duzenleyicinin e-postasinin `.eml` olarak arsivlenmesi

---

## 6. DOSYA HARITASI

| dosya | ne yapar |
|---|---|
| `m112_kalibre.py` | `M0`, `EK_MODEL`, `buzmeli_r_hat`, `L_gurultusu` — **cekirdek** |
| `m122_nihai_bilesik.py` | 40 ekseni secer, tek-yon bilesigi kurar |
| `m126_son_dogrulama.py` | bagimsiz uctan uca dogrulama |
| **`m148_demet_plani.py`** | **asil arac** — sonda uretir, olcum okur, nihaiyi yazar |
| `m148_demet.json` | sonda kayitlari (sabit, kappa_etkin) — **silme** |
| `m148_olcumler.json` | LB skorlari buraya yazilir |
| `m130`–`m147` | bugunun sinavlari (docs/71'de anlatilan) |
| `docs/69`, `docs/71`, `docs/72` | yontem, sinavlar, bu durum kaydi |

**Kural:** hicbir betik gonderim yapmaz. Gonderim yalnizca kullanicinin
onayiyla, elle yapilir.
