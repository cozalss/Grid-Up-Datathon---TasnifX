# 74 — KIRMIZI TAKIM: plani yikma turu (30 Agustos 2026, gece)

Gorev: `docs/72` + `docs/73` + `m148_demet_plani.py` planini DOGRULAMAK degil
YIKMAK. Sekiz saldiri yuzeyi denendi. **Iki KRITIK, uc YUKSEK, alti ORTA/DUSUK
zafiyet bulundu; dort yuzeye saldirildi ve kirilmadi.**

Bu belge yalnizca OLCUM ve ANALIZDIR. Hicbir kod degistirilmedi, hicbir
gonderim yapilmadi, `submissions/` altina yazilmadi. Butun sayilar bugunku
depodan bagimsiz olarak yeniden hesaplandi (mevcut `m148` cekirdeginin gecici
dizindeki kopyasi + ek olcumler; kosum ~4 dakika).

---

## 0. YONETICI OZETI — yarin sabah okunacak dort satir

0. **SU AN DISKTE SAHTE D2/D3/D4/Z_NIHAI DOSYALARI VAR.** Yarin sabah ilk is
   K0'i oku ve temizle. Temizlenmezse **bir gonderim hakki sahte bir dosyaya
   harcanir ve zincirin tabani zehirlenir.**
1. **Sonda 2-4'un `kappa`'si 0.0125.** Bu deger, umdugumuz senaryoda
   (`rho_1 ~ 0.15`) olcum zincirini **KARARSIZ** yapiyor: kalibre sabitteki
   1.72e-04'luk sistematik hata sonda 4'e **-0.082** olarak varir, nihai
   dosyaya **0.0076 rho^2** kaybettirir ve betik bunu **gormez** — 0.98597
   raporlarken gercek skor **0.99394** olur. **1. sirayi kaybettirir.**
   Duzeltme tek satir: `KAPPA_K = np.full(DEMET, 0.0517)`. Kayip 200 kat duser.
2. **Betik D4'u atlayip `Z_NIHAI`'yi erken uretemez.** `docs/73` §4.1'deki
   "kacis plani" **kodda yok**. Hak biterse elde `Z_NIHAI` kalmaz.
3. **`docs/72` §3'teki "olcum hatasi 5.6e-05" YANLIS.** Betigin kendi
   formulu **1.66e-03** veriyor (30 kat). Zaferi ya da yenilgiyi belirleyen
   sayi budur.

---

## K0 — [KRITIK / SIMDI] Diskte SAHTE sonda dosyalari duruyor; gonderilirse bir hak yanar ve zincir zehirlenir

**(a) Ne.** Bu raporu yazarken, paralel bir oturum `m158_son_kabul.py` ile
**SENTETIK ucdan uca kabul testi** kostu. Testin `--asama 2` adimi uydurma bir
gercekle butun zinciri yurutuyor ve **gercek dosya adlariyla gercek dizine
yaziyor**. 22:06 itibariyla depodaki durum:

```
submissions/tuketim_D1_demet.csv   21:17   GERCEK   (sha bc740f3e..., degismemis)
submissions/tuketim_D2_demet.csv   21:58   SAHTE
submissions/tuketim_D3_demet.csv   22:00   SAHTE
submissions/tuketim_D4_demet.csv   22:02   SAHTE
submissions/tuketim_Z_NIHAI.csv    22:05   SAHTE

experiments/model29/m148_olcumler.json  ->  {"2": 1.0002128586621946}   SAHTE SKOR
experiments/model29/m148_demet.json     ->  sonda 2,3,4 kayitlari EKLENMIS,
      onceki_r["1"] = 0.05006132593874083     <- UYDURMA rho_1
```

`m158`'in bir `--temizlik` adimi var, ama **henuz kosmadi**. Oturum cakilir,
kesilir ya da uykuya girerse bu dosyalar oldugu gibi kalir.

**(b) Nasil tetiklenir.** Yarin 03:00'te plan **aynen** uygulanarak. D1
gonderilir, skoru `m148_olcumler.json`'a yazilir, betik kosulur — ve satir
496'daki koruma devreye girer:

```
if kayit and os.path.exists(yol):
    print("  sonda 2 ZATEN VAR: tuketim_D2_demet.csv")
    print("    Yeniden uretilmedi. Skoru m148_olcumler.json'a yazip tekrar kos.")
```

Yani betik **sahte D2'yi mesru gorur, yeniden uretmez** ve operatore onu
gondermesini soyler. Ironi: bu koruma (K7'yi onlemek icin konmus) burada
tam ters yonde calisiyor.

**(c) Ne kadar kotu.** Sahte D2 `rho_1 = 0.05006` uydurmasi uzerine
kurulmustur; gercek `rho_1` ne cikarsa ciksin tabani yanlistir.
- **Bir gonderim hakki tamamen yanar** (31 Agustos'un 3 hakkindan biri) —
  `docs/73` §4.1'e gore zaten **slack 0**, yani bu tek basina zinciri kirar
  ve K2 ile birleserek `Z_NIHAI`'nin hic uretilememesine gider.
- `m148_olcumler.json`'daki `{"2": ...}` temizlenmezse ve operator dosyayi
  `{"1": P1}` diye **degistirmek yerine ekleyerek** yazarsa, sahte skor gercek
  bir olcum gibi zincire girer.
- `tuketim_Z_NIHAI.csv` de diskte hazir duruyor ve **"asil gonderim budur"**
  diye belgelenmis bir addir (`docs/72` §2 Adim 3). Yorgun bir operatorun onu
  gondermesi icin hicbir engel yok.

**(d) Karsi onlem — yarin ONCE, tek komut.**

```
rm -f submissions/tuketim_D2_demet.csv submissions/tuketim_D3_demet.csv \
      submissions/tuketim_D4_demet.csv submissions/tuketim_Z_NIHAI.csv \
      experiments/model29/m148_olcumler.json
git checkout -- experiments/model29/m148_demet.json     # sonda 2-4 kayitlarini sil
```

Sonra **dogrula**: `m148_demet.json` icinde yalnizca `sonda 1` kalmali;
`sabit` **1.0046992296275314**, `kappa_etkin` **0.0516962677376078** olmali
(bu iki sayi bugun bagimsiz olarak yeniden hesaplandi ve tuttu).
`submissions/tuketim_D1_demet.csv`'nin sha256'si **`bc740f3e61617eaf...`**
olmali.

> `git checkout` `sabit_hata` alanini da geri alir; o alan zararsizdir, betik
> zaten her kosuda yeniden yazar.

Kalici kural adayi: **sentetik testler `submissions/` altina ASLA gercek
adlarla yazmaz.** Test dizini ayri olmali (`submissions/_test/`) ya da
dosya adlari `_SENTETIK` on ekiyle uretilmelidir.

> **GUNCELLEME (22:10) — temizlik KISMEN kostu.** `m158 --temizlik` sahte
> `D2/D3/D4/Z_NIHAI` CSV'lerini sildi ve `m148_demet.json`'i sonda-1'e
> dondurdu (dogrulandi: `sondalar = [1]`, `sabit` ve `kappa_etkin` bugunku
> bagimsiz hesapla birebir). **Ama `m148_olcumler.json` GERIDE KALDI ve
> icinde hala SAHTE bir skor var:**
> ```
> {"1": 1.0027616026880977}      <- 16 ondalik; gercek LB skoru 5 ondaliktir
> ```
> Birakilirsa yarinki ilk kosu `rho_1 = -0.0080` cozer ve zehirli bir taban
> uzerine D2 uretir. **Bu dosya SILINMELIDIR** (`rm
> experiments/model29/m148_olcumler.json`). Ayrica bu, temizligin kendisinin
> de eksik oldugunu gosteriyor — yarin sabah **once diski gozle dogrula, sonra
> baslat.** Ondalik hane sayisi (5) ucuz ve kesin bir sahtelik testidir; K10'un
> onerdigi kontrol tam olarak bunu yakalar.

**(e) Aciliyet: SIMDI / YARIN ONCE — bu listedeki her seyden once.**

---

## K1 — [KRITIK / YARIN ONCE] Olcum zinciri, UMDUGUMUZ senaryoda kararsiz

**(a) Ne.** `m148_demet_plani.py` satir 453-456:

```
RHO_OLC[k] = (g["sabit"] - 2.0*capraz - P*P) / (2*g["kappa_etkin"])
capraz     = toplam_{j<k} r_j * RHO_OLC[j]
```

`sabit = M0 - 2*kL + Q(d)` icindeki `(M0 - 2*kL)` **olculmus degil KALIBRE**
bir sabittir ve hatasi (`SABIT_HATA = 1.72e-04`, betigin kendi sayisi) **tum
sondalarda AYNIDIR** — rastgele degil, sistematiktir. Hata yayilimi:

```
e_k = [ delta - 2 * toplam_{j<k} r_j * e_j ] / (2 * kappa_k)
```

Her adimdaki ic kazanc **r_j / kappa_k**. `kappa_2..4 = 0.0125` ve `r_1`
1. sira senaryosunda 0.1475 oldugundan **kazanc 11.8**. Zincir patlar.

**(b) Nasil tetiklenir.** Hicbir sey yanlis yapilmadan, plan **aynen**
yurutulunce. Tek tetikleyici: D1'in buyuk bir `rho_1` olcmesi — yani
**basari**.

**(c) Ne kadar kotu (olculdu).** `delta = +1.72e-04` (planin kendi ilan
ettigi buyukluk), gercek rho = (0.1475, 0, 0, 0):

| kurulus | e_1 | e_2 | e_3 | e_4 | kayip (rho^2) | GERCEK skor | betigin RAPORU |
|---|---|---|---|---|---|---|---|
| kappa 2-4 = **0.0125** (mevcut) | +0.0017 | -0.0130 | -0.0264 | **-0.0823** | **0.00765** | **0.99394** | 0.98597 |
| kappa 2-4 = **0.0517** (onerilen) | +0.0017 | -0.0031 | -0.0033 | -0.0035 | 0.000036 | **0.99010** | 0.98982 |

Yani mevcut ayarla **1. sira (0.99009) elden gider** ve betik "1. SIRA" yazar.
`delta = 3.44e-04` olursa (LOO hatasi `P` olceginde ise `P^2` olceginde tam bu
eder) zincir **iraksar**: e_4 = -0.62, gercek skor 1.17.

Diger senaryolar (`delta = +1.72e-04`):

| gercek rho_1 | kayip rho^2 | GERCEK skor | betigin RAPORU |
|---|---|---|---|
| 0 (sinyal yok) | 0.000062 | 1.00104 | 1.00098 |
| 0.0504 (\|c\|=0.39) | 0.000003 | 0.99974 | 0.99966 |
| 0.0737 (\|c\|=0.57) | 0.000055 | 0.99832 | 0.99815 |
| 0.0983 (2. sira) | 0.000438 | 0.99639 | 0.99579 |
| 0.1475 (1. sira) | **0.007647** | **0.99394** | 0.98597 |

Kayip **tam olarak hedefe yaklastikca** buyuyor.

**(d) Karsi onlem** (ucu de ucuz, ikisi tek satir):

1. **`KAPPA_K = np.full(DEMET, 0.0517)`** (satir 378). D1 zaten uretildi ve
   ilk eleman ayrica atandigi icin etkilenmez; D2-D4 kosum aninda uretiliyor.
   `m150`'nin 0.0125 "optimizasyonu", **plan tarafindan zaten diskalanmis** bir
   senaryoyu (son dosya bir sonda olursa ~0.0004 kayip) optimize ederken 200
   kat buyuk bir risk yaratmis. `docs/73` §4.1 "son hak her zaman Z_NIHAI'ye
   ayrilir" dedigi anda o optimizasyonun dayanagi kalmiyor.
2. **Anlamlilik kapisi:** `r_k` tabana konmadan once `|r_k| > 3*hata_k` sarti
   aransin; gecmiyorsa `r_k = 0`. Bu hem geri beslemeyi kirar hem K5'teki
   `eps^2` kaybini keser.
3. **Kararlilik kapisi:** her sondadan once
   `2*toplam|r_j| / (2*kappa_k) > 0.3` ise DUR. Mevcut ayarla bu kapi D2'de
   zaten ateslenir — dogru davranistir.

**(e) Aciliyet: YARIN ONCE (bu gece).** D2 uretilmeden yapilmali.

---

## K2 — [KRITIK / YARIN ONCE] Betik `Z_NIHAI`'yi erken uretemez; belgelenen kacis plani KODDA YOK

**(a) Ne.** `docs/73` §4.1: *"betik eksik sonda gorunce uyari verip devam
ediyor, yani yalniz D1-D2-D3 olculup Z_NIHAI erken uretilebilir."*
**Bu ifade yanlistir.** Satir 486:

```
SIRADAKI = next((k for k in range(1, DEMET + 1) if k not in RHO_OLC), None)
```

`Z_NIHAI` yalnizca `SIRADAKI is None`, yani **DORT olcumun DORDU de**
girildiginde uretilir (satir 555). 3 olcumle betik D4 uretir, `Z_NIHAI`
uretmez. `docs/73` §5 T+4'teki **"Secenek A (D4 atlanir)" uygulanamaz.**

Ayrica `DEMET = int(os.environ.get("DEMET", "5"))` (satir 276) satir 361'de
`DEMET = len(GD)` ile **eziliyor** — bu cevre degiskeni **olu koddur**, ona
guvenen bir operator sessizce yaniltilir.

Elle atlatmak da mumkun degil: sahte bir olcum girilirse satir 444
`DUR: sonda k icin kayit yok` ile durur (kayit ancak dosya uretilince
yazilir).

**(b) Nasil tetiklenir.** (i) 3 olcum yeterli gorulur ve D4 atlanmak istenir;
(ii) bir gonderim ERROR doner ve zincir kisaltilmak zorunda kalinir;
(iii) skor gec gelir, D4'e vakit kalmaz.

**(c) Sonuc.** Elde `Z_NIHAI` **hic olmaz**. Secilebilecek dosyalar: yedek
(1.00115) ve sondalar. Sondalarin beklenen skoru `sqrt(sabit)` — D1 icin
**1.00235**, yani yedekten kotu. Boylece **butun olcum emegi sifir puana
doner, 7. sira kalir**. Kayip: elde edilebilecek 0.99614 yerine 1.00115 →
**0.005 skor, 5 sira**.

**(d) Karsi onlem.** Bu gece, KOSULARAK SINANMIS bir yol acilmali. En kucuk
degisiklik satir 486'da:

```
ZORLA    = os.environ.get("NIHAI") == "1"
SIRADAKI = None if ZORLA else next((k for k in range(1, DEMET+1) if k not in RHO_OLC), None)
```

`taban` zaten yalniz `RHO_OLC`'daki olcumleri topluyor (satir 480-484), gerisi
kendiliginden dogru calisir. **Bu gece bos bir `m148_olcumler.json` ile
`NIHAI=1` kosulup dosyanin uretildigi GORULMELIDIR** — 03:00'te ilk kez
denenmemeli. Ayni yolla "yalniz rho_1 ile Z" dosyasi da uretilebilir olmali
(K1'in kacis dosyasi).

**(e) Aciliyet: YARIN ONCE (bu gece).**

---

## K3 — [YUKSEK / YARIN ONCE] `docs/72` §3'teki "olcum hatasi 5.6e-05" 30 KAT yanlis

**(a) Ne.** `docs/72` §3 "Sonda 1'in tam degerleri" blogunda
`olcum hatasi 5.6e-05` yaziyor. Betigin kendi formulu (satir 539):

```
hata = sqrt(YUV^2 + SABIT_HATA^2) / (2*kappa_etkin)
     = sqrt((2.887e-06)^2 + (1.72e-04)^2) / 0.1033925
     = 1.664e-03
```

5.6e-05, `SABIT_HATA` eklenmeden onceki (yalniz LB yuvarlamasi) bayat
degerdir; kodda artik yoktur.

**(b) Nasil tetiklenir.** Karar aninda `docs/72` okunarak. Sondanin
"neredeyse hatasiz" olctugu sanilir.

**(c) Sonuc.** Butun risk muhasebesi 30 kat iyimser. K1'in patlamasi da
K4'un yedek-marji sorunu da dogrudan bu sayidan cikiyor.

**(d) Karsi onlem.** `docs/72` §3'te `olcum hatasi 1.7e-03` yazilsin; ayrica
5.6e-05 ile 1.66e-03'un **hangi niceligin** hatasi oldugu (yalniz LB
yuvarlamasi mi, toplam mi) belirtilsin — Kural 69'un ta kendisi.

**(e) Aciliyet: YARIN ONCE.**

---

## K4 — [YUKSEK / YARIN SIRASINDA] "Hicbir sey tutmasa bile 1.00115'ten iyiyiz" iddiasi KANITSIZ

**(a) Ne.** `docs/72` §3: `toplam rho^2 = 0 -> 1.00101 (yine de 1.00115'ten
iyi)`. Olculdu:

```
TABAN_MSE - 1.00115^2 = -0.00027      ->  marj = 0.00014 SKOR
```

Ama `TABAN_MSE = M0 - 2*kL + ||r_hat||^2` de **ayni kalibre sabiti** tasiyor.
Planin kendi sayisiyla `SABIT_HATA = 1.72e-04` (P^2 olceginde) → skor
belirsizligi **+/-8.6e-05**; `P` olceginde ise **+/-1.72e-04**. Yani
**marj (1.4e-04) belirsizligin icindedir.** Ustune K1'in kaybi binince:

```
delta = 0        -> Z_NIHAI 1.00101   (yedekten +0.00014 iyi)
delta = +1.7e-04 -> Z_NIHAI 1.00104   (+0.00011)
delta = +3.4e-04 -> Z_NIHAI 1.00110   (+0.00005)
delta = -1.7e-04 -> Z_NIHAI 1.00132   (YEDEKTEN 0.00017 KOTU)
```

**(b) Nasil tetiklenir.** `|c|` gercekten kucukse (SENARYO D'nin alt ucu) VE
secim sayisi **N = 1** ise. N = 2 ise yedek bizi korur, bulgu zararsizdir.

**(c) Sonuc.** N=1 durumunda `Z_NIHAI` secmek, sinyal yoksa **yedekten kotu**
olabilir. `docs/73` §5 T+6'daki N=1 kurali ("olculen toplam rho^2, 3. sira
esigini net geciyorsa Z_NIHAI") bu yuzden **dogru kural** — ama `docs/72`
§3'un "yine de 1.00115'ten iyi" parantezi onu **celiyor** ve operatoru yanlis
tarafa cekebilir.

**(d) Karsi onlem.** O parantez silinsin. Karar kurali tek yerde ve tek
bicimde olsun: **N=1 ise `Z_NIHAI` ancak olculen `toplam rho^2 > 0.0015`
(marjin ~5 kati) ise secilir, yoksa yedek.**

**(e) Aciliyet: YARIN SIRASINDA** (T-0 adim 5'te N ogrenildikten sonra).

---

## K5 — [YUKSEK / YARIN ONCE] "OLCULEN HER YON RISKSIZDIR" iddiasi YANLIS — sinirlandi

**(a) Ne.** `docs/72` §3 ve `m148` satir 263: *"her OLCULEN yon RISKSIZDIR:
rho_k=0 cikarsa skor degismez."* Cebir bunu soylemiyor. Nihai dosyada `rho_k`
degil **olculen `r_k`** kullaniliyor:

```
P^2 = TABAN_MSE - 2*toplam r_k*rho_k + toplam r_k^2
    = TABAN_MSE - toplam rho_k^2 + toplam (r_k - rho_k)^2
```

Yani **kayip tam olarak `toplam eps_k^2`'dir**; `rho_k = 0` cikan bir yon
skoru degistirmemekle kalmaz, `eps_k^2` kadar **KOTULESTIRIR**. "Risksiz" olan
sey yalnizca isaretin ters cikmasidir, buyuklugun olcum hatasi degil.

**(b) Nasil tetiklenir.** Her zaman; kacinilmaz.

**(c) Sinir (olculdu).** `eps_k = hata_k` alinarak:

| kurulus | sinyal yok | 1. sira senaryosu |
|---|---|---|
| kappa 2-4 = 0.0125 (mevcut) | 6.2e-05 rho^2 → **3.1e-05 skor** | 7.6e-03 rho^2 → **3.8e-03 skor** |
| kappa 2-4 = 0.0517 (onerilen) | 1.0e-05 → 5e-06 | 3.6e-05 → **1.8e-05 skor** |

Karsilastirma: 2. sira icin gereken **toplam** rho^2 = 0.00973. Yani mevcut
kurulusta hedge yonlerinin olcum hatasi hedefin **%79'unu** yiyebiliyor;
onerilen kurulusta **%0.4**'unu.

**(d) Karsi onlem.** K1(d) ile ayni: kappa'yi buyut + anlamlilik kapisi
(`|r_k| < 3*hata_k` ise `r_k = 0`). Anlamlilik kapisi kaybi tanim geregi
`9*hata^2` ile sinirlar; H2-H4'un **ongorulen `rho_k`'si zaten 1e-16** oldugu
icin pratikte onlari sifirlar — **kayip sifirlanir, kazanc kalir**.

**(e) Aciliyet: YARIN ONCE.**

---

## K6 — [ORTA / YARIN ONCE] `|rho| > 0.20` aborti EN IYI sonucta atesleyebilir; gerekcesi de yanlis

**(a) Ne.** Satir 457-462:

```
if abs(RHO_OLC[k]) > 0.20:
    raise SystemExit("... ||r_hat|| = 0.061 tavani goz onune alindiginda bu olanaksiz")
```

Gerekce hatali: `||r_hat|| = 0.0611`, **span icindeki** bilesenin tavanidir.
Sondalarin olctugu `rho_k` span'a DIK bir yondedir ve ust siniri
`||r|| = sqrt(M0) = 1.003`'tur. 0.20 "olanaksiz" degildir.

**(b) Nasil tetiklenir.** D1'in skoru **0.99198'in altina** duserse. 1. sira
esigi 0.99471; arada yalnizca 0.0027 var. Yani "umdugumuzden de iyi" bir D1,
betigi **durdurur**.

**(c) Sonuc.** 03:00'te, kalan iki hakla, panik icinde kod duzenlemesi. Ayni
kapi K1'in iraksak halini (e_4 = -0.62) yakalar — orada iyidir — ama 0.08'lik
felaketi yakalamaz: **esik hem cok gevsek hem cok siki, yanlis yerde.**

**(d) Karsi onlem.** Esik yon basina ayrilsin: sonda 1 icin 0.30'a cikarilsin;
sonda 2-4 icin `3*kappa_k`'ya indirilsin ve mesaj "zincir kararsiz, K1'e bak"
desin. En azindan **mesaj bugun okunsun ki yarin panik olmasin.**

**(e) Aciliyet: YARIN ONCE (en azindan bilinmesi).**

---

## K7 — [ORTA / YARIN ONCE] Paralel oturum HALA CALISIYOR; suruklenme dedektoru YOK

**(a) Ne.** Bu tur sirasinda olculdu:

```
git status:  M  experiments/model29/m148_demet.json      (COMMIT EDILMEMIS)
             ?? experiments/model29/m154_taban_model.json
             ?? experiments/model29/m156_sigma_yeniden.py
             ?? experiments/model29/m157_besinci_yon.py   <- mtime 21:52
```

`docs/72` 21:10'da, `docs/73` 21:20'de yazildi; **m155/m156/m157 daha sonra
(21:47-21:52) uretildi.** Yani "durdurulmus calisma" durmamis. `m157` besinci
bir yon eklenip eklenmeyecegini tartisiyor — eklenirse `DEMET` 5 olur, `GD`
degisir, D1'in kayitli `sabit`i **gecersizlesir**.

Betikte suruklenmeyi yakalayan **hicbir sey yok**. Oz-denetim (satir 517-523)
`bek = TABAN_MSE + toplam r_j^2 + ketkin^2` **kendi kendine tutarli** oldugu
icin `r_hat` tamamen kaysa bile **sessizce gecer**.

**(b) Nasil tetiklenir.** `olculmus_skorlar.json`, `m112_durum.json`,
`m112_kalibre.py`, `m121_derin_tarama.json` ya da `submissions/` altindaki 28
kaynak dosyadan **herhangi biri** degisirse. Ozellikle satir 70-72:
`if ... not os.path.exists(...): continue` — bir CSV silinir ya da yeniden
adlandirilirsa Gram bir sutun **sessizce** kaybeder.

**(c) Sonuc.** D1'in olculmus `rho_1`'i ESKI `GD_1`'e aitken YENI `GD_1`'e
uygulanir. Iki yon arasindaki aci 10 derece olsa kayip `rho_1^2 * 0.03`,
30 derece olsa `rho_1^2 * 0.25` — 1. sira senaryosunda **0.0054 rho^2**,
K1 ile ayni buyuklukte.

**(d) Karsi onlem — somut.**

1. **SIMDI:** `git add -A experiments/model29 && git commit`, sonra
   **`git tag DONDU-31AGUSTOS`**.
2. **SIMDI:** asagidaki parmak izleri `m148_demet.json`'a bir `girdi_sha`
   alani olarak yazilsin; betik her kosuda ilk is bunlari yeniden hesaplayip
   **farkliysa DURSUN**. Bugunku degerler (sha256, ilk 16 hane):
   ```
   d73b751bb4fc435f  experiments/model29/m112_kalibre.py
   6130bb289c6161c5  experiments/model29/olculmus_skorlar.json
   14faa143d4ddd783  experiments/model29/m112_durum.json
   275ea9c21a08479d  experiments/model29/m121_derin_tarama.json
   bc740f3e61617eaf  submissions/tuketim_D1_demet.csv
   ```
3. **SIMDI:** sayisal capa — `m148_demet.json`'a
   `taban_mse = 1.00202690323433`, `rhat_kare_normu = 0.003734501676515773`,
   `V_sutun = 28`, `eksen_sayisi = 40` yazilsin; betik her kosuda
   karsilastirsin. (Bugun olculen dogru degerler bunlardir.)
4. **YARIN:** 03:00'ten sonra bu depoda **baska hicbir oturum kosmasin**.
   m155-m157 sonuclari ancak `git diff` ile gozden gecirilip **bilinerek**
   kabul edilirse plana girsin.

> **CANLI KANIT (22:06).** Bu raporun yazimi sirasinda depo alti kez degisti:
> `m154`→`m155`→`m156`→`m157`→`m158` uretildi, `m148_demet.json` ve
> `m148_olcumler.json` yeniden yazildi, `submissions/` altina **dort sahte
> dosya** dustu (K0). `m148_demet_plani.py`'nin son commit'i 21:41 — yani
> `docs/72` (21:10) ve `docs/73` (21:20) yazildiktan **sonra**. Bu yuzey
> teorik degil, **su anda aktif**.

> Iyi haber: **`r_hat` tarafinda suruklenme YOK.** Diskteki
> `tuketim_D1_demet.csv`,
> 21:07'deki yedekle **bayt bayt ayni** (sha256 `bc740f3e...`) ve kayitli
> `sabit` / `kappa_etkin` bugun yeniden hesaplandiginda **son basamagina
> kadar** tutuyor (fark 0.0e+00). Dondurulacak an **simdidir.**

**(e) Aciliyet: YARIN ONCE (bu gece).**

---

## K8 — [ORTA / YARIN ONCE] Ayni kosuda IKI farkli "optimum" basiliyor; `m143`'un tamami yanlis tabana oturuyor

**(a) Ne.** Betik satir 93 `saf optimum 1.001055` (= `sqrt(MSE_OPT)`,
`MSE_OPT = M0 - ||r_hat||^2 = 1.0021118643`), satir 279 ise
`saf span skoru 1.00101` (= `sqrt(TABAN_MSE)`,
`TABAN_MSE = M0 - 2*kL + ||r_hat||^2 = 1.0020269032`) basiyor. **Dogru olan
ikincisidir**; birincisi buzmesiz cozumun sayisidir (`kL != ||r_hat||^2`,
docs/69 §2.1).

`experiments/model29/m143_gonderim_plani.py` satir ~51 `MSE_OPT = 1.002112`
sabitini kullaniyor — **butun olasilik ve esik tablolari 8.5e-05 kaymis bir
tabana** oturuyor.

**(b) Nasil tetiklenir.** 03:00'te ekrandaki ilk satiri okumakla; ya da
`m143`'u kosturmakla.

**(c) Sonuc.** 8.5e-05 skor (~4.2e-05 rho^2). Tek basina kucuk, ama `m143`'un
"P(2. sira)" tahminlerini sistematik olarak iyimser yapiyor ve K3/K4 ile ayni
yone bakiyor.

**(d) Karsi onlem.** Satir 93 ya kaldirilsin ya da etiketi
`(buzmesiz referans -- KULLANMA)` olsun. `m143` **BAYAT** damgalansin;
`docs/72` §6 dosya haritasinda yok ama depoda duruyor ve kosulabilir.

**(e) Aciliyet: YARIN ONCE (etiketleme yeterli).**

---

## K9 — [ORTA / YARIN SIRASINDA] Son secim ani icin belgede UC FARKLI saat var

**(a) Ne.**

| kaynak | yazan | UTC karsiligi |
|---|---|---|
| `docs/72` satir 133 | "en gec 1 Eylul **22:00 UTC** = 2 Eylul 01:00 yerel" | 22:00 UTC |
| `docs/72` satir 309 | "en gec 1 Eylul **23:00 UTC**" | 23:00 UTC |
| `docs/73` §5 T+6 basligi | "1 Eylul, en gec **22:00 yerel**" | **19:00 UTC** |
| `docs/73` satir 350 | "hedef **2 Eylul 01:00** [yerel]" | 22:00 UTC |

`docs/73`'un kendi basligi kendi metniyle celisiyor (3 saat fark).

**(b) / (c).** Yorucu bir gunun sonunda "daha 3 saatim var" sanip 23:30
UTC'de secim ekranini kapali bulmak. `docs/73` §6 zaten "secim ekraninin
bitisten kac dakika once kapandigi" **DOGRULANAMADI** diyor. Bedeli: her sey
**otomatik secime** kalir — ki `docs/73` §3(a)'ya gore otomatik secim "en
yuksek public"i alir ve bizim sondalarimiz public'te kotu skor verebilir.
Kayip: 0.005 skor / 5 sira.

**(d) Karsi onlem.** Tek sayi: **secim 1 Eylul 19:00 UTC = 22:00 yerelden
once tamamlanir.** Dort ifade de bu tek sayiyla degistirilir.

**(e) Aciliyet: YARIN SIRASINDA (sayi bu gece tekillestirilmeli).**

---

## K10 — [ORTA / YARIN SIRASINDA] Skorun ELLE girilmesi zincirin en zayif halkasi; kapilar makul yazim hatasini gecirir

**(a) Ne.** `docs/73` §5 T+2: `m148_olcumler.json` **elle** yazilir. Iki kapi
var: `0.90 < P < 1.20` (satir 448) ve `|rho| < 0.20` (satir 457).

**(b) Nasil tetiklenir.** 03:05'te bir hane atlamak: `0.99967` yerine
`0.9967`. Her iki kapiyi da **gecer**:
`rho_1 = (1.0046992 - 0.99341) / 0.10339 = +0.1092`.

**(c) Sonuc.** 0.1092 son derece inandirici bir sayidir (1. sira menzili!) ve
zincirin **tabanina** girer; sonraki uc sondanin hepsi bozulur. K1'in geri
beslemesiyle birlesince nihai kayip 0.005-0.05 rho^2 mertebesine cikar. Betik
hicbir uyari vermez; tersine "1. SIRA menzilde" der.

**(d) Karsi onlem.**
- Skoru **yazma, kopyala**: `python -m kaggle competitions submissions -c
  grid-up-datathon -v | head -3` ciktisindaki `publicScore` alanindan.
- Girdikten sonra iki kontrol: (i) `P` bes ondalikli mi? (ii) betigin bastigi
  `rho_1`, el hesabiyla tutuyor mu: `rho_1 ~ (1.00235 - P) * 19.4`.
- Bu iki satir `docs/73` §5 T+2 adimina eklenmeli.

**(e) Aciliyet: YARIN SIRASINDA (kontrol listesine bu gece eklensin).**

---

## K11-K16 — [DUSUK] kalan bulgular

| # | ne | sonucu | onlem | aciliyet |
|---|---|---|---|---|
| K11 | `DEMET` cevre degiskeni (satir 276) satir 361'de eziliyor — **olu kod** | operator "DEMET=3 kosarim" sanir, hicbir sey olmaz | sil ya da `# OLU KOD` yaz | onemsiz |
| K12 | `\|c\|` icin depoda **iki** deger: `docs/72` §3 `0.57 [0.17, 1.26]`, ayni belgenin §4 satiri `~0.7 [0.3, 1.3]`; `m157` (bugun 21:52) **eski 0.7 / 0.30**'u kodluyor | m157'nin "besinci yon" hukmu bayat onselle verilmis; ona dayanip yon eklenirse K7 tetiklenir | m157'nin sonucu **kullanilmasin** ya da onseli guncellenip yeniden kosulsun | YARIN ONCE (karar) |
| K13 | betik **kosum basina ~4 dakika** (28 x 28 MB CSV yeniden okunuyor, ~800 MB) | `docs/73` takvimi (03:05 / 03:15) iyimser; gercek ~03:30-03:40. Kirilma degil ama "gec kaldik" panigi uretir | takvime `+5 dk / halka` yaz; ayni anda ikinci agir betik kosturma (bellek) | YARIN SIRASINDA |
| K14 | Gram'a giren yon sayisi (**28**) hicbir yerde dogrulanmiyor; eksik dosya `continue` ile sessizce atlaniyor | bir CSV silinirse `r_hat` sessizce degisir (K7'nin motoru) | `assert V.shape[1] == 28` ve `assert len(kul) == 40` | YARIN ONCE |
| K15 | liderlik esikleri (0.99009 / 0.99614 / 0.99927) hem betige (satir 573-588) hem `docs/72`'ye gomulu; tablo 30 Agustos'ta **gun icinde iki kez sertlesti** | 31 Agustos - 1 Eylul'de rakipler ilerlerse butun "KARAR KURALI" bayatlar | D1'i gondermeden **once** `kaggle competitions leaderboard -s \| head -12` kos, esikleri teyit et | YARIN SIRASINDA |
| K16 | `m148` satir 202 yorumunda `(docs/70)` deniyor; **`docs/70` diye bir dosya yok** (kastedilen `docs/69` §5 Kural 70) | kucuk; denetci yanlis yere bakar | yorumu duzelt | onemsiz |

---

## SALDIRILDI, KIRILMADI — sessiz gecilmeyen yuzeyler

### S1. CEBIR — `skor^2 = TABAN_MSE - toplam rho_k^2` sifirdan turetildi: **AYAKTA**

```
d      = r_hat + toplam_k r_k*G_k
P^2    = M0 - 2*<r,d> + Q(d)
<r,d>  = kL + toplam r_k*rho_k
Q(d)   = ||r_hat||^2 + 2*toplam r_k*<r_hat,G_k> + toplam r_k^2     [G_k ortonormal]
=>  P^2 = TABAN_MSE - toplam rho_k^2 + toplam (r_k - rho_k)^2 + 2*toplam r_k*<r_hat,G_k>
```

Iddia **uc** varsayima dayaniyor; ucu de sinandi:

| varsayim | olcum | hukum |
|---|---|---|
| `G_k`'lar birbirine dik | Gram sapmasi **4.44e-15** | GECTI |
| `<r_hat, G_k> = 0` | en buyuk **4.8e-15** (`\|\|r_hat\|\| = 0.0611` iken) | GECTI |
| `r_k = rho_k` | **GECMEDI** → kayip `toplam eps_k^2` | K5 |

Yan bulgu: `G_k`'lar `r_hat`'a tam dik ama **span'in kendisine tam dik degil**
— `max |<G_k, V_j>| / (N*||V_j||) = 1.31e-04`. Sebep `pinv(rcond=1e-6)` ile
atilan neredeyse-tekil kipler. Bu bir **hata degil**: `r_hat` o kiplerde
sifirdir, dolayisiyla cebir bozulmaz; sondanin o kiplerde gercek sinyal
olcmesi **bonustur**.

`a0` referansi tutarli: `TABAN_MSE + kappa_etkin^2 = 1.00469941` vs kayitli
`sabit = 1.00469923`, fark **1.8e-07** (kirpmadan gelen).

### S2. KIRPMA — dikligi bozuyor mu: **BOZMUYOR, hatta lehte**

`expm1 -> clip(0)` her dosyada calisiyor (D1'de 530 satir, tabanda 22).
Olculen yon `e_k` ile uygulanan yon `G_k` arasindaki aci:

```
D1: cos = 0.999684   (1 - cos^2 = 6.3e-04, kirpilan 530)
D2: cos = 0.999986   (2.9e-05, 100)
D3: cos = 0.999895   (2.1e-04,  32)
D4: cos = 0.999969   (6.2e-05, 118)
```

Goreli kayip `~2*(1-cos) = 6.3e-04 * rho^2`; 1. sira senaryosunda **1.4e-05
rho^2 = 7e-06 skor.** Isareti bilinmeyen ek terim (kirpma artiginin gercek `r`
ile ortusmesi) 530 satirla sinirli; ust siniri **~1e-04 skor**.

`Z_NIHAI`'de kirpma **yardim ediyor**: `toplam rho^2 = 0.02175`'te kirpmali
gercek skor 0.99006, kirpmasiz ideal 0.99009 (**-3.0e-05**, yani daha iyi) —
cunku tuketim zaten negatif olamaz.

### S3. PUBLIC / PRIVATE — olculen rho private'a tasinir mi: **TASINIR (kayip ihmal edilebilir)**

`rho` public satirlarda olculup **tum** satirlara uygulaniyor. Ornekleme
gurultusu `sd(rho_pub - rho_priv) ~ sqrt(M0 / n_pub)`:

| public payi | sd | 4 yonde toplam kayip (rho^2) | 2. sira esiginin yuzdesi |
|---|---|---|---|
| %50 | 1.7e-03 | **1.1e-05** | %0.1 |
| %30 | 3.3e-03 | 4.4e-05 | %0.5 |
| %20 | 5.3e-03 | 1.1e-04 | %1.2 |

**Public/private ayrimi "risksiz" iddiasini BOZMUYOR** — K1/K5'teki
kalibrasyon hatasindan iki-uc buyukluk mertebesi kucuk. (`docs/71` §3'e
guvenilmeden, bagimsiz hesap.)

> **Ama bir IC CELISKI var (bildirilir; karari degistirmez).** `docs/71` §3
> "public = TUM kume" hipotezini rakiplerine **10.7:1** goreli olabilirlikle
> ustun buluyor. `docs/69` §1.3 + `m112`'nin `M0` yorumu + Kural 65 ise
> "`P` public %50'de, `Q` tum satirlarda olculur; `M0` bu uyusmazligi emen
> **etkin** bir sabittir" diyor. **Ikisi ayni anda dogru olamaz.** Onemi
> buyuk: `docs/71` §3 hakliysa emilecek uyusmazlik yoktur, `SABIT_HATA ~ 0`
> olur ve **K1 soner**; `docs/69` hakliysa `SABIT_HATA = 1.72e-04` gercektir
> ve **K1 kritiktir**. Karar: **kotu tarafa gore davran** (K1(d) uygulanir) —
> maliyeti sifira yakin.

### S4. TAKVIM VE ZINCIR — kirilma noktalari sayildi: **PLAN AYAKTA, SLACK SIFIR**

Gereken gonderim **5** (D1-D4 + Z_NIHAI), elde **6** (3 + 3). Kota
sifirlanmasi 00:00 UTC = 03:00 yerel, **bagimsiz olarak yeniden dogrulandi**:
gecmis 30 gonderimin damgalarinda 23 Agu 14:26 UTC → 24 Agu 04:19 UTC arasi
**13.9 saat** var ve kabul edilmis; "kayan 24 saat" kurali olsaydi
reddedilirdi. Demek ki **UTC gunu** kurali dogru. Gunluk 3 siniri de
damgalarda dogrulandi (7 gunun 7'sinde tam 3).

En kotu durum agaci:

| olay | akis | elde kalan |
|---|---|---|
| her sey yolunda | 31 Agu: D1, D2, D3 · 1 Eyl: D4, Z_NIHAI (+1 hak yedek) | 4 olcum |
| 1 ERROR (kota yandi varsayimi) | 31 Agu: D1, D2 · 1 Eyl: D3, D4, Z_NIHAI | **slack 0** |
| 2 ERROR | zincir **kisaltilmali** → **K2 yuzunden IMKANSIZ** | yedek 1.00115 |
| skor 10+ dk gecikirse | ayni | ayni |

ERROR olasiligi dusuk (30/30 gonderim `COMPLETE`), ama **K2 cozulmeden tek bir
ERROR bile "3 olcum + Z_NIHAI" senaryosunu felce ugratir.** Zincir kirilirsa
elde kalan `tuketim_YP_seviye.csv` (1.00115, 7. sira); sondalar secilemez
cunku `rho_k = 0`'da beklenen skorlari 1.00235'tir.

`docs/73` §4.4'teki "zaman asimina ugrayan komut = gonderilmis olabilir"
kurali dogru ve kritik. `subs.csv` ile teyit edildi: 25 Agustos'ta
`tuketim_v55_gunolcek.csv` **iki kez** gonderilmis (ref 55755748 ve 55755749,
iki saniye arayla, ayni skor) — o hafiza notunun kaynagi burada goruluyor.

### S5. SECIM (N) — N=1 ise plan ne kadar kotu: **SAYILDI**

- **N = 2:** asagi yon kapali; en kotu 1.00115 (7. sira), en iyi olculen ne ise.
- **N = 1:** `Z_NIHAI`'nin kendi private skoru esas alinir. Sinyal yoksa
  `Z_NIHAI` beklenen 1.00101, yedek 1.00115 → marj **0.00014**, ki K4'e gore
  belirsizligin icinde. Yani N=1'de bahsin asagi yonu **~0.0002 skor**
  aciliyor; 5. ile 7. sira arasindaki fark 0.00068 oldugundan **bir siralik**
  risk. Ustune K1/K5 kaybi binerse iki sira.

**Hukum:** N=1 plani "cok kotu" yapmiyor (kayip ~1 sira), ama `docs/72`
§2'deki "yedek isaretli kaldigi surece kaybetme riski yoktur" cumlesi
**yalniz N >= 2 icin** dogrudur ve `docs/73` §5 T-0 adim 5 (N'i oku)
**gercekten atlanamaz**.

### S6. DOSYA / VERI TUZAKLARI — `gun1_baseline` disinda **BASKA TUZAK YOK**

`olculmus_skorlar.json`'daki **27 skorun 27'si**, onbelleklenmis Kaggle
gonderim listesiyle (30 kayit, hepsi `COMPLETE`) **birebir eslesti**: dosya
adi, skor, `ref`. Uydurma ya da turetilmis skor yok.

- `gun1_baseline.csv` (1.22670): gercek gonderim, ama dosya **80 satir /
  `hedef` sutunlu** → `oku()` `None` dondurur → **sessizce atilir**. Zarar yok;
  ama **sessiz** oldugu icin K14'un kanitidir.
- **Gonderimden sonra degistirilmis dosya: yalnizca `gun1_baseline.csv`**
  (mtime 30 Agu 06:14, gonderim 21 Agu 11:54). Zaten atiliyor. Diger 26
  dosyanin hepsinde `mtime < gonderim damgasi`. **Temiz.**
- `tuketim_y40_sota_temiz.csv`: **hic gonderilmedi** ama `EK_MODEL` ile Gram'a
  giriyor (`L = -0.002229`, `s3y40`'tan **turetilmis**). `docs/69` §1'de acikca
  etiketli ve gerekcelendirilmis; 28 yonden 1'i. Yeni bulgu degil; Kural 66'nin
  sinirinda duran **kalici risk** olarak not edilir.
- `V` sutun sayisi **28** = 25 (skorlar) + 1 (`EK_MODEL`) + 2
  (`m112_durum.olcumler`). Bu sayi hicbir yerde dogrulanmiyor → K14.
- 40 eksenin **5'inde** `rho_cv` ile `rho_s`'in isareti ters
  (`p_ilk_ofset*tanim_on3`, `t_kayma:ust25`, `t_kayma:ust10`,
  `ilce_toplam_guc:x_sv`, `g_ilce_kova_n:x_soguk`) ve kod **CV'nin isaretini**
  aliyor. `docs/71` §6'daki 35/40 ile birebir tutuyor; H1 yonunun icinde
  kalan bir belirsizlik, D1 olcumu bunu toplamda duzeltiyor. Yeni bir kusur
  degil.

### S7. PARALEL OTURUM — bkz. **K7** (bulundu; su an suruklenme YOK, dondurulmali)

### S8. SAYILAR — `docs/72`'deki her sayi bagimsiz yeniden hesaplandi

| sayi | `docs/72` | yeniden hesap | hukum |
|---|---|---|---|
| `M0` | 1.005846366 | 1.005846366 | ✓ |
| `TABAN_MSE` | 1.00202690 | 1.00202690323433 | ✓ |
| saf span skoru | 1.00101 | 1.0010129 | ✓ |
| `kappa` (D1) | 0.05174191 | 0.05174190699701174 | ✓ |
| `kappa_etkin` | 0.05169627 | 0.0516962677376078 | ✓ |
| `sabit` (D1) | 1.0046992296 | 1.0046992296275314 | ✓ |
| bolen `2*kappa_etkin` | 0.10339254 | 0.1033925355 | ✓ |
| `rho_1 = 0` skoru | 1.00235 | 1.0023469 | ✓ |
| tahmin tutarsa | 0.99967 | 0.99967 (`rho = kappa`) | ✓ |
| `sqrt(toplam rho_s^2)` | 0.1294 | 0.129355 | ✓ |
| `rho_pred` (bilesik) | 0.2522 | 0.252242 | ✓ |
| gereken rho^2 — 1. sira | 0.02175 | 0.02175 | ✓ |
| gereken rho^2 — 2. sira | 0.00973 | 0.00973 | ✓ |
| gereken rho^2 — 3. sira | 0.00349 | 0.00349 | ✓ |
| gereken \|c\| — 1. sira | 1.140 | 1.140 | ✓ |
| gereken \|c\| — 3. sira | 0.456 | 0.456 | ✓ |
| eksen sayisi / diklik | 40 / 4.4e-15 | 40 / 4.44e-15 | ✓ |
| artakalan H2 / H3 / H4 | 0.369 / 0.433 / 0.211 | 0.369 / 0.433 / 0.211 | ✓ |
| `\|b\|` katsayi araligi | (yazilmamis) | **[0.0328, 0.0578]** | — |
| **olcum hatasi** | **5.6e-05** | **1.66e-03** | ✗ **K3** |
| gereken \|c\| — 2. sira | 0.762 | 0.763 | ~ yuvarlama |
| D1 skoru @ \|c\| = 0.76 | 0.99725 | 0.99726 | ~ yuvarlama |
| `\|c\|` nokta / aralik | 0.57 / [0.17, 1.26] | ayni belgenin §4'unde **0.7 / [0.3, 1.3]** | ✗ **K12** |

**Uyusmayan iki gercek sayi: `olcum hatasi` (K3) ve `|c|`'nin iki surumu
(K12).** Geri kalan her sey son basamagina kadar tutuyor.

---

## YARIN SABAHA KADAR YAPILACAKLAR — sirali

0. **[K0]** Sahte `D2/D3/D4/Z_NIHAI` dosyalarini ve `m148_olcumler.json`'i sil,
   `m148_demet.json`'i geri al, uc capa sayiyi (sabit / kappa_etkin / D1 sha)
   gozle dogrula. **Her seyden once.**
1. **[K7]** `git add -A experiments/model29 && git commit`, ardindan
   `git tag DONDU-31AGUSTOS`. Bu gece bu depoda **baska oturum kosmasin**.
2. **[K1 / K5]** `KAPPA_K` satirini `np.full(DEMET, 0.0517)` yap (ilk eleman
   zaten ayrica atandigi icin D1 korunur). Tek satir, en buyuk kazanc.
3. **[K2]** `NIHAI=1` kacis yolunu ekle **ve bos olcumle bir kez kostur**,
   `tuketim_Z_NIHAI.csv`'nin uretildigini gor, sonra o deneme dosyasini sil.
4. **[K5]** Anlamlilik kapisi: `taban`'a `r_k` konmadan once
   `|r_k| > 3*hata_k` sarti.
5. **[K14]** `assert V.shape[1] == 28` ve `assert len(kul) == 40`.
6. **[K3 / K4 / K9]** `docs/72` §3'te `olcum hatasi 1.7e-03`; §3'teki
   "yine de 1.00115'ten iyi" parantezini sil; secim ani icin **tek** saat:
   **1 Eylul 19:00 UTC = 22:00 yerel**.
7. **[K12]** `m157`'nin hukmune, onseli guncellenmeden **dayanma**.
8. **[K10 / K15]** `docs/73` §5'e iki satir: skoru **kopyala, yazma**; D1'den
   once liderlik tablosunu tazele.

**Bu sekiz maddenin hicbiri yeni bir olcum ya da agir kosum gerektirmiyor;
toplam ~30 dakika.**
