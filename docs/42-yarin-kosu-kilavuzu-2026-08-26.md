# Koşu kılavuzu — 26 Ağustos ve sonrası

**Bu dosya mekanik olarak izlenir.** Gerekçeler
[41-olculemeyen-kolonlar](41-olculemeyen-kolonlar-2026-08-25.md)'da.

---

## 0. Durum (25 Ağustos 07:20)

```
1. Grid Grinders      1.00635   <- 25 Agustos 00:18 UTC, onceki ilk-4'te YOKTU
2. TasnifX            1.01591   <- BIZ
3. Bilalcan Ustabas   1.01793
4. Churros y Cay      1.02138
5. Data4Win           1.02298
Yarisma 1 Eylul'de bitiyor. Gunde 3 hak, sifirlanma 00:00 UTC = yerel 03:00.
```

> **DURUM DEGISTI.** Gonderimimizden 17 dakika sonra Grid Grinders 1,00635 ile
> birinci oldu. Fark **0,0096**. Bugunku kazancimiz (-0,00159) Bilalcan'a karsi
> yeterliydi ama bu sicramanin yanina yaklasmiyor.

Bugün doğrulanan: **gün ekseni genlik düzeltmesi gerçek** (v50 1,01686 →
v55 1,01591, −0,00095).

### Stratejik sonuç — rötuşlar bu farkı KAPATMAZ

Elde hazır duran adayların toplam beklentisi ≈ **−0,0015** (soğuk gün ekseni
−0,0006 · λ −0,0004 · 50 tohum −0,00016 · sıcak c optimumu −0,00026). Bu bizi
~1,0144'e getirir; Grid Grinders 1,00635'te. **Rötuşlarla yetişilmiyor.**

Ölçümler kalan bütün *ölçülebilir* eksenlerin kapalı olduğunu söylüyor (§5).
Öyleyse iki iş birbirinden ayrılır:

- **Savunma (hazır, mekanik):** aşağıdaki zinciri koş, 2.'liği sağlamlaştır ve
  farkı biraz kapat. Risk yok, kod hazır.
- **Atak (yeni bilgi gerekir):** 0,0096'yı kapatmak için rötuş değil, **yeni
  bilgi kaynağı** lazım. Ölçülen duvar: toplam MSE'nin %46'sı soğuk sıfırlarda
  ve statik özniteliklerle öğrenilemiyor (AUC 0,565); soğuk trafo seviyesinde
  ofset std 1,9357 ve en iyi kestirici %1,6'sını açıklıyor. Bu duvarı ancak
  trafo düzeyinde YENI bir veri kaynağı ya da yarışmanın gözden kaçmış bir
  yapısal özelliği yıkar. Grid Grinders'ın 1,006'sı böyle bir şeyi bulmuş
  olduklarını gösteriyor.

---

## 1. ÖNCE: taban tazelenmiş mi?

Arkaplanda tohum 130-149 üretiliyor (`scripts/tuketim_model.py --tohum 5
--tohum-baslangic 13X --dogrulama-atla --cikti tuketim_v51_ekN.csv`).
**Biten her parti tabanı iyileştirir ve bütün son işlemler yeniden türetilmelidir.**

```bash
ls submissions/tuketim_v51_ek*.csv          # kac parti bitti?

# Ornek: 4 parti bittiyse 30 + 20 = 50 tohum
uv run python scripts/birlestir_tohum.py --cikis submissions/tuketim_v61_ham50.csv \
    submissions/tuketim_v50_ham30.csv:30 \
    submissions/tuketim_v51_ek1.csv:5 submissions/tuketim_v51_ek2.csv:5 \
    submissions/tuketim_v51_ek3.csv:5 submissions/tuketim_v51_ek4.csv:5

uv run python scripts/son_islem.py --giris submissions/tuketim_v61_ham50.csv \
    --cikis submissions/tuketim_v61_nihai50.csv
```

Beklenen kazanç 30 → 50 tohum: **−0,00016** (σ = 0,15671; `σ²(1/30 − 1/50)`).
Küçük ama kesin ve bedava.

---

## 2. Son işlemleri TAZE tabandan yeniden üret

Sıra önemli: **önce soğuk, sonra sıcak** (soğuk gün koruması `_ham` dosyadan
türer; sıcak ölçekleme son işlem sonrasına uygulanır).

```bash
B=submissions/tuketim_v61_nihai50.csv        # yoksa tuketim_v50_nihai30.csv

# (a) SICAK gun ekseni -- c FORMULDEN, elle verilmez
uv run python scripts/son_islem_gunolcek.py --giris $B \
    --cikis submissions/A_sicak.csv

# (b) + SOGUK gun ekseni, LB-kalibreli
#
# !!! DIKKAT -- 2026-08-25 gecesi (docs/45 tik 2) uc kusur bulundu !!!
# GUNCELLEME 2026-08-26 09:00: kusur 1 ve 2 KODDA DUZELTILDI
# (scripts/son_islem_gunolcek.py). Kusur 3 (yanlis capa nufusu) DURUYOR --
# bu adim hala SOGUK icin yanlis nufusa capaliyor. Yerine
# scripts/son_islem_soguk_gunolcek.py kullanin.
# Ayrica LB artik c*'i TAM cozdu (1,3301, docs/46) -- capa tahminine
# gerek kalmadi.
#
# 1. CALISMIYOR. Aynen kosuldugunda cokuyor:
#      RuntimeError: olcek beklendigi gibi degil: 1.414 yerine 1.458
#    (BUG 1: kirpma yuzunden ISTENEN olcek ULASILMIYOR; betigin kendi kapisi
#     ateşliyor. Ayni kusur v55'te 1,492->1,4760, v66'da 1,335->1,3241.)
#
# 2. --lb-kalibre YANLIS BICIM (BUG 2). Kalibreyi AFFIN uyguluyor
#    (1+k(c-1)) ama c* sigma_gercek ile ORANTILI oldugu icin dogru bicim
#    CARPIMSAL (k*c). LB'nin cozdugu optimuma karsi: carpimsal 1,3325
#    (hata ~0), affin 1,4395 (hata +0,109, dMSE +0,000277).
#    Tasinabilir sabit c* DEGIL, HEDEF GENLIK S* = 0,2204.
#
# 3. YANLIS CAPA NUFUSU -- en buyugu. Betigin capasi pencerenin >=%90'inda
#    VAR OLAN (YERLESIK) trafolardan geliyor: sigma = 0,2710. Ama SOGUK
#    satirlar pencerede YENI DOGMUS trafolar ve onlarin gercek gun ekseni
#    genligi 0,4255 -- 1,570 KAT buyuk. Bu yuzden bu adim c=1,411 seciyor,
#    dogrusu ~2,2-3,0.
#    docs/41'in "v50 ham soguk gun std = 0,1626/0,60 = 0,2710, gercek referans
#    da 0,2710 -> ham model genligi ZATEN DOGRU biliyor" cumlesi DONGUSEL bir
#    dogrulamaydi: 0,2710 bir olcum degil, buzme katsayisina bolunerek elde
#    edilmis bir CIKARIM ve tesadufen YANLIS referansa oturmus.
#
# YERINE: scripts/son_islem_soguk_gunolcek.py --c 2.20
#   (dogru nufustan capali, seviye-notr, ulasilan olcek dogrulanir)
#   Ayni panelde: bu adim -0,00657 test dMSE, yerine gelen -0,01486.
#
# uv run python scripts/son_islem_gunolcek.py --giris submissions/A_sicak.csv \
#     --cikis submissions/B_sicak_soguk.csv --yalniz-soguk --lb-kalibre 0.893

# (c) + etkilesim (lambda)
uv run python scripts/son_islem_lambda.py --giris submissions/B_sicak_soguk.csv \
    --cikis submissions/C_tam.csv

# (d) ayirici nokta: sicak c=2,00, soguk uretim
uv run python scripts/son_islem_gunolcek.py --giris $B \
    --cikis submissions/D_sicak20.csv --c 2.0
```

Her betik kendi kapılarını koşar. **Kapı hata verirse gönderme** — bugün üç
gerçek sızıntı yakaladılar (seviye kayması, tek yönlü merkezleme ×2).

---

## 3. Gönderim sırası ve neyi öğrettiği

| sıra | dosya | ne öğretir | beklenti |
|---|---|---|---|
| 1 | `B_sicak_soguk` | **soğuk gün bileşeni** (sıcak satırlar A ile birebir) | ~−0,0006 |
| 2 | `D_sicak20` | **(A) c\*=1,33 mü, (B) public LB alt küme mi** | (A) 1,0204 / (B) 1,0169 |
| 3 | `C_tam` veya 1-2'nin gösterdiği optimum | | ~−0,0004 |

**Gönderimden ÖNCE liste oku** (kalıcı kural 8):

```bash
uv run kaggle competitions submissions -c grid-up-datathon | head -5
```

Bugün bu yapılmadığı için bir hak mükerrer gönderime gitti.

### 2. hak nasıl okunur

`MSLE(c) = A + B(c−c*)²`, `B = pay_sıcak × σ_gün²`.

- **D ≈ 1,0204 çıkarsa (A):** B doğru, `c* = 1,33`. Sıcak eksen bitmiştir
  (v55 optimumun 0,0003 uzağında). Kalan günler soğuk + λ + tohuma gider.
- **D ≈ 1,0169 çıkarsa (B):** public LB bir **alt küme**. Gerçek `c*` 1,49
  civarı ve **private LB'de kazanç bugün göründüğünden büyük**. O zaman sıcak
  `c`'yi 1,49'da tut, hatta yukarı tara.

---

## 4. Elde duran, henüz sınanmamış

| dosya | ne | not |
|---|---|---|
| `tuketim_v53_ablasyon_ham.csv` | 4 ölçülemez kolon çıkarılmış, 5 tohum | `rejim_birlestir.py` ile soğuk tarafı v50'den al, sonra son işlem; tohum cezası ~+0,0007 |
| `tuketim_v60_lambda.csv` | λ düzeltmesi m=0,13 | kapsama %44,1 |

---

## 5. KAPANMIŞ eksenler — tekrar açma

| eksen | nasıl kapandı |
|---|---|
| soğuk sıfırlar (toplam MSE'nin %46'sı) | sınıflayıcı AUC 0,565; trafo düzeyinde en yüksek 10'un hiçbiri ölü değil |
| soğuk trafo seviyesi | ofset std 1,9357; en iyi kestirici (ilçe) R² 0,016 |
| **kimlik komşuluğu** | mesafe≤1'de R² 0,019 — ilçenin 0,016'sının yanında hiç |
| sıcak trafo seviyesi | bloklar arası korelasyon −0,21 … +0,17, taşınmıyor |
| sıcak kapasite | kazanç 4 trafodan, K=25'te t=−4,03 |
| kalibrasyon / beta | orakül tavanı −0,002, çapraz uydurulan her varyant kötü |
| takvim / tatil | günlük etki std'si 0,0525 log birimi |
| harman (power mean, NNLS) | üretim zaten optimumda |
| λ pencere genişletme | 2025 tam yıl kor +0,029 (dar aynı mevsim +0,400) |
| **ulusal yük endeksiyle gün faktörü kurulumu** | blok İÇİNDE R² 0,926→0,950, ama katsayılar taşınmıyor: çapraz uydurmada yaz25 −0,0152, guz25 −0,0333, kis26 −0,0021 — **3/3 zarar** |

---

### Not: modelin gün faktörünün ŞEKLİ zaten iyi

```
gercek gun faktorunu aciklama gucu (R^2)
blok     MODEL   ulusal_gunluk   ulusal_tepe   MODEL+ULUSAL (ORNEKLEM ICI)
yaz25    0,926      0,666           0,684          0,950
guz25    0,942      0,814           0,798          0,950
kis26    0,120      0,187           0,289          0,451
```

Yaz ve güzde model şekli %93-94 biliyor; **eksik olan yalnızca genlikti** ve
onu `son_islem_gunolcek.py` düzeltiyor. Kışta model kötü (0,120) ama test yaz.

---

## 6. Kalıcı kurallar (ihlal eden bulgu reddedilir)

1. Soğuk tarafta her kazanç **trafo bazında** ayrıştırılır; kırpılmış tablo
   (K = 0,1,5,10,25,50) verilmeden kabul edilmez.
2. Önerilen her kolonun **eğitim/test doluluk deseni** karşılaştırılır.
3. Soğuk tarafta **üç tohum yetmez** (3 tohumda t=+3,91 olan bulgu 6'da çöktü).
4. Hüküm **(blok, tohum) çiftleri** üzerinde eşlenik SH ile verilir.
5. LB'yi problayarak test etiketi çıkarmak **yasak**.
6. **Gün ekseni ölçümü trafo etkisi çıkarılmadan yapılmaz** — yoksa ölçülen
   şey mevsim değil kompozisyondur.
7. **Mevsime bağlı bir eksen tek blokta ölçülmez**; ölçüm test penceresinin
   mevsimsel ikizini içermelidir.
8. **Gönderimden önce gönderim listesi okunur**; zaman aşımına uğrayan bir
   betik için "hiçbir şey olmadı" varsayılmaz.
