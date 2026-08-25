# Koşu kılavuzu — 26 Ağustos ve sonrası

**Bu dosya mekanik olarak izlenir.** Gerekçeler
[41-olculemeyen-kolonlar](41-olculemeyen-kolonlar-2026-08-25.md)'da.

---

## 0. Durum (25 Ağustos 05:15)

```
1. TasnifX            1.01591   <- BIZ
2. Bilalcan Ustabas   1.01793      fark 0,00202
3. Churros y Cay      1.02138
Yarisma 1 Eylul'de bitiyor. Gunde 3 hak, sifirlanma 00:00 UTC = yerel 03:00.
```

Bugün doğrulanan: **gün ekseni genlik düzeltmesi gerçek** (v50 1,01686 →
v55 1,01591, −0,00095).

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
uv run python scripts/son_islem_gunolcek.py --giris submissions/A_sicak.csv \
    --cikis submissions/B_sicak_soguk.csv --yalniz-soguk --lb-kalibre 0.893

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
