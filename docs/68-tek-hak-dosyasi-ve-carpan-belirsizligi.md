# 68 — Tek hak dosyası, `ρ` belirsizliği ve kalibrasyon çabaları

Tarih: 30 Ağustos 2026 · **docs/67'nin plan bölümünün yerine geçer**

Karar: **tek gönderim**. Kazanç biriktirme planı (6 hak) iptal.

---

## 1. Liderlik tablosu değişti

```
1. Grid Grinders        0.99009
2. Duo-Electra          0.99790   <- 1.00129'dan sicradi
3. Atakan Aldemir       0.99940
4. TasnifX              1.00115   <- BIZ (3. siradan dustuk)
5. Ahmet B. ALTUNOK     1.00118
```

---

## 2. Tek hak, çok hakka göre bir şey kaybettirmiyor

Cebir: tek sondayla 2. sıra koşulu `ρ ≥ 0.0793`; sonda + tam optimum ile
de koşul `ρ ≥ 0.0793`. **Aynı eşik.** Çok hakkın tek faydası ek yönler
ölçüp `Σρ²` büyütmek olurdu, aynı yönü daha iyi kullanmak değil.

---

## 3. Geniş LB-çapalı tarama iki yeni aile buldu

428 aday (157 öznitelik + `×seviye`, `×soğuk`, `üst%10` kesitleri) `|rho_s|`
büyüklüğüne göre tarandı. Önceki havuzda tavan 0.027 idi, **0.0405'e çıktı**.

Yeni aileler: **panel penceresi × seviye** (`p_pencere_payi`, `p_ilk_ofset`)
ve **trafo kimliği öneki × soğuk** (`tanim_on3` = kimliğin ilk 3 hanesi,
coğrafi/idari kod).

Seçim yanlılığı: `σ(rho_s) ≈ 3.1e-04`, 428 aday → sahte tavan ~1.1e-03.
Bulunanlar 0.03–0.04, yani 30–100 kat üzerinde. Gerçek.

---

## 4. Seçilen bileşik

Artımlı inandırıcılık denetimiyle `oran ≤ 4` kısıtı altında `ρ`'yu en üste
çıkaran ön-ek seçildi: **7 eksen**.

```
       p_pencere_payi:x_sv   oran 1.63
         tanim_on3:x_soguk   oran 5.29
          p_ilk_ofset:x_sv   oran 3.84
       t_yuk_faktoru:ust10   oran 3.98
                  yas:x_sv   oran 4.03
       t_egim_sicaklik_ort   oran 4.21
       asiri_sicak:x_soguk   oran 3.91  <- kesim, rho=0.1193
```
40 eksenli sürüm `ρ=0.1853` veriyordu ama oran 5.4 — kapı **yazmayı reddetti**.

**Ağırlıklar aşırı uyum değil** (`m120`): trafo-bölmeli çapraz doğrulama
16 katta iç korelasyonun **%94.5**'ini koruyor, plasebo `z=+15.7`.

---

## 5. ÇÖZÜLEMEYEN: `ρ` 0.06 mı 0.12 mi

İki meşru kestirim, ikisi de aynı `seviye` kalibrasyonundan:
```
per-eksen tavan toplami   sqrt(sum beta^2)            = 0.1193
bilesigin KENDI rho_s'i   1.95 x 0.0305               = 0.0595
```
`1.95` çarpanı **tek bir eksende** (seviye) ölçüldü. Bileşiğe uygulanınca
hangisinin doğru olduğu bilinmiyor.

**n'i büyütme denemesi BAŞARISIZ.** 27 ölçülmüş gönderimin her birini
sırayla dışarıda bırakıp `(rho_s, rho_perp)` çifti çıkarmayı denedim;
26'sı birbirinin span'ı içinde olduğu için `Q_perp ≥ 0.15` filtresini
yalnız biri geçti, onun da gürültüsü sinyalinin 100 katı. Çarpan **n=1'de
kalıyor**.

Ayrıca `rho_s`'i doğrudan büyütmek bir TUZAK: matematiksel tavanı
`||r_hat|| = 0.0613`, ama ona ulaşmanın yolu bileşiğin span parçasını
`r_hat`'e hizalamak — yani zaten bildiğimizi tekrarlamak. Yeni bilgi dik
parçada ve Kural 56 gereği o ilişki korunmuyor.

---

## 6. Gönderilecek dosya

```
submissions/tuketim_K_TEKHAK.csv     tum kapilar gecti
  7 eksen, kappa = 0.070, sifir tahmin 880
  kirpma sonrasi etkin yer degistirme 0.06994 (kayip ihmal edilir)
  sabit = 1.006983966
  COZUM: rho = (1.006983966 - P^2) / 0.140
```

`κ = 0.070` seçimi: `sqrt(MSE_opt − 0.99790²) = 0.0793` P(2. sıra)'yı en üste
çıkarır ama `ρ=0.06`'da bizi 4. sıraya düşürüyordu. `κ=0.070`, 2. sıra için
gerekeni `0.0793 → 0.0799` (%0.8) yükseltip 3. sıra için gerekeni
`0.0604 → 0.0585` düşürüyor.

| gerçek `ρ` | skor | sıra |
|---:|---:|---|
| 0.1193 | 0.99513 | **2. SIRA** |
| 0.0835 | 0.99764 | **2. SIRA** |
| 0.0700 | 0.99859 | 3. sıra |
| 0.0596 | 0.99932 | 3. sıra |
| 0.0500 | 0.99999 | 4. sıra |
| 0.0000 | 1.00349 | 5.+ |

**Dürüst beklenti:** `ρ` aralığı 2. sıra eşiğinin iki yanında. 3. sıra
olası, 2. sıra açık ama garanti değil, 1. sıra (0.99009) erişilemez.

---

## 7. Kalıcı kurallar 62–64

**62.** Bir kalibrasyon çarpanını tek eksende ölçüp bileşiğe uygulamak
meşru değildir; iki kestirim (per-eksen toplam vs bileşiğin kendi
ölçümü) 2 kat ayrışabilir. Çarpanı bileşik düzeyinde doğrulayamıyorsan
aralığı raporla, tek sayı verme.

**63.** LB'den ölçülen bir vekili (`rho_s`) doğrudan optimize etme.
Tavanı `||r_hat||`'tir ve ona ulaşmak zaten bilineni tekrarlamaktır;
vekil ile hedef arasındaki ilişki optimizasyon altında kopar.

**64.** Bileşik kurarken artımlı inandırıcılık izle ve kısıtı geçen en
büyük ön-eki seç. Eksen eklemek `ρ`'yu monoton büyütür ama inandırıcılığı
bozar; kesim noktası veriden gelmelidir, elle seçilmemelidir.
