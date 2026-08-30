# KARAR AGACI — olcumler gelince ne yapilacak

Dort olcum yurutuluyor. Her birinin sonucu somut bir yapilandirmaya
baglanir. Amac: sonuc gelince TARTISMAK degil UYGULAMAK.

---

## Olcum 1 — `n09_K_karari.json` : eksen sayisi doyuyor mu?

**Soru:** gerceklesen `rho`, eksen sayisi `K` ile buyuyor mu yoksa
`K ~ 25`'te doyuyor mu?

| sonuc | yapilacak |
|---|---|
| **Doyum YOK** (rho K ile buyuyor) | `K_AZAMI` verme. 136 eksen kalir. Ayrica `yama_tum_adaylar.py` uygula (m121'in 300 adayi da havuza girer) ve ikinci nesil carpimlari da ekle. `‖BETA‖` daha da buyur. |
| **Doyum VAR, tepe K\*'ta** | `K_AZAMI=K*` ile kos. Eksen sayisi kirpilir. `yama_tum_adaylar.py` UYGULANMAZ (bilgi degil gurultu ekler). |
| **Belirsiz** (guven araligi ikisini de kapsiyor) | 136'da kal. Kirpma kaybettirmez ama kazandirdigi de kanitlanmadi; statuko en az pismanlik. |

**Ek soru — blok sayisi.** Ayni olcum su uc sayiyi verecek: TAVAN
(optimal agirlik), SABIT AGIRLIK (mevcut), ve B blokla ulasilan.

| B=5'in B=4'e kazanci | yapilacak |
|---|---|
| `> 0.001` rho^2 | `DEMET_HEDEF=5`. 5 sonda + 1 nihai = 6 hak. **Yedek hak KALMAZ.** |
| `<= 0.001` rho^2 | `DEMET_HEDEF=4` kalir. 6. hak dogrulama-ve-onarim icin durur. |

---

## Olcum 2 — `n10_c_carpani.json` : `|c|` gercekten kac?

`|c|` su an 0.57, %90 GA [0.17, 1.26] — **yedi kat genislik**, neredeyse
bilgisiz. LB'nin kendi 29 olcumu uzerinde birak-birini-disarida ile
dogrudan olculuyor.

| sonuc | yapilacak |
|---|---|
| `\|c\|` daha DAR olculdu | `n06_kappa.py`'deki onseli guncelle, kappa yeniden secilir. `n05`/`n13` beklentileri yeniden hesaplanir. |
| `\|c\|` merkezi cok DUSUK (< 0.3) | Beklenti duser ama plan DEGISMEZ — sondalar `rho`'yu zaten dogrudan olcuyor. Yalnizca kappa kuculur. |
| Olculemedi | 0.57 / [0.17, 1.26] kalir; belirsizlik acikca raporlanir. |

> **Onemli:** `|c|` planin DOGRULUGUNU etkilemez, yalnizca BEKLENTIMIZI.
> Sondalar `rho`'yu LB'de dogrudan olcer; `|c|` ne cikarsa ciksin
> yakalanir.

---

## Olcum 3 — `n11_eksen_secimi.json` : daha iyi agirlik var mi?

Karsilastirilan: mevcut `KATS = 1.95|rho_s|` agirligi, `rho_cv` agirligi,
ikisinin buzmeli birlesimi, ileri secim.

| sonuc | yapilacak |
|---|---|
| Bir yontem anlamli kazandiriyor | Eksen listesi + agirlik listesi JSON'dan m148'e takilir. Sonra **m161 zincir sinamasi TEKRAR kosulur.** |
| Anlamli fark yok | Statuko (`KATS`) kalir. |

---

## Olcum 4 — `n12` kirmizi takim : hata var mi?

| sonuc | yapilacak |
|---|---|
| Kritik hata bulundu | Once DUZELT, sonra m161 zincir sinamasi TEKRAR. Duzeltme dogrulanmadan hicbir gonderim yapilmaz. |
| Hata yok | Kayda gecer. |

---

## 6. hakkin ALTERNATIF kullanimi — UYARLANABILIR 5. SONDA

`docs/76` 6. hakki "dogrulama-ve-onarim" icin ayirmisti. Daha degerli bir
kullanim var ve karar D4 sonrasina birakilabilir.

**Fikir:** dort blok olculdukten sonra hangi blogun en buyuk `rho`'yu
tasidigini BILIYORUZ. O blogu ikiye bolup ALT YONUNU olcmek, onceden
planlanmis bir 5. bloktan kesinlikle daha iyidir.

**Neden:** blok icindeki agirliklandirma da yanlis olabilir. Blok `k`'yi
iki dik yarima bolersek, ayni Cauchy-Schwarz bir kat asagida isler:

```
rho_k1^2 + rho_k2^2  =  rho_k^2 / cos^2(phi)
```

`phi` = blok ICINDEKI gercek dagilimla bizim agirliklandirmamiz arasindaki
aci. `rho_k = 0.11` ve `cos^2(phi) = 0.7` ise kazanc `0.0052` rho^2 —
gereken toplamin (`0.0225`) dortte biri. Kucuk bir sayi degil.

**Sira:** D1,D2,D3 (31 Agu) → D4, D5(uyarlanabilir), Z_NIHAI (1 Eyl).
Tam 6 hak, **yedek kalmaz**.

**KARAR NE ZAMAN:** D4'ten sonra.

| durum | 6. hak |
|---|---|
| Zincir temiz yurudu, olculen `rho`'lar beklentiyle tutarli | **D5 uyarlanabilir sonda** |
| Herhangi bir tutarsizlik ya da surpriz oldu | **Yedek olarak sakla** |

**Uygulama gereksinimi:** `GD_5`, secilen blogun eksen kumesi ICINDE
yasayan ve `GD_1..GD_4`'e dik bir yon olmalidir. Blok `k`'nin eksenlerini
`|rho_s|` medyanindan ikiye bolup ust yarinin `BETA`'sini `GD_k`'ye
diklestirmek yeterlidir. Bu yetenek m148'e EKLENMELIDIR (henuz yok).

---

## Her degisiklikten sonra ZORUNLU

1. `n07_temiz_kurulum.py` kosulur (sahte dosyalari siler, D1'i yeniden
   uretir, dogrular).
2. `m161_zincir_testi.py` kosulur (sentetik gercekle uctan uca; sureyi
   SINIRLAMA — onceki kosu 30 dk sinirina takilip yarim kaldi).
3. `n05_beklenti.py` ve `n13_iki_model.py` yeniden kosulur.
4. Sonuclar `docs/75`'e islenir.

---

## DEGISMEYEN

- **Onay olmadan gonderim yok.** Komut kullaniciya verilir.
- Gonderimden sonra liste OKUNUR.
- Nihai 2 secim yalnizca tarayicidan; once "You selected X of N" okunur.
- Yedek secim `tuketim_YP_seviye.csv` (1.00115).
- En kotu durumda nihai skor ~1.00101 — yedekten iyi. **Asagi yonlu koruma
  her yapilandirmada gecerlidir.**
