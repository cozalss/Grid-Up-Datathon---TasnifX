# Grid Up Datathon — Yarışma Brifingi

**Düzenleyen:** Coderspace · **Sponsorlar:** GDZ Elektrik, ADM Elektrik
**Format:** Kaggle "In-Class" · **Takım:** en fazla 4 kişi

---

## Takvim

Aşağıdaki takvim **17:28'de gelen "Güncel Program Belli Oldu" e-postasından** alınmıştır
ve yarışma sayfasındaki daha eski takvimi (3–16 Ağustos) geçersiz kılar.

| Tarih | Aşama |
|---|---|
| **21 Ağustos, Cuma 14:00–15:00** | Açılış buluşması (YouTube canlı yayın) |
| **21 Ağustos** (buluşmadan sonra) | Kaggle linki e-posta ile paylaşılır |
| **21 Ağustos 15:00 – 1 Eylül 23:59** | Yarışma süreci (**12 gün**) |
| **1–4 Eylül** | Private leaderboard ilk 20'nin notebook'ları toplanır ve değerlendirilir |
| **7–10 Eylül** | İlk 10'un online final sunumları (net tarih sonra duyurulacak) |

> **Ödüller:** 1. 75.000 TL · 2. 50.000 TL · 3. 25.000 TL

---

## Değerlendirmenin üç ayağı

Bu yarışma **yalnızca leaderboard skoru değildir.** Üç kapı var ve her biri elemeli:

```
  Kapı 1: Private leaderboard ilk 20  →  skor
  Kapı 2: Notebook değerlendirmesi     →  kod kalitesi, açıklanabilirlik, yöntem
  Kapı 3: Final sunumu (ilk 10)        →  iş değeri, anlatım, model savunması
```

**Sonuç:** Notebook'u yarışmanın son günü toparlamaya çalışmak bir hatadır. Kod boyunca
temiz, açıklamalı ve tekrarlanabilir tutmak — bu repodaki yapının var oluş sebebi budur.

---

## Kritik kurallar

### E-posta eşleşmesi — en sık yapılan hata

Kaggle'a **yalnızca Coderspace'e kayıtlı e-posta** ile katılabilirsin. Kaggle profilindeki
e-posta farklıysa ve değiştiremiyorsan, `helloworld@coderspace.io` adresine yazıp manuel
eşleştirme iste. **Bunu 21 Ağustos'a bırakma.**

### Takım

- En fazla **4 kişi**
- Her üye **ayrı ayrı** başvuru formunu doldurmalı
- Herkes **aynı takım ismini** girmeli
- Kaggle'da takım kurma penceresi sınırlıdır — açılış buluşmasında net tarihi sor

### Kaggle'da takım kurma — adım adım

1. Herkes **kendi hesabından** yarışmaya girer ve kuralları kabul eder
   ("I Understand and Accept"). Bu yapılmadan takım kurulamaz.
2. Yarışma sayfasında **Team** sekmesine git.
3. Bir üye diğerine **"Send Merge Request"** gönderir.
4. Karşı taraf kendi Team sayfasından **kabul eder** (çift taraflı onay şart).
5. Takım adını **başvuru formundaki isimle birebir aynı** yap.

> Birleşmeden önce yapılan bireysel submission'lar bazı yarışmalarda birleşik takımın
> kotasına sayılır. Gereksiz submission yapmadan önce takımı kur.

### Diskalifiye riskleri

- Etik dışı davranış, hile, rahatsızlık verici söylem
- Birden fazla takıma katılım
- Takım kurma penceresi dışında takım oluşturma
- Sponsor çalışanları ve 1. derece akrabaları katılamaz

---

## Bilinmeyenler — açılış buluşmasında sorulacaklar

Bunları 21 Ağustos'ta YouTube sohbetinden sor. Cevapları pipeline kararlarını değiştirir:

1. **Veri seti neyi içeriyor?** Zaman serisi mi, kesitsel mi? Kaç satır?
2. **Resmi metrik nedir?** (RMSE / RMSLE / MAE / AUC / F1 — her biri farklı strateji gerektirir)
3. **Public/private leaderboard bölünmesi nasıl?** Yüzde kaç, hangi kritere göre?
4. **Günlük submission limiti kaç?**
5. **Final submission olarak kaç tane seçilebiliyor?**
6. **Harici veri kullanımı serbest mi?** (hava durumu, TÜİK nüfus, EPİAŞ)
7. **Kaggle takım kurma son tarihi nedir?**
8. **Notebook değerlendirme kriterleri neler?** Ağırlıklandırma nasıl?
9. **Final sunumu kaç dakika, hangi format?**
10. **Notebook'un Kaggle'da çalışması zorunlu mu, internet açık mı?**

---

## Hazırlık kontrol listesi

- [x] Coderspace ve Kaggle e-postaları **aynı** (en kritik madde)
- [x] Kaggle hesabı var, profil tamamlanmış
- [x] Takım üyeleri netleşti, herkes formu doldurdu, takım adı üzerinde anlaşıldı
- [x] Açılış buluşması takvime eklendi (21 Ağustos 14:00)
- [x] Yukarıdaki soru listesi hazır
- [x] Bu repo kuruldu, `pytest` yeşil, `smoke_test.py` çalışıyor — 330 test, duman testi 57 sn
- [x] Hava durumu verisi indirildi (`scripts/fetch_weather.py`) — 20 konum, 48.180 satır

> Kalan iş **yarışma günü**ne ait, hazırlığa değil: Kaggle'da takım birleştirme
> (son tarih 24 Ağustos 23:59, birleşmeden önce submission YOK) ve açılış
> yayınında on sorunun sorulması.
