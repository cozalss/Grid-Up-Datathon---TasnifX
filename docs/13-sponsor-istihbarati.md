# Sponsor İstihbaratı — GDZ ve ADM Elektrik

Bu belge, 21 Ağustos açılış buluşmasında ve **Kapı 3 (ilk 10 jüri sunumu, 7–10 Eylül)**
karşınıza çıkacak insanların kim olduğunu, neyle ölçüldüklerini ve şu anda neyin
peşinde olduklarını toplar.

Etiketler: `[D]` birincil/haber kaynağından doğrulandı · `[?]` tek kaynak, teyit edilmedi
· `[Ç]` kaynaklar çelişiyor.

Araştırma tarihi: **17 Ağustos 2026**.

---

## 0 · Tek cümlelik en önemli bulgu

**GDZ ve ADM rakip iki şirket değil — ikisi de Aydem/Bereket Enerji çatısı altında
kardeş şirket.** `[D]` 2026'da datathon'un iki markayı birleştirmesinin sebebi bu;
"iki ayrı müşteriye hitap eden bir çözüm" diye konumlanmak yanlış olur. **Tek bir
dağıtım grubu, beş il, ~10 milyon insan.**

---

## 1 · Kim oldukları

| | GDZ Elektrik | ADM Elektrik |
|---|---|---|
| İller | İzmir, Manisa | Aydın, Denizli, Muğla |
| İlçe | 46–47 `[Ç]` | 49 `[D]` |
| Tüketici | ~3,3–4 mn `[Ç]` | ~2,3–2,4 mn `[Ç]` |
| Hizmet alanı | ~25.000 km² `[Ç]` | ~33.000 km² `[D]` |
| Kuruluş / özelleştirme | Gediz EDAŞ 2005; 2012 ihalesi 1,231 mia USD `[D]` | Türkiye'nin **ilk** özel dağıtım lisanslı şirketi `[D]` |
| 2025 yatırım bütçesi | **8,1 mia TL** `[?]` | 500 mn USD Eurobond ihracı `[D]` |

**Birlikte:** ~58.000 km², beş il, ~10 milyon insan, 6 milyon+ abone. `[D]`

> **Sayı çelişkileri kasıtlı olarak bırakıldı.** "Tüketici" (hizmet verilen kişi) ile
> "abone" (sayaç/sözleşme) farklı şeyler ve kaynaklar ikisini karıştırıyor. Sunumda
> rakam kullanacaksanız **açılış buluşmasında teyit ettirin** — yanlış rakam, jürinin
> kendi şirketi hakkında sizi düzeltmesi demektir.

### Teknoloji altyapısı (ADM tarafı, açıklanmış) `[D]`

- **SCADA/DMS:** 616 trafo ve dağıtım merkezi izleniyor, **3.406 OG fideri** uzaktan kumandalı
- **OSOS:** ~32.013 sayaç uzaktan okunuyor (yüksek tüketimli müşteriler, üretim tesisleri, aydınlatma)
- **CBS:** Üç ilin şebekesi dijital modelde; SCADA/DMS, CRM ve varlık yönetiminin ana veri kaynağı
- **Çağrı merkezi:** 7/24, 2016'da kapasite dört katına çıkarıldı

---

## 2 · Yeni yönetim — ikisi de 2026'da geldi `[D]`

| | Ad | Nereden geldi |
|---|---|---|
| **GDZ Genel Müdürü** | **Ahmet Bayramoğlu** | Aydem Enerji Dağıtım Grubu YK Başkan Yardımcısı |
| **ADM Genel Müdürü** | **Emrah Kalkan** | Aydem grup içi üst görevler; operasyondan **teknoloji odaklı süreçlere** uzanan profil |

Atama **15 Ocak 2026**. İkisi de enerji sektöründe 20+ yıl. Açıklanan 2026 öncelikleri:
arz güvenliği, altyapı yatırımı, kesintisiz dağıtım.

**Bunun sizin için anlamı:** İki genel müdür de göreve yeni başladı ve **görünür kazanç**
arıyor olacaklar. Kalkan'ın profili özellikle teknoloji tarafında. "Bu bir araştırma
projesi" değil, **"bu önümüzdeki çeyrekte pilot edilebilir"** mesajı bu izleyicide daha
çok karşılık bulur.

---

## 3 · Neyle ölçülüyorlar — EPDK 5. Tarife Dönemi (2026–2030) `[D]`

Bu, sunumun iş değeri kısmının **omurgası**. Bir dağıtım şirketi kâr etmez, **gelir
tavanı** alır — ve o tavan hizmet kalitesine bağlıdır.

**Kalite faktörü dört bileşenden hesaplanıyor:**

1. **Tedarik sürekliliği** ← kesinti tahmininin doğrudan girdiği yer
2. Teknik kalite performansı
3. Kullanıcı memnuniyeti
4. İş sağlığı ve güvenliği

**İzlenen göstergeler:**

| Gösterge | Ne ölçer |
|---|---|
| **SAIDI** (OKSÜRE) | Kullanıcı başına ortalama kesinti **süresi** |
| **SAIFI** (OKSIK) | Kullanıcı başına ortalama kesinti **sıklığı** |
| **AENS** | Kesintiden kaynaklanan ortalama **dağıtılmayan enerji** (2021–25'te eklendi) |

Değerlendirme, şirketin **önceki üç yılın aritmetik ortalamasıyla** kıyaslanarak bir
iyileştirme oranı üzerinden yapılıyor.

**Ceza mekanizması:** Süresinde ödenmeyen tazminatların **1,2 katı gelir tavanından
düşülüyor.** Zamanında tamamlanmayan yatırımlara da ceza var.

**5. dönemde değişenler:**
- Yatırımlar reel olarak **~1,5 kat** artırıldı
- **Planlı bakım bütçesi reel 2,1 kat artırıldı** ← bu, sizin için en önemli tek cümle
- 1/1/2026'dan itibaren teknik kalite verisi sayaçlardan **kullanıcı ve tesisat numarası
  bazında, veri kaybı olmadan** toplanmak zorunda

---

## 4 · Şu anda ne yapıyorlar — ADMS projesi `[D]`

**20 Mayıs 2026'da imzalandı** (datathon'dan üç ay önce): ADM ve GDZ, **Schneider
Electric** ve **Inavitas** ile anlaşarak **EcoStruxure ADMS**'e geçiyor.

Kapsam:
- Mevcut **SCADA** tamamen yenileniyor
- **Kesinti Yönetim Sistemi (OMS)** yenileniyor
- **DMS** (dağıtım yönetimi) + **DERMS** (dağıtık enerji kaynakları) entegrasyonu
- CBS entegrasyonu, gerçek zamanlı izleme ve uzaktan kumanda
- 5 il, 6 milyon+ abone; anahtar teslim

Alıntılananlar: **Mehmet Özalp** (Schneider Electric Türkiye/Orta Asya Bölge Başkanı),
**Erman Terciyanlı** (Inavitas CEO).

> **Bu, datathon'un "neden şimdi"sidir.** Jüri şu anda kesinti yönetim sistemini
> sıfırdan kuruyor. Sizin modeliniz akademik bir egzersiz değil, **o sisteme
> beslenebilecek bir bileşen** olarak konumlanmalı.

---

## 5 · Grid Up aslında ne — datathon bir kol, program dokuz aylık `[D]`

Grid Up, **açık inovasyon / hızlandırma programı**. Datathon onun bir parçası.

- **Ar-Ge birimi** yürütüyor
- **9 aylık** yapılandırılmış süreç, "hızlı test et, hızlı öğren"
- Girişimcilere **saha erişimi, veri ve operasyonel destek** veriliyor
- İlan edilen 12 tema: **veri analitiği, yapay zekâ, otomasyon, IoT, öngörücü
  yaklaşımlar**, operasyonel verimlilik, yeşil dönüşüm, akıllı şebekeler, enerji
  verimliliği, siber güvenlik, karbon ayak izi, yenilenebilir entegrasyonu
- Başvuru portalı: `inovasyonplatformu.admelektrik.com.tr`

**Anlamı:** Datathon bir yetenek ve fikir hunisi. İlk 10'a girip sunum yaparken
muhatabınız "iyi model mi?" diye değil, **"bu bizim operasyonumuza girer mi?"** diye
bakan bir Ar-Ge birimi olacak.

---

## 6 · Takvim teyidi `[D]`

Coderspace'in İngilizce sayfası, deponun `docs/00`'daki e-posta takvimini **doğruluyor**:

| Tarih | Aşama |
|---|---|
| 21 Ağustos – 1 Eylül | Kaggle, 12 gün |
| **24 Ağustos** | Kaggle'da **takım kurma son tarihi** |
| 7–10 Eylül | İlk 10'un jüri sunumu |

> Web'de hâlâ dolaşan **3–16 Ağustos** takvimi eskidir; `docs/00` bunu zaten
> işaretlemişti, araştırma teyit etti.

---

## 7 · Bunları sunuma nasıl çevirirsiniz

Bu bölüm, yukarıdaki olguların **yarışma karşılığıdır** — asıl değer burada.

**1 · İş değerini TL cinsinden bir zincire bağlayın.**
`Kesinti tahmini → planlı bakımın doğru ilçeye/güne yönlendirilmesi → SAIDI/SAIFI
iyileşmesi → kalite faktörü → gelir tavanı`. Planlı bakım bütçesi **2,1 kat arttığına**
göre, "bu bütçeyi nereye harcayacağınızı söyleyen model" doğrudan onların gündemidir.
2024 birincisinin sunumunun son üç slaydının tamamen iş değeri olması tesadüf değil.

**2 · Çıktınızı ADMS'e bağlanabilir biçimde tarif edin.**
"İlçe–gün seviyesinde risk skoru, OMS'nin ekip yönlendirme modülüne girdi olacak
biçimde" cümlesi, mimari şeması göstermekten güçlüdür. Sistemi **şu anda** kuruyorlar.

**3 · Dağıtık üretimi (DERMS) anlatıya katın.**
Ege'de güneş yoğun ve DERMS projenin ilan edilmiş parçası. `solar.py`'deki güneş
geometrisi feature'larını yalnızca "hava durumu" diye değil, **dağıtık üretimin şebekeye
etkisi** bağlamında da konumlandırabilirsiniz.

**4 · Modelin sınırlarını siz söyleyin.**
`benchmark_gercek.json`'daki `kazanan: null` kararı ve ablasyondaki negatif aileler
(tatil, güneş, takvim ölçülen veride zarar veriyordu) bir zayıflık değil, **olgunluk
işaretidir.** Operasyona model koyacak bir Ar-Ge birimi, "her şey harika" diyen ekipten
çok "şunu ölçtük, şu kadarına güveniyoruz" diyen ekibe güvenir.

**5 · Dağıtım maliyetini somut söyleyin.**
2024 birincisi "~25 MB model kümesi, yeni veriyle eğitilebilir" demişti. Bu tür somut
cümleler, iş biriminden gelen jüri üyesi için soyut mimariden değerlidir.

---

## 8 · Açılış buluşmasında (21 Ağustos 14:00) sorulacaklar

`docs/01`'deki yarışma soruları duruyor; bunlar **şirket tarafı** sorular:

1. Hedef değişken hangi operasyonel karara besleniyor — bakım planlama mı, ekip
   yönlendirme mi, yatırım önceliklendirme mi?
2. Model çıktısı **ADMS/OMS programına** mı bağlanacak, yoksa ayrı mı değerlendirilecek?
3. Veri hangi sistemden geliyor — SCADA, CBS, çağrı merkezi, OSOS? (Kolon adlarının
   anlamını bu belirler.)
4. Beş ilin tamamı mı, yoksa tek şirketin bölgesi mi?
5. Kesinti kayıtlarında **planlı/plansız** ayrımı var mı, planlı olanlar test setinde de
   veriliyor mu? (2024'te veriliyordu — bedava kovaryat.)

---

## 9 · Kaynaklar

- ADM kurumsal ve yatırımlar: [admelektrik.com.tr](https://www.admelektrik.com.tr/hizmetler/yatirimlar)
- ADMS anlaşması: [Enerji Bülteni, 3 Haziran 2026](https://www.enerjibulteni.com/2026/06/03/adm-ve-gdz-elektrik-ecostruxure-adms-ile-yeni-doneme-geciyor/)
- Yönetim değişikliği: [Dünya Gazetesi, 15 Ocak 2026](https://www.dunya.com/sirketler/adm-ve-gdz-elektrikte-ust-yonetimde-degisim-yeni-genel-mudurler-atandi-haberi-811474)
- EPDK 5. tarife dönemi: [AA Enerji Terminali](https://www.aa.com.tr/tr/enerjiterminali/elektrik/epdk-elektrik-dagitim-sirketlerine-yonelik-2026-2030-tarife-doneminin-yol-haritasini-belirledi/53769)
- Kalite yönetmeliği (SAIDI/SAIFI/AENS): [Lexpera konsolide metin](https://www.lexpera.com.tr/mevzuat/yonetmelikler/elektrik-dagitimi-ve-perakende-satisina-iliskin-hizmet-kalitesi-yonetmeligi)
- Grid Up programı: [ADM basın bülteni](https://www.admelektrik.com.tr/medya-merkezi/basin-bultenleri/Adm%20-ve-Gdz%20-Elektrik%E2%80%99ten-%20Ac%C4%B1k%20-Inovasyonla%20-Elektrik%20-Dag%C4%B1t%C4%B1m%C4%B1nda-%20Yeni%20-Donem) · [Coderspace etkinlik sayfası](https://coderspace.io/en/events/grid-up-datathon/)
- Şirket profilleri: [Enerji Atlası — Gediz](https://www.enerjiatlasi.com/elektrik-dagitim-sirketleri/gediz.html) · [Aydem Enerji](https://www.aydemenerji.com.tr/bilgi/13/elektrik-dagitim/)
