# Kazananlar El Kitabı — 2. araştırma dalgası (15 Ağustos 2026)

> Beş paralel hat: rakip dosyası · M5 derin madeni · hava→kesinti bilimi ·
> zaman serisi temel modelleri · LB stratejisi. Her iddia kaynaklı;
> [docs/08](08-gdz-2024-birincisi-ve-2026-meta.md)'in devamıdır.

---

## 1 · Rakip dosyası

**Anıl Öztürk (Pikachow):** anilozturk.net/local-competitions = 16 datathon'luk
resmi envanter, çoğu sunum PDF'li (BTK 2024+2025 birincisi, AHE 2024 birincisi,
GDZ 2022 3.sü, GDZ 2024 birincisi...). GDZ 2022 çözümünün tam kodu:
kaggle.com/code/nlztrk/long-live-the-shallow-models. Oradan bizde olmayanlar:
sürekli hedefi kutulayıp stratified fold (`pd.cut`+`StratifiedKFold`);
TF-IDF ile şebeke-unsuru kodundan alt-dize bayrakları; **eşzamanlı aktif
kesinti sayacı** (aynı il/ilçede o an kaç kesinti daha var).

**İzmir Bombası (GDZ 2024 3.sü) — tam notebook bulundu:**
github.com/sercanyesiloz/Gdz-Elektrik-Datathon-2024. Bizde olmayanlar:
**zamana göre doğrusal örnek ağırlığı** (ilçe başına 0.05→0.95);
**yumuşak IQR aykırı harmanı** (0.38×ham + 0.62×Q3+1.5IQR'a kırpılmış);
spline döngüsel kodlama; wind-chill/çiy noktası/ısı indeksi türevleri.

**gunesevitan (CIBMTR 2.):** 2. aşamada **3 farklı hedef formülasyonu**
(log-süre, KM olasılığı, NA tehlikesi) + rank-transform harmanı; başarısız
denemeleri de README'ye yazma disiplini. Kaynak:
github.com/gunesevitan/cibmtr-equity-in-post-hct-survival-predictions.
**aerdem4:** LOFO importance (github.com/aerdem4/lofo-importance) — Türk
datathon çevresinin standart feature-seçim aracı; Öztürk de kullanıyor.

## 2 · M5 madeni — bizde olmayan ilk 5

1. **Toplu-olay bayrağı** (out-of-stock analoğu): ilçelerin >%X'inin aynı gün
   kesintili olduğu günleri ayrı işaretle → hurdle'ın p(0) aşaması keskinleşir.
2. **Ölçek-farkındalıklı örnek ağırlığı** (M5 14.): ilçenin son-N-gün hedef
   varyansı/aktiflik oranı `sample_weight` olur — hep-sıfır ilçeler eğitimi
   domine etmesin.
3. **Yukarıdan-aşağı hibrit üye**: bölge/toplam günlük kesintiyi ayrı modelle
   tahmin et, tarihsel payla ilçelere dağıt → düşük-varyans harman üyesi
   (arxiv 2311.00993 bu yaklaşımla tek başına M5 top-50).
4. **"Sihirli çarpan"** (M5 2.: {0.90–0.99} taraması): round+clip'ten önce
   OOF'ta dar aralıkta grid-search'lü kalibrasyon çarpanı. UYARI: M5 1.si
   çarpan KULLANMADI ve model seçimini "fold'lar arası std minimizasyonu"yla
   yaptı — sağlamlık > parlaklık; çarpan ancak OOF kanıtıyla girer.
5. **Range-Blended augmentasyon** (M5 Uncertainty 1., kodu ele geçti —
   `data/prior/av/` yanına scratchpad'den taşınabilir): satırın hedef ve
   ölçek-bağımlı feature'larını `trailing_vol × exp(0.5·N(0,1)·scale)` ile
   böl; az-verili serilerde CV +%5-15. Kantil merdivenimize uygulanabilir.

Teyitler: Tweedie 1.1-1.3, global havuzlama, entity-embedding NN, yıl-bazlı
GroupKFold + saf holdout, rekürsif+direkt harmanı (M5 1.'sinin ana içgörüsü).

## 3 · Hava→kesinti — gün-1 sonrası eklenecekler

Bizde ZATEN var: hamle rüzgârı (`firtina_max`), yağış saati, derece-günler,
"kuraklık sonrası ilk yağmur", ıslak-zemin×rüzgâr. Eksikler (hepsi mevcut/ucuz
veriyle):

1. **Yüzey basıncı günlük min** — EAGLE-I çalışmasında 3. en güçlü değişken
   (%13.1 önem; arxiv 2512.22699). Veri çekimine `surface_pressure` ekle.
2. **Eşik-üstü rüzgâr saati** (>15/20 m/s saat sayısı) — UConn OPM'nin 2.
   değişkeni; saatlik seriden türetilir.
3. **Ardışık sıcak gece sayacı** (min sıcaklık eşik-üstü ardışık gün) —
   trafo gece soğuyamama yorulması; `sicaklik_min`'den hesaplanır.
4. **Çok pencereli kuraklık indeksi** (7/30/90 gün yağış anomalisi) —
   mevcut ikili bayrağın sürekli hali.
5. **Rüzgâr yönü sapması** (hakim yönden fark) — UK çalışması: yön sapması
   riski 2-5 kat değiştiriyor (Nature s43247-025-02176-6).

Not: GDZ'nin kendi verisiyle DEÜ çalışması (PMC11244009) hava katkısının
mütevazı, asıl sinyalin şebeke/varlık verisinde olduğunu buldu — bizim +2.5 dk
ölçümümüzle tutarlı. Hava'ya aşırı yatırım yapma.

## 4 · Temel modeller hükmü: ATLA

96 seri × güçlü kovaryat bağımlılığı = GBDT+feature bölgesi. Kanıt üçlü:
VN1'de Moirai 15 bin seriyle ve 0.0008 farkla kazandı (bizde 96 seri);
INFORMS 2025'te (birebir problem şekli) transformer/LSTM/GNN overfit edip
elendi, kazanan hurdle; FM'lerin kovaryat desteği ya yok ya yamalı.
Tek meşru opsiyon: gün-1 sonrası **Chronos-2 zero-shot** ucuz harman üyesi
denemesi (kovaryat destekli tek olgun FM). TFT/TimesFM/Moirai/TiRex: atla.

## 5 · 12 günlük submission planı (36 gönderi)

| Gün | Odak | Gönderi |
|---|---|---|
| 1 | format kontrolü + split/metrik teyidi | 1 (sıfır/ortalama baseline) |
| 2-3 | panel+takvim+lag+hava | 1/gün |
| 4-5 | komşu, zoo (LGB/XGB/Cat) | 1-2/gün |
| 6-7 | Optuna, hurdle+medyan kuralı, yuvarlama eşiği OOF taraması | 2/gün |
| 8-9 | SHAP eleme, harman, grup çarpanı (dar aralık + shrink) | 2-3/gün |
| 10 | çoklu-seed (5→15→25, marjinal kazancı ölç) | 2 |
| 11 | full-refit, notebook temizliği — yeni fikir YOK artık | 2 |
| 12 | final seçim + writeup | ≤1 |

**Final-2 kuralı:** A = en iyi CV; B = A'dan METODOLOJİK olarak farklı aile
(iki benzer final birlikte batar — kaggle general/414544 vakası). LB sadece
tie-breaker. **Probing yapma:** split rastgele %50 ise public LB zaten kendi
OOF dağılımınla aynı popülasyon; 36'lık bütçede probing değersiz (Hardt
analizi: salt şans-sömürüsü ~700 gönderi ister).

## 6 · Uygulama kuyruğu

**Şimdi (yarışma öncesi, ölçülebilir):** toplu-olay bayrağı · örnek ağırlığı
(recency×aktiflik) · kalibrasyon çarpanı yardımcısı · yumuşak IQR harmanı ·
ardışık-sıcak-gece + çok-pencereli kuraklık feature'ları.
**Gün-1 sonrası:** yukarıdan-aşağı hibrit üye · Range-Blended augmentasyon ·
eşzamanlı kesinti sayacı (veri şekline bağlı) · Chronos-2/TabM üye denemesi ·
basınç/eşik-üstü-saat için hava verisini yeniden çek.
