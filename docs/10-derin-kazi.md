# Derin Kazı — 3. araştırma dalgası (15 Ağustos 2026)

> Beş hat: akademik literatür + Türkçe tezler · Kaggle taranmamış yarışmalar ·
> alan-dışı sayım bilimi · mekânsal/graf hükmü · ADM bölgesi istihbaratı.
> [docs/08](08-gdz-2024-birincisi-ve-2026-meta.md) ve
> [docs/09](09-kazananlar-el-kitabi.md)'un devamı; her iddia kaynaklı.

---

## 1 · Türkçe tezler (YÖK arşivi dogrudan tarandı — 6 tez okundu)

- **Sivas tezi (2025, No 972166)** — görevimizin neredeyse aynısı: gün-öncesi
  GÜNLÜK KESİNTİ SAYISI. Feature önem sırası: (1) son 24s maks rüzgâr,
  (2) yıldırım yoğunluğu, (3) **son bakımdan geçen gün sayısı**. Kesintilerin
  %80'i hava kaynaklı. CNN-LSTM R²=0.56.
- **Çankırı tezi (No 555912)** — TERS-SEZGISEL: aşırı sıfıra rağmen düz
  Poisson, ZIP'i VE hurdle-Poisson'u yendi (DIC + PPC). Ders: hurdle'ı
  varsayma, düz Poisson/NB baseline'ı hep yanında koştur.
- **Diyarbakır tezi (No 874955)** — hava-açıklamalı ve hava-bağımsız kalıntıyı
  ayrıştırıp kalıntıyı "şebeke sağlığı" göstergesi yapma fikri; CatBoost kazandı.
- **EPDK resmi eşikleri**: Kentsel ≥50.000 / Kentaltı ≥2.000 / Kırsal <2.000
  nüfus — DSO'ların tazminat rejiminin kendi sınıflaması; hazır ilçe özniteliği.
- Aras EDAŞ makalesi (dergipark kfbd.1482179): LGBM/XGB %93-96; Fırat EDAŞ'ın
  Ankara Üniv. ile EPDK onaylı "EnergyMind" arıza-tahmin platformu (2026) canlı emsal.

## 2 · Kaggle süpürmesi — bizde olmayanlar

1. **Kararlılık-cezalı harman seçimi** (Home Credit 2024): ağırlık seçiminde
   `ortalama MAE − λ·fold_std` — M5 1.'sinin ilkesiyle çifte kaynak.
2. **Power-mean harman** (ASHRAE 1.-2.): doğrusal yerine kuvvet ortalaması.
3. **Kamu-kaynak sızıntı taraması** (ASHRAE dersi): verinin ham kaynağı bir
   kamu sitesinde yayınlanmış olabilir — gün-1'de aktif ara.
4. **Çapa-lag** (Web Traffic 1.): "52/13 hafta önce aynı gün" tekil noktalar.
5. **Mekanizma eşleştirme** (Ventilator 1.): bilinen prosedür/SLA varsa
   kural-tabanlı düzeltme istatistiği yener.
6. Predict Future Sales: hedefin bilinen üst sınırı varsa tahmine de uygula.

## 3 · Alan-dışı sayım bilimi

1. **Exposure-offset** (sigorta pratiği, kanıtlı): `log(müşteri/nüfus)` →
   `init_score/base_margin`; Poisson/Tweedie log-link'te doğal offset. İlçe
   boyut farkını öğrenilecek şeyden verili öncüle çevirir.
2. **Monotonik kısıtlar**: kredi-PD benchmark'ında maliyet ~0-2.9; seyrek
   veride overfit sigortası.
3. **Hawkes-esinli üstel-azalan recency**: `Σ exp(-Δt/τ)` — art arda arıza
   kümelenmesi; tam Hawkes kanıtsız, ucuz feature olarak dene.
4. **Flusion tarzı** (CDC ensemble'ını yendi): havuzlanmış GBM + ilçe-başına
   basit mevsimsel-AR blend — az geçmişli ilçelerde en etkili.
- Teyitler: hurdle tercihi hidrolojide de standart; kantil merdiveni salgın
  tahmininin (WIS) pratiğiyle aynı hizada; DengAI/çağrı-merkezi yeni şey vermedi.

## 4 · Mekânsal hüküm: GNN YOK

96 düğümde GNN'in feature+GBDT'yi yendiği tek yarışma yok (Traffic4cast 2022:
LightGBM %1 farkla 2.; 2020-21'i U-Net kazandı). Kesinti mekânsal
otokorelasyonu literatürde de zayıf-orta (Moran I 0.04-0.36) — bizim +0.18
ölçümümüz tutarlı. Gün-1 ucuz deneyleri: (1) mesafe-ağırlıklı komşu agregatı
(düz kNN yerine), (2) komşu istatistiklerini genişlet (min/max/std — KDD Cup
2018 kazananı), (3) ikinci-derece komşu lag'i (zaman-kutulu dene).

## 5 · ADM bölgesi — 2026'nın yeni kozu

- **Muğla**: yaz nüfusu yerleşiğin **2-5 katı** (ADM'nin resmi açıklaması);
  yaz bakım/kesinti dalgaları haberli; 2021 Marmaris yangını 70k hektar —
  eğitim verisindeki sıçrama tarihleri belli.
- **Aydın**: jeotermal üretim üssü (tüketimin 2.1 katı üretim); pamuk/incir
  sulama sezonu Haz-Eyl; **2026 DSİ kısıtlı sulama kararı** pompa desenini
  değiştirecek.
- **Denizli**: tekstil OSB — hava değil sipariş güdümlü, hafta içi/sonu
  vardiya ritmi; ayrı mevsimsellik rejimi.
- **Açık veri katalogu** (17 kaynak; ayrıntı ajan raporunda): KTB ilçe-bazlı
  konaklama/geceleme, TÜİK ADNKS ilçe nüfusu + tarım sayımı, EFFIS yangın
  poligonları, AFAD/Kandilli deprem katalogu (Aydın-Denizli graben hattı!),
  TEİAŞ/EPDK raporları, ADM hizmet-kalite göstergeleri sayfası.
- Gün-1 avantajları: yaz-nüfus çarpanı feature'ı (KTB geceleme ile interpole),
  sulama-penceresi kodlaması, OSB sanayi-shift rejimi, yangın-riski
  etkileşimi, graben-hattı deprem mesafesi.

## 6 · Uygulama kuyruğu (3. dalga)

**Şimdi (ölçülebilir):** kararlılık-cezalı harman · power-mean harman ·
exposure-offset (nüfus vekiliyle) · Hawkes-decay recency · son-olaydan-geçen-gün
("bakım açığı" vekili, Sivas tezi #3) · mesafe-ağırlıklı komşu + geniş komşu
istatistikleri · monotonik kısıt desteği.
**Gün-1 sonrası:** düz Poisson/NB baseline (Çankırı dersi) · EPDK
kentsel/kırsal sınıfı · hava-bağımsız kalıntı trendi · çapa-lag (veri yeterse) ·
ADM feature'ları (KTB/TÜİK/EFFIS verisi çekilerek) · kamu-kaynak sızıntı taraması ·
Flusion tarzı ilçe-AR üyesi.
