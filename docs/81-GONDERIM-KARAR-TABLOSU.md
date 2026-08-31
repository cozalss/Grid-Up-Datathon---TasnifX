# 81 — GÖNDERİM KARAR TABLOSU (1 Eylül, 3 hak)

Amaç: yarın hesap YAPILMASIN. Her dalın dosyası hazır, formüller burada.
Bağlam: `docs/80-DEVIR-31agustos-2200.md`. ONAY OLMADAN GÖNDERİM YOK.

## HAZIR DOSYALAR (hepsi doğrulandı; `experiments/model29/p_kalici/aday_csv/`)

| dosya | içerik | ne zaman |
|---|---|---|
| **p21_harman311_olu50.csv** | soğuk 3/1/1 + ölü trafo ×0.5 | **HAK 1 — her koşulda** |
| p21_esit_olu50.csv | soğuk EŞİT + ölü trafo ×0.5 | Hak 2, oran yüksekse |
| p20_harman_ESKI_3_1_1_V1_seviyesiz.csv | yalnız soğuk 3/1/1 | yedek/karşılaştırma |
| p20_harman_ESIT_V1_seviyesiz.csv | yalnız soğuk EŞİT | yedek |
| `submissions/tuketim_YP_seviye.csv` | ölçülmüş 1.00115 | son seçim sigortası |
| (gece çıkabilir) p22_311huber_* | 3/1/1 + lgbm-huber üyesi | p19 agent ölçüyor |

V2_seviyeli dosyaları KULLANMA (çift sayım, docs/80 §4).

## HAK 1 → TAŞIMA ORANI ÖLÇÜMÜ

Gönder: `p21_harman311_olu50.csv`. Gelen skor S1 ile:

```
oran = (1.00115 − S1) / 0.00788   # p25 kirmizi takim duzeltmesi (eski 0.00964)
```

(0.00788 = p25 duzeltilmis kirpmasiz beklenen ΔRMSLE. ESKI 0.00964 hataliydi:
blok olcumunde delta AGIRLIKSIZ merkezlenmis, olcut test-agirlikliydi -> kacak
seviye. Duzeltilmis yapi kazanci +0.0712 (eski +0.0867), kis26 -0.0183 (isaret
2/3, tohum 6/9). Dosya DOGRU, kusur olcumdeydi.)

**3. sira icin gereken tasima orani: >= 0.71** (eski iddia 0.58).
Duzeltilmis bant: 0.99327 (oran 1.0) .. 0.99721 (oran 0.5).
Muhafazakar kose (K25 + oran 0.5): 1.00063 ~ notr.

Beklenen S1 senaryoları:
| oran | S1 | anlamı |
|---|---|---|
| 1.0 | 0.99151 | 2.'yi geçer |
| 0.75 | 0.99392 | 2.'yi geçer |
| 0.58 (ölçülü tipik) | 0.99556 | 3. sıra SINIRI |
| 0.5 | 0.99633 | 4. civarı |
| 0.0 | 1.00115 | değişmez |
| <0 | >1.00115 | ters — soğuk kohort CV'den kopuk |

## HAK 2 — S1'E GÖRE

**Dal A — S1 ≤ 0.9940 (oran ≥ 0.75):** taşıma güçlü → `p21_esit_olu50.csv`.
EŞİT'in yapı kazancı %38 daha büyük (+0.1199 vs +0.0867); riski kırpma
merdiveninde kis26 dönüşü. Oran yüksekken bu risk alınır; beklenen
S2 ≈ 1.00115 − oran×0.0133. Oran 1.0 → **~0.9878, 1. sıra menzili.**

**Dal B — 0.9940 < S1 ≤ 0.9963 (oran 0.5–0.75):** 3.'lük sınırda →
gece çıkan EK KATMAN varsa (p22 lgbm-huber üyeli 3/1/1, ya da sıcak cat-τ
doğrulanırsa onun test dosyası — p18_yeniden_egit.py P18_MOD=test ile üret,
birebir doğrulama şartıyla) 3/1/1 üstüne bindir ve gönder. Yoksa
`p21_esit_olu50.csv` yine en iyi ikinci bahis (beklenen kazanç farkı
oran×0.0037).

**Dal C — S1 > 0.9963 (oran < 0.5):** kazanç zayıf taşınıyor → EŞİT'in ek
kazancı da (oran×0.0037) hedefe yetmez. Hak 2'yi YÜKSEK VARYANSA ayır:
gece katmanlarından hangisi CV'de en büyükse onu 3/1/1 üstüne bindir.
Hiçbiri yoksa hak 2'yi beklet, sıcak kampanyanın sonucunu bekle.

**Dal D — S1 > 1.00115 (ters):** soğuk düzeltme LB'de zarar → p20/p21
ailesini bırak. Hak 2-3'ü sıcak taraf katmanına ya da span-demet yüksek
varyans bahsine ayır. Son seçim sigortası YP_seviye zaten duruyor.

## HAK 3 — İKİ ÖLÇÜMDEN SONRA

S1 ve S2 ile hem oran hem (A dalıysa) EŞİT-vs-3/1/1 gerçek farkı ölçülmüş
olur. Hak 3 = en iyi ölçülmüş bileşim + varsa doğrulanmış sıcak katmanı.
Formül: her adayın beklenen skoru = 1.00115 − oran × (CV ΔRMSLE toplamı).
CV ΔRMSLE değerleri: 3/1/1 yapı 0.00964, EŞİT yapı 0.01330, ölü trafo
0.00055, (p22/p15 katmanları gece raporlarından).

## SON SEÇİM (yarışma bitmeden, TARAYICIDAN, 2 dosya)

1. En iyi LB skorlu gönderim
2. `tuketim_YP_seviye.csv` (1.00115) — ANCAK üç hak da 1.00115'ten iyiyse
   ikinci en iyi gönderim seçilir
Önce "You selected X of N" satırını OKU. API'den yapılamaz.

## GÖNDERİM MEKANİĞİ (docs/73 + kurallar)

```
kaggle competitions submit grid-up-datathon -f <dosya> -m "<mesaj>"
kaggle competitions submissions grid-up-datathon | head -5   # HER gönderimden sonra ZORUNLU
kaggle competitions leaderboard grid-up-datathon --show | head -8
```
Zaman aşımı "gönderilmedi" demek DEĞİL — önce listeyi oku.
Her hak için kullanıcıdan AYRI onay al.
