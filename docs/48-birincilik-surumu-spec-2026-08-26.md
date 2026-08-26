# Spec: 27 Ağustos birincilik sürümü

## Objective

TasnifX'in ölçülmüş 1.01538 RMSLE tabanını, hazır son-işlem kazançlarını ve
üretim-sadık yeni model aramasını birleştirerek 27 Ağustos gönderimleri için en
güçlü, geri alınabilir sürümü üretmek. Birincilik eşiği 1.00635; mevcut MSE
açığı yaklaşık 0.01826'dır. Hazır v80 zinciri bu açığın yaklaşık 0.00420'sini
kapatır. Kalan açık için öncelik, daha önce yanlış deney rig'iyle ölçülmüş model
ayarlarını doğru üretim geometrisinde yeniden ölçmektir.

## Tech stack

- Python 3.10+ ve `uv`
- pandas/numpy
- CatBoost, XGBoost, LightGBM ve isteğe bağlı PyTorch üyesi
- pytest, ruff ve mypy
- Kaggle CLI (yalnız gönderim geçmişini okumak ve 27 Ağustos kotasında göndermek için)

## Commands

```powershell
# Test
uv run python -m pytest -q

# Lint ve tip kontrolü
uv run ruff check src tests scripts
uv run mypy src

# Üretim-sadık aday ölçümü (aday betiğinin --help çıktısı kesin komutu tanımlar)
uv run python scripts/deney_uretim_ayarlari.py --help

# Gönderim bütünlüğü
uv run python scripts/kapi_denetim.py --ref submissions/tuketim_v67_c1335_olay.csv `
  submissions/tuketim_v80_optimum.csv submissions/tuketim_v81_sicak08.csv `
  submissions/tuketim_v82_ayirici.csv

# Gönderim geçmişini oku; gönderimden hemen önce tekrar edilir
uv run python -m kaggle competitions submissions -c grid-up-datathon
```

## Project structure

- `scripts/tuketim_model.py`: yarışma üretim modeli ve gerçek model sözleşmesi
- `scripts/deney_*.py`: zaman kesitli deney tezgâhları
- `src/gridup/`: tekrar kullanılabilir ve test edilen çekirdek
- `tests/`: birim, bütünleşme, sızıntı ve submission sözleşmesi testleri
- `experiments/`: yapılandırılmış deney sonuçları
- `submissions/`: yerel Kaggle adayları; git'e eklenmez
- `docs/`: ölçüm defteri, karar gerekçesi ve devir planı

## Code style

```python
def aday_kazandi(taban_mse: float, aday_mse: float, esik: float) -> bool:
    """Yalnız önceden belirlenen asgari kazancı geçen adayı kabul et."""
    return taban_mse - aday_mse >= esik
```

Davranış açık adlarla ifade edilir; deney adayları veri güdümlü tanımlanır;
aynı tohum, aynı maske ve aynı fold ile eşleştirilmiş karşılaştırma yapılır.
Yeni soyutlama yalnız birden fazla gerçek kullanım varsa eklenir.

## Testing strategy

- Yeni deney seçme/hesaplama mantığı için önce başarısız birim testi yazılır.
- Adaylar en az `yaz25` ve `kis26` örtüşmeyen zaman kesmelerinde ölçülür.
- Soğuk tarafta en az 5 tohum ve trafo kırpma tablosu olmadan hüküm verilmez.
- İlk eleme tek/az tohumla yapılabilir; yalnız kazanan kol tam kapıdan geçer.
- Üretim adayında tüm pytest, ruff, mypy, submission ve determinism kapıları geçer.

## Boundaries

### Always

- Üretimdeki 105 kolon, rejim maskesi, rejime özel eğitim nüfusu ve harmanla ölç.
- Aynı aday/taban tohumlarını eşleştir ve sonucu MSE uzayında raporla.
- Her tam koşudan önce ucuz kapıyla zayıf adayları ele.
- Yalnız bu hedefe ait dosyaları açıkça stage et.

### Ask first

- Yeni ücretli servis veya harici veri satın almak.
- Yarışma kurallarını etkileyebilecek yeni dış veri kullanmak.
- Bugün Kaggle gönderimi yapmak ya da uzak repoya push etmek.

### Never

- Test/LB etiket sızıntısı, public skoru satır düzeyinde tersine çözme veya yasak veri kullanımı.
- Mevcut kullanıcının untracked dosyalarını silmek, değiştirmek ya da topluca commit etmek.
- Başarısız testi kapatmak, sırrı commit etmek veya ölçülmemiş kazancı gerçek diye sunmak.

## Implementation plan

1. Mevcut v80/v81/v82 dosyalarını ve taban testlerini doğrula.
2. Üretim-sadık, iki aşamalı hiperparametre tezgâhını kur: ucuz eleme, tam doğrulama.
3. Önce yüksek etkili `learning_rate/iterations`, örnek/kolon alt örnekleme,
   çocuk ve L2 düzenleme eksenlerini rejim/aile bazında ölç.
4. Kazanan model farkı en az 0.002 MSE ise yeni üretim tabanı çıkar ve v80 zincirini
   üstüne yeniden kur; değilse ölçülmüş v80 üçlüsünü koru.
5. Paket, test, sızıntı, determinism ve submission kapılarını çalıştır.
6. Atomik commitler oluştur; GitHub push ve Kaggle gönderimini 27 Ağustos'a bırak.

## Success criteria

- Hazır güvenli aday `tuketim_v80_optimum.csv` tüm bütünlük kapılarından geçer.
- Yeni model ancak en az iki örtüşmeyen zaman kesmesinde aynı yönü gösterir ve
  önceden belirlenen 0.002 MSE eşiğini geçerse sürüme alınır.
- 27 Ağustos için üç dosyalık gönderim sırası, ters çözüm formülleri ve geri dönüş
  adayı tek güncel devir belgesinde yer alır.
- Test/lint/type/submission kapıları yeşildir; çalışma ağacında yalnız hedefe ait
  kasıtlı değişiklikler stage/commit edilir.
- Uzak repoya bugün push yapılmaz.

## Open questions

- “Push” GitHub push olarak yorumlanmıştır. Kaggle gönderimi ayrı bir yarışma
  işlemi olarak 27 Ağustos kotasında, güncel leaderboard okunduktan sonra yapılır.
- Birincilik garanti edilemez; kabul edilebilir tek iddia ölçülmüş skor ve açıkça
  etiketlenmiş beklentidir.
