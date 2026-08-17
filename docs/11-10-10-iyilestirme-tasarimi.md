# GridUp 10/10 İyileştirme Tasarımı

## Amaç ve başarı ölçütü

Bu çalışma, GridUp deposunu yarışma-pipeline mühendisliği açısından fail-closed,
tekrarlanabilir ve bağımsız olarak denetlenebilir hâle getirir. "10/10" görülmemiş
2026 yarışma verisinde sıralama garantisi değil; aşağıdaki mühendislik kapılarının
tamamının kanıtla geçmesi anlamına gelir:

- zamansal/OOF veri sızıntısı yoktur;
- train ve test aynı öğrenilmiş dönüşüm sözleşmesini kullanır;
- model seçimi ham resmî metrikte yapılır ve erken durdurma metriği kaydedilir;
- lag, OOF kapsamı ve submission ID sırası açık ve fail-closed sözleşmelerdir;
- benchmark bağımsız dış değerlendirme ve belirsizlik olmadan kazanan ilan etmez;
- recipe, veri, fold, kod, ortam ve artifact kimlikleri deneyle birlikte kaydedilir;
- bağımlılık, wheel, harici veri, sır ve lisans kapıları CI'da doğrulanır;
- tam test, lint, coverage, paket ve uçtan uca kontroller temizdir.

## Kabul edilen yaklaşım

Cerrahi yamalar aynı hata sınıflarının tekrarını önlemeyeceği; tam sklearn/DAG ve
konteyner göçü ise yarışma bağlamında aşırı maliyetli olduğu için sözleşme-temelli
uygulama çekirdeği seçildi.

1. Veriyle öğrenilen dönüşümler `fit(train) -> transform(train/test)` kullanır.
2. Kısmi temporal OOF kapsamı varsayılan olarak hata verir.
3. Lag gerçek satır ofsetidir; hedef türevinde `shift >= horizon` zorunludur.
4. Tahminler ters dönüştürüldükten sonra ham resmî metrikte skorlanır.
5. Ansambl kapsamı uyarı değil, gerçek skor maskesidir.
6. Submission ID dizisi sample ile sıra ve çokluk bakımından birebir eşleşir.
7. Tüm giriş noktaları immutable, fingerprint'li tek bir pipeline recipe kullanır.
8. Deney ve artifact yazımları transactional/atomiktir; tam provenance zorunludur.
9. Doğrulanmamış bağımlılık, wheel veya veri artifact'i yayımlanmaz/kurulmaz.
10. Kritik doğruluk değişiklikleri hemen fail-closed; isimsel API göçleri uyarılıdır.

## Hedef mimari

```text
discover/read -> schema + leakage gate -> resolve recipe -> fit transforms
-> transform train/test -> folds -> train/evaluate(raw metric)
-> independent model selection -> postprocess -> strict submission
-> transactional experiment/artifact manifest
```

- `recipe.py`: CV, feature, model ve yürütme politikasının tek kaynağı.
- `pipeline.py`: saf/typed uygulama aşamaları; CLI veya terminal I/O içermez.
- `models.py`: backend adaptörleri, transform-aware CV ve fold özetleri.
- `experiment.py`: provenance modeli ve transactional store sözleşmesi.
- `scripts/*.py`: yalnız CLI/adaptör; iş kurallarını çoğaltmaz.
- `security/` ve `data/sources.yml`: wheel/veri hash, lisans ve provenance kaynağı.

## Test ve yayın kapıları

- Önce her doğrulanmış kusur için başarısız regresyon testi yazılır.
- Python 3.10-3.13: clean install, import, Ruff, test ve paket doğrulama.
- Statement ve branch coverage eşikleri gerilemeye kapalıdır.
- Secret/dependency/license/provenance kontrolleri fail-closed çalışır.
- Hızlı yerel kapı hedefi üç dakika, PR CI hedefi on iki dakikadır.
- Gerçek benchmark en az altı zaman anchor'ı, eşleştirilmiş fark ve güven aralığı
  taşır; sonuç belirsizse daha basit model seçilir.

## Karar günlüğü

- Bilimsel doğruluk, kritik durumlarda geriye uyumluluktan önce gelir.
- Train-only fit varsayılandır; transductive train+test referansı açık opt-in'dir.
- Aynı OOF üzerinde optimize edilip ölçülen fark nihai kanıt sayılmaz.
- Global geliştirici ortamı kapsam dışıdır; kanıt temiz ve kilitli ortamdan gelir.
- Kanıtlanamayan yeniden dağıtım lisansında artifact pakete alınmaz.
- Ağsız Kaggle çalışma kabiliyeti korunur.

