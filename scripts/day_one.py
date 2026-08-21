"""VERI GUNU: ham dosyadan ilk submission'a tek komut.

21 Agustos'ta veri geldiginde calistirilacak ilk sey budur. Amac ilk saatte
leaderboard'da bir skor gormek -- iyi bir skor degil, CALISAN bir hat.

    python scripts/day_one.py --data data/raw --target HEDEF --id ID --metric MAE

Betik yedi asamada ilerler ve her asamada DURUP ne bulduğunu soyler. Bir
asama karar gerektiriyorsa (CV semasi, hedef tipi) onerisini yazar ve
``--yes`` verilmediyse onay bekler.

TASARIM: Her asama kendi basina anlamlidir. Sonuna kadar calismasa bile
her calisan asama bir sey ogretir -- yarisma gununde en degerli sey budur.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # kardes betikler

from epias_panel import MERKEZ_KURTARMA  # noqa: E402

from gridup import (  # noqa: E402
    build_panel,
    cross_validate,
    default_pipeline_recipe,
    environment_report,
    leakage_report,
    panel_coverage,
    postprocess_predictions,
    profile,
    read_any,
    set_global_seed,
    sniff_dialect_shared,
    suggest_scheme,
    write_submission,
    zero_baseline_score,
)
from gridup.experiment import (  # noqa: E402
    DataArtifact,
    ExperimentProvenance,
    ExperimentRecord,
)
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_maruziyet_etkilesimleri,
    add_turkish_holiday_features,
    attach_external,
    shared_origin,
)
from gridup.metrics import get_metric, optimize_threshold  # noqa: E402
from gridup.models import esik_bagimli_mi, starter_params  # noqa: E402
from gridup.panel import PANEL_FLAG_COLUMN  # noqa: E402
from gridup.pipeline import FoldPlan, runtime_recipe_fingerprint  # noqa: E402
from gridup.recipe import CVRecipe, FeatureRecipe, ModelRecipe, PipelineRecipe  # noqa: E402
from gridup.refit import (  # noqa: E402
    estimate_full_data_rounds,
    extract_best_iterations,
    fold_train_fraction,
    multi_seed_refit,
)
from gridup.stores import SQLiteExperimentStore  # noqa: E402
from gridup.turkish import (  # noqa: E402
    grup_adayini_sec,
    hizala_ilce_anahtarlari,
    join_key,
    normalize_columns,
)
from gridup.validation import (  # noqa: E402
    build_splitter,
    forecast_geometry,
    hedefi_sayisallastir,
    parse_time_series,
    purged_time_series_split,
)

ROOT = Path(__file__).resolve().parents[1]


def banner(step: str, title: str) -> None:
    print(f"\n{'=' * 78}\n{step}  {title}\n{'=' * 78}")


def confirm(question: str, *, auto: bool) -> bool:
    if auto:
        print(f"  -> {question}  [otomatik: evet]")
        return True
    answer = input(f"  -> {question} [E/h] ").strip().lower()
    return answer in ("", "e", "evet", "y", "yes")


def require_official_metric(metric: str | None) -> str:
    """Resmi metrik bilinmeden sessiz bir varsayimla model kurmayi reddet."""
    if metric is None or not metric.strip():
        raise ValueError("resmi metrik zorunlu; --metric ile acikca belirt")
    normalized = metric.strip().lower()
    get_metric(normalized)
    return normalized


#: Resmi metrik bunlardan biriyse hedef REGRESYONDUR -- profil "multiclass"
#: dese bile. Olculdu (2026-08-18 denetimi): 2024 GDZ semasinda gunluk
#: kesinti adedi 0..8 arasi -> profil 20'den az benzersiz deger gorup
#: "multiclass" dedi -> LightGBM "Multiclass objective and metrics don't
#: match" ile coktu, submission uretilmedi.
REGRESYON_METRIKLERI = frozenset({"rmse", "rmsle", "mae", "mape", "smape", "r2"})

#: Regresyon metrigi -> egitim objective'i. MAE'de L2 egitmek, benchmark'ta
#: SIFIR TABANININ altinda kaldi (lgb_l2 400 vs sifir 367 vs lgb_mae 311).
#: metrics.py:10 "MAE -> L1" der; burasi onu day_one'a uygular.
METRIK_OBJECTIVE = {
    "mae": "mae",
    "mape": "mae",
    "smape": "mae",
    "rmse": "l2",
    "rmsle": "l2",
    "r2": "l2",
}


def resolve_task(
    explicit: str | None, target_summary: dict[str, Any], metric: str | None = None
) -> str:
    """Acik kullanici secimini koru; yoksa profil tahminini kullan.

    Profil dusuk kardinaliteli sayisal hedefi "multiclass" sanabilir; resmi
    metrik bir regresyon metrigiyse bu tahmin ezilir ve neden yazilir.
    """
    inferred = target_summary.get("gorev_tahmini")
    resolved = explicit or inferred or "regression"
    if (
        explicit is None
        and metric is not None
        and metric.strip().lower() in REGRESYON_METRIKLERI
        and resolved != "regression"
    ):
        print(
            f"  NOT: profil gorevi {resolved!r} tahmin etti ama resmi metrik "
            f"{metric!r} bir regresyon metrigi -> gorev 'regression' olarak cozuldu "
            "(dusuk kardinaliteli sayim hedefi tuzagi)."
        )
        resolved = "regression"
    if resolved not in {"regression", "binary", "multiclass"}:
        raise ValueError(f"desteklenmeyen gorev tipi: {resolved}")
    return str(resolved)


#: Bilesik id sentezinde denenen ayraclar ve tarih bicimleri (2024 GDZ emsali:
#: ``unique_id = "2025-07-01-izmir-aliağa"``; docs/01:35).
_ID_AYRACLARI = ("-", "_", "|", "/", "")
_ID_TARIH_BICIMLERI = ("%Y-%m-%d", "%d.%m.%Y", "%Y%m%d", "%d-%m-%Y", "%Y/%m/%d")


def _id_parca_adaylari(test: pd.DataFrame, time_column: str | None) -> list[tuple[str, pd.Series]]:
    """Id'yi olusturabilecek parcalar: tarih (birkac bicim) + metin kolonlari (ham/kucuk)."""
    parcalar: list[tuple[str, pd.Series]] = []
    if time_column is not None and time_column in test.columns:
        zaman = pd.to_datetime(test[time_column], errors="coerce")
        for bicim in _ID_TARIH_BICIMLERI:
            parcalar.append((f"{time_column}:{bicim}", zaman.dt.strftime(bicim)))
    for kolon in test.columns:
        dtype = test[kolon].dtype
        metin = (
            pd.api.types.is_string_dtype(dtype)  # pandas 3: str/StringDtype
            or pd.api.types.is_object_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
        )
        if kolon == time_column or not metin:
            continue
        ham = test[kolon].astype(str).str.strip()
        parcalar.append((kolon, ham))
        parcalar.append((f"{kolon}:kucuk", ham.map(lambda s: s.lower())))
    return parcalar


def synthesize_id_column(
    test: pd.DataFrame, sample: pd.DataFrame, id_column: str, time_column: str | None
) -> pd.Series | None:
    """sample_submission id'si test'te YOKSA test kolonlarindan yeniden kurmayi dener.

    2024 GDZ'de id bilesik bir dizgeydi ("il-ilce-tarih" gibi) ve test'te
    ayri kolon olarak yoktu. Yontem: tarih (5 bicim) ve metin kolonlarinin
    (ham/kucuk harf) 1-3'lu siralamalari 5 ayracla birlestirilir; uretilen
    kume sample id kumesiyle BIREBIR esitse o desen kabul edilir. Esitlik
    sarti sessiz yanlis desene izin vermez. Bulunamazsa None.
    """
    from itertools import permutations

    hedef = set(sample[id_column].astype(str).str.strip())
    if len(hedef) != len(test):
        return None
    parcalar = _id_parca_adaylari(test, time_column)
    for uzunluk in (1, 2, 3):
        for kombin in permutations(parcalar, uzunluk):
            adlar = [k[0].split(":")[0] for k in kombin]
            if len(set(adlar)) != uzunluk:
                continue  # ayni kolonun iki bicimi bir arada olmaz
            for ayrac in _ID_AYRACLARI:
                aday = kombin[0][1]
                for _, seri in kombin[1:]:
                    aday = aday + ayrac + seri
                if set(aday) == hedef:
                    print(
                        f"  id '{id_column}' test'te yoktu; desen bulundu: "
                        f"{' + '.join(k[0] for k in kombin)} (ayrac {ayrac!r})"
                    )
                    return aday.rename(id_column)
    return None


#: Harici gunluk seriler: (dosya, zaman kolonu). Test blogu bu dosyalarin son
#: gununu asarsa ilgili feature ailesi YALNIZCA testte NaN olur -- CV'de
#: gorunmez (2026-08-18 denetimi, P0-10). Ilk kontrol budur.
HARICI_GUNLUK_DOSYALAR: tuple[tuple[str, str], ...] = (
    ("data/external/hava_gunluk.parquet", "tarih"),
    ("data/external/hava_saatlik_turev.parquet", "tarih"),
    ("data/external/gunes_gunluk.parquet", "tarih"),
    ("data/external/yanginlar.parquet", "tarih"),
    ("data/external/depremler.parquet", "tarih"),
)


def harici_kapsam_raporu(
    test_min: pd.Timestamp, test_max: pd.Timestamp, root: Path = ROOT
) -> list[tuple[str, pd.Timestamp | None, bool]]:
    """Her harici gunluk serinin son gununu test blogunun sonuyla kiyaslar.

    Dondurdugu liste: (dosya, son_gun, kapsiyor_mu). Dosya yoksa son_gun None.
    ``hava_gunluk`` icin tahmin (forecast koprusu) satirlarinin test
    araligindaki payi da yazilir -- model o gunlerde ERA5 degil, tahmin gorur.
    """
    print("\n  HARICI VERI KAPSAMI (test bloguna gore):")
    sonuc: list[tuple[str, pd.Timestamp | None, bool]] = []
    for gorece, zaman_kolonu in HARICI_GUNLUK_DOSYALAR:
        yol = root / gorece
        if not yol.exists():
            print(f"    - {gorece}: YOK")
            sonuc.append((gorece, None, False))
            continue
        try:
            tablo = pd.read_parquet(yol, columns=[zaman_kolonu])
        except (OSError, ValueError, KeyError) as hata:
            print(f"    - {gorece}: okunamadi ({hata})")
            sonuc.append((gorece, None, False))
            continue
        zaman = pd.to_datetime(tablo[zaman_kolonu], errors="coerce")
        son = zaman.max()
        kapsiyor = bool(pd.notna(son) and son >= test_max.normalize())
        durum = "OK" if kapsiyor else f"UYARI: {(test_max.normalize() - son).days} gun ACIK"
        print(f"    - {gorece}: son gun {son.date() if pd.notna(son) else '?'}  [{durum}]")
        if gorece.endswith("hava_gunluk.parquet"):
            try:
                bayrak = pd.read_parquet(yol, columns=[zaman_kolonu, "hava_tahmin"])
                aralik = bayrak[
                    (bayrak[zaman_kolonu] >= test_min.normalize())
                    & (bayrak[zaman_kolonu] <= test_max.normalize())
                ]
                if len(aralik):
                    pay = float(aralik["hava_tahmin"].mean())
                    print(f"      test araliginda tahmin (forecast) satiri payi: %{100 * pay:.0f}")
            except (OSError, ValueError, KeyError):
                print("      (hava_tahmin bayragi yok: fetch_weather_bridge.py calistirilmamis)")
        sonuc.append((gorece, son, kapsiyor))
    if not all(k for _, _, k in sonuc):
        print(
            "    -> Acik olan seride test satirlari NaN alir; ya kaynagi guncelle "
            "(fetch_weather_bridge.py / fetch_*.py) ya da o aileyi ufuk-kaydirmali kullan."
        )
    return sonuc


def objective_for_metric(task_type: str, metric: str) -> str | None:
    """Regresyonda resmi metrige uygun objective; digerlerinde None (varsayilan)."""
    if task_type != "regression":
        return None
    return METRIK_OBJECTIVE.get(metric.strip().lower())


def _kolonu_coz(ad: str | None, frame: pd.DataFrame) -> str | None:
    """Kullanicinin yazdigi ham kolon adini normalize edilmis ada cevirir.

    Kullanici CSV'de ne goruyorsa onu yazar ("TARIH"); ``read_any`` ise
    kolonlari normalize eder ("tarih"). Ikisini burada baglıyoruz.
    """
    if ad is None or ad in frame.columns:
        return ad
    aday = normalize_columns([ad]).get(ad)
    if aday and aday in frame.columns:
        return aday
    # Ters yon: kullanici normalize edilmis adi yazdi ama frame ham duruyor.
    for ham, normal in (frame.attrs.get("original_columns") or {}).items():
        if ad in (ham, normal):
            return normal if normal in frame.columns else ham
    return ad


#: Dosya adi kaliplari -- kelime listesi BILEREK degistirilmedi; duzeltme
#: eslesme YONTEMINDE (alt dizgi -> kelime), kelime kumesinde degil.
#: Anahtar sirasi ONEM SIRASIDIR: ayni dosya birden fazla role uyarsa
#: listede once gelen rolu kazanir.
_AD_KALIPLARI = {
    "train": ("train", "egitim"),
    "test": ("test",),
    "sample": ("sample", "submission", "ornek"),
}
_VERI_UZANTILARI = {".csv", ".parquet", ".xlsx", ".txt"}


def _ad_belirtecleri(katlanmis: str) -> set[str]:
    """Katlanmis dosya adini KELIMELERE ayirir ("gdz_train_2023" -> {gdz, train, 2023})."""
    return {parca for parca in re.split(r"[^a-z0-9]+", katlanmis) if parca}


def _adaylari_bul(
    dosyalar: list[tuple[Path, set[str], str]], needles: tuple[str, ...]
) -> tuple[list[Path], bool]:
    """``(aday_listesi, kelime_eslesmesi_mi)``.

    ONCE kelime eslesmesi denenir; hicbir dosya kelime duzeyinde uymuyorsa
    alt dizgiye DUSULUR. Bu sira kritik: ``latest_train.csv`` alt dizgi
    duzeyinde "test" icerir ama gercek ``test.csv`` varken ona hic bakilmaz.
    """
    kesin = [path for path, belirtecler, _ in dosyalar if belirtecler.intersection(needles)]
    if kesin:
        return kesin, True
    zayif = [path for path, _, ad in dosyalar if any(needle in ad for needle in needles)]
    return zayif, False


def find_files(data_dir: Path) -> dict[str, Path]:
    """train/test/sample dosyalarini ad kalibina gore bulur.

    ESLESME ONCE KELIME BAZLIDIR. Eskiden yalnizca ``"test" in name`` alt
    dizgi denetimi vardi ve ``"latest"`` kelimesi ``"test"`` iceriyor
    (olculdu: ``'test' in 'latest'`` -> True). Sonuc olculdu:
    ``['latest_train.csv', 'test.csv', 'sample_submission.csv']`` dizininde
    ``{'train': 'latest_train.csv', 'test': 'latest_train.csv'}`` -- train ve
    test AYNI dosya, gercek ``test.csv`` sessizce elendi. sample_submission
    yoksa betik 500 satirlik (olmasi gereken 120) submission uretip EXIT=0
    ile bitiyordu.

    Alt dizgi dali ``sampleSubmission.csv`` gibi kelime ayraci olmayan klasik
    Kaggle adlari icin YEDEK olarak korunur, ama kullanildiginda sessiz
    kalmaz. Ayni dosya iki role atanamaz; elenen adaylar da ekrana yazilir,
    cunku veri gununde yanlis dosyayi okumak saatlere mal olur.
    """

    # SIG olan once: data/raw/train.csv, data/raw/_prova/train.csv'den ONCE
    # gelmeli. Duz sorted() ile "_" harfi "t"den once siralanir ve alt
    # dizindeki sentetik prova dosyasi gercek dosyayi GOLGELER (olculdu:
    # 2026-08-18 denetimi, ayni ad iki yolda -> uyari bile okunamiyordu).
    # Derinlik birincil anahtar; ayni derinlikte ad sirasi.
    def _sira(path: Path) -> tuple[int, str]:
        return (len(path.relative_to(data_dir).parts), str(path).lower())

    dosyalar: list[tuple[Path, set[str], str]] = []
    for path in sorted(data_dir.rglob("*"), key=_sira):
        if path.suffix.lower() not in _VERI_UZANTILARI:
            continue
        katlanmis = join_key(path.stem)
        dosyalar.append((path, _ad_belirtecleri(katlanmis), katlanmis))

    found: dict[str, Path] = {}
    kullanilan: set[Path] = set()
    for key, needles in _AD_KALIPLARI.items():
        liste, kelime_eslesmesi = _adaylari_bul(dosyalar, needles)
        kalan = [path for path in liste if path not in kullanilan]
        if not kalan:
            continue
        found[key] = kalan[0]
        kullanilan.add(kalan[0])
        secilen = kalan[0].relative_to(data_dir).as_posix()
        if not kelime_eslesmesi:
            print(
                f"  UYARI: '{key}' icin kelime eslesmesi yok; "
                f"'{secilen}' ALT DIZGI ile secildi -- dogru dosya mi kontrol et."
            )
        if len(kalan) > 1:
            # Goreli yol yaz: ayni ad farkli dizinlerde olabilir; yalnizca
            # dosya adi yazilirsa "train.csv secildi, yok sayilan: train.csv"
            # gibi anlamsiz bir uyari cikar.
            digerleri = ", ".join(path.relative_to(data_dir).as_posix() for path in kalan[1:])
            print(
                f"  UYARI: '{key}' icin {len(kalan)} aday var. "
                f"'{secilen}' secildi, yok sayilanlar: {digerleri}"
            )
    return found


#: Hizalanmis ilce anahtarinin yazildigi kolon. Panelin KENDI grup kolonuna
#: dokunmayiz -- o CV gruplamasinda ve submission'da kullaniliyor olabilir.
HARICI_ANAHTAR_KOLONU = "_harici_ilce_key"

#: Referans anahtar kumesinin okundugu tablolar. Statik ilce tablolari tam
#: 96 kanonik anahtari tasir; gunluk tablolar yedektir.
_REFERANS_TABLOLARI = (
    "data/external/arazi_ortusu_ilce.parquet",
    "data/external/osm_altyapi_ilce.parquet",
    "data/external/hava_gunluk.parquet",
)


def _referans_anahtarlar() -> set[str]:
    """Dis tablolarin ``ilce_key`` kumesini birlestirir."""
    anahtarlar: set[str] = set()
    for gorece in _REFERANS_TABLOLARI:
        yol = ROOT / gorece
        if not yol.exists():
            continue
        try:
            tablo = pd.read_parquet(yol, columns=["ilce_key"])
        except (ValueError, KeyError):
            continue
        anahtarlar.update(tablo["ilce_key"].dropna().astype(str))
    return anahtarlar


def anahtarlari_hizala(train: pd.DataFrame, test: pd.DataFrame | None, *, group_column: str) -> str:
    """Grup kolonunu dis tablolarin anahtarina hizalar; kolon adini dondurur.

    NEDEN BU ADIM VAR (2026-08-21, ``dusmanca_prova.py`` olctu)
    -----------------------------------------------------------
    ``attach_external`` %0 eslesmede durur, %50 altinda uyarir. Gercek veri
    tam ARADAKI kor banda dustu: 96 ilcenin 91'i esledi (%94,8), yani ne
    hata ne uyari. Eslesmeyen 5 ilce EPIAS'in kendi yazimindan geliyordu
    (``BOZKURT / DENIZLI``, ``AYDIN MERKEZ``, ...) ve o 5 ilcenin BUTUN dis
    kolonlari sessizce NaN kaldi. Model bunu "bilgi yok" diye degil "bu
    ilcede orman yok" diye okur.

    Hizalama basarisiz olursa hat DURMAZ: ham kolon adiyla devam edilir ve
    ``attach_external``in kendi kapilari isini yapar. Bu adim bir guvence
    katmanidir, bir on kosul degil.

    Returns:
        ``attach_external``a verilecek kolon adi. Hizalama bir sey
        degistirmediyse ``group_column``in kendisi.
    """
    referans = _referans_anahtarlar()
    if not referans:
        print("  Hizalama atlandi: referans tablo bulunamadi.")
        return group_column

    ham = pd.concat(
        [train[group_column]] + ([test[group_column]] if test is not None else []),
        ignore_index=True,
    )
    kurtarmalar = hizala_ilce_anahtarlari(
        ham.dropna().astype(str).unique(),
        referans=referans,
        takma_adlar={ham_ad: yeni for (_, ham_ad), yeni in MERKEZ_KURTARMA.items()},
    )
    esleme = {ad: kayit.anahtar for ad, kayit in kurtarmalar.items()}

    kurtarilan = [k for k in kurtarmalar.values() if k.yontem not in {"dogrudan", "BULUNAMADI"}]
    bulunamayan = [k for k in kurtarmalar.values() if k.yontem == "BULUNAMADI"]
    print(
        f"  Anahtar hizalama: {len(kurtarmalar)} benzersiz ad, "
        f"{len(kurtarilan)} kurtarildi, {len(bulunamayan)} bulunamadi."
    )
    for kayit in kurtarilan[:10]:
        print(f"    KURTARILDI {kayit}")
    for kayit in bulunamayan[:10]:
        print(f"    BULUNAMADI {kayit.ham!r} -- bu ilcenin dis kolonlari NaN olacak")
    if len(bulunamayan) > 10:
        print(f"    ... {len(bulunamayan) - 10} bulunamayan daha")

    if not kurtarilan:
        return group_column
    train[HARICI_ANAHTAR_KOLONU] = train[group_column].astype(str).map(esleme)
    if test is not None:
        test[HARICI_ANAHTAR_KOLONU] = test[group_column].astype(str).map(esleme)
    return HARICI_ANAHTAR_KOLONU


def baseline_karsilastir(
    y: np.ndarray,
    result: Any,
    *,
    metric: str,
    zero_share: float,
    log_uzayinda: bool,
) -> float:
    """'Hep sifir' baseline'ini CV skoruyla AYNI UZAYDA ve AYNI SATIRLARDA basar.

    Eskiden baseline HAM hedeften, CV skoru ise ``log1p`` uzayindan geliyordu
    ve ikisi ard arda ayni birimmis gibi yaziliyordu. Olculdu (bu betigin
    uctan uca kosusu, --metric mae):

        'Hep sifir' baseline mae : 24.131778   (HAM uzay, TUM veri)
        CV skoru (OOF)           :  0.803790   (LOG uzay, kapsanan satirlar)
        ekranda okunan "iyilesme": 30.02 kat   <-- tamamen sahte

    Iki duzeltme birlikte gerekiyor:
    1) UZAY. ``log1p(0) == 0`` oldugu icin "hep sifir" tahmini log uzayinda da
       gecerlidir; baseline'i DONUSTURULMUS hedefe karsi olcmek yeter.
    2) KAPSAM. purged TimeSeriesSplit'te ilk donem hicbir fold'un valid
       tarafinda degildir (olculdu: kapsam %27.5). ``overall_score`` yalnizca
       kapsanan satirlarda hesaplandigi icin baseline de orada olculmeli.

    Dondurulen deger baseline skorudur (test edilebilir olsun diye).
    """
    kapsanan, _ = result.covered_predictions()
    baseline = zero_baseline_score(np.asarray(y)[kapsanan], metric=metric)
    _, buyuk_daha_iyi, _ = get_metric(metric)
    uzay = "log1p uzayinda" if log_uzayinda else "ham uzayda"
    skor = float(result.overall_score)

    print(f"\n  KARSILASTIRMA -- ikisi de {uzay}, ayni {len(kapsanan):,} OOF satirinda:")
    print(f"    'Hep sifir' baseline {metric}: {baseline:.6f}")
    print(f"    Model CV skoru       {metric}: {skor:.6f}")
    print(f"    (sifir orani %{zero_share * 100:.1f} -- modelin baseline'i GECMESI gerek)")

    gecti = skor > baseline if buyuk_daha_iyi else skor < baseline
    if baseline == 0:
        # Sifira bolme yok: oransal fark tanimsiz, mutlak farki soyleriz.
        print(f"    Baseline tam 0 -- oransal fark tanimsiz. Fark: {skor - baseline:+.6f}")
    else:
        oran = (baseline - skor) / abs(baseline) * (-1 if buyuk_daha_iyi else 1)
        print(f"    Fark: %{oran * 100:+.1f}  ->  {'GECTI' if gecti else 'GECEMEDI'}")
    if not gecti:
        print("    UYARI: Model 'hep sifir' baseline'ini GECMIYOR. Sorun modelde")
        print("    degil yaklasimdadir -- iki asamali model (two_stage) dene.")
    if log_uzayinda:
        print("    NOT: Bu sayilar log1p uzayindadir; leaderboard skoruyla")
        print("    dogrudan karsilastirilamaz (LB ham uzayda olculur).")
    return baseline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw", help="Ham veri dizini")
    parser.add_argument("--target", help="Hedef kolon adi (bilinmiyorsa profil sonrasi sorulur)")
    parser.add_argument("--id", dest="id_column", help="ID kolonu")
    parser.add_argument("--time", dest="time_column", help="Zaman kolonu")
    parser.add_argument("--group", dest="group_column", help="Grup/varlik kolonu")
    parser.add_argument("--metric", help="Resmi metrik (zorunlu; sessiz varsayim yok)")
    parser.add_argument("--task", default=None, choices=("regression", "binary", "multiclass"))
    parser.add_argument(
        "--tek-tohum",
        dest="tek_tohum",
        action="store_true",
        help="Son adimda cok tohumlu yeniden egitimi ATLA (CV fold ortalamasi kullanilir)",
    )
    parser.add_argument(
        "--tohum",
        type=int,
        default=5,
        help=(
            "Yeniden egitimde kac tohum ortalanacak (varsayilan 5). Kazanc ~1/sqrt(n) "
            "ile doyar; buyutmeden once benchmark'in 'tohum_egrisi' ciktisina bak"
        ),
    )
    parser.add_argument(
        "--harici-yok",
        dest="harici_yok",
        action="store_true",
        help="Harici veri ailelerini (hava/yangin/turizm...) BAGLAMA",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Rutin onaylari otomatik gec (KRITIK sizinti kapisi HARIC)",
    )
    parser.add_argument(
        "--sizintiyi-kabul-ediyorum",
        dest="accept_leakage",
        action="store_true",
        help="Kritik sizinti bulgusuna RAGMEN devam et. Ayri bayrak, "
        "cunku --yes 'onaylari gec' demektir, 'sizintiyi gormezden "
        "gel' demek degil.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    recipe = default_pipeline_recipe()
    set_global_seed(recipe.seed)

    # ---------------------------------------------------------------- 0
    banner("0/7", "ORTAM")
    for key, value in environment_report().items():
        print(f"  {key:<26} {value}")

    # ---------------------------------------------------------------- 1
    banner("1/7", "DOSYALARI BUL VE OKU")
    data_dir = Path(args.data)
    if not data_dir.exists():
        print(f"  HATA: {data_dir} yok. --data ile dogru dizini ver.")
        return 1

    files = find_files(data_dir)
    if "train" not in files:
        print(f"  HATA: train dosyasi bulunamadi. {data_dir} icindekiler:")
        for path in sorted(data_dir.rglob("*"))[:20]:
            print(f"    {path.name}")
        return 1

    for key, path in files.items():
        print(f"  {key:<8} {path.name}")

    # ORTAK BICIM KARARI. sniff_dialect her dosyaya BAGIMSIZ bakar ve ayni
    # kolon train'de float64, test'te str olabilir (olculdu). Metin dosyalarinin
    # basliklarini havuzlayip TEK karar veriyoruz.
    metin_yollari = [
        yol for yol in files.values() if yol.suffix.lower() in {".csv", ".tsv", ".txt", ".dat", ""}
    ]
    ortak = sniff_dialect_shared(metin_yollari) if len(metin_yollari) > 1 else None
    if ortak is not None:
        print(f"  ortak bicim: {ortak}")

    def _oku(yol: Path) -> pd.DataFrame:
        if ortak is not None and yol in metin_yollari:
            return read_any(yol, dialect=ortak)
        return read_any(yol)

    train = _oku(files["train"])
    test = _oku(files["test"]) if "test" in files else None
    sample = _oku(files["sample"]) if "sample" in files else None

    # Okuyucu kolon adlarini normalize eder ("TARIH" -> "tarih",
    # "Dagitilan Enerji (MWh)" -> "dagitilan_enerji_mwh") ama kullanici
    # dosyada GORDUGU adi yazar. Eskiden bu ikisi eslesmiyordu ve belgelenen
    # veri gunu komutu ("--time TARIH --group ILCE") ilk adimda hata veriyordu:
    #     HATA: hedef kolon belirlenemedi. Kolonlar: ['id','ilce','tarih',...]
    # Yarismanin ilk saatinde kaybedilecek en pahali dakikalar bunlar.
    for alan in ("target", "id_column", "time_column", "group_column"):
        ham = getattr(args, alan)
        cozulen = _kolonu_coz(ham, train)
        if cozulen != ham:
            print(f"  --{alan}: '{ham}' -> '{cozulen}' (normalize edilmis ad)")
        setattr(args, alan, cozulen)

    print(f"\n  train {train.shape}" + (f"   test {test.shape}" if test is not None else ""))
    if sample is not None:
        print(f"  sample_submission kolonlari: {list(sample.columns)}")
        if args.id_column is None:
            args.id_column = str(sample.columns[0])
        if args.target is None:
            args.target = str(sample.columns[-1])
            print(f"  -> sample'dan cikarildi: id={args.id_column}  hedef={args.target}")

    if args.target is None or args.target not in train.columns:
        print(f"\n  HATA: hedef kolon belirlenemedi. Kolonlar: {list(train.columns)}")
        return 1

    # HEDEF SAYISALLIGI PROFILDEN ONCE DOGRULANIR (2026-08-21, dusmanca prova).
    # Ondalik-virgullu bir dosyada NOKTA ondalikli hedef kolon sessizce ``str``
    # kalir. Eski davranista hicbir sey durmuyordu: profil, CV, feature ve 5
    # tohumlu yeniden egitim sonuna kadar kosuyor, is ancak 7/7'de ``np.mod``
    # bir metin serisine uygulanirken coküyordu -- yani tum egitim maliyeti
    # odendikten sonra, gonderim uretmeden. Yukaridaki ID dogrulamasiyla ayni
    # ders, farkli kolon.
    try:
        hedef_kayit = hedefi_sayisallastir(train[args.target], ad=args.target)
    except ValueError as error:
        print(f"\n  HATA: {error}")
        return 1
    if hedef_kayit.donusum != "gerek yok":
        print(f"  {hedef_kayit}")
        train[args.target] = hedef_kayit.deger

    # ---------------------------------------------------------------- 2
    banner("2/7", "PROFIL")
    report = profile(train, test, target=args.target)
    print(report.report())

    try:
        args.metric = require_official_metric(args.metric)
    except ValueError as error:
        print(f"\n  HATA: {error}")
        return 1
    inferred_task = report.target_summary.get("gorev_tahmini")
    target_kind = resolve_task(args.task, report.target_summary, metric=args.metric)
    if args.task is not None and inferred_task not in (None, args.task):
        print(
            f"  UYARI: profil gorevi {inferred_task!r} tahmin etti; "
            f"acik --task={args.task!r} korunuyor."
        )
    zero_share = float(report.target_summary.get("sifir_orani", 0.0) or 0.0)
    skew = float(report.target_summary.get("carpiklik", 0.0) or 0.0)

    # ---------------------------------------------------------------- 3
    banner("3/7", "SIZINTI TARAMASI")
    time_column = args.time_column or (report.time_columns[0] if report.time_columns else None)

    # ID KOLONU CV'DEN ONCE DOGRULANIR. Olculdu (2026-08-18): 2024 semasinda
    # sample'in bilesik unique_id'si test'te yoktu -> tum CV bitti, 7/7'de
    # KeyError, dosya yok. Sentez basarisizsa simdi, net mesajla dur.
    if test is not None and sample is not None and args.id_column not in test.columns:
        sentez = synthesize_id_column(test, sample, args.id_column, time_column)
        if sentez is None:
            print(
                f"\n  HATA: sample_submission id kolonu '{args.id_column}' test'te yok ve "
                f"test kolonlarindan turetilemedi.\n"
                f"  sample id ornekleri: {sample[args.id_column].head(3).tolist()}\n"
                f"  test kolonlari: {list(test.columns)}\n"
                f"  Cozum: test'e id kolonunu elle kur "
                f"(or. test['{args.id_column}'] = tarih + '-' + il + '-' + ilce) ve yeniden kos."
            )
            return 1
        test = test.assign(**{args.id_column: sentez.to_numpy()})

    findings = leakage_report(train, args.target, test=test, time_column=time_column)
    print(f"  {findings['summary']}\n")
    for severity in ("critical", "warning", "info"):
        for message in findings[severity][:6]:
            print(f"  [{severity.upper()}] {message}")

    # KRITIK SIZINTI KAPISI --yes ILE ACILMAZ.
    #
    # --yes "rutin onaylari gec" demektir (panel doldurma gibi, geri
    # alinabilir kararlar). Kritik sizinti geri alinabilir bir karar DEGIL:
    # olculdu, hedefle 1.0000 korelasyonlu bir kolon enjekte edildiginde
    # --yes'li kosu "[otomatik: evet]" yazip EXIT=0 ile submission uretti.
    # Kaggle'a gonderilen dosya LB'de coker, sebebi gorunmez.
    #
    # Bilerek devam etmek icin AYRI ve uzun bir bayrak gerekir; boylece
    # "sizintiyi kabul ettim" karari komut satirinda YAZILI kalir.
    if findings["critical"]:
        if args.accept_leakage:
            print("\n  --sizintiyi-kabul-ediyorum verildi -- KRITIK bulguya ragmen devam.")
        elif args.yes or not confirm("Kritik bulgu var. Yine de devam?", auto=False):
            print("\n  DURDURULDU. Once sizintiyi coz.")
            print("  Bilerek devam etmek istiyorsan: --sizintiyi-kabul-ediyorum")
            print("  (--yes bu kapiyi ACMAZ -- kritik sizinti rutin bir onay degildir.)")
            return 1

    # ---------------------------------------------------------------- 4
    banner("4/7", "PANEL KONTROLU")
    if time_column and args.group_column:
        coverage = panel_coverage(
            train, entity_columns=[args.group_column], time_column=time_column
        )
        ratio = coverage.get("coverage", float("nan"))
        print(f"  Beklenen satir: {coverage.get('expected_rows', 0):,.0f}")
        print(f"  Gercek satir:   {coverage.get('actual_rows', 0):,.0f}")
        print(f"  Doluluk:        %{ratio * 100:.1f}")
        if ratio < 0.95 and confirm("Panel seyrek. Sifirla doldurulsun mu?", auto=args.yes):
            # value_columns=[hedef]: aksi halde TUM sayisal kolonlar toplanir
            # ve statik kovaryatlar (nufus, kVA, id) olay sayisiyla carpilir --
            # train'de toplanmis, test'te ham deger olur (sessiz kayma).
            train = build_panel(
                train,
                entity_columns=[args.group_column],
                time_column=time_column,
                value_columns=[args.target],
            )
            print(f"  Panel kuruldu: {train.shape}")
    else:
        print("  Zaman veya grup kolonu verilmedi -- panel kontrolu atlandi.")
        print("  (--time ve --group ile belirtirsen 'olay olmadi' gunleri doldurulur)")

    # ---------------------------------------------------------------- 5
    banner("5/7", "CV SEMASI")
    suggestion = suggest_scheme(
        train,
        target=args.target,
        task_type=target_kind,
        known_time=time_column,
        known_group=args.group_column,
    )
    print(suggestion)

    # SEZILEN GRUP KOLONU BENIMSENIR (2026-08-21, dusmanca prova olctu).
    #
    # Eskiden sezim yalnizca EKRANA yaziliyordu ("Grup kolonu: il") ama
    # ``args.group_column``a geri yazilmiyordu. Iki yer bu degiskene bakar:
    # panel kurulumu ve ``attach_external``. ``--group`` elle verilmediginde
    # her ikisi de SESSIZCE atlaniyordu -- hata yok, uyari yok, kosu
    # "basarili" bitiyor. Olculdu: 27 feature (beklenen ~230), hava/arazi/
    # altyapi/turizm ailelerinin hicbiri baglanmamis.
    #
    # Hangi adayin secilecegi de TAHMIN EDILMEZ, OLCULUR: sezici gercek
    # veride 'il' dedi (5 deger) ama butun dis tablolar 'ilce' anahtarli
    # (96 deger). ``grup_adayini_sec`` adaylari referans anahtar kumesine
    # vurup en cok eslesenini secer.
    if args.group_column is None:
        referans = _referans_anahtarlar()
        adaylar = [suggestion.group_column] if suggestion.group_column else []
        adaylar += [
            kolon
            for kolon in train.columns
            if kolon not in {args.target, args.id_column, time_column, *adaylar}
            and (
                pd.api.types.is_string_dtype(train[kolon].dtype)
                or pd.api.types.is_object_dtype(train[kolon].dtype)
                or isinstance(train[kolon].dtype, pd.CategoricalDtype)
            )
            and (test is None or kolon in test.columns)
        ]
        secim = grup_adayini_sec(train, adaylar=adaylar, referans=referans or {"__yok__"})
        if secim.kolon is not None:
            args.group_column = secim.kolon
            print(f"  {secim}")
            print(
                f"  -> grup kolonu OTOMATIK secildi: {secim.kolon!r}. "
                f"Yanlissa --group ILE EZ (panel ve harici veri buna bagli)."
            )
        else:
            print("  UYARI: grup kolonu sezilemedi -- panel ve harici veri ATLANACAK.")

    horizon = 1
    bosluk_gun = 0
    ic_ice = False
    if time_column and test is not None and time_column in test.columns:
        test_times = parse_time_series(test[time_column])
        # UFUK = train'in son gunu -> test blogunun son gunu; AMBARGO = aradaki
        # verisiz bosluk. Eski formul (test.max - test.min + 1) boslugu yok
        # sayiyordu: 10 gun boslukta CV lag'i 20 gun, test lag'i 30 gun bayatti
        # (olculdu, 2026-08-18 denetimi P1-3).
        geometri = forecast_geometry(parse_time_series(train[time_column]), test_times)
        horizon, bosluk_gun = geometri.horizon_days, geometri.gap_days
        ic_ice = geometri.interleaved
        print(f"\n  {geometri.summary()}")
        if ic_ice:
            # IC ICE (rastgele) BOLME: test tarihleri train'in ICINE giriyor.
            # Zaman-ileri sema uygulanamaz -- eski surumde ufuk 455 gun cikip
            # "hicbir fold uretilemedi" ile duruyordu (olculdu 2026-08-18, P1-8).
            print(
                "  -> test tarihleri train ICINE giriyor: zaman-ileri sema UYGULANAMAZ. "
                "Grup varsa GroupKFold, yoksa KFold kullanilacak."
            )
        else:
            print("  -> lag/rolling feature'lari bu ufka gore kaydirilmali; CV ambargosu = bosluk")
        harici_kapsam_raporu(test_times.min(), test_times.max())

    splitter_name: str
    test_span_days: int | None = None
    embargo_days = 0
    if time_column and ic_ice:
        # Zaman ekseni var ama bolme zamansal DEGIL: grup varsa gruba gore
        # (ayni ilce iki tarafta olmasin), yoksa duz KFold.
        if args.group_column:
            splitter_name = "GroupKFold"
            splitter = build_splitter("GroupKFold", n_splits=recipe.cv.n_splits)
            folds = list(splitter.split(train, groups=train[args.group_column]))
        else:
            scheme = "StratifiedKFold" if target_kind != "regression" else "KFold"
            splitter_name = scheme
            splitter = build_splitter(scheme, n_splits=recipe.cv.n_splits, seed=recipe.seed)
            stratify = train[args.target] if target_kind != "regression" else None
            folds = list(splitter.split(train, stratify))
        print(f"  Sema: {splitter_name} (ic ice test bolmesi)")
    elif time_column:
        train[time_column] = parse_time_series(train[time_column])
        embargo = pd.Timedelta(days=bosluk_gun)
        # DOGRULAMA PENCERESI = TEST BLOGU UZUNLUGU.
        #
        # 2023 GDZ birincisi TimeSeriesSplit(n_splits=3, test_size=744)
        # kullandi -- 744 saat, yani test blogunun TAM boyu. CV, tahmin
        # edilecek ufku birebir taklit etmelidir.
        #
        # Satir sayisina gore esit bolme PANEL veride yanlis pencere uretir:
        # 20 ilcelik bir "ay" 620 satirdir ve fold uzunlugu veri yogunluguna
        # gore kayar. test_span zamana gore boler, satira gore degil.
        test_span = pd.Timedelta(days=horizon) if horizon else None
        splitter_name = "purged_time_series"
        test_span_days = horizon
        embargo_days = bosluk_gun
        folds = purged_time_series_split(
            train[time_column],
            n_splits=recipe.cv.n_splits,
            embargo=embargo,
            test_span=test_span,
        )
    elif args.group_column:
        splitter_name = "GroupKFold"
        splitter = build_splitter("GroupKFold", n_splits=recipe.cv.n_splits)
        folds = list(splitter.split(train, groups=train[args.group_column]))
    else:
        scheme = "StratifiedKFold" if target_kind != "regression" else "KFold"
        splitter_name = scheme
        splitter = build_splitter(scheme, n_splits=recipe.cv.n_splits, seed=recipe.seed)
        stratify = train[args.target] if target_kind != "regression" else None
        folds = list(splitter.split(train, stratify))

    for index, (tr, va) in enumerate(folds, start=1):
        print(f"    fold {index}: train={len(tr):>8,}  valid={len(va):>8,}")

    # ---------------------------------------------------------------- 6
    banner("6/7", "FEATURE + BASELINE")
    origin = None
    if time_column and test is not None:
        test[time_column] = parse_time_series(test[time_column])
        origin = shared_origin(train, test, time_column=time_column)
        print(f"  Ortak zaman baslangici: {origin.date()}")

    def build_base(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        if time_column:
            out = add_calendar_features(out, time_column, include_year=False, origin=origin)
            out = add_turkish_holiday_features(out, time_column)
        return out

    train_features = build_base(train)
    test_features = build_base(test) if test is not None else None

    # HARICI VERI -- tek kapi (features.external.attach_external). Panel
    # anahtari ilce ise 12 aile (hava, hava kalitesi, CAPE, nem/toprak, gunes,
    # yangin, deprem, turizm x2, izsu, EPIAS) ayni cagriyla baglanir; eksik
    # kaynak SESSIZ NaN degil raporlanan atlamadir, %0 eslesme HATA verir.
    # --harici-yok ile kapatilir (sema tanimadiginda zaman kaybettirmesin).
    if not args.harici_yok and args.group_column and time_column:
        try:
            harici_anahtar = anahtarlari_hizala(
                train_features, test_features, group_column=args.group_column
            )
            ek_train = attach_external(
                train_features,
                key_column=harici_anahtar,
                time_column=time_column,
                horizon=horizon,
                root=ROOT,
            )
            for satir in ek_train.summary().splitlines():
                print(f"  {satir}")
            train_features = ek_train.frame
            if test_features is not None:
                ek_test = attach_external(
                    test_features,
                    key_column=harici_anahtar,
                    time_column=time_column,
                    horizon=horizon,
                    families=list(ek_train.families),
                    root=ROOT,
                )
                test_features = ek_test.frame

            # MARUZIYET ETKILESIMLERI -- dis veri BAGLANDIKTAN sonra, cunku
            # girdileri iki ayri aileden gelir (hava + arazi ortusu).
            #
            # Fizik: ruzgar hattin kendisini nadiren koparir, AGACI devirir.
            # Agac ortusu bu yuzden bir CARPANDIR. Yaprakli agac daha cok
            # ruzgar tutar, islak toprak koku gevsetir (NHESS 2023: ayni
            # ruzgarda 3-4x / 2-3x / birlikte 4-5x). GBDT bunu ogrenebilir
            # ama tam da onem tasiyan gunlerde -- siddetli firtinalarda --
            # ornek seyrektir. Fizik biliniyorsa acikca vermek ucuzdur.
            mar_train = add_maruziyet_etkilesimleri(
                train_features, time_column=time_column, key_column=harici_anahtar
            )
            for satir in mar_train.ozet().splitlines():
                print(f"  {satir}")
            train_features = mar_train.frame
            if test_features is not None:
                test_features = add_maruziyet_etkilesimleri(
                    test_features, time_column=time_column, key_column=harici_anahtar
                ).frame
        except (ValueError, KeyError, RuntimeError) as hata:
            print(f"  UYARI: harici veri baglanamadi ({hata}); harici kolonsuz devam ediliyor.")
    # Global frekans sayimi CV'den once yapilirsa erken temporal fold'lar
    # validation doneminin kategori dagilimini gorur. Fold-ici transformer
    # kurulana kadar day-one bu aileyi bilincli olarak kullanmaz.

    # PANEL_FLAG_COLUMN feature OLAMAZ: fill_value=0 iken hedefin sifir
    # olmasiyla birebir ayni seydir (olculdu: %100 ortusme, Spearman -0.9810).
    # Test kumesinde yoksa zaten elenirdi, ama test verilmediginde listeye
    # giriyordu -- adiyla dislamak tek guvenilir yol.
    drop = {args.target, args.id_column, time_column, PANEL_FLAG_COLUMN} - {None}
    columns = [
        c
        for c in train_features.columns
        if c not in drop and (test_features is None or c in test_features.columns)
    ]
    print(f"  {len(columns)} feature")

    y = train_features[args.target].to_numpy()
    use_log = target_kind == "regression" and args.metric.lower() == "rmsle"
    target_transform = "log1p" if use_log else None
    if use_log:
        print(
            f"  Hedef carpikligi {skew:.2f} -> model log1p uzayinda egitilecek; "
            f"{args.metric} HAM hedef uzayinda olculecek"
        )

    objective = objective_for_metric(target_kind, args.metric)
    params = starter_params("lightgbm", target_kind, objective=objective)
    if objective is not None:
        print(f"  objective: {params['objective']} (resmi metrik {args.metric} icin)")
    result = cross_validate(
        train_features[columns],
        y,
        folds,
        kind="lightgbm",
        task_type=target_kind,
        metric=args.metric,
        params=params,
        test=test_features[columns] if test_features is not None else None,
        target_transform=target_transform,
        early_stopping_metric=args.metric,
        early_stopping_rounds=200,
    )
    print("\n" + result.summary())

    if target_kind == "regression" and zero_share > 0.4:
        baseline_karsilastir(
            y, result, metric=args.metric, zero_share=zero_share, log_uzayinda=False
        )

    # ---------------------------------------------------------------- 7
    banner("7/7", "SUBMISSION")
    if test_features is None or result.test_predictions is None:
        print("  Test kumesi yok -- submission uretilmedi.")
        return 0

    predictions = result.test_predictions

    # 5-TOHUMLU YENIDEN EGITIM (varsayilan). CV test tahmini fold modellerinin
    # duz ortalamasidir: en eski fold'un modeli, test'e en yakin fold'unkiyle
    # ayni agirligi alir. multi_seed_refit TUM veriyle egitir ve tohum
    # ortalamasi alir -- benchmark'ta olculdu: catboost 5 tohumda 301.21-304.80
    # (yayilim 1.24), ortalamasi 302.22 = tekil ortalamadan 0.90 daha iyi,
    # YAPISAL YANLILIK OLMADAN (harmanin aksine; harman yuvalanmis kontrolde
    # gecmedi). Tur sayisi CV'nin erken durdurmasindan devralinir.
    # --tek-tohum ile kapatilir (hizli tur icin).
    if not args.tek_tohum:
        try:
            turlar = estimate_full_data_rounds(
                extract_best_iterations(result.models),
                n_folds=len(folds),
                mean_train_fraction=fold_train_fraction(folds, n_rows=len(train_features)),
            )
            refit = multi_seed_refit(
                train_features[columns],
                y,
                test_features[columns],
                kind="lightgbm",
                params=params,
                n_estimators=turlar,
                seeds=tuple(range(args.tohum)),
                sample_weight=None,
                target_transform=target_transform,
                verbose=False,
            )
            print(
                f"  {args.tohum} tohumlu yeniden egitim: {turlar} tur, "
                f"{refit.summary().splitlines()[0]}"
            )
            predictions = refit.predictions
        except (ValueError, RuntimeError) as hata:
            print(f"  UYARI: yeniden egitim basarisiz ({hata}); CV fold ortalamasi kullanildi.")

    # ESIK BAGIMLI METRIK -> ESIGI OOF UZERINDE OPTIMIZE ET (2026-08-21).
    #
    # F1/precision/recall bir KARAR ESIGI secildikten sonra tanimlidir. Model
    # olasilik uretir; 0,5 esigi yalnizca siniflar dengeliyse VE model
    # kalibreyse dogrudur. Bu panelde gunlerin ~%65'i sifir -- yani 0,5 ile
    # gondermek olculebilir bir kayiptir.
    #
    # ``optimize_threshold`` depoda ZATEN VARDI ama hicbir yerden
    # cagrilmiyordu. GDZ'22 Case-1 (ayni problem) metrik olarak F1
    # kullanmisti; o senaryo gelirse bu blok devreye girer.
    #
    # KRITIK: esik FOLD-DISI tahminlerde aranir, egitim tahminlerinde degil --
    # aksi halde esik de asiri uyum yapar. ``covered_predictions()`` yalnizca
    # gercekten bir fold'un valid tarafinda olan satirlari verir; kapsanmayan
    # satirlarda oof SIFIRDIR ve onlari da katmak esigi asagi ceker.
    esik = None
    if esik_bagimli_mi(args.metric) and target_kind != "regression":
        kapsam = result.oof_covered
        if kapsam.any():
            secim = optimize_threshold(
                y[kapsam], result.oof_predictions[kapsam], metric=args.metric
            )
            esik = float(secim["best_threshold"])
            print(
                f"  Esik optimizasyonu (OOF, {int(kapsam.sum()):,} satir): "
                f"esik={esik:.3f}  {args.metric}={secim['best_score']:.4f} "
                f"(0,5'te {secim['score_at_half']:.4f})"
            )
            predictions = (predictions >= esik).astype(int)
        else:
            print("  UYARI: OOF kapsami bos -- esik optimize edilemedi, 0,5 kullanildi.")
            predictions = (predictions >= 0.5).astype(int)

    predictions = postprocess_predictions(
        predictions,
        round_to_integer=(
            args.metric == "mae"
            and float(np.mod(train_features[args.target].dropna(), 1).max()) == 0.0
        ),
        clip_min=0.0,
    )

    path = write_submission(
        test_features[args.id_column].to_numpy(),
        predictions,
        ROOT / "submissions" / "gun1_baseline.csv",
        sample=sample,
        id_column=args.id_column,
        target_column=args.target,
        # Test sirasi sample sirasindan farkliysa hizala; id'ler tekilse
        # guvenlidir (yardimci tekrarli id'de zaten reddeder). Olculdu:
        # karisik sample ile "ID sirasi uyusmuyor" hatasi, EXIT=1, dosya yok.
        align_to_sample=sample is not None,
    )

    fold_plan = FoldPlan.from_folds(folds, n_rows=len(train_features))
    run_recipe = PipelineRecipe(
        seed=recipe.seed,
        cv=CVRecipe(
            n_splits=len(folds),
            splitter=splitter_name,
            test_span_days=test_span_days,
            embargo_days=embargo_days,
        ),
        features=FeatureRecipe(
            horizon=horizon,
            target_shifts=(),
            rolling_windows=(),
            families=("calendar", "holiday") if time_column else (),
        ),
        model=ModelRecipe(
            kind="lightgbm",
            objective=str(params.get("objective", target_kind)),
            metric=args.metric,
            early_stopping_metric=args.metric,
            n_estimators=int(params.get("n_estimators", 2000)),
            early_stopping_rounds=200,
        ),
        execution=recipe.execution,
    )
    run_fingerprint = runtime_recipe_fingerprint(
        run_recipe.to_dict(),
        target_transform=target_transform,
        splitter=splitter_name,
        estimator="lightgbm",
        n_estimators=int(params.get("n_estimators", 2000)),
        early_stopping_rounds=200,
    )
    artifacts = [
        DataArtifact.from_path(files[role]) for role in ("train", "test", "sample") if role in files
    ]
    artifacts.append(DataArtifact.from_path(path))
    provenance = ExperimentProvenance.capture(
        recipe_fingerprint=run_fingerprint,
        data_artifacts=artifacts,
        feature_names=columns,
        fold_fingerprint=fold_plan.fingerprint,
    )
    record = SQLiteExperimentStore(ROOT / "experiments" / "experiments.db").add(
        ExperimentRecord(
            name=f"day_one_lightgbm_{run_fingerprint[:8]}",
            cv_score=result.overall_score,
            metric=args.metric,
            model_kind="lightgbm",
            n_features=len(columns),
            fold_scores=list(result.fold_scores),
            params=dict(params),
            features=list(columns),
            notes=f"day_one; score_space={result.score_space}; horizon={horizon}",
            submission_path=str(path),
            provenance=provenance,
        )
    )
    print(f"  Deney kaniti: experiments/experiments.db  run_id={record.run_id}")

    elapsed = time.perf_counter() - started
    banner("TAMAM", f"{elapsed / 60:.1f} dakika")
    print(f"  Submission: {path}")
    print("\n  SIRADAKI ADIMLAR:")
    print("   1. Bu dosyayi Kaggle'a gonder -- format dogru mu gor")
    print(f"   2. LB skorunu run_id={record.run_id} icin SQLite deposuna yaz")
    print("   3. adversarial_validation ile train/test kaymasini olc")
    print(f"   4. Ufuk-farkindalikli lag/rolling ekle (horizon={horizon})")
    print("   5. Hava verisini birlestir (data/external/hava_gunluk.parquet)")
    print("   6. Komsu ilce sinyali (data/reference/ilceler_gdz_adm.parquet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
