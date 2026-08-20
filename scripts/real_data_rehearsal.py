"""GERCEK GDZ VERISIYLE TAM PROVA.

NEDEN BU BETIK VAR
------------------
Bugune kadar her sey SENTETIK veride dogrulandi. Sentetik veri, yazan kisinin
aklindaki hatalari icerir -- akli almadigi hatalari icermez. Bu betik hatti
GERCEK bir GDZ kesinti kaydi uzerinde kosar:

    68.257 kesinti kaydi, 2021-05-08 .. 2022-08-22, Izmir + Manisa, 47 ilce
    kaynak: kaggle.com/datasets/tmlalper/manisa-izmir-plansiz-elektrik-kesintileri
    (distributioncompanyname = GDZ_EDAS)

Bu, 2026 yarismasinin bekledigimiz veri sekliyle AYNI ailedendir: olay kaydi,
saat damgali, Turkce ilce adlari, Turkce ariza sebepleri.

ILK KOSUDA BULDUKLARI (ikisi de gercek, sentetik veride cikmamisti)
-------------------------------------------------------------------
1. 'Koprubasi / Manisa' niteleyici eki -- referans tablosu yalin 'Koprubasi'
   tutuyor, join 284 satiri sessizce dusuruyordu. strip_qualifier eklendi.
2. Olay kaydi -> panel donusumu: 68.257 kaydin gunluk panele oturmasi
   saat damgasinin izgaraya cekilmesini GEREKTIRIYOR (tur 5 duzeltmesi).

KULLANIM
    python scripts/real_data_rehearsal.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup import (  # noqa: E402
    build_panel,
    cross_validate,
    leakage_report,
    panel_coverage,
    read_table,
    set_global_seed,
    suggest_scheme,
)
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_turkish_holiday_features,
)
from gridup.features.external import attach_external  # noqa: E402
from gridup.features.outage_reason import reason_family_report  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.panel import PANEL_FLAG_COLUMN  # noqa: E402
from gridup.turkish import diagnose_join, join_key, strip_qualifier  # noqa: E402
from gridup.validation import purged_time_series_split  # noqa: E402

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "data" / "prior" / "ayna" / "MANISA_IZMIR_PLANSIZ_KESINTILER.csv"
REFERANS = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
HAVA = KOK / "data" / "external" / "hava_gunluk.parquet"

HEDEF = "kesinti_dk"
ZAMAN = "gun"
GRUP = "ilce_key"


def basamak(no: str, baslik: str) -> None:
    print(f"\n{'=' * 78}\n{no}  {baslik}\n{'=' * 78}")


def main() -> int:
    if not VERI.exists():
        print(f"HATA: {VERI} yok.")
        print(
            "Indir: kaggle datasets download -d "
            "tmlalper/manisa-izmir-plansiz-elektrik-kesintileri --unzip"
        )
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()

    # ---------------------------------------------------------------- 1
    basamak("1/7", "OKU VE HEDEFI KUR")
    ham = read_table(VERI, verbose=True)
    print(f"  {len(ham):,} kesinti kaydi, {ham.shape[1]} kolon")

    # Kesinti SURESI hedefimiz: bitis - baslangic (dakika).
    bas = pd.to_datetime(ham["starttime"], utc=True, format="mixed")
    bit = pd.to_datetime(ham["endtime"], utc=True, format="mixed")
    ham[HEDEF] = (bit - bas).dt.total_seconds() / 60.0
    ham[ZAMAN] = (
        pd.to_datetime(ham["date"], utc=True, format="mixed")
        .dt.tz_convert("Europe/Istanbul")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    negatif = int((ham[HEDEF] < 0).sum())
    if negatif:
        print(f"  UYARI: {negatif} kayitta bitis < baslangic -- disariya aliniyor")
        ham = ham[ham[HEDEF] >= 0]
    print(
        f"  hedef '{HEDEF}': medyan={ham[HEDEF].median():.0f} dk  "
        f"max={ham[HEDEF].max():,.0f} dk ({ham[HEDEF].max() / 1440:.1f} gun)"
    )

    # ---------------------------------------------------------------- 2
    basamak("2/7", "TURKCE JOIN -- referans tablosuyla eslesme")
    ref = pd.read_parquet(REFERANS)
    tani = diagnose_join(ham["ilce"].unique(), ref["ilce"].unique())
    print(f"  veri {tani['left_unique']} ilce, referans {tani['right_unique']} ilce")
    print(f"  normalize eslesme : {tani['normalized_matched']}")
    print(f"  eslesmeyen        : {tani['left_only']}")
    print(f"  ek atilinca kurtulan: {tani['qualifier_recoverable']}")

    ham[GRUP] = ham["ilce"].map(lambda x: join_key(strip_qualifier(str(x))))
    ref_key = set(ref["ilce"].map(join_key))
    eslesen = int(ham[GRUP].isin(ref_key).sum())
    print(f"  -> eslesecek satir: {eslesen:,}/{len(ham):,} (%{eslesen / len(ham) * 100:.1f})")

    # ---------------------------------------------------------------- 3
    basamak("3/7", "ARIZA SEBEBI TAKSONOMISI")
    rapor = reason_family_report(ham["reason"])
    print(rapor.head(10).to_string(index=False))

    # ---------------------------------------------------------------- 4
    basamak("4/7", "PANEL -- olay kaydi -> ilce x gun")
    kapsam = panel_coverage(ham, entity_columns=[GRUP], time_column=ZAMAN)
    print(
        f"  beklenen {kapsam['expected_rows']:,.0f}  gercek {kapsam['actual_rows']:,.0f}"
        f"  doluluk %{kapsam['coverage'] * 100:.1f}"
    )
    if kapsam["coverage"] > 1.0:
        print("  HATA: doluluk %100'u asamaz -- izgaraya oturtma bozuk.")
        return 1

    panel = build_panel(
        ham,
        entity_columns=[GRUP],
        time_column=ZAMAN,
        value_columns=[HEDEF, "effectedsubscribers", "hourlyloadavg"],
        verbose=True,
    )
    print(f"  panel: {panel.shape}")
    korunan = panel[HEDEF].sum() / ham[HEDEF].sum()
    print(f"  hedef kutlesi korundu mu: %{korunan * 100:.2f}  (100 olmali)")
    if abs(korunan - 1.0) > 0.001:
        print("  HATA: panel hedef kutlesini kaybetti.")
        return 1

    # ---------------------------------------------------------------- 5
    basamak("5/7", "SIZINTI TARAMASI + CV SEMASI")
    bulgular = leakage_report(panel, HEDEF, time_column=ZAMAN)
    print(f"  {bulgular['summary']}")
    for seviye in ("critical", "warning"):
        for mesaj in bulgular[seviye][:4]:
            print(f"  [{seviye.upper()}] {mesaj[:110]}")

    oneri = suggest_scheme(panel, target=HEDEF, known_time=ZAMAN, known_group=GRUP)
    print(f"\n  {oneri.scheme}  (grup={oneri.group_column}, zaman={oneri.time_column})")

    ufuk = 31
    folds = purged_time_series_split(
        panel[ZAMAN],
        embargo=pd.Timedelta(days=max(ufuk, 30)),
        n_splits=4,
        test_span=pd.Timedelta(days=ufuk),
        verbose=True,
    )
    for i, (tr, va) in enumerate(folds, start=1):
        print(f"    fold {i}: train={len(tr):>7,}  valid={len(va):>6,}")

    # ---------------------------------------------------------------- 6
    basamak("6/7", "FEATURE + HAVA BIRLESTIRME")
    ozellik = add_calendar_features(panel, ZAMAN, include_year=False)
    ozellik = add_turkish_holiday_features(ozellik, ZAMAN)

    # HARICI VERI -- ``attach_external`` UZERINDEN (2026-08-21 duzeltmesi).
    #
    # KOR NOKTA: bu betik havayi ELLE merge ediyordu ve ``attach_external``i
    # hic cagirmiyordu. Yani "gercek veri provasi gecti" demek, gun-1'de
    # gercekten kosacak olan harici veri yolunun test edildigi anlamina
    # GELMIYORDU -- day_one.py attach_external kullanir, bu betik kullanmazdi.
    # Ayni gun eklenen iki statik aile (arazi_ortusu, osm_altyapi) provadan
    # gecmis GORUNUP hic denenmemisti. Prova artik gun-1 ile ayni kapiyi kullanir.
    oncesi = len(ozellik)
    ek = attach_external(
        ozellik,
        key_column=GRUP,
        time_column=ZAMAN,
        horizon=ufuk,
        root=KOK,
    )
    ozellik = ek.frame
    assert len(ozellik) == oncesi, "attach_external satir sayisini degistirdi"
    print(f"  {ek.summary()}")
    if ek.skipped:
        print(f"  ATLANAN aile: {', '.join(ek.skipped)}")
    # Dagilim/frekans ozellikleri temporal CV'den once tum panelde fit edilmez.
    # Fold-ici encoder entegrasyonu gelene kadar bu aile fail-closed kapali.

    # SIZINTI DUVARI (cekismeli denetim yakaladi): ham olay kaydinin TUM
    # kolonlari ayni gunun bilgisidir ve feature olamaz. Ilk surum id,
    # effectedsubscribers ve hourlyloadavg'i feature aliyordu:
    #   * effectedsubscribers/hourlyloadavg -> AYNI GUNUN kesinti bilgisi
    #   * id -> build_panel'in 'first' tasidigi kolon; NaN deseni _dolduruldu
    #     bayraginin birebir kopyasi (benchmark'ta olculdu: uyum 1.000000)
    # O yuzden ilk provanin MAE=266.60 sayisi IYIMSERDI; duzeltilmis sayi
    # asagida her kosuda yeniden olculur ve yalnizca kendi baseline'iyla
    # kiyaslanir.
    ham_kolonlar = {
        "id",
        "il",
        "ilce",
        "date",
        "starttime",
        "endtime",
        "reason",
        "effectedsubscribers",
        "hourlyloadavg",
        "effectedneighbourhoods",
        "distributioncompanyname",
    }
    dus = {HEDEF, ZAMAN, GRUP, PANEL_FLAG_COLUMN, "tarih", "ilce_key", *ham_kolonlar}
    kolonlar = [
        c for c in ozellik.columns if c not in dus and pd.api.types.is_numeric_dtype(ozellik[c])
    ]
    print(f"  {len(kolonlar)} sayisal feature (ham olay kolonlari sizinti duvarinin arkasinda)")

    # ---------------------------------------------------------------- 7
    basamak("7/7", "MODEL")
    y = ozellik[HEDEF].to_numpy()
    sonuc = cross_validate(
        ozellik[kolonlar],
        y,
        folds,
        kind="lightgbm",
        metric="mae",
        params=starter_params("lightgbm", "regression", objective="mae"),
        verbose=False,
    )
    print(sonuc.summary())
    kapsanan, _ = sonuc.covered_predictions()
    sifir_baseline = float(np.abs(y[kapsanan]).mean())
    print(f"\n  'hep sifir' baseline mae : {sifir_baseline:.4f}")
    print(f"  model mae                : {sonuc.overall_score:.4f}")
    kazanc = (1 - sonuc.overall_score / sifir_baseline) * 100
    print(f"  -> baseline'a gore %{kazanc:.1f} daha az sapma")

    print(f"\n{'=' * 78}")
    print(f"PROVA TAMAM  ({time.perf_counter() - baslangic:.0f} sn)")
    print(f"{'=' * 78}")
    print("  Hat GERCEK GDZ verisinde uctan uca calisti.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
