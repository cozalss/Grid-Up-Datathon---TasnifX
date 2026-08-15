"""GERCEK GDZ VERISINDE FEATURE AILESI ABLASYONU.

NEDEN BU BETIK VAR
------------------
Yarisma gunu her feature ailesine zaman yetmeyecek. Bu betik, 68.257 gercek
GDZ kesinti kaydinda (Izmir+Manisa, 47 ilce, 2021-05..2022-08) her ailenin
GERCEK katkisini olcer ve veri gununun oncelik listesini uretir: hangi aile
sinyal tasiyor, hangisi susleme.

Feature onemi (gain) bunu SOYLEYEMEZ: birbirinin yerini tutan korele kolonlar
tek tek "onemli" gorunur ama biri silinince digeri isi devralir. Bu yuzden
olcu leave-one-group-out'tur: aile TUMUYLE silinir, ayni fold'larla MAE
yeniden olculur. delta = mae_ailesiz - mae_tam; pozitif delta = aile katki
veriyor.

VERI RECETESI scripts/real_data_rehearsal.py ile BIREBIR aynidir (kanitlanmis):
hedef kesinti_dk = endtime - starttime (dakika, negatifler disari), gun
Istanbul gunune normalize, ilce_key = join_key(strip_qualifier(ilce)),
panel = 47 ilce x gun izgarasi, fold'lar = purged split (embargo=31 gun,
4 fold x 31 gun test).

SIZINTI KURALLARI
-----------------
* effectedsubscribers / hourlyloadavg AYNI GUNUN bilgisidir -- feature listesine
  GIREMEZ (ilk -- sonradan duzeltilen -- provanin 266.60 MAE'si bu iki kolonu
  iceriyordu; buradaki tam model
  bilerek onlarsiz olculur, o yuzden sayilar birebir karsilastirilamaz).
* PANEL_FLAG_COLUMN (_dolduruldu) feature olamaz.
* Hedef turevli her sey (lag/rolling/komsu) horizon=31 kaydirmali.

OLCULEN (bu makinede, 2026-08-15)
---------------------------------
  panel 22.184 satir (47 ilce x 472 gun), tam model 76 feature
  tam MAE=313.64  sifir-baseline=366.97  fold_std=94.31 (GURULTULU, %30)
  siralama: lag(+22.34) > hava(+2.47) > komsu(+0.18) > frekans(+0.00)
            > takvim(-0.19) > tatil(-4.66) > gunes(-5.03)
  toplam sure 39 sn

  Okuma notlari:
  * lag TEK BASINA tum diger ailelerin toplamindan buyuk katki veriyor --
    veri gununde ILK kurulacak aile budur.
  * frekans deltasi TAM sifir: panel tam izgara oldugu icin her ilce ayni
    satir sayisina sahip, kodlama sabit 1/47 cikiyor -- sifir bilgi.
  * tatil ve gunes NEGATIF: silinince MAE dusuyor. fold_std=94 gurultusunde
    kucuk deltalar kesin hukum degil, ama bu ikisi ilk elenecek adaylar.
  * Ilk provanin 266.60 MAE'si ile karsilastirilamaz: o kosu ayni-gun sizintili
    effectedsubscribers/hourlyloadavg kolonlarini feature almisti.

KULLANIM
    python scripts/ablation_gercek.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gridup import (  # noqa: E402
    build_panel,
    cross_validate,
    read_table,
    set_global_seed,
)
from gridup.ablation import FeatureGroup, leave_one_group_out  # noqa: E402
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_cyclical_features,
    add_frequency_encoding,
    add_lag_features,
    add_neighbour_target_lag,
    add_physical_derivatives,
    add_rolling_features,
    add_turkish_holiday_features,
    nearest_neighbours,
)
from gridup.features.temporal import add_ramadan_features  # noqa: E402
from gridup.models import starter_params  # noqa: E402
from gridup.panel import PANEL_FLAG_COLUMN  # noqa: E402
from gridup.turkish import join_key, strip_qualifier  # noqa: E402
from gridup.validation import purged_time_series_split  # noqa: E402

KOK = Path(__file__).resolve().parents[1]
VERI = KOK / "data" / "prior" / "ayna" / "MANISA_IZMIR_PLANSIZ_KESINTILER.csv"
REFERANS = KOK / "data" / "reference" / "ilceler_gdz_adm.parquet"
HAVA = KOK / "data" / "external" / "hava_gunluk.parquet"
GUNES = KOK / "data" / "external" / "gunes_gunluk.parquet"
CIKTI = KOK / "experiments" / "ablasyon_gercek.json"

HEDEF = "kesinti_dk"
ZAMAN = "gun"
GRUP = "ilce_key"

#: Test blogu 31 gunluk -- lag/rolling/komsu HEPSI bu ufukla kaydirilir.
UFUK = 31

#: Ayni gunun bilgisi: tahmin aninda BILINMEZ, feature listesine giremez.
AYNI_GUN_SIZINTISI = ("effectedsubscribers", "hourlyloadavg")

#: Risk katmani ablasyon tablosunda rapor icindir (LOGO hesabini degistirmez):
#: takvim/tatil/lag/frekans yarisma verisinin kendisinden turer (cekirdek),
#: hava/gunes dis kaynaktan gelir (harici), komsu kanitlanmamis (deneysel).
RISKLER = {
    "takvim": "cekirdek", "tatil": "cekirdek", "lag": "cekirdek",
    "frekans": "cekirdek", "hava": "harici", "gunes": "harici",
    "komsu": "deneysel",
}


def panel_kur() -> pd.DataFrame:
    """Olay kaydini gunluk ilce paneline cevirir -- provayla ayni recete."""
    ham = read_table(VERI, verbose=False)

    bas = pd.to_datetime(ham["starttime"], utc=True, format="mixed")
    bit = pd.to_datetime(ham["endtime"], utc=True, format="mixed")
    ham[HEDEF] = (bit - bas).dt.total_seconds() / 60.0
    ham[ZAMAN] = (
        pd.to_datetime(ham["date"], utc=True, format="mixed")
        .dt.tz_convert("Europe/Istanbul")
        .dt.tz_localize(None)
        .dt.normalize()
    )
    # Savunma filtresi: bitis < baslangic olan kayitlar veri hatasidir.
    # BU veride 0 cikti (denetimde olculdu: 68.257 kaydin tamami temiz);
    # filtre gelecekte gelebilecek bozuk kayitlar icin durur.
    ham = ham[ham[HEDEF] >= 0]
    # 'Koprubasi / Manisa' niteleyici kurtarmasi (provada 284 satir kurtardi).
    ham[GRUP] = ham["ilce"].map(lambda x: join_key(strip_qualifier(str(x))))

    panel = build_panel(
        ham, entity_columns=[GRUP], time_column=ZAMAN,
        value_columns=[HEDEF, *AYNI_GUN_SIZINTISI], verbose=False,
    )
    korunan = panel[HEDEF].sum() / ham[HEDEF].sum()
    if abs(korunan - 1.0) > 0.001:
        raise RuntimeError(f"Panel hedef kutlesini kaybetti: %{korunan * 100:.2f}")
    return panel


def _takvim(frame: pd.DataFrame) -> pd.DataFrame:
    """Takvim + dongusel kodlama. include_year=False: test donemi gelecekte."""
    out = add_calendar_features(frame, ZAMAN, include_year=False)
    # Aralik 31 -> Ocak 1 komsulugunu koruyan sin/cos ciftleri.
    return add_cyclical_features(
        out, {f"{ZAMAN}_ay": 12, f"{ZAMAN}_haftanin_gunu": 7, f"{ZAMAN}_yilin_gunu": 366}
    )


def _tatil(frame: pd.DataFrame) -> pd.DataFrame:
    """Resmi tatiller + Ramazan (hicri kayma yuzunden takvimden OGRENILEMEZ)."""
    out = add_turkish_holiday_features(frame, ZAMAN)
    return add_ramadan_features(out, ZAMAN)


def _hava(frame: pd.DataFrame) -> pd.DataFrame:
    """Gunluk hava + fiziksel turevler.

    Ayni gunun havasi MESRUDUR: tahmin aninda hava TAHMINI (forecast) elde
    olur ve deterministik kaynaktan gelir -- hedeften turemez.
    """
    hava = (
        pd.read_parquet(HAVA)
        .drop(columns=["konum", "konum_key", "il_key"])
        .rename(columns={"tarih": ZAMAN})
    )
    once = len(frame)
    out = frame.merge(hava, on=[GRUP, ZAMAN], how="left", validate="many_to_one")
    if len(out) != once:
        raise RuntimeError("hava merge satir sayisini degistirdi")
    return add_physical_derivatives(out, group_columns=[GRUP], time_column=ZAMAN)


def _gunes(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Gunes geometrisi (gun uzunlugu, GHI...) -- tamamen deterministik."""
    esleme = dict(zip(ref[GRUP], ref["anahtar"], strict=True))
    gunes = pd.read_parquet(GUNES).rename(columns={"tarih": ZAMAN})
    once = len(frame)
    out = (
        frame.assign(anahtar=frame[GRUP].map(esleme))
        .merge(gunes, on=["anahtar", ZAMAN], how="left", validate="many_to_one")
        .drop(columns=["anahtar"])
    )
    if len(out) != once:
        raise RuntimeError("gunes merge satir sayisini degistirdi")
    return out


def _lag(frame: pd.DataFrame) -> pd.DataFrame:
    """Hedefin gecmisi -- horizon=31 kaydirmali oldugu icin MESRU."""
    out = add_lag_features(
        frame, HEDEF, [31, 62, 93],
        time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP],
    )
    return add_rolling_features(
        out, HEDEF, [31, 93],
        time_column=ZAMAN, horizon=UFUK, group_columns=[GRUP],
    )


def _komsu(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Komsu ilcelerin GECMIS hedefi (ufuk=31) -- firtina mekansal yayilir."""
    koordinat = ref.loc[
        ref[GRUP].isin(frame[GRUP].unique()), [GRUP, "lat", "lon"]
    ].reset_index(drop=True)
    komsular = nearest_neighbours(koordinat, key_column=GRUP)
    return add_neighbour_target_lag(
        frame, komsular, key_column=GRUP, time_column=ZAMAN,
        target_column=HEDEF, horizon=UFUK,
    )


def _frekans(frame: pd.DataFrame) -> pd.DataFrame:
    """Ilce frekans kodlamasi -- hedefi kullanmaz, sizintisiz."""
    return add_frequency_encoding(frame, [GRUP])


def aileleri_kur(
    panel: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Yedi aileyi sirayla ekler; her ailenin SAYISAL kolonlarini kaydeder.

    Fold'lar KONUMSAL indekstir ve panel uzerinde hesaplanir. Feature adimi
    satir sayisini/sirasini degistirirse fold'lar baska satirlara isaret eder
    ve hata VERMEZ -- o yuzden her adimdan sonra satir kimligi dogrulanir.
    """
    adimlar = [
        ("takvim", _takvim),
        ("tatil", _tatil),
        ("hava", _hava),
        ("gunes", lambda f: _gunes(f, ref)),
        ("lag", _lag),
        ("komsu", lambda f: _komsu(f, ref)),
        ("frekans", _frekans),
    ]
    yasak = {HEDEF, ZAMAN, GRUP, PANEL_FLAG_COLUMN, *AYNI_GUN_SIZINTISI}
    frame = panel
    aile_kolonlari: dict[str, list[str]] = {}
    for ad, kur in adimlar:
        onceki = set(frame.columns)
        frame = kur(frame)
        if len(frame) != len(panel):
            raise RuntimeError(f"'{ad}' ailesi satir sayisini degistirdi")
        if not (frame[GRUP].to_numpy() == panel[GRUP].to_numpy()).all():
            raise RuntimeError(f"'{ad}' ailesi satir sirasini bozdu")
        aile_kolonlari[ad] = [
            c for c in frame.columns
            if c not in onceki and c not in yasak
            and pd.api.types.is_numeric_dtype(frame[c])
        ]
        print(f"  {ad:<8} {len(aile_kolonlari[ad]):>2} kolon")
    return frame, aile_kolonlari


def main() -> int:
    if not VERI.exists():
        print(f"HATA: {VERI} yok.")
        print("Indir: kaggle datasets download -d "
              "tmlalper/manisa-izmir-plansiz-elektrik-kesintileri --unzip")
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()

    print("1/4  PANEL")
    panel = panel_kur()
    n_ilce = panel[GRUP].nunique()
    n_gun = panel[ZAMAN].nunique()
    print(f"  {len(panel):,} satir = {n_ilce} ilce x {n_gun} gun")

    # Fold'lar TUM kosularda AYNIDIR -- provayla ayni sema (embargo=31, 4x31).
    folds = purged_time_series_split(
        panel[ZAMAN], embargo=pd.Timedelta(days=UFUK),
        n_splits=4, test_span=pd.Timedelta(days=UFUK), verbose=False,
    )

    print("\n2/4  AILELER")
    ref = pd.read_parquet(REFERANS)
    ozellik, aile_kolonlari = aileleri_kur(panel, ref)

    kolonlar = [c for kolonlar in aile_kolonlari.values() for c in kolonlar]
    if len(kolonlar) != len(set(kolonlar)):
        raise RuntimeError("Aileler arasinda kolon cakismasi var")
    for kolon in kolonlar:
        if kolon in AYNI_GUN_SIZINTISI or kolon == PANEL_FLAG_COLUMN:
            raise RuntimeError(f"Sizintili kolon feature listesine girdi: {kolon}")
    print(f"  toplam {len(kolonlar)} feature")

    print("\n3/4  TAM MODEL (tum aileler)")
    params = starter_params("lightgbm", "regression", objective="mae")
    y = ozellik[HEDEF].to_numpy()
    tam = cross_validate(
        ozellik[kolonlar], y, folds, kind="lightgbm", metric="mae",
        params=params, verbose=False,
    )
    kapsanan, _ = tam.covered_predictions()
    sifir_baseline = float(np.abs(y[kapsanan]).mean())
    print(f"  tam MAE      : {tam.overall_score:.4f}")
    print(f"  sifir-baseline: {sifir_baseline:.4f}")
    print(f"  fold_std     : {tam.fold_std:.4f}"
          f"  ({'STABIL' if tam.is_stable else 'GURULTULU'})")

    print("\n4/4  LEAVE-ONE-GROUP-OUT")
    gruplar = [
        FeatureGroup(ad, tuple(kols), risk=RISKLER[ad])
        for ad, kols in aile_kolonlari.items()
    ]
    tablo = leave_one_group_out(
        ozellik[kolonlar], y, folds, groups=gruplar,
        kind="lightgbm", metric="mae", params=params, verbose=True,
    )
    # LOGO kendi taban kosusunu yapar; ayni fold+params+seed ile tam modelle
    # AYNI cikmali. Cikmadiysa determinizm bozuktur ve deltalar guvenilmez.
    taban = float(tablo.attrs["taban_skor"])
    if abs(taban - tam.overall_score) > 1e-6:
        print(f"  UYARI: LOGO tabani ({taban:.6f}) tam modelden "
              f"({tam.overall_score:.6f}) farkli -- determinizm kontrol et")

    aileler = {}
    for satir in tablo.itertuples():
        aileler[satir.grup] = {
            "mae_ailesiz": round(float(satir.skor_grupsuz), 4),
            # delta = mae_ailesiz - mae_tam; pozitif = aile katki veriyor.
            "delta": round(float(satir.skor_grupsuz) - float(tam.overall_score), 4),
            "kolon_sayisi": int(satir.feature_sayisi),
        }
    siralama = sorted(aileler, key=lambda ad: aileler[ad]["delta"], reverse=True)

    onem = tam.feature_importance.sort_values("importance", ascending=False).head(15)
    gain_top15 = [
        [str(satir["feature"]), round(float(satir["importance"]), 1)]
        for _, satir in onem.iterrows()
    ]

    sonuc = {
        "tam_mae": round(float(tam.overall_score), 4),
        "sifir_baseline": round(sifir_baseline, 4),
        "fold_std": round(float(tam.fold_std), 4),
        "aileler": aileler,
        "siralama": siralama,
        "gain_top15": gain_top15,
        "panel": {"satir": int(len(panel)), "ilce": int(n_ilce), "gun": int(n_gun)},
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text(
        json.dumps(sonuc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )

    print(f"\n{'=' * 70}")
    print("VERI GUNU ONCELIK LISTESI (delta = aile silinince MAE kaybi)")
    for sira, ad in enumerate(siralama, start=1):
        bilgi = aileler[ad]
        print(f"  {sira}. {ad:<8} delta={bilgi['delta']:+8.2f}  "
              f"({bilgi['kolon_sayisi']} kolon, ailesiz MAE={bilgi['mae_ailesiz']:.2f})")
    print(f"\nYazildi: {CIKTI.relative_to(KOK)}")
    print(f"TAMAM ({time.perf_counter() - baslangic:.0f} sn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
