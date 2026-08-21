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

import argparse
import json
import math
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
from gridup.ablation import FeatureGroup, aile_hukmu, leave_one_group_out  # noqa: E402
from gridup.features import (  # noqa: E402
    add_calendar_features,
    add_cyclical_features,
    add_lag_features,
    add_neighbour_target_lag,
    add_rolling_features,
    add_turkish_holiday_features,
    nearest_neighbours,
)
from gridup.features.external import attach_external  # noqa: E402
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
    "takvim": "cekirdek",
    "tatil": "cekirdek",
    "lag": "cekirdek",
    "frekans": "cekirdek",
    "komsu": "deneysel",
    # Harici aileler artik features.external.attach_external'dan TEK kaynaktan
    # gelir (2026-08-18 denetimi P1-5: on bir kaynagin sekizi hicbir pipeline
    # tarafindan cagrilmiyordu). Her biri ayri aile => LOGO tablosunda ayri
    # satir => "hangi harici veri ise yariyor" olcumle yanitlanir.
    "hava": "harici",
    "hava_saatlik": "harici",
    "hava_kalitesi": "harici",
    "konvektif": "harici",
    "nem_toprak": "harici",
    "gunes": "harici",
    "yangin": "harici",
    "deprem": "harici",
    "turizm_yillik": "harici",
    "turizm_aylik": "harici",
    "izsu": "harici",
    "epias": "harici",
}


#: EPIAS tam izgara paneli -- AYNANIN yerine gecen prova zemini.
EPIAS_PANEL = KOK / "data" / "external" / "epias" / "panel_ilce_gun_tam.parquet"

KAYNAKLAR = ("epias", "ayna")


def panel_kur_epias() -> pd.DataFrame:
    """EPIAS ilce x gun panelini okur -- 96 ilce, 2022-2026.

    NEDEN AYNANIN YERINE (2026-08-21)
    ----------------------------------
    Ayna dosyasi (``MANISA_IZMIR_PLANSIZ_KESINTILER.csv``) 47 ilce ve
    2021-2022 kapsar; yalnizca iki il. EPIAS paneli AYNI OLCUMUN genisidir:

        ayna   47 ilce · 2 il · 2021-2022 · 68.257 olay
        epias  96 ilce · 5 il · 2022-2026 · 405.819 olay  (GDZ + ADM)

    Yani ayni fenomen, iki kat ilce ve iki kat sure. Olcum aletini
    kalibre ederken genis olani kullanmamak icin sebep yok.

    KURAL NOTU: bu PROVA kullanimidir, model girdisi degildir. Kesinti
    verisi ``gridup.uygunluk`` tarafindan modele kapatilmistir; oradaki
    statik tarama yalnizca ``src/gridup/`` altina bakar cunku prova ve
    olcum betiklerinin bu veriyi okumasi gereklidir (bkz. TARANAN_DIZINLER).
    """
    panel = pd.read_parquet(EPIAS_PANEL)
    panel[ZAMAN] = pd.to_datetime(panel[ZAMAN])
    # Ayna receteyle ayni sozlesme: GRUP + ZAMAN + HEDEF, artigi atilir.
    # 'etkilenen_abone' AYNI GUN bilgisidir ve tahmin aninda bilinmez --
    # aynadaki 'effectedsubscribers' ile ayni sebeple feature olamaz.
    tutulan = [GRUP, ZAMAN, HEDEF]
    return panel[tutulan].sort_values([GRUP, ZAMAN]).reset_index(drop=True)


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
        ham,
        entity_columns=[GRUP],
        time_column=ZAMAN,
        value_columns=[HEDEF, *AYNI_GUN_SIZINTISI],
        verbose=False,
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


def _lag(frame: pd.DataFrame) -> pd.DataFrame:
    """Hedefin gecmisi -- horizon=31 kaydirmali oldugu icin MESRU."""
    out = add_lag_features(
        frame,
        HEDEF,
        shifts=[31, 62, 93],
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
    )
    return add_rolling_features(
        out,
        HEDEF,
        [31, 93],
        time_column=ZAMAN,
        horizon=UFUK,
        group_columns=[GRUP],
    )


def _komsu(frame: pd.DataFrame, ref: pd.DataFrame) -> pd.DataFrame:
    """Komsu ilcelerin GECMIS hedefi (ufuk=31) -- firtina mekansal yayilir."""
    koordinat = ref.loc[ref[GRUP].isin(frame[GRUP].unique()), [GRUP, "lat", "lon"]].reset_index(
        drop=True
    )
    komsular = nearest_neighbours(koordinat, key_column=GRUP)
    return add_neighbour_target_lag(
        frame,
        komsular,
        key_column=GRUP,
        time_column=ZAMAN,
        target_column=HEDEF,
        horizon=UFUK,
    )


def aileleri_kur(
    panel: pd.DataFrame, ref: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Yedi aileyi sirayla ekler; her ailenin SAYISAL kolonlarini kaydeder.

    Fold'lar KONUMSAL indekstir ve panel uzerinde hesaplanir. Feature adimi
    satir sayisini/sirasini degistirirse fold'lar baska satirlara isaret eder
    ve hata VERMEZ -- o yuzden her adimdan sonra satir kimligi dogrulanir.
    """
    # Cekirdek aileler yerel; HARICI aileler tek kapidan (attach_external).
    adimlar = [
        ("takvim", _takvim),
        ("tatil", _tatil),
        ("lag", _lag),
        ("komsu", lambda f: _komsu(f, ref)),
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
            c
            for c in frame.columns
            if c not in onceki and c not in yasak and pd.api.types.is_numeric_dtype(frame[c])
        ]
        print(f"  {ad:<14} {len(aile_kolonlari[ad]):>3} kolon")

    # HARICI AILELER -- tek cagri, aile bazli kolon haritasiyla.
    ek = attach_external(
        frame,
        key_column=GRUP,
        time_column=ZAMAN,
        horizon=UFUK,
        root=KOK,
    )
    if len(ek.frame) != len(panel):
        raise RuntimeError("attach_external satir sayisini degistirdi")
    if not (ek.frame[GRUP].to_numpy() == panel[GRUP].to_numpy()).all():
        raise RuntimeError("attach_external satir sirasini bozdu")
    frame = ek.frame
    for ad, kolonlar in ek.families.items():
        sayisal = [
            c for c in kolonlar if c not in yasak and pd.api.types.is_numeric_dtype(frame[c])
        ]
        if sayisal:
            aile_kolonlari[ad] = sayisal
            print(f"  {ad:<14} {len(sayisal):>3} kolon")
    for ad, neden in ek.skipped.items():
        print(f"  {ad:<14} ATLANDI: {neden}")
    return frame, aile_kolonlari


#: Tohum listesi. 42 basta (deponun kanonik tohumu) ki --tohum buyutuldugunde
#: onceki kosunun ilk k tohumu KORUNSUN ve sonuclar karsilastirilabilir kalsin.
TOHUMLAR: tuple[int, ...] = (42, 0, 1, 2, 3, 4, 5, 6, 7, 8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tohum",
        type=int,
        default=5,
        help=(
            "Kac tohumla olculsun (varsayilan 5). 1 verirsen hukum verilemez -- "
            "yayilim bilinmeden 'faydali' demek eski hatanin ta kendisi"
        ),
    )
    parser.add_argument(
        "--kaynak",
        choices=KAYNAKLAR,
        default="epias",
        help=(
            "Prova zemini. 'epias': 96 ilce x 2022-2026 (varsayilan). "
            "'ayna': eski 47 ilce x 2021-2022 Kaggle dosyasi -- eski "
            "olcumlerle karsilastirmak icin korunuyor."
        ),
    )
    args = parser.parse_args()
    if not 1 <= args.tohum <= len(TOHUMLAR):
        parser.error(f"--tohum 1..{len(TOHUMLAR)} arasinda olmali")

    if args.kaynak == "epias" and not EPIAS_PANEL.exists():
        print(f"HATA: {EPIAS_PANEL} yok. Once: python scripts/epias_panel.py")
        return 1
    if args.kaynak == "ayna" and not VERI.exists():
        print(f"HATA: {VERI} yok.")
        print(
            "Indir: kaggle datasets download -d "
            "tmlalper/manisa-izmir-plansiz-elektrik-kesintileri --unzip"
        )
        return 1

    set_global_seed(42)
    baslangic = time.perf_counter()

    print(f"1/4  PANEL  (kaynak: {args.kaynak})")
    panel = panel_kur_epias() if args.kaynak == "epias" else panel_kur()
    n_ilce = panel[GRUP].nunique()
    n_gun = panel[ZAMAN].nunique()
    print(f"  {len(panel):,} satir = {n_ilce} ilce x {n_gun} gun")

    # Fold'lar TUM kosularda AYNIDIR -- provayla ayni sema (embargo=31, 4x31).
    folds = purged_time_series_split(
        panel[ZAMAN],
        embargo=pd.Timedelta(days=UFUK),
        n_splits=4,
        test_span=pd.Timedelta(days=UFUK),
        verbose=False,
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
        ozellik[kolonlar],
        y,
        folds,
        kind="lightgbm",
        metric="mae",
        params=params,
        verbose=False,
    )
    kapsanan, _ = tam.covered_predictions()
    sifir_baseline = float(np.abs(y[kapsanan]).mean())
    print(f"  tam MAE      : {tam.overall_score:.4f}")
    print(f"  sifir-baseline: {sifir_baseline:.4f}")
    print(f"  fold_std     : {tam.fold_std:.4f}  ({'STABIL' if tam.is_stable else 'GURULTULU'})")

    print("\n4/4  LEAVE-ONE-GROUP-OUT")
    gruplar = [
        FeatureGroup(ad, tuple(kols), risk=RISKLER.get(ad, "harici"))
        for ad, kols in aile_kolonlari.items()
    ]
    # COK TOHUMLU, ESLESTIRILMIS OLCUM (2026-08-21).
    #
    # NEDEN: ayni ayna verisinde ayni ablasyon iki kez kosuldu ve YEDI ailenin
    # BESINDE isaret degisti (konvektif +4,12 -> -0,41; epias +3,13 -> -1,51).
    # Tohum gurultusu ~1,24 MAE, aile etkileri +-2 MAE -- yani kucuk etkilerin
    # isareti yazi-turaydi. Tek kosunun ham deltasini SIRALAMA olarak
    # raporlamak yanlis guven uretiyordu.
    #
    # ESLESTIRME: her tohum icin hem tam model hem ailesiz model AYNI tohum ve
    # AYNI fold'larla kosulur; delta o cift icinde alinir. Ortak gurultu boylece
    # sadelesir ve geriye ailenin kendi etkisi kalir.
    tohum_deltalari: dict[str, list[float]] = {g.ad: [] for g in gruplar}
    kolon_sayilari: dict[str, int] = {}
    ailesiz_son: dict[str, float] = {}
    taban_skorlar: list[float] = []

    for tohum in TOHUMLAR[: args.tohum]:
        tohum_params = dict(params)
        tohum_params["random_state"] = int(tohum)
        tablo = leave_one_group_out(
            ozellik[kolonlar],
            y,
            folds,
            groups=gruplar,
            kind="lightgbm",
            metric="mae",
            params=tohum_params,
            verbose=False,
        )
        taban = float(tablo.attrs["taban_skor"])
        taban_skorlar.append(taban)
        for satir in tablo.itertuples():
            tohum_deltalari[satir.grup].append(float(satir.skor_grupsuz) - taban)
            kolon_sayilari[satir.grup] = int(satir.feature_sayisi)
            ailesiz_son[satir.grup] = float(satir.skor_grupsuz)
        print(f"  tohum {tohum}: taban MAE {taban:.2f}")

    # OLCULEN gurultu tabani: tohumlar arasi TAM MODEL yayilimi. Sabit bir
    # varsayim degil, bu kosunun kendi olcumu -- veri degisince esik de degisir.
    tohum_gurultusu = (
        float(np.std(taban_skorlar, ddof=1)) if len(taban_skorlar) > 1 else float("nan")
    )
    print(f"\n  OLCULEN tohum gurultusu (tam model yayilimi): {tohum_gurultusu:.3f} MAE")

    aileler = {}
    for ad, deltalar in tohum_deltalari.items():
        # Tek tohumda gurultu olculemez (nan); esigi 0 yaparak aile_hukmu'nun
        # kendi "tek olcum -> KARARSIZ" kuralina birakiyoruz.
        esik = 0.0 if math.isnan(tohum_gurultusu) else tohum_gurultusu
        hukum = aile_hukmu(deltalar, gurultu=esik)
        aileler[ad] = {
            "mae_ailesiz_son_tohum": round(ailesiz_son[ad], 4),
            "delta": round(hukum.ortalama, 4),
            "delta_sapma": None if math.isnan(hukum.sapma) else round(hukum.sapma, 4),
            "delta_tohumlar": [round(d, 4) for d in deltalar],
            "karar": hukum.karar,
            "gerekce": hukum.gerekce,
            "kolon_sayisi": kolon_sayilari[ad],
        }
    # Siralama once KARARA, sonra buyukluge gore: "kararsiz" bir aile, ham
    # deltasi buyuk diye faydali bir ailenin ustune cikmamali.
    karar_sirasi = {"FAYDALI": 0, "KARARSIZ": 1, "ZARARLI": 2}
    siralama = sorted(
        aileler,
        key=lambda ad: (karar_sirasi[aileler[ad]["karar"]], -aileler[ad]["delta"]),
    )

    onem = tam.feature_importance.sort_values("importance", ascending=False).head(15)
    gain_top15 = [
        [str(satir["feature"]), round(float(satir["importance"]), 1)]
        for _, satir in onem.iterrows()
    ]

    sonuc = {
        "tam_mae": round(float(tam.overall_score), 4),
        "sifir_baseline": round(sifir_baseline, 4),
        "fold_std": round(float(tam.fold_std), 4),
        "n_tohum": int(args.tohum),
        "tohum_gurultusu": round(tohum_gurultusu, 4),
        "taban_skorlar": [round(v, 4) for v in taban_skorlar],
        "aileler": aileler,
        "siralama": siralama,
        "gain_top15": gain_top15,
        "panel": {"satir": int(len(panel)), "ilce": int(n_ilce), "gun": int(n_gun)},
    }
    CIKTI.parent.mkdir(parents=True, exist_ok=True)
    CIKTI.write_text(json.dumps(sonuc, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")

    print(f"\n{'=' * 78}")
    print(
        f"VERI GUNU ONCELIK LISTESI -- {args.tohum} tohum, eslestirilmis fark\n"
        f"  delta = MAE(ailesiz) - MAE(tam), ayni tohum ve fold'larda.\n"
        f"  Hukum icin etki HEM yayilimi HEM olculen tohum gurultusunu "
        f"({tohum_gurultusu:.2f}) asmali."
    )
    print("=" * 78)
    for sira, ad in enumerate(siralama, start=1):
        bilgi = aileler[ad]
        sapma = bilgi["delta_sapma"]
        sapma_metni = f"+-{sapma:.2f}" if sapma is not None else "  ?  "
        print(
            f"  {sira:>2}. {ad:<16} {bilgi['karar']:<9} "
            f"{bilgi['delta']:+7.2f} {sapma_metni}  ({bilgi['kolon_sayisi']} kolon)"
        )
    kararli = [ad for ad in siralama if aileler[ad]["karar"] != "KARARSIZ"]
    print(
        f"\n  {len(kararli)}/{len(siralama)} aile hukum aldi; "
        f"{len(siralama) - len(kararli)} tanesi KARARSIZ (gurultuden ayirt edilemedi)."
    )
    if not kararli:
        print("  DIKKAT: hicbir aile gurultuyu gecmedi -- bu siralamaya gore KARAR VERME.")
    print(f"\nYazildi: {CIKTI.relative_to(KOK)}")
    print(f"TAMAM ({time.perf_counter() - baslangic:.0f} sn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
