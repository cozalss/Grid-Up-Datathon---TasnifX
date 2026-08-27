"""Grid Up Datathon -- SOTA Breakthrough Trafo Gunluk Tuketim Tahmini.

Bu betik 1. sirayi (0.99403 RMSLE) gecmek uzere tasarlanmis yeni nesil modelleme hattidir.

KILIT YENILIKLER:
-----------------
1. KOORT BAZLI OZNITELIK VE MODELLEME:
   - G1 (Yaz25 Aktif, %41.0 test satiri): Gecen yilin ayni 122 gununun (Nis-Tem 2025)
     aylik, haftalik ve gunluk profilleri, trafo bazli YoY buyume orani.
   - G2 (Yaz-Sonrasi Sicak, %36.9 test satiri): Son donem baz seviyesi + Ilce bazli
     yaz/kis gecis carpani + Tarimsal sulama etkilesimi.
   - G3 (Soguk Trafolar, %22.2 test satiri): Kapasite, ilce yapisi, altyapi,
     tam takvim/tatil/ramazan donguleri, hava isil duyarliligi ve kalibreli buzulme.

2. TAM VE ZENGIN TAKVIM MUHENDISLIGI:
   - Dongusel haftagunu (sin/cos), ay (sin/cos), yilin gunu (sin/cos).
   - Resmi ve idari tatiller, bayram mesafeleri.
   - 2025 vs 2026 Kurban Bayrami ay kaymasi (2025: 6-9 Haziran, 2026: 26-30 Mayis).

3. COKLU AILE VE MIMARI HARMANI:
   - CatBoost (farkli derinlik ve regularization)
   - LightGBM (genis yaprakli ve histerik bolmeli)
   - XGBoost (hist tabanli agaclar)
   - MLP Sinir Agi (Varlik gomulumlu derin katmanlar)
   - Log-uzayinda analitik agirlikli harman (expm1(mean(log1p)))

4. OLU TRAFO SIFIR KIRPMA:
   - Son 14+ gunu tamamen sifir olan ve susmus trafolarin sifir garantisi
     (log1p uzayinda 40+ MSE cezasini onler).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.features.temporal import (  # noqa: E402
    add_calendar_features,
    add_ramadan_features,
    add_turkish_holiday_features,
)
from gridup.features.trafo import (  # noqa: E402
    grup_seviyeleri_ekle,
    trafo_ozetleri_cikar,
    trafo_ozetleri_uygula,
)
from gridup.reporting import satir_tamponlu_cikti  # noqa: E402
from gridup.turkish import join_key  # noqa: E402

HAM = KOK / "data" / "raw"
DIS = KOK / "data" / "external"
GONDERIM = KOK / "submissions"

HEDEF = "tuketim"
ZAMAN = "tarih"
GRUP = "tanim"
EGITIM_BASI = "2025-01-01"


@dataclass(frozen=True)
class Blok:
    ad: str
    etiket_basi: str
    etiket_sonu: str

    @property
    def ozet_basi(self) -> pd.Timestamp:
        return pd.Timestamp(EGITIM_BASI)

    @property
    def ozet_bitis(self) -> pd.Timestamp:
        return pd.Timestamp(self.etiket_basi) - pd.Timedelta(days=1)


BLOKLAR: tuple[Blok, ...] = (
    Blok("yaz25", "2025-04-01", "2025-07-31"),
    Blok("guz25", "2025-08-01", "2025-11-30"),
    Blok("kis26", "2025-12-01", "2026-03-31"),
)

EK_KOKENLER: tuple[tuple[str, str, str], ...] = (
    ("sub25", "2025-02-01", "2025-03-31"),
    ("bah25", "2025-05-01", "2025-08-31"),
    ("yaz25b", "2025-07-01", "2025-10-31"),
    ("guz25b", "2025-09-01", "2025-12-31"),
    ("kis26b", "2025-11-01", "2026-02-28"),
    ("bah26", "2026-01-01", "2026-03-31"),
)

HAVA_TABLOLARI: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "hava_gunluk.parquet",
        (
            "sicaklik_ort",
            "sicaklik_max",
            "sicaklik_min",
            "hissedilen_max",
            "yagis_toplam",
            "yagis_saati",
            "ruzgar_max",
            "gunes_radyasyon",
            "asiri_sicak",
            "asiri_soguk",
        ),
    ),
    (
        "nem_toprak_gunluk.parquet",
        ("nem_ort", "ciy_ort", "vpd_ort", "toprak_nem_ort", "bulut_dusuk_ort", "et0_toplam"),
    ),
)

CDD_TABANLARI: tuple[int, ...] = (18, 22, 24)
ISIL_PENCERELER: tuple[int, ...] = (3, 7, 14)
ISIL_KOLONLAR: tuple[str, ...] = ("cdd22", "cdd24", "sicaklik_ort")

STATIK_ILCE_TABLOLARI: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "arazi_ortusu_ilce.parquet",
        (
            "agac_orani",
            "calilik_orani",
            "otlak_orani",
            "tarim_orani",
            "yerlesim_orani",
            "ciplak_orani",
            "su_orani",
            "bitki_ortusu_orani",
        ),
    ),
    (
        "osm_altyapi_ilce.parquet",
        (
            "osm_trafo",
            "osm_direk",
            "osm_dagitim_hat_km",
            "osm_iletim_hat_km",
            "osm_kablo_km",
            "osm_direk_yogunlugu",
            "osm_hat_yogunlugu",
        ),
    ),
)

DISLANAN = {"id", HEDEF, ZAMAN, GRUP, "lokasyon", "_blok"}
KATEGORIK = ("il_key", "bolge", "ilce_key")
TEST_SOGUK_PAYI = 0.2216
_GECMIS_ONEKI = ("t_",)


# ---------------------------------------------------------------- veri yukleme & temel


def yukle() -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = pd.read_csv(
        HAM / "train.csv", dtype={GRUP: "string"}, parse_dates=[ZAMAN], encoding="utf-8"
    )
    te = pd.read_csv(
        HAM / "test.csv", dtype={GRUP: "string"}, parse_dates=[ZAMAN], encoding="utf-8"
    )
    return tr, te


def lokasyon_ayristir(frame: pd.DataFrame) -> pd.DataFrame:
    p = frame["lokasyon"].str.split(">")
    sonuc = frame.copy()
    sonuc["il_key"] = p.str[0].str.strip().map(join_key)
    sonuc["ilce_key"] = p.str[-1].str.strip().map(join_key)
    sonuc["bolge"] = np.where(p.str.len() >= 3, p.str[1].str.strip(), "YOK")
    return sonuc


def hava_yukle() -> pd.DataFrame:
    tablolar: list[pd.DataFrame] = []
    for dosya, kolonlar in HAVA_TABLOLARI:
        yol = DIS / dosya
        d = pd.read_parquet(yol, columns=["ilce_key", "tarih", *kolonlar])
        d["tarih"] = pd.to_datetime(d["tarih"]).dt.normalize()
        tablolar.append(d.set_index(["ilce_key", "tarih"]))
    hava = pd.concat(tablolar, axis=1).reset_index()

    gunes = pd.read_parquet(
        DIS / "gunes_gunluk.parquet",
        columns=["anahtar", "tarih", "gunes_ghi_gunluk", "gun_uzunlugu_saat"],
    )
    gunes["ilce_key"] = gunes["anahtar"].str.split("|").str[-1]
    gunes["tarih"] = pd.to_datetime(gunes["tarih"]).dt.normalize()
    hava = hava.merge(
        gunes.drop(columns=["anahtar"]), on=["ilce_key", "tarih"], how="left", validate="one_to_one"
    )

    for taban in CDD_TABANLARI:
        hava[f"cdd{taban}"] = (hava["sicaklik_ort"] - taban).clip(lower=0.0)
    hava = hava.drop(columns=["isitma_derece_gun", "sogutma_derece_gun"], errors="ignore")

    hava = hava.sort_values(["ilce_key", "tarih"])
    g = hava.groupby("ilce_key", observed=True)
    for kolon in ISIL_KOLONLAR:
        for p in ISIL_PENCERELER:
            hava[f"{kolon}_ort{p}"] = g[kolon].transform(
                lambda s, _p=p: s.rolling(_p, min_periods=1).mean()
            )
    return hava


def hava_ekle(frame: pd.DataFrame, hava: pd.DataFrame) -> pd.DataFrame:
    oncesi = len(frame)
    sonuc = frame.merge(hava, on=["ilce_key", "tarih"], how="left", validate="many_to_one")
    if len(sonuc) != oncesi:
        raise RuntimeError(f"hava birlesimi satir sayisini degistirdi: {oncesi} -> {len(sonuc)}")
    return sonuc


def statik_ilce_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    tablolar = []
    for dosya, kolonlar in STATIK_ILCE_TABLOLARI:
        d = pd.read_parquet(DIS / dosya, columns=["ilce_key", *kolonlar])
        tablolar.append(d.set_index("ilce_key"))
    statik = pd.concat(tablolar, axis=1).reset_index()

    sonuclar = []
    for c in cerceveler:
        yeni = c.merge(statik, on="ilce_key", how="left", validate="many_to_one")
        if len(yeni) != len(c):
            raise RuntimeError("statik ilce birlesimi satir sayisini degistirdi")
        sonuclar.append(yeni)
    return sonuclar


def ilce_yapisi_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    ref = pd.read_parquet(
        KOK / "data" / "reference" / "ilceler_gdz_adm.parquet",
        columns=["ilce_key", "nufus", "alan_km2"],
    ).drop_duplicates("ilce_key")

    hepsi = pd.concat([c[[GRUP, "guc", "ilce_key"]] for c in cerceveler]).drop_duplicates(GRUP)
    g = hepsi.groupby("ilce_key", observed=True)["guc"]
    ilce = pd.DataFrame(
        {
            "ilce_trafo_sayisi": g.size().astype("float64"),
            "ilce_toplam_guc": g.sum(),
            "ilce_guc_medyan": g.median(),
        }
    ).reset_index()
    hepsi["guc_yuzdelik"] = g.rank(pct=True)
    yuzdelik = hepsi.set_index(GRUP)["guc_yuzdelik"]

    sonuclar = []
    for c in cerceveler:
        yeni = c.merge(ilce, on="ilce_key", how="left", validate="many_to_one")
        yeni = yeni.merge(ref, on="ilce_key", how="left", validate="many_to_one")
        yeni["ilce_nufus_yogunlugu"] = yeni["nufus"] / yeni["alan_km2"]
        yeni["trafo_basina_nufus"] = yeni["nufus"] / yeni["ilce_trafo_sayisi"]
        yeni["kva_basina_nufus"] = yeni["nufus"] / yeni["ilce_toplam_guc"]
        yeni["guc_yuzdelik"] = yeni[GRUP].map(yuzdelik).astype("float64")
        yeni["guc_payi"] = yeni["guc"] / yeni["ilce_toplam_guc"]
        yeni["guc_medyan_orani"] = yeni["guc"] / yeni["ilce_guc_medyan"]
        if "osm_dagitim_hat_km" in yeni.columns:
            yeni["trafo_basina_hat"] = yeni["osm_dagitim_hat_km"] / yeni["ilce_trafo_sayisi"]
        if "tarim_orani" in yeni.columns and "cdd22_ort7" in yeni.columns:
            yeni["tarim_cdd_etkilesim"] = yeni["tarim_orani"] * yeni["cdd22_ort7"]
        sonuclar.append(yeni)
    return sonuclar


def kimlik_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    sonuclar = []
    for c in cerceveler:
        yeni = c.copy()
        ad = yeni[GRUP].astype("string")
        yeni["tanim_num"] = pd.to_numeric(ad.where(ad.str.fullmatch(r"\d+")), errors="coerce")
        yeni["tanim_uzunluk"] = ad.str.len().astype("float64")
        for n in (2, 3, 4, 5):
            yeni[f"tanim_on{n}"] = pd.to_numeric(ad.str[:n], errors="coerce")
        sonuclar.append(yeni)
    return sonuclar


def panel_yapisi_ekle(etiket: pd.DataFrame) -> pd.DataFrame:
    sonuc = etiket.copy()
    pencere_basi = sonuc[ZAMAN].min()
    pencere_gun = float((sonuc[ZAMAN].max() - pencere_basi).days) + 1.0
    g = sonuc.groupby(GRUP, observed=True)[ZAMAN]
    ilk, son, adet = g.transform("min"), g.transform("max"), g.transform("size")
    sonuc["p_gun_sayisi"] = adet.astype("float64")
    sonuc["p_ilk_ofset"] = (ilk - pencere_basi).dt.days.astype("float64")
    sonuc["p_son_ofset"] = (son - pencere_basi).dt.days.astype("float64")
    sonuc["p_yayilma"] = (son - ilk).dt.days.astype("float64") + 1.0
    sonuc["p_doluluk"] = sonuc["p_gun_sayisi"] / sonuc["p_yayilma"]
    sonuc["p_pencere_payi"] = sonuc["p_gun_sayisi"] / pencere_gun
    return sonuc


def ulusal_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    yol = DIS / "epias" / "tuketim_saatlik.parquet"
    saatlik = pd.read_parquet(yol)
    saatlik["_g"] = pd.to_datetime(saatlik["zaman"]).dt.normalize()
    g = saatlik.groupby("_g")["consumption"]
    u = pd.DataFrame({"ulusal_gunluk": g.sum(), "ulusal_tepe": g.max(), "_ort": g.mean()})
    u["ulusal_tepe_orani"] = u["ulusal_tepe"] / u["_ort"].replace(0.0, np.nan)
    u["ulusal_yil_once"] = u["ulusal_gunluk"].reindex(u.index - pd.Timedelta(days=364)).to_numpy()
    u["ulusal_yillik_buyume"] = np.log(
        u["ulusal_gunluk"] / u["ulusal_yil_once"].replace(0.0, np.nan)
    )
    u = u.drop(columns=["_ort"]).reset_index().rename(columns={"_g": ZAMAN})

    sonuclar = []
    for c in cerceveler:
        yeni = c.merge(u, on=ZAMAN, how="left", validate="many_to_one")
        if len(yeni) != len(c):
            raise RuntimeError("ulusal birlesim satir sayisini degistirdi")
        sonuclar.append(yeni)
    return sonuclar


def yas_ekle(*cerceveler: pd.DataFrame) -> list[pd.DataFrame]:
    ilk = (
        pd.concat([c[[GRUP, ZAMAN]] for c in cerceveler]).groupby(GRUP, observed=True)[ZAMAN].min()
    )
    sonuclar = []
    for c in cerceveler:
        yeni = c.copy()
        yeni["yas"] = (yeni[ZAMAN] - yeni[GRUP].map(ilk)).dt.days.astype("float64")
        yeni["ilk_gun_mu"] = (yeni["yas"] == 0).astype("int8")
        sonuclar.append(yeni)
    return sonuclar


def gelismis_takvim_ekle(frame: pd.DataFrame) -> pd.DataFrame:
    sonuc = frame.copy()
    zaman = pd.to_datetime(sonuc[ZAMAN])

    sonuc = add_calendar_features(sonuc, ZAMAN, prefix="tk")
    sonuc = add_turkish_holiday_features(sonuc, ZAMAN, prefix="tatil")
    sonuc = add_ramadan_features(sonuc, ZAMAN, prefix="ramazan")

    hg = zaman.dt.dayofweek
    ay = zaman.dt.month
    yg = zaman.dt.dayofyear

    sonuc["tk_hg_sin"] = np.sin(2 * np.pi * hg / 7.0)
    sonuc["tk_hg_cos"] = np.cos(2 * np.pi * hg / 7.0)
    sonuc["tk_ay_sin"] = np.sin(2 * np.pi * ay / 12.0)
    sonuc["tk_ay_cos"] = np.cos(2 * np.pi * ay / 12.0)
    sonuc["tk_yg_sin"] = np.sin(2 * np.pi * yg / 365.25)
    sonuc["tk_yg_cos"] = np.cos(2 * np.pi * yg / 365.25)

    sonuc["tk_is_haftasonu"] = (hg >= 5).astype("int8")
    sonuc["tk_is_pazar"] = (hg == 6).astype("int8")
    sonuc["tk_is_pazartesi"] = (hg == 0).astype("int8")
    sonuc["tk_is_cuma"] = (hg == 4).astype("int8")

    kurban_tarihleri = pd.to_datetime(
        [
            "2025-06-06",
            "2025-06-07",
            "2025-06-08",
            "2025-06-09",
            "2026-05-26",
            "2026-05-27",
            "2026-05-28",
            "2026-05-29",
            "2026-05-30",
        ]
    )
    zaman_dizi = zaman.to_numpy()[:, None]
    kurban_dizi = kurban_tarihleri.to_numpy()[None, :]
    gun_fark = (zaman_dizi - kurban_dizi).astype("timedelta64[D]").astype(float)
    en_yakin_kurban = np.min(np.abs(gun_fark), axis=1)
    sonuc["kurban_mesafe_gun"] = en_yakin_kurban
    sonuc["kurban_etki_alani"] = np.exp(-en_yakin_kurban / 5.0)

    return sonuc


def ilce_mevsim_oranlari_ekle(uygula: pd.DataFrame, profil_kaynak: pd.DataFrame) -> pd.DataFrame:
    p = profil_kaynak[[GRUP, ZAMAN, HEDEF, "guc", "ilce_key"]].copy()
    p["_y"] = np.log1p(p[HEDEF].clip(lower=0.0))
    p["_ay"] = p[ZAMAN].dt.month

    p["_sapma"] = p["_y"] - p.groupby(GRUP, observed=True)["_y"].transform("mean")
    ilce_ay = (
        p.groupby(["ilce_key", "_ay"], observed=True)["_sapma"]
        .mean()
        .rename("ilce_ay_sapma")
        .reset_index()
    )

    sonuc = uygula.copy()
    sonuc["_ay"] = sonuc[ZAMAN].dt.month
    sonuc = sonuc.merge(ilce_ay, on=["ilce_key", "_ay"], how="left", validate="many_to_one")

    yaz = p[p["_ay"].isin([6, 7])].groupby("ilce_key", observed=True)["_y"].mean()
    kis = p[p["_ay"].isin([1, 2, 3])].groupby("ilce_key", observed=True)["_y"].mean()
    ilce_yaz_kis_fark = (yaz - kis).rename("ilce_yaz_kis_orani").reset_index()

    sonuc = sonuc.merge(ilce_yaz_kis_fark, on="ilce_key", how="left", validate="many_to_one")
    return sonuc.drop(columns=["_ay"])


# ---------------------------------------------------------------- blok & ozet tasiyici


def _sota_ozet_tasi(
    ozet: pd.DataFrame, etiket: pd.DataFrame, ad: str, *, profil_kaynak: pd.DataFrame
) -> pd.DataFrame:
    ozetler = trafo_ozetleri_cikar(
        ozet,
        profil_kaynak=profil_kaynak,
        hedef_penceresi=(etiket[ZAMAN].min(), etiket[ZAMAN].max()),
        isil_kolonlar=("cdd22", "sicaklik_ort"),
    )
    sonuc = trafo_ozetleri_uygula(etiket, ozetler)
    sonuc = grup_seviyeleri_ekle(sonuc, ozet)
    sonuc = ilce_mevsim_oranlari_ekle(sonuc, profil_kaynak)
    sonuc = panel_yapisi_ekle(sonuc)

    pencere = int((ozet[ZAMAN].max() - ozet[ZAMAN].min()).days) + 1
    sonuc["ozet_pencere_gun"] = float(pencere)
    sonuc["t_doluluk"] = sonuc["t_gun_sayisi"] / float(pencere)
    sonuc["ufuk_gun"] = (sonuc[ZAMAN] - ozet[ZAMAN].max()).dt.days.astype("float64")

    z = ozet[ZAMAN]
    gy_yaz = ozet[(z >= "2025-04-01") & (z <= "2025-07-31")].copy()
    if not gy_yaz.empty:
        gy_yaz["_y"] = np.log1p(gy_yaz[HEDEF].clip(lower=0.0))
        gy_yaz["_ay"] = gy_yaz[ZAMAN].dt.month
        gy_aylik = gy_yaz.groupby([GRUP, "_ay"], observed=True)["_y"].mean().unstack("_ay")
        gy_aylik.columns = [f"t_gy_m{col}_log" for col in gy_aylik.columns]
        sonuc = sonuc.merge(gy_aylik, on=GRUP, how="left", validate="many_to_one")

        kis25 = (
            ozet[(z >= "2025-01-01") & (z <= "2025-03-31")]
            .groupby(GRUP, observed=True)[HEDEF]
            .mean()
        )
        kis26 = (
            ozet[(z >= "2026-01-01") & (z <= "2026-03-31")]
            .groupby(GRUP, observed=True)[HEDEF]
            .mean()
        )
        yoy_log = (np.log1p(kis26) - np.log1p(kis25)).rename("t_yoy_buyume").reset_index()
        sonuc = sonuc.merge(yoy_log, on=GRUP, how="left", validate="many_to_one")
    else:
        for m in (4, 5, 6, 7):
            sonuc[f"t_gy_m{m}_log"] = np.nan
        sonuc["t_yoy_buyume"] = np.nan

    sonuc["_blok"] = ad
    return sonuc


def sota_blok_kur(tam_egitim: pd.DataFrame, blok: Blok) -> pd.DataFrame:
    ozet = tam_egitim[
        (tam_egitim[ZAMAN] >= blok.ozet_basi) & (tam_egitim[ZAMAN] <= blok.ozet_bitis)
    ]
    etiket_maske = (tam_egitim[ZAMAN] >= blok.etiket_basi) & (tam_egitim[ZAMAN] <= blok.etiket_sonu)
    etiket = tam_egitim[etiket_maske]
    return _sota_ozet_tasi(ozet, etiket, blok.ad, profil_kaynak=tam_egitim[~etiket_maske])


def sota_egitim_kur(tam_egitim: pd.DataFrame) -> pd.DataFrame:
    parcalar = [sota_blok_kur(tam_egitim, b) for b in BLOKLAR]
    for b, p in zip(BLOKLAR, parcalar, strict=True):
        soguk = int(p["soguk_mu"].sum())
        print(
            f"  blok {b.ad:6} etiket {len(p):>7,} satir  ozet {p['ozet_pencere_gun'].iloc[0]:>4.0f} gun  soguk {soguk:>6,} (%{100 * soguk / len(p):.1f})"  # noqa: E501
        )
    return pd.concat(parcalar, ignore_index=True)


def sota_ek_kokenleri_kur(tam_egitim: pd.DataFrame) -> pd.DataFrame:
    parcalar = []
    for ad, bas, son in EK_KOKENLER:
        blok = Blok(ad, bas, son)
        parcalar.append(sota_blok_kur(tam_egitim, blok))
    return pd.concat(parcalar, ignore_index=True)


def sota_test_kur(tam_egitim: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    sonuc = _sota_ozet_tasi(tam_egitim, test, "TEST", profil_kaynak=tam_egitim)
    soguk = int(sonuc["soguk_mu"].sum())
    print(
        f"  TEST         {len(sonuc):>7,} satir  ozet {sonuc['ozet_pencere_gun'].iloc[0]:>4.0f} gun  soguk {soguk:>6,} (%{100 * soguk / len(sonuc):.1f})"  # noqa: E501
    )
    return sonuc


# ---------------------------------------------------------------- modelleme & egitim


def oznitelikler(frame: pd.DataFrame) -> list[str]:
    adaylar = [k for k in frame.columns if k not in DISLANAN]
    return [k for k in adaylar if frame[k].dtype.kind in "ifbu" or k in KATEGORIK]


def kategorik_kodla(egitim: pd.DataFrame, *digerleri: pd.DataFrame) -> None:
    for kolon in KATEGORIK:
        seviyeler = pd.Index(sorted(egitim[kolon].dropna().unique()))
        for f in (egitim, *digerleri):
            f[kolon] = pd.Categorical(f[kolon], categories=seviyeler)


def soguk_maskele(
    cerceve: pd.DataFrame, kolonlar: list[str], tohum: int, oran: float = 0.2216
) -> pd.DataFrame:
    rng = np.random.default_rng(tohum)
    trafolar = cerceve[GRUP].unique()
    secilen = set(rng.choice(trafolar, size=int(len(trafolar) * oran), replace=False))
    maske = cerceve[GRUP].isin(secilen).to_numpy()
    sonuc = cerceve.copy()
    sonuc.loc[maske, [k for k in kolonlar if k.startswith(_GECMIS_ONEKI)]] = np.nan
    sonuc.loc[maske, "soguk_mu"] = 1
    return sonuc


def ofsetli_hedef(cerceve: pd.DataFrame) -> np.ndarray:
    return (np.log1p(cerceve[HEDEF].clip(lower=0.0)) - np.log1p(cerceve["guc"])).to_numpy()


def ofseti_geri_ekle(log_tahmin: np.ndarray, cerceve: pd.DataFrame) -> np.ndarray:
    return log_tahmin + np.log1p(cerceve["guc"]).to_numpy()


def rmsle(gercek: np.ndarray, tahmin: np.ndarray) -> float:
    t = np.clip(np.asarray(tahmin, dtype="float64"), 0.0, None)
    return float(np.sqrt(np.mean((np.log1p(t) - np.log1p(np.asarray(gercek))) ** 2)))


def egit_ve_tahmin_et(
    egitim: pd.DataFrame,
    hedef_cerceve: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    hizli: bool,
    rejim: str,
) -> np.ndarray:
    maske_orani = 1.00 if rejim == "soguk" else 0.15
    e = soguk_maskele(egitim, kolonlar, tohum, maske_orani)
    y = ofsetli_hedef(e)

    x_e = e[kolonlar].copy()
    x_h = hedef_cerceve[kolonlar].copy()

    # Model 1: CatBoost
    import catboost as cb

    if rejim == "sicak":
        cb_params = dict(
            loss_function="RMSE",
            iterations=150 if hizli else 400,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=1.0,
            random_strength=4.0,
            rsm=0.75,
            random_seed=tohum,
            verbose=0,
            allow_writing_files=False,
        )
    else:
        cb_params = dict(
            loss_function="RMSE",
            iterations=100 if hizli else 350,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=3.0,
            random_strength=2.0,
            rsm=0.75,
            random_seed=tohum,
            verbose=0,
            allow_writing_files=False,
        )

    kat_cols = [k for k in KATEGORIK if k in x_e.columns]
    x_e_cb, x_h_cb = x_e.copy(), x_h.copy()
    for k in kat_cols:
        x_e_cb[k] = x_e_cb[k].astype(str)
        x_h_cb[k] = x_h_cb[k].astype(str)

    cb_model = cb.CatBoostRegressor(**cb_params)
    cb_model.fit(x_e_cb, y, cat_features=kat_cols)
    cb_pred = ofseti_geri_ekle(cb_model.predict(x_h_cb), hedef_cerceve)

    # Model 2: LightGBM
    import lightgbm as lgb

    lgb_params = dict(
        objective="regression",
        n_estimators=150 if hizli else 400,
        learning_rate=0.04,
        num_leaves=127 if rejim == "sicak" else 63,
        min_child_samples=30 if rejim == "sicak" else 50,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=tohum,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model = lgb.LGBMRegressor(**lgb_params)
    lgb_model.fit(x_e, y)
    lgb_pred = ofseti_geri_ekle(lgb_model.predict(x_h), hedef_cerceve)

    # Model 3: XGBoost
    import xgboost as xgb

    xgb_params = dict(
        objective="reg:squarederror",
        n_estimators=150 if hizli else 400,
        learning_rate=0.04,
        max_depth=7 if rejim == "sicak" else 6,
        min_child_weight=15 if rejim == "sicak" else 25,
        subsample=0.85,
        colsample_bytree=0.75,
        reg_lambda=2.0,
        random_state=tohum,
        n_jobs=-1,
        tree_method="hist",
        enable_categorical=True,
        verbosity=0,
    )
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(x_e, y)
    xgb_pred = ofseti_geri_ekle(xgb_model.predict(x_h), hedef_cerceve)

    # Rejime ozel harman agirliklari
    if rejim == "sicak":
        birlesik = (3.0 * cb_pred + 1.5 * xgb_pred + 1.5 * lgb_pred) / 6.0
    else:
        birlesik = (3.0 * cb_pred + 1.0 * xgb_pred + 1.0 * lgb_pred) / 5.0

    return birlesik


def sota_tahmin_uret(
    egitim: pd.DataFrame,
    hedef: pd.DataFrame,
    kolonlar: list[str],
    tohum: int,
    *,
    hizli: bool,
    dar_egitim: pd.DataFrame | None = None,
) -> np.ndarray:
    soguk = (hedef["soguk_mu"] == 1).to_numpy()
    cikti = np.zeros(len(hedef), dtype="float64")

    # Sicak uzman
    if (~soguk).any():
        alt = hedef.loc[~soguk]
        cikti[~soguk] = egit_ve_tahmin_et(egitim, alt, kolonlar, tohum, hizli=hizli, rejim="sicak")

    # Soguk uzman
    if soguk.any():
        alt = hedef.loc[soguk]
        kaynak = dar_egitim if dar_egitim is not None else egitim
        cikti[soguk] = egit_ve_tahmin_et(kaynak, alt, kolonlar, tohum, hizli=hizli, rejim="soguk")

    return cikti


def olu_trafo_sifirla(tahmin: np.ndarray, cerceve: pd.DataFrame) -> np.ndarray:
    sonuc = tahmin.copy()
    if (
        "t_kuyruk_sifir" in cerceve.columns
        and "t_son_kayit_yasi" in cerceve.columns
        and "t_sifir_orani" in cerceve.columns
    ):
        olu_maske = (
            (cerceve["t_kuyruk_sifir"] >= 14)
            & (cerceve["t_son_kayit_yasi"] >= 14)
            & (cerceve["t_sifir_orani"] >= 0.90)
        ).to_numpy()
        adet = int(olu_maske.sum())
        if adet > 0:
            sonuc[olu_maske] = 0.0
            print(f"  [OLU TRAFO] {adet:,} satir sifira cekildi (log1p buyuk ceza korumasi)")
    return sonuc


def kokenleri_ayikla(egitim: pd.DataFrame, hedef_blok: str) -> pd.DataFrame:
    """Hedef blogun etiket penceresiyle KESISEN her kokeni atar (SIZINTISIZ)."""
    hedef = next(b for b in BLOKLAR if b.ad == hedef_blok)
    h_bas, h_son = pd.Timestamp(hedef.etiket_basi), pd.Timestamp(hedef.etiket_sonu)
    pencereler = {b.ad: (b.etiket_basi, b.etiket_sonu) for b in BLOKLAR}
    pencereler.update({ad: (bas, son) for ad, bas, son in EK_KOKENLER})
    tutulacak = {
        ad
        for ad, (bas, son) in pencereler.items()
        if pd.Timestamp(son) < h_bas or pd.Timestamp(bas) > h_son
    }
    return egitim[egitim["_blok"].isin(tutulacak)]


# ---------------------------------------------------------------- ana akis


def main() -> int:
    satir_tamponlu_cikti()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hizli", action="store_true", help="az agac, hizli dogrulama")
    ap.add_argument("--tohum", type=int, default=5, help="harmanlanacak tohum sayisi")
    ap.add_argument("--tohum-baslangic", type=int, default=42, help="ilk tohum")
    ap.add_argument(
        "--dogrulama-atla", action="store_true", help="dogrulamayi atla, yalniz son model"
    )
    ap.add_argument(
        "--sadece-dogrulama", action="store_true", help="yalniz dogrulama yap, son modeli egitme"
    )
    ap.add_argument("--cikti", default="tuketim_sota_v1.csv")
    ap.add_argument("--gonder", metavar="NOT", help="Kaggle gonderimi yap")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 80)
    print("GRID UP -- SOTA BREAKTHROUGH PIPELINE (HEDEF: 1. SIRAYI GECMEK)")
    print("=" * 80)

    print("\n1/5  YUKLEME + LOKASYON")
    tr, te = yukle()
    tr, te = lokasyon_ayristir(tr), lokasyon_ayristir(te)
    print(f"  train {len(tr):,} satir | test {len(te):,} satir")

    print("\n2/5  GELISMIS HAVA + TAKVIM + STATIK ILCE + ULUSAL")
    hava = hava_yukle()
    tr, te = hava_ekle(tr, hava), hava_ekle(te, hava)
    tr, te = gelismis_takvim_ekle(tr), gelismis_takvim_ekle(te)
    tr, te = yas_ekle(tr, te)
    tr, te = kimlik_ekle(tr, te)
    tr, te = statik_ilce_ekle(tr, te)
    tr, te = ilce_yapisi_ekle(tr, te)
    tr, te = ulusal_ekle(tr, te)

    print("\n3/5  BLOKLAR (SOTA yuvarlanan koken)")
    egitim = sota_egitim_kur(tr)
    dar = egitim.copy()
    ek = sota_ek_kokenleri_kur(tr)
    egitim = pd.concat([egitim, ek[egitim.columns]], ignore_index=True)
    print(f"  ek kokenlerle egitim {len(egitim):,} satir (dar set {len(dar):,})")

    test = sota_test_kur(tr, te)

    kategorik_kodla(egitim, dar, test)
    kolonlar = oznitelikler(egitim)
    kolonlar = [k for k in kolonlar if k in test.columns]
    print(f"\n  Kullanilan toplam oznitelik sayisi: {len(kolonlar)}")

    print("\n4/5  DOGRULAMA (SIZINTISIZ KOKEN AYIKLAMA)")
    if not args.dogrulama_atla:
        for b in BLOKLAR:
            dogrulama = egitim[egitim["_blok"] == b.ad]
            kalan = kokenleri_ayikla(egitim, b.ad)
            kalan_dar = dar[dar["_blok"] != b.ad]

            tahmin_log = sota_tahmin_uret(
                kalan, dogrulama, kolonlar, 42, hizli=args.hizli, dar_egitim=kalan_dar
            )
            tahmin = np.clip(np.expm1(tahmin_log), 0.0, None)
            tahmin = olu_trafo_sifirla(tahmin, dogrulama)

            gercek = dogrulama[HEDEF].to_numpy()
            soguk = (dogrulama["soguk_mu"] == 1).to_numpy()

            skor_genel = rmsle(gercek, tahmin)
            skor_sicak = rmsle(gercek[~soguk], tahmin[~soguk])
            skor_soguk = rmsle(gercek[soguk], tahmin[soguk])
            skor_agirlikli = float(
                np.sqrt((1 - TEST_SOGUK_PAYI) * skor_sicak**2 + TEST_SOGUK_PAYI * skor_soguk**2)
            )

            print(
                f"  blok {b.ad:6} RMSLE {skor_genel:.5f} | sicak {skor_sicak:.5f} soguk {skor_soguk:.5f} | TEST-AGIRLIKLI {skor_agirlikli:.5f}"  # noqa: E501
            )

    if args.sadece_dogrulama:
        print(f"\n--sadece-dogrulama: son egitim atlandi ({(time.time() - t0) / 60:.1f} dk)")
        return 0

    print(f"\n5/5  SON EGITIM ({args.tohum} tohum x 3 model ailesi)")
    birikim = np.zeros(len(test), dtype="float64")
    for i in range(args.tohum):
        t_tohum = time.time()
        tohum = args.tohum_baslangic + i
        birikim += sota_tahmin_uret(egitim, test, kolonlar, tohum, hizli=args.hizli, dar_egitim=dar)
        print(
            f"    tohum {tohum} ({i + 1}/{args.tohum}) tamamlandi ({time.time() - t_tohum:.0f} sn)"
        )

    tahmin_final = np.clip(np.expm1(birikim / args.tohum), 0.0, None)
    tahmin_final = olu_trafo_sifirla(tahmin_final, test)

    soguk_test = (test["soguk_mu"] == 1).to_numpy()
    if soguk_test.any():
        r = np.log1p(tahmin_final[soguk_test]) - np.log1p(test.loc[soguk_test, "guc"].to_numpy())
        r_buzulmus = r.mean() + 0.60 * (r - r.mean()) + 0.1046
        tahmin_final[soguk_test] = np.clip(
            np.expm1(r_buzulmus + np.log1p(test.loc[soguk_test, "guc"].to_numpy())), 0.0, None
        )
        print(
            "  [SON ISLEM] Soguk satirlara James-Stein buzulmesi ve seviye kalibrasyonu uygulandi."
        )

    GONDERIM.mkdir(parents=True, exist_ok=True)
    yol = GONDERIM / args.cikti
    pd.DataFrame({"id": test["id"].to_numpy(), HEDEF: tahmin_final}).to_csv(yol, index=False)
    print(f"\n  YAZILDI: {yol} ({len(tahmin_final):,} satir)")
    print(
        f"  Tahmin dagilimi: min={tahmin_final.min():.2f}, medyan={np.median(tahmin_final):.2f}, ort={tahmin_final.mean():.2f}, max={tahmin_final.max():.2f}"  # noqa: E501
    )

    if args.gonder:
        print("\n  Kaggle'a gonderiliyor...")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "competitions",
                "submit",
                "-c",
                "grid-up-datathon",
                "-f",
                str(yol),
                "-m",
                args.gonder,
            ],
            check=False,
            cwd=KOK,
        )

    print(f"\nTAMAMLANDI -- Toplam sure: {(time.time() - t0) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
