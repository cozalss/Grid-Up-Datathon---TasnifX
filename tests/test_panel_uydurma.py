"""SIZINTI AVI TUR 5 -- PANEL UYDURMA sinifi (``panel.py``).

NEDEN AYRI BIR SIZINTI SINIFI
-----------------------------
Onceki turlar "yanlis satirlar birbirini goruyor" turu sizintilara bakti.
Bu sinif farklidir: **satirlarin kendisi uydurmadir.** Panel kurma, olay
kaydini varlik x zaman izgarasina oturtur; izgara ile kayit hizalanmazsa
gercek gozlemler sessizce dusup yerlerine sifir dolgulu sahte satirlar gecer.

Skor bozulmaz, hata cikmaz, CV temiz gorunur -- model yalnizca sifir ogrenir.

Her test ``t5_repro.py`` ile ONCE-SONRA olculmus bir bulguya karsilik gelir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gridup.panel import PANEL_FLAG_COLUMN, build_panel, panel_coverage

KOK = Path(__file__).resolve().parents[1]


def _olay_kaydi(*, saatli: bool, gunluk_olay: int = 1, tohum: int = 0) -> pd.DataFrame:
    """20 ilce x 60 gun uzerinde seyrek olay kaydi."""
    rng = np.random.default_rng(tohum)
    satirlar = []
    for ilce in [f"ilce_{i:02d}" for i in range(20)]:
        for gun in pd.date_range("2024-01-01", periods=60, freq="D"):
            for _ in range(gunluk_olay if gunluk_olay > 1 else int(rng.random() < 0.3)):
                damga = (
                    gun + pd.Timedelta(hours=int(rng.integers(0, 24)),
                                       minutes=int(rng.integers(0, 60)))
                    if saatli
                    else gun
                )
                satirlar.append(
                    {
                        "ilce": ilce,
                        "tarih": damga,
                        "sure_dk": float(rng.integers(10, 200)),
                        "sebep": str(rng.choice(["kablo", "trafo", "hava"])),
                    }
                )
    return pd.DataFrame(satirlar)


# --------------------------------------------------------------------------
# B9 -- hedef kutlesi
# --------------------------------------------------------------------------


def test_saat_damgali_kayitta_hedef_kutlesi_korunuyor():
    """OLCULDU: saat damgali olay kaydinda hedefin %90.4'u kayboluyordu.

        ham 332 kayit,  sure_dk toplami 35.576
        panel 1.200 satir, sure_dk toplami 3.416   -> 27 satir sifir disi

    Sebep: ``freq="D"`` izgarasi gece yarilarindan olusuyor, kayitlar ise
    14:23 gibi damgalar tasiyor; ``merge`` neredeyse hicbirini bulamiyordu.
    Hata da uyari da yoktu -- model tamamen sifir ogrenirdi.
    """
    olaylar = _olay_kaydi(saatli=True)
    panel = build_panel(
        olaylar, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    assert panel["sure_dk"].sum() == pytest.approx(olaylar["sure_dk"].sum())
    assert int((panel["sure_dk"] > 0).sum()) == len(olaylar)


def test_gunde_birden_cok_olay_toplanarak_korunuyor():
    """Ayni gunun farkli saatlerindeki kayitlar tek satira TOPLANMALI."""
    olaylar = _olay_kaydi(saatli=True, gunluk_olay=3)
    panel = build_panel(
        olaylar, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    assert len(panel) == 20 * 60
    assert panel["sure_dk"].sum() == pytest.approx(olaylar["sure_dk"].sum())


# --------------------------------------------------------------------------
# B10 -- doluluk olcumu
# --------------------------------------------------------------------------


def test_panel_coverage_yuzde_yuzu_asamaz():
    """OLCULDU: saat damgali, gunde ~3 olayli veride DOLULUK %304.8 cikiyordu.

    ``day_one`` buna bakip "doluluk >= %95, panel gerekmez" diyordu; oysa ayni
    veride panel kurulsa hedefin %99.8'i kaybolacakti. Yani hatali olcum,
    hatali kurulumu GIZLIYORDU.
    """
    olaylar = _olay_kaydi(saatli=True, gunluk_olay=3)
    kapsam = panel_coverage(olaylar, entity_columns=["ilce"], time_column="tarih")
    assert kapsam["coverage"] <= 1.0
    assert kapsam["coverage"] == pytest.approx(1.0)


def test_panel_coverage_seyrek_veride_gercegi_soyluyor():
    """Ters yon: gercekten seyrek veride doluluk dusuk cikmali."""
    olaylar = _olay_kaydi(saatli=True)
    kapsam = panel_coverage(olaylar, entity_columns=["ilce"], time_column="tarih")
    assert 0.2 < kapsam["coverage"] < 0.4
    assert kapsam["expected_rows"] == 20 * 60


# --------------------------------------------------------------------------
# B11 -- metin kolonlari
# --------------------------------------------------------------------------


def test_metin_kolonlari_panelden_dusmuyor():
    """OLCULDU: girdi ['ilce','tarih','sure_dk','sebep'] iken cikti
    ['ilce','tarih','sure_dk','_dolduruldu'] oluyordu -- 'sebep' sessizce
    yok oluyordu. Arizanin SEBEBI, ariza probleminde en degerli kolondur."""
    olaylar = _olay_kaydi(saatli=False)
    panel = build_panel(
        olaylar, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    assert "sebep" in panel.columns
    # Gercek satirlarda deger var, uydurulan satirlarda NaN -- sifir DEGIL,
    # cunku o gun bir gozlem yoktur.
    gercek = panel[panel[PANEL_FLAG_COLUMN] == 0]
    uydurma = panel[panel[PANEL_FLAG_COLUMN] == 1]
    assert gercek["sebep"].notna().all()
    assert uydurma["sebep"].isna().all()


# --------------------------------------------------------------------------
# B14 -- NaN varlik anahtari
# --------------------------------------------------------------------------


def test_nan_varlik_anahtari_hayalet_satir_uretmiyor(capsys):
    """OLCULDU: NaN anahtarli 50 satirin hedefi (5.483 dk) sessizce dusuyor,
    ustune 60 satirlik NaN anahtarli HAYALET bir varlik uretiliyordu."""
    olaylar = _olay_kaydi(saatli=False)
    bozuk = olaylar.copy()
    bozuk.loc[bozuk.index[:50], "ilce"] = np.nan

    panel = build_panel(
        bozuk, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    assert int(panel["ilce"].isna().sum()) == 0
    # Kayip SESSIZ olmamali.
    assert "eksik deger" in capsys.readouterr().out


def test_hedef_kolonu_bos_olan_gercek_satir_sentetik_sayilmiyor():
    """Bayrak eskiden ilk deger kolonunun NaN olmasindan turetiliyordu, yani o
    kolonu bos olan GERCEK bir kayit da 'uydurma' isaretleniyordu."""
    olaylar = pd.DataFrame(
        {
            "ilce": ["a", "a", "b"],
            "tarih": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-01"]),
            "sure_dk": [np.nan, 5.0, 3.0],
        }
    )
    panel = build_panel(
        olaylar, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    kayit = panel[(panel["ilce"] == "a") & (panel["tarih"] == pd.Timestamp("2024-01-01"))]
    assert int(kayit[PANEL_FLAG_COLUMN].iloc[0]) == 0


# --------------------------------------------------------------------------
# B15 -- doldurma bayragi
# --------------------------------------------------------------------------


def test_doldurma_bayragi_hedefle_ayni_sey_ve_sizinti_raporunda_gorunuyor():
    """OLCULDU: bayrak (hedef==0) ile %100 ortusuyor, Pearson -0.8378 (esik
    ALTINDA) ama Spearman -0.9810. Tur 4'te eklenen Spearman kontrolu onu
    yakaliyor; bu test o zincirin kopmadigini garanti eder."""
    from gridup.validation import leakage_report

    olaylar = _olay_kaydi(saatli=False)
    panel = build_panel(
        olaylar, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    ortusme = (
        panel[PANEL_FLAG_COLUMN] == (panel["sure_dk"] == 0).astype("int8")
    ).mean()
    assert ortusme == 1.0

    rapor = leakage_report(panel.drop(columns=["ilce", "tarih", "sebep"]), "sure_dk")
    assert any(PANEL_FLAG_COLUMN in bulgu for bulgu in rapor["critical"])


# --------------------------------------------------------------------------
# Tur 5'te GUN-1 PROVASINDA cikan iki yeni bulgu
# --------------------------------------------------------------------------


def test_panel_turkce_tarih_kolonunu_dogru_cozuyor():
    """Tur 4'un tarih duzeltmesi panel.py'ye ULASMAMISTI.

    Gercek bir gun-1 kosusunda olculdu (1.091 satir, 120 gunluk veri):

        ONCE : [build_panel] UYARI: 641 satirda gecersiz tarih var
               20 varlik x 339 zaman adimi          <- 120 gunluk veri
        SONRA: uyari yok, 20 varlik x 120 zaman adimi

    Yani panelin ucte ikisi atiliyor, kalani yanlis takvime yayiliyordu --
    ve bu, ambargonun uzerinde calistigi takvimdi.
    """
    rng = np.random.default_rng(3)
    satirlar = []
    for ilce in [f"ilce_{i:02d}" for i in range(5)]:
        for gun in pd.date_range("2024-01-01", periods=120, freq="D"):
            if rng.random() < 0.5:
                damga = gun + pd.Timedelta(hours=int(rng.integers(0, 24)))
                satirlar.append(
                    {
                        "ilce": ilce,
                        # TR bicimi: gun once, ustune saat
                        "tarih": damga.strftime("%d.%m.%Y %H:%M"),
                        "sure_dk": float(rng.integers(10, 200)),
                    }
                )
    olaylar = pd.DataFrame(satirlar)

    panel = build_panel(
        olaylar, entity_columns=["ilce"], time_column="tarih", verbose=False
    )
    assert len(panel) == 5 * 120
    assert panel["tarih"].nunique() == 120
    assert panel["sure_dk"].sum() == pytest.approx(olaylar["sure_dk"].sum())


def test_day_one_kullanicinin_yazdigi_ham_kolon_adini_cozuyor():
    """Belgelenen veri gunu komutu ILK ADIMDA hata veriyordu.

    Kullanici CSV'de gordugu adi yazar (``--time TARIH``); ``read_any`` ise
    kolonlari normalize eder (``tarih``). Ikisi eslesmiyordu:

        HATA: hedef kolon belirlenemedi. Kolonlar: ['id','ilce','tarih',...]

    docs/07 tam olarak "--time TARIH --group ILCE" yazdigi icin bu, yarismanin
    ilk saatinde kaybedilecek dakikalardi.
    """
    kaynak = (KOK / "scripts" / "day_one.py").read_text(encoding="utf-8")
    assert "_kolonu_coz" in kaynak

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gun1_modul", KOK / "scripts" / "day_one.py"
    )
    modul = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(modul)

    frame = pd.DataFrame({"id": [1], "tarih": ["2024-01-01"], "hedef": [1.0]})
    frame.attrs["original_columns"] = {"ID": "id", "TARIH": "tarih", "HEDEF": "hedef"}
    assert modul._kolonu_coz("TARIH", frame) == "tarih"
    assert modul._kolonu_coz("HEDEF", frame) == "hedef"
    assert modul._kolonu_coz("tarih", frame) == "tarih"
    assert modul._kolonu_coz(None, frame) is None


def test_day_one_doldurma_bayragini_feature_listesine_almiyor():
    """Yapisal koruma: bayrak adiyla dislanmali.

    Test kumesi verildiginde zaten elenirdi (test'te o kolon yok), ama test
    verilmediginde listeye giriyordu.
    """
    kaynak = (KOK / "scripts" / "day_one.py").read_text(encoding="utf-8")
    assert "PANEL_FLAG_COLUMN" in kaynak
    assert "drop = {args.target, args.id_column, time_column, PANEL_FLAG_COLUMN}" in kaynak
