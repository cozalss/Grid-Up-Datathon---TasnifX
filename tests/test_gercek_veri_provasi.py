"""GERCEK GDZ VERISININ ORTAYA CIKARDIGI BULGULAR.

NEDEN AYRI DOSYA
----------------
Bu repodaki her sey sentetik veride dogrulandi. Sentetik veri, onu yazan
kisinin AKLINDAKI hatalari icerir -- aklina gelmeyenleri icermez.

Buradaki testler, 68.257 satirlik GERCEK bir GDZ kesinti kaydi uzerinde
hat kosturuldugunda ortaya cikan kusurlara karsilik gelir. Sentetik veride
hicbiri gorunmemisti.

Kaynak: kaggle.com/datasets/tmlalper/manisa-izmir-plansiz-elektrik-kesintileri
        (distributioncompanyname = GDZ_EDAS, Izmir + Manisa, 2021-05..2022-08)
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.turkish import diagnose_join, join_key, strip_qualifier

# --------------------------------------------------------------------------
# Niteleyici eki: "Koprubasi / Manisa"
# --------------------------------------------------------------------------


def test_niteleyici_ekli_ilce_adi_sade_hale_indirgeniyor():
    """OLCULDU: gercek veride 'Köprübaşı / Manisa' yaziyor, referans tablosu
    yalin 'Köprübaşı' tutuyor.

    Ayni ilce adi Turkiye'de birden fazla ilde bulunabildigi icin (Koprubasi
    hem Manisa'da hem Trabzon'da) kaynak sistemler adi il ile niteler. Bu tek
    ilcede 284 kayit vardi ve hava/komsu join'inde SESSIZCE duserdi.
    """
    assert strip_qualifier("Köprübaşı / Manisa") == "Köprübaşı"
    assert strip_qualifier("Merkez (Aydın)") == "Merkez"
    assert strip_qualifier("Seydikemer - Muğla") == "Seydikemer"
    # Nitelenmemis adlar DEGISMEZ.
    assert strip_qualifier("Bornova") == "Bornova"
    assert strip_qualifier("Şehzadeler") == "Şehzadeler"


def test_bos_sonuc_uretmiyor():
    """Ayirici basta ise ad oldugu gibi kalmali -- bos anahtar her seyi bozar."""
    assert strip_qualifier("/ Manisa") == "/ Manisa"
    assert strip_qualifier("(Aydın)") == "(Aydın)"


def test_diagnose_join_niteleyiciyle_kurtarilabilenleri_raporluyor():
    """Tani, kullaniciya HANGI donusumu uygulayacagini soylemeli.

    OLCULDU (gercek veri, 47 ilce vs 96 ilcelik referans):
        normalize eslesme : 46/47
        eslesmeyen        : ['koprubasi / manisa']
        ek atilinca       : {'koprubasi / manisa': 'koprubasi'}
        satir kazanci     : 67.973 -> 68.257  (+284)
    """
    veri = ["Bornova", "Köprübaşı / Manisa", "Menemen"]
    referans = ["Bornova", "Köprübaşı", "Menemen", "Çeşme"]

    tani = diagnose_join(veri, referans)
    assert tani["qualifier_recoverable"] == {"koprubasi / manisa": "koprubasi"}
    # Kurtarilabilen anahtar 'eslesmeyen' listesinde TEKRAR gorunmemeli --
    # aksi halde kullanici cozumu elinde dururken sorunu ariyor olur.
    assert tani["left_only"] == []


def test_join_key_degistirilmedi():
    """``strip_qualifier`` AYRI bir adimdir; ``join_key`` sadelestirmez.

    Kesme islemi ("/" oncesini al) her veri setinde dogru olmayabilir --
    karari kullanici verir. join_key'i degistirmek, bu kararı sessizce
    herkes adina almak olurdu.
    """
    assert join_key("Köprübaşı / Manisa") == "koprubasi / manisa"
    assert join_key("Köprübaşı") == "koprubasi"


# --------------------------------------------------------------------------
# Olay kaydi -> panel: gercek veri sekli
# --------------------------------------------------------------------------


def test_saat_damgali_olay_kaydi_gunluk_panele_kutle_kaybetmeden_oturuyor():
    """Gercek verinin sekli: her satir bir kesinti, saat damgali, seyrek.

    Gercek kosuda olculdu: 47 ilce x 472 gun = 22.184 satir panel, hedef
    kutlesi %100.00 korundu, doluluk %65.2.
    """
    from gridup.panel import PANEL_FLAG_COLUMN, build_panel, panel_coverage

    kayitlar = []
    for ilce in ("bornova", "menemen", "aliaga"):
        for gun in pd.date_range("2024-01-01", periods=30, freq="D"):
            # Gunlerin ucte ikisinde kayit yok (seyrek), olan gunlerde saat damgasi
            if gun.day % 3 == 0:
                continue
            kayitlar.append(
                {
                    "ilce_key": ilce,
                    "gun": gun + pd.Timedelta(hours=gun.day % 24),
                    "kesinti_dk": float(gun.day * 7),
                }
            )
    olaylar = pd.DataFrame(kayitlar)

    kapsam = panel_coverage(olaylar, entity_columns=["ilce_key"], time_column="gun")
    assert kapsam["coverage"] <= 1.0

    panel = build_panel(
        olaylar,
        entity_columns=["ilce_key"],
        time_column="gun",
        value_columns=["kesinti_dk"],
        verbose=False,
    )
    # Izgara ILK kayittan SON kayda uzanir; son gun (30) atlandigi icin
    # 1..29 = 29 gun x 3 ilce.
    beklenen_gun = (
        olaylar["gun"].dt.normalize().max() - olaylar["gun"].dt.normalize().min()
    ).days + 1
    assert len(panel) == 3 * beklenen_gun
    assert panel["kesinti_dk"].sum() == pytest.approx(olaylar["kesinti_dk"].sum())
    # Asil sozlesme: saat damgali kayitlarin HEPSI gunluk izgaraya oturmali.
    assert int((panel[PANEL_FLAG_COLUMN] == 0).sum()) == len(olaylar)
