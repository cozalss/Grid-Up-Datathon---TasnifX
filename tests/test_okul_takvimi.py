"""MEB okul takvimi feature testleri.

Tarihler 2026-08-15'te MEB duyurulari ve basin arsiviyle dogrulandi
(kaynak listesi: ``gridup/features/school.py`` docstring'i). Testler uc
sinif iddiayi zorlar:

  1. YAPISAL: MEB takviminin degismez desenleri (acilis hep pazartesi,
     donem sonu hep cuma, ara tatil hep 5 is gunu). Bir kayit yanlis
     girilirse buradan yakalanir.
  2. NOKTASAL: dogrulanmis tekil gunler (deprem ertelemesi, birlesik
     bayram aralari, COVID kuyrugu).
  3. SOZLESME: girdi degistirilmez, satir sayisi/sirasi korunur --
     tests/test_sozlesme.py'nin FEATURE_MODULLERI listesine "school"
     eklenene kadar ayni guvenceyi burada tasiyoruz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridup.features.school import (
    COVERAGE_END,
    COVERAGE_START,
    MISSING_OPENING_DISTANCE,
    SCHOOL_DAY_TYPES,
    SCHOOL_YEARS,
    add_school_calendar_features,
    school_calendar,
)


def _frame(dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"tarih": pd.to_datetime(dates)})


def _tur(gun: str) -> str:
    takvim = school_calendar(gun, gun)
    return str(takvim["tatil_turu"].iloc[0])


class TestYapisalDesenler:
    """MEB takviminin degismezleri -- kayit yanlis girilirse burada patlar."""

    def test_tum_donem_acilislari_pazartesi(self):
        # 2021-2026 arasi ISTISNASIZ: guz baslangici da ikinci donem
        # baslangici da (deprem ertelemesi 2023-02-20 dahil) pazartesidir.
        for kayit in SCHOOL_YEARS:
            for baslangic, _ in kayit["donemler"]:
                gun = pd.Timestamp(baslangic)
                assert gun.dayofweek == 0, f"{kayit['ad']}: {baslangic} pazartesi degil"

    def test_tum_donem_sonlari_cuma(self):
        for kayit in SCHOOL_YEARS:
            for _, bitis in kayit["donemler"]:
                gun = pd.Timestamp(bitis)
                assert gun.dayofweek == 4, f"{kayit['ad']}: {bitis} cuma degil"

    def test_aralar_bes_is_gunu_pazartesi_cuma(self):
        for kayit in SCHOOL_YEARS:
            for baslangic, bitis in kayit["aralar"]:
                b, s = pd.Timestamp(baslangic), pd.Timestamp(bitis)
                assert b.dayofweek == 0, f"{kayit['ad']}: ara {baslangic} pazartesi degil"
                assert s.dayofweek == 4, f"{kayit['ad']}: ara {bitis} cuma degil"
                assert (s - b).days == 4, f"{kayit['ad']}: ara 5 gun degil"

    def test_bes_tam_yil_ve_bir_kuyruk_kapsaniyor(self):
        adlar = [kayit["ad"] for kayit in SCHOOL_YEARS]
        assert adlar == [
            "2020-2021",
            "2021-2022",
            "2022-2023",
            "2023-2024",
            "2024-2025",
            "2025-2026",
        ]
        # 2019'dan beri her tam yilda iki ara tatil var; kuyruk kaydinda yok.
        for kayit in SCHOOL_YEARS[1:]:
            assert len(kayit["aralar"]) == 2, f"{kayit['ad']}: iki ara tatil bekleniyor"

    def test_okul_hicbir_hafta_sonu_acik_degil(self):
        takvim = school_calendar(COVERAGE_START, COVERAGE_END)
        hafta_sonu = takvim["tarih"].dt.dayofweek >= 5
        assert not (takvim.loc[hafta_sonu, "okul_acik"] == 1).any()

    def test_tam_kapsam_dagilimi(self):
        """Docstring'deki OLCULDU tablosunun kaynagi -- sayilar kayarsa
        belge de kod da ayni anda yakalanir."""
        takvim = school_calendar(COVERAGE_START, COVERAGE_END)
        sayim = takvim["tatil_turu"].value_counts()
        assert len(takvim) == 1949
        assert sayim["yok"] == 945
        assert sayim["yaz"] == 482
        assert sayim["hafta_sonu"] == 378
        assert sayim["yariyil"] == 94
        assert sayim["ara"] == 50
        assert sayim["bilinmiyor"] == 0


class TestSchoolCalendar:
    def test_kolonlar_ve_satir_sayisi(self):
        takvim = school_calendar("2023-09-01", "2023-09-30")
        assert list(takvim.columns) == ["tarih", "okul_acik", "tatil_turu"]
        assert len(takvim) == 30

    def test_donem_ici_hafta_ici_acik(self):
        takvim = school_calendar("2023-10-04", "2023-10-04")
        assert takvim["okul_acik"].iloc[0] == 1
        assert _tur("2023-10-04") == "yok"

    def test_donem_ici_hafta_sonu(self):
        assert _tur("2023-10-07") == "hafta_sonu"

    def test_kasim_ara_tatili(self):
        # 15-19 Kasim 2021: okul kapali ama resmi tatil DEGIL -- bu modulun
        # var olma sebebi olan hafta.
        assert _tur("2021-11-16") == "ara"

    def test_nisan_ara_tatili_2022(self):
        assert _tur("2022-04-13") == "ara"

    def test_yariyil(self):
        assert _tur("2022-01-26") == "yariyil"

    def test_yariyil_icindeki_hafta_sonu_yariyil_sayilir(self):
        # 22 Ocak 2022 cumartesi: karneden sonraki ilk gun. Kesintisiz kapali
        # blogun parcasi -- hafta_sonu degil yariyil (gerekce: docstring).
        assert _tur("2022-01-22") == "yariyil"

    def test_yaz(self):
        assert _tur("2022-07-15") == "yaz"

    def test_eylul_baslangicindan_onceki_gunler_yaz(self):
        assert _tur("2021-09-05") == "yaz"
        assert _tur("2021-09-06") == "yok"

    def test_ders_yilinin_son_gunu_acik(self):
        # 26 Haziran 2026 cuma: karne gunu, okul acik.
        assert _tur("2026-06-26") == "yok"
        assert _tur("2026-06-29") == "yaz"


class TestDepremYili:
    """2022-2023: 6 Subat depremi Ege illerinde ikinci donemi 20 Subat'a
    erteledi. Resmi takvimi degil YASANANI kodluyoruz."""

    def test_resmi_takvimde_acik_olacak_gun_kapali(self):
        # 13 Subat 2023 pazartesi: resmi takvime gore donem basalmisti,
        # gercekte tum yurtta okullar kapaliydi.
        assert _tur("2023-02-13") == "yariyil"

    def test_yirmi_subat_acilis(self):
        takvim = school_calendar("2023-02-20", "2023-02-20")
        assert takvim["okul_acik"].iloc[0] == 1

    def test_nisan_2023_arasi_iptal_edilmedi(self):
        # 17-21 Nisan 2023 uygulandi ve Ramazan Bayrami (21-23 Nis) ile
        # birlesti -- iptal soylentisi dogrulanmadi.
        assert _tur("2023-04-18") == "ara"

    def test_bahar_2025_ve_2026_aralari_bayramla_birlesik(self):
        assert _tur("2025-04-02") == "ara"  # 31 Mar - 4 Nis 2025
        assert _tur("2026-03-18") == "ara"  # 16-20 Mar 2026


class TestCovidKuyrugu:
    def test_2021_mayis_resmi_takvimde_acik(self):
        # DIKKAT: fiilen buyuk olcude uzaktan egitimdi; kolon resmi takvimi
        # kodlar (docstring'de isaretli). 20 Mayis 2021 persembe.
        assert _tur("2021-05-20") == "yok"

    def test_2021_yaz_baslangici(self):
        # 18 Haziran 2021 cuma karne; 21 Haziran pazartesi artik yaz.
        assert _tur("2021-06-18") == "yok"
        assert _tur("2021-06-21") == "yaz"


class TestKapsamSiniri:
    def test_kapsam_oncesi_hata(self):
        with pytest.raises(ValueError, match="kapsam"):
            school_calendar("2020-01-01", "2021-06-01")

    def test_kapsam_sonrasi_hata(self):
        # 2026-2027 takvimi modulde YOK -- sessizce yaz uydurmak yerine hata.
        with pytest.raises(ValueError, match="kapsam"):
            school_calendar("2026-01-01", "2027-01-01")

    def test_ters_aralik_hata(self):
        with pytest.raises(ValueError, match="sonra"):
            school_calendar("2023-05-10", "2023-05-01")

    def test_nat_hata(self):
        with pytest.raises(ValueError, match="NaT"):
            school_calendar(pd.NaT, "2023-05-01")


class TestAddSchoolCalendarFeatures:
    def test_uc_kolon_ekleniyor(self):
        sonuc = add_school_calendar_features(_frame(["2023-10-04"]), "tarih")
        for kolon in ("okul_acik_mi", "okul_tatil_turu", "okul_acilisa_gun"):
            assert kolon in sonuc.columns, kolon

    def test_girdi_frame_degistirilmiyor(self):
        frame = _frame(["2023-10-04", "2022-07-15"])
        onceki = frame.copy(deep=True)
        add_school_calendar_features(frame, "tarih")
        pd.testing.assert_frame_equal(frame, onceki)

    def test_yeni_frame_donuyor_satir_sayisi_ve_sira_korunuyor(self):
        frame = _frame(["2022-07-15", "2023-10-04", "2021-11-16"])
        sonuc = add_school_calendar_features(frame, "tarih")
        assert sonuc is not frame
        assert len(sonuc) == len(frame)
        assert list(sonuc["tarih"]) == list(frame["tarih"])

    def test_acik_gunde_acilisa_sifir(self):
        sonuc = add_school_calendar_features(_frame(["2023-10-04"]), "tarih")
        assert sonuc["okul_acik_mi"].iloc[0] == 1
        assert sonuc["okul_acilisa_gun"].iloc[0] == 0

    def test_hafta_sonu_mesafeleri(self):
        # 7 Ekim 2023 cumartesi -> pazartesiye 2; 8 Ekim pazar -> 1.
        sonuc = add_school_calendar_features(_frame(["2023-10-07", "2023-10-08"]), "tarih")
        assert list(sonuc["okul_acilisa_gun"]) == [2, 1]

    def test_yariyil_baslangicinda_mesafe(self):
        # 22 Ocak 2022 -> ikinci donem 7 Subat 2022: 16 gun.
        sonuc = add_school_calendar_features(_frame(["2022-01-22"]), "tarih")
        assert sonuc["okul_acilisa_gun"].iloc[0] == 16

    def test_ara_tatil_cumasinda_mesafe(self):
        # 19 Kasim 2021 cuma (ara) -> pazartesi 22 Kasim: 3 gun.
        sonuc = add_school_calendar_features(_frame(["2021-11-19"]), "tarih")
        assert sonuc["okul_acilisa_gun"].iloc[0] == 3

    def test_yaz_ortasinda_mesafe(self):
        # 1 Agustos 2022 -> acilis 12 Eylul 2022: 42 gun.
        sonuc = add_school_calendar_features(_frame(["2022-08-01"]), "tarih")
        assert sonuc["okul_acilisa_gun"].iloc[0] == 42

    def test_2026_yaz_kuyrugu_sentinel(self):
        # Kapsam icinde sonraki acilis yok (Eylul 2026 kapsam disi) --
        # sahte bir mesafe uydurmak yerine sentinel.
        sonuc = add_school_calendar_features(_frame(["2026-07-15"]), "tarih")
        assert sonuc["okul_acilisa_gun"].iloc[0] == MISSING_OPENING_DISTANCE

    def test_kapsam_disi_bilinmiyor_ve_uyari(self, capsys):
        sonuc = add_school_calendar_features(_frame(["2019-05-05"]), "tarih")
        satir = sonuc.iloc[0]
        assert str(satir["okul_tatil_turu"]) == "bilinmiyor"
        assert satir["okul_acik_mi"] == 0
        assert satir["okul_acilisa_gun"] == MISSING_OPENING_DISTANCE
        assert "UYARI" in capsys.readouterr().out

    def test_bozuk_tarih_bilinmiyor(self):
        sonuc = add_school_calendar_features(
            pd.DataFrame({"tarih": ["2023-10-04", "bozuk-tarih"]}), "tarih"
        )
        assert str(sonuc["okul_tatil_turu"].iloc[1]) == "bilinmiyor"
        assert sonuc["okul_acik_mi"].iloc[1] == 0

    def test_kapsam_icinde_uyari_yok(self, capsys):
        add_school_calendar_features(_frame(["2023-10-04"]), "tarih")
        assert "UYARI" not in capsys.readouterr().out

    def test_dtype_lar_kompakt(self):
        sonuc = add_school_calendar_features(
            _frame(["2023-10-04", "2022-07-15", "2019-01-01"]), "tarih"
        )
        assert sonuc["okul_acik_mi"].dtype == np.int8
        assert sonuc["okul_acilisa_gun"].dtype == np.int16
        assert isinstance(sonuc["okul_tatil_turu"].dtype, pd.CategoricalDtype)
        assert tuple(sonuc["okul_tatil_turu"].cat.categories) == SCHOOL_DAY_TYPES

    def test_prefix_degistirilebilir(self):
        sonuc = add_school_calendar_features(_frame(["2023-10-04"]), "tarih", prefix="mektep")
        assert "mektep_acik_mi" in sonuc.columns
        assert "okul_acik_mi" not in sonuc.columns

    def test_kolon_yoksa_keyerror(self):
        with pytest.raises(KeyError, match="tarih"):
            add_school_calendar_features(pd.DataFrame({"gun": []}), "tarih")

    def test_tekrarli_indeks_karismiyor(self):
        """pd.concat([train, test]) sonrasi tekrarli indeks -- solar'daki
        P0 regresyonun (etiket hizalamasiyla sessiz takas) okul versiyonu."""
        parca = _frame(["2023-10-04", "2022-07-15"])
        panel = pd.concat([parca, parca])
        assert not panel.index.is_unique
        sonuc = add_school_calendar_features(panel, "tarih")
        assert list(sonuc["okul_acik_mi"]) == [1, 0, 1, 0]

    def test_gercek_veri_araligi_tamamen_kapsaniyor(self, capsys):
        """Prova paneli 2021-05..2022-08 -- tek bir 'bilinmiyor' satiri bile
        cikmamali, aksi halde en degerli veride feature deliktir."""
        gunler = pd.date_range("2021-05-01", "2022-08-31", freq="D")
        sonuc = add_school_calendar_features(pd.DataFrame({"tarih": gunler}), "tarih")
        assert (sonuc["okul_tatil_turu"] != "bilinmiyor").all()
        assert "UYARI" not in capsys.readouterr().out
