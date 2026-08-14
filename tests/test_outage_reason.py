"""Ariza sebebi taksonomisi testleri.

Ornekler GERCEK EPIAS verisinden alindi (17.578 kayit, 724 farkli sebep metni).
Taksonomi bu kume uzerinde %0,88 siniflandirilamayan orani veriyor.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.features.outage_reason import (
    REASON_CODES,
    add_reason_features,
    classify_reason,
    reason_family_report,
)


class TestWeatherFamily:
    """HAVA ailesi en onemlisi: hava feature'lariyla etkilesecek olan bu."""

    @pytest.mark.parametrize(
        "text",
        [
            "Olumsuz Hava Sartlari - Sigorta Arıza",
            "Olumsuz Hava Sartlari - Kesici Arıza",
            "Elverişsiz Hava Koşulları",
            "FIRTINA",
            "Yıldırım Düşmesi",
        ],
    )
    def test_weather_variants_all_map_to_hava(self, text):
        assert classify_reason(text)[1] == "HAVA"

    def test_weather_wins_over_equipment_in_compound_text(self):
        """'Olumsuz Hava - Sigorta Ariza': hava daha ACIKLAYICI bilgidir.

        Hangi parcanin bozuldugu degil, NEDEN bozuldugu modele ogretilmeli --
        cunku hava kolonlariyla etkilesecek olan odur.
        """
        code, family, _ = classify_reason("Olumsuz Hava Sartlari - Sigorta Arıza")
        assert family == "HAVA"
        assert code == REASON_CODES["HAVA"]


class TestNonFaultRecords:
    """Ariza OLMAYAN kayitlar: hedef tanimini bozarlar."""

    def test_scada_switching_is_not_a_fault(self):
        """En sik kayit '-SCADA - MANEVRA' -- bu bir ariza degil, operasyon."""
        _, family, is_fault = classify_reason("-SCADA - MANEVRA")
        assert family == "MANEVRA"
        assert is_fault is False

    def test_debt_disconnection_is_not_a_fault(self):
        _, family, is_fault = classify_reason("Borçtan Kesme - Hattan kesme")
        assert family == "IDARI_KESME"
        assert is_fault is False

    def test_debt_disconnection_does_not_fall_into_cable_family(self):
        """'Hattan kesme' metninde 'hat' gecer -- KABLO'ya dusmemeli.

        Sira onemli: idari kesme ekipman ailelerinden ONCE eslenmeli.
        """
        assert classify_reason("Borçtan Kesme - Hattan kesme")[1] != "KABLO"

    def test_safety_outage_is_not_a_fault(self):
        assert classify_reason("Emniyet Amaçlı Kesinti")[2] is False

    def test_planned_work_is_not_a_fault(self):
        assert classify_reason("Şebeke Çalışması")[2] is False

    def test_new_connection_is_not_a_fault(self):
        assert classify_reason("Yeni Bağlantı - AG-Yeni bağlantı")[2] is False


class TestEquipmentFamilies:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("Kablo Arizasi", "KABLO"),
            ("İletken; AG İletken Kopması", "KABLO"),
            ("HAVAİ HAT ARIZASI", "KABLO"),
            ("Sigorta Atması", "SIGORTA"),
            ("SIGORTA ARIZASI", "SIGORTA"),
            ("Sigorta; NH Sigorta Atma/Değişim", "SIGORTA"),
            ("Kesici; Geçici Arıza", "KESICI"),
            ("TMŞ; TMŞ Açma", "KESICI"),
            ("Klemens / Oksit Arızası", "KLEMENS"),
            ("KUŞ ÇARPMASI", "HAYVAN"),
            ("-3.Şahıs Kaynaklı Hasar - Kablo Kopma/Sıyrılma", "UCUNCU_SAHIS"),
            ("Asiri Yuk - Sigorta Arıza", "ASIRI_YUK"),
            ("Agac ve Dis Etkenler - Sigorta Arıza", "AGAC"),
            ("Malzemelerin Aşınması (Korozyon)", "EKONOMIK_OMUR"),
            ("TEIAS Kesintisi", "UST_SEBEKE"),
        ],
    )
    def test_real_epias_strings_classify_correctly(self, text, expected):
        assert classify_reason(text)[1] == expected


class TestTurkishCaseHandling:
    """Metinler BUYUK, kucuk ve karisik geliyor -- hepsi ayni aileye dusmeli."""

    def test_case_variants_agree(self):
        variants = ["SIGORTA ARIZASI", "Sigorta Arızası", "sigorta arizasi"]
        families = {classify_reason(text)[1] for text in variants}
        assert families == {"SIGORTA"}

    def test_dotted_i_does_not_break_matching(self):
        """'İLETKEN' ham .lower() ile 'i̇letken' olur ve eslesme kacar."""
        assert classify_reason("İLETKEN KOPMASI")[1] == "KABLO"

    def test_accent_variants_agree(self):
        assert classify_reason("Agac Temasi")[1] == classify_reason("Ağaç Teması")[1]


class TestEdgeCases:
    def test_empty_and_none_are_handled(self):
        for value in ("", None, float("nan"), "   "):
            code, family, is_fault = classify_reason(value)
            assert family == "DIGER"
            assert code == 0
            assert is_fault is True

    def test_unknown_text_falls_back_to_diger(self):
        assert classify_reason("Alpek Arızası")[1] == "DIGER"

    def test_missing_reason_recorded_as_unknown(self):
        assert classify_reason("Kesinti Sebebi Girilmemiştir")[1] == "BILINMEYEN"


class TestFrameFeatures:
    @pytest.fixture
    def frame(self):
        return pd.DataFrame(
            {
                "reason": [
                    "Olumsuz Hava Sartlari - Sigorta Arıza",
                    "-SCADA - MANEVRA",
                    "Kablo Arizasi",
                    "KUŞ ÇARPMASI",
                    None,
                ]
            }
        )

    def test_generates_expected_columns(self, frame):
        result = add_reason_features(frame, "reason")
        for column in ("sebep_kod", "sebep_aile", "sebep_ariza", "sebep_hava"):
            assert column in result.columns

    def test_weather_flag_is_isolated(self, frame):
        """sebep_hava ayri bir kolon: en guclu etkilesim bu."""
        result = add_reason_features(frame, "reason")
        assert result["sebep_hava"].tolist() == [1, 0, 0, 0, 0]

    def test_fault_flag_excludes_switching(self, frame):
        result = add_reason_features(frame, "reason")
        assert result["sebep_ariza"].tolist() == [1, 0, 1, 1, 1]

    def test_code_column_is_integer(self, frame):
        result = add_reason_features(frame, "reason")
        assert pd.api.types.is_integer_dtype(result["sebep_kod"])

    def test_input_not_mutated(self, frame):
        before = frame.copy()
        add_reason_features(frame, "reason")
        pd.testing.assert_frame_equal(frame, before)

    def test_missing_column_raises(self, frame):
        with pytest.raises(KeyError):
            add_reason_features(frame, "yok")


class TestFamilyReport:
    def test_report_sums_to_input_length(self):
        reasons = ["Kablo Arizasi"] * 3 + ["-SCADA - MANEVRA"] * 2 + ["Alpek"]
        report = reason_family_report(reasons)
        assert int(report["kayit"].sum()) == 6

    def test_report_marks_switching_as_non_fault(self):
        report = reason_family_report(["-SCADA - MANEVRA"] * 5)
        row = report[report["aile"] == "MANEVRA"].iloc[0]
        assert bool(row["ariza_mi"]) is False

    def test_percentages_add_up(self):
        report = reason_family_report(["Kablo Arizasi", "Sigorta Atması"])
        assert report["yuzde"].sum() == pytest.approx(100.0)
