"""Turkce metin tehlikeleri testleri.

Bu testler bir stil tercihini degil, SESSIZ VERI KAYBINI korumaya alir:
``.lower()`` ile yapilan bir il adi join'i istisna firlatmadan 0 satir doner.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridup.turkish import (
    codepoints,
    diagnose_join,
    has_combining_dot,
    join_key,
    normalize_columns,
    tr_lower,
    tr_sorted,
    tr_upper,
)


class TestDottedI:
    """Noktali/noktasiz i ciftinin dogru eslenmesi."""

    def test_capital_dotted_i_lowercases_to_single_codepoint(self):
        # Arrange / Act
        result = tr_lower("İ")

        # Assert -- cikplak .lower() burada IKI kod noktasi uretir (U+0069 U+0307)
        assert result == "i"
        assert len(result) == 1

    def test_naive_lower_produces_combining_dot(self):
        """Hatanin kendisini belgeleyen test: bu davranis degisirse haberimiz olsun."""
        naive = "İ".lower()
        assert len(naive) == 2
        assert codepoints(naive) == ["U+0069", "U+0307"]
        assert naive != "i"

    def test_capital_dotless_i_lowercases_to_dotless(self):
        assert tr_lower("I") == "ı"
        assert tr_lower("IŞIK") == "ışık"

    def test_province_names_lowercase_correctly(self):
        assert tr_lower("İZMİR") == "izmir"
        assert tr_lower("AYDIN") == "aydın"
        assert tr_lower("MUĞLA") == "muğla"

    def test_upper_round_trips(self):
        for word in ("ışık", "şişli", "izmir", "aydın"):
            assert tr_lower(tr_upper(word)) == word

    def test_has_combining_dot_detects_bad_lowercase(self):
        assert has_combining_dot("İ".lower()) is True
        assert has_combining_dot(tr_lower("İ")) is False


class TestJoinKey:
    """Aksan farkliliklarina ragmen eslesme."""

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            ("MUĞLA", "Mugla"),
            ("İzmir", "IZMIR"),
            ("Aydın", "AYDIN"),
            ("Denizli", "DENİZLİ"),
            ("Çeşme", "Cesme"),
            ("Ödemiş", "Odemis"),
        ],
    )
    def test_diacritic_variants_produce_same_key(self, left, right):
        assert join_key(left) == join_key(right)

    def test_whitespace_is_normalized(self):
        assert join_key("  İzmir  ") == "izmir"
        assert join_key("İzmir   Konak") == "izmir konak"

    def test_the_zero_match_scenario_is_fixed(self):
        """Skill'de olculen tam senaryo: naif join 0, join_key 3 eslesme verir."""
        # Arrange
        upper = {"İSTANBUL": 34, "İZMİR": 35, "DİYARBAKIR": 21}
        lower = ["istanbul", "izmir", "diyarbakir"]

        # Act
        naive_keys = {key.lower() for key in upper}
        smart_keys = {join_key(key) for key in upper}

        # Assert
        assert len(naive_keys & set(lower)) == 0, "naif join sessizce 0 eslesme verir"
        assert len(smart_keys & {join_key(k) for k in lower}) == 3


class TestDiagnoseJoin:
    def test_reports_recovered_matches(self):
        diagnosis = diagnose_join(["İZMİR", "MUĞLA"], ["izmir", "mugla"])
        assert diagnosis["raw_matched"] == 0
        assert diagnosis["normalized_matched"] == 2
        assert diagnosis["recovered"] == 2

    def test_reports_genuinely_unmatched_keys(self):
        diagnosis = diagnose_join(["İzmir", "Bursa"], ["izmir", "ankara"])
        assert diagnosis["normalized_matched"] == 1
        assert "bursa" in diagnosis["left_only"]
        assert "ankara" in diagnosis["right_only"]


class TestTurkishSorting:
    def test_dotless_i_sorts_before_dotted_i(self):
        result = tr_sorted(["işlem", "ıspanak"])
        assert result == ["ıspanak", "işlem"]

    def test_turkish_letters_sort_in_alphabet_order_not_codepoint_order(self):
        words = ["güneş", "havuç", "işlem", "zurna", "çilek", "ördek", "ütü", "ıspanak", "şeker"]

        # Cikplak sorted() Turkce harfleri z'den sonraya atar
        assert sorted(words)[-1] == "şeker"

        assert tr_sorted(words) == [
            "çilek", "güneş", "havuç", "ıspanak", "işlem", "ördek", "şeker", "ütü", "zurna",
        ]


class TestNormalizeColumns:
    def test_turkish_column_names_become_ascii_snake_case(self):
        mapping = normalize_columns(["İL", "İLÇE", "KESİNTİ_SÜRESİ_DK", "TÜKETİM (kWh)"])
        assert mapping["İL"] == "il"
        assert mapping["İLÇE"] == "ilce"
        assert mapping["KESİNTİ_SÜRESİ_DK"] == "kesinti_suresi_dk"
        assert mapping["TÜKETİM (kWh)"] == "tuketim_kwh"

    def test_no_normalized_name_contains_combining_dot(self):
        mapping = normalize_columns(["İL", "İLÇE", "TESİS_YILI", "ARIZA_TİPİ"])
        for normalized in mapping.values():
            assert not has_combining_dot(normalized)
            assert normalized.isascii()

    def test_collisions_are_suffixed_not_dropped(self):
        """Iki farkli ham ad ayni normalize ada dusuyorsa kolon KAYBEDILMEZ."""
        mapping = normalize_columns(["İL", "IL", "Il"])
        assert len(set(mapping.values())) == 3, "cakisan adlar sessizce birlesmemeli"

    def test_leading_digit_gets_prefix(self):
        mapping = normalize_columns(["2024_TUKETIM"])
        assert mapping["2024_TUKETIM"].startswith("k_")

    def test_applying_mapping_to_dataframe_preserves_column_count(self):
        frame = pd.DataFrame({"İL": [1], "İLÇE": [2], "TÜKETİM": [3]})
        renamed = frame.rename(columns=normalize_columns(frame.columns))
        assert len(renamed.columns) == len(frame.columns)
        assert list(renamed.columns) == ["il", "ilce", "tuketim"]
