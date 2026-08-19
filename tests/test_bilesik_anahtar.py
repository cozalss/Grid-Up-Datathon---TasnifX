"""Bilesik "il-ilce" anahtarlari (P1-12, 2026-08-18 denetimi).

2024 GDZ semasinda ilce kolonu "izmir-aliaga" bicimindeydi; referans tablosu
yalin "aliaga" tutuyor. ``strip_qualifier`` SOL parcayi alir (niteleyici
sagda oldugu icin dogru davranis) ve bilesik anahtarda YANLIS parcayi verir:
"izmir-karabaglar" -> "izmir". Sonuc: join %0, sessiz NaN. ``split_il_ilce``
bu ikinci sinifi cozer; ``diagnose_join`` artik kurtarilabilir bilesikleri
raporlar.
"""

from __future__ import annotations

import pytest

from gridup.turkish import diagnose_join, join_key, split_il_ilce, strip_qualifier


@pytest.mark.parametrize(
    "girdi, beklenen",
    [
        ("izmir-karabaglar", ("izmir", "karabaglar")),
        ("İZMİR / KARABAĞLAR", ("İZMİR", "KARABAĞLAR")),
        ("aydin|bozdogan", ("aydin", "bozdogan")),
        ("mugla_mentese", ("mugla", "mentese")),
        ("Bornova", (None, "Bornova")),
        ("  Çeşme  ", (None, "Çeşme")),
    ],
)
def test_bilesik_ayirma(girdi: str, beklenen: tuple[str | None, str]) -> None:
    assert split_il_ilce(girdi) == beklenen


def test_strip_qualifier_ile_karistirilmamali() -> None:
    """Iki fonksiyon TERS parcayi alir; ikisi de kendi sinifinda dogrudur."""
    # Niteleyici SAGDA: sol parca dogru
    assert strip_qualifier("Koprubasi / Manisa") == "Koprubasi"
    # Bilesik anahtar: ilce SAGDA
    assert split_il_ilce("izmir / karabaglar")[1] == "karabaglar"
    assert strip_qualifier("izmir / karabaglar") == "izmir"  # bu sinifta YANLIS parca


def test_diagnose_join_bilesikleri_kurtarilabilir_diye_raporlar() -> None:
    sol = ["izmir-karabaglar", "izmir-bornova", "mugla|bodrum", "yok-ilce"]
    sag = ["karabaglar", "bornova", "bodrum", "cesme"]
    rapor = diagnose_join(sol, sag)
    kurtarilabilir = rapor["composite_recoverable"]
    assert set(kurtarilabilir) == {
        join_key("izmir-karabaglar"),
        join_key("izmir-bornova"),
        join_key("mugla|bodrum"),
    }
    assert kurtarilabilir[join_key("izmir-karabaglar")] == "karabaglar"
    # Gercekten eslesmeyen anahtar left_only'de kalir
    assert rapor["left_only"] == [join_key("yok-ilce")]
    assert rapor["normalized_matched"] == 0  # ham haliyle hicbiri eslesmiyor


def test_yalin_anahtarlar_bilesik_kurtarmayi_tetiklemez() -> None:
    rapor = diagnose_join(["bornova", "cesme"], ["bornova", "cesme"])
    assert rapor["composite_recoverable"] == {}
    assert rapor["normalized_matched"] == 2
