"""``deney.AILELER`` KAPSAM testi.

NEDEN BU TEST VAR
-----------------
21-22 Agustos 2026 gecesi ayni sinifdan IKI hata bulundu ve ikisi de aylarca
sessiz durabilirdi:

1. ``AILELER["hava"]`` onekleri ``isitma_derece`` ve ``sogutma_derece``
   iceriyordu. O isimde HIC KOLON YOK -- gercek isimler ``cdd18``,
   ``cdd22``, ``cdd24`` ve hareketli ortalamalari. Sonuc: ``-hava``
   ablasyonu 10 CDD kolonunu hic cikarmadi ve havanin degerini eksik
   olctu. Ustelik kacanlar, yaz elektrik tuketiminin fizik olarak en
   onemli degiskeni.

2. 144 kolonun 8'i hicbir aileye girmiyordu (``t_gy_*``, ``t_yayilma``,
   ``t_kayma``, ``t_hg_genligi``, ``ozet_pencere_gun``, ``t_doluluk``).
   Yani "136 kolonun 19'u is yapiyor" ablasyonu onlari HIC sinamadi.

Ikisi de ayni koke sahip: ablasyon onek eslesmesiyle calisiyordu ve
**hicbir kapsam denetimi yoktu**. Bir ablasyonun sonucu ancak kapsami
kadar gecerlidir; kapsanmayan kolon "olculdu ve onemsiz" degil,
"olculmedi" demektir ve ikisi cok farkli seylerdir.

Bu test o denetimi kalicilastirir.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

ONBELLEK = KOK / "data" / "interim" / "deney" / "egitim.parquet"
KAYNAK = KOK / "src" / "gridup" / "features" / "trafo.py"


def _onbellek_bayat() -> bool:
    """Onbellek, oznitelikleri ureten koddan ESKI mi?

    Onbellek bir YAPI ARTEFAKTI ve koddan geri kalabilir: yeni bir kolon
    eklendiginde ``--yenile`` calistirilana kadar diskteki cerceve onu
    icermez. O durumda kapsam testi kodun degil, bayat artefaktin
    kapsamini olcer -- ve YANLIS bir basarisizlik uretir.

    Bu yuzden bayatlik sessizce tolere edilmiyor, ACIKCA atlaniyor:
    testin mesaji "onbellegi yenile" der.
    """
    return ONBELLEK.exists() and ONBELLEK.stat().st_mtime < KAYNAK.stat().st_mtime


BAYAT = pytest.mark.skipif(
    _onbellek_bayat(),
    reason="onbellek trafo.py'den eski -- 'python scripts/deney.py --yenile' calistir",
)

#: Bilerek aileye girmeyen kolonlar. Ablasyon bunlari cikaramaz cunku
#: model onlarsiz anlamsiz olur (``guc`` hedefin ofseti, ``soguk_mu``
#: rejim bayragi) ya da kategorik altyapiya ait.
YAPISAL = frozenset({"guc", "il_key", "bolge", "soguk_mu"})


@BAYAT
@pytest.mark.skipif(not ONBELLEK.exists(), reason="deney onbellegi yok")
def test_aile_kapsami_tam() -> None:
    """Yapisal olanlar disinda HER kolon bir aileye ait olmali."""
    import deney as d
    import pandas as pd
    import tuketim_model as tm

    cerceve = pd.read_parquet(ONBELLEK)
    kolonlar = tm.oznitelikler(cerceve)

    kapsanan: set[str] = set()
    for onek in d.AILELER.values():
        kapsanan |= {k for k in kolonlar if k.startswith(onek)}

    kacan = sorted(k for k in kolonlar if k not in kapsanan and k not in YAPISAL)
    assert not kacan, (
        f"{len(kacan)} kolon hicbir ablasyon ailesine girmiyor: {kacan}. "
        "Ablasyon bunlari HIC sinamiyor, yani 'olculdu ve onemsiz' degil "
        "'olculmedi' durumundalar. Ya AILELER'e ekle ya YAPISAL'a."
    )


@BAYAT
@pytest.mark.skipif(not ONBELLEK.exists(), reason="deney onbellegi yok")
def test_her_aile_en_az_bir_kolon_eslesiyor() -> None:
    """Bir onek hicbir kolonla eslesmiyorsa, o onek OLU demektir.

    ``isitma_derece`` hatasi tam olarak boyleydi: onek listede duruyordu,
    kimse eslesmedigini fark etmedi, ve ablasyon o kolonlari cikardigini
    SANDI.
    """
    import deney as d
    import pandas as pd
    import tuketim_model as tm

    cerceve = pd.read_parquet(ONBELLEK)
    kolonlar = tm.oznitelikler(cerceve)

    olu: list[str] = []
    for aile, onekler in d.AILELER.items():
        for onek in onekler:
            if not any(k.startswith(onek) for k in kolonlar):
                olu.append(f"{aile}:{onek}")
    assert not olu, (
        f"su onekler hicbir kolonla eslesmiyor: {olu}. "
        "Ablasyon o kolonlari cikardigini sanip cikarmiyor."
    )
