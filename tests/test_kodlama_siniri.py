"""64 KB ornekleme siniri UTF-8 karakterini bolerse SESSIZ cp1254 okumasi (P1-10).

2026-08-18 denetimi yeniden uretti: ornek tam bir cok baytli karakterin
ortasinda biterse utf-8 cozumu patlar, cp1254/iso-8859-9 her bayti kabul
ettigi icin DOSYANIN TAMAMI yanlis kodlamayla okunur; "Menteşe" -> "MenteÅŸe"
olur ve join_key eslesmesi %0'a duser. Gercek 11 MB'lik GDZ dosyasinda bu
kesme noktasina denk gelme olasiligi ~%2,8 olarak olculdu.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from gridup.io_utils import _SNIFF_BYTES, _decode_head, read_table


def _sinirda_bolen_dosya(yol: Path) -> None:
    """64 KB sinirinin TAM ustune cok baytli bir karakter denk getirir."""
    baslik = "ilce;deger\n"
    satir = "Bornova;1\n"
    govde = satir * ((_SNIFF_BYTES // len(satir)) + 10)
    ham = (baslik + govde).encode("utf-8")
    # Sinirdan hemen once bitecek sekilde kirp, sonra 2 baytli 'ş' koy.
    kirpik = ham[: _SNIFF_BYTES - 1]
    icerik = kirpik + "şehir;2\n".encode() + ham[_SNIFF_BYTES:]
    yol.write_bytes(icerik)


def test_sinirda_bolunen_utf8_dosya_yine_utf8_cozulur(tmp_path: Path) -> None:
    yol = tmp_path / "sinir.csv"
    _sinirda_bolen_dosya(yol)
    metin, kodlama = _decode_head(yol)
    assert kodlama.startswith("utf-8"), f"cp1254'e dustu: {kodlama}"
    assert "Bornova" in metin


def test_turkce_karakterler_bozulmadan_okunuyor(tmp_path: Path) -> None:
    """Uctan uca: read_table Turkce adlari dogru okumali (mojibake yok)."""
    yol = tmp_path / "ilceler.csv"
    _sinirda_bolen_dosya(yol)
    tablo = read_table(yol, verbose=False)
    adlar = set(tablo[tablo.columns[0]].astype(str))
    assert "şehir" in adlar or "Bornova" in adlar
    assert not any("Ã" in ad or "Å" in ad for ad in adlar), "mojibake tespit edildi"


def test_kisa_dosyada_davranis_degismiyor(tmp_path: Path) -> None:
    yol = tmp_path / "kisa.csv"
    yol.write_bytes("ilce;deger\nMenteşe;1\nÇeşme;2\n".encode())
    metin, kodlama = _decode_head(yol)
    assert kodlama.startswith("utf-8") and "Menteşe" in metin
    tablo = read_table(yol, verbose=False)
    assert list(tablo[tablo.columns[0]]) == ["Menteşe", "Çeşme"]
    assert isinstance(tablo, pd.DataFrame)
