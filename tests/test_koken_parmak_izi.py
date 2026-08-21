"""Git parmak izi HESAPLANAMAYINCA tamamlanmis kosu COPE GITMEMELI.

NEDEN BU TEST DOSYASI (2026-08-21, olculdu)
--------------------------------------------
``day_one.py`` gecerli bir submission uretti, 5 tohumlu yeniden egitimi
bitirdi, dosyayi diske yazdi -- ve SONRA su hatayla cikti::

    ValueError: Deney yeniden uretim metadatasi eksik:
                provenance.git_diff_fingerprint

Sebep: ``git diff HEAD --binary`` 10 saniyelik zaman asimina takildi. Ayni
makinede arka planda bir olcum kosusu (ablasyon, 96 ilce x 1690 gun x 5
tohum) butun cekirdekleri doyuruyordu; git surecine zamaninda sira gelmedi.
Bu depoda agir olcumlerle es zamanli calismak ISTISNA DEGIL, KURAL --
yarisma gunu daha da beter olacak.

IKI AYRI SORUN VARDI
--------------------
1. SURE. 10 saniye, yuklu bir makine icin dar. 60'a cikarildi.

2. UC DURUMUN IKIYE SIKISTIRILMASI. Fonksiyon ``None`` donuyordu hem
   "agac temiz, kaydedilecek diff yok" hem "agac kirli ama diff'i
   alamadim" durumunda. Kapi ikincisini birincisinden ayirt edemiyordu;
   ayirt edebilseydi kosuyu atmak yerine kaydi DURUSTCE isaretlerdi.

   temiz agac        -> None
   diff alindi       -> sha256 hex
   alinamadi         -> "HESAPLANAMADI:<sebep>"     <- yeni

Ucuncu durum yeniden uretilebilirligi ZAYIFLATIR ve bunu saklamaz: kayit
"agac kirliydi, diff yakalanamadi, sebep zaman asimi" der. Sessizce yok
saymaktan da, saatlerce suren bir kosuyu son adimda atmaktan da iyidir.
"""

from __future__ import annotations

import subprocess

import pytest

from gridup.experiment import HESAPLANAMADI_ONEKI, _git_diff_fingerprint
from gridup.stores.sqlite import _provenance_errors


class _SahteProvenance:
    """Kapiyi test etmek icin asgari provenance yuzeyi."""

    def __init__(self, *, git_dirty: bool, git_diff_fingerprint: str | None) -> None:
        self.git_sha = "a" * 40
        self.git_dirty = git_dirty
        self.git_diff_fingerprint = git_diff_fingerprint
        self.data_artifacts = (_SahteArtifact(),)
        self.recipe_fingerprint = "0" * 64
        self.fold_fingerprint = "0" * 64
        self.feature_names = ("a",)
        self.environment = {"python": "3.12"}


class _SahteArtifact:
    sha256 = "0" * 64
    size_bytes = 1


class _SahteKayit:
    """``_provenance_errors`` bir KAYIT alir, provenance degil."""

    def __init__(self, provenance: _SahteProvenance) -> None:
        self.provenance = provenance
        self.features = ("a",)


def test_zaman_asiminda_none_degil_sentinel_doner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Olculen ariza: yuklu makinede git zaman asimina ugruyor."""

    def _patlat(*_args: object, **_kwargs: object) -> None:
        raise subprocess.TimeoutExpired(cmd="git diff", timeout=60)

    monkeypatch.setattr(subprocess, "run", _patlat)

    sonuc = _git_diff_fingerprint()

    assert sonuc is not None
    assert sonuc.startswith(HESAPLANAMADI_ONEKI)
    assert "zaman" in sonuc.lower() or "timeout" in sonuc.lower()


def test_git_yoksa_da_sentinel_doner(monkeypatch: pytest.MonkeyPatch) -> None:
    def _yok(*_args: object, **_kwargs: object) -> None:
        raise OSError("git bulunamadi")

    monkeypatch.setattr(subprocess, "run", _yok)

    sonuc = _git_diff_fingerprint()

    assert sonuc is not None
    assert sonuc.startswith(HESAPLANAMADI_ONEKI)


def test_kirli_agacta_sentinel_kapiyi_gecer() -> None:
    """Tamamlanmis kosu, diff alinamadi diye ATILMAZ."""
    prov = _SahteProvenance(
        git_dirty=True, git_diff_fingerprint=f"{HESAPLANAMADI_ONEKI}zaman_asimi"
    )

    assert "provenance.git_diff_fingerprint" not in _provenance_errors(_SahteKayit(prov))


def test_kirli_agacta_eksik_parmak_izi_hala_reddedilir() -> None:
    """Guvence korunur: gercekten YOK olan parmak izi hala hatadir.

    Sentinel bir kacamak degil; yalnizca "denendi ve olmadi" durumunu
    isaretler. Hic denenmemis (None) durum reddedilmeye devam eder.
    """
    prov = _SahteProvenance(git_dirty=True, git_diff_fingerprint=None)

    assert "provenance.git_diff_fingerprint" in _provenance_errors(_SahteKayit(prov))


def test_temiz_agacta_none_sorun_degil() -> None:
    prov = _SahteProvenance(git_dirty=False, git_diff_fingerprint=None)

    assert "provenance.git_diff_fingerprint" not in _provenance_errors(_SahteKayit(prov))


def test_sentinel_gercek_hexle_karistirilmaz() -> None:
    """Sentinel, gecerli bir sha256 hex'i gibi GORUNMEMELI."""
    assert not all(c in "0123456789abcdef" for c in HESAPLANAMADI_ONEKI)
    assert len(HESAPLANAMADI_ONEKI) > 0
