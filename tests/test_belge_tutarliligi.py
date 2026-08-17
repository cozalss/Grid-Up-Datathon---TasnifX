"""BELGE-GERCEK tutarliligi.

NEDEN BU DOSYA VAR
------------------
README "426 test" diyordu, gercek 544'tu. docs/07 "349 test" diyordu.
README "100k ve 500k satirda olculdu" diyordu ama experiments dosyasinda
yalnizca 100k vardi.

Bu, kod hatasi degil ama yarismada BEDELI VAR: kapi 2 notebook
degerlendirmesidir ve tutmayan bir rakam, juride gereksiz bir guven kaybi
yaratir. Ayrica kendi kararlarimizi bayat sayilara dayandirmis oluruz.

Buradaki testler belgedeki OLCULEBILIR iddialari gercege karsi dogrular.
Kirildiklarinda yapilacak sey belgeyi guncellemektir -- testi degil.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]
README = KOK / "README.md"
VERI_GUNU = KOK / "docs" / "07-veri-gunu-kontrol-listesi.md"


def _toplanan_test_sayisi() -> int:
    """pytest'in TOPLADIGI test sayisi -- kosmadan, hizli.

    ``--collect-only -q`` bu surumde dosya basina "yol: N" satirlari basiyor,
    tek bir "N tests collected" ozeti DEGIL. Ilk surumde ozet satirini
    ariyordum ve bulamayinca test SESSIZCE ATLANIYORDU -- yani koruma hic
    calismiyordu. Artik dosya basina sayilari topluyoruz.
    """
    sonuc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        capture_output=True,
        text=True,
        cwd=KOK,
        timeout=300,
        check=False,
    )
    # Once ozet satirini dene (baska pytest surumleri onu basar).
    ozet = re.search(r"(\d+)\s+tests? collected", sonuc.stdout)
    if ozet:
        return int(ozet.group(1))

    satirlar = re.findall(r"^\S+\.py:\s*(\d+)\s*$", sonuc.stdout, re.MULTILINE)
    if not satirlar:
        pytest.fail(
            "test sayisi cikarilamadi -- koruma calismiyor demektir.\n"
            f"pytest ciktisi: {sonuc.stdout[-300:]}"
        )
    return sum(int(n) for n in satirlar)


@pytest.mark.slow
@pytest.mark.parametrize("belge", [README, VERI_GUNU])
def test_belgedeki_test_sayisi_gercegi_yansitiyor(belge: Path):
    """Belgede gecen 'N test' ifadeleri gercek test sayisiyla ayni olmali."""
    if not belge.exists():
        pytest.skip(f"{belge.name} yok")

    metin = belge.read_text(encoding="utf-8")
    iddialar = {int(m) for m in re.findall(r"(\d+)\s*test\b", metin)}
    if not iddialar:
        pytest.skip("belgede test sayisi iddiasi yok")

    gercek = _toplanan_test_sayisi()
    # Toplama sirasinda atlanan testler kosuda gorunmeyebilir; kucuk bir
    # tolerans birakiyoruz ama BUYUK kaymayi yakaliyoruz.
    for iddia in iddialar:
        assert abs(iddia - gercek) <= 5, (
            f"{belge.name}: '{iddia} test' yaziyor ama gercek {gercek}. Belgeyi guncelle."
        )


def test_readme_olcek_provasi_iddiasi_dosyayla_ortusuyor():
    """README hangi olceklerin olculdugunu iddia ediyorsa, dosyada OLMALI."""
    yol = KOK / "experiments" / "olcek_provasi.json"
    if not yol.exists():
        pytest.skip("olcek provasi henuz calistirilmamis")

    veri = json.loads(yol.read_text(encoding="utf-8"))
    olculen = {
        int(k)
        for k, v in veri.items()
        if not (v.get("olcumler") and v["olcumler"][0].get("tahmin"))
    }
    metin = README.read_text(encoding="utf-8")
    # README'de olcek iddiasi nerede gecerse gecsin yakala (tablo satiri,
    # baslik veya duz cumle). Ilk surum yalnizca "Ölçek provası" iceren
    # satira bakiyordu ve README yeniden yazilinca SESSIZCE atlandi.
    ilgili = [
        s
        for s in metin.splitlines()
        if re.search(r"\d+k\s+satır", s) and ("ölç" in s.lower() or "prova" in s.lower())
    ]
    if not ilgili:
        pytest.skip("README'de olcek iddiasi yok")

    for iddia_k in re.findall(r"(\d+)k\s+satır", " ".join(ilgili)):
        iddia = int(iddia_k) * 1000
        assert any(abs(iddia - o) / o < 0.1 for o in olculen), (
            f"README '{iddia_k}k satır' iddia ediyor ama olculen olcekler: "
            f"{sorted(olculen)}. Ya provayi kos ya iddiayi duzelt."
        )


def test_veri_gunu_hava_konum_sayisi_dogru():
    """docs/07'deki hava konum sayisi gercek dosyayla ayni olmali."""
    parquet = KOK / "data" / "external" / "hava_gunluk.parquet"
    if not parquet.exists() or not VERI_GUNU.exists():
        pytest.skip("hava verisi veya belge yok")

    gercek = pd.read_parquet(parquet)["konum"].nunique()
    satir = next(
        (
            s
            for s in VERI_GUNU.read_text(encoding="utf-8").splitlines()
            if "hava_gunluk.parquet" in s and "|" in s
        ),
        "",
    )
    if not satir:
        pytest.skip("belgede hava satiri yok")

    sayilar = [int(m) for m in re.findall(r"\*\*(\d+)\s*(?:ilçe|konum)\*\*", satir)]
    sayilar += [int(m) for m in re.findall(r"(\d+)\s*(?:ilçe|konum)", satir)]
    assert sayilar, f"belgede konum sayisi bulunamadi: {satir[:80]}"
    assert gercek in sayilar, f"docs/07 {sayilar} konum diyor, gercek {gercek}. Belgeyi guncelle."


def test_readme_bahsettigi_betikler_var():
    """README'de gecen her script gercekten var olmali."""
    metin = README.read_text(encoding="utf-8")
    betikler = set(re.findall(r"`?(scripts/[a-z_]+\.py)`?", metin))
    eksik = [b for b in betikler if not (KOK / b).exists()]
    assert not eksik, f"README var olmayan betiklere atif yapiyor: {eksik}"


# --------------------------------------------------------------------------
# Hava verisi: iki granulariteyi de desteklemeli
# --------------------------------------------------------------------------


def test_hava_verisi_il_ve_ilce_anahtari_iceriyor():
    """Yarisma verisinin hangi seviyede gelecegini BILMIYORUZ.

    Konum adi "Il-Ilce" bicimine gecince il bazli join'in eslesme orani
    %0.0'a dustu (full_pipeline'da olculdu). Iki anahtari da yazmak, veri
    gunu hangi granularite gelirse gelsin calisan tek cozumdur.
    """
    parquet = KOK / "data" / "external" / "hava_gunluk.parquet"
    if not parquet.exists():
        pytest.skip("hava verisi yok")

    hava = pd.read_parquet(parquet)
    for kolon in ("konum_key", "il_key", "ilce_key"):
        assert kolon in hava.columns, f"'{kolon}' kolonu yok"

    assert hava["il_key"].nunique() == 5, "bes il beklenir"
    assert hava["ilce_key"].nunique() >= 90, "ilce kapsami eksik"
    # Anahtarlar join_key ile normalize edilmis olmali (Turkce tuzagi).
    assert not hava["il_key"].str.contains(r"[çğıöşüÇĞİÖŞÜ]", regex=True).any()
