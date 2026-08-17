"""Veri saglik kapisinin KENDISININ sozlesmeleri.

``scripts/veri_sagligi.py`` harici verinin kapsam/butunluk/fizik kapisidir.
Ama bir kapinin en tehlikeli hali, KAPSAMADIGI seyi sessizce gecirmesidir:
yeni bir veri kaynagi eklenir, saglik listesine yazilmaz ve kapi "her sey
yolunda" der. Buradaki testler tam olarak bunu imkansiz kilar.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

KOK = Path(__file__).resolve().parents[1]
MANIFEST = KOK / "data" / "sources.yml"


def _saglik_modulu():
    yol = KOK / "scripts" / "veri_sagligi.py"
    spec = importlib.util.spec_from_file_location("veri_sagligi", yol)
    assert spec is not None and spec.loader is not None
    modul = importlib.util.module_from_spec(spec)
    sys.modules["veri_sagligi"] = modul
    spec.loader.exec_module(modul)
    return modul


def test_manifestteki_her_artefakt_saglik_kapisinda_kayitli() -> None:
    """Manifeste eklenen bir veri, saglik denetiminden KACAMAZ.

    Bu testin kirilmasi su demektir: yeni bir kaynak eklendi ama
    ``veri_sagligi.KAYNAKLAR`` listesine yazilmadi. O kaynak icin kapsam,
    NaN ve fizik kontrolu HIC kosmayacak; kapi yanlis yere "temiz" diyecek.
    """
    if not MANIFEST.is_file():
        pytest.skip("sources.yml yok")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    beklenen = {a["path"] for a in manifest["artifacts"]}
    modul = _saglik_modulu()
    kayitli = {k.yol for k in modul.KAYNAKLAR}

    eksik = sorted(beklenen - kayitli)
    assert not eksik, (
        f"Su artefaktlar manifestte var ama saglik kapisinda YOK: {eksik}. "
        "scripts/veri_sagligi.py icindeki KAYNAKLAR listesine ekle."
    )


def test_saglik_kapisi_gercek_veride_geciyor() -> None:
    """Depodaki gercek veri, kendi saglik sozlesmesini SAGLAMALI.

    Bu test veri dosyalari yoksa atlanir (CI checkout'unda data/ olmayabilir),
    ama varsa hatasiz gecmek ZORUNDADIR.
    """
    modul = _saglik_modulu()
    mevcut = [k for k in modul.KAYNAKLAR if (KOK / k.yol).is_file()]
    if not mevcut:
        pytest.skip("harici veri dosyalari bu ortamda yok")

    bugun = pd.Timestamp.today().normalize()
    tum_hatalar: list[str] = []
    for kaynak in mevcut:
        hatalar, _ = modul.denetle(kaynak, bugun)
        tum_hatalar.extend(f"{kaynak.ad}: {h}" for h in hatalar)

    assert not tum_hatalar, "Veri saglik kapisi HATA verdi:\n  " + "\n  ".join(tum_hatalar)


def test_her_kaynagin_asgari_bir_kontrolu_var() -> None:
    """Sozlesmesiz bir kaynak, listede olsa bile denetlenmiyor demektir.

    Yalnizca dosya varligina bakan bir girdi, kapiyi gecerken hicbir sey
    kanitlamaz -- "listeye ekledim" ile "denetleniyor" ayni sey degildir.
    """
    modul = _saglik_modulu()
    zayif = [
        k.ad
        for k in modul.KAYNAKLAR
        if not (k.tarih_kolonu or k.anahtar_kolonu or k.fizik) and k.asgari_satir <= 1
    ]
    assert not zayif, (
        f"Su kaynaklarin hicbir anlamli kontrolu yok: {zayif}. "
        "En az bir tarih/anahtar/fizik kontrolu ya da asgari satir sayisi ver."
    )


def test_fizik_kontrolleri_bozuk_veriyi_yakaliyor() -> None:
    """Kapinin ISE YARADIGININ kaniti: kasitli bozulmus veri REDDEDILMELI.

    'Her sey gecti' ciktisi, kontrollerin gercekten calistigi anlamina
    gelmez -- her zaman True donen bir kontrol de ayni ciktiyi verir.
    Burada hava tablosunu bilerek bozup kapinin bagirdigini dogruluyoruz.
    """
    modul = _saglik_modulu()
    hava = next((k for k in modul.KAYNAKLAR if k.ad == "hava_gunluk"), None)
    assert hava is not None
    yol = KOK / hava.yol
    if not yol.is_file():
        pytest.skip("hava_gunluk yok")

    d = pd.read_parquet(yol)
    bozuk = d.copy()
    # Sicakliklari 30 derece dusur: Ege yazi artik 5 C gorunur.
    bozuk["sicaklik_max"] = bozuk["sicaklik_max"] - 30.0

    ay = pd.to_datetime(bozuk["tarih"]).dt.month
    yaz_ort = bozuk.loc[ay.isin([7, 8]), "sicaklik_max"].mean()
    assert yaz_ort < 30, "Test kurgusu hatali: bozulmus veri hala makul aralikta"

    aciklama, kontrol = hava.fizik[0]
    assert not bool(kontrol(bozuk)), (
        f"Fizik kontrolu bozuk veriyi GECIRDI: '{aciklama}'. "
        "Kontrol her zaman True donuyor olabilir -- kapi ise yaramaz."
    )
    assert bool(kontrol(d)), "Ayni kontrol SAGLAM veriyi reddediyor -- esik yanlis."
