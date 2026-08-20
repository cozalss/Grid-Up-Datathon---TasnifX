"""BENCHMARK SONUCUNUN HAFIF DOGRULAMASI.

NEDEN BU TESTLER VAR
--------------------
scripts/benchmark_gercek.py gercek GDZ verisinde alti receteyi karsilastirir
ve sonucu experiments/benchmark_gercek.json'a yazar. Koşu dakikalar surer;
testler koşuyu TEKRARLAMAZ. Bunun yerine urunun SOZLESMESINI dogrular:

  * sema tam mi (alan adlari, tipler) -- json'u okuyan otomasyon buna guvenir
  * sayilar ic tutarli mi -- harman uyelerinden kotu olamaz, OOF'ta gorunen
    en iyi aday dogru mu, bagimsiz kanit yokken kazanan kapisi kapali mi
  * sizinti kokusu var mi -- MAE sifira yakinsa ayni gunun bilgisi sizmistir

Sayilarin KENDISI test edilmez (veri guncellenince degisir); ILISKILERI
test edilir.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

SONUC_YOLU = Path(__file__).resolve().parents[1] / "experiments" / "benchmark_gercek.json"

BEKLENEN_MODELLER = {
    "lgb_l2",
    "lgb_mae",
    "lgb_tweedie",
    "lgb_sqrt",
    "catboost_mae",
    "xgb",
    "iki_asama",
    "iki_asama_medyan",
    "iki_asama_medyan_kalibre",
}
BEKLENEN_ALANLAR = {
    "modeller",
    "harman",
    "stack_mae",
    "kazanan",
    "sifir_baseline",
    "sifir_orani",
    "gun1_recetesi",
    "feature_kolonlari",
    "kalibrasyon",
    "tohum_kararliligi",
    "statistically_conclusive",
    "decision_reason",
    "benchmark_decision",
}


@pytest.fixture(scope="module")
def sonuc() -> dict:
    if not SONUC_YOLU.exists():
        pytest.fail(
            f"{SONUC_YOLU} yok -- once 'python scripts/benchmark_gercek.py' kos. "
            "Bu dosya teslimatin parcasidir, yoklugu hata sayilir."
        )
    return json.loads(SONUC_YOLU.read_text(encoding="utf-8"))


def test_sema_tam(sonuc: dict) -> None:
    """JSON'u okuyan otomasyon (deney gunlugu, gun-1 betigi) bu alanlara guvenir."""
    assert set(sonuc) == BEKLENEN_ALANLAR
    assert set(sonuc["modeller"]) == BEKLENEN_MODELLER
    for ad, bilgi in sonuc["modeller"].items():
        # fold_scores 2026-08-18 denetiminde eklendi: eslestirilmis karsilastirma
        # (ayni fold'da A-B farki) artefaktlardan yeniden hesaplanabilmeli.
        assert set(bilgi) == {"mae", "fold_std", "fold_scores", "sure_sn"}, ad
        assert isinstance(bilgi["fold_scores"], list) and bilgi["fold_scores"], ad
        assert all(isinstance(v, float) and math.isfinite(v) for v in bilgi["fold_scores"]), ad
        for alan in ("mae", "fold_std", "sure_sn"):
            deger = bilgi[alan]
            assert isinstance(deger, float) and math.isfinite(deger), f"{ad}.{alan}"


def test_skorlar_makul_aralikta(sonuc: dict) -> None:
    """MAE dakika cinsindendir: sifira yakin bir deger sizinti kokusudur.

    Ayni gunun effectedsubscribers'i feature alindiginda MAE yapay olarak
    duser (prova bunu yasadi). Hedefin medyani ~100 dk mertebesinde; MAE'nin
    1 dk altina inmesi ancak gelecek bilgisiyle mumkundur.
    """
    for ad, bilgi in sonuc["modeller"].items():
        assert bilgi["mae"] > 1.0, f"{ad} MAE={bilgi['mae']} -- sizinti kokusu"
        assert bilgi["fold_std"] >= 0.0, ad
        assert bilgi["sure_sn"] > 0.0, ad
    assert sonuc["sifir_baseline"] > 1.0
    assert 0.0 < sonuc["sifir_orani"] < 1.0


def test_oof_minimumu_yalnizca_gorunen_en_iyi_olarak_etiketleniyor(sonuc: dict) -> None:
    """Ayni OOF'ta secilen minimum bilimsel kazanan diye sunulamaz."""
    adaylar = {ad: bilgi["mae"] for ad, bilgi in sonuc["modeller"].items()}
    adaylar["harman"] = sonuc["harman"]["mae"]
    adaylar["stack"] = sonuc["stack_mae"]
    karar = sonuc["benchmark_decision"]
    gorunen = karar["apparent_oof_best"]

    assert adaylar[gorunen] == pytest.approx(min(adaylar.values()))
    assert sonuc["kazanan"] is None
    assert karar["winner"] is None
    assert sonuc["statistically_conclusive"] is False
    assert karar["statistically_conclusive"] is False
    assert (karar["n_anchors"] < karar["required_anchors"]) or (
        not karar["statistically_conclusive"]
    )
    assert sonuc["decision_reason"] == karar["decision_reason"]


def test_oof_ta_gorunen_en_iyi_baseline_i_geciyor(sonuc: dict) -> None:
    """Gorunen en iyi aday baseline'i gecse bile bagimsiz kanit kapisi ayridir."""
    adaylar = {ad: bilgi["mae"] for ad, bilgi in sonuc["modeller"].items()}
    adaylar["harman"] = sonuc["harman"]["mae"]
    adaylar["stack"] = sonuc["stack_mae"]
    gorunen = sonuc["benchmark_decision"]["apparent_oof_best"]
    assert adaylar[gorunen] < sonuc["sifir_baseline"]


def test_harman_ic_tutarli(sonuc: dict) -> None:
    """Hill climbing en iyi tek uyeden BASLAR -- ayni satir kumesinde uyelerinden
    kotu bir harman matematiksel olarak mumkun degildir. Agirliklar toplami 1."""
    harman = sonuc["harman"]
    assert set(harman) == {"mae", "uyeler", "agirliklar", "yuvalanmis"}
    # Harman artik TUM uyeler uzerinde hill-climb yapar; agirligi 0 cikanlar
    # raporda yer almaz. En az 1, en fazla uye sayisi kadar olabilir.
    assert 1 <= len(harman["uyeler"]) <= len(BEKLENEN_MODELLER)
    assert set(harman["uyeler"]) <= BEKLENEN_MODELLER
    assert set(harman["agirliklar"]) == set(harman["uyeler"])
    assert all(agirlik >= 0.0 for agirlik in harman["agirliklar"].values())
    assert sum(harman["agirliklar"].values()) == pytest.approx(1.0, abs=1e-2)

    uye_maeleri = [sonuc["modeller"][ad]["mae"] for ad in harman["uyeler"]]
    assert harman["mae"] <= min(uye_maeleri) + 1e-6


def test_harman_yuvalanmis_kontrolden_geciyor_mu_kaydediliyor(sonuc: dict) -> None:
    """Ornek-ici harman skoru yanlidir; karar YUVALANMIS olcume dayanmali.

    2026-08-18 denetimi (P1-1): agirliklar tum OOF'ta tirmanilip ayni OOF'ta
    skorlaniyordu. Artik agirliklar gecmis fold'larda ogrenilip SONRAKI fold'da
    skorlaniyor ve sonuc JSON'a yaziliyor -- gun-1 recetesi buna bakar.
    """
    nested = sonuc["harman"]["yuvalanmis"]
    assert set(nested) >= {
        "mae",
        "fold_kayitlari",
        "ayni_satirlarda_en_iyi_tekil",
        "ayni_satirlarda_tekil_mae",
        "harman_tekilden_iyi_mi",
        "fark",
    }
    assert isinstance(nested["harman_tekilden_iyi_mi"], bool)
    assert nested["mae"] > 1.0
    # Yuvalanmis skor ornek-ici skordan IYI OLAMAZ (ayni veriye bakmiyor).
    assert nested["mae"] >= sonuc["harman"]["mae"]
    # Recete metni kararla tutarli olmali.
    recete = sonuc["gun1_recetesi"]
    if nested["harman_tekilden_iyi_mi"]:
        assert "YUVALANMIS kontrolde de geciyor" in recete
    else:
        assert "GECMIYOR" in recete and "5 TOHUMLA" in recete


def test_tohum_kararliligi_gurultuyu_olcuyor(sonuc: dict) -> None:
    """Kucuk MAE farklarinin anlamli olup olmadigini tohum yayilimi belirler."""
    tohum = sonuc["tohum_kararliligi"]
    assert len(tohum["tohumlar"]) == len(set(tohum["tohumlar"])) >= 3
    assert len(tohum["tekil_mae"]) == len(tohum["tohumlar"])
    assert tohum["tohum_yayilimi"] >= 0.0
    assert tohum["tohum_araligi"] >= 0.0
    # Tohum ortalamasi tekil skorlarin en kotusunden iyi olmali.
    assert tohum["tohum_ortalamasi_mae"] <= max(tohum["tekil_mae"]) + 1e-6


def test_gun1_recetesi_olculen_sayilarla_konusuyor(sonuc: dict) -> None:
    """Recete bos slogan degil; kanit sinirini ve baslangic adayini soylemeli."""
    recete = sonuc["gun1_recetesi"]
    assert isinstance(recete, str) and len(recete) > 100
    assert "Bilimsel kazanan ilan edilmedi" in recete
    assert sonuc["decision_reason"] in recete
    # 2023 birinci recetesinin hukmu (tasinir mi, tasinamaz mi) yazili olmali.
    assert "catboost_mae" in recete


def test_yasak_ham_kolonlar_feature_listesine_sizmamis(sonuc: dict) -> None:
    """CEKISMELI DENETIMIN BULDUGU SIZINTIYA KARSI KALICI KORUMA.

    Ilk surumde ham kaydin 'id' kolonu 50 feature arasindaydi: build_panel
    onu 'first' ile tasiyip dolgu satirlarinda NaN biraktigi icin NaN deseni
    _dolduruldu bayraginin BIREBIR kopyasiydi (olculdu: uyum 1.000000, y==0
    ile 0.9975) ve tum skorlari ~60 dk iyimser gosterdi (harman 251 -> 308).

    Onceki test yalnizca 'MAE > 1.0' esigine bakiyordu ve iki dunyayi ayirt
    edemiyordu (denetim kaniti: id'li JSON ile 6 test de geciyordu). Bu test
    feature listesinin KENDISINE bakar -- sizinti skora yansimadan yakalanir.
    """
    yasak = {
        "id",
        "il",
        "ilce",
        "date",
        "starttime",
        "endtime",
        "reason",
        "effectedsubscribers",
        "hourlyloadavg",
        "effectedneighbourhoods",
        "distributioncompanyname",
        "_dolduruldu",
    }
    kolonlar = sonuc.get("feature_kolonlari")
    assert kolonlar, "JSON feature_kolonlari tasimali -- sizinti denetimi makinelesir"
    sizanlar = sorted(yasak & set(kolonlar))
    assert not sizanlar, f"Ham/ayni-gun kolon feature listesine sizdi: {sizanlar}"
