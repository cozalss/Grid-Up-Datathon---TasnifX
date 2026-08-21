"""GDZ'nin kamuya acik kesinti haritasindan CBS noktalari indirir.

Bu kaynak TRAFO envanteri degildir. Her satir bir kesinti-ilce kaydidir;
``cbs_lat``/``cbs_lon`` kesinti kaydinin resmi CBS noktasidir. API trafo
kodunu dondurmedigi icin bu noktalar yarisma ``tanim`` degerleriyle dogrudan
eslestirilemez. Betik bu siniri kolonlarda ve konsol raporunda acik tutar.

Kullanim::

    python scripts/fetch_gdz_kesinti_cbs.py
    python scripts/fetch_gdz_kesinti_cbs.py --out data/research/gdz_snapshot.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))

from gridup.io_utils import publish_dataframe  # noqa: E402

SAYFA = "https://www.gdzelektrik.com.tr/bilgi-merkezi/planli-bakim-ve-iyilestirme-calismalari"
VARSAYILAN_CIKTI = KOK / "data" / "research" / "gdz_kesinti_cbs_snapshot.parquet"
SEHIRLER = ("\u0130ZM\u0130R", "MAN\u0130SA")
KAYNAK = "https://www.gdzelektrik.com.tr/api/outages-v2"
YEDEK_KAYNAK = "https://www.gdzelektrik.com.tr/api/test-outages"


def _uygulama_bilgisi(oturum: requests.Session) -> tuple[str, str]:
    """Acik web istemcisinden guncel uygulama JS adresini ve token'i bulur."""
    sayfa = oturum.get(SAYFA, timeout=60)
    sayfa.raise_for_status()
    js_eslesme = re.search(r'<script\s+src=["\']([^"\']*?/js/app\.js[^"\']*)', sayfa.text)
    if js_eslesme is None:
        raise RuntimeError("GDZ kesinti sayfasinda uygulama JS adresi bulunamadi")

    js_adresi = urljoin(SAYFA, js_eslesme.group(1))
    javascript = oturum.get(js_adresi, timeout=120)
    javascript.raise_for_status()
    token_eslesme = re.search(r'var\s+token\s*=\s*["\']([^"\']+)["\']', javascript.text)
    if token_eslesme is None:
        raise RuntimeError("Acik web istemcisinin API yetkilendirme degeri bulunamadi")
    return js_adresi, token_eslesme.group(1)


def _wkt_nokta(deger: Any) -> tuple[float | None, float | None]:
    """``POINT(lon lat)`` degerini (lat, lon) olarak ayirir."""
    eslesme = re.fullmatch(
        r"\s*POINT\(\s*(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*\)\s*",
        str(deger or ""),
        flags=re.IGNORECASE,
    )
    if eslesme is None:
        return None, None
    return float(eslesme.group(2)), float(eslesme.group(1))


def _liste(deger: Any) -> list[Any]:
    if isinstance(deger, list):
        return deger
    if deger in (None, ""):
        return []
    return [deger]


def _json_utf8(yanit: requests.Response) -> dict[str, Any]:
    """GDZ JSON yanitini HTTP'nin eksik charset bilgisinden bagimsiz oku."""
    veri = json.loads(yanit.content.decode("utf-8-sig"))
    if not isinstance(veri, dict):
        raise RuntimeError("GDZ API yaniti JSON nesnesi degil")
    return veri


def _satira_cevir(kayit: dict[str, Any], *, cekilme_zamani: str) -> dict[str, Any]:
    lat, lon = _wkt_nokta(kayit.get("CBS_Koordinat"))
    mahalleler = _liste(kayit.get("Mahalle"))
    sokaklar = _liste(kayit.get("Sokak"))
    adres_noktalari = _liste(kayit.get("Musteri_Koordinat"))
    return {
        "kesinti_id": str(kayit.get("Kesinti_ID") or ""),
        "durum": kayit.get("Durum"),
        "il": kayit.get("Sehir"),
        "ilce": kayit.get("Ilce"),
        "neden": kayit.get("Kesinti_Nedeni"),
        "planlanan_baslangic": kayit.get("Planlanan_Baslangic_Zamani"),
        "planlanan_bitis": kayit.get("Planlanan_Sona_Erme_Zamani"),
        "gercek_baslangic": kayit.get("Kesinti_Baslangic_Zamani"),
        "gercek_bitis": kayit.get("Kesinti_Sona_Erme_Zamani"),
        "cbs_lat": lat,
        "cbs_lon": lon,
        "koordinat_semantigi": "kesinti_cbs_noktasi_trafo_degildir",
        "mahalle_sayisi": len(mahalleler),
        "sokak_sayisi": len(sokaklar),
        "adres_nokta_sayisi": len(adres_noktalari),
        "mahalleler_json": json.dumps(mahalleler, ensure_ascii=False),
        "sokaklar_json": json.dumps(sokaklar, ensure_ascii=False),
        # Arayuz bu alani destekliyor; 2026-08-21 planli-kesinti yaniti
        # fiilen dondurmedi. Kaynak ileride doldurursa otomatik yakalanir.
        "etkilenen_abone_sayisi": kayit.get("Kesintiden_Etkilenen_Abone_Sayisi"),
        "kesinti_turu": kayit.get("type"),
        "trafo_kodu": pd.NA,
        "trafo_tipi": pd.NA,
        "trafo_eslesme_durumu": "eslesmedi_api_trafo_kodu_dondurmuyor",
        "cekilme_zamani_utc": cekilme_zamani,
    }


def kesintileri_cek(oturum: requests.Session | None = None) -> pd.DataFrame:
    """Izmir ve Manisa icin acik kesinti-CBS snapshot'ini dondurur."""
    session = oturum or requests.Session()
    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": "GridUpDatathon/1.0 (acik veri arastirmasi)",
            "Referer": SAYFA,
        }
    )
    yeniden_deneme = Retry(
        total=5,
        connect=5,
        read=5,
        status=5,
        backoff_factor=1.0,
        status_forcelist=(429, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
        raise_on_status=True,
    )
    session.mount("https://", HTTPAdapter(max_retries=yeniden_deneme))
    _js_adresi, token = _uygulama_bilgisi(session)
    session.headers["Authorization"] = f"Bearer {token}"
    cekilme_zamani = datetime.now(timezone.utc).isoformat()

    satirlar: list[dict[str, Any]] = []
    for sehir in SEHIRLER:
        ilce_yaniti = session.get(KAYNAK, params={"city": sehir}, timeout=60)
        ilce_yaniti.raise_for_status()
        ilce_verisi = _json_utf8(ilce_yaniti)
        if ilce_verisi.get("success") is not True or not isinstance(ilce_verisi.get("data"), list):
            raise RuntimeError(f"{sehir}: ilce listesi beklenen semada degil")
        for ilce in ilce_verisi["data"]:
            parametreler = {"city": sehir, "district": ilce, "action": "outages"}
            yanit = session.get(KAYNAK, params=parametreler, timeout=60)
            if yanit.status_code >= 400:
                # V2 2026-08-21'de bir ilcede Laravel cache dizini hatasi ile
                # 500 verdi. Eski acik uc ayni kayitlari ``type`` haricinde
                # donduruyor; kapsama kaybetmek yerine acikca bu uca don.
                yanit = session.get(YEDEK_KAYNAK, params=parametreler, timeout=60)
            yanit.raise_for_status()
            veri = _json_utf8(yanit)
            if veri.get("success") is not True or not isinstance(veri.get("data"), list):
                raise RuntimeError(f"{sehir}/{ilce}: kesinti yaniti beklenen semada degil")
            satirlar.extend(
                _satira_cevir(kayit, cekilme_zamani=cekilme_zamani) for kayit in veri["data"]
            )

    sonuc = pd.DataFrame(satirlar)
    if sonuc.empty:
        raise RuntimeError("GDZ API gecerli ama snapshot'ta hic kesinti kaydi yok")
    sonuc["etkilenen_abone_sayisi"] = pd.to_numeric(
        sonuc["etkilenen_abone_sayisi"], errors="coerce"
    ).astype("Int64")
    return sonuc.sort_values(
        ["planlanan_baslangic", "kesinti_id", "ilce"], kind="stable"
    ).reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=VARSAYILAN_CIKTI)
    args = parser.parse_args()

    tablo = kesintileri_cek()
    metadata = publish_dataframe(
        tablo,
        args.out,
        required_columns=(
            "kesinti_id",
            "il",
            "ilce",
            "cbs_lat",
            "cbs_lon",
            "etkilenen_abone_sayisi",
            "trafo_eslesme_durumu",
        ),
        min_rows=1,
        source=KAYNAK,
    )

    print(f"Yayinlandi: {args.out}")
    print(f"  satir: {len(tablo):,}  benzersiz kesinti: {tablo['kesinti_id'].nunique():,}")
    print(f"  CBS koordinati dolu: {tablo['cbs_lat'].notna().sum():,}")
    print(
        f"  etkilenen abone dolu: {tablo['etkilenen_abone_sayisi'].notna().sum():,}/{len(tablo):,}"
    )
    print("  trafo kodu/tipi: 0 (API bu alanlari dondurmuyor; eslesme uydurulmadi)")
    print(f"  sha256: {metadata.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
