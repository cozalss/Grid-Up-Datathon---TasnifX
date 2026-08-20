"""EPIAS kimlik servisi DONUNCE veriyi otomatik ceker -- nobetci betik.

NEDEN BU BETIK
--------------
2026-08-20 23:47'de olculdu: EPIAS'in kimlik (CAS) ucu kapali.

    https://giris.epias.com.tr/            -> HTTP 302, server=EPIAS  (site AYAKTA)
    https://giris.epias.com.tr/cas/v1/tickets -> HTTP 503, ciplak nginx sayfasi
    https://cas.epias.com.tr/cas/v1/tickets   -> HTTP 401 (eski CAS, bu hesabi tanimiyor)
    https://seffaflik.epias.com.tr/        -> HTTP 200

Tarayici User-Agent'i ile de ayni 503 geliyor, yani istek suzme degil GERCEK
kesinti. Kimlik bilgilerinin bicimi de dogrulandi (e-posta, bosluksuz).

Yani sorun bizde degil ve yapilacak tek sey BEKLEMEK. Ama beklemeyi insanin
yapmasi gereksiz: bu betik ucu yoklar, ayaga kalktigi anda cekimi baslatir ve
kapilari kosar.

NEDEN ONEMLI
------------
EPIAS ulusal tuketimi, panelin ufkunu SINIRLAYAN tek kaynak: hava verisi
2026-08-26'ya kadar dolu, EPIAS 2026-08-15'te duruyor ve
``scripts/kapsam_deseni.py`` paneli en dar kaynaga kadar kurdugu icin
kullanilabilir panel de orada bitiyor. EPIAS donerse ufuk dort gun genisler.

KULLANIM
--------
::

    python scripts/epias_bekle.py                  # varsayilan: 10 dk'da bir, 12 saat
    python scripts/epias_bekle.py --aralik 300 --azami-saat 6
    python scripts/epias_bekle.py --yalniz-kontrol # bir kez bak, cikis kodu ver

Cikis kodu: 0 = veri cekildi (ya da --yalniz-kontrol'de servis ayakta),
1 = sure doldu / servis hala kapali.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable

from gridup.epias import TGT_URL  # noqa: E402

#: Yoklama araligi (sn). 10 dakika: kesinti pencereleri saatler surer, daha
#: sik yoklamak servise nezaketsizlik ve bize hicbir sey kazandirmaz.
VARSAYILAN_ARALIK_SN = 600

#: En fazla ne kadar beklenecek (saat).
VARSAYILAN_AZAMI_SAAT = 12.0

#: 503 = servis kapali. 401/405 = servis AYAKTA (yalnizca istegimizi
#: begenmedi) -- yani beklemeyi bitirmemiz gereken durum budur.
KAPALI_KODLARI = (500, 502, 503, 504)


def uc_ayakta_mi(zaman_asimi: int = 20) -> tuple[bool, str]:
    """Kimlik ucu istek kabul ediyor mu? ``(ayakta, aciklama)``.

    Kimlik bilgisi GONDERMEZ -- yalnizca ucun cevap verip vermedigine bakar.
    """
    try:
        yanit = requests.get(TGT_URL, timeout=zaman_asimi)
    except requests.RequestException as hata:
        return False, f"baglanti hatasi ({type(hata).__name__})"
    if yanit.status_code in KAPALI_KODLARI:
        return False, f"HTTP {yanit.status_code} -- servis kapali"
    return True, f"HTTP {yanit.status_code} -- servis cevap veriyor"


def _dun() -> str:
    return (date.today() - timedelta(days=1)).isoformat()


def cek_ve_dogrula(end: str) -> int:
    """EPIAS'i ceker, ardindan kapilari kosar. Cikis kodunu doner."""
    adimlar = [
        (
            "epias ulusal yuk",
            [PY, "scripts/fetch_epias_load.py", "--start", "2020-01-01", "--end", end],
        ),
        ("kapi: veri sagligi", [PY, "scripts/veri_sagligi.py"]),
        ("kapi: kapsam deseni", [PY, "scripts/kapsam_deseni.py"]),
        ("kapi: manifest tazeleme", [PY, "scripts/manifest_tazele.py"]),
    ]
    for ad, komut in adimlar:
        print(f"\n{'=' * 70}\n>>> {ad}\n")
        sonuc = subprocess.run(komut, cwd=ROOT, check=False)  # noqa: S603
        if sonuc.returncode != 0:
            print(f"\n<<< {ad}: BASARISIZ (exit {sonuc.returncode})")
            return int(sonuc.returncode)
        print(f"\n<<< {ad}: TAMAM")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aralik", type=int, default=VARSAYILAN_ARALIK_SN, help="Yoklama araligi (sn)")
    ap.add_argument("--azami-saat", type=float, default=VARSAYILAN_AZAMI_SAAT)
    ap.add_argument("--yalniz-kontrol", action="store_true", help="Bir kez bak ve cik")
    ap.add_argument("--end", default=None, help="Cekim bitisi (varsayilan: dun)")
    args = ap.parse_args()

    if args.yalniz_kontrol:
        ayakta, aciklama = uc_ayakta_mi()
        print(f"EPIAS kimlik ucu: {aciklama}")
        return 0 if ayakta else 1

    bitis = time.monotonic() + args.azami_saat * 3600
    deneme = 0
    print(f"EPIAS NOBETI  ({datetime.now():%Y-%m-%d %H:%M})")
    print("=" * 70)
    print(f"  uc      : {TGT_URL}")
    print(f"  yoklama : {args.aralik} sn'de bir, en fazla {args.azami_saat} saat")
    print("  servis ayaga kalkinca cekim + kapilar OTOMATIK kosacak.\n")

    while time.monotonic() < bitis:
        deneme += 1
        ayakta, aciklama = uc_ayakta_mi()
        print(f"[{datetime.now():%H:%M:%S}] deneme {deneme:3d}: {aciklama}", flush=True)
        if ayakta:
            print("\nServis dondu -- cekim baslatiliyor.")
            return cek_ve_dogrula(args.end or _dun())
        time.sleep(args.aralik)

    print(
        f"\n{args.azami_saat} saat doldu, servis hala kapali. "
        "Veri MEVCUT HALIYLE saglam; yalnizca EPIAS 2026-08-15'te kalir."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
