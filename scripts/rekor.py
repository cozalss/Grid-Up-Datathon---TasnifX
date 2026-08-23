"""REKOR DEFTERI: bir gonderim adayini TEK bir sayiyla olcer ve kaydeder.

Neden gerekli
-------------
"Daha iyi" iddiasinin nesnel bir olcusu olmali. LB gunde 3 kez ve gecikmeli
cevap veriyor; ara kararlar icin yerel, tekrarlanabilir bir sayi lazim.

Olcut: ``kis26`` uzerinde TEST-AGIRLIKLI RMSLE.

    genel = sqrt(0,7784 * sicak^2 + 0,2216 * soguk^2)

Neden kis26: ezber orani %0 olan TEK durust kat (docs/35). Neden test
agirlikli: test'in %22,16'si soguk, kis26'nin yalnizca %13,9'u -- ham
ortalama sogugu sistematik olarak hafife alir.

Neden bir GONDERIM DOSYASI degil de blok skoru: gonderim dosyasinin
gercek etiketi yok. Bu yuzden defter iki tur kayit tutar:

    tur=blok   kis26 uzerinde olculmus (yerel, her zaman hesaplanabilir)
    tur=lb     Kaggle'dan donen gercek skor (kesin ama gunde 3 tane)

    python scripts/rekor.py --liste
    python scripts/rekor.py --ekle v33 --tur blok --sicak 0.74263 --soguk 1.83606 \
        --not "cat-only soguk + sinir agi + gun korumali son islem"
    python scripts/rekor.py --ekle v30 --tur lb --skor 1.02639
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KOK = Path(__file__).resolve().parents[1]
DEFTER = KOK / "experiments" / "rekor.jsonl"

#: Test karisimindaki soguk pay. Dogrulama raporlariyla ayni sabit.
TEST_SOGUK_PAY = 0.2216


def test_agirlikli(sicak: float, soguk: float) -> float:
    return float((( 1.0 - TEST_SOGUK_PAY) * sicak**2 + TEST_SOGUK_PAY * soguk**2) ** 0.5)


def kayitlari_oku() -> list[dict]:
    if not DEFTER.exists():
        return []
    with DEFTER.open(encoding="utf-8") as fh:
        return [json.loads(s) for s in fh if s.strip()]


def main() -> int:
    a = argparse.ArgumentParser(description="rekor defteri")
    a.add_argument("--liste", action="store_true")
    a.add_argument("--ekle", metavar="AD")
    a.add_argument("--tur", choices=("blok", "lb"), default="blok")
    a.add_argument("--sicak", type=float)
    a.add_argument("--soguk", type=float)
    a.add_argument("--skor", type=float, help="tur=lb icin Kaggle skoru")
    a.add_argument("--not", dest="aciklama", default="")
    ar = a.parse_args()

    if ar.ekle:
        if ar.tur == "blok":
            if ar.sicak is None or ar.soguk is None:
                raise SystemExit("tur=blok icin --sicak ve --soguk gerekli")
            skor = test_agirlikli(ar.sicak, ar.soguk)
            kayit = {"ad": ar.ekle, "tur": "blok", "sicak": ar.sicak,
                     "soguk": ar.soguk, "skor": skor, "not": ar.aciklama}
        else:
            if ar.skor is None:
                raise SystemExit("tur=lb icin --skor gerekli")
            kayit = {"ad": ar.ekle, "tur": "lb", "skor": ar.skor, "not": ar.aciklama}
        DEFTER.parent.mkdir(parents=True, exist_ok=True)
        with DEFTER.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(kayit, ensure_ascii=False) + "\n")
        print(f"  eklendi: {kayit['ad']}  {kayit['tur']}  {kayit['skor']:.5f}")

    kayitlar = kayitlari_oku()
    if ar.liste or ar.ekle:
        for tur, baslik in (("lb", "KAGGLE LB (kesin)"), ("blok", "kis26 TEST-AGIRLIKLI (yerel)")):
            alt = [k for k in kayitlar if k["tur"] == tur]
            if not alt:
                continue
            print(f"\n  {baslik}")
            en_iyi = min(alt, key=lambda k: k["skor"])
            for k in sorted(alt, key=lambda k: k["skor"]):
                yildiz = "  <- REKOR" if k is en_iyi else ""
                detay = ""
                if tur == "blok":
                    detay = f"  (sicak {k['sicak']:.5f} soguk {k['soguk']:.5f})"
                print(f"    {k['ad']:22} {k['skor']:.5f}{detay}{yildiz}")
                if k["not"]:
                    print(f"      {k['not']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
