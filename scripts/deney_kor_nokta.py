"""KOR NOKTA KOLONLARI: riskin BUYUKLUGUNU olcer (yonunu degil).

SORUN
-----
Uc kolon ailesi testte egitimden BAMBASKA bir rejimde:

  t_ay_sapma     egitimde dolu oldugu satirlarin etiket ayi %100 Kas-Mar,
                 TESTte %100 Nis-Tem. yaz25 ve guz25'te kolon %0,00 dolu,
                 yani ogrenilen iliskinin yaza tasinip tasinmadigi HICBIR
                 blokta olculemez.
  t_gy_*         egitimde %24,4 dolu, testte %52,6. yaz25/guz25 %0,00,
                 yalniz kis26 %49,9.
  t_egim_cdd22   yaz25'te %0,00 -- "test'in mevsimsel ikizi" dedigimiz blok
                 bu kolonun yoklugunda olcum yapiyor.

Bu betik "kaldiralim mi" sorusunu YANITLAMAZ -- yanit test etiketi
olmadan bilinemez. Yanitladigi soru: BU KOLONLAR NE KADAR DEGERLI?
Kaldirmanin maliyeti kucukse, testte yanlis calisma riski de kucuktur;
buyukse hem kazanc hem risk buyuktur ve karar bilinerek verilmelidir.

Karar DEGIL, BUYUKLUK olcumu. Sonuc dokumantasyona gider.

    python scripts/deney_kor_nokta.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
sys.path.insert(0, str(KOK / "scripts"))

import deney as d  # noqa: E402
import deney_ileri as di  # noqa: E402
import tuketim_model as tm  # noqa: E402

from gridup.reporting import satir_tamponlu_cikti  # noqa: E402

SICAK_USTYAZIM: dict[str, object] = {"random_strength": 4.0, "l2_leaf_reg": 1.0, "depth": 6}
SICAK_MASKE = 0.15
KOR = ("t_ay_sapma", "t_gy_log_ort", "t_gy_sifir_orani", "t_gy_gun", "t_egim_cdd22")
KAYIT = KOK / "experiments" / "kor_nokta.jsonl"


def main() -> int:
    satir_tamponlu_cikti()
    t_bas = time.time()
    print("=" * 96)
    print("KOR NOKTA KOLONLARI: kaldirmanin MALIYETI (riskin buyuklugu)")
    print("=" * 96)

    egitim, test = d.cerceveleri_kur()
    tum = [k for k in tm.oznitelikler(egitim) if k in test.columns]
    uretim = [k for k in tum if not k.startswith(tm.YALIN_CIKARILAN)]
    var = [k for k in KOR if k in uretim]
    tm.kategorik_kodla(egitim, test)
    print(f"  uretim {len(uretim)} kolon | kor nokta {len(var)}: {var}")

    parcalar = {b.ad: di.blok_parcalari(egitim, b.ad) for b in tm.BLOKLAR}
    for b in tm.BLOKLAR:
        _, dog, _, sog = parcalar[b.ad]
        dolu = {k: float(dog.loc[~sog, k].notna().mean()) for k in var}
        print(f"  {b.ad} sicak doluluk: " + "  ".join(f"{k.split('_', 1)[1]} %{100 * v:.1f}"
                                                     for k, v in dolu.items()))
    dolu_test = {k: float(test[k].notna().mean()) for k in var}
    print("  TEST doluluk:        " + "  ".join(f"{k.split('_', 1)[1]} %{100 * v:.1f}"
                                                for k, v in dolu_test.items()))

    adaylar = (("TABAN", uretim), ("-KOR NOKTA", [k for k in uretim if k not in var]))
    maskeli = {
        (b.ad, t): d.soguk_maskele(parcalar[b.ad][0], uretim, SICAK_MASKE, t)
        for b in tm.BLOKLAR
        for t in di.TOHUMLAR
    }
    tekil: dict[str, dict[tuple[str, int], float]] = {}
    for ad, kol in adaylar:
        t0 = time.time()
        tekil[ad] = {}
        blok = {}
        for b in tm.BLOKLAR:
            _, dogrulama, gercek, soguk = parcalar[b.ad]
            sicak = ~soguk
            loglar = []
            for tohum in di.TOHUMLAR:
                log_t = di.egit_tahmin("cat", maskeli[(b.ad, tohum)], dogrulama, kol,
                                       tohum, **SICAK_USTYAZIM)
                loglar.append(log_t)
                tek = np.clip(np.expm1(log_t), 0.0, None)
                tekil[ad][(b.ad, tohum)] = tm.rmsle(gercek[sicak], tek[sicak])
            harman = np.clip(np.expm1(np.mean(loglar, axis=0)), 0.0, None)
            blok[b.ad] = tm.rmsle(gercek[sicak], harman[sicak])
        ort = float(np.mean(list(blok.values())))
        detay = "  ".join(f"{k} {v:.5f}" for k, v in blok.items())
        print(f"  {ad:14} {ort:.5f}   {detay}  ({time.time() - t0:.0f} sn)")

    f = np.array([tekil["TABAN"][k] - tekil["-KOR NOKTA"][k] for k in tekil["TABAN"]])
    o, sh = float(f.mean()), float(f.std(ddof=1) / np.sqrt(len(f)))
    t_d = o / sh if sh > 0 else 0.0
    print(f"\n  ESLENIK FARK {o:+.5f}  SH {sh:.5f}  t {t_d:+.2f}")
    for b in tm.BLOKLAR:
        bb = np.array([tekil["TABAN"][(b.ad, t)] - tekil["-KOR NOKTA"][(b.ad, t)]
                       for t in di.TOHUMLAR])
        yon = "YARARLI" if bb.mean() > 0 else "gereksiz"
        print(f"     {b.ad:6} {bb.mean():+.5f}  "
              f"({(bb > 0).sum()}/{len(bb)} tohum kolonlar {yon})")
    print("\n  YORUM: |fark| = kolonlarin OLCULEBILIR degeri. Testte yanlis")
    print("  calisma riski de bu buyuklukte sinirli. Karar test etiketi")
    print("  olmadan verilemez; bu sayi yalnizca BAHSI olcer.")
    print(f"  genel skora yansimasi {-o * 0.528:+.5f}")

    KAYIT.parent.mkdir(parents=True, exist_ok=True)
    with KAYIT.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"kolonlar": var, "fark": o, "sh": sh, "t": t_d},
                            ensure_ascii=False) + "\n")
    print(f"\nTAMAM  {(time.time() - t_bas) / 60:.1f} dakika")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
